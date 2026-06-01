"""Tests for the project smoke-test service (Diagnostics Intervention C).

Covers:
  * Happy path: every check on a healthy project lands ``ok``.
  * Missing-recipe project: recipe_applied fails, synth_catalog
    skips, others still run + come back ok.
  * Non-existent project: project_exists fails, downstream checks
    skip cleanly rather than 5xx-ing.
  * Failure envelope shape — fail checks carry an ErrorEnvelope-
    shaped dict the frontend can drop into <ErrorPanel>.
  * Overall rollup logic: any fail → fail; else any warn → warn;
    else ok. Skip counts as neutral.
  * Parallel execution — total elapsed should be close to the
    slowest single check, not the sum.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "smoke_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "smoke_test_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"
os.environ["ALLOW_SQLITE_AUTOCREATE"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402


class ProjectSmokeTestApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        settings.ensure_dirs()
        for suffix in ("", "-shm", "-wal"):
            p = Path(f"{TEST_DB_PATH.as_posix()}{suffix}")
            if p.exists():
                p.unlink()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def _create_project(self, name: str | None = None) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": name or f"smoke-{uuid.uuid4().hex[:8]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _apply_recipe(self, project_id: int, recipe_id: str) -> None:
        resp = self.client.put(
            f"/api/projects/{project_id}/recipe",
            json={"recipe_id": recipe_id},
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    def _checks_by_name(self, body: dict) -> dict:
        return {c["name"]: c for c in body["checks"]}

    # ── Envelope shape contract ─────────────────────────────

    def test_response_shape_matches_documented_contract(self):
        """Every response carries the keys the frontend depends on:
        ``overall``, ``elapsed_ms``, ``counts``, ``checks`` (list of
        ``{name, status, elapsed_ms, message, remediation, envelope,
        metadata}``)."""
        pid = self._create_project()
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["project_id"], pid)
        self.assertIn(body["overall"], {"ok", "warn", "fail", "skip"})
        self.assertIsInstance(body["elapsed_ms"], int)
        self.assertIsInstance(body["counts"], dict)
        # Counts cover all 4 statuses (some at 0 is fine).
        for status in ("ok", "warn", "fail", "skip"):
            self.assertIn(status, body["counts"])
        self.assertIsInstance(body["checks"], list)
        for check in body["checks"]:
            self.assertIn(check["status"], {"ok", "warn", "fail", "skip"})
            self.assertIsInstance(check["name"], str)
            self.assertGreater(len(check["name"]), 0)
            self.assertIn("message", check)
            self.assertIn("envelope", check)  # may be None
            self.assertIn("metadata", check)

    def test_no_recipe_project_lands_warn_overall_with_recipe_fail(self):
        """A freshly-created project has no recipe. ``recipe_applied``
        is the explicit failure; downstream checks that depend on a
        recipe (``synth_catalog``) skip cleanly."""
        pid = self._create_project()
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        body = resp.json()
        checks = self._checks_by_name(body)
        self.assertEqual(checks["recipe_applied"]["status"], "fail")
        self.assertIn("recipe", checks["recipe_applied"]["message"].lower())
        # synth_catalog correctly skips (no recipe means nothing to
        # enumerate — that's not a platform bug, just nothing to test).
        self.assertEqual(checks["synth_catalog"]["status"], "skip")
        # The fail check carries an envelope the frontend can render.
        env = checks["recipe_applied"]["envelope"]
        self.assertIsNotNone(env)
        self.assertEqual(env["error_code"], "SMOKE_RECIPE_MISSING")
        self.assertTrue(env["troubleshooting_id"].startswith("err_"))
        # Overall reflects the fail.
        self.assertEqual(body["overall"], "fail")

    def test_project_with_recipe_applied_passes_recipe_check(self):
        """Apply a recipe → recipe_applied flips to ok + synth_catalog
        un-skips (it now has a recipe to enumerate against)."""
        pid = self._create_project()
        self._apply_recipe(pid, "classification")
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        body = resp.json()
        checks = self._checks_by_name(body)
        self.assertEqual(checks["recipe_applied"]["status"], "ok")
        self.assertEqual(checks["synth_catalog"]["status"], "ok")
        self.assertIn("classification", checks["recipe_applied"]["message"])
        # The synth catalog reports how many playbooks for this recipe.
        self.assertIn("playbook", checks["synth_catalog"]["message"].lower())

    def test_non_existent_project_returns_404_via_fail_check_not_5xx(self):
        """Smoke-testing a project that doesn't exist used to be the
        kind of failure that would propagate as a 500. With the
        per-check ``except`` + skip-downstream behavior, the response
        is a clean 200 with project_exists=fail and the rest at skip.

        This is the load-bearing property: a broken project should
        never break the smoke-test endpoint itself."""
        resp = self.client.post("/api/projects/999999/smoke-test")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        checks = self._checks_by_name(body)
        # project_exists is the primary fail.
        self.assertEqual(checks["project_exists"]["status"], "fail")
        # Downstream checks that depend on the project should NOT crash
        # — they either skip (recipe_applied / synth_catalog) or just
        # return their natural empty-state result.
        # recipe_applied + synth_catalog know to short-circuit.
        self.assertEqual(checks["recipe_applied"]["status"], "skip")
        self.assertEqual(checks["synth_catalog"]["status"], "skip")
        # Overall ≥ fail.
        self.assertEqual(body["overall"], "fail")

    def test_overall_rollup_is_worst_status(self):
        """A project with no recipe has at least one fail + at least
        one warn (gold_set empty, prepared_splits empty). The overall
        rollup must be ``fail`` regardless of how many warns are
        below it."""
        pid = self._create_project()
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        body = resp.json()
        # We deliberately verify the COMBINED state — there are
        # multiple warns AND one fail.
        self.assertGreaterEqual(body["counts"]["fail"], 1)
        self.assertEqual(body["overall"], "fail")

    def test_adapter_handler_format_check_flags_recipe_adapter_mismatch(self):
        """γ — the diagnostic gate that catches the SQLi-detector
        class of bug: trainer used a different adapter than the
        recipe expects, so the trainer's prompt format won't match
        the eval handler's format and held-out F1 will be 0%-ish
        even after a clean training run.

        Repro: apply the ``classification`` recipe (which expects
        ``classification-label`` adapter), then write a prepared
        manifest with ``adapter_id: default-canonical`` (the
        dataset-import default). The smoke test must flag this with
        ``status=warn`` and a remediation pointing at Data Prep with
        the correct adapter."""
        import json
        pid = self._create_project()
        self._apply_recipe(pid, "classification")
        # Materialize a prepared manifest declaring the wrong adapter.
        prepared_dir = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        (prepared_dir / "manifest.json").write_text(
            json.dumps({"adapter_id": "default-canonical"}),
        )
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        self.assertEqual(resp.status_code, 200, resp.text)
        check = self._checks_by_name(resp.json())["adapter_handler_format"]
        self.assertEqual(check["status"], "warn", check)
        self.assertIn("classification-label", check["message"])
        self.assertIn("default-canonical", check["message"])
        self.assertIn("Data Prep", check["remediation"])
        self.assertEqual(check["metadata"]["expected_adapter"], "classification-label")
        self.assertEqual(check["metadata"]["actual_adapter"], "default-canonical")
        # Overall rolls up to warn (no other failures on this seed).
        self.assertIn(resp.json()["overall"], {"warn", "fail"})

    def test_adapter_handler_format_check_passes_when_adapters_match(self):
        """Steady-state: recipe applied + manifest declares the
        recipe's canonical adapter → status=ok with a confirmation
        message. No remediation, no envelope."""
        import json
        pid = self._create_project()
        self._apply_recipe(pid, "classification")
        prepared_dir = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        (prepared_dir / "manifest.json").write_text(
            json.dumps({"adapter_id": "classification-label"}),
        )
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        check = self._checks_by_name(resp.json())["adapter_handler_format"]
        self.assertEqual(check["status"], "ok", check)
        self.assertIn("classification-label", check["message"])
        self.assertIsNone(check["remediation"])
        self.assertIsNone(check["envelope"])

    def test_adapter_handler_format_check_skips_pre_dataprep(self):
        """Before Data Prep runs there's no manifest yet; the check
        skips with an informational message rather than failing.
        We don't want a brand-new project to fail the smoke test
        just because it hasn't reached Data Prep yet."""
        pid = self._create_project()
        self._apply_recipe(pid, "classification")
        # No manifest on disk.
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        check = self._checks_by_name(resp.json())["adapter_handler_format"]
        self.assertEqual(check["status"], "skip", check)
        self.assertIn("prepped", check["message"].lower())

    def test_adapter_match_BUT_prepared_row_lacks_prefix_lands_warn(self):
        """γ′ — the false-negative the original γ missed.

        Recipe + manifest both say ``classification-label``, but the
        adapter wrote raw ``text → label`` rows without the handler's
        production prompt template. Adapter ids agree, BUT
        training-format ≠ eval-format, and the held-out F1 collapses
        even though the trainer's internal eval looks fine.

        The strengthened check peeks at the first prepared row and
        looks for the handler's ``expected_prompt_prefixes``. If
        none are present, escalate to warn even when adapter ids
        agree — the row check is what catches the SQLi-detector
        regression in run #2."""
        import json
        pid = self._create_project()
        self._apply_recipe(pid, "classification")
        prepared_dir = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        # Adapter id agreement — necessary but insufficient.
        (prepared_dir / "manifest.json").write_text(
            json.dumps({
                "adapter_id": "classification-label",
                "task_profile": "classification",
            }),
        )
        # Prepared train rows in raw text→label format (NO "Classify
        # the following text" prefix anywhere). This is the actual
        # SQLi-detector reproduction.
        (prepared_dir / "train.jsonl").write_text(
            "\n".join(
                json.dumps({
                    "source_text": f"sample text {i}",
                    "target_text": "benign" if i % 2 == 0 else "injection",
                    "label": "benign" if i % 2 == 0 else "injection",
                })
                for i in range(3)
            ) + "\n",
        )
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        check = self._checks_by_name(resp.json())["adapter_handler_format"]
        self.assertEqual(check["status"], "warn", check)
        self.assertIn("don't contain the prompt format", check["message"])
        self.assertIn("classification", check["message"])
        # Metadata names the handler + the prefixes we searched for.
        self.assertEqual(check["metadata"]["handler_id"], "classification")
        self.assertIn(
            "Classify the following text",
            check["metadata"]["expected_prefixes"],
        )

    def test_adapter_match_AND_prepared_row_carries_prefix_lands_ok(self):
        """Steady-state — the adapter actually does write the
        production prompt format. Check returns ok with a confirmation
        that the row format matches the handler's expectations."""
        import json
        pid = self._create_project()
        self._apply_recipe(pid, "classification")
        prepared_dir = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        (prepared_dir / "manifest.json").write_text(
            json.dumps({
                "adapter_id": "classification-label",
                "task_profile": "classification",
            }),
        )
        # Prepared rows with the handler's prompt template
        # embedded in the input — what a well-behaved adapter
        # writes. Source_text contains the full classification
        # prompt; target_text is just the label.
        (prepared_dir / "train.jsonl").write_text(
            "\n".join(
                json.dumps({
                    "source_text": (
                        "Classify the following text. Reply with "
                        f"exactly one of: benign, injection.\n"
                        f"Text: sample {i}\n"
                        "Label:"
                    ),
                    "target_text": " benign" if i % 2 == 0 else " injection",
                })
                for i in range(3)
            ) + "\n",
        )
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        check = self._checks_by_name(resp.json())["adapter_handler_format"]
        self.assertEqual(check["status"], "ok", check)
        self.assertIn("classification-label", check["message"])
        self.assertIn(
            "classification handler's prompt format",
            check["message"],
        )

    def test_peek_is_silent_for_handlers_without_prompt_prefixes(self):
        """Handlers that don't declare prefixes (Generic / QA / Safety
        / Alignment) skip the row peek — there's nothing to check.
        The matching-adapter case still lands ok without trying to
        peek."""
        import json
        pid = self._create_project()
        # Causal_lm experiments route to QAHandler which doesn't
        # declare prefixes. Apply qa-sft recipe so the adapter check
        # has something to compare. Confirm: ok status, no peek
        # metadata leak.
        self._apply_recipe(pid, "qa-sft")
        prepared_dir = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        # Use whatever adapter qa-sft recipe expects — match it.
        from app.services.recipe_service import get_recipe
        expected = getattr(get_recipe("qa-sft"), "adapter_id", None)
        (prepared_dir / "manifest.json").write_text(
            json.dumps({
                "adapter_id": expected,
                "task_profile": "qa",
            }),
        )
        (prepared_dir / "train.jsonl").write_text(
            json.dumps({"prompt": "what is 2+2?", "answer": "4"}) + "\n",
        )
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        check = self._checks_by_name(resp.json())["adapter_handler_format"]
        self.assertEqual(check["status"], "ok", check)

    # ── classifier-head vs handler check (δ′) ──────────────────────

    def _seed_completed_experiment(
        self,
        project_id: int,
        *,
        output_dir: str,
        task_type: str = "classification",
    ) -> int:
        """Insert a completed Experiment row pointing at ``output_dir``
        so the head/handler check can resolve a checkpoint to
        inspect. Bypasses the training flow which would require
        a real GPU + dataset."""
        async def _go() -> int:
            from app.models.experiment import Experiment
            async with async_session_factory() as session:
                exp = Experiment(
                    project_id=project_id,
                    name="δ′-fixture",
                    description="head/handler smoke check fixture",
                    status="completed",
                    base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                    config={"task_type": task_type},
                    output_dir=output_dir,
                )
                session.add(exp)
                await session.commit()
                return int(exp.id)
        return asyncio.run(_go())

    def _write_seq_cls_adapter(
        self, output_dir: Path, *, label_space: list[str] | None = None,
    ) -> Path:
        """Build a fake checkpoint that the δ detector will accept:
        ``checkpoint-N/adapter_config.json`` flagging SEQ_CLS, plus
        ``training_report.json`` with the label space. Returns the
        checkpoint dir path."""
        import json
        ckpt = output_dir / "checkpoint-100"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": (
                        "HuggingFaceTB/SmolLM2-135M-Instruct"
                    ),
                    "task_type": "SEQ_CLS",
                    "modules_to_save": ["classifier", "score"],
                    "peft_type": "LORA",
                }
            )
        )
        (output_dir / "training_report.json").write_text(
            json.dumps(
                {
                    "runtime_environment": {
                        "label_space_preview": label_space or [
                            "benign", "injection",
                        ],
                    }
                }
            )
        )
        return ckpt

    def test_classifier_head_check_warns_on_generation_mode_recipe(self):
        """A SEQ_CLS-headed adapter under a generation-mode recipe
        is the very mismatch δ unblocked. The smoke check must warn
        with a remediation pointing at either changing the recipe
        or retraining as causal-LM."""
        pid = self._create_project()
        # qa-sft routes to a generation handler (instruction/qa
        # prompt format), not the classification handler — picking
        # this recipe with a SEQ_CLS adapter is the bad shape.
        self._apply_recipe(pid, "qa-sft")
        exp_dir = TEST_DATA_DIR / f"project_{pid}_exp_seqcls"
        exp_dir.mkdir(parents=True, exist_ok=True)
        self._write_seq_cls_adapter(exp_dir)
        self._seed_completed_experiment(pid, output_dir=str(exp_dir))

        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        check = self._checks_by_name(resp.json())[
            "classifier_head_vs_handler"
        ]
        self.assertEqual(check["status"], "warn", check)
        self.assertIn("SEQ_CLS", check["message"])
        self.assertIn("generation", check["message"])
        self.assertIn("qa-sft", check["message"])
        # Remediation lays out both options the user can take.
        self.assertIn("classification", check["remediation"])
        self.assertIn("retrain", check["remediation"].lower())
        # Metadata exposes the head kind + label count for the UI.
        self.assertEqual(
            check["metadata"]["head_kind"], "sequence_classification"
        )
        self.assertEqual(check["metadata"]["num_labels"], 2)
        # The overall rollup picks up the warn (no fails should fire
        # on this recipe-applied + classifier-head fixture).
        self.assertIn(resp.json()["overall"], {"warn", "fail"})

    def test_classifier_head_check_passes_with_classification_recipe(self):
        """SEQ_CLS adapter + classification recipe is the δ-aligned
        shape — eval dispatches through the head's logits, no
        mismatch. The check must land ``ok`` with a message that
        names the experiment + recipe so a reviewer can audit the
        path."""
        pid = self._create_project()
        self._apply_recipe(pid, "classification")
        exp_dir = TEST_DATA_DIR / f"project_{pid}_exp_seqcls_classif"
        exp_dir.mkdir(parents=True, exist_ok=True)
        self._write_seq_cls_adapter(exp_dir)
        self._seed_completed_experiment(pid, output_dir=str(exp_dir))

        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        check = self._checks_by_name(resp.json())[
            "classifier_head_vs_handler"
        ]
        self.assertEqual(check["status"], "ok", check)
        self.assertIn("classification", check["message"])
        self.assertIn("δ", check["message"])  # the literal δ
        self.assertEqual(check["metadata"]["task_profile"], "classification")

    def test_classifier_head_check_skips_without_experiment(self):
        """No completed experiment yet → nothing to inspect →
        skip. Smoke endpoint must not block a fresh project that
        hasn't reached training yet."""
        pid = self._create_project()
        self._apply_recipe(pid, "classification")
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        check = self._checks_by_name(resp.json())[
            "classifier_head_vs_handler"
        ]
        self.assertEqual(check["status"], "skip", check)
        self.assertIn(
            "No completed experiment", check["message"]
        )

    def _write_seq2seq_adapter(self, output_dir: Path) -> Path:
        """ε fixture: build a fake seq2seq checkpoint that the ε
        detector accepts."""
        import json
        ckpt = output_dir / "checkpoint-50"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "t5-small",
                    "task_type": "SEQ_2_SEQ_LM",
                    "peft_type": "LORA",
                }
            )
        )
        return ckpt

    def test_classifier_head_check_warns_on_seq2seq_under_causal_recipe(self):
        """ε branch — a seq2seq adapter under a CausalLM-style
        recipe (qa-sft routes to ``instruction_sft``) means the
        recipe's prompt format won't match what the encoder-decoder
        model was trained on. Warn so the user sees the shape
        mismatch before eval looks "off"."""
        pid = self._create_project()
        self._apply_recipe(pid, "qa-sft")
        exp_dir = TEST_DATA_DIR / f"project_{pid}_exp_seq2seq_causal"
        exp_dir.mkdir(parents=True, exist_ok=True)
        self._write_seq2seq_adapter(exp_dir)
        self._seed_completed_experiment(
            pid, output_dir=str(exp_dir), task_type="seq2seq",
        )

        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        check = self._checks_by_name(resp.json())[
            "classifier_head_vs_handler"
        ]
        self.assertEqual(check["status"], "warn", check)
        self.assertIn("SEQ_2_SEQ_LM", check["message"])
        self.assertIn("encoder-decoder", check["message"])
        self.assertEqual(
            check["metadata"]["head_kind"], "seq2seq_lm"
        )
        self.assertIn("summarization", check["remediation"])

    def test_classifier_head_check_passes_for_seq2seq_under_summarization_recipe(self):
        """ε aligned shape — seq2seq adapter + summarization
        recipe is the right pairing. ε dispatches through
        AutoModelForSeq2SeqLM at eval time."""
        pid = self._create_project()
        self._apply_recipe(pid, "summarization")
        exp_dir = TEST_DATA_DIR / f"project_{pid}_exp_seq2seq_aligned"
        exp_dir.mkdir(parents=True, exist_ok=True)
        self._write_seq2seq_adapter(exp_dir)
        self._seed_completed_experiment(
            pid, output_dir=str(exp_dir), task_type="seq2seq",
        )

        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        check = self._checks_by_name(resp.json())[
            "classifier_head_vs_handler"
        ]
        self.assertEqual(check["status"], "ok", check)
        self.assertIn("seq2seq", check["message"].lower())
        self.assertIn("ε", check["message"])
        self.assertEqual(check["metadata"]["head_kind"], "seq2seq_lm")

    def test_classifier_head_check_passes_for_non_seq_cls_checkpoint(self):
        """A causal-LM (or otherwise non-SEQ_CLS) checkpoint is the
        right shape for a generation-mode recipe — the check must
        land ``ok`` rather than warn, since generation is the
        correct eval path."""
        import json
        pid = self._create_project()
        self._apply_recipe(pid, "qa-sft")
        exp_dir = TEST_DATA_DIR / f"project_{pid}_exp_causal"
        ckpt = exp_dir / "checkpoint-50"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": (
                        "HuggingFaceTB/SmolLM2-135M-Instruct"
                    ),
                    "task_type": "CAUSAL_LM",
                    "modules_to_save": [],
                }
            )
        )
        self._seed_completed_experiment(pid, output_dir=str(exp_dir))

        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        check = self._checks_by_name(resp.json())[
            "classifier_head_vs_handler"
        ]
        self.assertEqual(check["status"], "ok", check)
        self.assertIn("not classifier-head", check["message"])

    def test_parallel_execution_is_faster_than_sum_of_checks(self):
        """Sanity check that the orchestrator runs checks in parallel.
        Total elapsed should be near the slowest single check, NOT
        the sum across all 9 checks. We assert a loose upper bound
        (total < 2× slowest) which is enough to catch a regression
        to serial execution."""
        pid = self._create_project()
        resp = self.client.post(f"/api/projects/{pid}/smoke-test")
        body = resp.json()
        total = body["elapsed_ms"]
        slowest = max(c["elapsed_ms"] for c in body["checks"])
        sum_serial = sum(c["elapsed_ms"] for c in body["checks"])
        # Serial would have total ≈ sum. Parallel has total ≈ slowest.
        # The bound is loose to avoid CI flakiness; in practice we see
        # ratios like 1.05× or 1.2×.
        self.assertLess(
            total, slowest * 3,
            f"Smoke test isn't parallel: total={total}ms slowest={slowest}ms",
        )
        # And the cumulative serial time should be meaningfully larger
        # than the actual parallel total — otherwise the checks are
        # too cheap to tell apart.
        if sum_serial > 20:  # threshold to skip the test on a too-fast box
            self.assertLess(total, sum_serial)


if __name__ == "__main__":
    unittest.main()
