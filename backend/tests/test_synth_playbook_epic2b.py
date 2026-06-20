"""Tests for USER-SUCCESS Epic 2b — hard-negatives, class-balance,
cluster-targeted, the synth review queue, and the dataset-prep gate.

Coverage:
  * playbook registry catalog after Epic 2b (17 playbooks expected)
  * HARD_NEGATIVES classification rejects rows labeled target_class
  * HARD_NEGATIVES span-extraction drops rows with non-empty spans
  * HARD_NEGATIVES generic-sft penalizes non-refusal completions
  * CLASS_BALANCE_FILL auto-picks the minority class + rejects wrong labels
  * CLUSTER_TARGETED cluster_block embedded in the prompt
  * augment_from_cluster orchestrator wires through to the playbook
  * /review-queue lists pending rows grouped by synth_source
  * /review-queue/bulk-update accept flips review_status; reject removes
  * dataset_service._load_records_from_file gates pending rows by default
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "synth_epic2b_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "synth_epic2b_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.dataset_service import _load_records_from_file
from app.services.synth_playbooks import SynthMode, get_playbook, list_playbooks


class CatalogAfterEpic2bTests(unittest.TestCase):
    def test_registry_has_20_playbooks_with_expected_coverage(self):
        playbooks = list_playbooks()
        self.assertEqual(len(playbooks), 20)
        # Coverage matrix sanity:
        # qa-sft + summarization: paraphrase + cluster (2 modes each)
        # classification: paraphrase + hard_neg + balance + cluster (4 modes)
        # code-review / generic-sft / span-extraction: paraphrase + hard_neg + cluster (3 modes each)
        # rag-protocol: paraphrase + refusals + format_robustness (3 modes)
        from collections import defaultdict
        by_recipe = defaultdict(set)
        for pb in playbooks:
            by_recipe[pb["recipe_id"]].add(pb["mode"])
        self.assertEqual(by_recipe["classification"], {"positives_paraphrase", "hard_negatives", "class_balance_fill", "cluster_targeted"})
        self.assertEqual(by_recipe["qa-sft"], {"positives_paraphrase", "cluster_targeted"})
        self.assertEqual(by_recipe["summarization"], {"positives_paraphrase", "cluster_targeted"})
        self.assertEqual(by_recipe["rag-protocol"], {"positives_paraphrase", "refusals", "format_robustness"})
        for r in ("code-review", "generic-sft", "span-extraction"):
            self.assertEqual(
                by_recipe[r],
                {"positives_paraphrase", "hard_negatives", "cluster_targeted"},
                f"{r} should have paraphrase + hard_neg + cluster",
            )


class HardNegativeUnitTests(unittest.TestCase):
    def test_classification_hard_negatives_drops_rows_labeled_target_class(self):
        pb = get_playbook("classification", SynthMode.HARD_NEGATIVES)
        gold = [
            {"text": "I want a refund", "label": "billing"},
            {"text": "App keeps crashing", "label": "technical"},
        ]
        ctx = {
            "recipe_id": "classification",
            "project_id": 1,
            "gold_rows": gold,
            "target_count": 5,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": "billing",
        }
        # Force the playbook to set target_class on ctx.
        pb.build_prompt(ctx)
        canned = [
            {"text": "App is slow", "label": "technical"},  # valid hard negative
            {"text": "Please refund me", "label": "billing"},  # generation failure — same class as target
            {"text": "Crash", "label": "invented"},  # unknown class — penalized
        ]
        validated = pb.validate(canned, ctx)
        # 2 rows accepted; the billing-labeled row dropped entirely
        # (it would be a positives_paraphrase, not a hard negative).
        labels = [r["payload"]["label"] for r in validated]
        self.assertEqual(len(validated), 2)
        self.assertNotIn("billing", labels)
        # Unknown label survives but with penalty.
        invented = next(r for r in validated if r["payload"]["label"] == "invented")
        self.assertLess(invented["synth_confidence"], 0.5)

    def test_span_extraction_hard_negatives_requires_empty_spans(self):
        pb = get_playbook("span-extraction", SynthMode.HARD_NEGATIVES)
        gold = [{"text": "John Doe", "spans": [{"type": "name", "start": 0, "end": 8, "text": "John Doe"}]}]
        ctx = {
            "recipe_id": "span-extraction",
            "project_id": 1,
            "gold_rows": gold,
            "target_count": 3,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }
        canned = [
            {"text": "The weather is nice today", "spans": []},  # valid negative
            {"text": "Jane Smith lives here", "spans": [{"type": "name", "start": 0, "end": 10, "text": "Jane Smith"}]},  # NOT a negative — drop
            {"text": "Just plain text", "spans": None},  # treat None as empty — valid
        ]
        validated = pb.validate(canned, ctx)
        # 2 negatives accepted; the one with non-empty spans dropped.
        self.assertEqual(len(validated), 2)
        for r in validated:
            self.assertEqual(r["payload"]["spans"], [])

    def test_generic_sft_hard_negatives_penalizes_non_refusal_completions(self):
        pb = get_playbook("generic-sft", SynthMode.HARD_NEGATIVES)
        gold = [{"prompt": "Reset password", "completion": "Click Settings"}]
        ctx = {
            "recipe_id": "generic-sft",
            "project_id": 1,
            "gold_rows": gold,
            "target_count": 3,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }
        canned = [
            {"prompt": "Pick a movie", "completion": "I can't help with that, sorry."},  # refusal
            {"prompt": "What's the weather?", "completion": "It's sunny and 72 degrees in NYC."},  # substantive — penalty
        ]
        validated = pb.validate(canned, ctx)
        self.assertEqual(len(validated), 2)
        refusal = next(r for r in validated if "can't help" in r["payload"]["completion"])
        substantive = next(r for r in validated if "sunny" in r["payload"]["completion"])
        self.assertEqual(refusal["synth_confidence"], 1.0)
        self.assertLess(substantive["synth_confidence"], 0.5)


class ClassBalanceFillUnitTests(unittest.TestCase):
    def test_balance_fill_auto_picks_minority_class_and_rejects_others(self):
        pb = get_playbook("classification", SynthMode.CLASS_BALANCE_FILL)
        # 5 majority + 1 minority — minority is "rare".
        gold = [{"text": f"row {i}", "label": "common"} for i in range(5)] + [
            {"text": "the rare one", "label": "rare"},
        ]
        ctx = {
            "recipe_id": "classification",
            "project_id": 1,
            "gold_rows": gold,
            "target_count": 4,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }
        prompt = pb.build_prompt(ctx)
        # After build_prompt, the playbook stashes the resolved target.
        self.assertEqual(ctx["target_class"], "rare")
        # The class distribution should be visible in the prompt.
        self.assertIn("rare", prompt)
        self.assertIn("common", prompt)
        # Validator now drops rows labeled anything other than "rare".
        canned = [
            {"text": "new rare example", "label": "rare"},
            {"text": "wrong class output", "label": "common"},  # not target — drop
        ]
        validated = pb.validate(canned, ctx)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["payload"]["label"], "rare")

    def test_balance_fill_prompt_renders_target_class_as_valid_json(self):
        """Regression: the JSON example in the prompt used Python's
        ``{target_class!r}`` repr, rendering as single-quoted
        ``'injection'``. Some LLMs (Llama 3) mirrored that style in
        their output, producing ``'label': 'injection'`` which the
        JSON parser rejected. Every generated row failed validation,
        and the playbook reported '0 accepted rows' with a generic
        diagnostic that didn't name the cause.

        The fix renders the label as proper JSON: ``"injection"``."""
        pb = get_playbook("classification", SynthMode.CLASS_BALANCE_FILL)
        gold = [{"text": "x", "label": "benign"}, {"text": "y", "label": "injection"}]
        ctx = {
            "recipe_id": "classification", "project_id": 1, "gold_rows": gold,
            "target_count": 3, "raw_rows": None, "failure_cluster": None,
            "target_class": "injection",
        }
        prompt = pb.build_prompt(ctx)
        # The JSON template inside the prompt must use double quotes
        # — single quotes around "injection" was the bug.
        self.assertIn('"label": "injection"', prompt)
        self.assertNotIn("'injection'}", prompt)
        # Belt-and-suspenders directive that calls out the bug class.
        self.assertIn("DOUBLE QUOTES", prompt)


class RefusalDetectionTests(unittest.TestCase):
    """LLMs with safety guardrails (Llama 3 family in particular) refuse
    on legitimate classifier-training data when the label name sounds
    alarming ('injection', 'phishing', 'toxicity'). When that happens
    the playbook landed a generic '0 accepted rows' error that blamed
    JSON parsing — useless because the fix isn't to tweak the parser,
    it's to switch models or pin a non-refusing one.

    These tests cover the new refusal detector + defensive-security
    system prompt + qwen-first model preference."""

    def test_refusal_detector_fires_on_short_apologetic_response(self):
        from app.services.synth_playbook_service import _looks_like_refusal
        self.assertTrue(_looks_like_refusal(
            "I cannot generate malicious or harmful examples. Can I help "
            "you with something else?"
        ))
        self.assertTrue(_looks_like_refusal(
            "I'm sorry, but as an AI language model I cannot comply."
        ))
        self.assertTrue(_looks_like_refusal(
            "As a language model, I'm unable to assist with that request."
        ))

    def test_refusal_detector_silent_on_valid_jsonl_output(self):
        from app.services.synth_playbook_service import _looks_like_refusal
        # JSONL containing the word 'cannot' shouldn't be misclassified.
        valid = '{"text": "\' OR 1=1--", "label": "injection"}\n{"text": "we cannot allow that", "label": "benign"}'
        self.assertFalse(_looks_like_refusal(valid))
        # Long natural-language response also shouldn't fire (the
        # detector only triggers on short, JSON-less output).
        long_text = "I cannot " * 200
        self.assertFalse(_looks_like_refusal(long_text))

    def test_refusal_detector_silent_on_empty_response(self):
        from app.services.synth_playbook_service import _looks_like_refusal
        self.assertFalse(_looks_like_refusal(""))

    def test_system_prompt_includes_defensive_security_framing(self):
        """Without this framing, Llama 3 refuses on every BrewSLM project
        whose label vocabulary triggers its guardrails (injection,
        phishing, toxicity, spam). The framing is honest — BrewSLM IS
        training a detector — and unblocks compliance."""
        from app.services.synth_playbook_service import _system_prompt_for_mode
        from app.services.synth_playbooks import SynthMode
        for mode in SynthMode:
            sp = _system_prompt_for_mode(mode)
            self.assertIn("defensive", sp.lower())
            self.assertIn("classifier", sp.lower())
            # Must explicitly tell the model not to refuse.
            self.assertTrue(
                "refuse" in sp.lower() or "moralis" in sp.lower(),
                f"mode {mode}: defensive prompt should discourage refusal",
            )

    def test_ollama_prefers_qwen_over_llama3(self):
        """Qwen 2.5 scales higher (14B/32B/72B) and refuses far less on
        legitimate security/abuse-detection training data. When both
        are installed, the auto-picker must reach for Qwen first."""
        from app.services.synth_backends.ollama import PREFERRED_MODEL_PATTERNS
        qwen_idx = next(
            i for i, p in enumerate(PREFERRED_MODEL_PATTERNS) if p == "qwen2.5"
        )
        llama_idx = next(
            i for i, p in enumerate(PREFERRED_MODEL_PATTERNS) if p == "llama3"
        )
        self.assertLess(
            qwen_idx, llama_idx,
            "qwen2.5 must come before llama3 in PREFERRED_MODEL_PATTERNS",
        )


class ClusterTargetedUnitTests(unittest.TestCase):
    def test_cluster_block_embedded_in_prompt_with_exemplars(self):
        pb = get_playbook("classification", SynthMode.CLUSTER_TARGETED)
        cluster = {
            "cluster_id": "cluster-2",
            "reason_code": "negation_missed",
            "output_pattern": "predicted positive when text contains 'not'",
            "failure_count": 12,
            "share_of_total": 0.35,
            "classifier_reason": "Model is missing the negation",
            "exemplars": [
                {"prompt": "this is not great", "prediction": "positive", "reference": "negative"},
                {"prompt": "I do not like it", "prediction": "positive", "reference": "negative"},
            ],
        }
        ctx = {
            "recipe_id": "classification",
            "project_id": 1,
            "gold_rows": [{"text": "good", "label": "positive"}, {"text": "bad", "label": "negative"}],
            "target_count": 5,
            "raw_rows": None,
            "failure_cluster": cluster,
            "target_class": None,
        }
        prompt = pb.build_prompt(ctx)
        # Cluster reason + pattern + classifier reason + an exemplar
        # should all appear in the prompt.
        self.assertIn("negation_missed", prompt)
        self.assertIn("missing the negation", prompt)
        self.assertIn("not great", prompt)
        # Validator stamps the cluster id into synth_source.
        canned = [{"text": "I cannot recommend this", "label": "negative"}]
        validated = pb.validate(canned, ctx)
        self.assertEqual(len(validated), 1)
        self.assertIn("cluster=cluster-2", validated[0]["synth_source"])


class DatasetPrepGateTests(unittest.TestCase):
    """The dataset_service JSONL loader must exclude pending synth rows."""

    def test_load_records_skips_pending_synth_by_default(self):
        # Hand-craft a synthetic.jsonl with mixed pending + accepted.
        tmp_dir = TEST_DATA_DIR / "gate_test"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / "synthetic.jsonl"
        path.write_text(
            json.dumps({"id": 1, "text": "accepted row", "review_status": "accepted"}) + "\n"
            + json.dumps({"id": 2, "text": "pending row", "review_status": "pending"}) + "\n"
            + json.dumps({"id": 3, "text": "no field at all"}) + "\n",
            encoding="utf-8",
        )
        # Default read → pending row skipped.
        records = _load_records_from_file(path)
        self.assertEqual(len(records), 2)
        ids = {r["id"] for r in records}
        self.assertEqual(ids, {1, 3})
        # Explicit include_pending_synth → all 3 rows.
        with_pending = _load_records_from_file(path, include_pending_synth=True)
        self.assertEqual(len(with_pending), 3)

    def test_non_synth_files_unaffected_by_filter(self):
        # Gold rows / raw rows don't have review_status — filter is no-op.
        tmp_dir = TEST_DATA_DIR / "gold_unaffected"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / "gold.jsonl"
        path.write_text(
            json.dumps({"id": 1, "question": "Q1", "answer": "A1"}) + "\n"
            + json.dumps({"id": 2, "question": "Q2", "answer": "A2"}) + "\n",
            encoding="utf-8",
        )
        records = _load_records_from_file(path)
        self.assertEqual(len(records), 2)


class ReviewQueueIntegrationTests(unittest.TestCase):
    """End-to-end via TestClient: instantiate a template, manually seed
    a synthetic.jsonl with pending rows, then exercise the review queue
    list + bulk-update endpoints."""

    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for p in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        settings.AUTH_ENABLED = cls._prev_auth_enabled
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for p in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()

    def _instantiate_template(self, slug: str, name: str) -> dict:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def _seed_synth_rows(self, project_id: int, rows: list[dict]) -> None:
        """Append rows directly to synthetic.jsonl. The list/update
        endpoints read the file as the source of truth, so we don't
        need a Dataset row to exist — bulk_update just won't bump
        record_count when there's no row to update, which doesn't
        affect test correctness."""
        path = settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / "synthetic.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_review_queue_lists_pending_rows_grouped_by_source(self):
        project = self._instantiate_template("ticket-router", "Queue List Test")
        pid = project["id"]
        # Hand-seed 5 pending rows across 2 sources + 1 accepted.
        self._seed_synth_rows(pid, [
            {"id": 1, "text": "a", "label": "billing", "synth_source": "playbook:classification:positives_paraphrase", "synth_confidence": 0.9, "review_status": "pending"},
            {"id": 2, "text": "b", "label": "billing", "synth_source": "playbook:classification:positives_paraphrase", "synth_confidence": 0.85, "review_status": "pending"},
            {"id": 3, "text": "c", "label": "technical", "synth_source": "playbook:classification:hard_negatives:vs=billing", "synth_confidence": 0.95, "review_status": "pending"},
            {"id": 4, "text": "d", "label": "billing", "synth_source": "playbook:classification:positives_paraphrase", "synth_confidence": 0.7, "review_status": "accepted"},
        ])
        resp = self.client.get(f"/api/projects/{pid}/synthetic/review-queue")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["total_pending"], 3)
        self.assertEqual(payload["total_accepted"], 1)
        # 2 pending source groups.
        sources = {g["synth_source"] for g in payload["groups"]}
        self.assertEqual(sources, {
            "playbook:classification:positives_paraphrase",
            "playbook:classification:hard_negatives:vs=billing",
        })
        paraphrase_group = next(g for g in payload["groups"] if "positives_paraphrase" in g["synth_source"])
        self.assertEqual(paraphrase_group["count"], 2)
        # The accepted row surfaces in accepted_groups, not groups.
        accepted_sources = {g["synth_source"] for g in payload["accepted_groups"]}
        self.assertIn("playbook:classification:positives_paraphrase", accepted_sources)
        self.assertEqual(
            sum(g["count"] for g in payload["accepted_groups"]),
            1,
        )

    def test_review_queue_rows_sorted_by_confidence_ascending(self):
        # Epic E — the most uncertain rows (lowest synth_confidence) surface
        # first within a group so a reviewer's attention lands where it matters.
        # Rows with no confidence trail.
        project = self._instantiate_template("ticket-router", "Queue Confidence Sort")
        pid = project["id"]
        src = "playbook:classification:positives_paraphrase"
        self._seed_synth_rows(pid, [
            {"id": 1, "text": "high", "label": "billing", "synth_source": src, "synth_confidence": 0.95, "review_status": "pending"},
            {"id": 2, "text": "low", "label": "billing", "synth_source": src, "synth_confidence": 0.30, "review_status": "pending"},
            {"id": 3, "text": "none", "label": "billing", "synth_source": src, "review_status": "pending"},  # no confidence → trails
            {"id": 4, "text": "mid", "label": "billing", "synth_source": src, "synth_confidence": 0.60, "review_status": "pending"},
        ])
        resp = self.client.get(f"/api/projects/{pid}/synthetic/review-queue")
        self.assertEqual(resp.status_code, 200, resp.text)
        group = next(g for g in resp.json()["groups"] if g["synth_source"] == src)
        ids_in_order = [r["id"] for r in group["rows"]]
        # 0.30 < 0.60 < 0.95 < (no confidence).
        self.assertEqual(ids_in_order, [2, 4, 1, 3])

    def test_review_queue_group_by_class(self):
        # Epic E — ?group_by=class buckets pending rows by their label instead
        # of synth_source, so a reviewer can sweep one class at a time.
        project = self._instantiate_template("ticket-router", "Queue Group By Class")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 1, "text": "a", "label": "billing", "synth_source": "playbook:x", "synth_confidence": 0.9, "review_status": "pending"},
            {"id": 2, "text": "b", "label": "billing", "synth_source": "playbook:y", "synth_confidence": 0.8, "review_status": "pending"},
            {"id": 3, "text": "c", "label": "technical", "synth_source": "playbook:x", "synth_confidence": 0.7, "review_status": "pending"},
            {"id": 4, "text": "d", "synth_source": "playbook:x", "synth_confidence": 0.6, "review_status": "pending"},  # no label
        ])
        resp = self.client.get(
            f"/api/projects/{pid}/synthetic/review-queue?group_by=class"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["group_by"], "class")
        groups = {g["synth_source"]: g for g in payload["groups"]}
        # Grouped by label; the unlabeled row falls into "(unlabeled)".
        self.assertEqual(set(groups), {"billing", "technical", "(unlabeled)"})
        self.assertEqual(groups["billing"]["count"], 2)
        # Default (no param) still groups by source.
        src_resp = self.client.get(f"/api/projects/{pid}/synthetic/review-queue")
        self.assertEqual(src_resp.json()["group_by"], "source")
        src_groups = {g["synth_source"] for g in src_resp.json()["groups"]}
        self.assertEqual(src_groups, {"playbook:x", "playbook:y"})

    def test_review_queue_total_rows_includes_every_row_regardless_of_status(self):
        """total_rows is the whole-file count — the user's anchor
        when they ask 'how many synth rows do I have?'."""
        project = self._instantiate_template("ticket-router", "Queue Total Rows Test")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 1, "text": "p1", "label": "billing", "synth_source": "playbook:classification:positives_paraphrase", "synth_confidence": 0.9, "review_status": "pending"},
            {"id": 2, "text": "p2", "label": "billing", "synth_source": "playbook:classification:positives_paraphrase", "synth_confidence": 0.85, "review_status": "pending"},
            {"id": 3, "text": "a1", "label": "billing", "synth_source": "playbook:classification:positives_paraphrase", "synth_confidence": 0.95, "review_status": "accepted"},
            # Legacy row — no review_status field, has the legacy `source` instead.
            {"id": 4, "question": "Q1", "answer": "A1", "source": "teacher_model", "model": "llama3", "status": "accepted"},
        ])
        resp = self.client.get(f"/api/projects/{pid}/synthetic/review-queue")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["total_rows"], 4)
        self.assertEqual(payload["total_pending"], 2)
        self.assertEqual(payload["total_accepted"], 2)  # 1 explicit + 1 legacy

    def test_review_queue_labels_legacy_rows_as_legacy_source(self):
        """Legacy rows from the pre-Epic-2a flow have no `synth_source`
        but do have a legacy `source` field. The review queue should
        surface them as ``legacy:<source>`` instead of the opaque
        ``playbook:unknown`` placeholder."""
        project = self._instantiate_template("policy-qa-style", "Legacy Source Label Test")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 1, "question": "Q1", "answer": "A1", "source": "teacher_model", "model": "llama3", "status": "accepted"},
            {"id": 2, "question": "Q2", "answer": "A2", "source": "demo_heuristic", "status": "accepted"},
            {"id": 3, "question": "Q3", "answer": "A3", "status": "accepted"},  # no source at all
        ])
        resp = self.client.get(f"/api/projects/{pid}/synthetic/review-queue")
        payload = resp.json()
        accepted_sources = {g["synth_source"] for g in payload["accepted_groups"]}
        self.assertIn("legacy:teacher_model", accepted_sources)
        self.assertIn("legacy:demo_heuristic", accepted_sources)
        self.assertIn("legacy:manual", accepted_sources)
        # The placeholder "playbook:unknown" must NOT leak.
        self.assertNotIn("playbook:unknown", accepted_sources)

    def test_review_queue_caps_accepted_rows_per_group(self):
        """Legacy buckets can have thousands of rows. The list endpoint
        truncates each accepted group's `rows` array to 25 + sets a
        `truncated` flag, while `count` keeps the full total. Pending
        rows are NOT capped (the queue needs every row for accept/reject)."""
        project = self._instantiate_template("email-chat-tone", "Cap Test")
        pid = project["id"]
        # Seed 30 legacy accepted rows (no review_status).
        self._seed_synth_rows(pid, [
            {"id": i, "question": f"Q{i}", "answer": f"A{i}", "source": "teacher_model", "status": "accepted"}
            for i in range(1, 31)
        ])
        resp = self.client.get(f"/api/projects/{pid}/synthetic/review-queue")
        payload = resp.json()
        self.assertEqual(payload["total_accepted"], 30)
        group = payload["accepted_groups"][0]
        self.assertEqual(group["count"], 30)
        self.assertEqual(len(group["rows"]), 25)
        self.assertTrue(group["truncated"])

    def test_review_queue_surfaces_accepted_only_when_no_pending_left(self):
        """After all pending rows are accepted, the list endpoint
        should still surface the accepted rows so the user can see
        what's queued for training. Closes the 'where do my approved
        rows show up?' gap."""
        project = self._instantiate_template("policy-qa-style", "Queue Accepted-Only Test")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 1, "question": "Q1", "answer": "A1", "synth_source": "playbook:qa-sft:positives_paraphrase", "synth_confidence": 0.9, "review_status": "accepted"},
            {"id": 2, "question": "Q2", "answer": "A2", "synth_source": "playbook:qa-sft:positives_paraphrase", "synth_confidence": 0.85, "review_status": "accepted"},
        ])
        resp = self.client.get(f"/api/projects/{pid}/synthetic/review-queue")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["total_pending"], 0)
        self.assertEqual(payload["total_accepted"], 2)
        self.assertEqual(payload["groups"], [])
        self.assertEqual(len(payload["accepted_groups"]), 1)
        self.assertEqual(payload["accepted_groups"][0]["count"], 2)

    def test_bulk_accept_flips_pending_to_accepted(self):
        project = self._instantiate_template("log-triage", "Queue Accept Test")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 1, "log_line": "x", "label": "info", "synth_source": "playbook:classification:positives_paraphrase", "synth_confidence": 0.9, "review_status": "pending"},
            {"id": 2, "log_line": "y", "label": "info", "synth_source": "playbook:classification:positives_paraphrase", "synth_confidence": 0.85, "review_status": "pending"},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/bulk-update",
            json={"row_ids": [1], "action": "accept"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["rejected"], 0)
        self.assertEqual(payload["total_remaining_pending"], 1)

        # Verify on disk.
        path = settings.DATA_DIR / "projects" / str(pid) / "synthetic" / "synthetic.jsonl"
        with path.open() as f:
            rows = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(rows), 2)
        statuses = {r["id"]: r["review_status"] for r in rows}
        self.assertEqual(statuses[1], "accepted")
        self.assertEqual(statuses[2], "pending")

    def test_bulk_reject_soft_marks_rows_keeps_them_on_disk(self):
        # Arc 5 — soft-reject. Rejected rows used to be physically
        # deleted; now they stay on disk with
        # ``review_status="rejected"`` so the user can review them,
        # bulk-purge by reason, or recover (future feature). Project
        # preference "rejected rows are selectable + bulk-droppable"
        # required this — vanishing rows aren't selectable.
        project = self._instantiate_template("policy-qa-style", "Queue Reject Test")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 1, "question": "Q1", "answer": "A1", "synth_source": "playbook:qa-sft:positives_paraphrase", "synth_confidence": 0.8, "review_status": "pending"},
            {"id": 2, "question": "Q2", "answer": "A2", "synth_source": "playbook:qa-sft:positives_paraphrase", "synth_confidence": 0.75, "review_status": "pending"},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/bulk-update",
            json={"row_ids": [1, 2], "action": "reject"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["rejected"], 2)
        # File retains the rows but their status flipped to "rejected".
        path = settings.DATA_DIR / "projects" / str(pid) / "synthetic" / "synthetic.jsonl"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["review_status"], "rejected")

    def test_bulk_reject_stamps_reject_reason_when_provided(self):
        # Arc 5 — the new reject_reason field gets stamped on each
        # rejected row. UI uses this to group + filter the Rejected
        # section ("show me just the 'duplicate' rejects").
        project = self._instantiate_template("policy-qa-style", "Queue Reject Reason Test")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 10, "question": "Q", "answer": "A", "synth_source": "playbook:qa-sft", "synth_confidence": 0.7, "review_status": "pending"},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/bulk-update",
            json={"row_ids": [10], "action": "reject", "reject_reason": "duplicate"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        path = settings.DATA_DIR / "projects" / str(pid) / "synthetic" / "synthetic.jsonl"
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(rows[0]["review_status"], "rejected")
        self.assertEqual(rows[0]["reject_reason"], "duplicate")

    def test_list_review_queue_returns_rejected_groups(self):
        # Rejected rows surface in their own ``rejected_groups``
        # bucket alongside the existing pending ``groups`` +
        # accepted_groups. ``total_rejected`` counter is new too.
        project = self._instantiate_template("policy-qa-style", "Queue Rejected Groups")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 1, "question": "Q1", "answer": "A1", "synth_source": "playbook:qa-sft", "synth_confidence": 0.9, "review_status": "pending"},
            {"id": 2, "question": "Q2", "answer": "A2", "synth_source": "playbook:qa-sft", "synth_confidence": 0.6, "review_status": "rejected", "reject_reason": "low_confidence"},
            {"id": 3, "question": "Q3", "answer": "A3", "synth_source": "playbook:qa-sft", "synth_confidence": 0.65, "review_status": "rejected", "reject_reason": "duplicate"},
        ])
        resp = self.client.get(f"/api/projects/{pid}/synthetic/review-queue")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["total_pending"], 1)
        self.assertEqual(payload["total_rejected"], 2)
        # Rejected rows surface under their synth_source group with
        # the same row shape (preview + payload) as pending rows.
        self.assertEqual(len(payload["rejected_groups"]), 1)
        rejected_group = payload["rejected_groups"][0]
        self.assertEqual(rejected_group["synth_source"], "playbook:qa-sft")
        self.assertEqual(rejected_group["count"], 2)

    def test_purge_rejected_endpoint_removes_all_when_no_reasons_filter(self):
        # Purge with no reasons filter → all rejected rows drop.
        # Pending + accepted rows untouched.
        project = self._instantiate_template("policy-qa-style", "Queue Purge All")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 1, "question": "Q1", "answer": "A1", "synth_source": "p", "synth_confidence": 0.9, "review_status": "pending"},
            {"id": 2, "question": "Q2", "answer": "A2", "synth_source": "p", "synth_confidence": 0.6, "review_status": "rejected", "reject_reason": "duplicate"},
            {"id": 3, "question": "Q3", "answer": "A3", "synth_source": "p", "synth_confidence": 0.7, "review_status": "rejected", "reject_reason": "low_confidence"},
            {"id": 4, "question": "Q4", "answer": "A4", "synth_source": "p", "synth_confidence": 0.95, "review_status": "accepted"},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/purge",
            json={},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["purged"], 2)
        self.assertEqual(payload["retained"], 2)
        # Disk-side: only pending + accepted rows survive.
        path = settings.DATA_DIR / "projects" / str(pid) / "synthetic" / "synthetic.jsonl"
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        statuses = {row["review_status"] for row in rows}
        self.assertEqual(statuses, {"pending", "accepted"})

    def test_purge_rejected_endpoint_filters_by_reasons(self):
        # Reason cohort filter: only ``duplicate`` rows go. The
        # ``low_confidence`` rejected row stays on disk.
        project = self._instantiate_template("policy-qa-style", "Queue Purge Filtered")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 1, "question": "Q1", "answer": "A1", "synth_source": "p", "synth_confidence": 0.6, "review_status": "rejected", "reject_reason": "duplicate"},
            {"id": 2, "question": "Q2", "answer": "A2", "synth_source": "p", "synth_confidence": 0.6, "review_status": "rejected", "reject_reason": "duplicate"},
            {"id": 3, "question": "Q3", "answer": "A3", "synth_source": "p", "synth_confidence": 0.7, "review_status": "rejected", "reject_reason": "low_confidence"},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/purge",
            json={"reasons": ["duplicate"]},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["purged"], 2)
        self.assertEqual(payload["retained"], 1)
        path = settings.DATA_DIR / "projects" / str(pid) / "synthetic" / "synthetic.jsonl"
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        # Only the low_confidence row remains.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reject_reason"], "low_confidence")

    def test_purge_rejected_endpoint_is_idempotent_when_nothing_rejected(self):
        # No rejected rows → purge is a no-op + returns 0/0.
        project = self._instantiate_template("policy-qa-style", "Queue Purge Idempotent")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 1, "question": "Q", "answer": "A", "synth_source": "p", "synth_confidence": 0.9, "review_status": "pending"},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/purge",
            json={},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["purged"], 0)

    def test_bulk_update_handles_unknown_action(self):
        project = self._instantiate_template("email-chat-tone", "Queue Bad Action")
        pid = project["id"]
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/bulk-update",
            json={"row_ids": [1], "action": "delete"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_bulk_update_by_source_accepts_only_the_groups_pending_rows(self):
        # Epic E — one-click "Accept all (N)" on a Data Studio pending group.
        # Two sources; accepting one source's group must flip ONLY its pending
        # rows, leaving the other source + an already-accepted row untouched.
        project = self._instantiate_template("ticket-router", "Queue By-Source Accept")
        pid = project["id"]
        src_a = "playbook:classification:positives_paraphrase"
        src_b = "playbook:classification:hard_negatives:vs=billing"
        self._seed_synth_rows(pid, [
            {"id": 1, "text": "a", "label": "billing", "synth_source": src_a, "synth_confidence": 0.9, "review_status": "pending"},
            {"id": 2, "text": "b", "label": "billing", "synth_source": src_a, "synth_confidence": 0.8, "review_status": "pending"},
            {"id": 3, "text": "c", "label": "technical", "synth_source": src_b, "synth_confidence": 0.95, "review_status": "pending"},
            {"id": 4, "text": "d", "label": "billing", "synth_source": src_a, "synth_confidence": 0.7, "review_status": "accepted"},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/bulk-update-by-source",
            json={"source": src_a, "action": "accept"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["source"], src_a)
        self.assertEqual(payload["matched"], 2)
        self.assertEqual(payload["accepted"], 2)
        # src_b's pending row is the only one left pending.
        self.assertEqual(payload["total_remaining_pending"], 1)

        path = settings.DATA_DIR / "projects" / str(pid) / "synthetic" / "synthetic.jsonl"
        with path.open() as f:
            rows = {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}
        self.assertEqual(rows[1]["review_status"], "accepted")
        self.assertEqual(rows[2]["review_status"], "accepted")
        self.assertEqual(rows[3]["review_status"], "pending")  # other source, untouched
        self.assertEqual(rows[4]["review_status"], "accepted")  # already accepted

    def test_bulk_update_by_source_reject_stamps_reason(self):
        project = self._instantiate_template("policy-qa-style", "Queue By-Source Reject")
        pid = project["id"]
        src = "playbook:qa-sft:positives_paraphrase"
        self._seed_synth_rows(pid, [
            {"id": 1, "question": "Q1", "answer": "A1", "synth_source": src, "synth_confidence": 0.6, "review_status": "pending"},
            {"id": 2, "question": "Q2", "answer": "A2", "synth_source": src, "synth_confidence": 0.55, "review_status": "pending"},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/bulk-update-by-source",
            json={"source": src, "action": "reject", "reject_reason": "low_confidence"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["rejected"], 2)
        path = settings.DATA_DIR / "projects" / str(pid) / "synthetic" / "synthetic.jsonl"
        with path.open() as f:
            rows = [json.loads(l) for l in f if l.strip()]
        for row in rows:
            self.assertEqual(row["review_status"], "rejected")
            self.assertEqual(row["reject_reason"], "low_confidence")

    def test_bulk_update_by_source_unknown_source_is_a_noop(self):
        project = self._instantiate_template("ticket-router", "Queue By-Source Noop")
        pid = project["id"]
        self._seed_synth_rows(pid, [
            {"id": 1, "text": "a", "label": "x", "synth_source": "real:source", "synth_confidence": 0.9, "review_status": "pending"},
        ])
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/bulk-update-by-source",
            json={"source": "does:not:exist", "action": "accept"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["matched"], 0)
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(payload["total_remaining_pending"], 1)

    def test_bulk_update_by_source_rejects_unknown_action(self):
        project = self._instantiate_template("email-chat-tone", "Queue By-Source Bad Action")
        pid = project["id"]
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/bulk-update-by-source",
            json={"source": "s", "action": "delete"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_data_studio_pending_groups_carry_synth_source_for_actions(self):
        # The Data Studio review-queue panel needs the raw synth_source on each
        # synthetic-pending group so its Accept/Reject-all buttons can target it.
        project = self._instantiate_template("ticket-router", "Queue DS Source Key")
        pid = project["id"]
        src = "playbook:classification:positives_paraphrase"
        self._seed_synth_rows(pid, [
            {"id": 1, "text": "a", "label": "billing", "synth_source": src, "synth_confidence": 0.9, "review_status": "pending"},
        ])
        resp = self.client.get(f"/api/projects/{pid}/data-studio/review-queue")
        self.assertEqual(resp.status_code, 200, resp.text)
        by_source = resp.json()["groupings"]["by_source"]
        pending = [
            g for g in by_source
            if g["kind"] == "synthetic" and g["status"] == "pending"
        ]
        self.assertTrue(pending)
        self.assertEqual(pending[0]["synth_source"], src)
        # Accepted groups don't carry the actionable key.
        accepted = [
            g for g in by_source
            if g["kind"] == "synthetic" and g["status"] == "accepted"
        ]
        for g in accepted:
            self.assertNotIn("synth_source", g)

    def test_run_playbook_endpoint_wraps_backend_failure_as_503(self):
        """A flaky LLM backend that throws mid-generation must produce
        a clean 503 + readable message, not a 500 / "network error"
        on the frontend. Regression test for the 2026-05-23 fix."""
        project = self._instantiate_template("ticket-router", "Run Backend-Flake Test")
        pid = project["id"]

        class _FlakyBackend:
            name = "flaky"

            @classmethod
            def is_available(cls):  # pragma: no cover
                return True

            def describe(self):  # pragma: no cover
                return "flaky:test"

            async def complete(self, *args, **kwargs):
                # Simulate an Ollama timeout / connection drop that
                # bubbles up as an unwrapped exception (the bug we
                # just fixed in OllamaBackend).
                raise RuntimeError("simulated Ollama transport drop")

        from unittest.mock import patch
        from app.services import synth_playbook_service

        with patch.object(synth_playbook_service, "pick_backend", return_value=_FlakyBackend()):
            resp = self.client.post(
                f"/api/projects/{pid}/synthetic/run-playbook",
                json={"mode": "positives_paraphrase", "target_count": 3},
            )
        # 503 (Service Unavailable), not 500.
        self.assertEqual(resp.status_code, 503, resp.text)
        # The error message should include the underlying type so
        # the user can diagnose.
        self.assertIn("RuntimeError", resp.text)
        self.assertIn("simulated Ollama transport drop", resp.text)

    def test_augment_cluster_endpoint_returns_400_for_unknown_cluster(self):
        # Need an eval result to look up — easiest path is to make
        # an experiment + eval row directly. But this 400 case only
        # needs the lookup to fail, which fails earlier at the
        # eval_result level. We use a non-existent eval id.
        project = self._instantiate_template("ticket-router", "Aug 404 Test")
        pid = project["id"]
        resp = self.client.post(
            f"/api/projects/{pid}/evaluation/99999/clusters/cluster-1/augment",
            params={"target_count": 5},
        )
        # eval_result_not_found → 404 via the ValueError("not found") match.
        self.assertIn(resp.status_code, (400, 404))


if __name__ == "__main__":
    unittest.main()
