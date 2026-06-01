"""Tests for the synthetic-data playbook framework
(USER-SUCCESS Epic 2).

11 backend tests covering:
  - Playbook lookup for each of the 6 recipes returns a valid class
  - QA paraphrase preserves the answer text verbatim
  - Classification paraphrase rejects rows whose label isn't a known class
  - Span-extraction paraphrase validates declared offsets against text
  - Generic-sft paraphrase drops malformed rows
  - JSONL parser handles markdown fences + bare lines + extra prose
  - run_playbook persists accepted rows into the synthetic dataset
  - run_playbook raises 400-ish ValueErrors for the bad-input cases
  - Backend picker falls back when first choice is unavailable
  - SynthBackendError when nothing's available
  - GET /playbooks lists modes filtered by project recipe
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

TEST_DB_PATH = Path(__file__).resolve().parent / "synth_playbook_test.db"
TEST_DATA_DIR = Path(__file__).resolve().parent / "synth_playbook_data"

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"
os.environ["DATA_DIR"] = TEST_DATA_DIR.as_posix()
os.environ["DEBUG"] = "false"
os.environ["DB_REQUIRE_ALEMBIC_HEAD"] = "false"

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.synth_backends import (
    OllamaBackend,
    SynthBackend,
    SynthBackendError,
    TeacherModelBackend,
    pick_backend,
)
from app.services.synth_playbooks import (
    SynthMode,
    get_playbook,
    list_playbooks,
)
from app.services.synth_playbooks.base import parse_jsonl_lines


# ─────────────────────────────────────────────────────────────────────
# Fake backend for tests — returns a canned LLM output.
# ─────────────────────────────────────────────────────────────────────


class _CannedBackend:
    name = "canned"

    def __init__(self, response: str):
        self._response = response
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    @classmethod
    def is_available(cls) -> bool:  # pragma: no cover
        return True

    def describe(self) -> str:
        return "canned:test"

    async def complete(self, prompt: str, *, system_prompt: str | None = None, max_tokens: int = 1024, temperature: float = 0.7, response_schema: dict | None = None) -> str:
        self.last_prompt = prompt
        self.last_system = system_prompt
        self.last_response_schema = response_schema
        return self._response


# ─────────────────────────────────────────────────────────────────────
# Unit tests — pure-function pieces. No DB, no API client.
# ─────────────────────────────────────────────────────────────────────


class PlaybookUnitTests(unittest.TestCase):
    def test_playbook_lookup_for_each_recipe_returns_valid_class(self):
        # Roadmap spec: each of the 6 recipes must have a paraphrase
        # playbook registered. This is the calibration anchor that
        # confirms the v1 coverage matrix is complete.
        for recipe_id in (
            "qa-sft",
            "classification",
            "span-extraction",
            "summarization",
            "code-review",
            "generic-sft",
        ):
            with self.subTest(recipe=recipe_id):
                pb = get_playbook(recipe_id, SynthMode.POSITIVES_PARAPHRASE)
                self.assertIsNotNone(pb, f"missing paraphrase playbook for {recipe_id}")
                self.assertEqual(pb.recipe_id, recipe_id)
                self.assertEqual(pb.mode, SynthMode.POSITIVES_PARAPHRASE)

        # Catalog read covers all registered playbooks; every recipe
        # has at least POSITIVES_PARAPHRASE.
        catalog = list_playbooks()
        self.assertGreaterEqual(len(catalog), 6)
        modes_present = {p["mode"] for p in catalog}
        self.assertIn("positives_paraphrase", modes_present)
        # Every recipe must have positives_paraphrase available.
        recipes_with_paraphrase = {
            p["recipe_id"] for p in catalog if p["mode"] == "positives_paraphrase"
        }
        self.assertEqual(len(recipes_with_paraphrase), 6)

    def test_qa_sft_paraphrase_preserves_answer_text(self):
        # The QA paraphrase playbook must drop confidence when the
        # generated row's answer differs from the source row's answer.
        pb = get_playbook("qa-sft", SynthMode.POSITIVES_PARAPHRASE)
        gold_rows = [{"question": "How do I reset my password?", "answer": "Visit Settings → Security."}]
        ctx = {
            "recipe_id": "qa-sft",
            "project_id": 1,
            "gold_rows": gold_rows,
            "target_count": 5,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }
        canned_rows = [
            {"question": "What's the password reset process?", "answer": "Visit Settings → Security."},
            {"question": "Where do I change my password?", "answer": "Click forgot password."},  # answer drifted
        ]
        validated = pb.validate(canned_rows, ctx)
        # Both rows accepted (paraphrasing the answer doesn't drop the
        # row outright), but the second has lower confidence.
        self.assertEqual(len(validated), 2)
        preserved = next(v for v in validated if v["payload"]["answer"] == "Visit Settings → Security.")
        drifted = next(v for v in validated if v["payload"]["answer"] == "Click forgot password.")
        self.assertEqual(preserved["synth_confidence"], 1.0)
        self.assertLess(drifted["synth_confidence"], 1.0)

    def test_classification_paraphrase_rejects_unknown_label(self):
        pb = get_playbook("classification", SynthMode.POSITIVES_PARAPHRASE)
        gold_rows = [
            {"text": "I want a refund", "label": "billing"},
            {"text": "App is crashing", "label": "technical"},
        ]
        ctx = {
            "recipe_id": "classification",
            "project_id": 1,
            "gold_rows": gold_rows,
            "target_count": 5,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }
        canned_rows = [
            {"text": "Please refund me", "label": "billing"},
            {"text": "App keeps freezing", "label": "technical"},
            {"text": "Invented label row", "label": "fraud"},  # label not in known classes
        ]
        validated = pb.validate(canned_rows, ctx)
        self.assertEqual(len(validated), 3)
        fraud = next(v for v in validated if v["payload"]["label"] == "fraud")
        valid = next(v for v in validated if v["payload"]["label"] == "billing")
        # Unknown labels get a sharp confidence penalty.
        self.assertLess(fraud["synth_confidence"], 0.5)
        self.assertEqual(valid["synth_confidence"], 1.0)

    def test_span_extraction_paraphrase_validates_offsets(self):
        # Critical test: when the LLM declares spans, the text at
        # the declared offsets must equal the span's text field.
        pb = get_playbook("span-extraction", SynthMode.POSITIVES_PARAPHRASE)
        gold_rows = [{
            "text": "John Doe lives at 123 Main St.",
            "spans": [{"type": "name", "start": 0, "end": 8, "text": "John Doe"}],
        }]
        ctx = {
            "recipe_id": "span-extraction",
            "project_id": 1,
            "gold_rows": gold_rows,
            "target_count": 4,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }
        canned_rows = [
            # Row 1: valid — offsets match.
            {
                "text": "Jane Smith works downtown.",
                "spans": [{"type": "name", "start": 0, "end": 10, "text": "Jane Smith"}],
            },
            # Row 2: offsets lie — text[5:13] is not "Jane Smith"
            {
                "text": "Jane Smith works downtown.",
                "spans": [{"type": "name", "start": 5, "end": 13, "text": "Jane Smith"}],
            },
        ]
        validated = pb.validate(canned_rows, ctx)
        # Both rows produced output, but row 2's spans got dropped
        # (offsets didn't match) so its synth_confidence is lower.
        self.assertEqual(len(validated), 2)
        good = validated[0]
        bad = validated[1]
        self.assertEqual(len(good["payload"]["spans"]), 1)
        # Bad row's span list is empty because the offset-mismatched
        # span got dropped during validation.
        self.assertEqual(len(bad["payload"]["spans"]), 0)
        self.assertLess(bad["synth_confidence"], good["synth_confidence"])

    def test_generic_sft_paraphrase_drops_malformed_rows(self):
        pb = get_playbook("generic-sft", SynthMode.POSITIVES_PARAPHRASE)
        ctx = {
            "recipe_id": "generic-sft",
            "project_id": 1,
            "gold_rows": [{"prompt": "Hello", "completion": "Hi there"}],
            "target_count": 5,
            "raw_rows": None,
            "failure_cluster": None,
            "target_class": None,
        }
        canned_rows = [
            {"prompt": "Hey", "completion": "Hi there"},  # valid
            {"prompt": "", "completion": "Hi there"},  # empty prompt — drop
            {"prompt": "Greet me", "completion": ""},  # empty completion — drop
            {"completion": "Hi"},  # missing prompt — drop
            {"prompt": "x" * 5000, "completion": "Hi"},  # prompt too long — drop
        ]
        validated = pb.validate(canned_rows, ctx)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["payload"]["prompt"], "Hey")

    def test_jsonl_parser_handles_markdown_fences_and_extra_prose(self):
        # The LLM frequently wraps output in ```json fences and adds
        # explanatory prose. The parser must strip both.
        raw = """\
Sure, here are the paraphrases:

```json
{"question": "How do I reset?", "answer": "Visit settings."}
{"question": "Where do I reset?", "answer": "Visit settings."}
```

Hope that helps!
"""
        parsed = parse_jsonl_lines(raw)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["question"], "How do I reset?")

    def test_jsonl_parser_skips_malformed_lines(self):
        raw = (
            '{"question": "valid", "answer": "ok"}\n'
            'not valid json at all\n'
            '{"question": "also valid", "answer": "ok"}\n'
            '{broken json\n'
        )
        parsed = parse_jsonl_lines(raw)
        self.assertEqual(len(parsed), 2)

    def test_backend_picker_falls_back_when_first_unavailable(self):
        # Walk a custom 2-element registry: first unavailable, second
        # available. Picker should pick the second.
        class _Unavailable:
            name = "fake-unavailable"
            @classmethod
            def is_available(cls) -> bool:
                return False
            def describe(self) -> str:  # pragma: no cover
                return self.name

        class _Available:
            name = "fake-available"
            @classmethod
            def is_available(cls) -> bool:
                return True
            def describe(self) -> str:
                return self.name

        picked = pick_backend(None, registry=[_Unavailable, _Available])
        self.assertEqual(picked.name, "fake-available")

    def test_backend_picker_raises_when_nothing_available(self):
        class _Unavailable:
            name = "x"
            @classmethod
            def is_available(cls) -> bool:
                return False

        with self.assertRaises(SynthBackendError):
            pick_backend(None, registry=[_Unavailable])

    def test_backend_picker_supports_pinned_name(self):
        class _A:
            name = "alpha"
            @classmethod
            def is_available(cls) -> bool:
                return True
            def describe(self) -> str:
                return "alpha"
        class _B:
            name = "beta"
            @classmethod
            def is_available(cls) -> bool:
                return True
            def describe(self) -> str:
                return "beta"

        picked = pick_backend("beta", registry=[_A, _B])
        self.assertEqual(picked.name, "beta")
        # Unknown pin raises.
        with self.assertRaises(SynthBackendError):
            pick_backend("gamma", registry=[_A, _B])


# ─────────────────────────────────────────────────────────────────────
# Integration tests — full stack with a project template + canned LLM.
# ─────────────────────────────────────────────────────────────────────


class RunPlaybookIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth_enabled = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False

        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        if TEST_DATA_DIR.exists():
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
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
            for path in sorted(TEST_DATA_DIR.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()

    def _instantiate_template(self, slug: str, name: str) -> dict:
        resp = self.client.post(
            f"/api/project-templates/{slug}/instantiate",
            json={"project_name": name},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_run_playbook_persists_accepted_rows_into_synthetic_dataset(self):
        """End-to-end: instantiate a classification template,
        inject a canned backend, run the paraphrase playbook,
        verify rows land in the synthetic.jsonl file."""
        project = self._instantiate_template("ticket-router", "Synth E2E 1")
        pid = project["id"]

        # Patch the orchestrator's pick_backend to return our canned backend.
        canned_response = (
            '{"text": "Could I please get a refund?", "label": "billing"}\n'
            '{"text": "Please credit my account.", "label": "billing"}\n'
            '{"text": "Cancel and refund", "label": "billing"}\n'
        )
        canned = _CannedBackend(canned_response)

        from unittest.mock import patch
        from app.services import synth_playbook_service

        with patch.object(synth_playbook_service, "pick_backend", return_value=canned):
            resp = self.client.post(
                f"/api/projects/{pid}/synthetic/run-playbook",
                json={"mode": "positives_paraphrase", "target_count": 3},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        # 3 rows generated, all should pass validation (billing is a known class).
        self.assertEqual(len(payload["rows"]), 3)
        for row in payload["rows"]:
            self.assertEqual(row["payload"]["label"], "billing")
            self.assertGreater(row["synth_confidence"], 0.5)
            self.assertTrue(row["synth_source"].startswith("playbook:classification:"))
        self.assertEqual(payload["backend_used"], "canned:test")
        self.assertGreater(payload["elapsed_sec"], -0.01)  # >= 0
        self.assertGreater(len(payload["prompt_snippet"]), 20)

        # Verify the rows landed on disk in the synthetic.jsonl.
        synthetic_path = Path(settings.DATA_DIR) / "projects" / str(pid) / "synthetic" / "synthetic.jsonl"
        self.assertTrue(synthetic_path.exists())
        with synthetic_path.open() as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertGreaterEqual(len(lines), 3)
        # Provenance fields present.
        for line in lines[-3:]:
            self.assertEqual(line["review_status"], "pending")
            self.assertIn("synth_source", line)
            self.assertIn("synth_confidence", line)

    def test_run_playbook_400s_on_unknown_mode(self):
        project = self._instantiate_template("log-triage", "Synth E2E Bad Mode")
        resp = self.client.post(
            f"/api/projects/{project['id']}/synthetic/run-playbook",
            json={"mode": "doesnt_exist", "target_count": 5},
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_run_playbook_400s_on_unsupported_mode_for_recipe(self):
        project = self._instantiate_template("policy-qa-style", "Synth E2E Bad Combo")
        # HARD_NEGATIVES is registered as a SynthMode but no
        # playbook exists for it in v1 (Epic 2b).
        resp = self.client.post(
            f"/api/projects/{project['id']}/synthetic/run-playbook",
            json={"mode": "hard_negatives", "target_count": 5},
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("playbook", resp.text.lower())

    def test_list_playbooks_endpoint_filters_by_project_recipe(self):
        project = self._instantiate_template("ticket-router", "Synth Catalog Test")
        resp = self.client.get(f"/api/projects/{project['id']}/synthetic/playbooks")
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertEqual(payload["recipe_id"], "classification")
        # Classification recipe has the full set: paraphrase + hard_neg + balance + cluster.
        modes = {p["mode"] for p in payload["playbooks"]}
        self.assertEqual(
            modes,
            {"positives_paraphrase", "hard_negatives", "class_balance_fill", "cluster_targeted"},
        )
        # Every entry should be scoped to the project's recipe.
        for pb in payload["playbooks"]:
            self.assertEqual(pb["recipe_id"], "classification")
        # New contract: brief-driven + magic-create always populate
        # selected_recipe at create time, so legacy NULL is no longer
        # the common path. Flag is False when a recipe IS set.
        self.assertFalse(payload.get("recipe_required", False))

    def test_list_playbooks_endpoint_signals_recipe_required_on_legacy_null(self):
        """Legacy projects without ``selected_recipe`` (pre-dating the
        auto-apply-on-create fix) should land an empty playbook list +
        ``recipe_required=True`` so the UI can render a 'pick a recipe'
        CTA instead of a confusing dump of every playbook across every
        task shape. Simulates the legacy state by clearing the recipe
        from a freshly-instantiated template project."""
        import asyncio

        from app.database import async_session_factory
        from app.services.recipe_apply_service import clear_recipe_from_project

        project = self._instantiate_template(
            "ticket-router", "Synth Catalog Legacy NULL",
        )

        async def _clear() -> None:
            async with async_session_factory() as db:
                await clear_recipe_from_project(db, int(project["id"]))
                await db.commit()

        asyncio.run(_clear())

        resp = self.client.get(
            f"/api/projects/{project['id']}/synthetic/playbooks",
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        self.assertIsNone(payload["recipe_id"])
        self.assertTrue(payload.get("recipe_required"))
        # Empty list — not the full cross-task-shape catalog, which
        # was the pre-fix behavior that misled legacy-project users.
        self.assertEqual(payload["playbooks"], [])

    # ── Pre-flight dry-run + Ollama models endpoints ───────────────

    def test_dry_run_returns_ok_without_persisting_when_backend_complies(self):
        """A successful dry-run returns ``ok=True`` + the 1 generated
        row in the response, but DOES NOT write to synthetic.jsonl —
        that's the contract the frontend's pre-flight check relies on."""
        from unittest.mock import patch
        from app.services import synth_playbook_service

        project = self._instantiate_template("ticket-router", "Dry-run OK")
        pid = project["id"]
        canned = _CannedBackend(
            '{"text": "Refund my charge.", "label": "billing"}\n'
        )
        with patch.object(synth_playbook_service, "pick_backend", return_value=canned):
            resp = self.client.post(
                f"/api/projects/{pid}/synthetic/run-playbook/dry-run",
                json={"mode": "positives_paraphrase", "target_count": 5},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["accepted_count"], 1)
        self.assertFalse(body["refusal_detected"])
        self.assertGreater(len(body["raw_llm_snippet"]), 0)
        self.assertEqual(body["backend_used"], "canned:test")
        # No persistence: synthetic.jsonl is either missing or unchanged.
        synthetic_path = (
            Path(settings.DATA_DIR) / "projects" / str(pid) / "synthetic" / "synthetic.jsonl"
        )
        if synthetic_path.exists():
            with synthetic_path.open() as f:
                lines = [l for l in f if l.strip()]
            self.assertEqual(len(lines), 0, "dry-run must not write to synthetic.jsonl")

    def test_dry_run_target_count_is_always_1_regardless_of_request(self):
        """The frontend passes ``target_count=1`` explicitly, but even
        if a buggy caller sends 50, the server must clamp to 1 so the
        pre-flight stays fast."""
        from unittest.mock import patch
        from app.services import synth_playbook_service

        project = self._instantiate_template("ticket-router", "Dry-run Clamp")
        pid = project["id"]
        canned = _CannedBackend('{"text": "x", "label": "billing"}\n')
        with patch.object(synth_playbook_service, "pick_backend", return_value=canned):
            self.client.post(
                f"/api/projects/{pid}/synthetic/run-playbook/dry-run",
                json={"mode": "positives_paraphrase", "target_count": 50},
            )
        # The prompt the backend received must mention target=1.
        self.assertIn("1", canned.last_prompt or "")
        self.assertNotIn("50", (canned.last_prompt or ""))

    def test_dry_run_reports_refusal_with_200_not_500(self):
        """When the LLM refuses, the dry-run must return 200 with
        ``ok=False, refusal_detected=True`` so the frontend can render
        an inline error + Retry-with-Qwen button. Returning a 5xx
        would push the diagnostic into the notification bell instead
        of the inline panel."""
        from unittest.mock import patch
        from app.services import synth_playbook_service

        project = self._instantiate_template("ticket-router", "Dry-run Refusal")
        pid = project["id"]
        # Canonical Llama-style refusal — short, no JSON.
        canned = _CannedBackend(
            "I cannot generate malicious or harmful examples. Can I help "
            "you with something else?"
        )
        with patch.object(synth_playbook_service, "pick_backend", return_value=canned):
            resp = self.client.post(
                f"/api/projects/{pid}/synthetic/run-playbook/dry-run",
                json={"mode": "positives_paraphrase", "target_count": 1},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["accepted_count"], 0)
        self.assertTrue(body["refusal_detected"])
        # raw_llm_snippet should carry through so the panel can show
        # the user what the model actually said.
        self.assertIn("cannot generate", body["raw_llm_snippet"])

    def test_dry_run_unknown_mode_returns_400(self):
        project = self._instantiate_template("ticket-router", "Dry-run Bad Mode")
        resp = self.client.post(
            f"/api/projects/{project['id']}/synthetic/run-playbook/dry-run",
            json={"mode": "make-coffee", "target_count": 1},
        )
        self.assertEqual(resp.status_code, 400, resp.text)

    def test_cloud_models_endpoint_returns_three_providers_with_curated_models(self):
        """The synth panel renders a cloud picker even before any key
        is saved — the user needs to see what providers exist + what
        models each one offers BEFORE going to Project Settings to
        save a key. Endpoint contract: always 200, always 3 providers,
        always a non-empty curated models list per provider."""
        project = self._instantiate_template("ticket-router", "Cloud Models")
        pid = project["id"]
        resp = self.client.get(
            f"/api/projects/{pid}/synthetic/backends/cloud/models",
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        providers = {p["provider"] for p in body["providers"]}
        self.assertEqual(providers, {"openai", "anthropic", "deepseek"})
        for entry in body["providers"]:
            # Fresh project — no keys saved.
            self.assertFalse(entry["key_saved"])
            self.assertGreater(len(entry["models"]), 0)
            for m in entry["models"]:
                self.assertIn("id", m)
                self.assertIn("label", m)

    def test_cloud_backend_pin_without_saved_key_returns_402(self):
        """402 (payment required) is reserved for 'no API key saved' on
        cloud-pinned playbook runs. The frontend renders a 'save key
        first' affordance for 402; a 400 would show a generic toast.
        Critical that the status code is stable here."""
        project = self._instantiate_template("ticket-router", "Cloud No Key")
        pid = project["id"]
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/run-playbook/dry-run",
            json={
                "mode": "positives_paraphrase",
                "target_count": 1,
                "backend": "cloud:openai:gpt-4o-mini",
            },
        )
        self.assertEqual(resp.status_code, 402, resp.text)
        self.assertIn("API key", resp.json()["detail"])
        self.assertIn("openai", resp.json()["detail"])

    def test_malformed_cloud_pin_returns_400(self):
        """``cloud:something`` (only 2 colons-separated parts) is
        malformed — frontend bug, not user-actionable. 400 with a
        clear remediation string."""
        project = self._instantiate_template("ticket-router", "Cloud Malformed")
        pid = project["id"]
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/run-playbook/dry-run",
            json={
                "mode": "positives_paraphrase",
                "target_count": 1,
                "backend": "cloud:openai",
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("Malformed", resp.json()["detail"])

    def test_unknown_cloud_provider_returns_400(self):
        project = self._instantiate_template("ticket-router", "Cloud Unknown Provider")
        pid = project["id"]
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/run-playbook/dry-run",
            json={
                "mode": "positives_paraphrase",
                "target_count": 1,
                "backend": "cloud:gemini:flash",
            },
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("Unknown cloud provider", resp.json()["detail"])

    def test_cloud_backend_dispatches_to_provider_when_key_saved(self):
        """End-to-end: stash a fake OpenAI key in project secrets,
        mock the cloud_llm_service call, run a cloud-pinned dry-run,
        verify the CloudLlmBackend was constructed + the response
        round-tripped through the playbook parser."""
        import asyncio
        from unittest.mock import patch
        from app.database import async_session_factory
        from app.services.secret_service import upsert_project_secret
        from app.services.cloud_llm_service import CloudLlmResponse

        project = self._instantiate_template("ticket-router", "Cloud Dispatch")
        pid = project["id"]

        async def _save_key():
            async with async_session_factory() as db:
                await upsert_project_secret(
                    db, pid, "cloud_llm_openai", "api_key",
                    value="sk-test-fake",
                )
                await db.commit()
        asyncio.run(_save_key())

        canned_resp = CloudLlmResponse(
            content='{"text": "Refund please", "label": "billing"}\n',
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=20,
        )
        with patch(
            "app.services.synth_backends.cloud_llm.call_openai_chat",
            return_value=canned_resp,
        ) as mock_call:
            resp = self.client.post(
                f"/api/projects/{pid}/synthetic/run-playbook/dry-run",
                json={
                    "mode": "positives_paraphrase",
                    "target_count": 1,
                    "backend": "cloud:openai:gpt-4o-mini",
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["backend_used"], "cloud:openai:gpt-4o-mini")
        mock_call.assert_called_once()
        # The CloudLlmBackend forwarded the saved key + chosen model.
        kwargs = mock_call.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "sk-test-fake")
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        # force_json=False so the JSONL playbook prompt isn't
        # constrained into a single top-level JSON object.
        self.assertFalse(kwargs["force_json"])

    def test_ollama_models_endpoint_returns_structured_payload_when_daemon_up(self):
        """Smoke test against the actual local Ollama daemon if
        running. Tolerates 'daemon down' (returns ollama_available=False)
        because CI environments won't have Ollama installed."""
        project = self._instantiate_template("ticket-router", "Ollama Models")
        pid = project["id"]
        resp = self.client.get(
            f"/api/projects/{pid}/synthetic/backends/ollama/models",
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # Contract: response always has these keys regardless of
        # daemon state.
        self.assertIn("models", body)
        self.assertIn("default", body)
        self.assertIn("ollama_available", body)
        self.assertIsInstance(body["models"], list)
        # When daemon is up, each model entry carries the standard
        # set of fields the picker needs to render labels.
        if body["ollama_available"]:
            for m in body["models"]:
                self.assertIn("name", m)
                self.assertIn("parameter_size", m)


if __name__ == "__main__":
    unittest.main()
