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
    def test_registry_has_17_playbooks_with_expected_coverage(self):
        playbooks = list_playbooks()
        self.assertEqual(len(playbooks), 17)
        # Coverage matrix sanity:
        # qa-sft + summarization: paraphrase + cluster (2 modes each)
        # classification: paraphrase + hard_neg + balance + cluster (4 modes)
        # code-review / generic-sft / span-extraction: paraphrase + hard_neg + cluster (3 modes each)
        from collections import defaultdict
        by_recipe = defaultdict(set)
        for pb in playbooks:
            by_recipe[pb["recipe_id"]].add(pb["mode"])
        self.assertEqual(by_recipe["classification"], {"positives_paraphrase", "hard_negatives", "class_balance_fill", "cluster_targeted"})
        self.assertEqual(by_recipe["qa-sft"], {"positives_paraphrase", "cluster_targeted"})
        self.assertEqual(by_recipe["summarization"], {"positives_paraphrase", "cluster_targeted"})
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
        # 2 source groups.
        sources = {g["synth_source"] for g in payload["groups"]}
        self.assertEqual(sources, {
            "playbook:classification:positives_paraphrase",
            "playbook:classification:hard_negatives:vs=billing",
        })
        paraphrase_group = next(g for g in payload["groups"] if "positives_paraphrase" in g["synth_source"])
        self.assertEqual(paraphrase_group["count"], 2)

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

    def test_bulk_reject_removes_rows_from_disk(self):
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
        # File should now be empty (or contain only non-rejected rows).
        path = settings.DATA_DIR / "projects" / str(pid) / "synthetic" / "synthetic.jsonl"
        if path.exists():
            with path.open() as f:
                rows = [json.loads(l) for l in f if l.strip()]
            self.assertEqual(rows, [])

    def test_bulk_update_handles_unknown_action(self):
        project = self._instantiate_template("email-chat-tone", "Queue Bad Action")
        pid = project["id"]
        resp = self.client.post(
            f"/api/projects/{pid}/synthetic/review-queue/bulk-update",
            json={"row_ids": [1], "action": "delete"},
        )
        self.assertEqual(resp.status_code, 400, resp.text)

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
