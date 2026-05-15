"""Story 1.6 — promote labeled rows → training dataset.

Pins the contract that closes the Epic 1 annotation loop. Before this
story shipped, ``submit_label`` persisted into ``label_rows`` and that
was the end of the line — the trainer never read those rows.

Parameterized across the three label_types Story 1.1 supports so the
test suite catches a regression in any branch:

- Classification → SYNTHETIC dataset, ``{question, answer}`` shape
  where ``answer`` is JSON-encoded ``{label}``.
- Span → SYNTHETIC dataset, ``{question, answer}`` shape where
  ``answer`` is JSON-encoded ``{entities: [...]}``; spans without
  explicit ``text`` get materialized from the source string by
  offset.
- Preference pair → alignment ``preference_pairs.jsonl`` with
  ``{prompt, chosen, rejected}`` rows. Ties / both-bad rows skip
  the file entirely but still get marked promoted (idempotency).

Plus cross-cutting invariants:

- Idempotency: a second promote call reports
  ``skipped_already_promoted`` rather than duplicating rows.
- Job stats grow a ``promoted`` field.
- Each promotion emits a RunEvent with reason
  ``annotation_rows_promoted``.
- Unknown ``target_dataset_type`` → 400.
- Missing job → 404 wrapped in the existing error translator.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.dataset import Dataset, DatasetType  # noqa: E402
from app.models.run_event import (  # noqa: E402
    SEVERITY_INFO,
    STAGE_INGESTION,
    RunEvent,
)


TEST_DATA_DIR = (
    Path(tempfile.gettempdir())
    / f"brewslm-promotion-{uuid.uuid4().hex[:8]}"
)


def _read_events_for_project(project_id: int) -> list[RunEvent]:
    async def _go():
        async with async_session_factory() as session:
            result = await session.execute(
                select(RunEvent)
                .where(RunEvent.project_id == project_id)
                .order_by(RunEvent.id.asc())
            )
            return list(result.scalars().all())

    return asyncio.run(_go())


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class AnnotationPromotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.AUTH_ENABLED = False
        settings.DEBUG = False
        settings.DATA_DIR = TEST_DATA_DIR.resolve()
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ensure_dirs()
        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def _create_project(self, label: str) -> int:
        resp = self.client.post(
            "/api/projects",
            json={"name": f"{label}-{uuid.uuid4().hex[:6]}"},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return int(resp.json()["id"])

    def _create_job(
        self,
        project_id: int,
        *,
        label_type: str,
        name: str = "promo-test",
        label_schema: dict | None = None,
    ) -> dict:
        body = {
            "name": name,
            "label_type": label_type,
            "label_schema": label_schema or {},
        }
        resp = self.client.post(
            f"/api/projects/{project_id}/label-jobs/", json=body
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def _seed_row_directly(
        self, job_id: int, raw_payload: dict
    ) -> int:
        """Bypass the seed-from-dataset path (which requires a real
        Dataset on disk) and insert one LabelRow with raw_payload set
        directly. Returns the row id."""
        from app.models.label_job import LabelRow

        async def _go():
            async with async_session_factory() as session:
                row = LabelRow(
                    job_id=job_id,
                    source_row_id=str(uuid.uuid4().hex[:6]),
                    raw_payload=raw_payload,
                )
                session.add(row)
                await session.commit()
                return int(row.id)

        return asyncio.run(_go())

    def _submit_label(
        self, pid: int, job_id: int, row_id: int, label_payload: dict
    ):
        # Need to assign before submitting (matches the UI flow).
        assign = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/next-row",
            json={"user_id": 1},
        )
        self.assertEqual(assign.status_code, 200, assign.text)
        # Re-fetch row id from the assignment so we submit against
        # whatever the queue handed us.
        assigned_row_id = assign.json()["row"]["id"]
        submit = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/rows/{assigned_row_id}/submit",
            json={"label_payload": label_payload},
        )
        self.assertEqual(submit.status_code, 200, submit.text)
        return assigned_row_id

    def _promote(self, pid: int, job_id: int, *, target: str = "synthetic"):
        resp = self.client.post(
            f"/api/projects/{pid}/label-jobs/{job_id}/promote",
            json={"target_dataset_type": target},
        )
        return resp

    # ── Classification ─────────────────────────────────────────────

    def test_classification_promote_writes_to_synthetic(self):
        pid = self._create_project("promo-cls")
        job = self._create_job(
            pid,
            label_type="classification",
            label_schema={"allowed_labels": ["positive", "negative"]},
        )
        # Seed three rows + label each.
        for i in range(3):
            row_id = self._seed_row_directly(
                job["id"], {"text": f"review {i}"}
            )
            self._submit_label(
                pid, job["id"], row_id, {"label": "positive"}
            )

        resp = self._promote(pid, job["id"])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["promoted_count"], 3)
        self.assertEqual(body["skipped_already_promoted"], 0)
        self.assertEqual(body["target_dataset_type"], "synthetic")

        rows = _read_jsonl(Path(body["written_path"]))
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertIn("question", r)
            answer = json.loads(r["answer"])
            self.assertEqual(answer["label"], "positive")
            self.assertEqual(r["source"], "annotation_job")
            self.assertEqual(r["annotation_job_id"], job["id"])

    # ── Span ───────────────────────────────────────────────────────

    def test_span_promote_writes_entities_with_materialized_text(self):
        pid = self._create_project("promo-span")
        job = self._create_job(
            pid,
            label_type="span",
            label_schema={"span_types": ["PERSON"]},
        )
        text = "Alice met Bob in Paris."
        row_id = self._seed_row_directly(job["id"], {"text": text})
        # Submit with {start, end, type} — no explicit "text" field on
        # the span; promotion must materialize it from offsets.
        self._submit_label(
            pid,
            job["id"],
            row_id,
            {
                "spans": [
                    {"start": 0, "end": 5, "type": "PERSON"},
                    {"start": 10, "end": 13, "type": "PERSON"},
                ]
            },
        )

        resp = self._promote(pid, job["id"])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["promoted_count"], 1)

        [row] = _read_jsonl(Path(body["written_path"]))
        self.assertEqual(row["question"], text)
        answer = json.loads(row["answer"])
        ents = answer["entities"]
        self.assertEqual(len(ents), 2)
        self.assertEqual(ents[0]["text"], "Alice")
        self.assertEqual(ents[1]["text"], "Bob")
        # entities field also denormalized at the top of the row for
        # the structured-extraction eval handler.
        self.assertEqual(len(row["entities"]), 2)

    # ── Preference pair ────────────────────────────────────────────

    def test_preference_pair_promote_writes_alignment_file(self):
        pid = self._create_project("promo-pref")
        job = self._create_job(
            pid, label_type="preference_pair"
        )
        row_id = self._seed_row_directly(
            job["id"],
            {
                "prompt": "Translate hello to French",
                "completion_a": "Bonjour",
                "completion_b": "Salut",
            },
        )
        self._submit_label(
            pid,
            job["id"],
            row_id,
            {"chosen": "A", "tie": False, "both_bad": False},
        )

        resp = self._promote(pid, job["id"])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["promoted_count"], 1)
        # Alignment path goes into the preference_pairs.jsonl file,
        # NOT synthetic.jsonl.
        self.assertIn("preference_pairs.jsonl", body["written_path"])
        [row] = _read_jsonl(Path(body["written_path"]))
        self.assertEqual(row["prompt"], "Translate hello to French")
        self.assertEqual(row["chosen"], "Bonjour")
        self.assertEqual(row["rejected"], "Salut")

    def test_preference_pair_tie_is_skipped_but_marked_promoted(self):
        pid = self._create_project("promo-pref-tie")
        job = self._create_job(pid, label_type="preference_pair")
        row_id = self._seed_row_directly(
            job["id"],
            {
                "prompt": "Which is better?",
                "completion_a": "x",
                "completion_b": "y",
            },
        )
        self._submit_label(
            pid,
            job["id"],
            row_id,
            {"chosen": None, "tie": True, "both_bad": False},
        )
        resp = self._promote(pid, job["id"])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        # Tie row: not written to file, but counted as unrenderable so
        # a second promote doesn't keep retrying it.
        self.assertEqual(body["promoted_count"], 0)
        self.assertEqual(body["skipped_unrenderable"], 1)
        second = self._promote(pid, job["id"])
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["skipped_already_promoted"], 1)

    # ── Idempotency ────────────────────────────────────────────────

    def test_promoting_twice_does_not_duplicate(self):
        pid = self._create_project("promo-idempotent")
        job = self._create_job(pid, label_type="classification")
        for _ in range(2):
            row_id = self._seed_row_directly(
                job["id"], {"text": "x"}
            )
            self._submit_label(
                pid, job["id"], row_id, {"label": "positive"}
            )

        first = self._promote(pid, job["id"]).json()
        self.assertEqual(first["promoted_count"], 2)

        second = self._promote(pid, job["id"]).json()
        self.assertEqual(second["promoted_count"], 0)
        self.assertEqual(second["skipped_already_promoted"], 2)

        rows = _read_jsonl(Path(first["written_path"]))
        # File still has exactly 2 rows after the second call.
        self.assertEqual(len(rows), 2)

    # ── job_stats integration ──────────────────────────────────────

    def test_job_stats_reports_promoted_count(self):
        pid = self._create_project("promo-stats")
        job = self._create_job(pid, label_type="classification")
        row_id = self._seed_row_directly(job["id"], {"text": "x"})
        self._submit_label(
            pid, job["id"], row_id, {"label": "positive"}
        )

        # Pre-promotion: promoted == 0.
        detail = self.client.get(
            f"/api/projects/{pid}/label-jobs/{job['id']}"
        ).json()
        self.assertEqual(detail["stats"]["promoted"], 0)

        self._promote(pid, job["id"])

        # Post-promotion: promoted == labeled.
        detail = self.client.get(
            f"/api/projects/{pid}/label-jobs/{job['id']}"
        ).json()
        self.assertEqual(detail["stats"]["promoted"], 1)
        self.assertEqual(detail["stats"]["labeled"], 1)

    # ── Audit hook ─────────────────────────────────────────────────

    def test_promotion_emits_run_event(self):
        pid = self._create_project("promo-audit")
        job = self._create_job(pid, label_type="classification")
        row_id = self._seed_row_directly(job["id"], {"text": "x"})
        self._submit_label(
            pid, job["id"], row_id, {"label": "positive"}
        )
        self._promote(pid, job["id"])

        events = _read_events_for_project(pid)
        promos = [
            ev
            for ev in events
            if ev.reason_code == "annotation_rows_promoted"
            and ev.severity == SEVERITY_INFO
            and ev.stage == STAGE_INGESTION
        ]
        self.assertEqual(len(promos), 1)
        payload = promos[0].payload
        self.assertEqual(payload["job_id"], job["id"])
        self.assertEqual(payload["promoted_count"], 1)
        self.assertEqual(payload["label_type"], "classification")

    # ── Error paths ────────────────────────────────────────────────

    def test_missing_job_returns_404(self):
        pid = self._create_project("promo-missing")
        resp = self._promote(pid, 9999)
        self.assertEqual(resp.status_code, 404, resp.text)

    def test_invalid_target_dataset_type_returns_400(self):
        pid = self._create_project("promo-bad-target")
        job = self._create_job(pid, label_type="classification")
        resp = self._promote(pid, job["id"], target="train")
        self.assertEqual(resp.status_code, 400, resp.text)
        # Should mention what's allowed in the detail.
        self.assertIn("synthetic", resp.json()["detail"])

    def test_unlabeled_rows_are_skipped_not_promoted(self):
        pid = self._create_project("promo-unlabeled")
        job = self._create_job(pid, label_type="classification")
        # Seed but never label.
        self._seed_row_directly(job["id"], {"text": "unlabeled-1"})
        self._seed_row_directly(job["id"], {"text": "unlabeled-2"})
        resp = self._promote(pid, job["id"])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["promoted_count"], 0)
        self.assertEqual(body["skipped_unlabeled"], 2)

    # ── GOLD_DEV target path ───────────────────────────────────────

    def test_classification_promote_to_gold_dev(self):
        pid = self._create_project("promo-gold")
        job = self._create_job(pid, label_type="classification")
        row_id = self._seed_row_directly(job["id"], {"text": "x"})
        self._submit_label(
            pid, job["id"], row_id, {"label": "positive"}
        )
        resp = self._promote(pid, job["id"], target="gold_dev")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["target_dataset_type"], "gold_dev")
        self.assertIn("gold_dev.jsonl", body["written_path"])

        # Verify a GOLD_DEV dataset row exists in the DB.
        async def _check():
            async with async_session_factory() as session:
                result = await session.execute(
                    select(Dataset).where(
                        Dataset.project_id == pid,
                        Dataset.dataset_type == DatasetType.GOLD_DEV,
                    )
                )
                return result.scalar_one_or_none()

        gold = asyncio.run(_check())
        self.assertIsNotNone(gold)
        self.assertGreaterEqual(int(gold.record_count or 0), 1)


if __name__ == "__main__":
    unittest.main()
