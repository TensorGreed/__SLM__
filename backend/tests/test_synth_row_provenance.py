"""Synthetic row provenance: every playbook row records the Job and backend that wrote it.

``synth_job_id`` (int, or null when not run as a Job), ``synth_backend`` (the backend pin string) and
``synth_batch: true`` for a Message Batch run are stamped at the single append point. The review
queue returns them per row, groups by ``job`` / ``backend`` with a label (Job id + backend + mode +
direct / batch), and filters by ``synth_job_id`` / ``synth_backend``.

pytest-only (the Jobs these tests start are awaited in tearDown).
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.main import app
from app.services.synth_backends import TemplateBackend
from app.services.synth_playbooks import SynthMode
from test_code_findings_playbooks import TAXONOMY, clean_row
from test_playbook_message_batch import FakeBatchBackend

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


class AppendStampTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp()) / "synthetic.jsonl"

    def rows(self):
        return [{"payload": {"text": "x"}, "synth_confidence": 0.9, "synth_source": "playbook:r:m"}]

    def read(self):
        return [json.loads(line) for line in self.tmp.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_provenance_is_stamped(self):
        from app.services.synth_playbook_service import _append_to_synthetic_jsonl

        _append_to_synthetic_jsonl(self.tmp, self.rows(), synth_job_id=7, synth_backend="cloud:anthropic:m", synth_batch=True)
        row = self.read()[0]
        self.assertEqual((row["synth_job_id"], row["synth_backend"], row["synth_batch"]), (7, "cloud:anthropic:m", True))

    def test_a_synchronous_direct_run_has_null_job_and_no_batch_flag(self):
        from app.services.synth_playbook_service import _append_to_synthetic_jsonl

        _append_to_synthetic_jsonl(self.tmp, self.rows(), synth_backend="ollama:x")
        row = self.read()[0]
        self.assertIsNone(row["synth_job_id"])
        self.assertEqual(row["synth_backend"], "ollama:x")
        self.assertNotIn("synth_batch", row)


class ProjectBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls._patches = [
            patch("app.services.synth_backends.BACKEND_REGISTRY", [TemplateBackend]),
            patch("app.services.synth_playbook_service.BACKEND_REGISTRY", [TemplateBackend]),
            patch.dict("os.environ", {"BREWSLM_TEMPLATE_SYNTH": "1", "ANTHROPIC_BATCH_POLL_SECONDS": "0"}),
        ]
        for p in cls._patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        settings.AUTH_ENABLED = cls._prev_auth

    def tearDown(self):
        wait_for_jobs_idle(client)

    def setUp(self):
        r = client.post("/api/projects", json={"name": f"provenance-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        self.pdir = Path(settings.DATA_DIR) / "projects" / str(self.pid)
        resp = client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "code-findings"})
        self.assertEqual(resp.status_code, 200, resp.text)
        put = client.put(f"/api/projects/{self.pid}/label-taxonomy", json={"entries": TAXONOMY["entries"]})
        self.assertEqual(put.status_code, 200, put.text)
        self.add_gold([clean_row(code=f"def f{i}(x):\n    return x + {i}", path=f"mod{i}.py") for i in range(8)])

    def add_gold(self, rows):
        from app.database import async_session_factory
        from app.models.dataset import Dataset, DatasetType

        path = self.pdir / "test_inputs" / "dev.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        async def go():
            async with async_session_factory() as db:
                db.add(Dataset(project_id=self.pid, name="dev", dataset_type=DatasetType.GOLD_DEV,
                               file_path=str(path), record_count=len(rows)))
                await db.commit()

        asyncio.run(go())

    def direct_run(self, backend, count=2, **kw):
        from app.database import async_session_factory
        from app.services.synth_playbook_service import run_playbook

        async def go():
            async with async_session_factory() as db:
                kw.setdefault("use_batch", bool(getattr(backend, "use_batch", False)))
                result = await run_playbook(db, self.pid, SynthMode.SEEDED_DEFECT, target_count=count,
                                            backend_override=backend, params={"label_id": "injection.query"}, **kw)
                await db.commit()
                return result

        return asyncio.run(go())

    def stored(self):
        path = self.pdir / "synthetic" / "synthetic.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def queue(self, **params):
        resp = client.get(f"/api/projects/{self.pid}/synthetic/review-queue", params=params)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()


class PlaybookStampTests(ProjectBase):
    def test_a_job_run_stamps_the_job_id_and_the_backend(self):
        resp = client.post(f"/api/projects/{self.pid}/synthetic/run-playbook?async_job=true", json={
            "mode": "seeded_defect", "target_count": 2, "params": {"label_id": "injection.query"}})
        self.assertEqual(resp.status_code, 202, resp.text)
        job_id = resp.json()["id"]
        wait_for_jobs_idle(client)
        done = client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(done["status"], "succeeded", done)
        rows = self.stored()
        self.assertTrue(rows)
        self.assertEqual({r["synth_job_id"] for r in rows}, {job_id})
        self.assertEqual({r["synth_backend"] for r in rows}, {TemplateBackend().describe()})
        self.assertTrue(all("synth_batch" not in r for r in rows))

    def test_a_synchronous_run_has_a_null_job_id(self):
        resp = client.post(f"/api/projects/{self.pid}/synthetic/run-playbook", json={
            "mode": "seeded_defect", "target_count": 2, "params": {"label_id": "injection.query"}})
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = self.stored()
        self.assertTrue(rows)
        self.assertEqual({r["synth_job_id"] for r in rows}, {None})
        self.assertTrue(all(r["synth_backend"] for r in rows))

    def test_a_message_batch_run_is_flagged(self):
        backend = FakeBatchBackend(name="cloud:anthropic:fake-model", rows_per_call=2)
        self.direct_run(backend, count=2)
        rows = self.stored()
        self.assertTrue(rows)
        self.assertEqual({r["synth_batch"] for r in rows}, {True})
        self.assertEqual({r["synth_backend"] for r in rows}, {"cloud:anthropic:fake-model"})

    def test_a_dry_run_writes_nothing(self):
        self.direct_run(FakeBatchBackend(use_batch=False), count=1, dry_run=True)
        self.assertFalse((self.pdir / "synthetic" / "synthetic.jsonl").exists())


class QueueProvenanceTests(ProjectBase):
    def seed(self):
        base = {"synth_confidence": 0.9, "review_status": "pending"}
        rows = [
            {"id": 1, "text": "a", **base, "synth_source": "playbook:r:alpha", "synth_job_id": 11, "synth_backend": "cloud:anthropic:m", "synth_batch": True},
            {"id": 2, "text": "b", **base, "synth_source": "playbook:r:alpha", "synth_job_id": 11, "synth_backend": "cloud:anthropic:m", "synth_batch": True},
            {"id": 3, "text": "c", **base, "synth_source": "playbook:r:alpha", "synth_job_id": 12, "synth_backend": "command:x"},
            {"id": 4, "text": "d", **base, "synth_source": "playbook:r:beta", "synth_job_id": None},
        ]
        path = self.pdir / "synthetic" / "synthetic.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_rows_carry_provenance_and_it_is_not_in_the_payload_or_preview(self):
        self.seed()
        groups = self.queue()["groups"]
        row = next(r for g in groups for r in g["rows"] if r["id"] == 1)
        self.assertEqual((row["synth_job_id"], row["synth_backend"], row["synth_batch"]), (11, "cloud:anthropic:m", True))
        self.assertNotIn("synth_job_id", row["payload"])
        self.assertNotIn("synth_backend", row["preview"])
        plain = next(r for g in groups for r in g["rows"] if r["id"] == 4)
        self.assertEqual((plain["synth_job_id"], plain["synth_backend"], plain["synth_batch"]), (None, None, False))

    def test_group_by_job_labels_each_group(self):
        self.seed()
        data = self.queue(group_by="job")
        self.assertEqual(data["group_by"], "job")
        by_key = {g["synth_source"]: g for g in data["groups"]}
        self.assertEqual(set(by_key), {"job:11", "job:12", "(no job)"})
        self.assertEqual(by_key["job:11"]["count"], 2)
        self.assertEqual(by_key["job:11"]["label"], "job:11 · cloud:anthropic:m · alpha · batch")
        self.assertEqual(by_key["job:12"]["label"], "job:12 · command:x · alpha · direct")
        self.assertIn("(unknown backend)", by_key["(no job)"]["label"])

    def test_group_by_backend_labels_each_group(self):
        self.seed()
        by_key = {g["synth_source"]: g for g in self.queue(group_by="backend")["groups"]}
        self.assertEqual(set(by_key), {"cloud:anthropic:m", "command:x", "(unknown backend)"})
        self.assertEqual(by_key["cloud:anthropic:m"]["label"], "cloud:anthropic:m · alpha · batch · 1 job")

    def test_source_grouping_has_no_label(self):
        self.seed()
        self.assertTrue(all(g["label"] is None for g in self.queue()["groups"]))

    def test_filters_narrow_the_lists_but_not_the_totals(self):
        self.seed()
        data = self.queue(synth_job_id=11)
        self.assertEqual(data["total_pending"], 2)
        self.assertEqual(data["total_rows"], 4)
        self.assertEqual(sum(g["count"] for g in data["groups"]), 2)
        data = self.queue(synth_backend="command:x")
        self.assertEqual([r["id"] for g in data["groups"] for r in g["rows"]], [3])
        # Facets always describe every pending row, so the pickers keep their options.
        self.assertEqual([j["synth_job_id"] for j in data["facets"]["jobs"]], [11, 12])
        self.assertEqual({b["synth_backend"]: b["count"] for b in data["facets"]["backends"]},
                         {"cloud:anthropic:m": 2, "command:x": 1, "(unknown backend)": 1})


if __name__ == "__main__":
    unittest.main()
