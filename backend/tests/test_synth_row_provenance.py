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


class CredentialSafetyTests(unittest.TestCase):
    def test_label_sanitiser(self):
        from app.services.synth_playbook_service import safe_backend_label as f

        self.assertEqual(f("cloud:anthropic:claude-x"), "cloud:anthropic:claude-x")
        self.assertEqual(f("teacher:m@https://api.example.com/v1"), "teacher:m@https://api.example.com/v1")
        self.assertEqual(f("teacher:m@https://user:s3cret@api.example.com/v1"), "teacher:m@https://api.example.com/v1")
        self.assertEqual(f("teacher:m@https://tok123@api.example.com/v1?api_key=sk-abc&region=us"),
                         "teacher:m@https://api.example.com/v1?region=us")
        out = f("x@https://h.example/v1?foo=sk-ant-0123456789abcdef0123456789")
        self.assertNotIn("sk-ant", out)
        self.assertEqual(f(None), "")

    def test_the_append_point_never_stamps_credentials(self):
        import tempfile

        from app.services.synth_playbook_service import _append_to_synthetic_jsonl

        path = Path(tempfile.mkdtemp()) / "synthetic.jsonl"
        _append_to_synthetic_jsonl(path, [{"payload": {"t": 1}, "synth_confidence": 0.9, "synth_source": "s"}],
                                   synth_backend="teacher:m@https://u:pw@h.example/v1?token=abc")
        row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["synth_backend"], "teacher:m@https://h.example/v1")
        self.assertNotIn("pw", path.read_text(encoding="utf-8"))

    def test_the_teacher_backends_label_is_sanitised_before_stamping(self):
        from app.services.synth_backends.teacher import TeacherModelBackend
        from app.services.synth_playbook_service import safe_backend_label

        label = TeacherModelBackend(model="m", api_url="https://u:hunter2@teach.example/v1?api_key=zzz").describe()
        self.assertIn("hunter2", label)  # the raw label does leak: that is why it is sanitised before stamping
        self.assertNotIn("hunter2", safe_backend_label(label))
        self.assertNotIn("zzz", safe_backend_label(label))


class DocumentsFlowProvenanceTests(unittest.TestCase):
    def test_flow_rows_carry_job_backend_and_batch(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        pid = 987622
        root = Path(tempfile.mkdtemp())
        ds = SimpleNamespace(record_count=0, file_path=None)
        db = SimpleNamespace(flush=AsyncMock())

        class LLM:
            def describe(self):
                return "teacher:m@https://u:pw@h.example/v1"

        from app.services import documents_qa_flow_service as flow

        with patch.object(settings, "DATA_DIR", root), \
                patch("app.services.synthetic_service.get_or_create_synthetic_dataset", AsyncMock(return_value=ds)):
            asyncio.run(flow._save_training_pairs(db, pid, [{"question": "q", "answer": "a"}], backend=LLM(), batch=True))
        row = json.loads((root / "projects" / str(pid) / "synthetic" / "synthetic.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["synth_backend"], "teacher:m@https://h.example/v1")
        self.assertIs(row["synth_batch"], True)
        self.assertIn("synth_job_id", row)
        self.assertIsNone(row["synth_job_id"])
        self.assertEqual(row["synth_source"], flow.FLOW_SOURCE)


class DuplicateDropsJobIdsTests(unittest.TestCase):
    def test_a_copied_project_keeps_the_backend_but_not_the_source_jobs_id(self):
        import tempfile

        from app.services.project_duplicate_service import copy_data_tree

        root = Path(tempfile.mkdtemp())
        with patch.object(settings, "DATA_DIR", root):
            src = root / "projects" / "1" / "synthetic"
            src.mkdir(parents=True)
            (src / "synthetic.jsonl").write_text(
                json.dumps({"id": 1, "synth_job_id": 41, "synth_backend": "cloud:anthropic:m", "synth_batch": True, "t": "x"}) + "\n"
                + json.dumps({"id": 2, "t": "no job"}) + "\n", encoding="utf-8")
            (src / "review_assist.json").write_text(json.dumps({"1": {"verdict": "pass", "job_id": 50, "sig": "s"}}), encoding="utf-8")
            copy_data_tree(1, 2)
            dst = root / "projects" / "2" / "synthetic"
            rows = [json.loads(line) for line in (dst / "synthetic.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertNotIn("synth_job_id", rows[0])
            self.assertEqual((rows[0]["synth_backend"], rows[0]["synth_batch"]), ("cloud:anthropic:m", True))
            self.assertEqual(rows[1], {"id": 2, "t": "no job"})
            self.assertIsNone(json.loads((dst / "review_assist.json").read_text(encoding="utf-8"))["1"]["job_id"])
            # the source is untouched
            self.assertIn("41", (src / "synthetic.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
