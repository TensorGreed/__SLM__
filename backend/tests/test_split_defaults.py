"""Wave 2d — safe data defaults a newcomer doesn't have to know about.

Pins:
  * a fresh split drops exact / near-duplicate rows by default (reported in
    the manifest) and stratifies by ``label`` on its own when the rows carry
    a categorical label; explicit choices still win;
  * when synthetic / imported rows exist, only UNLABELLED cleaned document
    passages are left out — labelled cleaned rows (CSV/XLSX uploads) stay;
  * imbalanced classification gets inverse-frequency class weights
    (>= 3x imbalance, mean 1, capped), balanced data gets none;
  * autopilot takes the task shape from the project's data over keywords
    in the goal text, and the brief's task keywords feed the same prior.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.services.dataset_service import _looks_categorical
from app.services.newbie_autopilot_service import resolve_newbie_autopilot_intent
from app.services.task_shape_service import _intent_profiles
from scripts.train import _class_weights


def setUpModule():
    global _client_cm, client, _prev_auth
    _prev_auth = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    _client_cm = TestClient(app)
    client = _client_cm.__enter__()


def tearDownModule():
    _client_cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = _prev_auth


class PureHelperTests(unittest.TestCase):
    def test_class_weights_only_for_real_imbalance(self):
        self.assertIsNone(_class_weights([0] * 50 + [1] * 40, 2))  # 1.25x → balanced enough
        weights = _class_weights([0] * 90 + [1] * 10, 2)
        self.assertAlmostEqual(sum(weights) / 2, 1.0, places=6)
        self.assertGreater(weights[1], weights[0] * 8)
        capped = _class_weights([0] * 10_000 + [1] * 1, 2, cap=10.0)
        self.assertLessEqual(max(capped), 10.0)
        # A class that never appears in train gets weight 1.
        self.assertEqual(_class_weights([0] * 90 + [1] * 10, 3)[2], 1.0)

    def test_categorical_label_detection(self):
        labelled = [{"label": "billing" if i % 3 else "shipping"} for i in range(30)]
        self.assertTrue(_looks_categorical(labelled, "label"))
        unique = [{"label": f"free text {i}"} for i in range(30)]
        self.assertFalse(_looks_categorical(unique, "label"))
        self.assertFalse(_looks_categorical([{"text": "x"}] * 30, "label"))


class SplitDefaultsTests(unittest.TestCase):
    def setUp(self):
        resp = client.post("/api/projects", json={"name": f"split-{uuid4().hex[:6]}", "description": ""})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.pid = int(resp.json()["id"])

    def _upload_clean(self, name: str, content: bytes) -> None:
        doc = client.post(
            f"/api/projects/{self.pid}/ingestion/upload",
            files={"file": (name, content, "application/octet-stream")},
        ).json()
        client.post(f"/api/projects/{self.pid}/ingestion/documents/{doc['id']}/process")
        resp = client.post(
            f"/api/projects/{self.pid}/cleaning/clean",
            json={"document_id": doc["id"], "chunk_size": 200, "chunk_overlap": 0},
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    def _split(self, **body) -> dict:
        resp = client.post(f"/api/projects/{self.pid}/dataset/split", json=body)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def test_fresh_split_dedups_and_auto_stratifies_labels(self):
        client.post(f"/api/projects/{self.pid}/task-shape/confirm", json={"task_profile": "classification"})
        rows = ["text,label"]
        for i in range(36):
            rows.append(f"ticket {i} about my invoice being wrong,billing")
        for i in range(6):
            rows.append(f"parcel {i} never arrived at my door,shipping")
        rows += ["ticket 0 about my invoice being wrong,billing"] * 3  # exact dups
        self._upload_clean("tickets.csv", ("\n".join(rows) + "\n").encode())

        manifest = self._split()
        # Dedup runs by default and is reported. (These exact duplicates
        # were already dropped row-by-row at cleaning, so nothing is left
        # for the split's own dedup here.)
        self.assertTrue(manifest["dedup_requested"])
        self.assertEqual(manifest["dedup_report"]["input_count"], 42)
        self.assertEqual(manifest["stratify_by"], "label")
        self.assertTrue(manifest["stratify_auto"])
        prep = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "prepared"
        test_labels = {json.loads(l).get("label") for l in (prep / "test.jsonl").read_text().splitlines() if l}
        self.assertIn("shipping", test_labels)  # the rare class reached test

    def test_explicit_choices_win(self):
        client.post(f"/api/projects/{self.pid}/task-shape/confirm", json={"task_profile": "classification"})
        rows = ["text,label"] + [f"row {i} text here,{'a' if i % 4 else 'b'}" for i in range(40)]
        self._upload_clean("t.csv", ("\n".join(rows) + "\n").encode())
        manifest = self._split(auto_stratify=False)
        self.assertIsNone(manifest["stratify_by"])
        self.assertFalse(manifest["stratify_auto"])

    def test_labelled_cleaned_rows_survive_when_synthetic_exists(self):
        csv = "question,answer\n" + "".join(f"How do I reset unit {i}?,Hold the button for {i} seconds.\n" for i in range(20))
        self._upload_clean("faq.csv", csv.encode())
        from docx import Document
        import io

        doc = Document()
        for i in range(8):
            doc.add_paragraph(f"Section {i}: the warranty covers the motor housing for fourteen months. " * 3)
        buf = io.BytesIO()
        doc.save(buf)
        self._upload_clean("manual.docx", buf.getvalue())

        synth_dir = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "synthetic"
        synth_dir.mkdir(parents=True, exist_ok=True)
        synth_path = synth_dir / "synthetic.jsonl"
        synth_path.write_text("".join(
            json.dumps({"question": f"Synthetic q {i}?", "answer": f"Synthetic a {i}."}) + "\n" for i in range(10)
        ))

        async def _seed():
            async with async_session_factory() as db:
                db.add(Dataset(project_id=self.pid, name="Synthetic", dataset_type=DatasetType.SYNTHETIC,
                               file_path=str(synth_path), record_count=10))
                await db.commit()

        asyncio.run(_seed())
        manifest = self._split()
        cleaned_filter = manifest["include_types_resolution"]["cleaned_filter"]
        self.assertEqual(cleaned_filter["kept"], 20)  # every labelled CSV row kept
        self.assertGreater(cleaned_filter["dropped_unlabelled_passages"], 0)  # docx passages left out


    def test_gold_set_stays_out_of_the_default_split(self):
        """The gold set is the answer key: the default split must not train on
        it (the train↔gold leakage check would flag every gold row that did).
        An explicit include_types can still opt gold dev in."""
        csv = "question,answer\n" + "".join(f"How do I reset unit {i}?,Hold the button for {i} seconds.\n" for i in range(20))
        self._upload_clean("faq.csv", csv.encode())
        gold_dir = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "gold"
        gold_dir.mkdir(parents=True, exist_ok=True)
        gold_path = gold_dir / "gold_dev.jsonl"
        gold_path.write_text("".join(
            json.dumps({"question": f"Gold-only question {i}?", "answer": f"Gold answer {i}."}) + "\n" for i in range(10)
        ))

        async def _seed():
            async with async_session_factory() as db:
                db.add(Dataset(project_id=self.pid, name="Gold dev", dataset_type=DatasetType.GOLD_DEV,
                               file_path=str(gold_path), record_count=10))
                await db.commit()

        asyncio.run(_seed())
        prep = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "prepared"

        def _prepared_text() -> str:
            return "".join((prep / f"{name}.jsonl").read_text() for name in ("train", "val", "test")
                           if (prep / f"{name}.jsonl").exists())

        manifest = self._split()
        self.assertNotIn("gold_dev", manifest["include_types_resolution"].get("resolved_types", ["cleaned", "synthetic"]))
        self.assertNotIn("Gold-only question", _prepared_text())
        self.assertEqual(sum(1 for line in _prepared_text().splitlines() if line.strip()), 20)

        self._split(include_types=["cleaned", "gold_dev"])
        self.assertIn("Gold-only question", _prepared_text())


class DetectorPriorTests(unittest.TestCase):
    def test_data_beats_goal_keywords(self):
        plan = resolve_newbie_autopilot_intent(
            intent="summarize meeting notes",
            data_task_profile="classification",
            data_task_profile_source="confirmed",
        )
        self.assertEqual(plan["task_profile"], "classification")
        self.assertEqual(plan["task_profile_source"], "data_confirmed")
        self.assertEqual(plan["intent_task_profile"], "summarization")
        self.assertEqual(plan["safe_training_config"]["task_type"], "classification")

    def test_documents_only_plan_continues_pretraining(self):
        plan = resolve_newbie_autopilot_intent(
            intent="learn our product manuals", data_task_profile="language_modeling",
        )
        cfg = plan["safe_training_config"]
        self.assertEqual(cfg["training_mode"], "domain_pretrain")
        self.assertEqual(cfg["target_modules"], "all-linear")

    def test_keywords_still_work_without_data(self):
        plan = resolve_newbie_autopilot_intent(intent="answer customer faq questions")
        self.assertEqual(plan["task_profile"], "qa")
        self.assertEqual(plan["task_profile_source"], "intent_keywords")

    def test_brief_keywords_feed_the_same_prior(self):
        self.assertIn("classification", _intent_profiles("triage incoming support emails"))


if __name__ == "__main__":
    unittest.main()
