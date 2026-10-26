"""code-findings end to end on the simulate runtime: create a project, apply
the recipe, upload a taxonomy, import findings rows through a G3 mapper,
prepare the split, train (simulated: no weights), score injected predictions
for the test split through the evaluation API and read per-label metrics back.

Real model inference is out of scope here (simulate runs have no weights and
the auto lift eval skips them)."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from _job_wait import wait_for_jobs_idle
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.synth_backends import TemplateBackend

client: TestClient | None = None
_prev: dict = {}

TAXONOMY = [
    {"id": "injection", "title": "Injection"},
    {"id": "injection.query", "title": "Query injection", "parent": "injection", "aliases": ["sql concat"]},
    {"id": "secret.hardcoded", "title": "Hard-coded secret"},
    {"id": "logging.sensitive", "title": "Sensitive logging"},
]


def setUpModule():
    global client
    for key in ("TRAINING_BACKEND", "ALLOW_SIMULATED_TRAINING"):
        _prev[key] = getattr(settings, key)
    settings.TRAINING_BACKEND = "simulate"
    settings.ALLOW_SIMULATED_TRAINING = True
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    wait_for_jobs_idle(client, timeout_s=60)
    client.__exit__(None, None, None)
    for key, value in _prev.items():
        setattr(settings, key, value)


def fix_rows(n: int) -> list[dict]:
    """``n`` before/after fix pairs -> n flagged + n clean chunks, each unique."""
    rows = []
    for i in range(n):
        before = (
            f"def load_{i}(cur, key_{i}):\n"
            f"    table = 'items_{i}'\n"
            f"    q = 'SELECT * FROM ' + table + ' WHERE k = ' + key_{i}\n"
            f"    cur.execute(q)\n"
            f"    return cur.fetchall()  # variant {i * 7919}"
        )
        after = before.replace(
            f"q = 'SELECT * FROM ' + table + ' WHERE k = ' + key_{i}\n    cur.execute(q)",
            f"q = 'SELECT * FROM items_{i} WHERE k = ?'\n    cur.execute(q, (key_{i},))",
        )
        rows.append({"id": i, "func_before": before, "func_after": after,
                     "label": "sql concat", "path": f"svc/mod_{i}.py"})
    return rows


class CodeFindingsE2ETests(unittest.TestCase):
    def _add_gold_dev_clean_rows(self, pid: int, n: int) -> None:
        from app.database import async_session_factory
        from app.models.dataset import Dataset, DatasetType

        rows = [{"row_kind": "findings", "code": f"def clean_{i}(a, b):\n    return a + b + {i}", "language": "python",
                 "path": f"svc/clean_{i}.py", "start_line": 20 + 50 * i, "end_line": 21 + 50 * i,
                 "findings_json": {"findings": []}, "labels_raw": [], "taxonomy_version": None} for i in range(n)]
        path = Path(settings.DATA_DIR) / "projects" / str(pid) / "gold_dev_clean.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        async def go():
            async with async_session_factory() as db:
                db.add(Dataset(project_id=pid, name="gold dev", dataset_type=DatasetType.GOLD_DEV,
                               file_path=str(path), record_count=len(rows)))
                await db.commit()

        asyncio.run(go())

    def setUp(self):
        self.pid = client.post("/api/projects", json={"name": f"cf-e2e-{time.time_ns()}", "description": "x"}).json()["id"]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def tearDown(self):
        wait_for_jobs_idle(client, timeout_s=60)

    def test_pipeline(self):
        pid = self.pid
        # 1. recipe -> base model, adapter preset, guided-json hint (no enum yet)
        resp = client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "code-findings"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["base_model_name"], "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(resp.json()["selected_recipe"]["scoring_mode"], "finding_set")
        hint = client.get(f"/api/projects/{pid}/guided-json").json()["guided_json"]
        self.assertNotIn("enum", hint["schema"]["properties"]["findings"]["items"]["properties"]["label"])

        # 2. taxonomy -> hint refreshed with the label enum
        put = client.put(f"/api/projects/{pid}/label-taxonomy", json={"entries": TAXONOMY})
        self.assertEqual(put.status_code, 200, put.text)
        version = put.json()["version"]
        hint = client.get(f"/api/projects/{pid}/guided-json").json()["guided_json"]
        self.assertEqual(hint["taxonomy_version"], version)
        self.assertEqual(
            hint["schema"]["properties"]["findings"]["items"]["properties"]["label"]["enum"],
            ["injection", "injection.query", "secret.hardcoded", "logging.sensitive"],
        )

        # 3. import through the G3 mapper (alias label resolves to a taxonomy id)
        path = Path(self.tmp.name) / "fixes.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in fix_rows(20)) + "\n", encoding="utf-8")
        imp = client.post(f"/api/projects/{pid}/dataset-import/run",
                          json={"locator": f"jsonl:{path}", "mapper_id": "fix_pair", "field_map": {}})
        self.assertEqual(imp.status_code, 200, imp.text)
        self.assertEqual(imp.json()["accepted_count"], 40)

        # 3b. synth: seeded_defect from a GOLD_DEV answer key of clean chunks (template
        #     backend stubbed in BOTH registries), accepted through the review queue so the
        #     rows enter the split next to the imported ones.
        self._add_gold_dev_clean_rows(pid, 3)
        with patch("app.services.synth_backends.BACKEND_REGISTRY", [TemplateBackend]), patch(
            "app.services.synth_playbook_service.BACKEND_REGISTRY", [TemplateBackend]
        ), patch.dict("os.environ", {"BREWSLM_TEMPLATE_SYNTH": "1"}):
            synth = client.post(f"/api/projects/{pid}/synthetic/run-playbook", json={
                "mode": "seeded_defect", "target_count": 6, "params": {"label_id": "injection.query"}})
        self.assertEqual(synth.status_code, 200, synth.text)
        self.assertEqual((len(synth.json()["rows"]), synth.json()["rejected_count"]), (6, 0))
        source = synth.json()["rows"][0]["synth_source"]
        accepted = client.post(f"/api/projects/{pid}/synthetic/review-queue/bulk-update-by-source",
                               json={"source": source, "action": "accept"})
        self.assertEqual(accepted.status_code, 200, accepted.text)

        # 4. prepare the split
        split = client.post(f"/api/projects/{pid}/dataset/split",
                            json={"train_ratio": 0.6, "val_ratio": 0.1, "test_ratio": 0.3, "dedup_rows": False})
        self.assertEqual(split.status_code, 200, split.text)
        manifest = split.json()
        schema = manifest["output_schema"]
        self.assertEqual(manifest["adapter_id"], "structured-extraction")
        self.assertEqual(schema["scoring_mode"], "finding_set")
        self.assertEqual(schema["required"], ["findings"])
        self.assertEqual(schema["finding_match"], {"iou_threshold": 0.5, "attrs": [], "hierarchy": False, "parent_credit": 0.5})
        self.assertEqual(schema["labels"], ["injection", "injection.query", "secret.hardcoded", "logging.sensitive"])
        self.assertEqual(schema["label_parents"], {"injection.query": "injection"})
        self.assertEqual(schema["prompt_labels"], schema["labels"])

        prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        train = [json.loads(x) for x in (prepared / "train.jsonl").read_text().splitlines() if x.strip()]
        test = [json.loads(x) for x in (prepared / "test.jsonl").read_text().splitlines() if x.strip()]
        self.assertGreaterEqual(len(train), 20)
        self.assertGreaterEqual(len(test), 8)
        val = [json.loads(x) for x in (prepared / "val.jsonl").read_text().splitlines() if x.strip()] \
            if (prepared / "val.jsonl").exists() else []
        self.assertEqual(len(train) + len(val) + len(test), 40 + 6)  # imported + accepted synthetic rows
        synthetic = [r for r in train + val + test if str(r.get("synth_source", "")).startswith("playbook:code-findings")]
        self.assertEqual(len(synthetic), 6)
        for row in train + test:
            self.assertTrue(row["source_text"].startswith("Inspect the code chunk below"))
            self.assertIn("Allowed labels: injection, injection.query", row["source_text"])
            self.assertEqual(json.loads(row["target_text"]), row["findings_json"])

        # 5. train on the simulate runtime and wait for it
        exp = client.post(f"/api/projects/{pid}/training/experiments", json={
            "name": "cf", "description": "d", "config": {"base_model": "Qwen/Qwen2.5-Coder-7B-Instruct", "num_epochs": 1}})
        self.assertEqual(exp.status_code, 201, exp.text)
        eid = exp.json()["id"]
        start = client.post(f"/api/projects/{pid}/training/experiments/{eid}/start")
        self.assertEqual(start.status_code, 200, start.text)
        self.assertEqual(start.json()["backend"], "simulate")
        wait_for_jobs_idle(client, timeout_s=90)
        status = client.get(f"/api/projects/{pid}/training/experiments/{eid}/status").json()
        self.assertEqual(status["status"], "completed", status)

        # 6. score injected predictions for the test split (the handler builds
        #    the prompt + reference exactly as the held-out eval would) and
        #    persist the result through the evaluation API.
        from app.services.eval_task_handler_service import StructuredExtractionHandler, build_eval_context

        ctx, handler = build_eval_context(project_id=pid, experiment_id=eid, eval_type="f1",
                                          dataset_name="test", experiment_task_type="causal_lm")
        self.assertIsInstance(handler, StructuredExtractionHandler)
        built = handler.build_prompts(test, ctx)
        for row, bp in zip(test, built):
            self.assertEqual(bp.prompt, row["source_text"])  # train and eval prompts are byte-identical
        self.assertFalse(handler.wraps_own_prompt(ctx))

        predictions = []
        for i, bp in enumerate(built):
            ref = json.loads(bp.reference)
            if i % 4 == 3:  # one row in four: a hallucinated finding, the real one dropped
                pred = {"findings": [{"label": "secret.hardcoded", "start_line": 1, "end_line": 1, "evidence": "x", "attrs": {}}]}
            elif ref["findings"] and i % 4 == 2:  # right label, a parent label instead of the leaf
                pred = {"findings": [{**ref["findings"][0], "label": "injection"}]}
            else:
                pred = ref
            predictions.append({"prompt": bp.prompt, "reference": bp.reference, "prediction": json.dumps(pred)})

        run = client.post(f"/api/projects/{pid}/evaluation/run", json={
            "experiment_id": eid, "dataset_name": "test", "eval_type": "f1", "predictions": predictions})
        self.assertEqual(run.status_code, 201, run.text)
        metrics = run.json()["metrics"]
        self.assertEqual(metrics["scoring_mode"], "finding_set")
        self.assertEqual(metrics["total"], len(test))
        self.assertIn("injection.query", metrics["per_class"])
        self.assertEqual(metrics["per_class"]["injection.query"]["support"], sum(
            1 for bp in built if json.loads(bp.reference)["findings"]))
        self.assertGreater(metrics["f1"], 0.0)
        self.assertLess(metrics["f1"], 1.0)
        for key in ("precision_macro", "clean_chunk_fp_rate", "malformed_rate", "json_validity_rate", "iou_threshold"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["malformed_count"], 0)

        # the eval API returns what was persisted
        listed = client.get(f"/api/projects/{pid}/evaluation/results/{eid}").json()
        self.assertEqual(listed[0]["metrics"]["per_class"], metrics["per_class"])
        self.assertEqual(listed[0]["metrics"]["f1"], metrics["f1"])

        # per-label gates resolve through per_class
        from app.services.evaluation_pack_service import _flatten_per_class_metrics

        class _Row:
            id, experiment_id, eval_type, dataset_name = 1, eid, "f1", "test"

        values: dict = {}
        _flatten_per_class_metrics(values, {}, payload_key="per_class", payload_value=metrics["per_class"],
                                   row=_Row(), eval_type="f1")
        self.assertIn("f1_injection.query", values)


if __name__ == "__main__":
    unittest.main()
