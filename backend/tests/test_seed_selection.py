"""Ship the best seed of a multi-seed group (``seed_selection_service``).

* the pick reads VALIDATION metrics only (span F1 / macro-F1 / accuracy /
  lowest loss by task type), ties → lower run id, a child without the metric
  is listed but never picked;
* GET /seed-groups/{leader}/pick and POST .../ship (records the pick on the
  leader, runs the normal export of the picked child);
* the export says how the seed was picked: span model card "Seed selection"
  line, manifest ``seed_selection``, release-notes line for any task type.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.database import async_session_factory
from app.main import app
from app.models.experiment import EvalResult, Experiment, ExperimentStatus
from app.services.seed_selection_service import (
    pick_seed,
    selection_text,
    validation_values,
)
from app.services.span_model_package_service import build_span_model_card

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def _child(eid, seed, **values):
    return {"experiment_id": eid, "seed_value": seed, "values": values}


class PickSeedTests(unittest.TestCase):
    def test_span_f1_highest_wins(self):
        out = pick_seed(
            [_child(10, 42, span_f1=0.81, eval_loss=0.1), _child(11, 43, span_f1=0.84, eval_loss=0.3), _child(12, 44, span_f1=0.82)],
            task_type="token_classification",
        )
        self.assertEqual(out["metric_id"], "span_f1")
        self.assertEqual((out["picked_experiment_id"], out["picked_seed_value"]), (11, 43))
        self.assertEqual([c["picked"] for c in out["children"]], [False, True, False])
        self.assertIn("highest validation span F1", out["rule"])
        self.assertIn("answer key", out["rule"])

    def test_classification_macro_f1_then_accuracy(self):
        out = pick_seed([_child(1, 1, macro_f1=0.6, accuracy=0.9), _child(2, 2, macro_f1=0.7, accuracy=0.8)], task_type="classification")
        self.assertEqual((out["metric_id"], out["picked_experiment_id"]), ("macro_f1", 2))
        out = pick_seed([_child(1, 1, accuracy=0.9), _child(2, 2, accuracy=0.8)], task_type="classification")
        self.assertEqual((out["metric_id"], out["picked_experiment_id"]), ("accuracy", 1))

    def test_other_tasks_pick_lowest_loss(self):
        out = pick_seed([_child(1, 1, eval_loss=1.2), _child(2, 2, eval_loss=0.9), _child(3, 3, eval_loss=1.0)], task_type="causal_lm")
        self.assertEqual((out["metric_id"], out["picked_experiment_id"]), ("eval_loss", 2))
        self.assertIn("lowest validation loss", out["rule"])

    def test_tie_goes_to_lower_run_id(self):
        out = pick_seed([_child(7, 3, span_f1=0.8), _child(5, 1, span_f1=0.8), _child(9, 2, span_f1=0.7)], task_type="token_classification")
        self.assertEqual(out["picked_experiment_id"], 5)
        out = pick_seed([_child(7, 3, eval_loss=0.5), _child(5, 1, eval_loss=0.5)], task_type=None)
        self.assertEqual(out["picked_experiment_id"], 5)

    def test_missing_metric_is_listed_not_picked(self):
        out = pick_seed([_child(1, 1), _child(2, 2, span_f1=0.5)], task_type="token_classification")
        self.assertEqual(out["picked_experiment_id"], 2)
        self.assertEqual(out["n_scored"], 1)
        skipped = out["children"][0]
        self.assertIsNone(skipped["value"])
        self.assertIn("no validation span F1", skipped["skipped_reason"])

    def test_nothing_to_pick_by(self):
        out = pick_seed([_child(1, 1), _child(2, 2)], task_type="token_classification")
        self.assertIsNone(out["picked_experiment_id"])
        self.assertIsNone(selection_text(out))

    def test_metric_override_must_be_a_validation_metric(self):
        out = pick_seed([_child(1, 1, span_f1=0.9, eval_loss=0.5), _child(2, 2, span_f1=0.8, eval_loss=0.2)], task_type="token_classification", metric="eval_loss")
        self.assertEqual(out["picked_experiment_id"], 2)
        with self.assertRaises(ValueError):
            pick_seed([_child(1, 1)], metric="test_f1")

    def test_validation_values_read_only_validation_keys(self):
        report = {
            "task_metrics": {"span_set": {"f1": 0.83}},
            "final_eval_macro_f1": 0.7,
            "final_eval_accuracy": 0.9,
            "final_eval_loss": None,
            "test_f1": 0.99,
        }
        self.assertEqual(
            validation_values(report, fallback_eval_loss=0.4),
            {"span_f1": 0.83, "macro_f1": 0.7, "accuracy": 0.9, "eval_loss": 0.4},
        )
        self.assertEqual(validation_values(None), {})

    def test_selection_text_names_seed_count_metric_values(self):
        out = pick_seed([_child(10, 42, span_f1=0.81), _child(11, 43, span_f1=0.84)], task_type="token_classification")
        text = selection_text(out)
        self.assertIn("Picked seed 43 (run #11) of 2 by validation span F1", text)
        self.assertIn("seed 42 0.81, seed 43 0.84", text)
        self.assertIn("not used to pick", text)

    def test_span_card_has_seed_selection_line(self):
        card = build_span_model_card(entity_types=["phone"], training={"train_records": 3}, seed_selection="Picked seed 43 (run #11) of 2.")
        self.assertIn("- **Seed selection:** Picked seed 43 (run #11) of 2.", card)
        self.assertNotIn("Seed selection", build_span_model_card(entity_types=["phone"]))


# ── Endpoints ───────────────────────────────────────────────────────────


def _model_dir(root: Path, *, span: bool) -> None:
    model = root / "model"
    model.mkdir(parents=True)
    if span:
        labels = ["O", "B-phone", "I-phone"]
        (model / "config.json").write_text(json.dumps({
            "architectures": ["BertForTokenClassification"],
            "id2label": {str(i): l for i, l in enumerate(labels)},
        }))
        (model / "label_space.json").write_text(json.dumps({"labels": labels, "entity_types": ["phone"]}))
    else:
        (model / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (model / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 64}))
    (model / "model.safetensors").write_bytes(b"w")


def _seed_group(pid: int, root: Path, *, task_type: str, reports: list[dict], evals: bool = True) -> tuple[int, list[int]]:
    group = uuid.uuid4().hex

    async def _go():
        async with async_session_factory() as db:
            leader = Experiment(
                project_id=pid, name="group", status=ExperimentStatus.COMPLETED, base_model="base/model",
                seed_group_id=group, config={"task_type": task_type, "max_seq_length": 64},
            )
            db.add(leader)
            await db.flush()
            ids = []
            for i, report in enumerate(reports):
                out = root / f"seed{i}"
                _model_dir(out, span=task_type == "token_classification")
                (out / "training_report.json").write_text(json.dumps({"task_type": task_type, **report}))
                child = Experiment(
                    project_id=pid, name=f"seed {i}", status=ExperimentStatus.COMPLETED, base_model="base/model",
                    seed_group_id=group, seed_value=40 + i, output_dir=str(out),
                    config={"task_type": task_type, "max_seq_length": 64},
                )
                db.add(child)
                await db.flush()
                if evals and task_type == "token_classification":
                    # A test score that would pick a DIFFERENT seed: must be ignored.
                    db.add(EvalResult(
                        experiment_id=child.id, dataset_name="test", eval_type="f1",
                        metrics={"f1": 0.9 - i * 0.1, "precision": 0.9, "recall": 0.9, "total": 5, "scoring_mode": "span_set",
                                 "per_class": {"phone": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "tp": 4, "fp": 0, "fn": 1}}},
                    ))
                ids.append(child.id)
            await db.commit()
            return leader.id, ids

    return asyncio.run(_go())


def _new_project() -> int:
    return client.post("/api/projects", json={"name": f"seed-pick-{time.time_ns()}", "description": "x"}).json()["id"]


class SeedGroupEndpointTests(unittest.TestCase):
    def test_pick_uses_validation_not_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = _new_project()
            leader, ids = _seed_group(pid, Path(tmp), task_type="token_classification", reports=[
                {"task_metrics": {"span_set": {"f1": 0.70}}},
                {"task_metrics": {"span_set": {"f1": 0.80}}},
                {"task_metrics": {}},
            ])
            resp = client.get(f"/api/projects/{pid}/seed-groups/{leader}/pick")
            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertEqual(body["metric_id"], "span_f1")
            self.assertEqual(body["picked_experiment_id"], ids[1], "validation F1, not the test F1 (which favours seed 0)")
            self.assertEqual(body["n_seeds"], 3)
            self.assertIsNone(body["children"][2]["value"])
            self.assertIsNone(body["recorded"])

    def test_pick_errors(self):
        pid = _new_project()
        self.assertEqual(client.get(f"/api/projects/{pid}/seed-groups/999999/pick").status_code, 404)

        async def _single():
            async with async_session_factory() as db:
                exp = Experiment(project_id=pid, name="one", status=ExperimentStatus.COMPLETED, base_model="b", config={})
                db.add(exp)
                await db.commit()
                return exp.id

        single = asyncio.run(_single())
        self.assertEqual(client.get(f"/api/projects/{pid}/seed-groups/{single}/pick").status_code, 400)
        with tempfile.TemporaryDirectory() as tmp:
            leader, _ = _seed_group(pid, Path(tmp), task_type="causal_lm", reports=[{"final_eval_loss": 1.0}])
            self.assertEqual(client.get(f"/api/projects/{pid}/seed-groups/{leader}/pick?metric=test_f1").status_code, 400)

    def test_ship_span_seed_exports_with_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = _new_project()
            leader, ids = _seed_group(pid, Path(tmp), task_type="token_classification", reports=[
                {"task_metrics": {"span_set": {"f1": 0.70}}},
                {"task_metrics": {"span_set": {"f1": 0.80}}},
            ])
            resp = client.post(f"/api/projects/{pid}/seed-groups/{leader}/ship", json={"run_smoke_tests": False})
            self.assertEqual(resp.status_code, 201, resp.text)
            body = resp.json()
            self.assertEqual(body["picked_experiment_id"], ids[1])
            export = body["export"]
            self.assertEqual((export["status"], export["format"], export["experiment_id"]), ("completed", "huggingface", ids[1]))
            self.assertIn("Picked seed 41", body["selection"]["text"])

            run_dir = Path(export["output_path"]) / f"run-{export['run_id']}"
            card = (run_dir / "model" / "README.md").read_text()
            self.assertIn("**Seed selection:** Picked seed 41 (run #%d) of 2 by validation span F1" % ids[1], card)
            self.assertIn("seed 40 0.7, seed 41 0.8", card)
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["seed_selection"]["picked_experiment_id"], ids[1])
            self.assertEqual(manifest["seed_selection"]["export_id"], export["id"])
            self.assertIn("Seed selection: Picked seed 41", (run_dir / "RELEASE_NOTES.md").read_text())

            recorded = client.get(f"/api/projects/{pid}/seed-groups/{leader}/pick").json()["recorded"]
            self.assertEqual(recorded["picked_experiment_id"], ids[1])
            self.assertEqual(recorded["export_format"], "huggingface")

    def test_ship_other_task_puts_selection_in_manifest_and_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = _new_project()
            leader, ids = _seed_group(pid, Path(tmp), task_type="causal_lm", reports=[
                {"final_eval_loss": 1.1}, {"final_eval_loss": 0.9}, {"final_eval_loss": 0.9},
            ])
            resp = client.post(f"/api/projects/{pid}/seed-groups/{leader}/ship", json={"run_smoke_tests": False})
            self.assertEqual(resp.status_code, 201, resp.text)
            body = resp.json()
            self.assertEqual(body["picked_experiment_id"], ids[1], "tie on loss → lower run id")
            export = body["export"]
            self.assertEqual(export["status"], "completed", export)
            run_dir = Path(export["output_path"]) / f"run-{export['run_id']}"
            notes = (run_dir / "RELEASE_NOTES.md").read_text()
            self.assertIn("Seed selection: Picked seed 41 (run #%d) of 3 by validation loss (lower is better" % ids[1], notes)
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["seed_selection"]["metric_id"], "eval_loss")
            self.assertFalse((run_dir / "model" / "README.md").exists())

    def test_ship_without_any_metric_is_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = _new_project()
            leader, _ = _seed_group(pid, Path(tmp), task_type="token_classification", reports=[{}, {}], evals=False)
            resp = client.post(f"/api/projects/{pid}/seed-groups/{leader}/ship", json={"run_smoke_tests": False})
            self.assertEqual(resp.status_code, 400, resp.text)

    def test_eval_summary_flags_the_leader(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = _new_project()
            leader, ids = _seed_group(pid, Path(tmp), task_type="token_classification", reports=[
                {"task_metrics": {"span_set": {"f1": 0.70}}}, {"task_metrics": {"span_set": {"f1": 0.80}}},
            ])
            self.assertTrue(client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": leader}).json()["seed_group_leader"])
            self.assertFalse(client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": ids[0]}).json()["seed_group_leader"])


if __name__ == "__main__":
    unittest.main()
