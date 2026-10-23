"""Packaging a passage reader (a causal LM trained with retrieved passages):

* the model card is built only from the run's own evidence — its lift
  check record, passage-prompt report, training report, adapter config and
  the training file (tallied only while its sha256 matches the report);
* the headline claim follows the paired CI verdict, a run with no lift
  check says so, and a retrieval change since training is called out;
* HF export of a passage run writes README.md into model/ and records a
  ``reader_package``; a plain causal-LM run still ships without a card.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.experiment import Experiment, ExperimentStatus
from app.services.reader_model_package_service import (
    build_reader_model_card,
    package_passage_reader,
    training_file_tallies,
)

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


RETRIEVAL = {"k": 3, "reranker": None, "pool": None, "chunk_size": 600}


def _record(verdict="better", **overrides):
    record = {
        "experiment_id": 9, "base_model": "org/base-1b", "split": "test", "judge": "ollama:judge:12b",
        "retrieval": RETRIEVAL, "retrieval_label": "top-3 · 600-char passages", "n_rows": 157, "comparable": True,
        "arms": {
            "base_with_passages": {"score": 0.5701, "counts": {"correct": 82, "partial": 15, "wrong": 60}},
            "base_without_passages": {"score": 0.0732, "counts": {"correct": 6, "partial": 11, "wrong": 140}},
            "finetune_with_passages": {
                "score": 0.6879, "counts": {"correct": 100, "partial": 16, "wrong": 41},
                "retrieval": {"judged": 157, "found": 123, "wrong_missed": 28, "wrong_misread": 13},
            },
            "finetune_without_passages": {"score": 0.1847, "counts": {"correct": 23, "partial": 12, "wrong": 122}},
        },
        "evidence": {"n": 157, "better": 32, "worse": 12, "same": 113, "mean_diff": 0.1178,
                     "ci_low": 0.0539, "ci_high": 0.1818, "verdict": verdict},
    }
    record.update(overrides)
    return record


def _passage_report(**overrides):
    report = {
        "applied": True, "targets": "teacher", "teacher": "ollama:teacher:120b", "retrieval": RETRIEVAL,
        "retrieval_label": "top-3 · 600-char passages",
        "counts": {"train": {"rows_in": 3168, "rows_out": 2779, "abstained": 272, "filtered": 389},
                   "val": {"rows_out": 157}},
    }
    report.update(overrides)
    return report


class ReaderCardTests(unittest.TestCase):
    def test_full_evidence_copies_numbers_and_ci_verdict(self):
        card = build_reader_model_card(
            base_model="org/base-1b", experiment_id=9, project_name="Docs", reader_record=_record(),
            passage_report=_passage_report(),
            training={"epochs": 2.0, "total_steps": 348, "final_train_loss": 0.4458, "final_eval_loss": 0.531,
                      "reproducibility": {"seed": 44},
                      "runtime_environment": {"micro_batch": {"effective_batch": 16},
                                              "max_seq_length_fit": {"to": 2048}, "loss_masking": "completion_only"}},
            adapter={"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["v_proj", "q_proj"]},
            tallies={"generators": {"ollama:teacher:120b": 2779}, "sources": {"doc.pdf": 2779}},
            merged=True,
        )
        self.assertTrue(card.startswith("---\nlibrary_name: transformers\npipeline_tag: text-generation\nbase_model: org/base-1b"))
        self.assertIn("- merged", card)
        self.assertIn("| Base model + passages (the bar) | 0.570 (82 / 15 / 60) |", card)
        self.assertIn("| **This model + passages** | **0.688 (100 / 16 / 41)** |", card)
        self.assertIn("better on 32, worse on 12 and the same on 113: +0.118, 95% CI +0.054 to +0.182", card)
        self.assertIn("CI of the paired difference is above 0", card)
        self.assertIn("held the answer for 123 of 157 questions", card)
        self.assertIn("28 of this model's 41 wrong answers", card)
        self.assertIn("without passages it scores 0.185", card)
        self.assertIn("272 of them say the passages don't hold the answer", card)
        self.assertIn("389 were left out of training", card)
        self.assertIn("LoRA r 16, alpha 32, dropout 0.05 on `q_proj`, `v_proj` (merged into the base weights)", card)
        self.assertIn("2 epochs, 348 steps, effective batch 16, max sequence length 2048", card)
        self.assertIn("Seed 44.", card)
        self.assertIn("272 of 2,779 training answers (10%)", card)
        # Section order: result before data before training before prompt.
        order = [card.index(h) for h in ("## Result", "## Training data", "## Training\n", "## Prompt format", "## Limitations")]
        self.assertEqual(order, sorted(order))

    def test_prompt_layout_is_the_served_template(self):
        from app.services.auto_rag_service import document_preamble

        card = build_reader_model_card(reader_record=_record())
        served = document_preamble([{"payload": {"text": "x", "source_doc": "d", "chunk_id": 0}}])
        self.assertIn(served.split("\n")[0], card)
        self.assertIn(served.split("\n")[-1], card, "the instruction sentence is the served one")

    def test_no_lift_check_says_so(self):
        card = build_reader_model_card(base_model="org/base-1b", passage_report=_passage_report())
        self.assertIn("No held-out evaluation is recorded for this run", card)
        self.assertNotIn("| Base model + passages", card)
        self.assertNotIn("in-distribution", card, "no test claim without a test")

    def test_within_noise_is_not_called_better(self):
        card = build_reader_model_card(reader_record=_record(verdict="within_noise"))
        self.assertIn("not distinguishable from the bar", card)
        self.assertNotIn("better than the bar", card)

    def test_retrieval_change_and_incomparable_arms_are_called_out(self):
        record = _record(retrieval={**RETRIEVAL, "k": 5}, retrieval_label="top-5 · 600-char passages", comparable=False)
        card = build_reader_model_card(reader_record=record, passage_report=_passage_report())
        self.assertIn("trained with top-3 · 600-char passages but evaluated with top-5 · 600-char passages", card)
        self.assertIn("not scored under identical conditions", card)
        same = build_reader_model_card(reader_record=_record(), passage_report=_passage_report())
        self.assertNotIn("Retrieval differs from training", same)

    def test_reference_targets_describe_withheld_rows(self):
        report = _passage_report(targets="reference", teacher=None,
                                 counts={"train": {"rows_out": 100, "withheld": 20}})
        card = build_reader_model_card(passage_report=report)
        self.assertIn("the reference answer with the passage that holds it cited", card)
        self.assertIn("For 20 rows the passage holding the answer was withheld", card)

    def test_tallies_need_a_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(json.dumps({"generated_by": "gen", "source_doc": "a.pdf"}) + "\n")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(training_file_tallies(path, sha), {"generators": {"gen": 1}, "sources": {"a.pdf": 1}})
            self.assertEqual(training_file_tallies(path, "0" * 64), {})
            self.assertEqual(training_file_tallies(path, None), {})

    def test_plain_run_gets_no_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = SimpleNamespace(id=1, base_model="b", output_dir=tmp, config={"task_type": "causal_lm"})
            self.assertIsNone(package_passage_reader(project_id=1, experiment=exp, model_dir=Path(tmp)))
            self.assertFalse((Path(tmp) / "README.md").exists())


# ── Export ──────────────────────────────────────────────────────────────


def _reader_run_dir(root: Path) -> Path:
    """A passage-trained run dir: a weightless causal-LM model/, the
    passage-prompt report + training file, and a training report whose
    fingerprint matches that file."""
    exp_dir = root / "exp"
    model = exp_dir / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"]}))
    (model / "model.safetensors").write_bytes(b"w")
    prompts = exp_dir / "passage_prompts"
    prompts.mkdir()
    rows = [{"question": "q", "answer": "a [1]", "generated_by": "ollama:gen", "source_doc": "doc.pdf"}] * 3
    train = prompts / "train.jsonl"
    train.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (prompts / "report.json").write_text(json.dumps(_passage_report(counts={"train": {"rows_out": 3, "abstained": 1}})))
    (exp_dir / "training_report.json").write_text(json.dumps({
        "epochs": 2.0, "total_steps": 4, "train_file": str(train),
        "reproducibility": {"seed": 7, "train_file_sha256": hashlib.sha256(train.read_bytes()).hexdigest()},
    }))
    return exp_dir


def _seed_run(pid: int, output_dir: Path, config: dict) -> int:
    async def _go():
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=pid, name="reader run", status=ExperimentStatus.COMPLETED,
                base_model="org/base-1b", output_dir=str(output_dir), config=config,
            )
            db.add(exp)
            await db.commit()
            return exp.id

    return asyncio.run(_go())


def _new_project() -> int:
    return client.post("/api/projects", json={"name": f"reader-pkg-{time.time_ns()}", "description": "x"}).json()["id"]


def _export(pid: int, exp_id: int):
    from app.models.export import ExportFormat
    from app.services.export_service import create_export, run_export

    async def _go():
        async with async_session_factory() as db:
            export = await create_export(db, pid, exp_id, ExportFormat.HUGGINGFACE)
            export = await run_export(db, pid, export.id, run_smoke_tests=False)
            await db.commit()
            return export.status, export.manifest, export.output_path

    return asyncio.run(_go())


class ReaderExportTests(unittest.TestCase):
    def test_hf_export_of_a_passage_run_writes_the_card(self):
        from app.models.export import ExportStatus

        with tempfile.TemporaryDirectory() as tmp:
            pid = _new_project()
            exp_id = _seed_run(pid, _reader_run_dir(Path(tmp)), {"task_type": "causal_lm", "passage_prompts": True})
            record_path = settings.DATA_DIR / "projects" / str(pid) / "auto_rag" / "passage_reader" / f"{exp_id}.json"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(_record(experiment_id=exp_id)))

            status, manifest, output_path = _export(pid, exp_id)
            self.assertEqual(status, ExportStatus.COMPLETED)
            run_dir = Path(output_path) / manifest["run_dir"]
            card = (run_dir / "model" / "README.md").read_text()
            self.assertIn("| **This model + passages** | **0.688 (100 / 16 / 41)** |", card)
            self.assertIn("written by `ollama:gen` (3 rows)", card)
            self.assertIn("Seed 7.", card)
            self.assertEqual(manifest["reader_package"]["verdict"], "better")
            self.assertEqual(manifest["reader_package"]["judge"], "ollama:judge:12b")
            self.assertIsNone(manifest["span_package"])
            self.assertIn("README.md", {Path(a["path"]).name for a in manifest["artifacts"]})
            self.assertIn("Passage reader: README.md model card (test split", (run_dir / "RELEASE_NOTES.md").read_text())

    def test_plain_causal_lm_export_has_no_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = _new_project()
            exp_id = _seed_run(pid, _reader_run_dir(Path(tmp)), {"task_type": "causal_lm"})
            _status, manifest, output_path = _export(pid, exp_id)
            self.assertIsNone(manifest["reader_package"])
            self.assertFalse((Path(output_path) / manifest["run_dir"] / "model" / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
