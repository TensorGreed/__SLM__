"""Packaging a passage reader (a causal LM trained with retrieved passages):

* the model card is built only from the run's own evidence — its lift
  check record, passage-prompt report, training report, adapter config and
  the training file (tallied only while its sha256 matches the report);
* the headline claim follows the paired CI verdict, a run with no lift
  check says so, and a retrieval change since training is called out;
* HF export of a passage run writes README.md into model/ and records a
  ``reader_package``; a plain causal-LM run still ships without a card;
* who filtered / revised a teacher-target run's answers is frozen in its
  passage-prompt report (``teacher_provenance``: only cache records whose
  answer is the trained one count) and shown on the card, with the
  same-judge caveat; an older run falls back to the verified cache;
* seed siblings (same base model, byte-identical train / val files, same
  config bar the seed) get a per-seed table + the across-seed interval,
  only for siblings scored exactly like this run.
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
        self.assertIn("389 answers were left out of training", card)
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


# ── Filter / revise provenance ──────────────────────────────────────────

RET = {"k": 3}


def _cache_rec(question, answer, *, revised=False, dropped=False, abstained=False, judge="ollama:judge:12b"):
    rec = {"question": question, "answer": answer, "system_prompt": "Passages…", "teacher": "ollama:teacher:120b",
           "abstained": abstained, "dropped": dropped,
           "filter": {"judge": judge, "verdict": "wrong" if dropped else "correct", "keep": "correct"}}
    if revised:
        rec["revised"] = True
        rec["original_answer"] = "first draft"
        rec["revisions"] = [{"round": 1, "answer": answer, "critic": "ollama:critic", "judge": judge,
                             "verdict": rec["filter"]["verdict"]}]
    return rec


def _answers(*recs):
    from app.services.teacher_passage_answers_service import answer_key

    return {answer_key(r["question"], RET): r for r in recs}


class TeacherProvenanceTests(unittest.TestCase):
    def test_teacher_rows_count_revisions_kept_and_still_dropped(self):
        from app.services.teacher_passage_answers_service import teacher_rows

        answers = _answers(
            _cache_rec("q1", "a1 [1]"),
            _cache_rec("q2", "a2 [1]", revised=True),
            _cache_rec("q3", "a3", revised=True, dropped=True),
            _cache_rec("q4", "I don't know.", revised=True, dropped=True, abstained=True),
            _cache_rec("q5", "a5", dropped=True),
        )
        rows, counts = teacher_rows([{"question": f"q{i}", "answer": "ref"} for i in range(1, 6)], answers, RET)
        self.assertEqual(len(rows), 2)
        self.assertEqual((counts["filtered"], counts["revised_kept"], counts["filtered_revised"],
                          counts["filtered_revised_abstained"]), (3, 1, 2, 1))

    def test_provenance_reads_matching_records_only(self):
        from app.services.teacher_passage_answers_service import teacher_provenance, teacher_rows

        answers = _answers(_cache_rec("q1", "a1 [1]"), _cache_rec("q2", "a2 [1]", revised=True))
        rows, _ = teacher_rows([{"question": "q1", "answer": "r"}, {"question": "q2", "answer": "r"}], answers, RET)
        self.assertEqual(teacher_provenance(rows, answers, RET), {
            "filter": {"judge": "ollama:judge:12b", "keep": "correct", "judged": 2},
            "revise": {"critic": "ollama:critic", "judge": "ollama:judge:12b", "rounds": 1, "kept": 1},
        })
        # A later flow rewrote one answer: the cache no longer describes this run.
        drifted = _answers(_cache_rec("q1", "a1 [1]"), _cache_rec("q2", "a different answer [2]", revised=True))
        self.assertIsNone(teacher_provenance(rows, drifted, RET))
        # A row the cache never held.
        self.assertIsNone(teacher_provenance(rows + [{"question": "q9", "answer": "x"}], answers, RET))

    def test_provenance_survives_a_normaliser_upgrade(self):
        from app.services.teacher_passage_answers_service import normalize_teacher_answer, teacher_provenance

        cached = "Yes, it is taxable【1†L1-L4】."
        self.assertNotEqual(normalize_teacher_answer(cached), cached, "the cached form is one the normaliser rewrites")
        answers = _answers(_cache_rec("q1", cached))
        # The run trained on the raw (pre-fix) form.
        self.assertIsNotNone(teacher_provenance([{"question": "q1", "answer": cached}], answers, RET))

    def test_mixed_judges_are_listed_and_no_filter_is_none(self):
        from app.services.teacher_passage_answers_service import teacher_provenance

        answers = _answers(_cache_rec("q1", "a"), _cache_rec("q2", "b", judge="ollama:other"))
        rows = [{"question": "q1", "answer": "a"}, {"question": "q2", "answer": "b"}]
        self.assertEqual(teacher_provenance(rows, answers, RET)["filter"]["judge"], ["ollama:judge:12b", "ollama:other"])
        bare = {k: {**v, "filter": None} for k, v in answers.items()}
        self.assertEqual(teacher_provenance(rows, bare, RET), {"filter": None, "revise": None})

    def test_report_freezes_filter_and_revise(self):
        from app.services.passage_prompts_service import _apply_teacher_targets
        from app.services.teacher_passage_answers_service import cache_path

        pid = 990_000 + (time.time_ns() % 9_999)
        recs = [_cache_rec("q1", "a1 [1]"), _cache_rec("q2", "a2 [1]", revised=True), _cache_rec("q3", "a3", dropped=True)]
        path = cache_path(pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        from app.services.teacher_passage_answers_service import answer_key

        path.write_text("".join(json.dumps({**r, "key": answer_key(r["question"], RET), "retrieval": RET}) + "\n" for r in recs))
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train.jsonl"
            train.write_text("".join(json.dumps({"question": r["question"], "answer": "ref"}) + "\n" for r in recs))
            block = _apply_teacher_targets(pid, retrieval=RET, train_file=train, val_file=None, output_dir=Path(tmp))
            saved = json.loads((Path(tmp) / "passage_prompts" / "report.json").read_text())
        self.assertEqual(saved["filter"], {"judge": "ollama:judge:12b", "keep": "correct", "judged": 2})
        self.assertEqual(saved["revise"]["kept"], 1)
        self.assertEqual(block["counts"]["train"]["filtered"], 1)

    def test_card_shows_filter_revise_and_same_judge_caveat(self):
        report = _passage_report(
            counts={"train": {"rows_out": 2779, "abstained": 272, "filtered": 389, "revised_kept": 242,
                              "filtered_revised": 386, "filtered_revised_abstained": 69}},
            filter={"judge": "ollama:judge:12b", "keep": "correct", "judged": 2507},
            revise={"critic": "ollama:teacher:120b", "judge": "ollama:judge:12b", "rounds": 1, "kept": 242},
        )
        card = build_reader_model_card(reader_record=_record(), passage_report=report)
        self.assertIn("**Filter:** `ollama:judge:12b` graded", card)
        self.assertIn("only answers graded correct were kept", card)
        self.assertIn("up to 1 revision round: `ollama:teacher:120b` critiqued it", card)
        self.assertIn("242 revisions passed and are training targets; 386 still failed and stayed out "
                      "(69 of them because the revision said it didn't know).", card)
        self.assertIn("revisions written under critique, not first drafts", card)
        self.assertIn("The same model (`ollama:judge:12b`) filtered the training answers and judges the test", card)
        other = build_reader_model_card(reader_record=_record(judge="ollama:another"), passage_report=report)
        self.assertNotIn("filtered the training answers and judges the test", other)


# ── Seed siblings ───────────────────────────────────────────────────────


def _seed_record(score, diff, lo, hi, bar=0.5701, **overrides):
    rec = _record()
    rec["arms"] = {**rec["arms"], "base_with_passages": {**rec["arms"]["base_with_passages"], "score": bar},
                   "finetune_with_passages": {**rec["arms"]["finetune_with_passages"], "score": score}}
    rec["evidence"] = {**rec["evidence"], "mean_diff": diff, "ci_low": lo, "ci_high": hi}
    rec.update(overrides)
    return rec


class SeedTableTests(unittest.TestCase):
    def _runs(self):
        return [
            {"experiment_id": 162, "seed": 44, "val_loss": 0.531, "record": None},
            {"experiment_id": 160, "seed": 42, "val_loss": 0.526, "record": _seed_record(0.6369, 0.0668, 0.003, 0.131)},
            {"experiment_id": 161, "seed": 43, "val_loss": 0.5263, "record": _seed_record(0.6592, 0.0891, 0.026, 0.152)},
        ]

    def test_table_and_across_seed_interval(self):
        card = build_reader_model_card(experiment_id=162, reader_record=_seed_record(0.6879, 0.1178, 0.0539, 0.1818),
                                       seed_runs=self._runs())
        self.assertIn("one of 3 runs trained on the identical training and validation files", card)
        rows = [l for l in card.splitlines() if l.startswith("| 4") or l.startswith("| **4")]
        self.assertEqual(rows, [
            "| 42 | #160 | 0.637 | +0.067 (+0.003 to +0.131) | 0.5260 |",
            "| 43 | #161 | 0.659 | +0.089 (+0.026 to +0.152) | 0.5263 |",
            "| **44** | **#162 (this model)** | **0.688** | **+0.118 (+0.054 to +0.182)** | **0.5310** |",
        ])
        self.assertIn("mean score is 0.661 (range 0.637 to 0.688), +0.091 against the bar "
                      "(95% CI across seeds +0.028 to +0.155), so the gain holds across seeds", card)
        self.assertIn("All 3 seeds beat the bar.", card)
        self.assertLess(card.index("### Across seeds"), card.index("## Prompt format"))

    def test_siblings_scored_differently_or_unscored_are_named_not_tabled(self):
        runs = self._runs()
        runs[1]["record"] = _seed_record(0.7, 0.1, 0.05, 0.15, bar=0.60)  # another bar
        runs.append({"experiment_id": 170, "seed": 45, "val_loss": 0.53, "record": None})
        card = build_reader_model_card(experiment_id=162, reader_record=_seed_record(0.6879, 0.1178, 0.0539, 0.1818),
                                       seed_runs=runs)
        self.assertNotIn("| 42 | #160", card)
        self.assertIn("run #160 (scored on a different split, judge, retrieval or bar)", card)
        self.assertIn("run #170 (no lift check)", card)
        self.assertIn("Across the 2 seeds", card)

    def test_no_table_without_two_comparable_seeds(self):
        rec = _seed_record(0.6879, 0.1178, 0.0539, 0.1818)
        lone = [{"experiment_id": 162, "seed": 44, "record": None}, {"experiment_id": 160, "seed": 42, "record": None}]
        self.assertNotIn("### Across seeds", build_reader_model_card(experiment_id=162, reader_record=rec, seed_runs=lone))
        self.assertNotIn("### Across seeds", build_reader_model_card(experiment_id=162, seed_runs=self._runs()),
                         "no lift check on this run → no seed table")

    def test_export_finds_siblings_by_files_and_config(self):
        from app.database import async_session_factory
        from app.services.reader_model_package_service import seed_siblings

        config = {"task_type": "causal_lm", "passage_prompts": True, "passage_prompts_targets": "teacher"}
        with tempfile.TemporaryDirectory() as tmp:
            pid = _new_project()

            def _run(name, seed, train_sha="a" * 64, **cfg):
                d = Path(tmp) / name
                d.mkdir()
                (d / "training_report.json").write_text(json.dumps({
                    "final_eval_loss": 0.5, "reproducibility": {"seed": seed, "train_file_sha256": train_sha,
                                                                "val_file_sha256": "b" * 64}}))
                return _seed_run(pid, d, {**config, "seed": seed, "_runtime": {"x": seed}, **cfg})

            me = _run("me", 44)
            sib = _run("sib", 42)
            _run("other-data", 43, train_sha="c" * 64)
            _run("other-lr", 41, learning_rate=1e-4)
            _run("same-seed", 44)
            not_passage = _run("plain", 40, passage_prompts=False)

            async def _go():
                async with async_session_factory() as db:
                    exp = await db.get(Experiment, me)
                    return await seed_siblings(db, pid, exp)

            runs = asyncio.run(_go())
            self.assertEqual(sorted(r["experiment_id"] for r in runs), [me, sib])
            self.assertNotIn(not_passage, [r["experiment_id"] for r in runs])
            self.assertEqual({r["seed"] for r in runs}, {42, 44})


if __name__ == "__main__":
    unittest.main()
