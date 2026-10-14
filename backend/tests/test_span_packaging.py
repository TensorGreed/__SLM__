"""Epic P5 — packaging a span (token-classification) model:

* the model card is built only from evidence (no invented numbers; a
  section without data is left out), answer key before test examples;
* HF export of a span run writes README.md + thresholds.json into model/;
* the scored decoder: default spans unchanged, per-span scores, per-type
  thresholds, recall bias;
* POST /projects/{id}/redact masks / tags with an offsets map (the model
  loader is patched — a tiny word-level tokenizer + a rule model).
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_token_classification as tc  # noqa: E402
from test_train_token_classification import ID2LABEL, LABELS, _tiny_tokenizer  # noqa: E402

from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import EvalResult, Experiment, ExperimentStatus  # noqa: E402
from app.services import redaction_service  # noqa: E402
from app.services.span_model_package_service import (  # noqa: E402
    build_span_model_card,
    default_thresholds,
    pipeline_stride,
)

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


# ── A model whose scores are set per word ───────────────────────────────


class _ScoreModel:
    """Logits by word: ``scores[word]`` = a probability vector over LABELS
    (log-probs as logits); unknown words are confidently O."""

    def __init__(self, tokenizer, scores):
        self.tokenizer = tokenizer
        self.scores = scores

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None, **_):
        import math
        from types import SimpleNamespace

        import torch

        rows = []
        for ids in input_ids.tolist():
            row = []
            for word in self.tokenizer.convert_ids_to_tokens(ids):
                probs = self.scores.get(word) or [0.96, 0.01, 0.01, 0.01, 0.01]
                row.append([math.log(p) for p in probs])
            rows.append(row)
        return SimpleNamespace(logits=torch.tensor(rows))


SCORES = {
    "555-0101": [0.02, 0.95, 0.01, 0.01, 0.01],    # sure phone
    "Ada": [0.05, 0.01, 0.01, 0.9, 0.03],          # sure name
    "Lovelace": [0.1, 0.01, 0.01, 0.03, 0.85],
    "maybe-0102": [0.55, 0.40, 0.03, 0.01, 0.01],  # borderline phone (O wins)
    "Bob": [0.3, 0.01, 0.01, 0.65, 0.03],          # weak name
}
TEXT = "call 555-0101 or maybe-0102 and ask Ada Lovelace or Bob ."


def _predict(**kwargs):
    tok = _tiny_tokenizer(TEXT.split())
    return tc.predict_spans([TEXT], _ScoreModel(tok, SCORES), tok, ID2LABEL, max_length=32, device="cpu", **kwargs)[0]


class ScoredDecoderTests(unittest.TestCase):
    def test_scores_do_not_change_the_default_spans(self):
        plain = _predict()
        scored = _predict(with_scores=True)
        self.assertEqual([s["text"] for s in plain], ["555-0101", "Ada Lovelace", "Bob"])
        self.assertNotIn("score", plain[0], "the default output is unchanged")
        self.assertEqual(
            [(s["type"], s["start"], s["end"], s["text"]) for s in scored],
            [(s["type"], s["start"], s["end"], s["text"]) for s in plain],
        )
        by_text = {s["text"]: s for s in scored}
        self.assertAlmostEqual(by_text["555-0101"]["score"], 0.96, places=2)  # B + I mass
        self.assertAlmostEqual(by_text["Ada Lovelace"]["min_score"], 0.88, places=2)
        self.assertLess(by_text["Bob"]["score"], 0.7)

    def test_per_type_threshold_drops_weak_spans_of_that_type_only(self):
        spans = _predict(thresholds={"person_name": 0.8})
        self.assertEqual([s["text"] for s in spans], ["555-0101", "Ada Lovelace"])
        spans = _predict(min_score=0.97)
        self.assertEqual(spans, [])

    def test_recall_bias_tags_borderline_tokens(self):
        self.assertNotIn("maybe-0102", [s["text"] for s in _predict(with_scores=True)])
        spans = _predict(recall_bias=0.2)
        self.assertIn("maybe-0102", [s["text"] for s in spans])
        borderline = next(s for s in spans if s["text"] == "maybe-0102")
        self.assertEqual(borderline["type"], "phone")
        self.assertAlmostEqual(borderline["score"], 0.43, places=2, msg="scores stay unbiased")


# ── Model card ──────────────────────────────────────────────────────────


def _eval_block(label, name, f1, rows, per_type, written_by=None):
    return {
        "label": label, "dataset_name": name, "precision": f1 + 0.02, "recall": f1 - 0.02, "f1": f1,
        "rows": rows, "per_type": per_type, "written_by": written_by,
    }


class ModelCardTests(unittest.TestCase):
    def test_full_card_carries_only_the_given_numbers(self):
        card = build_span_model_card(
            entity_types=["phone", "email"],
            base_model="microsoft/deberta-v3-base",
            experiment_id=58, project_name="Sensitive data", window=512,
            training={
                "train_records": 486, "train_records_labelled": 470, "eval_records": 61, "epochs": 13.0,
                "sources": {"teacher_llm": 331, "passage_labels:ollama": 111},
                "included_types": ["synthetic"],
            },
            evaluations=[
                _eval_block("answer key", "gold_test", 0.79, 44,
                            {"phone": {"precision": 0.9, "recall": 1.0, "f1": 0.96, "tp": 20, "fp": 2, "fn": 0},
                             "email": {"precision": 1.0, "recall": 0.5, "f1": 0.67, "tp": 2, "fp": 0, "fn": 2}},
                            written_by="cloud:anthropic:claude-haiku-4-5"),
                _eval_block("test examples", "test", 0.89, 44, {}),
            ],
        )
        self.assertTrue(card.startswith("---\nlibrary_name: transformers"))
        self.assertIn("base_model: microsoft/deberta-v3-base", card)
        self.assertIn("`email`, `phone`", card)
        self.assertIn("**Training rows:** 486 (470 with at least one span)", card)
        self.assertIn("`teacher_llm` 331", card)
        self.assertLess(card.index("### Answer key (44 texts)"), card.index("### Test examples (44 texts)"))
        self.assertIn("written by `cloud:anthropic:claude-haiku-4-5`", card)
        self.assertIn("F1 0.79", card)
        self.assertIn("| `phone` | 0.90 | 1.00 | 0.96 | 20 |", card)
        self.assertIn("stride=128", card)
        self.assertIn("text[start].isspace()", card)
        self.assertIn("`email`) have unreliable per-type scores", card, "4 gold spans < 10")
        self.assertIn("synthetic or model-labelled", card)
        self.assertNotIn("no independent answer key", card)

    def test_sections_without_data_are_left_out(self):
        card = build_span_model_card(entity_types=["phone"], base_model="/local/ckpt")
        self.assertNotIn("## Evaluation", card)
        self.assertNotIn("## Training data", card)
        self.assertNotIn("base_model:", card, "a local path is not a hub id")
        self.assertIn("no held-out span evaluation recorded", card)
        self.assertNotIn("synthetic", card)
        for number in ("0.", "texts)"):
            self.assertNotIn(f"F1 {number}", card)

    def test_test_only_evaluation_says_there_is_no_answer_key(self):
        card = build_span_model_card(entity_types=["phone"], evaluations=[_eval_block("test examples", "test", 0.9, 12, {})])
        self.assertIn("no independent answer key", card)
        self.assertIn("12 texts (test examples)", card)

    def test_stride_follows_the_window(self):
        self.assertEqual(pipeline_stride(512), 128)
        self.assertEqual(pipeline_stride(64), 16)
        self.assertIn("stride=16", build_span_model_card(entity_types=["x"], window=64))

    def test_default_thresholds_keep_argmax(self):
        th = default_thresholds(["phone", "email"])
        self.assertEqual(th["per_type"], {"email": 0.0, "phone": 0.0})
        self.assertEqual((th["default_min_score"], th["recall_bias"]), (0.0, 0.0))


# ── Export + redact endpoint ────────────────────────────────────────────


def _span_run_dir(root: Path) -> Path:
    """An experiment dir with a (weightless) token-classification model/,
    a training report and the prepared train file it read."""
    exp_dir = root / "exp"
    model = exp_dir / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({
        "architectures": ["BertForTokenClassification"],
        "id2label": {str(i): l for i, l in enumerate(LABELS)},
        "max_position_embeddings": 64,
    }))
    (model / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 64}))
    (model / "label_space.json").write_text(json.dumps({"labels": LABELS, "entity_types": ["person_name", "phone"]}))
    (model / "model.safetensors").write_bytes(b"w")
    prepared = root / "prepared"
    prepared.mkdir()
    rows = [{"text": "call 555-0101", "entities": [], "source": "teacher_llm"}] * 3
    (prepared / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (prepared / "manifest.json").write_text(json.dumps({"splits": {"train": 3}, "included_types": ["synthetic"]}))
    (exp_dir / "training_report.json").write_text(json.dumps({
        "train_records": 3, "train_records_labelled": 3, "eval_records": 1, "epochs": 5.0,
        "train_file": str(prepared / "train.jsonl"),
    }))
    return exp_dir


def _seed_run(pid: int, output_dir: Path, evals=()) -> int:
    async def _go():
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=pid, name="tc run", status=ExperimentStatus.COMPLETED,
                base_model="microsoft/deberta-v3-base", output_dir=str(output_dir),
                config={"task_type": "token_classification", "max_seq_length": 64},
            )
            db.add(exp)
            await db.flush()
            for name, f1 in evals:
                db.add(EvalResult(
                    experiment_id=exp.id, dataset_name=name, eval_type="f1",
                    metrics={"precision": f1, "recall": f1, "f1": f1, "total": 7, "scoring_mode": "span_set",
                             "per_class": {"phone": {"precision": f1, "recall": f1, "f1": f1, "tp": 6, "fp": 1, "fn": 1}}},
                ))
            await db.commit()
            return exp.id

    return asyncio.run(_go())


def _new_project() -> int:
    return client.post("/api/projects", json={"name": f"span-pkg-{time.time_ns()}", "description": "x"}).json()["id"]


class ExportPackagingTests(unittest.TestCase):
    def test_hf_export_writes_model_card_and_thresholds(self):
        from app.models.export import ExportFormat, ExportStatus
        from app.services.export_service import create_export, run_export

        with tempfile.TemporaryDirectory() as tmp:
            pid = _new_project()
            exp_id = _seed_run(pid, _span_run_dir(Path(tmp)), evals=[("test", 0.81), ("gold_test", 0.74)])

            async def _export():
                async with async_session_factory() as db:
                    export = await create_export(db, pid, exp_id, ExportFormat.HUGGINGFACE)
                    export = await run_export(db, pid, export.id, run_smoke_tests=False)
                    await db.commit()
                    return export.status, export.manifest

            status, manifest = asyncio.run(_export())
            self.assertEqual(status, ExportStatus.COMPLETED)
            model_dir = Path(manifest["run_dir"]) / "model"
            card = (model_dir / "README.md").read_text()
            self.assertLess(card.index("### Answer key (7 texts)"), card.index("### Test examples (7 texts)"))
            self.assertIn("F1 0.74", card)
            self.assertIn("**Training rows:** 3", card)
            self.assertIn("`teacher_llm` 3", card)
            self.assertIn("stride=16", card, "window 64 → stride 16")
            thresholds = json.loads((model_dir / "thresholds.json").read_text())
            self.assertEqual(thresholds["per_type"], {"person_name": 0.0, "phone": 0.0})
            self.assertEqual(manifest["span_package"]["evaluations"], ["gold_test", "test"])
            names = {Path(a["path"]).name for a in manifest["artifacts"]}
            self.assertTrue({"README.md", "thresholds.json"} <= names)


class RedactEndpointTests(unittest.TestCase):
    def setUp(self):
        redaction_service._CACHE.update(key=None, bundle=None)

    def _patched(self):
        def _fake_load(artifacts):
            tok = _tiny_tokenizer(TEXT.split())
            return redaction_service.TaggerBundle(_ScoreModel(tok, SCORES), tok, artifacts["id2label"], 32, "cpu")

        return mock.patch.object(redaction_service, "load_tagger", side_effect=_fake_load)

    def test_mask_tag_and_spans(self):
        with tempfile.TemporaryDirectory() as tmp, self._patched():
            pid = _new_project()
            exp_id = _seed_run(pid, _span_run_dir(Path(tmp)))
            body = client.post(f"/api/projects/{pid}/redact", json={"text": TEXT}).json()
            self.assertEqual(body["experiment_id"], exp_id, "defaults to the newest span run")
            self.assertEqual(body["text"], "call [PHONE] or maybe-0102 and ask [PERSON_NAME] or [PERSON_NAME] .")
            first = body["offsets"][0]
            self.assertEqual(TEXT[first["start"]:first["end"]], "555-0101")
            self.assertEqual(body["text"][first["out_start"]:first["out_end"]], "[PHONE]")
            self.assertTrue(all("score" in s for s in body["spans"]))

            tagged = client.post(f"/api/projects/{pid}/redact", json={"text": TEXT, "mode": "tag", "min_score": 0.8}).json()
            self.assertEqual(tagged["text"], "call <phone>555-0101</phone> or maybe-0102 and ask <person_name>Ada Lovelace</person_name> or Bob .")

            biased = client.post(f"/api/projects/{pid}/redact", json={"text": TEXT, "mode": "spans", "recall_bias": 0.2}).json()
            self.assertEqual(biased["text"], TEXT)
            self.assertIn("maybe-0102", [s["text"] for s in biased["spans"]])
            self.assertEqual(biased["settings"]["recall_bias"], 0.2)

    def test_saved_thresholds_apply(self):
        with tempfile.TemporaryDirectory() as tmp, self._patched():
            pid = _new_project()
            run_dir = _span_run_dir(Path(tmp))
            (run_dir / "model" / "thresholds.json").write_text(json.dumps({"per_type": {"person_name": 0.8}, "recall_bias": 0.0}))
            _seed_run(pid, run_dir)
            body = client.post(f"/api/projects/{pid}/redact", json={"text": TEXT, "mode": "spans"}).json()
            self.assertEqual([s["text"] for s in body["spans"]], ["555-0101", "Ada Lovelace"])
            self.assertTrue(body["settings"]["thresholds_file"])

    def test_no_span_run_is_a_404(self):
        pid = _new_project()
        resp = client.post(f"/api/projects/{pid}/redact", json={"text": "hello"})
        self.assertEqual(resp.status_code, 404)

    def test_apply_redaction_skips_overlaps(self):
        out = redaction_service.apply_redaction(
            "abcdef", [{"type": "x", "start": 0, "end": 3}, {"type": "y", "start": 2, "end": 5}], "mask",
        )
        self.assertEqual(out["text"], "[X]def")
        self.assertEqual(len(out["offsets"]), 1)


if __name__ == "__main__":
    unittest.main()
