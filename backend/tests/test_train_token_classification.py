"""Token classification for span extraction — the pure parts:
rows → (text, spans) with offset validation, BIO label space, char spans
↔ token labels round-trip, strict span P/R/F1 (the span_set match), and
train.py knowing the task type."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_token_classification as tc  # noqa: E402


TEXT = "Call Ada at ada@example.test or 555-0101 about card 4111 1111 1111 1111."


class RowParsingTests(unittest.TestCase):
    def test_row_shapes_and_offset_validation(self):
        spans = [
            {"type": "email", "start": 12, "end": 28, "text": "ada@example.test"},
            {"type": "phone", "start": 32, "end": 40, "text": "555-0101"},
            {"type": "credit_card", "start": 52, "end": 71, "text": "4111 1111 1111 1111"},
            {"type": "person", "start": 5, "end": 8, "text": "Bob"},  # text mismatch → dropped
            {"type": "phone", "start": 70, "end": 99, "text": ""},  # runs off the end → dropped
        ]
        text, kept = tc.row_to_example({"text": TEXT, "entities": spans})
        self.assertEqual(text, TEXT)
        self.assertEqual([s["type"] for s in kept], ["email", "phone", "credit_card"])
        # JSON-string entities and the canonical input/expected shape.
        import json

        text2, kept2 = tc.row_to_example({"input": TEXT, "expected": json.dumps({"entities": spans[:2]})})
        self.assertEqual(len(kept2), 2)
        text3, kept3 = tc.row_to_example({"text": TEXT, "entities_json": "```json\n" + json.dumps(spans[:1]) + "\n```"})
        self.assertEqual(kept3[0]["type"], "email")
        self.assertIsNone(tc.row_to_example({"entities": spans}))
        # Overlapping spans: the first wins (BIO can't hold both).
        _, kept4 = tc.row_to_example({"text": TEXT, "entities": [{"type": "a", "start": 12, "end": 28, "text": "ada@example.test"}, {"type": "b", "start": 16, "end": 28, "text": "example.test"}]})
        self.assertEqual([s["type"] for s in kept4], ["a"])
        self.assertEqual(tc.build_label_space([(TEXT, kept)]), ["O", "B-credit_card", "I-credit_card", "B-email", "I-email", "B-phone", "I-phone"])


class BioRoundTripTests(unittest.TestCase):
    def test_spans_to_labels_and_back(self):
        # Fake word-level tokenizer offsets (with a special token at each end).
        words = [(0, 4), (5, 8), (9, 11), (12, 28), (29, 31), (32, 40), (41, 46), (47, 51), (52, 56), (57, 61), (62, 66), (67, 71), (71, 72)]
        offsets = [(0, 0), *words, (0, 0)]
        spans = [
            {"type": "email", "start": 12, "end": 28},
            {"type": "credit_card", "start": 52, "end": 71},
        ]
        labels = tc.build_label_space([(TEXT, spans)])
        l2i = {l: i for i, l in enumerate(labels)}
        i2l = {i: l for l, i in l2i.items()}
        ids = tc.spans_to_token_labels(offsets, spans, l2i)
        self.assertEqual(ids[0], -100)
        self.assertEqual(ids[-1], -100)
        self.assertEqual(i2l[ids[4]], "B-email")
        self.assertEqual([i2l[i] for i in ids[9:13]], ["B-credit_card", "I-credit_card", "I-credit_card", "I-credit_card"])
        self.assertEqual(i2l[ids[1]], "O")
        back = tc.token_labels_to_spans(offsets, ids, i2l, TEXT)
        self.assertEqual([(s["type"], s["start"], s["end"], s["text"]) for s in back], [
            ("email", 12, 28, "ada@example.test"),
            ("credit_card", 52, 71, "4111 1111 1111 1111"),
        ])
        # A stray I- opens a span; a type change closes one.
        stray = [l2i["O"]] * len(offsets)
        stray[4] = l2i["I-email"]
        stray[5] = l2i["I-credit_card"]
        self.assertEqual([(s["type"], s["start"]) for s in tc.token_labels_to_spans(offsets, stray, i2l, TEXT)], [("email", 12), ("credit_card", 29)])

    def test_span_prf_is_strict_and_per_type(self):
        gold = [[{"type": "email", "start": 12, "end": 28}, {"type": "phone", "start": 32, "end": 40}]]
        pred = [[{"type": "email", "start": 12, "end": 28}, {"type": "phone", "start": 32, "end": 39}, {"type": "card", "start": 52, "end": 71}]]
        prf = tc.span_prf(gold, pred)
        self.assertEqual((prf["tp"], prf["fp"], prf["fn"]), (1, 2, 1))
        self.assertEqual(prf["precision"], 0.3333)
        self.assertEqual(prf["recall"], 0.5)
        self.assertEqual(prf["per_type"]["email"]["f1"], 1.0)
        self.assertEqual(prf["per_type"]["phone"]["f1"], 0.0)
        self.assertEqual(prf["per_type"]["card"]["fp"], 1)
        self.assertEqual(tc._auto_epochs(100), 15.0)
        self.assertEqual(tc._auto_epochs(5000), 3.0)


class TrainScriptWiringTests(unittest.TestCase):
    def test_train_py_knows_the_task_type(self):
        import train

        self.assertIn("token_classification", train.SUPPORTED_TASK_TYPES)
        self.assertEqual(train._normalize_task_type("token_classification", []), "token_classification")


if __name__ == "__main__":
    unittest.main()


class EvaluationPathTests(unittest.TestCase):
    """The held-out eval routes a token-classification checkpoint to its own
    inference (tag the raw input, emit span_set JSON) instead of generation."""

    def test_detects_a_token_classification_checkpoint_under_model(self):
        import json
        import tempfile

        from app.services.evaluation_service import _resolve_token_classifier_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.assertIsNone(_resolve_token_classifier_artifacts(str(run)))
            (run / "model").mkdir()
            (run / "model" / "config.json").write_text(json.dumps({
                "architectures": ["DebertaV2ForTokenClassification"],
                "id2label": {"0": "O", "1": "B-email", "2": "I-email"},
                "max_position_embeddings": 512,
            }))
            found = _resolve_token_classifier_artifacts(str(run))
            self.assertEqual(found["model_path"], str(run / "model"))
            self.assertEqual(found["id2label"], {0: "O", 1: "B-email", 2: "I-email"})
            (run / "model" / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"], "id2label": {"0": "LABEL_0"}}))
            self.assertIsNone(_resolve_token_classifier_artifacts(str(run)))

    def test_inference_tags_the_raw_input_and_emits_span_set_json(self):
        import json
        from unittest import mock

        from app.services import evaluation_service as ev

        fake_spans = [[{"type": "email", "start": 12, "end": 28, "text": "ada@example.test"}], []]
        seen: dict = {}

        def fake_predict(texts, model, tokenizer, id2label, **kw):
            seen["texts"] = list(texts)
            return fake_spans

        class FakeTok:
            @staticmethod
            def from_pretrained(path):
                return "tok"

        class FakeModel:
            @staticmethod
            def from_pretrained(path, **kw):
                return FakeModel()

            def to(self, device):
                return self

            def eval(self):
                return self

        import types

        fake_mod = types.SimpleNamespace(predict_spans=fake_predict)
        with mock.patch.dict(sys.modules, {"train_token_classification": fake_mod}), \
                mock.patch("transformers.AutoTokenizer", FakeTok), \
                mock.patch("transformers.AutoModelForTokenClassification", FakeModel):
            preds, runtime = ev._run_token_classification_inference(
                {"model_path": "/x/model", "id2label": {0: "O"}, "max_length": 512},
                [
                    {"prompt": "Extract entities…\n\n" + TEXT, "reference": json.dumps({"entities": fake_spans[0]}), "structured_input": TEXT},
                    {"prompt": "Extract entities…\n\nNothing here.", "reference": json.dumps({"entities": []}), "structured_input": "Nothing here."},
                ],
            )
        self.assertEqual(seen["texts"], [TEXT, "Nothing here."], "the raw input is tagged, not the wrapped prompt")
        self.assertEqual(json.loads(preds[0]["prediction"]), {"entities": fake_spans[0]})
        self.assertEqual(json.loads(preds[1]["prediction"]), {"entities": []})
        self.assertEqual(runtime["inference"]["mode"], "token_classification")
        self.assertEqual(runtime["inference"]["rows"], 2)


class LiftCheckSkipTests(unittest.TestCase):
    def test_lift_check_skips_a_token_classification_run(self):
        from types import SimpleNamespace

        from app.services.post_training_eval_service import auto_lift_skip_reason

        exp = SimpleNamespace(config={"task_type": "token_classification", "base_model": "microsoft/deberta-v3-base"}, status="COMPLETED", output_dir="/tmp/x", seed_group_id=None, seed_value=None)
        reason = auto_lift_skip_reason(exp, SimpleNamespace(runtime_config={}))
        self.assertEqual(reason, "token_classification_no_baseline")
