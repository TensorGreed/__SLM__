"""Per-type score thresholds tuned on the validation split (P1t)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from app.services import span_threshold_service as svc


def _s(t, a, b, score=1.0):
    return {"type": t, "start": a, "end": b, "score": score}


class TuneTests(unittest.TestCase):
    def test_raises_a_threshold_only_when_it_helps(self):
        gold = [[_s("acct", 0, 5)], [_s("acct", 0, 5)], [_s("name", 0, 3)]]
        pred = [
            [_s("acct", 0, 5, 0.95), _s("acct", 10, 14, 0.35)],  # a weak false positive
            [_s("acct", 0, 5, 0.9)],
            [_s("name", 0, 3, 0.4)],  # a weak TRUE positive — a threshold would hurt
        ]
        th, report = svc.tune_from_scored(gold, pred, ["acct", "name"])
        self.assertEqual(th["acct"], 0.4, "lowest threshold that drops the weak false positive")
        self.assertEqual(th["name"], 0.0, "no gain → stays at 0")
        self.assertEqual((report["acct"]["f1_before"], report["acct"]["f1_after"]), (0.8, 1.0))
        self.assertEqual(svc.apply_thresholds(pred, th)[0], [_s("acct", 0, 5, 0.95)])

    def test_decoding_reads_only_non_zero_thresholds(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "thresholds.json").write_text(json.dumps({"per_type": {"acct": 0.7, "name": 0.0}}))
            self.assertEqual(svc.thresholds_for_decoding(tmp), {"acct": 0.7})
            self.assertEqual(svc.thresholds_for_decoding(Path(tmp) / "nowhere"), {})


class EvalAppliesThresholdsTests(unittest.TestCase):
    def test_token_classification_inference_passes_the_models_thresholds(self):
        import types as _types

        from app.services import evaluation_service as ev

        seen = {}

        def fake_predict(texts, model, tokenizer, id2label, **kw):
            seen.update(kw)
            return [[] for _ in texts]

        class Fake:
            @staticmethod
            def from_pretrained(path, **kw):
                return Fake()

            def to(self, d):
                return self

            def eval(self):
                return self

        saved = sys.modules.get("train_token_classification")
        sys.modules["train_token_classification"] = _types.SimpleNamespace(predict_spans=fake_predict)
        try:
            with mock.patch("transformers.AutoTokenizer", Fake), mock.patch("transformers.AutoModelForTokenClassification", Fake), \
                    mock.patch.object(svc, "thresholds_for_decoding", return_value={"acct": 0.7}):
                _, runtime = ev._run_token_classification_inference(
                    {"model_path": "/x/model", "id2label": {0: "O"}, "max_length": 512},
                    [{"prompt": "p", "reference": "{}", "structured_input": "t"}],
                )
        finally:
            if saved is None:
                sys.modules.pop("train_token_classification", None)
            else:
                sys.modules["train_token_classification"] = saved
        self.assertEqual(seen["thresholds"], {"acct": 0.7})
        self.assertEqual(runtime["inference"]["thresholds_applied"], {"acct": 0.7})


if __name__ == "__main__":
    unittest.main()
