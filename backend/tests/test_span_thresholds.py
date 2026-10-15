"""Per-type score thresholds tuned on the validation split (P1t), with
per-type recall floors (P1u): a floored type keeps at least its floor's
recall; an unreachable floor gets a recall_bias suggestion from re-decoding
the same scored split."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services import span_threshold_service as svc  # noqa: E402

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


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
        self.assertEqual(seen["recall_bias"], 0.0, "no thresholds.json → no bias")
        self.assertEqual(runtime["inference"]["recall_bias_applied"], 0.0)

    def test_recall_bias_for_decoding_reads_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "thresholds.json").write_text(json.dumps({"per_type": {}, "recall_bias": 0.2}))
            self.assertEqual(svc.recall_bias_for_decoding(tmp), 0.2)
            self.assertEqual(svc.recall_bias_for_decoding(Path(tmp) / "nowhere"), 0.0)


# ── Recall floors (pure) ─────────────────────────────────────────────────


class FloorTests(unittest.TestCase):
    # acct: 4 reference spans; predictions at scores .95 .9 .5 (true) + .45 .35 (false)
    GOLD = [[_s("acct", 0, 5)], [_s("acct", 0, 5)], [_s("acct", 0, 5)], [_s("acct", 0, 5)]]
    PRED = [
        [_s("acct", 0, 5, 0.95), _s("acct", 10, 14, 0.45)],
        [_s("acct", 0, 5, 0.9), _s("acct", 10, 14, 0.35)],
        [_s("acct", 0, 5, 0.5)],
        [],  # one span the model never predicts
    ]

    def test_no_floor_keeps_the_f1_behaviour(self):
        th, report = svc.tune_from_scored(self.GOLD, self.PRED, ["acct"])
        self.assertEqual(th["acct"], 0.5, "F1: drop both false positives, keep the .5 true one")
        self.assertIsNone(report["acct"]["floor"])
        self.assertIsNone(report["acct"]["floor_status"])
        th2, _ = svc.tune_from_scored(self.GOLD, self.PRED, ["acct"], floors={})
        self.assertEqual(th2, th)

    def test_a_met_floor_takes_the_best_precision_threshold_that_keeps_it(self):
        th, report = svc.tune_from_scored(self.GOLD, self.PRED, ["acct"], floors={"acct": 0.5})
        # .5 → .9 all reach precision 1.0 and meet the floor; .5 keeps more recall (.75 vs .5)
        self.assertEqual(th["acct"], 0.5)
        r = report["acct"]
        self.assertEqual((r["floor"], r["floor_status"]), (0.5, "met"))
        self.assertEqual((r["recall_before"], r["recall_after"]), (0.75, 0.75))
        self.assertEqual((r["precision_before"], r["precision_after"]), (0.6, 1.0))

        # Precision and recall tied across thresholds → the HIGHEST one (F1 stops at the first gain).
        gold, pred = [[_s("a", 0, 3)]], [[_s("a", 0, 3, 0.95), _s("a", 5, 8, 0.65)]]
        self.assertEqual(svc.tune_from_scored(gold, pred, ["a"])[0]["a"], 0.7)
        self.assertEqual(svc.tune_from_scored(gold, pred, ["a"], floors={"a": 1.0})[0]["a"], 0.9)

    def test_a_floor_can_hold_a_threshold_below_the_f1_choice(self):
        _, report = svc.tune_from_scored(self.GOLD, self.PRED, ["acct"], floors={"acct": 0.75})
        self.assertEqual(report["acct"]["threshold"], 0.5, "a .75 floor allows .5 (keeps 3 of 4)")
        gold = [[_s("a", 0, 3)], [_s("a", 0, 3)]]
        pred = [
            [_s("a", 0, 3, 0.95), _s("a", 5, 8, 0.6), _s("a", 9, 11, 0.6), _s("a", 12, 14, 0.6)],
            [_s("a", 0, 3, 0.45), _s("a", 5, 8, 0.35)],
        ]
        self.assertEqual(svc.tune_from_scored(gold, pred, ["a"])[0]["a"], 0.7, "F1 drops the weak true span too")
        th, report = svc.tune_from_scored(gold, pred, ["a"], floors={"a": 1.0})
        self.assertEqual(th["a"], 0.4, "the floor keeps the .45 true span: best precision with recall 1.0")
        self.assertEqual(report["a"]["recall_after"], 1.0)

    def test_default_floor_applies_to_types_without_their_own(self):
        gold = self.GOLD + [[_s("name", 0, 3)]]
        pred = self.PRED + [[_s("name", 0, 3, 0.4)]]
        _, report = svc.tune_from_scored(gold, pred, ["acct", "name"], floors={"_default": 0.9, "name": 0.5})
        self.assertEqual((report["acct"]["floor"], report["acct"]["floor_status"]), (0.9, "not_reachable"))
        self.assertEqual(report["acct"]["threshold"], 0.0, "thresholds only drop spans — stay at 0")
        self.assertEqual((report["name"]["floor"], report["name"]["floor_status"]), (0.5, "met"))
        _, report = svc.tune_from_scored(gold, pred, ["acct", "zip"], floors={"zip": 0.9})
        self.assertEqual(report["zip"]["floor_status"], "no_support")

    def test_unreachable_floor_suggests_the_smallest_bias_and_its_cost(self):
        missing = _s("acct", 0, 5, 0.3)
        extra = _s("acct", 20, 24, 0.2)
        calls = []

        def decode(bias):
            calls.append(bias)
            pred = [list(p) for p in self.PRED]
            if bias >= 0.15:
                pred[3] = [missing]  # the missed span appears
            if bias >= 0.1:
                pred[2] = pred[2] + [extra]  # but extra spans appear sooner
            return pred

        result = svc.tune_scored_split(self.GOLD, decode, ["acct"], floors={"acct": 1.0})
        self.assertEqual(result["report"]["acct"]["floor_status"], "not_reachable")
        self.assertEqual(result["floors_not_reachable"], ["acct"])
        sug = result["recall_bias_suggestion"]
        self.assertEqual(sug["recall_bias"], 0.15, "smallest grid bias that meets the floor")
        per = sug["per_type"]["acct"]
        self.assertEqual((per["recall_before"], per["recall_at_bias"]), (0.75, 1.0))
        self.assertEqual((per["precision_before"], per["precision_at_bias"]), (0.6, round(4 / 7, 4)))
        self.assertEqual(per["precision_cost"], round(0.6 - round(4 / 7, 4), 4))
        self.assertEqual([t["recall_bias"] for t in sug["tried"]], [0.05, 0.1, 0.15], "stops at the first that works")
        self.assertEqual(result["recall_bias"], 0.0, "suggested, not applied")
        self.assertEqual(result["thresholds"]["acct"], 0.0)

        applied = svc.tune_scored_split(self.GOLD, decode, ["acct"], floors={"acct": 1.0}, apply_recall_bias=True)
        self.assertEqual(applied["recall_bias"], 0.15)
        r = applied["report"]["acct"]
        self.assertEqual((r["floor_status"], r["recall_before"], r["recall_after"]), ("met", 0.75, 1.0))
        self.assertEqual(applied["floors_met"], ["acct"])

    def test_floor_not_reachable_at_any_bias(self):
        result = svc.tune_scored_split(self.GOLD, lambda bias: self.PRED, ["acct"], floors={"acct": 1.0})
        sug = result["recall_bias_suggestion"]
        self.assertIsNone(sug["recall_bias"])
        self.assertIsNone(sug["per_type"]["acct"]["bias"])
        self.assertEqual(len(sug["tried"]), len(svc.BIAS_GRID))
        applied = svc.tune_scored_split(self.GOLD, lambda bias: self.PRED, ["acct"], floors={"acct": 1.0}, apply_recall_bias=True)
        self.assertEqual(applied["recall_bias"], 0.0)

    def test_normalise_floors_validates(self):
        self.assertEqual(svc.normalise_floors({"Bank Account": 0.9, "_default": 0.5}), {"bank_account": 0.9, "_default": 0.5})
        with self.assertRaises(ValueError):
            svc.normalise_floors({"email": 1.5})
        with self.assertRaises(ValueError):
            svc.normalise_floors({"email": "high"})
        with self.assertRaisesRegex(ValueError, "Unknown entity type"):
            svc.normalise_floors({"zip": 0.9}, ["email"])
        self.assertEqual(svc.floor_for({"_default": 0.8}, "x"), 0.8)
        self.assertIsNone(svc.floor_for({}, "x"))


# ── Scored once, decoded many times ──────────────────────────────────────


class DecodeRefactorTests(unittest.TestCase):
    def test_score_then_decode_equals_predict_spans(self):
        import train_token_classification as tc
        from test_span_packaging import SCORES, TEXT, _ScoreModel
        from test_train_token_classification import ID2LABEL, _tiny_tokenizer

        tok = _tiny_tokenizer(TEXT.split())
        model = _ScoreModel(tok, SCORES)
        scores = tc.score_tokens([TEXT], model, tok, max_length=32, device="cpu")
        for kwargs in ({}, {"with_scores": True}, {"recall_bias": 0.2}, {"thresholds": {"person_name": 0.8}}):
            self.assertEqual(
                tc.decode_token_scores([TEXT], scores, ID2LABEL, **kwargs),
                tc.predict_spans([TEXT], model, tok, ID2LABEL, max_length=32, device="cpu", **kwargs),
                kwargs,
            )


# ── Endpoints ────────────────────────────────────────────────────────────


def _new_project() -> int:
    return client.post("/api/projects", json={"name": f"span-floors-{time.time_ns()}", "description": "x"}).json()["id"]


class RecallFloorEndpointTests(unittest.TestCase):
    def test_get_put_validate_and_suggest_without_applying(self):
        pid = _new_project()
        base = f"/api/projects/{pid}/span-recall-floors"
        got = client.get(base).json()
        self.assertEqual((got["floors"], got["default"], got["types"]), ({}, None, []))

        client.post(f"/api/projects/{pid}/compliance/apply", json={"bundle_ids": ["pci_dss"]})
        got = client.get(base).json()
        self.assertIn("credit_card", got["types"])
        self.assertEqual(got["suggestions"]["credit_card"], {"floor": 0.95, "regimes": ["pci_dss"]})
        self.assertEqual(got["floors"], {}, "suggestions are never applied")

        put = client.put(base, json={"floors": {"credit_card": 0.95, "cvv": 0.9}, "default": 0.8})
        self.assertEqual(put.status_code, 200, put.text)
        self.assertEqual((put.json()["floors"], put.json()["default"]), ({"credit_card": 0.95, "cvv": 0.9}, 0.8))
        self.assertEqual(client.get(base).json()["floors"], {"credit_card": 0.95, "cvv": 0.9})

        bad = client.put(base, json={"floors": {"credit_card": 1.2}})
        self.assertEqual(bad.status_code, 422)
        unknown = client.put(base, json={"floors": {"shoe_size": 0.9}})
        self.assertEqual(unknown.status_code, 422)
        self.assertIn("shoe_size", unknown.text)
        self.assertEqual(client.get(base).json()["default"], 0.8, "a rejected PUT changes nothing")


class TuneEndpointTests(unittest.TestCase):
    """The whole tune path with the model call patched: a word-score model
    (test_span_packaging) tags the validation split; floors come from the
    project setting; thresholds.json records floors + report."""

    def _setup(self, tmp):
        from test_span_packaging import SCORES, TEXT, _ScoreModel, _seed_run, _span_run_dir
        from test_train_token_classification import _tiny_tokenizer

        pid = _new_project()
        run_dir = _span_run_dir(Path(tmp))
        exp_id = _seed_run(pid, run_dir)
        prepared = settings.DATA_DIR / "projects" / str(pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)

        def ent(t, word):
            start = TEXT.index(word)
            return {"type": t, "start": start, "end": start + len(word), "text": word}

        # Gold: both phones; "Ada Lovelace" is a name, "Bob" is not.
        row = {"text": TEXT, "entities": [ent("phone", "555-0101"), ent("phone", "maybe-0102"), ent("person_name", "Ada Lovelace")]}
        (prepared / "val.jsonl").write_text(json.dumps(row) + "\n")
        (prepared / "train.jsonl").write_text(json.dumps(row) + "\n")
        tok = _tiny_tokenizer(TEXT.split())
        patch = mock.patch.object(svc, "_load_model", return_value=(_ScoreModel(tok, SCORES), tok, "cpu"))
        return pid, exp_id, run_dir / "model" / "thresholds.json", patch

    def test_floors_from_the_project_and_the_written_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid, exp_id, th_file, patch = self._setup(tmp)
            put = client.put(f"/api/projects/{pid}/span-recall-floors", json={"floors": {"phone": 1.0, "person_name": 1.0}})
            self.assertEqual(put.status_code, 200, put.text)
            with patch:
                resp = client.post(f"/api/projects/{pid}/span-thresholds/tune", json={"experiment_id": exp_id})
            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            # person_name: floor met; best precision keeping recall 1.0 is the highest
            # threshold under Ada Lovelace's .905 (F1 alone would stop at .7).
            self.assertEqual(body["report"]["person_name"]["threshold"], 0.9)
            self.assertEqual(body["report"]["person_name"]["floor_status"], "met")
            self.assertEqual(body["report"]["person_name"]["precision_after"], 1.0)
            # phone: maybe-0102 is never tagged at bias 0 → not reachable; bias .2 tags it.
            self.assertEqual(body["floors_not_reachable"], ["phone"])
            self.assertEqual(body["recall_bias_suggestion"]["recall_bias"], 0.2)
            self.assertEqual(body["recall_bias"], 0.0)
            saved = json.loads(th_file.read_text())
            self.assertEqual(saved["recall_floors"], {"phone": 1.0, "person_name": 1.0})
            self.assertEqual(saved["per_type"]["person_name"], 0.9)
            self.assertEqual(saved["recall_bias"], 0.0)
            self.assertEqual(saved["tuned"]["report"]["phone"]["floor_status"], "not_reachable")

            with patch:
                applied = client.post(
                    f"/api/projects/{pid}/span-thresholds/tune",
                    json={"experiment_id": exp_id, "floors": {"phone": 1.0}, "apply_recall_bias": True},
                ).json()
            self.assertEqual(applied["recall_bias"], 0.2)
            self.assertEqual(applied["report"]["phone"]["floor_status"], "met")
            self.assertEqual((applied["report"]["phone"]["recall_before"], applied["report"]["phone"]["recall_after"]), (0.5, 1.0))
            saved = json.loads(th_file.read_text())
            self.assertEqual((saved["recall_bias"], saved["recall_floors"]), (0.2, {"phone": 1.0}))

            from app.services.span_model_package_service import build_span_model_card

            card = build_span_model_card(entity_types=["person_name", "phone"], thresholds=saved)
            self.assertIn("### Recall floors", card)
            self.assertIn("| `phone` | 1.00 | 0.50 → 1.00 |", card)
            self.assertIn("| met |", card)
            self.assertIn("`recall_bias` is 0.2", card)

    def test_bad_floors_in_the_request_are_a_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid, exp_id, _, patch = self._setup(tmp)
            with patch:
                resp = client.post(f"/api/projects/{pid}/span-thresholds/tune", json={"experiment_id": exp_id, "floors": {"phone": 2}})
            self.assertEqual(resp.status_code, 400)


class ModelCardFloorTests(unittest.TestCase):
    def test_card_says_when_a_floor_was_not_reachable(self):
        from app.services.span_model_package_service import build_span_model_card

        thresholds = {
            "recall_floors": {"iban": 0.95, "_default": 0.5}, "recall_bias": 0.0,
            "tuned": {"rows": 40, "recall_bias_suggestion": {"recall_bias": 0.3}, "report": {
                "iban": {"floor": 0.95, "floor_status": "not_reachable", "recall_before": 0.8, "recall_after": 0.8,
                         "precision_before": 0.9, "precision_after": 0.9, "threshold": 0.0},
                "email": {"floor": None, "floor_status": None, "threshold": 0.6},
            }},
        }
        card = build_span_model_card(entity_types=["iban", "email"], thresholds=thresholds)
        self.assertIn("validation split (40 texts)", card)
        self.assertIn("not reachable", card)
        self.assertIn("Types without their own floor use 0.50", card)
        self.assertNotIn("`email`", card.split("### Recall floors")[1].split("##")[0])
        self.assertIn("A `recall_bias` of 0.3 would meet the unmet floors", card)
        self.assertNotIn("### Recall floors", build_span_model_card(entity_types=["iban"], thresholds={"per_type": {}}))
        untuned = build_span_model_card(entity_types=["iban"], thresholds={"recall_floors": {"iban": 0.9}})
        self.assertIn("whether they are met is not known", untuned)


if __name__ == "__main__":
    unittest.main()
