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
        # ~400 optimizer steps: 228 rows / 16 = 15 steps per epoch → 27 epochs;
        # a big set still gets the 5-epoch floor, a tiny one the 40 cap.
        self.assertEqual(tc._auto_epochs(228, 16), 27.0)
        self.assertEqual(tc._auto_epochs(50000, 16), 5.0)
        self.assertEqual(tc._auto_epochs(20, 16), 40.0)
        self.assertTrue(tc._coerce_bool(None, True) and tc._coerce_bool("false", True) is False)


class TrainScriptWiringTests(unittest.TestCase):
    def test_train_py_knows_the_task_type(self):
        import train

        self.assertIn("token_classification", train.SUPPORTED_TASK_TYPES)
        self.assertEqual(train._normalize_task_type("token_classification", []), "token_classification")


if __name__ == "__main__":
    unittest.main()


class _swap_module:
    """Replace one ``sys.modules`` entry for a block and restore just that
    entry afterwards (unlike ``mock.patch.dict``, which also removes every
    module imported inside the block)."""

    def __init__(self, name, module):
        self.name, self.module = name, module

    def __enter__(self):
        self.saved = sys.modules.get(self.name)
        sys.modules[self.name] = self.module
        return self.module

    def __exit__(self, *exc):
        if self.saved is None:
            sys.modules.pop(self.name, None)
        else:
            sys.modules[self.name] = self.saved
        return False


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
        # Swap only this one module: patch.dict(sys.modules) would also drop
        # every module first imported inside the block (numpy, tokenizers),
        # and those refuse a second import in the same process.
        with _swap_module("train_token_classification", fake_mod), \
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


class LiftCheckSpanEvalTests(unittest.TestCase):
    """"Evaluate this run" on a token-classification run: no base-model lift
    (the base is an untrained head) — the run is scored on its own, on the
    test examples and the answer key when the project has one."""

    def test_a_token_classification_run_is_no_longer_skipped(self):
        from types import SimpleNamespace

        from app.services.post_training_eval_service import auto_lift_skip_reason, is_token_classification

        exp = SimpleNamespace(config={"task_type": "token_classification", "base_model": "microsoft/deberta-v3-base"}, status="COMPLETED", output_dir="/tmp/nowhere", seed_group_id=None, seed_value=None)
        self.assertTrue(is_token_classification(exp))
        self.assertNotEqual(auto_lift_skip_reason(exp, SimpleNamespace(runtime_config={})), "token_classification_no_baseline")

    def test_span_block_reports_the_weakest_types(self):
        from types import SimpleNamespace

        from app.services.post_training_eval_service import span_block

        result = SimpleNamespace(id=7, dataset_name="gold_test", metrics={
            "precision": 0.8, "recall": 0.7, "f1": 0.75, "total": 44,
            "per_class": {
                "email": {"f1": 1.0, "tp": 27, "fn": 0},
                "bank_account": {"f1": 0.21, "tp": 3, "fn": 6},
                "person_name": {"f1": 0.65, "tp": 33, "fn": 13},
                "cvv": {"f1": 0.65, "tp": 9, "fn": 1},
            },
        })
        block = span_block(result)
        self.assertEqual((block["f1"], block["rows"], block["eval_result_id"]), (0.75, 44, 7))
        self.assertEqual([t["type"] for t in block["weakest_types"]], ["bank_account", "person_name", "cvv", "email"],
                         "lowest F1 first; a tie goes to the type with more gold spans")
        self.assertEqual(block["weakest_types"][1]["support"], 46)

    def test_run_span_evaluation_scores_test_and_answer_key(self):
        import asyncio
        from types import SimpleNamespace
        from unittest import mock

        from app.services import post_training_eval_service as pt

        calls = []

        async def fake_heldout(**kw):
            calls.append((kw["dataset_name"], kw["eval_type"], kw["answer_judge_enabled"], kw["max_samples"]))
            return SimpleNamespace(id=len(calls), dataset_name=kw["dataset_name"], metrics={"precision": 0.9, "recall": 0.8, "f1": 0.85, "total": 44, "per_class": {}})

        class Db:
            async def commit(self):
                return None

        async def has_key(db, pid):
            return True

        with mock.patch("app.services.evaluation_service.run_heldout_evaluation", fake_heldout), \
                mock.patch.object(pt, "project_has_answer_key", has_key):
            out = asyncio.run(pt.run_span_evaluation(Db(), project_id=3, experiment=SimpleNamespace(id=59)))
        self.assertEqual(calls, [("test", "f1", False, pt.SPAN_EVAL_MAX_ROWS), ("gold_test", "f1", False, pt.SPAN_EVAL_MAX_ROWS)])
        self.assertEqual(out["kind"], "span_eval")
        self.assertEqual((out["test"]["dataset_name"], out["answer_key"]["dataset_name"]), ("test", "gold_test"))


class PreflightTests(unittest.TestCase):
    def test_encoder_base_model_accepts_token_classification(self):
        from app.services import training_preflight_service as pf

        tables = [v for v in vars(pf).values() if isinstance(v, list) and v and isinstance(v[0], dict) and "family" in v[0]]
        caps = next(c for table in tables for c in table if c.get("family") == "encoder_only")
        self.assertIn("token_classification", caps["supported_task_types"])
        from app.services.capability_contract_service import SUPPORTED_TRAINING_TASK_TYPES

        self.assertIn("token_classification", SUPPORTED_TRAINING_TASK_TYPES)


class SpanEvalPlumbingTests(unittest.TestCase):
    """What the held-out eval needs from a span-extraction project: the
    requested task profile survives an adapter that doesn't list it, the
    extraction handler reads `entities` as the reference, and the split
    manifest carries the recipe's scoring mode."""

    def test_explicit_task_profile_wins_over_the_adapter_list(self):
        from app.services.data_adapter_service import resolve_task_profile_for_adapter

        self.assertEqual(resolve_task_profile_for_adapter("default-canonical", requested_task_profile="structured_extraction"), "structured_extraction")
        self.assertNotEqual(resolve_task_profile_for_adapter("default-canonical", requested_task_profile="auto"), "structured_extraction")

    def test_extraction_handler_reads_entities_as_the_reference(self):
        import json

        from app.services.eval_task_handler_service import StructuredExtractionHandler

        h = StructuredExtractionHandler()
        ents = [{"type": "email", "start": 12, "end": 28, "text": "ada@example.test"}]
        self.assertEqual(json.loads(h._reference_as_string({"text": TEXT, "entities": ents})), {"entities": ents})
        self.assertEqual(json.loads(h._reference_as_string({"text": TEXT, "entities_json": json.dumps({"entities": ents})})), {"entities": ents})

    def test_recipe_output_schema_for_span_extraction(self):
        import asyncio
        from types import SimpleNamespace

        from app.services.dataset_service import _recipe_output_schema

        class FakeDb:
            def __init__(self, project):
                self.project = project

            async def get(self, model, pid):
                return self.project

        span = asyncio.run(_recipe_output_schema(FakeDb(SimpleNamespace(selected_recipe={"recipe_id": "span-extraction"})), 1))
        self.assertEqual(span, {"scoring_mode": "span_set", "required": ["entities"]})
        self.assertIsNone(asyncio.run(_recipe_output_schema(FakeDb(SimpleNamespace(selected_recipe=None)), 1)))


# ── Long texts: overlapping windows (Epic P1) ────────────────────────────
#
# A tiny word-level fast tokenizer and a rule-based "model" make the
# windowing testable offline: every word is one token, so window edges are
# easy to place, and the "model" labels a token by its word.

LABELS = ["O", "B-phone", "I-phone", "B-person_name", "I-person_name"]
ID2LABEL = dict(enumerate(LABELS))


def _tiny_tokenizer(words):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = WhitespaceSplit()
    tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
    )


class _RuleModel:
    """Labels a token by its word: phone-shaped words → B-phone, "Ada" →
    B-person_name, "Lovelace" → I-person_name, everything else O. Counts
    the windows it was shown and their longest length."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.windows = 0
        self.longest = 0

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None, **_):
        import re
        from types import SimpleNamespace

        import torch

        rows = []
        for ids in input_ids.tolist():
            self.windows += 1
            self.longest = max(self.longest, len(ids))
            row = []
            for word in self.tokenizer.convert_ids_to_tokens(ids):
                label = 0
                if re.fullmatch(r"555-\d{4}", word or ""):
                    label = 1
                elif word == "Ada":
                    label = 3
                elif word == "Lovelace":
                    label = 4
                row.append([5.0 if i == label else 0.0 for i in range(len(LABELS))])
            rows.append(row)
        return SimpleNamespace(logits=torch.tensor(rows))


def _multi_page_document(pages: int = 4, filler_words: int = 40):
    """A document of ``pages`` pages: filler, then a phone and a name near
    the end of each page — most of them far past the first window."""
    parts, spans, cursor = [], [], 0
    for page in range(pages):
        filler = " ".join(f"w{page}x{i}" for i in range(filler_words))
        phone = f"555-01{page:02d}"
        chunk = f"{filler} call {phone} or Ada Lovelace ."
        if page:
            chunk = "\n\n" + chunk
        p_start = cursor + chunk.index(phone)
        n_start = cursor + chunk.index("Ada Lovelace")
        spans.append({"type": "phone", "start": p_start, "end": p_start + len(phone)})
        spans.append({"type": "person_name", "start": n_start, "end": n_start + len("Ada Lovelace")})
        parts.append(chunk)
        cursor += len(chunk)
    return "".join(parts), spans


class LongTextWindowTests(unittest.TestCase):
    def test_a_multi_page_document_is_tagged_end_to_end(self):
        text, gold = _multi_page_document(pages=4, filler_words=40)
        tok = _tiny_tokenizer((text + " short call 555-0999 .").split())
        model = _RuleModel(tok)
        self.assertGreater(len(tok(text)["input_ids"]), 150, "the document is ~10 windows long")
        pred = tc.predict_spans([text, "short call 555-0999 ."], model, tok, ID2LABEL, max_length=16, stride=4, device="cpu")
        got = [(s["type"], s["start"], s["end"]) for s in pred[0]]
        self.assertEqual(got, [(s["type"], s["start"], s["end"]) for s in gold], "every page's entities, exact offsets")
        self.assertEqual([s["text"] for s in pred[1]], ["555-0999"], "a short text in the same batch still works")
        self.assertGreater(model.windows, 10)
        self.assertLessEqual(model.longest, 16, "no window is longer than max_length")

    def test_a_name_cut_by_a_window_edge_is_one_span(self):
        # Place "Ada Lovelace" so a window boundary falls between the two words
        # for several strides; the merged prediction must be one span.
        for lead in range(10, 18):
            text = " ".join(f"f{i}" for i in range(lead)) + " Ada Lovelace " + " ".join(f"g{i}" for i in range(20))
            tok = _tiny_tokenizer(text.split())
            pred = tc.predict_spans([text], _RuleModel(tok), tok, ID2LABEL, max_length=12, stride=3, device="cpu")
            self.assertEqual([s["text"] for s in pred[0]], ["Ada Lovelace"], f"lead={lead}")

    def test_merge_takes_the_window_where_a_token_is_most_central(self):
        offsets_a = [(0, 0), (0, 2), (3, 5), (6, 8), (0, 0)]   # token (6,8) at A's edge
        offsets_b = [(0, 0), (3, 5), (6, 8), (9, 11), (0, 0)]  # token (6,8) central in B
        merged = tc.merge_window_scores(1, [offsets_a, offsets_b], [0, 0], [
            ["-", "a1", "a2", "edge", "-"],
            ["-", "b1", "centre", "b3", "-"],
        ])
        offsets, scores = merged[0]
        self.assertEqual(offsets, [(0, 2), (3, 5), (6, 8), (9, 11)])
        self.assertEqual(scores[2], "centre")

    def test_a_window_starting_inside_a_span_continues_it(self):
        label_to_id = {label: i for i, label in enumerate(LABELS)}
        span = [{"type": "person_name", "start": 10, "end": 22}]
        # Window 1 holds the span's first word; window 2 starts at its second.
        first = tc.spans_to_token_labels([(0, 0), (10, 13), (14, 22)], span, label_to_id)
        second = tc.spans_to_token_labels([(0, 0), (14, 22), (23, 25)], span, label_to_id)
        self.assertEqual([LABELS[i] if i >= 0 else None for i in first], [None, "B-person_name", "I-person_name"])
        self.assertEqual([LABELS[i] if i >= 0 else None for i in second], [None, "I-person_name", "O"])

    def test_window_stride_defaults_and_bounds(self):
        self.assertEqual(tc.window_stride(512), 128)
        self.assertEqual(tc.window_stride(512, 64), 64)
        self.assertEqual(tc.window_stride(16, 100), 6, "always below half the window")


class LongTextTrainingTests(unittest.TestCase):
    """The real training path on a tiny random BERT (built here, no
    download): long rows become several windows, the report says so, and
    the saved tokenizer's max length is the window, so the HF pipeline
    windows the exported model."""

    def test_long_rows_train_in_windows(self):
        try:
            import torch  # noqa: F401
            from transformers import BertConfig, BertModel
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"torch/transformers unavailable: {exc}")
        import json
        import tempfile

        text, gold = _multi_page_document(pages=3, filler_words=60)
        short = "please call 555-0123 today ."
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tok = _tiny_tokenizer((text + " " + short).split())
            base = tmp / "tiny-bert"
            tok.save_pretrained(base)
            cfg = BertConfig(
                vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                intermediate_size=64, max_position_embeddings=128,
            )
            # An encoder without a head, like a hub base model.
            BertModel(cfg).save_pretrained(base)
            rows = [{"text": text, "entities": gold}] * 3 + [
                {"text": short, "entities": [{"type": "phone", "start": 12, "end": 20}]}
            ] * 3
            train = tmp / "train.jsonl"
            train.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            report = tc.run_token_classification(
                base_model=str(base),
                config={"max_seq_length": 64, "num_epochs": 1, "auto_epochs": False, "batch_size": 4},
                output_dir=tmp / "out", model_dir=tmp / "out" / "model",
                train_file=train, val_file=train, started_at="now",
            )
            windowing = report["windowing"]
            self.assertEqual(windowing["max_length"], 64)
            self.assertEqual(windowing["stride"], 16)
            self.assertEqual(windowing["train_long_rows"], 3, "each long row became several windows")
            self.assertGreater(windowing["train_windows"], 6)
            self.assertIn("span_set", report["task_metrics"], "validation merged the windows back into texts")
            saved = json.loads((tmp / "out" / "model" / "tokenizer_config.json").read_text())
            self.assertEqual(saved["model_max_length"], 64)


class ExportWindowTests(unittest.TestCase):
    def test_export_caps_an_unlimited_tokenizer_length_for_token_classifiers(self):
        import json
        import tempfile

        from app.services.export_service import cap_token_classifier_window

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "config.json").write_text(json.dumps({"architectures": ["DebertaV2ForTokenClassification"]}))
            (d / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 1e30, "do_lower_case": False}))
            note = cap_token_classifier_window(d, 512)
            self.assertEqual(note["model_max_length"], 512)
            saved = json.loads((d / "tokenizer_config.json").read_text())
            self.assertEqual((saved["model_max_length"], saved["do_lower_case"]), (512, False))
            self.assertIsNone(cap_token_classifier_window(d, 256), "a real window is left alone")
            (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
            (d / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 1e30}))
            self.assertIsNone(cap_token_classifier_window(d, 512), "only token classifiers")


class OcrNoiseAugmentTests(unittest.TestCase):
    def test_noisy_copies_keep_every_span_offset(self):
        examples = [(TEXT, [
            {"type": "email", "start": TEXT.index("ada@"), "end": TEXT.index("ada@") + len("ada@example.test")},
            {"type": "phone", "start": TEXT.index("555-0101"), "end": TEXT.index("555-0101") + 8},
        ])]
        out = tc.augment_with_ocr_noise(examples, rate=0.3, copies=3, seed=1)
        self.assertEqual(len(out), 3)
        for noisy, spans in out:
            self.assertEqual(len(noisy), len(TEXT), "length-preserving")
            self.assertNotEqual(noisy, TEXT)
            for sp in spans:
                self.assertEqual(noisy[sp["start"]:sp["end"]], sp["text"], "span text follows the noised characters")
        changed = sum(a != b for noisy, _ in out for a, b in zip(noisy, TEXT))
        self.assertGreater(changed, 3)
        self.assertEqual(tc.augment_with_ocr_noise(examples, rate=0.3, copies=3, seed=1), out, "deterministic per seed")
        self.assertTrue(set(tc.TRAIN_CONFUSIONS) < set(tc.TRAIN_CONFUSIONS) | {"h", "g", "Z", "@", "-"})
        for held_out in ("h", "g", "Z", "@", "-"):
            self.assertNotIn(held_out, tc.TRAIN_CONFUSIONS, "kept out of training so measured noise is partly unseen")

    def test_training_reports_the_augmentation(self):
        try:
            from transformers import BertConfig, BertModel
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"transformers unavailable: {exc}")
        import json
        import tempfile

        text = "please call 555-0123 or mail ada@example.test today ."
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tok = _tiny_tokenizer(text.split())
            base = tmp / "tiny-bert"
            tok.save_pretrained(base)
            BertModel(BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                 intermediate_size=64, max_position_embeddings=128)).save_pretrained(base)
            rows = [{"text": text, "entities": [{"type": "phone", "start": 12, "end": 20}]}] * 4
            train = tmp / "train.jsonl"
            train.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            report = tc.run_token_classification(
                base_model=str(base),
                config={"max_seq_length": 64, "num_epochs": 1, "auto_epochs": False, "batch_size": 4,
                        "ocr_noise_augment_rate": 0.3, "ocr_noise_augment_copies": 2},
                output_dir=tmp / "out", model_dir=tmp / "out" / "model",
                train_file=train, val_file=train, started_at="now",
            )
            self.assertEqual(report["augmentation"]["copies"], 2)
            self.assertEqual(report["augmentation"]["rows_added"], 8)
            self.assertEqual(report["windowing"]["train_windows"], 12, "4 rows + 8 noisy copies")
