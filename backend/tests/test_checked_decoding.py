"""Checked decoding (Epic Y1 + Y4): verify-and-retry at eval time, opt-in.

Pins:
  * the setting's validation (422 at the API) and resolution (per-eval override > project setting);
  * the verifier interface: a verifier never receives the reference / answer key — a sentinel in
    the reference and in reference-like row fields never reaches a (spy) verifier;
  * attempt-1 equivalence: with the setting off the eval loop is unchanged, and with it on a row
    that passes keeps its first answer with one generation;
  * resample (sampled, seeded per row / attempt) and repair (one chat follow-up turn listing the
    failed checks) paths with a stub model; none pass -> attempt 1 kept; unknown shape -> no retry;
  * verifiers by answer shape (code checks, grounding only when registered, JSON schema, label set,
    citations);
  * the lift check's base-model cache and Compare key on the policy.

No real model runs: a fake tokenizer / model stands in for transformers (CPU tensors only).
"""

from __future__ import annotations

import json
from uuid import uuid4
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from app.services import checked_decoding_service as cds

SENTINEL = "REFERENCE-SENTINEL-7f3a"

_client_cm = None


def setUpModule():  # noqa: N802 - unittest hook
    global _client_cm
    from app.main import app

    _client_cm = TestClient(app)
    _client_cm.__enter__()


def tearDownModule():  # noqa: N802 - unittest hook
    if _client_cm is not None:
        _client_cm.__exit__(None, None, None)


def _policy(**kw):
    return cds.normalize_policy({"enabled": True, **kw})


# ── policy ────────────────────────────────────────────────────────────────


class PolicyTests(unittest.TestCase):
    def test_defaults_are_off(self):
        self.assertEqual(cds.normalize_policy(None), cds.DEFAULTS)
        self.assertFalse(cds.DEFAULTS["enabled"])
        self.assertIsNone(cds.resolve_policy({}, None))
        self.assertIsNone(cds.resolve_policy(None, None))
        self.assertIsNone(cds.policy_record(cds.DEFAULTS))

    def test_bad_values_raise(self):
        for bad in (
            {"mode": "beam"}, {"max_attempts": 0}, {"max_attempts": 6}, {"max_attempts": True},
            {"max_attempts": 2.0}, {"verifiers": ["nope"]}, {"verifiers": "label_set"},
            {"sample_temperature": 0}, {"sample_temperature": 3}, {"seed": -1}, {"enabled": "yes"},
            {"unknown": 1}, "on",
        ):
            with self.subTest(bad=bad), self.assertRaises(cds.PolicyError):
                cds.normalize_policy(bad)

    def test_override_replaces_project_setting(self):
        rc = cds.with_project_policy({"other": 1}, {"enabled": True, "mode": "repair"})
        self.assertEqual(rc["other"], 1)
        self.assertEqual(cds.resolve_policy(rc)["mode"], "repair")
        self.assertIsNone(cds.resolve_policy(rc, {"enabled": False}))
        self.assertEqual(cds.resolve_policy({}, {"enabled": True, "max_attempts": 2})["max_attempts"], 2)

    def test_record_and_label(self):
        rec = cds.policy_record(_policy(mode="resample", max_attempts=4))
        self.assertEqual(rec, {"mode": "resample", "max_attempts": 4, "verifiers": "auto",
                               "sample_temperature": 0.7, "seed": 0})
        self.assertEqual(cds.policy_label(rec), "Checked decoding · up to 4 tries · resample")
        rec = cds.policy_record(_policy(mode="repair", verifiers=["label_set"]))
        self.assertEqual(rec, {"mode": "repair", "max_attempts": 3, "verifiers": ["label_set"]})

    def test_lift_override_is_always_explicit(self):
        self.assertEqual(cds.lift_policy_override({}, None), cds.DEFAULTS)
        rc = cds.with_project_policy({}, {"enabled": True})
        self.assertTrue(cds.lift_policy_override(rc, None)["enabled"])
        self.assertFalse(cds.lift_policy_override(rc, {"enabled": False})["enabled"])


# ── the verifier allow-list ──────────────────────────────────────────────


def _pair_with_sentinels() -> dict:
    """A pair whose reference and reference-like fields carry the sentinel."""
    return {
        "prompt": "Classify the text. Reply with exactly one of: alpha, beta.\nText: hello\nLabel:",
        "reference": SENTINEL,
        "_row_index": 3,
        "classification_candidates": ["alpha", "beta"],
        "plan_row": {"scenario": "s", "passages": [{"text": "passage one"}], "acceptable": {"x": SENTINEL}},
        "code_example_row": {"task": "t", "language": "python", "passages": [{"text": "doc"}]},
        "tool_gold_calls": [{"name": SENTINEL}],
        "alignment_chosen": SENTINEL,
        "remediation_row": {"finding": {"label": SENTINEL}},
        "judge": {"verdict": SENTINEL},
    }


class AllowListTests(unittest.TestCase):
    def test_builder_reads_only_prompt_visible_fields(self):
        vin = cds.build_verifier_input(_pair_with_sentinels(), {"json_schema": None})
        self.assertNotIn(SENTINEL, repr(vin))
        self.assertEqual(vin.code_language, "python")
        self.assertEqual(vin.passage_ids, ("P1",))
        self.assertEqual(vin.label_set, ("alpha", "beta"))
        # The dataclass has no field a reference could live in.
        self.assertEqual(
            set(vin.__dataclass_fields__), {"prompt", "code_language", "passage_ids", "label_set", "json_schema"}
        )

    def test_labels_not_shown_in_the_prompt_are_not_passed(self):
        pair = {"prompt": "Classify: hello", "reference": "alpha", "classification_candidates": ["alpha", "beta"]}
        self.assertEqual(cds.build_verifier_input(pair).label_set, ())

    def test_reference_never_reaches_a_verifier_end_to_end(self):
        seen: list[tuple] = []

        def spy(vin, answer):
            seen.append((vin, answer))
            return {"verifier_id": "label_set", "ok": False, "messages": ["no"]}

        pair = _pair_with_sentinels()
        policy = _policy(max_attempts=3, verifiers=["label_set"])
        with mock.patch.dict(cds._VERIFIERS, {"label_set": spy}):
            preds, runtime = _infer([pair], ["gamma", "delta", "omega"], checked=_checked(policy, [pair]))
        self.assertEqual(len(seen), 3)
        for vin, answer in seen:
            self.assertNotIn(SENTINEL, repr(vin))
            self.assertNotIn(SENTINEL, answer)
        self.assertEqual(preds[0]["prediction"], "gamma")  # none passed: attempt 1 kept


# ── the eval loop with a fake model ───────────────────────────────────────


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, answers, chat_template=None):
        self.answers = answers
        self.chat_template = chat_template
        self.rendered: list[list[dict]] = []
        self.texts: list[str] = []

    def __len__(self):
        return 8

    def __call__(self, text, return_tensors=None, **kw):
        import torch

        self.texts.append(text)
        return {"input_ids": torch.tensor([[2, 3, 4]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.rendered.append([dict(m) for m in messages])
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def decode(self, ids, skip_special_tokens=True):
        return self.answers[int(ids[0]) - 100]


class _FakeModel:
    def __init__(self):
        import torch

        self.dtype = torch.float32
        self.generation_config = SimpleNamespace()
        self.calls: list[dict] = []

    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=8)

    def eval(self):
        return self

    def to(self, *_a, **_k):
        return self

    def generate(self, **kw):
        import torch

        self.calls.append({k: v for k, v in kw.items() if k not in ("input_ids", "attention_mask")})
        # The generated token encodes which answer the fake tokenizer decodes (call order).
        return torch.tensor([[2, 3, 4, 100 + len(self.calls) - 1]])


def _checked(policy, pairs, eval_spec=None):
    return {"policy": policy, "inputs": [cds.build_verifier_input(p, eval_spec) for p in pairs]}


def _infer(pairs, answers, *, chat_template=None, apply_chat_template=False, **kwargs):
    from app.services import evaluation_service

    tok, model = _FakeTok(answers, chat_template), _FakeModel()
    seeds: list[int] = []
    import torch

    real_seed = torch.manual_seed
    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tok), \
         mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), \
         mock.patch("torch.cuda.is_available", return_value=False), \
         mock.patch("torch.manual_seed", side_effect=lambda s: (seeds.append(s), real_seed(s))[1]):
        preds, runtime = evaluation_service._run_transformers_inference(
            "fake/model", pairs, 16, 0.0, apply_chat_template=apply_chat_template, **kwargs,
        )
    _infer.last = SimpleNamespace(tok=tok, model=model, seeds=seeds)
    return preds, runtime


def _label_pair(i=0):
    return {"prompt": "Classify the text. Reply with exactly one of: alpha, beta.\nText: x\nLabel:",
            "reference": "alpha", "_row_index": i, "classification_candidates": ["alpha", "beta"]}


def _strip_timing(runtime):
    return {k: v for k, v in runtime.items() if k not in (
        "model_load_seconds", "generation_seconds", "average_latency_ms", "token_throughput_tps")}


def _strip_latency(preds):
    return [{k: v for k, v in p.items() if k != "latency_ms"} for p in preds]


class EquivalenceTests(unittest.TestCase):
    def test_off_is_unchanged(self):
        pairs = [_label_pair(0), _label_pair(1)]
        base_preds, base_rt = _infer(pairs, ["gamma", "alpha"])
        base_calls = _infer.last.model.calls
        off_preds, off_rt = _infer(pairs, ["gamma", "alpha"], checked=None)
        self.assertEqual(_strip_latency(off_preds), _strip_latency(base_preds))
        self.assertEqual(_strip_timing(off_rt), _strip_timing(base_rt))
        self.assertEqual(_infer.last.model.calls, base_calls)
        self.assertNotIn("checked_decoding", off_rt)
        self.assertNotIn("checked_decoding", off_rt["decoding"])
        self.assertTrue(all("checked_decoding" not in p for p in off_preds))

    def test_passing_rows_keep_attempt_one_with_one_generation(self):
        pairs = [_label_pair(0), _label_pair(1)]
        base_preds, _ = _infer(pairs, ["alpha", "beta"])
        preds, runtime = _infer(pairs, ["alpha", "beta"], checked=_checked(_policy(), pairs))
        self.assertEqual([p["prediction"] for p in preds], [p["prediction"] for p in base_preds])
        self.assertEqual(len(_infer.last.model.calls), 2)
        self.assertEqual([p["checked_decoding"]["attempts"] for p in preds], [1, 1])
        summary = runtime["checked_decoding"]
        self.assertEqual((summary["rows_retried"], summary["rows_fixed_by_retry"], summary["mean_attempts"]), (0, 0, 1.0))
        self.assertEqual(runtime["decoding"]["checked_decoding"], cds.policy_record(_policy()))


class RetryTests(unittest.TestCase):
    def test_resample_retries_with_seeded_sampling_until_a_pass(self):
        pairs = [_label_pair(5)]
        preds, runtime = _infer(pairs, ["gamma", "delta", "beta"], checked=_checked(_policy(seed=11), pairs))
        calls = _infer.last.model.calls
        self.assertEqual(len(calls), 3)
        self.assertFalse(calls[0]["do_sample"])  # attempt 1: the normal greedy answer
        self.assertTrue(calls[1]["do_sample"] and calls[2]["do_sample"])
        self.assertEqual(calls[1]["temperature"], 0.7)
        self.assertEqual(calls[1]["repetition_penalty"], 1.0)
        self.assertEqual(_infer.last.seeds, [cds.attempt_seed(11, 5, 2), cds.attempt_seed(11, 5, 3)])
        self.assertEqual(preds[0]["prediction"], "beta")
        rec = preds[0]["checked_decoding"]
        self.assertEqual((rec["attempts"], rec["chosen_attempt"]), (3, 3))
        self.assertEqual([d["attempt"] for d in rec["attempts_detail"]], [1, 2, 3])
        self.assertEqual(rec["attempts_detail"][0]["failed"][0]["verifier_id"], "label_set")
        self.assertEqual(rec["attempts_detail"][2]["failed"], [])
        s = runtime["checked_decoding"]
        self.assertEqual((s["rows_retried"], s["rows_fixed_by_retry"], s["rows_still_failing"]), (1, 1, 0))
        self.assertEqual(s["mean_attempts"], 3.0)
        self.assertIn("generation_seconds_total", s)
        # Latency stats stay attempt-1 numbers: the row's latency is its first generation.
        self.assertEqual(runtime["total_generated_tokens"], 1)

    def test_seeds_are_deterministic(self):
        self.assertEqual(cds.attempt_seed(0, 1, 2), cds.attempt_seed(0, 1, 2))
        self.assertNotEqual(cds.attempt_seed(0, 1, 2), cds.attempt_seed(0, 1, 3))
        self.assertNotEqual(cds.attempt_seed(0, 1, 2), cds.attempt_seed(1, 1, 2))

    def test_none_pass_keeps_attempt_one(self):
        pairs = [_label_pair()]
        preds, runtime = _infer(pairs, ["gamma", "delta"], checked=_checked(_policy(max_attempts=2), pairs))
        self.assertEqual(preds[0]["prediction"], "gamma")
        rec = preds[0]["checked_decoding"]
        self.assertEqual((rec["attempts"], rec["chosen_attempt"], rec["passed"]), (2, 1, False))
        s = runtime["checked_decoding"]
        self.assertEqual((s["rows_retried"], s["rows_fixed_by_retry"], s["rows_still_failing"]), (1, 0, 1))

    def test_repair_appends_one_follow_up_turn_through_the_chat_template(self):
        pairs = [_label_pair()]
        policy = _policy(mode="repair")
        preds, _ = _infer(pairs, ["gamma", "alpha"], checked=_checked(policy, pairs),
                          chat_template="{{ messages }}", apply_chat_template=True)
        tok, calls = _infer.last.tok, _infer.last.model.calls
        self.assertEqual(preds[0]["prediction"], "alpha")
        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[1]["do_sample"])  # repair decodes like attempt 1
        follow = tok.rendered[-1]
        self.assertEqual([m["role"] for m in follow], ["user", "assistant", "user"])
        self.assertEqual(follow[1]["content"], "gamma")
        self.assertIn("Your previous answer failed these automatic checks:", follow[2]["content"])
        self.assertIn("- label_set:", follow[2]["content"])
        self.assertIn("Return a corrected answer in the same format.", follow[2]["content"])
        self.assertNotIn(SENTINEL, json.dumps(tok.rendered))
        self.assertEqual(_infer.last.seeds, [])

    def test_repair_without_a_chat_template_resamples(self):
        pairs = [_label_pair()]
        preds, _ = _infer(pairs, ["gamma", "beta"], checked=_checked(_policy(mode="repair"), pairs))
        self.assertTrue(_infer.last.model.calls[1]["do_sample"])
        self.assertEqual(preds[0]["checked_decoding"]["modes_used"], ["resample"])

    def test_unknown_shape_is_not_retried(self):
        pairs = [{"prompt": "Say something.", "reference": "x", "_row_index": 0}]
        preds, runtime = _infer(pairs, ["anything"], checked=_checked(_policy(), pairs))
        self.assertEqual(len(_infer.last.model.calls), 1)
        self.assertEqual(preds[0]["checked_decoding"]["skipped"], "no_verifiers")
        self.assertEqual(runtime["checked_decoding"]["rows_without_verifiers"], 1)

    def test_replayed_turns_are_not_checked(self):
        pairs = [_label_pair(), {**_label_pair(), "_replay": True}]
        checked = {"policy": _policy(), "inputs": [cds.build_verifier_input(pairs[0]), None]}
        preds, _ = _infer(pairs, ["alpha", "gamma"], checked=checked)
        self.assertNotIn("checked_decoding", preds[1])
        self.assertEqual(len(_infer.last.model.calls), 2)


# ── verifiers by answer shape ─────────────────────────────────────────────


def _code_vin(**kw):
    base = {"prompt": "Language: python\nTask: add\n[P1] doc", "code_language": "python", "passage_ids": ("P1",)}
    base.update(kw)
    return cds.VerifierInput(**base)


class VerifierTests(unittest.TestCase):
    def test_code_checks_parse_the_code_field(self):
        good = json.dumps({"code": "def f(a, b):\n    return a + b\n", "citations": ["P1"]})
        bad = json.dumps({"code": "def f(:\n", "citations": ["P1"]})
        self.assertTrue(cds.verify_code_checks(_code_vin(), good)["ok"])
        res = cds.verify_code_checks(_code_vin(), bad)
        self.assertFalse(res["ok"])
        self.assertTrue(res["messages"][0].startswith("python_ast:"))
        self.assertFalse(cds.verify_code_checks(_code_vin(), "not json")["ok"])

    def test_grounding_only_when_opted_in_and_registered_and_gets_the_prompt(self):
        policy = _policy()
        opted = {**policy, cds.GROUNDING_OPT_IN_KEY: True}
        with mock.patch("app.services.code_check_service.get_code_check", return_value=None):
            self.assertEqual(cds.applicable_verifiers(_code_vin(), opted), ["code_checks", "citations"])
        with mock.patch("app.services.code_check_service.get_code_check", return_value=object()):
            # Auto mode: a registered (core built-in) grounding check is NOT on by default.
            self.assertEqual(cds.applicable_verifiers(_code_vin(), policy), ["code_checks", "citations"])
            self.assertEqual(
                cds.applicable_verifiers(_code_vin(), opted), ["code_checks", "context_grounding", "citations"]
            )
            self.assertEqual(
                cds.applicable_verifiers(_code_vin(), _policy(verifiers=["code_checks"])), ["code_checks"]
            )
            self.assertEqual(
                cds.applicable_verifiers(_code_vin(), _policy(verifiers=["code_checks", "context_grounding"])),
                ["code_checks", "context_grounding"],
            )
        seen = {}

        def fake_run(code, *, language, check_ids=None, context=None):
            seen.update(check_ids=check_ids, context=context)
            return {"ok": None, "results": [{"check_id": "context_grounding", "ok": None, "messages": []}]}

        with mock.patch("app.services.code_check_service.run_code_checks", fake_run):
            res = cds.verify_context_grounding(_code_vin(), json.dumps({"code": "x = 1"}))
        self.assertIsNone(res["ok"])  # skipped is not a failure
        self.assertEqual(seen["check_ids"], ["context_grounding"])
        self.assertEqual(seen["context"]["grounding_text"], _code_vin().prompt)

    def test_code_checks_never_request_grounding_themselves(self):
        seen = {}

        def fake_run(code, *, language, check_ids=None, context=None):
            seen["ids"] = check_ids
            return {"ok": True, "results": []}

        fake_checks = [SimpleNamespace(id="python_ast"), SimpleNamespace(id="context_grounding")]
        with mock.patch("app.services.code_check_service.checks_for_language", return_value=fake_checks), \
             mock.patch("app.services.code_check_service.run_code_checks", fake_run):
            cds.verify_code_checks(_code_vin(), json.dumps({"code": "x = 1"}))
        self.assertEqual(seen["ids"], ["python_ast"])

    def test_json_schema_from_the_declared_output_schema(self):
        manifest = {"output_schema": {"properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
                                      "required": ["name"]}}
        schema = cds.output_json_schema(manifest)
        vin = cds.VerifierInput(prompt="p", json_schema=schema)
        self.assertTrue(cds.verify_json_schema(vin, '{"name": "a", "count": 2}')["ok"])
        self.assertTrue(cds.verify_json_schema(vin, '{"name": null}')["ok"])  # null always allowed
        self.assertFalse(cds.verify_json_schema(vin, '{"count": 2}')["ok"])  # required key missing
        self.assertFalse(cds.verify_json_schema(vin, '{"name": "a", "count": "two"}')["ok"])
        self.assertFalse(cds.verify_json_schema(vin, "no json here")["ok"])
        # Other scoring modes / no declared properties: no schema verifier.
        self.assertIsNone(cds.output_json_schema({"output_schema": {"scoring_mode": "span_set", "properties": {"a": {}}}}))
        self.assertIsNone(cds.output_json_schema({}))

    def test_citations(self):
        vin = cds.VerifierInput(prompt="p", passage_ids=("P1", "P2"))
        self.assertTrue(cds.verify_citations(vin, '{"citations": ["P2"]}')["ok"])
        self.assertTrue(cds.verify_citations(vin, '{"citations": []}')["ok"])
        self.assertFalse(cds.verify_citations(vin, '{"citations": ["P9"]}')["ok"])
        self.assertIsNone(cds.verify_citations(vin, '{"code": "x"}')["ok"])

    def test_a_crashing_verifier_is_skipped(self):
        def boom(vin, answer):
            raise RuntimeError("x")

        with mock.patch.dict(cds._VERIFIERS, {"label_set": boom}):
            out = cds.run_verifiers(cds.VerifierInput(prompt="p", label_set=("a",)), "a", ["label_set"])
        self.assertIsNone(out[0]["ok"])

    def test_failure_messages_are_capped(self):
        def many(vin, answer):
            return {"verifier_id": "label_set", "ok": False, "messages": ["m" * 500] * 6}

        with mock.patch.dict(cds._VERIFIERS, {"label_set": many}):
            _, rec = cds.run_checked_row(
                policy=_policy(max_attempts=1), vin=cds.VerifierInput(prompt="p", label_set=("a",)),
                first={"text": "b", "tokens": 1, "seconds": 0.0}, regenerate=lambda *a: {},
            )
        failed = rec["attempts_detail"][0]["failed"][0]
        self.assertEqual(len(failed["messages"]), 3)
        self.assertEqual(len(failed["messages"][0]), 300)


# ── run_heldout_evaluation end to end (stub inference) ─────────────────────


def _run_heldout(rows, *, runtime_config=None, runtime_extra=None, **kw):
    """One eval scenario on a throwaway engine in a single event loop (the model is stubbed)."""
    import asyncio
    import tempfile
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    import app.models  # noqa: F401
    from app.database import Base
    from app.models.dataset import Dataset, DatasetType
    from app.models.experiment import Experiment, ExperimentStatus
    from app.models.project import Project
    from app.services import evaluation_service as ev

    seen: dict = {}

    def fake_inference(model_ref, pairs, *a, **kwargs):
        seen["kwargs"] = kwargs
        seen["pairs"] = [dict(p) for p in pairs]
        preds = [{"prompt": p["prompt"], "reference": p["reference"], "prediction": p["reference"], "latency_ms": 1.0}
                 for p in pairs]
        return preds, {"engine": "stub", **(runtime_extra or {})}

    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = create_async_engine(f"sqlite+aiosqlite:///{root / 'cd.db'}", future=True)
            sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            try:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                heldout = root / "test.jsonl"
                heldout.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
                async with sf() as db:
                    project = Project(id=20_000 + uuid4().int % 10**9, name=f"cd-{uuid4().hex[:8]}", description="x",
                                      base_model_name="toy/model", runtime_config=runtime_config or {})
                    db.add(project)
                    await db.flush()
                    exp = Experiment(project_id=project.id, name="e", description="", status=ExperimentStatus.COMPLETED,
                                     base_model="toy/model", output_dir=str(root))
                    db.add(exp)
                    db.add(Dataset(project_id=project.id, name="t", dataset_type=DatasetType.TEST,
                                   file_path=str(heldout), record_count=len(rows)))
                    await db.commit()
                    pid, eid = project.id, exp.id
                with mock.patch("app.services.evaluation_service._run_local_inference", fake_inference), \
                        mock.patch("app.services.evaluation_service._resolve_model_reference", return_value="toy"):
                    async with sf() as db:
                        result = await ev.run_heldout_evaluation(
                            db=db, project_id=pid, experiment_id=eid, dataset_name="test", eval_type="exact_match",
                            max_samples=100, answer_judge_enabled=False, **kw)
                        await db.commit()
                return result, seen
            finally:
                await engine.dispose()

    return asyncio.run(scenario())


ROWS = [{"question": "What is 2+2?", "answer": SENTINEL}, {"question": "Capital?", "answer": SENTINEL}]


class HeldoutEndToEndTests(unittest.TestCase):
    def test_off_by_default_passes_nothing(self):
        result, seen = _run_heldout(ROWS)
        self.assertNotIn("checked", seen["kwargs"])
        self.assertNotIn("checked_decoding", result.metrics)

    def test_project_setting_reaches_inference_without_the_reference(self):
        rc = cds.with_project_policy({}, {"enabled": True, "mode": "repair", "max_attempts": 2})
        summary = {"policy": {"mode": "repair"}, "rows_fixed_by_retry": 0}
        result, seen = _run_heldout(ROWS, runtime_config=rc, runtime_extra={"checked_decoding": summary})
        checked = seen["kwargs"]["checked"]
        self.assertEqual((checked["policy"]["mode"], checked["policy"]["max_attempts"]), ("repair", 2))
        self.assertEqual(len(checked["inputs"]), 2)
        self.assertNotIn(SENTINEL, repr(checked["inputs"]))
        self.assertEqual(result.metrics["checked_decoding"], summary)

    def test_override_turns_it_off_and_an_engine_without_support_is_reported(self):
        rc = cds.with_project_policy({}, {"enabled": True})
        _, seen = _run_heldout(ROWS, runtime_config=rc, checked_decoding={"enabled": False})
        self.assertNotIn("checked", seen["kwargs"])
        result, _ = _run_heldout(ROWS, checked_decoding={"enabled": True})
        self.assertEqual(result.metrics["checked_decoding"]["skipped"], "engine_unsupported")


# ── lift cache + Compare ──────────────────────────────────────────────────


class LiftCacheTests(unittest.TestCase):
    def _result(self, checked):
        from app.services.decoding_settings import decoding_record, generation_kwargs

        dec = decoding_record(generation_kwargs(max_new_tokens=1, temperature=0.0))
        if checked is not None:
            dec["checked_decoding"] = checked
        return SimpleNamespace(metrics={"inference": {"decoding": dec}})

    def test_cache_keys_on_the_policy(self):
        from app.services.post_training_eval_service import _baseline_decoding_matches

        rec = cds.policy_record(_policy())
        self.assertTrue(_baseline_decoding_matches(self._result(None), "neutral", None))
        self.assertFalse(_baseline_decoding_matches(self._result(None), "neutral", rec))
        self.assertFalse(_baseline_decoding_matches(self._result(rec), "neutral", None))
        self.assertTrue(_baseline_decoding_matches(self._result(rec), "neutral", rec))
        other = cds.policy_record(_policy(mode="repair"))
        self.assertFalse(_baseline_decoding_matches(self._result(rec), "neutral", other))
        self.assertFalse(_baseline_decoding_matches(SimpleNamespace(metrics={}), "neutral", rec))


class CompareTests(unittest.TestCase):
    def _prov(self, checked):
        return {"answer_key": {"dataset_type": "test", "sha256": "a" * 64}, "judge": {},
                "scoring": {"eval_type": "f1", "scoring_mode": None, "fingerprints": {},
                            "decoding": {"do_sample": False}, "checked_decoding": checked}}

    def test_different_policies_are_not_comparable(self):
        from app.services.compare_row_evidence_service import compare_provenance

        rec = cds.policy_record(_policy(max_attempts=3))
        out = compare_provenance(self._prov(None), self._prov(rec))
        self.assertFalse(out["comparable"])
        self.assertEqual([d["kind"] for d in out["differences"]], ["decoding_policy"])
        self.assertIn("different decoding policy", out["differences"][0]["message"])
        self.assertIn("off vs up to 3 tries · resample", out["differences"][0]["message"])
        self.assertTrue(compare_provenance(self._prov(rec), self._prov(dict(rec)))["comparable"])
        self.assertTrue(compare_provenance(self._prov(None), self._prov(None))["comparable"])
        self.assertEqual(compare_provenance(self._prov(None), self._prov(rec))["caveats"], [])

    def test_scoring_block_splits_the_policy_from_the_decoding(self):
        from app.services.compare_row_evidence_service import _scoring

        rec = cds.policy_record(_policy())
        er = SimpleNamespace(eval_type="f1", metrics={"inference": {"decoding": {"do_sample": False, "checked_decoding": rec}}},
                             details={})
        block = _scoring(er)
        self.assertEqual(block["decoding"], {"do_sample": False})
        self.assertEqual(block["checked_decoding"], rec)


# ── API ───────────────────────────────────────────────────────────────────


def _make_project() -> int:
    import asyncio

    from app.database import async_session_factory
    from app.models.project import Project

    async def _go():
        async with async_session_factory() as db:
            p = Project(name=f"checked-{uuid4().hex[:10]}", description="")
            db.add(p)
            await db.commit()
            return p.id

    return asyncio.run(_go())


class ApiTests(unittest.TestCase):
    def test_settings_round_trip_and_422(self):
        client = _client_cm
        pid = _make_project()
        got = client.get(f"/api/projects/{pid}/evaluation/checked-decoding")
        self.assertEqual(got.status_code, 200, got.text)
        self.assertFalse(got.json()["settings"]["enabled"])
        self.assertIn("label_set", got.json()["verifier_ids"])
        put = client.put(f"/api/projects/{pid}/evaluation/checked-decoding",
                         json={"enabled": True, "mode": "repair", "max_attempts": 4})
        self.assertEqual(put.status_code, 200, put.text)
        self.assertEqual(put.json()["label"], "Checked decoding · up to 4 tries · repair")
        for bad in ({"mode": "x"}, {"max_attempts": 9}, {"verifiers": ["nope"]}, {"sample_temperature": 0}):
            with self.subTest(bad=bad):
                r = client.put(f"/api/projects/{pid}/evaluation/checked-decoding", json=bad)
                self.assertEqual(r.status_code, 422, r.text)

    def test_run_heldout_validates_and_forwards_the_override(self):
        client = _client_cm
        pid = _make_project()
        bad = client.post(f"/api/projects/{pid}/evaluation/run-heldout",
                          json={"experiment_id": 1, "checked_decoding": {"enabled": True, "max_attempts": 0}})
        self.assertEqual(bad.status_code, 422, bad.text)
        seen = {}

        async def fake_run(**kwargs):
            seen.update(kwargs)
            raise ValueError("stop here")

        with mock.patch("app.api.evaluation.run_heldout_evaluation", fake_run):
            r = client.post(f"/api/projects/{pid}/evaluation/run-heldout",
                            json={"experiment_id": 1, "checked_decoding": {"enabled": True, "mode": "repair"}})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(seen["checked_decoding"]["mode"], "repair")
        self.assertEqual(seen["checked_decoding"]["max_attempts"], 3)

    def test_lift_check_validates_the_override(self):
        r = _client_cm.post(f"/api/projects/{_make_project()}/evaluation/summary/lift-check",
                            json={"experiment_id": 1, "checked_decoding": {"mode": "nope"}})
        self.assertEqual(r.status_code, 422, r.text)


def _make_experiment(project_id: int, config: dict) -> int:
    import asyncio

    from app.database import async_session_factory
    from app.models.experiment import Experiment, ExperimentStatus

    async def _go():
        async with async_session_factory() as db:
            exp = Experiment(project_id=project_id, name="e", description="", status=ExperimentStatus.COMPLETED,
                             base_model="toy/model", output_dir="/nonexistent", config=config)
            db.add(exp)
            await db.commit()
            return exp.id

    return asyncio.run(_go())


class ApiReviewFixTests(unittest.TestCase):
    def test_put_merges_a_partial_update_into_the_saved_setting(self):
        client = _client_cm
        pid = _make_project()
        url = f"/api/projects/{pid}/evaluation/checked-decoding"
        self.assertEqual(client.put(url, json={"enabled": True, "mode": "repair", "max_attempts": 4}).status_code, 200)
        r = client.put(url, json={"max_attempts": 2})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual({k: r.json()["settings"][k] for k in ("enabled", "mode", "max_attempts")},
                         {"enabled": True, "mode": "repair", "max_attempts": 2})
        self.assertEqual(client.put(url, json={"verifiers": ["label_set"]}).json()["settings"]["verifiers"],
                         ["label_set"])
        r = client.put(url, json={"verifiers": None})  # explicit null: back to auto
        self.assertIsNone(r.json()["settings"]["verifiers"])
        self.assertEqual(r.json()["settings"]["mode"], "repair")
        self.assertEqual(client.put(url, json={"max_attempts": 9}).status_code, 422)
        self.assertEqual(client.get(url).json()["settings"]["max_attempts"], 2, "a rejected update saves nothing")

    def test_lift_check_422_for_an_enabled_override_on_a_passage_trained_run(self):
        pid = _make_project()
        eid = _make_experiment(pid, {"passage_prompts": True})
        url = f"/api/projects/{pid}/evaluation/summary/lift-check"
        r = _client_cm.post(url, json={"experiment_id": eid, "checked_decoding": {"enabled": True}})
        self.assertEqual(r.status_code, 422, r.text)
        self.assertIn("passages", r.json()["detail"])

        async def not_started(db, **kwargs):
            return {"started": False, "skipped_reason": "stubbed"}

        with mock.patch("app.services.post_training_eval_service.start_post_training_lift_job", not_started):
            r = _client_cm.post(url, json={"experiment_id": eid, "checked_decoding": {"enabled": False}})
            self.assertEqual(r.status_code, 400, r.text)  # off: no 422
            other = _make_experiment(pid, {})
            r = _client_cm.post(url, json={"experiment_id": other, "checked_decoding": {"enabled": True}})
            self.assertEqual(r.status_code, 400, r.text)  # a plain run takes it


class RetryErrorTests(unittest.TestCase):
    def _vin(self):
        return cds.VerifierInput(prompt="p", label_set=("a",))

    def test_a_retry_that_raises_keeps_attempt_one_and_stops(self):
        calls = []

        def boom(n, previous, failures):
            calls.append(n)
            raise RuntimeError("generation failed")

        chosen, rec = cds.run_checked_row(policy=_policy(max_attempts=3), vin=self._vin(),
                                          first={"text": "b", "tokens": 2, "seconds": 0.1}, regenerate=boom)
        self.assertEqual(chosen["text"], "b")
        self.assertEqual(calls, [2])
        self.assertEqual(rec["attempts_detail"][1], {"attempt": 2, "error": "RuntimeError"})
        self.assertEqual((rec["attempts"], rec["chosen_attempt"], rec["passed"]), (1, 1, False))
        self.assertEqual(rec["retry_error"], "RuntimeError")
        self.assertNotIn("chosen_stats", rec)
        s = cds.summarize([rec], _policy())
        self.assertEqual((s["rows_failed_first_try"], s["rows_still_failing"], s["rows_retried"]), (1, 1, 0))

    def test_a_cuda_out_of_memory_retry_frees_the_cache(self):
        import torch

        oom_cls = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)

        def oom(*_a):
            raise oom_cls("CUDA out of memory. Tried to allocate 2.00 GiB")

        def other(*_a):
            raise ValueError("bad")

        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.empty_cache") as empty:
            _, rec = cds.run_checked_row(policy=_policy(), vin=self._vin(),
                                         first={"text": "b", "tokens": 1, "seconds": 0.0}, regenerate=oom)
            self.assertEqual(empty.call_count, 1)
            self.assertEqual(rec["attempts_detail"][-1]["error"], oom_cls.__name__)
            cds.run_checked_row(policy=_policy(), vin=self._vin(),
                                first={"text": "b", "tokens": 1, "seconds": 0.0}, regenerate=other)
            self.assertEqual(empty.call_count, 1, "only an out-of-memory error frees the cache")

    def test_the_eval_loop_survives_a_retry_generation_error(self):
        orig = _FakeModel.generate

        def flaky(self, **kw):
            if self.calls:
                self.calls.append({})
                raise RuntimeError("boom")
            return orig(self, **kw)

        pairs = [_label_pair(0)]
        with mock.patch.object(_FakeModel, "generate", flaky):
            preds, runtime = _infer(pairs, ["gamma"], checked=_checked(_policy(), pairs))
        self.assertEqual(preds[0]["prediction"], "gamma")
        self.assertEqual(preds[0]["checked_decoding"]["retry_error"], "RuntimeError")
        self.assertEqual(runtime["checked_decoding"]["rows_still_failing"], 1)


class RngAndStatsTests(unittest.TestCase):
    def test_retries_restore_the_rng_state(self):
        import torch

        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()
        pairs = [_label_pair(5)]
        _infer(pairs, ["gamma", "delta", "beta"], checked=_checked(_policy(seed=11), pairs))
        self.assertEqual(len(_infer.last.seeds), 2, "two seeded retries ran")
        self.assertTrue(torch.equal(torch.get_rng_state(), before), "the retries left the RNG as they found it")

    def test_a_later_chosen_attempt_records_its_own_stats(self):
        pairs = [_label_pair(5)]
        preds, _ = _infer(pairs, ["gamma", "delta", "beta"], checked=_checked(_policy(), pairs))
        rec = preds[0]["checked_decoding"]
        self.assertEqual(rec["chosen_attempt"], 3)
        self.assertEqual(set(rec["chosen_stats"]), {"generated_tokens", "latency_ms"})
        self.assertEqual(rec["chosen_stats"]["generated_tokens"], 1)
        self.assertEqual(preds[0]["generated_tokens"], 1)  # top-level stays attempt 1's
        passing, _ = _infer(pairs, ["alpha"], checked=_checked(_policy(), pairs))
        self.assertNotIn("chosen_stats", passing[0]["checked_decoding"])


class GroundingOptInTests(unittest.TestCase):
    RC = {"remediation": {"checks": ["python_ast", "context_grounding"]}}

    def test_sources(self):
        from app.config import settings

        p = _policy()
        self.assertFalse(cds.grounding_opted_in(p))
        self.assertFalse(cds.grounding_opted_in(p, runtime_config={"remediation": {"checks": ["python_ast"]}}))
        self.assertTrue(cds.grounding_opted_in(_policy(verifiers=["context_grounding"])))
        self.assertTrue(cds.grounding_opted_in(p, runtime_config=self.RC))
        self.assertTrue(cds.grounding_opted_in(
            p, manifest={"output_schema": {"remediation": {"checks": ["context_grounding"]}}}))
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "context_grounding"):
            self.assertTrue(cds.grounding_opted_in(p))

    def test_auto_mode_leaves_the_registered_check_out_unless_opted_in(self):
        p = _policy()
        self.assertNotIn("context_grounding", cds.applicable_verifiers(_code_vin(), p))
        self.assertNotIn("context_grounding", cds.applicable_verifiers(_code_vin(), cds.with_grounding_opt_in(p)))
        self.assertIn("context_grounding",
                      cds.applicable_verifiers(_code_vin(), cds.with_grounding_opt_in(p, runtime_config=self.RC)))
        self.assertEqual(cds.policy_record(cds.with_grounding_opt_in(p)), cds.policy_record(p))
        self.assertTrue(cds.policy_record(cds.with_grounding_opt_in(p, runtime_config=self.RC))["auto_includes_grounding"])
        self.assertIsNone(cds.with_grounding_opt_in(None))

    def test_heldout_eval_resolves_the_opt_in(self):
        rc = cds.with_project_policy({}, {"enabled": True})
        _, seen = _run_heldout(ROWS, runtime_config=rc)
        self.assertFalse(seen["kwargs"]["checked"]["policy"][cds.GROUNDING_OPT_IN_KEY])
        _, seen = _run_heldout(ROWS, runtime_config={**rc, **self.RC})
        self.assertTrue(seen["kwargs"]["checked"]["policy"][cds.GROUNDING_OPT_IN_KEY])


class SchemaAndLabelTests(unittest.TestCase):
    def test_json_schema_requires_only_what_the_schema_declares(self):
        schema = cds.output_json_schema({"output_schema": {"properties": {"a": {"type": "string"}, "b": {}}}})
        self.assertEqual(schema["required"], [])
        vin = cds.VerifierInput(prompt="p", json_schema=schema)
        self.assertTrue(cds.verify_json_schema(vin, "{}")["ok"])
        self.assertFalse(cds.verify_json_schema(vin, '{"a": 3}')["ok"])

    def test_label_answer_must_be_exactly_one_label(self):
        vin = cds.VerifierInput(prompt="p", label_set=("alpha", "very_positive"))
        for ok in ("alpha", " Alpha. ", '"alpha"', "**ALPHA**", "very_positive", "'Very_Positive'"):
            with self.subTest(ok=ok):
                self.assertTrue(cds.verify_label_set(vin, ok)["ok"])
        for bad in ("alpha or beta", "not alpha", "alphabet", "The label is alpha", "", "positive"):
            with self.subTest(bad=bad):
                self.assertFalse(cds.verify_label_set(vin, bad)["ok"])

    def test_label_set_only_when_the_prompt_enumerates_the_candidates(self):
        from app.services.eval_task_handler_service import ClassificationHandler

        h = ClassificationHandler()
        cands = ["alpha", "beta"]
        listed = {"prompt": h._build_prompt_text("hello", cands), "classification_candidates": cands}
        self.assertEqual(cds.build_verifier_input(listed).label_set, ("alpha", "beta"))
        # The labels appear in the input text only: no label verifier.
        unlisted = {"prompt": "Classify the following text. Reply with just the class label, nothing else.\n"
                              "Text: alpha or beta?\nLabel:", "classification_candidates": cands}
        self.assertEqual(cds.build_verifier_input(unlisted).label_set, ())
        # More candidates than the prompt lists: no label verifier, even if every label is in the text.
        many = [f"label{i}" for i in range(h.LABEL_LIST_PROMPT_CAP + 1)]
        prompt = h._build_prompt_text(" ".join(many), many)
        self.assertNotIn("Reply with exactly one of", prompt)
        self.assertEqual(cds.build_verifier_input({"prompt": prompt, "classification_candidates": many}).label_set, ())


if __name__ == "__main__":
    unittest.main()
