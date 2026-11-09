"""Stronger checks for code-example evals (all opt-in; defaults unchanged).

Pins:
  * clearer ``code_checks`` messages: a ``code`` given as a JSON object / list, and a fenced ``code``
    string when a check failed -- pass / fail unchanged, per-failure message cap kept;
  * ``properties_parse`` (gated in-process check): JSON text and JSON / YAML structure lines fail,
    ``key=value`` / ``key: value`` / ``key value`` / bare ``key`` pass, continuations + comments honoured;
    not in the default selection unless ``CODE_CHECK_ALLOWED`` lists it;
  * checked decoding ``resample_on_repeat`` (default off): a repeated retry makes the next attempt a
    resample; ``repeats`` / ``rows_with_repeats``; policy records unchanged when off;
  * ``remediation.code_example_instruction`` (default none): unset = the prompt byte for byte as before;
    set = its own line after the first instruction line, rendered identically for training rows and eval.

Pytest-only; no real model runs (a fake tokenizer / model stands in for transformers).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.services import checked_decoding_service as cds
from app.services import code_check_service as ccs
from app.services import code_remediation_prompt as crp
from app.services.code_checks.base import normalize_language
from app.services.code_checks.builtin import PropertiesParse, properties_logical_lines

client: TestClient | None = None


def setUpModule():  # noqa: N802 - unittest hook
    global client
    from app.main import app

    client = TestClient(app)
    client.__enter__()


def tearDownModule():  # noqa: N802 - unittest hook
    if client is not None:
        client.__exit__(None, None, None)


def _policy(**kw):
    return cds.normalize_policy({"enabled": True, **kw})


# ── 1. clearer code_checks messages ──────────────────────────────────────


def _yaml_vin():
    return cds.VerifierInput(prompt="Language: yaml\nTask: a config", code_language="yaml")


class CodeCheckMessageTests(unittest.TestCase):
    def test_code_given_as_object_or_list_names_the_mistake(self):
        for code in ({"apiVersion": "v1"}, ["a", "b"]):
            with self.subTest(code=code):
                res = cds.verify_code_checks(_yaml_vin(), json.dumps({"code": code, "rationale": "r"}))
                self.assertIs(res["ok"], False)
                self.assertEqual(res["messages"], [
                    'the "code" field must be a string holding the yaml source text, not a JSON object or list'
                ])

    def test_missing_code_keeps_the_old_message(self):
        res = cds.verify_code_checks(_yaml_vin(), "not json at all")
        self.assertIs(res["ok"], False)
        self.assertEqual(res["messages"], ['the answer is not a JSON object with a "code" string'])

    def test_fenced_code_that_fails_gets_the_fence_hint(self):
        answer = json.dumps({"code": "```yaml\nkey: value\n```", "rationale": "r", "citations": []})
        res = cds.verify_code_checks(_yaml_vin(), answer)
        self.assertIs(res["ok"], False)
        self.assertTrue(res["messages"][0].startswith("yaml_parse:"))
        self.assertEqual(
            res["messages"][-1],
            'the "code" string starts with a markdown fence; put only the yaml source text in it, without ``` fences',
        )

    def test_unfenced_failure_has_no_fence_hint(self):
        res = cds.verify_code_checks(_yaml_vin(), json.dumps({"code": "key: [unclosed"}))
        self.assertIs(res["ok"], False)
        self.assertFalse(any("markdown fence" in m for m in res["messages"]))

    def test_a_passing_fenced_answer_still_passes_without_the_hint(self):
        report = {"ok": True, "results": [{"check_id": "yaml_parse", "ok": True, "messages": []}]}
        with mock.patch.object(ccs, "run_code_checks", return_value=report):
            res = cds.verify_code_checks(_yaml_vin(), json.dumps({"code": "```yaml\nk: v\n```"}))
        self.assertIs(res["ok"], True)
        self.assertEqual(res["messages"], [])

    def test_the_hint_survives_the_per_failure_cap(self):
        report = {"ok": False, "results": [
            {"check_id": "yaml_parse", "ok": False, "messages": [f"problem {i}" for i in range(6)]}]}
        with mock.patch.object(ccs, "run_code_checks", return_value=report):
            res = cds.verify_code_checks(_yaml_vin(), json.dumps({"code": "```\nk: v\n```"}))
        failed = cds._failures([res])[0]["messages"]
        self.assertEqual(len(failed), cds.MAX_MESSAGES_PER_FAILURE)
        self.assertIn("markdown fence", failed[-1])


# ── 2. properties_parse ──────────────────────────────────────────────────


def _props(code):
    return PropertiesParse().run(code, language="properties", context={})


class PropertiesParseTests(unittest.TestCase):
    def test_valid_forms_pass(self):
        code = (
            "# comment {not checked}\n! also a comment [x]\n\n"
            "a=b\nc: d\ne f\ng\n  indented.key = v\n"
            "long=one, \\\n    {continued text is part of the value}\n"
            "path=C:\\\\dir\n"
        )
        self.assertEqual(_props(code), {"check_id": "properties_parse", "ok": True, "messages": [], "skipped": None})
        self.assertIs(_props("")["ok"], True)

    def test_json_text_fails(self):
        for code, kind in (('{"KEY": "v"}', "object"), ('  [1, 2]\n', "array")):
            with self.subTest(code=code):
                res = _props(code)
                self.assertIs(res["ok"], False)
                self.assertIn(f"JSON {kind}", res["messages"][0])

    def test_structure_lines_fail_and_name_the_line(self):
        res = _props("a=1\n{\n- item\n  ]\n# [fine]\nb=2\n[section]\n")
        self.assertIs(res["ok"], False)
        self.assertEqual([m.split(":")[0] for m in res["messages"]], ["line 2", "line 3", "line 4", "line 7"])
        self.assertIn("starts with '-'", res["messages"][1])

    def test_continuation_reports_the_first_physical_line(self):
        self.assertEqual(properties_logical_lines("x=\\\\\n{y\nz=1 \\\n  2"), [(1, "x=\\\\"), (2, "{y"), (3, "z=1 2")])
        res = _props("x=\\\\\n{y")  # an even number of backslashes does not continue
        self.assertEqual(res["messages"][0].split(":")[0], "line 2")

    def test_many_bad_lines_are_summarised(self):
        res = _props("\n".join("- x" for _ in range(9)))
        self.assertEqual(len(res["messages"]), 6)
        self.assertEqual(res["messages"][-1], "and 4 more line(s) like these")

    def test_aliases(self):
        for alias in ("properties", "java-properties", "Java_Properties"):
            self.assertEqual(normalize_language(alias), "properties")

    def test_gated_like_the_other_in_process_check(self):
        check = ccs.get_code_check("properties_parse")
        self.assertIsNotNone(check)
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", ""):
            self.assertFalse(ccs.in_default_selection(check))
            self.assertNotIn("properties_parse", [c.id for c in ccs.checks_for_language("properties")])
            res = cds.verify_code_checks(
                cds.VerifierInput(prompt="p", code_language="properties"), json.dumps({"code": '{"K": "v"}'}))
            self.assertIsNone(res["ok"])  # no runnable check: not a failure
        with mock.patch.object(settings, "CODE_CHECK_ALLOWED", "properties_parse"):
            self.assertTrue(ccs.in_default_selection(check))
            res = cds.verify_code_checks(
                cds.VerifierInput(prompt="p", code_language="java-properties"), json.dumps({"code": '{"K": "v"}'}))
            self.assertIs(res["ok"], False)
            self.assertTrue(res["messages"][0].startswith("properties_parse: the text is a JSON object"))
            ok = cds.verify_code_checks(
                cds.VerifierInput(prompt="p", code_language="properties"), json.dumps({"code": "K=v\nL: w"}))
            self.assertIs(ok["ok"], True)


# ── 3. resample_on_repeat ────────────────────────────────────────────────


class RepeatPolicyTests(unittest.TestCase):
    def test_default_off_and_validated(self):
        self.assertIs(cds.DEFAULTS["resample_on_repeat"], False)
        self.assertIs(cds.normalize_policy({})["resample_on_repeat"], False)
        self.assertIs(_policy(resample_on_repeat=True)["resample_on_repeat"], True)
        with self.assertRaises(cds.PolicyError):
            cds.normalize_policy({"resample_on_repeat": "yes"})

    def test_record_when_off(self):
        # Repair mode carries the repair-message revision (the repair text changed); no flag, no sampling knobs.
        self.assertEqual(cds.policy_record(_policy(mode="repair")),
                         {"mode": "repair", "max_attempts": 3, "verifiers": "auto", "repair_messages": 2})
        self.assertEqual(cds.policy_record(_policy()),
                         {"mode": "resample", "max_attempts": 3, "verifiers": "auto", "sample_temperature": 0.7, "seed": 0})

    def test_record_carries_the_flag_and_sampling_knobs_when_on(self):
        rec = cds.policy_record(_policy(mode="repair", resample_on_repeat=True, seed=4))
        self.assertEqual(rec, {"mode": "repair", "max_attempts": 3, "verifiers": "auto", "repair_messages": 2,
                               "resample_on_repeat": True, "sample_temperature": 0.7, "seed": 4})

    def test_flag_not_recorded_in_resample_mode(self):
        # Resample mode behaves the same with or without the flag: its record (cache key) must not change.
        self.assertEqual(cds.policy_record(_policy(resample_on_repeat=True)), cds.policy_record(_policy()))


def _label_vin():
    return cds.VerifierInput(prompt="Reply with one of: alpha, beta", label_set=("alpha", "beta"))


def _run(policy, texts, *, with_resample=True):
    calls: list[tuple[str, int]] = []
    queue = list(texts[1:])

    def regenerate(n, previous, failures):
        calls.append(("repair", n))
        return {"text": queue.pop(0), "tokens": 1, "seconds": 0.0, "mode": "repair"}

    def resample(n):
        calls.append(("resample", n))
        return {"text": queue.pop(0), "tokens": 1, "seconds": 0.0, "mode": "resample"}

    chosen, record = cds.run_checked_row(
        policy=policy, vin=_label_vin(), first={"text": texts[0], "tokens": 1, "seconds": 0.0},
        regenerate=regenerate, resample=resample if with_resample else None,
    )
    return chosen, record, calls


class RepeatLoopTests(unittest.TestCase):
    def test_off_always_repairs_and_records_repeats_only_when_seen(self):
        _, rec, calls = _run(_policy(mode="repair"), ["gamma", " gamma\n", "alpha"])
        self.assertEqual(calls, [("repair", 2), ("repair", 3)])
        self.assertEqual(rec["repeats"], 1)
        _, clean, _ = _run(_policy(mode="repair"), ["gamma", "delta", "alpha"])
        self.assertNotIn("repeats", clean)

    def test_on_resamples_after_a_repeat(self):
        policy = _policy(mode="repair", resample_on_repeat=True, max_attempts=4)
        chosen, rec, calls = _run(policy, ["gamma", "gamma", "delta", "alpha"])
        self.assertEqual(calls, [("repair", 2), ("resample", 3), ("repair", 4)])
        self.assertEqual(chosen["text"], "alpha")
        self.assertEqual(rec["repeats"], 1)
        self.assertEqual(rec["modes_used"], ["repair", "resample", "repair"])

    def test_on_without_repeats_records_zero(self):
        _, rec, calls = _run(_policy(mode="repair", resample_on_repeat=True), ["gamma", "alpha"])
        self.assertEqual(calls, [("repair", 2)])
        self.assertEqual(rec["repeats"], 0)

    def test_a_repeat_of_any_earlier_attempt_counts(self):
        policy = _policy(mode="repair", resample_on_repeat=True, max_attempts=4)
        _, rec, calls = _run(policy, ["gamma", "delta", "gamma", "alpha"])
        self.assertEqual(calls, [("repair", 2), ("repair", 3), ("resample", 4)])
        self.assertEqual(rec["repeats"], 1)

    def test_no_resample_callable_falls_back_to_regenerate(self):
        _, _, calls = _run(_policy(mode="repair", resample_on_repeat=True), ["gamma", "gamma", "alpha"],
                           with_resample=False)
        self.assertEqual(calls, [("repair", 2), ("repair", 3)])

    def test_summary_rows_with_repeats(self):
        off = _policy(mode="repair")
        _, rec_a, _ = _run(off, ["gamma", "gamma", "alpha"])
        _, rec_b, _ = _run(off, ["alpha"])
        self.assertEqual(cds.summarize([rec_a, rec_b], off)["rows_with_repeats"], 1)
        self.assertNotIn("rows_with_repeats", cds.summarize([rec_b], off))
        on = _policy(mode="repair", resample_on_repeat=True)
        self.assertEqual(cds.summarize([rec_b], on)["rows_with_repeats"], 0)
        self.assertEqual(cds.checked_decoding_summary({"checked_decoding": cds.summarize([rec_a], off)})
                         ["rows_with_repeats"], 1)


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = "{{ messages }}"

    def __init__(self, answers):
        self.answers = answers

    def __len__(self):
        return 8

    def __call__(self, text, return_tensors=None, **kw):
        import torch

        return {"input_ids": torch.tensor([[2, 3, 4]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
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
        return torch.tensor([[2, 3, 4, 100 + len(self.calls) - 1]])


class RepeatEvalLoopTests(unittest.TestCase):
    def test_eval_loop_resamples_after_a_repeated_repair(self):
        import torch

        from app.services import evaluation_service

        pair = {"prompt": "Classify the text. Reply with exactly one of: alpha, beta.\nText: x\nLabel:",
                "reference": "alpha", "_row_index": 3, "classification_candidates": ["alpha", "beta"]}
        policy = _policy(mode="repair", resample_on_repeat=True, seed=9)
        tok, model, seeds = _FakeTok(["gamma", "gamma", "beta"]), _FakeModel(), []
        real_seed = torch.manual_seed
        with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tok), \
             mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), \
             mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("torch.manual_seed", side_effect=lambda s: (seeds.append(s), real_seed(s))[1]):
            preds, runtime = evaluation_service._run_transformers_inference(
                "fake/model", [pair], 16, 0.0, apply_chat_template=True,
                checked={"policy": policy, "inputs": [cds.build_verifier_input(pair)]},
            )
        self.assertEqual(len(model.calls), 3)
        self.assertFalse(model.calls[1]["do_sample"])  # attempt 2: a repair turn
        self.assertTrue(model.calls[2]["do_sample"])  # attempt 3: resampled after the repeat
        self.assertEqual(seeds, [cds.attempt_seed(9, 3, 3)])
        self.assertEqual(preds[0]["prediction"], "beta")
        self.assertEqual(preds[0]["checked_decoding"]["repeats"], 1)
        self.assertEqual(runtime["checked_decoding"]["rows_with_repeats"], 1)
        self.assertIs(runtime["decoding"]["checked_decoding"]["resample_on_repeat"], True)


def _make_project(recipe_id: str | None = None) -> int:
    from app.database import async_session_factory
    from app.models.project import Project

    async def _go():
        async with async_session_factory() as db:
            p = Project(name=f"code-example-checks-{uuid4().hex[:10]}", description="",
                        selected_recipe={"recipe_id": recipe_id} if recipe_id else None)
            db.add(p)
            await db.commit()
            return p.id

    return asyncio.run(_go())


class RepeatApiTests(unittest.TestCase):
    def test_put_round_trip_and_422(self):
        url = f"/api/projects/{_make_project()}/evaluation/checked-decoding"
        got = client.get(url)
        self.assertIs(got.json()["settings"]["resample_on_repeat"], False)
        put = client.put(url, json={"enabled": True, "mode": "repair", "resample_on_repeat": True})
        self.assertEqual(put.status_code, 200, put.text)
        self.assertIs(put.json()["settings"]["resample_on_repeat"], True)
        self.assertEqual(client.put(url, json={"resample_on_repeat": [1]}).status_code, 422)


# ── 4. code-example extra instruction ────────────────────────────────────

PASSAGES = [{"id": "P1", "text": "db.get(id) returns a row", "source": "api.md"}]
# sha256 of the prompts rendered by the builder before the setting existed.
GOLDEN = {
    ("Read a row by id.", "python", True): "283ac10a85354ad1ee8e812848ee9a4569260254a8bc6dcbf2c05ebf9d52bf1a",
    ("Write a config.", "yaml", False): "202d9d821a338bd429099f32f6a22103eef1317f7c674ee2ca128fcc3528ba4f",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class InstructionPromptTests(unittest.TestCase):
    def test_unset_is_byte_identical(self):
        for (task, lang, with_passages), digest in GOLDEN.items():
            passages = PASSAGES if with_passages else None
            for extra in (None, "", "   \n ", 7):
                with self.subTest(task=task, extra=extra):
                    self.assertEqual(_sha(crp.build_code_example_prompt(task, lang, passages, extra)), digest)
            self.assertEqual(_sha(crp.build_code_example_prompt(task, lang, passages)), digest)

    def test_set_is_one_line_after_the_first_instruction(self):
        plain = crp.build_code_example_prompt("Write a config.", "yaml", None)
        got = crp.build_code_example_prompt("Write a config.", "yaml", None, '  Put only  the source\ntext in "code". ')
        lines, base = got.split("\n"), plain.split("\n")
        self.assertEqual(lines[1], 'Put only the source text in "code".')
        self.assertEqual(lines[:1] + lines[2:], base)
        long = crp.build_code_example_prompt("t", "yaml", None, "x" * 900).split("\n")[1]
        self.assertEqual(len(long), crp.MAX_CODE_EXAMPLE_INSTRUCTION_CHARS)

    def test_effective_config(self):
        self.assertIsNone(crp.effective_remediation_config({})["code_example_instruction"])
        cfg = crp.effective_remediation_config({"remediation": {"code_example_instruction": " a \n b "}})
        self.assertEqual(cfg["code_example_instruction"], "a b")

    def test_training_and_eval_render_the_same_prompt(self):
        from app.services.data_adapter_service import _map_remediation
        from app.services.eval_task_handler_service import StructuredExtractionHandler

        row = {"row_kind": "code_example", "task": "Read a row by id.", "language": "python", "passages": PASSAGES,
               "code_example_json": {"code": "x = 1", "rationale": "r", "citations": ["P1"]}}
        for extra in (None, "Use only the standard library."):
            with self.subTest(extra=extra):
                rem = {"extra_fields": {}, **({"code_example_instruction": extra} if extra else {})}
                mapped = _map_remediation(row, {"remediation": rem}, row["task"], "{}", None)
                built = StructuredExtractionHandler()._build_remediation_prompt(
                    row, {"extra_fields": {}, "remediation": dict(rem)})
                self.assertEqual(mapped["source_text"], built.prompt)
                self.assertEqual(mapped["source_text"],
                                 crp.build_code_example_prompt("Read a row by id.", "python", PASSAGES, extra))
                if extra:
                    self.assertEqual(mapped["source_text"].split("\n")[1], extra)

    def test_manifest_stamp_only_when_set(self):
        from app.services.dataset_service import _remediation_schema_stamp

        bare = SimpleNamespace(id=None, runtime_config={"remediation": {"retrieve_passages": False}}, selected_recipe=None)
        self.assertNotIn("code_example_instruction", _remediation_schema_stamp(bare, {})["remediation"])
        set_ = SimpleNamespace(id=None, runtime_config={"remediation": {
            "retrieve_passages": False, "code_example_instruction": "Be brief."}}, selected_recipe=None)
        self.assertEqual(_remediation_schema_stamp(set_, {})["remediation"]["code_example_instruction"], "Be brief.")

    def test_fingerprint_unchanged_when_unset(self):
        from app.services.remediation_detector import remediation_scoring_fingerprint

        async def fp(schema):
            from app.database import async_session_factory

            async with async_session_factory() as db:
                return await remediation_scoring_fingerprint(db, 0, {"remediation": {"retrieve_passages": False}}, schema)

        base = {"properties": {}, "required": [], "remediation": {"checks": None}}
        unset = asyncio.run(fp(base))
        self.assertEqual(asyncio.run(fp({**base, "remediation": {"checks": None, "code_example_instruction": ""}})), unset)
        with_line = asyncio.run(fp({**base, "remediation": {"checks": None, "code_example_instruction": "Be brief."}}))
        self.assertNotEqual(with_line, unset)


class InstructionApiTests(unittest.TestCase):
    def test_put_get_validate_and_clear(self):
        url = f"/api/projects/{_make_project('code-remediation')}/remediation-config"
        got = client.get(url)
        self.assertEqual(got.status_code, 200, got.text)
        self.assertIsNone(got.json()["config"]["code_example_instruction"])
        self.assertIsNone(got.json()["effective"]["code_example_instruction"])
        put = client.put(url, json={"code_example_instruction": "  Return only\n the source text. "})
        self.assertEqual(put.status_code, 200, put.text)
        self.assertEqual(put.json()["config"]["code_example_instruction"], "Return only the source text.")
        self.assertEqual(put.json()["effective"]["code_example_instruction"], "Return only the source text.")
        too_long = client.put(url, json={"code_example_instruction": "x" * 501})
        self.assertEqual(too_long.status_code, 422, too_long.text)
        self.assertEqual(client.get(url).json()["config"]["code_example_instruction"], "Return only the source text.")
        cleared = client.put(url, json={"code_example_instruction": "   "})
        self.assertIsNone(cleared.json()["config"]["code_example_instruction"])



class InstructionSplitConfigTests(unittest.TestCase):
    """The split's adapter config carries the instruction, so prepared training rows render the prompt the eval
    handler renders from the manifest stamp (and the live overlay keeps that stamp)."""

    def test_adapter_config_carries_the_instruction_and_prompts_match(self):
        from app.database import async_session_factory
        from app.services import dataset_service as ds
        from app.services.data_adapter_service import _map_remediation
        from app.services.eval_task_handler_service import StructuredExtractionHandler
        from app.services.evaluation_service import _apply_remediation_live_config

        pid = _make_project("code-remediation")
        put = client.put(f"/api/projects/{pid}/remediation-config",
                         json={"retrieve_passages": False, "code_example_instruction": "Use only the standard library."})
        self.assertEqual(put.status_code, 200, put.text)

        async def go():
            async with async_session_factory() as db:
                cfg = await ds._apply_recipe_adapter_config(db, pid, "structured-extraction", None)
                schema = await ds._recipe_output_schema(db, pid)
                ctx = SimpleNamespace(manifest={"output_schema": json.loads(json.dumps(schema))})
                await _apply_remediation_live_config(db, pid, ctx)
                return cfg, schema, ctx.manifest["output_schema"]

        cfg, schema, overlaid = asyncio.run(go())
        self.assertEqual(cfg["remediation"]["code_example_instruction"], "Use only the standard library.")
        self.assertEqual(schema["remediation"]["code_example_instruction"], "Use only the standard library.")
        self.assertEqual(overlaid["remediation"]["code_example_instruction"], "Use only the standard library.")

        row = {"row_kind": "code_example", "task": "Read a row by id.", "language": "python", "passages": PASSAGES,
               "code_example_json": {"code": "x = 1", "rationale": "r", "citations": ["P1"]}}
        mapped = _map_remediation(row, cfg, row["task"], "{}", None)
        built = StructuredExtractionHandler()._build_remediation_prompt(row, overlaid)
        self.assertEqual(mapped["source_text"].split("\n")[1], "Use only the standard library.")
        self.assertEqual(mapped["source_text"], built.prompt)


if __name__ == "__main__":
    unittest.main()
