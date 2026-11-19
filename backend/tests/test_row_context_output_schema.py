"""Per-row output schema (``row_context.output_schema``, hardening Phase 4) and the opt-in "null counts as
absent" JSON scoring setting.

- Rendering: identical in training and eval through the shared renderers, for structured extraction
  (``render_extraction_prompt``) and a chat-wrapped JSON answer on the ``qa`` handler
  (``render_output_schema_lead``); rows without a schema are untouched; chat rows never get one.
- Scoring: extraction field match takes the row's fields / required / types; every consuming task gets
  ``metrics.output_schema`` (the answer validated against the row's schema); evals without one keep their
  metrics. Import rejects a bad schema as a grouped ``row_context_schema_invalid``. Data health treats
  the block as context. Prepared splits store it as JSON text.
- Null setting: extraction (nested + whole object) and tool-call args (default + strict); provenance in
  ``metrics.json_scoring``; Compare / lift refuse to pair across settings; per-run override; project
  setting endpoint.

pytest-only. Model inference is stubbed (``_run_local_inference``); no network."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from _row_context_golden import CASES, FakeTokenizer, NoTemplateTokenizer, train_rows, training_render
from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus
from app.models.project import Project
from app.services import null_scoring
from app.services import row_context_service as rcs
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler, ToolCallHandler

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": ["integer", "null"]}},
    "required": ["name"],
    "additionalProperties": False,
}
OTHER = {"type": "object", "properties": {"city": {"type": "string", "enum": ["Paris", "Rome"]}}, "required": ["city"]}

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    from _job_wait import wait_for_jobs_idle

    wait_for_jobs_idle(client, timeout_s=60)
    client.__exit__(None, None, None)


def run_async(fn):
    async def go():
        async with async_session_factory() as db:
            return await fn(db)

    return asyncio.run(go())


def ctx(manifest: dict | None = None, profile: str = "structured_extraction") -> EvalContext:
    return EvalContext(project_id=0, experiment_id=0, eval_type="exact_match", task_profile=profile,
                       handler_id=profile, prepared_dir=Path("."), dataset_name="test", manifest=manifest or {})


# ── rendering ────────────────────────────────────────────────────────────


class RenderingTests(unittest.TestCase):
    def test_consumers_and_modes(self):
        self.assertEqual(rcs.output_schema_mode("structured_extraction"), "extraction")
        for profile in ("qa", "instruction_sft", "chat_sft"):
            self.assertEqual(rcs.output_schema_mode(profile), "lead", profile)
        for profile in ("tool_calling", "classification", None, "unknown_profile"):
            self.assertIsNone(rcs.output_schema_mode(profile), profile)

    def test_block_and_prompts(self):
        block = rcs.output_schema_block(SCHEMA)
        self.assertEqual(block.splitlines()[0], rcs.OUTPUT_SCHEMA_INSTRUCTION)
        self.assertEqual(json.loads(block.splitlines()[1][len(rcs.OUTPUT_SCHEMA_PREFIX):]), SCHEMA)
        p = rcs.render_extraction_prompt("Ann, 30", SCHEMA)
        self.assertTrue(p.startswith("Extract the following fields as JSON: age, name.\n"))
        self.assertTrue(p.endswith("\nInput: Ann, 30\nOutput:"))
        self.assertTrue(rcs.render_extraction_prompt("x", {"type": "object"}).startswith(
            "Extract the relevant fields from the input"))
        self.assertEqual(rcs.render_output_schema_lead("Q?", SCHEMA), f"{block}\n\nQ?")

    def test_import_rejects_a_bad_schema(self):
        for bad in ({"type": "nope"}, ["not", "an", "object"], {"type": "object", "properties": {"a": {"type": 5}}}):
            with self.subTest(bad=bad):
                self.assertEqual(rcs.normalize({"output_schema": bad})[1], "row_context_schema_invalid")
        huge = {"type": "object", "description": "x" * (rcs.MAX_OUTPUT_SCHEMA_CHARS + 1)}
        self.assertEqual(rcs.normalize({"output_schema": huge})[1], "row_context_schema_invalid")
        self.assertEqual(rcs.normalize({"output_schema": SCHEMA}), ({"output_schema": SCHEMA}, None))


class TrainEvalParityTests(unittest.TestCase):
    """Training text == eval text for rows with their own schema, with and without a chat template."""

    def _parity(self, name: str, rows: list[dict]) -> tuple[list[dict], list[str], list[dict]]:
        from _row_context_golden import eval_render

        case = {**CASES[name], "rows": rows}
        prepared = training_render(case)
        self.assertEqual(len(prepared), len(rows))
        with tempfile.TemporaryDirectory() as d:
            pairs, texts = eval_render(case, prepared, Path(d))
        trained = [train_rows(prepared, FakeTokenizer()), train_rows(prepared, NoTemplateTokenizer())]
        for i in range(len(prepared)):
            for j in range(2):
                got = texts[i * 2 + j]
                want = trained[j][i]["source_text"]
                self.assertEqual(want if j == 0 else want.rstrip("\n"), got, f"{name} row {i} tokenizer {j}")
        return prepared, texts, pairs

    def test_extraction(self):
        rows = [{"text": "Ann is 30", "json": {"name": "Ann", "age": 30}, "row_context": {"output_schema": SCHEMA}},
                {"text": "Lives in Rome", "json": {"city": "Rome"}, "row_context": {"output_schema": OTHER}},
                {"text": "Invoice 12 from ACME", "json": {"vendor": "ACME"}}]
        prepared, texts, pairs = self._parity("extraction", rows)
        self.assertIsInstance(prepared[0]["row_context"], str)  # JSON text in prepared splits
        self.assertIn("Extract the following fields as JSON: age, name.", texts[0])
        self.assertIn(rcs.canonical_schema(SCHEMA), texts[0])
        self.assertIn("Extract the following fields as JSON: city.", texts[2])
        self.assertNotIn("JSON Schema", texts[4])  # the row without one keeps the project prompt
        self.assertEqual(pairs[0]["row_output_schema"], SCHEMA)
        self.assertNotIn("row_output_schema", pairs[2])
        self.assertEqual(pairs[0]["row_context_view"]["output_schema_source"], "row")

    def test_json_answer_on_the_qa_handler(self):
        rows = [{"question": "Who is 30?", "answer": '{"name": "Ann", "age": 30}',
                 "row_context": {"output_schema": SCHEMA, "system": "Answer in JSON."}},
                {"question": "Capital of France?", "answer": "Paris"}]
        prepared, texts, pairs = self._parity("qa", rows)
        self.assertEqual(prepared[0]["user_prompt"], "Who is 30?")
        self.assertTrue(prepared[0]["question"].startswith(rcs.output_schema_block(SCHEMA) + "\n\n"))
        self.assertIn("<system>Answer in JSON.</system><user>Reply with a single JSON", texts[0])
        self.assertIn("\n\nWho is 30?</user>", texts[0])
        self.assertNotIn("user_prompt", prepared[1])
        self.assertNotIn("JSON Schema", texts[2])
        self.assertEqual(pairs[0]["row_output_schema"], SCHEMA)

    def test_raw_gold_row_renders_like_a_prepared_one(self):
        from _row_context_golden import eval_render

        raw = [{"question": "Who is 30?", "answer": "{}", "row_context": {"output_schema": SCHEMA}}]
        prepared = training_render({**CASES["qa"], "rows": raw})
        with tempfile.TemporaryDirectory() as d:
            _p1, from_raw = eval_render(CASES["qa"], raw, Path(d))
            _p2, from_prepared = eval_render(CASES["qa"], prepared, Path(d))
        self.assertEqual(from_raw, from_prepared)

    def test_chat_rows_and_non_consumers_are_untouched(self):
        chat = [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "{}"}],
                 "row_context": {"output_schema": SCHEMA}}]
        prepared = training_render({**CASES["chat"], "rows": chat})
        self.assertNotIn("user_prompt", prepared[0])
        tool_rows = [{"prompt": "Get item 7", "tool_name": "get_item", "tool_args": {"id": "7"}, "result": "ok",
                      "row_context": {"output_schema": SCHEMA}}]
        prepared = training_render({**CASES["tool_calling"], "rows": tool_rows})
        self.assertNotIn("JSON Schema", prepared[0]["question"])
        view = rcs.eval_view(tool_rows[0], profile="tool_calling")
        self.assertEqual(view["unused_kinds"], ["output_schema"])


# ── scoring ──────────────────────────────────────────────────────────────


def ext_pred(prediction: dict | str, reference: dict, schema: dict | None = None) -> dict:
    p = {"prediction": prediction if isinstance(prediction, str) else json.dumps(prediction),
         "reference": json.dumps(reference)}
    if schema is not None:
        p["row_output_schema"] = schema
    return p


class ExtractionScoringTests(unittest.TestCase):
    def test_rows_are_scored_on_their_own_schema(self):
        preds = [ext_pred({"name": "Ann", "age": "30"}, {"name": "Ann", "age": 30}, SCHEMA),
                 ext_pred({"town": "Rome"}, {"city": "Rome"}, OTHER),
                 ext_pred({"vendor": "ACME"}, {"vendor": "ACME"})]
        m = StructuredExtractionHandler().score(preds, ctx())
        self.assertEqual(m["row_output_schema_rows"], 2)
        self.assertEqual(preds[0]["row_field_results"].keys(), {"age", "name"})
        self.assertEqual(preds[0]["schema_violations"], {"age": ["type_violation"]})
        self.assertEqual(preds[1]["missing_required_fields"], ["city"])
        self.assertEqual(set(preds[2]["row_field_results"]), {"city", "name", "age", "vendor"} & set(m["schema"]["fields"]))
        self.assertIn("city", m["per_field"])
        self.assertEqual(m["per_field"]["city"]["support"], 2)  # the row schema + the derived project fields

    def test_without_row_schemas_the_metrics_keep_their_shape(self):
        preds = [ext_pred({"vendor": "ACME", "total": 4}, {"vendor": "ACME", "total": 4})]
        m = StructuredExtractionHandler().score(preds, ctx())
        self.assertNotIn("row_output_schema_rows", m)

    def test_generic_block_validates_against_the_row_schema(self):
        preds = [ext_pred({"name": "Ann", "age": 30}, {}, SCHEMA), ext_pred({"name": "Ann", "x": 1}, {}, SCHEMA),
                 ext_pred("not json", {}, OTHER), ext_pred({"vendor": "A"}, {})]
        block = rcs.score_output_schema(preds)
        self.assertEqual(block, {"rows": 3, "json_valid_rate": 0.6667, "schema_valid_rate": 0.3333})
        self.assertTrue(preds[0]["row_output_schema_valid"])
        self.assertFalse(preds[1]["row_output_schema_valid"])
        self.assertTrue(preds[1]["row_output_schema_errors"])
        self.assertNotIn("row_output_schema_valid", preds[3])
        self.assertIsNone(rcs.score_output_schema([ext_pred({"a": 1}, {"a": 1})]))


class NullAsAbsentTests(unittest.TestCase):
    def test_extraction_nested_and_whole_object(self):
        nested = [ext_pred({"name": "Ann", "meta": {"x": None, "y": 1}}, {"name": "Ann", "meta": {"y": 1}})]
        off = StructuredExtractionHandler().score([dict(p) for p in nested], ctx())
        on = StructuredExtractionHandler().score([dict(p) for p in nested], ctx({"null_as_absent": True}))
        self.assertEqual(off["overall_em"], 0.0)
        self.assertEqual(on["overall_em"], 1.0)
        # no fields known (empty references): whole-object compare
        blob = [{"prediction": json.dumps({"a": 1, "b": None}), "reference": json.dumps({"a": 1})}]
        self.assertEqual(StructuredExtractionHandler().score([dict(p) for p in blob], ctx({"output_schema": None}))["overall_em"],
                         1.0)  # top-level null already matched a missing field

    def test_tool_call_args_default_and_strict(self):
        from app.services import tool_call_scoring as tcs

        pred = {"tool": "get", "args": {"id": "1", "limit": None}}
        gold = {"tool": "get", "args": {"id": "1"}}
        self.assertFalse(tcs.args_match_pair(pred, gold, strict=True))
        self.assertTrue(tcs.args_match_pair(pred, gold, strict=True, null_as_absent=True))
        gold_null = {"tool": "get", "args": {"id": "1", "limit": None}}
        self.assertFalse(tcs.args_match_pair({"tool": "get", "args": {"id": "1"}}, gold_null))
        self.assertTrue(tcs.args_match_pair({"tool": "get", "args": {"id": "1"}}, gold_null, null_as_absent=True))

        def score(manifest):
            preds = [{"prediction": json.dumps(pred), "reference": json.dumps(gold),
                      "tool_gold_calls": [{"tool": "get", "args": {"id": "1"}}]}]
            return ToolCallHandler().score(preds, ctx(manifest, "tool_calling"))

        strict_cfg = {"tool_scoring_cfg": {"strict_args": True}}
        self.assertEqual(score(strict_cfg)["tool_call_strict_pass_rate"], 0.0)
        self.assertEqual(score({**strict_cfg, "null_as_absent": True})["tool_call_strict_pass_rate"], 1.0)

    def test_setting_helpers(self):
        self.assertEqual(null_scoring.effective({}, None), (False, "project"))
        rc = null_scoring.with_project_setting({"other": 1}, True)
        self.assertEqual(rc, {"other": 1, "json_scoring": {"null_as_absent": True}})
        self.assertEqual(null_scoring.effective(rc, None), (True, "project"))
        self.assertEqual(null_scoring.effective(rc, False), (False, "run"))
        self.assertEqual(null_scoring.with_project_setting(rc, None), {"other": 1})
        self.assertIsNone(null_scoring.difference({}, {"json_scoring": None}))
        self.assertIn("null", null_scoring.difference({"json_scoring": {"null_as_absent": True}}, {}))

    def test_compare_and_lift_cache_refuse_another_setting(self):
        from app.services.compare_row_evidence_service import compare_provenance
        from app.services.post_training_eval_service import _baseline_scoring_matches

        on = SimpleNamespace(metrics={"prompt_format_version": 2, "json_scoring": {"null_as_absent": True}},
                             eval_type="exact_match")
        off = SimpleNamespace(metrics={"prompt_format_version": 2}, eval_type="exact_match")
        self.assertTrue(_baseline_scoring_matches(off, None))
        self.assertFalse(_baseline_scoring_matches(on, None))
        self.assertTrue(_baseline_scoring_matches(on, None, null_as_absent=True))
        prov = lambda m: {"scoring": {"eval_type": "exact_match", "null_as_absent": null_scoring.result_setting(m)}}  # noqa: E731
        verdict = compare_provenance(prov(on.metrics), prov(off.metrics))
        self.assertFalse(verdict["comparable"])
        self.assertIn("null", verdict["differences"][0]["message"])
        self.assertTrue(compare_provenance(prov(off.metrics), prov(off.metrics))["comparable"])


# ── data health ──────────────────────────────────────────────────────────


class DataHealthTests(unittest.TestCase):
    def test_schema_block_is_context_only_for_rows_that_carry_it(self):
        from app.services.data_health_service import context_block_lines

        rows = training_render({**CASES["qa"], "rows": [
            {"question": "Who is 30?", "answer": "{}", "row_context": {"output_schema": SCHEMA}},
            {"question": "Who is 31?", "answer": "{}", "row_context": {"output_schema": SCHEMA}},
        ]})
        self.assertEqual(context_block_lines(rows), frozenset(rcs.output_schema_lines(SCHEMA)))
        self.assertEqual(context_block_lines([{"question": "q", "answer": "a", "row_context": {"system": "S"}}]),
                         frozenset())

    def test_leak_across_different_schemas_is_still_caught(self):
        from app.services.data_health_service import _scan_leakage_rows, leakage_boilerplate

        q = "What is the full name and age of the person in record 4417 of the city register?"

        def row(schema: dict) -> dict:
            return {"question": rcs.render_output_schema_lead(q, schema), "user_prompt": q,
                    "answer": '{"name": "Ann"}', "row_context": rcs.encode({"output_schema": schema})}

        train, test = [row(SCHEMA)], [row(OTHER)]
        boiler = leakage_boilerplate(train + test, {})
        self.assertTrue(set(rcs.output_schema_lines(OTHER)) <= boiler)
        self.assertEqual(_scan_leakage_rows(train, {"test": test}, boiler)["total_leaked"], 1)


# ── end to end (held-out eval with stubbed inference) ────────────────────


def seed(rows: list[dict], *, profile: str, adapter: str, runtime_config: dict | None = None) -> tuple[int, int]:
    from app.services.dataset_service import _normalize_rows_for_training

    pid = client.post("/api/projects", json={"name": f"outschema-{time.time_ns()}", "description": "x"}).json()["id"]
    if runtime_config:
        async def setup(db):
            project = await db.get(Project, pid)
            project.runtime_config = {**(project.runtime_config or {}), **runtime_config}
            await db.commit()

        run_async(setup)
    prepared_rows = _normalize_rows_for_training(rows, DatasetType.RAW, "llama3", adapter_id=adapter,
                                                 task_profile=profile)
    prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "manifest.json").write_text(json.dumps({"task_profile": profile, "adapter_id": adapter}))
    (prepared / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prepared_rows))

    async def add(db):
        db.add(Dataset(project_id=pid, name="test", dataset_type=DatasetType.TEST,
                       file_path=str(prepared / "test.jsonl"), record_count=len(prepared_rows)))
        exp = Experiment(project_id=pid, name="trained", status=ExperimentStatus.COMPLETED, base_model="toy",
                         config={}, output_dir=str(prepared), completed_at=datetime.now(timezone.utc))
        db.add(exp)
        await db.commit()
        return exp.id

    return pid, run_async(add)


def evaluate(pid: int, exp_id: int, outputs: list[str], pairs_out: list | None = None, **extra) -> dict:
    def run(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True,
            decoding="neutral"):
        if pairs_out is not None:
            pairs_out.extend(dict(p) for p in pairs)
        preds = [{**p, "formatted_prompt": "", "prediction": outputs[i % len(outputs)], "generated_tokens": 5,
                  "latency_ms": 1.0} for i, p in enumerate(pairs)]
        return preds, {"engine": "stub", "decoding": {"mode": decoding}}

    with patch("app.services.evaluation_service._run_local_inference", run):
        resp = client.post(f"/api/projects/{pid}/evaluation/run-heldout", json={
            "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 100,
            **extra})
    assert resp.status_code == 201, resp.text
    return resp.json()


class EndToEndTests(unittest.TestCase):
    def test_extraction_eval(self):
        rows = [{"text": "Ann is 30", "json": {"name": "Ann", "age": 30}, "row_context": {"output_schema": SCHEMA}},
                {"text": "Invoice from ACME", "json": {"vendor": "ACME"}}]
        pid, exp = seed(rows, profile="structured_extraction", adapter="structured-extraction")
        pairs: list[dict] = []
        out = evaluate(pid, exp, [json.dumps({"name": "Ann", "age": 30}), json.dumps({"vendor": "ACME"})], pairs)
        m = out["metrics"]
        self.assertIn(rcs.canonical_schema(SCHEMA), pairs[0]["prompt"])
        self.assertEqual(m["output_schema"], {"rows": 1, "json_valid_rate": 1.0, "schema_valid_rate": 1.0})
        self.assertEqual(m["row_output_schema_rows"], 1)
        self.assertEqual(m["row_context"]["kinds"]["output_schema"], {"row": 1, "none": 1})
        prev = out["details"]["predictions_preview"]
        self.assertTrue(prev[0]["row_output_schema_valid"])
        self.assertEqual(prev[0]["row_context"]["output_schema_source"], "row")
        self.assertNotIn("row_output_schema_valid", prev[1])
        self.assertNotIn("json_scoring", m)

    def test_qa_json_answer_eval(self):
        rows = [{"question": "Who is 30?", "answer": '{"name": "Ann"}', "row_context": {"output_schema": SCHEMA}},
                {"question": "Capital of France?", "answer": "Paris"}]
        pid, exp = seed(rows, profile="qa", adapter="default-canonical")
        pairs: list[dict] = []
        out = evaluate(pid, exp, ['{"name": 5}', "Paris"], pairs)
        self.assertEqual(pairs[0]["prompt"], rcs.render_output_schema_lead("Who is 30?", SCHEMA))
        self.assertEqual(pairs[1]["prompt"], "Capital of France?")
        self.assertEqual(out["metrics"]["output_schema"], {"rows": 1, "json_valid_rate": 1.0, "schema_valid_rate": 0.0})

    def test_no_schema_no_new_keys(self):
        pid, exp = seed([{"question": "Capital of France?", "answer": "Paris"}], profile="qa",
                        adapter="default-canonical")
        m = evaluate(pid, exp, ["Paris"])["metrics"]
        for key in ("output_schema", "json_scoring", "row_context", "row_output_schema_rows"):
            self.assertNotIn(key, m)

    def test_null_setting_project_and_per_run(self):
        rows = [{"text": "Ann", "json": {"name": "Ann", "meta": {"y": 1}}}]
        out_null = json.dumps({"name": "Ann", "meta": {"x": None, "y": 1}})
        pid, exp = seed(rows, profile="structured_extraction", adapter="structured-extraction")
        self.assertEqual(evaluate(pid, exp, [out_null])["metrics"]["overall_em"], 0.0)
        m = evaluate(pid, exp, [out_null], null_as_absent=True)["metrics"]
        self.assertEqual(m["overall_em"], 1.0)
        self.assertEqual(m["json_scoring"], {"null_as_absent": True, "source": "run"})
        resp = client.put(f"/api/projects/{pid}/settings/null-as-absent", json={"enabled": True})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"enabled": True})
        self.assertEqual(client.get(f"/api/projects/{pid}/settings/null-as-absent").json(), {"enabled": True})
        m = evaluate(pid, exp, [out_null])["metrics"]
        self.assertEqual(m["json_scoring"], {"null_as_absent": True, "source": "project"})
        m = evaluate(pid, exp, [out_null], null_as_absent=False)["metrics"]
        self.assertNotIn("json_scoring", m)
        self.assertEqual(m["overall_em"], 0.0)
        resp = client.put(f"/api/projects/{pid}/settings/null-as-absent", json={"enabled": None})
        self.assertEqual(resp.json(), {"enabled": False})

    def test_import_groups_bad_schemas_and_an_eval_set_scores_per_row(self):
        rows = [{"text": "Ann is 30", "structured_output": {"name": "Ann", "age": 30}, "row_context": {"output_schema": SCHEMA}},
                {"text": "Lives in Rome", "structured_output": {"city": "Rome"}, "row_context": {"output_schema": OTHER}},
                {"text": "bad", "structured_output": {"a": 1}, "row_context": {"output_schema": {"type": "nope"}}}]
        pid, exp = seed(rows[:2], profile="structured_extraction", adapter="structured-extraction")
        body = "".join(json.dumps(r) + "\n" for r in rows).encode()
        resp = client.post(f"/api/projects/{pid}/gold/import-file", data={"dataset_type": "gold_dev"},
                           files={"file": ("gold.jsonl", body, "application/octet-stream")})
        self.assertEqual(resp.status_code, 422, resp.text)
        self.assertEqual(resp.json()["metadata"]["rejected"]["by_reason"]["row_context_schema_invalid"]["lines"], [3])
        resp = client.post(f"/api/projects/{pid}/eval-sets", data={"name": "Schemas", "drop_rejected": "true"},
                           files={"file": ("set.jsonl", body, "application/jsonl")})
        self.assertIn(resp.status_code, (200, 201), resp.text)
        slug = resp.json().get("slug") or resp.json()["eval_set"]["slug"]
        pairs: list[dict] = []

        def run(model_ref, prs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True,
                decoding="neutral"):
            pairs.extend(dict(p) for p in prs)
            outs = [json.dumps({"name": "Ann", "age": 30}), json.dumps({"city": "Milan"})]  # not in the row schema enum
            return [{**p, "formatted_prompt": "", "prediction": outs[i], "generated_tokens": 5, "latency_ms": 1.0}
                    for i, p in enumerate(prs)], {"engine": "stub", "decoding": {"mode": decoding}}

        with patch("app.services.evaluation_service._run_local_inference", run):
            resp = client.post(f"/api/projects/{pid}/evaluation/run-heldout", json={
                "experiment_id": exp, "dataset_name": f"eval_set:{slug}", "eval_type": "exact_match", "max_samples": 10})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertIn("Extract the following fields as JSON: city.", pairs[1]["prompt"])
        m = resp.json()["metrics"]
        self.assertEqual(m["output_schema"], {"rows": 2, "json_valid_rate": 1.0, "schema_valid_rate": 0.5})
        self.assertEqual(m["row_output_schema_rows"], 2)


if __name__ == "__main__":
    unittest.main()
