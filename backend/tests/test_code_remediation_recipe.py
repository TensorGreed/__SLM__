"""The ``code-remediation`` recipe: catalog entry, detection, the shared prompt (train == eval,
chat template), passage retrieval, adapter rows, the manifest stamp, remediation scoring, the
``issue_resolved`` detector, the remediation judge, the lift headline and the settings API.

Neutral toy taxonomy and code only; external commands, model inference and judges are stubbed."""

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

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus
from app.services import answer_judge_service as ajs
from app.services import code_remediation_prompt as crp
from app.services import recipe_service, remediation_detector as rd, remediation_scoring as rs
from app.services.data_adapter_service import map_record_with_adapter, unmappable_reason
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler
from app.services.task_shape_service import structured_recipe_for_rows
from app.services.unified_diff import make_unified_diff
from scripts.train import _WRAPS_OWN_PROMPT_PREFIXES

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


TAXONOMY = [
    {"id": "injection", "title": "Injection"},
    {"id": "injection.query", "title": "Query injection", "parent": "injection"},
    {"id": "secret.hardcoded", "title": "Hard-coded secret"},
    {"id": "logging.sensitive", "title": "Sensitive logging"},
]
PARENTS = {"injection.query": "injection"}
CODE = 'def f(cur, uid):\n    q = "SELECT * FROM t WHERE id = " + uid\n    cur.execute(q)\n    return cur.fetchone()\n'
FIXED = 'def f(cur, uid):\n    q = "SELECT * FROM t WHERE id = ?"\n    cur.execute(q, (uid,))\n    return cur.fetchone()\n'
BROKEN = 'def f(cur, uid):\n    q = "SELECT * FROM t WHERE id = ?"\n    cur.execute(q, (uid,\n    return cur.fetchone()\n'
PASSAGES = [
    {"text": "Pass user input as bound parameters.", "source": "query-guidelines"},
    {"text": "def run(cur, sql, args):\n    cur.execute(sql, args)", "path": "lib/db.py", "start_line": 3, "end_line": 4, "symbol": "run"},
]
FINDING = {"label": "injection.query", "start_line": 11, "end_line": 12, "evidence": "q = ... + uid", "attrs": {}}
EXTRA = {"severity": {"type": ["string", "null"], "enum": ["low", "high", None], "description": "How serious the finding is."}}


def patch_of(old: str, new: str, start: int = 10) -> str:
    return make_unified_diff(old, new, path="app/db.py", start_line=start)


def remediation_row(*, passages=True, citations=("P1",), new_code: str = FIXED, extra: dict | None = None, code: str = CODE,
                    language: str | None = "python") -> dict:
    payload = {"patch": patch_of(code, new_code), "rationale": "Bind the id as a parameter.", "citations": list(citations)}
    payload.update(extra or {})
    row = {"row_kind": "remediation", "code": code, "language": language, "path": "app/db.py", "start_line": 10,
           "end_line": 13, "finding": dict(FINDING), "remediation_json": payload}
    if passages:
        row["passages"] = [dict(p) for p in PASSAGES]
    return row


def adapter_config(**over) -> dict:
    cfg = {"extra_fields": {}, "project_id": 0, "retrieve_passages": False, "top_k": 3, "retrieval": None, "label_titles": {}}
    cfg.update(over)
    return {"remediation": cfg}


def map_row(row, config=None):
    return map_record_with_adapter(row, adapter_id="structured-extraction", adapter_config=config or adapter_config())


def prepared_row(row, config=None) -> dict:
    return {**row, **map_row(row, config)}  # what prepared/*.jsonl carries


def manifest_schema(**over) -> dict:
    schema = {
        "scoring_mode": "remediation", "required": ["patch", "rationale", "citations"], "properties": {},
        "remediation": {"checks": None, "retrieve_passages": False, "top_k": 3, "retrieval": None},
        "label_parents": dict(PARENTS), "label_titles": {"injection.query": "Query injection"},
    }
    schema.update(over)
    return schema


def handler_and_ctx(schema=None, project_id: int = 0):
    ctx = EvalContext(project_id=project_id, experiment_id=0, eval_type="exact_match", task_profile="structured_extraction",
                      handler_id="structured_extraction", prepared_dir=Path("."), dataset_name="t",
                      manifest={"output_schema": schema or manifest_schema()})
    return StructuredExtractionHandler(), ctx


def prediction_for(row: dict, payload: dict | str, handler=None, ctx=None) -> dict:
    """A scored-ready prediction: what ``run_heldout_evaluation`` hands the scorer."""
    handler, ctx = (handler, ctx) if handler else handler_and_ctx()
    built = handler.build_prompts([prepared_row(row)], ctx)[0]
    return {"prompt": built.prompt, "reference": built.reference,
            "prediction": payload if isinstance(payload, str) else json.dumps(payload),
            "remediation_row": built.extras.get("remediation_row")}


def score(predictions, schema=None):
    handler, ctx = handler_and_ctx(schema)
    handler.build_prompts([], ctx)  # resolve the schema like the eval does
    return handler.score(predictions, ctx)


GOOD = {"patch": patch_of(CODE, FIXED), "rationale": "Bind it.", "citations": ["P1"]}


class RecipeCatalogTests(unittest.TestCase):
    def test_recipe_fields(self):
        r = recipe_service.get_recipe("code-remediation")
        self.assertEqual(r.name, "Code remediation")
        self.assertEqual((r.task_profile, r.scoring_mode), ("structured_extraction", "remediation"))
        self.assertEqual((r.default_input_column, r.default_output_column), ("code", "remediation_json"))
        self.assertEqual(r.adapter_id, "structured-extraction")
        self.assertEqual(r.suggested_base_model, "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(r.alt_base_models, ["Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct"])
        self.assertEqual(r.gold_template.shape_label, "code_remediation")
        self.assertTrue(r.sample_eval_prompts and r.data_acquisition_hints)
        findings = recipe_service.get_recipe("code-findings")
        self.assertEqual(findings.scoring_mode, "finding_set")

    def test_example_row_is_a_canonical_remediation_row(self):
        row = recipe_service.get_recipe("code-remediation").gold_template.example_row
        self.assertTrue(crp.is_remediation_row(row))
        payload = crp.remediation_payload(row)
        self.assertEqual(payload["citations"], ["P1"])
        result = rs.analyze_row({"prediction": json.dumps(payload), "remediation_row": row}, {}, StructuredExtractionHandler()._parse_json_safely)
        self.assertTrue(result["applies"], result["reason"])
        self.assertTrue(result["parses"])

    def test_sniffing(self):
        top = recipe_service.sniff_recipe_from_headers(["code", "finding", "remediation_json", "language"])[0]
        self.assertEqual(top["recipe_id"], "code-remediation")
        top = recipe_service.sniff_recipe_from_headers(["code", "findings_json"])[0]
        self.assertEqual(top["recipe_id"], "code-findings")
        top = recipe_service.sniff_recipe_from_headers(["diff", "review"])[0]
        self.assertEqual(top["recipe_id"], "code-review")

    def test_detection_from_rows(self):
        self.assertEqual(structured_recipe_for_rows([remediation_row()]), "code-remediation")
        self.assertEqual(structured_recipe_for_rows([{"code": CODE, "remediation_json": '{"patch": "x"}'}]), "code-remediation")
        self.assertEqual(structured_recipe_for_rows([{"text": "t", "structured_output": {"patch": "p", "rationale": "r"}}]),
                         "code-remediation")
        # a record that merely has a "patch" key stays json-extraction; findings rows stay findings
        self.assertEqual(structured_recipe_for_rows([{"text": "a", "json": {"patch": "p"}}]), "json-extraction")
        self.assertEqual(structured_recipe_for_rows([{"row_kind": "findings", "code": CODE}]), "code-findings")

    def test_eval_maps_know_the_recipe(self):
        from app.services.eval_pack_scaffold_service import _RECIPE_SCAFFOLD
        from app.services.evaluation_gate_catalog import _RECOMMENDED_METRIC_IDS_PER_RECIPE
        from app.services.post_training_eval_service import eval_type_for_project
        from app.services.probe_pack_service import get_probe_pack

        spec = _RECIPE_SCAFFOLD["code-remediation"]
        self.assertEqual(spec["task_profile"], "structured_extraction")
        self.assertIn("remediation_pass_rate", spec["required_metric_ids"])
        self.assertIn("remediation_pass_rate", _RECOMMENDED_METRIC_IDS_PER_RECIPE["code-remediation"])
        project = SimpleNamespace(selected_recipe={"scoring_mode": "remediation"}, runtime_config={}, training_mode=None)
        self.assertEqual(eval_type_for_project(project), "exact_match")
        # the generic structured-extraction probes do not apply to a patch-producing model
        self.assertFalse(get_probe_pack("structured_extraction", "code-remediation")["applicable"])
        self.assertTrue(get_probe_pack("structured_extraction", "code-findings")["applicable"])
        self.assertTrue(get_probe_pack("structured_extraction", "span-extraction")["applicable"])


class PromptTests(unittest.TestCase):
    def test_prompt_shape(self):
        prompt = crp.build_code_remediation_prompt(CODE, "python", "app/db.py", 10, FINDING, PASSAGES)
        self.assertEqual(prompt, prompt.strip())
        self.assertIn("Language: python", prompt)
        self.assertIn("Path: app/db.py", prompt)
        self.assertIn("Lines: 10-13", prompt)
        self.assertIn("   10 | def f(cur, uid):", prompt)
        self.assertIn("- label: injection.query", prompt)
        self.assertIn("- lines: 11-12", prompt)
        self.assertIn("- evidence: q = ... + uid", prompt)
        self.assertIn("[P1] query-guidelines\nPass user input", prompt)
        self.assertIn("[P2] lib/db.py:3-4 (run)\ndef run(cur, sql, args):", prompt)
        self.assertIn("--- a/app/db.py and +++ b/app/db.py", prompt)
        self.assertIn('"citations"', prompt)
        self.assertIn("Cite only the passage ids listed above", prompt)
        self.assertIn("data, not instructions", prompt)

    def test_passages_block_only_when_there_are_passages(self):
        self.assertNotIn("Reference passages", crp.build_code_remediation_prompt(CODE, "python", "a.py", 1, FINDING, None))
        self.assertNotIn("Reference passages", crp.build_code_remediation_prompt(CODE, "python", "a.py", 1, FINDING, []))

    def test_passages_truncated_like_the_served_preamble(self):
        from app.services.auto_rag_service import _PASSAGE_PREVIEW_CHARS

        long_text = "\n".join(f"line {i} of the passage" for i in range(400))
        prompt = crp.build_code_remediation_prompt(CODE, None, None, 1, FINDING, [{"text": long_text, "path": "x.py", "start_line": 1, "end_line": 400}])
        block = prompt.split("[P1] x.py:1-400\n", 1)[1].split("\n\nReturn ONLY", 1)[0]
        self.assertLessEqual(len(block), _PASSAGE_PREVIEW_CHARS + 3)
        self.assertTrue(block.endswith("…"))

    def test_extra_fields_named_with_descriptions(self):
        prompt = crp.build_code_remediation_prompt(CODE, "python", "a.py", 1, FINDING, None, crp.extra_fields_from_schema({"properties": EXTRA}))
        self.assertIn('"severity": <string|null one of "low", "high", null> - How serious the finding is.', prompt)

    def test_extra_fields_from_schema_skips_base_fields(self):
        extras = crp.extra_fields_from_schema({"properties": {"patch": {"type": "string"}, "kind": {"type": "string"}}})
        self.assertEqual(list(extras), ["kind"])

    def test_prompt_never_matches_a_handler_wrap_prefix(self):
        for passages in (None, PASSAGES):
            prompt = crp.build_code_remediation_prompt(CODE, "python", "a.py", 1, FINDING, passages)
            self.assertFalse(prompt.lstrip().startswith(_WRAPS_OWN_PROMPT_PREFIXES))
            self.assertFalse(prompt.startswith(tuple(StructuredExtractionHandler().expected_prompt_prefixes())))
        self.assertFalse(crp.build_code_example_prompt("do it", "python", None).startswith(_WRAPS_OWN_PROMPT_PREFIXES))

    def test_canonical_json_is_stable_and_orders_declared_extras(self):
        out = crp.canonical_remediation_json({"citations": ["P1"], "rationale": "r", "severity": "low", "patch": "p", "zzz": 1},
                                             {"severity": {}})
        self.assertEqual(out, '{"patch":"p","rationale":"r","citations":["P1"],"severity":"low","zzz":1}')
        self.assertEqual(crp.canonical_remediation_json('{"patch": "p"}'), '{"patch":"p","rationale":"","citations":[]}')

    def test_normalize_passages_numbers_by_position(self):
        out = crp.normalize_passages([{"id": "P9", "text": "a"}, {"text": "  "}, "plain", {"text": "b", "path": "x.py"}])
        self.assertEqual([p["id"] for p in out], ["P1", "P2", "P3"])
        self.assertEqual(out[1]["text"], "plain")

    def test_code_example_prompt_and_payload(self):
        prompt = crp.build_code_example_prompt("Read a row by id.", "python", PASSAGES)
        self.assertIn("Task:\nRead a row by id.", prompt)
        self.assertIn("[P1] query-guidelines", prompt)
        self.assertIn('"code"', prompt)
        row = {"row_kind": "code_example", "task": "Read a row by id.", "language": "python", "passages": PASSAGES,
               "code_example_json": {"code": "x = 1", "rationale": "r", "citations": ["P1"]}}
        self.assertTrue(crp.is_code_example_row(row))
        self.assertFalse(crp.is_remediation_row(row))
        self.assertEqual(crp.build_code_example_prompt_from_row(row), prompt)
        self.assertEqual(crp.canonical_code_example_json(row["code_example_json"]), '{"code":"x = 1","rationale":"r","citations":["P1"]}')


class AdapterAndIdentityTests(unittest.TestCase):
    def test_remediation_row(self):
        row = remediation_row()
        mapped = map_row(row)
        self.assertEqual(mapped["text"], CODE)  # indentation intact
        self.assertEqual(mapped["target_text"], crp.canonical_remediation_json(row["remediation_json"]))
        self.assertEqual(json.loads(mapped["target_text"]), row["remediation_json"])
        self.assertEqual([p["id"] for p in mapped["passages"]], ["P1", "P2"])
        self.assertEqual(mapped["source_text"], crp.build_code_remediation_prompt_from_row({**row, "passages": mapped["passages"]}))
        self.assertNotIn("passages_retrieved", mapped)

    def test_extra_fields_ride_in_the_target_and_the_prompt(self):
        extras = crp.extra_fields_from_schema({"properties": EXTRA})
        row = remediation_row(extra={"severity": "high"})
        mapped = map_row(row, adapter_config(extra_fields=extras))
        self.assertEqual(json.loads(mapped["target_text"])["severity"], "high")
        self.assertIn('"severity"', mapped["source_text"])

    def test_code_example_row_is_a_training_row(self):
        row = {"row_kind": "code_example", "task": "Read a row by id.", "language": "python", "passages": PASSAGES,
               "code_example_json": {"code": "x = 1", "rationale": "r", "citations": ["P1"]}}
        mapped = map_row(row)
        self.assertEqual(mapped["source_text"], crp.build_code_example_prompt_from_row(row))
        self.assertEqual(json.loads(mapped["target_text"]), row["code_example_json"])

    def test_unusable_rows_rejected_with_reasons(self):
        cfg = adapter_config()
        no_json = {k: v for k, v in remediation_row().items() if k != "remediation_json"}
        self.assertIsNone(map_row(no_json))
        self.assertEqual(unmappable_reason(no_json, adapter_id="structured-extraction", adapter_config=cfg), "remediation_json_missing")
        no_finding = {k: v for k, v in remediation_row().items() if k != "finding"}
        self.assertEqual(unmappable_reason(no_finding, adapter_id="structured-extraction", adapter_config=cfg), "remediation_finding_missing")
        self.assertIsNone(map_row({**remediation_row(), "code": "  "}))
        from app.services.dataset_service import INPUT_REJECT_REASONS

        for reason in ("remediation_json_missing", "remediation_finding_missing"):
            self.assertIn(reason, INPUT_REJECT_REASONS)

    def test_findings_and_other_modes_unchanged(self):
        mapped = map_record_with_adapter(
            {"row_kind": "findings", "code": CODE, "findings_json": {"findings": []}},
            adapter_id="structured-extraction", adapter_config={"finding_set": True})
        self.assertTrue(mapped["source_text"].startswith("Inspect the code chunk below"))

    def test_train_and_eval_prompts_are_byte_identical(self):
        for row in (remediation_row(), remediation_row(passages=False), remediation_row(citations=())):
            prepared = prepared_row(row)
            handler, ctx = handler_and_ctx()
            built = handler.build_prompts([prepared], ctx)[0]
            self.assertEqual(built.prompt, prepared["source_text"])
            self.assertEqual(built.reference, prepared["target_text"])
            self.assertEqual(built.extras["remediation_row"]["finding"]["label"], "injection.query")
            self.assertEqual(built.extras["path"], "app/db.py")
        # with declared extra fields on both sides
        extras = crp.extra_fields_from_schema({"properties": EXTRA})
        row = remediation_row(extra={"severity": "low"})
        prepared = prepared_row(row, adapter_config(extra_fields=extras))
        handler, ctx = handler_and_ctx(manifest_schema(properties=EXTRA))
        self.assertEqual(handler.build_prompts([prepared], ctx)[0].prompt, prepared["source_text"])

    def test_code_example_rows_are_skipped_by_the_scorer_not_the_prompt_builder(self):
        row = {"row_kind": "code_example", "task": "Read a row by id.", "language": "python", "passages": PASSAGES,
               "code_example_json": {"code": "x = 1", "rationale": "r", "citations": ["P1"]}}
        prepared = prepared_row(row)
        handler, ctx = handler_and_ctx()
        built = handler.build_prompts([prepared], ctx)[0]
        self.assertEqual(built.prompt, prepared["source_text"])
        self.assertNotIn("remediation_row", built.extras)

    def test_wraps_own_prompt_is_false_for_remediation(self):
        from app.services.eval_task_handler_service import handler_wraps_own_prompt

        self.assertFalse(handler_wraps_own_prompt(StructuredExtractionHandler(), {"output_schema": {"scoring_mode": "remediation"}}))
        self.assertTrue(handler_wraps_own_prompt(StructuredExtractionHandler(), {"output_schema": {"scoring_mode": "span_set"}}))

    def test_generation_budget_grows_for_patches(self):
        handler, ctx = handler_and_ctx()
        handler.build_prompts([], ctx)
        self.assertGreaterEqual(handler.max_new_tokens_override(128), 768)
        self.assertEqual(StructuredExtractionHandler().max_new_tokens_override(1000), 512)


class RetrievalTests(unittest.TestCase):
    @staticmethod
    def hit(text, **payload):
        return {"row_id": 1, "score": 1.0, "payload": {"text": text, **payload}}

    def setUp(self):
        crp._RETRIEVE_FNS.clear()
        self.addCleanup(crp._RETRIEVE_FNS.clear)

    def test_query_and_passages(self):
        seen = []

        def fn(query, cfg):
            seen.append((query, dict(cfg)))
            return [self.hit("guide one", source_doc="guide", id="d:0"),
                    self.hit("def f(cur, uid):", path="app/db.py", start_line=10, end_line=13),  # the row's own chunk
                    self.hit("guide two", path="lib/x.py", start_line=1, end_line=5, symbol="g", id="d:1")]

        row = remediation_row(passages=False)
        out = crp.retrieve_passages_for_row(7, row, {"k": 9, "reranker": None}, 2, label_title="Query injection", retrieve_fn=fn)
        self.assertEqual([p["id"] for p in out], ["P1", "P2"])
        self.assertEqual(out[0]["source"], "guide")
        self.assertEqual(out[0]["passage_id"], "d:0")
        self.assertEqual((out[1]["path"], out[1]["symbol"]), ("lib/x.py", "g"))  # own chunk dropped
        query, cfg = seen[0]
        self.assertTrue(query.startswith("injection.query Query injection q = ... + uid def f(cur, uid):"))
        self.assertLessEqual(len(query), len("injection.query Query injection q = ... + uid ") + crp.QUERY_CODE_CHARS)
        self.assertEqual(cfg["k"], 4)  # top_k + headroom for the dropped self-hit
        self.assertEqual(crp.retrieve_passages_for_row(7, row, None, 2, retrieve_fn=fn), crp.retrieve_passages_for_row(7, row, None, 2, retrieve_fn=fn))

    def test_failure_means_no_passages(self):
        def boom(query, cfg):
            raise ValueError("no index")

        self.assertEqual(crp.retrieve_passages_for_row(7, remediation_row(passages=False), None, 3, retrieve_fn=boom), [])

    def test_rows_with_passages_are_never_retrieved(self):
        calls = []
        cfg = {"retrieve_passages": True, "top_k": 3}
        with patch.object(crp, "retrieve_passages_for_row", side_effect=lambda *a, **k: calls.append(a) or []):
            kept = crp.with_retrieved_passages(1, remediation_row(), cfg)
            self.assertEqual([p["id"] for p in kept], ["P1", "P2"])
            crp.with_retrieved_passages(1, remediation_row(passages=False), cfg)
            crp.with_retrieved_passages(1, remediation_row(passages=False), {"retrieve_passages": False})
        self.assertEqual(len(calls), 1)

    def test_adapter_retrieves_once_and_the_row_carries_the_passages(self):
        retrieved = [{"id": "P1", "text": "from the index", "source": "idx", "path": None, "start_line": None, "end_line": None,
                      "symbol": None, "passage_id": "d:0"}]
        with patch.object(crp, "retrieve_passages_for_row", return_value=retrieved) as fetch:
            row = remediation_row(passages=False)
            mapped = map_row(row, adapter_config(retrieve_passages=True, project_id=5, top_k=2))
            fetch.assert_called_once()
            self.assertEqual(fetch.call_args.args[0], 5)
            self.assertEqual(fetch.call_args.args[3], 2)
            self.assertTrue(mapped["passages_retrieved"])
            self.assertIn("[P1] idx\nfrom the index", mapped["source_text"])
            # eval re-renders the identical prompt from the prepared row, without retrieving again
            prepared = {**row, **mapped}
            handler, ctx = handler_and_ctx(manifest_schema(remediation={"checks": None, "retrieve_passages": True, "top_k": 2, "retrieval": None}), project_id=5)
            fetch.reset_mock()
            built = handler.build_prompts([prepared], ctx)[0]
            fetch.assert_not_called()
            self.assertEqual(built.prompt, mapped["source_text"])
            # a row that already carries passages is never retrieved
            map_row(remediation_row(), adapter_config(retrieve_passages=True, project_id=5))
            fetch.assert_not_called()

    def test_handler_retrieves_for_a_row_the_adapter_never_saw(self):
        retrieved = [{"id": "P1", "text": "late passage", "source": "idx"}]
        with patch.object(crp, "retrieve_passages_for_row", return_value=retrieved) as fetch:
            schema = manifest_schema(remediation={"checks": None, "retrieve_passages": True, "top_k": 4, "retrieval": {"k": 4}})
            handler, ctx = handler_and_ctx(schema, project_id=9)
            built = handler.build_prompts([remediation_row(passages=False)], ctx)[0]
            fetch.assert_called_once()
            self.assertEqual((fetch.call_args.args[0], fetch.call_args.args[3]), (9, 4))
            self.assertEqual(fetch.call_args.kwargs["label_title"], "Query injection")
            self.assertIn("late passage", built.prompt)
            self.assertEqual(built.extras["remediation_row"]["passages"][0]["text"], "late passage")

    def test_default_follows_the_document_index(self):
        with patch.object(crp, "has_document_index", return_value=True):
            self.assertTrue(crp.effective_remediation_config({}, 3)["retrieve_passages"])
        with patch.object(crp, "has_document_index", return_value=False):
            self.assertFalse(crp.effective_remediation_config({}, 3)["retrieve_passages"])
            self.assertTrue(crp.effective_remediation_config({"remediation": {"retrieve_passages": True}}, 3)["retrieve_passages"])
        cfg = crp.effective_remediation_config({"remediation": {"top_k": 99, "judge": False, "checks": ["a"]}})
        self.assertEqual((cfg["top_k"], cfg["judge"], cfg["checks"], cfg["detector"]), (10, False, ["a"], None))


class ScoringTests(unittest.TestCase):
    def test_every_row_flag_and_rate(self):
        rows = [
            prediction_for(remediation_row(), GOOD),                                              # 0 passes
            prediction_for(remediation_row(), "not json at all"),                                 # 1 malformed
            prediction_for(remediation_row(), {"patch": patch_of(CODE.replace("uid", "key"), FIXED.replace("uid", "key")),
                                               "rationale": "r", "citations": []}),               # 2 context mismatch
            prediction_for(remediation_row(), {**GOOD, "patch": patch_of(CODE, BROKEN)}),         # 3 applies, breaks parsing
            prediction_for(remediation_row(), {**GOOD, "citations": ["P7"]}),                     # 4 invented citation
            prediction_for(remediation_row(passages=False, citations=()), {**GOOD, "citations": []}),  # 5 passes, no passages
            prediction_for(remediation_row(language="brainfuck", passages=False, citations=()), {**GOOD, "citations": []}),  # 6 not checkable
            {"prompt": "p", "reference": "{}", "prediction": "{}", "remediation_row": None},      # 7 not a remediation row
        ]
        m = score(rows)
        self.assertEqual(m["scoring_mode"], "remediation")
        self.assertEqual(m["total"], 7)
        self.assertEqual(m["skipped_non_remediation_rows"], 1)
        self.assertTrue(rows[7]["row_skipped"])

        r0, r1, r2, r3, r4, r5, r6 = rows[:7]
        self.assertTrue(r0["row_remediation_pass"] and r0["row_patch_applies"] and r0["row_code_checkable"] and r0["row_code_parses"])
        self.assertEqual(r0["row_exact_match"], 1.0)
        self.assertTrue(r0["row_citations_valid"] and r0["row_cited"])
        self.assertEqual(r0["row_patch_offsets"], [0])
        self.assertTrue(r1["row_malformed"] and not r1["row_patch_applies"] and not r1["row_remediation_pass"])
        self.assertEqual(r1["row_patch_reason"], "malformed")
        self.assertFalse(r2["row_patch_applies"])
        self.assertEqual(r2["row_patch_reason"], "context_mismatch")
        self.assertFalse(r2["row_code_parses"])  # checkable + patch did not apply = does not parse
        self.assertTrue(r3["row_patch_applies"] and r3["row_code_checkable"])
        self.assertFalse(r3["row_code_parses"])
        self.assertFalse(r3["row_remediation_pass"])
        self.assertTrue(r3["row_check_messages"] and r3["row_check_messages"][0].startswith("python_ast:"))
        self.assertLessEqual(len(r3["row_check_messages"]), 5)
        self.assertFalse(r4["row_citations_valid"])
        self.assertFalse(r4["row_remediation_pass"])
        self.assertEqual(r4["row_cited_ids"], ["P7"])
        self.assertIsNone(r5["row_citations_valid"])  # no passages: nothing to check
        self.assertTrue(r5["row_remediation_pass"])
        self.assertFalse(r6["row_code_checkable"])
        self.assertIsNone(r6["row_code_parses"])
        self.assertTrue(r6["row_remediation_pass"])  # applies; parsing is only required where checkable
        self.assertIsNone(r0["row_issue_resolved"])  # no detector configured
        self.assertIsNone(r0["row_detector"])

        self.assertEqual(m["correct"], 3)
        self.assertAlmostEqual(m["remediation_pass_rate"], round(3 / 7, 4))
        self.assertEqual(m["exact_match"], m["remediation_pass_rate"])
        self.assertAlmostEqual(m["patch_applies_rate"], round(5 / 7, 4))  # rows 0, 3, 4, 5, 6
        # checkable depends on the ORIGINAL chunk only (a malformed prediction does not shrink the
        # denominator): rows 0-5 are python
        self.assertEqual(m["code_checkable_count"], 6)
        self.assertTrue(r1["row_code_checkable"])
        self.assertIs(r1["row_code_parses"], False)  # malformed on a checkable row = does not parse
        self.assertAlmostEqual(m["code_parses_rate"], 0.5)  # 0, 4 and 5 parse; 1, 2 and 3 do not
        self.assertAlmostEqual(m["malformed_rate"], round(1 / 7, 4))
        self.assertAlmostEqual(m["json_validity_rate"], round(6 / 7, 4))
        self.assertEqual(m["patch_reasons"], {"context_mismatch": 1, "malformed": 1})
        self.assertEqual(m["rows_with_passages"], 5)  # rows 0-4
        self.assertAlmostEqual(m["citations_valid_rate"], 0.6)  # 0, 2, 3 (a malformed row cites nothing validly; 7 is invented)
        self.assertAlmostEqual(m["citation_rate"], 0.6)  # 0, 3, 4 cite something
        for absent in ("issue_resolved_rate", "new_findings_rate", "detector_evaluated_count"):
            self.assertNotIn(absent, m)

    def test_empty_and_all_skipped(self):
        self.assertEqual(score([])["total"], 0)
        m = score([{"prompt": "p", "reference": "{}", "prediction": "{}"}])
        self.assertEqual((m["total"], m["skipped_non_remediation_rows"]), (0, 1))

    def test_a_patch_that_is_not_a_diff_or_names_another_file(self):
        rows = [
            prediction_for(remediation_row(), {**GOOD, "patch": "just prose"}),
            prediction_for(remediation_row(), {**GOOD, "patch": patch_of(CODE, FIXED).replace("app/db.py", "other.py")}),
            prediction_for(remediation_row(), {**GOOD, "patch": ""}),
        ]
        m = score(rows)
        self.assertEqual(m["patch_applies_rate"], 0.0)
        self.assertEqual(set(m["patch_reasons"]), {"not_a_diff", "other_file", "empty_patch"})
        self.assertEqual(m["malformed_rate"], 0.0)  # a string patch is well-formed, however wrong

    def test_missing_patch_key_is_malformed(self):
        m = score([prediction_for(remediation_row(), {"rationale": "r", "citations": []})])
        self.assertEqual(m["malformed_rate"], 1.0)

    def test_check_selection_follows_the_manifest(self):
        rows = [prediction_for(remediation_row(), GOOD)]
        schema = manifest_schema(remediation={"checks": ["json_parse"], "retrieve_passages": False, "top_k": 3, "retrieval": None})
        m = score(rows, schema)
        self.assertEqual(m["code_checkable_count"], 0)  # python code is not json: that check does not pass the original
        self.assertTrue(rows[0]["row_remediation_pass"])

    def test_extra_fields_are_field_matched(self):
        schema = manifest_schema(properties=EXTRA)
        ref = remediation_row(extra={"severity": "high"})
        rows = [
            prediction_for(ref, {**GOOD, "severity": "high"}, *handler_and_ctx(schema)),
            prediction_for(ref, {**GOOD, "severity": "low"}, *handler_and_ctx(schema)),
        ]
        m = score(rows, schema)
        self.assertEqual(rows[0]["row_extra_fields"], {"severity": True})
        self.assertEqual(rows[1]["row_extra_fields"], {"severity": False})
        self.assertEqual(m["extra_fields"]["severity"], {"correct": 1, "total": 2, "accuracy": 0.5})
        self.assertEqual(m["extra_field_accuracy_mean"], 0.5)
        self.assertEqual(m["remediation_pass_rate"], 1.0)  # extras are reported, not gating

    def test_analysis_is_cached_and_precompute_matches(self):
        rows = [prediction_for(remediation_row(), GOOD)]
        handler, ctx = handler_and_ctx()
        schema = handler._resolve_schema(rows, ctx)
        rs.precompute(rows, schema, handler._parse_json_safely)
        cached = rows[0]["_remediation_analysis"]
        with patch("app.services.code_check_service.run_code_checks", side_effect=AssertionError("re-run")):
            m = handler.score(rows, ctx)
        self.assertEqual(m["remediation_pass_rate"], 1.0)
        self.assertIs(rows[0]["_remediation_analysis"], cached)

    def test_checks_run_through_the_registry_so_a_stub_decides(self):
        calls = []

        def fake(code, *, language, check_ids=None, context=None):
            calls.append((code, check_ids))
            return {"ok": True, "results": [{"check_id": "stub", "ok": True, "messages": [], "skipped": None}]}

        with patch("app.services.code_check_service.run_code_checks", side_effect=fake):
            m = score([prediction_for(remediation_row(), GOOD)])
        self.assertEqual(m["code_checkable_count"], 1)
        self.assertEqual(calls[1][1], ["stub"])  # the patched code runs the same checks that ran on the original
        self.assertEqual(calls[1][0], FIXED)


class DetectorTests(unittest.TestCase):
    def run_detector(self, rows, outputs, *, schema=None, error=None):
        """Annotate ``rows`` with a stub detector. ``outputs(code) -> labels | None``."""
        handler, ctx = handler_and_ctx(schema)
        handler.build_prompts([], ctx)
        schema = handler._resolve_schema(rows, ctx)

        def detect(chunks):
            if error:
                raise RuntimeError(error)
            return [outputs(c["code"]) for c in chunks]

        det = rd.ResolvedDetector("project", "Toy detector · run 4", f"test:{time.time_ns()}", detect)

        async def resolve(db, cfg):
            return det

        pid = 900_000 + int(time.time_ns() % 100_000)
        with patch.object(rd, "resolve_detector", resolve):
            snapshot = asyncio.run(rd.annotate_predictions(None, pid, rows, {"kind": "project", "project_id": 1}, schema, handler._parse_json_safely))
        return handler.score(rows, ctx), snapshot

    def rows(self):
        return [prediction_for(remediation_row(), GOOD) for _ in range(1)]

    def test_resolved(self):
        rows = self.rows()
        m, snap = self.run_detector(rows, lambda code: ["injection.query"] if "+ uid" in code else [])
        self.assertTrue(rows[0]["row_issue_resolved"])
        self.assertEqual(rows[0]["row_new_findings"], [])
        self.assertEqual(rows[0]["row_detector"], "Toy detector · run 4")
        self.assertEqual((m["issue_resolved_rate"], m["new_findings_rate"], m["detector_evaluated_count"]), (1.0, 0.0, 1))
        self.assertEqual(m["detector"], "Toy detector · run 4")
        self.assertTrue(rows[0]["row_remediation_pass"])
        self.assertEqual(snap["detector"], "Toy detector · run 4")

    def test_not_resolved_fails_the_row(self):
        rows = self.rows()
        m, _ = self.run_detector(rows, lambda code: ["injection.query"])
        self.assertFalse(rows[0]["row_issue_resolved"])
        self.assertFalse(rows[0]["row_remediation_pass"])
        self.assertEqual((m["issue_resolved_rate"], m["remediation_pass_rate"]), (0.0, 0.0))

    def test_new_finding_is_reported_not_failing(self):
        rows = self.rows()
        m, _ = self.run_detector(rows, lambda code: ["injection.query"] if "+ uid" in code else ["logging.sensitive"])
        self.assertTrue(rows[0]["row_issue_resolved"])
        self.assertEqual(rows[0]["row_new_findings"], ["logging.sensitive"])
        self.assertEqual(m["new_findings_rate"], 1.0)
        self.assertTrue(rows[0]["row_remediation_pass"])

    def test_descendant_label_counts_as_the_finding(self):
        # the finding is the parent label; the patched code still shows a child of it
        ref = remediation_row()
        ref["finding"] = {**FINDING, "label": "injection"}
        rows = [prediction_for(ref, GOOD)]
        m, _ = self.run_detector(rows, lambda code: ["injection.query"])
        self.assertFalse(rows[0]["row_issue_resolved"])
        # an unrelated label does not
        rows = [prediction_for(ref, GOOD)]
        self.run_detector(rows, lambda code: ["injection"] if code == CODE else ["secret.hardcoded"])
        self.assertTrue(rows[0]["row_issue_resolved"])

    def test_detector_error_leaves_the_row_unevaluated(self):
        rows = self.rows()
        m, snap = self.run_detector(rows, lambda code: [], error="model failed to load")
        self.assertIsNone(rows[0]["row_issue_resolved"])
        self.assertTrue(rows[0]["row_remediation_pass"])  # an unevaluated check never fails a row
        self.assertNotIn("issue_resolved_rate", m)
        self.assertEqual(m["detector_error_count"], 1)
        self.assertIn("detector_error", m)
        self.assertIn("model failed to load", snap["error"])

    def test_unparseable_detector_output_is_an_error_for_that_row(self):
        rows = self.rows()
        m, _ = self.run_detector(rows, lambda code: None)
        self.assertIsNone(rows[0]["row_issue_resolved"])
        self.assertEqual(m["detector_error_count"], 1)

    def test_rows_whose_patch_does_not_apply_are_not_sent_to_the_detector(self):
        bad = prediction_for(remediation_row(), {**GOOD, "patch": "prose"})
        seen = []

        def outputs(code):
            seen.append(code)
            return []

        m, _ = self.run_detector([bad], outputs)
        self.assertEqual(seen, [])
        self.assertIsNone(bad["row_issue_resolved"])
        self.assertNotIn("detector_evaluated_count", m)

    def test_verdicts_are_cached_per_project(self):
        rows = self.rows()
        det_calls = []
        handler, ctx = handler_and_ctx()
        handler.build_prompts([], ctx)
        schema = handler._resolve_schema(rows, ctx)
        det = rd.ResolvedDetector("project", "Toy detector · run 4", "cache-test-identity",
                                  lambda chunks: det_calls.append(len(chunks)) or [[] for _ in chunks])

        async def resolve(db, cfg):
            return det

        pid = 910_000 + int(time.time_ns() % 100_000)
        with patch.object(rd, "resolve_detector", resolve):
            for _ in range(2):
                asyncio.run(rd.annotate_predictions(None, pid, rows, {"kind": "project"}, schema, handler._parse_json_safely))
        self.assertEqual(det_calls, [2])  # original + patched once; the second pass is all cache
        self.assertTrue(rd.cache_path(pid).exists())

    def test_parse_finding_labels(self):
        self.assertEqual(rd.parse_finding_labels('{"findings": [{"label": "a", "start_line": 1, "end_line": 2}]}'), ["a"])
        self.assertEqual(rd.parse_finding_labels('```json\n{"findings": []}\n```'), [])
        self.assertIsNone(rd.parse_finding_labels("no idea"))


class JudgeTests(unittest.TestCase):
    def test_default_prompt_is_unchanged_byte_for_byte(self):
        expected = (
            "QUESTION:\nq\n\n"
            "REFERENCE ANSWER (treat as ground truth):\nr\n\n"
            "MODEL ANSWER:\np\n\n"
            "Verdict rules:\n"
            "- correct: the model answer states the reference's key facts (paraphrase is fine) "
            "and nothing that contradicts them.\n"
            "- partial: it gets some key facts right but misses important ones, or adds a "
            "claim that is not in the reference.\n"
            "- wrong: it contradicts the reference, answers a different question, is empty, "
            "or is mostly unsupported.\n"
            "An answer that reuses the question's words without giving the facts is wrong."
        )
        self.assertEqual(ajs.build_judge_prompt("q", "r", "p"), expected)
        self.assertEqual(ajs.build_judge_prompt("q", "r", "p", None), expected)
        self.assertEqual(ajs.build_judge_prompt("q", "r", "p", "  "), expected)
        self.assertEqual(ajs.cache_key("j", "q", "r", "p"), ajs.cache_key("j", "q", "r", "p", None))
        retrieval = ajs.build_retrieval_judge_prompt("q", "r", ["a"])
        self.assertEqual(retrieval, ajs.build_retrieval_judge_prompt("q", "r", ["a"], None))
        self.assertNotIn("GRADING INSTRUCTION", retrieval)

    def test_instruction_lands_before_the_verdict_rules_and_changes_the_cache_key(self):
        prompt = ajs.build_judge_prompt("q", "r", "p", "Grade the patch.")
        self.assertIn("MODEL ANSWER:\np\n\nGRADING INSTRUCTION:\nGrade the patch.\n\nVerdict rules:", prompt)
        self.assertNotEqual(ajs.cache_key("j", "q", "r", "p"), ajs.cache_key("j", "q", "r", "p", "Grade the patch."))
        self.assertIn("GRADING INSTRUCTION:\nx", ajs.build_retrieval_judge_prompt("q", "r", ["a"], "x"))

    def test_three_argument_judges_keep_working_and_instruction_is_passed_only_when_set(self):
        seen = []

        async def three(question, reference, prediction):
            seen.append(("three", question))
            return 1.0, "correct", "ok", 0

        async def four(question, reference, prediction, instruction=None):
            seen.append(("four", instruction))
            return 0.5, "partial", "meh", 0

        preds = [{"prompt": "q", "reference": "r", "prediction": "p"}]
        asyncio.run(ajs.judge_predictions(preds, three, label="t"))
        asyncio.run(ajs.judge_predictions([dict(preds[0])], four, label="t"))
        asyncio.run(ajs.judge_predictions([dict(preds[0])], four, label="t", instruction="rubric"))
        self.assertEqual(seen, [("three", "q"), ("four", None), ("four", "rubric")])

    def test_should_judge_remediation(self):
        self.assertTrue(ajs.should_judge("structured_extraction", ["short"], "remediation"))
        self.assertFalse(ajs.should_judge("structured_extraction", [""], "remediation"))
        self.assertFalse(ajs.should_judge("structured_extraction", ["a long reference " * 10], None))

    def judged(self, project_runtime_config=None):
        seen = {"judge": [], "retrieval": []}

        async def judge(question, reference, prediction, instruction=None):
            seen["judge"].append((question, reference, prediction, instruction))
            return 1.0, "correct", "fine", 0

        async def retrieval(question, reference, passages, instruction=None):
            seen["retrieval"].append((question, passages, instruction))
            return "found", 1, "p1", 0

        rows = [prediction_for(remediation_row(), GOOD), prediction_for(remediation_row(passages=False, citations=()), {**GOOD, "rationale": "Other.", "citations": []})]
        rows.append({"prompt": "p", "reference": "{}", "prediction": "{}"})  # not a remediation row
        score(rows)
        resolved = ajs.ResolvedJudge(label="stub:judge", judge=judge, retrieval_judge=retrieval)
        project = SimpleNamespace(runtime_config=project_runtime_config or {})

        async def go():
            from app.services import evaluation_service as ev

            class Db:
                async def get(self, model, pk):
                    return project

            with patch.object(ajs, "project_cache", lambda pid: ajs.AnswerJudgeCache(None)):
                return await ev._safe_judge_answers(
                Db(), project_id=1, task_profile="structured_extraction", predictions=rows, judge=resolved, scoring_mode="remediation")

        return asyncio.run(go()), seen, rows

    def test_remediation_judge_uses_the_rubric_and_retrieval_judge(self):
        snap, seen, rows = self.judged()
        self.assertEqual(snap["judged"], 2)
        self.assertEqual(snap["score"], 1.0)
        self.assertEqual(len(seen["judge"]), 2)
        question, reference, prediction, instruction = seen["judge"][0]
        self.assertIn("injection.query", question)
        self.assertIn("   10 | def f(cur, uid):", question)
        self.assertIn("PATCH:", reference)
        self.assertIn("RATIONALE: Bind it.", prediction)
        self.assertEqual(instruction, rs.JUDGE_INSTRUCTION)
        self.assertEqual(rows[0]["row_judge_verdict"], "correct")
        self.assertNotIn("row_judge_verdict", rows[2])
        # only the row with passages goes to the retrieval judge
        self.assertEqual(len(seen["retrieval"]), 1)
        self.assertEqual(snap["retrieval"]["found_rate"], 1.0)
        self.assertEqual(len(seen["retrieval"][0][1]), 2)

    def test_judge_can_be_turned_off_per_project(self):
        snap, seen, _ = self.judged({"remediation": {"judge": False}})
        self.assertEqual(snap, {"skipped": "disabled_for_project"})
        self.assertEqual(seen["judge"], [])

    def test_a_remediation_pass_decides_failure_not_the_judge(self):
        from app.services.evaluation_service import _prediction_failed, _row_scores_for_pairing

        passing = {"row_remediation_pass": True, "row_judge_score": 0.0}
        failing = {"row_remediation_pass": False, "row_judge_score": 1.0}
        self.assertFalse(_prediction_failed(passing))
        self.assertTrue(_prediction_failed(failing))
        self.assertFalse(_prediction_failed({"row_skipped": True, "reference": "x", "prediction": "y"}))
        scores = _row_scores_for_pairing([{"prompt": "a", "reference": "r", **passing}, {"prompt": "b", "reference": "r", **failing},
                                          {"prompt": "c", "reference": "r", "row_skipped": True}])
        self.assertEqual(scores["remediation_pass"], [1.0, 0.0, None])


class LiftSummaryTests(unittest.TestCase):
    def test_headline_and_row_pairing(self):
        from app.services import sft_lift_summary_service as lift

        self.assertEqual(lift._PREFERRED_HEADLINE_METRICS[0], "remediation_pass_rate")
        self.assertEqual(lift._ROW_SCORE_FOR_METRIC["remediation_pass_rate"], "remediation_pass")
        base = {"remediation_pass_rate": 0.25, "patch_applies_rate": 0.5, "judge_correct": 0.4, "f1": 0.3,
                "code_checkable_count": 5.0, "new_findings_rate": 0.3}
        tuned = {"remediation_pass_rate": 0.75, "patch_applies_rate": 0.9, "judge_correct": 0.7, "f1": 0.35,
                 "code_checkable_count": 6.0, "new_findings_rate": 0.1}
        lifts = lift._compute_metric_lifts(lift._normalize_metrics(base), lift._normalize_metrics(tuned))
        self.assertEqual(lifts[0]["metric_id"], "remediation_pass_rate")
        self.assertTrue(lifts[0]["is_headline"])
        self.assertNotIn("code_checkable_count", [row["metric_id"] for row in lifts])
        by_id = {row["metric_id"]: row for row in lifts}
        # prediction-dependent denominators are not lift metrics
        self.assertNotIn("new_findings_rate", by_id)

        def result(keys, passes):
            return SimpleNamespace(details={"row_scores": {"keys": keys, "remediation_pass": passes}})

        paired = lift._paired_row_scores(
            result(["a", "b", "c", "d"], [0.0, 1.0, 0.0, None]), result(["a", "b", "c", "d"], [1.0, 1.0, 0.0, None]), "remediation_pass_rate")
        self.assertEqual(paired, ([0.0, 1.0, 0.0], [1.0, 1.0, 0.0]))


class ManifestAndConfigApiTests(unittest.TestCase):
    def new_project(self, recipe="code-remediation") -> int:
        pid = client.post("/api/projects", json={"name": f"cr-{time.time_ns()}", "description": "x"}).json()["id"]
        if recipe:
            put = client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": recipe})
            self.assertEqual(put.status_code, 200, put.text)
        return pid

    @staticmethod
    def run_async(coro_fn):
        async def go():
            async with async_session_factory() as db:
                return await coro_fn(db)

        return asyncio.run(go())

    def test_recipe_applies_and_accepts_an_output_schema(self):
        pid = self.new_project()
        project = client.get(f"/api/projects/{pid}").json()
        self.assertEqual(project["selected_recipe"]["scoring_mode"], "remediation")
        self.assertEqual(project["base_model_name"], "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertIsNone(client.get(f"/api/projects/{pid}/guided-json").json()["guided_json"])
        put = client.put(f"/api/projects/{pid}/recipe/output-schema", json={"output_schema": {"properties": EXTRA, "required": []}})
        self.assertEqual(put.status_code, 200, put.text)
        stored = client.get(f"/api/projects/{pid}").json()["selected_recipe"]["output_schema"]
        self.assertEqual(stored["properties"]["severity"]["description"], "How serious the finding is.")

    def test_manifest_stamp_merges_extra_fields(self):
        from app.services.dataset_service import _recipe_output_schema

        pid = self.new_project()
        client.put(f"/api/projects/{pid}/label-taxonomy", json={"entries": TAXONOMY})
        client.put(f"/api/projects/{pid}/recipe/output-schema", json={"output_schema": {
            "properties": {**EXTRA, "patch": {"type": "string"}}, "required": ["severity"]}})
        schema = self.run_async(lambda db: _recipe_output_schema(db, pid))
        self.assertEqual(schema["scoring_mode"], "remediation")
        self.assertEqual(schema["required"], ["patch", "rationale", "citations", "severity"])
        self.assertEqual(list(schema["properties"]), ["severity"])  # base fields are fixed, never re-declared
        self.assertEqual(schema["properties"]["severity"]["enum"], ["low", "high", None])
        self.assertEqual(schema["label_parents"], PARENTS)
        self.assertEqual(schema["label_titles"]["injection.query"], "Query injection")
        self.assertEqual(set(schema["remediation"]), {"checks", "retrieve_passages", "top_k", "retrieval"})
        self.assertEqual(schema["remediation"]["top_k"], 3)
        self.assertIsNone(schema["remediation"]["checks"])

        # the handler turns it into the same extras the training prompt names
        handler = StructuredExtractionHandler()
        ctx = EvalContext(project_id=pid, experiment_id=0, eval_type="exact_match", task_profile="structured_extraction",
                          handler_id="x", prepared_dir=Path("."), dataset_name="t", manifest={"output_schema": schema})
        resolved = handler._resolve_schema([], ctx)
        self.assertEqual(resolved["fields"], ["patch", "rationale", "citations", "severity"])
        self.assertEqual(resolved["extra_fields"]["severity"]["description"], "How serious the finding is.")
        self.assertEqual(resolved["properties"]["severity"]["enum"], ["low", "high", None])

    def test_manifest_without_extras_has_empty_properties(self):
        from app.services.dataset_service import _recipe_output_schema

        pid = self.new_project()
        schema = self.run_async(lambda db: _recipe_output_schema(db, pid))
        self.assertEqual(schema["required"], ["patch", "rationale", "citations"])
        self.assertEqual(schema["properties"], {})

    def test_other_recipes_unchanged(self):
        from app.services.dataset_service import _recipe_output_schema

        pid = self.new_project("code-findings")
        schema = self.run_async(lambda db: _recipe_output_schema(db, pid))
        self.assertEqual(schema["scoring_mode"], "finding_set")
        self.assertNotIn("remediation", schema)

    def test_split_prepares_training_rows_with_the_shared_prompt(self):
        """Synthetic remediation + code-example rows through the real split: adapter config built from the
        recipe (extra fields, retrieval settings), passages retrieved once and carried, manifest stamped."""
        pid = self.new_project()
        client.put(f"/api/projects/{pid}/label-taxonomy", json={"entries": TAXONOMY})
        client.put(f"/api/projects/{pid}/recipe/output-schema", json={"output_schema": {"properties": EXTRA, "required": []}})
        client.put(f"/api/projects/{pid}/remediation-config", json={"retrieve_passages": True, "top_k": 2})
        rows = []
        for i in range(14):
            code = CODE + f"# variant {i}\n"
            fixed = FIXED + f"# variant {i}\n"
            rows.append(remediation_row(passages=i % 2 == 0, code=code, new_code=fixed, extra={"severity": "high"}))
        rows.append({"row_kind": "code_example", "task": "Read a row by id.", "language": "python", "passages": PASSAGES,
                     "code_example_json": {"code": "x = 1", "rationale": "r", "citations": ["P1"]}})
        rows.append({k: v for k, v in remediation_row().items() if k != "remediation_json"})  # rejected: no fix
        path = Path(settings.DATA_DIR) / "projects" / str(pid) / "synthetic_remediation.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        async def add(db):
            db.add(Dataset(project_id=pid, name="synthetic", dataset_type=DatasetType.SYNTHETIC, file_path=str(path), record_count=len(rows)))
            await db.commit()

        self.run_async(add)
        retrieved = [{"id": "P1", "text": "from the index", "source": "idx", "passage_id": "d:0"}]
        with patch.object(crp, "retrieve_passages_for_row", return_value=retrieved) as fetch:
            split = client.post(f"/api/projects/{pid}/dataset/split", json={"train_ratio": 0.6, "val_ratio": 0.1, "test_ratio": 0.3, "dedup_rows": False})
        self.assertEqual(split.status_code, 200, split.text)
        manifest = split.json()
        schema = manifest["output_schema"]
        self.assertEqual(schema["scoring_mode"], "remediation")
        self.assertEqual(schema["remediation"]["top_k"], 2)
        self.assertTrue(schema["remediation"]["retrieve_passages"])
        prepared = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        loaded = []
        for name in ("train", "val", "test"):
            f = prepared / f"{name}.jsonl"
            if f.exists():
                loaded += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        self.assertEqual(len(loaded), 15)  # 14 remediation + 1 code example; the row without a fix is rejected
        # 7 rows had no passages: retrieval ran for exactly those (top_k from the config)
        self.assertEqual(fetch.call_count, 7)
        self.assertTrue(all(c.args[3] == 2 for c in fetch.call_args_list))
        remediation = [r for r in loaded if r.get("row_kind") == "remediation"]
        self.assertEqual(len(remediation), 14)
        extras = crp.extra_fields_from_schema({"properties": EXTRA})
        for r in remediation:
            self.assertTrue(r["source_text"].startswith("Write a patch that fixes the finding"))
            self.assertIn('"severity"', r["source_text"])
            self.assertEqual(json.loads(r["target_text"])["severity"], "high")
            self.assertEqual(r["source_text"], crp.build_code_remediation_prompt_from_row(r, extras))
            if r["passages"] and r["passages"][0]["source"] == "idx":
                self.assertTrue(r["passages_retrieved"])
        example = [r for r in loaded if r.get("row_kind") == "code_example"][0]
        self.assertEqual(example["source_text"], crp.build_code_example_prompt_from_row(example))
        # the eval handler renders the same prompts from the prepared rows
        handler = StructuredExtractionHandler()
        ctx = EvalContext(project_id=pid, experiment_id=0, eval_type="exact_match", task_profile="structured_extraction",
                          handler_id="x", prepared_dir=prepared, dataset_name="test", manifest=manifest)
        with patch.object(crp, "retrieve_passages_for_row", side_effect=AssertionError("eval must not re-retrieve")):
            for r, built in zip(loaded, handler.build_prompts(loaded, ctx)):
                self.assertEqual(built.prompt, r["source_text"])
                self.assertEqual(built.reference, r["target_text"])

    # ── settings API ──

    def add_detector_project(self, *, weights: bool = True, name: str | None = None, recipe="code-findings") -> tuple[int, int | None]:
        pid = self.new_project(recipe)
        client.put(f"/api/projects/{pid}/label-taxonomy", json={"entries": TAXONOMY})
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "run"
        (out / "model").mkdir(parents=True)
        if weights:
            (out / "model" / "config.json").write_text("{}")
            (out / "model" / "model.safetensors").write_bytes(b"x")

        async def add(db):
            exp = Experiment(project_id=pid, name="detector run", status=ExperimentStatus.COMPLETED, base_model="toy",
                             config={}, output_dir=str(out), completed_at=datetime.now(timezone.utc))
            db.add(exp)
            await db.commit()
            return exp.id

        return pid, self.run_async(add)

    def test_get_defaults(self):
        pid = self.new_project()
        body = client.get(f"/api/projects/{pid}/remediation-config").json()
        self.assertEqual(body["config"], {"detector": None, "checks": None, "judge": None, "retrieve_passages": None, "top_k": None,
                                          "score_code_examples": None})
        self.assertFalse(body["effective"]["score_code_examples"])
        self.assertEqual(body["effective"]["top_k"], 3)
        self.assertTrue(body["effective"]["judge"])
        self.assertIsNone(body["effective"]["detector"])
        self.assertIsNone(body["detector_label"])
        self.assertIsInstance(body["available_detectors"], list)
        self.assertEqual(client.get("/api/projects/999999/remediation-config").status_code, 404)

    def test_put_validation(self):
        pid = self.new_project()
        url = f"/api/projects/{pid}/remediation-config"
        for body, fragment in (
            ({"top_k": 0}, "top_k"),
            ({"top_k": 11}, "top_k"),
            ({"checks": ["no_such_check"]}, "Unknown code check"),
            ({"detector": {"kind": "command", "check_id": "python_ast"}}, "not a registered detector"),
            ({"detector": {"kind": "project"}}, "needs project_id"),
            ({"detector": {"kind": "project", "project_id": 999999}}, "not found"),
            ({"detector": {"kind": "weird"}}, "must be"),
        ):
            resp = client.put(url, json=body)
            self.assertEqual(resp.status_code, 422, (body, resp.text))
            self.assertIn(fragment, json.dumps(resp.json()))
            self.assertEqual(resp.json()["error_code"], "INVALID_REMEDIATION_CONFIG")
        self.assertEqual(client.get(url).json()["config"]["top_k"], None)  # nothing was saved
        # a project that is not code-findings / has no weighted run cannot be the detector
        plain = self.new_project("qa-sft")
        self.assertEqual(client.put(url, json={"detector": {"kind": "project", "project_id": plain}}).status_code, 422)
        empty, _ = self.add_detector_project(weights=False)
        resp = client.put(url, json={"detector": {"kind": "project", "project_id": empty}})
        self.assertEqual(resp.status_code, 422)
        self.assertIn("no completed run with weights", resp.text)
        # settings only apply to code-remediation projects
        other = self.new_project("qa-sft")
        self.assertEqual(client.put(f"/api/projects/{other}/remediation-config", json={"top_k": 2}).status_code, 409)

    def test_put_stores_and_get_reports_detector_label(self):
        pid = self.new_project()
        det_pid, exp_id = self.add_detector_project()
        url = f"/api/projects/{pid}/remediation-config"
        resp = client.put(url, json={"detector": {"kind": "project", "project_id": det_pid}, "checks": ["python_ast"],
                                     "judge": False, "retrieve_passages": False, "top_k": 5})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["config"]["detector"], {"kind": "project", "project_id": det_pid, "experiment_id": None})
        self.assertEqual(body["config"]["checks"], ["python_ast"])
        self.assertEqual((body["effective"]["judge"], body["effective"]["retrieve_passages"], body["effective"]["top_k"]), (False, False, 5))
        det_name = client.get(f"/api/projects/{det_pid}").json()["name"]
        self.assertEqual(body["detector_label"], f"{det_name} · run {exp_id}")
        listed = [d for d in body["available_detectors"] if d["project_id"] == det_pid]
        self.assertEqual(len(listed), 1)
        self.assertEqual((listed[0]["experiment_id"], listed[0]["project_name"], listed[0]["run_name"]), (exp_id, det_name, "detector run"))
        self.assertTrue(listed[0]["completed_at"])
        # reading it back from the project, and clearing it
        self.assertEqual(client.get(url).json()["config"]["top_k"], 5)
        cleared = client.put(url, json={}).json()
        self.assertEqual(cleared["config"]["detector"], None)
        self.assertIsNone(cleared["detector_label"])

    def test_explicit_experiment_must_belong_to_the_detector_project(self):
        pid = self.new_project()
        det_pid, exp_id = self.add_detector_project()
        _other_pid, other_exp = self.add_detector_project()
        url = f"/api/projects/{pid}/remediation-config"
        self.assertEqual(client.put(url, json={"detector": {"kind": "project", "project_id": det_pid, "experiment_id": other_exp}}).status_code, 422)
        ok = client.put(url, json={"detector": {"kind": "project", "project_id": det_pid, "experiment_id": exp_id}})
        self.assertEqual(ok.status_code, 200, ok.text)

    def test_command_detector_from_the_registry(self):
        from app.services import code_check_service as ccs

        class Toy:
            id = "toy_detector"
            languages = frozenset({"python"})
            kind = "detector"
            binary = None

            def available(self):
                return True

            def run(self, code, *, language, context):
                return {"ok": True, "messages": [], "findings": [{"label": "secret.hardcoded"}] if "secret" in code else []}

        ccs.register_code_check(Toy())
        self.addCleanup(lambda: ccs._REGISTRY.pop("toy_detector", None))
        pid = self.new_project()
        resp = client.put(f"/api/projects/{pid}/remediation-config", json={"detector": {"kind": "command", "check_id": "toy_detector"}})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["detector_label"], "toy_detector")
        # a detector is not a parse check
        bad = client.put(f"/api/projects/{pid}/remediation-config", json={"checks": ["toy_detector"]})
        self.assertEqual(bad.status_code, 422)
        # not allow-listed: resolving it refuses rather than runs
        with patch.object(settings, "CODE_CHECK_ALLOWED", ""):
            with self.assertRaises(rd.DetectorUnavailable):
                asyncio.run(rd.resolve_detector(None, {"kind": "command", "check_id": "toy_detector"}))
        with patch.object(settings, "CODE_CHECK_ALLOWED", "toy_detector"):
            det = asyncio.run(rd.resolve_detector(None, {"kind": "command", "check_id": "toy_detector"}))
            self.assertEqual(det.detect([{"code": "a secret", "language": "python"}, {"code": "b", "language": "python"}]),
                             [["secret.hardcoded"], []])


class HeldoutIntegrationTests(unittest.TestCase):
    """run-heldout end to end with the model inference stubbed: extras copy, scoring, preview whitelist,
    row scores for pairing, the headline metric and the detector."""

    def setUp(self):
        self.api = ManifestAndConfigApiTests()
        self.api.addCleanup = self.addCleanup
        self.pid = self.api.new_project()
        client.put(f"/api/projects/{self.pid}/label-taxonomy", json={"entries": TAXONOMY})

    def seed(self, rows: list[dict], schema_override: dict | None = None) -> int:
        from app.services.dataset_service import _recipe_output_schema

        schema = ManifestAndConfigApiTests.run_async(lambda db: _recipe_output_schema(db, self.pid))
        if schema_override:
            schema.update(schema_override)
        prepared = Path(settings.DATA_DIR) / "projects" / str(self.pid) / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / "manifest.json").write_text(json.dumps({"task_profile": "structured_extraction", "adapter_id": "structured-extraction",
                                                            "output_schema": schema}))
        mapped = [prepared_row(r) for r in rows]
        (prepared / "test.jsonl").write_text("\n".join(json.dumps(r) for r in mapped) + "\n")

        async def add(db):
            db.add(Dataset(project_id=self.pid, name="test", dataset_type=DatasetType.TEST, file_path=str(prepared / "test.jsonl"),
                           record_count=len(mapped)))
            exp = Experiment(project_id=self.pid, name="rem run", status=ExperimentStatus.COMPLETED, base_model="toy", config={},
                             output_dir=str(prepared), completed_at=datetime.now(timezone.utc))
            db.add(exp)
            await db.commit()
            return exp.id

        return ManifestAndConfigApiTests.run_async(add)

    @staticmethod
    def fake_inference(answers):
        def run(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True, decoding="neutral"):
            preds = []
            for i, pair in enumerate(pairs):
                preds.append({"prompt": pair["prompt"], "formatted_prompt": "", "reference": pair["reference"],
                              "prediction": answers(i), "generated_tokens": 5, "latency_ms": 1.0})
            return preds, {"engine": "stub", "decoding": {"mode": decoding}}

        return run

    def run_heldout(self, exp_id, answers, *, stub_detector=None):
        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(patch("app.services.evaluation_service._run_local_inference", self.fake_inference(answers)))
            stack.enter_context(patch.dict("os.environ", {"EVAL_JUDGE_BACKEND": "none"}))
            if stub_detector is not None:
                stack.enter_context(patch.object(rd, "_infer_findings_text", stub_detector))
            resp = client.post(f"/api/projects/{self.pid}/evaluation/run-heldout", json={
                "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 20, "model_path": None})
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_scores_preview_and_row_scores(self):
        rows = [remediation_row(), remediation_row(), remediation_row(passages=False, citations=()),
                {"row_kind": "code_example", "task": "Read a row.", "language": "python", "passages": PASSAGES,
                 "code_example_json": {"code": "x = 1", "rationale": "r", "citations": ["P1"]}}]
        exp_id = self.seed(rows)
        answers = {0: json.dumps(GOOD), 1: json.dumps({**GOOD, "patch": patch_of(CODE, BROKEN)}), 2: "garbage",
                   3: '{"code": "x", "rationale": "r", "citations": []}'}
        result = self.run_heldout(exp_id, lambda i: answers[i])
        metrics = result["metrics"]
        self.assertEqual(metrics["scoring_mode"], "remediation")
        self.assertEqual((metrics["total"], metrics["skipped_non_remediation_rows"]), (3, 1))
        self.assertAlmostEqual(metrics["remediation_pass_rate"], round(1 / 3, 4))
        self.assertAlmostEqual(metrics["code_parses_rate"], round(1 / 3, 4))  # 3 checkable originals; garbage = no parse
        self.assertEqual(metrics["malformed_rate"], round(1 / 3, 4))
        self.assertEqual(metrics["evaluated_samples"], 4)
        self.assertNotIn("issue_resolved_rate", metrics)
        self.assertEqual(result["pass_rate"], metrics["remediation_pass_rate"])
        previews = result["details"]["predictions_preview"]
        first = previews[0]
        self.assertTrue(first["row_patch_applies"] and first["row_code_parses"] and first["row_remediation_pass"])
        self.assertEqual(first["row_finding"]["label"], "injection.query")
        self.assertEqual(first["row_passage_ids"], ["P1", "P2"])
        self.assertEqual(first["row_passage_headers"][1]["where"], "lib/db.py:3-4 (run)")
        self.assertEqual((first["path"], first["language"], first["start_line"]), ("app/db.py", "python", 10))
        self.assertIn("--- a/app/db.py", first["row_patch"])
        self.assertTrue(previews[3]["row_skipped"])
        scores = result["details"]["row_scores"]
        self.assertEqual(scores["remediation_pass"], [1.0, 0.0, 0.0, None])
        self.assertEqual(result["details"]["failed_count"], 2)

    def test_detector_runs_after_the_model_and_decides_issue_resolved(self):
        det_pid, _exp = ManifestAndConfigApiTests.add_detector_project(self.api)
        put = client.put(f"/api/projects/{self.pid}/remediation-config", json={"detector": {"kind": "project", "project_id": det_pid}})
        self.assertEqual(put.status_code, 200, put.text)
        exp_id = self.seed([remediation_row(), remediation_row(code=CODE.replace("uid", "key"), new_code=FIXED.replace("uid", "key"))])
        prompts = []

        def detector(model_ref, texts):
            prompts.extend(texts)
            out = []
            for text in texts:
                bad = '" + uid' in text or '" + key' in text
                out.append(json.dumps({"findings": [{"label": "injection.query", "start_line": 11, "end_line": 12}] if bad else []}))
            return out

        # the second answer leaves the finding in place: it fixes a different line
        wrong = patch_of(CODE.replace("uid", "key"), CODE.replace("uid", "key").replace("fetchone", "fetchall"))
        answers = {0: json.dumps(GOOD), 1: json.dumps({"patch": wrong, "rationale": "r", "citations": []})}
        result = self.run_heldout(exp_id, lambda i: answers[i], stub_detector=detector)
        metrics = result["metrics"]
        det_name = client.get(f"/api/projects/{det_pid}").json()["name"]
        self.assertEqual(metrics["detector_evaluated_count"], 2)
        self.assertEqual(metrics["issue_resolved_rate"], 0.5)
        self.assertEqual(metrics["detector"], f"{det_name} · run {_exp}")
        self.assertEqual(metrics["remediation_pass_rate"], 0.5)
        previews = result["details"]["predictions_preview"]
        self.assertTrue(previews[0]["row_issue_resolved"])
        self.assertFalse(previews[1]["row_issue_resolved"])
        self.assertEqual(previews[0]["row_detector"], f"{det_name} · run {_exp}")
        # the detector prompt is the shared findings prompt with the detector project's labels
        self.assertTrue(all(p.startswith("Inspect the code chunk below") for p in prompts))
        self.assertTrue(any("Allowed labels: injection, injection.query" in p for p in prompts))

    def test_a_failing_detector_never_fails_the_eval(self):
        det_pid, _exp = ManifestAndConfigApiTests.add_detector_project(self.api)
        client.put(f"/api/projects/{self.pid}/remediation-config", json={"detector": {"kind": "project", "project_id": det_pid}})
        exp_id = self.seed([remediation_row()])

        def boom(model_ref, texts):
            raise RuntimeError("CUDA out of memory")

        result = self.run_heldout(exp_id, lambda i: json.dumps(GOOD), stub_detector=boom)
        metrics = result["metrics"]
        self.assertEqual(metrics["remediation_pass_rate"], 1.0)
        self.assertNotIn("issue_resolved_rate", metrics)
        self.assertEqual(metrics["detector_error_count"], 1)
        self.assertIn("detector_error", metrics)


if __name__ == "__main__":
    unittest.main()
