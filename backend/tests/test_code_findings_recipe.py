"""The ``code-findings`` recipe: catalog entry, header sniffing, detection from
G3 mapper rows, the shared prompt (train == eval, chat template), adapter rows,
the manifest stamp, the guided-JSON hint and its export-manifest carry.
Neutral toy taxonomy only."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.database import async_session_factory
from app.main import app
from app.models.experiment import Experiment, ExperimentStatus
from app.services import code_findings_prompt as cfp
from app.services import label_taxonomy_service as lts
from app.services import recipe_service
from app.services.data_adapter_service import map_record_with_adapter
from app.services.dataset_import import resolve_mapper
from app.services.dataset_import.protocols import ImportContext, TransformedRow
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler
from app.services.task_shape_service import detect_task_shape, structured_recipe_for_rows
from scripts.train import _WRAPS_OWN_PROMPT_PREFIXES, _adapt_record_to_text, _build_data_adapter_contract, _rewrap_prompt_completion_row

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


TAXONOMY = [
    {"id": "injection", "title": "Injection"},
    {"id": "injection.query", "title": "Query injection", "parent": "injection", "aliases": ["sql concat"]},
    {"id": "secret.hardcoded", "title": "Hard-coded secret"},
    {"id": "logging.sensitive", "title": "Sensitive logging"},
]
LABELS = [e["id"] for e in TAXONOMY]
CODE = 'def f(cur, uid):\n    q = "SELECT * FROM t WHERE id = " + uid\n    cur.execute(q)\n    return cur.fetchone()\n'


def findings_row(*, clean: bool = False, start_line: int = 10) -> dict:
    items = [] if clean else [{"label": "injection.query", "start_line": start_line + 1, "end_line": start_line + 2,
                               "evidence": "q = ...", "attrs": {"severity": "high"}}]
    return {"row_kind": "findings", "code": CODE, "language": "python", "path": "app/db.py",
            "start_line": start_line, "end_line": start_line + 3, "findings_json": {"findings": items},
            "labels_raw": ["sql concat"] if items else [], "taxonomy_version": None}


def g3_rows() -> list[dict]:
    """Rows as the G3 mapper produces them (code_label_to_findings)."""
    ctx = ImportContext(project_id=0, project_task_profile=None, source_id="jsonl", mapper_id="code_label_to_findings",
                        locator="jsonl:/x", extra={"taxonomy": lts.TaxonomyIndex(lts.parse_taxonomy({"entries": TAXONOMY}))})
    out = list(resolve_mapper("code_label_to_findings").transform(
        [{"id": 1, "code": CODE, "label": "sql concat", "path": "a.py"}], {}, ctx=ctx))
    return [o.payload for o in out if isinstance(o, TransformedRow)]


class RecipeCatalogTests(unittest.TestCase):
    def test_recipe_fields(self):
        r = recipe_service.get_recipe("code-findings")
        self.assertEqual(r.name, "Code findings")
        self.assertEqual((r.task_profile, r.scoring_mode), ("structured_extraction", "finding_set"))
        self.assertEqual((r.default_input_column, r.default_output_column), ("code", "findings_json"))
        self.assertEqual(r.adapter_id, "structured-extraction")
        self.assertEqual(r.suggested_base_model, "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(r.alt_base_models, ["Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct"])
        self.assertEqual(r.gold_template.example_row["row_kind"], "findings")
        self.assertTrue(r.sample_eval_prompts and r.data_acquisition_hints)
        # the code-review recipe is untouched
        review = recipe_service.get_recipe("code-review")
        self.assertEqual((review.scoring_mode, review.default_output_column), ("field_match", "review"))

    def test_example_row_is_a_canonical_findings_row(self):
        row = recipe_service.get_recipe("code-findings").gold_template.example_row
        self.assertTrue(cfp.is_findings_row(row))
        self.assertIsNotNone(cfp.findings_payload(row))

    def test_task_maps_keep_structured_extraction_on_span_extraction(self):
        self.assertEqual(recipe_service.default_recipe_for_task_family("structured_extraction"), "span-extraction")
        self.assertEqual(recipe_service.default_recipe_for_task_profile("structured_extraction"), "span-extraction")

    def test_sniffing_prefers_code_findings_and_not_code_review(self):
        top = recipe_service.sniff_recipe_from_headers(["code", "findings_json", "language"])[0]
        self.assertEqual(top["recipe_id"], "code-findings")
        top = recipe_service.sniff_recipe_from_headers(["code", "findings"])[0]
        self.assertEqual(top["recipe_id"], "code-findings")
        for headers in (["diff", "review"], ["code", "review"], ["code", "comment"]):
            ids = [s["recipe_id"] for s in recipe_service.sniff_recipe_from_headers(headers)]
            self.assertEqual(ids[0], "code-review", headers)
            self.assertNotIn("code-findings", ids)

    def test_eval_maps_know_the_recipe(self):
        from app.services.coach_service import _PER_CLASS_RECIPE_IDS
        from app.services.eval_pack_scaffold_service import _RECIPE_SCAFFOLD
        from app.services.evaluation_gate_catalog import _RECOMMENDED_METRIC_IDS_PER_RECIPE, known_metric_ids
        from app.services.post_training_eval_service import eval_type_for_project

        self.assertIn("code-findings", _PER_CLASS_RECIPE_IDS)
        spec = _RECIPE_SCAFFOLD["code-findings"]
        self.assertEqual(spec["task_profile"], "structured_extraction")
        for metric in ("f1", "clean_chunk_fp_rate", "malformed_rate", "attr_accuracy_mean"):
            self.assertIn(metric, _RECOMMENDED_METRIC_IDS_PER_RECIPE["code-findings"])
            self.assertIn(metric, known_metric_ids())
        gates = {g[1]: g for g in spec["gates"]}
        self.assertTrue(gates["f1"][3])
        self.assertTrue(gates["clean_chunk_fp_rate"][3])
        project = SimpleNamespace(selected_recipe={"scoring_mode": "finding_set"}, runtime_config={}, training_mode=None)
        self.assertEqual(eval_type_for_project(project), "f1")


class DetectionTests(unittest.TestCase):
    def test_detects_g3_mapper_rows(self):
        rows = g3_rows()
        self.assertEqual(rows[0]["row_kind"], "findings")
        self.assertEqual(structured_recipe_for_rows(rows), "code-findings")

    def test_detects_payload_variants(self):
        self.assertEqual(structured_recipe_for_rows([{"code": CODE, "findings_json": {"findings": []}}]), "code-findings")
        self.assertEqual(structured_recipe_for_rows([{"code": CODE, "findings_json": '{"findings": []}'}]), "code-findings")
        self.assertEqual(structured_recipe_for_rows([{"text": "x", "structured_output": {"findings": [{"label": "a"}]}}]),
                         "code-findings")
        self.assertEqual(structured_recipe_for_rows([{"row_kind": "findings", "code": CODE}]), "code-findings")

    def test_other_structured_rows_keep_their_recipe(self):
        self.assertEqual(structured_recipe_for_rows([{"text": "a", "entities": [{"type": "x"}]}]), "span-extraction")
        self.assertEqual(structured_recipe_for_rows([{"text": "a", "json": {"name": "n"}}]), "json-extraction")
        self.assertEqual(structured_recipe_for_rows([{"text": "a", "json": {"findings": "none"}}]), "json-extraction")
        self.assertEqual(structured_recipe_for_rows([]), "span-extraction")

    def test_task_shape_detection_offers_code_findings(self):
        found = detect_task_shape(g3_rows() * 3)
        top = found["candidates"][0] if isinstance(found, dict) and found.get("candidates") else None
        self.assertIsNotNone(top, found)
        self.assertEqual(top["task_profile"], "structured_extraction")
        self.assertEqual(top.get("recipe_id"), "code-findings")

    def test_confirm_accepts_code_findings_and_detects_from_imported_rows(self):
        pid = client.post("/api/projects", json={"name": f"cf-shape-{time.time_ns()}", "description": "x"}).json()["id"]
        explicit = client.post(f"/api/projects/{pid}/task-shape/confirm",
                               json={"task_profile": "structured_extraction", "recipe_id": "code-findings"})
        self.assertEqual(explicit.status_code, 200, explicit.text)
        self.assertEqual(explicit.json()["recipe_id"], "code-findings")
        project = client.get(f"/api/projects/{pid}").json()
        self.assertEqual(project["selected_recipe"]["recipe_id"], "code-findings")
        self.assertEqual(project["selected_recipe"]["scoring_mode"], "finding_set")
        self.assertIsNotNone(client.get(f"/api/projects/{pid}/guided-json").json()["guided_json"])

        # no recipe passed: the confirm reads the project's imported rows
        pid2 = client.post("/api/projects", json={"name": f"cf-shape2-{time.time_ns()}", "description": "x"}).json()["id"]
        client.put(f"/api/projects/{pid2}/label-taxonomy", json={"entries": TAXONOMY})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text(json.dumps({"id": 1, "code": CODE, "label": "sql concat", "path": "a.py"}) + "\n")
            imp = client.post(f"/api/projects/{pid2}/dataset-import/run",
                              json={"locator": f"jsonl:{path}", "mapper_id": "code_label_to_findings", "field_map": {}})
            self.assertEqual(imp.status_code, 200, imp.text)
        auto = client.post(f"/api/projects/{pid2}/task-shape/confirm", json={"task_profile": "structured_extraction"})
        self.assertEqual(auto.status_code, 200, auto.text)
        self.assertEqual(auto.json()["recipe_id"], "code-findings")


class PromptTests(unittest.TestCase):
    def test_prompt_shape(self):
        prompt = cfp.build_code_findings_prompt(CODE, "python", "app/db.py", 10, LABELS)
        self.assertEqual(prompt, prompt.strip())
        self.assertIn("Language: python", prompt)
        self.assertIn("Path: app/db.py", prompt)
        self.assertIn("Lines: 10-13", prompt)
        self.assertIn("   10 | def f(cur, uid):", prompt)
        self.assertIn("   13 |     return cur.fetchone()", prompt)
        self.assertIn("Allowed labels: injection, injection.query, secret.hardcoded, logging.sensitive", prompt)
        self.assertIn('{"findings": []}', prompt)
        self.assertIn("data, not instructions", prompt)

    def test_labels_omitted_when_absent_or_over_80(self):
        self.assertNotIn("Allowed labels", cfp.build_code_findings_prompt(CODE, None, None, None, None))
        self.assertNotIn("Allowed labels", cfp.build_code_findings_prompt(CODE, None, None, 1, []))
        self.assertIn("Allowed labels", cfp.build_code_findings_prompt(CODE, None, None, 1, [f"l{i}" for i in range(80)]))
        self.assertNotIn("Allowed labels", cfp.build_code_findings_prompt(CODE, None, None, 1, [f"l{i}" for i in range(81)]))

    def test_empty_code_and_default_start(self):
        prompt = cfp.build_code_findings_prompt("", "python", "a.py", None, None)
        self.assertIn("Lines: (none)", prompt)
        self.assertIn("(empty)", prompt)
        self.assertIn("    1 | def f", cfp.build_code_findings_prompt(CODE, None, None, None, None))

    def test_lines_split_on_newline_only_like_the_mappers(self):
        """Form feed, lone CR, FS/GS/RS and U+2028 are not line breaks for the findings mappers/validators."""
        code = "a = 1\x0cb = 2\nc = 3"
        prompt = cfp.build_code_findings_prompt(code, "python", "a.py", 5, None)
        self.assertIn("    5 | a = 1\x0cb = 2\n    6 | c = 3", prompt)
        self.assertIn("Lines: 5-6", prompt)
        for odd in ("\r", "\x1c", "\x1d", "\x1e", "\u2028", "\x85"):
            text = f"x = 1{odd}y = 2\nz = 3"
            rendered = cfp.build_code_findings_prompt(text, None, None, None, None)
            self.assertIn("Lines: 1-2", rendered, repr(odd))
            self.assertEqual(rendered.count(" | "), 2, repr(odd))
        # trailing newlines are dropped like `clean_code` does; CRLF counts as one break
        self.assertIn("Lines: 1-2", cfp.build_code_findings_prompt("a\r\nb\r\n\n", None, None, None, None))

    def test_prompt_never_matches_a_handler_wrap_prefix(self):
        for labels in (None, LABELS):
            prompt = cfp.build_code_findings_prompt(CODE, "python", "a.py", 1, labels)
            self.assertFalse(prompt.lstrip().startswith(_WRAPS_OWN_PROMPT_PREFIXES))
            self.assertFalse(prompt.startswith(tuple(StructuredExtractionHandler().expected_prompt_prefixes())))

    def test_canonical_json_is_stable(self):
        a = cfp.canonical_findings_json({"findings": [{"attrs": {}, "label": "x", "end_line": 2, "start_line": 1, "evidence": "e"}]})
        self.assertEqual(a, '{"findings":[{"label":"x","start_line":1,"end_line":2,"evidence":"e","attrs":{}}]}')
        self.assertEqual(cfp.canonical_findings_json('{"findings": []}'), '{"findings":[]}')


class AdapterAndIdentityTests(unittest.TestCase):
    CONFIG = {"finding_labels": LABELS}

    def map_row(self, row, config=None):
        return map_record_with_adapter(row, adapter_id="structured-extraction", adapter_config=config or self.CONFIG)

    def test_flagged_row(self):
        row = findings_row()
        mapped = self.map_row(row)
        self.assertEqual(mapped["text"], CODE)  # indentation intact
        self.assertEqual(json.loads(mapped["target_text"]), row["findings_json"])
        self.assertEqual(mapped["target_text"], cfp.canonical_findings_json(row["findings_json"]))
        self.assertEqual(mapped["structured_output"], row["findings_json"])
        self.assertEqual(mapped["source_text"], cfp.build_code_findings_prompt_from_row(row, LABELS))

    def test_clean_row_is_a_valid_training_row(self):
        mapped = self.map_row(findings_row(clean=True))
        self.assertEqual(mapped["target_text"], '{"findings":[]}')

    def test_unusable_rows_rejected(self):
        self.assertIsNone(self.map_row({"row_kind": "findings", "code": "  ", "findings_json": {"findings": []}}))
        self.assertIsNone(self.map_row({"row_kind": "findings", "code": CODE}))

    def test_without_labels_the_prompt_names_none(self):
        mapped = map_record_with_adapter(findings_row(), adapter_id="structured-extraction", adapter_config={"finding_set": True})
        self.assertNotIn("Allowed labels", mapped["source_text"])

    def test_findings_prompt_needs_the_finding_set_scoring_mode(self):
        """A json-extraction project whose rows happen to carry a findings_json list keeps the field-match prompt."""
        payload = {"findings": [{"name": "alpha"}]}
        row = {"text": "alpha beta gamma", "structured_output": payload, "findings_json": payload}
        mapped = map_record_with_adapter(row, adapter_id="structured-extraction", adapter_config={"fields": ["findings"]})
        self.assertIsNotNone(mapped)
        self.assertTrue(mapped["source_text"].startswith("Extract the following fields as JSON"))
        self.assertNotIn("Inspect the code chunk", mapped["source_text"])

    def test_findings_mode_rejects_rows_without_findings_json_with_a_reason(self):
        from app.services.data_adapter_service import unmappable_reason

        row = {"row_kind": "findings", "code": CODE, "language": "python"}
        for config in ({"finding_set": True}, {"finding_labels": LABELS}):
            self.assertIsNone(map_record_with_adapter(row, adapter_id="structured-extraction", adapter_config=config))
            self.assertEqual(
                unmappable_reason(row, adapter_id="structured-extraction", adapter_config=config), "findings_json_missing")
        self.assertIn("findings_json_missing", __import__("app.services.dataset_service", fromlist=["x"]).INPUT_REJECT_REASONS)

    def test_other_structured_rows_unchanged(self):
        mapped = map_record_with_adapter({"text": "hi", "entities": [{"type": "a"}]}, adapter_id="structured-extraction",
                                         adapter_config={"fields": ["entities"]})
        self.assertTrue(mapped["source_text"].startswith("Extract the following fields as JSON"))
        self.assertTrue(mapped["target_text"].startswith(" "))

    def handler_and_ctx(self, schema=None):
        schema = schema or {"scoring_mode": "finding_set", "required": ["findings"], "labels": LABELS, "prompt_labels": LABELS}
        ctx = EvalContext(project_id=0, experiment_id=0, eval_type="f1", task_profile="structured_extraction",
                          handler_id="structured_extraction", prepared_dir=Path("."), dataset_name="t",
                          manifest={"output_schema": schema})
        return StructuredExtractionHandler(), ctx

    def test_train_and_eval_prompts_are_byte_identical(self):
        for row in (findings_row(), findings_row(clean=True), findings_row(start_line=1)):
            prepared = {**row, **self.map_row(row)}  # what prepared/*.jsonl carries
            handler, ctx = self.handler_and_ctx()
            built = handler.build_prompts([prepared], ctx)[0]
            self.assertEqual(built.prompt, prepared["source_text"])
            self.assertEqual(built.reference, prepared["target_text"])
            self.assertEqual(built.extras["path"], "app/db.py")
            self.assertEqual(built.extras["language"], "python")
            self.assertEqual(built.extras["start_line"], row["start_line"])
        # without labels in the manifest both sides name none
        bare = {**findings_row(), **map_record_with_adapter(findings_row(), adapter_id="structured-extraction", adapter_config={"finding_set": True})}
        handler, ctx = self.handler_and_ctx({"scoring_mode": "finding_set", "required": ["findings"], "labels": ["a"]})
        self.assertEqual(handler.build_prompts([bare], ctx)[0].prompt, bare["source_text"])

    def test_span_and_field_modes_keep_their_prompts(self):
        handler, ctx = self.handler_and_ctx({"scoring_mode": "span_set", "properties": {"entities": {}}, "required": ["entities"]})
        built = handler.build_prompts([{"text": "hello", "answer": '{"entities": []}'}], ctx)[0]
        self.assertTrue(built.prompt.startswith("Extract the following fields as JSON"))
        self.assertTrue(handler.wraps_own_prompt(ctx))
        self.assertTrue(handler.wraps_own_prompt())

    def test_wraps_own_prompt_helper(self):
        from app.services.eval_task_handler_service import ClassificationHandler, handler_wraps_own_prompt

        find = {"output_schema": {"scoring_mode": "finding_set"}}
        span = {"output_schema": {"scoring_mode": "span_set"}}
        self.assertFalse(handler_wraps_own_prompt(StructuredExtractionHandler(), find))
        self.assertTrue(handler_wraps_own_prompt(StructuredExtractionHandler(), span))
        self.assertTrue(handler_wraps_own_prompt(StructuredExtractionHandler(), None))
        self.assertTrue(handler_wraps_own_prompt(ClassificationHandler(), find))

    def test_trainer_rewraps_the_row_with_the_chat_template(self):
        class Tok:
            chat_template = "x"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"

        row = findings_row()
        prepared = {**row, **self.map_row(row)}
        contract = _build_data_adapter_contract("causal_lm", "llama3")
        adapted = _adapt_record_to_text(prepared, contract, "llama3")
        out = _rewrap_prompt_completion_row({**prepared, **adapted}, Tok())
        self.assertNotEqual(out["text"], prepared["source_text"] + prepared["target_text"])  # re-wrapped, not passthrough
        self.assertEqual(out["source_text"], f"<user>{prepared['source_text']}</user><assistant>")
        self.assertTrue(out["text"].endswith(prepared["target_text"]))
        # eval sends exactly the same user message through the same template
        from app.services.evaluation_service import _apply_chat_template_if_present

        formatted, applied = _apply_chat_template_if_present(
            SimpleNamespace(chat_template="x", apply_chat_template=Tok().apply_chat_template),
            self.handler_and_ctx()[0].build_prompts([prepared], self.handler_and_ctx()[1])[0].prompt)
        self.assertTrue(applied)
        self.assertEqual(formatted, out["source_text"])


class ManifestStampTests(unittest.TestCase):
    def test_stamp_with_taxonomy_and_override(self):
        from app.services.dataset_service import _finding_schema_stamp

        tax = lts.parse_taxonomy({"entries": TAXONOMY})
        project = SimpleNamespace(runtime_config={"label_taxonomy": tax, "finding_match": {"iou_threshold": 0.7, "hierarchy": True}})
        stamp = _finding_schema_stamp(project, None)
        self.assertEqual(stamp["labels"], LABELS)
        self.assertEqual(stamp["label_parents"], {"injection.query": "injection"})
        self.assertEqual(stamp["prompt_labels"], LABELS)
        self.assertEqual(stamp["taxonomy_version"], lts.taxonomy_version(tax))
        self.assertEqual(stamp["finding_match"], {"iou_threshold": 0.7, "attrs": [], "hierarchy": True, "parent_credit": 0.5})

    def test_stamp_without_taxonomy_uses_labels_seen(self):
        from app.services.dataset_service import _finding_schema_stamp

        splits = {"train": [findings_row(), findings_row(clean=True)], "test": [{"findings_json": {"findings": [
            {"label": "secret.hardcoded", "start_line": 1, "end_line": 1}]}}]}
        stamp = _finding_schema_stamp(SimpleNamespace(runtime_config={}), splits)
        self.assertEqual(stamp["labels"], ["injection.query", "secret.hardcoded"])
        self.assertEqual(stamp["label_parents"], {})
        self.assertNotIn("prompt_labels", stamp)
        self.assertEqual(stamp["finding_match"]["iou_threshold"], 0.5)

    def test_split_manifest_carries_the_stamp(self):
        pid = client.post("/api/projects", json={"name": f"cf-stamp-{time.time_ns()}", "description": "x"}).json()["id"]
        client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "code-findings"})

        async def override():  # no API sets finding_match: it is a runtime_config override
            from app.models.project import Project

            async with async_session_factory() as db:
                project = await db.get(Project, pid)
                project.runtime_config = {**(project.runtime_config or {}), "finding_match": {"iou_threshold": 0.6}}
                await db.commit()

        asyncio.run(override())
        put = client.put(f"/api/projects/{pid}/label-taxonomy", json={"entries": TAXONOMY})
        self.assertEqual(put.status_code, 200, put.text)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text("\n".join(json.dumps({"id": i, "code": CODE + f"# {i}\n", "label": "sql concat", "path": f"a{i}.py"})
                                      for i in range(12)) + "\n")
            client.post(f"/api/projects/{pid}/dataset-import/run",
                        json={"locator": f"jsonl:{path}", "mapper_id": "code_label_to_findings", "field_map": {}})
        split = client.post(f"/api/projects/{pid}/dataset/split", json={"dedup_rows": False})
        self.assertEqual(split.status_code, 200, split.text)
        schema = split.json()["output_schema"]
        self.assertEqual(schema["scoring_mode"], "finding_set")
        self.assertEqual(schema["required"], ["findings"])
        self.assertEqual(schema["labels"], LABELS)
        self.assertEqual(schema["finding_match"]["iou_threshold"], 0.6)
        self.assertEqual(schema["label_parents"], {"injection.query": "injection"})


class GuidedJsonTests(unittest.TestCase):
    def new_project(self) -> int:
        return client.post("/api/projects", json={"name": f"cf-gj-{time.time_ns()}", "description": "x"}).json()["id"]

    def hint(self, pid):
        return client.get(f"/api/projects/{pid}/guided-json").json()["guided_json"]

    def enum(self, hint):
        return hint["schema"]["properties"]["findings"]["items"]["properties"]["label"].get("enum")

    def test_null_without_the_recipe(self):
        pid = self.new_project()
        self.assertIsNone(self.hint(pid))
        client.put(f"/api/projects/{pid}/label-taxonomy", json={"entries": TAXONOMY})
        self.assertIsNone(self.hint(pid), "a taxonomy alone does not create the hint")
        self.assertEqual(client.get("/api/projects/999999/guided-json").status_code, 404)

    def test_lifecycle(self):
        pid = self.new_project()
        client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "code-findings"})
        hint = self.hint(pid)
        self.assertEqual(hint["source"], "recipe:code-findings")
        self.assertIsNone(hint["taxonomy_version"])
        self.assertIsNone(self.enum(hint))
        self.assertEqual(hint["schema"]["required"], ["findings"])
        self.assertTrue(hint["updated_at"])

        v1 = client.put(f"/api/projects/{pid}/label-taxonomy", json={"entries": TAXONOMY}).json()["version"]
        hint = self.hint(pid)
        self.assertEqual((hint["taxonomy_version"], self.enum(hint)), (v1, LABELS))

        v2 = client.put(f"/api/projects/{pid}/label-taxonomy", json={"entries": TAXONOMY[:2]}).json()["version"]
        self.assertEqual((self.hint(pid)["taxonomy_version"], self.enum(self.hint(pid))), (v2, LABELS[:2]))

        self.assertEqual(client.post(f"/api/projects/{pid}/label-taxonomy/restore", json={"version": v1}).status_code, 200)
        self.assertEqual((self.hint(pid)["taxonomy_version"], self.enum(self.hint(pid))), (v1, LABELS))

        client.delete(f"/api/projects/{pid}/label-taxonomy")
        hint = self.hint(pid)
        self.assertIsNone(hint["taxonomy_version"])
        self.assertIsNone(self.enum(hint))

        # an invalid save changes nothing
        bad = client.put(f"/api/projects/{pid}/label-taxonomy", json={"entries": [{"id": "a", "parent": "missing"}]})
        self.assertEqual(bad.status_code, 422)

    def test_switching_recipe_drops_the_hint(self):
        pid = self.new_project()
        client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "code-findings"})
        self.assertIsNotNone(self.hint(pid))
        client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": "qa-sft"})
        self.assertIsNone(self.hint(pid))

    def test_schema_validates_findings(self):
        schema = cfp.findings_json_schema(LABELS)
        item = schema["properties"]["findings"]["items"]
        self.assertEqual(item["required"], ["label", "start_line", "end_line"])
        self.assertEqual(item["properties"]["label"]["enum"], LABELS)

    def test_export_manifest_carries_the_hint(self):
        import tempfile as _tf

        from app.models.export import ExportFormat, ExportStatus
        from app.services.export_service import create_export, run_export

        def export_manifest(pid, model_root):
            async def go():
                async with async_session_factory() as db:
                    exp = Experiment(project_id=pid, name="run", status=ExperimentStatus.COMPLETED, base_model="org/base-1b",
                                     output_dir=str(model_root), config={"task_type": "causal_lm"})
                    db.add(exp)
                    await db.commit()
                    export = await create_export(db, pid, exp.id, ExportFormat.HUGGINGFACE)
                    export = await run_export(db, pid, export.id, run_smoke_tests=False)
                    await db.commit()
                    return export.status, export.manifest

            return asyncio.run(go())

        with _tf.TemporaryDirectory() as tmp:
            model = Path(tmp) / "exp" / "model"
            model.mkdir(parents=True)
            (model / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"]}))
            (model / "model.safetensors").write_bytes(b"w")

            with_hint = self.new_project()
            client.put(f"/api/projects/{with_hint}/recipe", json={"recipe_id": "code-findings"})
            client.put(f"/api/projects/{with_hint}/label-taxonomy", json={"entries": TAXONOMY})
            status, manifest = export_manifest(with_hint, model.parent)
            self.assertEqual(status, ExportStatus.COMPLETED)
            self.assertEqual(manifest["guided_json"], self.hint(with_hint))
            self.assertEqual(self.enum(manifest["guided_json"]), LABELS)

            without = self.new_project()
            status, manifest = export_manifest(without, model.parent)
            self.assertEqual(status, ExportStatus.COMPLETED)
            self.assertNotIn("guided_json", manifest)


if __name__ == "__main__":
    unittest.main()
