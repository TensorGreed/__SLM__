"""T2 (2026-11-20): a plain Q&A project's prepared manifest was stamped ``output_schema.scoring_mode:
field_match`` (the qa-sft recipe declares that mode) though its eval scores free text. The stamp is now
left off when it is bare (no declared fields) and the rows are not structured extraction; a real
field-match project (json-extraction with a declared schema, or structured-extraction rows) keeps it
unchanged, and every reader of the stamp behaves the same with or without the bare one."""

from __future__ import annotations

import json
import time
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.services import checked_decoding_service as cds
from app.services import multi_option_scoring as mos
from app.services import null_scoring
from app.services.dataset_service import _manifest_output_schema, manifest_scoring_mode
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler
from app.services.evaluation_service import _manifest_scoring_mode

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    from _job_wait import wait_for_jobs_idle

    wait_for_jobs_idle(client, timeout_s=60)
    client.__exit__(None, None, None)


BARE = {"scoring_mode": "field_match", "required": []}
QA = [{"question": f"What is item {i}?", "answer": f"Item {i} is a widget."} for i in range(12)]
RECORDS = [{"text": f"Report {i}: unit {i} failed at stage {i % 3}.", "structured_output": {"unit": str(i), "stage": i % 3}}
           for i in range(12)]
RECORD_SCHEMA = {"properties": {"unit": {"type": "string"}, "stage": {"type": "integer"}}}


def jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


def new_project(recipe_id: str | None) -> int:
    pid = client.post("/api/projects", json={"name": f"t2-{time.time_ns()}", "description": "x"}).json()["id"]
    if recipe_id:
        resp = client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": recipe_id})
        assert resp.status_code == 200, resp.text
    return pid


def predefined(pid: int, rows: list[dict]) -> dict:
    files = {"train": ("train.jsonl", jsonl(rows[:8]), "application/octet-stream"),
             "val": ("val.jsonl", jsonl(rows[8:10]), "application/octet-stream"),
             "test": ("test.jsonl", jsonl(rows[10:]), "application/octet-stream")}
    resp = client.post(f"/api/projects/{pid}/dataset/predefined-split", files=files)
    assert resp.status_code == 200, resp.text
    return resp.json()


class HelperTests(unittest.TestCase):
    def test_bare_field_match_on_non_extraction_rows_is_dropped(self):
        for profile in ("qa", "classification", "summarization", "generic_sft", None, ""):
            self.assertIsNone(_manifest_output_schema(dict(BARE), profile), profile)

    def test_real_field_match_and_other_modes_are_kept_unchanged(self):
        declared = {"scoring_mode": "field_match", "required": ["unit"], "properties": RECORD_SCHEMA["properties"]}
        self.assertEqual(_manifest_output_schema(declared, "qa"), declared)
        self.assertEqual(_manifest_output_schema(dict(BARE), "structured_extraction"), BARE)
        for schema in ({"scoring_mode": "span_set", "required": ["entities"]},
                       {"scoring_mode": "field_match", "required": [], "subtask": "x"},
                       {"scoring_mode": "field_match", "required": ["a"]}):
            self.assertEqual(_manifest_output_schema(dict(schema), "qa"), schema)
        self.assertIsNone(_manifest_output_schema(None, "qa"))

    def test_every_reader_treats_the_bare_stamp_like_no_stamp(self):
        with_stamp, without = {"task_profile": "qa", "output_schema": dict(BARE)}, {"task_profile": "qa", "output_schema": None}
        ctx = lambda m: type("Ctx", (), {"manifest": m})()  # noqa: E731
        # The handler-independent mode branches: none of remediation / structured_plan / multi_option.
        self.assertNotIn(_manifest_scoring_mode(ctx(with_stamp)), ("remediation", "structured_plan", "multi_option", "span_set"))
        self.assertIsNone(_manifest_scoring_mode(ctx(without)))
        self.assertNotIn(manifest_scoring_mode(with_stamp), ("remediation",))
        for profile in ("qa", "classification", None):
            self.assertEqual(null_scoring.reads(profile, "field_match"), null_scoring.reads(profile, None))
        self.assertEqual(mos.uses_multi_option({}, with_stamp), mos.uses_multi_option({}, without))
        self.assertIsNone(cds.output_json_schema(with_stamp))
        self.assertIsNone(cds.output_json_schema(without))

    def test_structured_extraction_scoring_is_identical_without_the_bare_stamp(self):
        preds = [{"prompt": "p", "reference": '{"a": "1", "b": 2}', "prediction": '{"a": "1", "b": 3}'},
                 {"prompt": "q", "reference": '{"a": "x"}', "prediction": '{"a": "x"}'}]

        def score(schema):
            ctx = EvalContext(project_id=0, experiment_id=0, eval_type="exact_match", task_profile="structured_extraction",
                              handler_id="structured_extraction", prepared_dir=None, dataset_name="test",
                              manifest={"output_schema": schema} if schema is not None else {})
            return StructuredExtractionHandler().score([dict(p) for p in preds], ctx)

        self.assertEqual(score(dict(BARE)), score(None))


class ManifestStampTests(unittest.TestCase):
    def test_plain_qa_import_and_random_split_carry_no_field_match_stamp(self):
        pid = new_project("qa-sft")
        manifest = predefined(pid, QA)
        self.assertIsNone(manifest["output_schema"])
        stored = client.get(f"/api/projects/{pid}/prepared-manifest").json()
        self.assertIsNone(stored.get("output_schema"))

    def test_json_extraction_with_a_declared_schema_keeps_field_match(self):
        pid = new_project("json-extraction")
        resp = client.put(f"/api/projects/{pid}/recipe/output-schema", json={"output_schema": RECORD_SCHEMA})
        self.assertEqual(resp.status_code, 200, resp.text)
        manifest = predefined(pid, RECORDS)
        self.assertEqual(manifest["output_schema"]["scoring_mode"], "field_match")
        self.assertEqual(manifest["output_schema"]["properties"], RECORD_SCHEMA["properties"])

    def test_no_recipe_is_unchanged(self):
        pid = new_project(None)
        self.assertIsNone(predefined(pid, QA)["output_schema"])


if __name__ == "__main__":
    unittest.main()
