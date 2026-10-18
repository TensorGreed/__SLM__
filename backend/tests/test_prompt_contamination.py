"""Seed-documents / span-synthesis prompts are refused when they carry a
value from a benchmark's gold rows (the 2026-10-17 Gretel incident, caught
at the prompt instead of after training)."""

from __future__ import annotations

import json
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import prompt_contamination_service as svc

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    from _job_wait import wait_for_jobs_idle

    wait_for_jobs_idle(client)
    client.__exit__(None, None, None)


SPAN_ROWS = [
    {"text": "Badge EMP-48213 for Grace Hopper", "entities": [
        {"type": "employee_id", "start": 6, "end": 15, "text": "EMP-48213"},
        {"type": "person_name", "start": 20, "end": 32, "text": "Grace Hopper"}]},
    {"text": "Plate KA-01-AB-1234 seen", "entities": [{"type": "plate", "start": 6, "end": 19, "text": "KA-01-AB-1234"}]},
]


class PureTests(unittest.TestCase):
    def test_index_merges_benchmarks_and_drops_short_values(self):
        index = svc.build_gold_index([
            ("gretel", "structured_extraction", SPAN_ROWS),
            ("conll", "structured_extraction", [{"text": "Grace Hopper and Al", "entities": [
                {"type": "person_name", "start": 0, "end": 12, "text": "Grace Hopper"}, {"type": "person_name", "start": 17, "end": 19, "text": "Al"}]}]),
            ("squad", "qa", [{"question": "q", "answer": "Paris is the capital"}]),
            ("yelp", "classification", [{"text": "t", "label": "positive"}]),
        ])
        self.assertEqual(index["grace hopper"]["benchmarks"], ["gretel", "conll"])
        self.assertEqual(index["paris is the capital"]["type"], "answer")
        self.assertNotIn("al", index)
        self.assertNotIn("positive", index)

    def test_whole_word_match_across_fields(self):
        index = svc.build_gold_index([("gretel", "structured_extraction", SPAN_ROWS)])
        fields = {"kinds": "employee ids like EMP-48213, plates such as ka-01-ab-1234 and names", "genres": ["ticket", "Grace Hopper's diary"],
                  "domain_hint": "EMP-482130 is not the same value"}
        offending = svc.find_benchmark_values(fields, index)
        self.assertEqual([(o["value"], o["fields"]) for o in offending],
                         [("Grace Hopper", ["genres"]), ("EMP-48213", ["kinds"]), ("KA-01-AB-1234", ["kinds"])])
        self.assertEqual(offending[1]["benchmarks"], ["gretel"])
        self.assertEqual(svc.find_benchmark_values({"kinds": "employee ids (EMP-nnnnn), plates"}, index), [])

    def test_gate_raises_with_the_list_unless_overridden(self):
        index = svc.build_gold_index([("gretel", "structured_extraction", SPAN_ROWS)])
        report = svc.check_prompt_fields({"kinds": "use EMP-48213"}, index, ["gretel"])
        self.assertFalse(report["clean"])
        exc = svc.PromptCarriesBenchmarkValues(report)
        self.assertIn("EMP-48213 (employee_id, gretel)", str(exc))
        self.assertIn("ignore_benchmark_values", str(exc))
        detail = svc.http_detail(exc)
        self.assertEqual(detail["error_code"], "PROMPT_CARRIES_BENCHMARK_VALUES")
        self.assertEqual(detail["metadata"]["offending"][0]["value"], "EMP-48213")
        self.assertTrue(svc.check_prompt_fields({"kinds": ""}, index, ["gretel"])["clean"])


def _project_with_benchmark() -> int:
    pid = client.post("/api/projects", json={"name": f"prompt-gate-{time.time_ns()}", "description": "x"}).json()["id"]
    root = settings.DATA_DIR / "projects" / str(pid)
    (root / "prepared").mkdir(parents=True, exist_ok=True)
    train = [{"text": "Hi Ada Lovelace", "entities": [{"type": "person_name", "start": 3, "end": 15}]}]
    (root / "prepared" / "train.jsonl").write_text(json.dumps(train[0]) + "\n")
    (root / "prepared" / "test.jsonl").write_text(json.dumps(train[0]) + "\n")
    (root / "imports").mkdir(exist_ok=True)
    path = root / "imports" / "pii.jsonl"
    path.write_text("".join(json.dumps({"text": r["text"], "spans": [
        {"start": e["start"], "end": e["end"], "label": e["type"]} for e in r["entities"]]}) + "\n" for r in SPAN_ROWS))
    created = client.post(f"/api/projects/{pid}/benchmarks", json={
        "locator": f"file:{path}", "name": "Gretel PII",
        "mapping": {"employee_id": "person_name", "person_name": "person_name", "plate": "person_name"}})
    assert created.status_code == 200, created.text
    return pid


class EndpointTests(unittest.TestCase):
    def test_seed_documents_is_refused_then_allowed_with_the_override(self):
        pid = _project_with_benchmark()
        body = {"kinds": "employee ids like EMP-48213 and plates", "count": 2, "genres": ["ticket"]}
        res = client.post(f"/api/projects/{pid}/flows/seed-documents", json=body)
        self.assertEqual(res.status_code, 400, res.text)
        payload = res.json()
        self.assertEqual(payload.get("error_code"), "PROMPT_CARRIES_BENCHMARK_VALUES")
        self.assertIn("EMP-48213", payload.get("message") or payload.get("detail"))
        self.assertEqual(payload["metadata"]["offending"][0]["fields"], ["kinds"])
        self.assertEqual(payload["metadata"]["override_field"], "ignore_benchmark_values")

        async def fake_run(db, project_id, **kwargs):
            from app.services import seed_documents_service as sds

            return sds.SeedDocumentsSummary(project_id=project_id, requested=2, written=2, failed=0, backend="fake",
                                            filename="seed-documents-x.txt", document_id=1, passages=3)

        with mock.patch("app.api.flows.run_seed_documents", fake_run):
            ok = client.post(f"/api/projects/{pid}/flows/seed-documents", json={**body, "ignore_benchmark_values": True})
            self.assertEqual(ok.status_code, 202, ok.text)
            self.assertEqual(ok.json()["params"]["prompt_check"], {"clean": False, "offending_total": 1, "overridden": True})
            clean = client.post(f"/api/projects/{pid}/flows/seed-documents", json={"kinds": "employee ids (EMP-nnnnn)", "count": 2, "genres": ["memo"]})
            self.assertIn(clean.status_code, (202, 409), "the gate passed (409 only while the first job is still in flight)")

    def test_span_synthesis_is_refused_on_types_and_on_the_conventions(self):
        pid = _project_with_benchmark()
        res = client.post(f"/api/projects/{pid}/synthetic/generate-spans-async", json={
            "target_rows": 10, "entity_types": ["employee_id", "plate"], "focus_types": ["KA-01-AB-1234"]})
        self.assertEqual(res.status_code, 400, res.text)
        self.assertEqual(res.json()["metadata"]["offending"][0]["fields"], ["focus_types"])

        put = client.put(f"/api/projects/{pid}/synthetic/type-definitions", json={
            "definitions": {"employee_id": "an employee badge number, for example EMP-48213"}})
        self.assertEqual(put.status_code, 200, put.text)
        dry = client.post(f"/api/projects/{pid}/synthetic/prompt-check", json={"fields": {"entity_types": ["employee_id"]}, "include_conventions": True}).json()
        self.assertFalse(dry["clean"])
        self.assertEqual(dry["offending"][0]["fields"], ["definition:employee_id"])
        self.assertEqual(dry["benchmarks"], ["gretel-pii"])
        self.assertTrue(client.post(f"/api/projects/{pid}/synthetic/prompt-check", json={"fields": {"kinds": "badge numbers"}}).json()["clean"])

        sync = client.post(f"/api/projects/{pid}/synthetic/generate-spans", json={"source_text": "x" * 20, "entity_types": ["employee_id"]})
        self.assertEqual(sync.status_code, 400)
        self.assertEqual(sync.json()["error_code"], "PROMPT_CARRIES_BENCHMARK_VALUES")

        with mock.patch("app.api.synthetic.start_span_generation_task") as start, \
                mock.patch("app.api.synthetic._spawn_legacy_synth_shadow_job", mock.AsyncMock(return_value=None)):
            start.return_value = mock.Mock(task_id="t1", status="pending", target_rows=10, batches_total=1)
            ok = client.post(f"/api/projects/{pid}/synthetic/generate-spans-async", json={
                "target_rows": 10, "entity_types": ["employee_id"], "ignore_benchmark_values": True})
            self.assertEqual(ok.status_code, 202, ok.text)


if __name__ == "__main__":
    unittest.main()
