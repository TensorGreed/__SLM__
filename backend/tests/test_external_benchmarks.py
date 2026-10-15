"""External benchmarks (Epic G-B1): import a labelled set someone else made,
map its labels onto the project's, keep it eval-only, score runs on it."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.services import external_benchmark_service as svc

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


class PureTests(unittest.TestCase):
    def test_bio_tokens_become_char_spans(self):
        text, spans = svc.bio_to_spans(
            ["Peter", "Blackburn", "met", "EU", "staff", "in", "Bonn"],
            ["B-PER", "I-PER", "O", "B-ORG", "O", "O", "B-LOC"],
        )
        self.assertEqual(text, "Peter Blackburn met EU staff in Bonn")
        self.assertEqual([(s["type"], s["text"]) for s in spans], [("PER", "Peter Blackburn"), ("ORG", "EU"), ("LOC", "Bonn")])
        _, spans = svc.bio_to_spans(["a", "b"], ["I-PER", "I-LOC"])
        self.assertEqual([s["type"] for s in spans], ["PER", "LOC"], "an I- with no matching B- starts a span")

    def test_detects_the_three_formats(self):
        names = {"ner_tags": ["O", "B-PER", "I-PER"]}
        bio = [{"tokens": ["Ada", "Lovelace", "wrote"], "ner_tags": [1, 2, 0]}]
        self.assertEqual(svc.detect_shape(bio, names)["format"], "bio_tokens")
        spans = [{"source_text": "Mail ada@x.io now", "privacy_mask": [{"start": 5, "end": 13, "label": "EMAIL", "value": "ada@x.io"}]}]
        shape = svc.detect_shape(spans)
        self.assertEqual((shape["format"], shape["fields"]["text"], shape["fields"]["span_label"]), ("char_spans", "source_text", "label"))
        cls = [{"text": f"review {i}", "label": i % 5} for i in range(20)]
        self.assertEqual(svc.detect_shape(cls)["task_profile"], "classification")
        with self.assertRaises(svc.BenchmarkError):
            svc.detect_shape([{"prompt": f"p{i}", "chosen": "a good reply", "rejected": "a bad reply"} for i in range(5)])

    def test_offsetless_entity_strings_are_located(self):
        rows = [{"text": "Mail jo@x.io or call 555-0101; jo@x.io again.",
                 "entities": "[{'entity': 'jo@x.io', 'types': ['email']}, {'entity': '555-0101', 'types': ['phone_number', 'x']}]"}]
        shape = svc.detect_shape(rows)
        self.assertEqual(shape["format"], "entity_strings")
        out = svc.convert_rows(rows, shape, {})
        self.assertEqual([(e["type"], e["start"]) for e in out[0]["entities"]], [("email", 5), ("phone_number", 21), ("email", 31)])
        self.assertEqual(svc.locate_entities("Ann Lee and Lee", [("Lee", "LAST"), ("Ann Lee", "NAME")]),
                         [{"type": "NAME", "start": 0, "end": 7, "text": "Ann Lee"}, {"type": "LAST", "start": 12, "end": 15, "text": "Lee"}])

    def test_many_to_one_ignore_and_join(self):
        rows = [{"text": "t1", "label": "1 star"}, {"text": "t2", "label": "2 star"}, {"text": "t3", "label": "3 stars"}, {"text": "t5", "label": "5 stars"}]
        mapping = {"1 star": "negative", "2 star": "negative", "3 stars": "ignore", "5 stars": "positive"}
        out, summary = svc.apply_mapping(rows, mapping)
        self.assertEqual([r["label"] for r in out], ["negative", "negative", "positive"])
        self.assertEqual(summary["rows_dropped_by_label"], {"3 stars": 1})
        span_rows = [{"text": "Ada Lovelace in Bonn", "entities": [
            {"type": "FIRST", "start": 0, "end": 3}, {"type": "LAST", "start": 4, "end": 12}, {"type": "LOC", "start": 16, "end": 20}]}]
        m = {"FIRST": "person_name", "LAST": "person_name", "LOC": "ignore"}
        joined, s = svc.apply_mapping(span_rows, m, join_adjacent=True)
        self.assertEqual([(e["type"], e["text"]) for e in joined[0]["entities"]], [("person_name", "Ada Lovelace")])
        self.assertEqual(s["entities_ignored"], {"LOC": 1})
        apart, _ = svc.apply_mapping(span_rows, m)
        self.assertEqual(len(apart[0]["entities"]), 2)

    def test_mapping_validation_and_suggestions(self):
        self.assertEqual(svc.suggest_mapping(["Positive", "1 star"], ["positive", "negative"]), {"Positive": "positive", "1 star": "ignore"})
        with self.assertRaises(svc.BenchmarkError):
            svc.validate_mapping({"a": "nope"}, ["a"], ["positive"])
        with self.assertRaises(svc.BenchmarkError):
            svc.validate_mapping({"a": "ignore"}, ["a"], ["positive"])

    def test_span_score_counts_only_mapped_types(self):
        per_class = {"person_name": {"tp": 8, "fp": 2, "fn": 2}, "email": {"tp": 0, "fp": 9, "fn": 0}}
        block = svc.score_block("structured_extraction", {"per_class": per_class}, ["person_name"])
        self.assertAlmostEqual(block["f1"], 0.8)
        self.assertEqual([t["type"] for t in block["per_type"]], ["person_name"], "email predictions are not penalised")


def _project(name: str, train_rows, test_rows):
    pid = client.post("/api/projects", json={"name": f"{name}-{time.time_ns()}", "description": "x"}).json()["id"]
    prepared = settings.DATA_DIR / "projects" / str(pid) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train_rows))
    (prepared / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test_rows))
    return pid


def _upload(pid: int, name: str, rows) -> str:
    imports = settings.DATA_DIR / "projects" / str(pid) / "imports"
    imports.mkdir(parents=True, exist_ok=True)
    path = imports / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f"file:{path}"


class EndToEndTests(unittest.TestCase):
    def test_classification_benchmark_import_leakage_and_scoring(self):
        train = [{"text": f"great product number {i}", "label": "positive"} for i in range(10)] + \
                [{"text": f"awful product number {i}", "label": "negative"} for i in range(10)]
        pid = _project("bench-cls", train, train[:4])
        bench_rows = [{"text": f"loved it, review {i}", "label": 4} for i in range(6)] + \
                     [{"text": f"hated it, review {i}", "label": 0} for i in range(6)] + \
                     [{"text": f"it was ok, review {i}", "label": 2} for i in range(4)] + \
                     [{"text": "great product number 3", "label": 4}]  # a training row → leakage
        locator = _upload(pid, "yelp.jsonl", bench_rows)

        self.assertEqual(client.post(f"/api/projects/{pid}/benchmarks/preview", json={"locator": "file:/etc/hosts.csv"}).status_code, 400)
        prev = client.post(f"/api/projects/{pid}/benchmarks/preview", json={"locator": locator}).json()
        self.assertEqual(prev["task_profile"], "classification")
        self.assertEqual(prev["project"], {"task_profile": "classification", "labels": ["negative", "positive"]})
        self.assertEqual(set(prev["source_labels"]), {"0", "2", "4"})

        bad = client.post(f"/api/projects/{pid}/benchmarks", json={"locator": locator, "name": "Yelp", "mapping": {"4": "great"}})
        self.assertEqual(bad.status_code, 400)
        created = client.post(f"/api/projects/{pid}/benchmarks", json={
            "locator": locator, "name": "Yelp stars", "mapping": {"0": "negative", "2": "ignore", "4": "positive"},
            "licence": "research only", "local_only": True,
        }).json()
        self.assertEqual(created["rows"], 12, "4 ignored, 1 leaked")
        self.assertEqual(created["leakage"]["exact"], 1)
        self.assertEqual(created["licence"], "research only")
        self.assertTrue(created["local_only"])

        async def _types():
            from sqlalchemy import select

            from app.models.dataset import Dataset

            async with async_session_factory() as db:
                rows = (await db.execute(select(Dataset).where(Dataset.project_id == pid))).scalars().all()
                return {(d.dataset_type.value, d.is_locked) for d in rows}

        self.assertIn(("benchmark", True), asyncio.run(_types()))
        from app.services.dataset_service import DEFAULT_TRAINING_SOURCE_TYPES
        from app.models.dataset import DatasetType

        self.assertNotIn(DatasetType.BENCHMARK, DEFAULT_TRAINING_SOURCE_TYPES)

        # Scoring: fake predictions, real handler — no model loaded.
        async def _exp():
            from app.models.experiment import Experiment, ExperimentStatus

            async with async_session_factory() as db:
                exp = Experiment(project_id=pid, name="clf", status=ExperimentStatus.COMPLETED, base_model="x",
                                 config={"task_type": "classification"}, output_dir=str(settings.DATA_DIR))
                db.add(exp)
                await db.commit()
                return exp.id

        eid = asyncio.run(_exp())

        async def fake_generate(db, *, pairs, **kw):
            preds = [{"prompt": p.get("prompt", ""), "reference": p.get("reference", ""),
                      "prediction": "positive" if "loved" in str(p.get("prompt", "")) else "positive"} for p in pairs]
            return preds, {}, {}, {}

        async def _score():
            async with async_session_factory() as db:
                return await svc.evaluate_benchmark(db, pid, created["id"], eid)

        with mock.patch("app.services.evaluation_service._generate_heldout_predictions", fake_generate):
            result = asyncio.run(_score())
        self.assertEqual(result["rows"], 12)
        self.assertAlmostEqual(result["accuracy"], 0.5, msg="always 'positive' → half right")
        summary = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": eid}).json()
        bench = summary["benchmarks"][0]
        self.assertEqual(bench["result"]["experiment_id"], eid)
        self.assertAlmostEqual(bench["result"]["score"], 0.5)
        self.assertEqual(client.delete(f"/api/projects/{pid}/benchmarks/{created['id']}").json(), {"deleted": created["id"]})
        self.assertEqual(client.get(f"/api/projects/{pid}/benchmarks").json(), {"benchmarks": []})

    def test_span_benchmark_from_bio_tokens_and_shape_mismatch(self):
        train = [{"text": "Hi Ada Lovelace", "entities": [{"type": "person_name", "start": 3, "end": 15}]}]
        pid = _project("bench-span", train, train)
        locator = _upload(pid, "conll.jsonl", [
            {"tokens": ["Grace", "Hopper", "visited", "Paris"], "ner_tags": ["B-PER", "I-PER", "O", "B-LOC"]},
            {"tokens": ["Nothing", "here"], "ner_tags": ["O", "O"]},
        ])
        prev = client.post(f"/api/projects/{pid}/benchmarks/preview", json={"locator": locator}).json()
        self.assertEqual((prev["task_profile"], prev["format"]), ("structured_extraction", "bio_tokens"))
        self.assertEqual(prev["source_labels"], {"PER": 1, "LOC": 1})
        created = client.post(f"/api/projects/{pid}/benchmarks", json={
            "locator": locator, "name": "CoNLL people", "mapping": {"PER": "person_name", "LOC": "ignore"}}).json()
        self.assertEqual(created["scored_labels"], ["person_name"])
        rows_file = settings.DATA_DIR / "projects" / str(pid) / "benchmarks" / created["slug"] / "rows.jsonl"
        first = json.loads(rows_file.read_text().splitlines()[0])
        self.assertEqual(first["entities"], [{"type": "person_name", "start": 0, "end": 12, "text": "Grace Hopper"}])

        cls_loc = _upload(pid, "cls.jsonl", [{"text": f"t{i}", "label": i % 2} for i in range(10)])
        r = client.post(f"/api/projects/{pid}/benchmarks", json={"locator": cls_loc, "name": "x", "mapping": {"0": "person_name"}})
        self.assertEqual(r.status_code, 400)
        self.assertIn("project's data is structured_extraction", r.json().get("message") or r.json().get("detail"))


class EvaluateJobTests(unittest.TestCase):
    def test_evaluate_runs_as_a_benchmark_eval_job(self):
        from _job_wait import wait_for_jobs_idle

        train = [{"text": f"good {i}", "label": "positive"} for i in range(3)] + [{"text": f"bad {i}", "label": "negative"} for i in range(3)]
        pid = _project("bench-job", train, train[:2])
        locator = _upload(pid, "b.jsonl", [{"text": f"fine {i}", "label": i % 2} for i in range(6)])
        created = client.post(f"/api/projects/{pid}/benchmarks", json={
            "locator": locator, "name": "B", "mapping": {"0": "negative", "1": "positive"}}).json()
        self.assertEqual(client.post(f"/api/projects/{pid}/benchmarks/{created['id']}/evaluate", json={}).status_code, 400,
                         "no trained run yet")

        async def fake_eval(db, project_id, benchmark_id, experiment_id, **kw):
            await asyncio.sleep(0.2)
            return {"benchmark_id": benchmark_id, "experiment_id": experiment_id, "score": 0.75}

        with mock.patch.object(svc, "evaluate_benchmark", fake_eval):
            r = client.post(f"/api/projects/{pid}/benchmarks/{created['id']}/evaluate", json={"experiment_id": 999})
            self.assertEqual(r.status_code, 202)
            job_id = r.json()["id"]
            wait_for_jobs_idle(client)
        job = client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual((job["kind"], job["status"]), ("benchmark_eval", "succeeded"))
        self.assertEqual(job["result"]["score"], 0.75)


class GenerativeShapeTests(unittest.TestCase):
    """G-B2: Q&A and summarization benchmarks — no labels to map."""

    def test_detects_and_converts_qa_squad_and_summarization(self):
        qa = [{"question": f"What does clause {i} say?", "answer": f"Clause {i} says the tenant pays rent monthly"} for i in range(6)]
        shape = svc.detect_shape(qa)
        self.assertEqual((shape["task_profile"], shape["format"]), ("qa", "qa_pair"))
        self.assertEqual(svc.convert_rows(qa[:1], shape, {}), [{"question": "What does clause 0 say?", "answer": "Clause 0 says the tenant pays rent monthly"}])

        squad = [{"id": str(i), "title": "t", "context": f"The capital of land {i} is City{i}.", "question": f"Capital of land {i}?",
                  "answers": {"text": [f"City{i}"], "answer_start": [24]}} for i in range(8)]
        shape = svc.detect_shape(squad)
        self.assertEqual((shape["task_profile"], shape["format"]), ("qa", "qa_context"))
        self.assertEqual(svc.convert_rows(squad[:1], shape, {}),
                         [{"question": "Capital of land 0?", "answer": "City0", "context": "The capital of land 0 is City0."}])
        self.assertEqual(svc.source_label_counts(svc.convert_rows(squad, shape, {})), {}, "nothing to map")

        news = [{"id": i, "article": f"Long article number {i} about the harbour. " * 10, "highlights": f"Harbour story {i} in short form"} for i in range(6)]
        shape = svc.detect_shape(news)
        self.assertEqual((shape["task_profile"], shape["fields"]), ("summarization", {"document": "article", "summary": "highlights"}))
        self.assertEqual(svc.convert_rows(news[:1], shape, {})[0]["answer"], "Harbour story 0 in short form")

    def test_label_like_answers_are_not_generative(self):
        rows = [{"input": f"text {i}", "output": ["pos", "neg"][i % 2]} for i in range(20)]
        self.assertIsNone(svc.detect_generative_shape(rows))

    def test_context_is_folded_unless_the_project_answers_from_context(self):
        rows = [{"question": "Who?", "answer": "Ada", "context": "Ada wrote it."}]
        folded, did = svc.fit_rows_to_project(rows, "qa")
        self.assertTrue(did)
        self.assertEqual(folded, [{"question": "Context: Ada wrote it.\nQuestion: Who?", "answer": "Ada"}])
        kept, did = svc.fit_rows_to_project(rows, "rag_qa")
        self.assertFalse(did)
        self.assertEqual(kept, rows)
        self.assertTrue(svc.shapes_compatible("summarization", "instruction_sft"))
        self.assertFalse(svc.shapes_compatible("qa", "classification"))


class GroupMappingTests(unittest.TestCase):
    """G-B2: one benchmark label → a group of project labels."""

    def test_group_mapping_validation_and_rows(self):
        labels = ["date_of_birth", "visit_date", "person_name"]
        clean = svc.validate_mapping({"DATE": ["visit_date", "date_of_birth"], "PER": "person_name", "X": "ignore"}, ["DATE", "PER", "X"], labels)
        self.assertEqual(clean["DATE"], ["date_of_birth", "visit_date"])
        self.assertEqual(svc.validate_mapping({"DATE": ["visit_date"]}, ["DATE"], labels), {"DATE": "visit_date"})
        with self.assertRaisesRegex(svc.BenchmarkError, "more than one group"):
            svc.validate_mapping({"DATE": ["visit_date", "date_of_birth"], "DOB": "date_of_birth"}, ["DATE", "DOB"], labels)
        with self.assertRaises(svc.BenchmarkError):
            svc.validate_mapping({"DATE": ["visit_date", "nope"]}, ["DATE"], labels)
        # Two benchmark labels onto the same group is fine.
        svc.validate_mapping({"DATE": ["visit_date", "date_of_birth"], "TIME": ["date_of_birth", "visit_date"]}, ["DATE", "TIME"], labels)

        rows = [{"text": "Seen 2024-01-02 by Ann", "entities": [{"type": "DATE", "start": 5, "end": 15}, {"type": "PER", "start": 19, "end": 22}]}]
        out, _ = svc.apply_mapping(rows, clean)
        self.assertEqual([e["type"] for e in out[0]["entities"]], ["date_of_birth|visit_date", "person_name"])
        self.assertEqual(svc.group_members(clean), {"date_of_birth": "date_of_birth|visit_date", "visit_date": "date_of_birth|visit_date"})

    def test_predictions_collapse_to_the_group(self):
        from app.services.eval_task_handler_service import ClassificationHandler, StructuredExtractionHandler

        members = {"positive": "positive|very_positive", "very_positive": "positive|very_positive"}
        preds = [{"prediction": "very_positive", "reference": "positive|very_positive"},
                 {"prediction": "negative.", "reference": "negative"}]
        candidates = svc.collapse_predictions("classification", preds, members, ClassificationHandler(), ["negative", "positive", "very_positive"])
        self.assertEqual(candidates, ["negative", "positive|very_positive"])
        self.assertEqual([p["prediction"] for p in preds], ["positive|very_positive", "negative."])
        self.assertEqual(preds[0]["prediction_before_grouping"], "very_positive")

        span_members = {"visit_date": "date_of_birth|visit_date", "date_of_birth": "date_of_birth|visit_date"}
        spans = [{"prediction": '```json\n{"entities": [{"type": "visit_date", "start": 5, "end": 15}, {"type": "person_name", "start": 19, "end": 22}]}\n```'}]
        svc.collapse_predictions("structured_extraction", spans, span_members, StructuredExtractionHandler())
        self.assertEqual([e["type"] for e in json.loads(spans[0]["prediction"])["entities"]], ["date_of_birth|visit_date", "person_name"])
        block = svc.score_block("structured_extraction", {"per_class": {"date_of_birth|visit_date": {"tp": 3, "fp": 1, "fn": 0}}},
                                ["date_of_birth|visit_date"])
        self.assertEqual(block["per_type"][0]["type"], "date_of_birth|visit_date")


def _exp(pid: int, name: str, task_type: str) -> int:
    async def _go():
        from app.models.experiment import Experiment, ExperimentStatus

        async with async_session_factory() as db:
            exp = Experiment(project_id=pid, name=name, status=ExperimentStatus.COMPLETED, base_model="x",
                             config={"task_type": task_type}, output_dir=str(settings.DATA_DIR))
            db.add(exp)
            await db.commit()
            return exp.id

    return asyncio.run(_go())


def _score(pid: int, bid: int, eid: int) -> dict:
    async def _go():
        async with async_session_factory() as db:
            return await svc.evaluate_benchmark(db, pid, bid, eid)

    return asyncio.run(_go())


class GenerativeEndToEndTests(unittest.TestCase):
    def test_qa_benchmark_scores_f1_judge_skip_for_local_only_and_history(self):
        from app.services.answer_judge_service import ResolvedJudge

        train = [{"question": f"How long is notice period {i}?", "answer": f"The notice period {i} lasts thirty days in writing"} for i in range(6)]
        pid = _project("bench-qa", train, train[:2])
        (settings.DATA_DIR / "projects" / str(pid) / "prepared" / "manifest.json").write_text(json.dumps({"task_profile": "qa"}))
        bench = [{"context": f"Section {i}: a landlord must repair heating within {i + 2} days of notice.",
                  "question": f"How fast must heating be repaired under section {i}?",
                  "answer": f"Within {i + 2} days of the tenant giving notice to the landlord"} for i in range(6)]
        locator = _upload(pid, "leases.jsonl", bench)
        prev = client.post(f"/api/projects/{pid}/benchmarks/preview", json={"locator": locator}).json()
        self.assertEqual((prev["task_profile"], prev["format"], prev["labels_to_map"]), ("qa", "qa_context", False))
        self.assertEqual(prev["source_labels"], {})
        self.assertTrue(prev["shape_matches_project"])
        news = _upload(pid, "news.jsonl", [{"article": f"Long harbour article {i}. " * 12, "summary": f"Harbour piece {i} summarised in brief"} for i in range(5)])
        self.assertEqual(client.post(f"/api/projects/{pid}/benchmarks/preview", json={"locator": news}).json()["task_profile"], "summarization")

        created = client.post(f"/api/projects/{pid}/benchmarks", json={"locator": locator, "name": "Lease QA", "local_only": True}).json()
        self.assertEqual((created["task_profile"], created["rows"], created["scored_labels"], created["context_folded"]), ("qa", 6, [], True))
        rows_file = settings.DATA_DIR / "projects" / str(pid) / "benchmarks" / created["slug"] / "rows.jsonl"
        first = json.loads(rows_file.read_text().splitlines()[0])
        self.assertTrue(first["question"].startswith("Context: Section 0"))

        async def fake_generate(db, *, pairs, **kw):
            preds = [{"prompt": p["prompt"], "reference": p["reference"],
                      "prediction": p["reference"] if i % 2 == 0 else "I am not sure about that at all"} for i, p in enumerate(pairs)]
            return preds, {}, {}, {}

        async def fake_judge(question, reference, prediction):
            return (1.0, "correct", "same", 3) if prediction == reference else (0.0, "wrong", "differs", 3)

        cloud = ResolvedJudge(label=f"anthropic:judge-{pid}", judge=fake_judge, local=False)
        local = ResolvedJudge(label=f"ollama:judge-{pid}", judge=fake_judge)
        e1, e2 = _exp(pid, "qa-run-1", "causal_lm"), _exp(pid, "qa-run-2", "causal_lm")
        with mock.patch("app.services.evaluation_service._generate_heldout_predictions", fake_generate), \
                mock.patch("app.services.answer_judge_service.resolve_answer_judge", mock.AsyncMock(return_value=cloud)):
            r1 = _score(pid, created["id"], e1)
        self.assertEqual(r1["judge"]["skipped"], "local_only_cloud_judge")
        self.assertIn("local-only", r1["judge"]["message"])
        self.assertEqual(r1["headline"], "f1")
        self.assertAlmostEqual(r1["exact_match"], 0.5)
        with mock.patch("app.services.evaluation_service._generate_heldout_predictions", fake_generate), \
                mock.patch("app.services.answer_judge_service.resolve_answer_judge", mock.AsyncMock(return_value=local)):
            r2 = _score(pid, created["id"], e2)
        self.assertEqual((r2["headline"], r2["score"]), ("judge_correct", 0.5))
        self.assertEqual(r2["judge"]["judge"], f"ollama:judge-{pid}")
        self.assertEqual(r2["failures"][0]["judge_verdict"], "wrong")

        hist = client.get(f"/api/projects/{pid}/benchmarks/{created['id']}/history").json()
        self.assertEqual([(h["experiment_id"], h["experiment_name"], h["headline"]) for h in hist["history"]],
                         [(e1, "qa-run-1", "f1"), (e2, "qa-run-2", "judge_correct")])
        self.assertAlmostEqual(hist["history"][1]["score"], 0.5)
        self.assertEqual(client.get(f"/api/projects/{pid}/benchmarks/999999/history").status_code, 404)

    def test_group_mapping_classification_scoring(self):
        train = [{"text": f"fine item {i}", "label": "positive"} for i in range(4)] + \
                [{"text": f"superb item {i}", "label": "very_positive"} for i in range(4)] + \
                [{"text": f"broken item {i}", "label": "negative"} for i in range(4)]
        pid = _project("bench-group", train, train[:3])
        locator = _upload(pid, "g.jsonl", [{"text": f"lovely gadget {i}", "label": "good"} for i in range(4)] +
                          [{"text": f"dreadful gadget {i}", "label": "bad"} for i in range(4)])
        bad = client.post(f"/api/projects/{pid}/benchmarks", json={"locator": locator, "name": "G", "mapping": {
            "good": ["positive", "very_positive"], "bad": ["negative", "positive"]}})
        self.assertEqual(bad.status_code, 400)
        created = client.post(f"/api/projects/{pid}/benchmarks", json={"locator": locator, "name": "G", "mapping": {
            "good": ["very_positive", "positive"], "bad": "negative"}}).json()
        self.assertEqual(created["scored_labels"], ["negative", "positive|very_positive"])
        self.assertEqual(created["groups"], {"positive|very_positive": ["positive", "very_positive"]})
        self.assertEqual(created["mapping"]["good"], ["positive", "very_positive"])
        eid = _exp(pid, "clf", "classification")

        async def fake_generate(db, *, pairs, **kw):
            self.assertIn("negative, positive, very_positive", pairs[0]["prompt"], "the prompt lists project labels, not groups")
            preds = [{"prompt": p["prompt"], "reference": p["reference"],
                      "prediction": ["very_positive", "positive"][i % 2] if "lovely" in p["prompt"] else "negative"}
                     for i, p in enumerate(pairs)]
            return preds, {}, {}, {}

        with mock.patch("app.services.evaluation_service._generate_heldout_predictions", fake_generate):
            result = _score(pid, created["id"], eid)
        self.assertAlmostEqual(result["accuracy"], 1.0, msg="either member counts as the group")
        self.assertIn("positive|very_positive", result["per_class"])


class HeadEvalSummaryTests(unittest.TestCase):
    def test_encoder_classifier_summary_is_its_own_score(self):
        pid = _project("head-sum", [{"text": "a", "label": "x"}], [{"text": "b", "label": "x"}])

        async def _seed():
            from app.models.experiment import EvalResult, Experiment, ExperimentStatus

            async with async_session_factory() as db:
                exp = Experiment(project_id=pid, name="enc", status=ExperimentStatus.COMPLETED, base_model="enc-base",
                                 config={"task_type": "classification"}, output_dir=str(settings.DATA_DIR))
                db.add(exp)
                await db.flush()
                db.add(EvalResult(experiment_id=exp.id, dataset_name="test", eval_type="exact_match",
                                  metrics={"accuracy": 0.73, "macro_f1": 0.72, "evaluated_samples": 100}, pass_rate=0.73, details={}))
                await db.commit()
                return exp.id

        eid = asyncio.run(_seed())
        from app.services import post_training_eval_service as pte

        with mock.patch.object(pte, "base_is_causal_lm", return_value=False):
            summary = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": eid}).json()
        self.assertEqual((summary["kind"], summary["verdict"]), ("head_eval", "evaluated"))
        self.assertEqual(summary["head"]["accuracy"], 0.73)
        self.assertEqual(summary["benchmarks"], [])
