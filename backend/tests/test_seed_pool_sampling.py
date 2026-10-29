"""Seed pool sampling + ``source_ids`` for the code / document seeded playbooks.

Unit: ``select_seed_chunks`` (below-cap identity, spread, determinism). API: ``source_ids``
validation (400 ``invalid_playbook_param``), filtering, the zero-chunk error, the licence gate
on top, and ``seed_pool`` reaching the direct response, the dry run and an async Job result.

pytest-only (the Jobs these tests start are awaited in tearDown).
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.main import app
from app.services import synth_playbook_service as sps
from app.services.synth_backends import TemplateBackend

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def chunk(doc, i, lang="python"):
    return {"text": f"def f_{doc}_{i}():\n    return {i}", "path": f"d{doc}/m{i}.x", "language": lang,
            "start_line": 1, "end_line": 2, "source_document_id": doc, "chunk_id": i, "source_doc": f"d{doc}.x"}


class SelectSeedChunksTests(unittest.TestCase):
    def setUp(self):
        # doc 1 -> source 10 (big, yaml), doc 2 and 3 -> source 20 (java, go)
        self.chunks = [chunk(1, i, "yaml") for i in range(50)] + [chunk(2, i, "java") for i in range(5)] + [
            chunk(3, i, "go") for i in range(5)]
        self.docs = {1: 10, 2: 20, 3: 20}

    def test_below_cap_is_unchanged_and_in_order(self):
        out, pool = sps.select_seed_chunks(self.chunks, self.docs, cap=100)
        self.assertEqual(json.dumps(out), json.dumps(self.chunks))
        self.assertFalse(pool["sampled"])
        self.assertEqual((pool["chunks_eligible"], pool["chunks_used"]), (60, 60))
        self.assertEqual(pool["by_source"], {"10": 50, "20": 10})
        self.assertNotIn("source_ids", pool)

    def test_over_cap_spreads_across_sources_and_languages(self):
        out, pool = sps.select_seed_chunks(self.chunks, self.docs, cap=12)
        self.assertEqual(len(out), 12)
        self.assertTrue(pool["sampled"])
        self.assertEqual(pool["by_language"], {"yaml": 4, "java": 4, "go": 4})
        self.assertEqual(set(pool["by_source"]), {"10", "20"})
        order = [self.chunks.index(c) for c in out]
        self.assertEqual(order, sorted(order))  # passage order kept

    def test_small_strata_are_exhausted_then_others_fill(self):
        _, pool = sps.select_seed_chunks(self.chunks, self.docs, cap=30)
        self.assertEqual(pool["by_language"], {"yaml": 20, "java": 5, "go": 5})

    def test_deterministic_and_seed_changes_the_sample(self):
        a, _ = sps.select_seed_chunks(self.chunks, self.docs, cap=12)
        b, _ = sps.select_seed_chunks(self.chunks, self.docs, cap=12)
        c, _ = sps.select_seed_chunks(self.chunks, self.docs, cap=12, seed=7)
        d, _ = sps.select_seed_chunks(self.chunks, self.docs, cap=12, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(c, d)
        self.assertNotEqual(a, c)

    def test_run_nonce_rotates_the_sample_unless_seed_given(self):
        # No params.seed: each run's seed_nonce picks a different sample (a resumed run, same nonce, agrees).
        self.assertEqual(sps._sample_seed({}, {"seed_nonce": 41}), 41)
        self.assertEqual(sps._sample_seed({"seed": 7}, {"seed_nonce": 41}), 7)
        self.assertEqual(sps._sample_seed({"seed": True}, {"seed_nonce": 41}), 41)
        runs = {tuple(c["chunk_id"] for c in sps.select_seed_chunks(self.chunks, self.docs, cap=12,
                                                                     seed=sps._sample_seed({}, {"seed_nonce": n}))[0])
                for n in (1, 2, 3)}
        self.assertEqual(len(runs), 3)

    def test_source_filter(self):
        out, pool = sps.select_seed_chunks(self.chunks, self.docs, cap=100, source_ids=[20])
        self.assertEqual(len(out), 10)
        self.assertEqual(pool["source_ids"], [20])
        self.assertEqual(pool["by_source"], {"20": 10})
        out, pool = sps.select_seed_chunks(self.chunks, self.docs, cap=100, source_ids=[99])
        self.assertEqual((out, pool["chunks_eligible"]), ([], 0))


class SeedFilterUnitTests(unittest.TestCase):
    def setUp(self):
        def mk(doc, path, text):
            return {"text": text, "path": path, "language": "python", "start_line": 1, "end_line": 2,
                    "source_document_id": doc, "chunk_id": path}

        self.chunks = [
            mk(1, "src/app/main.py", "widget builder"),
            mk(1, "src/app/util.py", "helper only"),
            mk(1, "docs/guide.md", "widget notes"),
            mk(1, "tests/test_main.py", "widget test"),
            mk(2, "setup.cfg", "nothing"),
            mk(2, "lib/core.py", "Widget core"),
        ]
        self.docs = {1: 10, 2: 20}

    def paths(self, **kw):
        out, pool = sps.select_seed_chunks(self.chunks, self.docs, filters=sps.seed_filters_from_params(kw), **{})
        return [c["path"] for c in out], pool

    def test_glob_semantics(self):
        self.assertTrue(sps.seed_glob_matches("src/a/b.py", "src/*.py"))
        self.assertTrue(sps.seed_glob_matches("src/a/b.py", "src/**/b.py"))
        self.assertTrue(sps.seed_glob_matches("b.py", "**/b.py"))
        self.assertFalse(sps.seed_glob_matches("B.py", "b.py"))
        self.assertFalse(sps.seed_glob_matches("src/b.py", "lib/*"))

    def test_no_filters_means_no_pool_extras(self):
        self.assertIsNone(sps.seed_filters_from_params({}))
        self.assertIsNone(sps.seed_filters_from_params({"seed_path_include": [], "seed_relevance_terms": [" "]}))
        _, pool = sps.select_seed_chunks(self.chunks, self.docs)
        self.assertNotIn("sources", pool)
        self.assertNotIn("filters", pool)

    def test_include_then_exclude(self):
        paths, pool = self.paths(seed_path_include=["src/**", "lib/*"], seed_path_exclude=["**/util.py"])
        self.assertEqual(paths, ["src/app/main.py", "lib/core.py"])
        self.assertEqual(pool["sources"]["10"], {"total": 4, "kept": 1, "dropped_glob": 3, "dropped_relevance": 0, "exempt": 0})
        self.assertEqual(pool["sources"]["20"]["dropped_glob"], 1)

    def test_relevance_chunk_scope_matches_path_or_text_case_insensitively(self):
        paths, pool = self.paths(seed_relevance_terms=["WIDGET", "setup"])
        self.assertEqual(paths, ["src/app/main.py", "docs/guide.md", "tests/test_main.py", "setup.cfg", "lib/core.py"])
        self.assertEqual(pool["sources"]["10"]["dropped_relevance"], 1)

    def test_relevance_file_scope_keeps_every_chunk_of_a_matching_file(self):
        self.chunks.append({"text": "no mention", "path": "src/app/main.py", "language": "python", "start_line": 9,
                            "end_line": 10, "source_document_id": 1, "chunk_id": "main-2"})
        chunk_only, _ = self.paths(seed_relevance_terms=["widget"])
        self.assertNotIn("main-2", [c["chunk_id"] for c in sps.select_seed_chunks(
            self.chunks, self.docs, filters=sps.seed_filters_from_params({"seed_relevance_terms": ["widget"]}))[0]])
        self.assertEqual(len(chunk_only), 4)
        out, pool = sps.select_seed_chunks(self.chunks, self.docs, filters=sps.seed_filters_from_params(
            {"seed_relevance_terms": ["widget"], "seed_relevance_scope": "file"}))
        self.assertIn("main-2", [c["chunk_id"] for c in out])
        self.assertEqual(pool["filters"]["scope"], "file")

    def test_exempt_source_skips_relevance_but_not_globs(self):
        paths, pool = self.paths(seed_relevance_terms=["widget"], relevance_exempt_source_ids=[20])
        self.assertIn("setup.cfg", paths)
        self.assertEqual(pool["sources"]["20"]["exempt"], 2)
        paths, _ = self.paths(seed_relevance_terms=["widget"], relevance_exempt_source_ids=[20],
                              seed_path_exclude=["*.cfg"])
        self.assertNotIn("setup.cfg", paths)

    def test_filters_apply_before_the_cap(self):
        many = [{"text": "keepme" if i % 10 == 0 else "x", "path": f"p/{i}.py", "language": "python",
                 "start_line": 1, "end_line": 2, "source_document_id": 1, "chunk_id": i} for i in range(100)]
        out, pool = sps.select_seed_chunks(many, {1: 10}, cap=10, filters=sps.seed_filters_from_params(
            {"seed_relevance_terms": ["keepme"]}))
        self.assertEqual(len(out), 10)
        self.assertTrue(all(c["text"] == "keepme" for c in out))
        self.assertEqual((pool["chunks_eligible"], pool["sampled"]), (10, False))
        self.assertEqual(pool["sources"]["10"]["kept"], 10)


class SeedPoolApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._prev_auth = settings.AUTH_ENABLED
        settings.AUTH_ENABLED = False
        cls._patches = [
            patch("app.services.synth_backends.BACKEND_REGISTRY", [TemplateBackend]),
            patch("app.services.synth_playbook_service.BACKEND_REGISTRY", [TemplateBackend]),
            patch.dict("os.environ", {"BREWSLM_TEMPLATE_SYNTH": "1"}),
        ]
        for p in cls._patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        settings.AUTH_ENABLED = cls._prev_auth

    def tearDown(self):
        wait_for_jobs_idle(client)

    def setUp(self):
        r = client.post("/api/projects", json={"name": f"seedpool-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "code-remediation"})
        self.base = f"/api/projects/{self.pid}/synthetic/run-playbook"
        self.pdir = Path(settings.DATA_DIR) / "projects" / str(self.pid)
        self.sources, self.docs = self._seed_sources()

    def _seed_sources(self):
        """Two code sources (python / go), one document each, plus an eval-only source."""
        from app.database import async_session_factory
        from app.models.dataset import Dataset, DatasetType, RawDocument
        from app.services import provenance_service as ps

        async def go():
            async with async_session_factory() as db:
                ds = Dataset(project_id=self.pid, name="docs", dataset_type=DatasetType.RAW, record_count=0)
                db.add(ds)
                await db.flush()
                out_src, out_docs = [], []
                for title, usage in (("repo-a", ps.USAGE_ANY), ("repo-b", ps.USAGE_ANY), ("repo-eval", ps.USAGE_EVAL)):
                    src = await ps.create_source(db, self.pid, kind=ps.KIND_UPLOAD, title=title, usage=usage)
                    doc = RawDocument(dataset_id=ds.id, filename=f"{title}.py", file_type="py",
                                      file_path=f"/tmp/{title}.py", data_source_id=src.id)
                    db.add(doc)
                    await db.flush()
                    out_src.append(src.id)
                    out_docs.append(doc.id)
                await db.commit()
                return out_src, out_docs

        sources, docs = asyncio.run(go())
        rows = [chunk(docs[0], i, "python") for i in range(6)] + [chunk(docs[1], i, "go") for i in range(6)] + [
            chunk(docs[2], i, "rust") for i in range(3)]
        cleaned = self.pdir / "cleaned" / "cleaned.jsonl"
        cleaned.parent.mkdir(parents=True, exist_ok=True)
        cleaned.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return sources, docs

    def _run(self, params=None, count=2, **kw):
        return client.post(self.base, json={"mode": "code_examples", "target_count": count, "params": params, **kw})

    def test_pool_reported_without_cap_hit_and_licence_gate_applies(self):
        resp = self._run({"per_chunk": 1})
        self.assertEqual(resp.status_code, 200, resp.text)
        pool = resp.json()["seed_pool"]
        self.assertFalse(pool["sampled"])
        self.assertEqual(pool["chunks_eligible"], 12)  # the eval-only source's 3 chunks are gated out
        self.assertEqual(pool["by_source"], {str(self.sources[0]): 6, str(self.sources[1]): 6})
        self.assertEqual(pool["by_language"], {"python": 6, "go": 6})

    def test_cap_samples_across_sources(self):
        with patch.object(sps, "MAX_DOCUMENT_CHUNKS", 4):
            resp = self._run({"per_chunk": 1})
        self.assertEqual(resp.status_code, 200, resp.text)
        pool = resp.json()["seed_pool"]
        self.assertTrue(pool["sampled"])
        self.assertEqual((pool["chunks_eligible"], pool["chunks_used"]), (12, 4))
        self.assertEqual(pool["by_source"], {str(self.sources[0]): 2, str(self.sources[1]): 2})

    def test_source_ids_filter(self):
        resp = self._run({"source_ids": [self.sources[1]]})
        self.assertEqual(resp.status_code, 200, resp.text)
        pool = resp.json()["seed_pool"]
        self.assertEqual(pool["source_ids"], [self.sources[1]])
        self.assertEqual(pool["by_source"], {str(self.sources[1]): 6})
        self.assertEqual(pool["by_language"], {"go": 6})

    def test_licence_gate_still_applies_with_source_ids(self):
        resp = self._run({"source_ids": [self.sources[2]]})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn(str(self.sources[2]), resp.json()["detail"])

    def test_zero_chunk_source_names_the_source(self):
        from app.database import async_session_factory
        from app.services import provenance_service as ps

        async def mk():
            async with async_session_factory() as db:
                src = await ps.create_source(db, self.pid, kind=ps.KIND_UPLOAD, title="empty")
                await db.commit()
                return src.id

        empty = asyncio.run(mk())
        resp = self._run({"source_ids": [empty]})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn(str(empty), resp.json()["detail"])

    def test_source_ids_validation(self):
        other = client.post("/api/projects", json={"name": f"other-{time.time_ns()}", "description": "x"}).json()["id"]
        from app.database import async_session_factory
        from app.services import provenance_service as ps

        async def mk():
            async with async_session_factory() as db:
                src = await ps.create_source(db, other, kind=ps.KIND_UPLOAD, title="foreign")
                await db.commit()
                return src.id

        foreign = asyncio.run(mk())
        for bad in ([], "x", [foreign], [self.sources[0], 987654], ["a"], [True], list(range(1, 102))):
            resp = self._run({"source_ids": bad})
            self.assertEqual(resp.status_code, 400, (bad, resp.text))
            body = resp.json()
            self.assertEqual(body["error_code"], "invalid_playbook_param", body)
            self.assertEqual(body["metadata"]["param"], "source_ids")
        dry = client.post(self.base + "/dry-run", json={"mode": "code_examples", "params": {"source_ids": []}})
        self.assertEqual(dry.status_code, 400)

    def test_dry_run_carries_seed_pool(self):
        resp = client.post(self.base + "/dry-run", json={"mode": "code_examples", "params": {"source_ids": [self.sources[0]]}})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["seed_pool"]["by_source"], {str(self.sources[0]): 6})

    def test_seed_pool_reaches_the_job_result(self):
        resp = client.post(self.base + "?async_job=true", json={
            "mode": "code_examples", "target_count": 2, "params": {"source_ids": [self.sources[0], self.sources[1]]}})
        self.assertEqual(resp.status_code, 202, resp.text)
        job_id = resp.json()["id"]
        wait_for_jobs_idle(client)
        job = client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(job["status"], "succeeded", job)
        pool = job["result"]["seed_pool"]
        self.assertEqual(pool["source_ids"], sorted(self.sources[:2]))
        self.assertEqual(pool["chunks_used"], 12)

    def test_seed_filters_through_the_api_and_the_pool_report(self):
        a, b = self.sources[0], self.sources[1]
        resp = self._run({"seed_path_include": ["d*/m[0-2].x"], "seed_path_exclude": ["d%d/*" % self.docs[1]]})
        self.assertEqual(resp.status_code, 200, resp.text)
        pool = resp.json()["seed_pool"]
        self.assertEqual(pool["chunks_used"], 3)
        self.assertEqual(pool["sources"][str(a)], {"total": 6, "kept": 3, "dropped_glob": 3, "dropped_relevance": 0, "exempt": 0})
        self.assertEqual(pool["sources"][str(b)]["kept"], 0)
        self.assertEqual(pool["filters"]["include"], ["d*/m[0-2].x"])
        dry = client.post(self.base + "/dry-run", json={"mode": "code_examples", "params": {"seed_relevance_terms": ["f_%d_1" % self.docs[0]]}})
        self.assertEqual(dry.status_code, 200, dry.text)
        self.assertEqual(dry.json()["seed_pool"]["chunks_used"], 1)
        self.assertEqual(dry.json()["seed_pool"]["sources"][str(a)]["dropped_relevance"], 5)

    def test_seed_filters_in_the_job_result(self):
        resp = client.post(self.base + "?async_job=true", json={
            "mode": "code_examples", "target_count": 2, "params": {"seed_path_exclude": ["**/m0.x"]}})
        self.assertEqual(resp.status_code, 202, resp.text)
        job_id = resp.json()["id"]
        wait_for_jobs_idle(client)
        job = client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(job["result"]["seed_pool"]["chunks_used"], 10)
        self.assertEqual(job["result"]["seed_pool"]["sources"][str(self.sources[0])]["dropped_glob"], 1)

    def test_filter_leaving_nothing_is_an_error_with_counts(self):
        resp = self._run({"seed_relevance_terms": ["no-such-term-anywhere"]})
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json()["detail"]
        self.assertIn("left 0 of 12", detail)
        self.assertIn("by seed_relevance_terms", detail)
        self.assertIn(f"source {self.sources[0]}: 6 chunks", detail)
        resp = client.post(self.base + "?async_job=true", json={
            "mode": "code_examples", "target_count": 2, "params": {"seed_path_include": ["nothing/**"]}})
        job_id = resp.json()["id"]
        wait_for_jobs_idle(client)
        job = client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(job["status"], "failed", job)
        self.assertIn("left 0 of 12", job["error"])

    def test_filters_with_sources_holding_no_chunks_keep_the_source_message(self):
        from app.database import async_session_factory
        from app.services import provenance_service as ps

        async def mk():
            async with async_session_factory() as db:
                src = await ps.create_source(db, self.pid, kind=ps.KIND_UPLOAD, title="empty2")
                await db.commit()
                return src.id

        empty = asyncio.run(mk())
        resp = self._run({"source_ids": [empty], "seed_path_include": ["**/*.x"]})
        self.assertEqual(resp.status_code, 400, resp.text)
        detail = resp.json()["detail"]
        self.assertIn("No code chunks in data source", detail)
        self.assertNotIn("seed filters left", detail)
        self.assertNotIn(". .", detail)
        again = self._run({"seed_relevance_terms": ["no-such-term-anywhere"]}).json()["detail"]
        self.assertNotIn(". .", again)

    def test_null_filter_params_are_unset(self):
        resp = self._run({"seed_path_include": None, "seed_path_exclude": None, "seed_relevance_terms": None,
                          "seed_relevance_scope": None, "relevance_exempt_source_ids": None})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertNotIn("sources", resp.json()["seed_pool"])
        self.assertEqual(resp.json()["seed_pool"]["chunks_used"], 12)

    def test_seed_filter_param_validation(self):
        cases = {
            "seed_path_include": ["x", [1], [""], ["a" * 201], ["a\x00b"], ["x"] * 51, "src/**"],
            "seed_path_exclude": ["x", [None], ["  "]],
            "seed_relevance_terms": ["x", [1], ["t" * 101], ["t"] * 101],
            "seed_relevance_scope": ["files", 3],
            "relevance_exempt_source_ids": ["x", [987654], [True]],
        }
        for param, bads in cases.items():
            for bad in bads:
                resp = self._run({param: bad})
                self.assertEqual(resp.status_code, 400, (param, bad, resp.text))
                body = resp.json()
                self.assertEqual(body["error_code"], "invalid_playbook_param", body)
                self.assertEqual(body["metadata"]["param"], param)
        ok = self._run({"seed_path_include": [], "seed_relevance_terms": [], "relevance_exempt_source_ids": [],
                        "seed_relevance_scope": "file"})
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertNotIn("sources", ok.json()["seed_pool"])


if __name__ == "__main__":
    unittest.main()
