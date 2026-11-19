"""SQL execution match (``sql_execution_scoring``): read-only enforcement, limits, result comparison,
special rows, path containment, multi-database scoring, the fingerprint, the settings API and the
held-out eval wiring (model inference stubbed; no real model).

pytest-only."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import EvalResult, Experiment, ExperimentStatus
from app.models.project import Project
from app.services import sql_execution_scoring as ses

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


def make_db(path: Path, people: list[tuple] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, score REAL, team TEXT)")
    conn.executemany(
        "INSERT INTO people VALUES (?, ?, ?, ?, ?)",
        people if people is not None else [
            (1, "ada", 36, 1.5, "a"), (2, "bob", 41, 2.25, "b"), (3, "cy", 29, None, "a"), (4, "dee", 41, 3.0, None),
        ],
    )
    conn.commit()
    conn.close()
    return path


CFG = ses.normalize_config({"enabled": True, "db_paths": ["x"]})


class ExecutionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sql-exec-"))
        self.db = str(make_db(self.tmp / "one.db"))

    def test_select_runs(self):
        r = ses.execute_query(self.db, "```sql\nSELECT name FROM people WHERE id = 1;\n```")
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["rows"], [("ada",)])

    def test_writes_attach_pragma_are_blocked(self):
        for sql in (
            "INSERT INTO people (id, name) VALUES (9, 'x')",
            "UPDATE people SET age = 1",
            "DELETE FROM people",
            "DROP TABLE people",
            "CREATE TABLE t (a)",
            f"ATTACH DATABASE '{self.tmp / 'other.db'}' AS o",
            "PRAGMA table_info(people)",
            "PRAGMA writable_schema = 1",
            "BEGIN",
            "SELECT 1; DELETE FROM people",
        ):
            with self.subTest(sql=sql):
                r = ses.execute_query(self.db, sql)
                self.assertFalse(r["ok"], r)
                self.assertTrue(r["blocked"], r)
        self.assertFalse((self.tmp / "other.db").exists())
        conn = sqlite3.connect(self.db)
        self.assertEqual(conn.execute("SELECT count(*), sum(age) FROM people").fetchone(), (4, 147))
        conn.close()

    def test_load_extension_is_unavailable(self):
        r = ses.execute_query(self.db, "SELECT load_extension('/nonexistent/libx')")
        self.assertFalse(r["ok"])

    def test_timeout_aborts_a_runaway_recursive_cte(self):
        sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
        start = time.monotonic()
        r = ses.execute_query(self.db, sql, timeout_ms=200)
        self.assertLess(time.monotonic() - start, 5.0)
        self.assertFalse(r["ok"])
        self.assertTrue(r["timed_out"], r)
        # a bounded recursive CTE is allowed
        ok = ses.execute_query(self.db, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 5) SELECT x FROM c")
        self.assertEqual([r[0] for r in ok["rows"]], [1, 2, 3, 4, 5])

    def test_row_cap(self):
        sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 50) SELECT x FROM c"
        self.assertTrue(ses.execute_query(self.db, sql, row_cap=50)["ok"])
        r = ses.execute_query(self.db, sql, row_cap=49)
        self.assertFalse(r["ok"])
        self.assertTrue(r["over_cap"])

    def test_one_huge_value_or_result_is_refused_not_buffered(self):
        # a single function call is one VM step (no progress-handler check): the size limits stop it
        for sql in ("SELECT randomblob(150000000)", "SELECT printf('%.*c', 300000000, 'x')", "SELECT zeroblob(50000000)"):
            with self.subTest(sql=sql):
                r = ses.execute_query(self.db, sql, timeout_ms=100)
                # refused, or (printf) the value collapses to NULL: never a buffered huge value
                if r["ok"]:
                    self.assertTrue(all(v is None or len(v) <= ses._MAX_VALUE_BYTES for row in r["rows"] for v in row), sql)
        with patch.object(ses, "_MAX_RESULT_BYTES", 10_000):
            sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 50) SELECT randomblob(500) FROM c"
            r = ses.execute_query(self.db, sql)
        self.assertTrue(r["over_cap"], r)

    def test_uri_characters_in_a_file_name_are_not_parameters(self):
        odd = make_db(self.tmp / "a?mode=rwc#x.db")
        r = ses.execute_query(str(odd), "SELECT count(*) FROM people")
        self.assertEqual(r["rows"], [(4,)], r)
        self.assertFalse(ses.execute_query(str(odd), "DELETE FROM people")["ok"])

    def test_path_outside_data_dir_is_rejected(self):
        root = self.tmp / "projects" / "7"
        inside = make_db(root / "dbs" / "in.db")
        outside = self.tmp / "one.db"
        self.assertEqual(ses.resolve_db_paths(["dbs/in.db"], root), [os.path.realpath(inside)])
        for bad in (str(outside), "../../one.db", "dbs/missing.db"):
            with self.subTest(path=bad), self.assertRaises(ses.SqlPathError):
                ses.resolve_db_paths([bad], root)
        link = root / "dbs" / "link.db"
        os.symlink(outside, link)
        with self.assertRaises(ses.SqlPathError):
            ses.resolve_db_paths(["dbs/link.db"], root)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sql-cmp-"))
        self.db = str(make_db(self.tmp / "one.db"))

    def row(self, pred, gold, paths=None, **cfg):
        return ses.score_row(pred, gold, paths or [self.db], {**CFG, **cfg})

    def test_unordered_is_a_multiset(self):
        gold = "SELECT name FROM people WHERE team = 'a'"
        self.assertTrue(self.row("SELECT name FROM people WHERE team = 'a' ORDER BY name DESC", gold)["pass"])
        # a missing duplicate fails a multiset compare
        self.assertFalse(self.row("SELECT DISTINCT age FROM people", "SELECT age FROM people")["pass"])
        # column names are ignored, positions are not
        self.assertTrue(self.row("SELECT name AS n, age AS a FROM people", "SELECT name, age FROM people")["pass"])
        self.assertFalse(self.row("SELECT age, name FROM people", "SELECT name, age FROM people")["pass"])

    def test_ordered_when_the_outer_query_orders(self):
        gold = "SELECT name FROM people ORDER BY id"
        r = self.row("SELECT name FROM people ORDER BY id DESC", gold)
        self.assertTrue(r["ordered"])
        self.assertFalse(r["pass"])
        self.assertTrue(self.row("SELECT name FROM people ORDER BY age, id", "SELECT name FROM people ORDER BY id", order="never")["pass"])
        self.assertFalse(self.row("SELECT name FROM people ORDER BY id DESC", "SELECT name FROM people", order="always")["pass"])

    def test_order_by_inside_subquery_or_window_is_not_outer(self):
        for sql in (
            "SELECT name FROM (SELECT name FROM people ORDER BY id)",
            "SELECT name, row_number() OVER (ORDER BY id) FROM people",
            "WITH t AS (SELECT * FROM people ORDER BY id) SELECT name FROM t",
            "SELECT name FROM people WHERE name <> 'order by' -- ORDER BY id",
        ):
            with self.subTest(sql=sql):
                self.assertFalse(ses.has_outer_order_by(sql))
        self.assertTrue(ses.has_outer_order_by("WITH t AS (SELECT 1) SELECT * FROM t ORDER BY 1"))

    def test_large_integers_compare_exactly_in_both_modes(self):
        big = 2**53
        for ordered in (True, False):
            with self.subTest(ordered=ordered):
                self.assertFalse(ses.compare_results([(big + 1,)], [(big,)], ordered=ordered, rel_tol=1e-6))
                self.assertTrue(ses.compare_results([(big + 1,)], [(big + 1,)], ordered=ordered, rel_tol=1e-6))
        self.assertTrue(ses.compare_results([(3,), (1,)], [(1.0,), (3.0,)], ordered=False, rel_tol=1e-6))

    def test_union_and_cte_outer_order_by(self):
        self.assertTrue(ses.has_outer_order_by("SELECT name FROM people UNION SELECT team FROM people ORDER BY 1"))
        self.assertFalse(ses.has_outer_order_by("WITH t AS (SELECT name FROM people ORDER BY id LIMIT 2) SELECT name FROM t"))

    def test_nulls_and_float_tolerance(self):
        self.assertTrue(self.row("SELECT score FROM people", "SELECT score FROM people")["pass"])  # NULL == NULL
        self.assertFalse(self.row("SELECT coalesce(score, 0) FROM people", "SELECT score FROM people")["pass"])
        self.assertTrue(self.row("SELECT 1.0000000001", "SELECT 1.0")["pass"])
        self.assertTrue(self.row("SELECT 3", "SELECT 3.0")["pass"])  # int vs float numerically
        self.assertFalse(self.row("SELECT 1.01", "SELECT 1.0")["pass"])
        self.assertTrue(self.row("SELECT 1.01", "SELECT 1.0", float_rel_tol=0.05)["pass"])
        self.assertFalse(self.row("SELECT '3'", "SELECT 3")["pass"])  # string vs number
        # tolerance applies in multiset compares too
        self.assertTrue(self.row(
            "SELECT score * (1 + 1e-9) FROM people WHERE score IS NOT NULL", "SELECT score FROM people WHERE score IS NOT NULL",
        )["pass"])

    def test_refusal_rows(self):
        r = self.row("NO_QUERY", "NO_QUERY")
        self.assertEqual((r["pass"], r["status"], r["refusal"]), (True, "refusal_correct", True))
        r = self.row("SELECT 1", "NO_QUERY")
        self.assertEqual((r["pass"], r["status"]), (False, "refusal_wrong"))
        r = self.row("NO_QUERY", "SELECT 1")
        self.assertEqual((r["pass"], r["status"]), (False, "unexpected_refusal"))
        self.assertTrue(self.row("CANNOT", "CANNOT", refusal_marker="CANNOT")["pass"])

    def test_gold_error_is_unscoreable(self):
        r = self.row("SELECT 1", "SELECT nope FROM missing_table")
        self.assertIsNone(r["pass"])
        self.assertEqual(r["status"], "gold_error")

    def test_blocked_prediction_fails_and_is_flagged(self):
        r = self.row("DELETE FROM people", "SELECT 1")
        self.assertEqual((r["pass"], r["status"], r["blocked"]), (False, "blocked", True))

    def test_all_databases_must_match(self):
        other = str(make_db(self.tmp / "two.db", [(1, "ada", 36, 1.5, "a"), (2, "bob", 41, 2.25, "a")]))
        gold = "SELECT name FROM people WHERE team = 'a'"
        pred = "SELECT name FROM people WHERE id IN (1, 3)"  # right on the first, wrong on the second
        self.assertTrue(self.row(pred, gold, [self.db])["pass"])
        r = self.row(pred, gold, [self.db, other])
        self.assertFalse(r["pass"])
        self.assertEqual([e["pass"] for e in r["per_db"]], [True, False])
        self.assertTrue(self.row(gold, gold, [self.db, other])["pass"])

    def test_block_aggregate(self):
        preds = [
            {"prediction": "SELECT name FROM people", "reference": "SELECT name FROM people"},
            {"prediction": "SELECT age FROM people", "reference": "SELECT name FROM people"},
            {"prediction": "DROP TABLE people", "reference": "SELECT 1"},
            {"prediction": "SELECT 1", "reference": "SELECT * FROM nope"},
            {"prediction": "NO_QUERY", "reference": "NO_QUERY"},
            {"prediction": "SELECT 1", "reference": "NO_QUERY"},
            {"prediction": "x", "reference": "y", "row_skipped": True},
        ]
        block = ses.score_sql_block(preds, [self.db], CFG)
        self.assertEqual(block["n_rows"], 6)
        self.assertEqual(block["n_scored"], 5)
        self.assertEqual(block["n_unscoreable"], 1)
        self.assertEqual(block["n_blocked"], 1)
        self.assertEqual(block["n_refusal_rows"], 2)
        self.assertEqual(block["refusal_correct_rate"], 0.5)
        self.assertEqual(block["exec_match_rate"], 0.4)
        self.assertEqual([p.get("row_sql_exec_pass") for p in preds], [True, False, False, None, True, False, None])
        self.assertNotIn("row_sql_exec", preds[-1])


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_follows_db_bytes_and_config(self):
        tmp = Path(tempfile.mkdtemp(prefix="sql-fp-"))
        db = str(make_db(tmp / "one.db"))
        first = ses.db_fingerprint(CFG, [db])
        self.assertEqual(first, ses.db_fingerprint(CFG, [db]))
        self.assertNotEqual(first, ses.db_fingerprint({**CFG, "row_cap": 5}, [db]))
        make_db(tmp / "one.db", [(1, "zed", 1, 1.0, "z")])
        self.assertNotEqual(first, ses.db_fingerprint(CFG, [db]))

    def test_off_means_no_fingerprint(self):
        self.assertIsNone(ses.sql_execution_fingerprint({}, 1))
        self.assertIsNone(ses.sql_execution_fingerprint({"sql_execution": {"enabled": True}}, 1))  # no databases
        self.assertIsNone(ses.effective_config({"sql_execution": {"enabled": False, "db_paths": ["a.db"]}}))

    def test_normalize_rejects_bad_values(self):
        cfg = ses.normalize_config({"timeout_ms": -1, "row_cap": True, "float_rel_tol": 5, "order": "x", "refusal_marker": " "})
        self.assertEqual({k: cfg[k] for k in ("timeout_ms", "row_cap", "float_rel_tol", "order", "refusal_marker")},
                         {"timeout_ms": 2000, "row_cap": 10000, "float_rel_tol": 1e-6, "order": "auto", "refusal_marker": "NO_QUERY"})

    def test_cache_key_rejects_a_result_scored_against_other_bytes(self):
        from app.services.post_training_eval_service import _baseline_scoring_matches

        r = EvalResult(metrics={"scoring_mode": "qa", "sql_execution_fingerprint": "sql_exec/v1+aaa", "prompt_format_version": 2})
        self.assertTrue(_baseline_scoring_matches(r, None, None, "sql_exec/v1+aaa"))
        self.assertFalse(_baseline_scoring_matches(r, None, None, "sql_exec/v1+bbb"))
        self.assertFalse(_baseline_scoring_matches(r, None, None, None))
        plain = EvalResult(metrics={"scoring_mode": "qa", "prompt_format_version": 2})
        self.assertTrue(_baseline_scoring_matches(plain, None, None, None))  # scoring off: unchanged
        self.assertFalse(_baseline_scoring_matches(plain, None, None, "sql_exec/v1+aaa"))


def run_async(coro_fn):
    async def go():
        async with async_session_factory() as db:
            return await coro_fn(db)

    return asyncio.run(go())


GOLD = [
    "SELECT name FROM people WHERE team = 'a'",
    "SELECT count(*) FROM people",
    "SELECT name FROM people ORDER BY age DESC, id",
    "NO_QUERY",
    "SELECT broken FROM nowhere",
]


class EvalWiringTests(unittest.TestCase):
    def seed(self) -> tuple[int, int]:
        pid = client.post("/api/projects", json={"name": f"sql-exec-{time.time_ns()}", "description": "x"}).json()["id"]
        root = Path(settings.DATA_DIR) / "projects" / str(pid)
        make_db(root / "dbs" / "main.db")
        prepared = root / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        rows = [{"question": f"q{i}", "answer": g} for i, g in enumerate(GOLD)]
        (prepared / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        async def add(db):
            db.add(Dataset(project_id=pid, name="test", dataset_type=DatasetType.TEST,
                           file_path=str(prepared / "test.jsonl"), record_count=len(rows)))
            exp = Experiment(project_id=pid, name="run", status=ExperimentStatus.COMPLETED, base_model="toy",
                             config={}, output_dir=str(prepared), completed_at=datetime.now(timezone.utc))
            db.add(exp)
            await db.commit()
            return exp.id

        return pid, run_async(add)

    def evaluate(self, pid: int, exp_id: int, answers: list[str]) -> dict:
        def run(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True, decoding="neutral"):
            preds = [{**p, "formatted_prompt": "", "prediction": answers[i], "generated_tokens": 5, "latency_ms": 1.0}
                     for i, p in enumerate(pairs)]
            return preds, {"engine": "stub", "decoding": {"mode": decoding}}

        with patch("app.services.evaluation_service._run_local_inference", run):
            resp = client.post(f"/api/projects/{pid}/evaluation/run-heldout", json={
                "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 10, "model_path": None})
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def test_settings_api_validates_paths(self):
        pid, _ = self.seed()
        bad = client.put(f"/api/projects/{pid}/sql-execution-config", json={"enabled": True, "db_paths": ["/etc/hosts"]})
        self.assertEqual(bad.status_code, 422, bad.text)
        self.assertEqual(bad.json()["error_code"], "INVALID_SQL_EXECUTION_CONFIG")
        bad = client.put(f"/api/projects/{pid}/sql-execution-config", json={"row_cap": 0, "order": "random"})
        self.assertEqual(bad.status_code, 422, bad.text)
        self.assertEqual(len(bad.json()["metadata"]["errors"]), 2)
        ok = client.put(f"/api/projects/{pid}/sql-execution-config", json={"enabled": True, "db_paths": ["dbs/main.db"]})
        self.assertEqual(ok.status_code, 200, ok.text)
        body = ok.json()
        self.assertTrue(body["effective"]["active"])
        self.assertTrue(body["databases"][0]["ok"])
        self.assertTrue(client.get(f"/api/projects/{pid}/sql-execution-config").json()["effective"]["active"])

    def test_heldout_eval_scores_by_execution(self):
        pid, exp_id = self.seed()
        answers = [
            "```sql\nSELECT name FROM people WHERE team='a' ORDER BY name DESC;\n```",  # same set, other order: pass
            "SELECT 4",  # pass
            "SELECT name FROM people ORDER BY age",  # wrong order: fail
            "NO_QUERY",  # refusal: pass
            "SELECT 1",  # gold errors: unscoreable
        ]
        # off by default: no execution block, nothing stamped
        off = self.evaluate(pid, exp_id, answers)["metrics"]
        self.assertNotIn("sql_execution", off)
        self.assertNotIn("sql_execution_fingerprint", off)

        put = client.put(f"/api/projects/{pid}/sql-execution-config", json={"enabled": True, "db_paths": ["dbs/main.db"]})
        self.assertEqual(put.status_code, 200, put.text)
        with patch("app.services.evaluation_service._safe_judge_answers") as judge:
            res = self.evaluate(pid, exp_id, answers)
        judge.assert_not_called()  # scored by execution: the answer judge never runs
        m = res["metrics"]
        block = m["sql_execution"]
        self.assertEqual(block["status"], "scored")
        self.assertEqual((block["n_scored"], block["n_passed"], block["n_unscoreable"]), (4, 3, 1))
        self.assertEqual(m["exec_match_rate"], 0.75)
        self.assertEqual(block["n_refusal_rows"], 1)
        self.assertTrue(m["sql_execution_fingerprint"].startswith("sql_exec/v1+"))
        previews = res["details"]["predictions_preview"]
        self.assertEqual([p["row_sql_exec_pass"] for p in previews], [True, True, False, True, None])
        row_scores = res["details"]["row_scores"]
        self.assertEqual(row_scores["correct"], [1, 1, 0, 1, None])
        self.assertEqual(row_scores["sql_exec_pass"], [1.0, 1.0, 0.0, 1.0, None])

        # the fingerprint follows the database bytes
        make_db(Path(settings.DATA_DIR) / "projects" / str(pid) / "dbs" / "main.db", [(1, "solo", 1, 1.0, "a")])
        again = self.evaluate(pid, exp_id, answers)["metrics"]
        self.assertNotEqual(again["sql_execution_fingerprint"], m["sql_execution_fingerprint"])

    def test_bad_path_in_stored_config_reports_and_scores_nothing(self):
        pid, exp_id = self.seed()

        async def put(db):
            project = await db.get(Project, pid)
            project.runtime_config = {**(project.runtime_config or {}), "sql_execution": {"enabled": True, "db_paths": ["../../x.db"]}}
            await db.commit()

        run_async(put)
        m = self.evaluate(pid, exp_id, ["SELECT 1"] * len(GOLD))["metrics"]
        self.assertEqual(m["sql_execution"]["status"], "config_error")
        self.assertNotIn("exec_match_rate", m)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
