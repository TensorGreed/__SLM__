"""Named, versioned eval sets + prompt-format variants (hardening Phase 3), end to end through
``/eval-sets`` and ``/evaluation/run-heldout`` (inference stubbed).

Covers: upload with grouped rejects + drop, immutable versions (upload / copy / row delete), sources
(built-in split, another set), variants (strip / replace / add; immutable; validation), lock + delete,
eval runs on a set + variant for QA, classification, tool calling and extraction (same code path, no
task-specific branch), result labels + provenance, decision D3 (latest-eval readers stay on the built-in
sets), Compare + lift pairing on one set and their refusal of mismatched set / variant pairs, the
0053 migration being a no-op on SQLite, and project duplication copying ``eval_sets/``.

pytest-only. Model inference is stubbed; no network."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import Experiment, ExperimentStatus
from app.services import eval_set_service as ess

client: TestClient | None = None
SYSTEM = "Follow the house style."
INSTRUCTED = "You are a careful assistant. Answer exactly."


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


SHAPES = {
    "qa": {
        "adapter": "default-canonical", "profile": "qa", "config": {},
        "rows": [{"question": "Capital of France?", "answer": "Paris", "row_context": {"system": SYSTEM}},
                 {"question": "2+2?", "answer": "4"}],
        "outputs": ["Paris", "4"],
    },
    "classification": {
        "adapter": "classification-label", "profile": "classification", "config": {"candidates": ["neg", "pos"]},
        "rows": [{"text": "great", "label": "pos", "row_context": {"system": SYSTEM}},
                 {"text": "awful", "label": "neg"}],
        "outputs": ["pos", "neg"],
    },
    "tool_calling": {
        "adapter": "tool-call-json", "profile": "tool_calling", "config": {},
        "rows": [{"prompt": "Get item 0", "tool_name": "get_item", "tool_args": {"id": "0"}, "result": "ok",
                  "row_context": {"system": SYSTEM}},
                 {"prompt": "Get item 1", "tool_name": "get_item", "tool_args": {"id": "1"}, "result": "ok"}],
        "outputs": [json.dumps({"tool": "get_item", "args": {"id": "0"}}),
                    json.dumps({"tool": "get_item", "args": {"id": "1"}})],
    },
    "extraction": {
        "adapter": "structured-extraction", "profile": "structured_extraction", "config": {},
        "rows": [{"text": "Invoice 12 from ACME, total 40", "json": {"vendor": "ACME", "total": 40},
                  "row_context": {"system": SYSTEM}},
                 {"text": "Invoice 13 from Initech, total 7", "json": {"vendor": "Initech", "total": 7}}],
        "outputs": [json.dumps({"vendor": "ACME", "total": 40}), json.dumps({"vendor": "Initech", "total": 7})],
    },
}


def prepared(shape: str) -> list[dict]:
    from app.services.dataset_service import _normalize_rows_for_training

    case = SHAPES[shape]
    rows = _normalize_rows_for_training([dict(r) for r in case["rows"]], DatasetType.RAW, "llama3",
                                        adapter_id=case["adapter"], adapter_config=dict(case["config"]),
                                        task_profile=case["profile"])
    assert len(rows) == len(case["rows"])
    return rows


def seed_project(shape: str = "qa", *, baseline: bool = False) -> tuple[int, int, int | None]:
    """A project whose prepared test split holds ``shape``'s rows, one trained run (+ a baseline run)."""
    case = SHAPES[shape]
    pid = client.post("/api/projects", json={"name": f"evalset-{time.time_ns()}", "description": "x"}).json()["id"]
    rows = prepared(shape)
    prep = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
    prep.mkdir(parents=True, exist_ok=True)
    (prep / "manifest.json").write_text(json.dumps({"task_profile": case["profile"], "adapter_id": case["adapter"]}))
    (prep / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    async def add(db):
        db.add(Dataset(project_id=pid, name="test", dataset_type=DatasetType.TEST,
                       file_path=str(prep / "test.jsonl"), record_count=len(rows)))
        exp = Experiment(project_id=pid, name="trained", status=ExperimentStatus.COMPLETED, base_model="toy",
                         config={}, output_dir=str(prep), completed_at=datetime.now(timezone.utc))
        db.add(exp)
        base = None
        if baseline:
            base = Experiment(project_id=pid, name="baseline", status=ExperimentStatus.COMPLETED, base_model="toy",
                              config={"is_baseline": True}, output_dir=str(prep),
                              completed_at=datetime.now(timezone.utc))
            db.add(base)
        await db.commit()
        return exp.id, (base.id if base is not None else None)

    exp_id, base_id = run_async(add)
    return pid, exp_id, base_id


def upload(pid: int, name: str, rows: list, *, filename: str = "set.jsonl", raw: str | None = None, **form):
    body = raw if raw is not None else "".join((json.dumps(r) if not isinstance(r, str) else r) + "\n" for r in rows)
    data = {"name": name, **{k: (json.dumps(v) if isinstance(v, (list, dict)) else str(v)) for k, v in form.items()}}
    return client.post(f"/api/projects/{pid}/eval-sets", data=data,
                       files={"file": (filename, body.encode("utf-8"), "application/jsonl")})


def file_rows(shape: str) -> list[dict]:
    """Prepared rows as a user would upload them (``row_context`` as an object)."""
    out = []
    for r in prepared(shape):
        r = dict(r)
        if isinstance(r.get("row_context"), str):
            r["row_context"] = json.loads(r["row_context"])
        out.append(r)
    return out


def evaluate(pid: int, exp_id: int, outputs: list[str], *, dataset_name: str = "test",
             prompt_variant: str | None = None, pairs_out: list | None = None):
    def run(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True,
            decoding="neutral"):
        if pairs_out is not None:
            pairs_out.extend(dict(p) for p in pairs)
        preds = [{**p, "formatted_prompt": "", "prediction": outputs[i % len(outputs)], "generated_tokens": 5,
                  "latency_ms": 1.0} for i, p in enumerate(pairs)]
        return preds, {"engine": "stub", "decoding": {"mode": decoding}}

    body = {"experiment_id": exp_id, "dataset_name": dataset_name, "eval_type": "exact_match", "max_samples": 100}
    if prompt_variant:
        body["prompt_variant"] = prompt_variant
    with patch("app.services.evaluation_service._run_local_inference", run), \
            patch("app.services.evaluation_service._safe_judge_answers", return_value=None):
        return client.post(f"/api/projects/{pid}/evaluation/run-heldout", json=body)


def add_variant(pid: int, slug: str, name: str, system: str, text: str | None = None):
    body = {"name": name, "system": system}
    if text is not None:
        body["text"] = text
    return client.post(f"/api/projects/{pid}/eval-sets/{slug}/variants", json=body)


# --------------------------------------------------------------------------
# Pure helpers.
# --------------------------------------------------------------------------


class VariantAndLabelTests(unittest.TestCase):
    def test_strip_replace_add_cover_every_system_source(self):
        rc_row = {"question": "q", "answer": "a", "row_context": {"system": "S", "tools": []}}
        legacy = {"question": "q", "answer": "a", "system_prompt": "L"}
        encoded = {"question": "q", "answer": "a", "row_context": json.dumps({"system": "E"})}
        chat = {"messages": [{"role": "system", "content": "C"}, {"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "yo"}]}
        bare = {"question": "q", "answer": "a"}
        strip = {"system": "strip"}
        self.assertEqual(ess.apply_variant(rc_row, strip)["row_context"], {"tools": []})
        self.assertNotIn("system_prompt", ess.apply_variant(legacy, strip))
        self.assertNotIn("row_context", ess.apply_variant(encoded, strip))
        self.assertEqual([t["role"] for t in ess.apply_variant(chat, strip)["messages"]], ["user", "assistant"])
        replace = {"system": "replace", "text": "R"}
        for row in (rc_row, legacy, encoded, chat, bare):
            out = ess.apply_variant(row, replace)
            self.assertEqual(out["row_context"]["system"], "R")
            self.assertNotIn("system_prompt", out)
        add = {"system": "add", "text": "A"}
        self.assertEqual(ess.apply_variant(bare, add)["row_context"], {"system": "A"})
        for row in (rc_row, legacy, encoded, chat):
            self.assertIs(ess.apply_variant(row, add), row)  # has its own: unchanged
        self.assertEqual(bare, {"question": "q", "answer": "a"})  # inputs are never changed
        self.assertIs(ess.apply_variant(bare, None), bare)

    def test_variant_validation(self):
        with self.assertRaises(ess.EvalSetError):
            ess.normalize_variant({"system": "rewrite"})
        with self.assertRaises(ess.EvalSetError):
            ess.normalize_variant({"system": "replace"})
        with self.assertRaises(ess.EvalSetError):
            ess.normalize_variant({"system": "strip", "extra": 1})
        self.assertEqual(ess.normalize_variant({"system": "strip", "text": None}), {"system": "strip"})

    def test_labels(self):
        self.assertEqual(ess.dataset_label("fresh", 2), "eval_set:fresh@v2")
        self.assertEqual(ess.dataset_label("fresh", 2, "native"), "eval_set:fresh@v2+native")
        self.assertEqual(ess.parse_label("eval_set:fresh@v2+native"), ("fresh", 2, "native"))
        self.assertEqual(ess.parse_label("eval_set:fresh"), ("fresh", None, None))
        self.assertEqual(ess.human_label("eval_set:fresh@v2+native"), "eval set fresh v2 (native)")
        self.assertTrue(ess.is_default_eval_dataset("test"))
        self.assertTrue(ess.is_default_eval_dataset(None))
        self.assertFalse(ess.is_default_eval_dataset("eval_set:fresh@v1"))
        with self.assertRaises(ess.EvalSetError):
            ess.parse_label("eval_set:fresh@latest")

    def test_difference_reasons(self):
        a = {"slug": "fresh", "version": 1, "variant": "native", "variant_digest": "d1"}
        self.assertIsNone(ess.eval_set_difference(None, None))
        self.assertIsNone(ess.eval_set_difference(a, dict(a)))
        self.assertIn("different eval set", ess.eval_set_difference(a, None))
        self.assertIn("different eval-set version", ess.eval_set_difference(a, {**a, "version": 2}))
        self.assertIn("different prompt-format variant", ess.eval_set_difference(a, {**a, "variant": None}))
        self.assertIn("redefined", ess.eval_set_difference(a, {**a, "variant_digest": "d2"}))


class MigrationTests(unittest.TestCase):
    def test_0053_is_a_no_op_on_sqlite_and_idempotent(self):
        path = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "20261119_0053_eval_set_dataset_type.py"
        spec = importlib.util.spec_from_file_location("mig0053", path)
        mod = importlib.util.module_from_spec(spec)
        fake_op = MagicMock()
        fake_op.get_bind.return_value.dialect.name = "sqlite"
        with patch.dict("sys.modules", {"alembic.op": fake_op}):
            with patch("alembic.op", fake_op, create=True):
                spec.loader.exec_module(mod)
                mod.op = fake_op
                mod.upgrade()
                mod.upgrade()
        fake_op.execute.assert_not_called()
        self.assertEqual(mod.down_revision, "20261112_0052")
        pg = MagicMock()
        pg.get_bind.return_value.dialect.name = "postgresql"
        mod.op = pg
        mod.upgrade()
        stmts = [c.args[0] for c in pg.execute.call_args_list]
        self.assertTrue(all("IF NOT EXISTS" in s for s in stmts))
        self.assertEqual(DatasetType("eval_set"), DatasetType.EVAL_SET)


# --------------------------------------------------------------------------
# CRUD.
# --------------------------------------------------------------------------


class EvalSetCrudTests(unittest.TestCase):
    def test_upload_groups_rejects_then_drops_selected_lines(self):
        pid, _exp, _ = seed_project("qa")
        rows = [{"question": "q1", "answer": "a1"}, "not json", {"question": "q2", "answer": "a2"}, "[1, 2]"]
        resp = upload(pid, "Fresh native", rows)
        self.assertEqual(resp.status_code, 422, resp.text)
        body = resp.json()
        self.assertEqual(body["error_code"], "EVAL_SET_IMPORT_REJECTED_ROWS")
        by_reason = body["metadata"]["rejected"]["by_reason"]
        self.assertEqual(by_reason["invalid_json"]["lines"], [2])
        self.assertEqual(by_reason["not_an_object"]["lines"], [4])
        self.assertEqual(client.get(f"/api/projects/{pid}/eval-sets").json()["eval_sets"], [])  # nothing written
        resp = upload(pid, "Fresh native", rows, drop_lines=[2, 4])
        self.assertEqual(resp.status_code, 201, resp.text)
        out = resp.json()
        self.assertEqual((out["slug"], out["current_version"], out["rows"]), ("fresh-native", 1, 2))
        self.assertEqual(out["dropped_lines"], [2, 4])
        self.assertEqual(out["dataset_name"], "eval_set:fresh-native@v1")
        self.assertEqual(upload(pid, "fresh native", rows, drop_rejected=True).status_code, 409)  # same slug

    def test_versions_are_immutable_snapshots(self):
        pid, _exp, _ = seed_project("qa")
        upload(pid, "fresh", [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}])
        resp = client.post(f"/api/projects/{pid}/eval-sets/fresh/versions",
                           files={"file": ("v2.jsonl", b'{"question": "q3", "answer": "a3"}\n', "application/jsonl")})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["current_version"], 2)
        v1 = ess.version_path(pid, "fresh", 1)
        self.assertTrue(v1.exists())
        self.assertEqual(len(ess.read_version_rows(v1)), 2)
        resp = client.post(f"/api/projects/{pid}/eval-sets/fresh/rows/delete", json={"positions": [1], "version": 1})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["version"]["source"], {"kind": "rows_deleted", "from_version": 1, "deleted": 1})
        page = client.get(f"/api/projects/{pid}/eval-sets/fresh/rows", params={"version": 3}).json()
        self.assertEqual([r["row"]["question"] for r in page["rows"]], ["q1"])
        self.assertEqual(len(ess.read_version_rows(v1)), 2)  # v1 untouched
        self.assertEqual(client.post(f"/api/projects/{pid}/eval-sets/fresh/rows/delete",
                                     json={"positions": [9]}).status_code, 400)
        listing = client.get(f"/api/projects/{pid}/eval-sets").json()["eval_sets"][0]
        self.assertEqual([v["version"] for v in listing["versions"]], [1, 2, 3])
        self.assertEqual(client.get(f"/api/projects/{pid}/eval-sets/fresh/rows", params={"version": 9}).status_code, 404)

    def test_sources_split_and_other_set(self):
        pid, _exp, _ = seed_project("classification")
        resp = client.post(f"/api/projects/{pid}/eval-sets/from-source",
                           json={"name": "test copy", "source": {"kind": "split", "split": "test"}})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["rows"], 2)
        self.assertEqual(resp.json()["versions"][0]["source"]["kind"], "split")
        resp = client.post(f"/api/projects/{pid}/eval-sets/from-source",
                           json={"name": "copy of copy", "source": {"kind": "eval_set", "slug": "test-copy", "version": 1}})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["versions"][0]["source"], {"kind": "eval_set", "slug": "test-copy", "version": 1})
        self.assertEqual(client.post(f"/api/projects/{pid}/eval-sets/from-source",
                                     json={"name": "x", "source": {"kind": "split", "split": "gold_test"}}).status_code, 404)
        resp = client.post(f"/api/projects/{pid}/eval-sets/test-copy/versions/from-source",
                           json={"source": {"kind": "split", "split": "test"}})
        self.assertEqual(resp.json()["current_version"], 2)

    def test_variants_lock_and_delete(self):
        pid, _exp, _ = seed_project("qa")
        upload(pid, "fresh", [{"question": "q1", "answer": "a1"}])
        self.assertEqual(add_variant(pid, "fresh", "native", "strip").status_code, 201)
        self.assertEqual(add_variant(pid, "fresh", "native", "strip").status_code, 409)
        self.assertEqual(add_variant(pid, "fresh", "bad", "replace").status_code, 400)
        self.assertEqual(add_variant(pid, "fresh", "Bad Name", "strip").status_code, 400)
        resp = add_variant(pid, "fresh", "instructed", "replace", INSTRUCTED)
        self.assertEqual(resp.status_code, 201)
        names = [v["name"] for v in resp.json()["eval_set"]["variants"]]
        self.assertEqual(names, ["instructed", "native"])
        self.assertEqual(client.post(f"/api/projects/{pid}/eval-sets/fresh/lock").status_code, 200)
        resp = client.post(f"/api/projects/{pid}/eval-sets/fresh/versions",
                           files={"file": ("v2.jsonl", b'{"question": "q", "answer": "a"}\n', "application/jsonl")})
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error_code"], "EVAL_SET_LOCKED")
        self.assertEqual(client.delete(f"/api/projects/{pid}/eval-sets/fresh").status_code, 409)
        self.assertEqual(add_variant(pid, "fresh", "added", "add", "x").status_code, 201)  # variants still allowed
        upload(pid, "scratch", [{"question": "q1", "answer": "a1"}])
        self.assertEqual(client.delete(f"/api/projects/{pid}/eval-sets/scratch").status_code, 200)
        self.assertFalse((ess.eval_sets_dir(pid) / "scratch").exists())
        self.assertEqual(client.get(f"/api/projects/{pid}/eval-sets/scratch").status_code, 404)

    def test_duplicate_project_copies_eval_set_files(self):
        from app.services.project_duplicate_service import copy_data_tree

        pid, _exp, _ = seed_project("qa")
        upload(pid, "fresh", [{"question": "q1", "answer": "a1"}])
        pid2 = client.post("/api/projects", json={"name": f"evalset-copy-{time.time_ns()}", "description": "x"}).json()["id"]
        self.assertIn("eval_sets", copy_data_tree(pid, pid2))
        self.assertTrue(ess.version_path(pid2, "fresh", 1).exists())


# --------------------------------------------------------------------------
# Eval runs: every task shape through the same path.
# --------------------------------------------------------------------------


class EvalSetRunTests(unittest.TestCase):
    def _run_shape(self, shape: str):
        pid, exp, _ = seed_project(shape)
        self.assertEqual(upload(pid, f"{shape} fresh", file_rows(shape)).status_code, 201)
        slug = ess.slugify(f"{shape} fresh")
        add_variant(pid, slug, "native", "strip")
        add_variant(pid, slug, "instructed", "replace", INSTRUCTED)
        outputs = SHAPES[shape]["outputs"]

        pairs: list[dict] = []
        resp = evaluate(pid, exp, outputs, dataset_name=f"eval_set:{slug}", pairs_out=pairs)
        self.assertEqual(resp.status_code, 201, resp.text)
        plain = resp.json()
        self.assertEqual(plain["dataset_name"], f"eval_set:{slug}@v1")
        self.assertEqual(pairs[0].get("row_system"), SYSTEM)  # the rows as stored

        pairs = []
        resp = evaluate(pid, exp, outputs, dataset_name=f"eval_set:{slug}@v1", prompt_variant="native", pairs_out=pairs)
        self.assertEqual(resp.status_code, 201, resp.text)
        native = resp.json()
        self.assertEqual(native["dataset_name"], f"eval_set:{slug}@v1+native")
        self.assertTrue(all("row_system" not in p for p in pairs))
        block = native["details"]["dataset"]["eval_set"]
        self.assertEqual((block["slug"], block["version"], block["variant"], block["rows"]), (slug, 1, "native", 2))
        self.assertEqual(block["variant_spec"], {"system": "strip"})
        self.assertTrue(block["variant_digest"])
        self.assertEqual(native["details"]["dataset"]["dataset_type"], "eval_set")
        self.assertTrue(native["details"]["dataset"]["content_sha256"])
        self.assertEqual(native["details"]["dataset"]["version"], 1)
        self.assertEqual(native["metrics"]["prompt_format_version"], 2)

        pairs = []
        resp = evaluate(pid, exp, outputs, dataset_name=f"eval_set:{slug}+instructed", pairs_out=pairs)
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["dataset_name"], f"eval_set:{slug}@v1+instructed")
        self.assertEqual([p.get("row_system") for p in pairs], [INSTRUCTED, INSTRUCTED])
        self.assertIsNotNone(resp.json()["pass_rate"])
        return pid, exp, slug

    def test_qa(self):
        self._run_shape("qa")

    def test_classification(self):
        self._run_shape("classification")

    def test_tool_calling(self):
        self._run_shape("tool_calling")

    def test_extraction(self):
        self._run_shape("extraction")

    def test_bad_requests(self):
        pid, exp, _ = seed_project("qa")
        upload(pid, "fresh", file_rows("qa"))
        self.assertEqual(evaluate(pid, exp, ["x"], dataset_name="eval_set:fresh", prompt_variant="nope").status_code, 404)
        self.assertEqual(evaluate(pid, exp, ["x"], dataset_name="eval_set:fresh@v7").status_code, 404)
        self.assertEqual(evaluate(pid, exp, ["x"], dataset_name="eval_set:missing").status_code, 404)
        self.assertEqual(evaluate(pid, exp, ["x"], prompt_variant="native").status_code, 400)  # built-in set
        # A plain name never resolves to an eval set (only the eval_set: prefix does).
        self.assertEqual(evaluate(pid, exp, ["x"], dataset_name="fresh").status_code, 400)

    def test_async_job_title_names_the_set(self):
        from _job_wait import wait_for_jobs_idle

        pid, exp, _ = seed_project("qa")
        upload(pid, "fresh", file_rows("qa"))
        add_variant(pid, "fresh", "native", "strip")

        def run(model_ref, pairs, *a, decoding="neutral", **k):
            return ([{**p, "formatted_prompt": "", "prediction": "Paris", "generated_tokens": 1, "latency_ms": 1.0}
                     for p in pairs], {"engine": "stub", "decoding": {"mode": decoding}})

        with patch("app.services.evaluation_service._run_local_inference", run), \
                patch("app.services.evaluation_service._safe_judge_answers", return_value=None):
            resp = client.post(f"/api/projects/{pid}/evaluation/run-heldout?async_job=true", json={
                "experiment_id": exp, "dataset_name": "eval_set:fresh", "prompt_variant": "native",
                "eval_type": "exact_match", "max_samples": 10})
            self.assertEqual(resp.status_code, 202, resp.text)
            job = resp.json()
            self.assertIn("eval set fresh v1 (native)", job["title"])
            wait_for_jobs_idle(client, timeout_s=60)
        done = client.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual(done["status"], "succeeded", done)
        self.assertEqual(done["result"]["eval_set_label"], "eval_set:fresh@v1+native")


# --------------------------------------------------------------------------
# D3 readers, Compare, lift.
# --------------------------------------------------------------------------


class ReadersCompareLiftTests(unittest.TestCase):
    def test_latest_eval_readers_stay_on_built_in_sets(self):
        pid, exp, base = seed_project("qa", baseline=True)
        upload(pid, "fresh", file_rows("qa"))
        add_variant(pid, "fresh", "native", "strip")
        builtin_trained = evaluate(pid, exp, ["Paris", "4"]).json()
        builtin_base = evaluate(pid, base, ["Paris", "nope"]).json()
        set_trained = evaluate(pid, exp, ["nope", "nope"], dataset_name="eval_set:fresh+native").json()
        set_base = evaluate(pid, base, ["Paris", "nope"], dataset_name="eval_set:fresh+native").json()
        self.assertEqual(set_trained["dataset_name"], "eval_set:fresh@v1+native")

        # Default lift + Compare: the built-in results, as before (the newer eval-set runs are ignored).
        lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary").json()
        self.assertEqual(lift["trained"]["dataset_name"], "test")
        self.assertEqual(lift["baseline"]["dataset_name"], "test")
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare", params={"a": base, "b": exp}).json()
        self.assertEqual((cmp["a"]["eval_result_id"], cmp["b"]["eval_result_id"]),
                         (builtin_base["id"], builtin_trained["id"]))
        self.assertIsNone(cmp["a"]["provenance"]["eval_set"])

        summary = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": exp}).json()
        self.assertEqual(summary.get("eval_result_id"), builtin_trained["id"], summary)

        # Picked explicitly: both legs on the eval set.
        lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary",
                          params={"dataset_name": "eval_set:fresh@v1+native"}).json()
        self.assertEqual(lift["status"], "ok", lift)
        self.assertEqual(lift["trained"]["dataset_name"], "eval_set:fresh@v1+native")
        self.assertEqual(lift["dataset_name"], "eval_set:fresh@v1+native")
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare",
                         params={"a": base, "b": exp, "dataset_name": "eval_set:fresh@v1+native"}).json()
        self.assertEqual((cmp["a"]["eval_result_id"], cmp["b"]["eval_result_id"]), (set_base["id"], set_trained["id"]))
        self.assertEqual(cmp["b"]["provenance"]["eval_set"]["variant"], "native")
        self.assertNotEqual(cmp["row_evidence"]["verdict"], "not_comparable", cmp["row_evidence"])

        # Mismatched pairs are refused with the reason.
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare",
                         params={"a": base, "b": exp, "eval_a": builtin_base["id"], "eval_b": set_trained["id"]}).json()
        self.assertEqual(cmp["row_evidence"]["verdict"], "not_comparable")
        kinds = {d["kind"]: d["message"] for d in cmp["row_evidence"]["comparability"]["differences"]}
        self.assertIn("different eval set", kinds["eval_set"])
        plain_base = evaluate(pid, base, ["Paris", "4"], dataset_name="eval_set:fresh").json()
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare",
                         params={"a": base, "b": exp, "eval_a": plain_base["id"], "eval_b": set_trained["id"]}).json()
        kinds = {d["kind"]: d["message"] for d in cmp["row_evidence"]["comparability"]["differences"]}
        self.assertIn("different prompt-format variant", kinds["eval_set"])

        # A variant redefined under the same name: the lift view refuses the pair.
        client.delete(f"/api/projects/{pid}/eval-sets/fresh/variants/native")
        add_variant(pid, "fresh", "native", "replace", INSTRUCTED)
        evaluate(pid, base, ["Paris", "4"], dataset_name="eval_set:fresh+native")
        lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary",
                          params={"dataset_name": "eval_set:fresh@v1+native"}).json()
        self.assertEqual(lift["status"], "not_comparable")
        self.assertIn("redefined", lift["message"])

        sets = client.get(f"/api/projects/{pid}/eval-sets").json()["eval_sets"]
        self.assertIn("eval_set:fresh@v1+native", sets[0]["last_evals"])


if __name__ == "__main__":
    unittest.main()
