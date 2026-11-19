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


# --------------------------------------------------------------------------
# Review fixes: writes under the lock, version numbers, content hashes, leakage rules.
# --------------------------------------------------------------------------


def _upload_file(name: str, rows: list[dict]):
    import io

    from starlette.datastructures import UploadFile

    return UploadFile(file=io.BytesIO("".join(json.dumps(r) + "\n" for r in rows).encode()), filename=name)


def _gather(*calls):
    """Run API endpoint coroutines concurrently in ONE event loop, each with its own session."""
    async def one(fn, args, kwargs):
        async with async_session_factory() as db:
            return await fn(*args, **kwargs, db=db)

    async def go():
        return await asyncio.gather(*(one(fn, a, k) for fn, a, k in calls), return_exceptions=True)

    return asyncio.run(go())


def _version_rows(dataset_id: int) -> list[int]:
    from sqlalchemy import select

    from app.models.dataset import DatasetVersion

    async def q(db):
        rows = await db.execute(select(DatasetVersion.version).where(DatasetVersion.dataset_id == dataset_id))
        return sorted(rows.scalars().all())

    return run_async(q)


class ConcurrentWriteTests(unittest.TestCase):
    """Two quick writes to one set: each re-reads the set inside the write lock, so neither overwrites
    the other's version file, drops it from the metadata or duplicates a DatasetVersion row."""

    def test_concurrent_row_deletes_make_two_distinct_versions(self):
        from app.api import eval_sets as api_es

        pid, _exp, _ = seed_project("qa")
        rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
        set_id = upload(pid, "fresh", rows).json()["id"]
        out = _gather(
            (api_es.delete_rows, (pid, "fresh", api_es.RowsDeleteBody(positions=[0])), {}),
            (api_es.delete_rows, (pid, "fresh", api_es.RowsDeleteBody(positions=[0])), {}),
        )
        self.assertTrue(all(not isinstance(o, BaseException) for o in out), out)
        listing = client.get(f"/api/projects/{pid}/eval-sets/fresh").json()
        self.assertEqual([v["version"] for v in listing["versions"]], [1, 2, 3])
        self.assertEqual(listing["current_version"], 3)
        self.assertEqual(_version_rows(set_id), [1, 2, 3])  # no duplicate DatasetVersion
        # The second delete started from the first one's version: two rows gone, in either order.
        self.assertEqual([r["question"] for r in ess.read_version_rows(ess.version_path(pid, "fresh", 3))],
                         ["q2", "q3", "q4"])
        self.assertEqual(len(ess.read_version_rows(ess.version_path(pid, "fresh", 2))), 4)
        sources = sorted(v["source"]["from_version"] for v in listing["versions"][1:])
        self.assertEqual(sources, [1, 2])

    def test_upload_and_variant_interleaved_keep_both(self):
        from app.api import eval_sets as api_es

        pid, _exp, _ = seed_project("qa")
        set_id = upload(pid, "fresh", [{"question": "q1", "answer": "a1"}]).json()["id"]
        out = _gather(
            (api_es.new_version_from_file, (pid, "fresh", _upload_file("v2.jsonl", [{"question": "q2", "answer": "a2"}])),
             {"field_mapping": None, "drop_rejected": False, "drop_lines": None}),
            (api_es.add_variant, (pid, "fresh", api_es.VariantBody(name="native", system="strip")), {}),
            (api_es.new_version_from_file, (pid, "fresh", _upload_file("v3.jsonl", [{"question": "q3", "answer": "a3"}])),
             {"field_mapping": None, "drop_rejected": False, "drop_lines": None}),
        )
        self.assertTrue(all(not isinstance(o, BaseException) for o in out), out)
        listing = client.get(f"/api/projects/{pid}/eval-sets/fresh").json()
        self.assertEqual([v["version"] for v in listing["versions"]], [1, 2, 3])
        self.assertEqual([v["name"] for v in listing["variants"]], ["native"])
        self.assertEqual(_version_rows(set_id), [1, 2, 3])
        files = {ess.read_version_rows(ess.version_path(pid, "fresh", v))[0]["question"] for v in (2, 3)}
        self.assertEqual(files, {"q2", "q3"})  # neither upload overwrote the other's file

    def test_eval_refuses_a_version_file_changed_after_it_was_made(self):
        pid, exp, _ = seed_project("qa")
        upload(pid, "fresh", file_rows("qa"))
        path = ess.version_path(pid, "fresh", 1)
        path.write_text(json.dumps({"question": "swapped", "answer": "x"}) + "\n")
        resp = evaluate(pid, exp, ["x"], dataset_name="eval_set:fresh")
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertIn("changed after the version was made", resp.text)
        path.unlink()
        resp = evaluate(pid, exp, ["x"], dataset_name="eval_set:fresh")
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertIn("missing", resp.text)


class DeleteRecreateTests(unittest.TestCase):
    def test_recreated_slug_never_reuses_a_version_and_never_pairs_old_results(self):
        pid, exp, base = seed_project("qa", baseline=True)
        upload(pid, "fresh", file_rows("qa"))
        old_base = evaluate(pid, base, ["Paris", "nope"], dataset_name="eval_set:fresh").json()
        self.assertEqual(old_base["dataset_name"], "eval_set:fresh@v1")
        old_hash = old_base["details"]["dataset"]["eval_set"]["content_sha256"]
        self.assertTrue(old_hash)
        self.assertEqual(client.delete(f"/api/projects/{pid}/eval-sets/fresh").status_code, 200)

        resp = upload(pid, "fresh", [{"question": "Largest planet?", "answer": "Jupiter"}])
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["current_version"], 2)  # continues after the deleted set's v1
        self.assertEqual(resp.json()["dataset_name"], "eval_set:fresh@v2")
        self.assertEqual(client.get(f"/api/projects/{pid}/eval-sets/fresh/rows", params={"version": 1}).status_code, 404)
        new_trained = evaluate(pid, exp, ["Jupiter"], dataset_name="eval_set:fresh").json()
        self.assertEqual(new_trained["dataset_name"], "eval_set:fresh@v2")
        self.assertNotEqual(new_trained["details"]["dataset"]["eval_set"]["content_sha256"], old_hash)

        # Same label, other rows (a result from before version numbers survived a delete): refused.
        a = {"slug": "fresh", "version": 1, "variant": None, "variant_digest": None, "content_sha256": "aaa"}
        self.assertIn("different rows", ess.eval_set_difference(a, {**a, "content_sha256": "bbb"}))
        self.assertIsNone(ess.eval_set_difference(a, {**a, "content_sha256": None}))  # unknown: not refused

        # The card's "last eval" shows only results on the set's rows today.
        sets = client.get(f"/api/projects/{pid}/eval-sets").json()["eval_sets"]
        self.assertEqual(sorted(sets[0]["last_evals"]), ["eval_set:fresh@v2"])
        self.assertEqual(sets[0]["last_evals"]["eval_set:fresh@v2"]["eval_result_id"], new_trained["id"])

    def test_lift_refuses_same_label_with_different_rows(self):
        pid, exp, base = seed_project("qa", baseline=True)
        upload(pid, "fresh", file_rows("qa"))
        evaluate(pid, base, ["Paris", "nope"], dataset_name="eval_set:fresh")
        trained = evaluate(pid, exp, ["Paris", "4"], dataset_name="eval_set:fresh").json()

        async def forge(db):  # a pre-fix result: same label, other rows
            from app.models.experiment import EvalResult

            er = await db.get(EvalResult, trained["id"])
            details = dict(er.details)
            details["dataset"] = {**details["dataset"],
                                  "eval_set": {**details["dataset"]["eval_set"], "content_sha256": "other-rows"}}
            er.details = details
            await db.commit()

        run_async(forge)
        lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary",
                          params={"dataset_name": "eval_set:fresh@v1"}).json()
        self.assertEqual(lift["status"], "not_comparable", lift)
        self.assertIn("different rows", lift["message"])


class EvalSetsNeverTrainTests(unittest.TestCase):
    def test_every_training_assembly_path_refuses_eval_sets(self):
        from app.services import dataset_service as dsvc

        pid, _exp, _ = seed_project("qa")
        upload(pid, "fresh", file_rows("qa"))
        resp = client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["eval_set"]})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("never training data", resp.text)
        resp = client.post(f"/api/projects/{pid}/dataset/split", json={"include_types": ["synthetic", "eval_set"]})
        self.assertEqual(resp.status_code, 400, resp.text)
        resp = client.get(f"/api/projects/{pid}/dataset/preview", params={"include_types": ["eval_set"]})
        self.assertEqual(resp.status_code, 400, resp.text)
        with self.assertRaises(ValueError):
            run_async(lambda db: dsvc.combine_datasets(db, pid, [DatasetType.EVAL_SET]))
        with self.assertRaises(ValueError):
            run_async(lambda db: dsvc.resolve_training_dataset_types(db, pid, ["eval_set"]))
        from app.services.alignment_dataset_service import resolve_alignment_dataset_path

        with self.assertRaises(ValueError):
            resolve_alignment_dataset_path(pid, str(ess.version_path(pid, "fresh", 1)))
        from app.services.alignment_dataset_service import (
            compose_alignment_training_dataset, filter_preference_dataset_by_quality,
        )

        es_file = str(ess.version_path(pid, "fresh", 1))
        with self.assertRaises(ValueError):
            compose_alignment_training_dataset(pid, source_path=es_file)
        with self.assertRaises(ValueError):
            compose_alignment_training_dataset(pid, target_path=es_file)
        with self.assertRaises(ValueError):
            filter_preference_dataset_by_quality(pid, source_path=es_file)
        dsvc.refuse_never_training_types(["cleaned", "synthetic", DatasetType.RAW])  # the others pass


class TombstoneSafetyTests(unittest.TestCase):
    def test_corrupt_file_fails_closed_and_is_untouched(self):
        pid, _exp, _ = seed_project("qa")
        upload(pid, "fresh", file_rows("qa"))
        path = ess.eval_sets_dir(pid) / ess.TOMBSTONES_FILE
        path.write_text("{not json", encoding="utf-8")
        resp = client.delete(f"/api/projects/{pid}/eval-sets/fresh")
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(resp.json().get("error_code"), "EVAL_SET_TOMBSTONES_CORRUPT", resp.json())
        self.assertEqual(path.read_text(encoding="utf-8"), "{not json")
        self.assertEqual(client.get(f"/api/projects/{pid}/eval-sets/fresh").status_code, 200)  # not deleted
        path.write_text("{}", encoding="utf-8")
        self.assertEqual(client.delete(f"/api/projects/{pid}/eval-sets/fresh").status_code, 200)
        self.assertEqual(ess.read_tombstones(pid), {"fresh": 1})

    def test_concurrent_writes_of_different_slugs_both_survive(self):
        import threading

        pid, _exp, _ = seed_project("qa")
        threads = [threading.Thread(target=ess._write_tombstone, args=(pid, f"s{i}", i + 1)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(ess.read_tombstones(pid), {f"s{i}": i + 1 for i in range(16)})
        self.assertEqual([p.name for p in ess.eval_sets_dir(pid).glob("*.tmp-*")], [])


class VariantSupportTests(unittest.TestCase):
    def test_variants_are_refused_where_no_system_message_is_rendered(self):
        for task_type in ("classification", "token_classification", "seq2seq"):
            pid, exp, _ = seed_project("qa")

            async def set_type(db, exp=exp, task_type=task_type):
                e = await db.get(Experiment, exp)
                e.config = {"task_type": task_type}
                await db.commit()

            run_async(set_type)
            upload(pid, "fresh", file_rows("qa"))
            add_variant(pid, "fresh", "native", "strip")
            support = client.get(f"/api/projects/{pid}/eval-sets/variant-support", params={"experiment_id": exp}).json()
            self.assertFalse(support["supported"], support)
            self.assertIn("change only the system message", support["reason"])
            resp = evaluate(pid, exp, ["Paris", "4"], dataset_name="eval_set:fresh", prompt_variant="native")
            self.assertEqual(resp.status_code, 400, resp.text)
            self.assertIn("would change nothing", resp.text)
            # Without a variant the set still evaluates.
            self.assertEqual(evaluate(pid, exp, ["Paris", "4"], dataset_name="eval_set:fresh").status_code, 201)
        pid, exp, _ = seed_project("qa")
        self.assertTrue(client.get(f"/api/projects/{pid}/eval-sets/variant-support",
                                   params={"experiment_id": exp}).json()["supported"])

    def test_detects_a_classifier_head_checkpoint(self):
        import tempfile

        pid, exp, _ = seed_project("qa")
        out = Path(tempfile.mkdtemp())
        (out / "adapter_config.json").write_text(json.dumps({"task_type": "SEQ_2_SEQ_LM", "base_model_name_or_path": "t5"}))

        class Exp:
            config = {}
            output_dir = str(out)

        self.assertIn("seq2seq", ess.variant_unsupported_reason(Exp()))


class D3ReaderFixTests(unittest.TestCase):
    def test_remediation_without_an_id_uses_the_newest_built_in_eval(self):
        from app.services.evaluation_remediation_service import _resolve_eval_result

        pid, exp, _ = seed_project("qa")
        upload(pid, "fresh", file_rows("qa"))
        builtin = evaluate(pid, exp, ["Paris", "4"]).json()
        evaluate(pid, exp, ["Paris", "4"], dataset_name="eval_set:fresh")
        picked = run_async(lambda db: _resolve_eval_result(db, project_id=pid, experiment_id=exp, evaluation_result_id=None))
        self.assertEqual(picked.id, builtin["id"])


class GoldSourceLockTests(unittest.TestCase):
    def test_copy_of_an_answer_key_waits_for_its_writers(self):
        from app.services.gold_service import gold_write_lock

        pid, _exp, _ = seed_project("qa")
        gold = Path(settings.DATA_DIR) / "projects" / str(pid) / "gold" / "gold_dev.jsonl"
        gold.parent.mkdir(parents=True, exist_ok=True)
        gold.write_text(json.dumps({"question": "q1", "answer": "a1"}) + "\n")

        async def add(db):
            db.add(Dataset(project_id=pid, name="gold dev", dataset_type=DatasetType.GOLD_DEV,
                           file_path=str(gold), record_count=1))
            await db.commit()

        run_async(add)

        async def go():
            order: list[str] = []
            async with gold_write_lock(pid, DatasetType.GOLD_DEV):
                async def copy():
                    async with async_session_factory() as db:
                        rows, prov = await ess.rows_from_source(db, pid, {"kind": "split", "split": "gold_dev"})
                    order.append("copied")
                    return rows, prov

                task = asyncio.create_task(copy())
                await asyncio.sleep(0.2)
                order.append("writer")
                gold.write_text("".join(json.dumps({"question": f"q{i}", "answer": "a"}) + "\n" for i in range(2)))
            rows, prov = await task
            return order, rows, prov

        order, rows, _prov = asyncio.run(go())
        self.assertEqual(order, ["writer", "copied"])
        self.assertEqual(len(rows), 2)  # the rows after the writer finished, not a torn read


class LeakScanTests(unittest.TestCase):
    def _with_train(self, shape: str, train_rows: list[dict]) -> tuple[int, int]:
        pid, exp, _ = seed_project(shape)
        prep = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        (prep / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train_rows))

        async def add(db):
            db.add(Dataset(project_id=pid, name="train", dataset_type=DatasetType.TRAIN,
                           file_path=str(prep / "train.jsonl"), record_count=len(train_rows)))
            await db.commit()

        run_async(add)
        return pid, exp

    def test_rows_that_leak_training_data_are_grouped_selectable_rejects(self):
        train = [{"question": "What is the boiling point of water at sea level in celsius?", "answer": "100"},
                 {"question": "Who wrote the play Hamlet about a Danish prince?", "answer": "Shakespeare"}]
        pid, _exp = self._with_train("qa", train)
        rows = [{"question": "What is the boiling point of water at sea level in celsius?", "answer": "100"},
                {"question": "Which gas do plants absorb from the air for photosynthesis?", "answer": "CO2"},
                {"question": "who wrote the play hamlet about a danish prince?", "answer": "Shakespeare"}]
        resp = upload(pid, "fresh", rows)
        self.assertEqual(resp.status_code, 422, resp.text)
        body = resp.json()
        group = body["metadata"]["rejected"]["by_reason"]["leaks_training_data"]
        self.assertEqual(group["lines"], [1, 3])
        self.assertIn("Leaks training data", group["label"])
        self.assertIn("matches a training row", group["rows"][0]["preview"])
        self.assertIn("leak training data", body["message"])
        # Drop one: the other still blocks (never silently kept or removed).
        resp = upload(pid, "fresh", rows, drop_lines=[1])
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["metadata"]["rejected"]["by_reason"]["leaks_training_data"]["lines"], [3])
        resp = upload(pid, "fresh", rows, drop_lines=[1, 3])
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["rows"], 1)
        self.assertEqual(resp.json()["dropped_lines"], [1, 3])
        self.assertEqual(resp.json()["leak_scan"], {"scanned": 1, "leaking_rows_dropped": 0})
        # New versions are scanned too.
        resp = client.post(f"/api/projects/{pid}/eval-sets/fresh/versions",
                           files={"file": ("v2.jsonl", (json.dumps(rows[0]) + "\n").encode(), "application/jsonl")})
        self.assertEqual(resp.status_code, 422)

    def test_leak_flagged_rows_can_be_kept_on_purpose_and_results_say_so(self):
        train = [{"question": "What is the boiling point of water at sea level in celsius?", "answer": "100"},
                 {"question": "Who wrote the play Hamlet about a Danish prince?", "answer": "Shakespeare"}]
        pid, exp = self._with_train("qa", train)
        rows = [{"id": "q-boil", "question": "What is the boiling point of water at sea level in celsius?", "answer": "100"},
                {"question": "Which gas do plants absorb from the air for photosynthesis?", "answer": "CO2"},
                {"question": "who wrote the play hamlet about a danish prince?", "answer": "Shakespeare"},
                "not json"]  # unusable: can be dropped, never kept
        resp = upload(pid, "seen", rows)
        self.assertEqual(resp.status_code, 422, resp.text)
        by_reason = resp.json()["metadata"]["rejected"]["by_reason"]
        self.assertEqual(by_reason["leaks_training_data"]["lines"], [1, 3])
        bad = [r for r in by_reason if r != "leaks_training_data"]
        self.assertTrue(bad)
        bad_line = by_reason[bad[0]]["lines"][0]
        # Only leak-flagged rows can be kept.
        resp = upload(pid, "seen", rows, keep_leak_lines=[1, bad_line])
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json()["error_code"], "EVAL_SET_KEEP_NOT_LEAK_FLAGGED")
        # Keep one leak row, drop the other leak + the unusable one: per row, never all-or-nothing.
        resp = upload(pid, "seen", rows, keep_leak_lines=[1], drop_lines=[3, bad_line])
        self.assertEqual(resp.status_code, 201, resp.text)
        body = resp.json()
        self.assertEqual(body["rows"], 2)
        self.assertEqual(body["leak_scan"]["leaking_rows_kept"], 1)
        kept = body["versions"][-1]["kept_despite_leak"]
        self.assertEqual(kept["count"], 1)
        self.assertEqual(kept["row_ids"], ["q-boil"])
        self.assertEqual(kept["row_indexes"], [0])
        self.assertEqual(kept["lines"], [1])
        # Shown on the result.
        resp = evaluate(pid, exp, ["100"], dataset_name="eval_set:seen")
        self.assertEqual(resp.status_code, 201, resp.text)
        block = resp.json()["details"]["dataset"]["eval_set"]
        self.assertEqual(block["kept_despite_leak"]["row_ids"], ["q-boil"])
        # Deleting the other row keeps the record (re-indexed); deleting the kept row clears it.
        resp = client.post(f"/api/projects/{pid}/eval-sets/seen/rows/delete", json={"positions": [1]})
        self.assertEqual(resp.status_code, 200, resp.text)
        sets = client.get(f"/api/projects/{pid}/eval-sets").json()["eval_sets"]
        latest = [s for s in sets if s["slug"] == "seen"][0]["versions"][-1]
        self.assertEqual(latest["kept_despite_leak"]["row_indexes"], [0])
        resp = client.post(f"/api/projects/{pid}/eval-sets/seen/rows/delete", json={"positions": [0, 1]})
        self.assertNotEqual(resp.status_code, 500)

    def test_a_clean_upload_has_no_kept_record(self):
        pid, _exp = self._with_train("qa", [{"question": "Totally unrelated training question here?", "answer": "x"}])
        resp = upload(pid, "clean", [{"question": "Which gas do plants absorb from the air?", "answer": "CO2"}])
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertNotIn("leaking_rows_kept", resp.json()["leak_scan"])
        self.assertIsNone(resp.json()["versions"][-1]["kept_despite_leak"])

    def test_a_shared_tool_list_is_context_not_a_leak(self):
        from app.services.dataset_service import _normalize_rows_for_training

        words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
                 "quebec romeo sierra tango uniform victor whiskey xray yankee zulu amber basil cedar dune ember fjord "
                 "garnet harbor iris jasper kelp lagoon maple nectar onyx pebble quartz raven saffron thistle umber "
                 "violet willow yarrow zephyr").split()
        # A long shared tool list (many distinct words): without the context-block rule every row would
        # look like a near-duplicate of every training row.
        tools = [{"name": f"tool_{i}", "description": " ".join(f"{words[i]}{w}" for w in words[:12]),
                  "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}}} for i in range(16)]

        def row(prompt, name, i):
            return {"prompt": prompt, "tool_name": name, "tool_args": {"id": str(i)}, "result": "ok",
                    "row_context": {"tools": tools}}

        train_raw = [row(f"Please run the nightly reconciliation for ledger {i}", "tool_1", i) for i in range(6)]
        train = _normalize_rows_for_training([dict(r) for r in train_raw], DatasetType.RAW, "llama3",
                                             adapter_id="tool-call-json", adapter_config={},
                                             task_profile="tool_calling")
        self.assertTrue(train and train[0]["question"].endswith(train[0]["user_prompt"]))
        pid, _exp = self._with_train("tool_calling", train)
        fresh = [row("Cancel the duplicate invoice for customer Mira", "tool_2", 1),
                 row("Please run the nightly reconciliation for ledger 3", "tool_1", 3)]
        resp = upload(pid, "tools", fresh)
        self.assertEqual(resp.status_code, 422, resp.text)
        # Only the real leak: the 16-tool block every row carries is ignored, as in data health.
        self.assertEqual(resp.json()["metadata"]["rejected"]["by_reason"]["leaks_training_data"]["lines"], [2])


class SpanTaggingTests(unittest.TestCase):
    """Decision D6: eval sets + variants checked on span tagging (generative span_set and a
    token-classification run, which refuses a variant)."""

    TEXT = "Call Ada at 555-0100 tomorrow."
    ROWS = [{"text": TEXT, "entities": [{"type": "person", "start": 5, "end": 8, "text": "Ada"},
                                         {"type": "phone", "start": 12, "end": 20, "text": "555-0100"}],
             "row_context": {"system": SYSTEM}},
            {"text": "Nothing to tag here.", "entities": []}]

    def _project(self):
        pid, exp, _ = seed_project("extraction")
        prep = Path(settings.DATA_DIR) / "projects" / str(pid) / "prepared"
        (prep / "manifest.json").write_text(json.dumps({
            "task_profile": "structured_extraction", "adapter_id": "structured-extraction",
            "output_schema": {"scoring_mode": "span_set", "required": ["entities"]},
        }))
        return pid, exp

    def test_span_set_eval_on_a_set_and_a_variant(self):
        pid, exp = self._project()
        self.assertEqual(upload(pid, "spans fresh", self.ROWS).status_code, 201)
        add_variant(pid, "spans-fresh", "native", "strip")
        outputs = [json.dumps({"entities": self.ROWS[0]["entities"]}), json.dumps({"entities": []})]
        pairs: list[dict] = []
        resp = evaluate(pid, exp, outputs, dataset_name="eval_set:spans-fresh", pairs_out=pairs)
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["metrics"].get("scoring_mode"), "span_set", resp.json()["metrics"])
        self.assertEqual(pairs[0].get("row_system"), SYSTEM)
        pairs = []
        resp = evaluate(pid, exp, outputs, dataset_name="eval_set:spans-fresh", prompt_variant="native", pairs_out=pairs)
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["dataset_name"], "eval_set:spans-fresh@v1+native")
        self.assertTrue(all("row_system" not in p for p in pairs))

    def test_token_classification_run_refuses_a_variant(self):
        pid, exp = self._project()

        async def set_type(db):
            e = await db.get(Experiment, exp)
            e.config = {"task_type": "token_classification"}
            await db.commit()

        run_async(set_type)
        upload(pid, "spans", self.ROWS)
        add_variant(pid, "spans", "native", "strip")
        resp = evaluate(pid, exp, ["{}"], dataset_name="eval_set:spans", prompt_variant="native")
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("token-classification", resp.text)


class BuiltInVariantCopyTests(unittest.TestCase):
    def test_copy_a_built_in_set_with_a_variant_in_one_call(self):
        pid, exp, _ = seed_project("qa")
        resp = client.post(f"/api/projects/{pid}/eval-sets/from-source", json={
            "name": "test native", "source": {"kind": "split", "split": "test"},
            "variant": {"name": "native", "system": "strip"}})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual([v["name"] for v in resp.json()["variants"]], ["native"])
        resp = evaluate(pid, exp, ["Paris", "4"], dataset_name="eval_set:test-native", prompt_variant="native")
        self.assertEqual(resp.status_code, 201, resp.text)
        # A bad variant writes nothing.
        resp = client.post(f"/api/projects/{pid}/eval-sets/from-source", json={
            "name": "other", "source": {"kind": "split", "split": "test"}, "variant": {"name": "x", "system": "replace"}})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(client.get(f"/api/projects/{pid}/eval-sets/other").status_code, 404)


if __name__ == "__main__":
    unittest.main()
