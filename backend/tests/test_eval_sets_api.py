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
             prompt_variant: str | None = None, pairs_out: list | None = None, model_path: str | None = None,
             max_samples: int = 100):
    def run(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True,
            decoding="neutral"):
        if pairs_out is not None:
            pairs_out.extend(dict(p) for p in pairs)
        preds = [{**p, "formatted_prompt": "", "prediction": outputs[i % len(outputs)], "generated_tokens": 5,
                  "latency_ms": 1.0} for i, p in enumerate(pairs)]
        return preds, {"engine": "stub", "decoding": {"mode": decoding}}

    body = {"experiment_id": exp_id, "dataset_name": dataset_name, "eval_type": "exact_match", "max_samples": max_samples}
    if prompt_variant:
        body["prompt_variant"] = prompt_variant
    if model_path:
        body["model_path"] = model_path
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

    def test_both_queues_the_base_leg_behind_the_fine_tuned_one_and_the_lift_pairs_them(self):
        """UI simplification phase 7b, gap N1: "Both, base vs fine-tuned" posts the fine-tuned body, then
        the base body (Model Path = the run's base model), both for the run. The in-flight guard used to key
        on (project, experiment) only, so the base leg always got 409 HELDOUT_EVAL_ALREADY_RUNNING. Now both
        are queued, run one after the other (never at once), both succeed, and the lift pairs them. The same
        target queued twice is still refused. Real service path: only model inference is stubbed."""
        from _job_wait import wait_for_jobs_idle

        pid, exp, _ = seed_project("qa")
        upload(pid, "fresh", file_rows("qa"))
        add_variant(pid, "fresh", "native", "strip")
        add_variant(pid, "fresh", "instructed", "replace", INSTRUCTED)
        spans: list[tuple[str, float, float]] = []

        def run(model_ref, pairs, *a, decoding="neutral", **k):
            started = time.monotonic()
            time.sleep(0.6)  # long enough that two legs running at once would overlap
            base = "toy" in str(model_ref)
            spans.append(("base" if base else "tuned", started, time.monotonic()))
            outputs = ["Paris", "nope"] if base else ["Paris", "4"]
            return ([{**p, "formatted_prompt": "", "prediction": outputs[i % 2], "generated_tokens": 1,
                      "latency_ms": 1.0} for i, p in enumerate(pairs)], {"engine": "stub", "decoding": {"mode": decoding}})

        body = {"experiment_id": exp, "dataset_name": "eval_set:fresh", "prompt_variant": "native",
                "eval_type": "exact_match", "max_samples": 10, "max_new_tokens": 128, "temperature": 0.0}
        with patch("app.services.evaluation_service._run_local_inference", run), \
                patch("app.services.evaluation_service._safe_judge_answers", return_value=None):
            tuned = client.post(f"/api/projects/{pid}/evaluation/run-heldout?async_job=true", json=body)
            base = client.post(f"/api/projects/{pid}/evaluation/run-heldout?async_job=true",
                               json={**body, "model_path": "toy"})
            self.assertEqual((tuned.status_code, base.status_code), (202, 202), (tuned.text, base.text))
            # The same target again while it is in flight: refused, naming the Job already queued.
            again = client.post(f"/api/projects/{pid}/evaluation/run-heldout?async_job=true",
                                json={**body, "model_path": "toy/"})
            self.assertEqual(again.status_code, 409, again.text)
            self.assertEqual(again.json().get("error_code"), "HELDOUT_EVAL_ALREADY_RUNNING")
            # A cross-prompt base leg (another variant, base model) also queues.
            cross = client.post(f"/api/projects/{pid}/evaluation/run-heldout?async_job=true",
                                json={**body, "model_path": "toy", "prompt_variant": "instructed"})
            self.assertEqual(cross.status_code, 202, cross.text)
            wait_for_jobs_idle(client, timeout_s=60)
        jobs = [client.get(f"/api/jobs/{j.json()['id']}").json() for j in (tuned, base, cross)]
        self.assertEqual([j["status"] for j in jobs], ["succeeded"] * 3, jobs)
        # One at a time, in the order they were queued.
        self.assertEqual([who for who, _s, _e in spans], ["tuned", "base", "base"])
        for (_a, _s1, end), (_b, start, _e2) in zip(spans, spans[1:]):
            self.assertGreaterEqual(start, end - 0.01, spans)
        label = "eval_set:fresh@v1+native"
        lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary",
                          params={"experiment_id": exp, "dataset_name": label}).json()
        self.assertEqual(lift["status"], "ok", lift)
        self.assertEqual((lift["baseline"]["eval_result_id"], lift["trained"]["eval_result_id"]),
                         (jobs[1]["result"]["eval_result_id"], jobs[0]["result"]["eval_result_id"]))
        # The cross-prompt pair (fine-tuned native vs base instructed) pairs too, labelled as such.
        cross_lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary", params={
            "experiment_id": exp, "dataset_name": label, "base_dataset_name": "eval_set:fresh@v1+instructed"}).json()
        self.assertEqual(cross_lift["status"], "ok", cross_lift)
        self.assertEqual(cross_lift["baseline"]["eval_result_id"], jobs[2]["result"]["eval_result_id"])
        self.assertIsNotNone(cross_lift.get("cross_variant"))

    def test_every_prompt_format_with_both_queues_four_legs_and_pairs_each_format(self):
        """UI simplification phase 7e (gap G1): "Every prompt format" + "Both" on a 2-format set posts fine-tuned
        native, base native, fine-tuned instructed, base instructed (the form's order). Each is its own target, so
        all four queue (no 409), run one after the other, and the lift pairs base vs fine-tuned per format. Real
        endpoint + Jobs; only model inference is stubbed."""
        from _job_wait import wait_for_jobs_idle

        pid, exp, _ = seed_project("qa")
        upload(pid, "fresh", file_rows("qa"))
        add_variant(pid, "fresh", "native", "strip")
        add_variant(pid, "fresh", "instructed", "replace", INSTRUCTED)
        spans: list[tuple[str, float, float]] = []

        def run(model_ref, pairs, *a, decoding="neutral", **k):
            started = time.monotonic()
            time.sleep(0.4)
            base = "toy" in str(model_ref)
            spans.append(("base" if base else "tuned", started, time.monotonic()))
            outputs = ["Paris", "nope"] if base else ["Paris", "4"]
            return ([{**p, "formatted_prompt": "", "prediction": outputs[i % 2], "generated_tokens": 1,
                      "latency_ms": 1.0} for i, p in enumerate(pairs)], {"engine": "stub", "decoding": {"mode": decoding}})

        body = {"experiment_id": exp, "dataset_name": "eval_set:fresh", "eval_type": "exact_match",
                "max_samples": 10, "max_new_tokens": 128, "temperature": 0.0}
        legs = [("native", None), ("native", "toy"), ("instructed", None), ("instructed", "toy")]
        url = f"/api/projects/{pid}/evaluation/run-heldout?async_job=true"
        with patch("app.services.evaluation_service._run_local_inference", run), \
                patch("app.services.evaluation_service._safe_judge_answers", return_value=None):
            posted = [client.post(url, json={**body, "prompt_variant": variant,
                                             **({"model_path": path} if path else {})}) for variant, path in legs]
            self.assertEqual([r.status_code for r in posted], [202] * 4, [r.text for r in posted])
            wait_for_jobs_idle(client, timeout_s=60)
        jobs = [client.get(f"/api/jobs/{r.json()['id']}").json() for r in posted]
        self.assertEqual([j["status"] for j in jobs], ["succeeded"] * 4, jobs)
        self.assertEqual([who for who, _s, _e in spans], ["tuned", "base", "tuned", "base"])
        for (_a, _s1, end), (_b, start, _e2) in zip(spans, spans[1:]):
            self.assertGreaterEqual(start, end - 0.01, spans)
        for i, variant in enumerate(("native", "instructed")):
            lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary", params={
                "experiment_id": exp, "dataset_name": f"eval_set:fresh@v1+{variant}"}).json()
            self.assertEqual(lift["status"], "ok", lift)
            self.assertEqual((lift["trained"]["eval_result_id"], lift["baseline"]["eval_result_id"]),
                             (jobs[2 * i]["result"]["eval_result_id"], jobs[2 * i + 1]["result"]["eval_result_id"]))

    def test_cancelling_a_waiting_eval_never_lets_the_one_behind_it_overtake(self):
        """Review M1 (phase 7b): queue A (fine-tuned), B (base) and C (base, another prompt format) on one run,
        then cancel B while it waits. B's row turns CANCELLED at once, but its runner is still waiting for A;
        C used to read B's terminal row and start while A was still running (two models loaded at once). Now C
        starts only after A ends, B never evaluates, and a waiting Job shows QUEUED with no ``started_at``
        until its work begins (review S4). Real endpoint + Jobs; only model inference is stubbed."""
        from _job_wait import wait_for_jobs_idle
        from sqlalchemy import select

        from app.models.experiment import EvalResult
        from app.services import eval_jobs_service

        pid, exp, _ = seed_project("qa")
        upload(pid, "fresh", file_rows("qa"))
        add_variant(pid, "fresh", "native", "strip")
        add_variant(pid, "fresh", "instructed", "replace", INSTRUCTED)
        spans: list[tuple[str, float, float]] = []

        def run(model_ref, pairs, *a, decoding="neutral", **k):
            started = time.monotonic()
            time.sleep(2.5)
            spans.append(("base" if "toy" in str(model_ref) else "tuned", started, time.monotonic()))
            return ([{**p, "formatted_prompt": "", "prediction": "Paris", "generated_tokens": 1, "latency_ms": 1.0}
                     for p in pairs], {"engine": "stub", "decoding": {"mode": decoding}})

        body = {"experiment_id": exp, "dataset_name": "eval_set:fresh", "prompt_variant": "native",
                "eval_type": "exact_match", "max_samples": 10, "max_new_tokens": 128, "temperature": 0.0}
        url = f"/api/projects/{pid}/evaluation/run-heldout?async_job=true"
        with patch("app.services.evaluation_service._run_local_inference", run), \
                patch("app.services.evaluation_service._safe_judge_answers", return_value=None), \
                patch.object(eval_jobs_service, "_CHAIN_RECHECK_SEC", 0.2):
            a = client.post(url, json=body)
            b = client.post(url, json={**body, "model_path": "toy"})
            c = client.post(url, json={**body, "model_path": "toy", "prompt_variant": "instructed"})
            self.assertEqual([r.status_code for r in (a, b, c)], [202] * 3, [r.text for r in (a, b, c)])
            ids = [r.json()["id"] for r in (a, b, c)]
            time.sleep(1.0)
            waiting = client.get(f"/api/jobs/{ids[2]}").json()
            self.assertEqual(waiting["status"], "queued", waiting)
            self.assertIsNone(waiting["started_at"], waiting)
            self.assertIn(f"Waiting for eval job #{ids[1]}", waiting["progress_message"] or "")
            self.assertEqual(client.post(f"/api/jobs/{ids[1]}/cancel").status_code, 200)
            wait_for_jobs_idle(client, timeout_s=60)
        jobs = [client.get(f"/api/jobs/{i}").json() for i in ids]
        self.assertEqual([j["status"] for j in jobs], ["succeeded", "cancelled", "succeeded"], jobs)
        # B never ran; A and C ran one after the other.
        self.assertEqual([who for who, _s, _e in spans], ["tuned", "base"], spans)
        (_a, _s, a_end), (_c, c_start, _e) = spans
        self.assertGreaterEqual(c_start, a_end - 0.01, spans)
        # C's start time is when its work began, not when it was queued.
        self.assertGreaterEqual(jobs[2]["started_at"], jobs[0]["completed_at"][:19], jobs)

        # B wrote no EvalResult: the project's only results (any run, the base model's too) are A's and C's.
        self.assertNotIn("eval_result_id", jobs[1]["result"] or {})
        made = {j["result"]["eval_result_id"] for j in (jobs[0], jobs[2])}

        async def results(db):
            runs = select(Experiment.id).where(Experiment.project_id == pid)
            rows = (await db.execute(select(EvalResult).where(EvalResult.experiment_id.in_(runs)))).scalars()
            return {r.id for r in rows}

        self.assertEqual(run_async(results), made)


# --------------------------------------------------------------------------
# D3 readers, Compare, lift.
# --------------------------------------------------------------------------


class ReadersCompareLiftTests(unittest.TestCase):
    def test_eval_set_results_say_when_the_base_model_has_a_result_there(self):
        """Phase 7b: ``eval_set_results[].has_base`` (the Ship Coach's note-vs-warning switch)."""
        pid, exp, base = seed_project("qa", baseline=True)
        upload(pid, "fresh", file_rows("qa"))
        evaluate(pid, exp, ["nope", "nope"], dataset_name="eval_set:fresh")
        summary = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": exp}).json()
        self.assertEqual([r["has_base"] for r in summary["eval_set_results"]], [False])
        evaluate(pid, base, ["Paris", "nope"], dataset_name="eval_set:fresh")
        summary = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": exp}).json()
        self.assertEqual([r["has_base"] for r in summary["eval_set_results"]], [True])

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
        # Another variant of the same version: refused by default (as before), pairable only on request.
        plain_base = evaluate(pid, base, ["Paris", "4"], dataset_name="eval_set:fresh").json()
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare",
                         params={"a": base, "b": exp, "eval_a": plain_base["id"], "eval_b": set_trained["id"]}).json()
        self.assertEqual(cmp["row_evidence"]["verdict"], "not_comparable")
        diff = cmp["row_evidence"]["comparability"]["differences"][0]
        self.assertIn("different prompt-format variant", diff["message"])
        self.assertTrue(diff["cross_variant_available"])
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare", params={
            "a": base, "b": exp, "eval_a": plain_base["id"], "eval_b": set_trained["id"], "cross_variant": "true"}).json()
        self.assertNotEqual(cmp["row_evidence"]["verdict"], "not_comparable", cmp["row_evidence"])
        self.assertIn("cross_variant", cmp["row_evidence"]["comparability"])

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

    def test_cross_variant_pairs_share_rows_only_on_one_version_and_say_both_differ(self):
        """P5 review: base on one prompt-format variant vs the fine-tune on another variant of the SAME
        eval-set version (same slug, version, content hash) pairs on the rows both evaluated, labelled;
        another version or set stays refused."""
        pid, exp, base = seed_project("qa", baseline=True)
        upload(pid, "fresh", file_rows("qa"))
        add_variant(pid, "fresh", "native", "strip")
        add_variant(pid, "fresh", "instructed", "replace", INSTRUCTED)
        tuned = evaluate(pid, exp, ["Paris", "4"], dataset_name="eval_set:fresh+native").json()
        base_one = evaluate(pid, base, ["Paris"], dataset_name="eval_set:fresh+instructed", max_samples=1).json()
        self.assertEqual(tuned["details"]["row_scores"]["set_rows"], [0, 1])
        self.assertEqual(base_one["details"]["row_scores"]["set_rows"], [0])

        # Compare: refused by default; opted in, only the shared row paired, labelled with both variants.
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare",
                         params={"a": base, "b": exp, "eval_a": base_one["id"], "eval_b": tuned["id"]}).json()
        self.assertEqual(cmp["row_evidence"]["verdict"], "not_comparable")
        self.assertNotIn("cross_variant", cmp["row_evidence"])
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare", params={
            "a": base, "b": exp, "eval_a": base_one["id"], "eval_b": tuned["id"], "cross_variant": "true"}).json()
        ev = cmp["row_evidence"]
        self.assertNotEqual(ev["verdict"], "not_comparable", ev)
        self.assertEqual((ev["n"], ev["only_in_a"], ev["only_in_b"]), (1, 0, 1), ev)
        label = ev["cross_variant"]["label"]
        self.assertEqual((ev["cross_variant"]["variant_a"], ev["cross_variant"]["variant_b"]), ("instructed", "native"))
        self.assertIn("'instructed' variant", label)
        self.assertIn("'native' variant", label)
        self.assertIn("Both the model and the prompt differ", label)
        self.assertIn(label, ev["comparability"]["notes"])
        # P5b gap 3: the Compare page's "Base model on" picker -- labels, not result ids; either order.
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare", params={
            "a": exp, "b": base, "dataset_name": "eval_set:fresh@v1+native",
            "base_dataset_name": "eval_set:fresh@v1+instructed"}).json()
        self.assertEqual((cmp["a"]["eval_result_id"], cmp["b"]["eval_result_id"]), (base_one["id"], tuned["id"]))
        self.assertEqual(cmp["lift"]["base_dataset_name"], "eval_set:fresh@v1+instructed")
        self.assertEqual(cmp["row_evidence"]["cross_variant"]["variant_a"], "instructed")
        # without it the same labels stay refused (off by default)
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare", params={
            "a": base, "b": exp, "dataset_name": "eval_set:fresh@v1+native"}).json()
        self.assertNotIn("cross_variant", cmp["row_evidence"] or {})

        # Base vs fine-tuned on this set, base leg on the other variant.
        lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary", params={
            "dataset_name": "eval_set:fresh@v1+native", "base_dataset_name": "eval_set:fresh@v1+instructed"}).json()
        self.assertEqual(lift["status"], "ok", lift)
        self.assertEqual((lift["baseline"]["eval_result_id"], lift["trained"]["eval_result_id"]),
                         (base_one["id"], tuned["id"]))
        self.assertIn("the base model answered the 'instructed' variant", lift["cross_variant"]["label"])
        self.assertIn("the fine-tuned run the 'native' variant", lift["cross_variant"]["label"])
        headline = lift["metric_lifts"][0]
        self.assertEqual(headline["evidence"]["n"], 1, headline)
        # One label for both legs: no cross-variant block (unchanged behaviour).
        same = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary",
                          params={"dataset_name": "eval_set:fresh@v1+native"}).json()
        self.assertNotIn("cross_variant", same)

        # Another version of the set: refused, in Compare and in the lift view.
        resp = client.post(f"/api/projects/{pid}/eval-sets/fresh/rows/delete", json={"positions": [1], "version": 1})
        self.assertIn(resp.status_code, (200, 201), resp.text)
        base_v2 = evaluate(pid, base, ["Paris"], dataset_name="eval_set:fresh@v2+instructed").json()
        self.assertEqual(base_v2["dataset_name"], "eval_set:fresh@v2+instructed", base_v2)
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare", params={
            "a": base, "b": exp, "eval_a": base_v2["id"], "eval_b": tuned["id"], "cross_variant": "true"}).json()
        self.assertEqual(cmp["row_evidence"]["verdict"], "not_comparable")
        kinds = {d["kind"]: d["message"] for d in cmp["row_evidence"]["comparability"]["differences"]}
        self.assertIn("different eval-set version", kinds["eval_set"])
        self.assertNotIn("cross_variant", cmp["row_evidence"]["comparability"])
        lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary", params={
            "dataset_name": "eval_set:fresh@v1+native", "base_dataset_name": "eval_set:fresh@v2+instructed"}).json()
        self.assertEqual(lift["status"], "not_comparable", lift)
        self.assertIn("different eval-set version", lift["message"])

    def test_cross_variant_rule(self):
        a = {"slug": "fresh", "version": 1, "variant": "native", "content_sha256": "c1"}
        b = {**a, "variant": "instructed"}
        self.assertIsNotNone(ess.cross_variant_pair(a, b))
        self.assertIsNone(ess.eval_set_difference(a, b, allow_cross_variant=True))
        self.assertIn("different prompt-format variant", ess.eval_set_difference(a, b))  # default: unchanged
        for other in ({**b, "version": 2}, {**b, "slug": "other"}, {**b, "content_sha256": "c2"},
                      {**b, "content_sha256": None}, dict(a, variant_digest="x"), None):
            self.assertIsNone(ess.cross_variant_pair(a, other), other)
        self.assertIsNotNone(ess.eval_set_difference(a, {**b, "version": 2}, allow_cross_variant=True))
        self.assertIn("rows as written (no variant)", ess.cross_variant_pair(a, {**b, "variant": None})["label"])
        rekeyed = ess.rekey_by_set_row({"keys": ["k1", "k2"], "correct": [1, 0], "set_rows": [3, 5]})
        self.assertEqual(rekeyed["keys"], ["set-row-3", "set-row-5"])
        self.assertEqual(ess.rekey_by_set_row({"keys": ["k1"], "correct": [1]}), {"keys": ["k1"], "correct": [1]})

    def test_a_run_evaluated_only_on_eval_sets_says_so_and_its_pass_rate_is_the_headline(self):
        """P5 gap 9: Verdict / scorecard / Ship read eval-set results with the D3 rule (never "the" eval) and
        say "evaluated on eval sets only"; the stored pass rate is the finalized tool-call pass (0.0 kept)."""
        pid, exp, _ = seed_project("tool_calling")
        upload(pid, "fresh", file_rows("tool_calling"))
        summary = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": exp}).json()
        self.assertNotIn("eval_set_results", summary)  # nothing evaluated: unchanged
        good = SHAPES["tool_calling"]["outputs"]
        res = evaluate(pid, exp, [good[0], json.dumps({"tool": "get_item", "args": {"id": "x"}})],
                       dataset_name="eval_set:fresh").json()
        self.assertEqual(res["pass_rate"], res["metrics"]["tool_call_pass_rate"])
        self.assertEqual(res["pass_rate"], 0.5)
        zero = evaluate(pid, exp, ["nope"], dataset_name="eval_set:fresh").json()
        self.assertEqual(zero["pass_rate"], 0.0)  # not "no value"
        summary = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": exp}).json()
        self.assertEqual(summary["verdict"], "not_evaluated")  # D3: still no built-in verdict
        self.assertEqual([r["eval_result_id"] for r in summary["eval_set_results"]], [zero["id"]])
        self.assertIn("Evaluated on eval sets only: eval set fresh v1", summary["message"])
        card = client.get(f"/api/projects/{pid}/evaluation/scorecard/{exp}").json()
        self.assertIn("evaluated on eval sets only (eval set fresh v1)", " ".join(card["reasons"]))
        self.assertFalse(summary["eval_set_results"][0]["has_base"])  # no base-model result on the set yet
        # a built-in result makes it the verdict's eval again (the list is not added then)
        evaluate(pid, exp, good)
        summary = client.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": exp}).json()
        self.assertNotIn("eval_set_results", summary)

    def test_slice_breakdown_from_row_tags_with_strict_and_wilson(self):
        """P5b gap 4: per-slice pass and strict counts (Wilson 95%) from the eval-set rows' own tags, on the
        final row verdicts; a result with no slice carries no block."""
        pid, exp, _ = seed_project("tool_calling")
        client.put(f"/api/projects/{pid}/tool-scoring-config", json={"strict_args": True})
        rows = file_rows("tool_calling")
        rows[0]["slice"] = "seen"
        rows[1]["metadata"] = {"tags": "unseen, hard"}
        upload(pid, "fresh", rows)
        out = [json.dumps({"tool": "get_item", "args": {"id": "0", "extra": 1}}),
               json.dumps({"tool": "get_item", "args": {"id": "1"}})]
        res = evaluate(pid, exp, out, dataset_name="eval_set:fresh").json()
        block = res["metrics"]["slice_breakdown"]
        by = {s["id"]: s for s in block["slices"]}
        self.assertEqual(set(by), {"slice=seen", "metadata.tags=unseen", "metadata.tags=hard"})
        seen = by["slice=seen"]
        self.assertEqual((seen["rows"], seen["scored"], seen["pass"]), (1, 1, 1))
        self.assertEqual((seen["strict"]["pass"], seen["strict"]["scored"]), (0, 1))  # extra argument
        self.assertEqual(by["metadata.tags=unseen"]["strict"]["pass"], 1)
        lo, hi = seen["pass_ci"]
        self.assertTrue(0 < lo < 1 and hi == 1.0)
        self.assertEqual(block["overall"]["pass"], 2)
        plain = seed_project("tool_calling")
        upload(plain[0], "fresh", file_rows("tool_calling"))
        res = evaluate(plain[0], plain[1], out, dataset_name="eval_set:fresh").json()
        self.assertNotIn("slice_breakdown", res["metrics"])

    def test_tool_calling_recipe_rules_read_the_tool_call_pass_and_missing_metrics_are_not_a_fail(self):
        """P5b gap 1: with no rule set chosen, a tool-calling project's rules are its recipe scaffold (headline
        tool_call_pass_rate), and a rule whose metric is missing reads "not measured", never a FAIL; a real
        failure on a measured value still reads fail."""
        from app.models.project import Project

        pid, exp, _ = seed_project("tool_calling")

        async def set_recipe(db):
            project = await db.get(Project, pid)
            project.selected_recipe = {"recipe_id": "tool-calling"}
            await db.commit()

        run_async(set_recipe)
        report = client.get(f"/api/projects/{pid}/evaluation/gates/{exp}").json()
        self.assertEqual(report["pack_resolution"]["source"], "recipe_scaffold_default")
        self.assertEqual(report["task_spec"]["task_profile"], "tool_calling")
        self.assertEqual(report["status"], "not_measured")
        self.assertEqual(report["measured_failed_gate_ids"], [])
        self.assertFalse(report["passed"])  # unchanged: nothing measured is not a pass
        good = SHAPES["tool_calling"]["outputs"]
        evaluate(pid, exp, [good[0], json.dumps({"tool": "get_item", "args": {"id": "x"}})])
        report = client.get(f"/api/projects/{pid}/evaluation/gates/{exp}").json()
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["measured_failed_gate_ids"], ["min_tool_call_pass_rate"])
        evaluate(pid, exp, good)
        report = client.get(f"/api/projects/{pid}/evaluation/gates/{exp}").json()
        self.assertEqual((report["status"], report["passed"]), ("pass", True))
        # a project without such a recipe keeps its default rules
        other, other_exp, _ = seed_project("qa")
        report = client.get(f"/api/projects/{other}/evaluation/gates/{other_exp}").json()
        self.assertNotEqual(report["pack_resolution"]["source"], "recipe_scaffold_default")

    def test_a_base_model_eval_is_filed_under_the_baseline_run_and_pairs_with_the_fine_tune(self):
        """P5 gap 8: an eval run for a fine-tuned run with Model Path = its base model is the base model's
        result: filed under the project's baseline run for that model, labelled, and paired by lift /
        Compare / the eval-sets card on the same set and variant."""
        pid, exp, _ = seed_project("qa")
        upload(pid, "fresh", file_rows("qa"))
        add_variant(pid, "fresh", "native", "strip")
        label = "eval_set:fresh@v1+native"
        tuned = evaluate(pid, exp, ["Paris", "4"], dataset_name="eval_set:fresh+native").json()
        resp = evaluate(pid, exp, ["Paris", "nope"], dataset_name="eval_set:fresh+native", model_path="toy")
        self.assertEqual(resp.status_code, 201, resp.text)
        base = resp.json()
        self.assertNotEqual(base["experiment_id"], exp)
        self.assertEqual(base["metrics"]["evaluated_model"]["kind"], "base")
        self.assertEqual(base["metrics"]["evaluated_model"]["requested_for_experiment_id"], exp)
        # Labelled with the generation config that produced it; the lift reuses it only when that matches.
        generation = base["metrics"]["evaluated_model"]["generation"]
        self.assertEqual(set(generation), {"max_new_tokens", "temperature", "decoding", "max_samples"})
        from types import SimpleNamespace

        from app.services.post_training_eval_service import _filed_base_generation_matches

        filed = SimpleNamespace(metrics=base["metrics"])
        same = {"max_new_tokens": generation["max_new_tokens"], "temperature": generation["temperature"]}
        self.assertTrue(_filed_base_generation_matches(filed, same))
        self.assertFalse(_filed_base_generation_matches(filed, {**same, "max_new_tokens": same["max_new_tokens"] + 1}))
        self.assertFalse(_filed_base_generation_matches(filed, None))
        unlabelled = SimpleNamespace(metrics={"evaluated_model": {"kind": "base", "filed_under_baseline": 1}})
        self.assertFalse(_filed_base_generation_matches(unlabelled, same))
        lift_own = SimpleNamespace(metrics={"evaluated_model": {"kind": "base", "model_path": "m"}})
        self.assertTrue(_filed_base_generation_matches(lift_own, None))  # the lift check's own base evals

        async def baseline_cfg(db):
            row = await db.get(Experiment, base["experiment_id"])
            return row.config, row.base_model

        cfg, base_model = run_async(baseline_cfg)
        self.assertTrue(cfg.get("is_baseline"))
        self.assertEqual(base_model, "toy")
        # the fine-tuned run's latest result on the set is still its own
        lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary", params={"dataset_name": label}).json()
        self.assertEqual(lift["status"], "ok", lift)
        self.assertEqual((lift["baseline"]["eval_result_id"], lift["trained"]["eval_result_id"]), (base["id"], tuned["id"]))
        cmp = client.get(f"/api/projects/{pid}/evaluation/compare",
                         params={"a": base["experiment_id"], "b": exp, "dataset_name": label}).json()
        self.assertEqual((cmp["a"]["eval_result_id"], cmp["b"]["eval_result_id"]), (base["id"], tuned["id"]))
        last = client.get(f"/api/projects/{pid}/eval-sets").json()["eval_sets"][0]["last_evals"][label]
        self.assertEqual(last["eval_result_id"], tuned["id"])
        self.assertEqual(last["base"]["eval_result_id"], base["id"])
        self.assertEqual(last["base"]["base_model"], "toy")
        # a second base eval reuses the same baseline run; another model path stays under the run, labelled
        again = evaluate(pid, exp, ["Paris", "4"], dataset_name="eval_set:fresh+native", model_path="toy/").json()
        self.assertEqual(again["experiment_id"], base["experiment_id"])
        other = evaluate(pid, exp, ["Paris", "4"], dataset_name="eval_set:fresh+native", model_path="org/other").json()
        self.assertEqual(other["experiment_id"], exp)
        self.assertEqual(other["metrics"]["evaluated_model"]["kind"], "custom")
        # an eval without a model path carries no record (results unchanged)
        self.assertNotIn("evaluated_model", tuned["metrics"])

    def test_the_lift_never_cites_a_result_another_model_answered(self):
        """UI simplification phase 2, fix A: an eval run with another model path is filed under the run
        (``evaluated_model.kind == "custom"``); the lift's fine-tuned side is the newest result the run's own
        weights answered, never that one, on a set and on the built-in test examples alike."""
        pid, exp, _ = seed_project("qa")
        upload(pid, "fresh", file_rows("qa"))
        add_variant(pid, "fresh", "native", "strip")
        label = "eval_set:fresh@v1+native"
        base = evaluate(pid, exp, ["Paris", "nope"], dataset_name="eval_set:fresh+native", model_path="toy").json()
        tuned = evaluate(pid, exp, ["Paris", "4"], dataset_name="eval_set:fresh+native").json()
        other = evaluate(pid, exp, ["nope", "nope"], dataset_name="eval_set:fresh+native", model_path="org/other").json()
        self.assertEqual(other["experiment_id"], exp)
        self.assertGreater(other["id"], tuned["id"])
        self.assertEqual(other["metrics"]["evaluated_model"]["kind"], "custom")
        lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary",
                          params={"dataset_name": label, "experiment_id": exp}).json()
        self.assertEqual(lift["status"], "ok", lift)
        self.assertEqual((lift["baseline"]["eval_result_id"], lift["trained"]["eval_result_id"]), (base["id"], tuned["id"]))

        # Built-in test examples: the same rule (newest own result, not the newer custom one).
        base_t = evaluate(pid, exp, ["Paris", "nope"], model_path="toy").json()
        tuned_t = evaluate(pid, exp, ["Paris", "4"]).json()
        other_t = evaluate(pid, exp, ["nope", "nope"], model_path="org/other").json()
        self.assertGreater(other_t["id"], tuned_t["id"])
        lift = client.get(f"/api/projects/{pid}/evaluation/sft-lift-summary", params={"experiment_id": exp}).json()
        self.assertEqual(lift["status"], "ok", lift)
        self.assertEqual((lift["baseline"]["eval_result_id"], lift["trained"]["eval_result_id"]), (base_t["id"], tuned_t["id"]))

        # Only another model's result on a set: no fine-tuned side, and the message says why.
        pid2, exp2, _ = seed_project("qa")
        upload(pid2, "fresh", file_rows("qa"))
        evaluate(pid2, exp2, ["Paris", "nope"], dataset_name="eval_set:fresh", model_path="toy")
        evaluate(pid2, exp2, ["Paris", "4"], dataset_name="eval_set:fresh", model_path="org/other")
        lift = client.get(f"/api/projects/{pid2}/evaluation/sft-lift-summary",
                          params={"dataset_name": "eval_set:fresh@v1", "experiment_id": exp2}).json()
        self.assertEqual(lift["status"], "no_trained", lift)
        self.assertIn("another model", lift["message"])


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
        resp = client.post(f"/api/projects/{pid}/eval-sets/seen/rows/delete", json={"positions": [0]})
        self.assertEqual(resp.status_code, 422, resp.text)  # would leave no rows
        self.assertEqual(resp.json()["error_code"], "EVAL_SET_NO_ROWS")
        # Re-upload with an extra clean row, then delete the kept row: the new version has no record.
        rows2 = rows[:2] + [{"question": "Which planet is known as the red planet in our solar system?", "answer": "Mars"}]
        resp = upload(pid, "seen2", rows2, keep_leak_lines=[1])
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["versions"][-1]["kept_despite_leak"]["row_indexes"], [0])
        resp = client.post(f"/api/projects/{pid}/eval-sets/seen2/rows/delete", json={"positions": [0]})
        self.assertEqual(resp.status_code, 200, resp.text)
        sets = client.get(f"/api/projects/{pid}/eval-sets").json()["eval_sets"]
        latest = [s for s in sets if s["slug"] == "seen2"][0]["versions"][-1]
        self.assertEqual(latest["rows"], 2)
        self.assertIsNone(latest["kept_despite_leak"])

    def test_line_lists_must_be_integer_lists(self):
        pid, _exp = self._with_train("qa", [{"question": "Totally unrelated training question here?", "answer": "x"}])
        rows = [{"question": "Which gas do plants absorb from the air?", "answer": "CO2"}]
        for field in ("drop_lines", "keep_leak_lines"):
            for raw in ('"13"', "[1.7]", '["1"]', "[true]", "{}", "nope"):
                with self.subTest(field=field, raw=raw):
                    resp = client.post(f"/api/projects/{pid}/eval-sets", data={"name": "bad", field: raw},
                                       files={"file": ("set.jsonl", (json.dumps(rows[0]) + "\n").encode(), "application/jsonl")})
                    self.assertEqual(resp.status_code, 400, resp.text)
                    self.assertIn("integer line numbers", resp.text)

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
