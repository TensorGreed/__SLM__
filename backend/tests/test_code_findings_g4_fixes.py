"""Regression tests for the G4 review fixes of the `code-findings` playbooks and plumbing:
seeds come from the dev split only, context datasets cannot be answer keys, scaffolded
max_ gates use `lte`, the seeded-defect anchoring checks, seed rotation, per-chunk reject
persistence, findings-aware dedup, free labels without a taxonomy, and the dataset catalog.

pytest-only (no Jobs are started)."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import label_taxonomy_service as lts
from app.services import provenance_service
from app.services.synth_backends import TemplateBackend
from app.services.synth_backends.base import SynthBackendError
from app.services.synth_playbooks import SynthMode, get_playbook
from app.services.synth_playbooks._code_findings_common import resolve_seeds
from test_code_findings_playbooks import (
    BAD_LINE, CLEAN, SEEDED_CODE, TAXONOMY, clean_row, make_ctx, positive_row, reasons, seeded_line,
)

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


# ── unit ─────────────────────────────────────────────────────────────────


class SeededAnchoringTests(unittest.TestCase):
    def setUp(self):
        self.ctx = make_ctx(SynthMode.SEEDED_DEFECT, params={"label_id": "injection.query"})

    def run_mode(self, mode, rows, ctx=None):
        ctx = ctx or self.ctx
        return get_playbook("code-findings", mode).validate(rows, ctx), ctx

    def test_finding_on_unchanged_lines_is_rejected(self):
        unchanged = {"start_line": 10, "end_line": 10, "evidence": "def get_user(db, user_id):"}
        out, ctx = self.run_mode(SynthMode.SEEDED_DEFECT, [seeded_line(finding=unchanged)])
        self.assertEqual(out, [])
        self.assertEqual(reasons(ctx), ["lines_not_changed"])

    def test_range_overlapping_a_changed_line_is_accepted(self):
        span = {"start_line": 11, "end_line": 12, "evidence": BAD_LINE.strip()}
        out, ctx = self.run_mode(SynthMode.SEEDED_DEFECT, [seeded_line(finding=span)])
        self.assertEqual(reasons(ctx), [])
        self.assertEqual(len(out), 1)

    def test_in_place_edit_counts_as_changed(self):
        edited = 'def get_user(db, user_id):\n    return db.run("SELECT * FROM users WHERE id = " + user_id)'
        row = seeded_line(code=edited, finding={
            "start_line": 11, "end_line": 11, "evidence": 'db.run("SELECT * FROM users WHERE id = " + user_id)'})
        out, ctx = self.run_mode(SynthMode.SEEDED_DEFECT, [row])
        self.assertEqual((reasons(ctx), len(out)), ([], 1))

    def test_one_character_evidence_is_rejected(self):
        for evidence in ("q", " = ", "q ="):
            ctx = make_ctx(SynthMode.SEEDED_DEFECT, params={"label_id": "injection.query"})
            out, ctx = self.run_mode(SynthMode.SEEDED_DEFECT, [seeded_line(finding={"evidence": evidence})], ctx)
            self.assertEqual(out, [], evidence)
            self.assertEqual(reasons(ctx), ["evidence_too_short"], evidence)

    def test_cluster_targeted_uses_the_same_anchoring(self):
        ctx = make_ctx(SynthMode.CLUSTER_TARGETED, failure_cluster={"cluster_id": "c1", "label": "injection.query"})
        unchanged = {"start_line": 10, "end_line": 10, "evidence": "def get_user(db, user_id):"}
        out, ctx = self.run_mode(SynthMode.CLUSTER_TARGETED, [seeded_line(finding=unchanged), seeded_line(finding={"evidence": "q"})], ctx)
        self.assertEqual(out, [])
        self.assertEqual(sorted(reasons(ctx)), ["evidence_too_short", "lines_not_changed"])


class SeedRotationTests(unittest.TestCase):
    def pool(self, n=12):
        return [
            clean_row(code=f"def f{i}(x):\n    return x + {i}", path=f"mod{i}.py", start_line=10 + 100 * i)
            for i in range(n)
        ]

    def test_chunks_use_more_than_five_distinct_seeds(self):
        pb = get_playbook("code-findings", SynthMode.SEEDED_DEFECT)
        seen, per_chunk = set(), []
        for k in range(4):
            ctx = make_ctx(SynthMode.SEEDED_DEFECT, gold=self.pool(), params={"label_id": "injection.query"},
                           target_count=3, chunk_index=k, seed_nonce=7)
            prompt = pb.build_prompt(ctx)
            stored = ctx["code_seeds"]["seeds"]
            self.assertEqual(len(stored), 3)
            per_chunk.append([s["path"] for s in stored])
            for s in stored:
                self.assertIn(f'path="{s["path"]}"', prompt)
                seen.add(s["path"])
            # validate sees the very same seeds build_prompt showed
            self.assertIs(resolve_seeds(ctx, "clean"), stored)
        self.assertGreater(len(seen), 5)
        self.assertEqual(len(seen), 12)
        self.assertEqual(len({tuple(c) for c in per_chunk}), 4)

    def test_choice_is_deterministic_for_a_context_and_moves_with_the_nonce(self):
        def paths(nonce):
            ctx = make_ctx(SynthMode.SEEDED_DEFECT, gold=self.pool(), target_count=3, chunk_index=0, seed_nonce=nonce)
            return [s["path"] for s in resolve_seeds(ctx, "clean")]

        self.assertEqual(paths(0), paths(0))
        self.assertNotEqual(paths(0), paths(1))

    def test_validate_checks_against_the_chunk_seeds(self):
        pb = get_playbook("code-findings", SynthMode.SEEDED_DEFECT)
        ctx = make_ctx(SynthMode.SEEDED_DEFECT, gold=self.pool(), params={"label_id": "injection.query"},
                       target_count=3, chunk_index=2, seed_nonce=0)
        pb.build_prompt(ctx)
        seed = ctx["code_seeds"]["seeds"][0]
        code = seed["code"] + "\n" + BAD_LINE
        line = seed["start_line"] + seed["code"].count("\n") + 1
        out = pb.validate([{"chunk": 1, "code": code, "finding": {
            "label": "injection.query", "start_line": line, "end_line": line, "evidence": BAD_LINE.strip(), "attrs": {}}}], ctx)
        self.assertEqual((reasons(ctx), len(out)), ([], 1))
        self.assertEqual(out[0]["payload"]["path"], seed["path"])


class DedupKeyTests(unittest.TestCase):
    def key(self, payload):
        from app.services.synth_playbook_service import _row_key

        return _row_key({"payload": payload, "synth_confidence": 0.9, "synth_source": "x"})

    def test_findings_rows_key_on_code_and_findings(self):
        a = {"row_kind": "findings", "code": "x = 1\ny = 2", "findings_json": {"findings": []}}
        same = {**a, "code": "x = 1\n  y = 2"}
        other = {**a, "findings_json": {"findings": [{"label": "l", "start_line": 1, "end_line": 1}]}}
        self.assertEqual(self.key(a), self.key(same))
        self.assertNotEqual(self.key(a), self.key(other))

    def test_code_review_payloads_dedupe_as_before(self):
        one = {"code": "def f(): pass", "review": "needs a docstring"}
        two = {"code": "def f(): pass", "review": "rename f"}
        self.assertNotEqual(self.key(one), self.key(two))  # `code` is not a dedup field
        self.assertEqual(self.key({"text": "Hello  World", "code": "a"}), "hello world")


class ScaffoldOperatorTests(unittest.TestCase):
    def test_code_findings_max_gates_are_lte(self):
        from app.services.eval_pack_scaffold_service import scaffold_pack

        gates = {g["gate_id"]: g for g in scaffold_pack("code-findings")["task_specs"][0]["gates"]}
        self.assertEqual(gates["max_clean_chunk_fp_rate"]["operator"], "lte")
        self.assertEqual(gates["max_malformed_rate"]["operator"], "lte")
        for gate_id in ("min_f1", "min_attr_accuracy_mean", "min_safety_pass_rate"):
            self.assertEqual(gates[gate_id]["operator"], "gte", gate_id)

    def test_other_recipes_stay_gte(self):
        from app.services.eval_pack_scaffold_service import scaffold_pack

        for recipe in ("classification", "span-extraction", "summarization", "qa-sft", "generic-sft", "code-review"):
            gates = scaffold_pack(recipe)["task_specs"][0]["gates"]
            self.assertTrue(gates and all(g["operator"] == "gte" for g in gates), recipe)


class PreflightMediaRefTests(unittest.TestCase):
    def test_path_of_a_findings_row_is_not_a_media_ref(self):
        from app.services.training_preflight_service import _pick_media_ref

        self.assertEqual(_pick_media_ref({"row_kind": "findings", "code": "x", "path": "src/app/db.py"}, ("image", "path")), "")
        self.assertEqual(_pick_media_ref({"code": "x", "path": "icons/logo.png"}, ("path",)), "")

    def test_other_rows_keep_the_previous_behaviour(self):
        from app.services.training_preflight_service import _pick_media_ref

        self.assertEqual(_pick_media_ref({"path": "clips/a.wav"}, ("audio", "path")), "clips/a.wav")
        self.assertEqual(_pick_media_ref({"path": "data/x"}, ("path",)), "data/x")
        self.assertEqual(_pick_media_ref({"image": "a.png", "path": "b.png"}, ("image", "path")), "a.png")


class ChunkRejectResumeTests(unittest.TestCase):
    PROJECT_ID = 987001

    class Backend:
        playbook_rows_per_call = 2

        def __init__(self, explode_on: int | None):
            self.calls = 0
            self.explode_on = explode_on

        def describe(self):
            return "fake chunk backend"

        async def complete(self, prompt, **kw):
            self.calls += 1
            if self.explode_on == self.calls:
                raise KeyboardInterrupt("run interrupted")
            from app.services.synth_backends.template import code_findings_rows

            good = code_findings_rows(prompt, "seeded_defect").splitlines()
            bad = json.loads(good[0])
            bad["finding"] = {**bad["finding"], "label": "secret.hardcoded"}  # label_mismatch
            return "\n".join([*good, json.dumps(bad)])

    def run_it(self, backend):
        from app.services.synth_playbook_service import _run_chunked

        gold = [clean_row(code=f"def f{i}(x):\n    return x + {i}", path=f"mod{i}.py") for i in range(8)]
        ctx = make_ctx(SynthMode.SEEDED_DEFECT, gold=gold, params={"label_id": "injection.query"}, target_count=4,
                       project_id=self.PROJECT_ID)
        out = asyncio.run(_run_chunked(
            get_playbook("code-findings", SynthMode.SEEDED_DEFECT), ctx, SynthMode.SEEDED_DEFECT, backend, 2,
            progress=None, recipe_id="code-findings", target_class=None, failure_cluster=None))
        return ctx, out

    def tearDown(self):
        import shutil

        shutil.rmtree(Path(settings.DATA_DIR) / "projects" / str(self.PROJECT_ID), ignore_errors=True)

    def test_rejects_of_a_resumed_chunk_are_reported(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_it(self.Backend(explode_on=2))
        ctx, (_p, _r, rows, _e, meta, _path) = self.run_it(self.Backend(explode_on=None))
        self.assertEqual(meta["resumed_chunks"], 1)
        self.assertEqual(len(rows), 4)
        # one rejected candidate per chunk: the resumed chunk's reject survives the restart
        self.assertEqual(reasons(ctx), ["label_mismatch", "label_mismatch"])


# ── API ──────────────────────────────────────────────────────────────────


class ApiBase(unittest.TestCase):
    TAXONOMY_ON = True

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

    def setUp(self):
        r = client.post("/api/projects", json={"name": f"cf-g4-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        resp = client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "code-findings"})
        self.assertEqual(resp.status_code, 200, resp.text)
        if self.TAXONOMY_ON:
            put = client.put(f"/api/projects/{self.pid}/label-taxonomy", json={"entries": TAXONOMY["entries"]})
            self.assertEqual(put.status_code, 200, put.text)
        self.base = f"/api/projects/{self.pid}/synthetic/run-playbook"
        self.pdir = Path(settings.DATA_DIR) / "projects" / str(self.pid)
        self.prompts: list[str] = []
        original = TemplateBackend.complete
        prompts = self.prompts

        async def spy(this, prompt, **kw):
            prompts.append(prompt)
            return await original(this, prompt, **kw)

        patcher = patch.object(TemplateBackend, "complete", spy)
        patcher.start()
        self.addCleanup(patcher.stop)

    def add_dataset(self, dataset_type, name, rows):
        from app.database import async_session_factory
        from app.models.dataset import Dataset, DatasetType

        path = self.pdir / "test_inputs" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        async def go():
            async with async_session_factory() as db:
                ds = Dataset(project_id=self.pid, name=name, dataset_type=DatasetType(dataset_type),
                             file_path=str(path), record_count=len(rows))
                db.add(ds)
                await db.commit()
                return ds.id

        return asyncio.run(go())

    def run_mode(self, mode, params=None, count=2, **kw):
        return client.post(self.base, json={"mode": mode, "target_count": count, "params": params, **kw})


HELD_OUT = {"code": "def heldout_marker(token):\n    return token", "path": "heldout.py"}
HELD_OUT_BAD = {
    "code": 'def heldout_bad(pw):\n    logger.info("pw " + pw)', "path": "heldout_bad.py",
    "findings_json": {"findings": [{"label": "logging.sensitive", "start_line": 2, "end_line": 2,
                                    "evidence": 'logger.info("pw " + pw)', "attrs": {}}]},
}


class DevSplitOnlySeedTests(ApiBase):
    def test_a_test_split_chunk_never_reaches_a_prompt_in_any_mode(self):
        self.add_dataset("gold_dev", "dev", [clean_row(), positive_row()])
        self.add_dataset("gold_test", "test", [clean_row(**HELD_OUT), clean_row(**HELD_OUT_BAD)])
        for mode, params in [("seeded_defect", {"label_id": "injection.query"}), ("hard_negatives", None),
                             ("cluster_targeted", {"label_id": "injection.query"})]:
            for count in (1, 3, 6):
                resp = self.run_mode(mode, params, count=count)
                self.assertEqual(resp.status_code, 200, (mode, resp.text))
        self.assertTrue(self.prompts)
        for prompt in self.prompts:
            self.assertNotIn("heldout", prompt)
        self.assertTrue(any("get_user" in p for p in self.prompts))  # dev rows are used
        saved = (self.pdir / "synthetic" / "synthetic.jsonl").read_text()
        self.assertNotIn("heldout", saved)

    def test_only_test_split_rows_and_no_documents_raises_the_diagnostic(self):
        self.add_dataset("gold_test", "test", [clean_row(), positive_row()])
        resp = self.run_mode("seeded_defect", {"label_id": "injection.query"})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("No clean code chunks", resp.text)
        self.assertIn("dev set", resp.text)
        resp = self.run_mode("hard_negatives")
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("No labelled code chunks", resp.text)
        resp = self.run_mode("cluster_targeted", {"label_id": "injection.query"})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(self.prompts, [])

    def test_rows_from_eval_only_sources_are_not_seeds(self):
        self.add_dataset("gold_dev", "dev", [
            clean_row(code="def blocked_marker(a):\n    return a", path="blocked.py", data_source_id=5), clean_row()])
        usage = AsyncMock(return_value={5: provenance_service.USAGE_EVAL})
        with patch.object(provenance_service, "source_usage_map", usage):
            resp = self.run_mode("seeded_defect", {"label_id": "injection.query"}, count=3)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(self.prompts)
        for prompt in self.prompts:
            self.assertNotIn("blocked_marker", prompt)
            self.assertIn("get_user", prompt)

    def test_chunked_run_rotates_through_more_than_five_seeds(self):
        rows = [clean_row(code=f"def f{i}(x):\n    return x + {i}", path=f"mod{i}.py", start_line=10 + 100 * i)
                for i in range(12)]
        self.add_dataset("gold_dev", "dev", rows)
        with patch.object(TemplateBackend, "playbook_rows_per_call", 3, create=True):
            resp = self.run_mode("seeded_defect", {"label_id": "injection.query"}, count=12)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(len(self.prompts), 4)
        shown = {p for prompt in self.prompts for p in (f"mod{i}.py" for i in range(12)) if f'path="{p}"' in prompt}
        self.assertGreater(len(shown), 5)
        # every row validated against the seed its own chunk showed (start lines would disagree otherwise)
        self.assertEqual((len(body["rows"]), body["rejected_count"]), (12, 0))


class ContextDatasetTests(ApiBase):
    def test_answer_key_datasets_cannot_be_the_context(self):
        self.add_dataset("gold_dev", "dev", [clean_row()])
        test_id = self.add_dataset("gold_test", "test", [clean_row()])
        dev_id = self.add_dataset("gold_dev", "dev2", [clean_row()])
        for dataset_id in (test_id, dev_id):
            resp = self.run_mode("seeded_defect", {"label_id": "injection.query", "context_dataset_id": dataset_id,
                                                   "context_column": "path"})
            self.assertEqual(resp.status_code, 400, resp.text)
            self.assertEqual(resp.json().get("error_code"), "context_dataset_not_allowed")
            dry = client.post(self.base + "/dry-run", json={"mode": "seeded_defect", "params": {
                "label_id": "injection.query", "context_dataset_id": dataset_id, "context_column": "path"}})
            self.assertEqual(dry.status_code, 400, dry.text)
        self.assertEqual(self.prompts, [])

    def test_a_raw_dataset_is_still_fine(self):
        self.add_dataset("gold_dev", "dev", [clean_row()])
        raw_id = self.add_dataset("raw", "fields", [{"field_name": "customer_id"}, {"field_name": "order_total"}])
        resp = self.run_mode("seeded_defect", {"label_id": "injection.query", "context_dataset_id": raw_id,
                                               "context_column": "field_name"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIn("customer_id", self.prompts[0])


class FreeLabelTests(ApiBase):
    TAXONOMY_ON = False

    def test_without_a_taxonomy_the_label_is_free_text(self):
        self.add_dataset("gold_dev", "dev", [clean_row()])
        resp = self.run_mode("seeded_defect", {"label_id": "  custom.label  "}, count=2)
        self.assertEqual(resp.status_code, 200, resp.text)
        rows = resp.json()["rows"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["payload"]["findings_json"]["findings"][0]["label"], "custom.label")
        self.assertTrue(rows[0]["synth_source"].endswith("label=custom.label"))

    def test_a_blank_label_is_still_required(self):
        self.add_dataset("gold_dev", "dev", [clean_row()])
        for params in ({"label_id": "   "}, {}, None):
            resp = self.run_mode("seeded_defect", params)
            self.assertEqual(resp.status_code, 400, resp.text)
            self.assertEqual(resp.json().get("error_code"), "missing_label_id")


class ClusterLabelValidationTests(ApiBase):
    def test_a_given_label_is_validated_up_front(self):
        self.add_dataset("gold_dev", "dev", [clean_row()])
        resp = self.run_mode("cluster_targeted", {"label_id": "nope"})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json().get("error_code"), "unknown_label")
        ok = self.run_mode("cluster_targeted", {"label_id": "sqli-like"})
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertTrue(ok.json()["rows"][0]["synth_source"].startswith("playbook:code-findings:cluster_targeted"))
        self.assertIn("Target label id: injection.query", self.prompts[-1])  # canonical id, not the alias

    def test_no_label_is_fine_for_cluster_targeted(self):
        self.add_dataset("gold_dev", "dev", [clean_row()])
        self.assertEqual(self.run_mode("cluster_targeted", None).status_code, 200)
        self.assertEqual(self.run_mode("cluster_targeted", {"label_id": ""}).status_code, 200)


class DatasetCatalogTests(ApiBase):
    def test_list_and_columns(self):
        self.add_dataset("gold_dev", "answer", [clean_row()])
        raw_id = self.add_dataset("raw", "fields", [{"table_name": "t", "field_name": "a"}, {"field_name": "b", "extra": 1}])
        listing = client.get(f"/api/projects/{self.pid}/datasets")
        self.assertEqual(listing.status_code, 200, listing.text)
        by_name = {d["name"]: d for d in listing.json()["datasets"]}
        self.assertEqual(by_name["fields"]["dataset_type"], "raw")
        self.assertEqual(by_name["answer"]["dataset_type"], "gold_dev")
        cols = client.get(f"/api/projects/{self.pid}/datasets/{raw_id}/columns")
        self.assertEqual(cols.status_code, 200, cols.text)
        self.assertEqual(cols.json()["columns"], ["table_name", "field_name", "extra"])
        self.assertEqual(cols.json()["sampled_rows"], 2)

    def test_columns_refuses_the_answer_key_and_other_projects(self):
        gold_id = self.add_dataset("gold_test", "answer", [clean_row()])
        resp = client.get(f"/api/projects/{self.pid}/datasets/{gold_id}/columns")
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json().get("error_code"), "context_dataset_not_allowed")
        self.assertEqual(client.get(f"/api/projects/{self.pid}/datasets/999999/columns").status_code, 404)


if __name__ == "__main__":
    unittest.main()
