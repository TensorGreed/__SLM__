"""`code-remediation` synth playbooks: remediation_from_findings and code_examples.

Unit: each validator on crafted good / bad teacher outputs (every reject reason), fix-pair seeds
(real diff, teacher writes the rationale only), passages attached to the prompt, citations.
Template backend: a deterministic branch whose output the validators accept.
API: both modes through /run-playbook with the template backend stubbed into both registries;
the answer-key test split is never a seed; eval-only ledger sources are skipped; param validation.

pytest-only (the Jobs these tests start are awaited in tearDown).
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from _job_wait import wait_for_jobs_idle
from app.config import settings
from app.main import app
from app.services import label_taxonomy_service as lts
from app.services.synth_backends import TemplateBackend
from app.services.synth_playbooks import SynthMode, get_playbook
from app.services.synth_playbooks._code_remediation_common import build_seeds_from_rows

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    client.__exit__(None, None, None)


TAXONOMY = {
    "entries": [
        {"id": "injection", "title": "Injection"},
        {"id": "injection.query", "title": "Query built from input", "parent": "injection"},
        {"id": "secret.hardcoded", "title": "Hard-coded secret"},
        {"id": "logging.sensitive", "title": "Sensitive value logged"},
    ]
}

BAD = 'def find(db, name):\n    q = "SELECT * FROM users WHERE name = " + name\n    return db.run(q)'
GOOD = 'def find(db, name):\n    q = "SELECT * FROM users WHERE name = ?"\n    return db.run(q, [name])'
FINDING = {"label": "injection.query", "start_line": 11, "end_line": 11,
           "evidence": 'q = "SELECT * FROM users WHERE name = " + name', "attrs": {}}
PASSAGES = [{"text": "Use parameter placeholders; never concatenate input.", "path": "docs/db.md", "source": "db.md"},
            {"text": "db.run(sql, params) binds params safely.", "path": "lib/db.py", "start_line": 5, "end_line": 9,
             "symbol": "run", "passage_id": "3:1"}]


def findings_row(**kw):
    row = {"row_kind": "findings", "code": BAD, "language": "python", "path": "app/users.py", "start_line": 10,
           "end_line": 12, "findings_json": {"findings": [FINDING]}, "labels_raw": ["injection.query"]}
    row.update(kw)
    return row


def fix_pair_row(**kw):
    row = {"row_kind": "patch", "code": BAD, "fixed_code": GOOD, "language": "python", "path": "app/users.py",
           "start_line": 10, "labels": ["injection.query"]}
    row.update(kw)
    return row


def diff(old=BAD, new=GOOD, start=10, path="app/users.py"):
    from app.services.unified_diff import make_unified_diff

    return make_unified_diff(old, new, path=path, start_line=start)


def make_ctx(rows=None, *, params=None, taxonomy=True, **extra):
    seeds = build_seeds_from_rows(rows if rows is not None else [findings_row()], "findings_dataset")
    ctx = {
        "recipe_id": "code-remediation", "project_id": 1, "gold_rows": [], "raw_rows": None, "failure_cluster": None,
        "target_class": None, "target_count": 2, "params": params or {}, "rejects": [],
        "document_chunks": [], "remediation_seeds": seeds,
        "taxonomy": lts.TaxonomyIndex(lts.parse_taxonomy(TAXONOMY)) if taxonomy else None,
    }
    ctx.update(extra)
    return ctx


def reasons(ctx):
    return [r["reason"] for r in ctx["rejects"]]


NO_OP = (
    "--- a/app/users.py\n+++ b/app/users.py\n@@ -11,1 +11,1 @@\n"
    '-    q = "SELECT * FROM users WHERE name = " + name\n+    q = "SELECT * FROM users WHERE name = " + name'
)


def line(**over):
    row = {"task": 1, "patch": diff(), "rationale": "Bind the value instead of concatenating.", "citations": []}
    row.update(over)
    return row


class RemediationSeedTests(unittest.TestCase):
    def test_one_seed_per_finding_and_clean_rows_skipped(self):
        two = findings_row(findings_json={"findings": [FINDING, {**FINDING, "label": "secret.hardcoded"}]})
        clean = findings_row(findings_json={"findings": []})
        seeds = build_seeds_from_rows([two, clean, {"junk": 1}], "findings_dataset")
        self.assertEqual([s["finding"]["label"] for s in seeds], ["injection.query", "secret.hardcoded"])
        self.assertTrue(all(s["start_line"] == 10 and s["end_line"] == 12 for s in seeds))

    def test_fix_pair_seed_has_a_finding_over_the_changed_lines(self):
        (seed,) = build_seeds_from_rows([fix_pair_row()], "findings_dataset")
        self.assertEqual(seed["finding"]["label"], "injection.query")
        self.assertEqual((seed["finding"]["start_line"], seed["finding"]["end_line"]), (11, 12))
        self.assertEqual(seed["fix_after"], GOOD)

    def test_remediation_rows_keep_their_passages(self):
        row = {"row_kind": "remediation", "code": BAD, "language": "python", "path": "a.py", "start_line": 10,
               "finding": FINDING, "passages": PASSAGES, "remediation_json": {}}
        (seed,) = build_seeds_from_rows([row], "answer_key")
        self.assertEqual([p["id"] for p in seed["passages"]], ["P1", "P2"])


class RemediationValidatorTests(unittest.TestCase):
    def setUp(self):
        self.pb = get_playbook("code-remediation", SynthMode.REMEDIATION_FROM_FINDINGS)
        self.ctx = make_ctx()

    def validate(self, *rows, ctx=None):
        return self.pb.validate(list(rows), ctx or self.ctx)

    def test_registered(self):
        self.assertIsNotNone(self.pb)
        self.assertEqual(SynthMode("remediation_from_findings"), SynthMode.REMEDIATION_FROM_FINDINGS)
        self.assertEqual(SynthMode("code_examples"), SynthMode.CODE_EXAMPLES)
        self.assertIsNotNone(get_playbook("code-remediation", SynthMode.CODE_EXAMPLES))

    def test_good_row_is_a_canonical_remediation_row(self):
        out = self.validate(line())
        self.assertEqual(reasons(self.ctx), [])
        p = out[0]["payload"]
        self.assertEqual(p["row_kind"], "remediation")
        self.assertEqual((p["language"], p["path"], p["start_line"], p["end_line"]), ("python", "app/users.py", 10, 12))
        self.assertEqual(p["finding"]["label"], "injection.query")
        self.assertNotIn("passages", p)
        rj = p["remediation_json"]
        self.assertEqual(set(rj), {"patch", "rationale", "citations"})
        from app.services.unified_diff import apply_patch_to_chunk

        self.assertEqual(apply_patch_to_chunk(BAD, 10, rj["patch"], path="app/users.py").patched_code, GOOD)
        self.assertTrue(out[0]["synth_source"].startswith("playbook:code-remediation:remediation_from_findings"))

    def test_fenced_patch_is_normalised(self):
        out = self.validate(line(patch="Here you go:\n```diff\n" + diff() + "\n```"))
        self.assertEqual(reasons(self.ctx), [])
        self.assertNotIn("```", out[0]["payload"]["remediation_json"]["patch"])

    def test_every_reject_reason(self):
        cases = {
            "parse_error": [line(task=9), line(rationale=""), line(patch=5), line(citations="P1"), "x", {"patch": "p"}],
            "empty_patch": [line(patch="  \n")],
            "patch_does_not_apply": [line(patch=diff(BAD.replace("find", "other"), GOOD.replace("find", "other")))],
            "no_op_patch": [line(patch=NO_OP)],
            "bad_citation": [line(citations=["P1"]), line(citations=["nope"])],
            "code_too_long": [line(patch=diff() + "\n" + "+x = 1\n" * 6000)],
            "code_does_not_parse": [line(patch=diff(BAD, GOOD.replace("db.run(q, [name])", "db.run(q, [name]")))],
        }
        for reason, rows in cases.items():
            self.ctx["rejects"].clear()
            out = self.validate(*rows)
            self.assertEqual(out, [], reason)
            self.assertEqual(set(reasons(self.ctx)), {reason}, (reason, self.ctx["rejects"]))
            self.assertEqual(len(self.ctx["rejects"]), len(rows), reason)
            self.assertTrue(all("detail" in r and "payload" in r for r in self.ctx["rejects"]))

    def test_patch_does_not_apply_detail_is_the_applier_reason(self):
        self.validate(line(patch=diff(BAD.replace("find", "other"), GOOD.replace("find", "other"))))
        self.assertEqual(self.ctx["rejects"][0]["detail"], "context_mismatch")

    def test_unknown_label_when_the_project_has_a_taxonomy(self):
        ctx = make_ctx([findings_row(findings_json={"findings": [{**FINDING, "label": "made.up"}]})])
        self.assertEqual(self.validate(line(), ctx=ctx), [])
        self.assertEqual(reasons(ctx), ["unknown_label"])
        free = make_ctx([findings_row(findings_json={"findings": [{**FINDING, "label": "made.up"}]})], taxonomy=False)
        self.assertEqual(len(self.validate(line(), ctx=free)), 1)

    def test_code_does_not_parse_only_when_the_original_is_checkable(self):
        broken = "def find(db, name):\n    q = ??\n    return db.run(q)"
        fixed = "def find(db, name):\n    q = 1 +\n    return db.run(q)"
        ctx = make_ctx([findings_row(code=broken)])
        # original does not parse -> not checkable -> an equally broken patch is accepted
        out = self.validate(line(patch=diff(broken, fixed)), ctx=ctx)
        self.assertEqual((len(out), reasons(ctx)), (1, []))
        # original parses, patched does not -> rejected
        bad = self.validate(line(patch=diff(BAD, BAD.replace("db.run(q)", "db.run(q"))))
        self.assertEqual(bad, [])
        self.assertEqual(reasons(self.ctx), ["code_does_not_parse"])

    def test_passages_are_attached_and_citations_checked(self):
        row = findings_row()
        row["passages"] = PASSAGES
        ctx = make_ctx([row])
        prompt = self.pb.build_prompt(ctx)
        self.assertIn("[P1]", prompt)
        self.assertIn("never concatenate input", prompt)
        out = self.validate(line(citations=["P2"]), line(citations=["P3"]), ctx=ctx)
        self.assertEqual(len(out), 1)
        self.assertEqual(reasons(ctx), ["bad_citation"])
        payload = out[0]["payload"]
        self.assertEqual([p["id"] for p in payload["passages"]], ["P1", "P2"])
        self.assertEqual(payload["remediation_json"]["citations"], ["P2"])

    def test_fix_pair_uses_the_real_diff_and_ignores_the_teacher_patch(self):
        ctx = make_ctx([fix_pair_row()])
        prompt = self.pb.build_prompt(ctx)
        self.assertIn('fixed_patch="yes"', prompt)
        self.assertIn("@@", prompt)
        out = self.validate({"task": 1, "patch": "garbage", "rationale": "Uses a bound parameter.", "citations": []}, ctx=ctx)
        self.assertEqual(reasons(ctx), [])
        self.assertEqual(out[0]["payload"]["remediation_json"]["patch"], diff())
        self.assertEqual(out[0]["payload"]["code"], BAD)

    def test_prompt_uses_the_shared_builder_and_marker(self):
        prompt = self.pb.build_prompt(self.ctx)
        self.assertIn("[brewslm:code-remediation:remediation_from_findings]", prompt)
        self.assertIn("   11 |     q = ", prompt)
        self.assertIn("injection.query", prompt)

    def test_no_seeds_is_a_clear_error(self):
        ctx = make_ctx([])
        with self.assertRaisesRegex(ValueError, "No findings"):
            self.pb.build_prompt(ctx)

    def test_parse_output_records_unparseable_reply(self):
        self.assertEqual(self.pb.parse_output("I cannot do that.", self.ctx), [])
        self.assertEqual(reasons(self.ctx), ["parse_error"])


CHUNKS = [
    {"text": "def connect(url):\n    return Client(url)", "path": "lib/net.py", "language": "python", "start_line": 1,
     "end_line": 2, "source_document_id": 4, "chunk_id": 0, "source_doc": "net.py"},
    {"text": "def send(client, body):\n    return client.post(body)", "path": "lib/net.py", "language": "python",
     "start_line": 4, "end_line": 5, "source_document_id": 4, "chunk_id": 1, "source_doc": "net.py"},
    {"text": "def other():\n    return 1", "path": "lib/other.py", "language": "python", "start_line": 1, "end_line": 2,
     "source_document_id": 5, "chunk_id": 0},
]


def ex_ctx(**params):
    return {"recipe_id": "code-remediation", "project_id": 1, "gold_rows": [], "raw_rows": None, "failure_cluster": None,
            "target_class": None, "target_count": 4, "params": params, "rejects": [], "document_chunks": CHUNKS,
            "taxonomy": None}


def ex_line(**over):
    row = {"seed": 1, "task": "Open a connection and send a message", "citations": ["P1"],
           "code": "c = connect(URL)\nsend(c, 'hello')\n", "rationale": "Uses connect then send."}
    row.update(over)
    return row


class CodeExamplesTests(unittest.TestCase):
    def setUp(self):
        self.pb = get_playbook("code-remediation", SynthMode.CODE_EXAMPLES)
        self.ctx = ex_ctx()

    def test_good_row_shape_and_passages(self):
        out = self.pb.validate([ex_line()], self.ctx)
        self.assertEqual(reasons(self.ctx), [])
        p = out[0]["payload"]
        self.assertEqual(p["row_kind"], "code_example")
        self.assertEqual(p["language"], "python")
        self.assertEqual([x["id"] for x in p["passages"]], ["P1", "P2"])  # the chunk + its neighbour (same document)
        self.assertEqual(p["passages"][0]["passage_id"], "4:0")
        self.assertEqual(set(p["code_example_json"]), {"code", "rationale", "citations"})

    def test_every_reject_reason(self):
        copied = CHUNKS[0]["text"]
        cases = {
            "parse_error": [ex_line(seed=7), ex_line(task=""), ex_line(code=3), "x"],
            "empty_code": [ex_line(code=" \n")],
            "code_too_long": [ex_line(code="x = 1\n" * 5000)],
            "bad_citation": [ex_line(citations=["P9"]), ex_line(citations=[])],
            "copied_chunk": [ex_line(code=copied), ex_line(code=copied + "\n# note")],
            "code_does_not_parse": [ex_line(code="def f(:\n    pass")],
        }
        for reason, rows in cases.items():
            self.ctx["rejects"].clear()
            self.assertEqual(self.pb.validate(rows, self.ctx), [], reason)
            self.assertEqual(set(reasons(self.ctx)), {reason}, (reason, self.ctx["rejects"]))
            self.assertEqual(len(self.ctx["rejects"]), len(rows), reason)

    def test_unchecked_language_skips_the_parse_check(self):
        chunks = [{**CHUNKS[0], "language": "cobol"}]
        ctx = {**ex_ctx(), "document_chunks": chunks}
        out = self.pb.validate([ex_line(code="def f(:\n    pass")], ctx)
        self.assertEqual((len(out), reasons(ctx)), (1, []))

    def test_per_chunk_param_and_prompt(self):
        prompt = self.pb.build_prompt(ex_ctx(per_chunk=3))
        self.assertIn("[brewslm:code-remediation:code_examples]", prompt)
        self.assertIn('per_chunk="3"', prompt)
        self.assertIn("[P1] lib/net.py:1-2", prompt)
        self.assertIn("Output exactly 6 JSON lines", prompt)  # target 4 -> 2 chunks x 3 tasks
        self.assertIn('per_chunk="5"', self.pb.build_prompt(ex_ctx(per_chunk=99)))
        self.assertIn('per_chunk="2"', self.pb.build_prompt(ex_ctx()))

    def test_no_code_chunks_is_a_clear_error(self):
        ctx = {**ex_ctx(), "document_chunks": []}
        with self.assertRaisesRegex(ValueError, "No code chunks"):
            self.pb.build_prompt(ctx)


class TemplateBackendBranchTests(unittest.TestCase):
    def complete(self, prompt):
        return asyncio.run(TemplateBackend().complete(prompt))

    def test_remediation_round_trips(self):
        pb = get_playbook("code-remediation", SynthMode.REMEDIATION_FROM_FINDINGS)
        row = findings_row()
        row["passages"] = PASSAGES
        ctx = make_ctx([row, fix_pair_row(), findings_row(code="x = 1\ny = 2\nz = 3", start_line=1, language="shell",
                                                           findings_json={"findings": [{**FINDING, "start_line": 2, "end_line": 2}]})],
                       target_count=3)
        out = pb.validate(pb.parse_output(self.complete(pb.build_prompt(ctx)), ctx), ctx)
        self.assertEqual(reasons(ctx), [], ctx["rejects"])
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["payload"]["remediation_json"]["citations"], ["P1"])

    def test_code_examples_round_trips(self):
        pb = get_playbook("code-remediation", SynthMode.CODE_EXAMPLES)
        ctx = {**ex_ctx(per_chunk=2), "target_count": 6}
        out = pb.validate(pb.parse_output(self.complete(pb.build_prompt(ctx)), ctx), ctx)
        self.assertEqual(reasons(ctx), [], ctx["rejects"])
        self.assertEqual(len(out), 6)  # 3 chunks x 2
        self.assertEqual(len({r["payload"]["task"] for r in out}), 6)


class PlaybookApiTests(unittest.TestCase):
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
        r = client.post("/api/projects", json={"name": f"cr-synth-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        self._set_recipe()
        put = client.put(f"/api/projects/{self.pid}/label-taxonomy", json={"entries": TAXONOMY["entries"]})
        self.assertEqual(put.status_code, 200, put.text)
        self.base = f"/api/projects/{self.pid}/synthetic/run-playbook"
        self.pdir = Path(settings.DATA_DIR) / "projects" / str(self.pid)

    def _set_recipe(self):
        resp = client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "code-remediation"})
        if resp.status_code == 200:
            return
        from app.database import async_session_factory
        from app.models.project import Project

        async def go():
            async with async_session_factory() as db:
                project = await db.get(Project, self.pid)
                project.selected_recipe = {"recipe_id": "code-remediation"}
                await db.commit()

        asyncio.run(go())

    def _add_dataset(self, dataset_type, name, rows):
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

    def _run(self, mode, params=None, count=2, **kw):
        return client.post(self.base, json={"mode": mode, "target_count": count, "params": params, **kw})

    def _write_cleaned(self, rows):
        cleaned = self.pdir / "cleaned" / "cleaned.jsonl"
        cleaned.parent.mkdir(parents=True, exist_ok=True)
        cleaned.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def _spy(self, seen):
        async def spy(self_, prompt, **kw):
            seen.append(prompt)
            from app.services.synth_backends.template import code_remediation_rows

            return code_remediation_rows(prompt, "remediation_from_findings")

        return patch.object(TemplateBackend, "complete", spy)

    def test_remediation_from_a_findings_dataset_sync(self):
        ds = self._add_dataset("raw", "findings", [findings_row(), fix_pair_row()])
        resp = self._run("remediation_from_findings", {"findings_dataset_id": ds}, count=2)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(len(body["rows"]), 2, body)
        self.assertEqual(body["rejected_count"], 0)
        self.assertEqual({r["payload"]["row_kind"] for r in body["rows"]}, {"remediation"})
        saved = [json.loads(x) for x in (self.pdir / "synthetic" / "synthetic.jsonl").read_text().splitlines() if x.strip()]
        self.assertEqual(len(saved), 2)

    def test_remediation_async_job_carries_reject_fields(self):
        ds = self._add_dataset("raw", "findings", [findings_row()])
        resp = client.post(self.base + "?async_job=true", json={
            "mode": "remediation_from_findings", "target_count": 1, "params": {"findings_dataset_id": ds}})
        self.assertEqual(resp.status_code, 202, resp.text)
        job_id = resp.json()["id"]
        wait_for_jobs_idle(client)
        job = client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(job["status"], "succeeded", job)
        self.assertIn("reject_summary", job["result"])

    def test_gold_test_is_never_a_seed_gold_dev_is(self):
        self._add_dataset("gold_test", "gt", [findings_row(code=BAD.replace("find", "held_out_test"))])
        resp = self._run("remediation_from_findings", None, count=1)
        self.assertEqual(resp.status_code, 400, resp.text)  # nothing but the test split -> no seeds
        self.assertIn("No findings", resp.json()["detail"])
        self._add_dataset("gold_dev", "gd", [findings_row(code=BAD.replace("find", "dev_rows"))])
        seen: list[str] = []
        with self._spy(seen):
            resp = self._run("remediation_from_findings", None, count=1)
        self.assertEqual(resp.status_code, 200, resp.text)
        joined = "\n".join(seen)
        self.assertIn("dev_rows", joined)
        self.assertNotIn("held_out_test", joined)

    def test_answer_key_dataset_is_refused_as_findings_dataset(self):
        gold = self._add_dataset("gold_test", "gt2", [findings_row()])
        resp = self._run("remediation_from_findings", {"findings_dataset_id": gold})
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(resp.json().get("error_code"), "findings_dataset_not_allowed", resp.json())

    def test_eval_only_ledger_source_rows_are_skipped(self):
        from app.database import async_session_factory
        from app.services import provenance_service

        async def mk():
            async with async_session_factory() as db:
                src = await provenance_service.create_source(
                    db, self.pid, kind=provenance_service.KIND_UPLOAD, title="eval only", usage=provenance_service.USAGE_EVAL)
                await db.commit()
                return src.id

        sid = asyncio.run(mk())
        ds = self._add_dataset("raw", "mixed", [
            findings_row(code=BAD.replace("find", "blocked_src"), data_source_id=sid),
            findings_row(code=BAD.replace("find", "allowed_src")),
        ])
        seen: list[str] = []
        with self._spy(seen):
            resp = self._run("remediation_from_findings", {"findings_dataset_id": ds}, count=2)
        self.assertEqual(resp.status_code, 200, resp.text)
        joined = "\n".join(seen)
        self.assertIn("allowed_src", joined)
        self.assertNotIn("blocked_src", joined)

    def test_passages_are_retrieved_for_seeds_without_them(self):
        ds = self._add_dataset("raw", "findings", [findings_row()])
        seen: list[str] = []
        fake = [{"id": "P1", "text": "Bind parameters with db.run(sql, params).", "path": "docs/db.md"}]
        with self._spy(seen), patch("app.services.code_remediation_prompt.retrieve_passages_for_row", return_value=fake) as mock:
            resp = self._run("remediation_from_findings", {"findings_dataset_id": ds, "top_k": 2, "retrieve_passages": True}, count=1)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(mock.called)
        self.assertEqual(mock.call_args.args[3], 2)
        self.assertIn("Bind parameters with db.run", seen[0])
        self.assertEqual(resp.json()["rows"][0]["payload"]["passages"][0]["id"], "P1")
        self.assertEqual(resp.json()["rows"][0]["payload"]["remediation_json"]["citations"], ["P1"])
        # retrieve_passages=false skips retrieval entirely
        with patch("app.services.code_remediation_prompt.retrieve_passages_for_row", return_value=fake) as mock:
            resp = self._run("remediation_from_findings", {"findings_dataset_id": ds, "retrieve_passages": False}, count=1)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertFalse(mock.called)
        self.assertNotIn("passages", resp.json()["rows"][0]["payload"])

    def test_code_examples_from_code_documents(self):
        self._write_cleaned([{**c, "source_doc": c.get("source_doc", "x.py")} for c in CHUNKS])
        resp = self._run("code_examples", {"per_chunk": 2}, count=4)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertGreaterEqual(len(body["rows"]), 4, body)
        self.assertTrue(all(r["payload"]["row_kind"] == "code_example" for r in body["rows"]))
        self.assertIn("reject_summary", body)

    def test_code_examples_every_seed_skipped_raises_the_diagnostic(self):
        self._write_cleaned([{**c, "source_doc": c.get("source_doc", "x.py")} for c in CHUNKS])

        async def skip_all(self_, prompt, **kw):
            import re as _re

            idx = _re.findall(r'<seed index="(\d+)"', prompt)
            return "\n".join(json.dumps({"seed": int(i), "skip": True, "reason": "nothing to teach"}) for i in idx)

        with patch.object(TemplateBackend, "complete", skip_all):
            resp = self._run("code_examples", {"per_chunk": 1}, count=2)
        self.assertGreaterEqual(resp.status_code, 400, resp.text)
        self.assertIn("skipped", resp.text)
        self.assertIn("nothing to teach", resp.text)

    def test_code_examples_skip_eval_only_documents(self):
        from app.services import provenance_service

        self._write_cleaned([{**CHUNKS[2], "text": "def only_doc():\n    return 'blocked_doc'", "source_doc": "x.py"}])
        with patch.object(provenance_service, "training_excluded_document_ids", new=AsyncMock(return_value={5})):
            resp = self._run("code_examples", None, count=2)
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("No code chunks", resp.json()["detail"])

    def test_param_validation(self):
        for mode, params in [("code_examples", {"per_chunk": 9}), ("code_examples", {"per_chunk": "x"}),
                             ("remediation_from_findings", {"top_k": 0}),
                             ("remediation_from_findings", {"findings_dataset_id": "abc"}),
                             ("remediation_from_findings", {"retrieve_passages": "yes"})]:
            resp = self._run(mode, params)
            self.assertEqual(resp.status_code, 400, (mode, params, resp.text))
            self.assertEqual(resp.json().get("error_code"), "invalid_playbook_param", resp.json())
        self.assertEqual(self._run("not_a_mode").status_code, 400)

    def test_catalog_lists_both(self):
        modes = {p["mode"] for p in __import__("app.services.synth_playbooks", fromlist=["x"]).list_playbooks()
                 if p["recipe_id"] == "code-remediation"}
        self.assertEqual(modes, {"remediation_from_findings", "code_examples"})


if __name__ == "__main__":
    unittest.main()
