"""`code-findings` synth playbooks: seeded_defect, hard_negatives, cluster_targeted.

Unit: each validator on crafted good / bad teacher outputs (every reject reason),
the reject aggregation, and the template backend's deterministic branch.
API: all three modes through /run-playbook (sync and ?async_job=true) with the
template backend stubbed into both registries, a data-carrying context column,
the unknown-label 400 and the no-clean-chunks diagnostic.

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
from app.services import label_taxonomy_service as lts
from app.services.synth_backends import TemplateBackend
from app.services.synth_playbooks import SynthMode, get_playbook
from app.services.synth_playbooks.base import reject_report

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
        {"id": "injection.query", "title": "Query built from input", "parent": "injection", "aliases": ["sqli-like"],
         "attrs": {"sink": "database"}},
        {"id": "secret.hardcoded", "title": "Hard-coded secret"},
        {"id": "logging.sensitive", "title": "Sensitive value logged"},
    ]
}

CLEAN = 'def get_user(db, user_id):\n    return db.run("SELECT * FROM users WHERE id = ?", [user_id])'
BAD_LINE = '    q = "SELECT * FROM users WHERE name = " + name'
SEEDED_CODE = CLEAN + "\n" + BAD_LINE  # lines 10..12
POS = 'def log_login(user, password):\n    logger.info("login " + password)\n    return True'


def index():
    return lts.TaxonomyIndex(lts.parse_taxonomy(TAXONOMY))


def clean_row(**kw):
    row = {"row_kind": "findings", "code": CLEAN, "language": "python", "path": "app/users.py", "start_line": 10,
           "end_line": 11, "findings_json": {"findings": []}, "labels_raw": [], "taxonomy_version": None}
    row.update(kw)
    return row


def positive_row():
    return {"row_kind": "findings", "code": POS, "language": "python", "path": "app/auth.py", "start_line": 40,
            "end_line": 42,
            "findings_json": {"findings": [{"label": "logging.sensitive", "start_line": 41, "end_line": 41,
                                             "evidence": 'logger.info("login " + password)', "attrs": {}}]},
            "labels_raw": ["logging.sensitive"], "taxonomy_version": None}


def make_ctx(mode, *, gold=None, chunks=None, params=None, taxonomy=True, **extra):
    ctx = {
        "recipe_id": "code-findings", "project_id": 1, "gold_rows": gold if gold is not None else [clean_row(), positive_row()],
        "raw_rows": None, "failure_cluster": None, "target_class": None, "target_count": 2,
        "params": params or {}, "rejects": [], "document_chunks": chunks or [],
        "taxonomy": index() if taxonomy else None,
    }
    ctx.update(extra)
    return ctx


def seeded_line(**over):
    row = {"chunk": 1, "code": SEEDED_CODE,
           "finding": {"label": "injection.query", "start_line": 12, "end_line": 12, "evidence": BAD_LINE.strip(), "attrs": {}}}
    finding_over = over.pop("finding", {})
    row.update(over)
    row["finding"] = {**row["finding"], **finding_over} if isinstance(finding_over, dict) else finding_over
    return row


def reasons(ctx):
    return [r["reason"] for r in ctx["rejects"]]


class SeededDefectValidatorTests(unittest.TestCase):
    def setUp(self):
        self.pb = get_playbook("code-findings", SynthMode.SEEDED_DEFECT)
        self.ctx = make_ctx(SynthMode.SEEDED_DEFECT, params={"label_id": "injection.query"})

    def validate(self, *rows):
        return self.pb.validate(list(rows), self.ctx)

    def test_registered_with_new_mode(self):
        self.assertIsNotNone(self.pb)
        self.assertEqual(SynthMode("seeded_defect"), SynthMode.SEEDED_DEFECT)

    def test_good_row_is_a_canonical_findings_row(self):
        out = self.validate(seeded_line())
        self.assertEqual(reasons(self.ctx), [])
        self.assertEqual(len(out), 1)
        p = out[0]["payload"]
        self.assertEqual(p["row_kind"], "findings")
        self.assertEqual((p["language"], p["path"], p["start_line"], p["end_line"]), ("python", "app/users.py", 10, 12))
        self.assertEqual(p["labels_raw"], ["injection.query"])
        self.assertEqual(p["taxonomy_version"], index().version)
        f = p["findings_json"]["findings"][0]
        self.assertEqual((f["label"], f["start_line"], f["end_line"]), ("injection.query", 12, 12))
        self.assertEqual(out[0]["synth_source"], "playbook:code-findings:seeded_defect:label=injection.query")

    def test_alias_label_resolves_to_the_taxonomy_id(self):
        out = self.validate(seeded_line(finding={"label": "sqli-like"}))
        self.assertEqual(out[0]["payload"]["findings_json"]["findings"][0]["label"], "injection.query")

    def test_evidence_matches_ignoring_whitespace(self):
        out = self.validate(seeded_line(finding={"evidence": 'q  =  "SELECT * FROM users WHERE name = "   + name'}))
        self.assertEqual(len(out), 1)

    def test_every_reject_reason(self):
        cases = {
            "parse_error": [seeded_line(finding="oops"), {"code": 5}, "not a dict", seeded_line(chunk=9)],
            "empty_code": [seeded_line(code="  \n")],
            "unchanged_code": [seeded_line(code=CLEAN)],
            "label_mismatch": [seeded_line(finding={"label": "secret.hardcoded"}), seeded_line(finding={"label": "nope"})],
            "line_out_of_range": [seeded_line(finding={"start_line": 13, "end_line": 13}),
                                  seeded_line(finding={"start_line": 9, "end_line": 12}),
                                  seeded_line(finding={"start_line": 12, "end_line": 11}),
                                  seeded_line(finding={"start_line": "12"})],
            "evidence_not_in_lines": [seeded_line(finding={"evidence": "something absent"}),
                                      seeded_line(finding={"start_line": 10, "end_line": 10}),
                                      seeded_line(finding={"evidence": ""})],
            "code_too_long": [seeded_line(code=SEEDED_CODE + "\n" + "x = 1\n" * 4000)],
        }
        for reason, rows in cases.items():
            self.ctx["rejects"].clear()
            out = self.validate(*rows)
            self.assertEqual(out, [], reason)
            self.assertEqual(set(reasons(self.ctx)), {reason}, (reason, reasons(self.ctx)))
            self.assertEqual(len(self.ctx["rejects"]), len(rows), reason)
            self.assertTrue(all("detail" in r and "payload" in r for r in self.ctx["rejects"]))

    def test_parse_output_records_unparseable_reply(self):
        self.assertEqual(self.pb.parse_output("I cannot do that.", self.ctx), [])
        self.assertEqual(reasons(self.ctx), ["parse_error"])
        good = json.dumps(seeded_line())
        self.assertEqual(len(self.pb.parse_output(f"```json\n{good}\n```", self.ctx)), 1)

    def test_prompt_numbers_lines_and_names_the_label(self):
        ctx = make_ctx(SynthMode.SEEDED_DEFECT, params={"label_id": "injection.query"},
                       context_values=["customer_id", "9981"], context_column="field")
        prompt = self.pb.build_prompt(ctx)
        self.assertIn("[brewslm:code-findings:seeded_defect]", prompt)
        self.assertIn("   10 | def get_user(db, user_id):", prompt)
        self.assertIn("   11 |     return db.run", prompt)
        self.assertIn("Target label id: injection.query", prompt)
        self.assertIn("Query built from input", prompt)
        self.assertIn('"sink": "database"', prompt)
        self.assertIn("customer_id", prompt)
        self.assertIn("9981", prompt)
        self.assertNotIn("def log_login", prompt)  # positives are never seeds for a seeded defect

    def test_attrs_param_overrides_taxonomy_attrs(self):
        ctx = make_ctx(SynthMode.SEEDED_DEFECT, params={"label_id": "injection.query", "attrs": {"sink": "shell"}})
        self.assertIn('"sink": "shell"', self.pb.build_prompt(ctx))
        out = self.pb.validate([seeded_line()], ctx)
        self.assertEqual(out[0]["payload"]["findings_json"]["findings"][0]["attrs"], {"sink": "shell"})

    def test_document_chunks_are_assumed_clean_seeds(self):
        chunk = {"text": "int f(int x) {\n  return x;\n}", "path": "src/f.c", "language": "c", "start_line": 100, "end_line": 102}
        ctx = make_ctx(SynthMode.SEEDED_DEFECT, gold=[], chunks=[chunk], params={"label_id": "secret.hardcoded"})
        prompt = self.pb.build_prompt(ctx)
        self.assertIn("assumed clean", prompt)
        self.assertIn("  100 | int f(int x) {", prompt)
        code = chunk["text"] + '\n  char *key = "abc123";'
        out = self.pb.validate([{"chunk": 1, "code": code, "finding": {
            "label": "secret.hardcoded", "start_line": 103, "end_line": 103, "evidence": 'char *key = "abc123";', "attrs": {}}}], ctx)
        self.assertEqual(reasons(ctx), [])
        self.assertEqual((out[0]["payload"]["path"], out[0]["payload"]["start_line"], out[0]["payload"]["end_line"]), ("src/f.c", 100, 103))

    def test_no_clean_chunks_is_a_clear_error(self):
        ctx = make_ctx(SynthMode.SEEDED_DEFECT, gold=[positive_row()], params={"label_id": "injection.query"})
        with self.assertRaisesRegex(ValueError, "No clean code chunks"):
            self.pb.build_prompt(ctx)

    def test_label_must_resolve(self):
        with self.assertRaisesRegex(ValueError, "not in the project taxonomy"):
            self.pb.build_prompt(make_ctx(SynthMode.SEEDED_DEFECT, params={"label_id": "nope"}))
        with self.assertRaisesRegex(ValueError, "label_id"):
            self.pb.build_prompt(make_ctx(SynthMode.SEEDED_DEFECT))

    def test_response_schema_pins_the_label(self):
        schema = self.pb.response_schema(self.ctx)
        self.assertEqual(schema["properties"]["finding"]["properties"]["label"]["enum"], ["injection.query"])


class HardNegativeValidatorTests(unittest.TestCase):
    def setUp(self):
        self.pb = get_playbook("code-findings", SynthMode.HARD_NEGATIVES)
        self.ctx = make_ctx(SynthMode.HARD_NEGATIVES)
        self.fixed = POS.replace('"login " + password', '"login"')

    def row(self, **over):
        row = {"chunk": 1, "negative_kind": "fixed", "code": self.fixed, "findings": []}
        row.update(over)
        return row

    def test_good_row(self):
        out = self.pb.validate([self.row()], self.ctx)
        self.assertEqual(reasons(self.ctx), [])
        p = out[0]["payload"]
        self.assertEqual((p["row_kind"], p["path"], p["start_line"], p["end_line"]), ("findings", "app/auth.py", 40, 42))
        self.assertEqual(p["findings_json"], {"findings": []})
        self.assertEqual((p["labels_raw"], p["negative_kind"]), ([], "fixed"))
        self.assertEqual(out[0]["synth_source"], "playbook:code-findings:hard_negatives")

    def test_unknown_kind_is_recorded_as_unspecified(self):
        out = self.pb.validate([self.row(negative_kind="weird")], self.ctx)
        self.assertEqual(out[0]["payload"]["negative_kind"], "unspecified")

    def test_every_reject_reason(self):
        cases = {
            "parse_error": [{"chunk": 1, "code": 4}, "x", self.row(chunk=7)],
            "empty_code": [self.row(code="")],
            "has_findings": [self.row(findings=[{"label": "logging.sensitive"}]), self.row(findings="none")],
            "code_too_long": [self.row(code=self.fixed + "\n" + "y = 2\n" * 4000)],
            "unchanged_code": [self.row(code=POS)],
        }
        for reason, rows in cases.items():
            self.ctx["rejects"].clear()
            self.assertEqual(self.pb.validate(rows, self.ctx), [], reason)
            self.assertEqual(set(reasons(self.ctx)), {reason}, (reason, reasons(self.ctx)))

    def test_prompt_uses_positive_seeds_with_their_findings(self):
        prompt = self.pb.build_prompt(self.ctx)
        self.assertIn("[brewslm:code-findings:hard_negatives]", prompt)
        self.assertIn("   41 |     logger.info", prompt)
        self.assertIn("logging.sensitive at lines 41-41", prompt)
        self.assertNotIn("get_user", prompt)

    def test_needs_positive_chunks(self):
        ctx = make_ctx(SynthMode.HARD_NEGATIVES, gold=[clean_row()])
        with self.assertRaisesRegex(ValueError, "labelled code chunks"):
            self.pb.build_prompt(ctx)


class ClusterTargetedValidatorTests(unittest.TestCase):
    def setUp(self):
        self.pb = get_playbook("code-findings", SynthMode.CLUSTER_TARGETED)
        self.cluster = {"cluster_id": "cluster-3", "reason_code": "missed_finding", "label": "injection.query",
                        "exemplars": [{"prompt": "q = a + b", "reference": "injection.query", "prediction": "none"}]}
        self.ctx = make_ctx(SynthMode.CLUSTER_TARGETED, failure_cluster=self.cluster)

    def test_good_row_and_provenance(self):
        out = self.pb.validate([seeded_line()], self.ctx)
        self.assertEqual(reasons(self.ctx), [])
        self.assertEqual(out[0]["synth_source"], "playbook:code-findings:cluster_targeted:cluster-3")
        self.assertEqual(out[0]["payload"]["row_kind"], "findings")

    def test_every_reject_reason(self):
        cases = {
            "unknown_label": [seeded_line(finding={"label": "totally.unknown"})],
            "label_mismatch": [seeded_line(finding={"label": "secret.hardcoded"})],
            "parse_error": [seeded_line(finding="x")],
            "empty_code": [seeded_line(code="")],
            "unchanged_code": [seeded_line(code=CLEAN)],
            "line_out_of_range": [seeded_line(finding={"start_line": 99, "end_line": 99})],
            "evidence_not_in_lines": [seeded_line(finding={"evidence": "absent"})],
            "code_too_long": [seeded_line(code=SEEDED_CODE + "\n" + "z = 3\n" * 4000)],
        }
        for reason, rows in cases.items():
            self.ctx["rejects"].clear()
            self.assertEqual(self.pb.validate(rows, self.ctx), [], reason)
            self.assertEqual(set(reasons(self.ctx)), {reason}, (reason, reasons(self.ctx)))

    def test_unpinned_cluster_accepts_any_taxonomy_label(self):
        ctx = make_ctx(SynthMode.CLUSTER_TARGETED, failure_cluster={"cluster_id": "c9", "reason_code": "x"})
        out = self.pb.validate([seeded_line(finding={"label": "secret.hardcoded"})], ctx)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["payload"]["findings_json"]["findings"][0]["label"], "secret.hardcoded")

    def test_prompt_embeds_the_cluster(self):
        prompt = self.pb.build_prompt(self.ctx)
        self.assertIn("[brewslm:code-findings:cluster_targeted]", prompt)
        self.assertIn("Cluster reason: missed_finding", prompt)
        self.assertIn("Target label id: injection.query", prompt)

    def test_pinned_label_outside_taxonomy_raises(self):
        ctx = make_ctx(SynthMode.CLUSTER_TARGETED, params={"label_id": "nope"})
        with self.assertRaisesRegex(ValueError, "not in the project taxonomy"):
            self.pb.build_prompt(ctx)


class RejectReportTests(unittest.TestCase):
    def test_summary_and_cap(self):
        rejects = [{"reason": "a", "detail": "", "payload": i} for i in range(60)] + [{"reason": "b", "detail": "", "payload": 0}]
        report = reject_report(rejects)
        self.assertEqual(report["rejected_count"], 61)
        self.assertEqual(report["reject_summary"], {"a": 60, "b": 1})
        self.assertEqual(len(report["rejected_examples"]), 50)
        self.assertEqual(reject_report(None), {"rejected_count": 0, "reject_summary": {}, "rejected_examples": []})


class TemplateBackendBranchTests(unittest.TestCase):
    """The deterministic backend answers every code-findings prompt with rows its own validator accepts."""

    def complete(self, prompt):
        return asyncio.run(TemplateBackend().complete(prompt))

    def test_each_mode_round_trips(self):
        for mode, params, ctx_kw in [
            (SynthMode.SEEDED_DEFECT, {"label_id": "injection.query"}, {}),
            (SynthMode.CLUSTER_TARGETED, {"label_id": "injection.query"}, {"failure_cluster": {"cluster_id": "c1"}}),
            (SynthMode.CLUSTER_TARGETED, {}, {}),
            (SynthMode.HARD_NEGATIVES, {}, {}),
        ]:
            pb = get_playbook("code-findings", mode)
            ctx = make_ctx(mode, params=params, target_count=4, **ctx_kw)
            raw = self.complete(pb.build_prompt(ctx))
            out = pb.validate(pb.parse_output(raw, ctx), ctx)
            self.assertEqual(reasons(ctx), [], (mode, ctx["rejects"]))
            self.assertEqual(len(out), 4, mode)
            self.assertEqual(len({r["payload"]["code"] for r in out}), 4, mode)
            if mode == SynthMode.HARD_NEGATIVES:
                self.assertTrue(all(r["payload"]["findings_json"]["findings"] == [] for r in out))
                self.assertEqual({r["payload"]["negative_kind"] for r in out}, {"fixed", "safe_lookalike", "test_fixture"})
            else:
                self.assertTrue(all(len(r["payload"]["findings_json"]["findings"]) == 1 for r in out))

    def test_still_rejects_unrelated_prompts(self):
        from app.services.synth_backends.base import SynthBackendError

        with self.assertRaises(SynthBackendError):
            self.complete("write me a poem")


# ── API ──────────────────────────────────────────────────────────────────


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
        r = client.post("/api/projects", json={"name": f"cf-synth-{time.time_ns()}", "description": "x"})
        self.assertEqual(r.status_code, 201, r.text)
        self.pid = r.json()["id"]
        self._set_recipe()
        put = client.put(f"/api/projects/{self.pid}/label-taxonomy", json={"entries": TAXONOMY["entries"]})
        self.assertEqual(put.status_code, 200, put.text)
        self.base = f"/api/projects/{self.pid}/synthetic/run-playbook"
        self.pdir = Path(settings.DATA_DIR) / "projects" / str(self.pid)

    def _set_recipe(self):
        resp = client.put(f"/api/projects/{self.pid}/recipe", json={"recipe_id": "code-findings"})
        if resp.status_code == 200:
            return
        from app.database import async_session_factory
        from app.models.project import Project

        async def go():
            async with async_session_factory() as db:
                project = await db.get(Project, self.pid)
                project.selected_recipe = {"recipe_id": "code-findings"}
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

    def _seed_gold(self, rows=None):
        return self._add_dataset("gold_dev", "gold", rows if rows is not None else [clean_row(), positive_row()])

    def _synthetic_rows(self):
        path = self.pdir / "synthetic" / "synthetic.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

    def _run(self, mode, params=None, count=3, **kw):
        return client.post(self.base, json={"mode": mode, "target_count": count, "params": params, **kw})

    def _job(self, resp):
        self.assertEqual(resp.status_code, 202, resp.text)
        job_id = resp.json()["id"]
        wait_for_jobs_idle(client)
        job = client.get(f"/api/jobs/{job_id}").json()
        return job

    # sync
    def test_all_three_modes_sync(self):
        self._seed_gold()
        for mode, params, tag in [
            ("seeded_defect", {"label_id": "injection.query"}, "playbook:code-findings:seeded_defect:label=injection.query"),
            ("hard_negatives", None, "playbook:code-findings:hard_negatives"),
            ("cluster_targeted", {"label_id": "secret.hardcoded"}, "playbook:code-findings:cluster_targeted:noctx"),
        ]:
            resp = self._run(mode, params, count=3)
            self.assertEqual(resp.status_code, 200, (mode, resp.text))
            body = resp.json()
            self.assertEqual(len(body["rows"]), 3, mode)
            self.assertEqual((body["rejected_count"], body["reject_summary"], body["rejected_examples"]), (0, {}, []))
            self.assertTrue(all(r["synth_source"] == tag for r in body["rows"]), mode)
            self.assertEqual(body["backend_used"], "template (deterministic smoke backend)")
        saved = self._synthetic_rows()
        self.assertEqual(len(saved), 9)
        for row in saved:  # the nested findings payload survives the synthetic.jsonl round trip
            self.assertIsInstance(row["findings_json"], dict)
            self.assertIsInstance(row["findings_json"]["findings"], list)
            self.assertEqual(row["row_kind"], "findings")
            self.assertEqual(row["review_status"], "pending")
        seeded = [r for r in saved if r["synth_source"].startswith("playbook:code-findings:seeded_defect")]
        self.assertEqual(seeded[0]["findings_json"]["findings"][0]["label"], "injection.query")

    def test_alias_label_is_canonicalised_by_the_api(self):
        self._seed_gold()
        resp = self._run("seeded_defect", {"label_id": "sqli-like"}, count=1)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.json()["rows"][0]["synth_source"].endswith("label=injection.query"))

    # async
    def test_all_three_modes_async_job_carries_reject_fields(self):
        self._seed_gold()
        for mode, params in [("seeded_defect", {"label_id": "injection.query"}), ("hard_negatives", None),
                             ("cluster_targeted", {"label_id": "logging.sensitive"})]:
            job = self._job(client.post(
                self.base + "?async_job=true", json={"mode": mode, "target_count": 2, "params": params}))
            self.assertEqual(job["status"], "succeeded", (mode, job))
            result = job["result"]
            self.assertEqual(result["rows_generated"], 2, mode)
            self.assertEqual((result["rejected_count"], result["reject_summary"], result["rejected_examples"]), (0, {}, []))
            if params:
                self.assertEqual(job["params"]["params"], params)
        self.assertEqual(len(self._synthetic_rows()), 6)

    def test_rejected_rows_are_reported_in_the_sync_result_and_the_job(self):
        self._seed_gold()

        async def bad_complete(self, prompt, **kw):
            good = json.loads(TemplateBackend_complete_sync(prompt).splitlines()[0])
            bad_label = {**good, "finding": {**good["finding"], "label": "secret.hardcoded"}}
            bad_lines = {**good, "finding": {**good["finding"], "start_line": 500, "end_line": 500}}
            return "\n".join(json.dumps(x) for x in [good, bad_label, bad_lines, bad_lines]) + "\nnot json at all"

        def TemplateBackend_complete_sync(prompt):
            from app.services.synth_backends.template import code_findings_rows

            return code_findings_rows(prompt, "seeded_defect")

        with patch.object(TemplateBackend, "complete", bad_complete):
            resp = self._run("seeded_defect", {"label_id": "injection.query"}, count=4)
            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertEqual(len(body["rows"]), 1)
            self.assertEqual(body["rejected_count"], 3)
            self.assertEqual(body["reject_summary"], {"label_mismatch": 1, "line_out_of_range": 2})
            self.assertEqual(len(body["rejected_examples"]), 3)
            self.assertTrue(all({"reason", "detail", "payload"} <= set(e) for e in body["rejected_examples"]))

            job = self._job(client.post(self.base + "?async_job=true", json={
                "mode": "seeded_defect", "target_count": 4, "params": {"label_id": "injection.query"}}))
            self.assertEqual(job["status"], "succeeded", job)
            self.assertEqual(job["result"]["rejected_count"], 3)
            self.assertEqual(job["result"]["reject_summary"], {"label_mismatch": 1, "line_out_of_range": 2})

    def test_all_rejected_fails_the_job_with_the_reject_summary(self):
        self._seed_gold()

        async def only_bad(self, prompt, **kw):
            return json.dumps({"chunk": 1, "code": CLEAN, "finding": {"label": "injection.query", "start_line": 10,
                                                                      "end_line": 10, "evidence": "x", "attrs": {}}})

        with patch.object(TemplateBackend, "complete", only_bad):
            job = self._job(client.post(self.base + "?async_job=true", json={
                "mode": "seeded_defect", "target_count": 2, "params": {"label_id": "injection.query"}}))
        self.assertEqual(job["status"], "failed", job)
        self.assertIn("unchanged_code", json.dumps(job))

    # context dataset
    def test_context_dataset_column_values_reach_the_prompt(self):
        self._seed_gold()
        ctx_id = self._add_dataset("raw", "fields", [{"field": "customer_id"}, {"field": "customer_id"}, {"field": "iban_value"},
                                                      {"field": ""}, {"other": 1}])
        seen = {}

        async def spy(self, prompt, **kw):
            seen["prompt"] = prompt
            from app.services.synth_backends.template import code_findings_rows

            return code_findings_rows(prompt, "seeded_defect")

        with patch.object(TemplateBackend, "complete", spy):
            resp = self._run("seeded_defect", {"label_id": "injection.query", "context_dataset_id": ctx_id,
                                               "context_column": "field", "context_sample_size": 5}, count=2)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIn("customer_id", seen["prompt"])
        self.assertIn("iban_value", seen["prompt"])
        self.assertEqual(seen["prompt"].count("  - customer_id"), 1)  # distinct values

        for params in [{"context_dataset_id": ctx_id, "context_column": "missing"}, {"context_dataset_id": ctx_id},
                       {"context_dataset_id": 987654, "context_column": "field"}]:
            resp = self._run("seeded_defect", {"label_id": "injection.query", **params}, count=1)
            self.assertIn(resp.status_code, (400, 404), (params, resp.text))

    def test_eval_only_source_values_are_not_used_as_context(self):
        self._seed_gold()
        from app.database import async_session_factory
        from app.services import provenance_service

        async def mk():
            async with async_session_factory() as db:
                src = await provenance_service.create_source(
                    db, self.pid, kind=provenance_service.KIND_UPLOAD, title="eval only", usage=provenance_service.USAGE_EVAL)
                await db.commit()
                return src.id

        sid = asyncio.run(mk())
        ctx_id = self._add_dataset("raw", "restricted", [{"field": "secret_eval_value", "data_source_id": sid}])
        resp = self._run("seeded_defect", {"label_id": "injection.query", "context_dataset_id": ctx_id,
                                           "context_column": "field"}, count=1)
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("no usable values", resp.json()["detail"])

    # clean seeds from code documents
    def test_code_document_chunks_seed_without_an_answer_key(self):
        cleaned = self.pdir / "cleaned" / "cleaned.jsonl"
        cleaned.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"text": "def handler(req):\n    return respond(req)", "source_document_id": 1, "source_doc": "svc.py", "chunk_id": 0,
                 "path": "svc/handler.py", "language": "python", "start_line": 20, "end_line": 21},
                {"text": "plain prose paragraph", "source_document_id": 2, "source_doc": "notes.txt", "chunk_id": 0}]
        cleaned.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        resp = self._run("seeded_defect", {"label_id": "injection.query"}, count=2)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(len(body["rows"]), 2)
        self.assertTrue(all(r["payload"]["path"] == "svc/handler.py" and r["payload"]["start_line"] == 20 for r in body["rows"]))
        self.assertTrue(all(r["payload"]["end_line"] == 22 for r in body["rows"]))

    # errors
    def test_unknown_label_is_a_structured_400(self):
        self._seed_gold()
        for body in [{"mode": "seeded_defect", "target_count": 1, "params": {"label_id": "not.a.label"}},
                     {"mode": "seeded_defect", "target_count": 1, "params": {"label_id": "not.a.label"}, "async": True}]:
            url = self.base + ("?async_job=true" if body.pop("async", False) else "")
            resp = client.post(url, json=body)
            self.assertEqual(resp.status_code, 400, resp.text)
            data = resp.json()
            self.assertEqual(data["error_code"], "unknown_label")
            self.assertEqual(data["metadata"]["label_id"], "not.a.label")
            self.assertIn("injection.query", data["metadata"]["known_labels"])
            self.assertIsInstance(data["detail"], str)
        missing = client.post(self.base, json={"mode": "seeded_defect", "target_count": 1, "params": {}})
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(missing.json()["error_code"], "missing_label_id")
        dry = client.post(self.base + "/dry-run", json={"mode": "seeded_defect", "params": {"label_id": "zzz"}})
        self.assertEqual(dry.status_code, 400)
        self.assertEqual(dry.json()["error_code"], "unknown_label")
        self.assertEqual(self._synthetic_rows(), [])

    def test_no_clean_chunks_is_a_diagnostic_error(self):
        self._seed_gold([positive_row()])
        resp = self._run("seeded_defect", {"label_id": "injection.query"}, count=1)
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("No clean code chunks", resp.json()["detail"])
        job = self._job(client.post(self.base + "?async_job=true", json={
            "mode": "seeded_defect", "target_count": 1, "params": {"label_id": "injection.query"}}))
        self.assertEqual(job["status"], "failed")
        self.assertIn("No clean code chunks", json.dumps(job))

    def test_no_labelled_chunks_for_hard_negatives(self):
        self._seed_gold([clean_row()])
        resp = self._run("hard_negatives", None, count=1)
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("labelled code chunks", resp.json()["detail"])

    def test_dry_run_reports_rejects_and_persists_nothing(self):
        self._seed_gold()
        resp = client.post(self.base + "/dry-run", json={"mode": "seeded_defect", "params": {"label_id": "injection.query"}})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["ok"], body)
        self.assertEqual(body["rejected_count"], 0)
        self.assertEqual(self._synthetic_rows(), [])


if __name__ == "__main__":
    unittest.main()
