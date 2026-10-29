"""Predefined-split import on recipe projects whose prompt needs the recipe's adapter config
(``dataset_service._apply_recipe_adapter_config``): code-remediation (remediation / code_example rows) and
code-findings (finding_set). The imported rows must be prepared exactly as a random split prepares them, the
manifest must carry the same recipe stamp, and held-out eval must render each test row's training prompt.

Projects on any other recipe (or none) must import and random-split byte-identically to before the shared
helper existed (golden hashes recorded on the code before the change).

Neutral toy code only; model inference and the judge are stubbed."""

from __future__ import annotations

import asyncio
import hashlib
import json
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
from app.models.experiment import Experiment, ExperimentStatus
from app.services import answer_judge_service as ajs
from app.services import code_example_scoring as ces
from app.services.dataset_service import _prep_dir
from app.services.eval_task_handler_service import EvalContext, StructuredExtractionHandler
from app.services.unified_diff import make_unified_diff

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    from _job_wait import wait_for_jobs_idle

    wait_for_jobs_idle(client, timeout_s=60)
    client.__exit__(None, None, None)


CODE = 'def f(cur, uid):\n    q = "SELECT * FROM t WHERE id = " + uid\n    cur.execute(q)\n    return cur.fetchone()\n'
FIXED = 'def f(cur, uid):\n    q = "SELECT * FROM t WHERE id = ?"\n    cur.execute(q, (uid,))\n    return cur.fetchone()\n'
EX_REF = "def get(cur, i):\n    cur.execute('SELECT 1 WHERE id = ?', (i,))\n"


def passages(i: int) -> list[dict]:
    return [{"id": "P1", "path": f"guides/query-{i}.md", "text": f"Pass user input as bound parameters ({i}).",
             "source": "guides"}]


def code_example(i: int) -> dict:
    return {"row_kind": "code_example", "task": f"Read row {i} by id.", "language": "python", "passages": passages(i),
            "code_example_json": {"code": EX_REF + f"# variant {i}\n", "rationale": "Bound.", "citations": ["P1"]}}


def remediation(i: int) -> dict:
    start = 10 + i
    code, fixed = CODE.replace("uid", f"uid{i}"), FIXED.replace("uid", f"uid{i}")
    patch_text = make_unified_diff(code, fixed, path=f"app/db_{i}.py", start_line=start)
    return {"row_kind": "remediation", "code": code, "language": "python", "path": f"app/db_{i}.py",
            "start_line": start, "end_line": start + 3,
            "finding": {"label": "injection.query", "start_line": start + 1, "end_line": start + 2,
                        "evidence": "q = ... + uid", "attrs": {}},
            "passages": passages(i),
            "remediation_json": {"patch": patch_text, "rationale": "Bind it.", "citations": ["P1"]}}


def findings(i: int) -> dict:
    start = 10 + i
    items = [] if i % 3 == 0 else [{"label": "injection.query", "start_line": start + 1, "end_line": start + 2,
                                    "evidence": "q = ...", "attrs": {}}]
    return {"row_kind": "findings", "code": CODE + f"# chunk {i}\n", "language": "python", "path": f"app/m{i}.py",
            "start_line": start, "end_line": start + 4, "findings_json": {"findings": items}}


def jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


def preset_of(pid: int) -> dict:
    from app.models.project import Project

    async def get(db):
        return dict((await db.get(Project, pid)).dataset_adapter_preset or {})

    return run_async(get)


def run_async(fn):
    async def go():
        async with async_session_factory() as db:
            return await fn(db)

    return asyncio.run(go())


def new_project(recipe_id: str | None = None) -> int:
    pid = client.post("/api/projects", json={"name": f"psr-{time.time_ns()}", "description": "x"}).json()["id"]
    if recipe_id:
        resp = client.put(f"/api/projects/{pid}/recipe", json={"recipe_id": recipe_id})
        assert resp.status_code == 200, resp.text
    return pid


def predefined(pid: int, train: list[dict], val: list[dict] | None = None, test: list[dict] | None = None) -> dict:
    files = {"train": ("train.jsonl", jsonl(train), "application/octet-stream")}
    if val:
        files["val"] = ("val.jsonl", jsonl(val), "application/octet-stream")
    if test:
        files["test"] = ("test.jsonl", jsonl(test), "application/octet-stream")
    resp = client.post(f"/api/projects/{pid}/dataset/predefined-split", files=files)
    assert resp.status_code == 200, resp.text
    return resp.json()


def random_split(pid: int, rows: list[dict]) -> dict:
    path = Path(settings.DATA_DIR) / "projects" / str(pid) / f"synthetic_{time.time_ns()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    async def add(db):
        db.add(Dataset(project_id=pid, name="synthetic", dataset_type=DatasetType.SYNTHETIC, file_path=str(path),
                       record_count=len(rows)))
        await db.commit()

    run_async(add)
    resp = client.post(f"/api/projects/{pid}/dataset/split", json={
        "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "dedup_rows": False})
    assert resp.status_code == 200, resp.text
    return resp.json()


def prepared(pid: int) -> dict[str, list[dict]]:
    out = {}
    for name in ("train", "val", "test"):
        path = _prep_dir(pid) / f"{name}.jsonl"
        out[name] = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    return out


def handler_for(manifest: dict) -> tuple[StructuredExtractionHandler, EvalContext]:
    ctx = EvalContext(project_id=0, experiment_id=0, eval_type="exact_match", task_profile="structured_extraction",
                      handler_id="structured_extraction", prepared_dir=Path("."), dataset_name="test", manifest=manifest)
    return StructuredExtractionHandler(), ctx


def row_key(row: dict) -> str:
    return str(row.get("task") or row.get("path"))


class RemediationImportTests(unittest.TestCase):
    def rows(self) -> dict[str, list[dict]]:
        return {"train": [code_example(i) for i in range(4)] + [remediation(0), remediation(1)],
                "val": [code_example(4), remediation(2)],
                "test": [code_example(5), code_example(6), remediation(3)]}

    def test_rows_import_with_the_random_split_prompt_and_stamp(self):
        rows = self.rows()
        pid = new_project("code-remediation")
        manifest = predefined(pid, rows["train"], rows["val"], rows["test"])
        self.assertEqual(manifest["splits"], {"train": 6, "val": 2, "test": 3})
        got = prepared(pid)

        # the same rows through a random split of a sibling project
        ref_pid = new_project("code-remediation")
        ref_manifest = random_split(ref_pid, [r for split in rows.values() for r in split])
        ref = {row_key(r): r for split in prepared(ref_pid).values() for r in split}
        self.assertEqual(len(ref), 11)

        for name, split_rows in got.items():
            self.assertEqual([row_key(r) for r in split_rows], [row_key(r) for r in rows[name]])
            for r in split_rows:
                self.assertEqual(r["source_text"], ref[row_key(r)]["source_text"])
                self.assertEqual(r["target_text"], ref[row_key(r)]["target_text"])
                for key in ("row_kind", "passages", "language"):
                    self.assertEqual(r[key], ref[row_key(r)][key])
                if r["row_kind"] == "code_example":
                    self.assertEqual(r["code_example_json"], ref[row_key(r)]["code_example_json"])
                    self.assertTrue(r["source_text"])
                else:
                    self.assertEqual(r["finding"], ref[row_key(r)]["finding"])
                    self.assertEqual(r["remediation_json"], ref[row_key(r)]["remediation_json"])

        # same recipe stamp as the random split; the recipe config is not persisted into the wiring
        self.assertEqual(manifest["output_schema"], ref_manifest["output_schema"])
        self.assertEqual(manifest["output_schema"]["scoring_mode"], "remediation")
        self.assertIn("remediation", manifest["output_schema"])
        self.assertNotIn("remediation", manifest["adapter_config"])
        preset = preset_of(pid)
        self.assertEqual(preset["adapter_id"], "structured-extraction")
        self.assertNotIn("remediation", preset["adapter_config"])

        # held-out eval renders each test row's training prompt (code examples scored when opted in)
        sch = dict(manifest["output_schema"])
        sch["remediation"] = {**sch["remediation"], "score_code_examples": True}
        handler, ctx = handler_for({**manifest, "output_schema": sch})
        for r, built in zip(got["test"], handler.build_prompts(got["test"], ctx)):
            self.assertEqual(built.prompt, r["source_text"])
            self.assertEqual(built.reference, r["target_text"])

    def test_heldout_scores_imported_code_examples_when_opted_in(self):
        pid = new_project("code-remediation")
        put = client.put(f"/api/projects/{pid}/remediation-config",
                         json={"checks": ["python_ast"], "score_code_examples": True})
        self.assertEqual(put.status_code, 200, put.text)
        rows = self.rows()
        predefined(pid, rows["train"], None, [code_example(5), code_example(6)])
        test_rows = prepared(pid)["test"]
        self.assertEqual([r["row_kind"] for r in test_rows], ["code_example", "code_example"])

        async def add(db):
            exp = Experiment(project_id=pid, name="psr run", status=ExperimentStatus.COMPLETED, base_model="toy",
                             config={}, output_dir=str(_prep_dir(pid)), completed_at=datetime.now(timezone.utc))
            db.add(exp)
            await db.commit()
            return exp.id

        exp_id = run_async(add)
        answers = [json.dumps({"code": EX_REF, "rationale": "r", "citations": ["P1"]}),
                   json.dumps({"code": "x = 1\n", "rationale": "r", "citations": ["P1"]})]
        seen_prompts: list[str] = []

        def infer(model_ref, pairs, max_new_tokens, temperature, stop_sequences=None, *, apply_chat_template=True,
                  decoding="neutral"):
            seen_prompts.extend(p["prompt"] for p in pairs)
            return ([{"prompt": p["prompt"], "formatted_prompt": "", "reference": p["reference"],
                      "prediction": answers[i], "generated_tokens": 5, "latency_ms": 1.0} for i, p in enumerate(pairs)],
                    {"engine": "stub", "decoding": {"mode": decoding}})

        async def judge(question, reference, prediction, instruction=None):
            if instruction != ces.JUDGE_INSTRUCTION:
                return 1.0, "correct", "fine", 1
            return (1.0, "correct", "ok", 1) if "cur.execute" in prediction else (0.0, "wrong", "no", 1)

        resolved = ajs.ResolvedJudge(label="stub:judge", judge=judge)

        async def resolve(*args, **kwargs):
            return resolved

        with patch("app.services.evaluation_service._run_local_inference", infer), \
                patch.object(ajs, "resolve_answer_judge", resolve), \
                patch.object(ajs, "project_cache", lambda _pid: ajs.AnswerJudgeCache(None)):
            resp = client.post(f"/api/projects/{pid}/evaluation/run-heldout", json={
                "experiment_id": exp_id, "dataset_name": "test", "eval_type": "exact_match", "max_samples": 20,
                "model_path": None})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(seen_prompts, [r["source_text"] for r in test_rows])
        m = resp.json()["metrics"]
        self.assertEqual(m["code_example_count"], 2)
        self.assertEqual(m["code_example_pass_rate"], 0.5)
        self.assertEqual(resp.json()["details"]["row_scores"]["code_example_pass"], [1.0, 0.0])


class FindingSetImportTests(unittest.TestCase):
    TAXONOMY = [{"id": "injection", "title": "Injection"},
                {"id": "injection.query", "title": "Query injection", "parent": "injection"}]

    def check(self, with_taxonomy: bool) -> None:
        rows = {"train": [findings(i) for i in range(6)], "val": [findings(6)], "test": [findings(7), findings(8)]}
        pids = []
        for _ in range(2):
            pid = new_project("code-findings")
            if with_taxonomy:
                put = client.put(f"/api/projects/{pid}/label-taxonomy", json={"entries": self.TAXONOMY})
                self.assertEqual(put.status_code, 200, put.text)
            pids.append(pid)
        manifest = predefined(pids[0], rows["train"], rows["val"], rows["test"])
        ref_manifest = random_split(pids[1], [r for split in rows.values() for r in split])
        ref = {row_key(r): r for split in prepared(pids[1]).values() for r in split}
        got = prepared(pids[0])
        self.assertEqual(sum(len(v) for v in got.values()), 9)
        for split_rows in got.values():
            for r in split_rows:
                self.assertEqual(r["source_text"], ref[row_key(r)]["source_text"])
                self.assertEqual(r["target_text"], ref[row_key(r)]["target_text"])
        if with_taxonomy:
            self.assertIn("Allowed labels: injection, injection.query", got["train"][0]["source_text"])
        self.assertEqual(manifest["output_schema"], ref_manifest["output_schema"])
        self.assertEqual(manifest["output_schema"]["labels"],
                         ["injection", "injection.query"] if with_taxonomy else ["injection.query"])
        self.assertNotIn("finding_set", manifest["adapter_config"])
        self.assertNotIn("finding_set", preset_of(pids[0])["adapter_config"])
        handler, ctx = handler_for(manifest)
        for r, built in zip(got["test"], handler.build_prompts(got["test"], ctx)):
            self.assertEqual(built.prompt, r["source_text"])

    def test_with_taxonomy(self):
        self.check(True)

    def test_without_taxonomy_labels_come_from_the_imported_rows(self):
        self.check(False)


# ── every other project: byte-identical to before the shared helper ─────────

DROP_MANIFEST = {"created_at", "project_id", "file_paths", "file_hashes", "dataset_versions", "prepared_version",
                 "source_files"}


def fingerprint(pid: int, manifest: dict) -> str:
    rows = {name: [{k: v for k, v in r.items() if not k.startswith("_")} for r in split]
            for name, split in prepared(pid).items()}
    kept = {k: v for k, v in manifest.items() if k not in DROP_MANIFEST}
    preset = preset_of(pid)
    blob = json.dumps({"rows": rows, "manifest": kept, "preset": preset}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


QA = [{"question": f"What is item {i}?", "answer": f"Item {i} is a widget."} for i in range(12)]
RECORDS = [{"text": f"Report {i}: unit {i} failed at stage {i % 3}.", "structured_output": {"unit": str(i), "stage": i % 3}}
           for i in range(12)]
RECORD_SCHEMA = {"properties": {"unit": {"type": "string"}, "stage": {"type": "integer"}}}


class NonRecipeRegressionTests(unittest.TestCase):
    # Recorded on the code before ``_apply_recipe_adapter_config`` existed (commit 5d656f8).
    GOLDEN = {
        "none_predefined": "1554c62dd8000aae1c38cade2f84fe7a5e19abc20fbff6b0430f8374a65aded3",
        "none_random": "97cf86c18148d0439fc19e55e34b8d2d69e313e3dd5ba03d455bcfee077dc4b9",
        "qa_predefined": "852ef2bacad44eea6f04b44edd81e8d6a12a0abe14dd8a92037cab50b78de69e",
        "qa_random": "48bb1d7ae6a305613d9f897e04acf01a9a0f63533856d4f5a4f2119709ed30f1",
        "records_predefined": "cfd157b51db9c0d82eaae47295a9d3510d2bb23f4cda87aee43a6d66d0153c4e",
        "records_random": "9c585effdce3a82e211db201cebbcfea246da3b47b8c29cecb5cbc4c41a5c2cb",
        "remediation_random": "aac095647f4bfcff3e157fb174d682dd8124cd18e5b441cb1025edad34cd225c",
    }

    def project(self, kind: str) -> int:
        if kind == "none":
            return new_project()
        if kind == "qa":
            return new_project("qa-sft")
        pid = new_project("json-extraction")
        resp = client.put(f"/api/projects/{pid}/recipe/output-schema", json={"output_schema": RECORD_SCHEMA})
        self.assertEqual(resp.status_code, 200, resp.text)
        return pid

    def assert_golden(self, name: str, value: str) -> None:
        self.assertEqual(value, self.GOLDEN[name], name)

    def test_predefined_imports_are_unchanged(self):
        for kind, rows in (("none", QA), ("qa", QA), ("records", RECORDS)):
            pid = self.project(kind)
            manifest = predefined(pid, rows[:8], rows[8:10], rows[10:])
            self.assert_golden(f"{kind}_predefined", fingerprint(pid, manifest))

    def test_random_splits_are_unchanged(self):
        for kind, rows in (("none", QA), ("qa", QA), ("records", RECORDS)):
            pid = self.project(kind)
            manifest = random_split(pid, rows)
            self.assert_golden(f"{kind}_random", fingerprint(pid, manifest))

    def test_recipe_random_split_is_unchanged(self):
        pid = new_project("code-remediation")
        rows = [code_example(i) for i in range(6)] + [remediation(i) for i in range(4)]
        manifest = random_split(pid, rows)
        # the remediation adapter config carries this project's id; normalise it
        manifest = json.loads(json.dumps(manifest).replace(f'"project_id": {pid}', '"project_id": 0'))
        self.assert_golden("remediation_random", fingerprint(pid, manifest))


if __name__ == "__main__":
    unittest.main()
