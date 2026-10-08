"""The core training loop on the REAL runtime, end to end, through the API.

Every other training test either imports ``scripts/train.py`` from
``backend/`` or runs the simulate runtime. Neither caught the week in which
every real run failed with "No module named 'app'": the runtime launches
``python scripts/train.py`` as a subprocess from a Celery worker, and that is
the path this test drives — project → sample data → experiment → start →
Celery → train.py → COMPLETED → watcher Job → automatic lift check → summary.

It runs against a live stack (backend on the external runtime + Redis +
a Celery worker) given by ``BREWSLM_REAL_TRAINING_BASE_URL``; without it the
file skips, so the per-file CI loop and local pytest ignore it. CI's
``real-training`` job boots that stack (CPU, SmolLM2-135M, one epoch) and
runs this file. It is a path-integrity gate: the lift may go either way on
CPU in one epoch; what must hold is that the run trains, the model is written,
the lift check runs and the summary can say which way it went.
"""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path

import httpx

BASE_URL = os.environ.get("BREWSLM_REAL_TRAINING_BASE_URL", "").rstrip("/")
BASE_MODEL = os.environ.get("BREWSLM_REAL_TRAIN_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
TRAIN_TIMEOUT_S = int(os.environ.get("BREWSLM_REAL_TRAINING_TIMEOUT_S", "2400"))
LIFT_TIMEOUT_S = int(os.environ.get("BREWSLM_REAL_TRAINING_LIFT_TIMEOUT_S", "1500"))


def _wait(predicate, timeout_s: float, every_s: float = 5.0, what: str = "condition"):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(every_s)
    raise AssertionError(f"timed out after {timeout_s}s waiting for {what}; last={last!r}")


@unittest.skipUnless(BASE_URL, "BREWSLM_REAL_TRAINING_BASE_URL not set (live stack needed)")
class RealTrainingPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = httpx.Client(base_url=BASE_URL, timeout=60.0)
        cls.http.get("/api/health").raise_for_status()
        token = os.environ.get("BREWSLM_REAL_TRAINING_TOKEN")
        if token:
            cls.http.headers["Authorization"] = f"Bearer {token}"

    @classmethod
    def tearDownClass(cls):
        cls.http.close()

    def _json(self, resp: httpx.Response) -> dict:
        self.assertLess(resp.status_code, 300, f"{resp.request.method} {resp.request.url} → {resp.status_code}: {resp.text[:800]}")
        return resp.json()

    def test_sample_project_trains_and_gets_a_lift_check(self):
        project = self._json(self.http.post("/api/projects", json={
            "name": f"real-training-ci-{int(time.time())}", "description": "CI real-runtime path gate",
        }))
        pid = project["id"]
        sample = self._json(self.http.post(
            f"/api/projects/{pid}/quickstart/import-sample", json={"slug": "support-faq"}
        ))["summary"]
        self.assertGreaterEqual(sample["prepared_train_rows"], 100)
        self.assertGreaterEqual(sample["prepared_test_rows"], 20)

        # One explicit epoch keeps a CPU run short; everything else is the
        # default a user gets (LoRA "auto" targets, chat-template rendering,
        # completion-only loss, the lift check afterwards).
        exp = self._json(self.http.post(f"/api/projects/{pid}/training/experiments", json={
            "name": "CI · real runtime", "config": {"base_model": BASE_MODEL, "num_epochs": 1},
        }))
        exp_id = exp["id"]
        start = self._json(self.http.post(f"/api/projects/{pid}/training/experiments/{exp_id}/start"))
        self.assertEqual(start.get("status"), "running", start)
        self.assertIn("external", str(start.get("runtime_id", "")), "this gate must use the real runtime")

        def _experiment():
            rows = self._json(self.http.get(f"/api/projects/{pid}/training/experiments"))
            row = next(r for r in rows if r["id"] == exp_id)
            return row if row["status"].lower() in {"completed", "failed", "cancelled"} else None

        t_start = time.monotonic()
        final = _wait(_experiment, TRAIN_TIMEOUT_S, what="training to finish")
        t_trained = time.monotonic()
        print(f"[real-training] training finished in {t_trained - t_start:.0f}s", flush=True)
        if final["status"].lower() != "completed":
            log = Path(str(final.get("output_dir") or "")) / "external_training.log"
            tail = ""
            if log.exists():
                try:
                    payload = json.loads(log.read_text(encoding="utf-8"))
                    tail = (payload.get("stderr") or payload.get("stdout") or "")[-3000:]
                except Exception:  # noqa: BLE001
                    tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
            self.fail(f"training ended {final['status']}: {tail or final}")

        output_dir = Path(str(final["output_dir"]))
        model_dir = output_dir / "model"
        weights = [p.name for p in model_dir.iterdir()] if model_dir.is_dir() else []
        self.assertTrue(
            any(name.endswith((".safetensors", ".bin")) for name in weights),
            f"no weights written under {model_dir}: {weights}",
        )
        report = json.loads((output_dir / "training_report.json").read_text(encoding="utf-8"))
        env = report.get("runtime_environment") or {}
        self.assertEqual(env.get("loss_masking"), "completion_only")
        self.assertEqual(env.get("chat_template_rewrap"), "tokenizer_chat_template")
        self.assertEqual((env.get("lora_target_modules") or {}).get("target_modules"), "all-linear")

        # The watcher Job must have fired the automatic lift check, and the
        # check must finish: base model + fine-tuned run on the test split.
        def _jobs():
            jobs = self._json(self.http.get("/api/jobs/active", params={"include_recently_completed": "true", "limit": 50}))["jobs"]
            mine = [j for j in jobs if j.get("project_id") == pid]
            watcher = next((j for j in mine if j["kind"] == "training_start"), None)
            lift = next((j for j in mine if j["kind"] == "post_training_lift_eval"), None)
            if watcher and watcher["status"] in {"failed", "cancelled"}:
                raise AssertionError(f"watcher Job failed: {watcher.get('error')}")
            if lift and lift["status"] in {"failed", "cancelled"}:
                raise AssertionError(f"lift-check Job failed: {lift.get('error')}")
            if watcher and watcher["status"] == "succeeded" and lift and lift["status"] == "succeeded":
                return {"watcher": watcher, "lift": lift}
            return None

        jobs = _wait(_jobs, LIFT_TIMEOUT_S, what="the lift check to finish")
        t_lifted = time.monotonic()
        print(f"[real-training] lift check finished {t_lifted - t_trained:.0f}s after training", flush=True)
        report_env = report.get("runtime_environment") or {}
        metrics_path = output_dir / "metrics.jsonl"
        if metrics_path.exists():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if '"train_runtime"' in line:
                    print(f"[real-training] trainer: {line.strip()[:200]}", flush=True)
        print(f"[real-training] lora targets: {(report_env.get('lora_target_modules') or {}).get('reason')}", flush=True)
        self.assertTrue((jobs["watcher"].get("result") or {}).get("auto_lift_eval", {}).get("started"))
        lift_result = jobs["lift"].get("result") or {}
        self.assertEqual(lift_result.get("lift_status"), "ok", lift_result)
        self.assertIn(lift_result.get("headline", {}).get("direction"), {"improved", "regressed", "unchanged"})

        summary = self._json(self.http.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": exp_id}))
        self.assertIn(summary["verdict"], {"better", "worse", "same"}, summary)
        head = summary["headline"]
        # judge_correct when a judge is configured for the stack (CI pins the
        # deterministic template judge), token F1 otherwise.
        self.assertIn(head["metric_id"], {"f1", "judge_correct"})
        for key in ("baseline_value", "trained_value"):
            self.assertIsInstance(head[key], float)
            self.assertGreaterEqual(head[key], 0.0)
        # Row-level evidence needs per-row scores from BOTH evals — the part of
        # the loop that makes "better than base" honest.
        self.assertIsNotNone(summary.get("evidence"), "no row evidence: per-row scores were not recorded")
        expected_rows = min(sample["prepared_test_rows"], int(os.environ.get("AUTO_LIFT_EVAL_MAX_SAMPLES", "100")))
        self.assertEqual(summary["evidence"]["n"], expected_rows)
        print(
            f"[real-training] run #{exp_id}: {head['metric_id']} {head['baseline_value']:.3f} → "
            f"{head['trained_value']:.3f} ({summary['verdict']}, {summary['evidence']['verdict']})"
        )

# ── Documents project: the pre-training gate on the real stack ───────────


def _statute_html(sections: int = 30) -> str:
    """A small synthetic act: ``sections`` paragraphs, each with its own
    vocabulary (a distinct duty, object, deadline and penalty) so rows don't
    collapse under near-duplicate dedup and passage retrieval has something
    to find."""
    duties = ["publish", "audit", "seal", "index", "translate", "archive", "certify", "redact", "transmit", "catalogue",
              "inspect", "register", "appraise", "digitise", "witness", "annotate", "summarise", "escrow", "license", "notarise",
              "quarantine", "reconcile", "encrypt", "tabulate", "forecast", "benchmark", "garnish", "survey", "calibrate", "underwrite"]
    objects = ["the disclosure register", "the annual report", "each notice of refusal", "the fee schedule", "every access request",
               "the correction log", "the retention record", "the complaint file", "the exemption list", "the consent forms",
               "the mining ledger", "the harbour permits", "the grain tallies", "the vaccine lots", "the broadcast licences",
               "the pension rolls", "the ferry manifests", "the timber quotas", "the census returns", "the patent abstracts",
               "the rail tariffs", "the fishery charts", "the dam inspections", "the postal routes", "the bridge tolls",
               "the orchard grants", "the lighthouse logs", "the canal leases", "the museum loans", "the vineyard bonds"]
    deadlines = ["thirty days", "ninety days", "two years", "ten business days", "one fiscal year", "six weeks", "the next quarter"]
    penalties = ["a fine of two hundred dollars", "suspension of the licence", "a written reprimand", "forfeiture of the fee",
                 "referral to the tribunal", "publication of the default", "a daily penalty of fifty dollars"]
    paras = []
    for i in range(1, sections + 1):
        duty, obj = duties[(i - 1) % len(duties)], objects[(i - 1) % len(objects)]
        deadline, penalty = deadlines[i % len(deadlines)], penalties[(i * 2) % len(penalties)]
        paras.append(
            f"<p>Section {i}. The officer responsible for section {i} shall {duty} {obj} within {deadline} of receiving it. "
            f"Failure to {duty} {obj} in time carries {penalty} under section {i}. "
            f"Once {obj} has been {duty}d, the officer shall tell the applicant in writing that section {i} is complete.</p>"
        )
    return "<html><body><h1>Synthetic Information Act</h1>" + "".join(paras) + "</body></html>"


@unittest.skipUnless(BASE_URL, "BREWSLM_REAL_TRAINING_BASE_URL not set (live stack needed)")
class DocumentsProjectGateTests(unittest.TestCase):
    """documents → Q&A flow → the pre-training gate (base model with and
    without the passages, judged) → stop or train → lift check. Generation
    and judging use the deterministic template backend (the stack is booted
    with BREWSLM_TEMPLATE_SYNTH=1 and EVAL_JUDGE_BACKEND=template); the base
    model inference is real, on CPU."""

    @classmethod
    def setUpClass(cls):
        cls.http = httpx.Client(base_url=BASE_URL, timeout=120.0)
        token = os.environ.get("BREWSLM_REAL_TRAINING_TOKEN")
        if token:
            cls.http.headers["Authorization"] = f"Bearer {token}"

    @classmethod
    def tearDownClass(cls):
        cls.http.close()

    def _json(self, resp: httpx.Response) -> dict:
        self.assertLess(resp.status_code, 300, f"{resp.request.method} {resp.request.url} → {resp.status_code}: {resp.text[:800]}")
        return resp.json()

    def _job(self, job_id: int) -> dict:
        return self._json(self.http.get(f"/api/jobs/{job_id}"))

    def _wait_job(self, job_id: int, timeout_s: float, what: str) -> dict:
        def _done():
            job = self._job(job_id)
            return job if job["status"] in {"succeeded", "failed", "cancelled"} else None

        job = _wait(_done, timeout_s, what=what)
        self.assertEqual(job["status"], "succeeded", f"{what}: {job.get('error')}")
        return job

    def test_documents_flow_gates_on_retrieval_then_trains(self):
        project = self._json(self.http.post("/api/projects", json={
            "name": f"real-training-docs-{int(time.time())}",
            "description": "CI: documents project, pre-training retrieval gate",
        }))
        pid = project["id"]
        self._json(self.http.put(f"/api/projects/{pid}", json={"base_model_name": BASE_MODEL}))
        doc = self._json(self.http.post(
            f"/api/projects/{pid}/ingestion/upload",
            files={"file": ("synthetic-act.html", _statute_html().encode("utf-8"), "text/html")},
        ))
        self._json(self.http.post(f"/api/projects/{pid}/ingestion/documents/{doc['id']}/process"))
        self._json(self.http.post(f"/api/projects/{pid}/cleaning/clean", json={
            "document_id": doc["id"], "chunk_size": 320, "chunk_overlap": 0,
        }))
        passages = self._json(self.http.get(f"/api/projects/{pid}/auto-rag/documents"))["passages"]
        self.assertGreaterEqual(passages, 20, "cleaning should yield one passage per section")

        t0 = time.monotonic()
        job = self._json(self.http.post(f"/api/projects/{pid}/flows/documents-to-qa", json={
            "backend": "template", "max_passages": 30, "pairs_per_passage": 3,
        }))
        result = self._wait_job(job["id"], TRAIN_TIMEOUT_S, "the documents → Q&A flow")["result"]
        print(f"[real-training] flow finished in {time.monotonic() - t0:.0f}s: {result.get('training_pairs')} pairs, "
              f"split {result.get('split')}, gate {result.get('passages_gate', {}).get('status')}", flush=True)
        self.assertGreaterEqual(result["training_pairs"], 40)
        self.assertGreaterEqual((result.get("split") or {}).get("test") or 0, 5)

        gate = result.get("passages_gate") or {}
        # The gate must have RUN and been judged (template judge): a decision
        # either way, never not_run / not_judged / error on this stack.
        self.assertIn(gate.get("status"), {"retrieval_ready", "retrieval_weak"}, gate)
        self.assertEqual(gate.get("split"), "test")
        self.assertGreaterEqual(gate.get("judged", 0), 5)
        self.assertTrue(str(gate.get("judge") or "").startswith("template"), gate)
        print(f"[real-training] gate: {gate['status']} — {gate.get('correct')}/{gate.get('partial')}/{gate.get('wrong')} "
              f"of {gate.get('judged')}, judge score {gate.get('score')}", flush=True)

        if gate["status"] == "retrieval_ready":
            # Stopped before training — and the project says so everywhere.
            self.assertIsNone(result.get("experiment_id"))
            self.assertIn("Retrieval already answers", result.get("stopped_reason") or "")
            self.assertEqual(self._json(self.http.get(f"/api/projects/{pid}/training/experiments")), [])
            coach = self._json(self.http.get(f"/api/projects/{pid}/coach/training"))
            ids = [s["id"] for s in coach["suggestions"]]
            self.assertIn("training:retrieval-ready", ids, ids)
            summary = self._json(self.http.get(f"/api/projects/{pid}/evaluation/summary"))
            self.assertEqual(summary.get("kind"), "rag_passages")
            self.assertEqual(summary["pretraining_gate"]["status"], "retrieval_ready")
            # Train anyway: same pairs, same split, gate recorded, run started.
            job = self._json(self.http.post(f"/api/projects/{pid}/flows/documents-to-qa", json={
                "reuse_existing": True, "train_if_retrieval_ready": True,
            }))
            result = self._wait_job(job["id"], TRAIN_TIMEOUT_S, "the flow (train anyway)")["result"]
            self.assertEqual((result.get("passages_gate") or {}).get("status"), "retrieval_ready")
        exp_id = result.get("experiment_id")
        self.assertIsNotNone(exp_id, result)

        def _experiment():
            rows = self._json(self.http.get(f"/api/projects/{pid}/training/experiments"))
            row = next(r for r in rows if r["id"] == exp_id)
            return row if row["status"].lower() in {"completed", "failed", "cancelled"} else None

        final = _wait(_experiment, TRAIN_TIMEOUT_S, what="training to finish")
        self.assertEqual(final["status"].lower(), "completed", final)

        def _lift():
            jobs = self._json(self.http.get("/api/jobs/active", params={"include_recently_completed": "true", "limit": 50}))["jobs"]
            lift = next((j for j in jobs if j.get("project_id") == pid and j["kind"] == "post_training_lift_eval"), None)
            if lift and lift["status"] in {"failed", "cancelled"}:
                raise AssertionError(f"lift-check Job failed: {lift.get('error')}")
            return lift if lift and lift["status"] == "succeeded" else None

        _wait(_lift, LIFT_TIMEOUT_S, what="the lift check to finish")
        summary = self._json(self.http.get(f"/api/projects/{pid}/evaluation/summary", params={"experiment_id": exp_id}))
        self.assertIn(summary["verdict"], {"better", "worse", "same"}, summary)
        self.assertEqual(summary["headline"]["metric_id"], "judge_correct", "the template judge must have scored the lift")
        # Both candidates measured on the same test rows: the verdict can be read.
        verdict = self._json(self.http.get(f"/api/projects/{pid}/auto-rag/comparison"))["passages_vs_finetune"]
        # Same judge, same rows, both candidates judged: the verdict is readable.
        # Whether it CALLS a winner depends on row noise (a 7-row split can't).
        self.assertTrue(verdict["same_rows"], verdict)
        self.assertTrue((verdict.get("finetune") or {}).get("judged"), verdict)
        self.assertIn(verdict["reason"], {"passages_ahead", "finetune_ahead", "retrieval_gain_within_noise"}, verdict)
        print(f"[real-training] documents project: gate {gate['status']}, run #{exp_id} judge "
              f"{summary['headline']['baseline_value']:.3f} → {summary['headline']['trained_value']:.3f}; "
              f"passages vs fine-tune: {verdict['reason']}", flush=True)

