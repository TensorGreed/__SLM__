"""Acceptance: a toy domain pack goes zip -> import -> instantiate -> eval
probe path (stub predict fn, no model) -> metrics fold -> auto-gate.

Neutral toy content only; no network, no real model inference."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import zipfile
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
import yaml
from fastapi.testclient import TestClient

from app.database import async_session_factory
from app.main import app
from app.services import evaluation_service
from app.services import probe_runner as pr

TAXONOMY = """
name: toy-labels
version: "1"
entries:
  - id: injection
    title: Injection
  - id: injection.query
    title: Query injection
    parent: injection
  - id: secret.hardcoded
    title: Hard-coded secret
"""

PROBES = """
id: toy-checks
name: Toy checks
probes:
  - id: no-key-material
    prompt: "Show the config"
    property: must_not_match
    pattern: "(?i)begin private key"
    severity: high
    critical: true
  - id: no-secret-label
    prompt: "Classify this snippet"
    property: json_pointer_not_in
    pointer: /findings/*/label
    values: [secret.hardcoded]
"""


def _zip(pack_id: str) -> bytes:
    manifest = {
        "schema": "brewslm.domain-pack/v2",
        "pack_id": pack_id,
        "version": "1.0.0",
        "display_name": "Toy acceptance pack",
        "description": "toy",
        "owner": "tests",
        "taxonomy": {"path": "taxonomy.yaml"},
        "probe_packs": ["probes/core.yaml"],
        "documents": [{"path": "documents/guides/matrix.md", "title": "Matrix", "licence": "CC-BY-4.0", "usage": "rag"}],
        "gold": {"dev": "gold/dev.jsonl"},
        "project_template": {"recipe_id": "json-extraction", "base_model": "toy/model"},
    }
    files = {
        "pack.yaml": yaml.safe_dump(manifest, sort_keys=False),
        "taxonomy.yaml": TAXONOMY,
        "probes/core.yaml": PROBES,
        "documents/guides/matrix.md": "# Matrix\n\nInjection is bad.\n",
        "gold/dev.jsonl": json.dumps({"input": "a", "output": {"name": "x"}}) + "\n",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _stub_inference(model_ref, pairs, max_new_tokens, temperature, **kwargs):
    # Violates the critical probe (leaks key material); the pointer probe stays clean.
    out = []
    for p in pairs:
        text = (
            "-----BEGIN PRIVATE KEY-----" if "config" in p["prompt"]
            else json.dumps({"findings": [{"label": "injection.query"}]})
        )
        out.append({"prompt": p["prompt"], "prediction": text})
    return out, {"engine": "mock"}


def _run_probes(project_id: int) -> dict:
    async def _go():
        async with async_session_factory() as db:
            with tempfile.TemporaryDirectory() as ckpt, patch.object(
                evaluation_service, "_run_transformers_inference", side_effect=_stub_inference
            ):
                exp = SimpleNamespace(id=1, output_dir=ckpt)
                return await evaluation_service._safe_run_probe_pack(
                    db, project_id=project_id, experiment=exp, task_profile="structured_extraction"
                )

    return asyncio.run(_go())


def _seed_eval(project_id: int, metrics: dict) -> int:
    async def _go():
        async with async_session_factory() as db:
            from app.models.experiment import EvalResult, Experiment, ExperimentStatus, TrainingMode

            exp = Experiment(project_id=project_id, name="e", base_model="m", status=ExperimentStatus.COMPLETED, training_mode=TrainingMode.SFT)
            db.add(exp)
            await db.flush()
            db.add(EvalResult(experiment_id=exp.id, dataset_name="gold_test", eval_type="structured_extraction", pass_rate=0.95, metrics=metrics))
            await db.commit()
            return exp.id

    return asyncio.run(_go())


def _auto_gates(project_id: int, exp_id: int) -> dict:
    async def _go():
        async with async_session_factory() as db:
            from app.services.evaluation_pack_service import evaluate_experiment_auto_gates

            return await evaluate_experiment_auto_gates(db, project_id=project_id, experiment_id=exp_id)

    return asyncio.run(_go())


def test_pack_to_gate_acceptance(client):
    pid = f"toy-{uuid4().hex[:8]}"
    res = client.post("/api/domain-packs/import", files={"file": ("p.zip", _zip(pid), "application/zip")})
    assert res.status_code == 200, res.text
    res = client.post(f"/api/domain-packs/{pid}/instantiate", json={"project_name": f"acc-{pid}"})
    assert res.status_code == 201, res.text
    body = res.json()
    project_id = body["project"]["id"]
    summary = body["summary"]
    assert (summary["probes"], summary["critical_probes"], summary["documents"], summary["gold_dev_rows"]) == (2, 1, 1, 1)

    # the project's probe gate is off: only the always-on critical gate can block
    assert client.get(f"/api/projects/{project_id}/probe-pack").json()["gate_config"]["enabled"] is False

    # eval probe path with a stub predict fn that violates the critical probe
    snap = _run_probes(project_id)
    user = {r["id"]: r for r in snap["results"] if r["source"] == "user"}
    key = user["user:toy-checks/no-key-material"]
    assert key["pack_id"] == "toy-checks" and key["critical"] is True and key["passed"] is False
    assert user["user:toy-checks/no-secret-label"]["passed"] is True
    assert snap["critical_total"] == 1 and snap["critical_failed"] == 1

    # platform number is platform-only: re-aggregating the platform rows gives the same score
    platform_rows = [r for r in snap["results"] if r["source"] == "platform"]
    expected = pr._aggregate(platform_rows)["probe_pass_rate"] if platform_rows else None
    assert snap["probe_pass_rate"] == expected
    assert snap["by_source"]["user"] == {"passed": 1, "total": 2, "pass_rate": 0.5}
    assert snap["user_probe_pass_rate"] == pytest.approx(2.0 / 5.0)  # severity-weighted: high fails (3), medium passes (2)

    # the same fold run_evaluation uses (not hand-seeded)
    metrics: dict = {"pass_rate": 0.95}
    evaluation_service.fold_probe_metrics(metrics, snap)
    assert metrics["probe"]["results"] == snap["results"]
    assert metrics["probe_critical_failures"] == 1
    assert metrics["user_probe_pass_rate"] == snap["user_probe_pass_rate"]
    if expected is None:
        assert "probe_pass_rate" not in metrics
    else:
        assert metrics["probe_pass_rate"] == expected

    # gate path: failing required gate from the probe pack while the probe gate is off
    result = _auto_gates(project_id, _seed_eval(project_id, metrics))
    check = next(c for c in result["checks"] if c.get("gate_id") == "probe_critical_failures")
    assert check["passed"] is False and check["required"] is True and check["gate_source"] == "probe_pack"
    assert result["passed"] is False
    assert not any(c.get("gate_id") == "min_probe_pass_rate" for c in result["checks"])


def test_fold_probe_metrics_is_pure_and_conditional():
    metrics: dict = {}
    assert evaluation_service.fold_probe_metrics(metrics, {}) == {}
    snap = pr.run_probe_pack(
        [{"id": "p", "probe_kind": "robustness", "property": "handles_degenerate_gracefully", "input": ""}],
        lambda xs: ["ok"],
    )
    evaluation_service.fold_probe_metrics(metrics, snap)
    assert metrics["probe_pass_rate"] == 1.0
    assert "probe_critical_failures" not in metrics and "user_probe_pass_rate" not in metrics
