"""vLLM multi-LoRA serve plan (G11)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _gateway_helpers as h  # noqa: E402
from app.models.experiment import ExperimentStatus  # noqa: E402
from app.services.serve_service import _round_lora_rank  # noqa: E402


@pytest.fixture(scope="module")
def client():
    cm, c, prev = h.open_client()
    yield c
    h.close_client(cm, prev)


@pytest.fixture()
def root():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def plan(client, pid, ids, **extra):
    return client.post(f"/api/projects/{pid}/serve/multi-lora-plan", json={"experiment_ids": ids, **extra})


def test_rank_rounding():
    assert [_round_lora_rank(r) for r in (1, 8, 9, 16, 17, 33, 64, 100, 256, 999)] == [8, 8, 16, 16, 32, 64, 64, 128, 256, 256]


def test_shared_base_plan(client, root):
    pid = h.make_project(client)
    a = h.seed_run(pid, root, rank=8, name="alpha")
    b = h.seed_run(pid, root, rank=20, name="beta")
    resp = plan(client, pid, [a, b], port=8123, max_model_len=4096)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    slug = h.slug_of(pid)
    assert body["base_model"] == h.BASE
    assert [(x["run_id"], x["name"], x["module_name"], x["rank"]) for x in body["adapters"]] == [
        (a, "alpha", f"{slug}:run-{a}", 8), (b, "beta", f"{slug}:run-{b}", 20)]
    assert all(Path(x["path"]).name == "model" for x in body["adapters"])
    t = body["template"]
    assert t["template_id"] == "runner.vllm_multi_lora"
    argv = t["launch_spec"]["argv"]
    assert argv[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]

    def val(flag):
        return argv[argv.index(flag) + 1]

    assert val("--model") == h.BASE and val("--served-model-name") == f"{slug}:base"
    assert "--enable-lora" in argv and val("--max-loras") == "2" and val("--max-lora-rank") == "32"  # 20 -> 32
    mods = argv[argv.index("--lora-modules") + 1: argv.index("--host")]
    assert mods == [f"{slug}:run-{a}={body['adapters'][0]['path']}", f"{slug}:run-{b}={body['adapters'][1]['path']}"]
    assert val("--port") == "8123" and val("--max-model-len") == "4096" and val("--host") == "127.0.0.1"
    assert Path(t["launch_spec"]["cwd"]).is_dir() and "--enable-lora" in t["command"]
    assert {"healthcheck", "smoke_test", "notes"} <= set(t)
    assert t["smoke_test"]["json_body"]["model"] == f"{slug}:run-{a}"
    assert body["gateway_settings_hint"] == {"upstream": "openai_compatible", "base_url": "http://127.0.0.1:8123",
                                             "upstream_kind": "vllm"}
    assert plan(client, pid, [a]).json()["port"] == 8001  # default port


def test_mismatched_base_is_400(client, root):
    pid = h.make_project(client)
    a = h.seed_run(pid, root, base="org/base-1")
    b = h.seed_run(pid, root, base="org/base-2")
    resp = plan(client, pid, [a, b])
    assert resp.status_code == 400
    msg = resp.json()["detail"]
    assert f"run #{a} -> org/base-1" in msg and f"run #{b} -> org/base-2" in msg


def test_rejects_non_adapter_incomplete_and_foreign_runs(client, root):
    pid = h.make_project(client)
    full = h.seed_run(pid, root, adapter=False)
    running = h.seed_run(pid, root, status=ExperimentStatus.RUNNING)
    foreign = h.seed_run(h.make_project(client), root)
    r = plan(client, pid, [full])
    assert r.status_code == 400 and "not a LoRA adapter" in r.json()["detail"]
    assert plan(client, pid, [running]).status_code == 400
    assert plan(client, pid, [foreign]).status_code == 404
    assert plan(client, pid, []).status_code == 400


def test_start_run_uses_serve_runtime(client, root):
    pid = h.make_project(client)
    a = h.seed_run(pid, root)
    seen = {}

    async def fake_start(**kwargs):
        seen.update(kwargs)
        return {"run_id": "abc", "status": "running"}

    with mock.patch("app.services.serve_runtime_service.start_serve_run", fake_start):
        resp = client.post(f"/api/projects/{pid}/serve/multi-lora-runs/start", json={"experiment_ids": [a]})
    assert resp.status_code == 200 and resp.json()["run_id"] == "abc"
    assert seen["source"] == "multi_lora" and seen["export_id"] is None and seen["model_id"] is None
    assert seen["project_id"] == pid and seen["template"]["template_id"] == "runner.vllm_multi_lora"


def test_rank_above_256_is_400_naming_the_run(client, root):
    pid = h.make_project(client)
    big = h.seed_run(pid, root, rank=512)
    resp = plan(client, pid, [big])
    assert resp.status_code == 400 and f"Run #{big}" in resp.json()["detail"] and "512" in resp.json()["detail"]


def test_adapter_without_weights_is_rejected(client, root):
    pid = h.make_project(client)
    ok = h.seed_run(pid, root)
    empty = h.seed_run(pid, root, weights=False)
    resp = plan(client, pid, [ok, empty])
    assert resp.status_code == 400 and f"Run #{empty}" in resp.json()["detail"] and "weights" in resp.json()["detail"]


def test_plan_notes_warn_about_unauthenticated_vllm(client, root):
    pid = h.make_project(client)
    a = h.seed_run(pid, root)
    notes = plan(client, pid, [a]).json()["template"]["notes"]
    assert any("no authentication" in n and "127.0.0.1" in n for n in notes)
