"""Offline knowledge distillation through the experiment API.

``scripts/train.py`` trains ``training_mode="distillation"`` (and the
``distillation_offline`` flag) against a teacher-capture artifact, and the
``recipe.kd.*`` presets set exactly those keys — but ``POST /experiments``
used to reject the mode (422) and drop the offline keys from the stored
config. It now accepts and passes them through unchanged.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models.experiment import TrainingMode
from app.schemas.training import TrainingConfig
from app.services.training_recipe_service import list_training_recipes


@pytest.fixture(scope="module")
def client():
    prev = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    with TestClient(app) as c:
        yield c
    settings.AUTH_ENABLED = prev


@pytest.fixture()
def project_id(client):
    resp = client.post("/api/projects", json={"name": f"kd-{uuid4().hex[:6]}", "description": ""})
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["id"]


def _create(client, pid, config):
    return client.post(
        f"/api/projects/{pid}/training/experiments",
        json={"name": "offline kd", "config": {"base_model": "HuggingFaceTB/SmolLM2-135M-Instruct", **config}},
    )


def test_offline_kd_mode_is_accepted_and_passed_through(client, project_id):
    capture = settings.DATA_DIR / "projects" / str(project_id) / "distillation" / "teacher_capture.jsonl"
    resp = _create(client, project_id, {
        "task_type": "causal_lm",
        "training_mode": "distillation",
        "distillation_offline": True,
        "distillation_capture_path": str(capture),
        "distillation_offline_top_k": 10,
        "distillation_alpha": 0.5,
    })
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["training_mode"] == "distillation"
    cfg = body["config"]
    assert cfg["training_mode"] == "distillation"
    assert cfg["distillation_offline"] is True
    assert cfg["distillation_capture_path"] == str(capture)
    assert cfg["distillation_offline_top_k"] == 10


def test_every_kd_recipe_patch_validates():
    patches = [r for r in list_training_recipes(include_patch=True) if str(r.get("recipe_id", "")).startswith("recipe.kd.")]
    assert len(patches) >= 3
    for recipe in patches:
        cfg = TrainingConfig(base_model="m", **recipe["config_patch"])
        assert cfg.training_mode == TrainingMode.DISTILLATION
        assert cfg.distillation_offline is True


def test_defaults_are_unchanged_for_ordinary_runs():
    cfg = TrainingConfig(base_model="m").model_dump()
    assert cfg["training_mode"] == TrainingMode.SFT
    # Same defaults train.py applies when the keys are absent.
    assert cfg["distillation_offline"] is False
    assert cfg["distillation_capture_path"] == ""
    assert cfg["distillation_offline_top_k"] == 8


def test_offline_and_live_teacher_kd_together_are_rejected(client, project_id):
    resp = _create(client, project_id, {
        "training_mode": "distillation", "distillation_enabled": True, "distillation_teacher_model": "t",
    })
    assert resp.status_code == 422
    assert "distillation_enabled" in resp.text


def test_capture_path_outside_the_data_dir_is_rejected(client, project_id):
    resp = _create(client, project_id, {"training_mode": "distillation", "distillation_capture_path": "/etc/passwd"})
    assert resp.status_code == 422
    assert "distillation_capture_path" in resp.text
