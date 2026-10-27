"""Seeding helpers shared by the gateway tests (no network, no real models)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import settings
from app.database import async_session_factory
from app.main import app
from app.models.dataset import Dataset, DatasetType, RawDocument
from app.models.experiment import Experiment, ExperimentStatus, TrainingMode
from app.models.project import Project
from app.services import gateway_service

BASE = "org/base-1"


def open_client():
    prev = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    cm = TestClient(app)
    return cm, cm.__enter__(), prev


def close_client(cm, prev):
    cm.__exit__(None, None, None)
    settings.AUTH_ENABLED = prev


def reset():
    gateway_service.reset_rate_limits()
    gateway_service._HTTP_TRANSPORT = None


def make_project(client, base_model: str = BASE, recipe: str | None = None, name: str | None = None) -> int:
    resp = client.post("/api/projects", json={"name": name or f"gw-{uuid4().hex[:8]}", "description": ""})
    assert resp.status_code == 201, resp.text
    pid = int(resp.json()["id"])

    async def _go():
        async with async_session_factory() as db:
            p = await db.get(Project, pid)
            p.base_model_name = base_model
            if recipe:
                p.selected_recipe = {"recipe_id": recipe}
            await db.commit()

    asyncio.run(_go())
    return pid


def set_runtime(pid: int, **cfg) -> None:
    async def _go():
        async with async_session_factory() as db:
            p = await db.get(Project, pid)
            p.runtime_config = {**(p.runtime_config or {}), **cfg}
            await db.commit()

    asyncio.run(_go())


def slug_of(pid: int) -> str:
    async def _go():
        async with async_session_factory() as db:
            return (await gateway_service.all_slugs(db))[pid]

    return asyncio.run(_go())


def seed_run(pid: int, root: Path, *, adapter: bool = True, base: str = BASE, rank: int = 8,
             config: dict | None = None, name: str | None = None, status=ExperimentStatus.COMPLETED,
             weights: bool = True) -> int:
    out = root / f"run-{uuid4().hex[:6]}"
    model = out / "model"
    model.mkdir(parents=True, exist_ok=True)
    if weights:
        if adapter:
            (model / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base, "r": rank}))
            (model / "adapter_model.safetensors").write_bytes(b"\0" * 8)
        else:
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"\0" * 8)

    async def _go():
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=pid, name=name or f"run-{uuid4().hex[:6]}", status=status,
                training_mode=TrainingMode.SFT, base_model=base, output_dir=str(out), config=config or {},
            )
            db.add(exp)
            await db.commit()
            return exp.id

    return asyncio.run(_go())


def seed_document(pid: int, url: str | None) -> int:
    async def _go():
        async with async_session_factory() as db:
            ds = Dataset(project_id=pid, name=f"ds-{uuid4().hex[:4]}", dataset_type=DatasetType.RAW)
            db.add(ds)
            await db.flush()
            doc = RawDocument(dataset_id=ds.id, filename="guide.md", file_type="md", file_path="/x/guide.md",
                              source_url=url)
            db.add(doc)
            await db.commit()
            return doc.id

    return asyncio.run(_go())


def make_key(client, project_ids, rate: int | None = None) -> str:
    body = {"name": "test", "project_ids": project_ids}
    if rate is not None:
        body["rate_limit_per_minute"] = rate
    resp = client.post("/api/gateway/keys", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["key"]


def auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}
