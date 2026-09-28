"""Dataset-import API surface (Phase A).

Four endpoints:

- ``GET  /api/dataset-import/sources``  — list registered source ids
- ``GET  /api/dataset-import/mappers``  — list registered mapper ids
- ``POST /api/projects/{id}/dataset-import/preview`` — dry-run a
  source × mapper combination
- ``POST /api/projects/{id}/dataset-import/run``     — persist accepted
  rows to the project's synthetic dataset

Phase B adds an introspector endpoint; Phase F wires the UI wizard
to these endpoints.
"""

from __future__ import annotations

from typing import Any

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.services.dataset_import import (
    list_registered_mappers,
    list_registered_sources,
)
from app.services.dataset_import.configs import (
    config_to_dict,
    delete_config,
    get_config,
    list_configs,
    run_from_config,
    save_config,
)
from app.services.dataset_import.service import (
    introspect_locator,
    preview_import,
    result_to_dict,
    run_import,
)


# Two routers — the registry endpoints aren't project-scoped, but
# preview/run are. Keeps URL shapes clean.
catalog_router = APIRouter(prefix="/dataset-import", tags=["Dataset Import"])
project_router = APIRouter(
    prefix="/projects/{project_id}/dataset-import", tags=["Dataset Import"]
)


class IntrospectRequest(BaseModel):
    locator: str = Field(
        ...,
        min_length=3,
        description="Source-prefixed locator, e.g. 'jsonl:/tmp/data.jsonl'. "
        "The introspector samples the source and proposes a mapping.",
    )
    sample_size: int = Field(default=20, ge=1, le=100)
    llm_assist: bool = Field(
        default=False,
        description="Phase H opt-in: also ask the project's teacher "
        "model for a mapping suggestion. The deterministic sniffer "
        "still runs; the LLM proposal joins the ranked hypotheses and "
        "competes on confidence — never overrides them silently.",
    )


class ImportRequest(BaseModel):
    locator: str = Field(
        ...,
        min_length=3,
        description="Source-prefixed locator, e.g. 'jsonl:/tmp/data.jsonl' "
        "or 'csv:./reviews.csv'.",
    )
    mapper_id: str = Field(..., min_length=1)
    field_map: dict[str, Any] = Field(default_factory=dict)
    limit: int | None = Field(default=None, ge=1, le=200_000)
    drop_reasons: list[str] = Field(
        default_factory=list,
        description="Rejection reason codes to silently bulk-drop "
        "(per the bulk-drop UX contract). Rejections in this set "
        "still count in rejection_counts but don't show up in the "
        "rejected_sample.",
    )


class PreviewRequest(ImportRequest):
    sample_cap: int = Field(default=5, ge=1, le=50)


@catalog_router.get("/sources")
async def list_sources() -> dict[str, list[str]]:
    return {"sources": list_registered_sources()}


@catalog_router.get("/mappers")
async def list_mappers() -> dict[str, list[str]]:
    return {"mappers": list_registered_mappers()}


@catalog_router.post("/introspect")
async def introspect(req: IntrospectRequest) -> dict[str, Any]:
    """Sniff the dataset behind ``locator`` and propose a mapping.

    Returns ranked hypotheses + a ``proposal`` block ready to feed into
    ``/preview`` once the user confirms. Per the no-silent-auto-mapping
    rule, callers MUST check ``proposal.needs_force`` and require an
    explicit override when confidence < threshold.
    """

    try:
        return await introspect_locator(
            req.locator,
            sample_size=req.sample_size,
            llm_assist=req.llm_assist,
        )
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"introspect failed: {exc}") from exc


async def _load_project_profile(db: AsyncSession, project_id: int) -> str | None:
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, f"Project {project_id} not found")
    preset = project.dataset_adapter_preset or {}
    if isinstance(preset, dict):
        value = preset.get("task_profile")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


MAX_UPLOAD_BYTES = 512 * 1024 * 1024


@project_router.post("/upload", status_code=201)
async def upload_import_file(
    project_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Stage a browser-uploaded row file (CSV/TSV/JSONL/JSON/XLSX/Parquet)
    under the project and introspect it — the wizard then previews/runs the
    returned ``file:`` locator exactly like a server path. No more typing a
    server file path."""
    await _load_project_profile(db, project_id)
    from app.config import settings
    from app.utils.tabular_io import TABULAR_EXTENSIONS

    original = Path(file.filename or "upload").name
    suffix = Path(original).suffix.lower()
    if suffix not in TABULAR_EXTENSIONS:
        allowed = ", ".join(sorted(TABULAR_EXTENSIONS))
        raise HTTPException(
            400,
            f"'{original}' isn't a row file ({allowed}). Upload PDFs, Word "
            "documents and web pages under Data → Upload documents.",
        )
    staging = Path(settings.DATA_DIR) / "projects" / str(project_id) / "imports"
    staging.mkdir(parents=True, exist_ok=True)
    target = staging / f"{uuid4().hex[:8]}_{original}"
    size = 0
    with target.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, "File is larger than 512 MB.")
            out.write(chunk)
    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "The uploaded file is empty.")

    locator = f"file:{target}"
    try:
        introspection = await introspect_locator(locator)
    except (ValueError, KeyError) as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"Couldn't read '{original}': {exc}") from exc
    return {
        "locator": locator,
        "filename": original,
        "size_bytes": size,
        "introspection": introspection,
    }


@project_router.post("/preview")
async def preview(
    project_id: int,
    req: PreviewRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    task_profile = await _load_project_profile(db, project_id)
    try:
        result = preview_import(
            project_id=project_id,
            project_task_profile=task_profile,
            locator=req.locator,
            mapper_id=req.mapper_id,
            field_map=req.field_map,
            sample_cap=req.sample_cap,
            limit=req.limit,
            drop_reasons=set(req.drop_reasons),
        )
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"preview failed: {exc}") from exc
    return result_to_dict(result)


@project_router.post("/run")
async def run(
    project_id: int,
    req: ImportRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    task_profile = await _load_project_profile(db, project_id)
    try:
        result = await run_import(
            db,
            project_id=project_id,
            project_task_profile=task_profile,
            locator=req.locator,
            mapper_id=req.mapper_id,
            field_map=req.field_map,
            limit=req.limit,
            drop_reasons=set(req.drop_reasons),
        )
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"run failed: {exc}") from exc
    return result_to_dict(result)


# ── Saved configs (Phase G) ──────────────────────────────────────────


class ConfigCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    locator: str = Field(..., min_length=3)
    mapper_id: str = Field(..., min_length=1)
    field_map: dict[str, Any] = Field(default_factory=dict)
    drop_reasons: list[str] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1, le=10_000_000)


_CONFIG_ERROR_CODES = {
    "config_name_required": (400, "Config name is required."),
    "config_name_too_long": (400, "Config name is too long (max 120 chars)."),
    "config_locator_required": (400, "Config locator is required."),
    "config_mapper_id_required": (400, "Config mapper_id is required."),
    "config_name_taken": (
        409,
        "A saved mapping with that name already exists in this project.",
    ),
}


def _translate_config_error(exc: ValueError) -> HTTPException:
    code = str(exc)
    status, detail = _CONFIG_ERROR_CODES.get(code, (400, code))
    return HTTPException(status, detail)


@project_router.get("/configs")
async def list_saved_configs(
    project_id: int, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    rows = await list_configs(db, project_id)
    return {"configs": [config_to_dict(row) for row in rows]}


@project_router.post("/configs", status_code=201)
async def create_saved_config(
    project_id: int,
    req: ConfigCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    # Confirm the project exists before writing the row.
    await _load_project_profile(db, project_id)
    try:
        row = await save_config(
            db,
            project_id=project_id,
            name=req.name,
            description=req.description,
            locator=req.locator,
            mapper_id=req.mapper_id,
            field_map=req.field_map,
            drop_reasons=req.drop_reasons,
            limit=req.limit,
        )
    except ValueError as exc:
        raise _translate_config_error(exc) from exc
    await db.commit()
    return config_to_dict(row)


@project_router.delete("/configs/{config_id}", status_code=204)
async def delete_saved_config(
    project_id: int,
    config_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    deleted = await delete_config(db, project_id, config_id)
    if not deleted:
        raise HTTPException(404, f"Saved mapping {config_id} not found.")
    await db.commit()


@project_router.post("/configs/{config_id}/run")
async def run_saved_config(
    project_id: int,
    config_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    config = await get_config(db, project_id, config_id)
    if config is None:
        raise HTTPException(404, f"Saved mapping {config_id} not found.")
    task_profile = await _load_project_profile(db, project_id)
    try:
        result = await run_from_config(
            db,
            project_id=project_id,
            config=config,
            project_task_profile=task_profile,
        )
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"run failed: {exc}") from exc
    await db.commit()
    return result_to_dict(result)
