"""Demo project seeder (newbie UX Phase 3).

Materialises a real project from a pre-curated bundle of sample data,
a hand-labelled gold set, and project metadata. The goal: cut
time-to-first-result for a new ML engineer from "find a dataset, label
it, configure the pipeline" to one click.

Bundles live under ``backend/data/demo_samples/<slug>/`` and ship
three files:

- ``manifest.json`` — slug, display name, headline, description,
  task profile, target profile, suggested autopilot brief, input /
  output field names.
- ``<dataset_filename>`` — CSV of source rows (e.g. ``tickets.csv``).
- ``gold.jsonl`` — JSONL with ``{key, input, expected, rationale}``
  rows for the locked gold set.

The seeder is idempotent: re-running with the same slug returns the
existing project if its name is already taken. That keeps the "Try a
demo" tile click safe even on a misbehaving frontend.

Stable reason codes (raised as ``ValueError`` from the API path):

- ``demo_slug_unknown:<slug>`` (404) — no bundle dir for that slug.
- ``demo_manifest_invalid:<slug>`` (400) — manifest.json missing or
  malformed.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.models.gold_set_annotation import (
    GoldSetRow,
    GoldSetRowStatus,
    GoldSetVersion,
    GoldSetVersionStatus,
)
from app.models.project import PipelineStage, Project, ProjectStatus


BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DEMO_SAMPLES_DIR = BACKEND_DIR / "data" / "demo_samples"


def _resolve_demo_dir(slug: str) -> Path:
    """Resolve a demo slug to its sample-data directory, with safety check."""

    safe = "".join(ch for ch in slug if ch.isalnum() or ch in "-_").strip("-_")
    if not safe or safe != slug:
        raise ValueError(f"demo_slug_unknown:{slug}")
    candidate = (DEMO_SAMPLES_DIR / safe).resolve()
    if not candidate.is_dir() or DEMO_SAMPLES_DIR.resolve() not in candidate.parents:
        raise ValueError(f"demo_slug_unknown:{slug}")
    return candidate


def _load_manifest(slug: str) -> dict[str, Any]:
    demo_dir = _resolve_demo_dir(slug)
    manifest_path = demo_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"demo_manifest_invalid:{slug}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"demo_manifest_invalid:{slug}:{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"demo_manifest_invalid:{slug}:not_a_dict")
    payload["_dir"] = str(demo_dir)
    return payload


def list_demo_archetypes() -> list[dict[str, Any]]:
    """Catalog of available demos, read fresh from disk on every call."""

    if not DEMO_SAMPLES_DIR.is_dir():
        return []
    archetypes: list[dict[str, Any]] = []
    for child in sorted(DEMO_SAMPLES_DIR.iterdir()):
        if not child.is_dir():
            continue
        try:
            manifest = _load_manifest(child.name)
        except ValueError:
            continue
        archetypes.append(
            {
                "slug": manifest.get("slug") or child.name,
                "name": manifest.get("name") or child.name.title(),
                "headline": manifest.get("headline") or "",
                "description": manifest.get("description") or "",
                "task_profile": manifest.get("task_profile") or "",
                "target_profile": manifest.get("target_profile") or "",
                "suggested_brief": manifest.get("suggested_brief") or "",
            }
        )
    return archetypes


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _read_gold_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


async def _find_project_by_name(db: AsyncSession, name: str) -> Project | None:
    result = await db.execute(select(Project).where(Project.name == name))
    return result.scalar_one_or_none()


async def _next_gold_version(db: AsyncSession, gold_set_id: int) -> int:
    row = await db.execute(
        select(func.max(GoldSetVersion.version)).where(GoldSetVersion.gold_set_id == gold_set_id)
    )
    current = row.scalar_one_or_none()
    return (current or 0) + 1


async def seed_demo_project(
    db: AsyncSession,
    slug: str,
    *,
    actor_user_id: int | None = None,
) -> tuple[Project, dict[str, Any]]:
    """Create (or return) the demo project for ``slug``. Idempotent."""

    manifest = _load_manifest(slug)
    demo_dir = Path(manifest["_dir"])

    project_name = str(manifest.get("name") or slug)
    existing = await _find_project_by_name(db, project_name)
    if existing is not None:
        return existing, {"slug": slug, "created": False, "project_id": existing.id}

    description = str(manifest.get("description") or "")
    suggested_brief = str(manifest.get("suggested_brief") or "")
    target_profile = str(manifest.get("target_profile") or "vllm_server")
    plan_profile = str(manifest.get("training_preferred_plan_profile") or "balanced")
    eval_pack = manifest.get("evaluation_preferred_pack_id") or None

    project = Project(
        name=project_name,
        description=description,
        status=ProjectStatus.ACTIVE,
        pipeline_stage=PipelineStage.GOLD_SET,
        beginner_mode=True,
        target_profile_id=target_profile,
        training_preferred_plan_profile=plan_profile,
        evaluation_preferred_pack_id=eval_pack,
        dataset_adapter_preset={
            "demo_slug": slug,
            "suggested_brief": suggested_brief,
        },
    )
    db.add(project)
    await db.flush()  # populate project.id

    # 1) Cleaned source dataset ------------------------------------------------
    csv_filename = str(manifest.get("dataset_filename") or "data.csv")
    csv_src = demo_dir / csv_filename
    if not csv_src.exists():
        raise ValueError(f"demo_manifest_invalid:{slug}:missing_dataset_file")

    raw_dir = settings.DATA_DIR / "projects" / str(project.id) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dst = raw_dir / csv_filename
    shutil.copy(csv_src, csv_dst)

    csv_rows = _read_csv_rows(csv_src)
    dataset = Dataset(
        project_id=project.id,
        name=f"{project_name} · source",
        dataset_type=DatasetType.CLEANED,
        description="Pre-loaded source rows for the demo project.",
        record_count=len(csv_rows),
        file_path=str(csv_dst),
        metadata_={
            "demo_slug": slug,
            "input_field": manifest.get("dataset_input_field"),
            "output_field": manifest.get("dataset_output_field"),
        },
        is_locked=True,
    )
    db.add(dataset)
    await db.flush()

    # One RawDocument per CSV row keeps the per-stage tabs realistic.
    for idx, row in enumerate(csv_rows):
        payload = json.dumps(row, ensure_ascii=False)
        db.add(
            RawDocument(
                dataset_id=dataset.id,
                filename=f"{csv_filename}#row-{idx + 1}",
                file_type="csv-row",
                file_path=str(csv_dst),
                file_size_bytes=len(payload.encode("utf-8")),
                source="demo",
                sensitivity="public",
                status=DocumentStatus.ACCEPTED,
                metadata_={"row_index": idx, "row": row},
            )
        )

    # 2) Locked gold set -------------------------------------------------------
    gold_filename = str(manifest.get("gold_filename") or "gold.jsonl")
    gold_src = demo_dir / gold_filename
    if not gold_src.exists():
        raise ValueError(f"demo_manifest_invalid:{slug}:missing_gold_file")
    gold_rows = _read_gold_rows(gold_src)
    gold_dataset = Dataset(
        project_id=project.id,
        name=f"{project_name} · gold",
        dataset_type=DatasetType.GOLD_DEV,
        description="Hand-labelled gold-set rows for the demo project.",
        record_count=len(gold_rows),
        file_path=str(gold_src),
        metadata_={"demo_slug": slug, "frozen": True},
        is_locked=True,
    )
    db.add(gold_dataset)
    await db.flush()

    version_number = await _next_gold_version(db, gold_dataset.id)
    gold_version = GoldSetVersion(
        gold_set_id=gold_dataset.id,
        version=version_number,
        status=GoldSetVersionStatus.LOCKED,
        notes=f"Seeded from demo bundle '{slug}'.",
        created_by_user_id=actor_user_id,
    )
    db.add(gold_version)
    await db.flush()

    for row in gold_rows:
        db.add(
            GoldSetRow(
                gold_set_id=gold_dataset.id,
                version_id=gold_version.id,
                source_row_key=str(row.get("key") or "")[:128] or None,
                source_dataset_id=dataset.id,
                input=row.get("input") or {},
                expected=row.get("expected") or {},
                rationale=str(row.get("rationale") or ""),
                labels=row.get("labels") or {},
                status=GoldSetRowStatus.APPROVED,
                reviewer_id=actor_user_id,
            )
        )

    await db.flush()

    summary = {
        "slug": slug,
        "created": True,
        "project_id": project.id,
        "project_name": project.name,
        "source_dataset_id": dataset.id,
        "source_row_count": len(csv_rows),
        "gold_set_id": gold_dataset.id,
        "gold_version_id": gold_version.id,
        "gold_row_count": len(gold_rows),
        "suggested_brief": suggested_brief,
    }
    return project, summary


__all__ = ["list_demo_archetypes", "seed_demo_project"]
