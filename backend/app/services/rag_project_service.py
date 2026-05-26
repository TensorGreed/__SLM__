"""RAG-skeleton project service (USER-SUCCESS Epic 7 Phase 7b).

The smallest "RAG-first project" we can offer today:

  * Clone a qa-sft source project's gold set + prepared splits +
    raw source dataset onto a sibling project row.
  * Stamp the new project with
    ``runtime_config = {"rag_first": True, "auto_rag": {"enabled": True}}``
    and link back via ``parent_project_id``.
  * Build the BM25 index immediately (Phase 9b's
    ``build_index_for_project``) so the playground can answer
    queries without a training run — the base model + retrieved
    preamble do the work.

Why this shape:

  * Phase 9 already shipped the retrieval primitive
    (``auto_rag_service``), the playground preamble path
    (``api/training.playground_chat``), and the
    ``qa_with_auto_rag`` target profile. Phase 7b is mostly
    file-copy plumbing on top.
  * Refusing non-qa-sft sources keeps the surface narrow. The
    Phase 7a recommendation routes non-QA projects to
    ``try_prompt_engineering`` instead, so this constraint matches
    the analyzer's gating.
  * No SQL deep-copy. Dataset rows + on-disk files are copied
    directly; downstream artifacts (RawDocument, GoldSetVersion,
    GoldSetRow, DatasetVersion) are *not* copied — a RAG-first
    project doesn't walk the training pipeline, so those pipeline-
    stage tables would surface empty UI either way.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset
from app.models.project import Project, ProjectStatus


# Dataset types we copy over. Synthetic + RawDocument lineage stays
# behind — RAG-first projects don't generate synth nor walk the
# cleaning pipeline.
_CLONED_DATASET_TYPES = (
    "raw",
    "cleaned",
    "gold_dev",
    "gold_test",
    "train",
    "validation",
    "test",
)


class RagCloneError(ValueError):
    """Raised when the clone refuses for a deterministic reason.

    The single-string ``ValueError`` arg encodes the failure mode
    so API layers can map it to a status code without parsing
    free-form text.
    """


async def _pick_unique_name(db: AsyncSession, base_name: str) -> str:
    """``base_name``, ``base_name 2``, ``base_name 3``... until one
    is free. Bounded loop — we cap at 50 attempts to avoid a runaway
    on a pathologically full namespace."""
    candidate = base_name
    for attempt in range(1, 51):
        result = await db.execute(
            select(Project.id).where(Project.name == candidate)
        )
        if result.scalar_one_or_none() is None:
            return candidate
        candidate = f"{base_name} {attempt + 1}"
    raise RagCloneError("name_namespace_exhausted")


def _project_data_dir(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id)


def _rewrite_file_path(
    file_path: str | None,
    src_project_id: int,
    dst_project_id: int,
) -> str | None:
    """Swap the project-id segment of an on-disk file path so a copied
    Dataset row points at the cloned file. Returns ``None`` if the
    input was empty (Dataset.file_path is nullable)."""
    if not file_path:
        return None
    src_marker = f"/projects/{src_project_id}/"
    dst_marker = f"/projects/{dst_project_id}/"
    if src_marker in file_path:
        return file_path.replace(src_marker, dst_marker, 1)
    # Windows path separator as a defensive fallback. We don't ship
    # on Windows but this keeps the function robust to manual
    # data-dir surgery.
    src_win = f"\\projects\\{src_project_id}\\"
    dst_win = f"\\projects\\{dst_project_id}\\"
    if src_win in file_path:
        return file_path.replace(src_win, dst_win, 1)
    # Path doesn't reference the project's data dir at all — leave
    # alone (e.g. a Dataset row whose file_path was set to an
    # external upload location).
    return file_path


def _copy_project_data_tree(src_id: int, dst_id: int) -> dict[str, Any]:
    """Copy the source project's data dir onto the destination's.

    Mirrors only the subdirs a RAG-first project will exercise:
    ``raw/``, ``cleaned/``, ``gold/``, ``prepared/``. The ``auto_rag/``
    index is built fresh by Phase 9b after the clone — never copied,
    since the index file stamps the source recipe + would mis-route
    retrievals against the new project id otherwise.
    """
    src_root = _project_data_dir(src_id)
    dst_root = _project_data_dir(dst_id)
    if not src_root.exists():
        return {"copied_subdirs": [], "reason": "source_data_dir_missing"}
    dst_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for subdir in ("raw", "cleaned", "gold", "prepared"):
        src_sub = src_root / subdir
        if not src_sub.exists() or not src_sub.is_dir():
            continue
        dst_sub = dst_root / subdir
        # Tolerate a re-clone overwriting a stale dst (rare — caller
        # generally picks a fresh name).
        if dst_sub.exists():
            shutil.rmtree(dst_sub)
        shutil.copytree(src_sub, dst_sub)
        copied.append(subdir)
    return {"copied_subdirs": copied}


async def clone_project_for_rag(
    db: AsyncSession,
    source_project_id: int,
    *,
    name_suffix: str = " (RAG)",
) -> Project:
    """Create a sibling project carrying the source's gold set + prepared
    splits forward, configured as a RAG-first project.

    Refuses non-qa-sft sources with
    ``RagCloneError("source_recipe_not_eligible:<recipe_id>")`` —
    a non-QA recipe lands in the Phase 7a recommendation as
    ``try_prompt_engineering`` instead.

    Does NOT commit. The caller wraps in a transaction and commits.
    """
    source: Project | None = await db.get(Project, source_project_id)
    if source is None:
        raise RagCloneError("source_project_not_found")

    selected_recipe: dict[str, Any] = dict(source.selected_recipe or {})
    recipe_id = str(selected_recipe.get("recipe_id") or "").strip()
    if not recipe_id:
        raise RagCloneError("source_recipe_missing")
    if recipe_id != "qa-sft":
        raise RagCloneError(f"source_recipe_not_eligible:{recipe_id}")

    base_name = f"{source.name}{name_suffix}".strip()
    if not base_name:
        raise RagCloneError("name_suffix_produced_empty_name")
    new_name = await _pick_unique_name(db, base_name)

    # Build the runtime_config carefully — preserve any existing
    # runtime_config on the source (rare) but force rag_first + auto_rag
    # on. The auto_rag.enabled mirror is what Phase 9b's
    # playground_chat consumes; rag_first is the dedicated flag the
    # training-start gate + the playground base-model override read.
    source_runtime_config = dict(source.runtime_config or {})
    new_runtime_config: dict[str, Any] = {
        **source_runtime_config,
        "rag_first": True,
        "auto_rag": {"enabled": True},
    }

    new_project = Project(
        name=new_name,
        description=source.description or "",
        status=ProjectStatus.ACTIVE,
        base_model_name=source.base_model_name or "",
        domain_pack_id=source.domain_pack_id,
        domain_profile_id=source.domain_profile_id,
        training_preferred_plan_profile=source.training_preferred_plan_profile,
        target_profile_id="qa_with_auto_rag",
        evaluation_preferred_pack_id=source.evaluation_preferred_pack_id,
        beginner_mode=source.beginner_mode,
        dataset_adapter_preset=dict(source.dataset_adapter_preset or {}),
        gate_policy=dict(source.gate_policy or {}),
        budget_settings=dict(source.budget_settings or {}),
        selected_recipe=selected_recipe,
        runtime_config=new_runtime_config,
        parent_project_id=source.id,
    )
    db.add(new_project)
    await db.flush()  # need new_project.id for file paths

    # Copy on-disk files.
    copy_report = _copy_project_data_tree(source.id, new_project.id)

    # Duplicate Dataset rows with rewritten file paths.
    result = await db.execute(
        select(Dataset).where(Dataset.project_id == source.id)
    )
    cloned_dataset_ids: list[int] = []
    for src_dataset in result.scalars():
        type_value = (
            src_dataset.dataset_type.value
            if hasattr(src_dataset.dataset_type, "value")
            else str(src_dataset.dataset_type)
        )
        if type_value not in _CLONED_DATASET_TYPES:
            continue
        new_file_path = _rewrite_file_path(
            src_dataset.file_path, source.id, new_project.id
        )
        new_dataset = Dataset(
            project_id=new_project.id,
            name=src_dataset.name,
            dataset_type=src_dataset.dataset_type,
            description=src_dataset.description or "",
            record_count=src_dataset.record_count,
            file_path=new_file_path,
            metadata_={
                **dict(src_dataset.metadata_ or {}),
                "cloned_from_dataset_id": src_dataset.id,
                "cloned_from_project_id": source.id,
            },
            is_locked=bool(src_dataset.is_locked),
        )
        db.add(new_dataset)
        await db.flush()
        cloned_dataset_ids.append(new_dataset.id)

    # Build BM25 index immediately — without it the playground can't
    # ground answers in the gold set. Never raises (Phase 9b
    # contract), so a failure here doesn't tear down the clone.
    from app.services.auto_rag_service import build_index_for_project

    index_report = await build_index_for_project(db, new_project.id)

    # Stash observability on the new project's runtime_config so the
    # caller (and downstream debugging) can see what happened. Full
    # reassignment for SQLAlchemy JSON dirty-tracking.
    runtime_config = dict(new_project.runtime_config or {})
    runtime_config["clone_report"] = {
        "source_project_id": source.id,
        "copied_subdirs": copy_report.get("copied_subdirs", []),
        "cloned_dataset_count": len(cloned_dataset_ids),
        "auto_rag_index": index_report,
    }
    new_project.runtime_config = runtime_config

    return new_project


def is_rag_first(project: Project | None) -> bool:
    """True iff this project should bypass training and use base-
    model + auto-RAG retrieval at inference time. Tolerant of
    legacy-bool config shapes (mirrors the same pattern used by
    coach_service + post_eval_decision_engine_service)."""
    if project is None:
        return False
    cfg = project.runtime_config
    if not isinstance(cfg, dict):
        return False
    flag = cfg.get("rag_first")
    if isinstance(flag, bool):
        return flag
    return False
