"""Duplicate a project: the same documents, rows, answer key and prepared
split under a new project — to try another base model (or setting) without
overwriting the original's measurements.

Why a copy and not "change the base model in place": the base + passages
comparison, the lift-check baselines and the passages-vs-fine-tune verdict
are measured on the project's base model. Switching it in place replaces
the bar the earlier runs were compared against (2026-10-19: switching a
3B project to 1.5B overwrote its base + passages record, and the verdict
then compared the 1.5B bar with the 3B fine-tune). A duplicate keeps both.

Copied: ``raw/``, ``cleaned/``, ``gold/``, ``prepared/`` (incl. its
versions — the test examples stay byte-identical) and ``synthetic/``, every
Dataset row with its file path rewritten, the recipe, task shape and the
runtime settings (retrieval setting included). NOT copied: runs,
evaluations, ``auto_rag/`` (comparisons are measured per base model; the
indexes are rebuilt on demand), jobs, secrets (cloud keys stay with the
project they were saved on), and the RAG-sibling flags.
"""

from __future__ import annotations

import shutil
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, RawDocument
from app.models.project import Project, ProjectStatus
from app.services.rag_project_service import (
    RagCloneError,
    _pick_unique_name,
    _project_data_dir,
    _rewrite_file_path,
)

COPIED_SUBDIRS = ("raw", "cleaned", "gold", "prepared", "synthetic")
# runtime_config keys that describe the source's own history, not settings.
_DROPPED_RUNTIME_KEYS = ("rag_first", "clone_report", "plan_refinement", "auto_rag_corpus_reason")


def copy_data_tree(src_id: int, dst_id: int, subdirs: tuple[str, ...] = COPIED_SUBDIRS) -> list[str]:
    src_root, dst_root = _project_data_dir(src_id), _project_data_dir(dst_id)
    copied: list[str] = []
    if not src_root.exists():
        return copied
    dst_root.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        src_sub = src_root / subdir
        if not src_sub.is_dir():
            continue
        dst_sub = dst_root / subdir
        if dst_sub.exists():
            shutil.rmtree(dst_sub)
        shutil.copytree(src_sub, dst_sub)
        copied.append(subdir)
    return copied


async def duplicate_project(
    db: AsyncSession,
    source_project_id: int,
    *,
    name: str | None = None,
    base_model_name: str | None = None,
) -> Project:
    """Create the copy (see module doc). Does NOT commit."""
    source = await db.get(Project, source_project_id)
    if source is None:
        raise RagCloneError("source_project_not_found")
    new_name = await _pick_unique_name(db, (name or f"{source.name} (copy)").strip()[:240] or "Project copy")
    runtime_config = {k: v for k, v in dict(source.runtime_config or {}).items() if k not in _DROPPED_RUNTIME_KEYS}
    runtime_config["duplicated_from"] = {"project_id": source.id, "base_model": source.base_model_name}
    project = Project(
        name=new_name,
        description=source.description or "",
        status=ProjectStatus.ACTIVE,
        base_model_name=(base_model_name or source.base_model_name or "").strip(),
        domain_pack_id=source.domain_pack_id,
        domain_profile_id=source.domain_profile_id,
        training_preferred_plan_profile=source.training_preferred_plan_profile,
        target_profile_id=source.target_profile_id,
        evaluation_preferred_pack_id=source.evaluation_preferred_pack_id,
        beginner_mode=source.beginner_mode,
        dataset_adapter_preset=dict(source.dataset_adapter_preset or {}),
        gate_policy=dict(source.gate_policy or {}),
        budget_settings=dict(source.budget_settings or {}),
        selected_recipe=dict(source.selected_recipe or {}),
        runtime_config=runtime_config,
        parent_project_id=source.id,
    )
    db.add(project)
    await db.flush()
    copied = copy_data_tree(source.id, project.id)
    rows = await db.execute(select(Dataset).where(Dataset.project_id == source.id))
    count = 0
    new_by_old: dict[int, Dataset] = {}
    for src in rows.scalars():
        new_by_old[src.id] = Dataset(
            project_id=project.id,
            name=src.name,
            dataset_type=src.dataset_type,
            description=src.description or "",
            record_count=src.record_count,
            file_path=_rewrite_file_path(src.file_path, source.id, project.id),
            metadata_={**dict(src.metadata_ or {}), "cloned_from_dataset_id": src.id, "cloned_from_project_id": source.id},
            is_locked=bool(src.is_locked),
        )
        db.add(new_by_old[src.id])
        count += 1
    await db.flush()
    # The uploaded documents (so Data → Sources lists them in the copy). The
    # cleaned passages keep the source's document ids as their identifiers.
    documents = 0
    if new_by_old:
        docs = await db.execute(select(RawDocument).where(RawDocument.dataset_id.in_(list(new_by_old))))
        for doc in docs.scalars():
            db.add(RawDocument(
                dataset_id=new_by_old[doc.dataset_id].id,
                filename=doc.filename,
                file_type=doc.file_type,
                file_path=_rewrite_file_path(doc.file_path, source.id, project.id) or doc.file_path,
                file_size_bytes=doc.file_size_bytes,
                source=doc.source,
                sensitivity=doc.sensitivity,
                license_info=doc.license_info,
                status=doc.status,
                quality_score=doc.quality_score,
                chunk_count=doc.chunk_count,
                metadata_={**dict(doc.metadata_ or {}), "cloned_from_document_id": doc.id},
            ))
            documents += 1
        await db.flush()
    cfg: dict[str, Any] = dict(project.runtime_config or {})
    cfg["duplicated_from"] = {**cfg["duplicated_from"], "copied_subdirs": copied, "datasets": count, "documents": documents}
    project.runtime_config = cfg
    return project
