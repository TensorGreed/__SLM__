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

from app.models.data_source import DataSource
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
    if "synthetic" in copied:
        _drop_copied_job_ids(dst_id)
    return copied


def _drop_copied_job_ids(project_id: int) -> None:
    """A copied project's synthetic rows (and review-assist verdicts) name Jobs of the SOURCE project:
    drop ``synth_job_id`` / null the verdict's ``job_id`` so the copy's review queue doesn't group or
    filter by a Job that isn't its own. ``synth_backend`` / ``synth_batch`` (how it was written) stay."""
    import json

    from app.config import settings

    root = settings.DATA_DIR / "projects" / str(project_id) / "synthetic"
    rows_path = root / "synthetic.jsonl"
    if rows_path.exists():
        out: list[str] = []
        changed = False
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if '"synth_job_id"' in line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = None
                if isinstance(row, dict) and "synth_job_id" in row:
                    row.pop("synth_job_id")
                    line = json.dumps(row, ensure_ascii=False)
                    changed = True
            out.append(line)
        if changed:
            rows_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    sidecar = root / "review_assist.json"
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for entry in data.values():
                    # Old format: the verdict itself; new format: {judge: verdict}.
                    for e in ([entry] if isinstance(entry, dict) and "verdict" in entry else list(entry.values()) if isinstance(entry, dict) else []):
                        if isinstance(e, dict) and e.get("job_id") is not None:
                            e["job_id"] = None
                        # Entries a re-run replaced (``history``) name the source's Jobs too.
                        for h in (e.get("history") or []) if isinstance(e, dict) else []:
                            if isinstance(h, dict) and h.get("job_id") is not None:
                                h["job_id"] = None
                sidecar.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError):
            pass


def _remap_copied_source_ids(project_id: int, source_map: dict[int, int]) -> None:
    """The copied ``synthetic.jsonl`` rows and prepared manifest still carry the
    source project's ``data_source_id``s: point them at the copy's sources
    (a row whose source was not copied loses the key)."""
    import json

    from app.config import settings

    root = settings.DATA_DIR / "projects" / str(project_id)
    rows_path = root / "synthetic" / "synthetic.jsonl"
    if rows_path.exists():
        out: list[str] = []
        changed = False
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if '"data_source_id"' in line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    out.append(line)
                    continue
                if isinstance(row, dict) and "data_source_id" in row:
                    new_id = source_map.get(row["data_source_id"]) if isinstance(row["data_source_id"], int) else None
                    if new_id is None:
                        row.pop("data_source_id")
                    else:
                        row["data_source_id"] = new_id
                    line = json.dumps(row, ensure_ascii=False)
                    changed = True
            out.append(line)
        if changed:
            rows_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    manifest_path = root / "prepared" / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            by_source = (manifest.get("provenance") or {}).get("by_source")
            if isinstance(by_source, dict):
                manifest["provenance"]["by_source"] = {
                    str(source_map[int(k)]): v for k, v in by_source.items() if str(k).isdigit() and int(k) in source_map
                }
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except (OSError, json.JSONDecodeError, ValueError):
            pass


def _remap_copied_document_ids(project_id: int, doc_map: dict[int, int]) -> None:
    """The copied cleaned / synthetic / gold / prepared jsonl files (and the
    ``*.chunks.jsonl`` sidecars) name the source project's document ids in
    ``source_document_id`` / ``_source_document_id``: point them at the copy's
    documents, so an eval / rag mark on a copied document still reaches its
    rows. An id with no copied document is left as it was."""
    import json

    from app.config import settings

    if not doc_map:
        return
    root = settings.DATA_DIR / "projects" / str(project_id)
    for sub in ("cleaned", "synthetic", "gold", "prepared", "raw"):
        base = root / sub
        if not base.is_dir():
            continue
        for path in base.rglob("*.jsonl"):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            changed = False
            out: list[str] = []
            for line in lines:
                if "source_document_id" in line:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        row = None
                    if isinstance(row, dict):
                        hit = False
                        ids = row.get("source_document_ids")
                        if isinstance(ids, list) and any(isinstance(i, int) and i in doc_map for i in ids):
                            row["source_document_ids"] = [doc_map.get(i, i) if isinstance(i, int) else i for i in ids]
                            hit = True
                        for key in ("source_document_id", "_source_document_id"):
                            val = row.get(key)
                            if isinstance(val, int) and not isinstance(val, bool) and val in doc_map:
                                row[key] = doc_map[val]
                                hit = True
                        if hit:
                            line = json.dumps(row, ensure_ascii=False)
                            changed = True
                out.append(line)
            if changed:
                path.write_text("\n".join(out) + "\n", encoding="utf-8")


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
    # The licence & provenance ledger: same sources, new ids; documents follow.
    source_map: dict[int, int] = {}
    old_sources = await db.execute(select(DataSource).where(DataSource.project_id == source.id).order_by(DataSource.id))
    for old in old_sources.scalars():
        clone = DataSource(
            project_id=project.id, source_kind=old.source_kind, title=old.title, locator=old.locator,
            source_ref=old.source_ref, licence=old.licence, licence_verified=old.licence_verified,
            licence_verified_by=old.licence_verified_by, licence_verified_at=old.licence_verified_at,
            retrieved_at=old.retrieved_at, usage=old.usage,
            metadata_={**dict(old.metadata_ or {}), "cloned_from_source_id": old.id},
        )
        db.add(clone)
        await db.flush()
        source_map[int(old.id)] = int(clone.id)
    _remap_copied_source_ids(project.id, source_map)
    documents = 0
    if new_by_old:
        docs = await db.execute(select(RawDocument).where(RawDocument.dataset_id.in_(list(new_by_old))))
        pairs: list[tuple[int, RawDocument]] = []
        for doc in docs.scalars():
            clone_doc = RawDocument(
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
                data_source_id=source_map.get(int(doc.data_source_id)) if doc.data_source_id is not None else None,
                source_url=doc.source_url,
                source_ref=doc.source_ref,
                retrieved_at=doc.retrieved_at,
            )
            db.add(clone_doc)
            pairs.append((int(doc.id), clone_doc))
            documents += 1
        await db.flush()
        _remap_copied_document_ids(project.id, {old: int(new.id) for old, new in pairs})
    cfg: dict[str, Any] = dict(project.runtime_config or {})
    cfg["duplicated_from"] = {**cfg["duplicated_from"], "copied_subdirs": copied, "datasets": count, "documents": documents}
    project.runtime_config = cfg
    return project
