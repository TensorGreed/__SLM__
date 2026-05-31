"""D3 — safe auto-fix engine for the Data Health Report.

Three transforms, each idempotent, each predictable:

- ``drop_failed_docs`` — delete RawDocument rows with ``status=ERROR``
  plus their on-disk artefacts. Failed parses have no extracted text;
  they're already useless. Returns the count + dropped filenames so
  the UI can render a "we dropped these N files" summary.
- ``dedupe_duplicate_docs`` — group ACCEPTED docs by their
  ``metadata_.text_hash`` (computed during cleaning); for each group
  of >1, keep the lowest-id occurrence and delete the rest. Same
  cleanup as drop_failed_docs.
- ``redact_pii`` — re-run ``clean_document`` with ``redact=True`` on
  every doc that has ``metadata_.pii_findings`` populated but
  ``redact_pii`` flag not set. Cleaning is itself idempotent; this
  just re-renders the cleaned text with PII replaced by ``[REDACTED]``.

D3 ships *safe* transforms only — no data fabrication, no destructive
operations beyond removing already-broken or already-redundant rows.
The riskier transforms (truncation, drop-low-quality, label
canonicalisation) land in D4 with preview-diff confirmation.

Every fix returns a structured ``AutofixResult`` (applied count,
summary string, per-fix context) so the UI can render the post-fix
toast accurately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument


# Fix kinds the service understands. Other kinds raise ``ValueError``
# at the dispatcher so a typo in the API surfaces as a 400, not a
# silent no-op.
SUPPORTED_FIX_KINDS: tuple[str, ...] = (
    "drop_failed_docs",
    "dedupe_duplicate_docs",
    "redact_pii",
    # D4 — label canonicalisation. Merges case/whitespace duplicates
    # in classification gold-set labels (e.g. "Positive" + "positive"
    # → "positive") with the most-common variant as canonical.
    "canonicalise_labels",
)


class AutofixResult(TypedDict, total=False):
    """The shape every autofix-apply function returns.

    ``applied_count`` is the user-facing "N things changed" number;
    ``summary`` is one short sentence we'll surface as the toast.
    ``details`` carries per-fix context (e.g. dropped filenames) so
    the panel can render a richer post-action UI without a second
    API call.
    """
    fix_kind: str
    applied_count: int
    summary: str
    details: dict[str, Any]


class AutofixPreview(TypedDict, total=False):
    """The shape every autofix-preview function returns.

    Same fields as ``AutofixResult`` plus ``items`` — a list of
    per-row records the modal renders so the user can see exactly
    what will change before clicking Apply. The shape of each item
    varies per fix kind (filename for drops, merge-map entry for
    canonicalisation) but every item has a ``kind`` discriminator
    so the frontend can render a heterogeneous list.

    ``would_apply_count`` is the planned-changes count; it should
    match the ``applied_count`` the subsequent apply call returns
    in the steady-state (modulo race conditions if the user is
    mutating data in another tab between preview and apply).
    """
    fix_kind: str
    would_apply_count: int
    summary: str
    details: dict[str, Any]
    items: list[dict[str, Any]]
    safe_to_apply: bool


# ─────────────────────────────────────────────────────────────────────
# Helpers — file cleanup that mirrors the existing
# ingestion_service.delete_document pattern (delete the raw file +
# the extracted-text sidecar + cleaned + chunks). Keeping in sync
# means autofix doesn't leave orphan files behind.
# ─────────────────────────────────────────────────────────────────────


def _remove_doc_artifacts(doc: RawDocument) -> None:
    """Best-effort delete of every file on disk a doc owns. Each
    unlink is wrapped because the file may have been deleted manually
    or moved — the autofix shouldn't error mid-loop on a stale path."""
    if doc.file_path:
        path = Path(doc.file_path)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        # Extracted-text sidecar produced by ingestion processing.
        extracted = path.with_suffix(".extracted.txt")
        if extracted.exists():
            try:
                extracted.unlink()
            except OSError:
                pass
        # Cleaned + chunks sidecars produced by cleaning.
        for sidecar_suffix in (".cleaned.txt", ".chunks.jsonl"):
            sidecar = path.with_suffix(sidecar_suffix)
            if sidecar.exists():
                try:
                    sidecar.unlink()
                except OSError:
                    pass


async def _load_raw_docs(
    db: AsyncSession, project_id: int
) -> list[RawDocument]:
    """Every RawDocument under the project's RAW dataset."""
    result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.RAW,
        )
    )
    return list(result.scalars())


async def _decrement_dataset_count(
    db: AsyncSession, project_id: int, by: int
) -> None:
    """Pull the project's RAW dataset and adjust ``record_count`` by
    ``by`` (signed). Negative values shrink the count after a drop."""
    if by == 0:
        return
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.RAW,
        )
    )
    ds = result.scalar_one_or_none()
    if ds is None:
        return
    ds.record_count = max(0, int(ds.record_count or 0) + by)


# ─────────────────────────────────────────────────────────────────────
# Preview helpers — read-only selection of "what would change". Each
# returns an AutofixPreview the panel renders before the user
# commits. Every apply function below mirrors the same selection
# logic so the apply step touches the same rows the preview listed.
# ─────────────────────────────────────────────────────────────────────


async def _preview_drop_failed_docs(
    db: AsyncSession, project_id: int
) -> AutofixPreview:
    docs = await _load_raw_docs(db, project_id)
    failed = [d for d in docs if d.status == DocumentStatus.ERROR]
    items = [
        {
            "kind": "document",
            "id": int(d.id),
            "filename": d.filename or "",
            "file_type": d.file_type or "",
            "error": (d.metadata_ or {}).get("error") or "",
        }
        for d in failed
    ]
    return {
        "fix_kind": "drop_failed_docs",
        "would_apply_count": len(failed),
        "summary": (
            f"Would drop {len(failed)} failed document"
            f"{'s' if len(failed) != 1 else ''}."
            if failed
            else "No failed documents to drop."
        ),
        "details": {},
        "items": items,
        "safe_to_apply": True,
    }


async def _preview_dedupe_duplicate_docs(
    db: AsyncSession, project_id: int
) -> AutofixPreview:
    docs = await _load_raw_docs(db, project_id)
    groups: dict[str, list[RawDocument]] = {}
    for doc in docs:
        if doc.status != DocumentStatus.ACCEPTED:
            continue
        text_hash = (doc.metadata_ or {}).get("text_hash")
        if not isinstance(text_hash, str) or not text_hash:
            continue
        groups.setdefault(text_hash, []).append(doc)

    items: list[dict[str, Any]] = []
    drop_count = 0
    group_count = 0
    for text_hash, group in groups.items():
        if len(group) <= 1:
            continue
        group_count += 1
        group.sort(key=lambda d: int(d.id))
        keep = group[0]
        drops = group[1:]
        items.append({
            "kind": "dedup_group",
            "text_hash": text_hash[:12],
            "keep": {"id": int(keep.id), "filename": keep.filename or ""},
            "drop": [
                {"id": int(d.id), "filename": d.filename or ""}
                for d in drops
            ],
        })
        drop_count += len(drops)

    return {
        "fix_kind": "dedupe_duplicate_docs",
        "would_apply_count": drop_count,
        "summary": (
            f"Would drop {drop_count} duplicate document"
            f"{'s' if drop_count != 1 else ''} "
            f"across {group_count} group{'s' if group_count != 1 else ''}; "
            f"the lowest-id occurrence in each group is kept."
            if drop_count
            else "No duplicate documents to drop."
        ),
        "details": {"group_count": group_count},
        "items": items,
        "safe_to_apply": True,
    }


async def _preview_redact_pii(
    db: AsyncSession, project_id: int
) -> AutofixPreview:
    """Preview the PII redact fix.

    Mirrors the apply path's recipe-aware guard — if the project's
    recipe is structured_extraction, ``safe_to_apply`` is ``False``
    and the items list explains why. The frontend uses
    ``safe_to_apply`` to grey out the Apply button so the user can't
    accidentally destroy training data even when the modal opens.
    """
    from app.models.project import Project

    project = await db.get(Project, project_id)
    task_profile: str | None = None
    if project is not None:
        selected = project.selected_recipe or {}
        recipe_id = selected.get("recipe_id") if isinstance(selected, dict) else None
        if recipe_id:
            try:
                from app.services.recipe_service import get_recipe
                recipe = get_recipe(recipe_id)
                task_profile = getattr(recipe, "task_profile", None) if recipe else None
            except Exception:
                task_profile = None

    if task_profile == "structured_extraction":
        return {
            "fix_kind": "redact_pii",
            "would_apply_count": 0,
            "summary": (
                "PII redaction is unsafe for span-extraction recipes — "
                "the model needs PII in source documents to learn what "
                "to detect. Auto-redaction would destroy the training "
                "signal."
            ),
            "details": {
                "task_profile": task_profile,
                "blocked_reason": "span_extraction_needs_pii",
            },
            "items": [],
            "safe_to_apply": False,
        }

    docs = await _load_raw_docs(db, project_id)
    targets = []
    for doc in docs:
        meta = doc.metadata_ or {}
        findings = meta.get("pii_findings") or []
        if not isinstance(findings, list) or not findings:
            continue
        if meta.get("redact_pii"):
            continue
        if not meta.get("cleaned_path"):
            continue
        targets.append((doc, len(findings)))

    items = [
        {
            "kind": "pii_doc",
            "id": int(doc.id),
            "filename": doc.filename or "",
            "pii_findings": count,
        }
        for doc, count in targets
    ]
    total_findings = sum(c for _, c in targets)
    return {
        "fix_kind": "redact_pii",
        "would_apply_count": len(targets),
        "summary": (
            f"Would re-clean {len(targets)} document"
            f"{'s' if len(targets) != 1 else ''} with PII redaction "
            f"enabled, masking {total_findings} finding"
            f"{'s' if total_findings != 1 else ''} total."
            if targets
            else "No documents with unredacted PII to process."
        ),
        "details": {"total_findings": total_findings},
        "items": items,
        "safe_to_apply": True,
    }


async def _preview_canonicalise_labels(
    db: AsyncSession, project_id: int
) -> AutofixPreview:
    """D4 — merge case/whitespace label duplicates in classification gold.

    Picks the **most-common variant** as the canonical form for each
    group (ties broken by the variant that appears first in lexical
    order). Returns the merge map so the panel renders
    'Positive (3) + POSITIVE (1) → positive (15)' style rows.
    """
    from app.models.dataset import Dataset, DatasetType
    from app.models.project import Project
    from app.services.trainability_forecast_service import (
        _extract_classification_label,
        _load_gold_rows,
    )

    # Only fire for classification recipes — for other shapes the
    # concept of "label fragmentation" doesn't apply.
    project = await db.get(Project, project_id)
    task_profile: str | None = None
    if project is not None:
        selected = project.selected_recipe or {}
        recipe_id = selected.get("recipe_id") if isinstance(selected, dict) else None
        if recipe_id:
            try:
                from app.services.recipe_service import get_recipe
                recipe = get_recipe(recipe_id)
                task_profile = getattr(recipe, "task_profile", None) if recipe else None
            except Exception:
                task_profile = None
    if task_profile != "classification":
        return {
            "fix_kind": "canonicalise_labels",
            "would_apply_count": 0,
            "summary": (
                "Label canonicalisation only applies to classification "
                "recipes (this project is "
                f"{task_profile or 'unknown'})."
            ),
            "details": {"task_profile": task_profile},
            "items": [],
            "safe_to_apply": False,
        }

    gold_rows = await _load_gold_rows(db, project_id)
    label_counts: dict[str, int] = {}
    for row in gold_rows:
        label = _extract_classification_label(row)
        if label is None:
            continue
        label_counts[label] = label_counts.get(label, 0) + 1

    # Group by normalised form (lowercase + collapsed whitespace).
    import re as _re
    norm_re = _re.compile(r"\s+")

    def _normalise(label: str) -> str:
        return norm_re.sub(" ", label.strip().lower())

    groups: dict[str, list[tuple[str, int]]] = {}
    for label, count in label_counts.items():
        groups.setdefault(_normalise(label), []).append((label, count))

    items: list[dict[str, Any]] = []
    rows_touched = 0
    for normalised, variants in groups.items():
        if len(variants) <= 1:
            continue
        # Canonical = most common, ties broken alphabetically by the
        # original variant string so the choice is deterministic.
        variants.sort(key=lambda v: (-v[1], v[0]))
        canonical = variants[0][0]
        affected = sum(c for v, c in variants if v != canonical)
        rows_touched += affected
        items.append({
            "kind": "label_merge",
            "canonical": canonical,
            "canonical_count": variants[0][1],
            "merge_in": [
                {"label": v, "count": c} for v, c in variants[1:]
            ],
            "rows_touched": affected,
        })

    # Find the gold dataset file path for the apply step — we record
    # it now so the apply can rewrite the file without re-scanning.
    gold_dataset_result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                [DatasetType.GOLD_DEV, DatasetType.GOLD_TEST]
            ),
        )
    )
    gold_paths = [
        str(ds.file_path)
        for ds in gold_dataset_result.scalars()
        if ds.file_path
    ]

    return {
        "fix_kind": "canonicalise_labels",
        "would_apply_count": rows_touched,
        "summary": (
            f"Would merge {len(items)} fragmented label group"
            f"{'s' if len(items) != 1 else ''}, "
            f"rewriting {rows_touched} gold row"
            f"{'s' if rows_touched != 1 else ''}."
            if items
            else "No fragmented labels detected."
        ),
        "details": {
            "merge_group_count": len(items),
            "gold_files": gold_paths,
        },
        "items": items,
        "safe_to_apply": True,
    }


# ─────────────────────────────────────────────────────────────────────
# The three D3 autofixes.
# ─────────────────────────────────────────────────────────────────────


async def _drop_failed_docs(
    db: AsyncSession, project_id: int
) -> AutofixResult:
    docs = await _load_raw_docs(db, project_id)
    failed = [d for d in docs if d.status == DocumentStatus.ERROR]
    dropped_filenames: list[str] = []
    for doc in failed:
        dropped_filenames.append(doc.filename or "")
        _remove_doc_artifacts(doc)
        await db.delete(doc)
    await _decrement_dataset_count(db, project_id, -len(failed))
    return {
        "fix_kind": "drop_failed_docs",
        "applied_count": len(failed),
        "summary": (
            f"Dropped {len(failed)} failed document{'s' if len(failed) != 1 else ''}."
            if failed
            else "No failed documents to drop."
        ),
        "details": {"dropped_filenames": dropped_filenames},
    }


async def _dedupe_duplicate_docs(
    db: AsyncSession, project_id: int
) -> AutofixResult:
    docs = await _load_raw_docs(db, project_id)
    # Group ACCEPTED docs with a populated text_hash. Other docs
    # (pending, error, no hash) aren't considered duplicates we can
    # confidently dedupe.
    groups: dict[str, list[RawDocument]] = {}
    for doc in docs:
        if doc.status != DocumentStatus.ACCEPTED:
            continue
        text_hash = (doc.metadata_ or {}).get("text_hash")
        if not isinstance(text_hash, str) or not text_hash:
            continue
        groups.setdefault(text_hash, []).append(doc)

    dropped: list[str] = []
    group_count = 0
    for text_hash, group in groups.items():
        if len(group) <= 1:
            continue
        group_count += 1
        # Keep lowest-id doc (the first one ingested); delete the rest.
        group.sort(key=lambda d: int(d.id))
        for dup in group[1:]:
            dropped.append(dup.filename or "")
            _remove_doc_artifacts(dup)
            await db.delete(dup)

    await _decrement_dataset_count(db, project_id, -len(dropped))
    return {
        "fix_kind": "dedupe_duplicate_docs",
        "applied_count": len(dropped),
        "summary": (
            f"Dropped {len(dropped)} duplicate document{'s' if len(dropped) != 1 else ''} "
            f"across {group_count} dedup group{'s' if group_count != 1 else ''}."
            if dropped
            else "No duplicate documents to drop."
        ),
        "details": {
            "dropped_filenames": dropped,
            "group_count": group_count,
        },
    }


async def _redact_pii(
    db: AsyncSession, project_id: int
) -> AutofixResult:
    """Re-clean every doc that has PII findings but wasn't redacted.

    Imports clean_document lazily because the cleaning service pulls
    in a lot of regex + chunking helpers we don't otherwise need on
    the request path.

    Defence in depth: refuses to run for ``structured_extraction``
    projects. For PII detection / NER / entity-extraction tasks the
    source-document PII IS the training signal — auto-redacting it
    would destroy what the model is supposed to learn. The data-health
    report's recipe-aware signal logic also hides the button for this
    project shape, but we re-check here in case the API is hit
    directly (CLI, scripted client, etc.).
    """
    from app.models.project import Project
    from app.services.cleaning_service import clean_document

    project = await db.get(Project, project_id)
    if project is not None:
        selected = project.selected_recipe or {}
        recipe_id = selected.get("recipe_id") if isinstance(selected, dict) else None
        if recipe_id:
            try:
                from app.services.recipe_service import get_recipe
                recipe = get_recipe(recipe_id)
                task_profile = getattr(recipe, "task_profile", None) if recipe else None
            except Exception:
                task_profile = None
            if task_profile == "structured_extraction":
                raise ValueError(
                    "redact_pii is unsafe for span-extraction recipes: "
                    "the model needs PII in source documents to learn "
                    "what to detect. Auto-redaction would destroy the "
                    "training signal. If you need redaction for a "
                    "separate non-training use, do it manually on a "
                    "copy of the cleaned outputs."
                )

    docs = await _load_raw_docs(db, project_id)
    targets = []
    for doc in docs:
        meta = doc.metadata_ or {}
        findings = meta.get("pii_findings") or []
        if not isinstance(findings, list) or not findings:
            continue
        if meta.get("redact_pii"):
            # Already redacted on a previous run — skip.
            continue
        if not meta.get("cleaned_path"):
            # Doc hasn't been cleaned yet; the user should run cleaning
            # first. Redact-on-uncleaned would require running the full
            # pipeline from scratch and we don't want to surprise them.
            continue
        targets.append(doc)

    redacted_files: list[str] = []
    for doc in targets:
        try:
            await clean_document(
                db, project_id, int(doc.id),
                redact=True,
                redact_toxic=False,
            )
            # Annotate the redact flag so future autofix runs skip this
            # doc — the cleaning service writes the cleaned files but
            # doesn't set this flag itself.
            updated = dict(doc.metadata_ or {})
            updated["redact_pii"] = True
            doc.metadata_ = updated
            redacted_files.append(doc.filename or "")
        except ValueError:
            # Doc disappeared between the select and the clean call,
            # or its extracted text isn't on disk. Skip rather than
            # failing the whole batch.
            continue

    return {
        "fix_kind": "redact_pii",
        "applied_count": len(redacted_files),
        "summary": (
            f"Re-cleaned {len(redacted_files)} document{'s' if len(redacted_files) != 1 else ''} "
            f"with PII redaction enabled."
            if redacted_files
            else "No documents with unredacted PII to process."
        ),
        "details": {"redacted_filenames": redacted_files},
    }


async def _canonicalise_labels(
    db: AsyncSession, project_id: int
) -> AutofixResult:
    """D4 — rewrite gold-set JSONL with merged label variants.

    Selects fragmented label groups the same way ``_preview_canonicalise_labels``
    does (normalise = lowercase + collapsed whitespace), picks the
    most-common variant as canonical (ties broken alphabetically), and
    rewrites every gold JSONL row whose extracted label sits in a
    non-canonical bucket. Each row is rewritten in-place at whichever
    field carried the label (``label``, ``expected.label``,
    ``expected``, or ``answer``) so downstream loaders see the merged
    vocabulary without further changes.
    """
    import json
    import re as _re
    from app.models.project import Project
    from app.services.trainability_forecast_service import (
        _extract_classification_label,
    )

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    selected = project.selected_recipe or {}
    recipe_id = selected.get("recipe_id") if isinstance(selected, dict) else None
    task_profile: str | None = None
    if recipe_id:
        try:
            from app.services.recipe_service import get_recipe
            recipe = get_recipe(recipe_id)
            task_profile = getattr(recipe, "task_profile", None) if recipe else None
        except Exception:
            task_profile = None
    if task_profile != "classification":
        raise ValueError(
            "canonicalise_labels only applies to classification recipes "
            f"(this project is {task_profile or 'unknown'})."
        )

    gold_result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                [DatasetType.GOLD_DEV, DatasetType.GOLD_TEST]
            ),
        )
    )
    gold_datasets = list(gold_result.scalars())

    norm_re = _re.compile(r"\s+")

    def _normalise(label: str) -> str:
        return norm_re.sub(" ", label.strip().lower())

    # Aggregate counts across both gold files so canonical-pick is
    # consistent dev↔test.
    label_counts: dict[str, int] = {}
    rows_per_path: dict[str, list[dict[str, Any]]] = {}
    for ds in gold_datasets:
        if not ds.file_path:
            continue
        path = Path(ds.file_path)
        if not path.exists():
            continue
        rows: list[dict[str, Any]] = []
        with path.open() as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        rows_per_path[str(path)] = rows
        for row in rows:
            label = _extract_classification_label(row)
            if label is None:
                continue
            label_counts[label] = label_counts.get(label, 0) + 1

    groups: dict[str, list[tuple[str, int]]] = {}
    for label, count in label_counts.items():
        groups.setdefault(_normalise(label), []).append((label, count))

    # Build {variant: canonical} for every fragmented group.
    merge_map: dict[str, str] = {}
    merge_groups: list[dict[str, Any]] = []
    for variants in groups.values():
        if len(variants) <= 1:
            continue
        variants.sort(key=lambda v: (-v[1], v[0]))
        canonical = variants[0][0]
        for v, _c in variants[1:]:
            merge_map[v] = canonical
        merge_groups.append({
            "canonical": canonical,
            "merged_variants": [v for v, _c in variants[1:]],
        })

    if not merge_map:
        return {
            "fix_kind": "canonicalise_labels",
            "applied_count": 0,
            "summary": "No fragmented labels detected.",
            "details": {"merge_groups": []},
        }

    rewritten_paths: list[str] = []
    rows_touched = 0
    for path_str, rows in rows_per_path.items():
        path = Path(path_str)
        changed = False
        new_lines: list[str] = []
        for row in rows:
            label = _extract_classification_label(row)
            if label is not None and label in merge_map:
                target = merge_map[label]
                if isinstance(row.get("label"), str):
                    row["label"] = target
                expected = row.get("expected")
                if isinstance(expected, dict) and isinstance(
                    expected.get("label"), str
                ):
                    expected["label"] = target
                elif isinstance(expected, str):
                    row["expected"] = target
                if (
                    isinstance(row.get("answer"), str)
                    and row["answer"] == label
                ):
                    row["answer"] = target
                rows_touched += 1
                changed = True
            new_lines.append(json.dumps(row, ensure_ascii=False))
        if changed:
            path.write_text("\n".join(new_lines) + "\n")
            rewritten_paths.append(path.name)

    return {
        "fix_kind": "canonicalise_labels",
        "applied_count": rows_touched,
        "summary": (
            f"Merged {len(merge_groups)} fragmented label group"
            f"{'s' if len(merge_groups) != 1 else ''} across "
            f"{rows_touched} gold row"
            f"{'s' if rows_touched != 1 else ''}."
        ),
        "details": {
            "merge_groups": merge_groups,
            "rewritten_files": rewritten_paths,
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Public dispatcher.
# ─────────────────────────────────────────────────────────────────────


async def preview_autofix(
    db: AsyncSession, project_id: int, fix_kind: str
) -> AutofixPreview:
    """Return the would-change list for ``fix_kind`` without mutating
    anything. The frontend renders this as a per-item modal before
    asking the user to confirm. Raises ``ValueError`` for unknown
    kinds — API layer surfaces as 400."""
    from app.models.project import Project

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    if fix_kind not in SUPPORTED_FIX_KINDS:
        raise ValueError(
            f"Unknown fix_kind '{fix_kind}'. "
            f"Supported: {', '.join(SUPPORTED_FIX_KINDS)}."
        )

    if fix_kind == "drop_failed_docs":
        return await _preview_drop_failed_docs(db, project_id)
    if fix_kind == "dedupe_duplicate_docs":
        return await _preview_dedupe_duplicate_docs(db, project_id)
    if fix_kind == "redact_pii":
        return await _preview_redact_pii(db, project_id)
    if fix_kind == "canonicalise_labels":
        return await _preview_canonicalise_labels(db, project_id)
    # pragma: no cover — guarded by SUPPORTED_FIX_KINDS check above
    raise ValueError(f"Unknown fix_kind '{fix_kind}'.")


async def apply_autofix(
    db: AsyncSession, project_id: int, fix_kind: str
) -> AutofixResult:
    """Dispatch ``fix_kind`` to the matching autofix function. Raises
    ``ValueError`` for unknown kinds — the API layer surfaces as 400."""
    from app.models.project import Project

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    if fix_kind not in SUPPORTED_FIX_KINDS:
        raise ValueError(
            f"Unknown fix_kind '{fix_kind}'. "
            f"Supported: {', '.join(SUPPORTED_FIX_KINDS)}."
        )

    if fix_kind == "drop_failed_docs":
        result = await _drop_failed_docs(db, project_id)
    elif fix_kind == "dedupe_duplicate_docs":
        result = await _dedupe_duplicate_docs(db, project_id)
    elif fix_kind == "redact_pii":
        result = await _redact_pii(db, project_id)
    elif fix_kind == "canonicalise_labels":
        result = await _canonicalise_labels(db, project_id)
    else:  # pragma: no cover — guarded by SUPPORTED_FIX_KINDS check above
        raise ValueError(f"Unknown fix_kind '{fix_kind}'.")

    await db.flush()
    return result
