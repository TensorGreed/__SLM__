"""D3 — safe auto-fix engine for the Data Health Report.

Phase 4 expands the surface from 4 fix kinds to 9. Every kind below is
*safe* — no data fabrication, no destructive operations beyond removing
already-broken or already-redundant rows, no transforms that can't be
re-run idempotently:

- ``drop_failed_docs`` — delete RawDocument rows with ``status=ERROR``.
- ``dedupe_duplicate_docs`` — exact-hash dedup keyed on
  ``metadata_.text_hash`` (computed during cleaning).
- ``redact_pii`` — re-run ``clean_document`` with ``redact=True``.
- ``canonicalise_labels`` — merge case/whitespace label duplicates in
  classification gold-set labels.

Coach-stage-2 phase 4 adds five "rewrite cleaned text" fixes, each
operating on the cleaned-text file referenced by
``metadata_.cleaned_path`` and updating ``metadata_.text_hash`` to
match. Phase 1's ``compute_text_hash`` already normalises by lowercase
+ whitespace collapse, so the new fixes operate ABOVE that baseline:

- ``near_duplicate_dedup`` — group cleaned docs by an *aggressively*
  normalised hash (punctuation-stripped + 500-char prefix). Catches
  pairs the exact-hash dedup misses: paraphrases with shared openings,
  documents that differ only in trailing boilerplate, etc.
- ``normalize_whitespace`` — rewrite the cleaned-text file with
  whitespace runs collapsed, line endings normalised, BOM stripped.
  Idempotent — second run finds nothing to change.
- ``strip_html`` — strip ``<[^>]+>`` tags and decode HTML entities
  from the cleaned text. Idempotent.
- ``length_cap`` — truncate cleaned text to
  ``max_seq_length × CHARS_PER_TOKEN_APPROX`` chars, mirroring the
  phase-3 truncation signal. Closes the gap the truncation signal
  flags without manually editing every row.
- ``normalize_schema`` — rename common field variants in
  GOLD_DEV/GOLD_TEST JSONL rows to the canonical names the trainer
  expects (``class``/``target`` → ``label``, ``text`` → ``input``).
  Safe + reversible mapping; idempotent.

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
    # Coach-stage-2 phase 4 — five "rewrite cleaned text" fixes. Each
    # operates on the cleaned-text file referenced by
    # ``metadata_.cleaned_path`` and updates ``metadata_.text_hash`` to
    # match. Idempotent — second run finds nothing to change.
    "near_duplicate_dedup",
    "normalize_whitespace",
    "strip_html",
    "length_cap",
    "normalize_schema",
)


# Phase 4 — token approximation matches the training-config gap
# scanner's CHARS_PER_TOKEN_APPROX. Used by length_cap to translate
# the trainer's max_seq_length into a char-level truncation point.
CHARS_PER_TOKEN_APPROX = 4

# Aggressive-normalisation prefix length for near_duplicate_dedup.
# 500 chars covers ~the opening paragraph of most documents, which is
# where shared content usually lives in near-duplicates (think
# customer-support tickets that all open with the same boilerplate).
NEAR_DUP_PREFIX_CHARS = 500


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
# Phase 4 — rewrite-cleaned-text helpers.
# ─────────────────────────────────────────────────────────────────────


def _read_cleaned_text(doc: RawDocument) -> str | None:
    """Read the cleaned text file for a doc, or ``None`` when the path
    is missing/unreadable. Centralises the "if metadata.cleaned_path
    exists" guard every phase-4 fix needs."""
    meta = doc.metadata_ or {}
    cleaned_path = meta.get("cleaned_path")
    if not isinstance(cleaned_path, str) or not cleaned_path:
        return None
    path = Path(cleaned_path)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _write_cleaned_text(doc: RawDocument, new_text: str) -> bool:
    """Overwrite a cleaned-text file in place. Updates the doc's
    ``metadata_.text_hash`` so future scans / dedup runs see the new
    content. Returns True on success.
    """
    meta = dict(doc.metadata_ or {})
    cleaned_path = meta.get("cleaned_path")
    if not isinstance(cleaned_path, str) or not cleaned_path:
        return False
    path = Path(cleaned_path)
    try:
        path.write_text(new_text, encoding="utf-8")
    except OSError:
        return False
    # Lazy import — cleaning_service is heavy and we only need one
    # helper. Keeping it lazy avoids pulling chunking + regex tables
    # into every autofix request.
    from app.services.cleaning_service import compute_text_hash
    meta["text_hash"] = compute_text_hash(new_text)
    doc.metadata_ = meta
    return True


def _normalise_whitespace(text: str) -> str:
    """Apply the whitespace-normalize transform: strip BOM, collapse
    runs of whitespace, normalize line endings, strip trailing
    whitespace. Idempotent — second run is a no-op."""
    import re as _re
    # Strip BOM.
    if text.startswith("﻿"):
        text = text.lstrip("﻿")
    # Normalize Windows / Mac line endings to LF before collapsing.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines to a single blank (preserve paragraph
    # structure but kill garbled-PDF excess).
    text = _re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of horizontal whitespace within a line.
    text = _re.sub(r"[ \t]+", " ", text)
    # Strip trailing whitespace per line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _strip_html(text: str) -> str:
    """Apply the html-strip transform: remove ``<[^>]+>`` tags and
    decode the common HTML entities (``&amp;``, ``&lt;``, ``&gt;``,
    ``&quot;``, ``&#39;``, ``&nbsp;``). Idempotent."""
    import re as _re
    # Drop script/style blocks first (incl. their contents) so the
    # tag-strip pass doesn't leave behind their bodies.
    text = _re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        "",
        text,
        flags=_re.IGNORECASE | _re.DOTALL,
    )
    # Strip remaining tags.
    text = _re.sub(r"<[^>]+>", "", text)
    # Decode common entities — html.unescape covers everything but
    # NBSP, which leaves a U+00A0 that we don't want.
    import html as _html
    text = _html.unescape(text)
    text = text.replace("\xa0", " ")
    return text


# ─────────────────────────────────────────────────────────────────────
# Phase 4 — preview helpers.
# ─────────────────────────────────────────────────────────────────────


async def _preview_near_duplicate_dedup(
    db: AsyncSession, project_id: int,
) -> AutofixPreview:
    """Group cleaned docs by an aggressively-normalised hash of their
    first 500 chars. Catches near-duplicates the exact text_hash dedup
    misses (paraphrases with shared openings, etc.).
    """
    import hashlib
    import re as _re

    docs = await _load_raw_docs(db, project_id)
    groups: dict[str, list[tuple[RawDocument, str]]] = {}
    for doc in docs:
        if doc.status != DocumentStatus.ACCEPTED:
            continue
        text = _read_cleaned_text(doc)
        if text is None:
            continue
        # Aggressive normalisation: lowercase + strip punctuation +
        # collapse whitespace + prefix.
        normalised = text.lower()
        normalised = _re.sub(r"[^\w\s]+", " ", normalised)
        normalised = _re.sub(r"\s+", " ", normalised).strip()
        prefix = normalised[:NEAR_DUP_PREFIX_CHARS]
        if not prefix:
            continue
        soft_hash = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        # Don't group docs that already share the EXACT text_hash —
        # those are handled by the existing dedupe_duplicate_docs fix
        # (run it first if both fire).
        exact_hash = (doc.metadata_ or {}).get("text_hash")
        groups.setdefault(
            soft_hash, []
        ).append((doc, exact_hash or ""))

    items: list[dict[str, Any]] = []
    drop_count = 0
    for soft_hash, group in groups.items():
        if len(group) <= 1:
            continue
        # Exclude groups that are also exact duplicates — let the
        # existing dedup handle those. (All members share the same
        # exact_hash AND it's not empty.)
        exact_hashes = {h for _, h in group if h}
        if len(exact_hashes) == 1 and len(group) == sum(
            1 for _, h in group if h
        ):
            continue
        group_sorted = sorted(group, key=lambda x: int(x[0].id))
        keep = group_sorted[0][0]
        drops = [doc for doc, _ in group_sorted[1:]]
        items.append({
            "kind": "near_dup_group",
            "soft_hash": soft_hash[:12],
            "keep": {"id": int(keep.id), "filename": keep.filename or ""},
            "drop": [
                {"id": int(d.id), "filename": d.filename or ""}
                for d in drops
            ],
        })
        drop_count += len(drops)

    return {
        "fix_kind": "near_duplicate_dedup",
        "would_apply_count": drop_count,
        "summary": (
            f"Would drop {drop_count} near-duplicate document"
            f"{'s' if drop_count != 1 else ''} across {len(items)} "
            f"group{'s' if len(items) != 1 else ''}; keeps the "
            f"lowest-id occurrence in each."
            if drop_count
            else "No near-duplicate documents detected."
        ),
        "details": {"group_count": len(items)},
        "items": items,
        "safe_to_apply": True,
    }


async def _preview_normalize_whitespace(
    db: AsyncSession, project_id: int,
) -> AutofixPreview:
    docs = await _load_raw_docs(db, project_id)
    items: list[dict[str, Any]] = []
    for doc in docs:
        if doc.status != DocumentStatus.ACCEPTED:
            continue
        text = _read_cleaned_text(doc)
        if text is None:
            continue
        normalised = _normalise_whitespace(text)
        if normalised == text:
            continue
        items.append({
            "kind": "whitespace_rewrite",
            "id": int(doc.id),
            "filename": doc.filename or "",
            "chars_before": len(text),
            "chars_after": len(normalised),
        })
    return {
        "fix_kind": "normalize_whitespace",
        "would_apply_count": len(items),
        "summary": (
            f"Would rewrite {len(items)} cleaned document"
            f"{'s' if len(items) != 1 else ''} with whitespace "
            f"normalised."
            if items
            else "No whitespace cleanup needed."
        ),
        "details": {},
        "items": items,
        "safe_to_apply": True,
    }


async def _preview_strip_html(
    db: AsyncSession, project_id: int,
) -> AutofixPreview:
    docs = await _load_raw_docs(db, project_id)
    items: list[dict[str, Any]] = []
    import re as _re
    tag_re = _re.compile(r"<[^>]+>")
    for doc in docs:
        if doc.status != DocumentStatus.ACCEPTED:
            continue
        text = _read_cleaned_text(doc)
        if text is None:
            continue
        tag_count = len(tag_re.findall(text))
        if tag_count == 0:
            continue
        stripped = _strip_html(text)
        items.append({
            "kind": "html_strip",
            "id": int(doc.id),
            "filename": doc.filename or "",
            "tags_to_strip": tag_count,
            "chars_before": len(text),
            "chars_after": len(stripped),
        })
    return {
        "fix_kind": "strip_html",
        "would_apply_count": len(items),
        "summary": (
            f"Would strip HTML from {len(items)} cleaned document"
            f"{'s' if len(items) != 1 else ''}."
            if items
            else "No HTML tags detected in cleaned text."
        ),
        "details": {},
        "items": items,
        "safe_to_apply": True,
    }


async def _resolve_length_cap_chars(
    db: AsyncSession, project_id: int,
) -> int:
    """Resolve the char-cap for length_cap from the project's effective
    training config. Mirrors the gap scanner's overlay logic so the
    fix lands at the same number the gap scanner reports against."""
    from app.models.project import Project
    from app.services.training_config_gap_service import (
        _effective_training_config,
    )

    project = await db.get(Project, project_id)
    if project is None:
        return 8192  # safe default (2048 tokens × 4)
    cfg = _effective_training_config(project)
    return int(cfg.max_seq_length) * CHARS_PER_TOKEN_APPROX


async def _preview_length_cap(
    db: AsyncSession, project_id: int,
) -> AutofixPreview:
    cap = await _resolve_length_cap_chars(db, project_id)
    docs = await _load_raw_docs(db, project_id)
    items: list[dict[str, Any]] = []
    for doc in docs:
        if doc.status != DocumentStatus.ACCEPTED:
            continue
        text = _read_cleaned_text(doc)
        if text is None:
            continue
        if len(text) <= cap:
            continue
        items.append({
            "kind": "length_truncate",
            "id": int(doc.id),
            "filename": doc.filename or "",
            "chars_before": len(text),
            "chars_after": cap,
            "chars_dropped": len(text) - cap,
        })
    return {
        "fix_kind": "length_cap",
        "would_apply_count": len(items),
        "summary": (
            f"Would truncate {len(items)} cleaned document"
            f"{'s' if len(items) != 1 else ''} to {cap} chars "
            f"(~{cap // CHARS_PER_TOKEN_APPROX} tokens, the project's "
            f"effective max_seq_length)."
            if items
            else "No documents exceed the project's effective max_seq_length."
        ),
        "details": {"cap_chars": cap},
        "items": items,
        "safe_to_apply": True,
    }


# Canonical gold-row field renames. Phase 4 ships these three because
# they're the variants we've seen across the 8 shipped templates +
# user imports; phase 5 may extend.
GOLD_FIELD_RENAMES: dict[str, str] = {
    "class": "label",
    "target": "label",
    "text": "input",
    "prompt": "input",
}


async def _preview_normalize_schema(
    db: AsyncSession, project_id: int,
) -> AutofixPreview:
    """Detect GOLD_DEV/GOLD_TEST rows using non-canonical field names
    (``class``/``target`` for label, ``text``/``prompt`` for input)
    and report the count per file."""
    import json
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                [DatasetType.GOLD_DEV, DatasetType.GOLD_TEST]
            ),
        )
    )
    datasets = list(result.scalars())

    items: list[dict[str, Any]] = []
    total_renames = 0
    for ds in datasets:
        if not ds.file_path:
            continue
        path = Path(ds.file_path)
        if not path.exists():
            continue
        rename_counts: dict[str, int] = {}
        try:
            with path.open(encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    for old_key in GOLD_FIELD_RENAMES:
                        if old_key in row and (
                            GOLD_FIELD_RENAMES[old_key] not in row
                        ):
                            rename_counts[old_key] = (
                                rename_counts.get(old_key, 0) + 1
                            )
        except OSError:
            continue
        file_renames = sum(rename_counts.values())
        if file_renames == 0:
            continue
        items.append({
            "kind": "schema_rename",
            "file": path.name,
            "renames_per_field": rename_counts,
            "row_count": file_renames,
        })
        total_renames += file_renames

    return {
        "fix_kind": "normalize_schema",
        "would_apply_count": total_renames,
        "summary": (
            f"Would rename non-canonical fields on {total_renames} "
            f"gold row{'s' if total_renames != 1 else ''} across "
            f"{len(items)} file{'s' if len(items) != 1 else ''}."
            if total_renames
            else "Gold-row schema already canonical."
        ),
        "details": {"rename_map": GOLD_FIELD_RENAMES},
        "items": items,
        "safe_to_apply": True,
    }


# ─────────────────────────────────────────────────────────────────────
# Phase 4 — apply functions.
# ─────────────────────────────────────────────────────────────────────


async def _near_duplicate_dedup(
    db: AsyncSession, project_id: int,
) -> AutofixResult:
    """Apply near-duplicate dedup. Mirrors the preview's grouping +
    keeps the lowest-id doc per group; deletes the rest (DB row +
    on-disk artefacts)."""
    preview = await _preview_near_duplicate_dedup(db, project_id)
    items = preview.get("items", [])
    drop_ids: list[int] = []
    for item in items:
        for drop in item.get("drop", []):
            drop_ids.append(int(drop["id"]))
    if not drop_ids:
        return {
            "fix_kind": "near_duplicate_dedup",
            "applied_count": 0,
            "summary": "No near-duplicate documents to drop.",
            "details": {"group_count": 0},
        }

    docs = await _load_raw_docs(db, project_id)
    by_id = {int(d.id): d for d in docs}
    dropped_filenames: list[str] = []
    for doc_id in drop_ids:
        doc = by_id.get(doc_id)
        if doc is None:
            continue
        dropped_filenames.append(doc.filename or "")
        _remove_doc_artifacts(doc)
        await db.delete(doc)
    await _decrement_dataset_count(db, project_id, -len(dropped_filenames))
    return {
        "fix_kind": "near_duplicate_dedup",
        "applied_count": len(dropped_filenames),
        "summary": (
            f"Dropped {len(dropped_filenames)} near-duplicate "
            f"document{'s' if len(dropped_filenames) != 1 else ''} "
            f"across {len(items)} group"
            f"{'s' if len(items) != 1 else ''}."
        ),
        "details": {
            "dropped_filenames": dropped_filenames,
            "group_count": len(items),
        },
    }


async def _normalize_whitespace(
    db: AsyncSession, project_id: int,
) -> AutofixResult:
    docs = await _load_raw_docs(db, project_id)
    rewritten: list[str] = []
    for doc in docs:
        if doc.status != DocumentStatus.ACCEPTED:
            continue
        text = _read_cleaned_text(doc)
        if text is None:
            continue
        normalised = _normalise_whitespace(text)
        if normalised == text:
            continue
        if _write_cleaned_text(doc, normalised):
            rewritten.append(doc.filename or "")
    return {
        "fix_kind": "normalize_whitespace",
        "applied_count": len(rewritten),
        "summary": (
            f"Normalised whitespace on {len(rewritten)} cleaned "
            f"document{'s' if len(rewritten) != 1 else ''}."
            if rewritten
            else "No whitespace cleanup needed."
        ),
        "details": {"rewritten_filenames": rewritten},
    }


async def _strip_html_fix(
    db: AsyncSession, project_id: int,
) -> AutofixResult:
    docs = await _load_raw_docs(db, project_id)
    rewritten: list[str] = []
    import re as _re
    tag_re = _re.compile(r"<[^>]+>")
    for doc in docs:
        if doc.status != DocumentStatus.ACCEPTED:
            continue
        text = _read_cleaned_text(doc)
        if text is None:
            continue
        if not tag_re.search(text):
            continue
        stripped = _strip_html(text)
        if stripped == text:
            continue
        if _write_cleaned_text(doc, stripped):
            rewritten.append(doc.filename or "")
    return {
        "fix_kind": "strip_html",
        "applied_count": len(rewritten),
        "summary": (
            f"Stripped HTML from {len(rewritten)} cleaned "
            f"document{'s' if len(rewritten) != 1 else ''}."
            if rewritten
            else "No HTML tags detected in cleaned text."
        ),
        "details": {"rewritten_filenames": rewritten},
    }


async def _length_cap(
    db: AsyncSession, project_id: int,
) -> AutofixResult:
    cap = await _resolve_length_cap_chars(db, project_id)
    docs = await _load_raw_docs(db, project_id)
    rewritten: list[str] = []
    total_dropped = 0
    for doc in docs:
        if doc.status != DocumentStatus.ACCEPTED:
            continue
        text = _read_cleaned_text(doc)
        if text is None:
            continue
        if len(text) <= cap:
            continue
        truncated = text[:cap]
        total_dropped += len(text) - cap
        if _write_cleaned_text(doc, truncated):
            rewritten.append(doc.filename or "")
    return {
        "fix_kind": "length_cap",
        "applied_count": len(rewritten),
        "summary": (
            f"Truncated {len(rewritten)} cleaned document"
            f"{'s' if len(rewritten) != 1 else ''} to {cap} chars "
            f"(~{cap // CHARS_PER_TOKEN_APPROX} tokens); dropped "
            f"{total_dropped} chars total."
            if rewritten
            else "No documents exceed the project's effective max_seq_length."
        ),
        "details": {
            "rewritten_filenames": rewritten,
            "cap_chars": cap,
            "chars_dropped": total_dropped,
        },
    }


async def _normalize_schema(
    db: AsyncSession, project_id: int,
) -> AutofixResult:
    """Rename non-canonical field names in gold-set JSONL rows in
    place. Idempotent — second run finds nothing to rename."""
    import json
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                [DatasetType.GOLD_DEV, DatasetType.GOLD_TEST]
            ),
        )
    )
    datasets = list(result.scalars())
    rewritten: list[str] = []
    total_renames = 0
    for ds in datasets:
        if not ds.file_path:
            continue
        path = Path(ds.file_path)
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        file_renames = 0
        new_lines: list[str] = []
        for line in text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                new_lines.append(line)
                continue
            try:
                row = json.loads(line_stripped)
            except json.JSONDecodeError:
                new_lines.append(line)
                continue
            if not isinstance(row, dict):
                new_lines.append(line)
                continue
            new_row = dict(row)
            for old_key, new_key in GOLD_FIELD_RENAMES.items():
                if old_key in new_row and new_key not in new_row:
                    new_row[new_key] = new_row.pop(old_key)
                    file_renames += 1
            if file_renames == 0:
                # No change to this row; preserve original line.
                new_lines.append(line)
            else:
                new_lines.append(json.dumps(new_row, ensure_ascii=False))
        if file_renames > 0:
            path.write_text("\n".join(new_lines), encoding="utf-8")
            rewritten.append(path.name)
            total_renames += file_renames
    return {
        "fix_kind": "normalize_schema",
        "applied_count": total_renames,
        "summary": (
            f"Renamed {total_renames} non-canonical field"
            f"{'s' if total_renames != 1 else ''} across "
            f"{len(rewritten)} gold file"
            f"{'s' if len(rewritten) != 1 else ''}."
            if total_renames
            else "Gold-row schema already canonical."
        ),
        "details": {
            "rewritten_filenames": rewritten,
            "rename_map": GOLD_FIELD_RENAMES,
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
    # Phase 4.
    if fix_kind == "near_duplicate_dedup":
        return await _preview_near_duplicate_dedup(db, project_id)
    if fix_kind == "normalize_whitespace":
        return await _preview_normalize_whitespace(db, project_id)
    if fix_kind == "strip_html":
        return await _preview_strip_html(db, project_id)
    if fix_kind == "length_cap":
        return await _preview_length_cap(db, project_id)
    if fix_kind == "normalize_schema":
        return await _preview_normalize_schema(db, project_id)
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
    # Phase 4.
    elif fix_kind == "near_duplicate_dedup":
        result = await _near_duplicate_dedup(db, project_id)
    elif fix_kind == "normalize_whitespace":
        result = await _normalize_whitespace(db, project_id)
    elif fix_kind == "strip_html":
        result = await _strip_html_fix(db, project_id)
    elif fix_kind == "length_cap":
        result = await _length_cap(db, project_id)
    elif fix_kind == "normalize_schema":
        result = await _normalize_schema(db, project_id)
    else:  # pragma: no cover — guarded by SUPPORTED_FIX_KINDS check above
        raise ValueError(f"Unknown fix_kind '{fix_kind}'.")

    await db.flush()
    return result
