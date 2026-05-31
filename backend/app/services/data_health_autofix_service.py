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
)


class AutofixResult(TypedDict, total=False):
    """The shape every autofix function returns.

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
    the request path."""
    from app.services.cleaning_service import clean_document

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


# ─────────────────────────────────────────────────────────────────────
# Public dispatcher.
# ─────────────────────────────────────────────────────────────────────


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
    else:  # pragma: no cover — guarded by SUPPORTED_FIX_KINDS check above
        raise ValueError(f"Unknown fix_kind '{fix_kind}'.")

    await db.flush()
    return result
