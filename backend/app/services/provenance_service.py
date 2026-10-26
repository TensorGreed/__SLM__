"""Licence & provenance ledger.

One :class:`~app.models.data_source.DataSource` per place data came from
(upload, website crawl, git repo, HuggingFace / Kaggle / URL import, a
dataset-import run, a predefined split, a demo bundle, an external benchmark).
Documents point at their source through ``raw_documents.data_source_id``;
rows a source wrote straight into ``synthetic.jsonl`` carry ``data_source_id``.

What the ledger decides (``usage``):

- ``any``   - today's behaviour: the rows train / validate / test as usual.
- ``train`` - same as ``any`` today (named so the intent is on record).
- ``eval``  - held out of the train / val pool and appended to the test split
              (``dataset_service.combine_datasets`` / ``split_dataset``).
- ``rag``   - dropped from every split and from every synthetic-seeding path
              (``training_excluded_document_ids``); the document stays cleaned
              and indexed so retrieval still finds it.

A predefined-split source is always ``any``: its own role (the user's test
file) already governs it (``split_role_service``).

Everything here is generic: licence strings are free text (an SPDX id when
one is known) and the only licence rules are "unknown" and "non-commercial".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.data_source import DataSource
from app.models.dataset import Dataset, RawDocument

USAGES = ("any", "train", "eval", "rag")
USAGE_ANY = "any"
USAGE_EVAL = "eval"
USAGE_RAG = "rag"

KIND_UPLOAD = "upload"
KIND_WEB_CRAWL = "web_crawl"
KIND_GIT_REPO = "git_repo"
KIND_HF = "hf"
KIND_KAGGLE = "kaggle"
KIND_URL = "url"
KIND_DATASET_IMPORT = "dataset_import"
KIND_PREDEFINED_SPLIT = "predefined_split"
KIND_DEMO = "demo"
KIND_BENCHMARK = "benchmark"
SOURCE_KINDS = (
    KIND_UPLOAD, KIND_WEB_CRAWL, KIND_GIT_REPO, KIND_HF, KIND_KAGGLE, KIND_URL,
    KIND_DATASET_IMPORT, KIND_PREDEFINED_SPLIT, KIND_DEMO, KIND_BENCHMARK,
)

FLAG_UNKNOWN = "unknown"
FLAG_NON_COMMERCIAL = "non_commercial"

UNSET: Any = object()

_PLACEHOLDER_RE = re.compile(r"^\s*imported from\b", re.IGNORECASE)
_UNKNOWN_LICENCES = {"", "unknown", "noassertion", "none", "n/a"}
_NC_RE = re.compile(
    r"(?<![a-z0-9])nc(?![a-z0-9])|non[\s_-]?commercial|not[\s_-]+for[\s_-]+commercial"
    r"|research[\s_-]+(?:(?:use|purposes?)[\s_-]+)?only",
    re.IGNORECASE,
)


class ProvenanceError(ValueError):
    """Bad request to the ledger (unknown usage, ...)."""


class SourceNotFoundError(LookupError):
    """A source id that is not in the project."""


class UsageLockedError(Exception):
    """``usage`` of a predefined-split source cannot be changed."""

    def __init__(self, source_ids: list[int]):
        self.source_ids = source_ids
        super().__init__(
            "The usage of a predefined-split source is fixed: its own train / val / test roles already decide "
            f"where its rows go (sources {source_ids})."
        )


# ── pure helpers ────────────────────────────────────────────────────────


def normalize_licence(raw: Any) -> str | None:
    """Trimmed licence text, or ``None`` when it says nothing: empty,
    ``unknown``, ``NOASSERTION`` or the old ``Imported from ...`` placeholder."""
    if raw is None:
        return None
    text = " ".join(str(raw).split())
    if text.lower() in _UNKNOWN_LICENCES or _PLACEHOLDER_RE.match(text):
        return None
    return text[:255]


def licence_flags(licence: str | None) -> list[str]:
    """``["unknown"]`` for no licence, ``["non_commercial"]`` for an NC one."""
    text = normalize_licence(licence)
    if text is None:
        return [FLAG_UNKNOWN]
    return [FLAG_NON_COMMERCIAL] if _NC_RE.search(text) else []


def validate_usage(usage: Any) -> str:
    text = str(usage or "").strip().lower()
    if text not in USAGES:
        raise ProvenanceError(f"usage must be one of {', '.join(USAGES)} (got {usage!r}).")
    return text


def effective_usage(source: Any | None) -> str:
    """The usage a source imposes (``any`` for no source / a predefined split)."""
    if source is None:
        return USAGE_ANY
    kind = getattr(source, "source_kind", None)
    if kind is None and isinstance(source, dict):
        kind = source.get("source_kind")
    if kind == KIND_PREDEFINED_SPLIT:
        return USAGE_ANY
    usage = getattr(source, "usage", None) if not isinstance(source, dict) else source.get("usage")
    return usage if usage in USAGES else USAGE_ANY


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _licence_text(value: Any) -> str | None:
    """A licence from a HuggingFace / Kaggle metadata value (str, list, dict)."""
    if isinstance(value, (list, tuple)):
        parts = [p for p in (_licence_text(v) for v in value) if p]
        return ", ".join(parts) or None
    if isinstance(value, dict):
        return _licence_text(value.get("name") or value.get("id"))
    return normalize_licence(value) if isinstance(value, str) else None


# ── remote licence lookups (best-effort, never raise) ───────────────────


def hf_dataset_licence(dataset_id: str, token: str | None = None, config: str | None = None) -> str | None:
    """A HuggingFace dataset's licence: the dataset-info licence of ``config``
    (first config when unnamed), else the Hub card. ``None`` when offline /
    private / not declared."""
    try:
        from datasets import get_dataset_infos

        infos = get_dataset_infos(dataset_id, token=token)
        info = infos.get(config) if config and config in infos else next(iter(infos.values()), None)
        licence = _licence_text(getattr(info, "license", None)) if info is not None else None
        if licence:
            return licence
    except Exception:  # noqa: BLE001 - offline / script dataset: try the card
        pass
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).dataset_info(dataset_id)
        card = info.card_data.to_dict() if info.card_data else {}
        return _licence_text(card.get("license"))
    except Exception:  # noqa: BLE001 - offline / private: licence unknown
        return None


def kaggle_dataset_licence(api: Any, slug: str) -> str | None:
    """A Kaggle dataset's licence through an authenticated ``KaggleApi``. Uses
    ``dataset_list(search=slug)`` (each hit has ``ref`` and ``license_name``),
    then ``dataset_metadata(slug, path)`` — which *writes* the metadata json
    into ``path`` and returns its location — read from a scratch directory.
    ``None`` on any failure."""
    slug = (slug or "").strip().strip("/")
    lister = getattr(api, "dataset_list", None)
    if lister is not None:
        try:
            for item in lister(search=slug.split("/")[-1]) or []:
                ref = str(getattr(item, "ref", "") or (item.get("ref") if isinstance(item, dict) else "") or "")
                if ref.strip("/").endswith(slug):
                    name = getattr(item, "license_name", None) or (item.get("license_name") if isinstance(item, dict) else None)
                    licence = _licence_text(name)
                    if licence:
                        return licence
        except Exception:  # noqa: BLE001 - best effort
            pass
    fetch = getattr(api, "dataset_metadata", None)
    if fetch is not None:
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                written = fetch(slug, tmp)
                meta_path = Path(written) if written else Path(tmp) / "dataset-metadata.json"
                info = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(info, dict):
                return _licence_text(info.get("licenses") or info.get("licenseName") or info.get("license_name"))
        except Exception:  # noqa: BLE001 - best effort
            pass
    return None


# ── create / attach ─────────────────────────────────────────────────────


async def create_source(
    db: AsyncSession,
    project_id: int,
    *,
    kind: str,
    title: str,
    locator: str = "",
    ref: str | None = None,
    licence: str | None = None,
    retrieved_at: datetime | None = None,
    usage: str = USAGE_ANY,
    metadata: dict[str, Any] | None = None,
) -> DataSource:
    """Add a ledger entry (flushed, not committed)."""
    usage = validate_usage(usage)
    if kind == KIND_PREDEFINED_SPLIT:
        usage = USAGE_ANY
    source = DataSource(
        project_id=project_id,
        source_kind=kind,
        title=(title or locator or kind)[:512],
        locator=(locator or "")[:1024],
        source_ref=(ref or None) and str(ref)[:255],
        licence=normalize_licence(licence),
        licence_verified=False,
        retrieved_at=retrieved_at or _now(),
        usage=usage,
        metadata_=dict(metadata or {}),
    )
    db.add(source)
    await db.flush()
    return source


async def attach_document(
    doc: RawDocument,
    source: DataSource,
    *,
    source_url: str | None = None,
    source_ref: str | None = None,
    retrieved_at: datetime | None = None,
) -> None:
    """Point ``doc`` at ``source`` and record where / when it was fetched."""
    doc.data_source_id = source.id
    if source_url:
        doc.source_url = source_url[:1024]
    if source_ref:
        doc.source_ref = str(source_ref)[:255]
    doc.retrieved_at = retrieved_at or source.retrieved_at
    doc.license_info = source.licence or ""


def _infer_kind(doc: RawDocument) -> str:
    meta = doc.metadata_ if isinstance(doc.metadata_, dict) else {}
    source = doc.source or ""
    if meta.get("split_role") or source.startswith("predefined_split"):
        return KIND_PREDEFINED_SPLIT
    if source == "web_crawl":
        return KIND_WEB_CRAWL
    if source == "git_repo":
        return KIND_GIT_REPO
    source_type = str(meta.get("source_type") or "")
    if source_type == "huggingface":
        return KIND_HF
    if source_type in (KIND_KAGGLE, KIND_URL):
        return source_type
    if source in ("demo", "seed_documents"):
        return KIND_DEMO
    return KIND_UPLOAD


async def _project_documents(db: AsyncSession, project_id: int) -> list[RawDocument]:
    result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(Dataset.project_id == project_id)
        .order_by(RawDocument.id)
    )
    return list(result.scalars().all())


def _legacy_group_key(doc: RawDocument, kind: str) -> str:
    meta = doc.metadata_ if isinstance(doc.metadata_, dict) else {}
    licence = normalize_licence(doc.license_info) or ""
    if kind == KIND_UPLOAD:
        day = doc.ingested_at.date().isoformat() if getattr(doc, "ingested_at", None) else ""
        parts: tuple = (kind, doc.source or "", day, licence)
    elif kind == KIND_PREDEFINED_SPLIT:
        parts = (kind, meta.get("split_role") or doc.source or "")
    else:
        parts = (kind, doc.source or "", meta.get("source_type") or "", meta.get("identifier") or "", licence)
    return json.dumps(parts, ensure_ascii=False)


async def ensure_sources(
    db: AsyncSession, project_id: int, docs: list[RawDocument] | None = None
) -> int:
    """Attach every document that predates the ledger to a source. Documents
    group into one source per (kind, ``source`` string, import identifier,
    licence); uploads additionally group by ingest day (not one source per
    file). The group key is stored in the source's metadata, so a repeat call
    — or a concurrent one that lost the race — reuses (and, after the fact,
    merges into) the same source. Idempotent; returns how many documents were
    attached. ``docs`` is the project's documents when the caller already
    loaded them."""
    candidates = [d for d in (docs if docs is not None else await _project_documents(db, project_id))
                  if d.data_source_id is None]
    if not candidates:
        return 0
    by_key: dict[str, DataSource] = {}
    for existing in await _project_sources(db, project_id):
        key = (existing.metadata_ or {}).get("legacy_group") if isinstance(existing.metadata_, dict) else None
        if key and key not in by_key:
            by_key[key] = existing
    grouped: dict[str, list[RawDocument]] = {}
    kinds: dict[str, str] = {}
    for doc in candidates:
        kind = _infer_kind(doc)
        key = _legacy_group_key(doc, kind)
        grouped.setdefault(key, []).append(doc)
        kinds[key] = kind
    fresh: dict[str, DataSource] = {}
    for key, members in grouped.items():
        if key in by_key:
            continue
        doc = members[0]
        meta = doc.metadata_ if isinstance(doc.metadata_, dict) else {}
        kind = kinds[key]
        locator = str(meta.get("identifier") or meta.get("source_url") or doc.filename or "")
        ref = meta.get("commit") or meta.get("split") or None
        title = {
            KIND_UPLOAD: doc.filename if len(members) == 1 else f"{len(members)} uploaded files",
            KIND_WEB_CRAWL: "Website pages",
            KIND_GIT_REPO: f"Git repository {locator}".strip(),
            KIND_PREDEFINED_SPLIT: f"Predefined {meta.get('split_role') or 'split'} file",
            KIND_DEMO: "Demo data" if doc.source == "demo" else "Generated sample documents",
        }.get(kind) or f"{kind} {locator}".strip()
        if kind == KIND_UPLOAD and len(members) > 1:
            locator = doc.source or ""
        source = DataSource(
            project_id=project_id, source_kind=kind, title=title[:512], locator=locator[:1024],
            source_ref=(ref or None) and str(ref)[:255], licence=normalize_licence(doc.license_info),
            licence_verified=False,
            retrieved_at=_parse_iso(meta.get("retrieved_at")) or getattr(doc, "ingested_at", None) or _now(),
            usage=USAGE_ANY, metadata_={"legacy": True, "legacy_group": key},
        )
        db.add(source)
        fresh[key] = source
    if fresh:
        await db.flush()  # one flush for every new source
    attached = 0
    for key, members in grouped.items():
        source = by_key.get(key) or fresh[key]
        for doc in members:
            meta = doc.metadata_ if isinstance(doc.metadata_, dict) else {}
            doc.data_source_id = source.id
            if doc.source_url is None and meta.get("source_url"):
                doc.source_url = str(meta["source_url"])[:1024]
            if doc.source_ref is None and (meta.get("commit") or meta.get("ref")):
                doc.source_ref = str(meta.get("commit") or meta.get("ref"))[:255]
            if doc.retrieved_at is None:
                doc.retrieved_at = _parse_iso(meta.get("retrieved_at"))
            attached += 1
    await db.flush()
    # A concurrent first call may have created the same group meanwhile: keep the
    # lowest id per group key and fold ours into it.
    if fresh:
        for key, ours in fresh.items():
            rows = await db.execute(select(DataSource).where(DataSource.project_id == project_id).order_by(DataSource.id))
            twins = [
                s_ for s_ in rows.scalars()
                if isinstance(s_.metadata_, dict) and s_.metadata_.get("legacy_group") == key
            ]
            if len(twins) > 1 and twins[0].id != ours.id:
                for doc in grouped[key]:
                    doc.data_source_id = twins[0].id
                await db.delete(ours)
        await db.flush()
    return attached


async def prune_empty_sources(db: AsyncSession, project_id: int, candidate_ids: Iterable[int | None]) -> int:
    """Delete the ``candidate_ids`` sources that no document points at any more
    (flushes first). ``dataset_import`` / ``benchmark`` sources have no documents
    by design and are never pruned. Returns how many were deleted."""
    ids = {int(i) for i in candidate_ids if i is not None}
    if not ids:
        return 0
    await db.flush()
    in_use = await db.execute(select(RawDocument.data_source_id).where(RawDocument.data_source_id.in_(ids)))
    used = {int(i) for i in in_use.scalars() if i is not None}
    result = await db.execute(
        select(DataSource).where(DataSource.project_id == project_id, DataSource.id.in_(ids - used))
    )
    removed = 0
    for source in result.scalars():
        if source.source_kind in (KIND_DATASET_IMPORT, KIND_BENCHMARK):
            continue
        await db.delete(source)
        removed += 1
    if removed:
        await db.flush()
    return removed


# ── usage maps + the training-time context ──────────────────────────────


async def _project_sources(db: AsyncSession, project_id: int) -> list[DataSource]:
    result = await db.execute(
        select(DataSource).where(DataSource.project_id == project_id).order_by(DataSource.id)
    )
    return list(result.scalars().all())


async def source_usage_map(db: AsyncSession, project_id: int) -> dict[int, str]:
    return {int(s.id): effective_usage(s) for s in await _project_sources(db, project_id)}


async def document_usage_map(db: AsyncSession, project_id: int) -> dict[int, str]:
    """Effective usage per document id (documents with no source are ``any``)."""
    usage = await source_usage_map(db, project_id)
    return {
        int(d.id): usage.get(int(d.data_source_id), USAGE_ANY) if d.data_source_id is not None else USAGE_ANY
        for d in await _project_documents(db, project_id)
    }


async def training_excluded_document_ids(
    db: AsyncSession, project_id: int, *, include_eval: bool = False
) -> set[int]:
    """Documents that must never seed training data: ``eval`` and ``rag``.
    They stay cleaned and in the passage index. ``include_eval=True`` is for
    answer-key (gold) building, which is evaluation data: only ``rag`` documents
    are excluded then."""
    usage = await source_usage_map(db, project_id)
    blocked = (USAGE_RAG,) if include_eval else (USAGE_EVAL, USAGE_RAG)
    restricted = {sid for sid, u in usage.items() if u in blocked}
    if not restricted:
        return set()
    return {
        int(d.id) for d in await _project_documents(db, project_id)
        if d.data_source_id is not None and int(d.data_source_id) in restricted
    }


def drop_excluded_passages(passages: Iterable[dict[str, Any]], excluded_ids: set[int]) -> list[dict[str, Any]]:
    """``passages`` without those cut from an excluded document."""
    if not excluded_ids:
        return list(passages)
    out = []
    for passage in passages:
        try:
            doc_id = passage.get("source_document_id")
            if doc_id is not None and int(doc_id) in excluded_ids:
                continue
        except (TypeError, ValueError):
            pass
        out.append(passage)
    return out


@dataclass
class UsageContext:
    """Per-document / per-source usage for one prepare run."""

    doc_source: dict[int, int] = field(default_factory=dict)
    source_usage: dict[int, str] = field(default_factory=dict)
    sources: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def restricted(self) -> bool:
        return any(u in (USAGE_EVAL, USAGE_RAG) for u in self.source_usage.values())

    def row_source_id(self, row: Any) -> int | None:
        if not isinstance(row, dict):
            return None
        # A generated row cut from several documents: the most restrictive one decides.
        many = row.get("source_document_ids")
        if isinstance(many, (list, tuple)) and many:
            found: list[int] = []
            for doc_id in many:
                try:
                    if int(doc_id) in self.doc_source:
                        found.append(self.doc_source[int(doc_id)])
                except (TypeError, ValueError):
                    continue
            if found:
                rank = {USAGE_RAG: 0, USAGE_EVAL: 1}
                return min(found, key=lambda sid: rank.get(self.source_usage.get(sid, USAGE_ANY), 2))
        for key in ("_source_document_id", "source_document_id"):
            doc_id = row.get(key)
            try:
                if doc_id is not None and int(doc_id) in self.doc_source:
                    return self.doc_source[int(doc_id)]
            except (TypeError, ValueError):
                continue
        sid = row.get("data_source_id")
        try:
            if sid is not None and int(sid) in self.sources:
                return int(sid)
        except (TypeError, ValueError):
            pass
        return None

    def row_usage(self, row: Any) -> str:
        sid = self.row_source_id(row)
        return self.source_usage.get(sid, USAGE_ANY) if sid is not None else USAGE_ANY

    def document_usage(self, doc_id: int) -> str:
        sid = self.doc_source.get(int(doc_id))
        return self.source_usage.get(sid, USAGE_ANY) if sid is not None else USAGE_ANY

    def partition(
        self, rows: list[dict[str, Any]], state: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """The rows that may enter the train / val pool. ``eval`` rows go to
        ``state["eval_rows"]``, ``rag`` rows are counted in
        ``state["excluded_rag"]`` / ``state["excluded_by_source"]``. Returns
        ``rows`` itself when nothing is restricted."""
        if not self.restricted:
            return rows
        kept: list[dict[str, Any]] = []
        for row in rows:
            usage = self.row_usage(row)
            if usage == USAGE_EVAL:
                if state is not None:
                    state.setdefault("eval_rows", []).append(row)
            elif usage == USAGE_RAG:
                if state is not None:
                    state["excluded_rag"] = state.get("excluded_rag", 0) + 1
                    by = state.setdefault("excluded_by_source", {})
                    sid = self.row_source_id(row)
                    by[sid] = by.get(sid, 0) + 1
            else:
                kept.append(row)
        return kept


async def build_usage_context(db: AsyncSession, project_id: int) -> UsageContext:
    """Attach legacy documents, then read the project's usage maps."""
    await ensure_sources(db, project_id)
    sources = await _project_sources(db, project_id)
    ctx = UsageContext()
    for s in sources:
        ctx.source_usage[int(s.id)] = effective_usage(s)
        ctx.sources[int(s.id)] = {
            "title": s.title, "kind": s.source_kind, "licence": s.licence, "usage": effective_usage(s),
        }
    for doc in await _project_documents(db, project_id):
        if doc.data_source_id is not None:
            ctx.doc_source[int(doc.id)] = int(doc.data_source_id)
    return ctx


def dedup_key(row: dict[str, Any]) -> str:
    """The exact-duplicate key ``split_dataset``'s dedup uses."""
    from app.services.data_health_service import _normalise_for_exact
    from app.services.trainability_forecast_service import _row_to_text

    return _normalise_for_exact(_row_to_text(row))


def prepared_provenance(
    ctx: UsageContext,
    splits: dict[str, list[dict[str, Any]]],
    state: dict[str, Any],
    *,
    eval_routed_to_test: int,
    eval_not_added_pinned_test: int,
) -> dict[str, Any]:
    """The prepared manifest's ``provenance`` section."""
    by_source: dict[int, dict[str, int]] = {}
    unattributed = {"train": 0, "val": 0, "test": 0}
    for split_name, rows in splits.items():
        if split_name not in unattributed:
            continue
        for row in rows:
            sid = ctx.row_source_id(row)
            if sid is None:
                unattributed[split_name] += 1
                continue
            bucket = by_source.setdefault(sid, {"train": 0, "val": 0, "test": 0, "excluded": 0})
            bucket[split_name] += 1
    for sid, count in (state.get("excluded_by_source") or {}).items():
        if sid is None:
            continue
        bucket = by_source.setdefault(sid, {"train": 0, "val": 0, "test": 0, "excluded": 0})
        bucket["excluded"] += count
    out_sources: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    for sid in sorted(by_source):
        info = ctx.sources.get(sid, {})
        counts = by_source[sid]
        out_sources[str(sid)] = {
            "title": info.get("title"), "kind": info.get("kind"), "licence": info.get("licence"),
            "usage": info.get("usage", USAGE_ANY), **counts,
        }
        if counts["train"] + counts["val"] > 0:
            for flag in licence_flags(info.get("licence")):
                warnings.append(_warning(sid, info, flag))
    return {
        "by_source": out_sources,
        "unattributed": unattributed,
        "excluded_rag": int(state.get("excluded_rag", 0)),
        "eval_routed_to_test": int(eval_routed_to_test),
        "eval_not_added_pinned_test": int(eval_not_added_pinned_test),
        "licence_warnings": warnings,
    }


def _warning(source_id: Any, info: dict[str, Any], flag: str) -> dict[str, Any]:
    title = info.get("title") or f"source {source_id}"
    if flag == FLAG_UNKNOWN:
        message = f"{title}: licence unknown."
    else:
        message = f"{title}: non-commercial licence ({info.get('licence')})."
    return {"source_id": int(source_id) if str(source_id).isdigit() else source_id, "title": info.get("title"),
            "flag": flag, "licence": info.get("licence"), "message": message}


def provenance_totals(provenance: dict[str, Any] | None) -> dict[str, Any]:
    """Compact totals of a prepared ``provenance`` section for a training run."""
    if not isinstance(provenance, dict) or not provenance:
        return {}
    by_source = provenance.get("by_source") or {}
    return {
        "by_source": by_source,
        "unattributed": provenance.get("unattributed") or {},
        "excluded_rag": provenance.get("excluded_rag", 0),
        "eval_routed_to_test": provenance.get("eval_routed_to_test", 0),
        "eval_not_added_pinned_test": provenance.get("eval_not_added_pinned_test", 0),
        "licence_warnings": provenance.get("licence_warnings") or [],
        "sources": len(by_source),
    }


def read_prepared_provenance(project_id: int) -> dict[str, Any] | None:
    path = settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    prov = manifest.get("provenance") if isinstance(manifest, dict) else None
    return prov if isinstance(prov, dict) else None


# ── listing + editing ───────────────────────────────────────────────────


def _doc_rows(doc: RawDocument) -> int:
    meta = doc.metadata_ if isinstance(doc.metadata_, dict) else {}
    for key in ("num_samples", "row_count", "rows_kept"):
        try:
            value = int(meta.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value:
            return value
    return int(doc.chunk_count or 0)


def _import_row_counts(project_id: int) -> dict[int, int]:
    """Rows per ``data_source_id`` in the project's synthetic file."""
    from app.services.synthetic_service import _synthetic_dir

    path = _synthetic_dir(project_id) / "synthetic.jsonl"
    counts: dict[int, int] = {}
    if not path.exists():
        return counts
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if '"data_source_id"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = row.get("data_source_id") if isinstance(row, dict) else None
                if isinstance(sid, int):
                    counts[sid] = counts.get(sid, 0) + 1
    except OSError:
        pass
    return counts


def serialize_source(
    source: DataSource, *, document_count: int = 0, row_count: int = 0,
    in_prepared: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "id": int(source.id),
        "source_kind": source.source_kind,
        "title": source.title,
        "locator": source.locator,
        "source_ref": source.source_ref,
        "licence": source.licence,
        "licence_verified": bool(source.licence_verified),
        "licence_verified_by": source.licence_verified_by,
        "licence_verified_at": source.licence_verified_at.isoformat() if source.licence_verified_at else None,
        "retrieved_at": source.retrieved_at.isoformat() if source.retrieved_at else None,
        "usage": effective_usage(source),
        "usage_locked": source.source_kind == KIND_PREDEFINED_SPLIT,
        "document_count": document_count,
        "row_count": row_count,
        "licence_flags": licence_flags(source.licence),
        "in_prepared": in_prepared,
    }


async def list_sources(db: AsyncSession, project_id: int) -> dict[str, Any]:
    docs = await _project_documents(db, project_id)
    await ensure_sources(db, project_id, docs)
    sources = await _project_sources(db, project_id)
    doc_count: dict[int, int] = {}
    row_count: dict[int, int] = {}
    for doc in docs:
        if doc.data_source_id is None:
            continue
        sid = int(doc.data_source_id)
        doc_count[sid] = doc_count.get(sid, 0) + 1
        row_count[sid] = row_count.get(sid, 0) + _doc_rows(doc)
    if any(s.source_kind == KIND_DATASET_IMPORT for s in sources):
        for sid, n in _import_row_counts(project_id).items():
            row_count[sid] = row_count.get(sid, 0) + n
    prepared_prov = read_prepared_provenance(project_id)
    prepared = (prepared_prov or {}).get("by_source") or {}
    items = []
    for s in sources:
        counts = prepared.get(str(s.id))
        if isinstance(counts, dict):
            in_prepared = {k: int(counts.get(k, 0)) for k in ("train", "val", "test", "excluded")}
        elif prepared_prov is not None:
            # A prepared manifest exists and this source contributed nothing to it.
            in_prepared = {"train": 0, "val": 0, "test": 0, "excluded": 0}
        else:
            in_prepared = None
        items.append(serialize_source(
            s, document_count=doc_count.get(int(s.id), 0), row_count=row_count.get(int(s.id), 0),
            in_prepared=in_prepared,
        ))
    return {
        "sources": items,
        "totals": {
            "sources": len(items),
            "documents": sum(doc_count.values()),
            "unknown_licence": sum(1 for i in items if FLAG_UNKNOWN in i["licence_flags"]),
            "non_commercial": sum(1 for i in items if FLAG_NON_COMMERCIAL in i["licence_flags"]),
        },
    }


async def update_sources(
    db: AsyncSession,
    project_id: int,
    source_ids: list[int],
    *,
    licence: Any = UNSET,
    licence_verified: Any = UNSET,
    usage: Any = UNSET,
    user: str | None = None,
) -> list[DataSource]:
    """Bulk edit. All-or-nothing: every id and value is checked before any
    change. Setting a licence resets ``licence_verified`` unless the same call
    sets it; verifying stamps who and when. Member documents' ``license_info``
    follows the source licence."""
    wanted = list(dict.fromkeys(int(i) for i in source_ids))
    by_id = {int(s.id): s for s in await _project_sources(db, project_id)}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise SourceNotFoundError(f"Sources not in this project: {missing}")
    selected = [by_id[i] for i in wanted]
    if usage is not UNSET:
        usage = validate_usage(usage)
        locked = [int(s.id) for s in selected if s.source_kind == KIND_PREDEFINED_SPLIT]
        if locked:
            raise UsageLockedError(locked)
    now = _now()
    for source in selected:
        if licence is not UNSET:
            new_licence = normalize_licence(licence)
            if new_licence != source.licence:
                source.licence_verified = False
                source.licence_verified_by = None
                source.licence_verified_at = None
            source.licence = new_licence
        if licence_verified is not UNSET:
            source.licence_verified = bool(licence_verified)
            source.licence_verified_by = (user or "local") if licence_verified else None
            source.licence_verified_at = now if licence_verified else None
        if usage is not UNSET:
            source.usage = usage
        source.updated_at = now
    if licence is not UNSET:
        chosen = {int(s.id): s for s in selected}
        for doc in await _project_documents(db, project_id):
            if doc.data_source_id is not None and int(doc.data_source_id) in chosen:
                doc.license_info = chosen[int(doc.data_source_id)].licence or ""
    await db.flush()
    return selected


# ── documents API enrichment ────────────────────────────────────────────


async def enrich_document_responses(db: AsyncSession, project_id: int, responses: list[Any]) -> None:
    """Fill ``usage`` / ``licence`` / ``licence_flags`` of DocumentResponse
    objects from their source (in place)."""
    sources = {int(s.id): s for s in await _project_sources(db, project_id)}
    if not sources:
        return
    for response in responses:
        sid = getattr(response, "data_source_id", None)
        source = sources.get(int(sid)) if sid is not None else None
        if source is None:
            continue
        response.usage = effective_usage(source)
        response.licence = source.licence
        response.licence_flags = licence_flags(source.licence)


# ── export ──────────────────────────────────────────────────────────────


async def export_provenance(db: AsyncSession, project_id: int, experiment_id: int | None) -> dict[str, Any]:
    """The ``provenance`` block of an export manifest: every source with the
    rows it put in train / val / test, rows per licence, and licence warnings.
    Counts come from the training manifest captured for the experiment when
    it has them (what the run trained on), else the current prepared manifest;
    ``counts_from`` says which."""
    from app.models.training_manifest import TrainingManifest

    counts: dict[str, Any] | None = None
    counts_from: str | None = None
    if experiment_id is not None:
        row = (
            await db.execute(select(TrainingManifest).where(TrainingManifest.experiment_id == experiment_id))
        ).scalar_one_or_none()
        recorded = (row.data_provenance if row is not None else None) or {}
        if isinstance(recorded.get("by_source"), dict) and recorded["by_source"]:
            counts, counts_from = recorded["by_source"], "training_manifest"
    if counts is None:
        current = read_prepared_provenance(project_id)
        if current and isinstance(current.get("by_source"), dict):
            counts, counts_from = current["by_source"], "current_prepared_manifest"

    sources = await _project_sources(db, project_id)
    out: list[dict[str, Any]] = []
    licences: dict[str, int] = {}
    warnings: list[dict[str, Any]] = []
    for s in sources:
        c = (counts or {}).get(str(s.id)) or {}
        entry = {
            "id": int(s.id), "kind": s.source_kind, "title": s.title, "locator": s.locator, "ref": s.source_ref,
            "licence": s.licence, "licence_verified": bool(s.licence_verified), "usage": effective_usage(s),
            "retrieved_at": s.retrieved_at.isoformat() if s.retrieved_at else None,
            "train": int(c.get("train", 0)), "val": int(c.get("val", 0)), "test": int(c.get("test", 0)),
            "excluded": int(c.get("excluded", 0)),
        }
        out.append(entry)
        trains = entry["train"] + entry["val"]
        # With counts, only sources that fed the model; without, every source that could.
        feeds = trains > 0 if counts is not None else effective_usage(s) in (USAGE_ANY, "train")
        if feeds:
            key = s.licence or FLAG_UNKNOWN
            licences[key] = licences.get(key, 0) + trains
            for flag in licence_flags(s.licence):
                warnings.append(_warning(s.id, {"title": s.title, "licence": s.licence}, flag))
    return {"counts_from": counts_from, "sources": out, "licences": licences, "warnings": warnings}


def release_notes_section(provenance: dict[str, Any]) -> str:
    """"Data sources & licences" for RELEASE_NOTES.md ("" for no sources)."""
    sources = provenance.get("sources") or []
    if not sources:
        return ""
    lines = ["", "## Data sources & licences", ""]
    for s in sources:
        where = s.get("locator") or ""
        if s.get("ref"):
            where = f"{where} @ {str(s['ref'])[:12]}"
        verified = "verified" if s.get("licence_verified") else "not verified"
        lines.append(
            f"- {s.get('title')} ({s.get('kind')}{', ' + where if where else ''}): "
            f"licence {s.get('licence') or 'unknown'} ({verified}); usage {s.get('usage')}; "
            f"rows train {s.get('train', 0)} / val {s.get('val', 0)} / test {s.get('test', 0)}"
        )
    warnings = provenance.get("warnings") or []
    if warnings:
        lines += ["", "Warnings:", ""]
        lines += [f"- {w.get('message')}" for w in warnings]
    lines.append("")
    return "\n".join(lines)
