"""Protocols + dataclasses for the dataset-import pipeline.

Three contract types:

- ``DatasetSource`` — a loader for an external system (jsonl / csv /
  hf / kaggle / …). Streams raw row dicts.
- ``TargetMapper`` — transforms raw rows into the canonical shape one
  of BrewSLM's task handlers consumes (classification, span_set
  structured-extraction, qa, seq2seq, alignment preference pair, etc).
- ``ProposedMapping`` — the introspector's output (Phase B). Phase A
  ships the dataclass + the contract so callers can already start
  passing manually-constructed mappings.

Plus :class:`RawRow` / :class:`TransformedRow` / :class:`RejectedRow`
to give every layer a stable shape to talk through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable


# ── Row shapes ────────────────────────────────────────────────────────


RawRow = dict[str, Any]
"""One row exactly as the source emitted it. Column names + values
are source-defined."""


@dataclass
class TransformedRow:
    """A row after the target mapper has run.

    ``payload`` carries the canonical fields the task handler consumes
    (e.g. ``{text, entities_json}`` for span_set, ``{text, label}`` for
    classification). ``row_key`` is a source-stable identifier so
    re-imports can deduplicate; ``warnings`` collects soft issues that
    don't justify rejecting the row.
    """

    payload: dict[str, Any]
    row_key: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class RejectedRow:
    """A row the mapper couldn't process.

    Rejection reasons use stable codes (e.g. ``missing_field``,
    ``offset_mismatch``, ``schema_violation``) so the UI / CLI can
    group + bulk-drop by reason rather than treating rejects
    individually. Reference:
    [[rejected-rows-selectable]] for the bulk-drop UX contract.
    """

    raw_row: RawRow
    reason: str
    detail: str = ""
    row_index: int | None = None


# ── Source contract ──────────────────────────────────────────────────


@runtime_checkable
class DatasetSource(Protocol):
    """Connector for a single external system."""

    source_id: str

    def load(
        self,
        locator: str,
        *,
        limit: int | None = None,
    ) -> Iterable[RawRow]:
        """Stream raw rows from the external system.

        ``locator`` is the source-specific reference: a path for local
        files, a dataset id for HF, a competition slug for Kaggle, etc.
        Implementations should yield rows lazily so large datasets
        don't blow memory.
        """
        ...

    def describe(self, locator: str) -> dict[str, Any]:
        """Return metadata for the introspector (Phase B) and the UI:
        column names, sample rows, total row count when cheaply
        knowable, license string when available."""
        ...


# ── Target mapper contract ───────────────────────────────────────────


@runtime_checkable
class TargetMapper(Protocol):
    """Transforms raw rows into one of the task handler's canonical
    shapes. Mappers are domain-agnostic — a single ``bio_to_spans``
    serves PII, medical, legal, financial NER. The entity-type mapping
    is part of the per-field config, not the mapper's code.
    """

    mapper_id: str

    def declared_target(self) -> str:
        """The ``task_profile`` this mapper feeds (e.g. ``classification``,
        ``structured_extraction``, ``qa``, ``seq2seq``, ``dpo``).
        Used by the orchestrator to validate the project's
        ``task_profile`` matches what the mapper produces."""
        ...

    def transform(
        self,
        rows: Iterable[RawRow],
        field_map: dict[str, Any],
        *,
        ctx: "ImportContext",
    ) -> Iterable[TransformedRow | RejectedRow]:
        """Walk raw rows and yield ``TransformedRow`` for each accepted
        row or ``RejectedRow`` with a stable reason code for each
        skipped row. Mappers must NEVER silently drop a row — the
        ``rejected-rows-selectable`` UX contract requires per-row
        accountability.
        """
        ...


# ── Context + result ─────────────────────────────────────────────────


@dataclass
class ImportContext:
    """Carries project / mapping context to every layer.

    The orchestrator builds this once at the top of an import run and
    passes it through. Mappers read ``project_task_profile`` to
    validate they're being used in a compatible project; sources read
    ``project_id`` only to scope artifact paths.
    """

    project_id: int
    project_task_profile: str | None
    source_id: str
    mapper_id: str
    locator: str
    field_map: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposedMapping:
    """Phase B output — the introspector's mapping proposal.

    Shipped in Phase A so manual callers can construct one directly,
    populating ``mapper_id`` + ``field_map`` + ``target_task_profile``
    by hand. The Phase B introspector will produce these
    automatically from sampled rows.
    """

    target_task_profile: str
    mapper_id: str
    field_map: dict[str, Any]
    confidence: float = 1.0
    rationale: str = "manual"
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    """Summary returned by the orchestrator.

    ``accepted_rows`` is the canonical-shape sample (capped at
    ``preview_limit`` in preview mode, full run otherwise).
    ``rejection_counts`` groups rejected rows by reason code so the
    UI / CLI can render the bulk-drop selection — see
    [[rejected-rows-selectable]] for the contract.
    """

    accepted_rows: list[TransformedRow]
    rejected_rows: list[RejectedRow]
    rejection_counts: dict[str, int]
    accepted_count: int
    rejected_count: int
    source_id: str
    mapper_id: str
    target_task_profile: str
    locator: str
    written_path: str | None = None
    dry_run: bool = False
    warnings: list[str] = field(default_factory=list)
