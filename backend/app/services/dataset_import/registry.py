"""Source + mapper registry for the dataset-import pipeline.

Two parallel registries (sources, mappers) with the same dispatcher
shape as :mod:`app.services.eval_task_handler_service`. Connectors
register themselves at import time via ``register_source`` /
``register_mapper``; the orchestrator dispatches on the locator's
prefix for sources and the explicit ``mapper_id`` for mappers.

The lookup is intentionally tolerant: unknown ids raise ``KeyError``
with a list of registered alternatives, so the API + CLI can surface
a useful error message instead of a 500.
"""

from __future__ import annotations

from typing import Callable

from app.services.dataset_import.protocols import DatasetSource, TargetMapper


_SOURCES: dict[str, Callable[[], DatasetSource]] = {}
_MAPPERS: dict[str, Callable[[], TargetMapper]] = {}


# ── Sources ──────────────────────────────────────────────────────────


def register_source(source_id: str, factory: Callable[[], DatasetSource]) -> None:
    """Register a source connector. Replacing an existing id is allowed
    (useful for tests) — production code should never register the
    same id twice, but we don't enforce uniqueness in case a hot-
    reload needs to swap a factory."""

    key = _normalize_id(source_id)
    if not key:
        raise ValueError("register_source requires a non-empty source_id")
    _SOURCES[key] = factory


def resolve_source(source_id: str) -> DatasetSource:
    """Return an instance of the registered source connector, or raise
    ``KeyError`` with a useful message naming the registered
    alternatives."""

    key = _normalize_id(source_id)
    factory = _SOURCES.get(key)
    if factory is None:
        registered = sorted(_SOURCES.keys())
        raise KeyError(
            f"no source registered for '{source_id}'. "
            f"Registered sources: {registered or '<none>'}"
        )
    return factory()


def list_registered_sources() -> list[str]:
    return sorted(_SOURCES.keys())


# ── Mappers ──────────────────────────────────────────────────────────


def register_mapper(mapper_id: str, factory: Callable[[], TargetMapper]) -> None:
    key = _normalize_id(mapper_id)
    if not key:
        raise ValueError("register_mapper requires a non-empty mapper_id")
    _MAPPERS[key] = factory


def resolve_mapper(mapper_id: str) -> TargetMapper:
    key = _normalize_id(mapper_id)
    factory = _MAPPERS.get(key)
    if factory is None:
        registered = sorted(_MAPPERS.keys())
        raise KeyError(
            f"no mapper registered for '{mapper_id}'. "
            f"Registered mappers: {registered or '<none>'}"
        )
    return factory()


def list_registered_mappers() -> list[str]:
    return sorted(_MAPPERS.keys())


# ── Locator parsing ──────────────────────────────────────────────────


def split_locator(locator: str) -> tuple[str, str]:
    """Split a ``<source_id>:<rest>`` locator into ``(source_id, rest)``.

    Examples:
        >>> split_locator("jsonl:/tmp/data.jsonl")
        ('jsonl', '/tmp/data.jsonl')
        >>> split_locator("hf:ai4privacy/pii-masking-200k:train")
        ('hf', 'ai4privacy/pii-masking-200k:train')

    Raises ``ValueError`` when there's no colon — the orchestrator
    treats this as a user error worth surfacing rather than guessing.
    """

    if ":" not in locator:
        raise ValueError(
            f"locator '{locator}' is missing a source prefix "
            f"(expected '<source_id>:<rest>', e.g. 'jsonl:/path/to/file')"
        )
    source_id, _, rest = locator.partition(":")
    return _normalize_id(source_id), rest


# ── Helpers ──────────────────────────────────────────────────────────


def _normalize_id(value: str) -> str:
    return str(value or "").strip().lower()
