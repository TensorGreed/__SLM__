"""Generic dataset import pipeline (Phase A of DATASET_IMPORT_PLAN.md).

Three pluggable layers — source loaders, schema introspector (Phase B),
target mappers — orchestrated by :mod:`service`. Source connectors and
mappers register themselves at module-load time via
:func:`register_source` and :func:`register_mapper`; the orchestrator
dispatches on the locator's prefix (e.g. ``jsonl:`` / ``csv:``) and
the explicit ``mapper_id``.

Side-effect import: importing this package triggers registration of
the built-in sources + mappers. Tests that need a clean registry
should snapshot ``_SOURCES`` / ``_MAPPERS`` before each test.
"""

from __future__ import annotations

# Force registration of built-in sources + mappers on package import
# so the registry is populated by the time anyone calls
# ``resolve_source`` / ``resolve_mapper``.
from app.services.dataset_import import sources as _sources  # noqa: F401
from app.services.dataset_import import mappers as _mappers  # noqa: F401
from app.services.dataset_import.protocols import (
    DatasetSource,
    ImportContext,
    ImportResult,
    ProposedMapping,
    RawRow,
    RejectedRow,
    TargetMapper,
    TransformedRow,
)
from app.services.dataset_import.registry import (
    list_registered_mappers,
    list_registered_sources,
    register_mapper,
    register_source,
    resolve_mapper,
    resolve_source,
    split_locator,
)


__all__ = [
    "DatasetSource",
    "ImportContext",
    "ImportResult",
    "ProposedMapping",
    "RawRow",
    "RejectedRow",
    "TargetMapper",
    "TransformedRow",
    "list_registered_mappers",
    "list_registered_sources",
    "register_mapper",
    "register_source",
    "resolve_mapper",
    "resolve_source",
    "split_locator",
]
