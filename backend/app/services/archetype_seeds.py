"""Template-derived seed contributions for the archetype cohort
(USER-SUCCESS Epic 8 Phase 8a).

Each of the 8 shipped project templates carries a curated
``gold.jsonl`` file. We treat each template as a synthetic
"passing project" of its associated recipe so a fresh install
gets a usable archetype on day 1 — before the user has trained
anything of their own.

Mapping below is hand-stamped (not read from each manifest's
``recipe_id`` field) because (a) the mapping is stable, (b)
reading 8 manifest files at module import adds startup cost
for zero flexibility benefit, and (c) the recipe associations
are part of the platform's product surface — they're documented,
not derived. Verified against the manifests' ``recipe_id`` keys
at the time this module was written:

  ticket-router → classification
  log-triage → classification
  email-chat-tone → generic-sft
  data-to-sql → generic-sft
  agent-tool-call → generic-sft
  policy-qa-style → qa-sft
  security-alert-summarizer → summarization
  contract-clause-extractor → span-extraction

Per-recipe coverage from templates:
  * classification: 2 templates
  * generic-sft: 3 templates
  * qa-sft / summarization / span-extraction: 1 template each
  * code-review: 0 templates (no shipped template; relies entirely
    on user-grown projects)

Seeded contributions are computed lazily on first request per
recipe and memoised — the gold files don't change at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from app.config import settings


# Recipe → list of template slugs that seed it.
_RECIPE_TO_TEMPLATE_SLUGS: dict[str, list[str]] = {
    "classification": ["ticket-router", "log-triage"],
    "generic-sft": ["email-chat-tone", "data-to-sql", "agent-tool-call"],
    "qa-sft": ["policy-qa-style"],
    "summarization": ["security-alert-summarizer"],
    "span-extraction": ["contract-clause-extractor"],
    # code-review intentionally absent — no shipped template.
}


class SeedContribution(TypedDict):
    """One template's contribution to a recipe's archetype cohort.
    Shape matches what ``archetype_service.compute_recipe_archetype``
    feeds into its aggregator."""

    pseudo_id: int                        # negative, so it can't collide with real projects
    name: str                             # human-readable, e.g. "Template · ticket-router"
    template_slug: str
    features: dict[str, float | None]     # feature_id → value, per archetype_service.extract_features_from_rows


_SEED_CACHE: dict[str, list[SeedContribution]] = {}


def _template_gold_path(slug: str) -> Path:
    """Resolve a template's gold.jsonl path on disk. Templates live
    at ``backend/data/project_templates/<slug>/gold.jsonl``. We
    derive the root from the settings module to keep test-time
    overrides (``DATA_DIR``) honored where applicable, but the
    templates themselves ship inside the backend tree."""
    # Templates ship with the codebase, not in the user's runtime
    # data dir. Walk from this module up to backend/ then down.
    backend_root = Path(__file__).resolve().parents[2]
    return backend_root / "data" / "project_templates" / slug / "gold.jsonl"


def _read_gold_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one template's gold rows. Best-effort: malformed lines
    are silently skipped (the template files are curated, but
    defending against a single bad line is cheap)."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _pseudo_id_for(slug: str) -> int:
    """Stable negative integer derived from the template slug. Real
    project IDs are positive autoincrement; negatives can't
    collide. Hash-mod keeps the value small enough to display
    comfortably in tooltips."""
    return -(abs(hash(slug)) % 1_000_000 + 1)


def load_seed_contributions(recipe_id: str) -> list[SeedContribution]:
    """Return the seeded contributions for ``recipe_id`` — one per
    template mapped to that recipe. Empty list for recipes with
    no shipped template (today: ``code-review``).

    Memoised after first call; the gold files are immutable at
    runtime, so re-reading would waste cycles."""
    cached = _SEED_CACHE.get(recipe_id)
    if cached is not None:
        return cached

    slugs = _RECIPE_TO_TEMPLATE_SLUGS.get(recipe_id, [])
    if not slugs:
        _SEED_CACHE[recipe_id] = []
        return []

    # Import here to avoid a circular-import (archetype_service
    # imports this module from inside compute_recipe_archetype).
    from app.services.archetype_service import extract_features_from_rows

    contributions: list[SeedContribution] = []
    for slug in slugs:
        rows = _read_gold_jsonl(_template_gold_path(slug))
        if not rows:
            # Missing template gold file = skip silently. Could
            # happen in dev when working on a non-default DATA_DIR.
            continue
        features = extract_features_from_rows(rows, recipe_id=recipe_id)
        contributions.append({
            "pseudo_id": _pseudo_id_for(slug),
            "name": f"Template · {slug}",
            "template_slug": slug,
            "features": features,
        })

    _SEED_CACHE[recipe_id] = contributions
    return contributions


def clear_seed_cache() -> None:
    """Test hook so unit tests can re-trigger the load (e.g. after
    mocking a different template path)."""
    _SEED_CACHE.clear()
