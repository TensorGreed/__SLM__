"""Cross-project gold-set archetypes (USER-SUCCESS Epic 8 Phase 8a).

Walks the user's local successful projects and emits a per-recipe
"archetype" — a distribution-stats payload over a handful of
structural features (row count, class entropy, hard-negative ratio,
input/output length percentiles, gold-set diversity). A new user
starting a project of the same recipe can then ask "what shape do
successful projects have?" and get back concrete percentile ranges
+ the cohort that contributed them.

**Privacy-preserving by construction**: the payload contains ONLY
distribution stats (percentiles, ratios, counts) + project pointers
(id, name, f1, source). Never raw training rows. The original epic
called this out explicitly; the contract is enforced by the
``FeatureDistribution`` / ``RecipeArchetype`` shapes here — they
have no field for raw row data.

**Template seeds**: when fewer than ``_MIN_USER_PROJECTS_BEFORE_SEED_THRESHOLD``
user projects pass for a recipe, the archetype merges in seeded
contributions from the shipped project templates (see
``archetype_seeds.py``). A fresh install therefore gets a usable
archetype on day 1; user-grown data augments the seeds as it
accumulates.

**Cache**: module-level dict with a 5-minute TTL keyed by
``recipe_id``. Recomputing across all of a single user's projects
is ~50-200ms (per-project file read + tokenization); caching keeps
the endpoint snappy without per-project invalidation complexity.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType
from app.models.experiment import EvalResult, Experiment, ExperimentStatus
from app.models.project import Project
from app.services.dataset_service import _load_records_from_file

# Reuse the helpers that the trainability-forecast service already
# battle-tested on the same row shapes. Keeping these in one place
# means a future fix (e.g. a smarter label extractor) lands once.
from app.services.trainability_forecast_service import (
    _extract_classification_labels,
    _mean_pairwise_jaccard,
    _row_to_text,
    _tokenize,
)


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────


# A project counts as "passing" if its latest EvalResult's pass_rate
# is at or above this threshold. 0.6 chosen because Phase 9c's
# policy-qa-style ran at ~0.18 and would have polluted archetypes
# at a lower bar; 0.6 reads as "the model is producing something
# useful, not just gibberish."
_PASSING_F1_THRESHOLD: float = 0.6

# Below this user-project count for a given recipe, the archetype
# merges in seeded contributions from the shipped templates so a
# fresh install still produces useful output.
_MIN_USER_PROJECTS_BEFORE_SEED_THRESHOLD: int = 3

# Module-level cache TTL. Archetype computation is sub-second for
# any reasonable user-project count; 5 minutes is the right
# refresh cadence for "user just trained a new project."
_CACHE_TTL_SECONDS: float = 300.0


# Per-feature applicability — gates each check on the recipe it
# applies to. Recipe-agnostic plumbing per [keep-brewslm-general]:
# adding a new recipe means updating this map, not rewriting the
# orchestrator.
_FEATURE_APPLICABILITY: dict[str, frozenset[str]] = {
    "row_count": frozenset({
        "qa-sft", "classification", "span-extraction",
        "summarization", "code-review", "generic-sft",
    }),
    "class_entropy": frozenset({"classification"}),
    "class_balance_ratio": frozenset({"classification"}),
    "hard_negative_ratio": frozenset({
        "classification", "span-extraction", "code-review", "generic-sft",
    }),
    "input_length_chars": frozenset({
        "qa-sft", "classification", "span-extraction",
        "summarization", "code-review", "generic-sft",
    }),
    "output_length_chars": frozenset({
        "qa-sft", "summarization", "code-review", "generic-sft",
    }),
    "goldset_diversity": frozenset({
        "qa-sft", "classification", "span-extraction",
        "summarization", "code-review", "generic-sft",
    }),
}


_FEATURE_LABELS: dict[str, str] = {
    "row_count": "Total training rows",
    "class_entropy": "Class label entropy (bits)",
    "class_balance_ratio": "Min/max class balance",
    "hard_negative_ratio": "Hard-negative share of synth",
    "input_length_chars": "Input length (chars)",
    "output_length_chars": "Output length (chars)",
    "goldset_diversity": "Gold-set diversity (1 - mean pairwise Jaccard)",
}


_FEATURE_UNITS: dict[str, str] = {
    "row_count": "rows",
    "class_entropy": "bits",
    "class_balance_ratio": "ratio",
    "hard_negative_ratio": "ratio",
    "input_length_chars": "chars",
    "output_length_chars": "chars",
    "goldset_diversity": "ratio",
}


# ─────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────


class FeatureDistribution(TypedDict):
    """Per-feature percentile stats across the cohort. No raw values."""

    feature_id: str
    label: str
    n_projects: int                       # cohort projects contributing this feature
    p25: float | None
    p50: float | None
    p75: float | None
    mean: float | None
    min: float | None
    max: float | None
    unit: str


class CohortMember(TypedDict):
    """Pointer to one project in the cohort. Source = 'user' or
    'template'; templates carry synthetic ids (negative ints) so
    they never collide with real projects."""

    id: int
    name: str
    source: Literal["user", "template"]
    pass_rate: float | None               # None for template seeds


class RecipeArchetype(TypedDict):
    recipe_id: str
    n_passing_projects: int               # cohort size incl. template seeds
    n_user_projects: int                  # excluding seeds
    n_template_seeds: int                 # template-derived contributions
    computed_at: str                      # ISO timestamp
    features: list[FeatureDistribution]
    cohort_provenance: list[CohortMember] # for transparency surfaces


# ─────────────────────────────────────────────────────────────────────
# Per-project feature extraction — pure functions
# ─────────────────────────────────────────────────────────────────────


# A "project sample" is what feature checks operate on. Decoupled
# from the DB read so template seeds (which have no Project row)
# can be fed through the same pipeline.
ProjectSample = dict[str, Any]


def extract_features_from_rows(
    rows: list[dict[str, Any]],
    *,
    recipe_id: str,
) -> dict[str, float | None]:
    """Compute one project's contribution to every applicable
    feature. Returns a feature_id → value map; values are None
    when the feature applies but the project's data can't support
    it (e.g. zero classifiable labels for class_entropy)."""
    out: dict[str, float | None] = {}

    if "row_count" in _applicable_features(recipe_id):
        out["row_count"] = float(len(rows))

    if "class_entropy" in _applicable_features(recipe_id):
        labels = _extract_classification_labels(rows)
        out["class_entropy"] = _shannon_entropy(labels) if labels else None

    if "class_balance_ratio" in _applicable_features(recipe_id):
        labels = _extract_classification_labels(rows)
        out["class_balance_ratio"] = (
            _min_over_max_class_ratio(labels) if labels else None
        )

    if "hard_negative_ratio" in _applicable_features(recipe_id):
        out["hard_negative_ratio"] = _hard_negative_ratio(rows)

    if "input_length_chars" in _applicable_features(recipe_id):
        lengths = _per_row_input_lengths(rows)
        # For length features we want the project's MEDIAN length to
        # contribute to the cohort distribution — otherwise outliers
        # in any single project would dominate. The orchestrator
        # then takes percentiles across project-medians.
        out["input_length_chars"] = (
            float(statistics.median(lengths)) if lengths else None
        )

    if "output_length_chars" in _applicable_features(recipe_id):
        lengths = _per_row_output_lengths(rows)
        out["output_length_chars"] = (
            float(statistics.median(lengths)) if lengths else None
        )

    if "goldset_diversity" in _applicable_features(recipe_id):
        out["goldset_diversity"] = _diversity_score(rows)

    return out


def _applicable_features(recipe_id: str) -> set[str]:
    return {fid for fid, recipes in _FEATURE_APPLICABILITY.items() if recipe_id in recipes}


def _shannon_entropy(labels: list[str]) -> float:
    """Standard Shannon entropy in bits. 50/50 split → 1.0; all-one
    class → 0.0. Returns 0.0 for an empty list (caller gates that)."""
    if not labels:
        return 0.0
    total = len(labels)
    counter = Counter(labels)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
        if count > 0
    )


def _min_over_max_class_ratio(labels: list[str]) -> float | None:
    """min(class_count) / max(class_count). 1.0 = perfectly balanced;
    near-0 = one dominant class. Returns None when fewer than 2
    classes present (ratio is undefined / trivially 1.0)."""
    counter = Counter(labels)
    counts = sorted(counter.values())
    if len(counts) < 2:
        return None
    return counts[0] / counts[-1] if counts[-1] > 0 else None


def _hard_negative_ratio(rows: list[dict[str, Any]]) -> float | None:
    """Share of accepted synth rows whose ``synth_source`` contains
    'hard_negatives'. Returns None when the project has no synth
    rows at all — we can't say "0%" because the feature simply
    doesn't apply to that project."""
    synth_rows = [r for r in rows if r.get("synth_source")]
    if not synth_rows:
        return None
    hard = sum(
        1
        for r in synth_rows
        if "hard_negatives" in str(r.get("synth_source") or "").lower()
    )
    return hard / len(synth_rows)


def _per_row_input_lengths(rows: list[dict[str, Any]]) -> list[int]:
    """Per-row input character lengths. Walks the same row shapes as
    ``_row_to_text`` but pulls input-side fields only."""
    out: list[int] = []
    for row in rows:
        text = ""
        for key in ("input", "question", "prompt", "text", "source"):
            val = row.get(key)
            if isinstance(val, str):
                text += val
            elif isinstance(val, dict):
                for sub in val.values():
                    if isinstance(sub, str):
                        text += sub
        if text:
            out.append(len(text))
    return out


def _per_row_output_lengths(rows: list[dict[str, Any]]) -> list[int]:
    """Per-row output character lengths. Walks expected / answer /
    summary / completion-shape fields."""
    out: list[int] = []
    for row in rows:
        expected = row.get("expected")
        if isinstance(expected, dict):
            for key in ("answer", "summary", "response", "output", "completion"):
                val = expected.get(key)
                if isinstance(val, str):
                    out.append(len(val))
                    break
            else:
                continue
            continue
        if isinstance(expected, str):
            out.append(len(expected))
            continue
        for key in ("answer", "summary", "response", "output", "completion"):
            val = row.get(key)
            if isinstance(val, str):
                out.append(len(val))
                break
    return out


def _diversity_score(rows: list[dict[str, Any]]) -> float | None:
    """1.0 - mean_pairwise_jaccard. Higher = more diverse gold set.
    Returns None when fewer than 2 rows have meaningful text."""
    token_sets = [_tokenize(_row_to_text(r)) for r in rows]
    token_sets = [s for s in token_sets if s]
    if len(token_sets) < 2:
        return None
    return 1.0 - _mean_pairwise_jaccard(token_sets)


# ─────────────────────────────────────────────────────────────────────
# Cohort assembly — find passing user projects + load their rows
# ─────────────────────────────────────────────────────────────────────


async def _load_project_rows(
    db: AsyncSession,
    project_id: int,
) -> list[dict[str, Any]]:
    """Load the labeled rows the archetype operates on for one
    project: gold_dev + gold_test + accepted synthetic rows.
    Skips pending-review synth (same gate the training pipeline
    uses). Returns [] when nothing's on disk yet."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                [DatasetType.GOLD_DEV, DatasetType.GOLD_TEST, DatasetType.SYNTHETIC]
            ),
        )
    )
    rows: list[dict[str, Any]] = []
    for dataset in result.scalars():
        if not dataset.file_path:
            continue
        path = Path(dataset.file_path)
        if not path.exists():
            continue
        rows.extend(_load_records_from_file(path))
    return rows


async def _find_passing_projects(
    db: AsyncSession,
    recipe_id: str,
) -> list[tuple[Project, float]]:
    """Find projects whose ``selected_recipe.recipe_id == recipe_id``
    AND whose latest COMPLETED experiment has at least one EvalResult
    with pass_rate >= ``_PASSING_F1_THRESHOLD``. Returns a list of
    ``(project, pass_rate)`` pairs."""
    # Two-step query: find all projects with the recipe, then check
    # their experiment+eval state. Cheaper than a 3-table join with
    # JSON-key filtering for SQLite.
    project_result = await db.execute(select(Project))
    candidates = [
        p
        for p in project_result.scalars()
        if isinstance(p.selected_recipe, dict)
        and p.selected_recipe.get("recipe_id") == recipe_id
    ]
    if not candidates:
        return []

    passing: list[tuple[Project, float]] = []
    for project in candidates:
        exp_result = await db.execute(
            select(Experiment)
            .where(
                Experiment.project_id == project.id,
                Experiment.status == ExperimentStatus.COMPLETED,
            )
            .order_by(Experiment.completed_at.desc().nullslast())
            .limit(1)
        )
        latest_exp = exp_result.scalar_one_or_none()
        if latest_exp is None:
            continue
        eval_result = await db.execute(
            select(EvalResult)
            .where(EvalResult.experiment_id == latest_exp.id)
            .order_by(EvalResult.created_at.desc())
            .limit(1)
        )
        latest_eval = eval_result.scalar_one_or_none()
        if latest_eval is None or latest_eval.pass_rate is None:
            continue
        if float(latest_eval.pass_rate) >= _PASSING_F1_THRESHOLD:
            passing.append((project, float(latest_eval.pass_rate)))
    return passing


# ─────────────────────────────────────────────────────────────────────
# Cache + orchestrator
# ─────────────────────────────────────────────────────────────────────


_CACHE: dict[str, tuple[float, RecipeArchetype]] = {}


def clear_archetype_cache() -> None:
    """Test hook + manual invalidation entry point."""
    _CACHE.clear()


def _cache_get(recipe_id: str) -> RecipeArchetype | None:
    entry = _CACHE.get(recipe_id)
    if entry is None:
        return None
    cached_at, payload = entry
    if (time.monotonic() - cached_at) > _CACHE_TTL_SECONDS:
        _CACHE.pop(recipe_id, None)
        return None
    return payload


def _cache_put(recipe_id: str, payload: RecipeArchetype) -> None:
    _CACHE[recipe_id] = (time.monotonic(), payload)


def _aggregate_feature_values(
    feature_id: str,
    values: list[float],
) -> FeatureDistribution:
    """Compute percentile/mean/min/max from a list of per-project
    feature values. Empty list returns a zero-state distribution
    (caller gates on n_projects=0 for display)."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {
            "feature_id": feature_id,
            "label": _FEATURE_LABELS[feature_id],
            "n_projects": 0,
            "p25": None,
            "p50": None,
            "p75": None,
            "mean": None,
            "min": None,
            "max": None,
            "unit": _FEATURE_UNITS[feature_id],
        }
    quantiles = (
        statistics.quantiles(clean, n=4, method="inclusive")
        if len(clean) >= 2
        else [clean[0], clean[0], clean[0]]
    )
    return {
        "feature_id": feature_id,
        "label": _FEATURE_LABELS[feature_id],
        "n_projects": len(clean),
        "p25": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "mean": float(statistics.mean(clean)),
        "min": float(min(clean)),
        "max": float(max(clean)),
        "unit": _FEATURE_UNITS[feature_id],
    }


async def compute_recipe_archetype(
    db: AsyncSession,
    recipe_id: str,
) -> RecipeArchetype:
    """Build the archetype for ``recipe_id``. Merges user-project
    contributions with template seeds when the user-project cohort
    is thin (< ``_MIN_USER_PROJECTS_BEFORE_SEED_THRESHOLD``).
    Raises ``ValueError("empty_cohort")`` when no contributors
    exist at all (neither user projects nor template seeds for
    this recipe — rare; only for recipes with no shipped template
    AND no passing user project)."""
    cached = _cache_get(recipe_id)
    if cached is not None:
        return cached

    if recipe_id not in {
        rid
        for rids in _FEATURE_APPLICABILITY.values()
        for rid in rids
    }:
        raise ValueError(f"unknown_recipe_id:{recipe_id}")

    # ── 1. Load passing user projects + their rows ────────────────
    passing = await _find_passing_projects(db, recipe_id)
    user_contributions: list[dict[str, float | None]] = []
    cohort: list[CohortMember] = []
    for project, pass_rate in passing:
        rows = await _load_project_rows(db, project.id)
        features = extract_features_from_rows(rows, recipe_id=recipe_id)
        user_contributions.append(features)
        cohort.append({
            "id": int(project.id),
            "name": str(project.name),
            "source": "user",
            "pass_rate": round(pass_rate, 4),
        })

    # ── 2. Merge in template seeds when user cohort is thin ───────
    n_template_seeds = 0
    if len(user_contributions) < _MIN_USER_PROJECTS_BEFORE_SEED_THRESHOLD:
        from app.services.archetype_seeds import load_seed_contributions

        seeds = load_seed_contributions(recipe_id)
        for seed in seeds:
            user_contributions.append(seed["features"])
            cohort.append({
                "id": int(seed["pseudo_id"]),
                "name": str(seed["name"]),
                "source": "template",
                "pass_rate": None,
            })
            n_template_seeds += 1

    if not user_contributions:
        raise ValueError(f"empty_cohort:{recipe_id}")

    # ── 3. Aggregate per-feature percentiles across the cohort ────
    feature_values: dict[str, list[float]] = {
        fid: [] for fid in _applicable_features(recipe_id)
    }
    for contribution in user_contributions:
        for fid, value in contribution.items():
            if value is not None:
                feature_values[fid].append(float(value))

    features: list[FeatureDistribution] = [
        _aggregate_feature_values(fid, vals)
        for fid, vals in sorted(feature_values.items())
    ]

    payload: RecipeArchetype = {
        "recipe_id": recipe_id,
        "n_passing_projects": len(user_contributions),
        "n_user_projects": len(passing),
        "n_template_seeds": n_template_seeds,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "cohort_provenance": cohort,
    }
    _cache_put(recipe_id, payload)
    return payload


# ─────────────────────────────────────────────────────────────────────
# Per-project comparison (USER-SUCCESS Epic 8 Phase 8b)
# ─────────────────────────────────────────────────────────────────────


FeatureStatus = Literal["ok", "below", "above", "missing"]
ComparisonSummary = Literal["healthy", "below_cohort", "above_cohort", "mixed"]


class FeatureComparison(TypedDict):
    """One feature compared between the current project and the
    recipe's archetype distribution. Status drives the UI badge;
    ``suggested_action`` (when present) drives a one-click remediation
    button that matches the existing Coach action shape so frontend
    handlers can reuse them verbatim."""

    feature_id: str
    label: str
    unit: str
    your_value: float | None
    archetype_p25: float | None
    archetype_p50: float | None
    archetype_p75: float | None
    status: FeatureStatus
    suggestion: str | None
    suggested_action: dict | None        # {kind, params} shape used by Coach


class ProjectArchetypeComparison(TypedDict):
    project_id: int
    recipe_id: str
    archetype: RecipeArchetype           # full archetype for context (cohort, n_*, etc.)
    features: list[FeatureComparison]
    summary: ComparisonSummary


def _classify_feature_status(
    *,
    your_value: float | None,
    p25: float | None,
    p75: float | None,
) -> FeatureStatus:
    """Status logic per the Phase 8b spec — below_p25 / above_p75 /
    in-band / missing. Defensive on incomplete archetypes (e.g. a
    1-project cohort might not yield meaningful percentiles)."""
    if your_value is None:
        return "missing"
    if p25 is None or p75 is None:
        # Archetype lacks a percentile band for this feature — can't
        # classify; treat as missing rather than misleading "ok".
        return "missing"
    if your_value < p25:
        return "below"
    if your_value > p75:
        return "above"
    return "ok"


def _suggestion_for(
    *,
    feature_id: str,
    status: FeatureStatus,
    your_value: float | None,
    p50: float | None,
    minority_label: str | None,
) -> tuple[str | None, dict | None]:
    """Map (feature, status) to a human-readable suggestion +
    optional ``suggested_action`` payload. Action shapes match the
    Coach Mode contract so the frontend reuses the existing
    handlers (``run_playbook`` → fires runPlaybookAsync via the
    Jobs framework; ``navigate`` → window.location.assign).

    Returns ``(None, None)`` for ok / missing or for features
    where there's no obvious one-click fix (length features)."""
    if status not in ("below", "above"):
        return (None, None)

    # Row count below cohort → paraphrase more positives. Suggest
    # filling the gap to the cohort median.
    if feature_id == "row_count" and status == "below" and p50 is not None and your_value is not None:
        delta = max(20, int(p50 - your_value))
        delta = min(delta, 200)  # cap so the suggestion isn't a 500-row request
        return (
            f"Your project has {int(your_value)} rows; the cohort median is "
            f"{int(p50)}. Generate {delta} more via the positives-paraphrase "
            f"playbook.",
            {
                "kind": "run_playbook",
                "params": {
                    "mode": "positives_paraphrase",
                    "target_count": delta,
                },
            },
        )

    # Class entropy below → fill the minority class via
    # class_balance_fill. Only emits when we know the minority label.
    if feature_id == "class_entropy" and status == "below":
        if minority_label:
            return (
                f"Class distribution is more skewed than the cohort. Generate "
                f"examples for the minority class ({minority_label}) via the "
                f"class-balance-fill playbook.",
                {
                    "kind": "run_playbook",
                    "params": {
                        "mode": "class_balance_fill",
                        "target_count": 30,
                        "target_class": minority_label,
                    },
                },
            )
        return (
            "Class distribution is more skewed than the cohort. Generate "
            "examples for the minority class via the class-balance-fill "
            "playbook.",
            None,
        )

    # Class balance ratio below → same fix as entropy. The two
    # signals usually fire together, so the Coach card collapses
    # duplicates; here we still emit both for completeness.
    if feature_id == "class_balance_ratio" and status == "below" and minority_label:
        return (
            f"One class dominates more than the cohort. Top up the minority "
            f"class ({minority_label}) via class-balance-fill.",
            {
                "kind": "run_playbook",
                "params": {
                    "mode": "class_balance_fill",
                    "target_count": 30,
                    "target_class": minority_label,
                },
            },
        )

    # Hard-negative ratio below → run the hard-negatives playbook.
    if feature_id == "hard_negative_ratio" and status == "below":
        return (
            "Your hard-negative share is below the cohort. The hard-negatives "
            "playbook generates rows that look like one class but should be "
            "labeled another — high-signal training data.",
            {
                "kind": "run_playbook",
                "params": {
                    "mode": "hard_negatives",
                    "target_count": 30,
                },
            },
        )

    # Diversity below → navigate to Data Studio's diversity tools.
    # No automatic playbook here yet; this is the manual path.
    if feature_id == "goldset_diversity" and status == "below":
        return (
            "Your gold set is less diverse than the cohort — rows likely "
            "repeat similar wording. Open Data Studio's diversity tools to "
            "spot the clusters.",
            {
                "kind": "navigate",
                "params": {"target": "data-studio-diversity"},
            },
        )

    # Length features: diagnostic copy only. No automatic fix —
    # input/output length mismatches usually signal a recipe / data
    # mismatch the user has to investigate themselves.
    if feature_id in ("input_length_chars", "output_length_chars"):
        direction = "shorter" if status == "below" else "longer"
        which = "input" if feature_id == "input_length_chars" else "output"
        return (
            f"Your {which} lengths are {direction} than the cohort. Worth "
            f"reviewing a sample to confirm the recipe / dataset match.",
            None,
        )

    return (None, None)


def _summarise(features: list[FeatureComparison]) -> ComparisonSummary:
    """Aggregate per-feature statuses into a single verdict. We
    ignore 'missing' when counting majorities — they don't reflect
    drift, just unmeasurable features."""
    counted = [f["status"] for f in features if f["status"] != "missing"]
    if not counted:
        return "healthy"  # nothing measurable → don't lecture
    n_below = sum(1 for s in counted if s == "below")
    n_above = sum(1 for s in counted if s == "above")
    n_ok = sum(1 for s in counted if s == "ok")
    if n_below == 0 and n_above == 0:
        return "healthy"
    if n_below > 0 and n_above > 0:
        return "mixed"
    if n_below > n_ok:
        return "below_cohort"
    if n_above > n_ok:
        return "above_cohort"
    return "mixed"


def _minority_label_for(rows: list[dict[str, Any]]) -> str | None:
    """Return the smallest-count class label, or None when no
    classes / single class. Used to populate
    ``class_balance_fill`` actions with the right target_class."""
    labels = _extract_classification_labels(rows)
    if not labels:
        return None
    counter = Counter(labels)
    if len(counter) < 2:
        return None
    minority = min(counter.items(), key=lambda kv: kv[1])
    return minority[0]


async def compare_project_to_archetype(
    db: AsyncSession,
    project_id: int,
) -> ProjectArchetypeComparison:
    """Compute the per-project comparison against the recipe's
    archetype. Raises ``ValueError("project_not_found")`` on
    missing project, ``ValueError("no_recipe_selected")`` when
    the project hasn't picked a recipe yet."""
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError("project_not_found")
    recipe_id = (
        project.selected_recipe.get("recipe_id")
        if isinstance(project.selected_recipe, dict)
        else None
    )
    if not recipe_id:
        raise ValueError("no_recipe_selected")

    # 1. Compute the recipe's archetype (cached when warm).
    archetype = await compute_recipe_archetype(db, recipe_id)

    # 2. Load this project's rows + extract its feature values.
    rows = await _load_project_rows(db, project_id)
    your_values = extract_features_from_rows(rows, recipe_id=recipe_id)
    minority_label = _minority_label_for(rows)

    # 3. Build per-feature comparisons against the archetype's bands.
    archetype_by_id: dict[str, FeatureDistribution] = {
        f["feature_id"]: f for f in archetype["features"]
    }
    comparisons: list[FeatureComparison] = []
    for fid in sorted(_applicable_features(recipe_id)):
        archetype_feature = archetype_by_id.get(fid)
        your_value = your_values.get(fid)
        p25 = archetype_feature["p25"] if archetype_feature else None
        p50 = archetype_feature["p50"] if archetype_feature else None
        p75 = archetype_feature["p75"] if archetype_feature else None
        status = _classify_feature_status(
            your_value=your_value, p25=p25, p75=p75,
        )
        suggestion, action = _suggestion_for(
            feature_id=fid,
            status=status,
            your_value=your_value,
            p50=p50,
            minority_label=minority_label,
        )
        comparisons.append({
            "feature_id": fid,
            "label": _FEATURE_LABELS[fid],
            "unit": _FEATURE_UNITS[fid],
            "your_value": your_value,
            "archetype_p25": p25,
            "archetype_p50": p50,
            "archetype_p75": p75,
            "status": status,
            "suggestion": suggestion,
            "suggested_action": action,
        })

    return {
        "project_id": project_id,
        "recipe_id": recipe_id,
        "archetype": archetype,
        "features": comparisons,
        "summary": _summarise(comparisons),
    }
