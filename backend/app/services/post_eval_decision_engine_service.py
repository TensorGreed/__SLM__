"""Post-eval decision engine (USER-SUCCESS Epic 7 Phase 7a).

After an evaluation lands, this service inspects the result, the
project's brief, and the gold-set's shape, then categorically
recommends one of:

  * ``stay_the_course`` — pass_rate is healthy; no reroute needed
    (panel self-hides upstream)
  * ``try_rag`` — task looks retrieval-shaped; user should consider
    swapping to a RAG-first approach (Phase 7b ships the clone
    primitive, Phase 7d wires the one-click action)
  * ``try_prompt_engineering`` — output is a tiny pluck out of a
    much larger input; base-model prompting may suffice
  * ``expand_data`` — catch-all when no specific reroute signal
    fires; "more / better data" is the residual lever

Three signal checks compose the recommendation:

  * ``brief_mentions_retrieval`` — case-insensitive substring scan
    of ``Project.description`` against a curated list of phrases
    that read like "this is a knowledge-retrieval task"
  * ``goldset_answer_diversity_high`` — mean pairwise token-set
    Jaccard across the gold-set's *output* strings is below a
    threshold (each answer is mostly unique → retrieval beats
    memorization)
  * ``input_output_density_low`` — mean(len(output)/len(input))
    across gold rows is below a threshold (the answer is a small
    pluck of the input → may not need fine-tuning at all)

Recipe-agnostic by design (per the ``keep-brewslm-general`` rule):
no recipe-specific code paths. Recipe info is used only to suppress
the ``try_rag`` recommendation when the project is already a
``qa-sft`` project with ``auto_rag.enabled=true`` on its latest
experiment (no recommending what's already on).

Caching: the analysis is persisted onto the source ``EvalResult``'s
``details["reroute_analysis"]`` JSON column. This invalidates
naturally when the eval row is rewritten, mirrors how
``evaluation_service`` itself reassigns ``result.details`` to
trigger SQLAlchemy dirty-tracking, and avoids a separate cache
table. ``?refresh=true`` from the API bypasses the cache and
overwrites it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict


# Curated list of phrases that strongly suggest "this is a
# knowledge-retrieval task." Each entry is matched as a
# case-insensitive substring of ``Project.description``. Kept
# specific enough to avoid false positives on generic project
# briefs (no bare "documentation" or "policy" — those land in too
# many non-RAG contexts). Add new entries when the user-reported
# false-negative rate is meaningful.
_RETRIEVAL_KEYWORDS: tuple[str, ...] = (
    "answer questions about",
    "answer questions based on",
    "answers questions about",
    "answers questions based on",
    "answering questions about",
    "look up",
    "looks up",
    "looking up",
    "look in",
    "find in the",
    "find the answer in",
    "based on the documentation",
    "based on the docs",
    "based on the knowledge",
    "based on the policy",
    "according to the documentation",
    "according to the policy",
    "according to the manual",
    "knowledge base",
    "from these documents",
    "from the policy",
    "from the docs",
    "from the manual",
    "support faq",
    "support chatbot",
    "policy q",
    "policy qa",
    "internal docs",
    "retrieve from",
    "qa on documents",
)


# Below this mean pairwise Jaccard the answers in the gold set look
# distinct enough to make retrieval valuable. 0.20 chosen by eye on
# the existing template gold sets — policy-qa-style and the demo
# Support FAQ land well under (each row's answer is mostly unique);
# ticket-router (classification) lands well over (rows share the
# same labels).
_DIVERSITY_THRESHOLD: float = 0.20

# Below this output/input length ratio the task looks like
# "extract a small answer from a big context" — fine-tuning is
# overkill, a base model with prompting often suffices.
_DENSITY_THRESHOLD: float = 0.05

# Below this pass_rate the eval is treated as "the model is
# struggling, surface a reroute." At or above, we recommend
# stay_the_course and the UI panel self-hides.
_PASS_RATE_THRESHOLD: float = 0.5

# Bounded compute for the diversity pairwise scan. At n=200 the
# full pair count is 19,900; at n=500 it's 124,750. Cap the sample
# size so the analyzer is fast on any project size.
_DIVERSITY_MAX_ROWS: int = 200

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


# ─────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────


RerouteRecommendationKind = Literal[
    "try_rag",
    "try_prompt_engineering",
    "expand_data",
    "stay_the_course",
]


class RerouteSignal(TypedDict):
    id: str
    fired: bool
    detail: str
    evidence: dict[str, Any]


class RerouteRecommendation(TypedDict):
    kind: RerouteRecommendationKind
    confidence: float
    rationale: str


class RerouteAnalysis(TypedDict):
    eval_result_id: int
    project_id: int
    pass_rate: float | None
    signals: list[RerouteSignal]
    recommendation: RerouteRecommendation
    computed_at: str


# ─────────────────────────────────────────────────────────────────────
# Row-shape walkers (mirrors trainability_forecast_service patterns)
# ─────────────────────────────────────────────────────────────────────


def _extract_input_text(row: dict[str, Any]) -> str:
    """Pull the *input* string from a gold row across shapes."""
    parts: list[str] = []
    for key in ("input", "question", "prompt", "text", "source"):
        value = row.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for sub in value.values():
                if isinstance(sub, str):
                    parts.append(sub)
    return " ".join(p for p in parts if p)


def _extract_output_text(row: dict[str, Any]) -> str:
    """Pull the *output* string from a gold row across shapes."""
    expected = row.get("expected")
    if isinstance(expected, dict):
        for k in ("answer", "label", "summary", "response", "output", "completion"):
            v = expected.get(k)
            if isinstance(v, str):
                return v
        # Last resort: stringify the expected dict
        return " ".join(str(v) for v in expected.values() if isinstance(v, str))
    if isinstance(expected, str):
        return expected
    for key in ("answer", "label", "summary", "response", "output", "completion"):
        v = row.get(key)
        if isinstance(v, str):
            return v
    return ""


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(t.lower() for t in _TOKEN_RE.findall(text or ""))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _mean_pairwise_jaccard(token_sets: list[frozenset[str]]) -> float | None:
    """Mean pairwise Jaccard. Returns None when fewer than 2 non-empty
    token sets are available (the metric is undefined)."""
    sets = [s for s in token_sets if s]
    n = len(sets)
    if n < 2:
        return None
    if n > _DIVERSITY_MAX_ROWS:
        sets = sets[:_DIVERSITY_MAX_ROWS]
        n = _DIVERSITY_MAX_ROWS
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _jaccard(sets[i], sets[j])
            pairs += 1
    if pairs == 0:
        return None
    return total / pairs


# ─────────────────────────────────────────────────────────────────────
# Signal computation
# ─────────────────────────────────────────────────────────────────────


def _matching_retrieval_keywords(description: str) -> list[str]:
    if not description:
        return []
    lowered = description.lower()
    return [kw for kw in _RETRIEVAL_KEYWORDS if kw in lowered]


def _signal_brief_mentions_retrieval(description: str) -> RerouteSignal:
    matched = _matching_retrieval_keywords(description)
    fired = bool(matched)
    return {
        "id": "brief_mentions_retrieval",
        "fired": fired,
        "detail": (
            f"Your brief mentions retrieval-style language ({', '.join(matched[:3])!r})"
            if fired
            else "Your brief doesn't read like a knowledge-retrieval task."
        ),
        "evidence": {"matched_keywords": matched},
    }


def _signal_goldset_answer_diversity_high(
    gold_rows: list[dict[str, Any]],
) -> RerouteSignal:
    token_sets = [_tokenize(_extract_output_text(r)) for r in gold_rows]
    mean = _mean_pairwise_jaccard(token_sets)
    n_rows = sum(1 for s in token_sets if s)
    fired = mean is not None and mean < _DIVERSITY_THRESHOLD
    if mean is None:
        detail = "Not enough gold-set output text to assess diversity."
    elif fired:
        detail = (
            f"Gold-set answers are highly diverse "
            f"(mean pairwise Jaccard {mean:.2f} < {_DIVERSITY_THRESHOLD}) "
            f"— retrieval is likely to beat memorization."
        )
    else:
        detail = (
            f"Gold-set answers overlap a lot "
            f"(mean pairwise Jaccard {mean:.2f}); SFT memorization is viable."
        )
    return {
        "id": "goldset_answer_diversity_high",
        "fired": fired,
        "detail": detail,
        "evidence": {
            "mean_pairwise_jaccard": mean,
            "n_rows": n_rows,
            "threshold": _DIVERSITY_THRESHOLD,
        },
    }


def _signal_input_output_density_low(
    gold_rows: list[dict[str, Any]],
) -> RerouteSignal:
    ratios: list[float] = []
    for row in gold_rows:
        i = _extract_input_text(row)
        o = _extract_output_text(row)
        if i and len(i) > 0:
            ratios.append(len(o) / len(i))
    mean = sum(ratios) / len(ratios) if ratios else None
    fired = mean is not None and mean < _DENSITY_THRESHOLD
    if mean is None:
        detail = "Not enough gold-set rows with both input and output text."
    elif fired:
        detail = (
            f"Output is a tiny slice of input "
            f"(mean ratio {mean:.3f} < {_DENSITY_THRESHOLD}) "
            f"— this looks like extraction; prompting may suffice."
        )
    else:
        detail = (
            f"Output is a substantial slice of input "
            f"(mean ratio {mean:.3f}); fine-tuning is on-task."
        )
    return {
        "id": "input_output_density_low",
        "fired": fired,
        "detail": detail,
        "evidence": {
            "mean_density": mean,
            "n_rows": len(ratios),
            "threshold": _DENSITY_THRESHOLD,
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Recommendation logic
# ─────────────────────────────────────────────────────────────────────


def _classify_recommendation(
    *,
    signals: list[RerouteSignal],
    pass_rate: float | None,
    recipe_id: str | None,
    auto_rag_enabled: bool,
) -> RerouteRecommendation:
    """Deterministic priority — no scoring fuss.

    1. pass_rate >= 0.5 → stay_the_course (panel self-hides upstream)
    2. Any RAG-shaped signal fires AND project isn't already qa-sft+RAG → try_rag
    3. input_output_density_low fires → try_prompt_engineering
    4. Catch-all → expand_data
    """
    fired = {s["id"] for s in signals if s["fired"]}

    if pass_rate is not None and pass_rate >= _PASS_RATE_THRESHOLD:
        return {
            "kind": "stay_the_course",
            "confidence": 1.0,
            "rationale": (
                f"Pass rate {pass_rate:.2f} ≥ {_PASS_RATE_THRESHOLD}; no reroute needed."
            ),
        }

    rag_signals = {"brief_mentions_retrieval", "goldset_answer_diversity_high"}
    rag_signal_fired = bool(rag_signals & fired)
    already_rag = (recipe_id == "qa-sft" and auto_rag_enabled)

    if rag_signal_fired and not already_rag:
        # Confidence climbs when both RAG signals fire; floors at 0.5
        # when only one fires.
        both = len(rag_signals & fired) == 2
        confidence = 0.85 if both else 0.6
        which = ", ".join(sorted(rag_signals & fired))
        return {
            "kind": "try_rag",
            "confidence": confidence,
            "rationale": (
                f"Low pass rate ({pass_rate}) + retrieval-shaped signals fired ({which}). "
                f"A RAG-first project would retrieve from your gold set at inference "
                f"instead of relying on what the model memorized during fine-tuning."
            ),
        }

    if "input_output_density_low" in fired:
        return {
            "kind": "try_prompt_engineering",
            "confidence": 0.55,
            "rationale": (
                f"Output is a tiny slice of input. Try iterating on the base "
                f"model in the Playground with a careful prompt before committing "
                f"to another training run."
            ),
        }

    return {
        "kind": "expand_data",
        "confidence": 0.4,
        "rationale": (
            f"No specific approach-mismatch signal fired. Pass rate "
            f"({pass_rate}) suggests the model needs more or higher-quality "
            f"training data — open the Active Learning panel."
        ),
    }


# ─────────────────────────────────────────────────────────────────────
# Gold-row loader
# ─────────────────────────────────────────────────────────────────────


async def _load_gold_rows(db, project_id: int) -> list[dict[str, Any]]:
    """Load the rows used to compute diversity + density signals.

    Reuses ``dataset_service._load_records_from_file`` so pending
    synth rows are excluded — same loader the training pipeline uses
    so the analyzer reasons about what training actually saw.
    """
    from sqlalchemy import select

    from app.config import settings
    from app.models.dataset import Dataset, DatasetType
    from app.services.dataset_service import _load_records_from_file

    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                [
                    DatasetType.GOLD_DEV,
                    DatasetType.GOLD_TEST,
                ]
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
    if not rows:
        # Fall back to the prepared/train split for projects where
        # the Dataset rows haven't been registered yet (older seeds
        # / partial flows) — same fallback auto_rag_service uses.
        prepared = (
            settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "train.jsonl"
        )
        if prepared.exists():
            rows.extend(_load_records_from_file(prepared))
    return rows


# ─────────────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────────────


def _read_auto_rag_enabled(experiment_config: Any) -> bool:
    """Mirrors the legacy-bool tolerance in coach_service: accepts
    both ``{"auto_rag": {"enabled": True}}`` and the legacy
    ``{"auto_rag": True}`` shape."""
    if not isinstance(experiment_config, dict):
        return False
    ar = experiment_config.get("auto_rag")
    if isinstance(ar, dict):
        return bool(ar.get("enabled"))
    if isinstance(ar, bool):
        return ar
    return False


async def analyze_eval_for_reroute(
    db,
    eval_result_id: int,
    *,
    use_cache: bool = True,
) -> RerouteAnalysis:
    """Compute the post-eval reroute analysis for an EvalResult.

    Raises ``ValueError("eval_result_not_found")`` when the
    eval_result row doesn't exist. Raises
    ``ValueError("project_not_found")`` when the experiment links
    to a missing project (should be unreachable under FK
    constraints, kept for defensiveness).
    """
    from app.models.experiment import EvalResult, Experiment
    from app.models.project import Project

    eval_result = await db.get(EvalResult, eval_result_id)
    if eval_result is None:
        raise ValueError("eval_result_not_found")

    if use_cache:
        cached = (eval_result.details or {}).get("reroute_analysis")
        if isinstance(cached, dict) and cached.get("eval_result_id") == eval_result_id:
            return cached  # type: ignore[return-value]

    experiment = await db.get(Experiment, eval_result.experiment_id)
    if experiment is None:
        raise ValueError("experiment_not_found")
    project = await db.get(Project, experiment.project_id)
    if project is None:
        raise ValueError("project_not_found")

    description = project.description or ""
    gold_rows = await _load_gold_rows(db, project.id)

    signals: list[RerouteSignal] = [
        _signal_brief_mentions_retrieval(description),
        _signal_goldset_answer_diversity_high(gold_rows),
        _signal_input_output_density_low(gold_rows),
    ]

    selected_recipe = project.selected_recipe or {}
    recipe_id = selected_recipe.get("recipe_id") if isinstance(selected_recipe, dict) else None
    auto_rag_enabled = _read_auto_rag_enabled(experiment.config)

    recommendation = _classify_recommendation(
        signals=signals,
        pass_rate=eval_result.pass_rate,
        recipe_id=recipe_id,
        auto_rag_enabled=auto_rag_enabled,
    )

    analysis: RerouteAnalysis = {
        "eval_result_id": eval_result_id,
        "project_id": project.id,
        "pass_rate": eval_result.pass_rate,
        "signals": signals,
        "recommendation": recommendation,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Cache on eval_result.details — full reassignment so SQLAlchemy
    # JSON dirty-tracking fires (the project-wide convention).
    details = dict(eval_result.details or {})
    details["reroute_analysis"] = analysis
    eval_result.details = details
    await db.flush()

    return analysis
