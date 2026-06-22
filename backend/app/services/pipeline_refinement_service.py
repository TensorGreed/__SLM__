"""Pipeline plan refinement — Phase 1 (deterministic, no cloud LLM).

The post-data pass: once a project has data, assess how well its current
pipeline *plan* (recipe / task shape / base model / target) fits the *measured*
data, and assemble a privacy-safe aggregate profile that a future cloud-LLM
strategy pass (Phase 2) can reason over.

PRIVACY BOUNDARY — load-bearing
================================
``cloud_safe_profile`` is the **only** structure a Phase-2 cloud-LLM call may
ever send off-box. It contains exclusively aggregate signals — counts, rates,
distribution *shape*, severities, and config strings the user themselves chose
(recipe id, base model, target profile). It must NEVER contain ingested rows,
document text, gold answers, or even label names.

It's built by *whitelisting scalar fields*, never by copying a signal's nested
``context`` (which can carry raw row excerpts — e.g. the leakage signals stash
matched rows under ``context.examples``). ``test_pipeline_refinement`` asserts
no row/text/label-name leaks into ``cloud_safe_profile``.

Provider note (Phase 2): the cloud call will ride the existing
``cloud_llm_service`` OpenAI-compatible path — which already covers DeepSeek —
plus Qwen via a DashScope-compatible base URL or a local Ollama model. Phase 1
makes no call; ``cloud_refinement.available`` is always False here.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project


# Severity → plan-health bucket. Gap/forecast signals use ok / warn / block.
def _bucket(severity: str | None) -> str:
    s = str(severity or "").lower()
    if s == "block":
        return "mismatch"
    if s in ("warn", "attention"):
        return "attention"
    return "ready"


def _label_distribution_shape(counts: dict[str, int]) -> dict[str, Any] | None:
    """Distribution *shape* metrics from a {label: count} map — counts only,
    NO label names (the names are user-chosen vocabulary we keep off-box)."""
    from app.services.trainability_forecast_service import PER_CLASS_MINIMUM

    if not counts:
        return None
    values = list(counts.values())
    min_count = min(values)
    max_count = max(values)
    return {
        "num_classes": len(counts),
        "min_class_count": int(min_count),
        "max_class_count": int(max_count),
        # 1.0 = perfectly balanced, → 0 as the rarest class shrinks vs the biggest.
        "imbalance_ratio": round(min_count / max_count, 4) if max_count else 0.0,
        "classes_below_floor": sum(1 for v in values if v < PER_CLASS_MINIMUM),
    }


def _gap_severity(gaps_report: dict[str, Any], signal_id: str) -> str | None:
    """Pull a single training-config-gap signal's severity by id (aggregate)."""
    for group in gaps_report.get("groups") or []:
        for signal in group.get("signals") or []:
            if signal.get("id") == signal_id:
                return str(signal.get("severity") or "") or None
    return None


def _archetype_below_band_features(comparison: dict[str, Any]) -> list[str]:
    """Feature *names* that sit below the archetype's expected band (e.g.
    ``answer_length``). Feature names describe the task shape, not user data."""
    out: list[str] = []
    features = comparison.get("features") if isinstance(comparison, dict) else None
    for feat in features or []:
        if not isinstance(feat, dict):
            continue
        status = str(feat.get("status") or feat.get("band") or "").lower()
        below = bool(feat.get("below_band")) or status in ("below", "low", "under")
        if below:
            name = feat.get("name") or feat.get("feature") or feat.get("id")
            if name:
                out.append(str(name))
    return out


async def build_cloud_safe_profile(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """Aggregate, off-box-safe data + plan profile. See the module PRIVACY
    BOUNDARY note — only whitelisted scalars/shapes, never raw content."""
    from app.services.trainability_forecast_service import (
        _estimate_train_row_count,
        _label_counts,
        _load_gold_rows,
        forecast_training,
    )
    from app.services.training_config_gap_service import scan_training_config_gaps

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    selected_recipe = project.selected_recipe or {}
    recipe_id = selected_recipe.get("recipe_id")
    task_profile = selected_recipe.get("task_profile") or selected_recipe.get("task_family")

    gold_rows = await _load_gold_rows(db, project_id)
    label_shape = _label_distribution_shape(_label_counts(gold_rows))
    labelled_row_count = await _estimate_train_row_count(db, project_id)

    # Aggregate severities — best-effort; a producer failure must not sink the
    # whole profile (it just narrows it).
    truncation_risk: str | None = None
    tokenizer_oov: str | None = None
    try:
        gaps = await scan_training_config_gaps(db, project_id)
        truncation_risk = _gap_severity(gaps, "training_config.max_seq_truncation_risk")
        tokenizer_oov = _gap_severity(gaps, "training_config.tokenizer_oov_high")
    except Exception:  # noqa: BLE001 — narrow the profile, don't fail it
        pass

    forecast_verdict: str | None = None
    try:
        forecast = await forecast_training(db, project_id)
        forecast_verdict = str(forecast.get("overall") or "") or None
    except Exception:  # noqa: BLE001
        pass

    archetype_features: list[str] = []
    try:
        from app.services.archetype_service import compare_project_to_archetype

        comparison = await compare_project_to_archetype(db, project_id)
        archetype_features = _archetype_below_band_features(comparison)
    except Exception:  # noqa: BLE001
        pass

    # WHITELIST — every field here is an aggregate/scalar/enum or user-chosen
    # config. No raw rows, document text, gold answers, or label names.
    return {
        "recipe_id": recipe_id,
        "task_profile": task_profile,
        "base_model_name": project.base_model_name or None,
        "target_profile_id": project.target_profile_id or None,
        "labelled_row_count": int(labelled_row_count),
        "label_distribution_shape": label_shape,
        "truncation_risk": truncation_risk,
        "tokenizer_oov": tokenizer_oov,
        "archetype_below_band_features": archetype_features,
        "forecast_verdict": forecast_verdict,
    }


def _signal(sid: str, severity: str, headline: str, *, target_tab: str | None = None) -> dict[str, Any]:
    return {"id": sid, "severity": severity, "headline": headline, "target_tab": target_tab}


def assess_plan_health(profile: dict[str, Any], *, recipe_min_rows: int | None) -> dict[str, Any]:
    """Deterministic plan-fit lens: does the *current plan* suit the *measured
    data*? Distinct from the data-health / config-gap panels — this cross-checks
    plan ↔ data. Verdict = worst signal (mismatch > attention > ready)."""
    signals: list[dict[str, Any]] = []

    if not profile.get("recipe_id"):
        signals.append(_signal(
            "plan.no_recipe", "block",
            "No recipe selected — pick a task shape before refining the plan.",
            target_tab="data",
        ))

    verdict = str(profile.get("forecast_verdict") or "")
    if verdict == "likely_fail":
        signals.append(_signal(
            "plan.forecast_likely_fail", "block",
            "Trainability forecast says this run would likely fail on the current data.",
            target_tab="training-config",
        ))
    elif verdict == "borderline":
        signals.append(_signal(
            "plan.forecast_borderline", "warn",
            "Trainability forecast is borderline — the plan may underdeliver without more/cleaner data.",
            target_tab="training-config",
        ))

    rows = int(profile.get("labelled_row_count") or 0)
    if recipe_min_rows and rows < int(recipe_min_rows):
        signals.append(_signal(
            "plan.rows_below_recommended", "warn",
            f"{rows} labelled rows is below the ~{int(recipe_min_rows)} recommended for this task shape.",
            target_tab="goldset",
        ))

    shape = profile.get("label_distribution_shape") or {}
    below_floor = int(shape.get("classes_below_floor") or 0)
    if below_floor > 0:
        signals.append(_signal(
            "plan.classes_below_floor", "warn",
            f"{below_floor} class(es) sit below the per-class floor — the model will underfit them.",
            target_tab="synthetic",
        ))

    for sid, label in (("truncation_risk", "max_seq_length truncation"),
                       ("tokenizer_oov", "tokenizer OOV")):
        bucket = _bucket(profile.get(sid))
        if bucket == "mismatch":
            signals.append(_signal(
                f"plan.{sid}", "block",
                f"High {label} risk for this base model — the plan will train on corrupted inputs.",
                target_tab="training-config",
            ))
        elif bucket == "attention":
            signals.append(_signal(
                f"plan.{sid}", "warn",
                f"Elevated {label} risk — worth resolving before training.",
                target_tab="training-config",
            ))

    if any(s["severity"] == "block" for s in signals):
        overall = "mismatch"
    elif any(s["severity"] == "warn" for s in signals):
        overall = "attention"
    else:
        overall = "ready"
    return {"verdict": overall, "signals": signals}


async def refine_pipeline_plan(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """Phase 1 — deterministic plan-refinement report: current plan + the
    privacy-safe aggregate profile + a plan-fit roll-up. No cloud call."""
    from app.services.recipe_service import get_recipe

    profile = await build_cloud_safe_profile(db, project_id)

    recipe_min_rows: int | None = None
    recipe_id = profile.get("recipe_id")
    if recipe_id:
        try:
            recipe = get_recipe(str(recipe_id))
            if recipe is not None:
                recipe_min_rows = int(recipe.gold_template.min_rows_recommended)
        except Exception:  # noqa: BLE001
            recipe_min_rows = None

    plan_health = assess_plan_health(profile, recipe_min_rows=recipe_min_rows)

    return {
        "project_id": int(project_id),
        "plan": {
            "recipe_id": profile.get("recipe_id"),
            "task_profile": profile.get("task_profile"),
            "base_model_name": profile.get("base_model_name"),
            "target_profile_id": profile.get("target_profile_id"),
        },
        "cloud_safe_profile": profile,
        "plan_health": plan_health,
        "privacy": {
            "cloud_sharing": "aggregate_only",
            "note": (
                "Only the aggregate signals in cloud_safe_profile are ever eligible "
                "to be sent to a cloud model (Phase 2). Your ingested rows, document "
                "text, gold answers, and label names never leave BrewSLM."
            ),
        },
        "cloud_refinement": {
            # Phase 1 is deterministic-only; Phase 2 wires the cloud strategy pass.
            "available": False,
            "supported_providers": ["anthropic", "openai", "deepseek", "qwen", "ollama"],
            "reason": "phase_1_deterministic_only",
        },
    }
