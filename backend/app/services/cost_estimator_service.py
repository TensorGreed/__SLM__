"""P18 — Cost estimator with provenance (priority.md, RM3).

Returns ``{gpu_hours, usd, co2_kg, provenance, confidence, ...}`` for a
planned training run. Calibrates against this project's completed
historical runs whenever they exist, and falls back to a deterministic
heuristic when they don't — the response always tells the caller which
path it took via ``provenance: measured | estimated`` so the UI can show
a "measured-vs-estimated" badge on the pre-run confirm modal.

Design:

- **Cohort rules**: a historical ``Experiment`` is comparable if it has
  ``started_at + completed_at`` (so duration is real) AND its
  ``training_mode`` matches the planned mode AND its ``base_model``
  size-class is the same. We never pull cancelled/failed runs into the
  cohort — their duration is not a measurement of training cost.

- **gpu_hours**: median duration of the matching cohort × planned
  ``num_gpus`` (defaults to 1). Heuristic fallback is
  ``base_seconds_per_step × planned_total_steps × gpu_count`` where
  ``base_seconds_per_step`` is per-target-profile.

- **USD**: ``gpu_hours × hourly_rate``. Hourly rate is sourced from the
  cloud-burst pricing catalog when the target profile is a server GPU,
  zero for local CPU / browser targets, and a small heuristic for edge.

- **CO2**: ``gpu_hours × gpu_power_kw × grid_intensity_g_per_kwh / 1000``.
  Power draw is per-target-profile; grid intensity is a fixed global
  average (400 g/kWh) with provenance noted in the response so future
  work can plug in a region-aware factor.

- **Confidence** (0..1): rises with cohort sample count, drops with
  variance, and is biased toward the cohort match tightness.
  ``provenance="measured"`` requires ``confidence >= 0.6``; below that we
  flip to ``estimated`` even if cohort samples exist (small N or wide
  variance is not a measurement, just a hint).

The endpoint ``POST /projects/{pid}/training/plan/estimate-cost`` wraps
this service and is consumed by the Training Planner UI (P20).
"""

from __future__ import annotations

from statistics import median, mean, pstdev
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import Experiment, ExperimentStatus, TrainingMode


# -- Heuristic constants ---------------------------------------------------


# Wall-clock seconds-per-step used by the heuristic fallback. These are
# deliberately conservative — overestimates are better than underestimates
# for a "this is how much it'll cost" pre-run modal. Calibrated against
# what we've seen in test runs and the existing autopilot heuristic.
_TARGET_BASE_SECONDS_PER_STEP: dict[str, float] = {
    "mobile_cpu": 12.0,
    "browser_webgpu": 9.0,
    "edge_gpu": 2.2,
    "vllm_server": 0.6,
}

_DEFAULT_TARGET_PROFILE = "vllm_server"


# Per-mode multiplier on top of the SFT baseline. DPO/ORPO involve a
# reference model + preference pairs, so they're meaningfully slower.
_MODE_TIME_MULTIPLIER: dict[str, float] = {
    "sft": 1.0,
    "domain_pretrain": 1.4,
    "dpo": 1.8,
    "orpo": 1.6,
}


# Power draw at sustained training load (kW). Used for the CO2 path.
_TARGET_POWER_KW: dict[str, float] = {
    "mobile_cpu": 0.0,
    "browser_webgpu": 0.0,
    "edge_gpu": 0.05,
    "vllm_server": 0.30,  # a single A10G/L40S-class server GPU
}


# Global-average grid intensity (gCO2eq per kWh). Region-aware factors
# are out of scope for P18 — leaving the value here so a follow-up can
# plug in a regional lookup.
_GLOBAL_GRID_INTENSITY_G_PER_KWH = 400.0


# Cohort match weight feeds into the confidence score. Same shape as the
# autopilot estimator's cohort weights so the UI can normalize across
# both surfaces. "mode+model_size" is the tightest match the cost
# estimator considers.
_COHORT_CONFIDENCE_WEIGHT: dict[str, float] = {
    "mode+model_size": 1.00,
    "mode": 0.85,
    "model_size": 0.70,
    "global": 0.50,
    "none": 0.30,
}


# Threshold below which a cohort-derived estimate gets demoted to
# ``provenance="estimated"`` — small N or wide variance is not a
# measurement, just a hint.
_MEASURED_CONFIDENCE_THRESHOLD = 0.60


# -- Helpers --------------------------------------------------------------


def _coerce_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _normalize_target_profile(value: Any) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return _DEFAULT_TARGET_PROFILE
    return token


def _normalize_training_mode(value: Any) -> str:
    if isinstance(value, TrainingMode):
        return value.value
    token = str(value or "").strip().lower()
    if not token:
        return "sft"
    return token


def _confidence_band(confidence: float) -> str:
    value = max(0.0, min(float(confidence), 1.0))
    if value >= 0.8:
        return "high"
    if value >= 0.6:
        return "medium"
    return "low"


def _model_size_bucket(base_model: str | None) -> str:
    """Coarse size class derived from the model id text.

    We don't introspect the actual parameter count here — that's a slow
    network call and the registry already does it. The size bucket is
    used only for cohort matching, so a name-based heuristic is fine
    (and is what the existing target_profile_service uses too).
    """
    token = str(base_model or "").strip().lower()
    if not token:
        return "unknown"
    if any(marker in token for marker in ("70b", "65b", "72b", "13b-2x", "8x7b", "8x22b")):
        return "xl"
    if any(marker in token for marker in ("13b", "14b", "20b", "30b", "34b")):
        return "large"
    if any(marker in token for marker in ("7b", "8b", "9b", "11b", "phi-3", "phi-2", "gemma-2-2b", "qwen-2-7b")):
        return "medium"
    if any(marker in token for marker in ("3b", "1b", "1.3b", "0.5b", "1.5b", "tinyllama", "phi-1", "smollm")):
        return "small"
    return "medium"


def _hourly_rate_usd(target_profile: str) -> tuple[float, str]:
    """Resolve hourly USD pricing for a target profile.

    Returns ``(rate_per_hour, source_label)``. Source is stamped into the
    response so the UI can show "cloud-pricing-catalog" vs "fallback".
    """
    target = _normalize_target_profile(target_profile)
    if target in {"mobile_cpu", "browser_webgpu"}:
        return 0.0, "local_runtime"

    if target == "vllm_server":
        try:
            from app.services.cloud_burst_service import list_cloud_burst_catalog

            catalog = dict(list_cloud_burst_catalog() or {})
            gpu_rows = [item for item in list(catalog.get("gpu_skus") or []) if isinstance(item, dict)]
            preferred = next(
                (
                    item
                    for item in gpu_rows
                    if str(item.get("gpu_sku") or "").strip().lower() == "a10g.24gb"
                ),
                gpu_rows[0] if gpu_rows else {},
            )
            hourly_map = dict(preferred.get("hourly_usd") or {})
            rates = [
                float(value)
                for value in hourly_map.values()
                if _coerce_positive_float(value) is not None
            ]
            if rates:
                return round(sum(rates) / float(len(rates)), 4), "cloud_burst_catalog_avg"
        except Exception:
            pass
        return 1.0, "fallback_server_heuristic"

    if target == "edge_gpu":
        return 0.15, "edge_gpu_heuristic"

    return 0.20, "generic_heuristic"


def _planned_total_steps(config: dict[str, Any]) -> int:
    """Derive total training steps from the planned config.

    Falls back to ``num_epochs * 100`` (matches the simulate runtime's
    bookkeeping) when ``total_steps`` isn't explicitly authored.
    """
    explicit = _coerce_positive_int(config.get("total_steps"))
    if explicit is not None:
        return explicit
    epochs = _coerce_positive_int(config.get("num_epochs")) or 3
    steps_per_epoch = _coerce_positive_int(config.get("steps_per_epoch")) or 100
    return epochs * steps_per_epoch


def _planned_gpu_count(config: dict[str, Any]) -> int:
    explicit = _coerce_positive_int(config.get("num_gpus"))
    if explicit is not None:
        return explicit
    return 1


def _historical_duration_seconds(exp: Experiment) -> float | None:
    if exp.started_at is None or exp.completed_at is None:
        return None
    seconds = (exp.completed_at - exp.started_at).total_seconds()
    if seconds <= 0:
        return None
    return float(seconds)


# -- Cohort selection -----------------------------------------------------


def _build_cohort(
    *,
    history: list[Experiment],
    planned_mode: str,
    planned_size_bucket: str,
) -> tuple[str, list[Experiment]]:
    """Pick the tightest cohort that has at least one comparable run.

    Order of preference: ``mode+model_size`` → ``mode`` → ``model_size``
    → ``global``. Returns ``("none", [])`` when nothing useful is in
    history.
    """
    eligible = [exp for exp in history if _historical_duration_seconds(exp) is not None]
    if not eligible:
        return "none", []

    selectors: list[tuple[str, Any]] = [
        (
            "mode+model_size",
            lambda e: _normalize_training_mode(e.training_mode) == planned_mode
            and _model_size_bucket(e.base_model) == planned_size_bucket,
        ),
        (
            "mode",
            lambda e: _normalize_training_mode(e.training_mode) == planned_mode,
        ),
        (
            "model_size",
            lambda e: _model_size_bucket(e.base_model) == planned_size_bucket,
        ),
        (
            "global",
            lambda e: True,
        ),
    ]

    # Prefer cohorts with ≥2 samples for variance + median, but fall back
    # to single-sample cohorts so we still beat pure heuristic when we can.
    for label, selector in selectors:
        subset = [exp for exp in eligible if selector(exp)]
        if len(subset) >= 2:
            return label, subset
    for label, selector in selectors:
        subset = [exp for exp in eligible if selector(exp)]
        if subset:
            return label, subset
    return "none", []


# -- Main entry point ----------------------------------------------------


async def estimate_training_cost(
    db: AsyncSession,
    *,
    project_id: int,
    config: dict[str, Any] | None,
    base_model: str | None = None,
    target_profile_id: str | None = None,
    history_overrides: list[Experiment] | None = None,
) -> dict[str, Any]:
    """Estimate cost for a planned training run.

    Args:
        db: SQLAlchemy session — used to pull this project's completed runs.
        project_id: Restricts the calibration cohort to this project. We
            don't pull cross-project runs because their hardware/cloud
            context can be very different and would skew the estimate.
        config: The planned training config dict — ``num_epochs``,
            ``num_gpus``, ``training_mode``, ``total_steps``,
            ``steps_per_epoch``, etc.
        base_model: Planned base model id (used for size-bucket cohort
            matching). Falls back to ``config["base_model"]``.
        target_profile_id: Planned deployment target — picks the hourly
            rate and power draw. Defaults to ``vllm_server``.
        history_overrides: Optional injection point for tests that don't
            want to depend on a populated DB.
    """
    cfg = dict(config or {})
    planned_mode = _normalize_training_mode(cfg.get("training_mode") or "sft")
    planned_target = _normalize_target_profile(target_profile_id or cfg.get("target_profile_id"))
    resolved_base_model = str(base_model or cfg.get("base_model") or "").strip()
    planned_size_bucket = _model_size_bucket(resolved_base_model or None)
    planned_total_steps = _planned_total_steps(cfg)
    planned_gpu_count = _planned_gpu_count(cfg)

    # 1) Pull historical runs for this project. We only consider rows in
    # COMPLETED state with both timestamps present — anything else is not
    # a valid measurement of cost.
    if history_overrides is not None:
        history: list[Experiment] = list(history_overrides)
    else:
        history = list(
            (
                await db.execute(
                    select(Experiment)
                    .where(
                        Experiment.project_id == project_id,
                        Experiment.status == ExperimentStatus.COMPLETED,
                    )
                    .order_by(Experiment.completed_at.desc())
                    .limit(50)
                )
            )
            .scalars()
            .all()
        )

    cohort_label, cohort = _build_cohort(
        history=history,
        planned_mode=planned_mode,
        planned_size_bucket=planned_size_bucket,
    )
    cohort_durations: list[float] = []
    for exp in cohort:
        duration = _historical_duration_seconds(exp)
        if duration is not None:
            cohort_durations.append(duration)
    sample_count = len(cohort_durations)

    # 2) gpu_hours estimate
    heuristic_seconds_per_step = float(
        _TARGET_BASE_SECONDS_PER_STEP.get(planned_target, 1.0)
    )
    mode_multiplier = float(_MODE_TIME_MULTIPLIER.get(planned_mode, 1.0))
    heuristic_seconds = (
        heuristic_seconds_per_step * planned_total_steps * mode_multiplier * planned_gpu_count
    )
    heuristic_seconds = max(60.0, heuristic_seconds)

    metric_source = "estimated"
    estimated_seconds = heuristic_seconds
    median_cohort_seconds: float | None = None
    variability_cv: float | None = None

    if sample_count >= 2:
        median_cohort_seconds = float(median(cohort_durations))
        # Cohort runs already include their own gpu_count somewhere in the
        # mix; treat the median directly as the run wall-clock and scale
        # by the *ratio* of planned vs assumed-1 gpu count below if we
        # had richer telemetry. For now: trust the median as-is.
        estimated_seconds = float(median_cohort_seconds * planned_gpu_count)
        baseline = mean(cohort_durations)
        if baseline > 0:
            variability_cv = float(pstdev(cohort_durations) / baseline)
        metric_source = "measured"
    elif sample_count == 1:
        # Single-sample blend: 60% telemetry, 40% heuristic. Same blend
        # the autopilot estimator uses; keeps things from swinging too
        # hard on one data point.
        median_cohort_seconds = float(cohort_durations[0])
        telemetry_seconds = float(median_cohort_seconds * planned_gpu_count)
        estimated_seconds = (0.6 * telemetry_seconds) + (0.4 * heuristic_seconds)
        # Variability unknown with N=1 — leave None.

    estimated_seconds = max(60.0, estimated_seconds)
    gpu_hours = round(estimated_seconds / 3600.0, 4)

    # 3) Confidence
    cohort_weight = float(_COHORT_CONFIDENCE_WEIGHT.get(cohort_label, _COHORT_CONFIDENCE_WEIGHT["none"]))
    if sample_count >= 2:
        sample_score = min(1.0, sample_count / 8.0)
        variance_score = (
            0.6 if variability_cv is None else max(0.1, min(1.0, 1.0 - variability_cv))
        )
        confidence = 0.25 + (0.40 * sample_score) + (0.25 * cohort_weight) + (0.10 * variance_score)
    elif sample_count == 1:
        confidence = 0.45 + (0.10 * cohort_weight)
    else:
        confidence = 0.30
    confidence = round(max(0.05, min(confidence, 0.99)), 4)

    # If the cohort gave us a number but confidence is too low to call it
    # a measurement, demote provenance back to "estimated". The number
    # still incorporates the cohort, just doesn't get the badge.
    if metric_source == "measured" and confidence < _MEASURED_CONFIDENCE_THRESHOLD:
        metric_source = "estimated"

    # 4) USD
    hourly_rate_usd, pricing_source = _hourly_rate_usd(planned_target)
    usd = round(gpu_hours * hourly_rate_usd, 4) if hourly_rate_usd > 0 else 0.0

    # 5) CO2 (kg)
    power_kw = float(_TARGET_POWER_KW.get(planned_target, 0.30))
    co2_kg = round(
        (gpu_hours * power_kw * _GLOBAL_GRID_INTENSITY_G_PER_KWH) / 1000.0, 6
    )

    return {
        "gpu_hours": gpu_hours,
        "usd": usd,
        "co2_kg": co2_kg,
        "provenance": metric_source,
        "confidence": confidence,
        "confidence_band": _confidence_band(confidence),
        "calibration": {
            "cohort": cohort_label,
            "sample_count": sample_count,
            "median_cohort_seconds": (
                round(median_cohort_seconds, 2) if median_cohort_seconds is not None else None
            ),
            "variability_cv": round(variability_cv, 4) if variability_cv is not None else None,
            "heuristic_seconds": round(heuristic_seconds, 2),
            "fallback_used": metric_source != "measured",
            "planned_total_steps": planned_total_steps,
            "planned_gpu_count": planned_gpu_count,
            "planned_size_bucket": planned_size_bucket,
            "training_mode": planned_mode,
            "target_profile_id": planned_target,
        },
        "pricing": {
            "hourly_rate_usd": round(hourly_rate_usd, 4),
            "source": pricing_source,
        },
        "co2": {
            "power_kw": power_kw,
            "grid_intensity_g_per_kwh": _GLOBAL_GRID_INTENSITY_G_PER_KWH,
            "intensity_source": "global_average",
        },
    }
