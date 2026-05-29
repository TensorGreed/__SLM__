"""SLM-vs-frontier benchmark report (Track 1, Epic D).

Answers the production buying question — "is my cheap local SLM good enough vs.
just calling gpt-4o-mini?" — as an honest in-product report:

    "BrewSLM model is X% as good as gpt-4o-mini at Y% of the cost and Z× the latency."

**Quality** mirrors `student_teacher_comparison_service`: a pure ratio over stored
EvalResult rows (no new model calls). It needs a *frontier baseline run* — the
frontier model evaluated on the same eval set — resolved like the teacher
baseline (explicit arg → experiment `config.frontier_baseline_run_id` → eval
pack). Without one, quality is a soft-fallback `no_frontier_eval` (we never
fabricate "X% as good"); the cost/latency comparison still renders.

**Cost / latency** are honest-by-provenance, never fabricated:
  - frontier numbers come from a small published-reference table (public pricing
    + typical latency, with an ``as_of`` stamp);
  - SLM numbers come from the project's latest model-benchmark sweep
    (``estimated``) — throughput → $/1M tokens at the GPU's hourly rate, latency
    from the same sweep. When no sweep exists, the SLM side is marked
    ``unavailable`` with a CTA rather than guessed.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import Experiment
from app.models.project import Project
from app.services.sft_lift_summary_service import (
    _PREFERRED_HEADLINE_METRICS,
    _latest_eval_result,
    _normalize_metrics,
)

# ── Frontier reference table (PUBLISHED public numbers — not fabricated) ─────
# Pricing is per 1M tokens; latency is a typical short-completion total. Update
# the ``as_of`` stamp when refreshing. These are reference points the report
# labels as "published reference", not measured calls.
FRONTIER_REFERENCE: dict[str, dict[str, Any]] = {
    "gpt-4o-mini": {
        "display_name": "GPT-4o mini",
        "usd_per_1m_input_tokens": 0.15,
        "usd_per_1m_output_tokens": 0.60,
        "typical_latency_ms": 700.0,
        "source": "OpenAI published API pricing",
        "as_of": "2026-05",
    },
    "gpt-4o": {
        "display_name": "GPT-4o",
        "usd_per_1m_input_tokens": 2.50,
        "usd_per_1m_output_tokens": 10.00,
        "typical_latency_ms": 900.0,
        "source": "OpenAI published API pricing",
        "as_of": "2026-05",
    },
    "claude-3-5-haiku": {
        "display_name": "Claude 3.5 Haiku",
        "usd_per_1m_input_tokens": 0.80,
        "usd_per_1m_output_tokens": 4.00,
        "typical_latency_ms": 800.0,
        "source": "Anthropic published API pricing",
        "as_of": "2026-05",
    },
}
DEFAULT_FRONTIER_MODEL_ID = "gpt-4o-mini"


def _blended_usd_per_1m(ref: dict[str, Any]) -> float:
    """50/50 input/output blend — a single comparable $/1M-token figure."""
    return round(
        (float(ref["usd_per_1m_input_tokens"]) + float(ref["usd_per_1m_output_tokens"])) / 2.0,
        4,
    )


def _quality_pct(slm: float, frontier: float) -> float | None:
    """slm / frontier as a ratio (UI renders ×100 as a %). None when frontier==0
    and slm>0 (undefined — the SLM "exceeds" a zero baseline)."""
    if frontier == 0.0:
        return 1.0 if slm == 0.0 else None
    return round(slm / frontier, 4)


def _resolve_frontier_model_id(experiment: Experiment, explicit: str | None) -> str:
    token = str(explicit or "").strip().lower()
    if token in FRONTIER_REFERENCE:
        return token
    cfg = experiment.config or {}
    if isinstance(cfg, dict):
        cfg_id = str(cfg.get("frontier_model_id") or "").strip().lower()
        if cfg_id in FRONTIER_REFERENCE:
            return cfg_id
    return DEFAULT_FRONTIER_MODEL_ID


async def _experiment_in_project(
    db: AsyncSession, project_id: int, experiment_id: int
) -> Experiment | None:
    result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    return result.scalar_one_or_none()


async def _resolve_frontier_run_id(
    db: AsyncSession, project_id: int, slm_exp: Experiment, explicit: int | None
) -> int | None:
    if explicit is not None:
        return int(explicit)
    cfg = slm_exp.config or {}
    if isinstance(cfg, dict) and cfg.get("frontier_baseline_run_id") is not None:
        try:
            return int(cfg["frontier_baseline_run_id"])
        except (TypeError, ValueError):
            pass
    try:
        from app.services.evaluation_pack_service import resolve_project_evaluation_pack

        resolved = await resolve_project_evaluation_pack(db, project_id)
    except Exception:
        resolved = None
    for candidate in _iter_pack_dicts(resolved):
        value = candidate.get("frontier_baseline_run_id")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _iter_pack_dicts(resolved: Any):
    if isinstance(resolved, dict):
        yield resolved
        for key in ("pack", "active_pack"):
            nested = resolved.get(key)
            if isinstance(nested, dict):
                yield nested


def _representative_gpu_hourly_usd() -> tuple[float, str]:
    """Average $/hr for a small inference GPU from the cloud-burst catalog."""
    try:
        from app.services.cloud_burst_service import list_cloud_burst_catalog

        skus = list(list_cloud_burst_catalog().get("gpu_skus") or [])
        sku = next((s for s in skus if "a10g" in str(s.get("gpu_sku", ""))), skus[0] if skus else None)
        if sku:
            rates = [float(v) for v in (sku.get("hourly_usd") or {}).values() if float(v) > 0]
            if rates:
                return round(sum(rates) / len(rates), 4), f"cloud-burst catalog avg ({sku.get('gpu_sku')})"
    except Exception:
        pass
    return 0.75, "fallback heuristic ($0.75/GPU-hr)"


def _slm_perf_from_benchmark(project_id: int, base_model: str) -> dict[str, Any] | None:
    """Latest model-benchmark sweep row for the SLM's base model (estimated)."""
    try:
        from app.services.training_telemetry_service import list_model_benchmark_runs

        runs = list(list_model_benchmark_runs(project_id, limit=10).get("runs") or [])
    except Exception:
        return None
    target = str(base_model or "").strip().lower()
    for event in runs:  # newest first
        for row in list(event.get("matrix") or []):
            if str(row.get("model_id") or "").strip().lower() != target:
                continue
            latency = row.get("estimated_latency_ms")
            throughput = row.get("estimated_throughput_tps")
            try:
                latency_ms = float(latency) if latency is not None else None
                throughput_tps = float(throughput) if throughput is not None else None
            except (TypeError, ValueError):
                continue
            if latency_ms or throughput_tps:
                return {
                    "latency_ms": latency_ms,
                    "throughput_tps": throughput_tps,
                    "run_id": event.get("run_id"),
                }
    return None


def _compute_cost_latency(
    project_id: int, slm_exp: Experiment, frontier_ref: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the cost + latency comparison blocks with explicit provenance."""
    frontier_cost_per_1m = _blended_usd_per_1m(frontier_ref)
    frontier_latency_ms = float(frontier_ref["typical_latency_ms"])

    perf = _slm_perf_from_benchmark(project_id, slm_exp.base_model)

    cost: dict[str, Any] = {
        "frontier_usd_per_1m_tokens": frontier_cost_per_1m,
        "frontier_source": f"{frontier_ref['source']} (as of {frontier_ref['as_of']})",
        "slm_usd_per_1m_tokens": None,
        "cost_pct": None,
        "provenance": "unavailable",
        "message": None,
    }
    latency: dict[str, Any] = {
        "frontier_latency_ms": round(frontier_latency_ms, 1),
        "frontier_source": f"{frontier_ref['source']} (typical, as of {frontier_ref['as_of']})",
        "slm_latency_ms": None,
        "latency_ratio": None,
        "provenance": "unavailable",
        "message": None,
    }

    if perf is None:
        msg = (
            "Run a model benchmark sweep (Training → Power Tools) so the SLM's "
            "estimated latency/throughput can be compared to the frontier reference."
        )
        cost["message"] = msg
        latency["message"] = msg
        return cost, latency

    # Latency (estimated from the sweep).
    if perf.get("latency_ms"):
        slm_latency = float(perf["latency_ms"])
        latency["slm_latency_ms"] = round(slm_latency, 1)
        latency["latency_ratio"] = (
            round(slm_latency / frontier_latency_ms, 2) if frontier_latency_ms > 0 else None
        )
        latency["provenance"] = "estimated"
        latency["source"] = "model benchmark sweep (estimated)"

    # Cost: GPU $/hr ÷ throughput → $/1M tokens (estimated, full-utilization).
    if perf.get("throughput_tps") and float(perf["throughput_tps"]) > 0:
        gpu_hourly, gpu_source = _representative_gpu_hourly_usd()
        tokens_per_hour = float(perf["throughput_tps"]) * 3600.0
        slm_cost_per_1m = round((gpu_hourly / tokens_per_hour) * 1_000_000.0, 4)
        cost["slm_usd_per_1m_tokens"] = slm_cost_per_1m
        cost["cost_pct"] = (
            round((slm_cost_per_1m / frontier_cost_per_1m) * 100.0, 1)
            if frontier_cost_per_1m > 0
            else None
        )
        cost["provenance"] = "estimated"
        cost["gpu_hourly_usd"] = gpu_hourly
        cost["source"] = f"self-host estimate: {gpu_source} ÷ sweep throughput, full utilization"
        cost["message"] = (
            "Self-host cost assumes full GPU utilization; real $/token depends on "
            "your request volume."
        )

    return cost, latency


def _compute_metric_comparisons(
    slm_metrics: dict[str, float], frontier_metrics: dict[str, float]
) -> list[dict[str, Any]]:
    shared = set(slm_metrics) & set(frontier_metrics)
    if not shared:
        return []

    def _sort_key(metric_id: str) -> tuple[int, str]:
        idx = (
            _PREFERRED_HEADLINE_METRICS.index(metric_id)
            if metric_id in _PREFERRED_HEADLINE_METRICS
            else len(_PREFERRED_HEADLINE_METRICS)
        )
        return (idx, metric_id)

    rows: list[dict[str, Any]] = []
    for metric_id in sorted(shared, key=_sort_key):
        slm = float(slm_metrics[metric_id])
        frontier = float(frontier_metrics[metric_id])
        pct = _quality_pct(slm, frontier)
        rows.append(
            {
                "metric_id": metric_id,
                "slm_value": round(slm, 4),
                "frontier_value": round(frontier, 4),
                "quality_pct": pct,
                "direction": (
                    "exceeds" if pct is None
                    else "matches_or_better" if slm + 1e-4 >= frontier
                    else "behind"
                ),
                "is_headline": metric_id in _PREFERRED_HEADLINE_METRICS,
            }
        )
    return rows


def _serialize(exp: Experiment, eval_result: Any, metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "experiment_id": exp.id,
        "experiment_name": exp.name,
        "base_model": exp.base_model,
        "eval_result_id": eval_result.id,
        "dataset_name": eval_result.dataset_name,
        "eval_type": eval_result.eval_type,
        "metrics": dict(metrics),
        "pass_rate": float(eval_result.pass_rate) if eval_result.pass_rate is not None else None,
    }


def _build_headline(
    frontier_name: str,
    quality_pct: float | None,
    cost_pct: float | None,
    latency_ratio: float | None,
) -> str:
    """Compose the report sentence from whatever pieces are available (honest —
    only states a clause when its number exists)."""
    clauses: list[str] = []
    if quality_pct is not None:
        clauses.append(f"{round(quality_pct * 100)}% as good as {frontier_name}")
    if cost_pct is not None:
        clauses.append(f"{cost_pct:g}% of the cost")
    if latency_ratio is not None:
        clauses.append(f"{latency_ratio:g}× the latency")
    if not clauses:
        return f"Not enough data yet to compare this model against {frontier_name}."
    if quality_pct is not None:
        head, *rest = clauses
        return "Your model is " + head + (" at " + " and ".join(rest) if rest else "") + "."
    return "Your model runs at " + " and ".join(clauses) + f" vs {frontier_name}."


async def compute_frontier_comparison(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
    *,
    frontier_model_id: str | None = None,
    frontier_baseline_run_id: int | None = None,
) -> dict[str, Any]:
    """Compare a project's SLM experiment against a frontier model.

    Raises ``ValueError`` with ``project_not_found:`` / ``experiment_not_found:``
    prefixes (API → 404). Quality has soft-fallback ``status`` values; cost +
    latency always render (with provenance), and the headline is composed from
    whatever is available.
    """
    if (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none() is None:
        raise ValueError(f"project_not_found:{project_id}")
    slm_exp = await _experiment_in_project(db, project_id, experiment_id)
    if slm_exp is None:
        raise ValueError(f"experiment_not_found:{experiment_id}")

    frontier_id = _resolve_frontier_model_id(slm_exp, frontier_model_id)
    frontier_ref = FRONTIER_REFERENCE[frontier_id]
    frontier_name = str(frontier_ref["display_name"])

    cost, latency = _compute_cost_latency(project_id, slm_exp, frontier_ref)

    quality: dict[str, Any] = {
        "status": "ok",
        "metric_comparisons": [],
        "headline_quality_pct": None,
        "frontier_baseline_run_id": None,
        "message": None,
    }

    frontier_run_id = await _resolve_frontier_run_id(
        db, project_id, slm_exp, frontier_baseline_run_id
    )
    slm_er = await _latest_eval_result(db, slm_exp.id)
    slm_block = None
    frontier_block = None

    if slm_er is None:
        quality["status"] = "no_slm_eval"
        quality["message"] = "This experiment has no eval result yet. Run evaluation first."
    elif frontier_run_id is None:
        quality["status"] = "no_frontier_eval"
        quality["message"] = (
            f"No {frontier_name} baseline eval on this eval set. Evaluate {frontier_name} on "
            "the same gold set and set config.frontier_baseline_run_id (or the eval pack's), "
            "so quality can be compared honestly."
        )
    else:
        slm_metrics = _normalize_metrics(slm_er.metrics)
        slm_block = _serialize(slm_exp, slm_er, slm_metrics)
        frontier_exp = await _experiment_in_project(db, project_id, frontier_run_id)
        frontier_er = (
            await _latest_eval_result(db, frontier_exp.id) if frontier_exp is not None else None
        )
        quality["frontier_baseline_run_id"] = frontier_run_id
        if frontier_exp is None or frontier_er is None:
            quality["status"] = "no_frontier_eval"
            quality["message"] = (
                f"Frontier baseline run {frontier_run_id} has no eval result in this project."
            )
        else:
            frontier_metrics = _normalize_metrics(frontier_er.metrics)
            frontier_block = _serialize(frontier_exp, frontier_er, frontier_metrics)
            rows = _compute_metric_comparisons(slm_metrics, frontier_metrics)
            quality["metric_comparisons"] = rows
            if not rows:
                quality["status"] = "no_overlap"
                quality["message"] = (
                    "SLM and frontier evals share no comparable metrics. Re-run eval on both "
                    "with the same task profile / eval set."
                )
            else:
                headline = next((r for r in rows if r["is_headline"]), rows[0])
                quality["headline_quality_pct"] = headline["quality_pct"]

    headline = _build_headline(
        frontier_name,
        quality.get("headline_quality_pct"),
        cost.get("cost_pct"),
        latency.get("latency_ratio"),
    )

    return {
        "project_id": project_id,
        "frontier_model": {
            "id": frontier_id,
            "display_name": frontier_name,
            "source": frontier_ref["source"],
            "as_of": frontier_ref["as_of"],
        },
        "slm": slm_block or {
            "experiment_id": slm_exp.id,
            "experiment_name": slm_exp.name,
            "base_model": slm_exp.base_model,
        },
        "frontier": frontier_block,
        "quality": quality,
        "cost": cost,
        "latency": latency,
        "headline": headline,
    }


__all__ = ["compute_frontier_comparison", "FRONTIER_REFERENCE", "DEFAULT_FRONTIER_MODEL_ID"]
