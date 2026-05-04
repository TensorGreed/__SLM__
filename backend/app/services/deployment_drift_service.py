"""On-demand drift check against a deployed endpoint (priority.md P27).

Re-runs an exact-match scoring pass over the project's gold-set rows
(P10) and compares the resulting pass rate against the baseline
:class:`EvalResult` produced at training time (matching ``eval_type`` +
``dataset_name``). Persists the verdict to ``deployment_drift_checks``
(P27 model) so the deployment-assistant UI can render a trend without
re-running the eval.

Two ways to supply predictions:

- **offline** (default; the test path) — caller passes ``predictions``
  as ``[{row_id, prediction}]`` directly. Useful for replaying a known
  inference batch or for tests that don't want a live HTTP call.
- **live_url** — caller passes ``endpoint_url`` (and optionally
  ``endpoint_headers`` and ``request_template``); the service POSTs
  ``{prompt}`` per row, expects ``{prediction}`` back, and tolerates
  missing predictions by counting them as failures.

A future provider-SDK mode (HF / SageMaker / vLLM-managed) plugs in
behind ``_collect_live_predictions`` without touching callers.

Stable reason codes raised as ``ValueError`` (mapped to HTTP by the API):
- ``deployment_version_not_found`` (404)
- ``gold_set_not_found`` / ``gold_set_no_rows`` (404)
- ``endpoint_or_predictions_required`` (400)
- ``invalid_tolerance`` (400 — outside [0, 1])
- ``invalid_max_samples`` (400 — outside [1, 500])
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset
from app.models.deployment_drift_check import DeploymentDriftCheck
from app.models.deployment_version import DeploymentVersion
from app.models.experiment import EvalResult
from app.models.export import Export
from app.models.gold_set_annotation import GoldSetRow, GoldSetVersion


_DEFAULT_TOLERANCE = 0.05
_DEFAULT_MAX_SAMPLES = 50
_HARD_MAX_SAMPLES = 500
_PER_ROW_RESULT_CAP = 100


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


async def _load_deployment_version(
    db: AsyncSession, *, deployment_version_id: int
) -> DeploymentVersion:
    result = await db.execute(
        select(DeploymentVersion).where(
            DeploymentVersion.id == deployment_version_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError("deployment_version_not_found")
    return row


async def _load_gold_set(
    db: AsyncSession, *, gold_set_id: int
) -> Dataset:
    result = await db.execute(
        select(Dataset).where(Dataset.id == gold_set_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError("gold_set_not_found")
    return row


async def _load_latest_gold_set_version(
    db: AsyncSession, *, gold_set_id: int
) -> GoldSetVersion | None:
    result = await db.execute(
        select(GoldSetVersion)
        .where(GoldSetVersion.gold_set_id == gold_set_id)
        .order_by(GoldSetVersion.version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _load_gold_set_rows(
    db: AsyncSession,
    *,
    gold_set_version_id: int,
    max_samples: int,
) -> list[GoldSetRow]:
    result = await db.execute(
        select(GoldSetRow)
        .where(GoldSetRow.version_id == gold_set_version_id)
        .order_by(GoldSetRow.id.asc())
        .limit(max_samples)
    )
    return list(result.scalars().all())


async def _resolve_baseline(
    db: AsyncSession,
    *,
    deployment_version: DeploymentVersion,
    eval_type: str,
    gold_set_dataset_name: str | None,
) -> tuple[int | None, EvalResult | None]:
    """Most recent ``EvalResult`` for the deployment's underlying experiment.

    Returns ``(experiment_id, eval_result)`` — experiment_id is populated
    even when no eval result exists, so the persisted row still
    points at the right run.
    """
    export_result = await db.execute(
        select(Export).where(Export.id == deployment_version.export_id)
    )
    export = export_result.scalar_one_or_none()
    experiment_id = export.experiment_id if export is not None else None
    if experiment_id is None:
        return None, None

    stmt = select(EvalResult).where(
        EvalResult.experiment_id == experiment_id,
        EvalResult.eval_type == eval_type,
    )
    if gold_set_dataset_name:
        stmt = stmt.where(EvalResult.dataset_name == gold_set_dataset_name)
    stmt = stmt.order_by(EvalResult.created_at.desc()).limit(1)
    eval_result = (await db.execute(stmt)).scalar_one_or_none()

    if eval_result is None and gold_set_dataset_name is not None:
        # Retry without the dataset filter: better to match on eval_type
        # alone than to report "no baseline" when the operator picked a
        # different gold set than the one used at training time.
        broad_stmt = (
            select(EvalResult)
            .where(
                EvalResult.experiment_id == experiment_id,
                EvalResult.eval_type == eval_type,
            )
            .order_by(EvalResult.created_at.desc())
            .limit(1)
        )
        eval_result = (await db.execute(broad_stmt)).scalar_one_or_none()

    return experiment_id, eval_result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _flatten_to_string(value: Any) -> str:
    """Best-effort flattening of a JSON value to a string for exact match.

    - Strings pass through (stripped).
    - Numbers / bools become their ``str()`` form.
    - Dicts pick the first non-empty value among ``answer`` / ``text`` /
      ``output`` / ``response``; otherwise they get JSON-encoded with
      sorted keys so two semantically-identical dicts compare equal.
    - Lists get JSON-encoded.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for preferred in ("answer", "text", "output", "response"):
            v = value.get(preferred)
            if v is not None and v != "":
                return _flatten_to_string(v)
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return str(value)
    if isinstance(value, list):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def _exact_match(expected: Any, prediction: Any) -> bool:
    """Case-insensitive whitespace-tolerant exact-match scorer."""
    a = _flatten_to_string(expected).lower().strip()
    b = _flatten_to_string(prediction).lower().strip()
    if not a and not b:
        return True
    return a == b


# ---------------------------------------------------------------------------
# Prediction collection (offline + live_url)
# ---------------------------------------------------------------------------


def _normalise_offline_predictions(
    predictions: Iterable[Any] | None,
) -> dict[int, Any]:
    if not predictions:
        return {}
    by_row: dict[int, Any] = {}
    for entry in predictions:
        if not isinstance(entry, dict):
            continue
        row_id = entry.get("row_id")
        if row_id is None:
            continue
        try:
            row_id_int = int(row_id)
        except (TypeError, ValueError):
            continue
        by_row[row_id_int] = entry.get("prediction")
    return by_row


async def _collect_live_predictions(
    *,
    rows: list[GoldSetRow],
    endpoint_url: str,
    endpoint_headers: dict[str, str] | None,
) -> tuple[dict[int, Any], dict[int, str]]:
    """POST ``{prompt}`` per row to ``endpoint_url`` and collect predictions.

    Returns ``(predictions, errors)`` where ``errors`` maps row_id to a
    short error string for failed calls; failed rows are counted as
    ``samples_failed`` and treated as non-matches.
    """
    try:
        import httpx  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ValueError(f"httpx_required_for_live_drift: {exc}") from exc

    predictions: dict[int, Any] = {}
    errors: dict[int, str] = {}
    headers = dict(endpoint_headers or {})
    headers.setdefault("Content-Type", "application/json")

    async with httpx.AsyncClient(timeout=30.0) as client:
        for row in rows:
            prompt = _flatten_to_string(row.input)
            try:
                response = await client.post(
                    endpoint_url,
                    json={"prompt": prompt},
                    headers=headers,
                )
                if response.status_code >= 400:
                    errors[row.id] = (
                        f"http_{response.status_code}:"
                        f"{(response.text or '')[:200]}"
                    )
                    continue
                data = response.json()
            except Exception as exc:  # pragma: no cover - network paths
                errors[row.id] = f"request_failed:{exc!r}"
                continue
            if isinstance(data, dict):
                predictions[row.id] = data.get("prediction") or data.get(
                    "answer"
                ) or data.get("output") or data.get("text") or data
            else:
                predictions[row.id] = data
    return predictions, errors


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_drift_check(
    db: AsyncSession,
    *,
    deployment_version_id: int,
    gold_set_id: int,
    predictions: Iterable[Any] | None = None,
    endpoint_url: str | None = None,
    endpoint_headers: dict[str, str] | None = None,
    eval_type: str = "exact_match",
    tolerance: float = _DEFAULT_TOLERANCE,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    notes: str | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    if not (0.0 <= tolerance <= 1.0):
        raise ValueError("invalid_tolerance")
    if not (1 <= int(max_samples) <= _HARD_MAX_SAMPLES):
        raise ValueError("invalid_max_samples")

    dv = await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    gold_set = await _load_gold_set(db, gold_set_id=gold_set_id)
    version = await _load_latest_gold_set_version(
        db, gold_set_id=gold_set.id
    )
    if version is None:
        raise ValueError("gold_set_no_rows")

    rows = await _load_gold_set_rows(
        db,
        gold_set_version_id=version.id,
        max_samples=int(max_samples),
    )
    if not rows:
        raise ValueError("gold_set_no_rows")

    offline_predictions = _normalise_offline_predictions(predictions)
    use_offline = bool(offline_predictions)
    use_live = bool(endpoint_url) and not use_offline
    if not use_offline and not use_live:
        raise ValueError("endpoint_or_predictions_required")

    mode = "offline" if use_offline else "live_url"

    if use_offline:
        prediction_map = offline_predictions
        live_errors: dict[int, str] = {}
    else:
        prediction_map, live_errors = await _collect_live_predictions(
            rows=rows,
            endpoint_url=str(endpoint_url),
            endpoint_headers=endpoint_headers,
        )

    matches = 0
    failures = 0
    skipped = 0
    per_row_results: list[dict[str, Any]] = []

    for row in rows:
        if row.id in live_errors:
            failures += 1
            if len(per_row_results) < _PER_ROW_RESULT_CAP:
                per_row_results.append(
                    {
                        "row_id": row.id,
                        "match": False,
                        "expected": row.expected,
                        "prediction": None,
                        "error": live_errors[row.id],
                    }
                )
            continue
        if row.id not in prediction_map:
            skipped += 1
            if len(per_row_results) < _PER_ROW_RESULT_CAP:
                per_row_results.append(
                    {
                        "row_id": row.id,
                        "match": False,
                        "expected": row.expected,
                        "prediction": None,
                        "error": "no_prediction",
                    }
                )
            continue
        prediction = prediction_map[row.id]
        match = _exact_match(row.expected, prediction)
        if match:
            matches += 1
        if len(per_row_results) < _PER_ROW_RESULT_CAP:
            per_row_results.append(
                {
                    "row_id": row.id,
                    "match": match,
                    "expected": row.expected,
                    "prediction": prediction,
                    "error": None,
                }
            )

    samples_evaluated = max(len(rows) - failures - skipped, 0)
    current_pass_rate = (
        matches / samples_evaluated if samples_evaluated > 0 else 0.0
    )

    baseline_experiment_id, baseline_eval_result = await _resolve_baseline(
        db,
        deployment_version=dv,
        eval_type=eval_type,
        gold_set_dataset_name=gold_set.name,
    )
    baseline_pass_rate = (
        float(baseline_eval_result.pass_rate)
        if baseline_eval_result is not None
        and baseline_eval_result.pass_rate is not None
        else None
    )
    delta = (
        current_pass_rate - baseline_pass_rate
        if baseline_pass_rate is not None
        else None
    )
    drift_detected = (
        delta is not None and abs(delta) > float(tolerance)
    )

    summary = {
        "matches": matches,
        "samples_evaluated": samples_evaluated,
        "samples_failed": failures,
        "samples_skipped": skipped,
        "rows_loaded": len(rows),
        "mode": mode,
    }

    actor_str = (actor or "system").strip()[:128] or "system"

    row_obj = DeploymentDriftCheck(
        deployment_version_id=dv.id,
        project_id=dv.project_id,
        gold_set_id=gold_set.id,
        gold_set_version_id=version.id,
        baseline_experiment_id=baseline_experiment_id,
        baseline_eval_result_id=(
            baseline_eval_result.id if baseline_eval_result is not None else None
        ),
        eval_type=eval_type,
        baseline_pass_rate=baseline_pass_rate,
        current_pass_rate=current_pass_rate,
        delta=delta,
        tolerance=float(tolerance),
        drift_detected=drift_detected,
        samples_evaluated=samples_evaluated,
        samples_failed=failures,
        samples_skipped=skipped,
        mode=mode,
        notes=notes,
        actor=actor_str,
        per_row_results=per_row_results,
        summary=summary,
    )
    db.add(row_obj)
    await db.flush()
    await db.refresh(row_obj)

    return _serialize(row_obj)


# ---------------------------------------------------------------------------
# Read paths
# ---------------------------------------------------------------------------


def _serialize(row: DeploymentDriftCheck) -> dict[str, Any]:
    return {
        "id": row.id,
        "deployment_version_id": row.deployment_version_id,
        "project_id": row.project_id,
        "gold_set_id": row.gold_set_id,
        "gold_set_version_id": row.gold_set_version_id,
        "baseline_experiment_id": row.baseline_experiment_id,
        "baseline_eval_result_id": row.baseline_eval_result_id,
        "eval_type": row.eval_type,
        "baseline_pass_rate": row.baseline_pass_rate,
        "current_pass_rate": row.current_pass_rate,
        "delta": row.delta,
        "tolerance": row.tolerance,
        "drift_detected": row.drift_detected,
        "samples_evaluated": row.samples_evaluated,
        "samples_failed": row.samples_failed,
        "samples_skipped": row.samples_skipped,
        "mode": row.mode,
        "notes": row.notes,
        "actor": row.actor,
        "per_row_results": list(row.per_row_results or []),
        "summary": dict(row.summary or {}),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


async def list_drift_checks(
    db: AsyncSession, *, deployment_version_id: int, limit: int = 50
) -> dict[str, Any]:
    dv = await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    bounded = max(1, min(int(limit), 200))
    result = await db.execute(
        select(DeploymentDriftCheck)
        .where(
            DeploymentDriftCheck.deployment_version_id == dv.id
        )
        .order_by(DeploymentDriftCheck.created_at.desc())
        .limit(bounded)
    )
    rows = list(result.scalars().all())
    return {
        "deployment_version_id": dv.id,
        "limit": bounded,
        "drift_checks": [_serialize(r) for r in rows],
    }


async def get_drift_check(
    db: AsyncSession, *, drift_check_id: int
) -> dict[str, Any]:
    result = await db.execute(
        select(DeploymentDriftCheck).where(
            DeploymentDriftCheck.id == drift_check_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError("drift_check_not_found")
    return _serialize(row)
