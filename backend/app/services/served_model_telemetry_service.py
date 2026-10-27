"""Post-deploy telemetry service (priority.md P26).

Push-style ingestion of inference samples reported by a deployed
endpoint (or a sidecar scraping a provider's metrics API), and on-demand
aggregation over a sliding window:

- Ingest: ``ingest_samples`` accepts a list of sample dicts, validates +
  inserts the well-formed ones, and returns counts of accepted /
  rejected so a noisy collector doesn't poison the whole batch.
- Aggregate: ``compute_telemetry`` returns latency p50/p95/p99 + min /
  max / mean, error count + rate, request volume (total + per-second +
  per-minute), and token throughput (input / output / total per second)
  over a configurable window.

Stable reason codes (raised as ``ValueError``):
- ``deployment_version_not_found`` (404)
- ``samples_required`` (400 — empty or missing samples list)
- ``invalid_window`` (400 — window_seconds <= 0 or since > until)

----

**Production note — this service is push-only**. There is no built-in
scraper. The deployment plane in this codebase **will not collect any
telemetry on its own**; the data flow is exclusively:

    your inference client ──► POST /deployments/{id}/telemetry/ingest

Until something on your side calls that endpoint, every deployment
version will sit at ``sample_count=0`` and every P28 deployability
score will report ``telemetry_health: no signal``. Two practical
patterns to wire telemetry up:

1. **Inline reporting** — wrap your inference client so each request
   produces a sample dict (``latency_ms``, ``success``, ``status_code``,
   ``input_tokens``, ``output_tokens``) and post a batch every N
   requests or every M seconds.
2. **Provider-side scrape** — a sidecar that polls the provider's
   metrics API (HF Inference Endpoints analytics, SageMaker
   CloudWatch, vLLM Prometheus ``/metrics``), normalises into the
   sample shape, and posts batches. Each provider has a different
   API and no shared scraper exists in this codebase.

The endpoint accepts either a top-level list or ``{"samples": [...]}``,
in batches of up to ``_MAX_SAMPLES_PER_INGEST`` (5000). Invalid samples
in a batch are dropped with a per-sample reason rather than failing
the whole batch.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.deployment_telemetry import DeploymentTelemetrySample
from app.models.deployment_version import DeploymentVersion


_DEFAULT_WINDOW_SECONDS = 3600
_MAX_SAMPLES_PER_INGEST = 5000
_MAX_SAMPLES_PER_QUERY = 100_000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Sample validation
# ---------------------------------------------------------------------------


def _coerce_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    # Accept "...Z" by mapping to "...+00:00".
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _validate_sample(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(cleaned_sample, None)`` on success or ``(None, reason)``."""
    if not isinstance(raw, dict):
        return None, "not_an_object"

    latency = _coerce_float(raw.get("latency_ms"))
    if latency is None or latency < 0.0:
        return None, "invalid_latency_ms"

    success = raw.get("success")
    if success is None:
        # If the caller didn't tell us, infer from status_code: < 400 == ok.
        status_code = _coerce_int(raw.get("status_code"))
        success = bool(status_code is None or status_code < 400)
    success = bool(success)

    cleaned: dict[str, Any] = {
        "latency_ms": latency,
        "success": success,
        "status_code": _coerce_int(raw.get("status_code")),
        "error_code": (
            str(raw.get("error_code"))[:128]
            if raw.get("error_code") is not None
            else None
        ),
        "input_tokens": _coerce_int(raw.get("input_tokens")),
        "output_tokens": _coerce_int(raw.get("output_tokens")),
        "request_id": (
            str(raw.get("request_id"))[:128]
            if raw.get("request_id") is not None
            else None
        ),
        "ts": _coerce_ts(raw.get("ts")),
        "payload": raw.get("payload") if isinstance(raw.get("payload"), dict) else {},
    }
    return cleaned, None


# ---------------------------------------------------------------------------
# Ingestion
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


async def ingest_samples(
    db: AsyncSession,
    *,
    deployment_version_id: int,
    samples: Iterable[Any] | None,
) -> dict[str, Any]:
    """Persist a batch of telemetry samples for a deployment version.

    Lenient: invalid samples are dropped with a per-sample reason rather
    than failing the whole batch. Accepted-count and rejection reasons
    are returned so a collector can self-diagnose.
    """
    dv = await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    if not samples:
        raise ValueError("samples_required")

    materialised = list(samples)
    if not materialised:
        raise ValueError("samples_required")
    if len(materialised) > _MAX_SAMPLES_PER_INGEST:
        materialised = materialised[:_MAX_SAMPLES_PER_INGEST]

    accepted = 0
    rejected: list[dict[str, Any]] = []
    rows_to_add: list[DeploymentTelemetrySample] = []
    fallback_ts = _utcnow()

    for index, raw in enumerate(materialised):
        cleaned, reason = _validate_sample(raw)
        if cleaned is None:
            rejected.append({"index": index, "reason": reason})
            continue
        ts = cleaned["ts"] or fallback_ts
        row = DeploymentTelemetrySample(
            deployment_version_id=dv.id,
            project_id=dv.project_id,
            ts=ts,
            latency_ms=cleaned["latency_ms"],
            success=cleaned["success"],
            status_code=cleaned["status_code"],
            error_code=cleaned["error_code"],
            input_tokens=cleaned["input_tokens"],
            output_tokens=cleaned["output_tokens"],
            request_id=cleaned["request_id"],
            payload=cleaned["payload"],
        )
        rows_to_add.append(row)
        accepted += 1

    for row in rows_to_add:
        db.add(row)
    await db.flush()

    return {
        "deployment_version_id": dv.id,
        "accepted": accepted,
        "rejected": len(rejected),
        "rejected_details": rejected[:50],
        "received": len(materialised),
        "max_per_ingest": _MAX_SAMPLES_PER_INGEST,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return values[0]
    if p >= 100:
        return values[-1]
    k = (len(values) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[int(k)]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _resolve_window(
    *,
    window_seconds: int | None,
    since: datetime | str | None,
    until: datetime | str | None,
) -> tuple[datetime, datetime]:
    until_dt = _coerce_ts(until) if not isinstance(until, datetime) else until
    if until_dt is None:
        until_dt = _utcnow()

    since_dt = _coerce_ts(since) if not isinstance(since, datetime) else since
    if since_dt is None:
        seconds = (
            int(window_seconds)
            if window_seconds is not None
            else _DEFAULT_WINDOW_SECONDS
        )
        if seconds <= 0:
            raise ValueError("invalid_window")
        since_dt = until_dt - timedelta(seconds=seconds)

    if since_dt >= until_dt:
        raise ValueError("invalid_window")

    return since_dt, until_dt


async def compute_telemetry(
    db: AsyncSession,
    *,
    deployment_version_id: int,
    window_seconds: int | None = None,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
) -> dict[str, Any]:
    dv = await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    since_dt, until_dt = _resolve_window(
        window_seconds=window_seconds, since=since, until=until
    )

    result = await db.execute(
        select(DeploymentTelemetrySample)
        .where(
            DeploymentTelemetrySample.deployment_version_id == dv.id,
            DeploymentTelemetrySample.ts >= since_dt,
            DeploymentTelemetrySample.ts <= until_dt,
        )
        .order_by(DeploymentTelemetrySample.ts.asc())
        .limit(_MAX_SAMPLES_PER_QUERY)
    )
    rows = list(result.scalars().all())

    return _summarise(
        deployment_version_id=dv.id,
        rows=rows,
        since_dt=since_dt,
        until_dt=until_dt,
    )


def _summarise(
    *,
    deployment_version_id: int,
    rows: list[DeploymentTelemetrySample],
    since_dt: datetime,
    until_dt: datetime,
) -> dict[str, Any]:
    total = len(rows)
    window_seconds = max((until_dt - since_dt).total_seconds(), 1.0)

    if total == 0:
        return {
            "deployment_version_id": deployment_version_id,
            "window_start": since_dt.isoformat(),
            "window_end": until_dt.isoformat(),
            "window_seconds": window_seconds,
            "sample_count": 0,
            "request_volume": {
                "total": 0,
                "per_second": 0.0,
                "per_minute": 0.0,
            },
            "latency_ms": {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
            },
            "errors": {"count": 0, "rate": 0.0},
            "tokens": {
                "input_total": 0,
                "output_total": 0,
                "input_per_second": 0.0,
                "output_per_second": 0.0,
                "total_per_second": 0.0,
            },
        }

    latencies = sorted(float(r.latency_ms) for r in rows)
    error_count = sum(1 for r in rows if not r.success)
    input_total = sum(int(r.input_tokens or 0) for r in rows)
    output_total = sum(int(r.output_tokens or 0) for r in rows)

    return {
        "deployment_version_id": deployment_version_id,
        "window_start": since_dt.isoformat(),
        "window_end": until_dt.isoformat(),
        "window_seconds": window_seconds,
        "sample_count": total,
        "request_volume": {
            "total": total,
            "per_second": total / window_seconds,
            "per_minute": (total / window_seconds) * 60.0,
        },
        "latency_ms": {
            "p50": _percentile(latencies, 50.0),
            "p95": _percentile(latencies, 95.0),
            "p99": _percentile(latencies, 99.0),
            "min": latencies[0],
            "max": latencies[-1],
            "mean": sum(latencies) / total,
        },
        "errors": {
            "count": error_count,
            "rate": error_count / total,
        },
        "tokens": {
            "input_total": input_total,
            "output_total": output_total,
            "input_per_second": input_total / window_seconds,
            "output_per_second": output_total / window_seconds,
            "total_per_second": (input_total + output_total) / window_seconds,
        },
    }


# ---------------------------------------------------------------------------
# Read paths (raw samples, capped)
# ---------------------------------------------------------------------------


async def list_recent_samples(
    db: AsyncSession,
    *,
    deployment_version_id: int,
    limit: int = 100,
) -> dict[str, Any]:
    dv = await _load_deployment_version(
        db, deployment_version_id=deployment_version_id
    )
    bounded_limit = max(1, min(int(limit), 1000))
    result = await db.execute(
        select(DeploymentTelemetrySample)
        .where(DeploymentTelemetrySample.deployment_version_id == dv.id)
        .order_by(DeploymentTelemetrySample.ts.desc())
        .limit(bounded_limit)
    )
    rows = list(result.scalars().all())
    return {
        "deployment_version_id": dv.id,
        "limit": bounded_limit,
        "samples": [
            {
                "id": row.id,
                "ts": row.ts.isoformat() if row.ts else None,
                "latency_ms": row.latency_ms,
                "success": row.success,
                "status_code": row.status_code,
                "error_code": row.error_code,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "request_id": row.request_id,
            }
            for row in rows
        ],
    }


# ---------------------------------------------------------------------------
# Serving gateway (``/v1``): per-request log, keyed by project (G11)
# ---------------------------------------------------------------------------
#
# A gateway request has no DeploymentVersion (that needs an export), so it is
# logged to ``gateway_request_logs``. Only timing / status / token counts are
# stored -- never the prompt or the reply, which may be private code.


async def record_gateway_request(
    db: AsyncSession,
    *,
    project_id: int,
    key_id: int | None,
    model_id: str,
    run_id: int | None,
    latency_ms: float,
    status_code: int,
    error_code: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    retrieval_applied: bool = False,
    guided_json_mode: str | None = None,
    upstream: str = "in_process",
    request_id: str | None = None,
) -> None:
    """Insert one row. Never raises: a logging failure must not fail the request."""
    from app.models.gateway import GatewayRequestLog

    try:
        db.add(
            GatewayRequestLog(
                project_id=int(project_id),
                key_id=key_id,
                model_id=str(model_id or "")[:160],
                run_id=run_id,
                ts=_utcnow(),
                latency_ms=max(0.0, float(latency_ms or 0.0)),
                success=int(status_code) < 400,
                status_code=int(status_code),
                error_code=(str(error_code)[:128] if error_code else None),
                input_tokens=_coerce_int(input_tokens),
                output_tokens=_coerce_int(output_tokens),
                retrieval_applied=bool(retrieval_applied),
                guided_json_mode=(str(guided_json_mode)[:32] if guided_json_mode else None),
                upstream=str(upstream or "in_process")[:32],
                request_id=(str(request_id)[:128] if request_id else None),
            )
        )
        await db.commit()
    except Exception:  # noqa: BLE001 - telemetry is best-effort
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


async def compute_gateway_telemetry(
    db: AsyncSession,
    *,
    project_id: int,
    window_seconds: int | None = None,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
) -> dict[str, Any]:
    """Latency / error / volume summary of a project's gateway requests: the
    deployment-version ``_summarise`` over all rows, plus one entry per model id."""
    from app.models.gateway import GatewayRequestLog

    since_dt, until_dt = _resolve_window(window_seconds=window_seconds, since=since, until=until)
    result = await db.execute(
        select(GatewayRequestLog)
        .where(
            GatewayRequestLog.project_id == int(project_id),
            GatewayRequestLog.ts >= since_dt,
            GatewayRequestLog.ts <= until_dt,
        )
        .order_by(GatewayRequestLog.ts.asc())
        .limit(_MAX_SAMPLES_PER_QUERY)
    )
    rows = list(result.scalars().all())
    total = _summarise(deployment_version_id=0, rows=rows, since_dt=since_dt, until_dt=until_dt)
    total.pop("deployment_version_id", None)
    by_model: dict[str, list] = {}
    for row in rows:
        by_model.setdefault(row.model_id, []).append(row)
    entries = []
    for model_id, group in sorted(by_model.items()):
        part = _summarise(deployment_version_id=0, rows=group, since_dt=since_dt, until_dt=until_dt)
        entries.append(
            {
                "model_id": model_id,
                "requests": part["sample_count"],
                "error_rate": part["errors"]["rate"],
                "p50_ms": part["latency_ms"]["p50"],
                "p95_ms": part["latency_ms"]["p95"],
            }
        )
    return {
        "project_id": int(project_id),
        "by_model": entries,
        "total": total["sample_count"],
        "overall": total,
    }


async def list_recent_gateway_requests(
    db: AsyncSession, *, project_id: int, limit: int = 50
) -> list[dict[str, Any]]:
    from app.models.gateway import GatewayApiKey, GatewayRequestLog

    bounded = max(1, min(int(limit), 500))
    result = await db.execute(
        select(GatewayRequestLog)
        .where(GatewayRequestLog.project_id == int(project_id))
        .order_by(GatewayRequestLog.ts.desc(), GatewayRequestLog.id.desc())
        .limit(bounded)
    )
    rows = list(result.scalars().all())
    key_ids = {r.key_id for r in rows if r.key_id is not None}
    prefixes: dict[int, str] = {}
    if key_ids:
        keys = await db.execute(select(GatewayApiKey).where(GatewayApiKey.id.in_(key_ids)))
        prefixes = {k.id: k.key_prefix for k in keys.scalars().all()}
    return [
        {
            "ts": row.ts.isoformat() if row.ts else None,
            "model_id": row.model_id,
            "run_id": row.run_id,
            "status_code": row.status_code,
            "latency_ms": row.latency_ms,
            "retrieval_applied": bool(row.retrieval_applied),
            "guided_json_mode": row.guided_json_mode,
            "key_prefix": prefixes.get(row.key_id) if row.key_id is not None else None,
        }
        for row in rows
    ]
