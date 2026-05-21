"""Per-cluster failure explanations (Theme 8 Epic 3).

When the user expands a failure cluster on the Eval tab's
`FailureClustersPanel`, we generate a one-sentence "why this
cluster fails" explanation using the configured judge model and
cache it on the eval result so the LLM call doesn't repeat on
every expand.

Design choices:

- The eval-side clusters are computed on-the-fly per request
  (cluster_id is derived as `cluster-<idx>` based on count-ordered
  buckets), so we cache by ``(eval_result_id, cluster_id)`` rather
  than on a persisted FailureCluster row. Cache lives in
  ``EvalResult.details["cluster_explanations"][cluster_id]``.
- The existing ``_judge_with_local_serve`` / ``_judge_with_remote_model``
  helpers are dataset-row scorers (return ``(score, rationale)``).
  This service uses a different rubric — "explain the common
  failure pattern in one sentence" — and parses a free-form text
  response, so it talks to the judge transport directly via
  ``_call_judge_freeform`` rather than reusing the scoring helpers.
- When no judge is configured, the endpoint returns 200 with
  ``status="judge_unavailable"`` so the UI can render a soft
  fallback message instead of breaking the expand interaction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.config import settings
from app.models.experiment import EvalResult
from app.services.evaluation_service import (
    _build_judge_endpoint,
    _extract_local_response_text,
    _resolve_local_judge_target,
    _wait_for_local_judge_ready,
)
from app.services.failure_cluster_service import cluster_eval_result_failures
from app.services.secret_service import get_project_secret_value


# Cap on exemplars per cluster fed to the judge — too few and the
# judge can't see the pattern, too many and the prompt balloons.
# Five matches the per-cluster cap the existing P12 clusterer uses
# for UI display so we don't have to do a separate fetch.
DEFAULT_EXPLAIN_EXEMPLARS = 5

# Free-form explanation rubric. Pinned short to keep tokens cheap
# and so the chip stays a one-liner in the UI.
_EXPLAIN_RUBRIC = (
    "You are analyzing why a fine-tuned language model fails on a "
    "group of eval examples. Look at the examples below and find "
    "the COMMON PATTERN across them — the specific mistake the "
    "model is consistently making. Respond with ONE SHORT SENTENCE "
    "(max 25 words). Be specific. Do not preamble. Do not enumerate. "
    "Just the pattern."
)


# ─────────────────────────────────────────────────────────────────────
# Judge transport
# ─────────────────────────────────────────────────────────────────────


async def _call_judge_freeform(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    method: str,
    transport: str,
    judge_model: str,
    rubric: str,
    user_content: str,
    api_key: str | None = None,
) -> str:
    """Send a free-form prompt to the judge model and return its raw
    text response. Mirrors the transport branching in
    ``_judge_with_local_serve`` / ``_judge_with_remote_model`` but
    doesn't require strict-JSON parsing — perfect for one-line
    free-form explanations.

    Raises httpx errors / ValueError for empty responses; callers
    handle these as soft failures."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if transport == "openai_chat":
        body: dict[str, Any] = {
            "model": judge_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": rubric},
                {"role": "user", "content": user_content},
            ],
        }
    elif transport == "ollama_generate":
        body = {
            "model": judge_model,
            "stream": False,
            "prompt": f"{rubric}\n\n{user_content}",
        }
    else:
        body = {
            "prompt": f"{rubric}\n\n{user_content}",
            "max_tokens": 160,
            "temperature": 0,
        }

    resp = await client.request(
        method=method or "POST",
        url=endpoint,
        json=body,
        headers=headers,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError("Judge response is not a JSON object")
    content = _extract_local_response_text(payload, transport=transport)
    if not content:
        raise ValueError("Judge response was empty")
    # Trim the response to one line max — the judge sometimes
    # ignores the "one sentence" instruction.
    first_line = next(
        (line.strip() for line in content.splitlines() if line.strip()),
        "",
    )
    return first_line or content.strip()


# ─────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────


def _build_user_content(
    cluster: dict[str, Any],
    *,
    max_exemplars: int = DEFAULT_EXPLAIN_EXEMPLARS,
) -> str:
    """Render the cluster's exemplars into a numbered block the
    judge model can scan for the common failure pattern."""
    exemplars = list(cluster.get("exemplars") or [])[:max_exemplars]
    lines: list[str] = [
        f"Reason code: {cluster.get('reason_code') or 'unknown'}",
        f"Output pattern: {cluster.get('output_pattern') or 'unknown'}",
        f"Failure count: {cluster.get('failure_count') or len(exemplars)}",
        "",
        f"Here are {len(exemplars)} failing examples:",
        "",
    ]
    for idx, ex in enumerate(exemplars, start=1):
        prompt_text = str(ex.get("prompt") or "").strip()
        reference_text = str(ex.get("reference") or "").strip()
        prediction_text = str(ex.get("prediction") or "").strip()
        lines.extend([
            f"Example {idx}:",
            f"  Question: {prompt_text}",
            f"  Expected: {reference_text}",
            f"  Model said: {prediction_text}",
            "",
        ])
    lines.append(
        "What is the single common failure pattern across these examples?"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────


def _read_cached_explanation(
    eval_result: EvalResult, cluster_id: str
) -> dict[str, Any] | None:
    details = dict(eval_result.details or {})
    bucket = (details.get("cluster_explanations") or {}).get(cluster_id)
    if isinstance(bucket, dict) and bucket.get("explanation"):
        return bucket
    return None


def _write_cached_explanation(
    eval_result: EvalResult,
    cluster_id: str,
    payload: dict[str, Any],
) -> None:
    details = dict(eval_result.details or {})
    bucket = dict(details.get("cluster_explanations") or {})
    bucket[cluster_id] = payload
    details["cluster_explanations"] = bucket
    eval_result.details = details
    # SQLAlchemy JSON columns don't auto-detect in-place mutation.
    flag_modified(eval_result, "details")


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


class JudgeUnavailableError(RuntimeError):
    """Raised when no judge model is configured for this project."""


async def _resolve_judge_transport(
    db: AsyncSession,
    project_id: int,
    judge_model: str,
) -> tuple[str, dict[str, Any] | None, str | None, str | None]:
    """Decide whether to route through a local serve run or a remote
    judge API. Returns (provider, local_target, endpoint, api_key).

    `local_target` is set for local serve runs (carries endpoint /
    method / transport). For remote API path, (endpoint, api_key)
    are set instead. Raises JudgeUnavailableError when neither is
    configured."""
    # Prefer a local serve run if any are healthy in this project.
    try:
        local_target = await _resolve_local_judge_target(
            project_id=project_id,
            run_id=None,
            model_override=None,
            judge_model=judge_model,
        )
    except Exception:
        local_target = None

    if local_target and local_target.get("endpoint"):
        # Best-effort readiness ping — if the serve run is warming
        # up we want to wait briefly rather than bail.
        try:
            await _wait_for_local_judge_ready(
                project_id=project_id,
                run_id=str(local_target.get("run_id") or "") or None,
                timeout_seconds=20,
            )
        except Exception:
            pass
        return "local_serve", local_target, None, None

    secret_api_url = await get_project_secret_value(
        db, project_id, "judge_model", "api_url"
    )
    secret_api_key = await get_project_secret_value(
        db, project_id, "judge_model", "api_key"
    )
    resolved_api_url = secret_api_url or settings.JUDGE_MODEL_API_URL
    resolved_api_key = secret_api_key or settings.JUDGE_MODEL_API_KEY
    judge_endpoint = (
        _build_judge_endpoint(resolved_api_url) if resolved_api_url else ""
    )
    if judge_endpoint:
        return "remote_api", None, judge_endpoint, resolved_api_key

    raise JudgeUnavailableError(
        "No judge model is configured. Set JUDGE_MODEL_API_URL + "
        "JUDGE_MODEL_API_KEY (or wire a local serve run) to generate "
        "per-cluster failure explanations."
    )


async def explain_failure_cluster(
    db: AsyncSession,
    *,
    project_id: int,
    eval_result_id: int,
    cluster_id: str,
    max_exemplars: int = DEFAULT_EXPLAIN_EXEMPLARS,
    force_refresh: bool = False,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """Generate (or return cached) one-line explanation for a
    failure cluster.

    Returns a dict shaped:
        {
            cluster_id,
            explanation,       # str, possibly empty if unavailable
            status,            # 'ok' | 'judge_unavailable' | 'cluster_not_found' | 'error'
            cached,            # bool — True when served from cache
            generated_at,      # iso8601 UTC, when first generated
            model,             # judge model id used
            exemplar_count,    # how many exemplars the judge saw
            note,              # human-friendly status detail
        }

    Raises ValueError("eval_result_not_found:{id}") for unknown
    eval_result_id; the API maps to 404.
    """
    result = await db.execute(
        select(EvalResult).where(EvalResult.id == eval_result_id)
    )
    eval_result = result.scalar_one_or_none()
    if eval_result is None:
        raise ValueError(f"eval_result_not_found:{eval_result_id}")

    if not force_refresh:
        cached = _read_cached_explanation(eval_result, cluster_id)
        if cached is not None:
            return {
                "cluster_id": cluster_id,
                "explanation": str(cached.get("explanation") or ""),
                "status": str(cached.get("status") or "ok"),
                "cached": True,
                "generated_at": cached.get("generated_at"),
                "model": cached.get("model"),
                "exemplar_count": int(cached.get("exemplar_count") or 0),
                "note": cached.get("note"),
            }

    # Recompute clusters to find the one matching cluster_id. Pulling
    # from the same service the UI uses keeps the cluster_id stable
    # within a single eval result.
    cluster_payload = await cluster_eval_result_failures(
        db,
        eval_result_id=eval_result_id,
        max_exemplars_per_cluster=max(1, int(max_exemplars)),
    )
    matching = next(
        (
            c for c in (cluster_payload.get("clusters") or [])
            if str(c.get("cluster_id")) == str(cluster_id)
        ),
        None,
    )
    if matching is None:
        return {
            "cluster_id": cluster_id,
            "explanation": "",
            "status": "cluster_not_found",
            "cached": False,
            "generated_at": None,
            "model": None,
            "exemplar_count": 0,
            "note": (
                f"No cluster with id '{cluster_id}' in the latest "
                "computation of this eval result."
            ),
        }

    effective_model = (judge_model or "").strip() or "auto"
    try:
        provider, local_target, judge_endpoint, judge_api_key = (
            await _resolve_judge_transport(db, project_id, effective_model)
        )
    except JudgeUnavailableError as e:
        unavailable = {
            "cluster_id": cluster_id,
            "explanation": "",
            "status": "judge_unavailable",
            "cached": False,
            "generated_at": None,
            "model": None,
            "exemplar_count": int(matching.get("failure_count") or 0),
            "note": str(e),
        }
        # Don't cache an unavailable verdict — the user might
        # configure the judge later and try again.
        return unavailable

    user_content = _build_user_content(matching, max_exemplars=max_exemplars)
    timeout = max(15.0, float(settings.JUDGE_MODEL_TIMEOUT_SECONDS or 120.0))

    explanation_text = ""
    judge_model_used = effective_model
    error_note: str | None = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if provider == "local_serve" and local_target is not None:
                judge_model_used = (
                    str(local_target.get("model") or "local-judge")
                )
                explanation_text = await _call_judge_freeform(
                    client,
                    endpoint=str(local_target.get("endpoint") or ""),
                    method=str(local_target.get("method") or "POST"),
                    transport=str(local_target.get("transport") or "openai_chat"),
                    judge_model=judge_model_used,
                    rubric=_EXPLAIN_RUBRIC,
                    user_content=user_content,
                )
            else:
                judge_model_used = effective_model
                explanation_text = await _call_judge_freeform(
                    client,
                    endpoint=str(judge_endpoint or ""),
                    method="POST",
                    transport="openai_chat",
                    judge_model=judge_model_used,
                    rubric=_EXPLAIN_RUBRIC,
                    user_content=user_content,
                    api_key=judge_api_key or None,
                )
        except Exception as exc:
            error_note = f"Judge call failed: {exc.__class__.__name__}: {exc}"

    generated_at = datetime.now(timezone.utc).isoformat()

    if explanation_text:
        payload = {
            "explanation": explanation_text,
            "status": "ok",
            "generated_at": generated_at,
            "model": judge_model_used,
            "exemplar_count": min(
                int(matching.get("failure_count") or 0),
                int(max_exemplars),
            ),
            "note": None,
        }
        _write_cached_explanation(eval_result, cluster_id, payload)
        await db.flush()
        return {
            "cluster_id": cluster_id,
            **payload,
            "cached": False,
        }

    return {
        "cluster_id": cluster_id,
        "explanation": "",
        "status": "error",
        "cached": False,
        "generated_at": generated_at,
        "model": judge_model_used,
        "exemplar_count": 0,
        "note": error_note or "Judge returned an empty response.",
    }


__all__ = [
    "DEFAULT_EXPLAIN_EXEMPLARS",
    "JudgeUnavailableError",
    "explain_failure_cluster",
]
