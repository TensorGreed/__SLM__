"""Drift-triggered hallucination-trap refresh (E4).

Generates fresh hallucination traps targeting the project's most
recent failure-cluster patterns and persists them to
``gold_drift_review_queue`` for the user to triage. Triggered:

  * On demand via ``POST /api/projects/{id}/drift/refresh-traps``.
  * Automatically by the per-deployment drift-check runner when the
    project's ``runtime_config.drift_refresh_traps.enabled`` flag is
    true.

The default generator path calls ``gold_llm_service.generate_gold_qa_via_llm``
with one ``focus_hint`` per cluster pattern (last 7 days, top-N by
``last_seen_at``). When no LLM credentials are configured for the
project the runner falls back to ``_simulate_traps`` which produces
deterministic placeholder rows — never fail-loud, so a project that
hasn't wired its API key yet still gets a queue populated when
``simulate=True`` is requested explicitly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.failure_cluster import FailureCluster
from app.models.gold_drift_review_queue import (
    GoldDriftQueueStatus,
    GoldDriftReviewQueueRow,
)
from app.models.project import Project


# Project's runtime_config knob path. Stored as a nested dict so we
# can add other drift-refresh tuning (interval, max queue depth)
# without expanding the top-level config.
RUNTIME_CONFIG_KEY = "drift_refresh_traps"

# Defaults when the flag dict is missing or partial.
DEFAULT_COUNT = 5
MAX_COUNT = 20
MAX_CLUSTER_LOOKBACK_DAYS = 7
MAX_CLUSTERS_FETCHED = 8


def is_trap_refresh_enabled(project: Project) -> bool:
    """Whether automatic trap refresh is opted in for the project.

    Reads ``project.runtime_config[RUNTIME_CONFIG_KEY]["enabled"]``.
    Default OFF so the LLM-cost surface stays opt-in.
    """
    config = (project.runtime_config or {}).get(RUNTIME_CONFIG_KEY) or {}
    return bool(config.get("enabled"))


def resolved_target_count(project: Project, override: int | None = None) -> int:
    """The per-refresh trap-count target, clamped to [1, MAX_COUNT].

    Order of precedence:
      1. Explicit override (the manual endpoint accepts ``?count=``).
      2. ``runtime_config.drift_refresh_traps.count``.
      3. ``DEFAULT_COUNT``.
    """
    if override is not None:
        return max(1, min(MAX_COUNT, int(override)))
    config = (project.runtime_config or {}).get(RUNTIME_CONFIG_KEY) or {}
    raw = config.get("count")
    try:
        n = int(raw) if raw is not None else DEFAULT_COUNT
    except (TypeError, ValueError):
        n = DEFAULT_COUNT
    return max(1, min(MAX_COUNT, n))


async def _load_recent_clusters(
    db: AsyncSession,
    *,
    project_id: int,
    lookback_days: int = MAX_CLUSTER_LOOKBACK_DAYS,
    limit: int = MAX_CLUSTERS_FETCHED,
) -> list[FailureCluster]:
    """Last ``lookback_days`` of failure clusters for the project,
    newest-first. Empty list when no clusters exist (project hasn't
    run any evals yet) — the runner then generates generic traps."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    result = await db.execute(
        select(FailureCluster)
        .where(
            FailureCluster.project_id == project_id,
            FailureCluster.last_seen_at >= cutoff,
        )
        .order_by(FailureCluster.last_seen_at.desc())
        .limit(max(1, int(limit)))
    )
    return list(result.scalars())


# ─────────────────────────────────────────────────────────────────────
# Generator strategies — pluggable so tests can inject deterministic
# output without spinning up a real LLM provider.
# ─────────────────────────────────────────────────────────────────────


TrapGenerator = Callable[
    [AsyncSession, Project, list[FailureCluster], int],
    Awaitable[list[dict[str, Any]]],
]


async def _simulate_traps(
    db: AsyncSession,
    project: Project,
    clusters: list[FailureCluster],
    count: int,
) -> list[dict[str, Any]]:
    """Deterministic placeholder traps. Used in dev/test when no LLM
    credentials are wired, and as the explicit ``simulate=True`` path
    in the manual endpoint.

    Each trap is shaped by the project's recipe so the downstream
    triage-accept flow can produce a valid gold_test row regardless
    of recipe.
    """
    recipe_id = ((project.selected_recipe or {}).get("recipe_id") or "qa-sft").lower()
    rows: list[dict[str, Any]] = []
    # Round-robin over clusters so each one gets at least one trap
    # before any cluster gets two. When there are no clusters,
    # emit generic traps tagged with cluster=None.
    pool = clusters or [None] * count
    for idx in range(count):
        cluster = pool[idx % len(pool)] if pool else None
        cluster_reason = cluster.reason_code if cluster else None
        cluster_signature = cluster.signature if cluster else None
        cluster_label = cluster.reason_code if cluster else "general"
        if recipe_id == "classification":
            payload: dict[str, Any] = {
                "text": f"Drift-refresh trap #{idx + 1} targeting {cluster_label}. The model "
                        f"should not infer a class from this ambiguous phrasing.",
                "label": "uncertain",
                "is_hallucination_trap": True,
                "rationale": f"Synthetic trap probing the '{cluster_label}' failure pattern.",
            }
        elif recipe_id == "span-extraction":
            payload = {
                "text": f"Drift-refresh trap #{idx + 1} for cluster {cluster_label}. No entities present.",
                "entities": [],
                "is_hallucination_trap": True,
                "rationale": f"Negative example for '{cluster_label}' — model must not over-extract.",
            }
        elif recipe_id == "summarization":
            payload = {
                "document": f"Drift-refresh trap #{idx + 1}. This document deliberately omits the "
                            f"information about {cluster_label} that the model has been hallucinating.",
                "summary": f"No information available about {cluster_label}.",
                "is_hallucination_trap": True,
                "rationale": f"Tests refusal-to-fabricate for '{cluster_label}' cluster.",
            }
        else:
            # qa-sft + generic-sft + code-review all read Q+A.
            payload = {
                "question": f"Drift-refresh trap #{idx + 1}: "
                            f"what is the answer to a question about {cluster_label} that "
                            f"isn't supported by your training data?",
                "answer": "I don't have reliable information about this — please consult an authoritative source.",
                "difficulty": "hard",
                "is_hallucination_trap": True,
                "rationale": f"Probes hallucination on '{cluster_label}' pattern.",
            }
        rows.append({
            "payload": payload,
            "cluster_reason_code": cluster_reason,
            "cluster_signature": cluster_signature,
        })
    return rows


async def _llm_generated_traps(
    db: AsyncSession,
    project: Project,
    clusters: list[FailureCluster],
    count: int,
) -> list[dict[str, Any]]:
    """Real-LLM generator. Calls
    ``gold_llm_service.generate_gold_qa_via_llm`` once with a
    composite focus_hint built from the cluster patterns, requesting
    ``distribution.hallucination_traps = count``.

    Falls back to ``_simulate_traps`` on any caller-fixable error
    (missing recipe, missing stored API key, provider rate-limited)
    so a single misconfigured project doesn't break the weekly tick.
    """
    try:
        from app.services.gold_llm_service import (
            GoldGenerationError,
            generate_gold_qa_via_llm,
        )
    except Exception:
        # gold_llm module unimportable for some reason — degrade to
        # simulate so the queue still populates.
        return await _simulate_traps(db, project, clusters, count)

    # Build a multi-line focus_hint so the LLM sees each cluster
    # pattern as a separate target. The composite is intentionally
    # short — too many bullets dilutes the LLM's attention.
    hints: list[str] = []
    for cluster in clusters[:5]:
        bullet = f"- {cluster.reason_code}"
        if cluster.exemplar_summaries:
            bullet += f": {str(cluster.exemplar_summaries[0])[:160]}"
        hints.append(bullet)
    focus_hint = "Generate hallucination traps targeting these recent failure patterns:\n" + (
        "\n".join(hints) if hints else "- generic hallucination probing"
    )

    # Look up the stored API key + provider from the project's
    # secret store. Without a key we can't make the LLM call.
    api_key, provider, api_url = await _resolve_stored_llm_key(db, project.id)
    if not api_key:
        # Caller can still ask for simulate explicitly via the
        # manual endpoint; for the auto-trigger we fall back so the
        # queue isn't perpetually empty for projects that opted in
        # without storing a key.
        return await _simulate_traps(db, project, clusters, count)

    try:
        result = await generate_gold_qa_via_llm(
            db,
            project_id=project.id,
            provider=provider,
            model=_default_model_for(provider),
            api_key=api_key,
            api_url=api_url,
            count=count,
            focus_hint=focus_hint,
            ground_in_source=True,
            # All-traps distribution so the LLM produces only
            # hallucination-probing rows.
            distribution=(0, 0, 0, count),
        )
    except GoldGenerationError:
        # Recipe missing / config bad — runner is best-effort.
        return await _simulate_traps(db, project, clusters, count)
    except Exception:
        return await _simulate_traps(db, project, clusters, count)

    # Tag rows by round-robin cluster assignment so the queue rows
    # carry the same diagnostic context the simulate path provides.
    out: list[dict[str, Any]] = []
    rows = list(result.rows or [])
    for idx, row in enumerate(rows):
        cluster = clusters[idx % len(clusters)] if clusters else None
        out.append({
            "payload": {
                **dict(row),
                "is_hallucination_trap": True,
            },
            "cluster_reason_code": cluster.reason_code if cluster else None,
            "cluster_signature": cluster.signature if cluster else None,
        })
    return out


async def _resolve_stored_llm_key(
    db: AsyncSession, project_id: int
) -> tuple[str | None, str, str | None]:
    """Look up the project's stored LLM key for the trap refresh.
    Tries openai → anthropic → deepseek in order so the first
    configured provider wins. Returns (api_key | None, provider tag,
    api_url | None)."""
    try:
        from app.services.secret_service import get_project_secret_value
    except Exception:
        return None, "openai", None

    for provider_tag, (secret_provider, key_name), api_url in (
        ("openai", ("cloud_llm_openai", "api_key"), None),
        ("anthropic", ("cloud_llm_anthropic", "api_key"), None),
        ("deepseek", ("cloud_llm_deepseek", "api_key"), "https://api.deepseek.com/v1"),
    ):
        try:
            value = await get_project_secret_value(
                db,
                project_id,
                secret_provider,
                key_name,
                touch=False,
            )
        except Exception:
            value = None
        if value:
            return value, provider_tag, api_url
    return None, "openai", None


def _default_model_for(provider: str) -> str:
    """Cheap default model per provider. The trap-refresh path is
    cost-sensitive (runs unattended) so we never spin up GPT-4-class
    models unless the user has explicitly configured something else."""
    return {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
        "deepseek": "deepseek-chat",
    }.get(provider, "gpt-4o-mini")


# ─────────────────────────────────────────────────────────────────────
# Public entry points.
# ─────────────────────────────────────────────────────────────────────


async def refresh_traps_for_project(
    db: AsyncSession,
    *,
    project_id: int,
    count: int | None = None,
    simulate: bool = False,
    source_drift_check_id: int | None = None,
    generator: TrapGenerator | None = None,
) -> dict[str, Any]:
    """Generate ``count`` fresh traps and persist them to the drift
    review queue. Returns a summary payload suitable for the API
    response::

        {
            "project_id": int,
            "generated": int,
            "clusters_targeted": [reason_code, ...],
            "simulated": bool,
            "row_ids": [int, ...],
        }

    Raises ``ValueError("project_not_found")`` when the project is
    missing. ``ValueError("recipe_required")`` when the project has
    no recipe selected — we can't shape a trap row without one.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError("project_not_found")
    selected = project.selected_recipe or {}
    if not selected.get("recipe_id"):
        raise ValueError("recipe_required")

    target = resolved_target_count(project, override=count)
    clusters = await _load_recent_clusters(db, project_id=project_id)

    # Pick the generator: explicit param wins, then simulate flag, then
    # LLM path (which falls back to simulate when no key is stored).
    gen: TrapGenerator
    if generator is not None:
        gen = generator
    elif simulate:
        gen = _simulate_traps
    else:
        gen = _llm_generated_traps

    rows = await gen(db, project, clusters, target)

    row_ids: list[int] = []
    for row in rows[:target]:
        entity = GoldDriftReviewQueueRow(
            project_id=project_id,
            source_drift_check_id=source_drift_check_id,
            cluster_reason_code=row.get("cluster_reason_code"),
            cluster_signature=row.get("cluster_signature"),
            payload=row.get("payload") or {},
            source_confidence="rough",
            status=GoldDriftQueueStatus.PENDING,
        )
        db.add(entity)
        await db.flush()
        row_ids.append(entity.id)

    return {
        "project_id": project_id,
        "generated": len(row_ids),
        "clusters_targeted": [c.reason_code for c in clusters],
        "simulated": gen is _simulate_traps,
        "row_ids": row_ids,
    }


async def list_review_queue(
    db: AsyncSession,
    *,
    project_id: int,
    status: GoldDriftQueueStatus | None = GoldDriftQueueStatus.PENDING,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Newest-first list of queue rows for the project. Filters by
    ``status`` (default ``PENDING``) so the UI's "triage me" list is
    cheap to render. Pass ``status=None`` to see every row (admin
    audit path)."""
    query = select(GoldDriftReviewQueueRow).where(
        GoldDriftReviewQueueRow.project_id == project_id
    )
    if status is not None:
        query = query.where(GoldDriftReviewQueueRow.status == status)
    query = query.order_by(GoldDriftReviewQueueRow.created_at.desc()).limit(
        max(1, min(500, int(limit)))
    )
    rows = (await db.execute(query)).scalars().all()
    return [
        {
            "id": row.id,
            "project_id": row.project_id,
            "source_drift_check_id": row.source_drift_check_id,
            "cluster_reason_code": row.cluster_reason_code,
            "cluster_signature": row.cluster_signature,
            "payload": row.payload or {},
            "status": row.status.value,
            "source_confidence": row.source_confidence,
            "triage_note": row.triage_note,
            "created_at": row.created_at.isoformat(),
            "triaged_at": row.triaged_at.isoformat() if row.triaged_at else None,
        }
        for row in rows
    ]


async def triage_queue_row(
    db: AsyncSession,
    *,
    project_id: int,
    row_id: int,
    accept: bool,
    note: str | None = None,
) -> GoldDriftReviewQueueRow:
    """Mark a queue row accepted or rejected. Accepted rows are also
    appended to the project's gold_test JSONL via
    ``_append_to_gold_test`` so the trap lands where the user expects.

    Raises ``ValueError`` when the row is missing, belongs to a
    different project, or is already triaged. The endpoint translates
    to 404 / 409.
    """
    row = await db.get(GoldDriftReviewQueueRow, row_id)
    if row is None or row.project_id != project_id:
        raise ValueError("queue_row_not_found")
    if row.status is not GoldDriftQueueStatus.PENDING:
        raise ValueError("queue_row_already_triaged")

    row.status = (
        GoldDriftQueueStatus.ACCEPTED if accept
        else GoldDriftQueueStatus.REJECTED
    )
    row.triage_note = note
    row.triaged_at = datetime.now(timezone.utc)
    await db.flush()

    if accept:
        await _append_to_gold_test(db, project_id=project_id, payload=row.payload or {})
    return row


async def _append_to_gold_test(
    db: AsyncSession,
    *,
    project_id: int,
    payload: dict[str, Any],
) -> None:
    """Append the triage-accepted row to the project's gold_test
    JSONL on disk. Best-effort: if the dataset isn't materialized
    yet (fresh project without a gold_test file) we record the row
    via the dataset model so the next gold-set materialisation
    picks it up.

    Implementation is intentionally minimal — we don't bump the
    Dataset.record_count or rewrite metadata; the gold_set endpoint
    already handles those concerns when the user formally imports
    the row. This keeps the queue's "accept" affordance lossless
    without coupling it to the GoldSet workbench plumbing.
    """
    import json as _json
    from pathlib import Path

    from app.models.dataset import Dataset, DatasetType
    from sqlalchemy import select as _select

    result = await db.execute(
        _select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.GOLD_TEST,
        ).limit(1)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None or not dataset.file_path:
        # Project has no gold_test yet — the row will be picked up
        # by the next gold-set materialisation if the user wires
        # one up. The queue row stays accepted so the audit log is
        # honest about what the user decided.
        return
    target = Path(dataset.file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = _json.dumps(payload, ensure_ascii=False)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    # Refresh the record_count so the gold-set workbench reflects
    # the new row count on the next read.
    dataset.record_count = (dataset.record_count or 0) + 1
    await db.flush()
