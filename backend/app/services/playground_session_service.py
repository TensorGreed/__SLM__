"""Persistence and model-discovery helpers for chat playground."""

from __future__ import annotations

from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.artifact import ArtifactRecord, ArtifactStatus
from app.models.experiment import Experiment
from app.models.playground import PlaygroundSession
from app.models.project import Project
from app.models.registry import ModelRegistryEntry


ALLOWED_TRANSCRIPT_ROLES = {"system", "user", "assistant"}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _normalize_transcript(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = _coerce_text(item.get("role")).lower()
        content = _coerce_text(item.get("content"))
        if role not in ALLOWED_TRANSCRIPT_ROLES:
            continue
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _derive_title(title: str | None, transcript: list[dict[str, str]]) -> str:
    supplied = _coerce_text(title)
    if supplied:
        return supplied[:255]
    for item in transcript:
        if item.get("role") == "user":
            raw = _coerce_text(item.get("content"))
            if raw:
                compact = " ".join(raw.split())
                return compact[:80] or "Untitled Session"
    return "Untitled Session"


def _last_message_preview(transcript: list[dict[str, str]]) -> str:
    for item in reversed(transcript):
        role = _coerce_text(item.get("role")).lower()
        if role not in {"user", "assistant"}:
            continue
        content = _coerce_text(item.get("content"))
        if content:
            compact = " ".join(content.split())
            return compact[:120]
    return ""


def _serialize_session_common(session: PlaygroundSession) -> tuple[list[dict[str, str]], dict[str, Any]]:
    transcript = _normalize_transcript(list(session.transcript or []))
    payload: dict[str, Any] = {
        "id": session.id,
        "project_id": session.project_id,
        "title": _coerce_text(session.title) or "Untitled Session",
        "provider": _coerce_text(session.provider) or "mock",
        "model_name": _coerce_text(session.model_name),
        "api_url": _coerce_text(session.api_url) or None,
        "system_prompt": _coerce_text(session.system_prompt),
        "temperature": float(session.temperature),
        "max_tokens": int(session.max_tokens),
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }
    return transcript, payload


def serialize_playground_session_summary(session: PlaygroundSession) -> dict[str, Any]:
    transcript, payload = _serialize_session_common(session)
    payload.update(
        {
            "message_count": len(transcript),
            "last_message_preview": _last_message_preview(transcript),
        }
    )
    return payload


def serialize_playground_session_detail(session: PlaygroundSession) -> dict[str, Any]:
    transcript, payload = _serialize_session_common(session)
    payload.update(
        {
            "message_count": len(transcript),
            "messages": transcript,
        }
    )
    return payload


async def get_playground_session(
    db: AsyncSession,
    *,
    project_id: int,
    session_id: int,
) -> PlaygroundSession | None:
    result = await db.execute(
        select(PlaygroundSession).where(
            PlaygroundSession.project_id == project_id,
            PlaygroundSession.id == session_id,
        )
    )
    return result.scalar_one_or_none()


async def list_playground_sessions(
    db: AsyncSession,
    *,
    project_id: int,
    limit: int = 30,
) -> list[PlaygroundSession]:
    capped_limit = max(1, min(int(limit or 30), 200))
    result = await db.execute(
        select(PlaygroundSession)
        .where(PlaygroundSession.project_id == project_id)
        .order_by(desc(PlaygroundSession.updated_at), desc(PlaygroundSession.id))
        .limit(capped_limit)
    )
    return list(result.scalars().all())


async def save_playground_session_transcript(
    db: AsyncSession,
    *,
    project_id: int,
    session_id: int | None,
    title: str | None,
    provider: str,
    model_name: str,
    api_url: str | None,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
    transcript: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> PlaygroundSession:
    session: PlaygroundSession | None = None
    if session_id is not None:
        session = await get_playground_session(db, project_id=project_id, session_id=int(session_id))
        if session is None:
            raise ValueError(f"Playground session {session_id} not found in project {project_id}")

    normalized_transcript = _normalize_transcript(transcript)
    if session is None:
        session = PlaygroundSession(project_id=project_id)
        db.add(session)

    session.title = _derive_title(title, normalized_transcript)
    session.provider = _coerce_text(provider).lower() or "mock"
    session.model_name = _coerce_text(model_name)
    session.api_url = _coerce_text(api_url) or None
    session.system_prompt = _coerce_text(system_prompt)
    session.temperature = float(temperature)
    session.max_tokens = max(16, min(int(max_tokens), 4096))
    session.transcript = normalized_transcript
    if metadata is not None:
        session.metadata_ = dict(metadata)

    await db.flush()
    await db.refresh(session)
    return session


async def delete_playground_session(
    db: AsyncSession,
    *,
    project_id: int,
    session_id: int,
) -> bool:
    session = await get_playground_session(db, project_id=project_id, session_id=session_id)
    if session is None:
        return False
    await db.delete(session)
    await db.flush()
    return True


async def list_playground_model_options(
    db: AsyncSession,
    *,
    project_id: int,
    limit: int = 50,
) -> dict[str, Any]:
    project_row = await db.execute(select(Project).where(Project.id == project_id))
    project = project_row.scalar_one_or_none()
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    options: list[dict[str, Any]] = []
    seen: set[str] = set()

    def push(
        *,
        model_name: str | None,
        label: str,
        source: str,
        priority: int,
        source_ref: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        cleaned = _coerce_text(model_name)
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        options.append(
            {
                "model_name": cleaned,
                "label": label,
                "source": source,
                "priority": priority,
                "source_ref": _coerce_text(source_ref) or None,
                "meta": dict(meta or {}),
            }
        )

    if _coerce_text(project.base_model_name):
        push(
            model_name=project.base_model_name,
            label=f"Project Base Model • {project.base_model_name}",
            source="project_base",
            priority=0,
            source_ref=str(project.id),
        )

    registry_rows = await db.execute(
        select(ModelRegistryEntry)
        .where(ModelRegistryEntry.project_id == project_id)
        .order_by(desc(ModelRegistryEntry.updated_at), desc(ModelRegistryEntry.id))
        .limit(80)
    )
    for row in registry_rows.scalars().all():
        registry_model = _coerce_text(row.name) or _coerce_text(row.artifact_path)
        if not registry_model:
            continue
        push(
            model_name=registry_model,
            label=f"Registry • {row.name}@{row.version} ({row.stage.value})",
            source="registry",
            priority=1,
            source_ref=str(row.id),
            meta={
                "artifact_path": _coerce_text(row.artifact_path) or None,
                "stage": row.stage.value,
                "version": _coerce_text(row.version),
            },
        )
        if _coerce_text(row.artifact_path):
            push(
                model_name=row.artifact_path,
                label=f"Registry Artifact Path • {row.name}@{row.version}",
                source="registry_artifact_path",
                priority=2,
                source_ref=str(row.id),
                meta={"stage": row.stage.value, "version": _coerce_text(row.version)},
            )

    experiment_rows = await db.execute(
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .order_by(desc(Experiment.created_at), desc(Experiment.id))
        .limit(120)
    )
    for row in experiment_rows.scalars().all():
        push(
            model_name=row.base_model,
            label=f"Experiment Base • {row.name} (#{row.id})",
            source="experiment_base",
            priority=3,
            source_ref=str(row.id),
            meta={"experiment_status": row.status.value},
        )
        output_dir = _coerce_text(row.output_dir)
        if output_dir:
            candidate_path = output_dir.rstrip("/") + "/model"
            push(
                model_name=candidate_path,
                label=f"Experiment Artifact Path • {row.name} (#{row.id})",
                source="experiment_artifact_path",
                priority=4,
                source_ref=str(row.id),
                meta={"experiment_status": row.status.value},
            )

    artifact_rows = await db.execute(
        select(ArtifactRecord)
        .where(
            ArtifactRecord.project_id == project_id,
            ArtifactRecord.status == ArtifactStatus.MATERIALIZED,
        )
        .order_by(desc(ArtifactRecord.created_at), desc(ArtifactRecord.id))
        .limit(200)
    )
    for row in artifact_rows.scalars().all():
        metadata = row.metadata_ or {}
        model_ref = (
            _coerce_text(metadata.get("model_name"))
            or _coerce_text(metadata.get("model_id"))
            or _coerce_text(metadata.get("base_model"))
            or _coerce_text(metadata.get("name"))
            or _coerce_text(row.uri)
            or _coerce_text(row.artifact_key)
        )
        push(
            model_name=model_ref,
            label=f"Artifact • {row.artifact_key} v{row.version}",
            source="artifact_registry",
            priority=5,
            source_ref=str(row.id),
            meta={
                "artifact_key": row.artifact_key,
                "artifact_type": row.artifact_type,
                "version": row.version,
                "uri": _coerce_text(row.uri) or None,
            },
        )

    options.sort(key=lambda item: (int(item.get("priority", 99)), str(item.get("label", "")).lower()))
    capped_limit = max(1, min(int(limit or 50), 200))
    selected = options[:capped_limit]
    return {
        "project_id": project_id,
        "default_model_name": _coerce_text(project.base_model_name) or "microsoft/phi-2",
        "count": len(selected),
        "models": selected,
    }
