"""Judge-model selection for a project (``runtime_config["eval_judge"]``).

The judge scores free-text answers (and passage retrieval) in evaluations and
filters teacher answers. ``resolve_answer_judge`` picks it: env
``EVAL_JUDGE_BACKEND`` > this setting > the first available local backend >
a configured cloud key. These routes let the UI show and change the setting,
with the billing of each option (local / metered API credits / subscription).
"""

import os
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.judge_call_guard import MAX_WAIT_CAP_SECONDS
from app.models.project import Project

router = APIRouter(prefix="/projects/{project_id}/eval-judge", tags=["Evaluation"])

_CLOUD_PROVIDERS = ("anthropic", "openai", "deepseek")
_CLOUD_LABELS = {"anthropic": "Anthropic", "openai": "OpenAI", "deepseek": "DeepSeek"}


class JudgeResilienceUpdate(BaseModel):
    """What happens when the judge is unreachable (``judge_call_guard``). Unset fields keep the
    install default (``JUDGE_BREAKER_ENABLED`` / ``JUDGE_OUTAGE_MODE`` / ``JUDGE_OUTAGE_MAX_WAIT_SECONDS``)."""

    breaker: bool | None = None
    mode: Literal["stop", "wait"] | None = None
    # Capped at half the task queue's visibility timeout (``judge_call_guard.max_wait_cap_seconds``).
    max_wait_seconds: float | None = Field(default=None, ge=0, le=MAX_WAIT_CAP_SECONDS)


class EvalJudgeUpdate(BaseModel):
    enabled: bool = True
    backend: str | None = None
    # None = keep the stored block; {} = back to the install defaults.
    resilience: JudgeResilienceUpdate | None = None


def _billing_for_class(cls: Any) -> str:
    declared = getattr(cls, "billing", None)
    if declared in {"local", "metered", "subscription"}:
        return str(declared)
    if getattr(cls, "name", "") == "teacher":
        from app.config import settings
        from app.services.llm_usage import is_local_url

        return "local" if is_local_url(settings.TEACHER_MODEL_API_URL) else "metered"
    return "local"


async def _options(db: AsyncSession, project_id: int, auto_billing: str = "local") -> list[dict[str, Any]]:
    from app.services.answer_judge_service import _cloud_config
    from app.services.synth_backends import BACKEND_REGISTRY

    options: list[dict[str, Any]] = [
        {"value": "", "label": "Automatic (a local model first, then a configured cloud key)", "billing": auto_billing, "available": True},
    ]
    for cls in BACKEND_REGISTRY:
        try:
            if not cls.is_available():
                continue
            value = cls().describe()
        except Exception:  # noqa: BLE001 — a broken backend is simply not offered
            continue
        billing = _billing_for_class(cls)
        label = {"command": "Local CLI (subscription)"}.get(cls.name, cls.name.capitalize())
        options.append({"value": value, "label": label, "billing": billing, "available": True})
    for provider in _CLOUD_PROVIDERS:
        cfg = await _cloud_config(db, project_id, provider)
        options.append({
            "value": provider, "label": f"{_CLOUD_LABELS[provider]} (API credits)",
            "billing": "metered", "available": cfg is not None,
        })
    return options


async def _resolved(db: AsyncSession, project: Project) -> dict[str, Any] | None:
    from app.services.answer_judge_service import resolve_answer_judge

    try:
        judge = await resolve_answer_judge(db, int(project.id), project)
    except Exception:  # noqa: BLE001 — a status read never fails
        return None
    if judge is None:
        return None
    head = judge.label.split(":", 1)[0]
    billing = "subscription" if head == "command" else "local" if judge.local else "metered"
    return {"describe": judge.label, "local": bool(judge.local), "billing": billing}


async def _trust(db: AsyncSession, project: Project, resolved: dict[str, Any] | None) -> dict[str, Any] | None:
    """Calibration trust (V4) of the judge in use: ``{verdict, rubric | None}``; None when no judge is
    reachable or on any error (a status read never fails)."""
    if not resolved:
        return None
    import asyncio

    from app.services import eval_judge_calibration_service as ecal

    try:
        live = await ecal.live_result_ids(db, int(project.id))
        return await asyncio.to_thread(ecal.trust_summary, int(project.id), resolved["describe"], project.runtime_config, live)
    except Exception:  # noqa: BLE001
        return None


async def _auto_billing(db: AsyncSession, project_id: int) -> str:
    """What "Automatic" resolves to when nothing is pinned: the first available
    backend that may be auto-picked (its billing), else a configured cloud key
    (metered), else local."""
    from app.services.answer_judge_service import _cloud_config
    from app.services.synth_backends import BACKEND_REGISTRY

    for cls in BACKEND_REGISTRY:
        if getattr(cls, "auto_pick", True) is False:
            continue
        try:
            if cls.is_available():
                return _billing_for_class(cls)
        except Exception:  # noqa: BLE001
            continue
    return "metered" if await _cloud_config(db, project_id, None) is not None else "local"


async def _payload(db: AsyncSession, project: Project) -> dict[str, Any]:
    rc = project.runtime_config if isinstance(project.runtime_config, dict) else {}
    cfg = rc.get("eval_judge") if isinstance(rc.get("eval_judge"), dict) else {}
    backend = str(cfg.get("backend") or "").strip() or None
    resolved = await _resolved(db, project)
    from app.services.judge_call_guard import max_wait_cap_seconds, normalize_project_resilience, resolve_resilience

    effective = resolve_resilience(rc)
    return {
        # When the judge is unreachable: the project's stored choices (``stored``) and what applies.
        "resilience": {
            "stored": normalize_project_resilience(cfg.get("resilience")),
            "breaker": effective.breaker,
            "mode": effective.mode,
            "max_wait_seconds": effective.max_wait_seconds,
            "consecutive": effective.consecutive,
            # The longest wait a project may choose (half the task queue's visibility timeout).
            "max_wait_cap_seconds": max_wait_cap_seconds(),
        },
        "trust": await _trust(db, project, resolved),
        "enabled": cfg.get("enabled") is not False,
        "backend": backend,
        "env_override": (os.getenv("EVAL_JUDGE_BACKEND") or "").strip() or None,
        "resolved": resolved,
        "options": await _options(db, int(project.id), await _auto_billing(db, int(project.id))),
    }


async def _project_or_404(db: AsyncSession, project_id: int) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"Project {project_id} not found")
    return project


@router.get("")
async def get_eval_judge(project_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """The project's judge setting, what it resolves to now, and the choices."""
    return await _payload(db, await _project_or_404(db, project_id))


@router.put("")
async def put_eval_judge(project_id: int, body: EvalJudgeUpdate, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Set the judge: ``backend`` null/"" = automatic; a registry backend's pin
    (``ollama:…``, ``command``, ``teacher``…) or a cloud provider
    (``anthropic`` / ``openai`` / ``deepseek``, optionally ``:<model>``)."""
    from app.services.synth_backends import BACKEND_REGISTRY

    project = await _project_or_404(db, project_id)
    backend = (body.backend or "").strip() or None
    if backend is not None:
        head = backend.split(":", 1)[0].lower()
        known = {cls.name for cls in BACKEND_REGISTRY} | set(_CLOUD_PROVIDERS)
        if head not in known:
            raise HTTPException(422, f"Unknown judge backend {backend!r}. Use one of: {', '.join(sorted(known))}.")
        from app.services.synth_backends.command import validate_pin

        try:
            validate_pin(backend)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    from app.services.judge_call_guard import normalize_project_resilience

    cfg = dict(project.runtime_config or {})
    # Merge, never overwrite: other keys of the block (the resilience settings) survive a judge change.
    block = dict(cfg.get("eval_judge") or {}) if isinstance(cfg.get("eval_judge"), dict) else {}
    block.update({"enabled": bool(body.enabled), "backend": backend})
    if body.resilience is not None:
        resilience = normalize_project_resilience(body.resilience.model_dump(exclude_none=True))
        if resilience:
            block["resilience"] = resilience
        else:
            block.pop("resilience", None)
    cfg["eval_judge"] = block
    project.runtime_config = cfg
    await db.commit()
    await db.refresh(project)
    return await _payload(db, project)
