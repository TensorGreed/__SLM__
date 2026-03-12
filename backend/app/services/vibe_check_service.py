"""Vibe-check timeline service for qualitative prompt snapshots during training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.playground_service import run_playground_chat

VIBE_CHECK_CONFIG_VERSION = "slm.vibe_check.config/v1"
VIBE_CHECK_TIMELINE_VERSION = "slm.vibe_check.timeline/v1"

_DEFAULT_PROMPTS: tuple[str, ...] = (
    "Answer in one sentence: what is this model currently optimized to do?",
    "Format this as JSON with keys summary and confidence: this platform now supports export.",
    "Rewrite this customer support reply to sound calm and precise: we are looking into it.",
    "Extract entities from: John Doe signed contract C-204 on January 5, 2026.",
    "Give a concise risk checklist for deploying an LLM-backed API in production.",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off", ""}:
            return False
    return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int, *, min_value: int, max_value: int) -> int:
    try:
        token = int(float(value))
    except (TypeError, ValueError):
        token = default
    return max(min_value, min(max_value, token))


def _normalize_prompt_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return list(_DEFAULT_PROMPTS)
    rows: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = str(item or "").strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        rows.append(token[:6000])
        seen.add(key)
        if len(rows) >= 5:
            break
    return rows if rows else list(_DEFAULT_PROMPTS)


def _project_training_dir(project_id: int) -> Path:
    path = settings.DATA_DIR / "projects" / str(project_id) / "training"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _project_vibe_config_path(project_id: int) -> Path:
    return _project_training_dir(project_id) / "vibe_check_config.json"


def _timeline_path(output_dir: Path) -> Path:
    return output_dir / "vibe_check_timeline.json"


def _default_config() -> dict[str, Any]:
    return {
        "config_version": VIBE_CHECK_CONFIG_VERSION,
        "enabled": True,
        "interval_steps": 50,
        "prompts": list(_DEFAULT_PROMPTS),
        "provider": "mock",
        "model_name": "microsoft/phi-2",
        "api_url": "",
        "temperature": 0.2,
        "max_tokens": 220,
        "updated_at": _utcnow_iso(),
    }


def _normalize_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    merged = _default_config()
    merged.update(dict(raw or {}))
    merged["config_version"] = VIBE_CHECK_CONFIG_VERSION
    merged["enabled"] = _coerce_bool(merged.get("enabled"), True)
    merged["interval_steps"] = _coerce_int(
        merged.get("interval_steps"),
        50,
        min_value=10,
        max_value=1000,
    )
    merged["prompts"] = _normalize_prompt_list(merged.get("prompts"))
    merged["provider"] = str(merged.get("provider") or "mock").strip() or "mock"
    merged["model_name"] = str(merged.get("model_name") or "microsoft/phi-2").strip() or "microsoft/phi-2"
    merged["api_url"] = str(merged.get("api_url") or "").strip()
    merged["temperature"] = max(0.0, min(2.0, _coerce_float(merged.get("temperature"), 0.2)))
    merged["max_tokens"] = _coerce_int(merged.get("max_tokens"), 220, min_value=32, max_value=4096)
    merged["updated_at"] = str(merged.get("updated_at") or _utcnow_iso())
    return merged


def _extract_experiment_overrides(experiment_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(experiment_config, dict):
        return {}
    mapping = {
        "vibe_check_enabled": "enabled",
        "vibe_check_interval_steps": "interval_steps",
        "vibe_check_prompts": "prompts",
        "vibe_check_provider": "provider",
        "vibe_check_model_name": "model_name",
        "vibe_check_api_url": "api_url",
        "vibe_check_temperature": "temperature",
        "vibe_check_max_tokens": "max_tokens",
    }
    out: dict[str, Any] = {}
    for source_key, target_key in mapping.items():
        if source_key in experiment_config:
            out[target_key] = experiment_config.get(source_key)
    return out


def load_project_vibe_check_config(
    project_id: int,
    *,
    experiment_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load project vibe-check config with optional experiment-level overrides."""
    path = _project_vibe_config_path(project_id)
    file_payload: dict[str, Any] = {}
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                file_payload = parsed
        except Exception:
            file_payload = {}
    merged = _normalize_config(file_payload)
    if isinstance(experiment_config, dict):
        merged.update(_extract_experiment_overrides(experiment_config))
        merged = _normalize_config(merged)
    return merged


def save_project_vibe_check_config(
    project_id: int,
    updates: dict[str, Any] | None,
) -> dict[str, Any]:
    """Persist project-level vibe-check config."""
    current = load_project_vibe_check_config(project_id)
    payload = dict(current)
    if isinstance(updates, dict):
        for key in ("enabled", "interval_steps", "prompts", "provider", "model_name", "api_url", "temperature", "max_tokens"):
            if key in updates:
                payload[key] = updates.get(key)
    payload["updated_at"] = _utcnow_iso()
    normalized = _normalize_config(payload)
    path = _project_vibe_config_path(project_id)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def _load_timeline_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": VIBE_CHECK_TIMELINE_VERSION,
            "created_at": _utcnow_iso(),
            "updated_at": _utcnow_iso(),
            "snapshot_count": 0,
            "snapshots": [],
        }
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    snapshots = parsed.get("snapshots")
    if not isinstance(snapshots, list):
        snapshots = []
    parsed["schema"] = VIBE_CHECK_TIMELINE_VERSION
    parsed["snapshots"] = snapshots
    parsed.setdefault("created_at", _utcnow_iso())
    parsed["updated_at"] = _utcnow_iso()
    parsed["snapshot_count"] = len(snapshots)
    return parsed


def load_vibe_check_timeline(output_dir: Path, *, limit: int = 200) -> dict[str, Any]:
    """Load vibe-check timeline snapshots for an experiment output directory."""
    safe_limit = max(1, min(int(limit), 2000))
    payload = _load_timeline_payload(_timeline_path(output_dir))
    snapshots = [item for item in list(payload.get("snapshots") or []) if isinstance(item, dict)]
    snapshots.sort(key=lambda item: int(item.get("step") or 0))
    if len(snapshots) > safe_limit:
        snapshots = snapshots[-safe_limit:]
    payload["snapshots"] = snapshots
    payload["snapshot_count"] = len(snapshots)
    payload["latest_snapshot"] = snapshots[-1] if snapshots else None
    return payload


def _fallback_vibe_reply(prompt: str, *, step: int, total_steps: int, train_loss: float | None) -> str:
    progress = max(0.0, min(1.0, step / max(1, total_steps)))
    phase = "bootstrap"
    if progress >= 0.85:
        phase = "stabilizing"
    elif progress >= 0.55:
        phase = "refining"
    elif progress >= 0.25:
        phase = "learning"
    loss_hint = f"train_loss={train_loss:.4f}" if isinstance(train_loss, float) else "train_loss=n/a"
    return (
        f"[vibe:{phase}] Step {step}/{total_steps} ({progress * 100:.1f}%). "
        f"{loss_hint}. Prompt focus: {prompt[:140]}"
    )


async def capture_vibe_check_snapshot(
    *,
    project_id: int,
    experiment_id: int,
    output_dir: Path,
    step: int,
    total_steps: int,
    base_model: str,
    epoch: float | None = None,
    train_loss: float | None = None,
    eval_loss: float | None = None,
    experiment_config: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generate and persist one vibe-check snapshot."""
    safe_step = max(1, int(step))
    safe_total_steps = max(1, int(total_steps))
    cfg = load_project_vibe_check_config(project_id, experiment_config=experiment_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_file = _timeline_path(output_dir)

    if not bool(cfg.get("enabled")):
        return {
            "enabled": False,
            "timeline_path": str(timeline_file),
            "snapshot": None,
            "snapshot_count": 0,
            "config": cfg,
        }

    prompts = _normalize_prompt_list(cfg.get("prompts"))
    provider = str(cfg.get("provider") or "mock")
    model_name = str(cfg.get("model_name") or base_model or "microsoft/phi-2").strip() or "microsoft/phi-2"
    api_url = str(cfg.get("api_url") or "").strip() or None
    temperature = float(cfg.get("temperature") or 0.2)
    max_tokens = int(cfg.get("max_tokens") or 220)
    progress = max(0.0, min(1.0, safe_step / max(1, safe_total_steps)))

    outputs: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts, start=1):
        user_prompt = (
            f"{prompt}\n\n"
            f"[VIBE_CONTEXT step={safe_step}/{safe_total_steps} progress={progress:.3f} "
            f"train_loss={train_loss if isinstance(train_loss, float) else 'n/a'} "
            f"eval_loss={eval_loss if isinstance(eval_loss, float) else 'n/a'}]"
        )
        reply = ""
        latency_ms = None
        error_text = ""
        try:
            result = await run_playground_chat(
                provider=provider,
                model_name=model_name,
                messages=[{"role": "user", "content": user_prompt}],
                api_url=api_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=(
                    "You are a training vibe-check probe. Return concise output that reflects the model's "
                    "current style and quality at this training step."
                ),
            )
            reply = str(result.get("reply") or "").strip()
            latency_ms = result.get("latency_ms")
        except Exception as exc:
            error_text = str(exc)
            reply = _fallback_vibe_reply(prompt, step=safe_step, total_steps=safe_total_steps, train_loss=train_loss)

        outputs.append(
            {
                "prompt_id": f"p{idx}",
                "prompt": prompt,
                "reply": reply,
                "provider": provider,
                "model_name": model_name,
                "latency_ms": latency_ms,
                "error": error_text or None,
            }
        )

    snapshot = {
        "step": safe_step,
        "epoch": float(epoch) if isinstance(epoch, (float, int)) else None,
        "progress": round(progress, 4),
        "train_loss": float(train_loss) if isinstance(train_loss, (float, int)) else None,
        "eval_loss": float(eval_loss) if isinstance(eval_loss, (float, int)) else None,
        "created_at": _utcnow_iso(),
        "outputs": outputs,
    }

    timeline = _load_timeline_payload(timeline_file)
    snapshots = [item for item in list(timeline.get("snapshots") or []) if isinstance(item, dict)]
    snapshots = [item for item in snapshots if int(item.get("step") or 0) != safe_step]
    snapshots.append(snapshot)
    snapshots.sort(key=lambda item: int(item.get("step") or 0))
    timeline["schema"] = VIBE_CHECK_TIMELINE_VERSION
    timeline["updated_at"] = _utcnow_iso()
    timeline["snapshot_count"] = len(snapshots)
    timeline["snapshots"] = snapshots
    timeline_file.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "enabled": True,
        "timeline_path": str(timeline_file),
        "snapshot": snapshot,
        "snapshot_count": len(snapshots),
        "latest_step": int(snapshots[-1].get("step") or safe_step) if snapshots else safe_step,
        "config": cfg,
    }

