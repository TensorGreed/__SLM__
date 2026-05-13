"""Task-aware evaluation dispatcher (Phase 5.3.0).

Foundation for routing evaluation through per-task handlers. This phase
ships only the dispatcher + ``GenericHandler``, which preserves today's
behavior byte-for-byte. Future phases (5.3.1 classification, 5.3.3
seq2seq, …) register new handlers without touching this file.

The contract: ``task_profile`` is read **only** from
``prepared/manifest.json``. There is no row-shape sniffing — a seq2seq
dataset with few unique references can never be auto-mistaken for
classification. Missing tag → ``GenericHandler`` → identical behavior
to the pre-dispatcher pipeline.

Plan and per-phase user stories live in ``TASK_AWARE_EVAL_PLAN.md`` at
the repo root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from app.config import settings


# ── Data shapes ───────────────────────────────────────────────────────


@dataclass
class EvalContext:
    """Read-only context passed to every handler call.

    Carries everything a handler needs to build prompts or compute
    metrics without re-reading state. Handlers must not mutate it.
    """

    project_id: int
    experiment_id: int
    eval_type: str
    task_profile: str | None
    handler_id: str
    prepared_dir: Path
    dataset_name: str
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuiltPrompt:
    """One row's prompt + reference + auxiliary fields.

    ``extras`` carries handler-specific fields (image_path, context for
    RAG, candidate label set echoed for diagnostics, etc.) that the
    inference path or scorer may need. ``GenericHandler`` populates
    ``image_path`` and ``audio_path`` when present.
    """

    prompt: str
    reference: str
    extras: dict[str, Any] = field(default_factory=dict)

    def as_pair(self) -> dict[str, Any]:
        """Render as the legacy ``{prompt, reference, **extras}`` dict the
        existing inference path consumes."""

        pair: dict[str, Any] = {"prompt": self.prompt, "reference": self.reference}
        pair.update(self.extras)
        return pair


# ── Handler protocol ──────────────────────────────────────────────────


@runtime_checkable
class TaskHandler(Protocol):
    """Two-method interface every task handler implements."""

    profile_id: str

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        """Map dataset rows to prompt/reference pairs for inference."""

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        """Compute metric dict from predictions. Returned keys flow
        straight into ``EvalResult.metrics``."""


# ── GenericHandler — today's behavior, preserved verbatim ─────────────


class GenericHandler:
    """Fallback handler. Mirrors pre-5.3.0 behavior exactly.

    Delegates prompt extraction and scoring to the helpers that live in
    ``evaluation_service`` so the pre-dispatcher entry points (the
    ``/api/evaluation/run`` direct path, existing tests) score
    identically.
    """

    profile_id: str = "generic"

    def build_prompts(
        self,
        rows: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> list[BuiltPrompt]:
        from app.services.evaluation_service import (
            _extract_prompt_and_reference,
        )

        built: list[BuiltPrompt] = []
        for row in rows:
            prompt, reference = _extract_prompt_and_reference(row)
            extras: dict[str, Any] = {}
            image_path = str(row.get("image_path") or row.get("image") or "").strip()
            audio_path = str(row.get("audio_path") or row.get("audio") or "").strip()
            if image_path:
                extras["image_path"] = image_path
            if audio_path:
                extras["audio_path"] = audio_path
            built.append(BuiltPrompt(prompt=prompt, reference=reference, extras=extras))
        return built

    def score(
        self,
        predictions: list[dict[str, Any]],
        ctx: EvalContext,
    ) -> dict[str, Any]:
        from app.services.evaluation_service import (
            evaluate_safety_response,
            exact_match,
            f1_score,
        )

        eval_type = ctx.eval_type
        if eval_type == "exact_match":
            scores = [
                exact_match(p.get("prediction", ""), p.get("reference", ""))
                for p in predictions
            ]
            return {
                "exact_match": round(sum(scores) / len(scores), 4) if scores else 0,
                "total": len(scores),
                "correct": int(sum(scores)),
            }
        if eval_type == "f1":
            scores = [
                f1_score(p.get("prediction", ""), p.get("reference", ""))
                for p in predictions
            ]
            return {
                "f1": round(sum(scores) / len(scores), 4) if scores else 0,
                "total": len(scores),
            }
        if eval_type == "safety":
            results = [
                evaluate_safety_response(
                    p.get("response", ""), p.get("test_type", "unknown")
                )
                for p in predictions
            ]
            passed = sum(1 for r in results if r["passed"])
            return {
                "pass_rate": round(passed / len(results), 4) if results else 0,
                "total_tests": len(results),
                "passed": passed,
                "failed": len(results) - passed,
            }
        # llm_judge is handled by a separate code path; the dispatcher
        # doesn't get called for it. Return empty so callers can detect
        # an unknown eval_type and fall through.
        return {}


# ── Registry + dispatcher ─────────────────────────────────────────────


def _normalize_profile(value: Any) -> str:
    """Normalize a task_profile string for registry lookup. Empty / None
    becomes empty string, which dispatches to ``GenericHandler``."""

    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


# Maps normalized task_profile → handler factory. New handlers register
# themselves by appending to this dict (e.g. classification handler in
# Phase 5.3.1). Missing key falls through to ``GenericHandler``.
_HANDLER_FACTORIES: dict[str, Callable[[], TaskHandler]] = {}


def register_handler(profile: str, factory: Callable[[], TaskHandler]) -> None:
    """Register a handler factory for a given task profile.

    Intentionally tolerant: registering the same profile twice replaces
    the prior factory (useful for tests). The empty string is reserved
    for the GenericHandler fallback.
    """

    key = _normalize_profile(profile)
    if not key:
        raise ValueError("register_handler requires a non-empty profile id")
    _HANDLER_FACTORIES[key] = factory


def resolve_task_handler(task_profile: str | None) -> TaskHandler:
    """Return the handler matching ``task_profile``, or GenericHandler.

    The lookup is intentionally forgiving: unknown profiles, empty
    strings, malformed types all fall through to ``GenericHandler``. We
    log the fall-through at the call site, not here, so callers can
    decide what severity to attach.
    """

    key = _normalize_profile(task_profile)
    factory = _HANDLER_FACTORIES.get(key)
    if factory is None:
        return GenericHandler()
    try:
        return factory()
    except Exception:
        # A buggy handler factory must never break eval — fall back.
        return GenericHandler()


def list_registered_profiles() -> list[str]:
    """Diagnostic: which task profiles currently have a registered
    handler (excludes the implicit generic fallback)."""

    return sorted(_HANDLER_FACTORIES.keys())


# ── Manifest reading ──────────────────────────────────────────────────


def _project_prepared_dir(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "prepared"


def read_prepared_manifest(project_id: int) -> dict[str, Any]:
    """Load ``prepared/manifest.json`` for a project; empty dict on
    miss / parse failure. Never raises."""

    manifest_path = _project_prepared_dir(project_id) / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_task_profile_from_manifest(project_id: int) -> str | None:
    """Convenience: pull just the ``task_profile`` from the prepared
    manifest. Returns ``None`` if missing or unreadable."""

    manifest = read_prepared_manifest(project_id)
    value = manifest.get("task_profile")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def build_eval_context(
    *,
    project_id: int,
    experiment_id: int,
    eval_type: str,
    dataset_name: str,
) -> tuple[EvalContext, TaskHandler]:
    """One-shot helper: read manifest, resolve handler, return both.

    Callers in ``evaluation_service`` use this to keep the dispatch
    site short.
    """

    manifest = read_prepared_manifest(project_id)
    task_profile = read_task_profile_from_manifest(project_id)
    handler = resolve_task_handler(task_profile)
    ctx = EvalContext(
        project_id=project_id,
        experiment_id=experiment_id,
        eval_type=eval_type,
        task_profile=task_profile,
        handler_id=handler.profile_id,
        prepared_dir=_project_prepared_dir(project_id),
        dataset_name=dataset_name,
        manifest=manifest,
    )
    return ctx, handler


__all__ = [
    "BuiltPrompt",
    "EvalContext",
    "GenericHandler",
    "TaskHandler",
    "build_eval_context",
    "list_registered_profiles",
    "read_prepared_manifest",
    "read_task_profile_from_manifest",
    "register_handler",
    "resolve_task_handler",
]
