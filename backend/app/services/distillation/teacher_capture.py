"""Teacher logit capture for knowledge distillation (Track 1, Epic A, slice 1).

For each row of a source dataset, call a strong *teacher* model and record
its top-k token log-probabilities alongside the row's original payload. A
student model later trains against these soft targets (KD slice 2), which is
the single biggest quality lever a small model has.

Design notes that matter:

* **Reuses the existing teacher config.** The OpenAI-compatible
  ``TEACHER_MODEL_API_URL`` / ``TEACHER_MODEL_API_KEY`` that already power the
  synthetic generator drive capture too — no new provider plumbing.
* **Dedicated artifact, not the synthetic dataset.** Captured rows carry a
  heavyweight ``teacher_logits`` field that the normal SFT prep readers don't
  expect, so they land in ``projects/<id>/distillation/teacher_capture.jsonl``
  rather than polluting ``synthetic.jsonl``. Slice 2's KD trainer reads this
  file explicitly. (Deviation from the original Story 2.1 brief, which said
  "synthetic dataset"; keeping the artifacts separate avoids breaking SFT.)
* **Background-task pattern.** A teacher call *per row* over a few hundred rows
  on a slow local model would blow the dev proxy's 10-minute ceiling, so the
  API returns a ``task_id`` in milliseconds and the frontend polls a status
  endpoint. Mirrors ``cleaning_service`` exactly (in-memory registry +
  ``asyncio.create_task`` + its own DB session).
* **Per-row isolation.** A row whose teacher call fails lands in
  ``chunk_errors`` but the rest of the batch still runs — one bad row costs one
  row, not the whole capture.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypedDict
from uuid import uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory
from app.models.dataset import Dataset
from app.models.run_event import STAGE_INGESTION
from app.services import run_event_service
from app.services.synthetic_service import (
    DEFAULT_TEACHER_SYSTEM_PROMPT,
    _coerce_completion_content,
)


# OpenAI's chat-completions API caps ``top_logprobs`` at 20.
_MAX_TOP_K = 20
# Capture is about the teacher's distribution over its own answer tokens;
# a modest generation budget is plenty and keeps each call cheap.
_DEFAULT_MAX_TOKENS = 256


class CaptureResult(TypedDict):
    dataset_id: int
    teacher_model: str
    top_k: int
    captured_count: int
    skipped_count: int
    chunk_errors: list[dict]
    written_path: str | None


# ── Teacher call (the mockable seam) ───────────────────────────────────


def _parse_logprobs(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize an OpenAI-compatible logprobs payload into our row shape.

    Returns a list of ``{"token": str, "top_k": [[token, logprob], ...]}`` —
    one entry per generated token. Tolerant of providers that omit the field
    (returns an empty list rather than raising).
    """
    try:
        content = data["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return []
    if not isinstance(content, list):
        return []

    captured: list[dict[str, Any]] = []
    for entry in content:
        if not isinstance(entry, dict):
            continue
        token = entry.get("token")
        if not isinstance(token, str):
            continue
        alts: list[list[Any]] = []
        for alt in entry.get("top_logprobs") or []:
            if not isinstance(alt, dict):
                continue
            alt_token = alt.get("token")
            alt_logprob = alt.get("logprob")
            if isinstance(alt_token, str) and isinstance(alt_logprob, (int, float)):
                alts.append([alt_token, float(alt_logprob)])
        captured.append({"token": token, "top_k": alts})
    return captured


async def call_teacher_with_logprobs(
    prompt: str,
    *,
    system_prompt: str = DEFAULT_TEACHER_SYSTEM_PROMPT,
    api_url: str = "",
    api_key: str = "",
    model_name: str = "llama3",
    top_k: int = 10,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call an OpenAI-compatible teacher endpoint asking for top-k logprobs.

    Returns ``{"content", "teacher_logits", "model"}``. ``temperature``
    defaults to 0.0 — distillation wants the teacher's confident
    distribution, not sampling noise.

    This is the seam tests patch to avoid hitting a real model.
    """
    url = api_url or settings.TEACHER_MODEL_API_URL
    key = api_key or settings.TEACHER_MODEL_API_KEY
    if not url:
        raise ValueError(
            "Teacher model API URL not configured. Set TEACHER_MODEL_API_URL in .env"
        )

    bounded_k = max(1, min(int(top_k), _MAX_TOP_K))

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": bounded_k,
    }

    timeout_seconds = max(30.0, float(settings.TEACHER_MODEL_TIMEOUT_SECONDS or 600.0))
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    content = _coerce_completion_content(
        data.get("choices", [{}])[0].get("message", {}).get("content", "")
    )
    return {
        "content": content,
        "teacher_logits": _parse_logprobs(data),
        "model": data.get("model", model_name),
    }


# ── Source-row helpers ─────────────────────────────────────────────────


def _distillation_dir(project_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "distillation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_dataset_rows(dataset: Dataset) -> list[dict[str, Any]]:
    """Read a dataset's JSONL file into row dicts. Non-dict / unparseable
    lines are skipped — capture only makes sense over structured rows."""
    path_str = dataset.file_path or ""
    if not path_str:
        return []
    path = Path(path_str)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            token = line.strip()
            if not token:
                continue
            try:
                obj = json.loads(token)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _extract_prompt_text(row: dict[str, Any]) -> str:
    """The input the teacher should respond to (so we capture its
    distribution over the answer). Walks the canonical SFT shapes."""
    for key in ("question", "prompt", "instruction", "input_text", "text", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Fall back to the first non-empty string field.
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_handler_wrapped_prompts(
    project_id: int,
    rows: list[dict[str, Any]],
) -> tuple[list[str | None] | None, dict[str, Any]]:
    """Distillation β-fix — resolve the project's eval handler and
    pre-wrap each row's prompt with the handler's eval-time
    instruction template BEFORE sending to the teacher.

    Returns ``(wrapped_prompts, provenance)`` where:
      * ``wrapped_prompts`` is a list aligned with ``rows`` — each
        entry is the handler-wrapped prompt for that row, or
        ``None`` when the handler couldn't wrap (row missing
        required fields, etc.). The full list is ``None`` when
        the project has no eval handler that wraps its own
        prompt (caller falls back to raw extraction).
      * ``provenance`` carries ``task_profile`` + ``handler_id``
        so the captured row can record how it was wrapped (for
        later diagnostics + the KD record builder's preference
        check).

    Closes the β-shape gap audit OQ surfaced: pre-this-fix the
    teacher capture wrote raw row prompts; the student trained
    on `raw_prompt + teacher_completion`; held-out eval handler
    wrapped with `"Classify the following text. …\\nLabel:"`
    (or `"Extract …\\nOutput:"`, etc.). Train and eval prompts
    were different strings — same shape as the SQLi pre-β
    collapse. After this fix, the teacher sees the same wrapped
    prompt the student will see at eval, so the captured
    distribution is over the answer tokens the eval handler will
    elicit, and the KD record builder can pin train+eval to a
    single shared scaffold.
    """
    from app.services.eval_task_handler_service import (
        EvalContext,
        read_prepared_manifest,
        read_task_profile_from_manifest,
        resolve_task_handler,
    )

    task_profile = read_task_profile_from_manifest(project_id)
    if not task_profile:
        return None, {}
    handler = resolve_task_handler(task_profile)
    # Only handlers that build their own complete prompts at eval
    # are gap-bearing — non-wrapping handlers (QA / language_modeling)
    # have a separate chat-template gap addressed elsewhere.
    if not (
        hasattr(handler, "wraps_own_prompt")
        and bool(handler.wraps_own_prompt())
    ):
        return None, {}

    manifest = read_prepared_manifest(project_id)
    ctx = EvalContext(
        project_id=project_id,
        experiment_id=0,
        eval_type="exact_match",
        task_profile=task_profile,
        handler_id=getattr(handler, "profile_id", task_profile),
        prepared_dir=_distillation_dir(project_id),
        dataset_name="teacher_capture",
        manifest=dict(manifest or {}),
    )

    # Handlers expose ``build_prompts(rows, ctx) -> list[BuiltPrompt]``
    # as the unified interface across task shapes. Per-row failures
    # (missing required fields, etc.) raise; we catch and emit
    # ``None`` for that row so the caller can fall back to raw
    # extraction without killing the whole capture.
    try:
        built = handler.build_prompts(rows, ctx)
    except Exception:
        return None, {
            "task_profile": task_profile,
            "handler_id": getattr(handler, "profile_id", task_profile),
            "wrap_error": "build_prompts_failed",
        }

    wrapped: list[str | None] = []
    for entry in built:
        prompt_value = getattr(entry, "prompt", None)
        if isinstance(prompt_value, str) and prompt_value.strip():
            wrapped.append(prompt_value)
        else:
            wrapped.append(None)
    # Pad/truncate to match rows length — defensive against a
    # handler returning a different-length list.
    while len(wrapped) < len(rows):
        wrapped.append(None)
    wrapped = wrapped[: len(rows)]
    return wrapped, {
        "task_profile": task_profile,
        "handler_id": getattr(handler, "profile_id", task_profile),
    }


# ── Capture orchestrator ───────────────────────────────────────────────


async def capture_teacher_outputs(
    db: AsyncSession,
    project_id: int,
    dataset_id: int,
    *,
    top_k: int = 10,
    teacher_model_name: str | None = None,
    limit: int | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> CaptureResult:
    """Capture teacher top-k logprobs for every row of ``dataset_id``.

    Writes captured rows (original payload + ``teacher_logits`` + provenance)
    to ``projects/<id>/distillation/teacher_capture.jsonl`` and emits a
    ``distillation_teacher_capture`` RunEvent. Per-row failures are recorded
    in ``chunk_errors`` and don't abort the batch.

    Raises ``ValueError`` (stable codes) for caller mistakes: ``dataset_not_found``,
    ``teacher_not_configured``, ``dataset_empty``.
    """
    result = await db.execute(
        select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.project_id == project_id,
        )
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise ValueError("dataset_not_found")

    teacher_model = (teacher_model_name or "").strip() or "llama3"
    # Fail fast on a missing teacher rather than N identical row failures.
    if not (settings.TEACHER_MODEL_API_URL or "").strip():
        raise ValueError("teacher_not_configured")

    rows = _load_dataset_rows(dataset)
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    total = len(rows)
    if total == 0:
        raise ValueError("dataset_empty")

    out_path = _distillation_dir(project_id) / "teacher_capture.jsonl"
    captured_count = 0
    skipped_count = 0
    chunk_errors: list[dict] = []
    bounded_k = max(1, min(int(top_k), _MAX_TOP_K))

    # β-fix for KD — resolve handler wraps so each row's teacher
    # input is the same string the eval handler will build at
    # held-out time. ``wrapped_prompts`` is None when the project's
    # handler doesn't wrap (QA / language_modeling); rows then
    # fall back to raw extraction (no β-shape gap there since the
    # handler also doesn't wrap at eval).
    wrapped_prompts, wrap_provenance = _resolve_handler_wrapped_prompts(
        project_id, rows,
    )

    if on_progress is not None:
        on_progress(0, total, 0)

    with open(out_path, "w", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            wrapped_prompt: str | None = None
            if wrapped_prompts is not None and index < len(wrapped_prompts):
                wrapped_prompt = wrapped_prompts[index]
            prompt_text = wrapped_prompt or _extract_prompt_text(row)
            if not prompt_text:
                skipped_count += 1
                chunk_errors.append(
                    {"row_index": index, "error": "no_prompt_text"}
                )
                if on_progress is not None:
                    on_progress(index + 1, total, captured_count)
                continue
            try:
                teacher = await call_teacher_with_logprobs(
                    prompt_text,
                    model_name=teacher_model,
                    top_k=bounded_k,
                )
            except Exception as exc:  # noqa: BLE001 — isolate per-row failures
                skipped_count += 1
                chunk_errors.append({"row_index": index, "error": str(exc)})
                if on_progress is not None:
                    on_progress(index + 1, total, captured_count)
                continue

            captured_row = {
                **row,
                "teacher_logits": teacher["teacher_logits"],
                "teacher_completion": teacher["content"],
                "teacher_model": teacher["model"],
                "source": "teacher_capture",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "status": "accepted",
            }
            # β-fix provenance: persist the wrapped prompt + which
            # handler wrapped it so the KD record builder can train
            # the student on the exact same scaffold and so a future
            # audit can verify capture vs eval byte-for-byte.
            if wrapped_prompt:
                captured_row["wrapped_prompt"] = wrapped_prompt
                if wrap_provenance:
                    captured_row.setdefault(
                        "task_profile", wrap_provenance.get("task_profile"),
                    )
                    captured_row.setdefault(
                        "handler_id", wrap_provenance.get("handler_id"),
                    )
            captured_row.setdefault("id", index + 1)
            out.write(json.dumps(captured_row) + "\n")
            captured_count += 1
            if on_progress is not None:
                on_progress(index + 1, total, captured_count)

    written_path = str(out_path) if captured_count > 0 else None

    # Record the capture on the dataset for traceability without minting a
    # new DatasetType (KD capture is an attached artifact, not a split).
    dataset.metadata_ = {
        **(dataset.metadata_ or {}),
        "teacher_capture": {
            "written_path": str(out_path),
            "captured_count": captured_count,
            "teacher_model": teacher_model,
            "top_k": bounded_k,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    await db.flush()

    # Best-effort audit event (info severity — never breaks the capture).
    try:
        await run_event_service.emit_event(
            db,
            project_id=project_id,
            run_id=f"distill-capture-{dataset_id}",
            stage=STAGE_INGESTION,
            summary=(
                f"Captured teacher logits for {captured_count}/{total} rows "
                f"of dataset {dataset_id} ({teacher_model})"
            ),
            reason_code="distillation_teacher_capture",
            payload={
                "dataset_id": dataset_id,
                "teacher_model": teacher_model,
                "top_k": bounded_k,
                "captured_count": captured_count,
                "skipped_count": skipped_count,
                "chunk_error_count": len(chunk_errors),
                "written_path": written_path,
            },
        )
    except Exception as exc:  # pragma: no cover - observability must not break capture
        print(f"[distillation] emit_event failed project={project_id} err={exc!r}", flush=True)

    return CaptureResult(
        dataset_id=dataset_id,
        teacher_model=teacher_model,
        top_k=bounded_k,
        captured_count=captured_count,
        skipped_count=skipped_count,
        chunk_errors=chunk_errors,
        written_path=written_path,
    )


# ── Background-task plumbing (mirrors cleaning_service) ─────────────────


@dataclass
class DistillationCaptureTask:
    task_id: str
    project_id: int
    dataset_id: int
    top_k: int
    teacher_model_name: str | None
    limit: int | None

    status: str = "pending"  # pending | running | completed | failed
    total: int = 0
    completed: int = 0
    produced_count: int = 0
    written_path: str | None = None
    chunk_errors: list[dict] = field(default_factory=list)
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "top_k": self.top_k,
            "status": self.status,
            "total": self.total,
            "completed": self.completed,
            "produced_count": self.produced_count,
            "written_path": self.written_path,
            "chunk_errors": list(self.chunk_errors),
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
        }


_CAPTURE_TASKS: dict[str, DistillationCaptureTask] = {}
_CAPTURE_TASKS_LOCK = threading.Lock()
_MAX_TRACKED_TASKS: int = 64

# Hold strong refs to the live asyncio.Task objects so the GC can't collect a
# task whose only other reference is the local var in start_capture_task
# (the CLAUDE.md "runner caveat").
_RUNNING_TASKS: set[asyncio.Task] = set()


def _trim_finished_tasks() -> None:
    if len(_CAPTURE_TASKS) <= _MAX_TRACKED_TASKS:
        return
    finished = sorted(
        (t for t in _CAPTURE_TASKS.values() if t.finished_at is not None),
        key=lambda t: t.finished_at,  # type: ignore[arg-type]
    )
    overflow = len(_CAPTURE_TASKS) - _MAX_TRACKED_TASKS
    for task in finished[:overflow]:
        _CAPTURE_TASKS.pop(task.task_id, None)


async def _run_capture_task(task: DistillationCaptureTask) -> None:
    task.status = "running"
    task.updated_at = datetime.now(timezone.utc)

    def _progress(completed: int, total: int, produced: int) -> None:
        task.completed = completed
        task.total = total
        task.produced_count = produced
        task.updated_at = datetime.now(timezone.utc)

    try:
        async with async_session_factory() as db:
            result = await capture_teacher_outputs(
                db,
                task.project_id,
                task.dataset_id,
                top_k=task.top_k,
                teacher_model_name=task.teacher_model_name,
                limit=task.limit,
                on_progress=_progress,
            )
            await db.commit()
        task.written_path = result["written_path"]
        task.chunk_errors = list(result["chunk_errors"])
        task.produced_count = result["captured_count"]
        task.status = "completed"
    except Exception as exc:  # noqa: BLE001 — fatal (bad dataset / no teacher / DB)
        task.status = "failed"
        task.error = str(exc)
    finally:
        task.finished_at = datetime.now(timezone.utc)
        task.updated_at = task.finished_at


def start_capture_task(
    *,
    project_id: int,
    dataset_id: int,
    top_k: int = 10,
    teacher_model_name: str | None = None,
    limit: int | None = None,
) -> DistillationCaptureTask:
    """Register + launch a capture job; returns the record immediately so the
    API can hand back a ``task_id`` while the work runs on the event loop."""
    task = DistillationCaptureTask(
        task_id=f"distill-{uuid4().hex[:12]}",
        project_id=project_id,
        dataset_id=dataset_id,
        top_k=top_k,
        teacher_model_name=teacher_model_name,
        limit=limit,
    )
    with _CAPTURE_TASKS_LOCK:
        _CAPTURE_TASKS[task.task_id] = task
        _trim_finished_tasks()

    runner = asyncio.create_task(_run_capture_task(task))
    _RUNNING_TASKS.add(runner)
    runner.add_done_callback(_RUNNING_TASKS.discard)
    return task


def get_capture_task(task_id: str) -> DistillationCaptureTask | None:
    with _CAPTURE_TASKS_LOCK:
        return _CAPTURE_TASKS.get(task_id)


def get_capture_task_status(task_id: str) -> dict[str, Any] | None:
    task = get_capture_task(task_id)
    return task.to_dict() if task else None
