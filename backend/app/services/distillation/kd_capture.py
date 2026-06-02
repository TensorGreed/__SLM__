"""Read + align the slice-1 teacher-capture artifact for offline KD (slice 2).

Pure helpers (no torch, no DB) so they're unit-testable with a fake tokenizer:

- ``load_teacher_capture`` — parse the ``teacher_capture.jsonl`` produced by
  ``distillation/teacher_capture.py``.
- ``verify_capture_artifact`` — a pre-training readiness gate (mirrors the
  repo's other incident-driven gates): refuse a distillation run before the GPU
  spins up if the capture file is missing / empty / logit-free, with an
  actionable message.
- ``build_teacher_target_topk`` — turn one captured row's per-position top-k
  ``(token_string, logprob)`` pairs into padded ``(ids, logprobs)`` arrays over
  the *student* vocab, via an injected ``token_to_id`` map.

**Same-tokenizer assumption.** Offline KD aligns the teacher's per-token
distribution with the student's tokens position-by-position, which is exact only
when teacher and student share a tokenizer (the common KD setup — e.g. distil a
larger SmolLM/Qwen into the 135M sibling). Teacher tokens that don't resolve to
a student id are dropped; if most drop, the soft signal weakens — the gate
surfaces a low map-rate so the caller can warn. Cross-tokenizer alignment is a
follow-up (slice 2b/3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, TypedDict

# Ragged top-k padding sentinel. Must match scatter_topk_to_logits' pad_id so
# the trainer drops padded slots when densifying the teacher distribution.
TEACHER_PAD_ID = -1


class CaptureGate(TypedDict):
    ok: bool
    message: str
    row_count: int
    rows_with_logits: int


def load_teacher_capture(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``teacher_capture.jsonl`` file into row dicts. Blank and
    unparseable lines are skipped; a missing file yields an empty list."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8", errors="replace") as f:
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


def _row_has_logits(row: dict[str, Any]) -> bool:
    logits = row.get("teacher_logits")
    return isinstance(logits, list) and len(logits) > 0


def verify_capture_artifact(path: str | Path) -> CaptureGate:
    """Pre-training gate: is this capture artifact usable for offline KD?

    ``ok`` is True only when the file exists, parses, and has at least one row
    carrying a non-empty ``teacher_logits`` list. The message is written for a
    user who just picked a ``kd_*`` recipe without running capture first.
    """
    p = Path(path)
    if not p.exists():
        return CaptureGate(
            ok=False,
            message=(
                f"No teacher-capture artifact at {p}. Run teacher logit capture "
                "(POST /api/projects/{id}/distillation/capture) before training "
                "in distillation mode."
            ),
            row_count=0,
            rows_with_logits=0,
        )
    rows = load_teacher_capture(p)
    with_logits = sum(1 for r in rows if _row_has_logits(r))
    if not rows:
        return CaptureGate(
            ok=False,
            message=f"Teacher-capture artifact at {p} is empty.",
            row_count=0,
            rows_with_logits=0,
        )
    if with_logits == 0:
        return CaptureGate(
            ok=False,
            message=(
                f"Teacher-capture artifact at {p} has {len(rows)} rows but none "
                "carry teacher_logits. Re-run capture against a teacher that "
                "returns logprobs (top_logprobs)."
            ),
            row_count=len(rows),
            rows_with_logits=0,
        )
    return CaptureGate(
        ok=True,
        message=(
            f"{with_logits}/{len(rows)} rows carry teacher logits — ready for "
            "distillation."
        ),
        row_count=len(rows),
        rows_with_logits=with_logits,
    )


def _iter_positions(row: dict[str, Any]) -> list[list[list[Any]]]:
    """Return per-position top-k lists ``[[token, logprob], ...]`` for a row.
    Tolerant of malformed entries (skips them)."""
    positions: list[list[list[Any]]] = []
    for entry in row.get("teacher_logits") or []:
        if not isinstance(entry, dict):
            positions.append([])
            continue
        topk = entry.get("top_k")
        pairs: list[list[Any]] = []
        if isinstance(topk, list):
            for pair in topk:
                if (
                    isinstance(pair, (list, tuple))
                    and len(pair) == 2
                    and isinstance(pair[0], str)
                    and isinstance(pair[1], (int, float))
                ):
                    pairs.append([pair[0], float(pair[1])])
        positions.append(pairs)
    return positions


def _extract_prompt_text(row: dict[str, Any]) -> str:
    """The teacher's input prompt for a captured row.

    β-fix for KD: prefer ``wrapped_prompt`` when present — it
    carries the exact handler-built string the teacher saw (and
    that the eval handler will rebuild at held-out time). Falls
    back to the raw-field walk for legacy captures (pre-β-KD)
    that don't have the wrapped field, so existing
    teacher_capture.jsonl files still build records.
    """
    wrapped = row.get("wrapped_prompt")
    if isinstance(wrapped, str) and wrapped.strip():
        return wrapped.strip()
    for key in ("question", "prompt", "instruction", "input_text", "text", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_teacher_target_topk(
    row: dict[str, Any],
    token_to_id: Callable[[str], int | None],
    *,
    top_k: int,
    pad_id: int = TEACHER_PAD_ID,
    pad_logprob: float = 0.0,
) -> tuple[list[list[int]], list[list[float]], dict[str, int]]:
    """Align one captured row's per-position top-k onto student vocab ids.

    Returns ``(ids, logprobs, stats)`` where ``ids`` / ``logprobs`` are
    ``[num_positions, top_k]`` (padded with ``pad_id`` / ``pad_logprob``), and
    ``stats`` carries ``{"mapped", "dropped"}`` token counts so the caller can
    surface the teacher→student map rate. ``token_to_id`` returns ``None`` for
    tokens absent from the student vocab; those are dropped.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    ids_out: list[list[int]] = []
    logprobs_out: list[list[float]] = []
    mapped = 0
    dropped = 0

    for pairs in _iter_positions(row):
        row_ids: list[int] = []
        row_logprobs: list[float] = []
        for token_str, logprob in pairs:
            if len(row_ids) >= top_k:
                break
            student_id = token_to_id(token_str)
            if student_id is None:
                dropped += 1
                continue
            row_ids.append(int(student_id))
            row_logprobs.append(float(logprob))
            mapped += 1
        while len(row_ids) < top_k:
            row_ids.append(pad_id)
            row_logprobs.append(pad_logprob)
        ids_out.append(row_ids)
        logprobs_out.append(row_logprobs)

    return ids_out, logprobs_out, {"mapped": mapped, "dropped": dropped}


class OfflineKDRecord(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    teacher_topk_ids: list[list[int]]
    teacher_topk_logprobs: list[list[float]]


def build_offline_kd_records(
    capture_rows: list[dict[str, Any]],
    encode_fn: Callable[[str], list[int]],
    token_to_id: Callable[[str], int | None],
    *,
    top_k: int = 8,
    max_seq_length: int = 1024,
    ignore_index: int = -100,
    pad_id: int = TEACHER_PAD_ID,
    prompt_transform: Callable[[str], str] | None = None,
) -> tuple[list[OfflineKDRecord], dict[str, int]]:
    """Turn captured rows into causal-LM training records with aligned teacher
    targets, ready for ``OfflineKDCollator`` + ``OfflineDistillationTrainer``.

    Each record trains the student on ``prompt + teacher_completion``: prompt
    tokens are masked (``labels = ignore_index``), and each completion token
    position carries the teacher's top-k distribution for that token. Positions
    whose teacher data is missing (capture shorter than the re-tokenized
    completion) are masked out of the KD loss.

    ``encode_fn(text) -> list[int]`` and ``token_to_id(token) -> int | None``
    are injected (the student tokenizer) so this is testable with a fake vocab.
    The teacher position ↔ completion token alignment is exact only under the
    same-tokenizer assumption (see module docstring).

    ``prompt_transform`` closes the chat-template sub-gap of the KD β-fix.
    The capture-time β-fix (commit 3672f05) covered handlers whose
    ``wraps_own_prompt() == True`` by persisting a ``wrapped_prompt`` on the
    captured row. For the QA / language_modeling family (``wraps_own_prompt
    == False``), held-out eval applies the model's chat template at inference
    but the student previously trained on raw ``prompt + completion`` tokens.
    When provided, ``prompt_transform`` is applied to the row's prompt BEFORE
    ``encode_fn`` is called — typically a function that runs
    ``tokenizer.apply_chat_template`` so the student trains on the same
    chat-template-wrapped scaffold the eval will build. Rows that already
    carry a ``wrapped_prompt`` from capture-time wrapping skip the
    transform (already byte-aligned to a wraps-own-prompt handler).
    """
    records: list[OfflineKDRecord] = []
    stats = {
        "built": 0,
        "skipped": 0,
        "truncated": 0,
        "mapped": 0,
        "dropped": 0,
        "positions_without_teacher": 0,
    }

    for row in capture_rows:
        prompt = _extract_prompt_text(row)
        # Chat-template sub-gap fix: apply the transform only when
        # the row didn't already carry a handler-wrapped prompt from
        # capture time (the wraps_own_prompt branch — Classification,
        # Structured, RAG, Seq2Seq, VisionLanguage, AudioTranscript).
        # ``_extract_prompt_text`` already preferred ``wrapped_prompt``
        # when present, so we detect that by checking the raw field
        # directly rather than re-walking the alias list.
        has_capture_wrap = isinstance(row.get("wrapped_prompt"), str) and bool(
            row.get("wrapped_prompt", "").strip()
        )
        if prompt_transform is not None and not has_capture_wrap and prompt:
            try:
                transformed = prompt_transform(prompt)
            except Exception:
                # Transform shouldn't crash the whole build — fall
                # back to the untransformed prompt rather than the
                # entire record. Same defensive shape as the rest
                # of this builder's per-row error handling.
                transformed = prompt
            if isinstance(transformed, str) and transformed.strip():
                prompt = transformed
        completion = str(row.get("teacher_completion") or row.get("answer") or "").strip()
        pos_ids, pos_logprobs, mstats = build_teacher_target_topk(
            row, token_to_id, top_k=top_k, pad_id=pad_id
        )
        if not prompt or not completion or not pos_ids:
            stats["skipped"] += 1
            continue

        prompt_ids = list(encode_fn(prompt))
        completion_ids = list(encode_fn(completion))
        if not completion_ids:
            stats["skipped"] += 1
            continue

        pad_row_ids = [pad_id] * top_k
        pad_row_lp = [0.0] * top_k

        input_ids = prompt_ids + completion_ids
        labels = [ignore_index] * len(prompt_ids) + list(completion_ids)
        tk_ids: list[list[int]] = [list(pad_row_ids) for _ in prompt_ids]
        tk_lp: list[list[float]] = [list(pad_row_lp) for _ in prompt_ids]

        n_teacher = len(pos_ids)
        for j in range(len(completion_ids)):
            if j < n_teacher:
                tk_ids.append(pos_ids[j])
                tk_lp.append(pos_logprobs[j])
            else:
                # Completion re-tokenized longer than the captured positions —
                # no teacher signal here, so drop it from the KD loss.
                tk_ids.append(list(pad_row_ids))
                tk_lp.append(list(pad_row_lp))
                labels[len(prompt_ids) + j] = ignore_index
                stats["positions_without_teacher"] += 1

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
            tk_ids = tk_ids[:max_seq_length]
            tk_lp = tk_lp[:max_seq_length]
            stats["truncated"] += 1

        records.append(
            OfflineKDRecord(
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids),
                labels=labels,
                teacher_topk_ids=tk_ids,
                teacher_topk_logprobs=tk_lp,
            )
        )
        stats["built"] += 1
        stats["mapped"] += mstats["mapped"]
        stats["dropped"] += mstats["dropped"]

    return records, stats
