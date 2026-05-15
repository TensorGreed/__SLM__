"""Pre-training data-shape gate.

Refuses to start an SFT training run when the prepared train.jsonl is in
domain-pretrain shape — i.e. carries only the source/input text and no
``answer`` / ``completion`` / ``output`` / ``response`` target. Without
this gate, the trainer silently falls back to causal-LM continuation on
the input field and produces a model that's never seen the task target
schema. Eval then reports F1≈0 because the model's predictions are in
the wrong shape, and the user has burned hours of GPU time finding out.

Background
~~~~~~~~~~
This was added after a 10.5h Qwen-PII-V2 SFT run completed with eval
F1=0.0%. The training set (``ai4privacy/pii-masking-200k`` chunks) had
only a ``text`` field; the project's adapter contract resolved
``target_fields=[answer, completion, output, response]`` but every row
in the prepared file was missing every one. The trainer ran on garbage,
loss went down slightly (1.08 vs 1.06 — mostly noise), and the model
emitted ai4privacy-style mask-token continuations rather than the
project's ``{entities: [...]}`` JSON schema.

The check costs O(sample_n) JSON parses + four key membership tests per
row, so even sampling a few hundred rows is sub-second.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TARGET_FIELDS: tuple[str, ...] = (
    "answer",
    "completion",
    "output",
    "response",
)

# Training modes that *require* a target field. DOMAIN_PRETRAIN is
# legitimately a causal-LM continuation on raw text; the alignment
# modes (DPO / ORPO) carry their own preference-pair contract which
# the alignment-dataset pipeline already validates, so we skip them
# here rather than double-gate.
TARGET_REQUIRED_MODES: frozenset[str] = frozenset({"sft", "instruction_sft"})


def _coerce_str_field(value: Any) -> bool:
    """Treat the field as present iff it has any non-empty payload —
    a stringified empty string, an empty list, or None all count as
    missing. Conservative on purpose: a real target string is what the
    trainer needs to make a meaningful loss."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def sample_train_rows(
    train_file: Path, *, sample_n: int = 256
) -> tuple[int, list[dict[str, Any]]]:
    """Read up to ``sample_n`` rows from a JSONL file. Returns
    ``(rows_scanned, rows)`` where ``rows_scanned`` is the number of
    non-empty lines we inspected (rows with malformed JSON count toward
    scanned but are excluded from the returned list).
    """
    rows: list[dict[str, Any]] = []
    scanned = 0
    with train_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            scanned += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
            if len(rows) >= sample_n:
                break
    return scanned, rows


def count_rows_with_any_field(
    rows: Iterable[dict[str, Any]], fields: Iterable[str]
) -> int:
    """Count rows that carry at least one of ``fields`` with a
    non-empty value."""
    field_list = list(fields)
    matches = 0
    for row in rows:
        for key in field_list:
            if _coerce_str_field(row.get(key)):
                matches += 1
                break
    return matches


def verify_training_data_has_targets(
    train_file: Path,
    *,
    training_mode: str,
    target_fields: Iterable[str] | None = None,
    sample_n: int = 256,
) -> dict[str, Any]:
    """Inspect ``train_file`` and decide whether SFT training can
    safely start.

    Returns a dict shaped::

        {
            "ok": bool,
            "gate_applied": bool,
            "training_mode": str,
            "target_fields": list[str],
            "sample_size": int,
            "rows_with_target": int,
            "ratio": float,
            "message": str | None,
        }

    ``gate_applied=False`` when training_mode falls outside
    :data:`TARGET_REQUIRED_MODES` — the report is still returned so
    callers can log it without branching. ``ok=False`` only when the
    gate applied *and* zero sampled rows have any target field.
    """
    mode = (training_mode or "").strip().lower()
    fields = (
        list(target_fields)
        if target_fields is not None
        else list(DEFAULT_TARGET_FIELDS)
    )

    report: dict[str, Any] = {
        "ok": True,
        "gate_applied": False,
        "training_mode": mode,
        "target_fields": fields,
        "sample_size": 0,
        "rows_with_target": 0,
        "ratio": 0.0,
        "message": None,
    }

    if mode not in TARGET_REQUIRED_MODES:
        return report

    report["gate_applied"] = True

    if not train_file.exists():
        report["ok"] = False
        report["message"] = (
            f"Training data not found at {train_file}. Run dataset import + "
            f"data prep first."
        )
        return report

    _scanned, rows = sample_train_rows(train_file, sample_n=sample_n)
    report["sample_size"] = len(rows)

    if not rows:
        report["ok"] = False
        report["message"] = (
            f"{train_file} is empty or every line failed to parse. Re-run "
            f"data prep and verify train.jsonl is valid JSONL."
        )
        return report

    matches = count_rows_with_any_field(rows, fields)
    report["rows_with_target"] = matches
    report["ratio"] = matches / len(rows)

    if matches == 0:
        report["ok"] = False
        report["message"] = (
            f"Training data has zero rows with any target field "
            f"{sorted(fields)} across {len(rows)} sampled. The trainer "
            f"would optimize causal-LM continuation on the input field "
            f"and never see the task target schema — eval F1 will be 0. "
            f"Likely cause: dataset_import mapper produced "
            f"domain-pretrain-shape rows (text-only) instead of "
            f"{{question, answer}} pairs. Re-run dataset import with a "
            f"mapper that emits a target field, or switch training_mode "
            f"to domain_pretrain if pretraining is what you want."
        )

    return report


__all__ = [
    "DEFAULT_TARGET_FIELDS",
    "TARGET_REQUIRED_MODES",
    "count_rows_with_any_field",
    "sample_train_rows",
    "verify_training_data_has_targets",
]
