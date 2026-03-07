"""Prepared dataset contract validation for task-specific training readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings


SUPPORTED_DATASET_SHAPES = (
    "text",
    "qa_pair",
    "chat_messages",
    "seq2seq_pair",
    "classification_label",
)

SUPPORTED_TASK_TYPES = ("causal_lm", "seq2seq", "classification")

TASK_SHAPE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "causal_lm": ("text", "qa_pair", "chat_messages"),
    "seq2seq": ("seq2seq_pair", "qa_pair"),
    "classification": ("classification_label", "qa_pair"),
}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _pick_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in record:
            text = _coerce_text(record.get(key))
            if text:
                return text
    return ""


def _messages_as_text(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        messages = record.get("conversations")
    if not isinstance(messages, list):
        return ""

    parts: list[str] = []
    has_user = False
    has_assistant = False
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = _coerce_text(item.get("role") or item.get("from")).lower()
        content = _coerce_text(item.get("content") or item.get("value") or item.get("text"))
        if not content:
            continue
        if role in {"user", "human"}:
            has_user = True
        if role in {"assistant", "gpt", "model"}:
            has_assistant = True
        parts.append(content)
    if not (has_user and has_assistant):
        return ""
    return "\n".join(parts).strip()


def _row_shapes(row: dict[str, Any]) -> set[str]:
    shapes: set[str] = set()

    text = _pick_text(row, ("text", "content", "body", "document", "passage"))
    question = _pick_text(row, ("question", "prompt", "query", "instruction", "input", "source_text"))
    answer = _pick_text(row, ("answer", "response", "output", "target", "completion", "target_text"))
    source_text = _pick_text(row, ("source_text", "source", "input", "question", "prompt", "instruction", "text"))
    target_text = _pick_text(row, ("target_text", "target", "answer", "output", "completion", "response"))
    label = _pick_text(row, ("label", "class", "category", "output_label", "target", "answer", "output"))
    chat_text = _messages_as_text(row)

    if text:
        shapes.add("text")
    if question and answer:
        shapes.add("qa_pair")
    if chat_text:
        shapes.add("chat_messages")
    if source_text and target_text:
        shapes.add("seq2seq_pair")
    if (text or question or source_text) and label:
        shapes.add("classification_label")
    return shapes


def _row_task_compatible(row: dict[str, Any], task_type: str) -> bool:
    shapes = _row_shapes(row)
    accepted = TASK_SHAPE_REQUIREMENTS.get(task_type, TASK_SHAPE_REQUIREMENTS["causal_lm"])
    return any(shape in shapes for shape in accepted)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sample_jsonl(path: Path, *, sample_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= sample_size:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _suggest_adapter(task_type: str, shape_counts: dict[str, int]) -> str:
    task = str(task_type).strip().lower()
    if task == "classification":
        return "classification-label"
    if task == "seq2seq":
        return "seq2seq-pair"
    if int(shape_counts.get("qa_pair", 0)) > int(shape_counts.get("text", 0)):
        return "qa-pair"
    return "default-canonical"


def analyze_prepared_dataset_contract(
    *,
    project_id: int,
    task_type: str,
    sample_size: int = 400,
    min_coverage: float = 0.9,
) -> dict[str, Any]:
    """Validate prepared train split contract against requested task type."""
    task = str(task_type or "causal_lm").strip().lower()
    if task not in SUPPORTED_TASK_TYPES:
        task = "causal_lm"

    prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    train_file = prepared_dir / "train.jsonl"
    manifest_path = prepared_dir / "manifest.json"
    manifest = _load_json(manifest_path)

    errors: list[str] = []
    warnings: list[str] = []
    hints: list[str] = []

    if not train_file.exists():
        errors.append(
            f"Prepared training dataset missing at {train_file}. Run Dataset Prep split before training."
        )
        return {
            "ok": False,
            "task_type": task,
            "required_shapes": list(TASK_SHAPE_REQUIREMENTS.get(task, ())),
            "sampled_rows": 0,
            "compatible_rows": 0,
            "coverage": 0.0,
            "shape_counts": {shape: 0 for shape in SUPPORTED_DATASET_SHAPES},
            "errors": errors,
            "warnings": warnings,
            "hints": hints,
            "manifest_adapter_id": str(manifest.get("adapter_id") or "").strip() or None,
            "manifest_field_mapping": manifest.get("field_mapping") if isinstance(manifest.get("field_mapping"), dict) else {},
        }

    rows = _sample_jsonl(train_file, sample_size=max(20, min(5000, int(sample_size or 400))))
    if not rows:
        errors.append(
            f"Prepared training split at {train_file} has no readable JSONL rows."
        )
        return {
            "ok": False,
            "task_type": task,
            "required_shapes": list(TASK_SHAPE_REQUIREMENTS.get(task, ())),
            "sampled_rows": 0,
            "compatible_rows": 0,
            "coverage": 0.0,
            "shape_counts": {shape: 0 for shape in SUPPORTED_DATASET_SHAPES},
            "errors": errors,
            "warnings": warnings,
            "hints": hints,
            "manifest_adapter_id": str(manifest.get("adapter_id") or "").strip() or None,
            "manifest_field_mapping": manifest.get("field_mapping") if isinstance(manifest.get("field_mapping"), dict) else {},
        }

    shape_counts = {shape: 0 for shape in SUPPORTED_DATASET_SHAPES}
    compatible_rows = 0
    for row in rows:
        shapes = _row_shapes(row)
        for shape in shapes:
            if shape in shape_counts:
                shape_counts[shape] += 1
        if _row_task_compatible(row, task):
            compatible_rows += 1

    sampled_rows = len(rows)
    coverage = float(compatible_rows / sampled_rows) if sampled_rows > 0 else 0.0
    suggested_adapter = _suggest_adapter(task, shape_counts)
    required_shapes = list(TASK_SHAPE_REQUIREMENTS.get(task, ()))
    manifest_adapter_id = str(manifest.get("adapter_id") or "").strip() or None
    manifest_mapping = manifest.get("field_mapping") if isinstance(manifest.get("field_mapping"), dict) else {}

    if coverage < float(min_coverage):
        shape_snapshot = ", ".join(
            f"{shape}={shape_counts.get(shape, 0)}"
            for shape in required_shapes
        )
        errors.append(
            (
                f"Dataset contract mismatch for task_type={task}: "
                f"compatible coverage {coverage:.1%} is below required {float(min_coverage):.0%}. "
                f"Observed task-shape counts in sample: {shape_snapshot}."
            )
        )
        hints.append(
            f"Run Dataset Prep split with adapter '{suggested_adapter}' and re-run training preflight."
        )
        hints.append(
            "Use Adapter Lab auto-detect, then save adapter preset and re-split."
        )
        if not manifest_mapping:
            hints.append(
                "If source fields are custom, provide field_mapping JSON (for example {'question':'prompt','answer':'completion'})."
            )
    elif coverage < 0.98:
        warnings.append(
            (
                f"Dataset contract coverage for task_type={task} is {coverage:.1%}. "
                "Some rows may be dropped at train runtime."
            )
        )

    if manifest_adapter_id and manifest_adapter_id == "default-canonical" and task in {"seq2seq", "classification"}:
        warnings.append(
            (
                f"manifest adapter_id={manifest_adapter_id} for task_type={task}; "
                f"consider adapter '{suggested_adapter}' for stricter mapping."
            )
        )

    return {
        "ok": len(errors) == 0,
        "task_type": task,
        "required_shapes": required_shapes,
        "sampled_rows": sampled_rows,
        "compatible_rows": compatible_rows,
        "coverage": round(coverage, 6),
        "shape_counts": shape_counts,
        "suggested_adapter_id": suggested_adapter,
        "errors": errors,
        "warnings": warnings,
        "hints": hints,
        "manifest_adapter_id": manifest_adapter_id,
        "manifest_field_mapping": manifest_mapping,
    }
