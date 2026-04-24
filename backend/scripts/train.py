#!/usr/bin/env python3
"""External training runtime for real SLM finetuning.

This script runs supervised finetuning against prepared JSONL datasets.
It includes:
- trainer backend abstraction (`hf_trainer`, optional `trl_sft`)
- native TRL pairwise alignment objective trainers (`DPOTrainer`, `ORPOTrainer`)
- task/data adapter contract (`causal_lm`, `seq2seq`, `classification`)
- runtime planner with CUDA OOM auto-retry
"""

from __future__ import annotations

import argparse
import inspect
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

SUPPORTED_TASK_TYPES = {"causal_lm", "seq2seq", "classification"}
SUPPORTED_TRAINER_BACKENDS = {"auto", "hf_trainer", "trl_sft"}

TRAINING_METRIC_PREFIX = "SLM_METRIC "
TRAINING_EVENT_PREFIX = "SLM_EVENT "


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external SLM finetuning workflow")
    parser.add_argument("--project", type=int, required=True, help="Project ID")
    parser.add_argument("--experiment", type=int, required=True, help="Experiment ID")
    parser.add_argument("--output", type=str, required=True, help="Experiment output directory")
    parser.add_argument("--base-model", type=str, required=True, help="HF base model id/path")
    parser.add_argument("--config", type=str, default="", help="Path to training config JSON")
    parser.add_argument("--train-file", type=str, default="", help="Path to train JSONL file")
    parser.add_argument("--val-file", type=str, default="", help="Path to validation JSONL file")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Root data directory")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional train row cap for dry runs")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="Optional eval row cap for dry runs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a JSON object: {path}")
    return payload


def _resolve_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    project_dir = data_dir / "projects" / str(args.project) / "prepared"
    train_file = (
        Path(args.train_file).expanduser().resolve()
        if args.train_file
        else project_dir / "train.jsonl"
    )
    val_file = (
        Path(args.val_file).expanduser().resolve()
        if args.val_file
        else project_dir / "val.jsonl"
    )
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_file}. "
            "Run dataset split before training."
        )
    return train_file, val_file if val_file.exists() else None


def _emit_runtime_event(event: str, payload: dict[str, Any] | None = None) -> None:
    body = {"event": event}
    if isinstance(payload, dict):
        body.update(payload)
    print(f"{TRAINING_EVENT_PREFIX}{json.dumps(body, ensure_ascii=False)}", flush=True)


def _coerce_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _coerce_float(value: Any, default: float, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if minimum is not None and parsed < minimum:
        return minimum
    return parsed


def _coerce_bool(value: Any, default: bool = False) -> bool:
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
    return bool(default)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _pick_first_text(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


_IMAGE_FIELD_CANDIDATES = ["image_path", "image", "image_url", "image_file", "path"]
_AUDIO_FIELD_CANDIDATES = ["audio_path", "audio", "audio_url", "audio_file", "path"]
_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".avif",
}
_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".webm",
    ".aiff",
}


def _extract_multimodal_paths(row: dict[str, Any]) -> tuple[str, str]:
    image_path = _pick_first_text(row, _IMAGE_FIELD_CANDIDATES)
    audio_path = _pick_first_text(row, _AUDIO_FIELD_CANDIDATES)
    if image_path and audio_path and image_path == audio_path:
        # Disambiguate shared "path" aliases using extension hints when possible.
        suffix = Path(image_path).suffix.strip().lower()
        if suffix in _AUDIO_EXTENSIONS:
            image_path = ""
        elif suffix in _IMAGE_EXTENSIONS:
            audio_path = ""
        else:
            # Avoid ambiguous shared aliases resolving both modalities to one token.
            audio_path = ""
    return image_path, audio_path


def _infer_input_modality(
    *,
    text: str,
    image_path: str,
    audio_path: str,
) -> str:
    if image_path and audio_path:
        return "multimodal"
    if image_path:
        return "vision_language"
    if audio_path:
        return "audio_text"
    token = str(text or "").strip().lower()
    has_image_marker = "<image:" in token
    has_audio_marker = "<audio:" in token
    if has_image_marker and has_audio_marker:
        return "multimodal"
    if has_image_marker:
        return "vision_language"
    if has_audio_marker:
        return "audio_text"
    return "text"


def _attach_multimodal_fields(row: dict[str, Any], mapped: dict[str, Any]) -> dict[str, Any]:
    payload = dict(mapped or {})
    image_path, audio_path = _extract_multimodal_paths(row)
    if image_path:
        payload["image_path"] = image_path
    if audio_path:
        payload["audio_path"] = audio_path
    payload["input_modality"] = _infer_input_modality(
        text=str(payload.get("text") or ""),
        image_path=image_path,
        audio_path=audio_path,
    )
    return payload


PREFERENCE_PROMPT_KEYS = [
    "prompt",
    "question",
    "instruction",
    "input",
    "query",
]
PREFERENCE_CHOSEN_KEYS = [
    "chosen",
    "preferred",
    "accepted",
    "better",
    "response_chosen",
    "answer_chosen",
    "completion_chosen",
]
PREFERENCE_REJECTED_KEYS = [
    "rejected",
    "dispreferred",
    "worse",
    "response_rejected",
    "answer_rejected",
    "completion_rejected",
]


def _adapt_record_to_preference(row: dict[str, Any]) -> dict[str, str]:
    prompt = _pick_first_text(row, PREFERENCE_PROMPT_KEYS)
    chosen = _pick_first_text(row, PREFERENCE_CHOSEN_KEYS)
    rejected = _pick_first_text(row, PREFERENCE_REJECTED_KEYS)
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def _is_valid_preference_row(row: dict[str, Any]) -> bool:
    prompt = str(row.get("prompt", "")).strip()
    chosen = str(row.get("chosen", "")).strip()
    rejected = str(row.get("rejected", "")).strip()
    if not prompt or not chosen or not rejected:
        return False
    return chosen != rejected


def _qa_to_chat_text(question: str, answer: str, template_name: str) -> str:
    q = question.strip()
    a = answer.strip()
    if not q or not a:
        return ""
    if template_name == "llama3":
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{a}<|eot_id|>"
        )
    if template_name == "chatml":
        return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
    if template_name == "zephyr":
        return f"<|user|>\n{q}</s>\n<|assistant|>\n{a}</s>"
    if template_name == "phi3":
        return f"<|user|>\n{q}<|end|>\n<|assistant|>\n{a}<|end|>"
    return f"User: {q}\nAssistant: {a}"


def _build_data_adapter_contract(task_type: str, chat_template: str) -> dict[str, Any]:
    task = task_type.strip().lower()
    if task not in SUPPORTED_TASK_TYPES:
        task = "causal_lm"
    if task == "seq2seq":
        return {
            "task_type": "seq2seq",
            "input_fields": ["source", "input", "question", "prompt", "instruction", "text", "content"],
            "target_fields": ["target", "answer", "output", "completion", "response", "label_text"],
            "render_mode": f"chat:{chat_template}",
        }
    if task == "classification":
        return {
            "task_type": "classification",
            "input_fields": ["text", "content", "input", "question", "prompt", "instruction"],
            "label_fields": [
                "label",
                "class",
                "category",
                "output_label",
                "target",
                "answer",
                "completion",
                "output",
            ],
            "render_mode": "label_instruction",
        }
    return {
        "task_type": "causal_lm",
        "input_fields": ["text", "content", "question", "prompt", "instruction"],
        "target_fields": ["answer", "completion", "output", "response"],
        "render_mode": f"chat:{chat_template}",
    }


def _adapt_record_to_text(
    row: dict[str, Any],
    contract: dict[str, Any],
    chat_template: str,
) -> dict[str, Any]:
    task_type = str(contract.get("task_type", "causal_lm"))

    if task_type == "seq2seq":
        source = _pick_first_text(row, list(contract.get("input_fields", [])))
        target = _pick_first_text(row, list(contract.get("target_fields", [])))
        if source and target:
            return _attach_multimodal_fields(row, {
                "text": _qa_to_chat_text(source, target, chat_template),
                "source_text": source,
                "target_text": target,
            })
        if source:
            return _attach_multimodal_fields(
                row,
                {"text": source, "source_text": source, "target_text": ""},
            )
        return _attach_multimodal_fields(
            row,
            {"text": "", "source_text": "", "target_text": ""},
        )

    if task_type == "classification":
        source = _pick_first_text(row, list(contract.get("input_fields", [])))
        label_raw = ""
        for key in list(contract.get("label_fields", [])):
            if key not in row:
                continue
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                label_raw = value.strip()
                break
            if isinstance(value, (int, float, bool)):
                label_raw = str(value)
                break
        if source and label_raw:
            return _attach_multimodal_fields(row, {
                "text": f"Text: {source}\nLabel: {label_raw}",
                "source_text": source,
                "target_text": label_raw,
            })
        if source:
            return _attach_multimodal_fields(
                row,
                {"text": source, "source_text": source, "target_text": ""},
            )
        return _attach_multimodal_fields(
            row,
            {"text": "", "source_text": "", "target_text": ""},
        )

    direct_text = _pick_first_text(row, ["text", "content"])
    if direct_text:
        return _attach_multimodal_fields(
            row,
            {"text": direct_text, "source_text": direct_text, "target_text": ""},
        )

    question = _pick_first_text(row, ["question", "prompt", "instruction"])
    answer = _pick_first_text(
        row,
        [
            "answer",
            "completion",
            "output",
            "response",
            "chosen",
            "preferred",
            "accepted",
            "response_chosen",
        ],
    )
    if question and answer:
        rendered = _qa_to_chat_text(question, answer, chat_template)
        return _attach_multimodal_fields(
            row,
            {"text": rendered, "source_text": question, "target_text": answer},
        )

    if question:
        optional_input = _pick_first_text(row, ["input"])
        if optional_input:
            return _attach_multimodal_fields(row, {
                "text": _qa_to_chat_text(question, optional_input, chat_template),
                "source_text": question,
                "target_text": optional_input,
            })
        return _attach_multimodal_fields(
            row,
            {"text": question, "source_text": question, "target_text": ""},
        )

    return _attach_multimodal_fields(
        row,
        {"text": "", "source_text": "", "target_text": ""},
    )


def _row_has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _is_valid_adapted_row(row: dict[str, Any], task_type: str) -> bool:
    task = str(task_type or "causal_lm").strip().lower()
    if task == "causal_lm":
        return _row_has_text(row.get("text"))
    if task in {"seq2seq", "classification"}:
        return _row_has_text(row.get("source_text")) and _row_has_text(row.get("target_text"))
    return _row_has_text(row.get("text"))


def _summarize_adapted_modalities(
    rows,
    *,
    sample_limit: int = 512,
) -> dict[str, Any]:
    counts: dict[str, int] = {
        "text": 0,
        "vision_language": 0,
        "audio_text": 0,
        "multimodal": 0,
    }
    total = 0
    if rows is None:
        return {"total": 0, "counts": counts, "dominant": "text"}

    try:
        row_count = int(len(rows))
    except Exception:
        row_count = 0
    if row_count <= 0:
        return {"total": 0, "counts": counts, "dominant": "text"}

    max_items = max(1, min(row_count, int(sample_limit)))
    for idx in range(max_items):
        try:
            item = rows[idx]
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("image_path") or "").strip()
        audio_path = str(item.get("audio_path") or "").strip()
        modality = str(item.get("input_modality") or "").strip().lower() or _infer_input_modality(
            text=str(item.get("text") or ""),
            image_path=image_path,
            audio_path=audio_path,
        )
        if modality not in counts:
            modality = "text"
        counts[modality] += 1
        total += 1

    if total == 0:
        return {"total": 0, "counts": counts, "dominant": "text"}

    dominant = max(counts.items(), key=lambda item: item[1])[0]
    return {"total": total, "counts": counts, "dominant": dominant}


def _resolve_media_path(
    value: str,
    *,
    search_roots: list[Path],
) -> Path | None:
    token = str(value or "").strip()
    if not token:
        return None
    lowered = token.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return None
    candidate = Path(token).expanduser()
    if candidate.is_absolute():
        return candidate.resolve() if candidate.exists() else None
    for root in search_roots:
        try:
            resolved = (root / candidate).expanduser().resolve()
        except Exception:
            continue
        if resolved.exists():
            return resolved
    try:
        resolved = candidate.resolve()
    except Exception:
        return None
    return resolved if resolved.exists() else None


def _extract_label_space(
    train_rows,
    eval_rows,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    labels: list[str] = []
    seen: set[str] = set()
    for dataset in [train_rows, eval_rows]:
        if dataset is None:
            continue
        for value in dataset["target_text"]:
            label = str(value or "").strip()
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
    if not labels:
        raise ValueError("Classification task requires at least one non-empty label.")
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return labels, label_to_id, id_to_label


def _last_metric_value(log_history: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(log_history):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _build_checkpoint_index(output_dir: Path, log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    checkpoint_dirs = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-", 1)[1]) if p.name.split("-", 1)[1].isdigit() else 0,
    )

    for ck_dir in checkpoint_dirs:
        suffix = ck_dir.name.split("-", 1)[1]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        epoch = None
        train_loss = None
        eval_loss = None
        for row in reversed(log_history):
            row_step = row.get("step")
            if not isinstance(row_step, (int, float)):
                continue
            if int(row_step) > step:
                continue
            if epoch is None and isinstance(row.get("epoch"), (int, float)):
                epoch = int(row["epoch"]) if row["epoch"] else 1
            if train_loss is None and isinstance(row.get("loss"), (int, float)):
                train_loss = float(row["loss"])
            if eval_loss is None and isinstance(row.get("eval_loss"), (int, float)):
                eval_loss = float(row["eval_loss"])
            if epoch is not None and train_loss is not None and eval_loss is not None:
                break
        checkpoints.append(
            {
                "step": step,
                "epoch": epoch or 1,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "file_path": str(ck_dir),
                "is_best": False,
            }
        )

    best_eval = None
    best_idx = None
    for idx, ck in enumerate(checkpoints):
        loss = ck.get("eval_loss")
        if isinstance(loss, (int, float)):
            if best_eval is None or float(loss) < best_eval:
                best_eval = float(loss)
                best_idx = idx
    if best_idx is not None:
        checkpoints[best_idx]["is_best"] = True

    return checkpoints


def _latest_checkpoint_dir(output_dir: Path) -> Path | None:
    checkpoint_dirs = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.split("-", 1)[1] if "-" in path.name else ""
        if not suffix.isdigit():
            continue
        checkpoint_dirs.append((int(suffix), path))
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort(key=lambda item: item[0])
    return checkpoint_dirs[-1][1]


def _resolve_resume_checkpoint(
    output_dir: Path,
    resume_value: Any,
    warnings: list[str],
) -> Path | None:
    if resume_value is None:
        return _latest_checkpoint_dir(output_dir)

    if isinstance(resume_value, bool):
        return _latest_checkpoint_dir(output_dir) if resume_value else None

    token = str(resume_value).strip()
    if not token:
        return _latest_checkpoint_dir(output_dir)

    lowered = token.lower()
    if lowered in {"0", "false", "no", "off", "none", "null", "disable", "disabled"}:
        return None
    if lowered in {"1", "true", "yes", "on", "auto", "latest"}:
        return _latest_checkpoint_dir(output_dir)

    if token.isdigit():
        candidate = output_dir / f"checkpoint-{token}"
        if candidate.exists() and candidate.is_dir():
            return candidate
        warnings.append(
            f"Configured resume checkpoint step does not exist: {candidate}. Falling back to latest checkpoint."
        )
        return _latest_checkpoint_dir(output_dir)

    candidate = Path(token).expanduser()
    if not candidate.is_absolute():
        candidate = (output_dir / candidate).resolve()
    if candidate.exists() and candidate.is_dir():
        return candidate

    warnings.append(
        f"Configured resume checkpoint path not found: {candidate}. Falling back to latest checkpoint."
    )
    return _latest_checkpoint_dir(output_dir)


def _coerce_constructor_kwargs(
    kwargs: dict[str, Any],
    constructor_owner: type,
    *,
    alias_pairs: list[tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    accepted = inspect.signature(constructor_owner.__init__).parameters
    normalized = dict(kwargs)
    for old_name, new_name in list(alias_pairs or []):
        if old_name in normalized and old_name not in accepted and new_name in accepted:
            normalized[new_name] = normalized.pop(old_name)

    filtered: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in normalized.items():
        if key in accepted:
            filtered[key] = value
        else:
            dropped.append(key)
    return filtered, sorted(dropped)


def _coerce_training_arguments_kwargs(
    kwargs: dict[str, Any],
    training_args_cls: type,
) -> tuple[dict[str, Any], list[str]]:
    return _coerce_constructor_kwargs(
        kwargs,
        training_args_cls,
        alias_pairs=[
            ("evaluation_strategy", "eval_strategy"),
            ("eval_strategy", "evaluation_strategy"),
        ],
    )


def _coerce_trainer_kwargs(
    kwargs: dict[str, Any],
    trainer_cls: type,
) -> tuple[dict[str, Any], list[str]]:
    return _coerce_constructor_kwargs(
        kwargs,
        trainer_cls,
        alias_pairs=[
            ("tokenizer", "processing_class"),
            ("processing_class", "tokenizer"),
        ],
    )


def _normalize_task_type(task_type: str | None, warnings: list[str]) -> str:
    candidate = str(task_type or "causal_lm").strip().lower()
    if candidate in SUPPORTED_TASK_TYPES:
        return candidate
    warnings.append(f"Unknown task_type '{candidate}', defaulting to causal_lm.")
    return "causal_lm"


def _normalize_trainer_backend(trainer_backend: str | None, warnings: list[str]) -> str:
    candidate = str(trainer_backend or "auto").strip().lower()
    if candidate in SUPPORTED_TRAINER_BACKENDS:
        return candidate
    warnings.append(f"Unknown trainer_backend '{candidate}', defaulting to auto.")
    return "auto"


def _normalize_requested_backend(
    requested_backend: str,
    warnings: list[str],
) -> str:
    if requested_backend == "auto":
        return "hf_trainer"
    if requested_backend == "trl_sft":
        try:
            import trl  # noqa: F401
            return "trl_sft"
        except ImportError:
            raise RuntimeError(
                "trainer_backend=trl_sft requested but trl is not installed. "
                "Install trl or use trainer_backend=hf_trainer/auto."
            )
    return "hf_trainer"


def _is_cuda_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        ("out of memory" in message and "cuda" in message)
        or "cudaerrormemoryallocation" in message
    )


def _next_oom_retry_config(
    current_config: dict[str, Any],
    *,
    seq_shrink: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = dict(current_config)
    changes: list[dict[str, Any]] = []

    batch_size = _coerce_int(cfg.get("batch_size"), 1, minimum=1)
    grad_accum = _coerce_int(cfg.get("gradient_accumulation_steps"), 1, minimum=1)
    if batch_size > 1:
        new_batch = max(1, batch_size // 2)
        if new_batch < batch_size:
            cfg["batch_size"] = new_batch
            changes.append({"field": "batch_size", "from": batch_size, "to": new_batch})
            factor = max(1, batch_size // max(1, new_batch))
            new_grad_accum = min(1024, grad_accum * factor)
            if new_grad_accum != grad_accum:
                cfg["gradient_accumulation_steps"] = new_grad_accum
                changes.append(
                    {
                        "field": "gradient_accumulation_steps",
                        "from": grad_accum,
                        "to": new_grad_accum,
                    }
                )
            return cfg, changes

    max_seq_length = _coerce_int(cfg.get("max_seq_length"), 512, minimum=128)
    new_seq = max(128, int(max_seq_length * seq_shrink))
    if new_seq < max_seq_length:
        cfg["max_seq_length"] = new_seq
        changes.append({"field": "max_seq_length", "from": max_seq_length, "to": new_seq})
        return cfg, changes

    if bool(cfg.get("sequence_packing", True)):
        cfg["sequence_packing"] = False
        changes.append({"field": "sequence_packing", "from": True, "to": False})
        return cfg, changes

    if bool(cfg.get("flash_attention", True)):
        cfg["flash_attention"] = False
        changes.append({"field": "flash_attention", "from": True, "to": False})
        return cfg, changes

    lora_r = _coerce_int(cfg.get("lora_r"), 8, minimum=1)
    if bool(cfg.get("use_lora", False)) and lora_r > 4:
        new_lora_r = max(4, lora_r // 2)
        if new_lora_r < lora_r:
            cfg["lora_r"] = new_lora_r
            changes.append({"field": "lora_r", "from": lora_r, "to": new_lora_r})
            return cfg, changes

    return cfg, changes


def _parse_supported_sm_values(arch_list: list[str]) -> list[int]:
    values: list[int] = []
    for token in arch_list:
        item = str(token).strip().lower()
        if not item.startswith("sm_"):
            continue
        suffix = item.removeprefix("sm_")
        if not suffix.isdigit():
            continue
        values.append(int(suffix))
    return values


def _collect_runtime_environment(torch_mod, use_cuda: bool, warnings: list[str]) -> dict[str, Any]:
    runtime_env: dict[str, Any] = {
        "torch_version": getattr(torch_mod, "__version__", "unknown"),
        "torch_cuda_version": str(getattr(getattr(torch_mod, "version", object()), "cuda", None)),
        "cuda_available": bool(use_cuda),
        "device_name": None,
        "device_capability": None,
        "supported_sm": [],
    }
    if not use_cuda:
        return runtime_env

    try:
        capability = torch_mod.cuda.get_device_capability(0)
        cap_val = int(capability[0]) * 10 + int(capability[1])
        runtime_env["device_capability"] = f"sm_{cap_val}"
    except Exception:
        capability = None

    try:
        runtime_env["device_name"] = torch_mod.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        arch_list = torch_mod.cuda.get_arch_list()
    except Exception:
        arch_list = []

    if isinstance(arch_list, list):
        runtime_env["supported_sm"] = [str(x) for x in arch_list]
        supported_vals = _parse_supported_sm_values(arch_list)
        if capability and supported_vals:
            cap_val = int(capability[0]) * 10 + int(capability[1])
            max_supported = max(supported_vals)
            if cap_val > max_supported:
                warnings.append(
                    (
                        "GPU capability appears newer than the max arch in this torch build "
                        f"(device sm_{cap_val}, wheel max sm_{max_supported}). "
                        "Training may run with warnings or reduced stability/performance."
                    )
                )

    return runtime_env


def _load_training_runtime_dependencies() -> dict[str, Any]:
    try:
        import torch
        import transformers as hf_transformers
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            DataCollatorForSeq2Seq,
            DataCollatorWithPadding,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as e:
        raise RuntimeError(
            "Missing training dependencies. Install torch, datasets, transformers, and accelerate."
        ) from e

    return {
        "torch": torch,
        "transformers_module": hf_transformers,
        "load_dataset": load_dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def _run_training_attempt(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    output_dir: Path,
    model_dir: Path,
    config_path: Path,
    train_file: Path,
    val_file: Path | None,
    started_at: str,
    attempt_index: int,
    total_attempts: int,
    retry_history: list[dict[str, Any]],
) -> dict[str, Any]:
    training_mode = str(config.get("training_mode", "sft")).strip().lower() or "sft"
    chat_template = str(config.get("chat_template", "llama3"))
    task_type = str(config.get("task_type", "causal_lm"))
    trainer_backend = str(config.get("trainer_backend", "auto"))
    max_seq_length = _coerce_int(config.get("max_seq_length"), 2048, minimum=128)
    batch_size = _coerce_int(config.get("batch_size"), 4, minimum=1)
    grad_accum = _coerce_int(config.get("gradient_accumulation_steps"), 4, minimum=1)
    learning_rate = _coerce_float(config.get("learning_rate"), 2e-4, minimum=1e-12)
    num_epochs = float(config.get("num_epochs", 3))
    weight_decay = _coerce_float(config.get("weight_decay"), 0.01, minimum=0.0)
    warmup_ratio = _coerce_float(config.get("warmup_ratio"), 0.03, minimum=0.0)
    save_steps = _coerce_int(config.get("save_steps"), 100, minimum=1)
    eval_steps = _coerce_int(config.get("eval_steps"), 100, minimum=1)
    lr_scheduler = str(config.get("lr_scheduler", "cosine"))
    optimizer = str(config.get("optimizer", "adamw_torch"))
    seed = _coerce_int(config.get("seed"), args.seed, minimum=0)
    use_lora = _coerce_bool(config.get("use_lora"), False)
    lora_r = _coerce_int(config.get("lora_r"), 16, minimum=1)
    lora_alpha = _coerce_int(config.get("lora_alpha"), 32, minimum=1)
    lora_dropout = _coerce_float(config.get("lora_dropout"), 0.05, minimum=0.0)
    target_modules = config.get("target_modules", ["q_proj", "v_proj"])
    gradient_checkpointing = _coerce_bool(config.get("gradient_checkpointing"), True)
    want_flash_attention = _coerce_bool(config.get("flash_attention"), True)
    want_fp16 = _coerce_bool(config.get("fp16"), False)
    want_bf16 = _coerce_bool(config.get("bf16"), True)
    sequence_packing = _coerce_bool(config.get("sequence_packing"), True)
    max_train_samples = args.max_train_samples
    max_eval_samples = args.max_eval_samples
    distillation_enabled = _coerce_bool(config.get("distillation_enabled"), False)
    distillation_teacher_model = str(config.get("distillation_teacher_model") or "").strip()
    distillation_alpha = float(config.get("distillation_alpha", 0.6))
    distillation_temperature = _coerce_float(
        config.get("distillation_temperature"),
        2.0,
        minimum=0.1,
    )
    distillation_hidden_state_weight = _coerce_float(
        config.get("distillation_hidden_state_weight"),
        0.0,
        minimum=0.0,
    )
    distillation_hidden_state_loss = str(
        config.get("distillation_hidden_state_loss", "mse")
    ).strip().lower() or "mse"
    observability_enabled = _coerce_bool(config.get("observability_enabled"), True)
    observability_log_steps = _coerce_int(config.get("observability_log_steps"), 50, minimum=5)
    observability_max_layers = _coerce_int(config.get("observability_max_layers"), 12, minimum=1)
    observability_probe_attention = _coerce_bool(config.get("observability_probe_attention"), True)
    observability_probe_top_k = _coerce_int(config.get("observability_probe_top_k"), 6, minimum=1)
    observability_probe_prompt = str(
        config.get("observability_probe_prompt") or "Summarize domain policy facts accurately."
    ).strip() or "Summarize domain policy facts accurately."

    deps = _load_training_runtime_dependencies()
    torch = deps["torch"]
    hf_transformers = deps["transformers_module"]
    load_dataset = deps["load_dataset"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoModelForSeq2SeqLM = deps["AutoModelForSeq2SeqLM"]
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    AutoTokenizer = deps["AutoTokenizer"]
    DataCollatorForLanguageModeling = deps["DataCollatorForLanguageModeling"]
    DataCollatorForSeq2Seq = deps["DataCollatorForSeq2Seq"]
    DataCollatorWithPadding = deps["DataCollatorWithPadding"]
    Trainer = deps["Trainer"]
    TrainerCallback = deps["TrainerCallback"]
    TrainingArguments = deps["TrainingArguments"]
    set_seed = deps["set_seed"]

    warnings: list[str] = []
    if training_mode not in {"sft", "domain_pretrain", "dpo", "orpo"}:
        warnings.append(f"Unknown training_mode '{training_mode}', defaulting to sft.")
        training_mode = "sft"
    normalized_task_type = _normalize_task_type(task_type, warnings)
    normalized_backend = _normalize_trainer_backend(trainer_backend, warnings)
    resolved_backend = _normalize_requested_backend(normalized_backend, warnings)
    if training_mode in {"dpo", "orpo"} and normalized_task_type != "causal_lm":
        raise ValueError(
            f"training_mode={training_mode} requires task_type=causal_lm."
        )
    if distillation_enabled and training_mode in {"dpo", "orpo"}:
        raise ValueError(
            "distillation_enabled is incompatible with DPO/ORPO pairwise objectives."
        )
    if distillation_enabled and normalized_task_type != "causal_lm":
        raise ValueError(
            "distillation_enabled currently supports task_type=causal_lm only."
        )
    if distillation_enabled and not distillation_teacher_model:
        raise ValueError(
            "distillation_enabled=true requires distillation_teacher_model."
        )
    if distillation_hidden_state_loss not in {"mse", "cosine"}:
        warnings.append(
            f"Unknown distillation_hidden_state_loss='{distillation_hidden_state_loss}', defaulting to mse."
        )
        distillation_hidden_state_loss = "mse"
    distillation_alpha = max(0.0, min(float(distillation_alpha), 1.0))
    if resolved_backend == "trl_sft" and normalized_task_type != "causal_lm":
        warnings.append(
            (
                "trainer_backend=trl_sft supports causal_lm only; "
                f"falling back to hf_trainer for task_type={normalized_task_type}."
            )
        )
        resolved_backend = "hf_trainer"
    if training_mode in {"dpo", "orpo"}:
        resolved_backend = f"trl_{training_mode}"
    set_seed(seed)

    data_contract = _build_data_adapter_contract(normalized_task_type, chat_template)

    # Load JSONL datasets.
    data_files: dict[str, str] = {"train": str(train_file)}
    if val_file is not None:
        data_files["validation"] = str(val_file)
    raw_ds = load_dataset("json", data_files=data_files)

    train_ds = raw_ds["train"]
    eval_ds = raw_ds["validation"] if "validation" in raw_ds else None
    if max_train_samples > 0:
        train_ds = train_ds.select(range(min(len(train_ds), max_train_samples)))
    if eval_ds is not None and max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(len(eval_ds), max_eval_samples)))

    train_text = None
    eval_text = None
    train_preference = None
    eval_preference = None
    if training_mode in {"dpo", "orpo"}:
        data_contract = {
            "task_type": "preference_pair",
            "required_fields": ["prompt", "chosen", "rejected"],
            "render_mode": "trl_pairwise",
        }

        def to_preference_record(row: dict[str, Any]) -> dict[str, str]:
            return _adapt_record_to_preference(row)

        train_preference = train_ds.map(to_preference_record, remove_columns=train_ds.column_names)
        train_preference = train_preference.filter(_is_valid_preference_row)
        if len(train_preference) == 0:
            raise ValueError(
                (
                    "No valid prompt/chosen/rejected rows for alignment training. "
                    "Verify preference pair fields before DPO/ORPO training."
                )
            )

        if eval_ds is not None:
            eval_preference = eval_ds.map(to_preference_record, remove_columns=eval_ds.column_names)
            eval_preference = eval_preference.filter(_is_valid_preference_row)
            if len(eval_preference) == 0:
                eval_preference = None
    else:
        def to_text_record(row: dict[str, Any]) -> dict[str, Any]:
            return _adapt_record_to_text(row, data_contract, chat_template)

        train_text = train_ds.map(to_text_record, remove_columns=train_ds.column_names)
        train_text = train_text.filter(lambda row: _is_valid_adapted_row(row, normalized_task_type))
        if len(train_text) == 0:
            raise ValueError(
                (
                    f"No valid training rows after '{normalized_task_type}' adapter normalization. "
                    "Verify required fields are present for the selected task type."
                )
            )

        if eval_ds is not None:
            eval_text = eval_ds.map(to_text_record, remove_columns=eval_ds.column_names)
            eval_text = eval_text.filter(lambda row: _is_valid_adapted_row(row, normalized_task_type))
            if len(eval_text) == 0:
                eval_text = None

    train_modality_summary = _summarize_adapted_modalities(train_text)
    eval_modality_summary = _summarize_adapted_modalities(eval_text)
    train_modality_counts = dict(train_modality_summary.get("counts") or {})
    eval_modality_counts = dict(eval_modality_summary.get("counts") or {})
    train_modality_dominant = str(train_modality_summary.get("dominant") or "text")
    has_multimodal_rows = any(
        int(train_modality_counts.get(key) or 0) > 0
        for key in ("vision_language", "audio_text", "multimodal")
    )
    multimodal_native_beta = _coerce_bool(config.get("multimodal_native_beta"), True)
    multimodal_media_loading = _coerce_bool(config.get("multimodal_media_loading"), True)
    multimodal_require_media = _coerce_bool(config.get("multimodal_require_media"), False)
    use_multimodal_collator = bool(
        has_multimodal_rows
        and multimodal_native_beta
        and training_mode not in {"dpo", "orpo"}
        and normalized_task_type in {"causal_lm", "seq2seq"}
    )
    if has_multimodal_rows and multimodal_require_media:
        if training_mode in {"dpo", "orpo"}:
            raise ValueError(
                (
                    "multimodal_require_media=true is incompatible with training_mode "
                    f"{training_mode}; use SFT/domain_pretrain for multimodal batches."
                )
            )
        if normalized_task_type not in {"causal_lm", "seq2seq"}:
            raise ValueError(
                (
                    "multimodal_require_media=true currently supports multimodal rows "
                    "for task_type causal_lm/seq2seq only."
                )
            )
        if not multimodal_native_beta:
            raise ValueError(
                "multimodal_require_media=true requires multimodal_native_beta=true."
            )
        if not multimodal_media_loading:
            raise ValueError(
                "multimodal_require_media=true requires multimodal_media_loading=true."
            )
        if not use_multimodal_collator:
            raise ValueError(
                "multimodal_require_media=true requires multimodal collator runtime support."
            )
    if has_multimodal_rows and not multimodal_native_beta:
        warnings.append(
            "Detected multimodal rows, but multimodal_native_beta=false; using text-marker fallback path."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    processor = None
    processor_tokenizer = tokenizer
    if use_multimodal_collator:
        auto_processor_cls = getattr(hf_transformers, "AutoProcessor", None)
        if auto_processor_cls is None:
            if multimodal_require_media:
                raise ValueError(
                    (
                        "multimodal_require_media=true requires transformers AutoProcessor, "
                        "but runtime does not expose AutoProcessor."
                    )
                )
            warnings.append(
                "transformers runtime does not expose AutoProcessor; using tokenizer-only multimodal fallback."
            )
        else:
            try:
                processor = auto_processor_cls.from_pretrained(args.base_model, trust_remote_code=True)
                processor_candidate = getattr(processor, "tokenizer", None)
                if processor_candidate is not None:
                    processor_tokenizer = processor_candidate
                elif hasattr(processor, "pad_token_id"):
                    processor_tokenizer = processor
            except Exception as processor_error:  # noqa: BLE001
                if multimodal_require_media:
                    raise ValueError(
                        (
                            "multimodal_require_media=true requires loading AutoProcessor for multimodal batches. "
                            f"Details: {processor_error}"
                        )
                    ) from processor_error
                warnings.append(
                    f"AutoProcessor load failed for multimodal beta path; using tokenizer fallback. Details: {processor_error}"
                )
                processor = None
                processor_tokenizer = tokenizer
    if getattr(processor_tokenizer, "pad_token", None) is None:
        eos_token = getattr(processor_tokenizer, "eos_token", None)
        if eos_token:
            processor_tokenizer.pad_token = eos_token
        elif hasattr(processor_tokenizer, "add_special_tokens"):
            processor_tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    label_space: list[str] = []
    label_to_id: dict[str, int] = {}
    id_to_label: dict[int, str] = {}
    if normalized_task_type == "classification":
        label_space, label_to_id, id_to_label = _extract_label_space(train_text, eval_text)
        model_kwargs["num_labels"] = len(label_space)
        model_kwargs["label2id"] = dict(label_to_id)
        model_kwargs["id2label"] = dict(id_to_label)

    use_cuda = torch.cuda.is_available()
    runtime_environment = _collect_runtime_environment(torch, use_cuda, warnings)
    runtime_environment["distillation_enabled"] = bool(distillation_enabled)
    runtime_environment["observability_enabled"] = bool(observability_enabled)
    runtime_environment["observability_log_steps"] = int(observability_log_steps)
    runtime_environment["observability_max_layers"] = int(observability_max_layers)
    runtime_environment["observability_probe_attention"] = bool(observability_probe_attention)
    runtime_environment["observability_probe_top_k"] = int(observability_probe_top_k)
    runtime_environment["train_modality"] = train_modality_dominant
    runtime_environment["train_modality_counts"] = train_modality_counts
    runtime_environment["eval_modality_counts"] = eval_modality_counts
    runtime_environment["multimodal_native_beta"] = bool(multimodal_native_beta)
    runtime_environment["multimodal_media_loading"] = bool(multimodal_media_loading)
    runtime_environment["multimodal_require_media"] = bool(multimodal_require_media)
    runtime_environment["multimodal_processor_loaded"] = bool(processor is not None)
    runtime_environment["multimodal_processor_class"] = (
        processor.__class__.__name__ if processor is not None else None
    )
    runtime_environment["multimodal_adapter_collator"] = bool(use_multimodal_collator)
    if distillation_enabled:
        runtime_environment["distillation_teacher_model"] = distillation_teacher_model
        runtime_environment["distillation_alpha"] = round(distillation_alpha, 4)
        runtime_environment["distillation_temperature"] = round(distillation_temperature, 4)
        runtime_environment["distillation_hidden_state_weight"] = round(
            distillation_hidden_state_weight,
            4,
        )
        runtime_environment["distillation_hidden_state_loss"] = distillation_hidden_state_loss
    if normalized_task_type == "classification":
        runtime_environment["label_space_size"] = len(label_space)
        runtime_environment["label_space_preview"] = label_space[:50]
    use_bf16 = bool(use_cuda and want_bf16 and torch.cuda.is_bf16_supported())
    use_fp16 = bool(use_cuda and not use_bf16 and want_fp16)
    if use_bf16:
        model_kwargs["dtype"] = torch.bfloat16
    elif use_fp16:
        model_kwargs["dtype"] = torch.float16

    if want_flash_attention and use_cuda:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model_loader = AutoModelForCausalLM
    if normalized_task_type == "seq2seq":
        model_loader = AutoModelForSeq2SeqLM
    elif normalized_task_type == "classification":
        model_loader = AutoModelForSequenceClassification
    if use_multimodal_collator and multimodal_require_media:
        mixed_rows = int(train_modality_counts.get("multimodal") or 0)
        vision_rows = int(train_modality_counts.get("vision_language") or 0)
        audio_rows = int(train_modality_counts.get("audio_text") or 0)
        if mixed_rows > 0:
            raise ValueError(
                (
                    "multimodal_require_media=true does not currently support rows containing both "
                    "image and audio references in one example."
                )
            )
        if vision_rows > 0 and audio_rows > 0:
            raise ValueError(
                (
                    "multimodal_require_media=true does not support datasets containing both "
                    "vision_language and audio_text rows in one run."
                )
            )
    if use_multimodal_collator and normalized_task_type == "seq2seq":
        vision_rows = int(train_modality_counts.get("vision_language") or 0)
        audio_rows = int(train_modality_counts.get("audio_text") or 0)
        mixed_rows = int(train_modality_counts.get("multimodal") or 0)
        if mixed_rows > 0 or (vision_rows > 0 and audio_rows > 0):
            if multimodal_require_media:
                raise ValueError(
                    (
                        "multimodal_require_media=true does not support mixed modality seq2seq "
                        "datasets in the current beta runtime."
                    )
                )
            warnings.append(
                "Mixed modality dataset detected for seq2seq beta runtime; using AutoModelForSeq2SeqLM fallback."
            )
        elif vision_rows > 0:
            vision_loader = getattr(hf_transformers, "AutoModelForVision2Seq", None)
            if vision_loader is not None:
                model_loader = vision_loader
                runtime_environment["multimodal_model_loader"] = "AutoModelForVision2Seq"
        elif audio_rows > 0:
            audio_loader = getattr(hf_transformers, "AutoModelForSpeechSeq2Seq", None)
            if audio_loader is not None:
                model_loader = audio_loader
                runtime_environment["multimodal_model_loader"] = "AutoModelForSpeechSeq2Seq"
    if "multimodal_model_loader" not in runtime_environment:
        runtime_environment["multimodal_model_loader"] = getattr(model_loader, "__name__", str(model_loader))

    def _load_model_with_dtype_fallback(
        load_kwargs: dict[str, Any],
        *,
        model_name: str | None = None,
    ):
        resolved_model_name = str(model_name or args.base_model)
        try:
            loaded = model_loader.from_pretrained(resolved_model_name, **load_kwargs)
            return loaded, load_kwargs
        except TypeError as type_error:
            if "dtype" in load_kwargs and "unexpected keyword argument 'dtype'" in str(type_error):
                fallback_kwargs = dict(load_kwargs)
                fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
                warnings.append("transformers runtime rejected 'dtype'; falling back to legacy 'torch_dtype'.")
                loaded = model_loader.from_pretrained(resolved_model_name, **fallback_kwargs)
                return loaded, fallback_kwargs
            raise

    try:
        model, model_kwargs = _load_model_with_dtype_fallback(model_kwargs)
    except Exception as e:
        if model_kwargs.get("attn_implementation") == "flash_attention_2":
            warnings.append(f"flash_attention_2 unavailable; falling back. Details: {e}")
            model_kwargs.pop("attn_implementation", None)
            model, model_kwargs = _load_model_with_dtype_fallback(model_kwargs)
        else:
            raise

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and len(tokenizer) > input_embeddings.num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise RuntimeError(
                "LoRA training requested but peft is not installed. Install peft or set use_lora=false."
            ) from e

        if not isinstance(target_modules, list) or not target_modules:
            raise ValueError("target_modules must be a non-empty list when use_lora=true")
        peft_task_key = {
            "causal_lm": "CAUSAL_LM",
            "seq2seq": "SEQ_2_SEQ_LM",
            "classification": "SEQ_CLS",
        }.get(normalized_task_type, "CAUSAL_LM")
        peft_task_type = getattr(TaskType, peft_task_key, TaskType.CAUSAL_LM)
        if not hasattr(TaskType, peft_task_key):
            warnings.append(
                f"peft TaskType.{peft_task_key} unavailable, falling back to TaskType.CAUSAL_LM."
            )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=peft_task_type,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            ),
        )

    teacher_model = None
    if distillation_enabled:
        teacher_model_kwargs = dict(model_kwargs)
        if distillation_hidden_state_weight > 0:
            teacher_model_kwargs["output_hidden_states"] = True
        try:
            teacher_model, teacher_model_kwargs = _load_model_with_dtype_fallback(
                teacher_model_kwargs,
                model_name=distillation_teacher_model,
            )
        except Exception as teacher_error:
            if teacher_model_kwargs.get("attn_implementation") == "flash_attention_2":
                warnings.append(
                    (
                        "Teacher model does not support flash_attention_2; falling back. "
                        f"Details: {teacher_error}"
                    )
                )
                teacher_model_kwargs.pop("attn_implementation", None)
                teacher_model, teacher_model_kwargs = _load_model_with_dtype_fallback(
                    teacher_model_kwargs,
                    model_name=distillation_teacher_model,
                )
            else:
                raise
        if teacher_model.config.pad_token_id is None:
            teacher_model.config.pad_token_id = tokenizer.pad_token_id
        teacher_embeddings = teacher_model.get_input_embeddings()
        if teacher_embeddings is not None and len(tokenizer) > teacher_embeddings.num_embeddings:
            warnings.append(
                (
                    "Student tokenizer vocab exceeds teacher vocab size; "
                    "distillation may fail on out-of-range token ids."
                )
            )
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

    has_eval_records = (
        (eval_preference is not None and len(eval_preference) > 0)
        if training_mode in {"dpo", "orpo"}
        else (eval_text is not None and len(eval_text) > 0)
    )

    if optimizer == "paged_adamw_8bit":
        try:
            import bitsandbytes  # noqa: F401
            runtime_environment["bitsandbytes"] = True
        except ImportError:
            runtime_environment["bitsandbytes"] = False
            warnings.append("bitsandbytes not installed; using adamw_torch instead of paged_adamw_8bit.")
            optimizer = "adamw_torch"

    args_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler,
        "logging_steps": max(1, min(save_steps, 20)),
        "save_strategy": "steps",
        "save_steps": max(1, save_steps),
        "save_total_limit": 3,
        "optim": optimizer,
        "gradient_checkpointing": gradient_checkpointing,
        "report_to": [],
        "seed": seed,
        "data_seed": seed,
        "dataloader_num_workers": 0,
        "remove_unused_columns": bool(resolved_backend == "hf_trainer" and not use_multimodal_collator),
        "fp16": use_fp16,
        "bf16": use_bf16,
    }
    if has_eval_records:
        args_kwargs["evaluation_strategy"] = "steps"
        args_kwargs["eval_steps"] = max(1, eval_steps)
        args_kwargs["load_best_model_at_end"] = True
        if normalized_task_type == "classification":
            args_kwargs["metric_for_best_model"] = "eval_accuracy"
            args_kwargs["greater_is_better"] = True
        else:
            args_kwargs["metric_for_best_model"] = "eval_loss"
            args_kwargs["greater_is_better"] = False
        args_kwargs["per_device_eval_batch_size"] = batch_size
    else:
        args_kwargs["evaluation_strategy"] = "no"

    safe_args_kwargs, dropped_arg_keys = _coerce_training_arguments_kwargs(args_kwargs, TrainingArguments)
    if dropped_arg_keys:
        warnings.append(
            "Ignoring unsupported TrainingArguments keys "
            f"for transformers {hf_transformers.__version__}: {', '.join(dropped_arg_keys)}."
        )
    training_args = TrainingArguments(**safe_args_kwargs)

    class StreamingMetricCallback(TrainerCallback):
        """Emit parseable metrics/events to stdout for worker-side streaming."""

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ANN201
            if not isinstance(logs, dict):
                return
            train_loss = _safe_float(logs.get("loss"))
            eval_loss = _safe_float(logs.get("eval_loss"))
            learning_rate = _safe_float(logs.get("learning_rate"))
            epoch = _safe_float(logs.get("epoch"))
            if epoch is None:
                epoch = _safe_float(getattr(state, "epoch", None))
            step = getattr(state, "global_step", None)
            if not isinstance(step, int):
                step = logs.get("step") if isinstance(logs.get("step"), int) else None

            if (
                train_loss is None
                and eval_loss is None
                and learning_rate is None
                and epoch is None
            ):
                return

            metric: dict[str, Any] = {}
            if step is not None:
                metric["step"] = int(step)
            if epoch is not None:
                metric["epoch"] = round(epoch, 4)
            if train_loss is not None:
                metric["train_loss"] = train_loss
            if eval_loss is not None:
                metric["eval_loss"] = eval_loss
            if learning_rate is not None:
                metric["learning_rate"] = learning_rate
            if metric:
                print(f"{TRAINING_METRIC_PREFIX}{json.dumps(metric, ensure_ascii=False)}", flush=True)

        def on_epoch_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN201
            epoch = _safe_float(getattr(state, "epoch", None))
            if epoch is None:
                return
            _emit_runtime_event("epoch_end", {"epoch": max(1, int(round(epoch)))})

    class ObservabilityCallback(TrainerCallback):
        """Emit structured observability events for gradient/attention diagnostics."""

        def __init__(
            self,
            *,
            enabled: bool,
            log_steps: int,
            max_layers: int,
            probe_attention: bool,
            probe_top_k: int,
            probe_prompt: str,
        ) -> None:
            self.enabled = bool(enabled)
            self.log_steps = max(1, int(log_steps))
            self.max_layers = max(1, int(max_layers))
            self.probe_attention = bool(probe_attention)
            self.probe_top_k = max(1, int(probe_top_k))
            self.probe_prompt = str(probe_prompt or "").strip() or "Summarize domain policy facts accurately."
            self._last_emitted_step = -1
            self._last_attention_error: str | None = None

        def _layer_bucket(self, name: str) -> str:
            token = str(name or "")
            markers = ["layers.", "layer.", "h.", "block.", "encoder.layer.", "decoder.layer."]
            for marker in markers:
                idx = token.find(marker)
                if idx < 0:
                    continue
                suffix = token[idx:]
                parts = suffix.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    return f"{parts[0]}.{parts[1]}"
                return parts[0]
            parts = token.split(".")
            if len(parts) >= 2:
                return f"{parts[0]}.{parts[1]}"
            return token or "model"

        def _collect_layer_gradients(self, model_ref) -> list[dict[str, Any]]:  # noqa: ANN001
            buckets: dict[str, dict[str, float]] = {}
            for name, parameter in model_ref.named_parameters():
                grad = getattr(parameter, "grad", None)
                if grad is None:
                    continue
                try:
                    grad_norm = float(grad.detach().float().norm().item())
                    weight_norm = float(parameter.detach().float().norm().item())
                except Exception:
                    continue
                layer = self._layer_bucket(name)
                bucket = buckets.get(layer)
                if bucket is None:
                    bucket = {
                        "grad_norm_sum": 0.0,
                        "weight_norm_sum": 0.0,
                        "update_ratio_sum": 0.0,
                        "count": 0.0,
                    }
                    buckets[layer] = bucket
                update_ratio = grad_norm / max(weight_norm, 1e-8)
                bucket["grad_norm_sum"] += grad_norm
                bucket["weight_norm_sum"] += weight_norm
                bucket["update_ratio_sum"] += update_ratio
                bucket["count"] += 1.0

            rows: list[dict[str, Any]] = []
            for layer, aggregate in buckets.items():
                count = max(1.0, float(aggregate.get("count", 1.0)))
                rows.append(
                    {
                        "layer": layer,
                        "grad_norm": float(aggregate.get("grad_norm_sum", 0.0)) / count,
                        "weight_norm": float(aggregate.get("weight_norm_sum", 0.0)) / count,
                        "update_ratio": float(aggregate.get("update_ratio_sum", 0.0)) / count,
                    }
                )
            rows.sort(key=lambda item: float(item.get("grad_norm", 0.0)), reverse=True)
            return rows[: self.max_layers]

        def _collect_attention_focus(self, model_ref) -> list[dict[str, Any]]:  # noqa: ANN001
            if not self.probe_attention:
                return []
            prompt = self.probe_prompt
            if not prompt:
                return []
            try:
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(256, max_seq_length),
                )
                model_device = next(model_ref.parameters()).device
                encoded = {key: value.to(model_device) for key, value in encoded.items()}
                model_ref.eval()
                with torch.no_grad():
                    outputs = model_ref(**encoded, output_attentions=True)
                attentions = getattr(outputs, "attentions", None)
                if not isinstance(attentions, tuple) and not isinstance(attentions, list):
                    return []
                if len(attentions) == 0:
                    return []
                last = attentions[-1]
                if last is None or getattr(last, "ndim", 0) < 4:
                    return []
                scores = last[0].float().mean(dim=0)  # [seq, seq]
                if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
                    return []
                focus_weights = scores[-1]
                top_k = min(int(self.probe_top_k), int(focus_weights.shape[0]))
                if top_k <= 0:
                    return []
                values, indices = torch.topk(focus_weights, k=top_k)
                input_ids = encoded.get("input_ids")
                if input_ids is None:
                    return []
                ids_row = input_ids[0]
                rows: list[dict[str, Any]] = []
                for idx, val in zip(indices.tolist(), values.tolist()):
                    token_text = ""
                    if isinstance(idx, int) and 0 <= idx < int(ids_row.shape[0]):
                        token_id = int(ids_row[idx].item())
                        token_text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
                    rows.append(
                        {
                            "token": token_text or f"token_{idx}",
                            "weight": float(val),
                            "source": "probe_prompt",
                        }
                    )
                self._last_attention_error = None
                return rows
            except Exception as attention_error:  # noqa: BLE001
                self._last_attention_error = str(attention_error)
                return []

        def on_log(self, args, state, control, logs=None, model=None, **kwargs):  # noqa: ANN001, ANN201
            _ = args, control, kwargs
            if not self.enabled:
                return
            if model is None:
                return
            step = getattr(state, "global_step", None)
            if not isinstance(step, int):
                if isinstance(logs, dict) and isinstance(logs.get("step"), int):
                    step = int(logs["step"])
            if not isinstance(step, int) or step <= 0:
                return
            if step == self._last_emitted_step:
                return
            if step % self.log_steps != 0:
                return
            layer_gradients = self._collect_layer_gradients(model)
            if not layer_gradients:
                return
            attention_focus = self._collect_attention_focus(model)
            max_grad = max(float(item.get("grad_norm", 0.0)) for item in layer_gradients)
            notes = []
            if self._last_attention_error:
                notes.append(f"attention_probe_error: {self._last_attention_error}")
            _emit_runtime_event(
                "observability",
                {
                    "step": step,
                    "epoch": _safe_float(getattr(state, "epoch", None)),
                    "split": "train",
                    "layer_gradients": layer_gradients,
                    "attention_focus": attention_focus,
                    "gradient_anomaly": bool(max_grad >= 5.0),
                    "hallucination_signal": any(
                        "<unk>" in str(item.get("token", "")).lower() for item in attention_focus
                    ),
                    "notes": "; ".join(notes) if notes else "",
                },
            )
            self._last_emitted_step = step

    alignment_native_objective = False
    if training_mode in {"dpo", "orpo"}:
        try:
            import trl
        except ImportError as e:
            raise RuntimeError(
                (
                    f"training_mode={training_mode} requires TRL pairwise trainers. "
                    "Install trl in the training runtime environment."
                )
            ) from e

        trainer_name = "DPOTrainer" if training_mode == "dpo" else "ORPOTrainer"
        config_name = "DPOConfig" if training_mode == "dpo" else "ORPOConfig"
        pairwise_trainer_cls = getattr(trl, trainer_name, None)
        pairwise_config_cls = getattr(trl, config_name, None)
        if pairwise_trainer_cls is None:
            raise RuntimeError(
                f"Installed trl does not provide {trainer_name}; upgrade trl to use {training_mode}."
            )

        runtime_environment["trl_version"] = str(getattr(trl, "__version__", "unknown"))
        runtime_environment["alignment_trainer"] = trainer_name

        alignment_beta = _coerce_float(config.get("alignment_beta"), 0.1, minimum=1e-6)
        alignment_max_length = _coerce_int(config.get("alignment_max_length"), max_seq_length, minimum=128)
        alignment_max_prompt_length = _coerce_int(
            config.get("alignment_max_prompt_length"),
            max(64, min(max_seq_length, max_seq_length // 2)),
            minimum=32,
        )
        if alignment_max_prompt_length >= alignment_max_length:
            alignment_max_prompt_length = max(32, alignment_max_length - 16)

        pairwise_args = training_args
        if pairwise_config_cls is not None:
            pairwise_args_kwargs = dict(safe_args_kwargs)
            pairwise_args_kwargs["beta"] = alignment_beta
            pairwise_args_kwargs["max_length"] = alignment_max_length
            pairwise_args_kwargs["max_prompt_length"] = alignment_max_prompt_length
            safe_pairwise_args_kwargs, dropped_pairwise_arg_keys = _coerce_constructor_kwargs(
                pairwise_args_kwargs,
                pairwise_config_cls,
                alias_pairs=[
                    ("evaluation_strategy", "eval_strategy"),
                    ("eval_strategy", "evaluation_strategy"),
                ],
            )
            if dropped_pairwise_arg_keys:
                warnings.append(
                    (
                        f"Ignoring unsupported {config_name} keys for trl "
                        f"{runtime_environment['trl_version']}: {', '.join(dropped_pairwise_arg_keys)}."
                    )
                )
            pairwise_args = pairwise_config_cls(**safe_pairwise_args_kwargs)
        else:
            warnings.append(
                f"trl does not expose {config_name}; falling back to TrainingArguments for {trainer_name}."
            )

        pairwise_kwargs: dict[str, Any] = {
            "model": model,
            "args": pairwise_args,
            "train_dataset": train_preference,
            "eval_dataset": eval_preference if has_eval_records else None,
            "tokenizer": tokenizer,
            "max_length": alignment_max_length,
            "max_prompt_length": alignment_max_prompt_length,
        }
        if pairwise_config_cls is None:
            pairwise_kwargs["beta"] = alignment_beta
        if training_mode == "dpo":
            ref_model, _ = _load_model_with_dtype_fallback(dict(model_kwargs))
            if getattr(ref_model, "config", None) is not None and ref_model.config.pad_token_id is None:
                ref_model.config.pad_token_id = tokenizer.pad_token_id
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad_(False)
            pairwise_kwargs["ref_model"] = ref_model

        safe_pairwise_kwargs, dropped_pairwise_keys = _coerce_trainer_kwargs(
            pairwise_kwargs,
            pairwise_trainer_cls,
        )
        if dropped_pairwise_keys:
            warnings.append(
                "Ignoring unsupported pairwise trainer keys "
                f"for {trainer_name}: {', '.join(dropped_pairwise_keys)}."
            )

        trainer = pairwise_trainer_cls(**safe_pairwise_kwargs)
        alignment_native_objective = True
    elif resolved_backend == "trl_sft":
        from trl import SFTTrainer

        sft_kwargs: dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": train_text,
            "eval_dataset": eval_text if has_eval_records else None,
            "dataset_text_field": "text",
            "packing": sequence_packing,
            "max_seq_length": max_seq_length,
        }
        if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
            sft_kwargs["processing_class"] = tokenizer
        else:
            sft_kwargs["tokenizer"] = tokenizer
        safe_sft_kwargs, dropped_sft_keys = _coerce_trainer_kwargs(sft_kwargs, SFTTrainer)
        if dropped_sft_keys:
            warnings.append(
                "Ignoring unsupported SFTTrainer keys "
                f"for trl backend: {', '.join(dropped_sft_keys)}."
            )
        trainer = SFTTrainer(**safe_sft_kwargs)
    else:
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": training_args,
            "tokenizer": tokenizer,
        }
        trainer_cls: type = Trainer

        if distillation_enabled:
            if teacher_model is None:
                raise RuntimeError("Distillation is enabled but teacher model failed to load.")

            class DistillationTrainer(Trainer):
                def __init__(
                    self,
                    *args,
                    teacher_model_ref,
                    distill_alpha: float,
                    distill_temperature: float,
                    hidden_state_weight: float,
                    hidden_state_loss_type: str,
                    **kwargs,
                ):
                    super().__init__(*args, **kwargs)
                    self.teacher_model = teacher_model_ref
                    self.teacher_model.eval()
                    for parameter in self.teacher_model.parameters():
                        parameter.requires_grad_(False)
                    self.distill_alpha = max(0.0, min(float(distill_alpha), 1.0))
                    self.distill_temperature = max(0.1, float(distill_temperature))
                    self.hidden_state_weight = max(0.0, float(hidden_state_weight))
                    self.hidden_state_loss_type = hidden_state_loss_type
                    self._last_distill_log_step = -1

                def _ensure_teacher_device(self, target_device):  # noqa: ANN001
                    teacher_device = next(self.teacher_model.parameters()).device
                    if teacher_device != target_device:
                        self.teacher_model.to(target_device)

                def compute_loss(
                    self,
                    model,  # noqa: ANN001
                    inputs,  # noqa: ANN001
                    return_outputs: bool = False,
                    num_items_in_batch=None,  # noqa: ANN001
                ):
                    import torch.nn.functional as F

                    outputs = model(**inputs)
                    student_loss = outputs.get("loss")
                    if student_loss is None:
                        if return_outputs:
                            return outputs.loss, outputs
                        return outputs.loss

                    teacher_inputs = {}
                    for key in ("input_ids", "attention_mask", "position_ids"):
                        value = inputs.get(key)
                        if value is not None:
                            teacher_inputs[key] = value

                    self._ensure_teacher_device(outputs.logits.device)
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(
                            **teacher_inputs,
                            output_hidden_states=self.hidden_state_weight > 0.0,
                        )

                    student_logits = outputs.logits.float()
                    teacher_logits = teacher_outputs.logits.float()
                    seq_len = min(int(student_logits.shape[1]), int(teacher_logits.shape[1]))
                    vocab_size = min(int(student_logits.shape[-1]), int(teacher_logits.shape[-1]))
                    student_logits = student_logits[:, :seq_len, :vocab_size]
                    teacher_logits = teacher_logits[:, :seq_len, :vocab_size]

                    labels = inputs.get("labels")
                    if labels is not None and hasattr(labels, "shape") and len(labels.shape) == 2:
                        mask = (labels[:, :seq_len] != -100).unsqueeze(-1)
                        mask_f = mask.to(student_logits.dtype)
                        valid_count = float(mask_f.sum().item())
                        if valid_count > 0:
                            student_logits = student_logits * mask_f
                            teacher_logits = teacher_logits * mask_f

                    log_probs = F.log_softmax(
                        student_logits / self.distill_temperature,
                        dim=-1,
                    )
                    probs = F.softmax(
                        teacher_logits / self.distill_temperature,
                        dim=-1,
                    )
                    kd_loss = (
                        F.kl_div(log_probs, probs, reduction="batchmean")
                        * (self.distill_temperature ** 2)
                    )

                    total_loss = (
                        (self.distill_alpha * student_loss)
                        + ((1.0 - self.distill_alpha) * kd_loss)
                    )
                    hidden_loss_value = None
                    if self.hidden_state_weight > 0.0:
                        student_states = getattr(outputs, "hidden_states", None)
                        teacher_states = getattr(teacher_outputs, "hidden_states", None)
                        if student_states and teacher_states:
                            student_last = student_states[-1].float()
                            teacher_last = teacher_states[-1].float()
                            hs_seq = min(int(student_last.shape[1]), int(teacher_last.shape[1]))
                            hs_dim = min(int(student_last.shape[2]), int(teacher_last.shape[2]))
                            student_last = student_last[:, :hs_seq, :hs_dim]
                            teacher_last = teacher_last[:, :hs_seq, :hs_dim]
                            if self.hidden_state_loss_type == "cosine":
                                s_flat = student_last.reshape(-1, hs_dim)
                                t_flat = teacher_last.reshape(-1, hs_dim)
                                hidden_loss = 1.0 - F.cosine_similarity(s_flat, t_flat, dim=-1).mean()
                            else:
                                hidden_loss = F.mse_loss(student_last, teacher_last)
                            hidden_loss_value = hidden_loss
                            total_loss = total_loss + (self.hidden_state_weight * hidden_loss)

                    step = int(getattr(self.state, "global_step", 0) or 0)
                    if step != self._last_distill_log_step and step % 10 == 0:
                        metrics = {
                            "distill_total_loss": float(total_loss.detach().cpu()),
                            "distill_kd_loss": float(kd_loss.detach().cpu()),
                            "distill_ce_loss": float(student_loss.detach().cpu()),
                            "distill_alpha": self.distill_alpha,
                            "distill_temperature": self.distill_temperature,
                        }
                        if hidden_loss_value is not None:
                            metrics["distill_hidden_loss"] = float(hidden_loss_value.detach().cpu())
                        self.log(metrics)
                        self._last_distill_log_step = step

                    if return_outputs:
                        outputs.loss = total_loss
                        return total_loss, outputs
                    return total_loss

            trainer_cls = DistillationTrainer
            trainer_kwargs["teacher_model_ref"] = teacher_model
            trainer_kwargs["distill_alpha"] = distillation_alpha
            trainer_kwargs["distill_temperature"] = distillation_temperature
            trainer_kwargs["hidden_state_weight"] = distillation_hidden_state_weight
            trainer_kwargs["hidden_state_loss_type"] = distillation_hidden_state_loss

        if use_multimodal_collator and normalized_task_type in {"causal_lm", "seq2seq"}:
            media_roots = [
                train_file.parent,
                train_file.parent.parent,
                Path(args.data_dir).expanduser().resolve(),
            ]
            collator_stats: dict[str, int] = {
                "total_batches": 0,
                "text_only_batches": 0,
                "vision_batches": 0,
                "audio_batches": 0,
                "mixed_batches": 0,
                "media_fallback_batches": 0,
            }
            model_forward_params = set(inspect.signature(model.forward).parameters)
            accepts_pixel_values = "pixel_values" in model_forward_params
            accepts_input_features = "input_features" in model_forward_params
            accepts_input_values = "input_values" in model_forward_params
            runtime_environment["multimodal_model_forward_keys"] = sorted(
                [
                    key
                    for key in ("pixel_values", "input_features", "input_values")
                    if key in model_forward_params
                ]
            )
            if multimodal_require_media:
                vision_rows = int(train_modality_counts.get("vision_language") or 0)
                audio_rows = int(train_modality_counts.get("audio_text") or 0)
                if vision_rows > 0 and not accepts_pixel_values:
                    raise ValueError(
                        (
                            "multimodal_require_media=true requires model.forward to accept "
                            "'pixel_values' for vision_language rows."
                        )
                    )
                if audio_rows > 0 and not (accepts_input_features or accepts_input_values):
                    raise ValueError(
                        (
                            "multimodal_require_media=true requires model.forward to accept "
                            "'input_features' or 'input_values' for audio_text rows."
                        )
                    )

            class AdapterAwareMultimodalCollator:
                def __init__(self) -> None:
                    self._warned_image_error = False
                    self._warned_audio_error = False
                    self._warned_mixed_batch = False

                def _warn_or_fail(self, message: str, *, warned_attr: str | None = None) -> None:
                    if multimodal_require_media:
                        raise ValueError(message)
                    if warned_attr:
                        if bool(getattr(self, warned_attr, False)):
                            return
                        setattr(self, warned_attr, True)
                    warnings.append(message)

                def _encode_text_batch(
                    self,
                    texts: list[str],
                    targets: list[str],
                ) -> dict[str, Any]:
                    if normalized_task_type == "seq2seq":
                        try:
                            return processor_tokenizer(
                                texts,
                                text_target=targets,
                                truncation=True,
                                max_length=max_seq_length,
                                max_target_length=max_seq_length,
                                padding=True,
                                return_tensors="pt",
                            )
                        except TypeError:
                            model_inputs = processor_tokenizer(
                                texts,
                                truncation=True,
                                max_length=max_seq_length,
                                padding=True,
                                return_tensors="pt",
                            )
                            if hasattr(processor_tokenizer, "as_target_tokenizer"):
                                with processor_tokenizer.as_target_tokenizer():
                                    labels = processor_tokenizer(
                                        targets,
                                        truncation=True,
                                        max_length=max_seq_length,
                                        padding=True,
                                        return_tensors="pt",
                                    )
                            else:
                                labels = processor_tokenizer(
                                    targets,
                                    truncation=True,
                                    max_length=max_seq_length,
                                    padding=True,
                                    return_tensors="pt",
                                )
                            model_inputs["labels"] = labels["input_ids"]
                            return model_inputs

                    encoded = processor_tokenizer(
                        texts,
                        truncation=True,
                        max_length=max_seq_length,
                        padding=True,
                        return_tensors="pt",
                    )
                    labels = encoded["input_ids"].clone()
                    pad_id = getattr(processor_tokenizer, "pad_token_id", None)
                    if isinstance(pad_id, int):
                        labels[labels == pad_id] = -100
                    encoded["labels"] = labels
                    return encoded

                def _load_images(self, rows: list[dict[str, Any]]) -> list[Any] | None:
                    if processor is None:
                        if multimodal_require_media:
                            raise ValueError(
                                "multimodal_require_media=true requires AutoProcessor for vision batches."
                            )
                        return None
                    if not accepts_pixel_values:
                        if multimodal_require_media:
                            raise ValueError(
                                "multimodal_require_media=true requires model.forward to accept 'pixel_values' for vision batches."
                            )
                        return None
                    if not multimodal_media_loading:
                        if multimodal_require_media:
                            raise ValueError(
                                "multimodal_require_media=true requires multimodal_media_loading=true for vision batches."
                            )
                        return None
                    images: list[Any] = []
                    for row in rows:
                        token = str(row.get("image_path") or "").strip()
                        media_path = _resolve_media_path(token, search_roots=media_roots)
                        if media_path is None:
                            if multimodal_require_media:
                                raise ValueError(
                                    (
                                        "multimodal_require_media=true but image asset is unresolved: "
                                        f"'{token or '<empty>'}'."
                                    )
                                )
                            return None
                        try:
                            from PIL import Image
                        except Exception:
                            self._warn_or_fail(
                                "Pillow missing; vision media loading disabled for multimodal batches.",
                                warned_attr="_warned_image_error",
                            )
                            return None
                        try:
                            with Image.open(str(media_path)) as img:
                                images.append(img.convert("RGB"))
                        except Exception as image_error:  # noqa: BLE001
                            self._warn_or_fail(
                                (
                                    "Image media loading failed; using text-only fallback. "
                                    f"Details: {image_error}"
                                ),
                                warned_attr="_warned_image_error",
                            )
                            return None
                    return images if images else None

                def _load_audios(self, rows: list[dict[str, Any]]) -> tuple[list[Any], int] | tuple[None, None]:
                    if processor is None:
                        if multimodal_require_media:
                            raise ValueError(
                                "multimodal_require_media=true requires AutoProcessor for audio batches."
                            )
                        return None, None
                    if not multimodal_media_loading:
                        if multimodal_require_media:
                            raise ValueError(
                                "multimodal_require_media=true requires multimodal_media_loading=true for audio batches."
                            )
                        return None, None
                    if not accepts_input_features and not accepts_input_values:
                        if multimodal_require_media:
                            raise ValueError(
                                (
                                    "multimodal_require_media=true requires model.forward to accept "
                                    "'input_features' or 'input_values' for audio batches."
                                )
                            )
                        return None, None
                    audios: list[Any] = []
                    sample_rate: int | None = None
                    for row in rows:
                        token = str(row.get("audio_path") or "").strip()
                        media_path = _resolve_media_path(token, search_roots=media_roots)
                        if media_path is None:
                            if multimodal_require_media:
                                raise ValueError(
                                    (
                                        "multimodal_require_media=true but audio asset is unresolved: "
                                        f"'{token or '<empty>'}'."
                                    )
                                )
                            return None, None
                        waveform = None
                        current_sr = None
                        try:
                            import soundfile as sf

                            waveform, current_sr = sf.read(str(media_path), dtype="float32")
                            if getattr(waveform, "ndim", 1) > 1:
                                waveform = waveform.mean(axis=1)
                        except Exception:
                            try:
                                import torchaudio

                                tensor, current_sr = torchaudio.load(str(media_path))
                                if int(getattr(tensor, "ndim", 1)) > 1:
                                    tensor = tensor.mean(dim=0)
                                waveform = tensor.detach().cpu().numpy()
                            except Exception as audio_error:  # noqa: BLE001
                                self._warn_or_fail(
                                    (
                                        "Audio media loading failed; using text-only fallback. "
                                        f"Details: {audio_error}"
                                    ),
                                    warned_attr="_warned_audio_error",
                                )
                                return None, None
                        if waveform is None or current_sr is None:
                            if multimodal_require_media:
                                raise ValueError(
                                    "multimodal_require_media=true but audio decoder returned empty waveform/sample rate."
                                )
                            return None, None
                        sr_int = int(current_sr)
                        if sample_rate is None:
                            sample_rate = sr_int
                        if sr_int != sample_rate:
                            self._warn_or_fail(
                                "Audio samples use mixed sampling rates; using text-only fallback for multimodal batches.",
                                warned_attr="_warned_audio_error",
                            )
                            return None, None
                        audios.append(waveform)
                    return (audios, int(sample_rate)) if audios and sample_rate is not None else (None, None)

                def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
                    rows = [dict(item) for item in list(features or [])]
                    collator_stats["total_batches"] += 1
                    texts: list[str] = []
                    targets: list[str] = []
                    has_image = False
                    has_audio = False
                    for row in rows:
                        text = str(row.get("text") or row.get("source_text") or "").strip()
                        target = str(row.get("target_text") or "").strip()
                        texts.append(text)
                        targets.append(target)
                        has_image = has_image or bool(str(row.get("image_path") or "").strip())
                        has_audio = has_audio or bool(str(row.get("audio_path") or "").strip())

                    if has_image and has_audio:
                        collator_stats["mixed_batches"] += 1
                        self._warn_or_fail(
                            "Mixed image+audio batch detected; using text-marker fallback for this batch.",
                            warned_attr="_warned_mixed_batch",
                        )
                    elif has_image:
                        collator_stats["vision_batches"] += 1
                    elif has_audio:
                        collator_stats["audio_batches"] += 1
                    else:
                        collator_stats["text_only_batches"] += 1

                    batch = self._encode_text_batch(texts, targets)
                    if processor is None:
                        if multimodal_require_media and (has_image or has_audio):
                            raise ValueError(
                                (
                                    "multimodal_require_media=true but processor is unavailable for "
                                    "media rows in current batch."
                                )
                            )
                        return batch

                    if has_image and not has_audio:
                        loaded_images = self._load_images(rows)
                        if loaded_images:
                            try:
                                media_batch = processor(images=loaded_images, return_tensors="pt")
                            except Exception as media_error:  # noqa: BLE001
                                if multimodal_require_media:
                                    raise ValueError(
                                        (
                                            "multimodal_require_media=true failed while building image processor "
                                            f"batch. Details: {media_error}"
                                        )
                                    ) from media_error
                                collator_stats["media_fallback_batches"] += 1
                                return batch
                            pixel_values = media_batch.get("pixel_values")
                            if pixel_values is not None and accepts_pixel_values:
                                batch["pixel_values"] = pixel_values
                            else:
                                if multimodal_require_media:
                                    raise ValueError(
                                        (
                                            "multimodal_require_media=true requires processor output "
                                            "to include pixel_values for image rows."
                                        )
                                    )
                                collator_stats["media_fallback_batches"] += 1
                        else:
                            if multimodal_require_media:
                                raise ValueError(
                                    "multimodal_require_media=true could not load images for current batch."
                                )
                            collator_stats["media_fallback_batches"] += 1
                    elif has_audio and not has_image:
                        loaded_audios, sampling_rate = self._load_audios(rows)
                        if loaded_audios and sampling_rate:
                            media_batch = None
                            for audio_key in ("audios", "audio", "raw_speech", "speech"):
                                try:
                                    media_batch = processor(
                                        **{audio_key: loaded_audios},
                                        sampling_rate=int(sampling_rate),
                                        return_tensors="pt",
                                        padding=True,
                                    )
                                    break
                                except TypeError:
                                    media_batch = None
                                    continue
                            if isinstance(media_batch, dict):
                                if accepts_input_features and media_batch.get("input_features") is not None:
                                    batch["input_features"] = media_batch["input_features"]
                                elif accepts_input_values and media_batch.get("input_values") is not None:
                                    batch["input_values"] = media_batch["input_values"]
                                else:
                                    if multimodal_require_media:
                                        raise ValueError(
                                            (
                                                "multimodal_require_media=true requires processor output "
                                                "to include input_features/input_values for audio rows."
                                            )
                                        )
                                    collator_stats["media_fallback_batches"] += 1
                            else:
                                if multimodal_require_media:
                                    raise ValueError(
                                        (
                                            "multimodal_require_media=true failed to build audio processor batch "
                                            "for current rows."
                                        )
                                    )
                                collator_stats["media_fallback_batches"] += 1
                        else:
                            if multimodal_require_media:
                                raise ValueError(
                                    "multimodal_require_media=true could not load audios for current batch."
                                )
                            collator_stats["media_fallback_batches"] += 1
                    return batch

            trainer_kwargs["train_dataset"] = train_text
            trainer_kwargs["eval_dataset"] = eval_text if has_eval_records else None
            trainer_kwargs["data_collator"] = AdapterAwareMultimodalCollator()
            runtime_environment["multimodal_collator_strategy"] = "adapter_aware_dynamic"
            runtime_environment["multimodal_collator_stats"] = collator_stats
            runtime_environment["multimodal_accepts_pixel_values"] = bool(accepts_pixel_values)
            runtime_environment["multimodal_accepts_input_features"] = bool(accepts_input_features)
            runtime_environment["multimodal_accepts_input_values"] = bool(accepts_input_values)
            _emit_runtime_event(
                "multimodal_runtime",
                {
                    "enabled": True,
                    "train_modality_counts": train_modality_counts,
                    "eval_modality_counts": eval_modality_counts,
                    "processor_loaded": bool(processor is not None),
                    "media_loading": bool(multimodal_media_loading),
                    "require_media": bool(multimodal_require_media),
                    "model_loader": runtime_environment.get("multimodal_model_loader"),
                },
            )
        elif normalized_task_type == "seq2seq":
            def tokenize_rows(rows: dict[str, list[str]]) -> dict[str, Any]:
                sources = [str(x) for x in rows["source_text"]]
                targets = [str(x) for x in rows["target_text"]]
                try:
                    return processor_tokenizer(
                        sources,
                        text_target=targets,
                        truncation=True,
                        max_length=max_seq_length,
                        max_target_length=max_seq_length,
                        padding=False,
                    )
                except TypeError:
                    model_inputs = processor_tokenizer(
                        sources,
                        truncation=True,
                        max_length=max_seq_length,
                        padding=False,
                    )
                    if hasattr(processor_tokenizer, "as_target_tokenizer"):
                        with processor_tokenizer.as_target_tokenizer():
                            labels = processor_tokenizer(
                                targets,
                                truncation=True,
                                max_length=max_seq_length,
                                padding=False,
                            )
                    else:
                        labels = processor_tokenizer(
                            targets,
                            truncation=True,
                            max_length=max_seq_length,
                            padding=False,
                        )
                    model_inputs["labels"] = labels["input_ids"]
                    return model_inputs

            tokenized_train = train_text.map(tokenize_rows, batched=True, remove_columns=train_text.column_names)
            tokenized_eval = (
                eval_text.map(tokenize_rows, batched=True, remove_columns=eval_text.column_names)
                if eval_text is not None
                else None
            )
            trainer_kwargs["train_dataset"] = tokenized_train
            trainer_kwargs["eval_dataset"] = tokenized_eval if has_eval_records else None
            trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(tokenizer=processor_tokenizer, model=model)
        elif normalized_task_type == "classification":
            def tokenize_rows(rows: dict[str, list[str]]) -> dict[str, Any]:
                sources = [str(x) for x in rows["source_text"]]
                labels: list[int] = []
                for raw in rows["target_text"]:
                    key = str(raw or "").strip()
                    if key not in label_to_id:
                        raise ValueError(f"Unknown classification label encountered: '{key}'")
                    labels.append(label_to_id[key])
                tokenized = processor_tokenizer(
                    sources,
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                )
                tokenized["labels"] = labels
                return tokenized

            tokenized_train = train_text.map(tokenize_rows, batched=True, remove_columns=train_text.column_names)
            tokenized_eval = (
                eval_text.map(tokenize_rows, batched=True, remove_columns=eval_text.column_names)
                if eval_text is not None
                else None
            )

            trainer_kwargs["train_dataset"] = tokenized_train
            trainer_kwargs["eval_dataset"] = tokenized_eval if has_eval_records else None
            trainer_kwargs["data_collator"] = DataCollatorWithPadding(tokenizer=processor_tokenizer)

            if has_eval_records:
                import numpy as np

                def compute_metrics(eval_pred):  # noqa: ANN001
                    logits, labels = eval_pred
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    preds = np.argmax(logits, axis=-1)
                    labels_arr = np.asarray(labels)
                    preds_arr = np.asarray(preds)
                    if labels_arr.size == 0:
                        return {"accuracy": 0.0, "macro_f1": 0.0}
                    accuracy = float((preds_arr == labels_arr).mean())

                    f1_scores: list[float] = []
                    for label_id in np.unique(labels_arr):
                        tp = float(np.sum((preds_arr == label_id) & (labels_arr == label_id)))
                        fp = float(np.sum((preds_arr == label_id) & (labels_arr != label_id)))
                        fn = float(np.sum((preds_arr != label_id) & (labels_arr == label_id)))
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                        f1_scores.append(f1)
                    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
                    return {"accuracy": accuracy, "macro_f1": macro_f1}

                trainer_kwargs["compute_metrics"] = compute_metrics
        else:
            def tokenize_rows(rows: dict[str, list[str]]) -> dict[str, Any]:
                return processor_tokenizer(
                    rows["text"],
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                )

            tokenized_train = train_text.map(tokenize_rows, batched=True, remove_columns=train_text.column_names)
            tokenized_eval = (
                eval_text.map(tokenize_rows, batched=True, remove_columns=eval_text.column_names)
                if eval_text is not None
                else None
            )
            trainer_kwargs["train_dataset"] = tokenized_train
            trainer_kwargs["eval_dataset"] = tokenized_eval if has_eval_records else None
            trainer_kwargs["data_collator"] = DataCollatorForLanguageModeling(tokenizer=processor_tokenizer, mlm=False)

        if distillation_enabled:
            trainer = trainer_cls(**trainer_kwargs)
        else:
            safe_trainer_kwargs, dropped_trainer_keys = _coerce_trainer_kwargs(trainer_kwargs, trainer_cls)
            if dropped_trainer_keys:
                warnings.append(
                    "Ignoring unsupported Trainer keys "
                    f"for transformers {hf_transformers.__version__}: {', '.join(dropped_trainer_keys)}."
                )
            trainer = trainer_cls(**safe_trainer_kwargs)

    trainer.add_callback(StreamingMetricCallback())
    trainer.add_callback(
        ObservabilityCallback(
            enabled=observability_enabled,
            log_steps=observability_log_steps,
            max_layers=observability_max_layers,
            probe_attention=observability_probe_attention,
            probe_top_k=observability_probe_top_k,
            probe_prompt=observability_probe_prompt,
        )
    )

    resume_checkpoint = _resolve_resume_checkpoint(
        output_dir=output_dir,
        resume_value=config.get("resume_from_checkpoint", "auto"),
        warnings=warnings,
    )
    if resume_checkpoint is not None:
        _emit_runtime_event(
            "resume_from_checkpoint",
            {"path": str(resume_checkpoint)},
        )
        train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    else:
        train_result = trainer.train()
    eval_metrics: dict[str, Any] = {}
    if has_eval_records:
        try:
            eval_metrics = trainer.evaluate()
        except Exception as eval_error:
            warnings.append(f"Evaluation step failed: {eval_error}")

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    log_history = [row for row in trainer.state.log_history if isinstance(row, dict)]
    metrics_path = output_dir / "metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for row in log_history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    checkpoints = _build_checkpoint_index(output_dir, log_history)
    final_train_loss = _last_metric_value(log_history, "loss")
    final_eval_loss = _last_metric_value(log_history, "eval_loss")
    final_eval_accuracy = _last_metric_value(log_history, "eval_accuracy")
    if final_eval_accuracy is None and isinstance(eval_metrics.get("eval_accuracy"), (int, float)):
        final_eval_accuracy = float(eval_metrics["eval_accuracy"])
    final_eval_macro_f1 = _last_metric_value(log_history, "eval_macro_f1")
    if final_eval_macro_f1 is None and isinstance(eval_metrics.get("eval_macro_f1"), (int, float)):
        final_eval_macro_f1 = float(eval_metrics["eval_macro_f1"])

    best_eval_loss = None
    if checkpoints:
        eval_losses = [ck["eval_loss"] for ck in checkpoints if isinstance(ck.get("eval_loss"), (int, float))]
        if eval_losses:
            best_eval_loss = float(min(eval_losses))

    task_metrics: dict[str, Any] = {}
    if normalized_task_type == "classification":
        task_metrics = {
            "final_eval_accuracy": final_eval_accuracy,
            "final_eval_macro_f1": final_eval_macro_f1,
            "label_space": label_space,
        }
    train_record_count = (
        len(train_preference)
        if training_mode in {"dpo", "orpo"}
        else len(train_text)
    )
    eval_record_count = (
        len(eval_preference)
        if training_mode in {"dpo", "orpo"} and eval_preference is not None
        else (len(eval_text) if eval_text is not None else 0)
    )
    training_mode_effective = (
        training_mode if alignment_native_objective else ("sft" if training_mode in {"dpo", "orpo"} else training_mode)
    )

    report = {
        "project_id": args.project,
        "experiment_id": args.experiment,
        "backend": resolved_backend,
        "training_mode_requested": training_mode,
        "training_mode_effective": training_mode_effective,
        "alignment_native_objective": alignment_native_objective,
        "distillation": {
            "enabled": bool(distillation_enabled),
            "teacher_model": distillation_teacher_model if distillation_enabled else None,
            "alpha": distillation_alpha if distillation_enabled else None,
            "temperature": distillation_temperature if distillation_enabled else None,
            "hidden_state_weight": (
                distillation_hidden_state_weight if distillation_enabled else None
            ),
            "hidden_state_loss": (
                distillation_hidden_state_loss if distillation_enabled else None
            ),
        },
        "task_type": normalized_task_type,
        "adapter_contract": data_contract,
        "base_model": args.base_model,
        "started_at": started_at,
        "finished_at": utcnow(),
        "config_path": str(config_path),
        "effective_config": dict(config),
        "train_file": str(train_file),
        "val_file": str(val_file) if val_file is not None else None,
        "model_dir": str(model_dir),
        "metrics_path": str(metrics_path),
        "epochs": num_epochs,
        "total_steps": int(getattr(trainer.state, "global_step", 0) or 0),
        "train_records": train_record_count,
        "eval_records": eval_record_count,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss or eval_metrics.get("eval_loss"),
        "final_eval_accuracy": final_eval_accuracy,
        "final_eval_macro_f1": final_eval_macro_f1,
        "best_eval_loss": best_eval_loss,
        "task_metrics": task_metrics,
        "checkpoints": checkpoints,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "attempt_index": attempt_index + 1,
        "attempts_total": total_attempts,
        "retry_history": list(retry_history),
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "runtime_environment": runtime_environment,
        "warnings": warnings,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utcnow()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    config_path = Path(args.config).expanduser().resolve() if args.config else output_dir / "training_config.json"
    train_file, val_file = _resolve_dataset_paths(args)

    base_config = _load_json(config_path)
    auto_oom_retry = bool(base_config.get("auto_oom_retry", True))
    max_oom_retries = _coerce_int(base_config.get("max_oom_retries"), 2, minimum=0)
    seq_shrink = _coerce_float(base_config.get("oom_retry_seq_shrink"), 0.75, minimum=0.1)
    seq_shrink = min(seq_shrink, 0.95)

    _emit_runtime_event(
        "runtime_preflight",
        {
            "training_mode": str(base_config.get("training_mode", "sft")),
            "task_type": str(base_config.get("task_type", "causal_lm")),
            "trainer_backend": str(base_config.get("trainer_backend", "auto")),
            "auto_oom_retry": auto_oom_retry,
            "max_oom_retries": max_oom_retries,
        },
    )

    attempt_config = dict(base_config)
    retry_history: list[dict[str, Any]] = []
    total_attempts = max_oom_retries + 1
    last_error: Exception | None = None

    for attempt_index in range(total_attempts):
        _emit_runtime_event(
            "attempt_start",
            {
                "attempt": attempt_index + 1,
                "total_attempts": total_attempts,
                "batch_size": _coerce_int(attempt_config.get("batch_size"), 1, minimum=1),
                "max_seq_length": _coerce_int(attempt_config.get("max_seq_length"), 128, minimum=128),
                "trainer_backend": str(attempt_config.get("trainer_backend", "auto")),
            },
        )
        try:
            return _run_training_attempt(
                args,
                config=attempt_config,
                output_dir=output_dir,
                model_dir=model_dir,
                config_path=config_path,
                train_file=train_file,
                val_file=val_file,
                started_at=started_at,
                attempt_index=attempt_index,
                total_attempts=total_attempts,
                retry_history=retry_history,
            )
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            if not auto_oom_retry or attempt_index >= total_attempts - 1 or not _is_cuda_oom_error(exc):
                raise

            next_config, changes = _next_oom_retry_config(
                attempt_config,
                seq_shrink=seq_shrink,
            )
            if not changes:
                raise

            retry_item = {
                "attempt": attempt_index + 1,
                "reason": str(exc),
                "changes": changes,
                "timestamp": utcnow(),
            }
            retry_history.append(retry_item)
            _emit_runtime_event("oom_retry", retry_item)
            attempt_config = next_config

    if last_error is not None:
        raise last_error
    raise RuntimeError("Training failed before first attempt.")


def main() -> int:
    args = parse_args()
    try:
        report = run_training(args)
        print(json.dumps({"status": "completed", "report": report}))
        return 0
    except Exception as e:
        traceback.print_exc()
        print(json.dumps({"status": "failed", "error": str(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
