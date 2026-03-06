#!/usr/bin/env python3
"""External training runtime for real SLM finetuning.

This script runs supervised finetuning against prepared JSONL datasets.
It includes:
- trainer backend abstraction (`hf_trainer`, optional `trl_sft`)
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
            return {
                "text": _qa_to_chat_text(source, target, chat_template),
                "source_text": source,
                "target_text": target,
            }
        if source:
            return {"text": source, "source_text": source, "target_text": ""}
        return {"text": "", "source_text": "", "target_text": ""}

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
            return {
                "text": f"Text: {source}\nLabel: {label_raw}",
                "source_text": source,
                "target_text": label_raw,
            }
        if source:
            return {"text": source, "source_text": source, "target_text": ""}
        return {"text": "", "source_text": "", "target_text": ""}

    direct_text = _pick_first_text(row, ["text", "content"])
    if direct_text:
        return {"text": direct_text, "source_text": direct_text, "target_text": ""}

    question = _pick_first_text(row, ["question", "prompt", "instruction"])
    answer = _pick_first_text(row, ["answer", "completion", "output", "response"])
    if question and answer:
        rendered = _qa_to_chat_text(question, answer, chat_template)
        return {"text": rendered, "source_text": question, "target_text": answer}

    if question:
        optional_input = _pick_first_text(row, ["input"])
        if optional_input:
            return {
                "text": f"Instruction: {question}\nInput: {optional_input}",
                "source_text": question,
                "target_text": optional_input,
            }
        return {"text": question, "source_text": question, "target_text": ""}

    return {"text": "", "source_text": "", "target_text": ""}


def _row_has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _is_valid_adapted_row(row: dict[str, Any], task_type: str) -> bool:
    task = str(task_type or "causal_lm").strip().lower()
    if task == "causal_lm":
        return _row_has_text(row.get("text"))
    if task in {"seq2seq", "classification"}:
        return _row_has_text(row.get("source_text")) and _row_has_text(row.get("target_text"))
    return _row_has_text(row.get("text"))


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


def _coerce_training_arguments_kwargs(
    kwargs: dict[str, Any],
    training_args_cls: type,
) -> tuple[dict[str, Any], list[str]]:
    accepted = inspect.signature(training_args_cls.__init__).parameters
    normalized = dict(kwargs)

    alias_pairs = [
        ("evaluation_strategy", "eval_strategy"),
        ("eval_strategy", "evaluation_strategy"),
    ]
    for old_name, new_name in alias_pairs:
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


def _coerce_trainer_kwargs(
    kwargs: dict[str, Any],
    trainer_cls: type,
) -> tuple[dict[str, Any], list[str]]:
    accepted = inspect.signature(trainer_cls.__init__).parameters
    normalized = dict(kwargs)

    alias_pairs = [
        ("tokenizer", "processing_class"),
        ("processing_class", "tokenizer"),
    ]
    for old_name, new_name in alias_pairs:
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
    use_lora = bool(config.get("use_lora", False))
    lora_r = _coerce_int(config.get("lora_r"), 16, minimum=1)
    lora_alpha = _coerce_int(config.get("lora_alpha"), 32, minimum=1)
    lora_dropout = _coerce_float(config.get("lora_dropout"), 0.05, minimum=0.0)
    target_modules = config.get("target_modules", ["q_proj", "v_proj"])
    gradient_checkpointing = bool(config.get("gradient_checkpointing", True))
    want_flash_attention = bool(config.get("flash_attention", True))
    want_fp16 = bool(config.get("fp16", False))
    want_bf16 = bool(config.get("bf16", True))
    sequence_packing = bool(config.get("sequence_packing", True))
    max_train_samples = args.max_train_samples
    max_eval_samples = args.max_eval_samples

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

    warnings: list[str] = []
    normalized_task_type = _normalize_task_type(task_type, warnings)
    normalized_backend = _normalize_trainer_backend(trainer_backend, warnings)
    resolved_backend = _normalize_requested_backend(normalized_backend, warnings)
    if resolved_backend == "trl_sft" and normalized_task_type != "causal_lm":
        warnings.append(
            (
                "trainer_backend=trl_sft supports causal_lm only; "
                f"falling back to hf_trainer for task_type={normalized_task_type}."
            )
        )
        resolved_backend = "hf_trainer"
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

    eval_text = None
    if eval_ds is not None:
        eval_text = eval_ds.map(to_text_record, remove_columns=eval_ds.column_names)
        eval_text = eval_text.filter(lambda row: _is_valid_adapted_row(row, normalized_task_type))
        if len(eval_text) == 0:
            eval_text = None

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

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

    def _load_model_with_dtype_fallback(load_kwargs: dict[str, Any]):
        try:
            loaded = model_loader.from_pretrained(args.base_model, **load_kwargs)
            return loaded, load_kwargs
        except TypeError as type_error:
            if "dtype" in load_kwargs and "unexpected keyword argument 'dtype'" in str(type_error):
                fallback_kwargs = dict(load_kwargs)
                fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
                warnings.append("transformers runtime rejected 'dtype'; falling back to legacy 'torch_dtype'.")
                loaded = model_loader.from_pretrained(args.base_model, **fallback_kwargs)
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

    has_eval_records = eval_text is not None and len(eval_text) > 0

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
        "overwrite_output_dir": True,
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
        "remove_unused_columns": resolved_backend == "hf_trainer",
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

    if resolved_backend == "trl_sft":
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

        if normalized_task_type == "seq2seq":
            def tokenize_rows(rows: dict[str, list[str]]) -> dict[str, Any]:
                sources = [str(x) for x in rows["source_text"]]
                targets = [str(x) for x in rows["target_text"]]
                try:
                    return tokenizer(
                        sources,
                        text_target=targets,
                        truncation=True,
                        max_length=max_seq_length,
                        max_target_length=max_seq_length,
                        padding=False,
                    )
                except TypeError:
                    model_inputs = tokenizer(
                        sources,
                        truncation=True,
                        max_length=max_seq_length,
                        padding=False,
                    )
                    if hasattr(tokenizer, "as_target_tokenizer"):
                        with tokenizer.as_target_tokenizer():
                            labels = tokenizer(
                                targets,
                                truncation=True,
                                max_length=max_seq_length,
                                padding=False,
                            )
                    else:
                        labels = tokenizer(
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
            trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        elif normalized_task_type == "classification":
            def tokenize_rows(rows: dict[str, list[str]]) -> dict[str, Any]:
                sources = [str(x) for x in rows["source_text"]]
                labels: list[int] = []
                for raw in rows["target_text"]:
                    key = str(raw or "").strip()
                    if key not in label_to_id:
                        raise ValueError(f"Unknown classification label encountered: '{key}'")
                    labels.append(label_to_id[key])
                tokenized = tokenizer(
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
            trainer_kwargs["data_collator"] = DataCollatorWithPadding(tokenizer=tokenizer)

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
                return tokenizer(
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
            trainer_kwargs["data_collator"] = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        safe_trainer_kwargs, dropped_trainer_keys = _coerce_trainer_kwargs(trainer_kwargs, Trainer)
        if dropped_trainer_keys:
            warnings.append(
                "Ignoring unsupported Trainer keys "
                f"for transformers {hf_transformers.__version__}: {', '.join(dropped_trainer_keys)}."
            )
        trainer = Trainer(**safe_trainer_kwargs)

    trainer.add_callback(StreamingMetricCallback())

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

    report = {
        "project_id": args.project,
        "experiment_id": args.experiment,
        "backend": resolved_backend,
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
        "train_records": len(train_text),
        "eval_records": len(eval_text) if eval_text is not None else 0,
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
