"""Training recipe catalog and resolution helpers.

Recipes are reusable config patches for common SLM training patterns.
They are intentionally domain-agnostic and can be combined with project/domain
runtime defaults through existing effective-config resolution.
"""

from __future__ import annotations

import copy
from typing import Any


def _deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


_BUILTIN_RECIPES: list[dict[str, Any]] = [
    {
        "recipe_id": "recipe.sft.safe",
        "display_name": "SFT Safe Bootstrap",
        "description": "Most conservative SFT profile for first successful run on unknown hardware.",
        "category": "sft",
        "tags": ["safe", "bootstrap", "oom-resilient"],
        "required_fields": ["base_model"],
        "config_patch": {
            "task_type": "causal_lm",
            "trainer_backend": "hf_trainer",
            "training_runtime_id": "auto",
            "chat_template": "llama3",
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 1024,
            "num_epochs": 3,
            "learning_rate": 2e-4,
            "optimizer": "adamw_torch",
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "sequence_packing": False,
            "gradient_checkpointing": True,
            "flash_attention": False,
            "fp16": False,
            "bf16": True,
            "auto_oom_retry": True,
            "max_oom_retries": 3,
            "oom_retry_seq_shrink": 0.75,
            "save_steps": 100,
            "eval_steps": 100,
        },
    },
    {
        "recipe_id": "recipe.sft.balanced",
        "display_name": "SFT Balanced",
        "description": "General-purpose supervised fine-tuning baseline for most instruction datasets.",
        "category": "sft",
        "tags": ["balanced", "general"],
        "required_fields": ["base_model"],
        "config_patch": {
            "task_type": "causal_lm",
            "trainer_backend": "auto",
            "training_runtime_id": "auto",
            "chat_template": "llama3",
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "num_epochs": 3,
            "learning_rate": 2e-4,
            "optimizer": "paged_adamw_8bit",
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "sequence_packing": True,
            "gradient_checkpointing": True,
            "flash_attention": True,
            "fp16": False,
            "bf16": True,
            "auto_oom_retry": True,
            "max_oom_retries": 2,
            "oom_retry_seq_shrink": 0.75,
            "save_steps": 100,
            "eval_steps": 100,
        },
    },
    {
        "recipe_id": "recipe.lora.fast",
        "display_name": "LoRA Fast Iteration",
        "description": "Fast iteration recipe with smaller adapters and shorter context for quicker experiments.",
        "category": "lora",
        "tags": ["lora", "fast", "iteration"],
        "required_fields": ["base_model"],
        "config_patch": {
            "task_type": "causal_lm",
            "trainer_backend": "auto",
            "training_runtime_id": "auto",
            "chat_template": "llama3",
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "max_seq_length": 1024,
            "num_epochs": 2,
            "learning_rate": 3e-4,
            "optimizer": "paged_adamw_8bit",
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "sequence_packing": True,
            "gradient_checkpointing": True,
            "flash_attention": True,
            "fp16": False,
            "bf16": True,
            "auto_oom_retry": True,
            "max_oom_retries": 2,
            "oom_retry_seq_shrink": 0.8,
            "save_steps": 50,
            "eval_steps": 50,
        },
    },
    {
        "recipe_id": "recipe.distill.logits",
        "display_name": "Distillation (Teacher->Student)",
        "description": "SFT + KL distillation against a teacher model for stronger small-model transfer.",
        "category": "distillation",
        "tags": ["distillation", "teacher-student", "logits"],
        "required_fields": ["base_model", "distillation_teacher_model"],
        "config_patch": {
            "task_type": "causal_lm",
            "trainer_backend": "hf_trainer",
            "training_runtime_id": "auto",
            "chat_template": "llama3",
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "num_epochs": 3,
            "learning_rate": 2e-4,
            "optimizer": "paged_adamw_8bit",
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "sequence_packing": True,
            "gradient_checkpointing": True,
            "flash_attention": True,
            "fp16": False,
            "bf16": True,
            "distillation_enabled": True,
            "distillation_alpha": 0.6,
            "distillation_temperature": 2.0,
            "distillation_hidden_state_weight": 0.0,
            "distillation_hidden_state_loss": "mse",
            "save_steps": 100,
            "eval_steps": 100,
        },
    },
    {
        "recipe_id": "recipe.classification.encoder",
        "display_name": "Classification Baseline",
        "description": "Generic text classification baseline tuned for encoder/decoder model families.",
        "category": "classification",
        "tags": ["classification", "baseline"],
        "required_fields": ["base_model"],
        "config_patch": {
            "task_type": "classification",
            "trainer_backend": "hf_trainer",
            "training_runtime_id": "auto",
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "max_seq_length": 512,
            "num_epochs": 5,
            "learning_rate": 5e-5,
            "optimizer": "adamw_torch",
            "use_lora": False,
            "sequence_packing": False,
            "gradient_checkpointing": False,
            "flash_attention": False,
            "fp16": False,
            "bf16": False,
            "auto_oom_retry": True,
            "max_oom_retries": 1,
            "oom_retry_seq_shrink": 0.8,
            "save_steps": 100,
            "eval_steps": 100,
        },
    },
    {
        "recipe_id": "recipe.seq2seq.default",
        "display_name": "Seq2Seq Baseline",
        "description": "General seq2seq recipe for instruction rewriting, QA generation, and transformation tasks.",
        "category": "seq2seq",
        "tags": ["seq2seq", "baseline"],
        "required_fields": ["base_model"],
        "config_patch": {
            "task_type": "seq2seq",
            "trainer_backend": "hf_trainer",
            "training_runtime_id": "auto",
            "chat_template": "chatml",
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 1024,
            "num_epochs": 3,
            "learning_rate": 1e-4,
            "optimizer": "adamw_torch",
            "use_lora": False,
            "sequence_packing": False,
            "gradient_checkpointing": True,
            "flash_attention": False,
            "fp16": False,
            "bf16": True,
            "auto_oom_retry": True,
            "max_oom_retries": 2,
            "oom_retry_seq_shrink": 0.75,
            "save_steps": 100,
            "eval_steps": 100,
        },
    },
]


def _normalize_recipe_id(value: str | None) -> str:
    return str(value or "").strip().lower()


def _recipe_summary(recipe: dict[str, Any], *, include_patch: bool) -> dict[str, Any]:
    payload = {
        "recipe_id": str(recipe.get("recipe_id", "")),
        "display_name": str(recipe.get("display_name", "")),
        "description": str(recipe.get("description", "")),
        "category": str(recipe.get("category", "general")),
        "tags": [str(item) for item in list(recipe.get("tags") or []) if str(item).strip()],
        "required_fields": [
            str(item) for item in list(recipe.get("required_fields") or []) if str(item).strip()
        ],
    }
    if include_patch:
        payload["config_patch"] = _deepcopy(dict(recipe.get("config_patch") or {}))
    return payload


def list_training_recipes(*, include_patch: bool = False) -> list[dict[str, Any]]:
    """List built-in recipe metadata."""
    return [_recipe_summary(item, include_patch=include_patch) for item in _BUILTIN_RECIPES]


def get_training_recipe(recipe_id: str) -> dict[str, Any] | None:
    """Lookup recipe by id."""
    token = _normalize_recipe_id(recipe_id)
    if not token:
        return None
    for recipe in _BUILTIN_RECIPES:
        if _normalize_recipe_id(str(recipe.get("recipe_id"))) == token:
            return _deepcopy(recipe)
    return None


def resolve_training_recipe(
    recipe_id: str,
    *,
    base_config: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve final recipe config by layering base config + recipe patch + overrides."""
    recipe = get_training_recipe(recipe_id)
    if recipe is None:
        available = ", ".join(sorted(str(item.get("recipe_id", "")) for item in _BUILTIN_RECIPES))
        raise ValueError(f"Unknown recipe_id '{recipe_id}'. Available recipes: {available}")

    resolved_config: dict[str, Any] = {}
    if isinstance(base_config, dict):
        resolved_config.update(_deepcopy(base_config))
    resolved_config.update(_deepcopy(dict(recipe.get("config_patch") or {})))
    if isinstance(overrides, dict):
        resolved_config.update(_deepcopy(overrides))

    missing_required: list[str] = []
    for field in list(recipe.get("required_fields") or []):
        key = str(field).strip()
        if not key:
            continue
        value = resolved_config.get(key)
        if value is None:
            missing_required.append(key)
            continue
        if isinstance(value, str) and not value.strip():
            missing_required.append(key)

    return {
        "recipe": _recipe_summary(recipe, include_patch=True),
        "resolved_config": resolved_config,
        "missing_required_fields": sorted(set(missing_required)),
    }
