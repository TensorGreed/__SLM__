"""Pydantic schemas for training configuration and experiment APIs."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.models.experiment import ExperimentStatus, TrainingMode


class TrainingConfig(BaseModel):
    """Training hyperparameters and configuration."""
    base_model: str = Field(..., description="HuggingFace model ID or local path")
    training_mode: TrainingMode = TrainingMode.SFT
    chat_template: str = Field(
        "llama3",
        description=(
            "Fallback prompt preset (llama3, chatml, zephyr, phi3) used only "
            "when use_tokenizer_chat_template is False."
        ),
    )
    use_tokenizer_chat_template: bool = Field(
        True,
        description=(
            "Train QA-family rows (qa / chat_sft / instruction_sft) in the "
            "exact prompt shape held-out eval and serving use: the base "
            "model's own ``tokenizer.apply_chat_template`` (raw prompt when "
            "the tokenizer has none). ``chat_template`` is only used when "
            "this is False. No effect on wraps_own_prompt rows "
            "(Classification / Structured / RAG / Seq2Seq / Vision / Audio), "
            "which carry their handler's prompt already."
        ),
    )
    task_type: str = Field(
        "causal_lm",
        description="Task adapter type (causal_lm, seq2seq, classification)",
    )
    trainer_backend: str = Field(
        "auto",
        description="Trainer backend (auto, hf_trainer, trl_sft)",
    )
    training_runtime_id: str = Field(
        "auto",
        description="Training runtime plugin id (auto resolves server default).",
    )
    recommended_starting_checkpoint: str = Field(
        "",
        max_length=255,
        description=(
            "Registry name of a pre-fine-tuned warm-start checkpoint to resolve as the "
            "effective starting weights. Falls back to base_model when absent/unavailable."
        ),
    )

    # Hyperparameters
    batch_size: int = Field(4, ge=1, le=256)
    gradient_accumulation_steps: int = Field(4, ge=1)
    learning_rate: float = Field(2e-4, gt=0)
    optimizer: str = Field("paged_adamw_8bit", description="Optimizer type")
    lr_scheduler: str = Field("cosine", description="Learning rate scheduler")
    num_epochs: int = Field(3, ge=1, le=100)
    class_weighting: Literal["auto", "none"] = Field(
        "auto",
        description=(
            "Classification head only: weight the loss by inverse class "
            "frequency when the largest class is >= 3x the smallest "
            "(\"auto\"), or never (\"none\")."
        ),
    )
    auto_epochs: bool = Field(
        True,
        description=(
            "Scale epochs (and, for small datasets, gradient accumulation — "
            "it shrinks before epochs are added) to the training-row count "
            "via ``training_epoch_policy``; "
            "``num_epochs`` is ignored while on. Set False to use "
            "``num_epochs`` exactly."
        ),
    )
    max_seq_length: int = Field(2048, ge=128, le=32768)
    warmup_ratio: float = Field(0.03, ge=0, le=1)
    weight_decay: float = Field(0.01, ge=0)
    sequence_packing: bool = Field(True, description="Pack multiple sequences up to max_seq_length")
    
    # LoRA
    use_lora: bool = True
    lora_r: int = Field(16, ge=1, le=256)
    lora_alpha: int = Field(32, ge=1)
    lora_dropout: float = Field(0.05, ge=0, le=1)
    # "auto" (default): every linear layer for small causal-LM models, the
    # classic q_proj/v_proj otherwise — resolved by ``lora_target_policy``
    # in train.py from the loaded model's size. Or a list of module names,
    # or "all-linear" (every linear layer — the continued-pretraining default).
    target_modules: list[str] | Literal["all-linear", "auto"] = "auto"
    
    # Compute / System
    fp16: bool = False
    bf16: bool = True
    # "auto" (default): off for models up to 2B parameters (recompute buys no
    # memory there and costs ~1.6x per step), on for larger ones — resolved by
    # ``gradient_checkpointing_policy`` in train.py from the loaded model.
    gradient_checkpointing: bool | Literal["auto"] = "auto"
    flash_attention: bool = True

    # Runtime planner / retry
    auto_oom_retry: bool = Field(
        True,
        description="Auto-retry CUDA OOM with smaller memory profile.",
    )
    max_oom_retries: int = Field(2, ge=0, le=5)
    oom_retry_seq_shrink: float = Field(
        0.75,
        gt=0.1,
        lt=1.0,
        description="Per-retry max_seq_length shrink factor.",
    )
    multimodal_require_media: bool = Field(
        False,
        description=(
            "Strict multimodal loading mode: require resolved local media assets and disable text-only "
            "fallback markers for sampled vision/audio rows."
        ),
    )

    # Alignment (DPO/ORPO)
    alignment_auto_filter: bool = Field(
        False,
        description="Auto-run judge quality filter and train on kept preference rows.",
    )
    alignment_quality_threshold: float = Field(
        3.0,
        ge=1.0,
        le=5.0,
        description="Judge score threshold for keeping preference pairs.",
    )
    alignment_beta: float = Field(
        0.1,
        gt=0.0,
        le=5.0,
        description="Pairwise objective beta for DPO/ORPO TRL trainers.",
    )
    alignment_max_prompt_length: int = Field(
        1024,
        ge=32,
        le=32768,
        description="Prompt token cap for DPO/ORPO processing.",
    )
    alignment_max_length: int = Field(
        2048,
        ge=64,
        le=32768,
        description="Total token cap for DPO/ORPO prompt+response processing.",
    )
    alignment_min_keep_ratio: float = Field(
        0.4,
        ge=0.05,
        le=1.0,
        description="Minimum keep ratio required when applying alignment filter.",
    )
    alignment_dataset_path: str = Field(
        "",
        max_length=4096,
        description="Optional project-relative path to a preference JSONL file for DPO/ORPO.",
    )

    # Distillation
    distillation_enabled: bool = Field(
        False,
        description="Enable teacher-student distillation objective during SFT training.",
    )
    distillation_teacher_model: str = Field(
        "",
        max_length=255,
        description="Teacher model id/path used for distillation guidance.",
    )
    distillation_alpha: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Weight for supervised CE loss when distillation is enabled (KD weight = 1-alpha).",
    )
    distillation_temperature: float = Field(
        2.0,
        gt=0.1,
        le=20.0,
        description="Softmax temperature for teacher-student logit distillation.",
    )
    distillation_hidden_state_weight: float = Field(
        0.0,
        ge=0.0,
        le=10.0,
        description="Optional hidden-state alignment weight (0 disables hidden-state guidance).",
    )
    distillation_hidden_state_loss: str = Field(
        "mse",
        pattern="^(mse|cosine)$",
        description="Hidden-state loss type when distillation_hidden_state_weight > 0 (mse|cosine).",
    )
    
    # Checkpointing
    save_steps: int = Field(100, ge=1)
    eval_steps: int = Field(100, ge=1)
    early_stopping_patience: int = Field(3, ge=1)

    seed: int = 42

    # Multi-seed variance reporting (Quality-Lift phase 1).
    # ``num_seeds=1`` (default) preserves single-seed behavior — no fan-out, no
    # aggregate row. When ``num_seeds>1`` or ``seeds`` is set explicitly,
    # ``training_service.start_training`` creates N child Experiments sharing a
    # ``seed_group_id`` and rolls their EvalResults into a single aggregate row
    # carrying ``{mean, std, min, max, n}`` per metric so gates can be judged
    # honestly (lower-bound = mean−std, per the no-vanity-metrics rule).
    seeds: list[int] | None = Field(
        None,
        description=(
            "Explicit seed list. When set, runs len(seeds) independent "
            "trainings with these exact values. Wins over num_seeds."
        ),
    )
    num_seeds: int = Field(
        1,
        ge=1,
        le=8,
        description=(
            "Number of independent training runs for variance reporting. "
            "Ignored when seeds is set explicitly. When >1, seeds are derived "
            "deterministically from the base seed: [seed, seed+1, seed+2, ...]."
        ),
    )
    parallel_seeds: bool = Field(
        False,
        description=(
            "Run multi-seed children in parallel (concurrent GPU contention) "
            "vs. sequentially. Default False — on single-GPU boxes parallel "
            "buys nothing and can OOM. Flip on only for multi-GPU runtimes."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _explicit_epochs_disable_auto(cls, data: Any) -> Any:
        # A caller that names ``num_epochs`` (CLI ``--num-epochs``, API
        # config, UI field the user touched) means that exact count.
        if isinstance(data, dict) and "num_epochs" in data and "auto_epochs" not in data:
            data = {**data, "auto_epochs": False}
        return data


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    config: TrainingConfig


class ExperimentResponse(BaseModel):
    id: int
    project_id: int
    name: str
    description: str | None
    status: ExperimentStatus
    training_mode: TrainingMode
    base_model: str
    config: dict | None
    final_train_loss: float | None
    final_eval_loss: float | None
    total_epochs: int | None
    total_steps: int | None
    output_dir: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    domain_pack_applied: str | None = None
    domain_pack_source: str | None = None
    domain_profile_applied: str | None = None
    domain_profile_source: str | None = None
    profile_training_defaults: dict[str, Any] | None = None
    resolved_training_config: dict[str, Any] | None = None
    profile_defaults_applied: list[str] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class TrainingMetricsSnapshot(BaseModel):
    """Real-time training metrics pushed via WebSocket."""
    experiment_id: int
    epoch: float
    step: int
    train_loss: float
    eval_loss: float | None = None
    learning_rate: float | None = None
