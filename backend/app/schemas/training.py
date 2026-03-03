"""Pydantic schemas for training configuration and experiment APIs."""

from datetime import datetime
from pydantic import BaseModel, Field

from app.models.experiment import ExperimentStatus, TrainingMode


class TrainingConfig(BaseModel):
    """Training hyperparameters and configuration."""
    base_model: str = Field(..., description="HuggingFace model ID or local path")
    training_mode: TrainingMode = TrainingMode.SFT
    chat_template: str = Field("llama3", description="Chat template format (llama3, chatml, zephyr, phi3)")
    
    # Hyperparameters
    batch_size: int = Field(4, ge=1, le=256)
    gradient_accumulation_steps: int = Field(4, ge=1)
    learning_rate: float = Field(2e-4, gt=0)
    optimizer: str = Field("paged_adamw_8bit", description="Optimizer type")
    lr_scheduler: str = Field("cosine", description="Learning rate scheduler")
    num_epochs: int = Field(3, ge=1, le=100)
    max_seq_length: int = Field(2048, ge=128, le=32768)
    warmup_ratio: float = Field(0.03, ge=0, le=1)
    weight_decay: float = Field(0.01, ge=0)
    sequence_packing: bool = Field(True, description="Pack multiple sequences up to max_seq_length")
    
    # LoRA
    use_lora: bool = True
    lora_r: int = Field(16, ge=1, le=256)
    lora_alpha: int = Field(32, ge=1)
    lora_dropout: float = Field(0.05, ge=0, le=1)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Compute / System
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    
    # Checkpointing
    save_steps: int = Field(100, ge=1)
    eval_steps: int = Field(100, ge=1)
    early_stopping_patience: int = Field(3, ge=1)
    
    seed: int = 42


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

    model_config = {"from_attributes": True}


class TrainingMetricsSnapshot(BaseModel):
    """Real-time training metrics pushed via WebSocket."""
    experiment_id: int
    epoch: float
    step: int
    train_loss: float
    eval_loss: float | None = None
    learning_rate: float | None = None
