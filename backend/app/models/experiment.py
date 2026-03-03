"""Experiment, Checkpoint, and EvalResult ORM models."""

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExperimentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingMode(str, enum.Enum):
    SFT = "sft"
    DOMAIN_PRETRAIN = "domain_pretrain"
    DPO = "dpo"
    ORPO = "orpo"


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default="")
    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus), default=ExperimentStatus.PENDING
    )
    training_mode: Mapped[TrainingMode] = mapped_column(
        Enum(TrainingMode), default=TrainingMode.SFT
    )
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    # Training hyperparams
    config: Mapped[dict | None] = mapped_column(JSON, default=dict)
    # Tracked metrics
    final_train_loss: Mapped[float | None] = mapped_column(Float, default=None)
    final_eval_loss: Mapped[float | None] = mapped_column(Float, default=None)
    total_epochs: Mapped[int | None] = mapped_column(Integer, default=None)
    total_steps: Mapped[int | None] = mapped_column(Integer, default=None)
    output_dir: Mapped[str | None] = mapped_column(String(1024), default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # Relationships
    project = relationship("Project", back_populates="experiments")
    checkpoints = relationship("Checkpoint", back_populates="experiment", cascade="all, delete-orphan")
    eval_results = relationship("EvalResult", back_populates="experiment", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Experiment {self.id}: {self.name} [{self.status.value}]>"


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    train_loss: Mapped[float | None] = mapped_column(Float, default=None)
    eval_loss: Mapped[float | None] = mapped_column(Float, default=None)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    is_best: Mapped[bool] = mapped_column(default=False)
    metrics: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    experiment = relationship("Experiment", back_populates="checkpoints")

    def __repr__(self) -> str:
        return f"<Checkpoint {self.experiment_id}@epoch{self.epoch}>"


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    eval_type: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    pass_rate: Mapped[float | None] = mapped_column(Float, default=None)
    risk_severity: Mapped[str | None] = mapped_column(String(32), default=None)
    details: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    experiment = relationship("Experiment", back_populates="eval_results")

    def __repr__(self) -> str:
        return f"<EvalResult {self.id}: {self.eval_type} on {self.dataset_name}>"
