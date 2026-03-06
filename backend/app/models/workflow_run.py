"""Workflow run ORM models for DAG execution tracking."""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class WorkflowNodeStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    graph_id: Mapped[str] = mapped_column(String(128), default="default-linear-v1")
    graph_version: Mapped[str] = mapped_column(String(32), default="1.0.0")
    execution_backend: Mapped[str] = mapped_column(String(32), default="local")
    status: Mapped[WorkflowRunStatus] = mapped_column(
        Enum(WorkflowRunStatus),
        default=WorkflowRunStatus.PENDING,
        nullable=False,
        index=True,
    )
    run_config: Mapped[dict | None] = mapped_column(JSON, default=dict)
    summary: Mapped[dict | None] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:
        return f"<WorkflowRun {self.id}: project={self.project_id} [{self.status.value}]>"


class WorkflowRunNode(Base):
    __tablename__ = "workflow_run_nodes"
    __table_args__ = (
        UniqueConstraint("run_id", "node_id", name="uq_workflow_run_nodes_run_node"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("workflow_runs.id"), nullable=False, index=True)
    node_id: Mapped[str] = mapped_column(String(255), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    step_type: Mapped[str] = mapped_column(String(128), nullable=False)
    execution_backend: Mapped[str] = mapped_column(String(32), default="local")
    status: Mapped[WorkflowNodeStatus] = mapped_column(
        Enum(WorkflowNodeStatus),
        default=WorkflowNodeStatus.PENDING,
        nullable=False,
        index=True,
    )
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=0)
    dependencies: Mapped[list | None] = mapped_column(JSON, default=list)
    input_artifacts: Mapped[list | None] = mapped_column(JSON, default=list)
    output_artifacts: Mapped[list | None] = mapped_column(JSON, default=list)
    runtime_requirements: Mapped[dict | None] = mapped_column(JSON, default=dict)
    missing_inputs: Mapped[list | None] = mapped_column(JSON, default=list)
    missing_runtime_requirements: Mapped[list | None] = mapped_column(JSON, default=list)
    published_artifact_keys: Mapped[list | None] = mapped_column(JSON, default=list)
    error_message: Mapped[str | None] = mapped_column(Text, default="")
    node_log: Mapped[list | None] = mapped_column(JSON, default=list)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:
        return f"<WorkflowRunNode {self.id}: run={self.run_id} node={self.node_id} [{self.status.value}]>"
