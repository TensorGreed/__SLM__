"""Add workflow run tracking tables.

Revision ID: 20260305_0006
Revises: 20260305_0005
Create Date: 2026-03-05 17:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260305_0006"
down_revision = "20260305_0005"
branch_labels = None
depends_on = None


workflow_run_status_enum = sa.Enum(
    "pending",
    "running",
    "completed",
    "failed",
    "blocked",
    "cancelled",
    name="workflowrunstatus",
)

workflow_node_status_enum = sa.Enum(
    "pending",
    "running",
    "completed",
    "failed",
    "blocked",
    "skipped",
    name="workflownodestatus",
)


def upgrade() -> None:
    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("graph_id", sa.String(length=128), nullable=False),
        sa.Column("graph_version", sa.String(length=32), nullable=False),
        sa.Column("execution_backend", sa.String(length=32), nullable=False),
        sa.Column("status", workflow_run_status_enum, nullable=False),
        sa.Column("run_config", sa.JSON(), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workflow_runs_project_id", "workflow_runs", ["project_id"], unique=False)
    op.create_index("ix_workflow_runs_status", "workflow_runs", ["status"], unique=False)
    op.create_index("ix_workflow_runs_created_at", "workflow_runs", ["created_at"], unique=False)

    op.create_table(
        "workflow_run_nodes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("step_type", sa.String(length=128), nullable=False),
        sa.Column("execution_backend", sa.String(length=32), nullable=False),
        sa.Column("status", workflow_node_status_enum, nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_retries", sa.Integer(), nullable=False),
        sa.Column("dependencies", sa.JSON(), nullable=True),
        sa.Column("input_artifacts", sa.JSON(), nullable=True),
        sa.Column("output_artifacts", sa.JSON(), nullable=True),
        sa.Column("runtime_requirements", sa.JSON(), nullable=True),
        sa.Column("missing_inputs", sa.JSON(), nullable=True),
        sa.Column("missing_runtime_requirements", sa.JSON(), nullable=True),
        sa.Column("published_artifact_keys", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("node_log", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["workflow_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "node_id", name="uq_workflow_run_nodes_run_node"),
    )
    op.create_index("ix_workflow_run_nodes_run_id", "workflow_run_nodes", ["run_id"], unique=False)
    op.create_index("ix_workflow_run_nodes_status", "workflow_run_nodes", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_workflow_run_nodes_status", table_name="workflow_run_nodes")
    op.drop_index("ix_workflow_run_nodes_run_id", table_name="workflow_run_nodes")
    op.drop_table("workflow_run_nodes")

    op.drop_index("ix_workflow_runs_created_at", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_status", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_project_id", table_name="workflow_runs")
    op.drop_table("workflow_runs")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("DROP TYPE IF EXISTS workflownodestatus"))
        op.execute(sa.text("DROP TYPE IF EXISTS workflowrunstatus"))
