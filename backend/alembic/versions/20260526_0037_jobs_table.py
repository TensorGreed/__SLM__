"""Jobs table for the background-task framework (Hardening Phase H1).

Adds:
  - ``jobs`` — generic background job records. Long-running LLM /
    cloning / training operations are scheduled via this table so
    the UI's notification bell can render progress and the user is
    no longer blocked on a 30-180s HTTP request.

Idempotent — checks for the table's existence before creating
(mirrors the safety guard added in 0036 after the partial-apply
recovery).

Revision ID: 20260526_0037
Revises: 20260526_0036
Create Date: 2026-05-26 09:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260526_0037"
down_revision = "20260526_0036"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return name in inspector.get_table_names()


def upgrade() -> None:
    if _table_exists("jobs"):
        return
    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "QUEUED",
                "RUNNING",
                "SUCCEEDED",
                "FAILED",
                "CANCELLED",
                name="jobstatus",
            ),
            nullable=False,
            server_default="QUEUED",
        ),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("progress_message", sa.Text(), nullable=True),
        sa.Column(
            "project_id",
            sa.Integer(),
            sa.ForeignKey("projects.id"),
            nullable=True,
        ),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "queued_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_project_id", "jobs", ["project_id"])
    op.create_index("ix_jobs_queued_at", "jobs", ["queued_at"])


def downgrade() -> None:
    if not _table_exists("jobs"):
        return
    op.drop_index("ix_jobs_queued_at", table_name="jobs")
    op.drop_index("ix_jobs_project_id", table_name="jobs")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_table("jobs")
