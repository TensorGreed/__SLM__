"""Add autopilot snapshot persistence for rollback.

Revision ID: 20260422_0015
Revises: 20260422_0014
Create Date: 2026-04-22 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260422_0015"
down_revision = "20260422_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autopilot_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("decision_sequence", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=True),
        sa.Column("snapshot_type", sa.String(length=64), nullable=False, server_default="autopilot_generic"),
        sa.Column("pre_state", sa.JSON(), nullable=True),
        sa.Column("post_state", sa.JSON(), nullable=True),
        sa.Column("rollback_actions", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("restored_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("restored_by", sa.String(length=64), nullable=True),
        sa.Column("restored_reason", sa.Text(), nullable=True),
        sa.Column("restored_decision_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["restored_decision_id"], ["autopilot_decisions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "decision_sequence",
            name="uq_autopilot_snapshots_run_sequence",
        ),
    )
    op.create_index("ix_autopilot_snapshots_run_id", "autopilot_snapshots", ["run_id"], unique=False)
    op.create_index("ix_autopilot_snapshots_project_id", "autopilot_snapshots", ["project_id"], unique=False)
    op.create_index("ix_autopilot_snapshots_created_at", "autopilot_snapshots", ["created_at"], unique=False)
    op.create_index("ix_autopilot_snapshots_expires_at", "autopilot_snapshots", ["expires_at"], unique=False)
    op.create_index("ix_autopilot_snapshots_restored_at", "autopilot_snapshots", ["restored_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_autopilot_snapshots_restored_at", table_name="autopilot_snapshots")
    op.drop_index("ix_autopilot_snapshots_expires_at", table_name="autopilot_snapshots")
    op.drop_index("ix_autopilot_snapshots_created_at", table_name="autopilot_snapshots")
    op.drop_index("ix_autopilot_snapshots_project_id", table_name="autopilot_snapshots")
    op.drop_index("ix_autopilot_snapshots_run_id", table_name="autopilot_snapshots")
    op.drop_table("autopilot_snapshots")
