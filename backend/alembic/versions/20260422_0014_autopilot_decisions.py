"""Add autopilot decision-log persistence.

Revision ID: 20260422_0014
Revises: 20260313_0013
Create Date: 2026-04-22 10:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260422_0014"
down_revision = "20260313_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autopilot_decisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=True),
        sa.Column("sequence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("stage", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False, server_default="info"),
        sa.Column("reason_code", sa.String(length=128), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("actor", sa.String(length=64), nullable=False, server_default="autopilot"),
        sa.Column("changed", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("safe", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("blocker", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("dry_run", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("intent", sa.Text(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_autopilot_decisions_run_id", "autopilot_decisions", ["run_id"], unique=False)
    op.create_index("ix_autopilot_decisions_project_id", "autopilot_decisions", ["project_id"], unique=False)
    op.create_index("ix_autopilot_decisions_stage", "autopilot_decisions", ["stage"], unique=False)
    op.create_index("ix_autopilot_decisions_status", "autopilot_decisions", ["status"], unique=False)
    op.create_index("ix_autopilot_decisions_action", "autopilot_decisions", ["action"], unique=False)
    op.create_index("ix_autopilot_decisions_reason_code", "autopilot_decisions", ["reason_code"], unique=False)
    op.create_index("ix_autopilot_decisions_created_at", "autopilot_decisions", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_autopilot_decisions_created_at", table_name="autopilot_decisions")
    op.drop_index("ix_autopilot_decisions_reason_code", table_name="autopilot_decisions")
    op.drop_index("ix_autopilot_decisions_action", table_name="autopilot_decisions")
    op.drop_index("ix_autopilot_decisions_status", table_name="autopilot_decisions")
    op.drop_index("ix_autopilot_decisions_stage", table_name="autopilot_decisions")
    op.drop_index("ix_autopilot_decisions_project_id", table_name="autopilot_decisions")
    op.drop_index("ix_autopilot_decisions_run_id", table_name="autopilot_decisions")
    op.drop_table("autopilot_decisions")
