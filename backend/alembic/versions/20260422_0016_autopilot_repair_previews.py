"""Add autopilot repair-preview persistence for explicit apply step.

Revision ID: 20260422_0016
Revises: 20260422_0015
Create Date: 2026-04-22 14:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260422_0016"
down_revision = "20260422_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autopilot_repair_previews",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("plan_token", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("intent", sa.Text(), nullable=True),
        sa.Column("request_payload", sa.JSON(), nullable=True),
        sa.Column("config_diff", sa.JSON(), nullable=True),
        sa.Column("dry_run_response", sa.JSON(), nullable=True),
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("applied_run_id", sa.String(length=64), nullable=True),
        sa.Column("applied_by", sa.String(length=64), nullable=True),
        sa.Column("applied_reason", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("plan_token", name="uq_autopilot_repair_previews_plan_token"),
    )
    op.create_index(
        "ix_autopilot_repair_previews_plan_token",
        "autopilot_repair_previews",
        ["plan_token"],
        unique=False,
    )
    op.create_index(
        "ix_autopilot_repair_previews_project_id",
        "autopilot_repair_previews",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_autopilot_repair_previews_created_at",
        "autopilot_repair_previews",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_autopilot_repair_previews_expires_at",
        "autopilot_repair_previews",
        ["expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_autopilot_repair_previews_applied_at",
        "autopilot_repair_previews",
        ["applied_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_autopilot_repair_previews_applied_at", table_name="autopilot_repair_previews")
    op.drop_index("ix_autopilot_repair_previews_expires_at", table_name="autopilot_repair_previews")
    op.drop_index("ix_autopilot_repair_previews_created_at", table_name="autopilot_repair_previews")
    op.drop_index("ix_autopilot_repair_previews_project_id", table_name="autopilot_repair_previews")
    op.drop_index("ix_autopilot_repair_previews_plan_token", table_name="autopilot_repair_previews")
    op.drop_table("autopilot_repair_previews")
