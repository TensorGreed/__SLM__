"""Canonical RunEvent schema (priority.md P31, Wave G).

Creates:
  - run_events — append-only event log spanning every stage. Powers the
    P32 unified timeline, P33 failure clustering, P34 support bundles,
    and the Wave I P44 audit explorer.

Revision ID: 20260507_0025
Revises: 20260504_0024
Create Date: 2026-05-07 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260507_0025"
down_revision = "20260504_0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("parent_run_id", sa.String(length=128), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("reason_code", sa.String(length=128), nullable=True),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_run_events_project_id", "run_events", ["project_id"], unique=False
    )
    op.create_index(
        "ix_run_events_run_id", "run_events", ["run_id"], unique=False
    )
    op.create_index(
        "ix_run_events_parent_run_id",
        "run_events",
        ["parent_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_run_events_stage", "run_events", ["stage"], unique=False
    )
    op.create_index(
        "ix_run_events_severity", "run_events", ["severity"], unique=False
    )
    op.create_index(
        "ix_run_events_reason_code",
        "run_events",
        ["reason_code"],
        unique=False,
    )
    op.create_index("ix_run_events_ts", "run_events", ["ts"], unique=False)
    op.create_index(
        "ix_run_events_created_at", "run_events", ["created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_run_events_created_at", table_name="run_events")
    op.drop_index("ix_run_events_ts", table_name="run_events")
    op.drop_index("ix_run_events_reason_code", table_name="run_events")
    op.drop_index("ix_run_events_severity", table_name="run_events")
    op.drop_index("ix_run_events_stage", table_name="run_events")
    op.drop_index("ix_run_events_parent_run_id", table_name="run_events")
    op.drop_index("ix_run_events_run_id", table_name="run_events")
    op.drop_index("ix_run_events_project_id", table_name="run_events")
    op.drop_table("run_events")
