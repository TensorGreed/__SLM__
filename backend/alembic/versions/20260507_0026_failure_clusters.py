"""Persisted failure clusters (priority.md P33, Wave G).

Creates:
  - failure_clusters — one row per (project_id, stage, reason_code,
    signature) tuple. Aggregates ``severity in {error, critical}``
    RunEvents grouped by their normalised summary signature.

Revision ID: 20260507_0026
Revises: 20260507_0025
Create Date: 2026-05-07 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260507_0026"
down_revision = "20260507_0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "failure_clusters",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("reason_code", sa.String(length=128), nullable=False),
        sa.Column("signature", sa.String(length=64), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exemplar_event_ids", sa.JSON(), nullable=True),
        sa.Column("exemplar_summaries", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "last_computed_at", sa.DateTime(timezone=True), nullable=False
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "stage",
            "reason_code",
            "signature",
            name="uq_failure_clusters_pid_stage_reason_sig",
        ),
    )
    op.create_index(
        "ix_failure_clusters_project_id",
        "failure_clusters",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_failure_clusters_stage", "failure_clusters", ["stage"], unique=False
    )
    op.create_index(
        "ix_failure_clusters_reason_code",
        "failure_clusters",
        ["reason_code"],
        unique=False,
    )
    op.create_index(
        "ix_failure_clusters_signature",
        "failure_clusters",
        ["signature"],
        unique=False,
    )
    op.create_index(
        "ix_failure_clusters_last_seen_at",
        "failure_clusters",
        ["last_seen_at"],
        unique=False,
    )
    op.create_index(
        "ix_failure_clusters_last_computed_at",
        "failure_clusters",
        ["last_computed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_failure_clusters_last_computed_at", table_name="failure_clusters"
    )
    op.drop_index(
        "ix_failure_clusters_last_seen_at", table_name="failure_clusters"
    )
    op.drop_index(
        "ix_failure_clusters_signature", table_name="failure_clusters"
    )
    op.drop_index(
        "ix_failure_clusters_reason_code", table_name="failure_clusters"
    )
    op.drop_index("ix_failure_clusters_stage", table_name="failure_clusters")
    op.drop_index(
        "ix_failure_clusters_project_id", table_name="failure_clusters"
    )
    op.drop_table("failure_clusters")
