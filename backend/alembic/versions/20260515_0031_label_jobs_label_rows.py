"""Annotation foundation (Story 1.1) — label_jobs + label_rows.

Adds:
  - label_jobs   — one row per labeling pass (task shape + label set
                   + instructions + status + target_rows).
  - label_rows   — per-row work units seeded from a source dataset.
                   Carries reviewer assignment + submitted label.

Indexed columns:
  - label_jobs.project_id           (scoped lookups from the API)
  - label_rows.job_id               (the dominant filter in assign /
                                     stats queries)

Revision ID: 20260515_0031
Revises: 20260514_0030
Create Date: 2026-05-15 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260515_0031"
down_revision = "20260514_0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "label_jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "project_id",
            sa.Integer(),
            sa.ForeignKey("projects.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("label_type", sa.String(length=32), nullable=False),
        sa.Column("label_schema", sa.JSON(), nullable=False),
        sa.Column("instructions", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="active",
        ),
        sa.Column("target_rows", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_label_jobs_project_id", "label_jobs", ["project_id"]
    )

    op.create_table(
        "label_rows",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            sa.Integer(),
            sa.ForeignKey("label_jobs.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("source_row_id", sa.String(length=128), nullable=True),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column(
            "assigned_to",
            sa.Integer(),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("label_payload", sa.JSON(), nullable=True),
        sa.Column("labeled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reviewer_notes", sa.Text(), nullable=True),
    )
    op.create_index("ix_label_rows_job_id", "label_rows", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_label_rows_job_id", table_name="label_rows")
    op.drop_table("label_rows")
    op.drop_index("ix_label_jobs_project_id", table_name="label_jobs")
    op.drop_table("label_jobs")
