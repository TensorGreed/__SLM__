"""Label-row multi-reviewer capture (Epic F Phase 2).

Adds:
  - ``label_row_reviews`` — per-reviewer label submissions, keyed
    by (row_id, reviewer_id). Captures every reviewer's individual
    submission so we can compute Cohen's κ / span-F1 / preference-
    agreement when ≥2 distinct reviewers ever labeled the same row.

The primary ``label_rows`` table keeps its single ``label_payload``
field for back-compat — promotion + UI still read the primary view.
This side-table is additive and idempotent; existence-checked
before create.

Revision ID: 20260601_0044
Revises: 20260530_0043
Create Date: 2026-06-01 14:55:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260601_0044"
down_revision = "20260530_0043"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return name in inspector.get_table_names()


def upgrade() -> None:
    if _table_exists("label_row_reviews"):
        return
    op.create_table(
        "label_row_reviews",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "row_id",
            sa.Integer(),
            sa.ForeignKey("label_rows.id"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            sa.Integer(),
            sa.ForeignKey("label_jobs.id"),
            nullable=False,
        ),
        sa.Column(
            "reviewer_id",
            sa.Integer(),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column(
            "label_payload",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("reviewer_notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "row_id", "reviewer_id", name="uq_label_row_reviews_row_reviewer"
        ),
    )
    op.create_index(
        "ix_label_row_reviews_row_id",
        "label_row_reviews",
        ["row_id"],
    )
    op.create_index(
        "ix_label_row_reviews_job_id",
        "label_row_reviews",
        ["job_id"],
    )
    op.create_index(
        "ix_label_row_reviews_reviewer_id",
        "label_row_reviews",
        ["reviewer_id"],
    )


def downgrade() -> None:
    if not _table_exists("label_row_reviews"):
        return
    op.drop_index("ix_label_row_reviews_reviewer_id", table_name="label_row_reviews")
    op.drop_index("ix_label_row_reviews_job_id", table_name="label_row_reviews")
    op.drop_index("ix_label_row_reviews_row_id", table_name="label_row_reviews")
    op.drop_table("label_row_reviews")
