"""Failure-cluster ``exemplar_run_ids`` column (priority.md P36).

Adds a parallel JSON column to ``failure_clusters`` so the
failure-analysis UI can deep-link straight from a cluster exemplar to
the per-run event drilldown without a chained fetch.

Existing clusters get an empty list; the next call to
``compute_failure_clusters`` will populate it.

Revision ID: 20260507_0028
Revises: 20260507_0027
Create Date: 2026-05-07 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260507_0028"
down_revision = "20260507_0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "failure_clusters",
        sa.Column("exemplar_run_ids", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("failure_clusters", "exemplar_run_ids")
