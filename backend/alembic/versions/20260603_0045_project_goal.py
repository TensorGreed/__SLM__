"""Arc H — Project end-goal contract.

Adds:
  - ``projects.goal`` JSON column holding the user's stated goal:
    ``{target_metric, target_threshold, deadline, title, stated_at}``.

The goal_service computes a "% toward stated goal" progress ledger
that Coach + Data Studio render. Nullable so existing projects
round-trip unchanged; the service treats null as "no goal set" and
falls back to a sensible default (f1 ≥ 0.70).

Revision ID: 20260603_0045
Revises: 20260601_0044
Create Date: 2026-06-03 11:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260603_0045"
down_revision = "20260601_0044"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for col in inspector.get_columns(table):
        if col.get("name") == column:
            return True
    return False


def upgrade() -> None:
    if _column_exists("projects", "goal"):
        return
    op.add_column(
        "projects",
        sa.Column("goal", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    if not _column_exists("projects", "goal"):
        return
    with op.batch_alter_table("projects") as batch:
        batch.drop_column("goal")
