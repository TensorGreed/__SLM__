"""Project quickstart_tour_state JSON column (Theme 1 Epic 2).

Adds:
  - projects.quickstart_tour_state — nullable JSON. Stores per-project
    state for the project-guide quickstart tour nudges (dismissed
    nudge ids etc.). Nullable so existing projects round-trip
    unchanged.

Revision ID: 20260521_0034
Revises: 20260521_0033
Create Date: 2026-05-21 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260521_0034"
down_revision = "20260521_0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("quickstart_tour_state", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("projects", "quickstart_tour_state")
