"""Project gamification (Lab Journal) JSON column.

Adds:
  - projects.gamification — nullable JSON. Stores per-project XP /
    level / unlocked achievement ids / milestone timestamps. The
    gamification_service materializes the canonical empty shape on
    first read, so existing projects survive the migration with a
    null column.

Revision ID: 20260514_0030
Revises: 20260514_0029
Create Date: 2026-05-14 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260514_0030"
down_revision = "20260514_0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("gamification", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("projects", "gamification")
