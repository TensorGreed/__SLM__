"""Checkpoint promotion timestamp (priority.md P16).

Adds:
  - checkpoints.promoted_at — wall-clock when an operator promoted this
    checkpoint as the run's canonical "best". Populated on
    POST /training/runs/{id}/checkpoints/{step}/promote and cleared when
    promotion moves to a different step within the same run.

Revision ID: 20260425_0019
Revises: 20260424_0018
Create Date: 2026-04-25 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260425_0019"
down_revision = "20260424_0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "checkpoints",
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("checkpoints", "promoted_at")
