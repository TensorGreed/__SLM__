"""Story 1.6 — promote labeled rows → training dataset.

Adds two columns on ``label_rows`` to track promotion of submitted
labels into the project's synthetic / alignment training data:

  - ``promoted_at``           — when the row was promoted (idempotency
                                guard; ``None`` means not yet promoted).
  - ``promoted_to_dataset_id`` — FK to ``datasets.id`` of the dataset
                                 the row landed in (so the operator
                                 can trace which run consumed which
                                 label).

Revision ID: 20260515_0032
Revises: 20260515_0031
Create Date: 2026-05-15 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260515_0032"
down_revision = "20260515_0031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "label_rows",
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "label_rows",
        sa.Column(
            "promoted_to_dataset_id",
            sa.Integer(),
            sa.ForeignKey("datasets.id"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("label_rows", "promoted_to_dataset_id")
    op.drop_column("label_rows", "promoted_at")
