"""Project selected_recipe JSON column.

Stores the task-shape recipe (Theme 2) the user picked at first-
dataset-import time. Nullable so existing projects round-trip
unchanged; the column is materialized only when the user selects
a recipe in the DatasetImportWizard.

Revision ID: 20260521_0033
Revises: 20260515_0032
Create Date: 2026-05-21 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260521_0033"
down_revision = "20260515_0032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("selected_recipe", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("projects", "selected_recipe")
