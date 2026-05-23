"""Project training_forecast_cache JSON column (USER-SUCCESS Epic 1).

Adds:
  - projects.training_forecast_cache — nullable JSON. Caches the
    most-recently computed trainability forecast for the project,
    keyed by (dataset_version, recipe_id, base_model_name). The
    diversity-score embed pass is the expensive part (5-30 seconds
    depending on gold-set size), so we persist the result here and
    invalidate when any of the cache-key inputs change. Nullable so
    existing projects round-trip unchanged.

Revision ID: 20260523_0035
Revises: 20260521_0034
Create Date: 2026-05-23 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260523_0035"
down_revision = "20260521_0034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("training_forecast_cache", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("projects", "training_forecast_cache")
