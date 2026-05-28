"""Training manifest warm_start JSON column (Track 1, Epic B).

Adds:
  - training_manifests.warm_start — nullable JSON. Records which starting
    weights a run actually used: ``source`` (``checkpoint`` for a registered
    pre-fine-tuned warm start, ``base_model`` for a cold-start fallback),
    ``effective_base_model``, ``checkpoint_name``, and ``reason``. Nullable so
    manifests captured before warm-start wiring round-trip unchanged.

Revision ID: 20260528_0042
Revises: 20260527_0041
Create Date: 2026-05-28 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260528_0042"
down_revision = "20260527_0041"
branch_labels = None
depends_on = None


def _existing_columns(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    if "warm_start" not in _existing_columns("training_manifests"):
        op.add_column(
            "training_manifests",
            sa.Column("warm_start", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    if "warm_start" in _existing_columns("training_manifests"):
        op.drop_column("training_manifests", "warm_start")
