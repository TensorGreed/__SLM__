"""Named eval sets: 'eval_set' dataset type (hardening Phase 3).

Postgres only: SQLite stores the enum as an unchecked VARCHAR, so this is a no-op there. Both
spellings are added because the ORM writes enum NAMES ('EVAL_SET') while the baseline declared
lowercase values; IF NOT EXISTS keeps it idempotent (same pattern as 0049).

Revision ID: 20261119_0053
Revises: 20261112_0052
Create Date: 2026-11-19 00:00:00
"""

from __future__ import annotations

from alembic import op


revision = "20261119_0053"
down_revision = "20261112_0052"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute("ALTER TYPE datasettype ADD VALUE IF NOT EXISTS 'EVAL_SET'")
            op.execute("ALTER TYPE datasettype ADD VALUE IF NOT EXISTS 'eval_set'")


def downgrade() -> None:
    # Postgres can't drop an enum value without rebuilding the type.
    pass
