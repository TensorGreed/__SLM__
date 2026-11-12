"""Offline knowledge distillation: 'distillation' training mode.

Postgres only — SQLite stores the enum as an unchecked VARCHAR. Both
spellings are added because the ORM writes enum NAMES ('DISTILLATION') while
the baseline declared lowercase values; IF NOT EXISTS keeps it idempotent.

Revision ID: 20261112_0052
Revises: 20261027_0051
Create Date: 2026-11-12 00:00:00
"""

from __future__ import annotations

from alembic import op


revision = "20261112_0052"
down_revision = "20261027_0051"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute("ALTER TYPE trainingmode ADD VALUE IF NOT EXISTS 'DISTILLATION'")
            op.execute("ALTER TYPE trainingmode ADD VALUE IF NOT EXISTS 'distillation'")


def downgrade() -> None:
    # Postgres can't drop an enum value without rebuilding the type.
    pass
