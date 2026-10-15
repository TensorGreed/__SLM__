"""External benchmarks: 'benchmark' dataset type (Epic G-B1).

Postgres only — SQLite stores the enum as an unchecked VARCHAR. Both
spellings are added because the ORM writes enum NAMES ('BENCHMARK') while
the baseline declared lowercase values; IF NOT EXISTS keeps it idempotent.

Revision ID: 20261015_0049
Revises: 20260609_0048
Create Date: 2026-10-15 00:00:00
"""

from __future__ import annotations

from alembic import op


revision = "20261015_0049"
down_revision = "20260609_0048"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        with op.get_context().autocommit_block():
            op.execute("ALTER TYPE datasettype ADD VALUE IF NOT EXISTS 'BENCHMARK'")
            op.execute("ALTER TYPE datasettype ADD VALUE IF NOT EXISTS 'benchmark'")


def downgrade() -> None:
    # Postgres can't drop an enum value without rebuilding the type.
    pass
