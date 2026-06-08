"""Quality-Lift phase 2 slice 1 — Project.slice_definitions column.

Adds:
  - ``projects.slice_definitions`` (JSON, nullable) — user-defined named
    subsets of eval rows. See slice_definitions_service for the
    canonical payload shape. Nullable so existing projects round-trip
    unchanged (handler emits only overall metrics; gate evaluator
    skips per_slice resolution).

Idempotent — checks column existence before adding (mirrors 0045/0046).

Revision ID: 20260608_0047
Revises: 20260608_0046
Create Date: 2026-06-08 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260608_0047"
down_revision = "20260608_0046"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for col in inspector.get_columns(table):
        if col.get("name") == column:
            return True
    return False


def upgrade() -> None:
    if _column_exists("projects", "slice_definitions"):
        return
    op.add_column(
        "projects",
        sa.Column("slice_definitions", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    if not _column_exists("projects", "slice_definitions"):
        return
    with op.batch_alter_table("projects") as batch:
        batch.drop_column("slice_definitions")
