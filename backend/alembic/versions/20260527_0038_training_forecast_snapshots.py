"""Training-forecast snapshot history table (USER-SUCCESS Epic 1, T2).

Adds:
  - ``training_forecast_snapshots`` — per-cache-miss compute log of the
    trainability forecast. Used by the panel sparkline + verdict-delta
    strip; 60-day retention enforced by the service layer (no separate
    cron required).

Idempotent — checks for the table's existence before creating
(mirrors the safety guard in 0037).

Revision ID: 20260527_0038
Revises: 20260526_0037
Create Date: 2026-05-27 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260527_0038"
down_revision = "20260526_0037"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return name in inspector.get_table_names()


def upgrade() -> None:
    if _table_exists("training_forecast_snapshots"):
        return
    op.create_table(
        "training_forecast_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "project_id",
            sa.Integer(),
            sa.ForeignKey("projects.id"),
            nullable=False,
        ),
        sa.Column("cache_key", sa.String(32), nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("overall", sa.String(32), nullable=False),
        sa.Column("confidence_pct", sa.Integer(), nullable=False),
        sa.Column("signals", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_training_forecast_snapshots_project_id",
        "training_forecast_snapshots",
        ["project_id"],
    )
    op.create_index(
        "ix_training_forecast_snapshots_computed_at",
        "training_forecast_snapshots",
        ["computed_at"],
    )


def downgrade() -> None:
    if not _table_exists("training_forecast_snapshots"):
        return
    op.drop_index(
        "ix_training_forecast_snapshots_computed_at",
        table_name="training_forecast_snapshots",
    )
    op.drop_index(
        "ix_training_forecast_snapshots_project_id",
        table_name="training_forecast_snapshots",
    )
    op.drop_table("training_forecast_snapshots")
