"""Forecast vs reality calibration observations (USER-SUCCESS Epic 1, T5).

Adds:
  - ``forecast_calibration_observations`` — one row per experiment
    pairing the user's latest forecast snapshot at launch with the
    post-eval gate-pass verdict. Source data for the admin
    calibration endpoint.

Idempotent — mirrors the existence-check pattern from 0037 + 0038.

Revision ID: 20260527_0039
Revises: 20260527_0038
Create Date: 2026-05-27 14:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260527_0039"
down_revision = "20260527_0038"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return name in inspector.get_table_names()


def upgrade() -> None:
    if _table_exists("forecast_calibration_observations"):
        return
    op.create_table(
        "forecast_calibration_observations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "experiment_id",
            sa.Integer(),
            sa.ForeignKey("experiments.id"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            sa.Integer(),
            sa.ForeignKey("projects.id"),
            nullable=False,
        ),
        sa.Column(
            "snapshot_id",
            sa.Integer(),
            sa.ForeignKey("training_forecast_snapshots.id"),
            nullable=False,
        ),
        sa.Column("predicted_confidence_pct", sa.Integer(), nullable=False),
        sa.Column("predicted_overall", sa.String(32), nullable=False),
        sa.Column("recipe_id", sa.String(64), nullable=False),
        sa.Column("actual_passed", sa.Boolean(), nullable=True),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("experiment_id", name="uq_forecast_obs_experiment_id"),
    )
    op.create_index(
        "ix_forecast_calibration_observations_experiment_id",
        "forecast_calibration_observations",
        ["experiment_id"],
    )
    op.create_index(
        "ix_forecast_calibration_observations_project_id",
        "forecast_calibration_observations",
        ["project_id"],
    )
    op.create_index(
        "ix_forecast_calibration_observations_recipe_id",
        "forecast_calibration_observations",
        ["recipe_id"],
    )


def downgrade() -> None:
    if not _table_exists("forecast_calibration_observations"):
        return
    op.drop_index(
        "ix_forecast_calibration_observations_recipe_id",
        table_name="forecast_calibration_observations",
    )
    op.drop_index(
        "ix_forecast_calibration_observations_project_id",
        table_name="forecast_calibration_observations",
    )
    op.drop_index(
        "ix_forecast_calibration_observations_experiment_id",
        table_name="forecast_calibration_observations",
    )
    op.drop_table("forecast_calibration_observations")
