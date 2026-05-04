"""Post-deploy telemetry samples (priority.md P26).

Creates:
  - deployment_telemetry_samples — one row per ingested inference
    request, scoped per deployment version. Aggregations are computed
    on-demand by ``served_model_telemetry_service`` so we don't pre-bin.

Revision ID: 20260504_0022
Revises: 20260504_0021
Create Date: 2026-05-04 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260504_0022"
down_revision = "20260504_0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "deployment_telemetry_samples",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("deployment_version_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["deployment_version_id"], ["deployment_versions.id"]
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_deployment_telemetry_samples_deployment_version_id",
        "deployment_telemetry_samples",
        ["deployment_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_telemetry_samples_project_id",
        "deployment_telemetry_samples",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_telemetry_samples_ts",
        "deployment_telemetry_samples",
        ["ts"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_telemetry_samples_created_at",
        "deployment_telemetry_samples",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_deployment_telemetry_samples_created_at",
        table_name="deployment_telemetry_samples",
    )
    op.drop_index(
        "ix_deployment_telemetry_samples_ts",
        table_name="deployment_telemetry_samples",
    )
    op.drop_index(
        "ix_deployment_telemetry_samples_project_id",
        table_name="deployment_telemetry_samples",
    )
    op.drop_index(
        "ix_deployment_telemetry_samples_deployment_version_id",
        table_name="deployment_telemetry_samples",
    )
    op.drop_table("deployment_telemetry_samples")
