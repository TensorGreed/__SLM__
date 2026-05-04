"""On-demand drift-check runs against a deployed endpoint (priority.md P27).

Creates:
  - deployment_drift_checks — one row per ``POST /deployments/{id}/drift/check``
    invocation. Stores baseline + current pass rate, delta, tolerance,
    drift verdict, per-row scoring detail (capped), and a summary blob.

Revision ID: 20260504_0023
Revises: 20260504_0022
Create Date: 2026-05-04 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260504_0023"
down_revision = "20260504_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "deployment_drift_checks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("deployment_version_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("gold_set_id", sa.Integer(), nullable=True),
        sa.Column("gold_set_version_id", sa.Integer(), nullable=True),
        sa.Column("baseline_experiment_id", sa.Integer(), nullable=True),
        sa.Column("baseline_eval_result_id", sa.Integer(), nullable=True),
        sa.Column("eval_type", sa.String(length=64), nullable=False),
        sa.Column("baseline_pass_rate", sa.Float(), nullable=True),
        sa.Column("current_pass_rate", sa.Float(), nullable=False),
        sa.Column("delta", sa.Float(), nullable=True),
        sa.Column("tolerance", sa.Float(), nullable=False),
        sa.Column("drift_detected", sa.Boolean(), nullable=False),
        sa.Column("samples_evaluated", sa.Integer(), nullable=False),
        sa.Column("samples_failed", sa.Integer(), nullable=False),
        sa.Column("samples_skipped", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("per_row_results", sa.JSON(), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["deployment_version_id"], ["deployment_versions.id"]
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["gold_set_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(
            ["gold_set_version_id"], ["gold_set_versions.id"]
        ),
        sa.ForeignKeyConstraint(
            ["baseline_experiment_id"], ["experiments.id"]
        ),
        sa.ForeignKeyConstraint(
            ["baseline_eval_result_id"], ["eval_results.id"]
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_deployment_drift_checks_deployment_version_id",
        "deployment_drift_checks",
        ["deployment_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_drift_checks_project_id",
        "deployment_drift_checks",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_drift_checks_gold_set_id",
        "deployment_drift_checks",
        ["gold_set_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_drift_checks_gold_set_version_id",
        "deployment_drift_checks",
        ["gold_set_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_drift_checks_baseline_experiment_id",
        "deployment_drift_checks",
        ["baseline_experiment_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_drift_checks_baseline_eval_result_id",
        "deployment_drift_checks",
        ["baseline_eval_result_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_drift_checks_eval_type",
        "deployment_drift_checks",
        ["eval_type"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_drift_checks_drift_detected",
        "deployment_drift_checks",
        ["drift_detected"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_drift_checks_created_at",
        "deployment_drift_checks",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_deployment_drift_checks_created_at",
        table_name="deployment_drift_checks",
    )
    op.drop_index(
        "ix_deployment_drift_checks_drift_detected",
        table_name="deployment_drift_checks",
    )
    op.drop_index(
        "ix_deployment_drift_checks_eval_type",
        table_name="deployment_drift_checks",
    )
    op.drop_index(
        "ix_deployment_drift_checks_baseline_eval_result_id",
        table_name="deployment_drift_checks",
    )
    op.drop_index(
        "ix_deployment_drift_checks_baseline_experiment_id",
        table_name="deployment_drift_checks",
    )
    op.drop_index(
        "ix_deployment_drift_checks_gold_set_version_id",
        table_name="deployment_drift_checks",
    )
    op.drop_index(
        "ix_deployment_drift_checks_gold_set_id",
        table_name="deployment_drift_checks",
    )
    op.drop_index(
        "ix_deployment_drift_checks_project_id",
        table_name="deployment_drift_checks",
    )
    op.drop_index(
        "ix_deployment_drift_checks_deployment_version_id",
        table_name="deployment_drift_checks",
    )
    op.drop_table("deployment_drift_checks")
