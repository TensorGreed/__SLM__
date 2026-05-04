"""Deployability score (priority.md P28).

Creates:
  - deployment_scores — one row per ``POST /deployments/{id}/score/compute``,
    blending measured and estimated components with provenance.

Revision ID: 20260504_0024
Revises: 20260504_0023
Create Date: 2026-05-04 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260504_0024"
down_revision = "20260504_0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "deployment_scores",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("deployment_version_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("provenance", sa.String(length=32), nullable=False),
        sa.Column("confidence_band", sa.String(length=16), nullable=False),
        sa.Column("components", sa.JSON(), nullable=True),
        sa.Column("signals_summary", sa.JSON(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["deployment_version_id"], ["deployment_versions.id"]
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_deployment_scores_deployment_version_id",
        "deployment_scores",
        ["deployment_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_scores_project_id",
        "deployment_scores",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_scores_provenance",
        "deployment_scores",
        ["provenance"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_scores_created_at",
        "deployment_scores",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_deployment_scores_created_at", table_name="deployment_scores"
    )
    op.drop_index(
        "ix_deployment_scores_provenance", table_name="deployment_scores"
    )
    op.drop_index(
        "ix_deployment_scores_project_id", table_name="deployment_scores"
    )
    op.drop_index(
        "ix_deployment_scores_deployment_version_id",
        table_name="deployment_scores",
    )
    op.drop_table("deployment_scores")
