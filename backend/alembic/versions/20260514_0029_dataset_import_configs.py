"""Saved dataset-import mapping configs (DATASET_IMPORT_PLAN.md Phase G).

Creates:
  - dataset_import_configs — one row per (project, name) tuple. Stores
    a re-runnable mapping (locator + mapper + field_map + drop_reasons)
    + audit columns (last_run_at, last_run_accepted) so the "Saved
    mappings" UI can show the latest yield without joining run_events.

Revision ID: 20260514_0029
Revises: 20260507_0028
Create Date: 2026-05-14 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260514_0029"
down_revision = "20260507_0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_import_configs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("locator", sa.Text(), nullable=False),
        sa.Column("mapper_id", sa.String(length=64), nullable=False),
        sa.Column("field_map", sa.JSON(), nullable=False),
        sa.Column("drop_reasons", sa.JSON(), nullable=False),
        sa.Column("limit", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_accepted", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id", "name", name="uq_dataset_import_config_name"
        ),
    )
    op.create_index(
        "ix_dataset_import_configs_project_id",
        "dataset_import_configs",
        ["project_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_dataset_import_configs_project_id",
        table_name="dataset_import_configs",
    )
    op.drop_table("dataset_import_configs")
