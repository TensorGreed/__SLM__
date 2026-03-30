"""Add dataset adapter definition persistence for Adapter Studio.

Revision ID: 20260313_0013
Revises: 20260312_0012
Create Date: 2026-03-13 10:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260313_0013"
down_revision = "20260312_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_adapter_definitions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=True),
        sa.Column("adapter_name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=True),
        sa.Column("base_adapter_id", sa.String(length=128), nullable=False),
        sa.Column("task_profile", sa.String(length=64), nullable=True),
        sa.Column("field_mapping", sa.JSON(), nullable=True),
        sa.Column("adapter_config", sa.JSON(), nullable=True),
        sa.Column("output_contract", sa.JSON(), nullable=True),
        sa.Column("schema_profile", sa.JSON(), nullable=True),
        sa.Column("inference_summary", sa.JSON(), nullable=True),
        sa.Column("validation_report", sa.JSON(), nullable=True),
        sa.Column("export_template", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "adapter_name",
            "version",
            name="uq_dataset_adapter_definitions_project_name_version",
        ),
    )
    op.create_index(
        "ix_dataset_adapter_definitions_project_id",
        "dataset_adapter_definitions",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_dataset_adapter_definitions_adapter_name",
        "dataset_adapter_definitions",
        ["adapter_name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_dataset_adapter_definitions_adapter_name", table_name="dataset_adapter_definitions")
    op.drop_index("ix_dataset_adapter_definitions_project_id", table_name="dataset_adapter_definitions")
    op.drop_table("dataset_adapter_definitions")
