"""Add typed artifact registry table.

Revision ID: 20260305_0005
Revises: 20260305_0004
Create Date: 2026-03-05 16:45:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260305_0005"
down_revision = "20260305_0004"
branch_labels = None
depends_on = None


artifact_status_enum = sa.Enum(
    "materialized",
    "failed",
    name="artifactstatus",
)


def upgrade() -> None:
    op.create_table(
        "artifact_records",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("artifact_key", sa.String(length=255), nullable=False),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", artifact_status_enum, nullable=False),
        sa.Column("uri", sa.String(length=2048), nullable=True),
        sa.Column("schema_ref", sa.String(length=255), nullable=False),
        sa.Column("producer_stage", sa.String(length=64), nullable=True),
        sa.Column("producer_run_id", sa.String(length=128), nullable=True),
        sa.Column("producer_step_id", sa.String(length=255), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "artifact_key",
            "version",
            name="uq_artifact_project_key_version",
        ),
    )
    op.create_index("ix_artifact_records_project_id", "artifact_records", ["project_id"], unique=False)
    op.create_index("ix_artifact_records_artifact_key", "artifact_records", ["artifact_key"], unique=False)
    op.create_index("ix_artifact_records_created_at", "artifact_records", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_artifact_records_created_at", table_name="artifact_records")
    op.drop_index("ix_artifact_records_artifact_key", table_name="artifact_records")
    op.drop_index("ix_artifact_records_project_id", table_name="artifact_records")
    op.drop_table("artifact_records")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("DROP TYPE IF EXISTS artifactstatus"))
