"""Add model registry and project secret tables.

Revision ID: 20260305_0002
Revises: 20260304_0001
Create Date: 2026-03-05 09:40:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260305_0002"
down_revision = "20260304_0001"
branch_labels = None
depends_on = None


registry_stage_enum = sa.Enum(
    "candidate",
    "staging",
    "production",
    "archived",
    name="registrystage",
)
deployment_status_enum = sa.Enum(
    "not_deployed",
    "deployed",
    "failed",
    name="deploymentstatus",
)


def upgrade() -> None:
    op.create_table(
        "model_registry_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("export_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("stage", registry_stage_enum, nullable=False),
        sa.Column("deployment_status", deployment_status_enum, nullable=False),
        sa.Column("artifact_path", sa.String(length=1024), nullable=True),
        sa.Column("readiness", sa.JSON(), nullable=True),
        sa.Column("deployment", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deployed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
        sa.ForeignKeyConstraint(["export_id"], ["exports.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "project_secrets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("key_name", sa.String(length=64), nullable=False),
        sa.Column("encrypted_value", sa.Text(), nullable=False),
        sa.Column("value_hint", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "provider", "key_name", name="uq_project_secret_key"),
    )


def downgrade() -> None:
    op.drop_table("project_secrets")
    op.drop_table("model_registry_entries")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("DROP TYPE IF EXISTS deploymentstatus"))
        op.execute(sa.text("DROP TYPE IF EXISTS registrystage"))
