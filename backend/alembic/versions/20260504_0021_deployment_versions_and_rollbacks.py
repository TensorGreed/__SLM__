"""Deployment versions + rollback audit (priority.md P25).

Creates:
  - deployment_versions — one row per executed deploy of an export to a
    target, with lifecycle status (pending / promoted / rejected /
    rolled_back / superseded).
  - deployment_rollbacks — append-only audit log keyed by deployment
    version, mirroring the P1 ``autopilot_decisions`` shape.

Revision ID: 20260504_0021
Revises: 20260425_0020
Create Date: 2026-05-04 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260504_0021"
down_revision = "20260425_0020"
branch_labels = None
depends_on = None


_DEPLOYMENT_VERSION_STATUS = (
    "pending",
    "promoted",
    "rejected",
    "rolled_back",
    "superseded",
)
_ROLLBACK_ACTION = ("promote", "reject", "rollback")


def upgrade() -> None:
    op.create_table(
        "deployment_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("export_id", sa.Integer(), nullable=False),
        sa.Column("registry_entry_id", sa.Integer(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("target_id", sa.String(length=128), nullable=False),
        sa.Column("target_kind", sa.String(length=64), nullable=True),
        sa.Column("endpoint_name", sa.String(length=256), nullable=True),
        sa.Column("endpoint_handle", sa.String(length=512), nullable=True),
        sa.Column("region", sa.String(length=64), nullable=True),
        sa.Column("instance_type", sa.String(length=128), nullable=True),
        sa.Column(
            "status",
            sa.Enum(*_DEPLOYMENT_VERSION_STATUS, name="deploymentversionstatus"),
            nullable=False,
        ),
        sa.Column("plan_payload", sa.JSON(), nullable=True),
        sa.Column("promoted_reason", sa.Text(), nullable=True),
        sa.Column("rejected_reason", sa.Text(), nullable=True),
        sa.Column("rolled_back_reason", sa.Text(), nullable=True),
        sa.Column("rolled_back_to_id", sa.Integer(), nullable=True),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rejected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rolled_back_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["export_id"], ["exports.id"]),
        sa.ForeignKeyConstraint(["registry_entry_id"], ["model_registry_entries.id"]),
        sa.ForeignKeyConstraint(["rolled_back_to_id"], ["deployment_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "export_id", "version", name="uq_deployment_versions_export_version"
        ),
    )
    op.create_index(
        "ix_deployment_versions_project_id",
        "deployment_versions",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_versions_export_id",
        "deployment_versions",
        ["export_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_versions_registry_entry_id",
        "deployment_versions",
        ["registry_entry_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_versions_target_id",
        "deployment_versions",
        ["target_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_versions_status",
        "deployment_versions",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_versions_rolled_back_to_id",
        "deployment_versions",
        ["rolled_back_to_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_versions_created_at",
        "deployment_versions",
        ["created_at"],
        unique=False,
    )

    op.create_table(
        "deployment_rollbacks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("deployment_version_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column(
            "action",
            sa.Enum(*_ROLLBACK_ACTION, name="deploymentrollbackaction"),
            nullable=False,
        ),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("status_after", sa.String(length=64), nullable=True),
        sa.Column("rolled_back_to_id", sa.Integer(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["deployment_version_id"], ["deployment_versions.id"]
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["rolled_back_to_id"], ["deployment_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_deployment_rollbacks_deployment_version_id",
        "deployment_rollbacks",
        ["deployment_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_rollbacks_project_id",
        "deployment_rollbacks",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_rollbacks_action",
        "deployment_rollbacks",
        ["action"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_rollbacks_rolled_back_to_id",
        "deployment_rollbacks",
        ["rolled_back_to_id"],
        unique=False,
    )
    op.create_index(
        "ix_deployment_rollbacks_created_at",
        "deployment_rollbacks",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_deployment_rollbacks_created_at", table_name="deployment_rollbacks"
    )
    op.drop_index(
        "ix_deployment_rollbacks_rolled_back_to_id",
        table_name="deployment_rollbacks",
    )
    op.drop_index(
        "ix_deployment_rollbacks_action", table_name="deployment_rollbacks"
    )
    op.drop_index(
        "ix_deployment_rollbacks_project_id", table_name="deployment_rollbacks"
    )
    op.drop_index(
        "ix_deployment_rollbacks_deployment_version_id",
        table_name="deployment_rollbacks",
    )
    op.drop_table("deployment_rollbacks")

    op.drop_index(
        "ix_deployment_versions_created_at", table_name="deployment_versions"
    )
    op.drop_index(
        "ix_deployment_versions_rolled_back_to_id",
        table_name="deployment_versions",
    )
    op.drop_index(
        "ix_deployment_versions_status", table_name="deployment_versions"
    )
    op.drop_index(
        "ix_deployment_versions_target_id", table_name="deployment_versions"
    )
    op.drop_index(
        "ix_deployment_versions_registry_entry_id",
        table_name="deployment_versions",
    )
    op.drop_index(
        "ix_deployment_versions_export_id", table_name="deployment_versions"
    )
    op.drop_index(
        "ix_deployment_versions_project_id", table_name="deployment_versions"
    )
    op.drop_table("deployment_versions")

    bind = op.get_bind()
    sa.Enum(*_DEPLOYMENT_VERSION_STATUS, name="deploymentversionstatus").drop(
        bind, checkfirst=True
    )
    sa.Enum(*_ROLLBACK_ACTION, name="deploymentrollbackaction").drop(
        bind, checkfirst=True
    )
