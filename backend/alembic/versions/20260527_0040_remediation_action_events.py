"""Remediation action event tracking (E2).

Adds:
  - ``remediation_action_events`` — one row per user click on a
    forecast or failure-cluster suggested action. Lift-stamped by the
    post-eval pipeline so we can aggregate "did this fix help?" by
    action_kind.

Idempotent — checks existence before creating.

Revision ID: 20260527_0040
Revises: 20260527_0039
Create Date: 2026-05-27 16:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260527_0040"
down_revision = "20260527_0039"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return name in inspector.get_table_names()


def upgrade() -> None:
    if _table_exists("remediation_action_events"):
        return
    op.create_table(
        "remediation_action_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "project_id",
            sa.Integer(),
            sa.ForeignKey("projects.id"),
            nullable=False,
        ),
        sa.Column("action_kind", sa.String(64), nullable=False),
        sa.Column("params_hash", sa.String(32), nullable=False),
        sa.Column(
            "outcome",
            sa.Enum(
                "CLICKED",
                "DISMISSED",
                "APPLIED",
                "IGNORED",
                name="remediationoutcome",
            ),
            nullable=False,
            server_default="CLICKED",
        ),
        sa.Column("evaluation_lift_pct", sa.Float(), nullable=True),
        sa.Column(
            "experiment_id",
            sa.Integer(),
            sa.ForeignKey("experiments.id"),
            nullable=True,
        ),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_remediation_action_events_project_id",
        "remediation_action_events",
        ["project_id"],
    )
    op.create_index(
        "ix_remediation_action_events_action_kind",
        "remediation_action_events",
        ["action_kind"],
    )
    op.create_index(
        "ix_remediation_action_events_observed_at",
        "remediation_action_events",
        ["observed_at"],
    )


def downgrade() -> None:
    if not _table_exists("remediation_action_events"):
        return
    op.drop_index(
        "ix_remediation_action_events_observed_at",
        table_name="remediation_action_events",
    )
    op.drop_index(
        "ix_remediation_action_events_action_kind",
        table_name="remediation_action_events",
    )
    op.drop_index(
        "ix_remediation_action_events_project_id",
        table_name="remediation_action_events",
    )
    op.drop_table("remediation_action_events")
