"""Gold-drift review queue table (E4).

Adds:
  - ``gold_drift_review_queue`` — fresh hallucination traps awaiting
    user triage. Populated by the drift-check trap-refresh runner;
    drained by the per-row triage endpoint.

Idempotent; existence-checked before create.

Revision ID: 20260527_0041
Revises: 20260527_0040
Create Date: 2026-05-27 18:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260527_0041"
down_revision = "20260527_0040"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return name in inspector.get_table_names()


def upgrade() -> None:
    if _table_exists("gold_drift_review_queue"):
        return
    op.create_table(
        "gold_drift_review_queue",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "project_id",
            sa.Integer(),
            sa.ForeignKey("projects.id"),
            nullable=False,
        ),
        sa.Column(
            "source_drift_check_id",
            sa.Integer(),
            sa.ForeignKey("deployment_drift_checks.id"),
            nullable=True,
        ),
        sa.Column("cluster_reason_code", sa.String(128), nullable=True),
        sa.Column("cluster_signature", sa.String(64), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "source_confidence",
            sa.String(32),
            nullable=False,
            server_default="rough",
        ),
        sa.Column(
            "status",
            sa.Enum("PENDING", "ACCEPTED", "REJECTED", name="golddriftqueuestatus"),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("triage_note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("triaged_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_gold_drift_review_queue_project_id",
        "gold_drift_review_queue",
        ["project_id"],
    )
    op.create_index(
        "ix_gold_drift_review_queue_status",
        "gold_drift_review_queue",
        ["status"],
    )
    op.create_index(
        "ix_gold_drift_review_queue_source_drift_check_id",
        "gold_drift_review_queue",
        ["source_drift_check_id"],
    )
    op.create_index(
        "ix_gold_drift_review_queue_created_at",
        "gold_drift_review_queue",
        ["created_at"],
    )


def downgrade() -> None:
    if not _table_exists("gold_drift_review_queue"):
        return
    op.drop_index(
        "ix_gold_drift_review_queue_created_at",
        table_name="gold_drift_review_queue",
    )
    op.drop_index(
        "ix_gold_drift_review_queue_source_drift_check_id",
        table_name="gold_drift_review_queue",
    )
    op.drop_index(
        "ix_gold_drift_review_queue_status",
        table_name="gold_drift_review_queue",
    )
    op.drop_index(
        "ix_gold_drift_review_queue_project_id",
        table_name="gold_drift_review_queue",
    )
    op.drop_table("gold_drift_review_queue")
