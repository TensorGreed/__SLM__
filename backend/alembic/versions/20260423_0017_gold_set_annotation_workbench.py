"""Gold-set annotation workbench tables (priority.md P10).

Creates:
  - gold_set_versions — monotonic versions per gold set with draft/locked status.
  - gold_set_rows — per-row annotation state (input/expected/rationale/labels/status/reviewer).
  - gold_set_reviewer_queue — per-reviewer work queue, maintained by the service
    whenever a row's reviewer assignment changes.

Revision ID: 20260423_0017
Revises: 20260422_0016
Create Date: 2026-04-23 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260423_0017"
down_revision = "20260422_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gold_set_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("gold_set_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("DRAFT", "LOCKED", name="goldsetversionstatus"),
            nullable=False,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["gold_set_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "gold_set_id",
            "version",
            name="uq_gold_set_versions_gold_set_version",
        ),
    )
    op.create_index(
        "ix_gold_set_versions_gold_set_id",
        "gold_set_versions",
        ["gold_set_id"],
        unique=False,
    )
    op.create_index(
        "ix_gold_set_versions_status",
        "gold_set_versions",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_gold_set_versions_created_at",
        "gold_set_versions",
        ["created_at"],
        unique=False,
    )

    op.create_table(
        "gold_set_rows",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("gold_set_id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("source_row_key", sa.String(length=128), nullable=True),
        sa.Column("source_dataset_id", sa.Integer(), nullable=True),
        sa.Column("input", sa.JSON(), nullable=True),
        sa.Column("expected", sa.JSON(), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("labels", sa.JSON(), nullable=True),
        sa.Column("reviewer_id", sa.Integer(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "IN_REVIEW",
                "APPROVED",
                "REJECTED",
                "CHANGES_REQUESTED",
                name="goldsetrowstatus",
            ),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["gold_set_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(["version_id"], ["gold_set_versions.id"]),
        sa.ForeignKeyConstraint(["source_dataset_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "version_id",
            "source_row_key",
            name="uq_gold_set_rows_version_source_key",
        ),
    )
    op.create_index(
        "ix_gold_set_rows_gold_set_id",
        "gold_set_rows",
        ["gold_set_id"],
        unique=False,
    )
    op.create_index(
        "ix_gold_set_rows_version_id",
        "gold_set_rows",
        ["version_id"],
        unique=False,
    )
    op.create_index(
        "ix_gold_set_rows_reviewer_id",
        "gold_set_rows",
        ["reviewer_id"],
        unique=False,
    )
    op.create_index(
        "ix_gold_set_rows_status",
        "gold_set_rows",
        ["status"],
        unique=False,
    )

    op.create_table(
        "gold_set_reviewer_queue",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("gold_set_id", sa.Integer(), nullable=False),
        sa.Column("row_id", sa.Integer(), nullable=False),
        sa.Column("reviewer_id", sa.Integer(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "IN_PROGRESS",
                "COMPLETED",
                "SKIPPED",
                name="goldsetreviewerqueuestatus",
            ),
            nullable=False,
        ),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["gold_set_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(["row_id"], ["gold_set_rows.id"]),
        sa.ForeignKeyConstraint(["reviewer_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "row_id",
            "reviewer_id",
            name="uq_gold_set_reviewer_queue_row_reviewer",
        ),
    )
    op.create_index(
        "ix_gold_set_reviewer_queue_gold_set_id",
        "gold_set_reviewer_queue",
        ["gold_set_id"],
        unique=False,
    )
    op.create_index(
        "ix_gold_set_reviewer_queue_row_id",
        "gold_set_reviewer_queue",
        ["row_id"],
        unique=False,
    )
    op.create_index(
        "ix_gold_set_reviewer_queue_reviewer_id",
        "gold_set_reviewer_queue",
        ["reviewer_id"],
        unique=False,
    )
    op.create_index(
        "ix_gold_set_reviewer_queue_status",
        "gold_set_reviewer_queue",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_gold_set_reviewer_queue_status", table_name="gold_set_reviewer_queue")
    op.drop_index("ix_gold_set_reviewer_queue_reviewer_id", table_name="gold_set_reviewer_queue")
    op.drop_index("ix_gold_set_reviewer_queue_row_id", table_name="gold_set_reviewer_queue")
    op.drop_index("ix_gold_set_reviewer_queue_gold_set_id", table_name="gold_set_reviewer_queue")
    op.drop_table("gold_set_reviewer_queue")

    op.drop_index("ix_gold_set_rows_status", table_name="gold_set_rows")
    op.drop_index("ix_gold_set_rows_reviewer_id", table_name="gold_set_rows")
    op.drop_index("ix_gold_set_rows_version_id", table_name="gold_set_rows")
    op.drop_index("ix_gold_set_rows_gold_set_id", table_name="gold_set_rows")
    op.drop_table("gold_set_rows")

    op.drop_index("ix_gold_set_versions_created_at", table_name="gold_set_versions")
    op.drop_index("ix_gold_set_versions_status", table_name="gold_set_versions")
    op.drop_index("ix_gold_set_versions_gold_set_id", table_name="gold_set_versions")
    op.drop_table("gold_set_versions")

    bind = op.get_bind()
    sa.Enum(name="goldsetreviewerqueuestatus").drop(bind, checkfirst=True)
    sa.Enum(name="goldsetrowstatus").drop(bind, checkfirst=True)
    sa.Enum(name="goldsetversionstatus").drop(bind, checkfirst=True)
