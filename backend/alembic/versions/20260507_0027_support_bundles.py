"""Persisted support bundles (priority.md P34, Wave G).

Creates:
  - support_bundles — one row per generated support bundle. The actual
    zip lives on disk under DATA_DIR/support_bundles; this table holds
    metadata + a download token + per-section redaction stats.

Revision ID: 20260507_0027
Revises: 20260507_0026
Create Date: 2026-05-07 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260507_0027"
down_revision = "20260507_0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "support_bundles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("bundle_uid", sa.String(length=64), nullable=False),
        sa.Column("download_token", sa.String(length=64), nullable=False),
        sa.Column("file_path", sa.String(length=2048), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("redactions_applied", sa.JSON(), nullable=True),
        sa.Column("section_counts", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("bundle_uid", name="uq_support_bundles_uid"),
    )
    op.create_index(
        "ix_support_bundles_project_id",
        "support_bundles",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_support_bundles_bundle_uid",
        "support_bundles",
        ["bundle_uid"],
        unique=False,
    )
    op.create_index(
        "ix_support_bundles_created_at",
        "support_bundles",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_support_bundles_expires_at",
        "support_bundles",
        ["expires_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_support_bundles_expires_at", table_name="support_bundles"
    )
    op.drop_index(
        "ix_support_bundles_created_at", table_name="support_bundles"
    )
    op.drop_index(
        "ix_support_bundles_bundle_uid", table_name="support_bundles"
    )
    op.drop_index(
        "ix_support_bundles_project_id", table_name="support_bundles"
    )
    op.drop_table("support_bundles")
