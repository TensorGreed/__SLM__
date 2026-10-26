"""Licence & provenance ledger: data_sources + provenance columns (phase G2).

Adds the ``data_sources`` table, ``raw_documents.data_source_id`` /
``source_url`` / ``source_ref`` / ``retrieved_at`` and
``training_manifests.data_provenance``. The ``data_source_id`` foreign key is
declared on the ORM column only (SQLite cannot ADD CONSTRAINT later).
Idempotent: every step checks for the table / column / index first.

Revision ID: 20261026_0050
Revises: 20261015_0049
Create Date: 2026-10-26 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20261026_0050"
down_revision = "20261015_0049"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _column_exists(table: str, column: str) -> bool:
    if not _table_exists(table):
        return False
    return any(col.get("name") == column for col in sa.inspect(op.get_bind()).get_columns(table))


def _index_exists(table: str, index: str) -> bool:
    if not _table_exists(table):
        return False
    return any(ix.get("name") == index for ix in sa.inspect(op.get_bind()).get_indexes(table))


def upgrade() -> None:
    if not _table_exists("data_sources"):
        op.create_table(
            "data_sources",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id"), nullable=False),
            sa.Column("source_kind", sa.String(32), nullable=False),
            sa.Column("title", sa.String(512), nullable=False, server_default=""),
            sa.Column("locator", sa.String(1024), nullable=False, server_default=""),
            sa.Column("source_ref", sa.String(255), nullable=True),
            sa.Column("licence", sa.String(255), nullable=True),
            sa.Column("licence_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("licence_verified_by", sa.String(255), nullable=True),
            sa.Column("licence_verified_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("usage", sa.String(8), nullable=False, server_default="any"),
            sa.Column("metadata", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
    if not _index_exists("data_sources", "ix_data_sources_project_id"):
        op.create_index("ix_data_sources_project_id", "data_sources", ["project_id"])

    for name, type_ in (
        ("data_source_id", sa.Integer()),
        ("source_url", sa.String(1024)),
        ("source_ref", sa.String(255)),
        ("retrieved_at", sa.DateTime(timezone=True)),
    ):
        if not _column_exists("raw_documents", name):
            op.add_column("raw_documents", sa.Column(name, type_, nullable=True))
    if not _index_exists("raw_documents", "ix_raw_documents_data_source_id"):
        op.create_index("ix_raw_documents_data_source_id", "raw_documents", ["data_source_id"])

    if _table_exists("training_manifests") and not _column_exists("training_manifests", "data_provenance"):
        op.add_column("training_manifests", sa.Column("data_provenance", sa.JSON(), nullable=True))


def downgrade() -> None:
    # Additive and idempotent; dropping columns on SQLite needs a table
    # rebuild, so the downgrade leaves the schema in place.
    pass
