"""Serving gateway: scoped API keys + request log (phase G11).

Adds ``gateway_api_keys`` (sha256-hashed keys scoped to project ids, with a
per-minute rate limit) and ``gateway_request_logs`` (timing / status / token
counts per gateway request; never prompt or reply text). The log's
``project_id`` is an ORM-level link only (SQLite cannot ADD CONSTRAINT later).
Idempotent: every step checks for the table / index first.

Revision ID: 20261027_0051
Revises: 20261026_0050
Create Date: 2026-10-27 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20261027_0051"
down_revision = "20261026_0050"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _index_exists(table: str, index: str) -> bool:
    if not _table_exists(table):
        return False
    return any(ix.get("name") == index for ix in sa.inspect(op.get_bind()).get_indexes(table))


def upgrade() -> None:
    if not _table_exists("gateway_api_keys"):
        op.create_table(
            "gateway_api_keys",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(128), nullable=False, server_default="default"),
            sa.Column("key_hash", sa.String(128), nullable=False, unique=True),
            sa.Column("key_prefix", sa.String(16), nullable=False, server_default=""),
            sa.Column("project_ids", sa.JSON(), nullable=False),
            sa.Column("rate_limit_per_minute", sa.Integer(), nullable=False, server_default="60"),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_by", sa.String(128), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        )
    if not _table_exists("gateway_request_logs"):
        op.create_table(
            "gateway_request_logs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("project_id", sa.Integer(), nullable=False),
            sa.Column("key_id", sa.Integer(), nullable=True),
            sa.Column("model_id", sa.String(160), nullable=False, server_default=""),
            sa.Column("run_id", sa.Integer(), nullable=True),
            sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
            sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
            sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("status_code", sa.Integer(), nullable=True),
            sa.Column("error_code", sa.String(128), nullable=True),
            sa.Column("input_tokens", sa.Integer(), nullable=True),
            sa.Column("output_tokens", sa.Integer(), nullable=True),
            sa.Column("retrieval_applied", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("guided_json_mode", sa.String(32), nullable=True),
            sa.Column("upstream", sa.String(32), nullable=False, server_default="in_process"),
            sa.Column("request_id", sa.String(128), nullable=True),
        )
    for column in ("project_id", "ts"):
        name = f"ix_gateway_request_logs_{column}"
        if not _index_exists("gateway_request_logs", name):
            op.create_index(name, "gateway_request_logs", [column])


def downgrade() -> None:
    for column in ("ts", "project_id"):
        name = f"ix_gateway_request_logs_{column}"
        if _index_exists("gateway_request_logs", name):
            op.drop_index(name, table_name="gateway_request_logs")
    if _table_exists("gateway_request_logs"):
        op.drop_table("gateway_request_logs")
    if _table_exists("gateway_api_keys"):
        op.drop_table("gateway_api_keys")
