"""Quality-Lift phase 1 — multi-seed variance reporting columns.

Adds:
  - ``experiments.seed_value`` (Integer, nullable) — the exact PRNG seed
    this child run used. NULL for legacy single-seed experiments where
    the seed lives implicitly in ``config['seed']``.
  - ``experiments.seed_group_id`` (String(64), nullable, indexed) — UUID4
    shared by all N children fanned out from one ``start_training`` call
    with ``num_seeds>1``. NULL for legacy / single-seed experiments.
  - ``eval_results.is_aggregate`` (Boolean, default False) — set True on
    rows synthesized by the seed-group aggregator (one per (dataset, eval_type)
    per group). Aggregate rows carry ``metrics[<id>]`` as a dict
    ``{mean,std,min,max,n}`` rather than a scalar.
  - ``eval_results.seed_group_id`` (String(64), nullable, indexed) — links
    aggregate rows back to their seed group so the UI can show per-seed
    drill-down alongside the aggregate.

All columns nullable / default-valued so existing experiments and eval
results round-trip unchanged. Idempotent — checks column existence before
adding (mirrors the pattern from 0036 / 0045).

Revision ID: 20260608_0046
Revises: 20260603_0045
Create Date: 2026-06-08 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260608_0046"
down_revision = "20260603_0045"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for col in inspector.get_columns(table):
        if col.get("name") == column:
            return True
    return False


def _index_exists(table: str, index: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for ix in inspector.get_indexes(table):
        if ix.get("name") == index:
            return True
    return False


def upgrade() -> None:
    if not _column_exists("experiments", "seed_value"):
        op.add_column(
            "experiments",
            sa.Column("seed_value", sa.Integer(), nullable=True),
        )
    if not _column_exists("experiments", "seed_group_id"):
        op.add_column(
            "experiments",
            sa.Column("seed_group_id", sa.String(64), nullable=True),
        )
    if not _index_exists("experiments", "ix_experiments_seed_group_id"):
        op.create_index(
            "ix_experiments_seed_group_id",
            "experiments",
            ["seed_group_id"],
        )

    if not _column_exists("eval_results", "is_aggregate"):
        # SQLite has no native bool — alembic emits INTEGER with default 0,
        # which SQLAlchemy round-trips through the Python ``Boolean`` type.
        op.add_column(
            "eval_results",
            sa.Column(
                "is_aggregate",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )
    if not _column_exists("eval_results", "seed_group_id"):
        op.add_column(
            "eval_results",
            sa.Column("seed_group_id", sa.String(64), nullable=True),
        )
    if not _index_exists("eval_results", "ix_eval_results_seed_group_id"):
        op.create_index(
            "ix_eval_results_seed_group_id",
            "eval_results",
            ["seed_group_id"],
        )


def downgrade() -> None:
    if _index_exists("eval_results", "ix_eval_results_seed_group_id"):
        op.drop_index(
            "ix_eval_results_seed_group_id", table_name="eval_results"
        )
    if _column_exists("eval_results", "seed_group_id"):
        with op.batch_alter_table("eval_results") as batch:
            batch.drop_column("seed_group_id")
    if _column_exists("eval_results", "is_aggregate"):
        with op.batch_alter_table("eval_results") as batch:
            batch.drop_column("is_aggregate")

    if _index_exists("experiments", "ix_experiments_seed_group_id"):
        op.drop_index(
            "ix_experiments_seed_group_id", table_name="experiments"
        )
    if _column_exists("experiments", "seed_group_id"):
        with op.batch_alter_table("experiments") as batch:
            batch.drop_column("seed_group_id")
    if _column_exists("experiments", "seed_value"):
        with op.batch_alter_table("experiments") as batch:
            batch.drop_column("seed_value")
