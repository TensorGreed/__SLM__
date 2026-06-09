"""Quality-Lift phase 4 slice 1 — label_noise_scans table.

Adds:
  - ``label_noise_scans`` — one row per scan invocation. Persists
    history across re-scans (user trains → scan → cleans → re-train
    → scan again) so the Coach nudge can gate re-fire on "current
    labeled count meaningfully grew since latest scan" without
    rebuilding the comparison from JSON blobs.

Single new table; no column adds elsewhere. Idempotent — skip
``create_table`` if the table already exists (mirrors the pattern
from 0037_jobs_table.py).

Revision ID: 20260609_0048
Revises: 20260608_0047
Create Date: 2026-06-09 11:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260609_0048"
down_revision = "20260608_0047"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return name in inspector.get_table_names()


def _index_exists(table: str, index: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table not in inspector.get_table_names():
        return False
    for ix in inspector.get_indexes(table):
        if ix.get("name") == index:
            return True
    return False


def upgrade() -> None:
    if not _table_exists("label_noise_scans"):
        op.create_table(
            "label_noise_scans",
            sa.Column(
                "id", sa.Integer(), primary_key=True, autoincrement=True
            ),
            sa.Column(
                "project_id",
                sa.Integer(),
                sa.ForeignKey("projects.id"),
                nullable=False,
            ),
            sa.Column(
                "base_experiment_id",
                sa.Integer(),
                sa.ForeignKey("experiments.id"),
                nullable=True,
            ),
            sa.Column(
                "status",
                sa.Enum(
                    "QUEUED",
                    "RUNNING",
                    "SUCCEEDED",
                    "FAILED",
                    "CANCELLED",
                    name="labelnoisescanstatus",
                ),
                nullable=False,
                server_default="QUEUED",
            ),
            sa.Column("label_count_at_scan", sa.Integer(), nullable=True),
            sa.Column("suspected_count", sa.Integer(), nullable=True),
            sa.Column(
                "confidence_threshold",
                sa.Float(),
                nullable=False,
                server_default="0.85",
            ),
            sa.Column(
                "given_label_floor",
                sa.Float(),
                nullable=False,
                server_default="0.15",
            ),
            sa.Column("result_payload", sa.JSON(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column(
                "job_id",
                sa.Integer(),
                sa.ForeignKey("jobs.id"),
                nullable=True,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "completed_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
        )

    if not _index_exists("label_noise_scans", "ix_label_noise_scans_project_id"):
        op.create_index(
            "ix_label_noise_scans_project_id",
            "label_noise_scans",
            ["project_id"],
        )
    if not _index_exists(
        "label_noise_scans", "ix_label_noise_scans_base_experiment_id"
    ):
        op.create_index(
            "ix_label_noise_scans_base_experiment_id",
            "label_noise_scans",
            ["base_experiment_id"],
        )
    if not _index_exists("label_noise_scans", "ix_label_noise_scans_job_id"):
        op.create_index(
            "ix_label_noise_scans_job_id",
            "label_noise_scans",
            ["job_id"],
        )


def downgrade() -> None:
    if _index_exists("label_noise_scans", "ix_label_noise_scans_job_id"):
        op.drop_index(
            "ix_label_noise_scans_job_id", table_name="label_noise_scans"
        )
    if _index_exists(
        "label_noise_scans", "ix_label_noise_scans_base_experiment_id"
    ):
        op.drop_index(
            "ix_label_noise_scans_base_experiment_id",
            table_name="label_noise_scans",
        )
    if _index_exists("label_noise_scans", "ix_label_noise_scans_project_id"):
        op.drop_index(
            "ix_label_noise_scans_project_id",
            table_name="label_noise_scans",
        )
    if _table_exists("label_noise_scans"):
        op.drop_table("label_noise_scans")
