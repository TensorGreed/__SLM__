"""Immutable training-run manifest table (priority.md P14).

Creates:
  - training_manifests — one row per experiment (run), capturing every
    input reference + resolved config + environment digest needed to
    reproduce the run later (Wave D P15).

Revision ID: 20260424_0018
Revises: 20260423_0017
Create Date: 2026-04-24 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260424_0018"
down_revision = "20260423_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_manifests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("base_model_registry_id", sa.Integer(), nullable=True),
        sa.Column("base_model_cache_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("base_model_source_ref", sa.String(length=512), nullable=True),
        sa.Column("dataset_adapter_id", sa.Integer(), nullable=True),
        sa.Column("dataset_adapter_version", sa.Integer(), nullable=True),
        sa.Column("blueprint_revision_id", sa.Integer(), nullable=True),
        sa.Column("blueprint_version", sa.Integer(), nullable=True),
        sa.Column("dataset_snapshot_ids", sa.JSON(), nullable=True),
        sa.Column("recipe_id", sa.String(length=256), nullable=True),
        sa.Column("runtime_id", sa.String(length=128), nullable=True),
        sa.Column("training_mode", sa.String(length=64), nullable=True),
        sa.Column("tokenizer_name", sa.String(length=256), nullable=True),
        sa.Column("tokenizer_config_hash", sa.String(length=128), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("resolved_config", sa.JSON(), nullable=True),
        sa.Column("git_sha", sa.String(length=64), nullable=True),
        sa.Column("pip_freeze_blob", sa.Text(), nullable=True),
        sa.Column("pip_freeze_hash", sa.String(length=128), nullable=True),
        sa.Column("env_digest", sa.String(length=128), nullable=True),
        sa.Column("artifact_ids", sa.JSON(), nullable=True),
        sa.Column("capture_warnings", sa.JSON(), nullable=True),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["base_model_registry_id"], ["base_model_registry_entries.id"]),
        sa.ForeignKeyConstraint(["dataset_adapter_id"], ["dataset_adapter_definitions.id"]),
        sa.ForeignKeyConstraint(["blueprint_revision_id"], ["domain_blueprints.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("experiment_id", name="uq_training_manifests_experiment_id"),
    )
    op.create_index(
        "ix_training_manifests_experiment_id",
        "training_manifests",
        ["experiment_id"],
        unique=False,
    )
    op.create_index(
        "ix_training_manifests_project_id",
        "training_manifests",
        ["project_id"],
        unique=False,
    )
    op.create_index(
        "ix_training_manifests_base_model_registry_id",
        "training_manifests",
        ["base_model_registry_id"],
        unique=False,
    )
    op.create_index(
        "ix_training_manifests_dataset_adapter_id",
        "training_manifests",
        ["dataset_adapter_id"],
        unique=False,
    )
    op.create_index(
        "ix_training_manifests_blueprint_revision_id",
        "training_manifests",
        ["blueprint_revision_id"],
        unique=False,
    )
    op.create_index(
        "ix_training_manifests_runtime_id",
        "training_manifests",
        ["runtime_id"],
        unique=False,
    )
    op.create_index(
        "ix_training_manifests_git_sha",
        "training_manifests",
        ["git_sha"],
        unique=False,
    )
    op.create_index(
        "ix_training_manifests_env_digest",
        "training_manifests",
        ["env_digest"],
        unique=False,
    )
    op.create_index(
        "ix_training_manifests_captured_at",
        "training_manifests",
        ["captured_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_training_manifests_captured_at", table_name="training_manifests")
    op.drop_index("ix_training_manifests_env_digest", table_name="training_manifests")
    op.drop_index("ix_training_manifests_git_sha", table_name="training_manifests")
    op.drop_index("ix_training_manifests_runtime_id", table_name="training_manifests")
    op.drop_index("ix_training_manifests_blueprint_revision_id", table_name="training_manifests")
    op.drop_index("ix_training_manifests_dataset_adapter_id", table_name="training_manifests")
    op.drop_index("ix_training_manifests_base_model_registry_id", table_name="training_manifests")
    op.drop_index("ix_training_manifests_project_id", table_name="training_manifests")
    op.drop_index("ix_training_manifests_experiment_id", table_name="training_manifests")
    op.drop_table("training_manifests")
