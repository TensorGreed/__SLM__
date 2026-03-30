"""Add universal base-model registry table for import/compatibility workflows.

Revision ID: 20260312_0012
Revises: 20260311_0011
Create Date: 2026-03-12 11:45:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260312_0012"
down_revision = "20260311_0011"
branch_labels = None
depends_on = None


base_model_source_enum = sa.Enum(
    "huggingface",
    "local_path",
    "catalog",
    name="basemodelsourcetype",
)


def upgrade() -> None:
    op.create_table(
        "base_model_registry_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_key", sa.String(length=255), nullable=False),
        sa.Column("source_type", base_model_source_enum, nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("model_family", sa.String(length=64), nullable=False),
        sa.Column("architecture", sa.String(length=64), nullable=False),
        sa.Column("tokenizer", sa.String(length=255), nullable=True),
        sa.Column("chat_template", sa.Text(), nullable=True),
        sa.Column("context_length", sa.Integer(), nullable=True),
        sa.Column("parameter_count", sa.Integer(), nullable=True),
        sa.Column("params_estimate_b", sa.Float(), nullable=True),
        sa.Column("license", sa.String(length=128), nullable=True),
        sa.Column("modalities", sa.JSON(), nullable=True),
        sa.Column("quantization_support", sa.JSON(), nullable=True),
        sa.Column("peft_support", sa.Boolean(), nullable=False),
        sa.Column("full_finetune_support", sa.Boolean(), nullable=False),
        sa.Column("supported_task_families", sa.JSON(), nullable=True),
        sa.Column("training_mode_support", sa.JSON(), nullable=True),
        sa.Column("estimated_hardware_needs", sa.JSON(), nullable=True),
        sa.Column("deployment_target_compatibility", sa.JSON(), nullable=True),
        sa.Column("normalization_contract_version", sa.String(length=128), nullable=False),
        sa.Column("normalized_metadata", sa.JSON(), nullable=True),
        sa.Column("provenance", sa.JSON(), nullable=True),
        sa.Column("cache_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("cache_status", sa.String(length=32), nullable=False),
        sa.Column("refresh_count", sa.Integer(), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_refreshed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_key"),
    )
    op.create_index(
        "ix_base_model_registry_entries_model_key",
        "base_model_registry_entries",
        ["model_key"],
        unique=True,
    )
    op.create_index(
        "ix_base_model_registry_entries_source_type",
        "base_model_registry_entries",
        ["source_type"],
        unique=False,
    )
    op.create_index(
        "ix_base_model_registry_entries_source_ref",
        "base_model_registry_entries",
        ["source_ref"],
        unique=False,
    )
    op.create_index(
        "ix_base_model_registry_entries_model_family",
        "base_model_registry_entries",
        ["model_family"],
        unique=False,
    )
    op.create_index(
        "ix_base_model_registry_entries_architecture",
        "base_model_registry_entries",
        ["architecture"],
        unique=False,
    )
    op.create_index(
        "ix_base_model_registry_entries_license",
        "base_model_registry_entries",
        ["license"],
        unique=False,
    )
    op.create_index(
        "ix_base_model_registry_entries_last_refreshed_at",
        "base_model_registry_entries",
        ["last_refreshed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_base_model_registry_entries_last_refreshed_at",
        table_name="base_model_registry_entries",
    )
    op.drop_index(
        "ix_base_model_registry_entries_license",
        table_name="base_model_registry_entries",
    )
    op.drop_index(
        "ix_base_model_registry_entries_architecture",
        table_name="base_model_registry_entries",
    )
    op.drop_index(
        "ix_base_model_registry_entries_model_family",
        table_name="base_model_registry_entries",
    )
    op.drop_index(
        "ix_base_model_registry_entries_source_ref",
        table_name="base_model_registry_entries",
    )
    op.drop_index(
        "ix_base_model_registry_entries_source_type",
        table_name="base_model_registry_entries",
    )
    op.drop_index(
        "ix_base_model_registry_entries_model_key",
        table_name="base_model_registry_entries",
    )
    op.drop_table("base_model_registry_entries")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("DROP TYPE IF EXISTS basemodelsourcetype"))
