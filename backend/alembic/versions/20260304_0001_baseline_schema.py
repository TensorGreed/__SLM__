"""Baseline schema for SLM platform tables.

Revision ID: 20260304_0001
Revises:
Create Date: 2026-03-04 12:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260304_0001"
down_revision = None
branch_labels = None
depends_on = None


project_status_enum = sa.Enum(
    "draft", "active", "paused", "completed", "failed", name="projectstatus"
)
pipeline_stage_enum = sa.Enum(
    "ingestion",
    "cleaning",
    "gold_set",
    "synthetic",
    "dataset_prep",
    "tokenization",
    "training",
    "evaluation",
    "compression",
    "export",
    "completed",
    name="pipelinestage",
)
dataset_type_enum = sa.Enum(
    "raw",
    "cleaned",
    "gold_dev",
    "gold_test",
    "synthetic",
    "train",
    "validation",
    "test",
    name="datasettype",
)
document_status_enum = sa.Enum(
    "pending", "processing", "accepted", "rejected", "error", name="documentstatus"
)
experiment_status_enum = sa.Enum(
    "pending", "running", "completed", "failed", "cancelled", name="experimentstatus"
)
training_mode_enum = sa.Enum(
    "sft", "domain_pretrain", "dpo", "orpo", name="trainingmode"
)
export_format_enum = sa.Enum(
    "gguf", "onnx", "tensorrt", "huggingface", "docker", name="exportformat"
)
export_status_enum = sa.Enum(
    "pending", "in_progress", "completed", "failed", name="exportstatus"
)
global_role_enum = sa.Enum("admin", "engineer", "viewer", name="globalrole")
project_role_enum = sa.Enum("owner", "editor", "viewer", name="projectrole")


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", project_status_enum, nullable=False),
        sa.Column("pipeline_stage", pipeline_stage_enum, nullable=False),
        sa.Column("base_model_name", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=128), nullable=False),
        sa.Column("role", global_role_enum, nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )

    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("dataset_type", dataset_type_enum, nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("record_count", sa.Integer(), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("is_locked", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", experiment_status_enum, nullable=False),
        sa.Column("training_mode", training_mode_enum, nullable=False),
        sa.Column("base_model", sa.String(length=255), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("final_train_loss", sa.Float(), nullable=True),
        sa.Column("final_eval_loss", sa.Float(), nullable=True),
        sa.Column("total_epochs", sa.Integer(), nullable=True),
        sa.Column("total_steps", sa.Integer(), nullable=True),
        sa.Column("output_dir", sa.String(length=1024), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("key_hash", sa.String(length=128), nullable=False),
        sa.Column("key_prefix", sa.String(length=16), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_hash"),
    )

    op.create_table(
        "project_memberships",
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role", project_role_enum, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("project_id", "user_id"),
    )

    op.create_table(
        "raw_documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("file_type", sa.String(length=32), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("sensitivity", sa.String(length=64), nullable=True),
        sa.Column("license_info", sa.String(length=255), nullable=True),
        sa.Column("status", document_status_enum, nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("record_count", sa.Integer(), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "checkpoints",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("train_loss", sa.Float(), nullable=True),
        sa.Column("eval_loss", sa.Float(), nullable=True),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("is_best", sa.Boolean(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "eval_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("dataset_name", sa.String(length=255), nullable=False),
        sa.Column("eval_type", sa.String(length=64), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("pass_rate", sa.Float(), nullable=True),
        sa.Column("risk_severity", sa.String(length=32), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "exports",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("export_format", export_format_enum, nullable=False),
        sa.Column("status", export_status_enum, nullable=False),
        sa.Column("quantization", sa.String(length=32), nullable=True),
        sa.Column("output_path", sa.String(length=1024), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("manifest", sa.JSON(), nullable=True),
        sa.Column("eval_report", sa.JSON(), nullable=True),
        sa.Column("safety_scorecard", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("method", sa.String(length=16), nullable=False),
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("project_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=255), nullable=True),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("exports")
    op.drop_table("eval_results")
    op.drop_table("checkpoints")
    op.drop_table("dataset_versions")
    op.drop_table("raw_documents")
    op.drop_table("project_memberships")
    op.drop_table("api_keys")
    op.drop_table("experiments")
    op.drop_table("datasets")
    op.drop_table("users")
    op.drop_table("projects")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for enum_name in [
            "projectrole",
            "globalrole",
            "exportstatus",
            "exportformat",
            "trainingmode",
            "experimentstatus",
            "documentstatus",
            "datasettype",
            "pipelinestage",
            "projectstatus",
        ]:
            op.execute(sa.text(f"DROP TYPE IF EXISTS {enum_name}"))
