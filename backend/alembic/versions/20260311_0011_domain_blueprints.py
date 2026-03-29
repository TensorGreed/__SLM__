"""Add domain blueprint revisions and beginner-mode project fields.

Revision ID: 20260311_0011
Revises: 20260310_0010
Create Date: 2026-03-11 09:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260311_0011"
down_revision = "20260310_0010"
branch_labels = None
depends_on = None


domain_blueprint_status_enum = sa.Enum(
    "draft",
    "active",
    "archived",
    name="domainblueprintstatus",
)


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("beginner_mode", sa.Boolean(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "projects",
        sa.Column("active_domain_blueprint_version", sa.Integer(), nullable=True),
    )

    op.create_table(
        "domain_blueprints",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", domain_blueprint_status_enum, nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("brief_text", sa.Text(), nullable=True),
        sa.Column("domain_name", sa.String(length=255), nullable=True),
        sa.Column("problem_statement", sa.Text(), nullable=True),
        sa.Column("target_user_persona", sa.Text(), nullable=True),
        sa.Column("task_family", sa.String(length=64), nullable=True),
        sa.Column("input_modality", sa.String(length=64), nullable=True),
        sa.Column("expected_output_schema", sa.JSON(), nullable=True),
        sa.Column("expected_output_examples", sa.JSON(), nullable=True),
        sa.Column("safety_compliance_notes", sa.JSON(), nullable=True),
        sa.Column("deployment_target_constraints", sa.JSON(), nullable=True),
        sa.Column("success_metrics", sa.JSON(), nullable=True),
        sa.Column("glossary", sa.JSON(), nullable=True),
        sa.Column("unresolved_assumptions", sa.JSON(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("analysis_metadata", sa.JSON(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "version", name="uq_domain_blueprints_project_version"),
    )
    op.create_index("ix_domain_blueprints_project_id", "domain_blueprints", ["project_id"], unique=False)
    op.create_index("ix_domain_blueprints_status", "domain_blueprints", ["status"], unique=False)
    op.create_index("ix_domain_blueprints_created_at", "domain_blueprints", ["created_at"], unique=False)

    op.alter_column("projects", "beginner_mode", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_domain_blueprints_created_at", table_name="domain_blueprints")
    op.drop_index("ix_domain_blueprints_status", table_name="domain_blueprints")
    op.drop_index("ix_domain_blueprints_project_id", table_name="domain_blueprints")
    op.drop_table("domain_blueprints")
    op.drop_column("projects", "active_domain_blueprint_version")
    op.drop_column("projects", "beginner_mode")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("DROP TYPE IF EXISTS domainblueprintstatus"))
