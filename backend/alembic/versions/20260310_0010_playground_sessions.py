"""Add persisted playground sessions.

Revision ID: 20260310_0010
Revises: 20260307_0009
Create Date: 2026-03-10 11:15:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260310_0010"
down_revision = "20260307_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "playground_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=512), nullable=False),
        sa.Column("api_url", sa.String(length=2048), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("max_tokens", sa.Integer(), nullable=False),
        sa.Column("transcript", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_playground_sessions_project_id", "playground_sessions", ["project_id"], unique=False)
    op.create_index("ix_playground_sessions_created_at", "playground_sessions", ["created_at"], unique=False)
    op.create_index("ix_playground_sessions_updated_at", "playground_sessions", ["updated_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_playground_sessions_updated_at", table_name="playground_sessions")
    op.drop_index("ix_playground_sessions_created_at", table_name="playground_sessions")
    op.drop_index("ix_playground_sessions_project_id", table_name="playground_sessions")
    op.drop_table("playground_sessions")
