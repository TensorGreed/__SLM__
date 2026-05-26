"""Project parent_project_id + runtime_config columns (USER-SUCCESS Epic 7 Phase 7b).

Adds:
  - projects.parent_project_id — nullable self-referential FK. Set
    when a project was created by
    ``rag_project_service.clone_project_for_rag`` so the UI can
    render a "cloned from" provenance chip.
  - projects.runtime_config — nullable JSON. Carries the
    ``rag_first`` flag plus a mirrored ``auto_rag.enabled`` value
    that the playground inference path consults. When
    ``rag_first=true`` the playground uses the base model with
    auto-RAG preamble (no LoRA adapter) and the training-start
    endpoint refuses requests.

Both columns are nullable so existing projects round-trip
unchanged.

Revision ID: 20260526_0036
Revises: 20260523_0035
Create Date: 2026-05-26 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260526_0036"
down_revision = "20260523_0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("parent_project_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_projects_parent_project_id",
        source_table="projects",
        referent_table="projects",
        local_cols=["parent_project_id"],
        remote_cols=["id"],
    )
    op.add_column(
        "projects",
        sa.Column("runtime_config", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_projects_parent_project_id",
        "projects",
        type_="foreignkey",
    )
    op.drop_column("projects", "parent_project_id")
    op.drop_column("projects", "runtime_config")
