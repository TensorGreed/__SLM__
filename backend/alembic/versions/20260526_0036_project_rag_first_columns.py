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


def _existing_columns(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    # SQLite doesn't support adding a FK constraint via ALTER TABLE
    # (only at CREATE TABLE time), and using alembic's batch_alter_table
    # to recreate the projects table on every prod upgrade is heavier
    # than the constraint is worth. SQLite also doesn't enforce FKs by
    # default. We keep the ORM-level ForeignKey on Project.parent_project_id
    # so SQLAlchemy relationships still resolve correctly; a follow-on
    # migration can promote this to a real DB-level constraint when
    # the project moves off SQLite.
    #
    # Idempotent: skip ``add_column`` for any column already present.
    # An earlier draft of this migration added ``parent_project_id``
    # successfully before failing on ``create_foreign_key`` (SQLite
    # NotImplementedError), and the partial state survived the
    # rollback. The guard below makes the migration safe to re-apply
    # from that state.
    existing = _existing_columns("projects")
    if "parent_project_id" not in existing:
        op.add_column(
            "projects",
            sa.Column("parent_project_id", sa.Integer(), nullable=True),
        )
    if "runtime_config" not in existing:
        op.add_column(
            "projects",
            sa.Column("runtime_config", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    existing = _existing_columns("projects")
    if "parent_project_id" in existing:
        op.drop_column("projects", "parent_project_id")
    if "runtime_config" in existing:
        op.drop_column("projects", "runtime_config")
