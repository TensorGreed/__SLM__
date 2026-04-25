"""Pause / resume training (priority.md P17).

Adds:
  - experiments.pause_requested — boolean flag honored by runtime polling
    loops. Set true by POST /training/runs/{id}/pause; runtime observes
    it on its next iteration, writes a resume-capable checkpoint,
    transitions status to PAUSED, and clears the flag.
  - 'paused' enum value on experimentstatus (Postgres only). SQLite
    stores enums as VARCHAR + CHECK and ``Base.metadata.create_all``
    regenerates the constraint with the new value, so no SQLite branch
    is needed here.

Revision ID: 20260425_0020
Revises: 20260425_0019
Create Date: 2026-04-25 00:00:01
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260425_0020"
down_revision = "20260425_0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # New value on the experimentstatus enum — Postgres only. SQLite
    # treats Enum as VARCHAR + CHECK and the model-level enum lists the
    # full set, so re-creating the constraint in SQLite is handled by
    # Base.metadata.create_all on autocreate.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TYPE experimentstatus ADD VALUE IF NOT EXISTS 'paused'")

    op.add_column(
        "experiments",
        sa.Column(
            "pause_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    # Drop the server_default once the column exists so future inserts go
    # through the ORM default (kept identical at False).
    with op.batch_alter_table("experiments") as batch_op:
        batch_op.alter_column("pause_requested", server_default=None)


def downgrade() -> None:
    op.drop_column("experiments", "pause_requested")
    # Postgres has no clean way to remove a value from an enum without
    # rebuilding the type — leave 'paused' in place on downgrade.
