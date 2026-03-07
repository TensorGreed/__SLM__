"""Add project evaluation pack preference field.

Revision ID: 20260307_0009
Revises: 20260306_0008
Create Date: 2026-03-07 09:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260307_0009"
down_revision = "20260306_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "evaluation_preferred_pack_id",
            sa.String(length=128),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("projects", "evaluation_preferred_pack_id")
