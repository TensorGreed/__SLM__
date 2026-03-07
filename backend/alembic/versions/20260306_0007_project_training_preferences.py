"""Add project-level training preference fields.

Revision ID: 20260306_0007
Revises: 20260305_0006
Create Date: 2026-03-06 11:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260306_0007"
down_revision = "20260305_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "training_preferred_plan_profile",
            sa.String(length=32),
            nullable=True,
            server_default="balanced",
        ),
    )
    op.alter_column(
        "projects",
        "training_preferred_plan_profile",
        server_default=None,
    )


def downgrade() -> None:
    op.drop_column("projects", "training_preferred_plan_profile")
