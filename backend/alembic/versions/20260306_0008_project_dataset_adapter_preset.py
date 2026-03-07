"""Add project dataset adapter preset persistence.

Revision ID: 20260306_0008
Revises: 20260306_0007
Create Date: 2026-03-06 13:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260306_0008"
down_revision = "20260306_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "dataset_adapter_preset",
            sa.JSON(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("projects", "dataset_adapter_preset")
