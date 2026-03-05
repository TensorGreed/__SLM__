"""Add domain packs and project pack binding.

Revision ID: 20260305_0004
Revises: 20260305_0003
Create Date: 2026-03-05 23:59:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260305_0004"
down_revision = "20260305_0003"
branch_labels = None
depends_on = None


domain_pack_status_enum = sa.Enum(
    "draft",
    "active",
    "deprecated",
    name="domainpackstatus",
)


def upgrade() -> None:
    op.create_table(
        "domain_packs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pack_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("owner", sa.String(length=128), nullable=False),
        sa.Column("status", domain_pack_status_enum, nullable=False),
        sa.Column("schema_ref", sa.String(length=255), nullable=False),
        sa.Column("default_profile_id", sa.String(length=128), nullable=True),
        sa.Column("contract", sa.JSON(), nullable=False),
        sa.Column("is_system", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pack_id"),
    )

    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.add_column(sa.Column("domain_pack_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_projects_domain_pack_id",
            "domain_packs",
            ["domain_pack_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_constraint("fk_projects_domain_pack_id", type_="foreignkey")
        batch_op.drop_column("domain_pack_id")

    op.drop_table("domain_packs")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("DROP TYPE IF EXISTS domainpackstatus"))
