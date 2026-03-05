"""Domain pack ORM model for composable domain defaults and overlays."""

import enum
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DomainPackStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class DomainPack(Base):
    __tablename__ = "domain_packs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pack_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    version: Mapped[str] = mapped_column(String(32), default="1.0.0")
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    owner: Mapped[str] = mapped_column(String(128), default="platform")
    status: Mapped[DomainPackStatus] = mapped_column(
        Enum(DomainPackStatus),
        default=DomainPackStatus.ACTIVE,
        nullable=False,
    )
    schema_ref: Mapped[str] = mapped_column(String(255), default="slm.domain-pack/v1")
    default_profile_id: Mapped[str | None] = mapped_column(String(128), default=None)
    contract: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )

    projects = relationship("Project", back_populates="domain_pack")

    def __repr__(self) -> str:
        return f"<DomainPack {self.id}: {self.pack_id}@{self.version}>"
