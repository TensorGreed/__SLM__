"""Domain profile ORM model for profile-driven pipeline behavior."""

import enum
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DomainProfileStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class DomainProfile(Base):
    __tablename__ = "domain_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    version: Mapped[str] = mapped_column(String(32), default="1.0.0")
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    owner: Mapped[str] = mapped_column(String(128), default="platform")
    status: Mapped[DomainProfileStatus] = mapped_column(
        Enum(DomainProfileStatus),
        default=DomainProfileStatus.ACTIVE,
        nullable=False,
    )
    schema_ref: Mapped[str] = mapped_column(String(255), default="slm.domain-profile/v1")
    contract: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )

    projects = relationship("Project", back_populates="domain_profile")

    def __repr__(self) -> str:
        return f"<DomainProfile {self.id}: {self.profile_id}@{self.version}>"
