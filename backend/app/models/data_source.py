"""Licence & provenance ledger: one row per place data came from.

A ``DataSource`` is an upload, a website crawl, a git repository, a
HuggingFace / Kaggle / URL import, a dataset-import wizard run, a predefined
split, a demo bundle or an external benchmark. Documents point at it through
``raw_documents.data_source_id``; rows a source wrote straight into
``synthetic.jsonl`` carry ``data_source_id`` themselves. ``usage`` decides
where its rows may go: ``any`` (today's behaviour), ``train`` (never held
out), ``eval`` (held out of training, goes to the test split) or ``rag``
(indexed for retrieval, never trained on or evaluated). See
``services/provenance_service.py``.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    # URL / HuggingFace id / repo URL / filename.
    locator: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    # Commit / revision / split / config.
    source_ref: Mapped[str | None] = mapped_column(String(255), default=None)
    # SPDX id or free text; NULL means unknown.
    licence: Mapped[str | None] = mapped_column(String(255), default=None)
    licence_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    licence_verified_by: Mapped[str | None] = mapped_column(String(255), default=None)
    licence_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    usage: Mapped[str] = mapped_column(String(8), nullable=False, default="any")
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=_utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return f"<DataSource {self.id}: {self.source_kind} {self.title!r}>"
