"""Persisted support bundles (priority.md P34, Wave G).

One row per ``POST /projects/{id}/support-bundle`` invocation. The
service writes the actual bundle (zip of sectioned JSON files) to disk
under ``DATA_DIR/support_bundles/{project_id}/`` and records the
metadata here.

The download endpoint requires the row's ``download_token`` (random
hex). This is a thin substitute for true signed URLs (no object
storage in this codebase) — the download URL is unguessable, scoped
to one bundle, and expires.

Per-section redaction stats land in ``redactions_applied`` so the
operator can verify the bundle was scrubbed before forwarding it to
support.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SupportBundle(Base):
    __tablename__ = "support_bundles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )

    # Stable hex id used in the download URL (separate from the SQL
    # primary key so bundles aren't enumerable by id).
    bundle_uid: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )

    # Random token the download endpoint requires. Single-use is left
    # for a future hardening pass — for now, valid until ``expires_at``.
    download_token: Mapped[str] = mapped_column(String(64), nullable=False)

    file_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )

    actor: Mapped[str] = mapped_column(String(128), nullable=False, default="system")

    # Section-by-section redaction summary:
    #   {"run_events": {"redactions": 4}, "training_manifests": {...}, ...}
    redactions_applied: Mapped[dict] = mapped_column(JSON, default=dict)

    # Section-by-section row counts so the support-bundle UI can show
    # "47 events, 3 experiments, 1 training manifest" without re-opening
    # the zip.
    section_counts: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<SupportBundle id={self.id} pid={self.project_id} "
            f"uid={self.bundle_uid[:8]} size={self.size_bytes}>"
        )
