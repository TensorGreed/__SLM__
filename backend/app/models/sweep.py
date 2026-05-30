"""Sweep ORM model — first-class record for a hyperparameter bake-off.

Replaces the legacy ``config._sweep.sweep_id`` join (where the only
record of a sweep was a JSON breadcrumb scattered across every cell
``Experiment``). With a real ``sweeps`` table the readers stop scanning
all experiments and just look up by id; the history sidebar can list
past sweeps cheaply; the inconclusive-verdict coach nudge becomes a
``select(Sweep).order_by(Sweep.created_at.desc()).limit(1)``.

Cells are linked back to their parent sweep via ``Experiment.sweep_id``
(a nullable FK — most experiments aren't sweep cells). The legacy
``config._sweep`` JSON breadcrumb is still written on every cell for
backward-compat / debugging, but the FK is the authoritative source.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Sweep(Base):
    __tablename__ = "sweeps"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )
    # The 12-char hex token shared by every cell — surfaced in URLs and
    # the legacy config._sweep.sweep_id breadcrumb. Indexed because the
    # API still accepts the token for backward-compat with old clients.
    sweep_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    # The base model at dispatch time — used by the pre-flight budget
    # estimator's "same_base_model" basis even when later cells migrate
    # to a different model via the base_model_values axis.
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    # The project's selected recipe at dispatch — captured here so the
    # "same_base_and_recipe" basis is computable after a recipe change.
    recipe_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Axis values as supplied to expand_grid, stored verbatim so the
    # history sidebar can reconstruct the axes a sweep was run with.
    axes: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Stop-when-met threshold (0..1), null when the sweep should run
    # every cell to completion.
    quality_target: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Planned cell count from expand_grid. The actual dispatched count
    # is computed live as ``len(experiments WHERE sweep_id=...)``.
    requested_cells: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    cells = relationship(
        "Experiment",
        back_populates="sweep",
        foreign_keys="Experiment.sweep_id",
    )

    def __repr__(self) -> str:
        return f"<Sweep {self.id}: {self.sweep_id} ({self.requested_cells} cells)>"
