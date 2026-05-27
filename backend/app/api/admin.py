"""Admin API surface (USER-SUCCESS Epic 1, T5).

Admin-scoped endpoints that aggregate across projects — not surfaced
on the per-project workspace. The first endpoint here is the
forecast-vs-reality calibration view, intended to support evidence-
based retuning of the heuristic in ``trainability_forecast_service``.

Auth: when ``AUTH_ENABLED`` is true, requires the principal's role to
be ``GlobalRole.ADMIN``. In dev (auth disabled) the endpoints are open
— same convention as the rest of the admin surfaces (see ``auth.py``).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.auth import GlobalRole
from app.security import get_request_principal


router = APIRouter(prefix="/admin", tags=["Admin"])


def _require_admin(request: Request) -> None:
    """Raise 401/403 when the principal isn't an admin (only enforced
    when ``AUTH_ENABLED``)."""
    if not settings.AUTH_ENABLED:
        return
    principal = get_request_principal(request)
    if not principal:
        raise HTTPException(401, "Authentication required")
    if principal.role != GlobalRole.ADMIN:
        raise HTTPException(403, "Admin role required")


@router.get("/forecast/calibration")
async def get_forecast_calibration(
    request: Request,
    recipe: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Bucketed forecast-vs-reality calibration. USER-SUCCESS Epic 1, T5.

    Returns one entry per 10%-confidence bucket with the count of
    predictions in that bucket and the actual gate-pass rate for the
    resolved ones. Optionally filtered by ``recipe`` so we can spot
    per-recipe calibration drift (the heuristic uses a single
    difficulty coefficient per recipe; if classification is well-
    calibrated but span-extraction is not, that's a coefficient bug).

    No new UI in v1 — the admin reads the JSON and feeds it into the
    next coefficient-retuning pass.
    """
    _require_admin(request)
    from app.services.trainability_forecast_service import (
        compute_calibration_buckets,
    )

    return await compute_calibration_buckets(db, recipe_id=recipe)
