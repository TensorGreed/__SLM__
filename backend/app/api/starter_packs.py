"""Starter-pack catalog API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services import starter_pack_service

router = APIRouter(prefix="/starter-packs", tags=["Starter Packs"])


@router.get("")
async def list_starter_packs(
    include_registry_meta: bool = Query(
        default=False,
        description="When true, include catalog-level plugin and registry metadata.",
    )
):
    if include_registry_meta:
        return starter_pack_service.list_starter_pack_catalog()
    packs = starter_pack_service.list_starter_packs()
    return {
        "starter_packs": packs,
        "count": len(packs),
    }


@router.get("/catalog")
async def get_starter_pack_catalog():
    return starter_pack_service.list_starter_pack_catalog()


@router.get("/{starter_pack_id}")
async def get_starter_pack(starter_pack_id: str):
    pack = starter_pack_service.get_starter_pack_by_id(starter_pack_id)
    if pack is None:
        raise HTTPException(404, f"Starter pack '{starter_pack_id}' not found")
    return pack


@router.post("/reload")
async def reload_starter_pack_catalog(
    force_reload: bool = Query(
        default=True,
        description="Reload configured starter-pack plugin modules before returning catalog state.",
    )
):
    starter_pack_service.clear_starter_pack_plugins()
    reload_result = starter_pack_service.load_starter_pack_plugins_from_settings(
        force_reload=force_reload,
    )
    return {
        "reload": reload_result,
        "catalog": starter_pack_service.list_starter_pack_catalog(),
    }
