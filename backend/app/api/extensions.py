"""Extensions API — plugin contract validators (priority.md P37, Wave H).

Surfaces the plugin-contract service over HTTP so operators / CLI can
list configured extensions, validate a plugin module before adding it
to settings, and reload modules in-place.

Routes:

- ``GET  /api/extensions`` — list every plugin kind with its
  contract version, configured modules, load errors, and recognised
  hook names.
- ``POST /api/extensions/validate`` — body: ``{kind, module, force_reload?}``.
  Imports the module without registering it and runs the contract
  suite. Returns the full :class:`PluginContractReport` payload.
- ``POST /api/extensions/reload`` — body: ``{kind?}``. Re-imports plugin
  modules listed in settings; kinds without a live loader return
  ``status="not_supported"``.

Scaffold + write endpoints (`/extensions/scaffold`) land in P38.

Stable reason codes:
- ``unknown_plugin_kind`` (400)
- ``extension_module_required`` (400)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.plugin_contract_service import (
    list_extensions,
    reload_extensions,
    validate_extension,
)
from app.services.plugin_contracts import KNOWN_PLUGIN_KINDS


router = APIRouter(prefix="/extensions", tags=["Extensions"])


_BAD_REQUEST_CODES = {"unknown_plugin_kind", "extension_module_required"}


def _raise_for(exc: ValueError) -> None:
    detail = str(exc) or "extension_error"
    head = detail.split(":", 1)[0]
    if head in _BAD_REQUEST_CODES:
        raise HTTPException(400, detail=detail) from exc
    raise HTTPException(400, detail=detail) from exc


class ValidateExtensionRequest(BaseModel):
    kind: str = Field(..., min_length=1, max_length=64)
    module: str = Field(..., min_length=1, max_length=256)
    force_reload: bool = False


class ReloadExtensionRequest(BaseModel):
    kind: str | None = Field(default=None, max_length=64)


@router.get("")
async def list_extensions_route():
    payload = list_extensions()
    payload["known_kinds"] = list(KNOWN_PLUGIN_KINDS)
    return payload


@router.post("/validate")
async def validate_extension_route(req: ValidateExtensionRequest):
    try:
        return validate_extension(
            kind=req.kind,
            module_path=req.module,
            force_reload=req.force_reload,
        )
    except ValueError as exc:
        _raise_for(exc)


@router.post("/reload")
async def reload_extensions_route(req: ReloadExtensionRequest | None = None):
    payload = req or ReloadExtensionRequest()
    try:
        return reload_extensions(kind=payload.kind)
    except ValueError as exc:
        _raise_for(exc)
