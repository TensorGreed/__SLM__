"""Extensions API — plugin contract validators + scaffold generator
(priority.md P37, P38, Wave H).

Surfaces the plugin-contract service over HTTP so operators / CLI can
list configured extensions, validate a plugin module before adding it
to settings, reload modules in-place, and scaffold a contract-valid
starter module for any plugin kind.

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
- ``POST /api/extensions/scaffold`` — body:
  ``{kind, plugin_id, display_name?, description?, author?, version?, export_dir?, write?}``.
  Generates a contract-compliant scaffold (module + test stub + README)
  and (unless ``write=False``) writes it under
  ``DATA_DIR/extension_scaffolds/{kind}/{plugin_id}``. Returns the file
  contents inline plus the resolved output dir.

Stable reason codes:
- ``unknown_plugin_kind`` (400)
- ``extension_module_required`` (400)
- ``scaffold_plugin_id_required`` (400)
- ``scaffold_plugin_id_invalid`` (400)
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
from app.services.scaffold_service import generate_extension_scaffold


router = APIRouter(prefix="/extensions", tags=["Extensions"])


_BAD_REQUEST_CODES = {
    "unknown_plugin_kind",
    "extension_module_required",
    "scaffold_plugin_id_required",
    "scaffold_plugin_id_invalid",
}


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


class ScaffoldExtensionRequest(BaseModel):
    kind: str = Field(..., min_length=1, max_length=64)
    plugin_id: str = Field(..., min_length=1, max_length=128)
    display_name: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=1024)
    author: str | None = Field(default=None, max_length=128)
    version: str | None = Field(default=None, max_length=32)
    export_dir: str | None = Field(default=None, max_length=512)
    write: bool = True


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


@router.post("/scaffold")
async def scaffold_extension_route(req: ScaffoldExtensionRequest):
    try:
        return generate_extension_scaffold(
            kind=req.kind,
            plugin_id=req.plugin_id,
            display_name=req.display_name,
            description=req.description,
            author=req.author,
            version=req.version,
            export_dir=req.export_dir,
            write=req.write,
        )
    except ValueError as exc:
        _raise_for(exc)
