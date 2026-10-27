"""Tool catalog: the project's list of callable tools (imported from an MCP server or an OpenAPI
document, entered by hand, or shipped in a domain pack). Single owner of the store.

Stored in ``Project.runtime_config["tool_catalog"]``::

    {"tools": [normalised], "version": 12-hex sha of the canonical tools only, "source": {...},
     "server": {...} | None, "imported_at", "flags": {name: {"allowed", "read_only"}}, "flags_version"}

plus ``runtime_config["tool_catalog_history"]`` (newest 10 versions, newest 5 with full content; the
probe-pack pattern). A normalised tool is ``{name, description, input_schema, annotations, read_only,
destructive}`` (``read_only`` / ``destructive`` come from the annotations; ``None`` = unknown) and
``schema_error`` when its schema cannot be validated (the tool is kept; scoring treats its args as invalid).

Read-only rule (fail closed): a tool is *effectively read-only* only when ``read_only`` is true (server
``readOnlyHint`` or the user's override) AND ``destructive`` is not true. A destructive tool can never be
made read-only. ``dry_run_call`` is the ONLY function that can reach a server's ``tools/call`` and it
refuses everything that is not effectively read-only, allowed, and from an MCP source.

Secrets (env values, header values) live ONLY in ``secret_service`` under provider ``mcp_<slug>``;
the source config keeps the names (``env_names`` / ``header_names``), never the values.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.services import json_schema_bundle as jsb
from app.services import mcp_client, secret_service
from app.services import openapi_tools

_LOG = logging.getLogger(__name__)

CATALOG_KEY = "tool_catalog"
HISTORY_KEY = "tool_catalog_history"
HISTORY_KEEP = 10
HISTORY_FULL = 5
MAX_TOOLS = 1000
MAX_DESCRIPTION = 2000
MAX_OPENAPI_BYTES = openapi_tools.MAX_DOC_BYTES
MAX_SECRET_VALUE = 4096  # same cap as the project secrets API
MAX_CATALOG_BYTES = 4 * 1024 * 1024  # canonical JSON of the normalised tools
MAX_SCHEMA_DEPTH = 64
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_]{0,63}$")
MCP_KINDS = ("mcp_stdio", "mcp_http")
SOURCE_KINDS = ("mcp_stdio", "mcp_http", "openapi", "manual", "domain_pack")
_HINT_KEYS = ("title", "readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")


class ToolCatalogError(ValueError):
    """The catalog (or a request) is not usable; ``.errors`` lists ``{code, path, message}`` entries."""

    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = errors
        super().__init__(errors[0]["message"] if errors else "Invalid tool catalog.")


class MutatingToolRefused(RuntimeError):
    """A dry run was asked for a tool that is not effectively read-only / allowed / from an MCP source."""


class DryRunDisabled(RuntimeError):
    """``MCP_DRY_RUN_ENABLED`` is false."""


def _err(code: str, path: str, message: str) -> dict[str, Any]:
    return {"code": code, "path": path, "message": message}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha12(value: Any) -> str:
    return hashlib.sha256(_canon(value).encode("utf-8")).hexdigest()[:12]


# ── schema handling ────────────────────────────────────────────────────────

_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SCHEMA_SINGLE = ("items", "additionalProperties", "not", "contains", "propertyNames", "if", "then", "else", "unevaluatedProperties", "unevaluatedItems")
_DATA_KEYS = frozenset({"enum", "const", "default", "examples"})


def _json_depth_exceeds(value: Any, limit: int) -> bool:
    """Iterative (no recursion) check that dict / list nesting stays within ``limit`` levels."""
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, dict):
            children: Any = item.values()
        elif isinstance(item, list):
            children = item
        else:
            continue
        if depth > limit:
            return True
        stack.extend((c, depth + 1) for c in children)
    return False


def _modernize(node: Any) -> Any:
    """Draft-07 -> 2020-12 in schema positions: tuple ``items`` -> ``prefixItems`` (+ ``additionalItems``
    -> ``items``), boolean ``exclusiveMinimum`` / ``exclusiveMaximum`` -> numeric."""
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    skip: set[str] = set()
    for excl, bound in (("exclusiveMinimum", "minimum"), ("exclusiveMaximum", "maximum")):
        if isinstance(node.get(excl), bool):
            skip.add(excl)
            if node[excl] and bound in node:
                skip.add(bound)
                out[excl] = node[bound]
    tuple_items = isinstance(node.get("items"), list)
    dep_required: dict[str, Any] = {}
    dep_schemas: dict[str, Any] = {}
    if isinstance(node.get("dependencies"), dict):  # draft-07: array -> dependentRequired, schema -> dependentSchemas
        for dep_key, dep_value in node["dependencies"].items():
            if isinstance(dep_value, list):
                dep_required[dep_key] = copy.deepcopy(dep_value)
            elif isinstance(dep_value, (dict, bool)):
                dep_schemas[dep_key] = _modernize(dep_value)
    for key, value in node.items():
        if key in skip:
            continue
        if key == "dependencies" and isinstance(value, dict):
            continue
        if key == "items" and tuple_items:
            out["prefixItems"] = [_modernize(v) for v in value]
        elif key == "additionalItems":
            if tuple_items:
                out["items"] = _modernize(value) if isinstance(value, dict) else value
            continue  # without tuple items it has no effect in 2020-12
        elif key in _DATA_KEYS:
            out[key] = copy.deepcopy(value)
        elif key in _SCHEMA_MAPS and isinstance(value, dict):
            out[key] = {k: _modernize(v) for k, v in value.items()}
        elif key in _SCHEMA_LISTS and isinstance(value, list):
            out[key] = [_modernize(v) for v in value]
        elif key in _SCHEMA_SINGLE and isinstance(value, dict):
            out[key] = _modernize(value)
        else:
            out[key] = copy.deepcopy(value)
    if dep_required:
        out["dependentRequired"] = {**(out.get("dependentRequired") if isinstance(out.get("dependentRequired"), dict) else {}), **dep_required}
    if dep_schemas:
        out["dependentSchemas"] = {**(out.get("dependentSchemas") if isinstance(out.get("dependentSchemas"), dict) else {}), **dep_schemas}
    return out


def normalize_input_schema(schema: Any) -> dict[str, Any]:
    """An MCP ``inputSchema`` as a 2020-12-compatible object schema (``{"type": "object"}`` when missing).
    Does not validate: see ``_schema_error``."""
    if not isinstance(schema, dict) or not schema:
        return {"type": "object"}
    out = _modernize(schema)
    if out.get("$schema") not in jsb.SUPPORTED_DIALECTS:
        out.pop("$schema", None)  # draft-04 / 06 / 07 (or unknown): 2020-12 is assumed
    return out or {"type": "object"}


def _schema_error(schema: dict[str, Any]) -> str | None:
    try:
        jsb.validate_bundle({"tool.json": schema}, "tool.json")
    except jsb.SchemaBundleError as exc:
        return str((exc.errors or [str(exc)])[0])[:300]
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"[:300]
    return None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def destructive_effective(tool: dict[str, Any]) -> bool:
    """Fail closed on unknown destructiveness: besides an explicit ``destructive: true``, a tool whose
    annotations say ``readOnlyHint: false`` and do not say ``destructiveHint: false`` is destructive (the MCP
    default). A tool with no annotations at all stays 'unknown' and can still be marked read-only by the user."""
    if tool.get("destructive") is True:
        return True
    ann = tool.get("annotations") if isinstance(tool.get("annotations"), dict) else {}
    return ann.get("readOnlyHint") is False and ann.get("destructiveHint") is not False


def normalize_tool(raw: Any) -> dict[str, Any]:
    """One raw tool (MCP / OpenAPI / pack / already normalised) -> the normalised form.
    Raises ``ToolCatalogError`` for a missing or illegal name."""
    if not isinstance(raw, dict):
        raise ToolCatalogError([_err("bad_tool", "", "A tool must be an object.")])
    name = raw.get("name", raw.get("id"))
    if not isinstance(name, str) or not TOOL_NAME_RE.match(name):
        raise ToolCatalogError([_err("bad_name", str(name)[:64], f"Tool name {str(name)[:64]!r} must match [A-Za-z0-9_.-]{{1,64}}.")])
    ann_raw = raw.get("annotations") if isinstance(raw.get("annotations"), dict) else {}
    annotations = {k: ann_raw[k] for k in _HINT_KEYS if k in ann_raw and isinstance(ann_raw[k], (str, bool))}
    if isinstance(annotations.get("title"), str):
        annotations["title"] = annotations["title"][:200]
    read_only = _bool_or_none(annotations.get("readOnlyHint"))
    destructive = _bool_or_none(annotations.get("destructiveHint"))
    if read_only is None:
        read_only = _bool_or_none(raw.get("read_only"))
    if destructive is None:
        destructive = _bool_or_none(raw.get("destructive"))
    schema_raw = raw.get("inputSchema", raw.get("input_schema"))
    tool = {
        "name": name,
        "description": str(raw.get("description") or "").strip()[:MAX_DESCRIPTION],
        "input_schema": {},
        "annotations": annotations,
        "read_only": read_only,
        "destructive": destructive,
    }
    if schema_raw is not None and not isinstance(schema_raw, dict):
        tool["input_schema"] = {"type": "object"}
        tool["schema_error"] = "inputSchema must be an object."
        return tool
    if _json_depth_exceeds(schema_raw, MAX_SCHEMA_DEPTH):
        tool["input_schema"] = {"type": "object"}
        tool["schema_error"] = f"inputSchema is nested deeper than {MAX_SCHEMA_DEPTH} levels."
        return tool
    try:
        schema = normalize_input_schema(schema_raw)
        tool["input_schema"] = schema
        err = _schema_error(schema)
    except RecursionError:
        tool["input_schema"] = {"type": "object"}
        tool["schema_error"] = "inputSchema is nested too deeply."
        return tool
    if err:
        tool["schema_error"] = err
    return tool


def normalize_tools(raw_tools: Any, *, strict: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalise a list. ``strict`` raises on the first problem class (collected); otherwise bad entries
    are skipped and reported. Returns ``(tools sorted by name, problems)``."""
    problems: list[dict[str, Any]] = []
    if not isinstance(raw_tools, list):
        raise ToolCatalogError([_err("bad_tools", "tools", "tools must be a list.")])
    if len(raw_tools) > MAX_TOOLS:
        raise ToolCatalogError([_err("too_many_tools", "tools", f"At most {MAX_TOOLS} tools per project.")])
    tools: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(raw_tools):
        try:
            tool = normalize_tool(raw)
        except ToolCatalogError as exc:
            for e in exc.errors:
                problems.append({**e, "path": f"tools[{i}]" + (f".{e['path']}" if e.get("path") else "")})
            continue
        if tool["name"] in tools:
            problems.append(_err("duplicate_name", f"tools[{i}]", f"Tool name {tool['name']!r} appears more than once."))
            continue
        tools[tool["name"]] = tool
    if strict and problems:
        raise ToolCatalogError(problems)
    ordered = [tools[k] for k in sorted(tools)]
    try:
        size = len(_canon(ordered).encode("utf-8"))
    except RecursionError:
        size = MAX_CATALOG_BYTES + 1
    if size > MAX_CATALOG_BYTES:
        raise ToolCatalogError([_err("catalog_too_large", "tools", f"The tool catalog is larger than {MAX_CATALOG_BYTES // (1024 * 1024)} MB; import fewer or smaller tools.")])
    return ordered, problems


def tools_from_pack(content: Any) -> list[Any]:
    """A pack's ``tool_catalog`` content (a list, or a mapping with ``tools``) -> its raw tool items."""
    if isinstance(content, dict):
        content = content.get("tools")
    return list(content) if isinstance(content, list) else []


def check_pack_tools(content: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tolerant pack validation: ``(errors, warnings)`` with ``{code, message}`` entries. A duplicate name is
    an error; a nameless tool or a schema that cannot be validated is a warning."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    items = tools_from_pack(content)
    names: list[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            warnings.append(_err("invalid_tool", f"tools[{i}]", "A tool entry must be a mapping."))
            continue
        name = item.get("name", item.get("id"))
        if not isinstance(name, str) or not name:
            warnings.append(_err("tool_without_name", f"tools[{i}]", "A tool entry has no name (or id)."))
            continue
        names.append(name)
        try:
            tool = normalize_tool(item)
        except ToolCatalogError as exc:
            warnings.append(_err("invalid_tool", name, exc.errors[0]["message"]))
            continue
        if tool.get("schema_error"):
            warnings.append(_err("tool_schema_error", name, f"Tool {name!r}: {tool['schema_error']}"))
    for dup in sorted({n for n in names if names.count(n) > 1}):
        errors.append(_err("duplicate_id", dup, f"Duplicate tool id {dup!r}."))
    return errors, warnings


# ── store ──────────────────────────────────────────────────────────────────


def _rc(project_or_config: Any) -> dict[str, Any]:
    cfg = getattr(project_or_config, "runtime_config", project_or_config)
    return cfg if isinstance(cfg, dict) else {}


def _state(runtime_config: Any) -> dict[str, Any] | None:
    state = _rc(runtime_config).get(CATALOG_KEY)
    return state if isinstance(state, dict) and isinstance(state.get("tools"), list) and state["tools"] else None


def raw_state(runtime_config: Any) -> dict[str, Any] | None:
    """The stored catalog state (tools, version, source, flags...) or None."""
    return _state(runtime_config)


def _flags_of(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    flags = state.get("flags") if isinstance(state.get("flags"), dict) else {}
    return {k: v for k, v in flags.items() if isinstance(v, dict)}


def _tool_view(tool: dict[str, Any], flags: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flag = flags.get(tool["name"]) or {}
    allowed = flag.get("allowed") is not False
    override = _bool_or_none(flag.get("read_only"))
    read_only = override if override is not None else tool.get("read_only")
    destructive = destructive_effective(tool)
    read_only_effective = bool(read_only) and not destructive
    return {
        **tool,
        "destructive": True if destructive else tool.get("destructive"),
        "allowed": allowed,
        "read_only_effective": read_only_effective,
        "mutating": not read_only_effective,
    }


def effective_tools(runtime_config: Any) -> list[dict[str, Any]]:
    """Every catalog tool, normalised, plus ``allowed`` / ``read_only_effective`` / ``mutating``."""
    state = _state(runtime_config)
    if state is None:
        return []
    flags = _flags_of(state)
    return [_tool_view(t, flags) for t in state["tools"] if isinstance(t, dict) and t.get("name")]


def allowed_tools(runtime_config: Any) -> list[dict[str, Any]]:
    return [t for t in effective_tools(runtime_config) if t["allowed"]]


def catalog_fingerprint(runtime_config: Any) -> str:
    state = _state(runtime_config)
    if state is None:
        return "none"
    return f"{state.get('version')}+{state.get('flags_version') or _sha12({})}"


_BUNDLES: "OrderedDict[tuple[str, str], Any]" = OrderedDict()


def tool_bundle(runtime_config: Any, name: str) -> "jsb.SchemaBundle | None":
    """The validated schema bundle of one tool's arguments (cached by catalog version + name); ``None``
    for an unknown tool or one whose schema could not be validated."""
    state = _state(runtime_config)
    if state is None:
        return None
    key = (str(state.get("version")), str(name))
    if key in _BUNDLES:
        _BUNDLES.move_to_end(key)
        return _BUNDLES[key]
    tool = next((t for t in state["tools"] if isinstance(t, dict) and t.get("name") == name), None)
    bundle = None
    if tool is not None and not tool.get("schema_error"):
        try:
            bundle = jsb.validate_bundle({"tool.json": tool["input_schema"]}, "tool.json")
        except Exception:  # noqa: BLE001
            bundle = None
    _BUNDLES[key] = bundle
    while len(_BUNDLES) > 512:
        _BUNDLES.popitem(last=False)
    return bundle


def _new_state(
    tools: list[dict[str, Any]], source: dict[str, Any], server: dict[str, Any] | None,
    flags: dict[str, dict[str, Any]], imported_at: str,
) -> dict[str, Any]:
    names = {t["name"] for t in tools}
    kept = {k: dict(v) for k, v in sorted(flags.items()) if k in names and (v.get("allowed") is False or v.get("read_only") is not None)}
    return {
        "tools": tools,
        "version": _sha12(tools),
        "source": source,
        "server": server,
        "imported_at": imported_at,
        "flags": kept,
        "flags_version": _sha12(kept),
    }


def _history(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [h for h in (cfg.get(HISTORY_KEY) or []) if isinstance(h, dict)]


def _apply(
    project: Any, tools: list[dict[str, Any]], source: dict[str, Any], server: dict[str, Any] | None,
    *, flags: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write a new catalog. Same tools as the current one = no new version / history entry (flags kept)."""
    cfg = dict(_rc(project))
    current = _state(cfg)
    carried = flags if flags is not None else (_flags_of(current) if current else {})
    now = _now()
    state = _new_state(tools, source, server, carried, now)
    if current is not None and current.get("version") == state["version"]:
        state["imported_at"] = current.get("imported_at") or now
        cfg[CATALOG_KEY] = state
        history = _history(cfg)
        for h in history:
            if h.get("version") == state["version"] and isinstance(h.get("tools"), list):
                h["flags"] = state["flags"]
        cfg[HISTORY_KEY] = history
        project.runtime_config = cfg
        return state
    history = [h for h in _history(cfg) if h.get("version") != state["version"]]
    history.append({
        "version": state["version"],
        "imported_at": now,
        "source": source,
        "server": server,
        "source_kind": source.get("kind"),
        "tool_count": len(tools),
        "flags": state["flags"],
        "tools": copy.deepcopy(tools),
    })
    history = history[-HISTORY_KEEP:]
    for h in history[:-HISTORY_FULL]:
        h.pop("tools", None)
    cfg[CATALOG_KEY] = state
    cfg[HISTORY_KEY] = history
    project.runtime_config = cfg
    return state


def _clean_source(source: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(source) if isinstance(source, dict) and source else {"kind": "manual"}
    if src.get("kind") not in SOURCE_KINDS:
        src["kind"] = "manual"
    for secretish in ("env", "headers", "values", "secrets"):
        src.pop(secretish, None)  # a source never carries values
    return src


def set_catalog_tools(project: Any, tools: Any, source: dict[str, Any] | None = None, *, server: dict[str, Any] | None = None) -> dict[str, Any]:
    """Replace the catalog with ``tools`` (manual / domain pack). Raises ``ToolCatalogError``."""
    normalized, _ = normalize_tools(tools, strict=True)
    if not normalized:
        raise ToolCatalogError([_err("no_tools", "tools", "A tool catalog needs at least one tool.")])
    _apply(project, normalized, _clean_source(source), server)
    return catalog_status(project)


async def set_catalog_tools_async(project: Any, tools: Any, source: dict[str, Any] | None = None) -> dict[str, Any]:
    """``set_catalog_tools`` with normalisation off the event loop."""
    normalized, _ = await asyncio.to_thread(normalize_tools, tools, strict=True)
    if not normalized:
        raise ToolCatalogError([_err("no_tools", "tools", "A tool catalog needs at least one tool.")])
    _apply(project, normalized, _clean_source(source), None)
    return catalog_status(project)


def set_tool_flags(project: Any, flags: Any) -> dict[str, Any]:
    """Merge ``{name: {allowed?, read_only?}}``. ``read_only: null`` clears the override; a destructive tool
    can never be made read-only (422)."""
    state = _state(project)
    if state is None:
        raise ToolCatalogError([_err("no_catalog", "", "This project has no tool catalog.")])
    if not isinstance(flags, dict):
        raise ToolCatalogError([_err("bad_flags", "flags", "flags must be an object.")])
    by_name = {t["name"]: t for t in state["tools"] if isinstance(t, dict)}
    merged = _flags_of(state)
    errors: list[dict[str, Any]] = []
    for name, patch in flags.items():
        path = f"flags.{name}"
        if name not in by_name:
            errors.append(_err("unknown_tool", path, f"No tool named {name!r} in the catalog."))
            continue
        if not isinstance(patch, dict):
            errors.append(_err("bad_flag", path, "A flag entry must be an object."))
            continue
        entry = dict(merged.get(name) or {})
        if "allowed" in patch:
            if not isinstance(patch["allowed"], bool):
                errors.append(_err("bad_flag", path + ".allowed", "allowed must be true or false."))
                continue
            entry["allowed"] = patch["allowed"]
        if "read_only" in patch:
            value = patch["read_only"]
            if value is not None and not isinstance(value, bool):
                errors.append(_err("bad_flag", path + ".read_only", "read_only must be true, false or null."))
                continue
            if value is True and destructive_effective(by_name[name]):
                errors.append(_err("destructive_read_only", path + ".read_only", f"{name!r} is marked destructive by its server; it can never be made read-only."))
                continue
            entry["read_only"] = value
        merged[name] = entry
    if errors:
        raise ToolCatalogError(errors)
    cfg = dict(_rc(project))
    new_state = _new_state(state["tools"], state.get("source") or {"kind": "manual"}, state.get("server"), merged, state.get("imported_at") or _now())
    cfg[CATALOG_KEY] = new_state
    history = _history(cfg)
    for h in history:
        if h.get("version") == new_state["version"]:
            h["flags"] = new_state["flags"]
    cfg[HISTORY_KEY] = history
    project.runtime_config = cfg
    return catalog_status(project)


def restore_catalog_version(project: Any, version: str) -> dict[str, Any]:
    """Restore a remembered full snapshot. ``LookupError`` when the version is unknown or trimmed."""
    for h in reversed(_history(_rc(project))):
        if h.get("version") == version:
            if not isinstance(h.get("tools"), list):
                raise LookupError(f"Version {version} is no longer restorable (only the newest {HISTORY_FULL} keep their content).")
            source = {**(h.get("source") or {"kind": "manual"}), "restored_from": version}
            _apply(project, copy.deepcopy(h["tools"]), _clean_source(source), h.get("server"), flags=copy.deepcopy(h.get("flags") or {}))
            return catalog_status(project)
    raise LookupError(f"Version {version} was not found in this project's history.")


def clear_catalog(project: Any) -> dict[str, Any]:
    """Drop the catalog (history is kept so it can be restored)."""
    cfg = dict(_rc(project))
    cfg.pop(CATALOG_KEY, None)
    project.runtime_config = cfg
    return catalog_status(project)


def catalog_status(project: Any) -> dict[str, Any]:
    cfg = _rc(project)
    state = _state(cfg)
    catalog = None
    if state is not None:
        flags = _flags_of(state)
        views = []
        for t in state["tools"]:
            if isinstance(t, dict) and t.get("name"):
                v = _tool_view(t, flags)
                v["signature"] = compact_signature(v)
                views.append(v)
        catalog = {
            "tools": views,
            "version": state.get("version"),
            "flags_version": state.get("flags_version"),
            "source": state.get("source"),
            "server": state.get("server"),
            "imported_at": state.get("imported_at"),
        }
    current = state.get("version") if state else None
    history = [
        {
            "version": h.get("version"),
            "imported_at": h.get("imported_at"),
            "source_kind": h.get("source_kind") or (h.get("source") or {}).get("kind"),
            "tool_count": h.get("tool_count"),
            "current": h.get("version") == current,
            "restorable": isinstance(h.get("tools"), list),
        }
        for h in reversed(_history(cfg))
    ]
    return {"catalog": catalog, "history": history}


# ── compact signature / prompt block ───────────────────────────────────────

_ENUM_MAX = 8
_DESC_MAX = 160


def _resolve_local(ref: str, root: dict[str, Any]) -> Any:
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _enum_text(values: list[Any]) -> str:
    shown = [v if isinstance(v, str) else json.dumps(v) for v in values[:_ENUM_MAX]]
    return "|".join(shown) + ("|…" if len(values) > _ENUM_MAX else "")


def _type_text(schema: Any, root: dict[str, Any], depth: int = 0) -> str:
    if not isinstance(schema, dict) or depth > 6:
        return "any"
    if isinstance(schema.get("$ref"), str):
        target = _resolve_local(schema["$ref"], root)
        return _type_text(target, root, depth + 1) if isinstance(target, dict) else "any"
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return _enum_text(schema["enum"])
    if "const" in schema:
        return json.dumps(schema["const"])
    t = schema.get("type")
    if isinstance(t, list):
        return "|".join(_one_type({**schema, "type": x}, root, depth) for x in t if isinstance(x, str)) or "any"
    if isinstance(t, str):
        return _one_type(schema, root, depth)
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list) and schema[key]:
            parts: list[str] = []
            for sub in schema[key]:
                text = _type_text(sub, root, depth + 1)
                if text not in parts:
                    parts.append(text)
            return "|".join(parts)
    if isinstance(schema.get("allOf"), list) and len(schema["allOf"]) == 1:
        return _type_text(schema["allOf"][0], root, depth + 1)
    if "properties" in schema:
        return "object"
    return "any"


def _one_type(schema: dict[str, Any], root: dict[str, Any], depth: int) -> str:
    t = schema.get("type")
    if t == "array":
        items = schema.get("items")
        return f"array<{_type_text(items, root, depth + 1)}>" if isinstance(items, dict) else "array"
    if t == "object":
        return "object"
    if t == "string" and isinstance(schema.get("enum"), list):
        return _enum_text(schema["enum"])
    return str(t)


def _tag(tool: dict[str, Any]) -> str:
    if "read_only_effective" in tool:
        read_only = bool(tool["read_only_effective"])
    else:
        read_only = bool(tool.get("read_only")) and tool.get("destructive") is not True
    if tool.get("destructive") is True:
        return "[destructive]"
    return "[read-only]" if read_only else "[mutating]"


def compact_signature(tool: dict[str, Any]) -> str:
    """``get_service(name: string) — Fetch one service. [read-only]``"""
    schema = tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {}
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = set(schema.get("required") or []) if isinstance(schema.get("required"), list) else set()
    params = [
        f"{pname}{'' if pname in required else '?'}: {_type_text(pschema, schema)}"
        for pname, pschema in props.items()
    ]
    sig = f"{tool.get('name', '')}({', '.join(params)})"
    desc = " ".join(str(tool.get("description") or "").split())
    if len(desc) > _DESC_MAX:
        desc = desc[: _DESC_MAX - 1].rstrip() + "…"
    return sig + (f" — {desc}" if desc else "") + f" {_tag(tool)}"


def render_tool_block(runtime_config: Any, *, max_chars: int = 6000) -> str:
    """``Available tools (call only these):`` + one signature line per allowed tool (sorted by name).
    Empty string when there is no catalog; a truncation note when the block does not fit."""
    if _state(runtime_config) is None:
        return ""
    header = "Available tools (call only these):"
    tools = sorted(allowed_tools(runtime_config), key=lambda t: t["name"])
    if not tools:
        return header + "\n- (no tools are allowed)"
    lines = [f"- {compact_signature(t)}" for t in tools]
    out = header
    shown = 0
    for i, line in enumerate(lines):
        rest = len(lines) - i - 1
        note_len = len(f"\n- … {len(lines) - i} more tools not shown (truncated)")
        room_for_more = len(out) + 1 + len(line) + (note_len if rest else 0) <= max_chars
        if not room_for_more:
            break
        out += "\n" + line
        shown += 1
    if shown < len(lines):
        out += f"\n- … {len(lines) - shown} more tools not shown (truncated)"
    return out


# ── secrets + imports ──────────────────────────────────────────────────────


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label or "").lower()).strip("_")[:40] or "server"


def _secret_key(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def secret_provider_for(label: str, kind: str = "mcp_http") -> str:
    """``mcp_http_<slug>`` / ``mcp_stdio_<slug>``: the kind is part of the provider so an http source can
    never read a stdio source's secrets (or the reverse)."""
    return f"{kind if kind in MCP_KINDS else 'mcp'}_{_slug(label)}"


def endpoint_binding(kind: str, *, url: Any = None, command: Any = None) -> Any:
    """What a stored secret is bound to: http = scheme + host + port + path of the url (query ignored);
    stdio = the exact command list. A secret is only ever re-sent to the endpoint it was entered for."""
    if kind == "mcp_stdio":
        return [str(c) for c in (command or [])]
    from urllib.parse import urlsplit

    parts = urlsplit(str(url or "").strip())
    scheme = (parts.scheme or "").lower()
    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError:
        port = 0
    return f"{scheme}://{(parts.hostname or '').lower()}:{port}{parts.path or '/'}"


def _stored_binding(src: dict[str, Any]) -> Any:
    if src.get("endpoint_binding") is not None:
        return src["endpoint_binding"]
    return endpoint_binding(str(src.get("kind")), url=src.get("url"), command=src.get("command"))


def _default_label(req: dict[str, Any]) -> str:
    if req.get("kind") == "mcp_stdio":
        cmd = req.get("command") or []
        return str(cmd[0]).rsplit("/", 1)[-1] if cmd else "server"
    from urllib.parse import urlsplit

    return urlsplit(str(req.get("url") or "")).hostname or "server"


def validate_mcp_request(source_req: Any) -> dict[str, Any]:
    if not isinstance(source_req, dict):
        raise ToolCatalogError([_err("bad_source", "", "The import request must be an object.")])
    kind = source_req.get("kind")
    errors: list[dict[str, Any]] = []
    out: dict[str, Any] = {"kind": kind, "label": str(source_req.get("label") or "").strip()[:80]}
    if kind == "mcp_stdio":
        command = source_req.get("command")
        if not (isinstance(command, list) and command and all(isinstance(c, str) for c in command) and command[0].strip()):
            errors.append(_err("bad_command", "command", "command must be a non-empty list of strings (program, then arguments)."))
        elif len(command) > 64 or any(len(c) > 4096 for c in command):
            errors.append(_err("bad_command", "command", "command is too long."))
        else:
            out["command"] = list(command)
        values = source_req.get("env") or {}
        name_re, field = ENV_NAME_RE, "env"
    elif kind == "mcp_http":
        url = source_req.get("url")
        if not isinstance(url, str) or not re.match(r"^https?://", url.strip(), re.IGNORECASE):
            errors.append(_err("bad_url", "url", "url must be an http(s) URL."))
        else:
            out["url"] = url.strip()
        values = source_req.get("headers") or {}
        name_re, field = HEADER_NAME_RE, "headers"
    else:
        raise ToolCatalogError([_err("bad_source", "kind", "kind must be 'mcp_stdio' or 'mcp_http'.")])
    if not isinstance(values, dict) or len(values) > 32:
        errors.append(_err("bad_values", field, f"{field} must be an object with at most 32 entries."))
        values = {}
    clean: dict[str, str | None] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name_re.match(name):
            errors.append(_err("bad_name", f"{field}.{str(name)[:64]}", f"Illegal {field} name {str(name)[:64]!r}."))
        elif field == "env" and mcp_client.env_name_denied(name):
            errors.append(_err("env_name_denied", f"{field}.{name}", f"Environment variable {name!r} is not allowed: it could change how the server binary loads or runs code."))
        elif value is not None and not isinstance(value, str):
            errors.append(_err("bad_value", f"{field}.{name}", "Values must be strings or null (keep the stored secret)."))
        elif isinstance(value, str) and len(value.strip()) > MAX_SECRET_VALUE:
            errors.append(_err("secret_too_long", f"{field}.{name}", f"The value of {name!r} is longer than {MAX_SECRET_VALUE} characters."))
        else:
            clean[name] = value if value and value.strip() else None  # blank = keep the stored secret
    if errors:
        raise ToolCatalogError(errors)
    out["values"] = clean
    out["label"] = out["label"] or _default_label(out)
    return out


async def _resolve_secret_values(
    db: Any, project: Any, req: dict[str, Any], provider: str, binding: Any,
) -> tuple[dict[str, str], dict[str, str]]:
    """``(all values by original name, new values to store)``. A null value reuses the stored secret ONLY
    when the stored source has the same kind, provider and endpoint binding; there is no fallback to another
    label / kind / provider."""
    current_src = (_state(project) or {}).get("source") or {}
    reusable = (
        current_src.get("kind") == req["kind"]
        and current_src.get("secret_provider") == provider
        and _stored_binding(current_src) == binding
    )
    resolved: dict[str, str] = {}
    fresh: dict[str, str] = {}
    missing: list[tuple[str, str]] = []
    for name, value in req["values"].items():
        if value is not None:
            resolved[name] = value
            fresh[name] = value
            continue
        found = None
        if reusable:
            found = await secret_service.get_project_secret_value(db, project.id, provider, _secret_key(name))
        if found is None:
            missing.append((name, "No stored value" if reusable else "re-enter the value: the endpoint changed (or this is a different server)"))
        else:
            resolved[name] = found
    if missing:
        raise ToolCatalogError([
            _err("missing_secret", f"values.{n}", f"No stored value for {n!r}; enter a value for it." if why == "No stored value" else f"Cannot reuse the stored value for {n!r}: {why}.")
            for n, why in missing
        ])
    return resolved, fresh


async def import_from_mcp(db: Any, project: Any, source_req: Any) -> dict[str, Any]:
    """Connect to an MCP server, list its tools and store them. Secret values are stored (and only
    stored) via ``secret_service``; the saved source keeps names. Raises ``ToolCatalogError``,
    ``mcp_client.McpStdioDisabled`` (stdio not allow-listed) or ``mcp_client.McpError`` (server problem)."""
    req = validate_mcp_request(source_req)
    if req["kind"] == "mcp_stdio":
        mcp_client.check_stdio_allowed(req["command"][0])  # fail fast, before any secret is touched
    provider = secret_provider_for(req["label"], req["kind"])
    binding = endpoint_binding(req["kind"], url=req.get("url"), command=req.get("command"))
    values, fresh = await _resolve_secret_values(db, project, req, provider, binding)
    if req["kind"] == "mcp_stdio":
        listing = await mcp_client.list_tools_stdio(req["command"], values, timeout_s=30)
        source: dict[str, Any] = {
            "kind": "mcp_stdio", "label": req["label"], "command": req["command"],
            "env_names": list(req["values"]), "secret_provider": provider, "endpoint_binding": binding,
        }
    else:
        listing = await mcp_client.list_tools_http(req["url"], values, timeout_s=float(settings.MCP_HTTP_TIMEOUT_S))
        source = {
            "kind": "mcp_http", "label": req["label"], "url": req["url"],
            "header_names": list(req["values"]), "secret_provider": provider, "endpoint_binding": binding,
        }
    secrets_all = list(values.values())
    tools, problems = await asyncio.to_thread(normalize_tools, listing.get("tools") or [], strict=False)
    warnings = [mcp_client.redact(p["message"], secrets_all) for p in problems]
    if listing.get("truncated"):
        warnings.append(f"The server lists more than {mcp_client.MAX_TOOLS} tools; the rest were ignored.")
    if not tools:
        raise ToolCatalogError([_err("no_tools", "tools", "The server listed no usable tools.")])
    for name, value in fresh.items():
        await secret_service.upsert_project_secret(db, project.id, provider, _secret_key(name), value)
    server = listing.get("server") if isinstance(listing.get("server"), dict) else None
    _apply(project, tools, source, server)
    _LOG.info("tool catalog: imported %d tools from %s source %r (secret names: %s)", len(tools), req["kind"], req["label"], sorted(req["values"]))
    status = catalog_status(project)
    if warnings:
        status["warnings"] = warnings
    return status


def parse_openapi_text(text: Any) -> Any:
    import yaml

    from app.services.label_taxonomy_service import _NoAliasLoader

    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    if not isinstance(text, str) or not text.strip():
        raise ToolCatalogError([_err("empty", "content", "The OpenAPI document is empty.")])
    if len(text.encode("utf-8")) > MAX_OPENAPI_BYTES:
        raise ToolCatalogError([_err("too_large", "content", f"The OpenAPI document is larger than {MAX_OPENAPI_BYTES // (1024 * 1024)} MB.")])
    try:
        return yaml.load(text, Loader=_NoAliasLoader)  # noqa: S506 - alias-refusing SafeLoader; JSON is YAML
    except yaml.YAMLError as exc:
        first = str(exc).splitlines()[0] if str(exc) else "parse error"
        raise ToolCatalogError([_err("parse_error", "content", f"Not valid YAML/JSON: {first}")]) from exc
    except RecursionError as exc:
        raise ToolCatalogError([_err("parse_error", "content", "The OpenAPI document is nested too deeply.")]) from exc


def prepare_openapi(text: Any, filename: str | None = None) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """The CPU part of an OpenAPI import (parse, convert, normalise): ``(tools, warnings, source)``. Pure; safe
    to run in a worker thread."""
    doc = parse_openapi_text(text)
    try:
        raw_tools, warnings = openapi_tools.openapi_to_tools(doc)
    except ValueError as exc:
        raise ToolCatalogError([_err("bad_openapi", "content", str(exc))]) from exc
    except RecursionError as exc:
        raise ToolCatalogError([_err("bad_openapi", "content", "The OpenAPI document is nested too deeply.")]) from exc
    tools, problems = normalize_tools(raw_tools, strict=False)
    warnings = [*warnings, *[p["message"] for p in problems]]
    if not tools:
        raise ToolCatalogError([_err("no_tools", "content", "The document produced no usable tools." + (f" {warnings[0]}" if warnings else ""))])
    info = doc.get("info") if isinstance(doc.get("info"), dict) else {}
    title = str(info.get("title") or "")[:200]
    fname = str(filename or "")[:200]
    source = {
        "kind": "openapi", "label": title or fname or "openapi",
        "filename": fname, "title": title, "version": str(info.get("version") or "")[:100],
    }
    return tools, warnings, source


def _finish_openapi(project: Any, tools: list[dict[str, Any]], warnings: list[str], source: dict[str, Any]) -> dict[str, Any]:
    _apply(project, tools, source, None)
    status = catalog_status(project)
    if warnings:
        status["warnings"] = warnings
    return status


def import_from_openapi(project: Any, text: Any, filename: str | None = None) -> dict[str, Any]:
    return _finish_openapi(project, *prepare_openapi(text, filename))


async def import_from_openapi_async(project: Any, text: Any, filename: str | None = None) -> dict[str, Any]:
    """``import_from_openapi`` with the parse / convert / normalise work off the event loop."""
    prepared = await asyncio.to_thread(prepare_openapi, text, filename)
    return _finish_openapi(project, *prepared)


# ── dry run ────────────────────────────────────────────────────────────────


def _result_snippet(result: dict[str, Any], secrets: list[str]) -> str:
    parts: list[str] = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    text = "\n".join(parts) if parts else (json.dumps(result.get("structured"), default=str) if result.get("structured") is not None else "")
    return mcp_client.redact(" ".join(text.split()), secrets)[:300]


async def dry_run_call(db: Any, project: Any, name: str, args: Any, *, timeout_s: float = 15) -> dict[str, Any]:
    """Execute ONE read-only tool against its MCP server and return ``{ok, is_error, error, snippet}``.

    The only code path to ``tools/call``. Raises ``MutatingToolRefused`` unless the tool is effectively
    read-only AND allowed AND the source is MCP (decided from the stored tool, never from the request),
    then ``DryRunDisabled`` unless ``MCP_DRY_RUN_ENABLED``. Invalid args or a server failure come back as
    ``ok: false`` without raising."""
    state = _state(project)
    if state is None:
        raise MutatingToolRefused("There is no tool catalog.")
    tool = next((t for t in state["tools"] if isinstance(t, dict) and t.get("name") == name), None)
    if tool is None:
        raise MutatingToolRefused(f"{name!r} is not in the catalog.")
    view = _tool_view(tool, _flags_of(state))
    source = state.get("source") if isinstance(state.get("source"), dict) else {}
    if not view["allowed"]:
        raise MutatingToolRefused(f"{name!r} is not allowed in this catalog.")
    if view["mutating"] or not view["read_only_effective"] or destructive_effective(tool):
        raise MutatingToolRefused(f"{name!r} is not effectively read-only; it is never executed.")
    if source.get("kind") not in MCP_KINDS:
        raise MutatingToolRefused(f"{name!r} comes from a {source.get('kind')!r} source; only MCP servers can be called.")
    if not settings.MCP_DRY_RUN_ENABLED:
        raise DryRunDisabled("Dry-run calls are disabled: set MCP_DRY_RUN_ENABLED=true.")
    if not isinstance(args, dict):
        return {"ok": False, "is_error": True, "error": "arguments must be an object", "snippet": ""}
    arguments = args
    bundle = tool_bundle(project, name)
    if bundle is None:
        return {"ok": False, "is_error": True, "error": "the tool's schema could not be validated", "snippet": ""}
    problems = bundle.validate(arguments)
    if problems:
        first = problems[0]
        return {"ok": False, "is_error": True, "error": f"invalid arguments: {first['pointer'] or '/'}: {first['message']}"[:300], "snippet": ""}

    provider = str(source.get("secret_provider") or "")
    names = source.get("env_names" if source["kind"] == "mcp_stdio" else "header_names") or []
    values: dict[str, str] = {}
    for n in names:
        value = await secret_service.get_project_secret_value(db, project.id, provider, _secret_key(str(n))) if provider else None
        if value is not None:
            values[str(n)] = value
    secrets_all = list(values.values())
    try:
        if source["kind"] == "mcp_stdio":
            from app.services.mcp_client import call_tool_stdio

            result = await call_tool_stdio(list(source.get("command") or []), values, name, arguments, timeout_s=timeout_s)
        else:
            from app.services.mcp_client import call_tool_http

            result = await call_tool_http(str(source.get("url") or ""), values, name, arguments, timeout_s=timeout_s)
    except mcp_client.McpError as exc:
        return {"ok": False, "is_error": True, "error": mcp_client.redact(str(exc), secrets_all)[:300], "snippet": ""}
    snippet = _result_snippet(result, secrets_all)
    is_error = bool(result.get("is_error"))
    return {"ok": not is_error, "is_error": is_error, "error": (snippet or "tool reported an error") if is_error else None, "snippet": snippet}
