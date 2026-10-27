"""Convert an OpenAPI 3.0 / 3.1 document into tool-catalog tools (no OpenAPI library).

Each operation becomes one tool shaped like an MCP tool (``name``, ``description``, ``inputSchema``,
``annotations``). The caller parses JSON / YAML text (YAML with the no-alias safe loader) and passes
the resulting dict. Nothing is fetched: only local ``#/components/...`` refs are resolved.

* name: ``operationId`` or ``<method>_<path slug>``, sanitised to ``^[A-Za-z0-9_.-]{1,64}$``, deduped ``_2``
* inputSchema: object with properties from path / query / header parameters (path ones required) plus
  ``body`` from the JSON request body; referenced ``#/components/schemas/X`` become a ``$defs`` copy
  (only the transitively referenced ones); 3.0 ``nullable`` becomes a type union with ``"null"``;
  ``example`` is dropped
* annotations: GET / HEAD -> ``readOnlyHint: true``; DELETE -> ``destructiveHint: true``; others ->
  ``readOnlyHint: false``
* limits: at most 1000 operations, document at most 5 MB
* a tool whose schema fails ``json_schema_bundle.validate_bundle`` is skipped with a warning
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from app.services import json_schema_bundle as jsb

MAX_OPERATIONS = 1000
MAX_DOC_BYTES = 5 * 1024 * 1024
MAX_DESCRIPTION = 1000
_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_IGNORED_HEADERS = {"accept", "content-type", "authorization"}
_DATA_KEYS = frozenset({"enum", "const", "default", "examples"})
_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SCHEMA_SINGLE = ("items", "additionalProperties", "not", "contains", "propertyNames", "if", "then", "else", "unevaluatedProperties", "unevaluatedItems")
_DROP = ("example", "xml", "externalDocs", "discriminator")
_REF_PREFIX = "#/components/schemas/"


def _pointer_escape(name: str) -> str:
    return name.replace("~", "~0").replace("/", "~1")


def _pointer_unescape(name: str) -> str:
    return name.replace("~1", "/").replace("~0", "~")


def sanitize_name(raw: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(raw or "")).strip("_") or "operation"
    return name[:64]


def _slug(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_") or "root"


class _Converter:
    def __init__(self, doc: dict[str, Any], warnings: list[str]) -> None:
        self.doc = doc
        self.warnings = warnings
        self.schemas = (doc.get("components") or {}).get("schemas") if isinstance(doc.get("components"), dict) else None
        self.schemas = self.schemas if isinstance(self.schemas, dict) else {}
        self.wanted: set[str] = set()

    # ── local refs ──

    def resolve(self, node: Any, kind: str) -> Any:
        """Follow a ``#/components/<kind>/X`` $ref (a few hops); anything else is returned as-is."""
        for _ in range(5):
            if not (isinstance(node, dict) and isinstance(node.get("$ref"), str)):
                return node
            ref = node["$ref"]
            prefix = f"#/components/{kind}/"
            if not ref.startswith(prefix):
                return node
            comp = (self.doc.get("components") or {}).get(kind)
            target = comp.get(_pointer_unescape(ref[len(prefix):])) if isinstance(comp, dict) else None
            if target is None:
                self.warnings.append(f"unresolved reference {ref}")
                return {}
            node = target
        return node

    # ── schemas ──

    def convert(self, schema: Any) -> Any:
        if isinstance(schema, list):
            return [self.convert(s) for s in schema]
        if not isinstance(schema, dict):
            return schema
        out: dict[str, Any] = {}
        nullable = schema.get("nullable") is True
        skip: set[str] = set(_DROP) | {"nullable"}
        for excl, bound in (("exclusiveMinimum", "minimum"), ("exclusiveMaximum", "maximum")):
            if isinstance(schema.get(excl), bool):  # 3.0 boolean form -> 2020-12 numeric form
                skip.add(excl)
                if schema[excl] and bound in schema:
                    skip.add(bound)
                    out[excl] = schema[bound]
        for key, value in schema.items():
            if key in skip:
                continue
            if key == "$ref" and isinstance(value, str) and value.startswith(_REF_PREFIX):
                name = _pointer_unescape(value[len(_REF_PREFIX):])
                self.wanted.add(name)
                out[key] = "#/$defs/" + _pointer_escape(name)
            elif key in _DATA_KEYS:
                out[key] = copy.deepcopy(value)
            elif key in _SCHEMA_MAPS and isinstance(value, dict):
                out[key] = {k: self.convert(v) for k, v in value.items()}
            elif key in _SCHEMA_LISTS and isinstance(value, list):
                out[key] = [self.convert(v) for v in value]
            elif key in _SCHEMA_SINGLE:
                out[key] = self.convert(value) if isinstance(value, (dict, list)) else value
            else:
                out[key] = copy.deepcopy(value)
        if nullable:
            t = out.get("type")
            if isinstance(t, str):
                out["type"] = [t, "null"] if t != "null" else "null"
            elif isinstance(t, list):
                if "null" not in t:
                    out["type"] = [*t, "null"]
            elif "$ref" in out or "allOf" in out:
                out = {"anyOf": [out, {"type": "null"}]}
            if isinstance(out.get("enum"), list) and None not in out["enum"]:
                out["enum"] = [*out["enum"], None]
        return out

    def defs_for(self) -> dict[str, Any]:
        defs: dict[str, Any] = {}
        pending = sorted(self.wanted)
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            body = self.schemas.get(name)
            if body is None:
                self.warnings.append(f"unresolved reference {_REF_PREFIX}{name}")
                defs[name] = {}
                continue
            before = set(self.wanted)
            defs[name] = self.convert(body)
            pending.extend(sorted(self.wanted - before - seen))
        return defs


def _operation_description(op: dict[str, Any]) -> str:
    parts = [str(op.get(k)).strip() for k in ("summary", "description") if isinstance(op.get(k), str) and str(op.get(k)).strip()]
    if len(parts) == 2 and parts[1].startswith(parts[0]):
        parts = [parts[1]]
    return "\n\n".join(parts)[:MAX_DESCRIPTION]


def openapi_to_tools(doc: dict) -> tuple[list[dict], list[str]]:
    """Return ``(tools, warnings)``; raises ``ValueError`` for a document that is not OpenAPI 3.x."""
    warnings: list[str] = []
    if not isinstance(doc, dict):
        raise ValueError("The OpenAPI document must be a mapping.")
    version = str(doc.get("openapi") or "")
    if not (version.startswith("3.0") or version.startswith("3.1")):
        raise ValueError(f"Only OpenAPI 3.0 and 3.1 are supported (found {version or 'no openapi field'!r}).")
    try:
        size = len(json.dumps(doc, default=str))
    except (TypeError, ValueError):
        raise ValueError("The OpenAPI document is not plain data.") from None
    if size > MAX_DOC_BYTES:
        raise ValueError(f"The OpenAPI document is larger than {MAX_DOC_BYTES // (1024 * 1024)} MB.")
    paths = doc.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise ValueError("The OpenAPI document has no paths.")

    tools: list[dict[str, Any]] = []
    used: set[str] = set()
    operations = 0
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        item = _Converter(doc, warnings).resolve(item, "pathItems") if "$ref" in item else item
        for method in _METHODS:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            operations += 1
            if operations > MAX_OPERATIONS:
                warnings.append(f"more than {MAX_OPERATIONS} operations; the rest were ignored")
                return tools, warnings
            tool = _operation_to_tool(doc, str(path), method, op, item, used, warnings)
            if tool is not None:
                tools.append(tool)
    if not tools:
        warnings.append("no usable operations were found")
    return tools, warnings


def _operation_to_tool(
    doc: dict[str, Any], path: str, method: str, op: dict[str, Any], item: dict[str, Any],
    used: set[str], warnings: list[str],
) -> dict[str, Any] | None:
    conv = _Converter(doc, warnings)
    raw_name = op.get("operationId") if isinstance(op.get("operationId"), str) and op.get("operationId").strip() else f"{method}_{_slug(path)}"
    base = sanitize_name(raw_name)
    name, i = base, 2
    while name in used:
        suffix = f"_{i}"
        name, i = base[: 64 - len(suffix)] + suffix, i + 1
    used.add(name)

    props: dict[str, Any] = {}
    required: list[str] = []
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for src in (item.get("parameters"), op.get("parameters")):
        for p in src if isinstance(src, list) else []:
            p = conv.resolve(p, "parameters")
            if isinstance(p, dict) and isinstance(p.get("name"), str) and p.get("in") in ("path", "query", "header"):
                merged[(p["in"], p["name"])] = p
    for (loc, pname), p in merged.items():
        if loc == "header" and pname.lower() in _IGNORED_HEADERS:
            continue
        schema = p.get("schema")
        if schema is None and isinstance(p.get("content"), dict):
            media = next((m for m in p["content"].values() if isinstance(m, dict) and isinstance(m.get("schema"), dict)), None)
            schema = media.get("schema") if media else None
        converted = conv.convert(schema if isinstance(schema, dict) else {})
        if isinstance(p.get("description"), str) and p["description"].strip() and "description" not in converted:
            converted = {**converted, "description": p["description"].strip()[:300]}
        if pname in props:
            warnings.append(f"{name}: parameter {pname!r} appears in more than one location; kept the first")
            continue
        props[pname] = converted
        if loc == "path" or p.get("required") is True:
            required.append(pname)

    body = conv.resolve(op.get("requestBody"), "requestBodies") if op.get("requestBody") is not None else None
    if isinstance(body, dict):
        content = body.get("content") if isinstance(body.get("content"), dict) else {}
        media = content.get("application/json")
        if media is None:
            media = next((m for k, m in content.items() if isinstance(k, str) and (k.endswith("+json") or k.endswith("/json"))), None)
        if isinstance(media, dict):
            if "body" in props:
                warnings.append(f"{name}: a parameter is called 'body'; the request body was not added")
            else:
                props["body"] = conv.convert(media.get("schema") if isinstance(media.get("schema"), dict) else {})
                if body.get("required") is True:
                    required.append("body")
        else:
            warnings.append(f"{name}: request body has no JSON media type; the body was not added")

    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = list(dict.fromkeys(required))
    defs = conv.defs_for()
    if defs:
        schema["$defs"] = defs

    try:
        jsb.validate_bundle({"tool.json": schema}, "tool.json")
    except jsb.SchemaBundleError as exc:
        warnings.append(f"{name}: skipped, schema is not usable: {(exc.errors or [str(exc)])[0]}")
        return None
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{name}: skipped, schema is not usable: {exc}")
        return None

    if method in ("get", "head"):
        annotations: dict[str, Any] = {"readOnlyHint": True}
    elif method == "delete":
        annotations = {"readOnlyHint": False, "destructiveHint": True}
    else:
        annotations = {"readOnlyHint": False}
    description = _operation_description(op) or f"{method.upper()} {path}"
    return {"name": name, "description": description, "inputSchema": schema, "annotations": annotations}
