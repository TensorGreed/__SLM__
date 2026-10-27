"""Multi-file JSON-Schema bundles: validate offline, flatten for guided decoding, build minimal instances.

A bundle is ``{file name: schema object}`` plus the name of the root file. Files reference each
other by ``$ref`` (file name, or ``$id``); nothing is ever fetched: the ``referencing`` registry's
``retrieve`` raises, so a remote ref is a reported error, not a network call.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import unquote, urljoin

import regex as _regex
from jsonschema import validators as _js_validators
from jsonschema.exceptions import ValidationError as _ValidationError
from jsonschema.exceptions import relevance as _relevance
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT201909, DRAFT202012

from app.services.json_pointer import format_json_pointer, resolve_json_pointer, split_json_pointer

MAX_FILES = 20
MAX_BUNDLE_BYTES = 1024 * 1024
MAX_ERRORS = 50
_BASE = "https://bundle.invalid/"  # urljoin only resolves relative refs for schemes it knows
_DIALECT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_DIALECT_2019_09 = "https://json-schema.org/draft/2019-09/schema"
SUPPORTED_DIALECTS = (_DIALECT_2020_12, _DIALECT_2019_09)
# Keywords whose value is instance data, never a schema: refs inside them are not refs. They are data
# only in KEYWORD position: a property called "default" under ``properties`` is a schema.
_DATA_KEYS = frozenset({"enum", "const", "default", "examples"})
# Keywords whose value maps arbitrary names to schemas: every child of such a map is a schema.
_SCHEMA_MAP_KEYS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
# A user ``pattern`` runs under the ``regex`` module with a timeout (stdlib ``re`` can backtrack forever).
PATTERN_TIMEOUT_S = 0.5


class SchemaBundleError(ValueError):
    """The bundle is not usable; ``.errors`` lists every problem found."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors[:5]) + (f" (+{len(self.errors) - 5} more)" if len(self.errors) > 5 else ""))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _no_retrieve(uri: str):  # pragma: no cover - exercised through Unresolvable
    raise LookupError(f"remote or unknown reference {uri!r} (bundles never fetch anything)")


def _file_uri(name: str) -> str:
    return _BASE + name


def _base_uri(name: str, schema: dict[str, Any]) -> str:
    ident = schema.get("$id") if isinstance(schema.get("$id"), str) else schema.get("id") if isinstance(schema.get("id"), str) else None
    if ident:
        return urljoin(_file_uri(name), ident.split("#", 1)[0])
    return _file_uri(name)


def _iter_refs(node: Any, path: tuple = (), in_map: bool = False):
    """Yield ``(ref string, path)`` for every ``$ref`` in schema position. ``in_map``: ``node`` is a
    schema map (``properties`` etc.), so its keys are names and its values are schemas."""
    if isinstance(node, dict):
        for key, value in node.items():
            if in_map:
                yield from _iter_refs(value, path + (key,))
                continue
            if key in _DATA_KEYS:
                continue
            if key == "$ref" and isinstance(value, str):
                yield value, path
            else:
                yield from _iter_refs(value, path + (key,), key in _SCHEMA_MAP_KEYS and isinstance(value, dict))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _iter_refs(value, path + (i,))


def _dialect_of(schema: dict[str, Any]) -> str | None:
    """The supported dialect URL a schema declares (2020-12 when it declares none), else None."""
    declared = schema.get("$schema")
    if declared is None:
        return _DIALECT_2020_12
    if not isinstance(declared, str):
        return None
    norm = declared.strip().rstrip("#")
    return norm if norm in SUPPORTED_DIALECTS else None


def _dialect_classes(dialect: str):
    if dialect == _DIALECT_2019_09:
        return _js_validators.Draft201909Validator, DRAFT201909
    return _js_validators.Draft202012Validator, DRAFT202012


@lru_cache(maxsize=256)
def _rx(pattern: str):
    return _regex.compile(pattern)


def _rx_search(pattern: str, text: str) -> bool:
    return _rx(pattern).search(text, timeout=PATTERN_TIMEOUT_S) is not None


def _pattern_kw(validator, patrn, instance, schema):
    if not validator.is_type(instance, "string"):
        return
    try:
        if not _rx_search(patrn, instance):
            yield _ValidationError(f"{instance!r} does not match {patrn!r}")
    except TimeoutError:
        yield _ValidationError(f"pattern check timed out ({patrn!r} took longer than {PATTERN_TIMEOUT_S}s)")
    except _regex.error:
        yield _ValidationError(f"pattern {patrn!r} is not a usable regular expression")


def _pattern_properties_kw(validator, pattern_properties, instance, schema):
    if not validator.is_type(instance, "object"):
        return
    for pattern, subschema in pattern_properties.items():
        for k, v in instance.items():
            try:
                hit = _rx_search(pattern, k)
            except TimeoutError:
                yield _ValidationError(f"patternProperties check timed out ({pattern!r} took longer than {PATTERN_TIMEOUT_S}s)")
                continue
            except _regex.error:
                yield _ValidationError(f"pattern {pattern!r} is not a usable regular expression")
                continue
            if hit:
                yield from validator.descend(v, subschema, path=k, schema_path=pattern)


def _additional_properties_kw(validator, aP, instance, schema):
    if not validator.is_type(instance, "object"):
        return
    props = schema.get("properties", {})
    patterns = list(schema.get("patternProperties", {}) or {})
    extras = []
    for name in instance:
        if name in props:
            continue
        matched = False
        for pat in patterns:
            try:
                if _rx_search(pat, name):
                    matched = True
                    break
            except TimeoutError:
                yield _ValidationError(f"patternProperties check timed out ({pat!r} took longer than {PATTERN_TIMEOUT_S}s)")
                matched = True
                break
            except _regex.error:
                continue
        if not matched:
            extras.append(name)
    if validator.is_type(aP, "object"):
        for extra in extras:
            yield from validator.descend(instance[extra], aP, path=extra)
    elif aP is False and extras:
        which = ", ".join(repr(e) for e in extras)
        yield _ValidationError(f"Additional properties are not allowed ({which} {'was' if len(extras) == 1 else 'were'} unexpected)")


class _TimedRe:
    """Stands in for ``re`` inside ``jsonschema._utils``. Its ``find_evaluated_property_keys_by_schema``
    (``unevaluatedProperties``) matches ``patternProperties`` with a bare ``re.search`` that no keyword
    override reaches, so a user pattern could backtrack on model output there without a time limit.
    A timeout raises ``TimeoutError``; ``SchemaBundle.validate`` turns it into an error entry."""

    def __getattr__(self, name: str) -> Any:
        return getattr(re, name)

    @staticmethod
    def search(pattern: str, string: str, flags: int = 0):
        return _regex.compile(pattern, flags).search(string, timeout=PATTERN_TIMEOUT_S)


def _install_timed_re() -> None:
    from jsonschema import _utils as _js_utils

    if not isinstance(getattr(_js_utils, "re", None), _TimedRe):
        _js_utils.re = _TimedRe()


_install_timed_re()

_EXTENDED: dict[str, Any] = {}


def _validator_class(dialect: str):
    cls = _EXTENDED.get(dialect)
    if cls is None:
        base, _spec = _dialect_classes(dialect)
        cls = _js_validators.extend(
            base,
            {"pattern": _pattern_kw, "patternProperties": _pattern_properties_kw, "additionalProperties": _additional_properties_kw},
        )
        _EXTENDED[dialect] = cls
    return cls


def _registry_for(files: dict[str, Any], dialect: str = _DIALECT_2020_12) -> Registry:
    resources: list[tuple[str, Resource]] = []
    spec = _dialect_classes(dialect)[1]
    for name, schema in files.items():
        res = Resource.from_contents(schema, default_specification=spec)
        resources.append((_file_uri(name), res))
        base = _base_uri(name, schema)
        if base != _file_uri(name):
            resources.append((base, res))
    return Registry(retrieve=_no_retrieve).with_resources(resources)


@dataclass(frozen=True)
class SchemaBundle:
    files: dict[str, Any]
    root: str
    sha256: str

    def root_schema(self) -> dict[str, Any]:
        return self.files[self.root]

    # ── validation ──────────────────────────────────────────────────────

    def dialect(self) -> str:
        return _dialect_of(self.root_schema()) or _DIALECT_2020_12

    def _validator(self):
        with _CACHE_LOCK:
            cached = _VALIDATORS.get(self.sha256)
            if cached is not None:
                _VALIDATORS.move_to_end(self.sha256)
                return cached
        root = copy.deepcopy(self.root_schema())
        dialect = self.dialect()
        cls = _validator_class(dialect)
        # A root with no $id would resolve relative refs against "" -- give it its file URI.
        if not (isinstance(root.get("$id"), str) or isinstance(root.get("id"), str)):
            root["$id"] = _file_uri(self.root)
        validator = cls(root, registry=_registry_for(self.files, dialect))  # format_checker off: portable
        with _CACHE_LOCK:
            _VALIDATORS[self.sha256] = validator
            while len(_VALIDATORS) > 32:
                _VALIDATORS.popitem(last=False)
        return validator

    def validate(self, instance: Any) -> list[dict[str, Any]]:
        """Every schema violation of ``instance`` (most relevant first, at most 50), as
        ``{"pointer", "message", "validator", "schema_path"}``; ``[]`` when valid. Never raises on a
        schema/reference problem: that comes back as a single error entry."""
        try:
            errors = list(self._validator().iter_errors(instance))
        except TimeoutError:
            return [{
                "pointer": "",
                "message": f"pattern check timed out (a pattern took longer than {PATTERN_TIMEOUT_S}s)",
                "validator": "pattern",
                "schema_path": "",
            }]
        except Exception as exc:  # noqa: BLE001 - referencing errors, recursion, bad schema internals
            return [{
                "pointer": "",
                "message": f"the schema could not be evaluated: {type(exc).__name__}: {str(exc)[:300]}",
                "validator": "$ref",
                "schema_path": "",
            }]
        rows = [
            {
                "pointer": format_json_pointer(e.absolute_path),
                "message": str(e.message)[:500],
                "validator": str(e.validator) if e.validator is not None else "",
                "schema_path": format_json_pointer(e.absolute_schema_path),
                "_rel": e,
            }
            for e in errors
        ]
        rows.sort(key=lambda r: (r["pointer"], r["message"], r["schema_path"]))
        rows.sort(key=lambda r: _relevance(r["_rel"]), reverse=True)
        for r in rows:
            del r["_rel"]
        return rows[:MAX_ERRORS]

    # ── flattening ──────────────────────────────────────────────────────

    def dereferenced(self) -> dict[str, Any]:
        """One self-contained document in the bundle's dialect (2020-12 or 2019-09): other files inlined
        under ``$defs``, every ``$ref`` a local ``#/...`` pointer (recursive refs stay as local refs).
        Cached per bundle sha; callers get their own copy."""
        with _CACHE_LOCK:
            hit = _DEREF.get(self.sha256)
            if hit is not None:
                _DEREF.move_to_end(self.sha256)
        if hit is None:
            hit = self._dereference()
            with _CACHE_LOCK:
                _DEREF[self.sha256] = hit
                while len(_DEREF) > 32:
                    _DEREF.popitem(last=False)
        return copy.deepcopy(hit)

    def _dereference(self) -> dict[str, Any]:
        dialect = self.dialect()
        names = {n: _def_name(n) for n in self.files if n != self.root}
        out = copy.deepcopy(self.root_schema())
        out["$schema"] = dialect
        out.pop("id", None)
        out.pop("$id", None)
        used = set(out["$defs"]) if isinstance(out.get("$defs"), dict) else set()
        unique: dict[str, str] = {}
        for n, d in names.items():
            cand, i = d, 2
            while cand in used:
                cand, i = f"{d}_{i}", i + 1
            used.add(cand)
            unique[n] = cand
        files_by_uri = {_file_uri(n): n for n in self.files}
        for n, s in self.files.items():
            files_by_uri.setdefault(_base_uri(n, s), n)

        def target(ref_base: str, current: str) -> str:
            if not ref_base:
                return current
            absolute = urljoin(_base_uri(current, self.files[current]), ref_base)
            found = files_by_uri.get(absolute)
            if found is None:
                raise SchemaBundleError([f"{current}: $ref {ref_base!r} does not resolve inside the bundle"])
            return found

        def rewrite(node: Any, current: str, in_map: bool = False) -> Any:
            if isinstance(node, dict):
                res: dict[str, Any] = {}
                for key, value in node.items():
                    if in_map:
                        res[key] = rewrite(value, current)
                    elif key in _DATA_KEYS:
                        res[key] = copy.deepcopy(value)
                    elif key == "$ref" and isinstance(value, str):
                        base, _, frag = value.partition("#")
                        tgt = target(base, current)
                        if frag and not frag.startswith("/") and tgt != self.root:
                            raise SchemaBundleError([f"{current}: anchor ref {value!r} cannot be flattened: " + _ANCHOR_HINT])
                        if tgt == self.root:
                            res[key] = "#" + frag  # a plain-name anchor of the root stays valid in the merged document
                        else:
                            res[key] = "#/$defs/" + unique[tgt] + frag
                            need(tgt)
                    else:
                        res[key] = rewrite(value, current, key in _SCHEMA_MAP_KEYS and isinstance(value, dict))
                return res
            if isinstance(node, list):
                return [rewrite(v, current) for v in node]
            return node

        pending: list[str] = []
        done: set[str] = set()

        def need(name: str) -> None:
            if name not in done:
                done.add(name)
                pending.append(name)

        rewritten_root = rewrite(out, self.root)
        if not isinstance(rewritten_root.get("$defs"), dict):
            rewritten_root["$defs"] = {}
        out_defs = rewritten_root["$defs"]
        while pending:
            name = pending.pop()
            body = copy.deepcopy(self.files[name])
            body.pop("$schema", None)
            body.pop("$id", None)
            body.pop("id", None)
            out_defs[unique[name]] = rewrite(body, name)
        rewritten_root["$schema"] = dialect
        if not out_defs:
            rewritten_root.pop("$defs", None)
        return rewritten_root


_VALIDATORS: "OrderedDict[str, Any]" = OrderedDict()
_DEREF: "OrderedDict[str, Any]" = OrderedDict()
_ANCHOR_HINT = "plain-name anchors across files are not supported; use a JSON-pointer fragment (e.g. step.schema.json#/$defs/x)."
_CACHE_LOCK = threading.Lock()


def _def_name(file_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", file_name).strip("_") or "file"


def validate_bundle(files: dict[str, Any], root: str) -> SchemaBundle:
    """Check a bundle and return it, or raise ``SchemaBundleError`` listing every problem. Supported
    dialects: JSON Schema 2020-12 (the default when ``$schema`` is absent) and 2019-09; every file must
    use the same one."""
    try:
        return _validate_bundle(files, root)
    except RecursionError:
        raise SchemaBundleError(["The schema is nested too deeply to check."]) from None


def _validate_bundle(files: dict[str, Any], root: str) -> SchemaBundle:
    errors: list[str] = []
    if not isinstance(files, dict) or not files:
        raise SchemaBundleError(["The schema bundle has no files."])
    if len(files) > MAX_FILES:
        errors.append(f"{len(files)} schema files; the limit is {MAX_FILES}.")
    for name, schema in files.items():
        if not isinstance(name, str) or not name.strip():
            errors.append("Schema file names must be non-empty strings.")
        elif not isinstance(schema, dict):
            errors.append(f"{name}: must be a JSON object.")
    if root not in files:
        errors.append(f"root schema {root!r} is not one of the files.")
    try:
        canonical = _canonical({"root": root, "files": files})
    except (TypeError, ValueError) as exc:
        raise SchemaBundleError(errors + [f"The bundle is not JSON-serialisable: {exc}"]) from None
    if len(canonical.encode("utf-8")) > MAX_BUNDLE_BYTES:
        errors.append(f"The bundle is larger than {MAX_BUNDLE_BYTES // 1024} KB.")
    if errors:
        raise SchemaBundleError(errors)

    dialect_ok = True
    dialects: dict[str, str] = {}
    for name, schema in files.items():
        dialect = _dialect_of(schema)
        if dialect is None:
            errors.append(
                f"{name}: unsupported $schema dialect {schema.get('$schema')!r}; use JSON Schema 2020-12 "
                f"({_DIALECT_2020_12}) or 2019-09 ({_DIALECT_2019_09})."
            )
            dialect_ok = False
            continue
        dialects[name] = dialect
        cls = _dialect_classes(dialect)[0]
        meta = sorted(cls(cls.META_SCHEMA).iter_errors(schema), key=lambda e: (format_json_pointer(e.absolute_path), e.message))
        for e in meta[:5]:
            errors.append(f"{name}: not a valid schema at {format_json_pointer(e.absolute_path) or '/'}: {str(e.message)[:200]}")
    if len(set(dialects.values())) > 1:
        errors.append("All schema files must use the same JSON Schema dialect (2020-12 or 2019-09); the files mix both.")
        dialect_ok = False
    bundle_dialect = next(iter(dialects.values()), _DIALECT_2020_12)
    if dialect_ok:
        try:
            registry = _registry_for(files, bundle_dialect)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Could not index the bundle: {str(exc)[:200]}")
            registry = None
        if registry is not None:
            for name, schema in files.items():
                resolver = registry.resolver(_base_uri(name, schema))
                seen: set[str] = set()
                for ref, _path in _iter_refs(schema):
                    if ref in seen:
                        continue
                    seen.add(ref)
                    try:
                        resolver.lookup(ref)
                    except Unresolvable:
                        errors.append(f"{name}: $ref {ref!r} does not resolve inside the bundle (remote references are not allowed).")
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{name}: $ref {ref!r} is not usable: {str(exc)[:160]}")
    if errors:
        raise SchemaBundleError(errors)
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    bundle = SchemaBundle(files=copy.deepcopy(files), root=root, sha256=sha)
    # Everything a prompt build or a validation needs must work NOW, so a saved bundle never fails later.
    try:
        bundle.dereferenced()
        bundle._validator()
    except SchemaBundleError as exc:
        raise SchemaBundleError(exc.errors) from None
    except RecursionError:
        raise SchemaBundleError(["The schema is nested too deeply to flatten."]) from None
    except Exception as exc:  # noqa: BLE001
        raise SchemaBundleError([f"The bundle cannot be used: {type(exc).__name__}: {str(exc)[:200]}"]) from None
    return bundle


def bundle_from_config(cfg: dict | None) -> SchemaBundle | None:
    """The bundle saved in a structured-plan config (``schema_files`` + ``root_schema``), else None."""
    if not isinstance(cfg, dict):
        return None
    files, root = cfg.get("schema_files"), cfg.get("root_schema")
    if not isinstance(files, dict) or not files or not isinstance(root, str) or not root:
        return None
    return validate_bundle(files, root)


# ── minimal instance ────────────────────────────────────────────────────


def _resolve_local(ref: str, root: Any) -> Any:
    if not ref.startswith("#"):
        return None
    frag = unquote(ref[1:])
    if frag == "":
        return root
    tokens = split_json_pointer(frag)
    if tokens is None:
        return None
    found = resolve_json_pointer(root, [t for t in tokens])
    return found[0] if found else None


def _simple_pattern_string(pattern: str, min_len: int) -> str | None:
    p = pattern
    if p.startswith("^"):
        p = p[1:]
    if p.endswith("$"):
        p = p[:-1]
    if re.fullmatch(r"[A-Za-z0-9_\-]+", p):
        return p
    m = re.fullmatch(r"\[([^\]^\\]+)\](?:([+*])|\{(\d+)(?:,\d*)?\})?", p)
    if m:
        cls = m.group(1)
        ch = cls[0]
        if len(cls) >= 3 and cls[1] == "-":
            ch = cls[0]
        count = int(m.group(3)) if m.group(3) else 1
        return ch * max(count, 1, min_len)
    return None


def minimal_instance(schema: dict, *, root: dict | None = None) -> Any:
    """A best-effort instance that satisfies ``schema`` (required properties, first enum/const value,
    ``minItems``, local ``$ref``, first ``oneOf``/``anyOf`` branch, ``minimum``). It can miss exotic
    constraints (cross-field rules, odd patterns); callers validate the result."""
    return _minimal(schema, root if isinstance(root, dict) else schema, (), 0)


def _minimal(schema: Any, root: Any, ref_stack: tuple, depth: int) -> Any:
    if not isinstance(schema, dict):
        return None
    if depth > 24:
        return None
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return copy.deepcopy(schema["enum"][0])
    if isinstance(schema.get("$ref"), str):
        ref = schema["$ref"]
        if ref in ref_stack:
            return None
        target = _resolve_local(ref, root)
        if isinstance(target, dict):
            merged = {k: v for k, v in schema.items() if k != "$ref"}
            return _minimal({**target, **merged}, root, ref_stack + (ref,), depth + 1)
        return None
    for key in ("oneOf", "anyOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and branches:
            base = {k: v for k, v in schema.items() if k != key}
            return _minimal({**base, **branches[0]} if isinstance(branches[0], dict) else base, root, ref_stack, depth + 1)
    if isinstance(schema.get("allOf"), list) and schema["allOf"]:
        merged_obj: dict[str, Any] = {}
        base = {k: v for k, v in schema.items() if k != "allOf"}
        parts = [_minimal(p, root, ref_stack, depth + 1) for p in schema["allOf"]]
        if base:
            parts.append(_minimal(base, root, ref_stack, depth + 1))
        dicts = [p for p in parts if isinstance(p, dict)]
        if dicts:
            for d in dicts:
                merged_obj.update(d)
            return merged_obj
        return next((p for p in parts if p is not None), None)

    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0] if t else None)
    if t is None:
        if "properties" in schema or "required" in schema:
            t = "object"
        elif "items" in schema or "prefixItems" in schema:
            t = "array"
        elif any(k in schema for k in ("minimum", "maximum")):
            t = "number"
        elif any(k in schema for k in ("pattern", "minLength", "maxLength")):
            t = "string"
        else:
            return None
    if t == "object":
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        out: dict[str, Any] = {}
        extra = schema.get("additionalProperties")
        for key in schema.get("required") or []:
            if key in props:
                out[key] = _minimal(props[key], root, ref_stack, depth + 1)
            elif isinstance(extra, dict):
                out[key] = _minimal(extra, root, ref_stack, depth + 1)
            else:
                out[key] = "example"
        return out
    if t == "array":
        n = int(schema.get("minItems") or 0)
        prefix = schema.get("prefixItems") if isinstance(schema.get("prefixItems"), list) else []
        items = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        out_l = [_minimal(p, root, ref_stack, depth + 1) for p in prefix[:n]]
        while len(out_l) < n:
            out_l.append(_minimal(items, root, ref_stack, depth + 1))
        return out_l
    if t == "string":
        min_len = int(schema.get("minLength") or 0)
        value: str | None = None
        if isinstance(schema.get("pattern"), str):
            value = _simple_pattern_string(schema["pattern"], min_len)
            if value is not None and not re.search(schema["pattern"], value):
                value = None
        if value is None:
            value = "example"
        if len(value) < min_len:
            value = value + "x" * (min_len - len(value))
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
        return value
    if t in ("integer", "number"):
        lo = schema.get("minimum")
        if lo is None and isinstance(schema.get("exclusiveMinimum"), (int, float)) and not isinstance(schema.get("exclusiveMinimum"), bool):
            lo = schema["exclusiveMinimum"] + 1
        val: Any = lo if isinstance(lo, (int, float)) and not isinstance(lo, bool) else 0
        hi = schema.get("maximum")
        if isinstance(hi, (int, float)) and not isinstance(hi, bool) and val > hi:
            val = hi
        if t == "integer" and isinstance(val, float):
            val = int(val) if val == int(val) else int(val) + 1
        return val
    if t == "boolean":
        return False
    if t == "null":
        return None
    return None
