"""Label taxonomy: the project's controlled vocabulary of labels.

A taxonomy is a list of entries ``{id, title, parent, aliases, attrs,
crosswalk}`` — a label id, a human title, an optional parent (a tree), other
spellings that should resolve to the same label, free-form attributes, and a
``crosswalk`` that maps the label to reference ids in other namespaces
(``{"stdA": ["A-12"], "stdB": ["7.3"]}``). Importers resolve raw labels from
external datasets to taxonomy ids, so every row of a project speaks the same
label language no matter where it came from.

The taxonomy is stored as ``Project.runtime_config["label_taxonomy"]`` and is
versioned the way the span labelling conventions are
(``span_type_definitions_service``): the version is a short hash of the
canonical JSON, and ``runtime_config["taxonomy_history"]`` keeps what each
version said so an earlier one can be restored.

Nothing here knows what the labels are about — it is a general-purpose
vocabulary with a crosswalk, not a domain handler.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Iterable

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

TAXONOMY_KEY = "label_taxonomy"
HISTORY_KEY = "taxonomy_history"
MAX_HISTORY = 20
KEEP_FULL_SNAPSHOTS = 5
MAX_TEXT_BYTES = 1024 * 1024
MAX_ENTRIES = 5000

_META_KEYS = {"name", "version", "description"}
_ENTRY_FIELDS = {"id", "title", "parent", "aliases", "attrs", "crosswalk"}


class TaxonomyError(ValueError):
    """Parsing failed; ``errors`` is a list of ``{code, id, message}``."""

    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = errors
        super().__init__(errors[0]["message"] if errors else "invalid taxonomy")


def _err(code: str, id_: Any, message: str) -> dict[str, Any]:
    return {"code": code, "id": None if id_ is None else str(id_), "message": message}


def norm(value: Any) -> str:
    """Casefold, strip and drop every non-alphanumeric character, so
    ``Leak 1``, ``leak-1`` and ``LEAK_1`` all compare equal."""
    return "".join(ch for ch in str(value if value is not None else "").strip().casefold() if ch.isalnum())


# ── Parsing ─────────────────────────────────────────────────────────────


class _NoAliasLoader(yaml.SafeLoader):
    """``safe_load`` that refuses YAML aliases (``*x``): they let a tiny text
    expand into a huge object ("billion laughs"). Anchors alone are harmless."""

    def compose_node(self, parent, index):  # type: ignore[override]
        if self.check_event(yaml.events.AliasEvent):
            event = self.peek_event()
            raise yaml.YAMLError("YAML aliases are not allowed (alias '*%s')." % event.anchor)
        return super().compose_node(parent, index)


def _jsonable(value: Any, errors: list[dict[str, Any]], depth: int = 0) -> Any:
    """Make a loaded YAML object JSON-safe: dates -> ISO strings, sets ->
    sorted lists of strings, non-string mapping keys -> strings, bytes ->
    error. Floats used as keys are an error (``1.10`` becomes ``1.1``)."""
    if depth > 200:
        raise RecursionError("nesting too deep")
    if isinstance(value, bytes):
        errors.append(_err("bad_attrs", None, "Binary (!!binary) values are not allowed in a taxonomy."))
        return None
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(str(_jsonable(v, errors, depth + 1)) for v in value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, float):
                errors.append(_err("quote_number", None, f"The key {k} is a number; quote dotted numbers, e.g. '1.10'."))
            key = k.isoformat() if isinstance(k, (_dt.date, _dt.datetime)) else str(k)
            out[key] = _jsonable(v, errors, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, errors, depth + 1) for v in value]
    return value


def _as_id(value: Any, errors: list[dict[str, Any]] | None = None, owner: Any = None) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, float):
        if errors is not None:
            errors.append(_err("quote_number", owner, f"{value} is a non-integer number; quote dotted numbers, e.g. '1.10'."))
        return None
    text = str(value).strip()
    return text or None


def _str_list(value: Any, errors: list[dict[str, Any]] | None = None, owner: Any = None) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    out: list[str] = []
    for item in items:
        text = _as_id(item, errors, owner)
        if text is not None and text not in out:
            out.append(text)
    return out


def _collection(value: Any) -> bool:
    return isinstance(value, (list, dict))


def _entries_from(value: Any, errors: list[dict[str, Any]]) -> list[tuple[Any, Any]]:
    """``[(key_or_None, raw_entry)]`` from a list or an id-keyed mapping."""
    if isinstance(value, list):
        return [(None, item) for item in value]
    if isinstance(value, dict):
        return list(value.items())
    errors.append(_err("bad_layout", None, "entries must be a list or a mapping keyed by id."))
    return []


def _locate_entries(obj: Any, errors: list[dict[str, Any]]) -> tuple[list[tuple[Any, Any]], str | None, str | None]:
    name = version = None
    if isinstance(obj, list):
        return [(None, item) for item in obj], name, version
    if not isinstance(obj, dict):
        errors.append(_err("bad_layout", None, "A taxonomy is a list of entries or a mapping with an 'entries' collection."))
        return [], None, None
    name = _as_id(obj.get("name"))
    version = _as_id(obj.get("version"))
    if "entries" in obj:
        return _entries_from(obj["entries"], errors), name, version
    candidates = [k for k, v in obj.items() if k not in _META_KEYS and _collection(v)]
    if len(candidates) == 1:
        return _entries_from(obj[candidates[0]], errors), name, version
    if not candidates:
        errors.append(_err("bad_layout", None, "No entries found: expected an 'entries' collection or a single list of entries."))
    else:
        errors.append(_err("bad_layout", None, f"More than one collection ({', '.join(map(str, candidates))}); put the entries under 'entries'."))
    return [], name, version


def _normalise_entry(key: Any, raw: Any, crosswalk_from: list[str], errors: list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(raw, dict):
        fields = dict(raw)
    elif raw is None or isinstance(raw, str) and key is not None:
        # id-keyed shorthand: ``LEAK-1: Some title`` / ``LEAK-1:``
        fields = {"title": raw} if raw else {}
    else:
        fields = {"id": raw}
    if isinstance(key, float):
        errors.append(_err("quote_number", None, f"The id {key} is a non-integer number; quote dotted numbers, e.g. '1.10'."))
    ident = _as_id(key) if key is not None else _as_id(fields.get("id"), errors)
    title = fields.get("title")
    if title is None and "name" in fields:
        title = fields.get("name")
    title_s = " ".join(str(title).split()) if title is not None and not isinstance(title, (dict, list)) else ""
    attrs_raw = fields.get("attrs")
    attrs: Any = dict(attrs_raw) if isinstance(attrs_raw, dict) else ({} if attrs_raw is None else attrs_raw)
    crosswalk_raw = fields.get("crosswalk")
    crosswalk: Any = (
        {str(ns) if isinstance(ns, (str, int)) else ns: (list(refs) if isinstance(refs, (list, tuple)) else ([refs] if isinstance(refs, (str, int)) else refs))
         for ns, refs in crosswalk_raw.items()}
        if isinstance(crosswalk_raw, dict) else ({} if crosswalk_raw is None else crosswalk_raw)
    )
    if isinstance(crosswalk, dict):
        for refs in crosswalk.values():
            for r in refs if isinstance(refs, list) else []:
                if isinstance(r, float):
                    errors.append(_err("quote_number", ident, f"Crosswalk reference {r} of '{ident}' is a non-integer number; quote dotted numbers, e.g. '1.10'."))
        crosswalk = {ns: ([str(r) if isinstance(r, int) and not isinstance(r, bool) else r for r in refs] if isinstance(refs, list) else refs) for ns, refs in crosswalk.items()}
    used = _ENTRY_FIELDS | ({"name"} if "title" not in fields and "name" in fields else set())
    for field_name in crosswalk_from:
        if field_name not in fields:
            continue
        values = fields[field_name]
        for item in values if isinstance(values, (list, tuple)) else [values]:
            if item is None:
                continue
            if isinstance(item, float):
                errors.append(_err("quote_number", ident, f"{item} in field '{field_name}' is a non-integer number; quote dotted numbers, e.g. '1.10'."))
                continue
            text = str(item).strip()
            if not text:
                continue
            ns, _sep, ref = text.partition(":")
            ns, ref = ns.strip(), ref.strip()
            if not ns:
                errors.append(_err("bad_crosswalk_ref", ident, f"'{text}' in field '{field_name}' has no namespace."))
                continue
            if not isinstance(crosswalk, dict):
                break
            refs_list = crosswalk.setdefault(ns, [])
            # a value without ':' (or 'NS:') names a whole namespace: kept with no refs
            if ref and isinstance(refs_list, list) and ref not in refs_list:
                refs_list.append(ref)
    if isinstance(attrs, dict):
        for field_name, value in fields.items():
            if field_name in used or field_name in crosswalk_from:
                continue
            attrs.setdefault(str(field_name), value)
    parent = fields.get("parent")
    return {
        "id": ident,
        "title": title_s or (ident or ""),
        "parent": _as_id(parent, errors, ident),
        "aliases": _str_list(fields.get("aliases"), errors, ident),
        "attrs": attrs,
        "crosswalk": crosswalk,
    }


def parse_taxonomy_lenient(
    source: Any, *, fmt: str = "auto", crosswalk_from: Iterable[str] | None = None, name: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Parse text (YAML / JSON) or an already-loaded object into a taxonomy.
    Returns ``(taxonomy | None, errors)``; the taxonomy is not validated."""
    errors: list[dict[str, Any]] = []
    obj = source
    if isinstance(source, (str, bytes)):
        text = source.decode("utf-8", "replace") if isinstance(source, bytes) else source
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            return None, [_err("too_large", None, f"The taxonomy text is larger than {MAX_TEXT_BYTES // 1024} KB.")]
        try:
            obj = yaml.load(text, Loader=_NoAliasLoader)  # noqa: S506 - SafeLoader subclass
        except yaml.YAMLError as exc:
            label = {"json": "JSON", "yaml": "YAML"}.get(fmt, "YAML/JSON")
            return None, [_err("parse_error", None, f"Not valid {label}: {str(exc).splitlines()[0] if str(exc) else exc}")]
        except RecursionError:
            return None, [_err("parse_error", None, "The taxonomy is nested too deeply.")]
    try:
        obj = _jsonable(obj, errors)
    except RecursionError:
        return None, [_err("parse_error", None, "The taxonomy is nested too deeply.")]
    cw_fields = [str(f) for f in (crosswalk_from or []) if str(f).strip()]
    pairs, found_name, version = _locate_entries(obj, errors)
    if errors:
        return None, errors
    if len(pairs) > MAX_ENTRIES:
        return None, [_err("too_many_entries", None, f"{len(pairs)} entries; the limit is {MAX_ENTRIES}.")]
    entries = [_normalise_entry(k, raw, cw_fields, errors) for k, raw in pairs]
    return {"name": name or found_name, "source_version": version, "entries": entries}, errors


def parse_taxonomy(source: Any, *, fmt: str = "auto", crosswalk_from: Iterable[str] | None = None, name: str | None = None) -> dict[str, Any]:
    """Like :func:`parse_taxonomy_lenient` but raises :class:`TaxonomyError`
    when anything is wrong (parse errors, bad crosswalk refs) or the taxonomy
    fails :func:`validate_taxonomy`."""
    tax, errors = parse_taxonomy_lenient(source, fmt=fmt, crosswalk_from=crosswalk_from, name=name)
    if tax is not None:
        errors = errors + validate_taxonomy(tax)
    if errors or tax is None:
        raise TaxonomyError(errors)
    return tax


# ── Validation ──────────────────────────────────────────────────────────


def validate_taxonomy(tax: Any) -> list[dict[str, Any]]:
    """All-or-nothing validation errors (``[]`` = valid)."""
    errors: list[dict[str, Any]] = []
    entries = tax.get("entries") if isinstance(tax, dict) else None
    if not isinstance(entries, list) or not entries:
        return [_err("empty", None, "The taxonomy has no entries.")]
    if len(entries) > MAX_ENTRIES:
        errors.append(_err("too_many_entries", None, f"{len(entries)} entries; the limit is {MAX_ENTRIES}."))
    ids: dict[str, int] = {}
    for position, entry in enumerate(entries, start=1):
        eid = _as_id(entry.get("id")) if isinstance(entry, dict) else None
        if eid is None:
            errors.append(_err("missing_id", None, f"Entry {position} has no id."))
        elif eid in ids:
            errors.append(_err("duplicate_id", eid, f"Id '{eid}' is used by more than one entry."))
        else:
            ids[eid] = position
    valid = [e for e in entries if isinstance(e, dict) and _as_id(e.get("id")) is not None]
    for entry in valid:
        eid = str(entry["id"])
        parent = entry.get("parent")
        if parent not in (None, "") and str(parent) not in ids:
            errors.append(_err("unknown_parent", eid, f"Parent '{parent}' of '{eid}' is not an entry."))
        if not isinstance(entry.get("attrs", {}), dict):
            errors.append(_err("bad_attrs", eid, f"attrs of '{eid}' must be a mapping."))
        cw = entry.get("crosswalk", {})
        if not isinstance(cw, dict):
            errors.append(_err("bad_crosswalk", eid, f"crosswalk of '{eid}' must map a namespace to a list of reference ids."))
        else:
            for ns, refs in cw.items():
                if not isinstance(ns, str) or not ns.strip():
                    errors.append(_err("bad_crosswalk", eid, f"A crosswalk namespace of '{eid}' is not a non-empty string."))
                elif not isinstance(refs, list) or any(not isinstance(r, str) or not r.strip() for r in refs):
                    errors.append(_err("bad_crosswalk", eid, f"crosswalk '{ns}' of '{eid}' must be a list of non-empty strings."))
    # cycles: follow parent links; report each cycle once, by its path
    parent_of = {str(e["id"]): str(e["parent"]) for e in valid if e.get("parent") not in (None, "") and str(e["parent"]) in ids}
    done: set[str] = set()
    for start in parent_of:
        if start in done:
            continue
        path: list[str] = []
        seen_at: dict[str, int] = {}
        cur: str | None = start
        while cur is not None and cur not in done:
            if cur in seen_at:
                cycle = path[seen_at[cur]:] + [cur]
                errors.append(_err("cycle", cycle[0], "Parent cycle: " + " -> ".join(cycle)))
                break
            seen_at[cur] = len(path)
            path.append(cur)
            cur = parent_of.get(cur)
        done.update(path)
    # alias conflicts (normalised) against other entries' ids / aliases
    owner: dict[str, str] = {}
    for entry in valid:
        owner.setdefault(norm(entry["id"]), str(entry["id"]))
    for entry in valid:
        eid = str(entry["id"])
        for alias in entry.get("aliases") or []:
            key = norm(alias)
            if not key:
                continue
            other = owner.get(key)
            if other is not None and other != eid:
                errors.append(_err("alias_conflict", eid, f"Alias '{alias}' of '{eid}' is also the id or an alias of '{other}'."))
            else:
                owner.setdefault(key, eid)
    return errors


def taxonomy_warnings(tax: Any) -> list[dict[str, Any]]:
    """Non-blocking notes about a taxonomy."""
    out: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    titles: dict[str, str] = {}
    for entry in (tax.get("entries") if isinstance(tax, dict) else None) or []:
        if not isinstance(entry, dict) or _as_id(entry.get("id")) is None:
            continue
        eid = str(entry["id"])
        if not str(entry.get("title") or "").strip() or str(entry.get("title")) == eid:
            out.append(_err("no_title", eid, f"'{eid}' has no title (the id is shown instead)."))
        for ns, refs in (entry.get("crosswalk") or {}).items() if isinstance(entry.get("crosswalk"), dict) else []:
            if isinstance(refs, list) and not refs:
                out.append(_err("namespace_only_ref", eid, f"'{eid}' references the whole namespace '{ns}' (no specific ids)."))
        title_key = norm(entry.get("title"))
        if title_key:
            if title_key in titles and titles[title_key] != eid:
                out.append(_err("duplicate_title", eid, f"'{eid}' and '{titles[title_key]}' have the same title; titles only resolve when unambiguous."))
            titles.setdefault(title_key, eid)
        key = norm(eid)
        if key in seen and seen[key] != eid:
            out.append(_err("ambiguous_id", eid, f"'{eid}' and '{seen[key]}' look the same once punctuation is ignored; an exact id always wins when resolving, but near-matches are ambiguous."))
        seen.setdefault(key, eid)
    return out


# ── Index + resolution ──────────────────────────────────────────────────


class TaxonomyIndex:
    """O(1) label resolution + tree helpers over one taxonomy.

    Resolution runs in tiers; the first tier with a hit decides:
    exact id -> casefolded id/alias (punctuation kept) -> id/alias with
    punctuation ignored -> casefolded title -> title with punctuation ignored.
    A key that maps to more than one entry within a tier is *ambiguous*:
    :meth:`resolve` returns ``(None, "ambiguous")`` and :meth:`candidates`
    lists the entries."""

    def __init__(self, taxonomy: dict[str, Any], version: str | None = None):
        self.taxonomy = taxonomy
        self.version = version if version is not None else taxonomy_version(taxonomy)
        self.entries: dict[str, dict[str, Any]] = {}
        self._exact: dict[str, str] = {}
        # tier -> key -> {entry id: matched_by}
        self._tiers: list[dict[str, dict[str, str]]] = [{}, {}, {}, {}]
        self._children: dict[str | None, list[str]] = {}
        for entry in taxonomy.get("entries") or []:
            eid = str(entry["id"])
            self.entries[eid] = entry
            self._exact.setdefault(eid, eid)
        fold = lambda v: str(v if v is not None else "").strip().casefold()  # noqa: E731

        def add(tier: int, key: str, eid: str, by: str) -> None:
            if key:
                slot = self._tiers[tier].setdefault(key, {})
                if slot.get(eid) != "id":
                    slot[eid] = by

        for eid, entry in self.entries.items():
            add(0, fold(eid), eid, "id")
            add(1, norm(eid), eid, "id")
            for alias in entry.get("aliases") or []:
                add(0, fold(alias), eid, "alias")
                add(1, norm(alias), eid, "alias")
            title = entry.get("title")
            add(2, fold(title), eid, "title")
            add(3, norm(title), eid, "title")
            parent = entry.get("parent")
            self._children.setdefault(str(parent) if parent else None, []).append(eid)

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    def _lookup(self, raw: Any) -> tuple[list[str], str | None]:
        text = str(raw if raw is not None else "").strip()
        if not text:
            return [], None
        if text in self._exact:
            return [text], "id"
        keys = [text.casefold(), norm(text), text.casefold(), norm(text)]
        for tier, key in enumerate(keys):
            slot = self._tiers[tier].get(key) if key else None
            if slot:
                by = next(iter(slot.values())) if len(slot) == 1 else None
                return list(slot), by
        return [], None

    def candidates(self, raw: Any) -> list[str]:
        """Entry ids ``raw`` could mean (more than one = ambiguous)."""
        return self._lookup(raw)[0]

    def resolve(self, raw: Any) -> tuple[str | None, str | None]:
        ids, by = self._lookup(raw)
        if not ids:
            return None, None
        if len(ids) > 1:
            return None, "ambiguous"
        return ids[0], by

    def title(self, label_id: str) -> str | None:
        entry = self.entries.get(label_id)
        return entry.get("title") if entry else None

    def ancestors(self, label_id: str) -> list[str]:
        out: list[str] = []
        cur = self.entries.get(label_id, {}).get("parent")
        while cur and str(cur) in self.entries and str(cur) not in out:
            out.append(str(cur))
            cur = self.entries[str(cur)].get("parent")
        return out

    def children(self, label_id: str) -> list[str]:
        return list(self._children.get(label_id, []))

    def roots(self) -> list[str]:
        return list(self._children.get(None, []))


def resolve_label(tax: Any, raw: Any) -> tuple[str | None, str | None]:
    """``(id, matched_by)`` for ``raw`` — ``matched_by`` is ``id`` / ``alias``
    / ``title`` — ``(None, "ambiguous")`` when several entries match, or ``(None, None)``. Build a :class:`TaxonomyIndex` once for
    many lookups."""
    index = tax if isinstance(tax, TaxonomyIndex) else TaxonomyIndex(tax)
    return index.resolve(raw)


# ── Versioning + storage ────────────────────────────────────────────────


def taxonomy_version(tax: dict[str, Any] | None) -> str | None:
    if not tax or not tax.get("entries"):
        return None
    payload = json.dumps(tax, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def namespaces(tax: dict[str, Any] | None) -> list[str]:
    found: set[str] = set()
    for entry in (tax or {}).get("entries") or []:
        found.update(str(ns) for ns in (entry.get("crosswalk") or {}))
    return sorted(found)


def project_taxonomy(project: Any) -> dict[str, Any] | None:
    cfg = getattr(project, "runtime_config", None)
    tax = cfg.get(TAXONOMY_KEY) if isinstance(cfg, dict) else None
    return tax if isinstance(tax, dict) and tax.get("entries") else None


def remember_taxonomy(project: Any) -> str | None:
    """Record the current taxonomy under its version in
    ``runtime_config[HISTORY_KEY]`` (on the object; the caller commits)."""
    tax = project_taxonomy(project)
    version = taxonomy_version(tax)
    if version is None:
        return None
    cfg = dict(project.runtime_config or {})
    history = dict(cfg.get(HISTORY_KEY) or {})
    history.pop(version, None)  # re-saving a version moves it to the newest slot
    history[version] = {
        "taxonomy": tax,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(tax["entries"]),
    }
    while len(history) > MAX_HISTORY:
        history.pop(next(iter(history)))
    # only the newest few keep a full snapshot; older ones are stubs
    for old in list(history)[:-KEEP_FULL_SNAPSHOTS]:
        snap = history[old]
        if isinstance(snap, dict) and "taxonomy" in snap:
            history[old] = {k: v for k, v in snap.items() if k != "taxonomy"}
    cfg[HISTORY_KEY] = history
    project.runtime_config = cfg
    return version


def set_project_taxonomy(project: Any, tax: dict[str, Any] | None) -> str | None:
    """Replace (or clear, with ``None``) the project's taxonomy, remembering
    the old and the new version. Returns the new version."""
    remember_taxonomy(project)
    cfg = dict(project.runtime_config or {})
    if tax is None:
        cfg.pop(TAXONOMY_KEY, None)
    else:
        cfg[TAXONOMY_KEY] = tax
    project.runtime_config = cfg
    return remember_taxonomy(project)


def taxonomy_status(project: Any) -> dict[str, Any]:
    """The API's GET shape."""
    tax = project_taxonomy(project)
    version = taxonomy_version(tax)
    cfg = project.runtime_config if isinstance(getattr(project, "runtime_config", None), dict) else {}
    history = cfg.get(HISTORY_KEY) or {}
    return {
        "taxonomy": tax,
        "version": version,
        "entry_count": len(tax["entries"]) if tax else 0,
        "namespaces": namespaces(tax),
        "history": [
            {"version": v, "entry_count": int(h.get("entry_count") or 0), "saved_at": h.get("saved_at"), "current": v == version, "restorable": isinstance(h.get("taxonomy"), dict)}
            for v, h in reversed(list(history.items()))
            if isinstance(h, dict)
        ],
    }


def restore_version(project: Any, version: str) -> str:
    """``"ok"``, ``"unknown"`` (no such version) or ``"stub"`` (the version
    is listed but its snapshot is no longer kept)."""
    history = (project.runtime_config or {}).get(HISTORY_KEY) or {}
    snap = history.get(version)
    if not isinstance(snap, dict):
        return "unknown"
    if not isinstance(snap.get("taxonomy"), dict):
        return "stub"
    set_project_taxonomy(project, snap["taxonomy"])
    return "ok"


async def load_taxonomy(db: AsyncSession | None, project_id: int) -> tuple[dict[str, Any] | None, str | None]:
    """The project's taxonomy and its version; ``(None, None)`` when it has
    none (or without a session)."""
    if db is None:
        return None, None
    from app.models.project import Project

    project = await db.get(Project, project_id)
    tax = project_taxonomy(project) if project is not None else None
    return tax, taxonomy_version(tax)


async def load_taxonomy_index(db: AsyncSession | None, project_id: int) -> TaxonomyIndex | None:
    tax, version = await load_taxonomy(db, project_id)
    return TaxonomyIndex(tax, version) if tax else None


def taxonomy_summary(index: TaxonomyIndex | None) -> dict[str, Any] | None:
    """``{version, entry_count}`` for preview / run results."""
    return {"version": index.version, "entry_count": index.entry_count} if index else None
