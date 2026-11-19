"""Per-row task context: optional context blocks a single row carries for itself.

Any row (train / val / test / gold) may hold one reserved key::

    "row_context": {
      "system": "...",                 # a system message for this row
      "tools": [ {name, description, inputSchema|input_schema, annotations?} ],
      "output_schema": { ... },        # reserved: per-row JSON Schema (validated, not consumed yet)
      "labels": ["a", "b"],            # reserved: per-row label set (validated, not consumed yet)
      "schema_text": "CREATE TABLE ..."  # reserved: free-text context (validated, not consumed yet)
    }

The container is task-agnostic: each kind declares which task profiles consume it (``CONSUMERS``); a
kind a profile does not consume stays on the row and is reported as "not used by this task". A row
WITHOUT ``row_context`` goes through exactly today's code everywhere (rendering is byte-identical).

Rendering is shared by training and eval so the two cannot drift:

- ``system``: a system turn. With a chat template both sides render ``[system, user]``; a prompt that is
  not chat-wrapped (a handler that builds its own complete prompt, or a tokenizer with no chat template)
  becomes ``render_system_plain(system, prompt)``. ``scripts/train.py`` (standalone) carries a copy of
  ``render_system_plain``; ``tests/test_row_context_service.py`` pins that they agree. The legacy
  ``system_prompt`` field (passage prompts) is read as the same kind, with ``row_context.system`` winning,
  except where training never rendered it (``eval_system``): ahead of a prompt the handler builds itself,
  and in a multi-turn chat. A multi-turn chat row (one that trains on the chat path) takes
  ``row_context.system`` as its leading system turn (``with_leading_system``, replacing a system turn its
  messages already have), in training and eval alike; a single user->assistant exchange stays on the
  prompt/completion path and renders it as ``[system, user]``.
- ``tools`` (consumer: ``tool_calling``): the row's own tool list, rendered by
  ``tool_catalog_service.render_tool_block`` over a per-row catalog view, so a per-row list and the
  project catalog render in the same format. The project's tool flags still apply by tool name (a tool
  the project marks not-allowed stays not-allowed; a destructive one stays destructive): a row can never
  lift a safety flag. Tool-specific by necessity (tool lists have their own schema format and the tool
  scorer is the only consumer of "allowed").
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROW_CONTEXT_KEY = "row_context"
SYSTEM_ALIAS = "system_prompt"
KINDS: tuple[str, ...] = ("system", "tools", "output_schema", "labels", "schema_text")
# Which task profiles read each kind (``"*"`` = every profile). Kinds with no consumer yet are validated
# and kept, and shown as "not used by this task".
CONSUMERS: dict[str, tuple[str, ...]] = {
    "system": ("*",),
    "tools": ("tool_calling",),
    "output_schema": (),
    "labels": (),
    "schema_text": (),
}

# Bumped when the held-out eval prompt changes for some rows. Stamped on every new held-out eval as
# ``metrics.prompt_format_version``; results without it are version 1.
#   2 (2026-11-19): a row's system message (``row_context.system`` or ``system_prompt``) is rendered in
#     eval exactly as training renders it (before, eval dropped it: a train/eval mismatch).
EVAL_PROMPT_FORMAT_VERSION = 2

REJECT_REASONS: tuple[str, ...] = (
    "row_context_invalid",
    "row_context_unknown_kind",
    "row_context_system_invalid",
    "row_context_tools_invalid",
    "row_context_schema_invalid",
    "row_context_labels_invalid",
    "row_context_schema_text_invalid",
)

MAX_SYSTEM_CHARS = 32_000
MAX_SCHEMA_TEXT_CHARS = 64_000
MAX_LABELS = 1000
VIEW_SYSTEM_CHARS = 4000
VIEW_MAX_TOOLS = 100


def digest(value: Any) -> str:
    """sha256[:12] of the canonical JSON of ``value``."""
    canon = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def consumes(kind: str, profile: str | None) -> bool:
    who = CONSUMERS.get(kind, ())
    return "*" in who or (profile or "") in who


def kinds_read_by(profile: str | None) -> list[str]:
    """The ``row_context`` kinds a task profile reads (the one rule behind every "not used by this task" label)."""
    return [k for k in KINDS if consumes(k, profile)]


async def effective_task_profile(db: Any, project_id: int) -> str | None:
    """The task profile the evaluator would use for this project: the prepared manifest's, else the latest
    experiment's task type (as ``build_eval_context``), else the selected recipe's. ``None`` if unknown."""
    from sqlalchemy import select

    from app.models.experiment import Experiment
    from app.models.project import Project
    from app.services.eval_task_handler_service import read_task_profile_from_manifest

    profile = read_task_profile_from_manifest(project_id)
    if profile:
        return profile.strip().lower()
    exp = (await db.execute(
        select(Experiment).where(Experiment.project_id == project_id).order_by(Experiment.id.desc()).limit(1)
    )).scalar_one_or_none()
    task_type = str(((getattr(exp, "config", None) or {}).get("task_type")) or "").strip().lower()
    if task_type:
        return "language_modeling" if task_type == "causal_lm" else task_type
    project = await db.get(Project, project_id)
    selected = getattr(project, "selected_recipe", None)
    recipe_id = selected.get("recipe_id") if isinstance(selected, dict) else None
    if isinstance(recipe_id, str) and recipe_id:
        try:
            from app.services.recipe_service import get_recipe

            recipe = get_recipe(recipe_id)
            value = getattr(recipe, "task_profile", None) if recipe is not None else None
            if value:
                return str(value).strip().lower()
        except Exception:  # noqa: BLE001 - a label must not fail the listing
            return None
    return None


# ── normalize / validate ───────────────────────────────────────────────


# ── prepared-file encoding ─────────────────────────────────────────────
# In PREPARED splits (train / val / test) ``row_context`` is stored as a canonical JSON string, not an
# object: ``scripts/train.py`` loads splits with ``datasets.load_dataset("json")``, whose Arrow schema
# inference fails on (or silently null-fills) per-row objects with different shapes - e.g. tool lists
# whose input schemas differ row to row. A string column always loads. Every reader goes through
# ``decode`` (via ``_ctx``), which accepts both the string and an object (gold files, raw uploads, older
# prepared files). Rows without ``row_context`` are untouched.

_DECODE_CACHE: "OrderedDict[str, dict[str, Any] | None]" = OrderedDict()


def encode(normalized: dict[str, Any]) -> str:
    """The prepared-file form of a normalised ``row_context``: canonical JSON text."""
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def decode(raw: Any) -> dict[str, Any] | None:
    """A ``row_context`` as an object: the object itself, or a JSON-object string parsed (cached; callers
    must not mutate the result). None for anything else."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    if raw in _DECODE_CACHE:
        _DECODE_CACHE.move_to_end(raw)
        return _DECODE_CACHE[raw]
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    value = value if isinstance(value, dict) else None
    _DECODE_CACHE[raw] = value
    while len(_DECODE_CACHE) > 4096:
        _DECODE_CACHE.popitem(last=False)
    return value


def normalize(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """``(normalized row_context, None)`` or ``(None, reject reason)``. ``None`` / ``{}`` is no context:
    ``(None, None)``. Empty ``system`` / ``schema_text`` strings are dropped; an empty ``tools`` list is
    kept (the row offers no tools). A prepared-file string (``encode``) is read as its object."""
    if raw is None:
        return None, None
    if isinstance(raw, str):
        decoded = decode(raw)
        if decoded is None:
            return None, "row_context_invalid"
        raw = decoded
    if not isinstance(raw, dict):
        return None, "row_context_invalid"
    unknown = [k for k in raw if k not in KINDS]
    if unknown:
        return None, "row_context_unknown_kind"
    out: dict[str, Any] = {}
    if "system" in raw and raw["system"] is not None:
        value = raw["system"]
        if not isinstance(value, str) or len(value) > MAX_SYSTEM_CHARS:
            return None, "row_context_system_invalid"
        if value.strip():
            out["system"] = value.strip()
    if "tools" in raw and raw["tools"] is not None:
        tools = normalize_tools(raw["tools"])
        if tools is None:
            return None, "row_context_tools_invalid"
        out["tools"] = tools
    if "output_schema" in raw and raw["output_schema"] is not None:
        schema = raw["output_schema"]
        if not isinstance(schema, dict) or not _schema_usable(schema):
            return None, "row_context_schema_invalid"
        out["output_schema"] = schema
    if "labels" in raw and raw["labels"] is not None:
        labels = raw["labels"]
        if not isinstance(labels, list) or len(labels) > MAX_LABELS or not all(
            isinstance(x, str) and x.strip() for x in labels
        ):
            return None, "row_context_labels_invalid"
        out["labels"] = list(dict.fromkeys(x.strip() for x in labels))
    if "schema_text" in raw and raw["schema_text"] is not None:
        value = raw["schema_text"]
        if not isinstance(value, str) or len(value) > MAX_SCHEMA_TEXT_CHARS:
            return None, "row_context_schema_text_invalid"
        if value.strip():
            out["schema_text"] = value
    return (out or None), None


def _schema_usable(schema: dict[str, Any]) -> bool:
    from app.services import json_schema_bundle as jsb

    try:
        jsb.validate_bundle({"row.json": schema}, "row.json")
    except Exception:  # noqa: BLE001
        return False
    return True


_TOOLS_CACHE: "OrderedDict[str, list[dict[str, Any]] | None]" = OrderedDict()


def normalize_tools(raw_tools: Any) -> list[dict[str, Any]] | None:
    """A row's tool list in the catalog's normalised form (``tool_catalog_service.normalize_tools``,
    strict: a duplicate / nameless tool or an unusable input schema makes the list invalid). ``None`` when
    invalid. Cached by content."""
    if not isinstance(raw_tools, list):
        return None
    key = digest(raw_tools)
    if key in _TOOLS_CACHE:
        _TOOLS_CACHE.move_to_end(key)
        return _TOOLS_CACHE[key]
    from app.services import tool_catalog_service as tcat

    try:
        tools, _problems = tcat.normalize_tools(raw_tools, strict=True)
        if any(t.get("schema_error") for t in tools):
            tools = None
    except tcat.ToolCatalogError:
        tools = None
    _TOOLS_CACHE[key] = tools
    while len(_TOOLS_CACHE) > 2048:
        _TOOLS_CACHE.popitem(last=False)
    return tools


def validate_row(row: Any) -> tuple[dict[str, Any] | None, str | None]:
    """The row's normalised ``row_context`` (None when it has none) or a reject reason."""
    if not isinstance(row, dict) or ROW_CONTEXT_KEY not in row:
        return None, None
    return normalize(row.get(ROW_CONTEXT_KEY))


def apply_to_prepared_row(merged: dict[str, Any], normalized: dict[str, Any] | None) -> None:
    """Training side, on a row that HAS a valid ``row_context``: store the normalised context (as its
    ``encode``d JSON string: see "prepared-file encoding") and expose its system message as
    ``system_prompt`` (what ``scripts/train.py`` renders as the system turn). Rows without
    ``row_context`` are never touched."""
    if normalized is None:
        return
    merged[ROW_CONTEXT_KEY] = encode(normalized)
    if normalized.get("system"):
        merged[SYSTEM_ALIAS] = normalized["system"]


# ── readers ──────────────────────────────────────────────────────────────


def _ctx(row: Any) -> dict[str, Any]:
    rc = decode(row.get(ROW_CONTEXT_KEY)) if isinstance(row, dict) else None
    return rc if isinstance(rc, dict) else {}


def row_system(row: Any) -> tuple[str | None, str | None]:
    """``(system message, field it came from)``: ``row_context.system`` first, then the legacy
    ``system_prompt`` field (passage prompts). ``(None, None)`` when the row has neither."""
    value = _ctx(row).get("system")
    if isinstance(value, str) and value.strip():
        return value.strip(), "row_context.system"
    legacy = row.get(SYSTEM_ALIAS) if isinstance(row, dict) else None
    if isinstance(legacy, str) and legacy.strip():
        return legacy.strip(), SYSTEM_ALIAS
    return None, None


def row_context_system(row: Any) -> str | None:
    """The row's ``row_context.system`` only (not the legacy ``system_prompt`` field)."""
    value = _ctx(row).get("system")
    return value.strip() if isinstance(value, str) and value.strip() else None


def eval_system(row: Any, *, wraps_own_prompt: bool) -> tuple[str | None, str | None]:
    """``(system message, field)`` a prompt/completion eval pair renders for ``row`` - exactly what
    training renders for it. ``row_context.system`` always; the legacy ``system_prompt`` field only on a
    chat-wrapped prompt: training never rendered it ahead of a prompt the handler builds itself
    (``scripts/train.py`` passes those rows through), so eval does not either."""
    system, field = row_system(row)
    if field == SYSTEM_ALIAS and wraps_own_prompt:
        return None, None
    return system, field


def with_leading_system(messages: list[dict[str, str]], system: str) -> list[dict[str, str]]:
    """A multi-turn chat with ``system`` as its leading system turn (replacing a leading system turn the
    messages already have: ``row_context.system`` wins). ``scripts/train.py`` has a pinned copy."""
    rest = [dict(m) for m in messages]
    if rest and rest[0].get("role") == "system":
        rest = rest[1:]
    return [{"role": "system", "content": system}, *rest]


def row_tools(row: Any) -> list[dict[str, Any]] | None:
    """The row's own (normalised) tool list, ``[]`` for an explicit empty list, None when it has none
    (or an invalid one)."""
    raw = _ctx(row).get("tools")
    if raw is None:
        return None
    return normalize_tools(raw)


def kinds_present(row: Any) -> list[str]:
    return [k for k in KINDS if _ctx(row).get(k) not in (None, "")]


# ── rendering (shared by training and eval) ──────────────────────────────


def render_system_plain(system: str, prompt: str) -> str:
    """A system message ahead of a prompt that is not chat-wrapped. ``scripts/train.py`` has a pinned copy."""
    return f"{system}\n\n{prompt}"


def system_messages(system: str, prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def project_tool_overrides(runtime_config: Any) -> dict[str, dict[str, Any]]:
    """The project catalog's safety flags by tool name: ``allowed: False`` (the user turned the tool off),
    ``read_only`` (the user's override) and ``destructive: True`` (the catalog says so). Applied to a
    row's own list so a row cannot lift a project flag."""
    from app.services import tool_catalog_service as tcat

    state = tcat.raw_state(runtime_config)
    if state is None:
        return {}
    flags = state.get("flags") if isinstance(state.get("flags"), dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for tool in tcat.effective_tools(runtime_config):
        entry: dict[str, Any] = {}
        if not tool.get("allowed", True):
            entry["allowed"] = False
        flag = flags.get(tool["name"]) if isinstance(flags.get(tool["name"]), dict) else {}
        if isinstance(flag.get("read_only"), bool):
            entry["read_only"] = flag["read_only"]
        if tcat.destructive_effective(tool):
            entry["destructive"] = True
        if entry:
            out[tool["name"]] = entry
    return out


_VIEW_CACHE: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()


def tool_view_config(tools: list[dict[str, Any]], overrides: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    """A runtime-config-shaped dict whose ``tool_catalog`` is the row's list (with the project's flags
    applied by name), so every ``tool_catalog_service`` reader (render, allowed, schema bundle) works on
    it unchanged. Cached by (tools, overrides)."""
    overrides = overrides or {}
    key = (digest(tools), digest(overrides))
    if key in _VIEW_CACHE:
        _VIEW_CACHE.move_to_end(key)
        return _VIEW_CACHE[key]
    from app.services import tool_catalog_service as tcat

    view_tools = []
    flags: dict[str, dict[str, Any]] = {}
    for tool in tools:
        t = dict(tool)
        o = overrides.get(t.get("name")) or {}
        if o.get("destructive"):
            t["destructive"] = True
        view_tools.append(t)
        flag = {k: o[k] for k in ("allowed", "read_only") if k in o}
        if flag:
            flags[t["name"]] = flag
    state = tcat._new_state(view_tools, {"kind": "row"}, None, flags, "")
    view = {tcat.CATALOG_KEY: state}
    _VIEW_CACHE[key] = view
    while len(_VIEW_CACHE) > 512:
        _VIEW_CACHE.popitem(last=False)
    return view


def render_row_tool_block(tools: list[dict[str, Any]], overrides: dict[str, dict[str, Any]] | None = None) -> str:
    """The row's tool block, same format as the project block (``render_tool_block``)."""
    from app.services import tool_catalog_service as tcat

    if not tools:
        return "Available tools (call only these):\n- (no tools are allowed)"
    return tcat.render_tool_block(tool_view_config(tools, overrides))


# ── eval provenance ──────────────────────────────────────────────────────


_UNSET: Any = object()


def eval_view(
    row: Any,
    *,
    profile: str | None,
    overrides: dict[str, dict[str, Any]] | None = None,
    system_used: Any = _UNSET,
) -> dict[str, Any] | None:
    """What the drill-down shows for one eval row with row-level context (None without any): the system
    message (and the field it came from), the row's tools as compact signatures, the kinds this task does
    not use, and a digest of the whole context. ``system_used`` is the ``(system, field)`` the eval
    actually rendered (default: ``row_system(row)``), so the view never claims a system message the
    prompt did not carry."""
    system, system_field = row_system(row) if system_used is _UNSET else (system_used or (None, None))
    present = kinds_present(row)
    if not system and not present:
        return None
    view: dict[str, Any] = {}
    if system:
        view["system"] = system[:VIEW_SYSTEM_CHARS]
        view["system_source"] = system_field
        if len(system) > VIEW_SYSTEM_CHARS:
            view["system_truncated"] = True
    tools = row_tools(row)
    if tools is not None and consumes("tools", profile):
        from app.services import tool_catalog_service as tcat

        shown = tools[:VIEW_MAX_TOOLS]
        cfg = tool_view_config(shown, overrides) if shown else None
        eff = {t["name"]: t for t in tcat.effective_tools(cfg)} if cfg else {}
        view["tools"] = [
            {"name": t["name"], "signature": tcat.compact_signature(eff.get(t["name"], t)),
             "allowed": bool(eff.get(t["name"], {}).get("allowed", True))}
            for t in shown
        ]
        view["tools_count"] = len(tools)
        view["tools_source"] = "row"
    unused = [k for k in present if k != "system" and not consumes(k, profile)]
    if unused:
        view["unused_kinds"] = unused
    view["digest"] = digest({"system": system, **{k: _ctx(row).get(k) for k in present if k != "system"}})
    return view


def eval_metrics_block(predictions: list[dict[str, Any]], *, profile: str | None, project_has_catalog: bool) -> dict[str, Any] | None:
    """``metrics.row_context`` for an eval whose rows carry row-level context (None otherwise, so other
    evals' metrics are unchanged): rows with context, per-kind source counts, and one digest over every
    row's context."""
    views = [p.get("row_context_view") for p in predictions if isinstance(p, dict)]
    with_ctx = [v for v in views if isinstance(v, dict)]
    if not with_ctx:
        return None
    kinds: dict[str, dict[str, int]] = {"system": {"row": 0, "none": 0}}
    unused: dict[str, int] = {}
    for v in views:
        v = v if isinstance(v, dict) else {}
        kinds["system"]["row" if v.get("system") else "none"] += 1
        for k in v.get("unused_kinds") or []:
            unused[k] = unused.get(k, 0) + 1
    if consumes("tools", profile):
        t = {"row": 0, "project": 0, "none": 0}
        for v in views:
            if isinstance(v, dict) and v.get("tools_source") == "row":
                t["row"] += 1
            else:
                t["project" if project_has_catalog else "none"] += 1
        kinds["tools"] = t
    out: dict[str, Any] = {
        "rows": len(views),
        "rows_with_context": len(with_ctx),
        "kinds": kinds,
        "digest": digest(sorted(v.get("digest") or "" for v in with_ctx)),
    }
    if unused:
        out["unused_kinds"] = unused
    return out


def split_has_row_tools(path: Any, *, max_lines: int | None = None) -> bool:
    """Whether any row of a prepared JSONL split carries its own tool list (cheap substring pre-check)."""
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if max_lines is not None and i >= max_lines:
                    break
                if '"row_context"' not in line or '"tools"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row_tools(row) is not None:
                    return True
    except OSError:
        return False
    return False
