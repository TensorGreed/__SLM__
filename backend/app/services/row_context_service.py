"""Per-row task context: optional context blocks a single row carries for itself.

Any row (train / val / test / gold) may hold one reserved key::

    "row_context": {
      "system": "...",                 # a system message for this row
      "tools": [ {name, description, inputSchema|input_schema, annotations?} ],
      "output_schema": { ... },        # a JSON Schema for this row's expected JSON output
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
- ``output_schema`` (consumers: the eval handlers that score a JSON answer - ``structured_extraction`` and
  the ``qa`` handler, which serves qa / instruction_sft / chat_sft / language_modeling, whose answers may
  be a JSON object): the row's own JSON Schema, rendered as ``output_schema_block`` (one instruction line
  + the schema as canonical one-line JSON). Extraction rows: ``render_extraction_prompt`` (field list from
  the row schema's properties) is the ONE builder the ``structured-extraction`` adapter and the eval
  handler call. The ``qa`` handler's rows (``output_schema_mode == "lead"``):
  ``render_output_schema_lead`` puts the block ahead of the bare prompt (block + blank line + prompt, the
  tool block's shape); training keeps the bare prompt as
  ``user_prompt`` (as the tool path does) and eval re-renders from it, so both sides show the same text.
  Scoring: the prediction is validated against the row's schema (``validate_output``,
  ``metrics.output_schema``) and extraction field match takes the row's fields / required / types. Tool
  calling does not read it (each tool's own input schema already validates the arguments).
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
    "output_schema": ("structured_extraction", "qa"),
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
MAX_OUTPUT_SCHEMA_CHARS = 64_000
VIEW_SCHEMA_CHARS = 4000
MAX_SCHEMA_ERRORS = 5
MAX_SCHEMA_TEXT_CHARS = 64_000
MAX_LABELS = 1000
VIEW_SYSTEM_CHARS = 4000
VIEW_MAX_TOOLS = 100


def digest(value: Any) -> str:
    """sha256[:12] of the canonical JSON of ``value``."""
    canon = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def _ordered_key(value: Any) -> str:
    """Cache key of ``value`` that keeps key order: two tool lists that differ only in the order of a
    schema's ``properties`` render different signatures, so they must not share a cache entry."""
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def consumes(kind: str, profile: str | None) -> bool:
    who = CONSUMERS.get(kind, ())
    return "*" in who or (profile or "") in who


def kinds_read_by(profile: str | None) -> list[str]:
    """The ``row_context`` kinds a task profile reads (the one rule behind every "not used by this task" label)."""
    return [k for k in KINDS if consumes(k, profile)]


def reads_report(profile: str | None) -> dict[str, Any]:
    """What the answer-key / eval-set row lists tell the UI: the effective task profile, the kinds it reads
    and how it renders a row's output schema (``output_schema_mode``: ``"extraction"`` = every row,
    ``"lead"`` = prompt/completion rows only, not chat rows, None = not read)."""
    return {
        "effective_task_profile": profile,
        "row_context_reads": kinds_read_by(profile),
        "row_context_schema_mode": output_schema_mode(profile),
    }


async def effective_task_profile(db: Any, project_id: int) -> str | None:
    """The task profile the evaluator would use for this project: the prepared manifest's, else the latest
    experiment's task type (as ``build_eval_context``), else the project's saved dataset preset's (what the
    next prepare writes), else the selected recipe's. ``None`` if unknown."""
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
    preset = getattr(project, "dataset_adapter_preset", None)
    preset_profile = preset.get("task_profile") if isinstance(preset, dict) else None
    if isinstance(preset_profile, str) and preset_profile.strip() and preset_profile.strip().lower() != "auto":
        return preset_profile.strip().lower()
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
# In PREPARED splits (train / val / test) ``row_context`` is stored as a compact JSON string, not an
# object: ``scripts/train.py`` loads splits with ``datasets.load_dataset("json")``, whose Arrow schema
# inference fails on (or silently null-fills) per-row objects with different shapes - e.g. tool lists
# whose input schemas differ row to row. A string column always loads. Every reader goes through
# ``decode`` (via ``_ctx``), which accepts both the string and an object (gold files, raw uploads, older
# prepared files). Rows without ``row_context`` are untouched.

_DECODE_CACHE: "OrderedDict[str, dict[str, Any] | None]" = OrderedDict()


def encode(normalized: dict[str, Any]) -> str:
    """The prepared-file form of a normalised ``row_context``: compact JSON text in the row's own key
    order. The order is kept because the rendered prompt follows it (a tool's parameters are listed in its
    schema's ``properties`` order): eval re-renders from this text, so it must list them as training did.
    Splits prepared before 2026-11-19 stored it with sorted keys (see ``prepared_row_tool_order``)."""
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


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
        if not isinstance(schema, dict) or len(canonical_schema(schema)) > MAX_OUTPUT_SCHEMA_CHARS or not _schema_usable(schema):
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
        bundle = jsb.validate_bundle({"row.json": schema}, "row.json")
    except Exception:  # noqa: BLE001
        return False
    # A self-recursive / cyclic ``$ref`` that consumes no data (``{"$ref": "#"}``, a -> b -> a) passes the
    # structural check and then fails every row at eval ("the schema could not be evaluated"). Probe one
    # value of each JSON kind: a usable schema never answers that.
    try:
        for probe in ({}, [], "x", 1, True, None):
            if any("could not be evaluated" in str(e.get("message")) for e in bundle.validate(probe)):
                return False
    except RecursionError:
        return False
    return True


_TOOLS_CACHE: "OrderedDict[str, list[dict[str, Any]] | None]" = OrderedDict()


def normalize_tools(raw_tools: Any) -> list[dict[str, Any]] | None:
    """A row's tool list in the catalog's normalised form (``tool_catalog_service.normalize_tools``,
    strict: a duplicate / nameless tool or an unusable input schema makes the list invalid), in the ROW's
    order (the project catalog sorts by name; a row's list is rendered as the row gives it, so imported
    prompts reproduce byte for byte). ``None`` when invalid. Cached by content (order included)."""
    if not isinstance(raw_tools, list):
        return None
    key = _ordered_key(raw_tools)
    if key in _TOOLS_CACHE:
        _TOOLS_CACHE.move_to_end(key)
        return _TOOLS_CACHE[key]
    from app.services import tool_catalog_service as tcat

    try:
        tools, _problems = tcat.normalize_tools(raw_tools, strict=True, keep_order=True)
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
    key = (_ordered_key(tools), digest(overrides))
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


ROW_TOOL_ORDER_KEY = "row_tool_order"
# How a prepared split's per-row tool lists were stored and rendered (the manifest's ``row_tool_order``):
#   "kept" (2026-11-19 on): tools in the row's order, each schema's keys (its parameters) in the row's
#          order too -- eval re-renders the stored list and gets the training text.
#   "row"  (2026-11-19, earlier the same day) and "name" (before): the stored lists had their keys sorted
#          (``encode`` used ``sort_keys``) while training rendered the uploaded rows, parameters in schema
#          order, and ("name") tools sorted by name. The parameter order is lost from the stored list, so
#          eval uses the row's stored training prompt (``trained_tool_prompt``) instead.
ROW_TOOL_ORDERS = ("kept", "row", "name")
LEGACY_ROW_TOOL_ORDERS = ("row", "name")


def render_row_tool_block(
    tools: list[dict[str, Any]], overrides: dict[str, dict[str, Any]] | None = None, *, order: str = "kept",
) -> str:
    """The row's tool block, same line format as the project block (``render_tool_block``). ``order``:
    ``"kept"`` / ``"row"`` (default) list the tools in the row's own order; ``"name"`` sorts them by name,
    which is how splits prepared before 2026-11-19 rendered them -- see ``prepared_row_tool_order``. Each
    tool's parameters are listed in its stored schema order."""
    from app.services import tool_catalog_service as tcat

    if not tools:
        return "Available tools (call only these):\n- (no tools are allowed)"
    if order == "name":
        tools = sorted(tools, key=lambda t: str(t.get("name")))
    return tcat.render_tool_block(tool_view_config(tools, overrides), keep_order=True)


def trained_tool_prompt(row: Any) -> str | None:
    """A prepared row's own training prompt when it is a tool-block prompt (``question`` = tool block +
    blank line + ``user_prompt``, as the ``tool-call-json`` adapter writes it); None otherwise. Eval uses it
    for splits whose stored tool lists lost their parameter order (``LEGACY_ROW_TOOL_ORDERS``), so the
    prompt is the one the model was trained on, byte for byte."""
    if not isinstance(row, dict):
        return None
    question, user_prompt = row.get("question"), row.get("user_prompt")
    if not isinstance(question, str) or not isinstance(user_prompt, str) or not user_prompt:
        return None
    if not question.startswith("Available tools") or not question.endswith("\n\n" + user_prompt):
        return None
    return question


def tool_list_digest(tools: list[dict[str, Any]]) -> str:
    """Digest of a tool list as a SET (name order ignored): the same tools offered in another order are
    one distinct list in the per-row counts."""
    return digest(sorted((t for t in tools if isinstance(t, dict)), key=lambda t: str(t.get("name"))))


def prepared_row_tool_order(manifest: dict[str, Any] | None, prepared_dir: Any) -> str:
    """How this project's prepared split stored and rendered per-row tool lists (``ROW_TOOL_ORDERS``), so
    eval prompts match training: the prepared manifest's ``row_tool_order`` (stamped by a prepare whose
    split carries per-row lists); without it, ``"name"`` when the prepared train split has per-row lists
    (prepared before the row order was kept) and ``"kept"`` otherwise (no training render to match)."""
    value = (manifest or {}).get(ROW_TOOL_ORDER_KEY) if isinstance(manifest, dict) else None
    if value in ROW_TOOL_ORDERS:
        return str(value)
    if prepared_dir is not None and split_has_row_tools(Path(prepared_dir) / "train.jsonl"):
        return "name"
    return "kept"


def stamp_row_tool_order(manifest: dict[str, Any], prepared_dir: Any) -> None:
    """Prepare side: record ``row_tool_order: "kept"`` when a written split carries per-row tool lists
    (other manifests are unchanged)."""
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        if split_has_row_tools(Path(prepared_dir) / name):
            manifest[ROW_TOOL_ORDER_KEY] = "kept"
            return


# ── output schema (rendering + validation; shared by training and eval) ──

OUTPUT_SCHEMA_INSTRUCTION = "Reply with a single JSON object that matches this JSON Schema, nothing else."
OUTPUT_SCHEMA_PREFIX = "JSON Schema: "
# Large row schemas eat the context window: import warns (does not reject) above this many tokens (estimate).
SCHEMA_TOKEN_WARN = 4000
CHARS_PER_TOKEN = 4


def schema_root(schema: dict[str, Any]) -> str:
    """The kind of JSON value the schema describes at its root: ``"object"`` (``type: object``, or no type
    but ``properties``), ``"array"`` (``type: array``) or ``"value"`` (anything else: a union, a scalar, a
    ``$ref`` / ``anyOf`` root, no type)."""
    t = schema.get("type")
    if t == "object" or (t is None and isinstance(schema.get("properties"), dict)):
        return "object"
    if t == "array":
        return "array"
    return "value"


def output_schema_instruction(schema: dict[str, Any]) -> str:
    """The instruction line for the schema's root type; the object wording is ``OUTPUT_SCHEMA_INSTRUCTION``."""
    kind = schema_root(schema)
    if kind == "object":
        return OUTPUT_SCHEMA_INSTRUCTION
    return OUTPUT_SCHEMA_INSTRUCTION.replace("a single JSON object", f"a single JSON {kind}")


def schema_token_estimate(schema: dict[str, Any]) -> int:
    """Rough token count of the schema as it appears in the prompt (characters / 4)."""
    return -(-len(canonical_schema(schema)) // CHARS_PER_TOKEN)


def bare_question(row: Any, schema: dict[str, Any], fallback: str) -> str:
    """The question behind a row's schema lead: a prepared row's ``question`` with the block taken off
    (so training and eval show the same text whatever else the row carries), else ``fallback``."""
    q = row.get("question") if isinstance(row, dict) else None
    lead = output_schema_block(schema) + "\n\n"
    if isinstance(q, str) and q.startswith(lead) and q[len(lead):].strip():
        return q[len(lead):]
    return fallback


def row_output_schema(row: Any) -> dict[str, Any] | None:
    """The row's own ``row_context.output_schema`` (None when it has none)."""
    value = _ctx(row).get("output_schema")
    return value if isinstance(value, dict) and value else None


def canonical_schema(schema: dict[str, Any]) -> str:
    """The schema as canonical one-line JSON (sorted keys): what the prompt shows."""
    return json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(", ", ": "), default=str)


def output_schema_block(schema: dict[str, Any]) -> str:
    """The rendered schema block: one instruction line + ``JSON Schema: <canonical JSON>``."""
    return f"{output_schema_instruction(schema)}\n{OUTPUT_SCHEMA_PREFIX}{canonical_schema(schema)}"


def output_schema_lines(schema: dict[str, Any]) -> list[str]:
    """The block's lines (data health ignores them as context, not content)."""
    return [ln.strip() for ln in output_schema_block(schema).splitlines() if ln.strip()]


def schema_fields(schema: dict[str, Any]) -> list[str]:
    """The schema's top-level property names, sorted (the extraction field list)."""
    props = schema.get("properties")
    return sorted(str(k) for k in props) if isinstance(props, dict) and schema_root(schema) == "object" else []


def render_extraction_prompt(input_text: str, schema: dict[str, Any]) -> str:
    """The structured-extraction prompt of a row with its own schema: the field list from the schema, the
    schema block, the input. Called by the ``structured-extraction`` adapter (training) and
    ``StructuredExtractionHandler`` (eval), so the two are byte-identical."""
    fields = schema_fields(schema)
    root = schema_root(schema)
    if fields:
        head = f"Extract the following fields as JSON: {', '.join(fields)}.\n"
    elif root == "object":
        head = "Extract the relevant fields from the input as a single JSON object.\n"
    else:
        head = f"Extract the relevant data from the input as a single JSON {root}.\n"
    return f"{head}{output_schema_block(schema)}\nInput: {input_text}\nOutput:"


def render_output_schema_lead(prompt: str, schema: dict[str, Any]) -> str:
    """A chat-wrapped prompt (the ``qa`` handler's rows) led by the row's schema block: block + blank line +
    prompt, the shape of the tool block, so data health strips it as a context block."""
    return f"{output_schema_block(schema)}\n\n{prompt}"


def output_schema_mode(profile: str | None) -> str | None:
    """How a task profile renders a row's ``output_schema``: ``"extraction"`` (the extraction prompt),
    ``"lead"`` (ahead of the bare prompt) or None (not read). Resolved through the eval handler registry, so
    training (which knows the task profile) and eval (which knows the handler) decide the same way."""
    if not profile:
        return None
    from app.services.eval_task_handler_service import resolve_task_handler

    handler_profile = getattr(resolve_task_handler(profile), "profile_id", None)
    if not consumes("output_schema", handler_profile):
        return None
    return "extraction" if handler_profile == "structured_extraction" else "lead"


def lead_applies(row: Any) -> bool:
    """A ``lead``-mode row renders its schema only on the prompt/completion path: not a chat row
    (``messages`` / ``conversations``), whose prompt is its own history."""
    return isinstance(row, dict) and not isinstance(row.get("messages"), list) and not isinstance(
        row.get("conversations"), list
    )


def apply_output_schema_to_prepared_row(
    merged: dict[str, Any], normalized: dict[str, Any] | None, profile: str | None
) -> None:
    """Training side, ``lead`` mode: a row with its own schema gets the block ahead of its question; the
    bare question is kept as ``user_prompt`` (eval re-renders from it) and an adapter ``source_text`` equal
    to the question follows it. Rows without a schema, other modes and chat rows are never touched."""
    schema = (normalized or {}).get("output_schema")
    if not isinstance(schema, dict) or output_schema_mode(profile) != "lead" or not lead_applies(merged):
        return
    question = merged.get("question")
    if not isinstance(question, str) or not question.strip():
        return
    rendered = render_output_schema_lead(question, schema)
    # Always the question this row trains on (never a ``user_prompt`` the file carried): eval re-renders
    # from the question it finds behind the block (``bare_question``), so both sides see the same text.
    merged["user_prompt"] = question
    if merged.get("source_text") == question:
        merged["source_text"] = rendered
    merged["question"] = rendered


_BUNDLE_CACHE: "OrderedDict[str, Any]" = OrderedDict()


def _schema_bundle(schema: dict[str, Any]) -> Any:
    key = digest(schema)
    if key in _BUNDLE_CACHE:
        _BUNDLE_CACHE.move_to_end(key)
        return _BUNDLE_CACHE[key]
    from app.services import json_schema_bundle as jsb

    try:
        bundle = jsb.validate_bundle({"row.json": schema}, "row.json")
    except Exception:  # noqa: BLE001 - normalize() rejects these at import
        bundle = None
    _BUNDLE_CACHE[key] = bundle
    while len(_BUNDLE_CACHE) > 512:
        _BUNDLE_CACHE.popitem(last=False)
    return bundle


def _strict_json(text: str) -> Any:
    """``text`` (code fences stripped) as one JSON document; None when it is not exactly that."""
    from app.services.eval_task_handler_service import strip_code_fences

    try:
        return json.loads(strip_code_fences(text.strip()).strip())
    except (ValueError, RecursionError):
        return None


def parse_json_strict(text: Any) -> Any:
    """A model output that IS JSON (a dict / list as is, or the whole text, optionally inside a code
    fence). JSON wrapped in prose is None here (``parse_json_output`` finds it). A deeply nested output
    (RecursionError) is None, not an exception."""
    if isinstance(text, (dict, list)):
        return text
    raw = str(text or "").strip()
    return _strict_json(raw) if raw else None


def parse_json_output(text: Any) -> Any:
    """A model output read as JSON: a dict / list as is, else the text with code fences stripped, else
    its first balanced ``{...}`` object. None when nothing parses (a deeply nested output included)."""
    strict = parse_json_strict(text)
    if strict is not None or isinstance(text, (dict, list)):
        return strict
    raw = str(text or "").strip()
    if not raw:
        return None
    from app.services.eval_task_handler_service import strip_code_fences

    raw = strip_code_fences(raw).strip()
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(raw)):
        if raw[idx] == "{":
            depth += 1
        elif raw[idx] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : idx + 1])
                except (ValueError, RecursionError):
                    return None
    return None


def validate_output(value: Any, schema: dict[str, Any]) -> list[str]:
    """``value`` checked against ``schema`` (JSON Schema, the bundle validator): up to
    ``MAX_SCHEMA_ERRORS`` messages, ``[]`` when valid."""
    bundle = _schema_bundle(schema)
    if bundle is None:
        return ["(root): the row's schema is unusable"]
    out: list[str] = []
    for err in bundle.validate(value)[:MAX_SCHEMA_ERRORS]:
        out.append(f"{err.get('pointer') or '(root)'}: {str(err.get('message'))[:160]}")
    return out


def score_output_schema(predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Per-row check of the prediction against the row's own schema (``row_output_schema`` on the
    prediction): sets ``row_output_schema_valid`` / ``row_output_schema_errors`` in place and returns
    ``metrics.output_schema`` = ``{rows, json_valid_rate, schema_valid_rate}`` over those rows. None (and
    nothing set) when no row carries one, so other evals are unchanged. Any task profile.

    ``json_valid_rate`` counts outputs that ARE JSON (optionally inside a code fence); JSON wrapped in
    prose is found and checked against the schema (so ``schema_valid_rate`` still sees it) but only shows
    in ``json_found_rate`` (present only when it differs). A too-deeply-nested output counts as invalid."""
    rows = json_ok = json_found = valid = 0
    for p in predictions:
        schema = p.get("row_output_schema") if isinstance(p, dict) else None
        if not isinstance(schema, dict):
            continue
        rows += 1
        raw = p.get("prediction")
        strict = parse_json_strict(raw)
        parsed = strict if strict is not None else parse_json_output(raw)
        if parsed is None:
            errors = ["(root): the output is not JSON"]
        else:
            json_found += 1
            json_ok += int(strict is not None)
            errors = validate_output(parsed, schema)
        p["row_output_schema_valid"] = not errors
        p["row_output_schema_errors"] = errors
        valid += int(not errors)
    if not rows:
        return None
    out = {
        "rows": rows,
        "json_valid_rate": round(json_ok / rows, 4),
        "schema_valid_rate": round(valid / rows, 4),
    }
    if json_found != json_ok:
        out["json_found_rate"] = round(json_found / rows, 4)
    return out


# ── eval provenance ──────────────────────────────────────────────────────


_UNSET: Any = object()


def eval_view(
    row: Any,
    *,
    profile: str | None,
    overrides: dict[str, dict[str, Any]] | None = None,
    system_used: Any = _UNSET,
    output_schema_used: bool | None = None,
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
    schema = row_output_schema(row)
    schema_read = consumes("output_schema", profile) and output_schema_used is not False
    if schema is not None and schema_read:
        text = canonical_schema(schema)
        view["output_schema"] = text[:VIEW_SCHEMA_CHARS]
        view["output_schema_source"] = "row"
        view["output_schema_fields"] = len(schema_fields(schema))
        if len(text) > VIEW_SCHEMA_CHARS:
            view["output_schema_truncated"] = True
    unused = [
        k for k in present
        if k != "system" and not (schema_read if k == "output_schema" else consumes(k, profile))
    ]
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
    if consumes("output_schema", profile) and any(isinstance(v, dict) and v.get("output_schema_source") for v in views):
        # Only when some row has its own schema (other evals' metrics are unchanged). A result with this
        # block rendered the schema in the prompt; one from before Phase 4 listed it under unused_kinds.
        s = {"row": 0, "none": 0}
        for v in views:
            s["row" if isinstance(v, dict) and v.get("output_schema_source") == "row" else "none"] += 1
        kinds["output_schema"] = s
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
                # prepared rows store row_context as JSON text (escaped quotes): match the bare key
                if '"row_context"' not in line or 'tools' not in line:
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
