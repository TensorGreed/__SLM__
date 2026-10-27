"""Tool-call scoring: does a prediction call tools from the project's catalog, with valid arguments?

Two shapes share one scorer:

- ``plan`` mode (the ``structured-plan`` recipe): a JSON plan holds an array of steps (``steps_pointer``);
  each step names a tool (``tool_field``), its arguments (``args_field``), an optional id
  (``id_field``) and the ids it depends on (``depends_field``).
- ``call`` mode (the ``tool_calling`` task profile): one call ``{"tool", "args"}``, a list of them,
  ``{"tool_calls": [...]}`` or the ``{"name", "arguments"}`` spelling. ``{"tool": "none"}`` is an
  explicit "no call".

Per row (``score_calls``): ``tool_call_valid`` (every call names a known, allowed tool and its args
validate against the catalog schema), ``tool_args_match`` (against the gold calls, by position, typed
``value_match``) and ``plan_order_valid`` (unique ids, existing dependencies, no cycles). A row passes
when every ENABLED mode that applies is true. ``score_tool_block`` aggregates a whole eval into the
``tools`` metrics block; ``run_dry_run`` optionally executes read-only calls against the live server
(informational; it never changes a verdict and never executes a mutating tool).

Project settings live in ``Project.runtime_config["tool_scoring"]`` (see ``DEFAULT_CONFIG``); they only
apply when the project has a tool catalog (``tool_catalog_service``).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Callable

from app.services.json_pointer import resolve_json_pointer, split_json_pointer
from app.services.output_schema_service import value_match

CONFIG_KEY = "tool_scoring"
SCORING_MODE_CALL = "tool_call"
ALL_MODES = ("tool_call_valid", "tool_args_match", "plan_order_valid")
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "modes": list(ALL_MODES),
    "steps_pointer": "/steps",
    "tool_field": "tool",
    "args_field": "args",
    "id_field": "step",
    "depends_field": "depends_on",
    "arg_keys": [],
    "dry_run": False,
}
MAX_ARG_KEYS = 50
MAX_ROW_ERRORS = 3
MAX_TOP_ERRORS = 10
MAX_DRY_RUN_CALLS = 50
NO_CALL_NAMES = {"none"}
_MISSING = "(missing)"


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


# ── config ───────────────────────────────────────────────────────────────


def stored_config(runtime_config: Any) -> dict[str, Any]:
    raw = runtime_config.get(CONFIG_KEY) if isinstance(runtime_config, dict) else None
    return raw if isinstance(raw, dict) else {}


def normalize_config(raw: Any) -> dict[str, Any]:
    """Stored settings with defaults applied (bad values fall back to the default)."""
    raw = raw if isinstance(raw, dict) else {}
    cfg = dict(DEFAULT_CONFIG)
    cfg["modes"] = list(ALL_MODES)
    cfg["arg_keys"] = []
    if isinstance(raw.get("enabled"), bool):
        cfg["enabled"] = raw["enabled"]
    if isinstance(raw.get("modes"), list):
        cfg["modes"] = [m for m in ALL_MODES if m in raw["modes"]]
    ptr = raw.get("steps_pointer")
    if isinstance(ptr, str) and split_json_pointer(ptr) is not None:
        cfg["steps_pointer"] = ptr
    for key in ("tool_field", "args_field", "id_field", "depends_field"):
        if isinstance(raw.get(key), str) and raw[key].strip():
            cfg[key] = raw[key].strip()
    if isinstance(raw.get("arg_keys"), list):
        cfg["arg_keys"] = [str(k) for k in raw["arg_keys"] if isinstance(k, str) and k][:MAX_ARG_KEYS]
    if isinstance(raw.get("dry_run"), bool):
        cfg["dry_run"] = raw["dry_run"]
    return cfg


def has_catalog(runtime_config: Any) -> bool:
    cat = runtime_config.get("tool_catalog") if isinstance(runtime_config, dict) else None
    return isinstance(cat, dict) and bool(cat.get("tools"))


def effective_tool_scoring_config(runtime_config: Any) -> dict[str, Any] | None:
    """The settings with defaults applied, or ``None`` when the project has no tool catalog or the
    scoring is disabled (every caller then behaves exactly as before the catalog existed)."""
    if not has_catalog(runtime_config):
        return None
    cfg = normalize_config(stored_config(runtime_config))
    return cfg if cfg["enabled"] else None


def dry_run_env_enabled() -> bool:
    from app.config import settings

    return bool(getattr(settings, "MCP_DRY_RUN_ENABLED", False))


def tools_fingerprint(runtime_config: Any) -> str:
    """Catalog version + flags version + the canonical scoring settings; ``"none"`` without an active
    catalog. Part of ``structured_plan_scoring_fingerprint`` so a cached base result is not reused after
    the catalog, its flags or the settings change."""
    cfg = effective_tool_scoring_config(runtime_config)
    if cfg is None:
        return "none"
    from app.services import tool_catalog_service as tcs

    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return f"{tcs.catalog_fingerprint(runtime_config)}:{hashlib.sha256(canon.encode()).hexdigest()[:12]}"


TOOL_CALL_HANDLER_VERSION = "tool_call/v1"


def tool_call_fingerprint(runtime_config: Any) -> str:
    """What a cached ``tool_call`` base result depends on: the handler version + ``tools_fingerprint``
    (catalog, flags, scoring settings). Stamped into the metrics as ``tool_call_fingerprint`` even without
    a catalog; a cached result without it (older than this scoring) is never reused."""
    return f"{TOOL_CALL_HANDLER_VERSION}+{tools_fingerprint(runtime_config)}"


# ── parsing ──────────────────────────────────────────────────────────────

_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)
_TOOL_CALL_TEXT = re.compile(r"Tool Call:\s*([A-Za-z0-9_.\-]+)\s*\(")


def parse_json_value(value: Any) -> Any:
    """Best-effort JSON parse of a model output: dict/list passthrough, code fences stripped, else the
    first balanced ``{...}`` / ``[...]`` block. ``None`` when nothing parses."""
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(obj, (dict, list)):
                return obj
    return None


def parse_tool_call_text(text: Any) -> Any:
    """The adapter's training rendering ``Tool Call: name({...args})`` as ``{"tool", "args"}``; ``None``
    when the text has no such line."""
    s = str(text or "")
    m = _TOOL_CALL_TEXT.search(s)
    if not m:
        return None
    rest = s[m.end():].lstrip()
    if rest.startswith(")"):
        return {"tool": m.group(1), "args": {}}
    try:
        args, _ = json.JSONDecoder().raw_decode(rest)
    except ValueError:
        return {"tool": m.group(1), "args": None}
    return {"tool": m.group(1), "args": args}


def parse_call_output(text: Any) -> Any:
    """JSON first, then the ``Tool Call: name(args)`` rendering."""
    head = str(text or "").lstrip()[:1]
    if head not in ("{", "[", "`"):
        rendered = parse_tool_call_text(text)  # "Tool Call: name({...})" before any JSON inside its args
        if rendered is not None:
            return rendered
    parsed = parse_json_value(text)
    if parsed is not None:
        return parsed
    return parse_tool_call_text(text)


def _args_of(value: Any) -> Any:
    """A call's arguments as an object (a JSON string is parsed); anything unparsable stays as is."""
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except ValueError:
            return value
    return value


def _norm_id(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    return str(value)


def _depends(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _call_from_dict(item: Any, index: int, cfg: dict[str, Any], *, plan: bool) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"id": None, "tool": None, "args": {}, "depends_on": []}
    fn = item.get("function") if isinstance(item.get("function"), dict) else None
    src = fn if (fn is not None and not plan) else item
    if plan:
        tool, raw_args = src.get(cfg["tool_field"]), src.get(cfg["args_field"])
    else:
        tool = src.get("tool", src.get("name"))
        raw_args = src.get("args", src.get("arguments", src.get("parameters")))
    return {
        "id": _norm_id(item.get(cfg["id_field"])) if plan else None,
        "tool": tool if isinstance(tool, str) else None,
        "args": _args_of(raw_args),
        "depends_on": _depends(item.get(cfg["depends_field"])) if plan else [],
    }


def extract_calls(parsed: Any, cfg: dict[str, Any], *, mode: str = "plan") -> list[dict[str, Any]] | None:
    """The calls in a parsed model output: ``[{"id", "tool", "args", "depends_on"}]``. ``[]`` is an
    explicit empty plan / no-call; ``None`` means nothing extractable."""
    cfg = normalize_config(cfg)
    if mode == "plan":
        tokens = split_json_pointer(cfg["steps_pointer"])
        if parsed is None or tokens is None:
            return None
        for value in resolve_json_pointer(parsed, tokens):
            if isinstance(value, list):
                return [_call_from_dict(s, i, cfg, plan=True) for i, s in enumerate(value)]
        return None
    # call mode
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tool_calls"), list):
            items = parsed["tool_calls"]
        elif "tool" in parsed or "name" in parsed or isinstance(parsed.get("function"), dict):
            items = [parsed]
        else:
            return None
    elif isinstance(parsed, list):
        items = parsed
    else:
        return None
    calls: list[dict[str, Any]] = []
    no_calls: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        call = _call_from_dict(item, i, cfg, plan=False)
        if isinstance(call["tool"], str) and call["tool"].strip().lower() in NO_CALL_NAMES:
            no_calls.append(call)  # an explicit no-call contributes nothing...
            continue
        calls.append(call)
    if not items:
        return []
    if calls and no_calls:
        return [*calls, *no_calls]  # ...unless mixed with real calls: kept so scoring can reject the output
    return calls


def gold_calls_from_fields(tool_name: Any, tool_args: Any) -> list[dict[str, Any]] | None:
    """Gold from a row's ``tool_name`` + ``tool_args_json`` fields."""
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None
    if tool_name.strip().lower() in NO_CALL_NAMES:
        return []
    return [{"id": None, "tool": tool_name.strip(), "args": _args_of(tool_args), "depends_on": []}]


# ── per-row scoring ──────────────────────────────────────────────────────


def _short(text: Any, limit: int = 160) -> str:
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def args_match_pair(pred: dict[str, Any], gold: dict[str, Any], arg_keys: list[str] | None = None) -> bool:
    """Same tool name and every selected argument key matches (``arg_keys`` or all gold keys)."""
    from app.services.structured_plan_scoring import types_of

    if not pred.get("tool") or pred.get("tool") != gold.get("tool"):
        return False
    p_args, g_args = pred.get("args"), gold.get("args")
    if not isinstance(p_args, dict):
        return not g_args and not p_args
    if not isinstance(g_args, dict):
        return False
    keys = list(arg_keys) if arg_keys else list(g_args)
    for key in keys:
        in_p, in_g = key in p_args, key in g_args
        if not in_g:
            if in_p and p_args[key] is not None:
                return False
            continue
        if not in_p:
            return False
        if not value_match(p_args[key], g_args[key], types_of(g_args[key])):
            return False
    return True


def _order_errors(calls: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for c in calls:
        cid = c.get("id")
        if cid is None:
            continue
        if cid in seen:
            errors.append(f"duplicate step id {cid}")
        seen.add(cid)
    graph: dict[str, list[str]] = {}
    for c in calls:
        cid = c.get("id")
        for dep in c.get("depends_on") or []:
            if dep not in seen:
                errors.append(f"step {cid if cid is not None else '?'} depends on unknown step {dep}")
        if cid is not None:
            graph.setdefault(cid, []).extend(d for d in c.get("depends_on") or [] if d in seen)
    state: dict[str, int] = {}

    def has_cycle(node: str) -> bool:
        stack = [(node, iter(graph.get(node, [])))]
        state[node] = 1
        while stack:
            cur, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[cur] = 2
                stack.pop()
                continue
            if state.get(nxt) == 1:
                return True
            if state.get(nxt) is None:
                state[nxt] = 1
                stack.append((nxt, iter(graph.get(nxt, []))))
        return False

    if any(state.get(n) is None and has_cycle(n) for n in list(graph)):
        errors.append("dependency cycle")
    return errors


def score_calls(
    pred_calls: list[dict[str, Any]] | None,
    gold_calls: list[dict[str, Any]] | None,
    runtime_config: Any,
    cfg: dict[str, Any],
    *,
    mode: str = "plan",
) -> dict[str, Any]:
    from app.services import tool_catalog_service as tcs

    cfg = normalize_config(cfg)
    by_name = {t["name"]: t for t in tcs.effective_tools(runtime_config)}
    errors: list[str] = []
    calls_out: list[dict[str, Any]] = []

    if pred_calls is None:
        errors.append("no tool calls found in the output")
        valid: bool = False
        args_match: bool | None = False if gold_calls else None
        order: bool | None = None
        calls_out = []
    else:
        for i, call in enumerate(pred_calls):
            name = call.get("tool")
            tool = by_name.get(name) if isinstance(name, str) else None
            known = tool is not None
            allowed = bool(tool and tool.get("allowed", True))
            args_valid, args_errors = False, []
            if isinstance(name, str) and name.strip().lower() in NO_CALL_NAMES:
                known = allowed = False
                errors.append("no-call mixed with calls")
            elif not known:
                errors.append(f"unknown tool {_short(name if name else _MISSING, 60)}")
            elif not allowed:
                errors.append(f"tool {name} is not allowed")
            if known:
                args = call.get("args")
                bundle = tcs.tool_bundle(runtime_config, name)
                if not isinstance(args, dict):
                    args_errors = ["(root): arguments must be an object"]
                elif bundle is None or tool.get("schema_error"):
                    args_errors = ["(root): the tool's schema is unusable"]
                else:
                    for err in bundle.validate(args)[:MAX_ROW_ERRORS]:
                        args_errors.append(f"{err.get('pointer') or '(root)'}: {_short(err.get('message'), 120)}")
                args_valid = not args_errors
                if not args_valid:
                    errors.append(f"{name}: invalid arguments")
            calls_out.append({
                "tool": name, "allowed": allowed, "known": known, "args_valid": args_valid,
                "args_errors": args_errors[:MAX_ROW_ERRORS], "args_match": None, "dry_run": None,
            })
        if pred_calls:
            valid = all(c["known"] and c["allowed"] and c["args_valid"] for c in calls_out)
        elif mode == "call":
            valid = True  # an explicit no-call
        else:
            valid = gold_calls == []
            if not valid:
                errors.append("empty plan")

        if not gold_calls:
            args_match = None
        else:
            args_match = len(pred_calls) == len(gold_calls)
            for i, entry in enumerate(calls_out):
                ok = i < len(gold_calls) and args_match_pair(pred_calls[i], gold_calls[i], cfg["arg_keys"])
                entry["args_match"] = ok
                args_match = args_match and ok
            if not pred_calls:
                args_match = False
        if mode == "plan":
            order_errors = _order_errors(pred_calls)
            errors.extend(order_errors)
            order = not order_errors
        else:
            order = None

    flags = {"tool_call_valid": valid, "tool_args_match": args_match, "plan_order_valid": order}
    enabled = [m for m in cfg["modes"] if flags.get(m) is not None]
    passed = all(flags[m] for m in enabled)
    if pred_calls is None:
        passed = False
    return {
        "calls": calls_out,
        "tool_call_valid": valid,
        "tool_args_match": args_match,
        "plan_order_valid": order,
        "pass": bool(passed),
        "errors": [_short(e) for e in dict.fromkeys(errors)][:10],
    }


# ── eval aggregation ─────────────────────────────────────────────────────


def _gold_for(prediction: dict[str, Any], cfg: dict[str, Any], mode: str) -> list[dict[str, Any]] | None:
    pre = prediction.get("tool_gold_calls")
    if isinstance(pre, list):
        return pre
    parsed_ref = prediction.get("parsed_reference")
    if parsed_ref is None:
        parsed_ref = parse_call_output(prediction.get("reference"))
    return extract_calls(parsed_ref, cfg, mode=mode)


def _pred_for(prediction: dict[str, Any], cfg: dict[str, Any], mode: str) -> list[dict[str, Any]] | None:
    parsed = prediction.get("parsed_prediction")
    if parsed is None:
        parsed = parse_call_output(prediction.get("prediction"))
    return extract_calls(parsed, cfg, mode=mode)


def score_tool_block(
    predictions: list[dict[str, Any]], runtime_config: Any, cfg: dict[str, Any], *, mode: str = "plan"
) -> dict[str, Any]:
    """Score every prediction (sets ``row_tool_results`` / ``row_tools_pass`` / ``row_tool_calls`` in
    place) and return ``{"tools": {...}, "tool_call_valid_rate", "tool_args_match_rate",
    "plan_order_valid_rate"}``. Blocking (schema validation): run it in a worker thread."""
    cfg = normalize_config(cfg)
    counts = {m: [0, 0] for m in ALL_MODES}  # [true, applicable]
    n_pass = n_total = 0
    per_tool: dict[str, dict[str, int]] = {}
    unknown: Counter[str] = Counter()
    errors_top: Counter[str] = Counter()
    for prediction in predictions:
        if not isinstance(prediction, dict) or prediction.get("row_skipped"):
            continue
        n_total += 1
        pred = _pred_for(prediction, cfg, mode)
        gold = _gold_for(prediction, cfg, mode)
        result = score_calls(pred, gold, runtime_config, cfg, mode=mode)
        prediction["row_tool_calls"] = pred
        prediction["row_tool_results"] = result
        prediction["row_tools_pass"] = result["pass"]
        n_pass += int(result["pass"])
        for m in ALL_MODES:
            if result[m] is not None:
                counts[m][1] += 1
                counts[m][0] += int(bool(result[m]))
        for entry in result["calls"]:
            name = entry["tool"] if isinstance(entry["tool"], str) and entry["tool"] else _MISSING
            if not entry["known"]:
                unknown[name] += 1
                continue
            bucket = per_tool.setdefault(name, {"calls": 0, "valid": 0, "invalid_args": 0, "not_allowed": 0})
            bucket["calls"] += 1
            bucket["valid"] += int(entry["allowed"] and entry["args_valid"])
            bucket["invalid_args"] += int(not entry["args_valid"])
            bucket["not_allowed"] += int(not entry["allowed"])
        for e in result["errors"]:
            errors_top[e] += 1

    rates = {m: _rate(*counts[m]) for m in ALL_MODES}
    cat = runtime_config.get("tool_catalog") if isinstance(runtime_config, dict) else None
    cat = cat if isinstance(cat, dict) else {}
    src = cat.get("source")
    source = src.get("kind") if isinstance(src, dict) else None
    block = {
        "catalog_version": cat.get("version"),
        "flags_version": cat.get("flags_version"),
        "source_kind": source,
        "modes": list(cfg["modes"]),
        "tool_call_valid_rate": rates["tool_call_valid"],
        "tool_args_match_rate": rates["tool_args_match"],
        "plan_order_valid_rate": rates["plan_order_valid"],
        "pass_rate": _rate(n_pass, n_total),
        "per_tool": dict(sorted(per_tool.items())),
        "unknown_tools": dict(unknown.most_common()),
        "errors_top": [{"message": m, "count": c} for m, c in errors_top.most_common(MAX_TOP_ERRORS)],
        "dry_run": None,
    }
    return {
        "tools": block,
        "tool_call_valid_rate": rates["tool_call_valid"],
        "tool_args_match_rate": rates["tool_args_match"],
        "plan_order_valid_rate": rates["plan_order_valid"],
    }


async def run_dry_run(
    db: Any, project: Any, predictions: list[dict[str, Any]], tool_block: dict[str, Any], cfg: dict[str, Any]
) -> None:
    """Informational dry run (``cfg["dry_run"]`` and ``MCP_DRY_RUN_ENABLED``): for rows whose calls are
    all valid, run each READ-ONLY call through ``tool_catalog_service.dry_run_call`` (at most 50 per
    eval, in order). Mutating calls are skipped ("mutating: never executed"). Results land in
    ``row_tool_results["calls"][i]["dry_run"]`` and ``tools["dry_run"]``; a failure never raises and
    never changes a verdict."""
    cfg = normalize_config(cfg)
    if not cfg["dry_run"] or not isinstance(tool_block.get("tools"), dict):
        return
    summary = {"enabled": dry_run_env_enabled(), "calls": 0, "ok": 0, "errors": 0, "skipped": 0}
    tool_block["tools"]["dry_run"] = summary
    if not summary["enabled"]:
        return
    from app.services import tool_catalog_service as tcs

    rc = getattr(project, "runtime_config", None)
    by_name = {t["name"]: t for t in tcs.effective_tools(rc)}
    for prediction in predictions:
        result = prediction.get("row_tool_results") if isinstance(prediction, dict) else None
        calls = prediction.get("row_tool_calls") if isinstance(prediction, dict) else None
        if not isinstance(result, dict) or not result.get("tool_call_valid") or not calls:
            continue
        for i, call in enumerate(calls):
            if i >= len(result["calls"]):
                break
            tool = by_name.get(call.get("tool"))
            if tool is None:
                continue
            if tool.get("mutating", True):
                result["calls"][i]["dry_run"] = {"skipped": "mutating: never executed"}
                summary["skipped"] += 1
                continue
            if summary["calls"] >= MAX_DRY_RUN_CALLS:
                result["calls"][i]["dry_run"] = {"skipped": f"limit of {MAX_DRY_RUN_CALLS} dry-run calls reached"}
                summary["skipped"] += 1
                continue
            summary["calls"] += 1
            try:
                out = await tcs.dry_run_call(db, project, call["tool"], call.get("args") or {})
                entry = {
                    "ok": bool(out.get("ok")), "is_error": bool(out.get("is_error")),
                    "error": out.get("error"), "snippet": str(out.get("snippet") or "")[:300],
                }
            except Exception as exc:  # noqa: BLE001 - informational; never fails the eval
                entry = {"ok": False, "is_error": True, "error": _short(exc, 200), "snippet": ""}
            result["calls"][i]["dry_run"] = entry
            summary["ok" if entry["ok"] else "errors"] += 1


def finalize_with_tools(
    predictions: list[dict[str, Any]], metrics: dict[str, Any], tool_block: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge ``score_tool_block``'s result into a structured-plan ``metrics`` dict and fold each row's
    tool verdict into ``row_structured_plan_checks_pass``. Recomputes ``structured_plan_pass_rate`` /
    ``correct`` / ``exact_match`` (call before ``finalize_with_rubric``)."""
    from app.services.structured_plan_scoring import ROW_PASS_KEY

    if not isinstance(tool_block, dict) or not isinstance(tool_block.get("tools"), dict):
        return metrics
    metrics["tools"] = tool_block["tools"]
    for key in ("tool_call_valid_rate", "tool_args_match_rate", "plan_order_valid_rate"):
        metrics[key] = tool_block.get(key)
    n_pass = 0
    for prediction in predictions:
        checks = prediction.get("row_structured_plan_checks_pass")
        if not isinstance(checks, bool):
            continue
        passed = checks and prediction.get("row_tools_pass") is not False
        prediction["row_structured_plan_checks_pass"] = passed
        prediction[ROW_PASS_KEY] = passed
        prediction["row_exact_match"] = 1.0 if passed else 0.0
        n_pass += int(passed)
    total = int(metrics.get("total") or 0)
    rate = round(n_pass / total, 4) if total else 0.0
    metrics["correct"] = n_pass
    metrics["exact_match"] = rate
    metrics["structured_plan_pass_rate"] = rate
    return metrics


def finalize_tool_calls(
    predictions: list[dict[str, Any]], metrics: dict[str, Any], tool_block: dict[str, Any] | None, cfg: dict[str, Any]
) -> dict[str, Any]:
    """``tool_calling`` eval with a catalog: a row passes when the tool was selected correctly, the call
    is valid against the catalog (``tool_call_valid`` mode) and the args match (``tool_args_match``
    mode). Recomputes ``tool_call_pass_rate`` / ``exact_match`` / ``correct``."""
    if not isinstance(tool_block, dict) or not isinstance(tool_block.get("tools"), dict):
        return metrics
    cfg = normalize_config(cfg)
    metrics["tools"] = tool_block["tools"]
    metrics["tool_call_valid_rate"] = tool_block.get("tool_call_valid_rate")
    n_pass = 0
    for prediction in predictions:
        if not isinstance(prediction.get("row_tool_selection_correct"), bool):
            continue
        res = prediction.get("row_tool_results") or {}
        ok = bool(prediction["row_tool_selection_correct"])
        if "tool_call_valid" in cfg["modes"]:
            ok = ok and bool(res.get("tool_call_valid"))
        if "tool_args_match" in cfg["modes"]:
            ok = ok and bool(prediction.get("row_tool_args_match"))
        prediction["row_tool_call_pass"] = ok
        prediction["row_exact_match"] = 1.0 if ok else 0.0
        n_pass += int(ok)
    total = int(metrics.get("total") or 0)
    rate = round(n_pass / total, 4) if total else 0.0
    metrics.update({"correct": n_pass, "exact_match": rate, "tool_call_pass_rate": rate})
    return metrics


def tool_prompt(block: str | None, text: str) -> str:
    """The prompt a ``tool_calling`` row uses: the tool block (when the project has a catalog) ahead of
    the user text. Shared by the ``tool-call-json`` adapter and the eval handler so train and eval match."""
    return f"{block}\n\n{text}" if block else text
