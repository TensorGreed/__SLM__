"""Plan-rule engine: caller-supplied, deterministic consistency rules over structured plans.

A structured plan is a JSON object with a list of steps (``plan["steps"]`` by default). The caller
writes a *rules document* (data, validated here against ``RULES_SCHEMA``) and the engine reports
which rules a plan breaks: no LLM, no domain knowledge in the code. The same engine runs

* at generation (``params.plan_rules`` of the structured-plan ``scenario_grid`` playbook, after
  schema validation: ``reject`` rules drop the plan as ``rule_violation``, ``warn`` rules only
  annotate it), and
* over the pending rows of the review queue (``check_pending_rows`` / the ``check-plan-rules``
  endpoint: dry run = counts per rule, apply = soft-reject ``rule_violation:<rule id>``).

Rules document::

    {"version": 1,
     "paths": {"tool": "tool", "operation": "args.operation"},   # optional; these are the defaults
     "steps_path": "steps",                                      # optional; the default
     "rules": [{"id": "...", "type": "step_requires" | "ref_requires" | "requires_step"
                                      | "forbid_together",
                "severity": "reject" | "warn", "description": "...", ...}]}

*Match* (picks steps): ``{"tool": str|[str], "operation": str|[str], "operation_regex": str,
"where": <condition>}`` (at least one key). The tool / operation are read at ``paths.tool`` /
``paths.operation`` (dotted paths inside the step).

*Condition*: ``{"path": "args.arguments.body.kind", "op": <op>, "value": ..., "quantifier":
"any"|"all"}`` with ops ``equals not_equals in not_in exists not_exists regex contains max min``
(``max`` / ``min`` are numeric bounds, inclusive). A path is dotted; ``*`` (or ``[*]``) walks every
list item / object value, an integer segment indexes a list. With several resolved values the
condition holds when ANY (default) / ALL of them satisfy the op; ``exists`` / ``not_exists`` look
only at whether the path resolved. Combinators: ``{"all": [...]}``, ``{"any": [...]}``,
``{"not": {...}}``.

Rule types:

* ``step_requires``  ``{match, require}`` - every matching step must satisfy ``require``.
* ``ref_requires``   ``{match, ref: {path}, creator: {match, name_path}, require, before?,
  on_missing?}`` - each value a matching step holds at ``ref.path`` names an object created by
  a step matching ``creator.match`` (its ``name_path`` values); every such creating step must
  satisfy ``require``. A reference no step creates is a violation (``on_missing: "violate"``,
  the default) or skipped (``"ignore"``). ``before: true`` also demands the creator come earlier.
* ``requires_step``  ``{if: match, then: match, before?}`` - when some step matches ``if``, some
  step must match ``then`` (``before: true``: earlier than the first ``if`` step).
* ``forbid_together`` ``{a: match, b: match, same_step?}`` - a plan having steps for both
  selectors is a violation (``same_step: true``: one step matching both).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from jsonschema import Draft202012Validator

MAX_RULES_CHARS = 65_536
MAX_RULES = 200
DEFAULT_PATHS = {"tool": "tool", "operation": "args.operation"}
SEVERITIES = ("reject", "warn")
REASON_PREFIX = "rule_violation"

_OPS = ("equals", "not_equals", "in", "not_in", "exists", "not_exists", "regex", "contains", "max", "min")
_PATH = {"type": "string", "minLength": 1, "maxLength": 200}
_COND_REF = {"$ref": "#/$defs/condition"}
_MATCH = {
    "type": "object", "minProperties": 1, "additionalProperties": False,
    "properties": {
        "tool": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}, "minItems": 1}]},
        "operation": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}, "minItems": 1}]},
        "operation_regex": {"type": "string", "maxLength": 200},
        "where": _COND_REF,
    },
}
_RULE_BASE = {
    "id": {"type": "string", "pattern": r"^[A-Za-z0-9_.-]{1,48}$"},
    "description": {"type": "string", "maxLength": 500},
    "message": {"type": "string", "maxLength": 500},
    "severity": {"enum": list(SEVERITIES)},
}


def _rule(type_name: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["id", "type", *required],
        "properties": {"type": {"const": type_name}, **_RULE_BASE, **props},
    }


RULES_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object", "additionalProperties": False, "required": ["rules"],
    "properties": {
        "version": {"const": 1},
        "paths": {
            "type": "object", "additionalProperties": False,
            "properties": {"tool": _PATH, "operation": _PATH},
        },
        "steps_path": _PATH,
        "rules": {
            "type": "array", "minItems": 1, "maxItems": MAX_RULES,
            "items": {"oneOf": [
                _rule("step_requires", {"match": {"$ref": "#/$defs/match"}, "require": _COND_REF}, ["match", "require"]),
                _rule("ref_requires", {
                    "match": {"$ref": "#/$defs/match"},
                    "ref": {"type": "object", "additionalProperties": False, "required": ["path"], "properties": {"path": _PATH}},
                    "creator": {
                        "type": "object", "additionalProperties": False, "required": ["match", "name_path"],
                        "properties": {"match": {"$ref": "#/$defs/match"}, "name_path": _PATH},
                    },
                    "require": _COND_REF, "before": {"type": "boolean"},
                    "on_missing": {"enum": ["violate", "ignore"]},
                }, ["match", "ref", "creator", "require"]),
                _rule("requires_step", {
                    "if": {"$ref": "#/$defs/match"}, "then": {"$ref": "#/$defs/match"}, "before": {"type": "boolean"},
                }, ["if", "then"]),
                _rule("forbid_together", {
                    "a": {"$ref": "#/$defs/match"}, "b": {"$ref": "#/$defs/match"}, "same_step": {"type": "boolean"},
                }, ["a", "b"]),
            ]},
        },
    },
    "$defs": {
        "match": _MATCH,
        "condition": {"oneOf": [
            {
                "type": "object", "additionalProperties": False, "required": ["path", "op"],
                "properties": {
                    "path": _PATH, "op": {"enum": list(_OPS)}, "value": {},
                    "quantifier": {"enum": ["any", "all"]},
                },
            },
            {"type": "object", "additionalProperties": False, "required": ["all"],
             "properties": {"all": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/condition"}}}},
            {"type": "object", "additionalProperties": False, "required": ["any"],
             "properties": {"any": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/condition"}}}},
            {"type": "object", "additionalProperties": False, "required": ["not"],
             "properties": {"not": {"$ref": "#/$defs/condition"}}},
        ]},
    },
}

_VALIDATOR = Draft202012Validator(RULES_SCHEMA)


class PlanRulesError(ValueError):
    """The rules document is not valid. ``pointer`` locates the problem inside it."""

    def __init__(self, message: str, pointer: str = ""):
        super().__init__(message)
        self.message = message
        self.pointer = pointer


@dataclass(frozen=True)
class Violation:
    rule_id: str
    severity: str
    steps: tuple[int, ...]
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"rule_id": self.rule_id, "severity": self.severity, "steps": list(self.steps), "message": self.message}


@dataclass
class RuleSet:
    doc: dict[str, Any]
    tool_path: str
    operation_path: str
    steps_path: str

    @property
    def rules(self) -> list[dict[str, Any]]:
        return self.doc["rules"]


# ── loading + validation ────────────────────────────────────────────

def load_rules(raw: Any) -> RuleSet:
    """Parse + validate a rules document (a dict or its JSON text). Raises ``PlanRulesError``."""
    if isinstance(raw, (str, bytes)):
        if len(raw) > MAX_RULES_CHARS:
            raise PlanRulesError(f"The rules document is {len(raw)} characters (max {MAX_RULES_CHARS}).")
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            raise PlanRulesError(f"The rules are not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PlanRulesError("The rules document must be a JSON object with a `rules` list.")
    try:
        size = len(json.dumps(raw))
    except (TypeError, ValueError) as exc:
        raise PlanRulesError("The rules document is not JSON-serialisable.") from exc
    if size > MAX_RULES_CHARS:
        raise PlanRulesError(f"The rules document is {size} characters (max {MAX_RULES_CHARS}).")
    errors = sorted(_VALIDATOR.iter_errors(raw), key=lambda e: len(list(e.absolute_path)))
    if errors:
        top = errors[0]
        pointer = "/" + "/".join(str(p) for p in top.absolute_path)
        detail = top.message if len(top.message) <= 200 else top.message[:200] + "…"
        if top.validator == "oneOf" and top.context:
            deepest = max(top.context, key=lambda e: len(list(e.absolute_path)))
            detail = deepest.message[:200]
            pointer = "/" + "/".join(str(p) for p in deepest.absolute_path)
        raise PlanRulesError(f"{pointer or '(root)'}: {detail}", pointer)
    seen: set[str] = set()
    for i, rule in enumerate(raw["rules"]):
        if rule["id"] in seen:
            raise PlanRulesError(f"/rules/{i}/id: duplicate rule id {rule['id']!r}.", f"/rules/{i}/id")
        seen.add(rule["id"])
    for i, rule in enumerate(raw["rules"]):
        for regex in _regexes_of(rule):
            try:
                re.compile(regex)
            except re.error as exc:
                raise PlanRulesError(f"/rules/{i}: bad regular expression {regex!r}: {exc}", f"/rules/{i}") from exc
    paths = {**DEFAULT_PATHS, **(raw.get("paths") or {})}
    return RuleSet(raw, paths["tool"], paths["operation"], raw.get("steps_path") or "steps")


def _regexes_of(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        if node.get("op") == "regex" and isinstance(node.get("value"), str):
            yield node["value"]
        if isinstance(node.get("operation_regex"), str):
            yield node["operation_regex"]
        for v in node.values():
            yield from _regexes_of(v)
    elif isinstance(node, list):
        for v in node:
            yield from _regexes_of(v)


# ── paths + conditions ──────────────────────────────────────────────

def resolve(obj: Any, path: str) -> list[Any]:
    """All values at a dotted path. ``*`` / ``[*]`` fan out over list items / object values, an
    integer indexes a list; anything that does not resolve yields nothing."""
    current = [obj]
    for raw_segment in path.replace("[*]", ".*").replace("[", ".").replace("]", "").split("."):
        if raw_segment == "":
            continue
        nxt: list[Any] = []
        for node in current:
            if raw_segment == "*":
                if isinstance(node, list):
                    nxt.extend(node)
                elif isinstance(node, dict):
                    nxt.extend(node.values())
            elif isinstance(node, dict):
                if raw_segment in node:
                    nxt.append(node[raw_segment])
            elif isinstance(node, list) and raw_segment.lstrip("-").isdigit():
                idx = int(raw_segment)
                if -len(node) <= idx < len(node):
                    nxt.append(node[idx])
        current = nxt
        if not current:
            break
    return current


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def _value_op(op: str, value: Any, expected: Any) -> bool:
    if op == "equals":
        return _same(value, expected)
    if op == "not_equals":
        return not _same(value, expected)
    if op == "in":
        return isinstance(expected, list) and any(_same(value, e) for e in expected)
    if op == "not_in":
        return not (isinstance(expected, list) and any(_same(value, e) for e in expected))
    if op == "regex":
        return isinstance(value, str) and isinstance(expected, str) and re.search(expected, value) is not None
    if op == "contains":
        if isinstance(value, str):
            return isinstance(expected, str) and expected in value
        if isinstance(value, list):
            return any(_same(v, expected) for v in value)
        return False
    if op in ("max", "min"):
        n, bound = _num(value), _num(expected)
        if n is None or bound is None:
            return False
        return n <= bound if op == "max" else n >= bound
    return False


def holds(cond: dict[str, Any], node: Any) -> bool:
    """Whether a condition holds on ``node`` (a step, or any JSON value)."""
    if "all" in cond:
        return all(holds(c, node) for c in cond["all"])
    if "any" in cond:
        return any(holds(c, node) for c in cond["any"])
    if "not" in cond:
        return not holds(cond["not"], node)
    values = resolve(node, cond["path"])
    op = cond["op"]
    if op == "exists":
        return bool(values)
    if op == "not_exists":
        return not values
    if not values:
        return False
    check = (lambda v: _value_op(op, v, cond.get("value")))
    return all(map(check, values)) if cond.get("quantifier") == "all" else any(map(check, values))


# ── matching ────────────────────────────────────────────────────────

def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else [x]


def make_matcher(match: dict[str, Any], rs: RuleSet) -> Callable[[dict[str, Any]], bool]:
    tools = _as_list(match["tool"]) if "tool" in match else None
    ops = _as_list(match["operation"]) if "operation" in match else None
    op_re = re.compile(match["operation_regex"]) if match.get("operation_regex") else None
    where = match.get("where")

    def matches(step: dict[str, Any]) -> bool:
        if tools is not None and not any(v in tools for v in resolve(step, rs.tool_path)):
            return False
        if ops is not None and not any(v in ops for v in resolve(step, rs.operation_path)):
            return False
        if op_re is not None and not any(isinstance(v, str) and op_re.search(v) for v in resolve(step, rs.operation_path)):
            return False
        return where is None or holds(where, step)

    return matches


def _steps_of(plan: Any, rs: RuleSet) -> list[tuple[int, dict[str, Any]]]:
    found = resolve(plan, rs.steps_path)
    steps = found[0] if found and isinstance(found[0], list) else []
    out = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        number = step.get("step")
        out.append((number if isinstance(number, int) and not isinstance(number, bool) else i + 1, step))
    return out


def _describe(rule: dict[str, Any], default: str) -> str:
    return rule.get("message") or rule.get("description") or default


# ── evaluation ──────────────────────────────────────────────────────

def check_plan(plan: Any, rules: RuleSet) -> list[Violation]:
    """Every violation of ``rules`` in one plan, in rule order. A plan with no step list has none."""
    steps = _steps_of(plan, rules)
    out: list[Violation] = []
    for rule in rules.rules:
        severity = rule.get("severity") or "reject"
        handler = _HANDLERS[rule["type"]]
        for numbers, message in handler(rule, steps, rules):
            out.append(Violation(rule["id"], severity, tuple(numbers), message))
    return out


def _step_requires(rule, steps, rs):
    matches = make_matcher(rule["match"], rs)
    for number, step in steps:
        if matches(step) and not holds(rule["require"], step):
            yield [number], _describe(rule, f"step {number} does not meet {rule['id']}")


def _ref_requires(rule, steps, rs):
    matches = make_matcher(rule["match"], rs)
    creator_matches = make_matcher(rule["creator"]["match"], rs)
    name_path = rule["creator"]["name_path"]
    ignore_missing = rule.get("on_missing") == "ignore"
    for number, step in steps:
        if not matches(step):
            continue
        for ref in resolve(step, rule["ref"]["path"]):
            if not isinstance(ref, (str, int)) or isinstance(ref, bool) or ref == "":
                continue
            creators = [
                (n, s) for n, s in steps
                if creator_matches(s) and any(_same(v, ref) for v in resolve(s, name_path))
            ]
            if not creators:
                if not ignore_missing:
                    yield [number], _describe(rule, f"step {number} refers to {ref!r}, which no step creates")
                continue
            for c_number, c_step in creators:
                if rule.get("before") and c_number >= number:
                    yield [number, c_number], _describe(rule, f"step {number} refers to {ref!r} before step {c_number} creates it")
                elif not holds(rule["require"], c_step):
                    yield [number, c_number], _describe(rule, f"step {c_number} creates {ref!r} (used by step {number}) without meeting {rule['id']}")


def _requires_step(rule, steps, rs):
    if_matches = make_matcher(rule["if"], rs)
    then_matches = make_matcher(rule["then"], rs)
    triggers = [n for n, s in steps if if_matches(s)]
    if not triggers:
        return
    thens = [n for n, s in steps if then_matches(s)]
    if rule.get("before"):
        thens = [n for n in thens if n < triggers[0]]
    if not thens:
        yield triggers[:1], _describe(rule, f"step {triggers[0]} needs a matching step" + (" before it" if rule.get("before") else ""))


def _forbid_together(rule, steps, rs):
    a_matches = make_matcher(rule["a"], rs)
    b_matches = make_matcher(rule["b"], rs)
    a_steps = [n for n, s in steps if a_matches(s)]
    b_steps = [n for n, s in steps if b_matches(s)]
    if rule.get("same_step"):
        both = [n for n in a_steps if n in b_steps]
        if both:
            yield both[:1], _describe(rule, f"step {both[0]} matches both forbidden shapes")
    elif a_steps and b_steps:
        yield sorted({a_steps[0], b_steps[0]}), _describe(rule, f"steps {a_steps[0]} and {b_steps[0]} must not appear in the same plan")


_HANDLERS = {
    "step_requires": _step_requires, "ref_requires": _ref_requires,
    "requires_step": _requires_step, "forbid_together": _forbid_together,
}


def violations_by_severity(violations: list[Violation]) -> tuple[list[Violation], list[Violation]]:
    return [v for v in violations if v.severity == "reject"], [v for v in violations if v.severity == "warn"]


def rule_ids(violations: list[Violation]) -> list[str]:
    seen: list[str] = []
    for v in violations:
        if v.rule_id not in seen:
            seen.append(v.rule_id)
    return seen


def reject_reason_for(rule_id: str) -> str:
    return f"{REASON_PREFIX}:{rule_id}"


# ── pending-row check ───────────────────────────────────────────────

SAMPLE_ROW_IDS = 5


def _plan_of(row: dict[str, Any]) -> Any:
    plan = row.get("plan_json")
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except ValueError:
            return None
    return plan if isinstance(plan, dict) else None


def select_pending_plan_rows(
    rows: list[dict[str, Any]], *, row_ids: list[int] | None = None, source: str | None = None,
    synth_job_id: int | None = None, max_rows: int | None = None,
) -> list[dict[str, Any]]:
    wanted = set(row_ids) if row_ids else None
    out = []
    for row in rows:
        if row.get("review_status") != "pending" or _plan_of(row) is None:
            continue
        if wanted is not None and row.get("id") not in wanted:
            continue
        if source and row.get("synth_source") != source:
            continue
        if synth_job_id is not None and row.get("synth_job_id") != synth_job_id:
            continue
        out.append(row)
        if max_rows is not None and len(out) >= max_rows:
            break
    return out


def check_rows(rows: list[dict[str, Any]], rules: RuleSet) -> dict[str, Any]:
    """Evaluate rows. Returns ``{checked, flagged, warned, by_rule, row_violations}``;
    ``by_rule`` maps rule id -> ``{severity, count, sample_row_ids}`` (every row counted once per rule);
    ``row_violations`` maps row id -> its violations (as dicts)."""
    by_rule: dict[str, dict[str, Any]] = {
        r["id"]: {"severity": r.get("severity") or "reject", "description": r.get("description") or "", "count": 0, "sample_row_ids": []}
        for r in rules.rules
    }
    row_violations: dict[int, list[dict[str, Any]]] = {}
    flagged = warned = 0
    for row in rows:
        found = check_plan(_plan_of(row), rules)
        if not found:
            continue
        row_violations[row["id"]] = [v.as_dict() for v in found]
        rejects, warns = violations_by_severity(found)
        flagged += bool(rejects)
        warned += bool(warns and not rejects)
        for rid in rule_ids(found):
            entry = by_rule[rid]
            entry["count"] += 1
            if len(entry["sample_row_ids"]) < SAMPLE_ROW_IDS:
                entry["sample_row_ids"].append(row["id"])
    return {
        "checked": len(rows), "flagged": flagged, "warned_only": warned,
        "by_rule": {k: v for k, v in by_rule.items() if v["count"]},
        "row_violations": row_violations,
    }
