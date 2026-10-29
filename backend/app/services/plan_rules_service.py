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

Every op except ``not_exists`` FAILS when the path resolves to nothing (``exists`` and ``not_exists``
only look at presence); a leaf condition's optional ``missing: "pass"`` makes a value op pass instead
(default ``"fail"``). ``regex`` applies to at most the first 10,000 characters of a value; patterns are
capped at 200 characters and nested-quantifier patterns (``(a+)+``) are refused. A rule may carry its own
``steps_path`` (default: the document's), so rules can target another list such as verification items.
Steps are ordered by their position in the list; the model's ``step`` number only appears in messages.

* ``step_requires``  ``{match, require}`` - every matching step must satisfy ``require``.
* ``ref_requires``   ``{match, ref: {path}, creator: {match, name_path}, require, before?,
  on_missing?}`` - each value a matching step holds at ``ref.path`` (a list value: each item is a reference) names an object created by
  a step matching ``creator.match`` (its ``name_path`` values); every such creating step must
  satisfy ``require``. A reference no step creates is a violation (``on_missing: "violate"``,
  the default) or skipped (``"ignore"``). ``before: true`` demands a creator earlier in the list (with several creators, only the
  earlier ones must satisfy ``require``; it is a violation only when none precedes the reference).
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
    "steps_path": _PATH,
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
                    "missing": {"enum": ["pass", "fail"]},
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

MAX_REGEX_CHARS = 200
MAX_REGEX_INPUT = 10_000
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)\s*(?:[+*]|\{\d*,\d*\})")

_BASE_SCHEMA = {**RULES_SCHEMA, "properties": {**RULES_SCHEMA["properties"], "rules": {"type": "array", "minItems": 1, "maxItems": MAX_RULES, "items": {"type": "object"}}}}
_BASE_VALIDATOR = Draft202012Validator(_BASE_SCHEMA)
_RULE_VALIDATORS = {
    branch["properties"]["type"]["const"]: Draft202012Validator({**branch, "$defs": RULES_SCHEMA["$defs"]})
    for branch in RULES_SCHEMA["properties"]["rules"]["items"]["oneOf"]
}


def _first_error(validator: Draft202012Validator, instance: Any, prefix: str) -> PlanRulesError | None:
    errors = sorted(validator.iter_errors(instance), key=lambda e: len(list(e.absolute_path)))
    if not errors:
        return None
    top = errors[0]
    parts = list(top.absolute_path)
    detail = top.message
    if top.validator == "oneOf" and top.context:  # a condition: report the deepest branch's complaint
        deepest = max(top.context, key=lambda e: len(list(e.absolute_path)))
        detail, parts = deepest.message, parts + list(deepest.absolute_path)
    pointer = prefix + "".join(f"/{p}" for p in parts)
    return PlanRulesError(f"{pointer or '(root)'}: {detail[:200]}", pointer)


def _check_condition(cond: dict[str, Any], pointer: str) -> None:
    """Per-op value types (the schema leaves ``value`` free)."""
    for key in ("all", "any"):
        for i, sub in enumerate(cond.get(key) or []):
            _check_condition(sub, f"{pointer}/{key}/{i}")
    if "not" in cond:
        _check_condition(cond["not"], f"{pointer}/not")
    op = cond.get("op")
    if op is None:
        return
    value = cond.get("value")
    problem = None
    if op in ("equals", "not_equals", "contains") and "value" not in cond:
        problem = f"`{op}` needs a `value`"
    elif op in ("in", "not_in") and not isinstance(value, list):
        problem = f"`{op}` needs a list `value`"
    elif op in ("max", "min") and _num(value) is None:
        problem = f"`{op}` needs a numeric `value`"
    elif op == "regex":
        if not isinstance(value, str):
            problem = "`regex` needs a string `value`"
        else:
            problem = _regex_problem(value)
    if problem:
        raise PlanRulesError(f"{pointer}: {problem}.", pointer)


def _regex_problem(pattern: str) -> str | None:
    if len(pattern) > MAX_REGEX_CHARS:
        return f"regular expression is longer than {MAX_REGEX_CHARS} characters"
    if _NESTED_QUANTIFIER.search(pattern):
        return f"regular expression {pattern!r} has a nested quantifier (it can take exponential time)"
    try:
        re.compile(pattern)
    except re.error as exc:
        return f"bad regular expression {pattern!r}: {exc}"
    return None


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
    err = _first_error(_BASE_VALIDATOR, raw, "")
    if err:
        raise err
    seen: set[str] = set()
    for i, rule in enumerate(raw["rules"]):
        validator = _RULE_VALIDATORS.get(rule.get("type")) if isinstance(rule.get("type"), str) else None
        if validator is None:
            raise PlanRulesError(f"/rules/{i}/type: must be one of {', '.join(sorted(_RULE_VALIDATORS))}.", f"/rules/{i}/type")
        err = _first_error(validator, rule, f"/rules/{i}")
        if err:
            raise err
        if rule["id"] in seen:
            raise PlanRulesError(f"/rules/{i}/id: duplicate rule id {rule['id']!r}.", f"/rules/{i}/id")
        seen.add(rule["id"])
        for ptr, cond in _leaf_condition_nodes(rule, f"/rules/{i}"):
            _check_condition(cond, ptr)
        for ptr, pattern in _match_regexes(rule, f"/rules/{i}"):
            problem = _regex_problem(pattern)
            if problem:
                raise PlanRulesError(f"{ptr}: {problem}.", ptr)
    paths = {**DEFAULT_PATHS, **(raw.get("paths") or {})}
    return RuleSet(raw, paths["tool"], paths["operation"], raw.get("steps_path") or "steps")


_CONDITION_KEYS = ("require", "where")


def _leaf_condition_nodes(node: Any, pointer: str) -> Iterable[tuple[str, dict[str, Any]]]:
    """Every top-level condition object (under ``require`` / ``where``) with its pointer."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _CONDITION_KEYS and isinstance(v, dict):
                yield f"{pointer}/{k}", v
            else:
                yield from _leaf_condition_nodes(v, f"{pointer}/{k}")


def _match_regexes(node: Any, pointer: str) -> Iterable[tuple[str, str]]:
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "operation_regex" and isinstance(v, str):
                yield f"{pointer}/{k}", v
            else:
                yield from _match_regexes(v, f"{pointer}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _match_regexes(v, f"{pointer}/{i}")


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
        return isinstance(value, str) and isinstance(expected, str) and re.search(expected, value[:MAX_REGEX_INPUT]) is not None
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
    """Whether a condition holds on ``node`` (a step, or any JSON value). A path that resolves to
    nothing fails every op except ``not_exists`` (a leaf's ``missing: "pass"`` flips value ops)."""
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
        return cond.get("missing") == "pass"
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
        if op_re is not None and not any(isinstance(v, str) and op_re.search(v[:MAX_REGEX_INPUT]) for v in resolve(step, rs.operation_path)):
            return False
        return where is None or holds(where, step)

    return matches


Step = tuple[int, int, dict[str, Any]]  # (position in the list, number shown in messages, the step)


def _steps_of(plan: Any, steps_path: str) -> list[Step]:
    found = resolve(plan, steps_path)
    steps = found[0] if found and isinstance(found[0], list) else []
    out: list[Step] = []
    for pos, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        number = step.get("step")
        out.append((pos, number if isinstance(number, int) and not isinstance(number, bool) else pos + 1, step))
    return out


def _describe(rule: dict[str, Any], default: str) -> str:
    return rule.get("message") or rule.get("description") or default


# ── evaluation ──────────────────────────────────────────────────────

def check_plan(plan: Any, rules: RuleSet) -> list[Violation]:
    """Every violation of ``rules`` in one plan, in rule order. A plan with no step list has none."""
    out: list[Violation] = []
    cache: dict[str, list[Step]] = {}
    for rule in rules.rules:
        path = rule.get("steps_path") or rules.steps_path
        if path not in cache:
            cache[path] = _steps_of(plan, path)
        severity = rule.get("severity") or "reject"
        for numbers, message in _HANDLERS[rule["type"]](rule, cache[path], rules):
            out.append(Violation(rule["id"], severity, tuple(numbers), message))
    return out


def _step_requires(rule, steps, rs):
    matches = make_matcher(rule["match"], rs)
    for _pos, number, step in steps:
        if matches(step) and not holds(rule["require"], step):
            yield [number], _describe(rule, f"step {number} does not meet {rule['id']}")


def _references(step: dict[str, Any], path: str) -> list[Any]:
    """Reference values at ``path``; a list value contributes each of its items (one level)."""
    out: list[Any] = []
    for value in resolve(step, path):
        for item in (value if isinstance(value, list) else [value]):
            if isinstance(item, (str, int)) and not isinstance(item, bool) and item != "":
                out.append(item)
    return out


def _ref_requires(rule, steps, rs):
    matches = make_matcher(rule["match"], rs)
    creator_matches = make_matcher(rule["creator"]["match"], rs)
    name_path = rule["creator"]["name_path"]
    ignore_missing = rule.get("on_missing") == "ignore"
    for pos, number, step in steps:
        if not matches(step):
            continue
        for ref in _references(step, rule["ref"]["path"]):
            creators = [
                (c_pos, c_number, c_step) for c_pos, c_number, c_step in steps
                if creator_matches(c_step) and any(_same(v, ref) for v in resolve(c_step, name_path))
            ]
            if not creators:
                if not ignore_missing:
                    yield [number], _describe(rule, f"step {number} refers to {ref!r}, which no step creates")
                continue
            if rule.get("before"):
                creators = [c for c in creators if c[0] < pos]
                if not creators:
                    yield [number], _describe(rule, f"step {number} refers to {ref!r} before any step creates it")
                    continue
            for _c_pos, c_number, c_step in creators:
                if not holds(rule["require"], c_step):
                    yield [number, c_number], _describe(rule, f"step {c_number} creates {ref!r} (used by step {number}) without meeting {rule['id']}")


def _requires_step(rule, steps, rs):
    if_matches = make_matcher(rule["if"], rs)
    then_matches = make_matcher(rule["then"], rs)
    triggers = [(p, n) for p, n, s in steps if if_matches(s)]
    if not triggers:
        return
    thens = [p for p, _n, s in steps if then_matches(s)]
    if rule.get("before"):
        thens = [p for p in thens if p < triggers[0][0]]
    if not thens:
        yield [triggers[0][1]], _describe(rule, f"step {triggers[0][1]} needs a matching step" + (" before it" if rule.get("before") else ""))


def _forbid_together(rule, steps, rs):
    a_matches = make_matcher(rule["a"], rs)
    b_matches = make_matcher(rule["b"], rs)
    a_steps = [(p, n) for p, n, s in steps if a_matches(s)]
    b_steps = [(p, n) for p, n, s in steps if b_matches(s)]
    if rule.get("same_step"):
        both = [n for p, n in a_steps if any(p == q for q, _ in b_steps)]
        if both:
            yield both[:1], _describe(rule, f"step {both[0]} matches both forbidden shapes")
    elif a_steps and b_steps:
        a, b = a_steps[0][1], b_steps[0][1]
        yield sorted({a, b}), _describe(rule, f"steps {a} and {b} must not appear in the same plan")


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
