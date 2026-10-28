"""Helpers for the `structured-plan` `scenario_grid` playbook.

A *structured-plan row* is a scenario, optional reference passages and the plan the model must write::

    {"row_kind": "structured_plan", "scenario": str,
     "passages": [{"id": "P1", "text", "source", "passage_id"}],     # optional
     "plan_json": {...},                                              # validates against the saved schema
     "grid_cell": {"region": "north", "tier": "gold"}}                # the matrix cell it came from

The playbook reads a *matrix* (a markdown pipe table or CSV): every column is a dimension, its
non-empty cells are the dimension's values, and the grid is the cartesian product of the dimensions.
Large grids are sampled deterministically (by ``seed``), keeping the product order.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import random
import re
from typing import Any

RECIPE_ID = "structured-plan"
MARKER_PREFIX = "[brewslm:structured-plan:scenario_grid:"
STAGE_SCENARIOS = "scenarios"
STAGE_ANSWERS = "answers"

MAX_MATRIX_CHARS = 200_000
MAX_DIMENSIONS = 10
MAX_VALUES_PER_DIMENSION = 100
MAX_CELLS_LIMIT = 200
DEFAULT_MAX_CELLS = 20
MAX_SCENARIOS_PER_CELL = 5
DEFAULT_SCENARIOS_PER_CELL = 1
MAX_EXTRA_INSTRUCTIONS = 2000
CELLS_PER_CALL = 10
MAX_TASKS_PER_ANSWER_CALL = 5
MAX_ANSWER_PROMPT_CHARS = 40_000
MAX_PASSAGES = 10
MATRIX_FORMATS = ("auto", "markdown", "csv")


class MatrixError(ValueError):
    """The matrix text is not a usable grid (the message names the problem)."""


def marker(stage: str) -> str:
    return f"{MARKER_PREFIX}{stage}]"


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def norm_ws(text: Any) -> str:
    return " ".join(str(text if text is not None else "").split())


def tag_text(text: Any, limit: int = 300) -> str:
    """Text safe inside a one-line prompt block: whitespace collapsed, angle brackets removed."""
    return norm_ws(str(text if text is not None else "").replace("<", " ").replace(">", " "))[:limit]


# ─────────────────────────── matrix parsing ───────────────────────────


def _split_markdown_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells = re.split(r"(?<!\\)\|", line)
    return [c.replace("\\|", "|").strip() for c in cells]


_SEPARATOR_CELL = re.compile(r"^:?-{1,}:?$")


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(_SEPARATOR_CELL.match(c.strip()) for c in cells if c.strip()) and any(c.strip() for c in cells)


def _looks_markdown(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or "|" not in lines[0]:
        return False
    if lines[0].lstrip().startswith("|"):
        return True
    return len(lines) > 1 and _is_separator(_split_markdown_row(lines[1]))


def _table_rows(text: str, fmt: str) -> list[list[str]]:
    if fmt == "auto":
        fmt = "markdown" if _looks_markdown(text) else "csv"
    if fmt == "markdown":
        rows = []
        for line in text.splitlines():
            if not line.strip() or "|" not in line:
                continue
            cells = _split_markdown_row(line)
            if _is_separator(cells):
                continue
            rows.append(cells)
        return rows
    try:
        return [[c.strip() for c in row] for row in csv.reader(io.StringIO(text)) if any(c.strip() for c in row)]
    except csv.Error as exc:
        raise MatrixError(f"The matrix is not valid CSV: {exc}") from None


def parse_matrix(text: Any, fmt: str = "auto") -> dict[str, Any]:
    """``{"dimensions": [{"name", "values"}]}`` from a markdown pipe table or CSV.

    Each column is a dimension: the header is its name, the non-empty cells below it are its values
    (de-duplicated, in order). Raises ``MatrixError`` (a ValueError) with a clear message."""
    if fmt not in MATRIX_FORMATS:
        raise MatrixError(f"matrix_format must be one of {', '.join(MATRIX_FORMATS)}.")
    if not isinstance(text, str) or not text.strip():
        raise MatrixError("The matrix is empty: give a table with a header row of dimension names.")
    if len(text) > MAX_MATRIX_CHARS:
        raise MatrixError(f"The matrix is {len(text)} characters (max {MAX_MATRIX_CHARS}).")
    rows = _table_rows(text.replace("\r\n", "\n").lstrip("﻿"), fmt)
    if not rows:
        raise MatrixError("The matrix has no header row: the first row must name the dimensions.")
    header = [h.strip() for h in rows[0]]
    while header and not header[-1]:
        header.pop()
    if not header:
        raise MatrixError("The matrix has no header row: the first row must name the dimensions.")
    if any(not h for h in header):
        raise MatrixError("Every column of the matrix needs a header (dimension name); found an empty one.")
    if len(set(header)) != len(header):
        dup = next(h for h in header if header.count(h) > 1)
        raise MatrixError(f"Dimension name {dup!r} appears more than once in the header.")
    if len(header) > MAX_DIMENSIONS:
        raise MatrixError(f"The matrix has {len(header)} dimensions (max {MAX_DIMENSIONS}).")
    dims: list[dict[str, Any]] = []
    for col, name in enumerate(header):
        values: dict[str, None] = {}
        for row in rows[1:]:
            cell = norm_ws(row[col]) if col < len(row) else ""
            if cell:
                values.setdefault(cell, None)
        if not values:
            raise MatrixError(f"Dimension {name!r} has no values: add at least one non-empty cell below its header.")
        if len(values) > MAX_VALUES_PER_DIMENSION:
            raise MatrixError(f"Dimension {name!r} has {len(values)} values (max {MAX_VALUES_PER_DIMENSION}).")
        dims.append({"name": name, "values": list(values)})
    return {"dimensions": dims}


def grid_size(dimensions: list[dict[str, Any]]) -> int:
    size = 1
    for dim in dimensions:
        size *= len(dim["values"])
    return size


def build_grid(dimensions: list[dict[str, Any]], max_cells: int, seed: int = 0) -> list[dict[str, str]]:
    """The cells of the cartesian product, in product order (last dimension fastest).

    Over ``max_cells`` the grid is sampled: ``random.Random(seed)`` picks ``max_cells`` positions
    of the product, kept in ascending (product) order, so a fixed seed always gives the same cells.
    The product is never materialised in full."""
    sizes = [len(d["values"]) for d in dimensions]
    total = grid_size(dimensions)
    if total <= max_cells:
        positions: list[int] = list(range(total))
    else:
        rng = random.Random(seed)
        if total < 2**62:
            positions = sorted(rng.sample(range(total), max_cells))
        else:  # range() cannot hold the product: draw distinct positions one by one
            chosen: set[int] = set()
            while len(chosen) < max_cells:
                chosen.add(rng.randrange(total))
            positions = sorted(chosen)
    cells = []
    for pos in positions:
        rest, picks = pos, []
        for size in reversed(sizes):
            rest, i = divmod(rest, size)
            picks.append(i)
        picks.reverse()
        cells.append({d["name"]: d["values"][i] for d, i in zip(dimensions, picks)})
    return cells


# alias used by tests that want the whole product for a tiny grid
def full_product(dimensions: list[dict[str, Any]]) -> list[dict[str, str]]:
    names = [d["name"] for d in dimensions]
    return [dict(zip(names, combo)) for combo in itertools.product(*(d["values"] for d in dimensions))]


# ─────────────────────────── param helpers ───────────────────────────


def whole(value: Any) -> int | None:
    if is_int(value):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    return None


def grid_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """The scenario_grid params with defaults applied (assumes they were range-checked)."""
    params = params or {}
    return {
        "matrix_text": params.get("matrix_text"),
        "matrix_format": params.get("matrix_format") or "auto",
        "max_cells": whole(params.get("max_cells")) or DEFAULT_MAX_CELLS,
        "scenarios_per_cell": whole(params.get("scenarios_per_cell")) or DEFAULT_SCENARIOS_PER_CELL,
        "seed": whole(params.get("seed")) or 0,
        "extra_instructions": (params.get("extra_instructions") or "").strip(),
    }


def check_params(params: dict[str, Any]) -> tuple[str, str] | None:
    """``(param_name, message)`` for the first bad scenario_grid param, else None. Also parses the
    matrix, so a malformed table is reported before any work starts."""
    text = params.get("matrix_text")
    if not isinstance(text, str) or not text.strip():
        return "matrix_text", "params.matrix_text is required: a markdown table or CSV whose header row names the dimensions."
    if len(text) > MAX_MATRIX_CHARS:
        return "matrix_text", f"params.matrix_text is {len(text)} characters (max {MAX_MATRIX_CHARS})."
    fmt = params.get("matrix_format")
    if fmt is not None and fmt not in MATRIX_FORMATS:
        return "matrix_format", "params.matrix_format must be auto, markdown or csv."
    for name, low, high in (("max_cells", 1, MAX_CELLS_LIMIT), ("scenarios_per_cell", 1, MAX_SCENARIOS_PER_CELL)):
        if params.get(name) is not None:
            n = whole(params[name])
            if n is None or not low <= n <= high:
                return name, f"params.{name} must be a whole number from {low} to {high}."
    if params.get("seed") is not None and whole(params["seed"]) is None:
        return "seed", "params.seed must be a whole number."
    extra = params.get("extra_instructions")
    if extra is not None and (not isinstance(extra, str) or len(extra) > MAX_EXTRA_INSTRUCTIONS):
        return "extra_instructions", f"params.extra_instructions must be text of at most {MAX_EXTRA_INSTRUCTIONS} characters."
    try:
        parse_matrix(text, fmt or "auto")
    except MatrixError as exc:
        return "matrix_text", str(exc)
    return None


# ─────────────────────────── passages / output parsing ───────────────────────────


def normalise_passages(passages: Any, *, limit: int = MAX_PASSAGES) -> list[dict[str, Any]]:
    """Canonical passages ``{id: P1.., text, source, passage_id}`` in order; blanks dropped."""
    out: list[dict[str, Any]] = []
    if not isinstance(passages, list):
        return out
    for item in passages:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            continue
        out.append({
            "id": f"P{len(out) + 1}", "text": item["text"],
            "source": item.get("source") if isinstance(item.get("source"), str) else None,
            "passage_id": item.get("passage_id") if isinstance(item.get("passage_id"), str) else None,
        })
        if len(out) >= limit:
            break
    return out


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


# A long, deeply nested reply sometimes miscounts its closing brackets (one ``}`` too many after a
# request body, or one missing) — a third of a structured-plan batch was lost to it (2026-10-28).
MAX_BRACKET_REPAIRS = 3
_CLOSERS = {"{": "}", "[": "]"}


def _open_brackets(text: str, end: int) -> list[str]:
    """The brackets still open at ``text[:end]`` (outside strings), innermost last."""
    stack: list[str] = []
    in_str = esc = False
    for ch in text[:end]:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in _CLOSERS:
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    return stack


def repair_json_brackets(text: str, max_edits: int = MAX_BRACKET_REPAIRS) -> Any | None:
    """``text`` parsed after at most ``max_edits`` bracket fixes at the parse-error position: drop
    the stray closer just before it, or insert the closer the innermost open bracket needs. None
    when no such fix parses. Tries at most 2**max_edits parses, so it stays cheap."""
    frontier = [text]
    for _ in range(max_edits):
        nxt: list[str] = []
        for cand in frontier:
            try:
                return json.loads(cand)
            except json.JSONDecodeError as exc:
                pos = exc.pos
            k = len(cand[:pos].rstrip())
            k = max(cand.rfind("}", 0, k), cand.rfind("]", 0, k))
            if k >= 0:
                nxt.append(cand[:k] + cand[k + 1 :])
            stack = _open_brackets(cand, pos)
            if stack:
                nxt.append(cand[:pos] + _CLOSERS[stack[-1]] + cand[pos:])
        frontier = nxt
    for cand in frontier:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _items_of(obj: Any) -> list[Any] | None:
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        return obj["items"]
    return obj if isinstance(obj, list) else None


def parse_json_items(raw: str, notes: dict[str, Any] | None = None) -> list[Any] | None:
    """The ``items`` list of a teacher reply: a JSON object ``{"items": [...]}`` (fenced or with prose
    around it), a bare JSON list, or JSON lines. A reply whose only fault is a miscounted closing
    bracket is repaired (``repair_json_brackets``; counted in ``notes["repaired_json"]``) — the
    caller's schema check still applies to what comes out. None when nothing parses."""
    text = _FENCE_RE.sub("", (raw or "").strip()).strip()
    candidates = [text]
    first, last = text.find("{"), text.rfind("}")
    if 0 <= first < last:
        candidates.append(text[first : last + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (ValueError, TypeError):
            continue
        items = _items_of(obj)
        if items is not None:
            return items
    if 0 <= first < last:
        items = _items_of(repair_json_brackets(text[first : last + 1]))
        if items is not None:
            if notes is not None:
                notes["repaired_json"] = int(notes.get("repaired_json") or 0) + 1
            return items
    lines = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{"):
            try:
                lines.append(json.loads(line))
            except ValueError:
                continue
    return lines or None
