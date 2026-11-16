"""SQL execution match: does a predicted SQL query return the same result as the gold query?

Opt-in per project (``Project.runtime_config["sql_execution"]``, see ``DEFAULT_CONFIG``). For every
eval row the gold SQL (the reference) and the predicted SQL are executed against each of the project's
SQLite files (``db_paths``) and the result sets compared; a row passes only when the results match on
ALL databases. Works for any project whose answers are SQL queries.

Safety (every execution, ``execute_query``):

- each database path must resolve (realpath) under the project's data directory
  (``DATA_DIR/projects/<id>``), and is opened read-only + immutable through a ``file:`` URI;
- an authorizer allows only ``SELECT`` / ``READ`` / ``FUNCTION`` (plus ``RECURSIVE`` for recursive
  CTEs): ``ATTACH``, ``PRAGMA``, writes, schema changes and transactions are denied;
- one statement only, extension loading off, a progress handler aborts the query after
  ``timeout_ms``, and at most ``row_cap`` rows are fetched (more = the query fails).

Comparison (``compare_results``): ordered when the gold query's OUTERMOST statement has ORDER BY
(``order="auto"``; ``"always"`` / ``"never"`` force it), else as a multiset of rows. Columns are matched
by position (names ignored); int / float values numerically (``math.isclose`` with ``float_rel_tol``);
NULL equals NULL; strings exactly.

Special rows: a gold query that fails leaves the row unscoreable (``pass`` None: out of every
denominator, counted in ``n_unscoreable``); a gold equal to ``refusal_marker`` passes iff the
prediction is the marker too (never executed); a prediction the authorizer blocks (or one that is not a
single read-only statement) fails and is counted in ``n_blocked``.

Everything here is a pure function of paths + strings so an offline data script can reuse it
(``score_row``, ``score_sql_block``, ``execute_query``, ``compare_results``, ``db_fingerprint``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

CONFIG_KEY = "sql_execution"
HANDLER_VERSION = "sql_exec/v1"
ORDER_MODES = ("auto", "always", "never")
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "db_paths": [],
    "timeout_ms": 2000,
    "row_cap": 10000,
    "float_rel_tol": 1e-6,
    "order": "auto",
    "refusal_marker": "NO_QUERY",
}
MAX_DB_PATHS = 20
TIMEOUT_MS_RANGE = (50, 60000)
ROW_CAP_RANGE = (1, 1_000_000)
FLOAT_TOL_RANGE = (0.0, 0.1)
MAX_TOP_ERRORS = 10
_PROGRESS_STEPS = 1000  # VM instructions between deadline checks

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
_ALLOWED_ACTIONS.add(_RECURSIVE)


class SqlPathError(ValueError):
    """A configured database path is missing or outside the allowed directory."""


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _short(text: Any, limit: int = 160) -> str:
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ── config ───────────────────────────────────────────────────────────────


def stored_config(runtime_config: Any) -> dict[str, Any]:
    raw = runtime_config.get(CONFIG_KEY) if isinstance(runtime_config, dict) else None
    return raw if isinstance(raw, dict) else {}


def _clamp(value: Any, lo: float, hi: float, *, integer: bool) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if value < lo or value > hi:
        return None
    return int(value) if integer else float(value)


def normalize_config(raw: Any) -> dict[str, Any]:
    """Stored settings with defaults applied (bad values fall back to the default)."""
    raw = raw if isinstance(raw, dict) else {}
    cfg = dict(DEFAULT_CONFIG)
    cfg["db_paths"] = []
    if isinstance(raw.get("enabled"), bool):
        cfg["enabled"] = raw["enabled"]
    if isinstance(raw.get("db_paths"), list):
        paths = [p.strip() for p in raw["db_paths"] if isinstance(p, str) and p.strip()]
        cfg["db_paths"] = list(dict.fromkeys(paths))[:MAX_DB_PATHS]
    for key, (lo, hi), integer in (
        ("timeout_ms", TIMEOUT_MS_RANGE, True),
        ("row_cap", ROW_CAP_RANGE, True),
        ("float_rel_tol", FLOAT_TOL_RANGE, False),
    ):
        value = _clamp(raw.get(key), lo, hi, integer=integer)
        if value is not None:
            cfg[key] = value
    if raw.get("order") in ORDER_MODES:
        cfg["order"] = raw["order"]
    marker = raw.get("refusal_marker")
    if isinstance(marker, str) and marker.strip() and len(marker) <= 64:
        cfg["refusal_marker"] = marker.strip()
    return cfg


def effective_config(runtime_config: Any) -> dict[str, Any] | None:
    """The settings with defaults applied, or ``None`` when SQL execution scoring is off (not enabled,
    or no database configured) -- every caller then behaves exactly as before this scorer existed."""
    cfg = normalize_config(stored_config(runtime_config))
    return cfg if cfg["enabled"] and cfg["db_paths"] else None


def project_data_dir(project_id: int) -> Path:
    from app.config import settings

    return Path(settings.DATA_DIR) / "projects" / str(project_id)


def resolve_db_paths(db_paths: list[str], allowed_root: str | os.PathLike[str]) -> list[str]:
    """Each path (relative paths are relative to ``allowed_root``) resolved with realpath; raises
    ``SqlPathError`` when one is outside ``allowed_root`` (symlinks followed) or is not a file."""
    root = os.path.realpath(os.fspath(allowed_root))
    out: list[str] = []
    for raw in db_paths:
        p = os.fspath(raw)
        joined = p if os.path.isabs(p) else os.path.join(root, p)
        real = os.path.realpath(joined)
        if os.path.commonpath([root, real]) != root or real == root:
            raise SqlPathError(f"database path {raw!r} is outside the project data directory")
        if not os.path.isfile(real):
            raise SqlPathError(f"database file {raw!r} does not exist")
        out.append(real)
    return out


def db_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def db_fingerprint(cfg: dict[str, Any], resolved_paths: list[str]) -> str:
    """Handler version + the canonical settings + the sha256 of each database file: a cached base-model
    result scored against other settings or other database bytes must not be reused."""
    cfg = normalize_config(cfg)
    parts = [json.dumps(cfg, sort_keys=True, separators=(",", ":"))]
    for p in resolved_paths:
        try:
            parts.append(db_sha256(p))
        except OSError:
            parts.append(f"unreadable:{p}")
    digest = hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]
    return f"{HANDLER_VERSION}+{digest}"


def sql_execution_fingerprint(runtime_config: Any, project_id: int) -> str | None:
    """``None`` when the scoring is off (results stay comparable with results from before it existed);
    else ``db_fingerprint`` over the resolved databases (``...+error`` when a path does not resolve).
    Blocking (hashes the files): run it in a worker thread."""
    cfg = effective_config(runtime_config)
    if cfg is None:
        return None
    try:
        paths = resolve_db_paths(cfg["db_paths"], project_data_dir(project_id))
    except SqlPathError:
        return f"{HANDLER_VERSION}+error"
    return db_fingerprint(cfg, paths)


# ── SQL text helpers ─────────────────────────────────────────────────────

_FENCE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\n?(.*?)\n?```\s*$", re.DOTALL)


def clean_sql(text: Any) -> str:
    """Strip whitespace, one surrounding code fence (```sql ... ```) and trailing semicolons."""
    s = str(text if text is not None else "").strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def _depth0_text(sql: str) -> str:
    """The SQL with comments removed, string / identifier literals blanked and anything inside
    parentheses blanked: what remains is the outermost statement's own keywords."""
    out: list[str] = []
    i, n, depth = 0, len(sql), 0
    while i < n:
        ch = sql[i]
        if ch == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j
            out.append(" ")
            continue
        if ch == "/" and sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
            continue
        if ch in ("'", '"', "`", "["):
            close = "]" if ch == "[" else ch
            j = i + 1
            while j < n:
                if sql[j] == close:
                    if close != "]" and j + 1 < n and sql[j + 1] == close:
                        j += 2
                        continue
                    break
                j += 1
            i = j + 1
            out.append(" x ")
            continue
        if ch == "(":
            depth += 1
            out.append(" ")
        elif ch == ")":
            depth = max(0, depth - 1)
            out.append(" ")
        elif depth == 0:
            out.append(ch)
        i += 1
    return "".join(out)


_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def has_outer_order_by(sql: str) -> bool:
    """True when the OUTERMOST query has ORDER BY (one inside a sub-query, CTE body or window is ignored)."""
    return bool(_ORDER_BY.search(_depth0_text(clean_sql(sql))))


def is_ordered(gold_sql: str, order: str) -> bool:
    if order == "always":
        return True
    if order == "never":
        return False
    return has_outer_order_by(gold_sql)


# ── execution ────────────────────────────────────────────────────────────


def execute_query(db_path: str, sql: str, *, timeout_ms: int = 2000, row_cap: int = 10000) -> dict[str, Any]:
    """Run one read-only statement against one SQLite file (``db_path`` must already be resolved and
    allowed). Returns ``{"ok", "rows", "columns", "error", "blocked", "timed_out", "over_cap"}``; never
    raises for a query problem."""
    result: dict[str, Any] = {
        "ok": False, "rows": None, "columns": None, "error": None,
        "blocked": False, "timed_out": False, "over_cap": False,
    }
    text = clean_sql(sql)
    if not text:
        result["error"] = "empty query"
        result["blocked"] = True
        return result
    denied: list[int] = []
    deadline = time.monotonic() + max(1, int(timeout_ms)) / 1000.0
    timed_out = [False]

    def authorizer(action: int, _a1: Any, _a2: Any, _db: Any, _trigger: Any) -> int:
        if action in _ALLOWED_ACTIONS:
            return sqlite3.SQLITE_OK
        denied.append(action)
        return sqlite3.SQLITE_DENY

    def progress() -> int:
        if time.monotonic() > deadline:
            timed_out[0] = True
            return 1
        return 0

    uri = Path(os.path.abspath(db_path)).as_uri() + "?mode=ro&immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False)
    except sqlite3.Error as exc:
        result["error"] = _short(f"cannot open database: {exc}")
        return result
    try:
        try:
            conn.enable_load_extension(False)
        except AttributeError:  # built without extension support: nothing to disable
            pass
        conn.set_authorizer(authorizer)
        conn.set_progress_handler(progress, _PROGRESS_STEPS)
        try:
            cur = conn.execute(text)
            if cur.description is None:
                result["blocked"] = True
                result["error"] = "not a query that returns rows"
                return result
            rows = cur.fetchmany(int(row_cap) + 1)
        except (sqlite3.Error, sqlite3.Warning) as exc:
            msg = str(exc)
            if timed_out[0]:
                result["timed_out"] = True
                result["error"] = f"timed out after {int(timeout_ms)} ms"
            elif denied or "not authorized" in msg or "one statement at a time" in msg:
                result["blocked"] = True
                result["error"] = _short(f"blocked: {msg}")
            else:
                result["error"] = _short(msg)
            return result
        if len(rows) > int(row_cap):
            result["over_cap"] = True
            result["error"] = f"more than {int(row_cap)} rows"
            return result
        result["ok"] = True
        result["rows"] = [tuple(r) for r in rows]
        result["columns"] = [d[0] for d in cur.description]
        return result
    finally:
        conn.close()


# ── comparison ───────────────────────────────────────────────────────────


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def values_equal(a: Any, b: Any, rel_tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if _is_num(a) and _is_num(b):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return math.isnan(fa) and math.isnan(fb)
        return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=1e-12)
    if _is_num(a) or _is_num(b):
        return False
    return type(a) is type(b) and a == b


def _rows_equal(r1: tuple, r2: tuple, rel_tol: float) -> bool:
    return len(r1) == len(r2) and all(values_equal(x, y, rel_tol) for x, y in zip(r1, r2))


def _canon(v: Any) -> Any:
    if _is_num(v):
        return ("n", float(v))
    if v is None:
        return ("0",)
    return (type(v).__name__, v)


def _sort_key(row: tuple) -> tuple:
    out = []
    for v in row:
        if v is None:
            out.append((0, 0))
        elif _is_num(v):
            out.append((1, float(v)))
        elif isinstance(v, str):
            out.append((2, v))
        else:
            out.append((3, bytes(v) if isinstance(v, (bytes, bytearray, memoryview)) else str(v)))
    return tuple(out)


def compare_results(pred_rows: list[tuple], gold_rows: list[tuple], *, ordered: bool, rel_tol: float) -> bool:
    """Same rows (ordered, or as a multiset); columns by position; numeric values within ``rel_tol``."""
    if len(pred_rows) != len(gold_rows):
        return False
    if ordered:
        return all(_rows_equal(p, g, rel_tol) for p, g in zip(pred_rows, gold_rows))
    if Counter(tuple(_canon(v) for v in r) for r in pred_rows) == Counter(
        tuple(_canon(v) for v in r) for r in gold_rows
    ):
        return True
    # Within-tolerance floats hash differently: compare the sorted rows pairwise.
    try:
        ps, gs = sorted(pred_rows, key=_sort_key), sorted(gold_rows, key=_sort_key)
    except TypeError:
        return False
    return all(_rows_equal(p, g, rel_tol) for p, g in zip(ps, gs))


# ── per-row scoring ──────────────────────────────────────────────────────


def score_row(pred_sql: Any, gold_sql: Any, db_paths: list[str], cfg: dict[str, Any]) -> dict[str, Any]:
    """Score one row against every (already resolved) database. ``pass`` is True / False, or None when
    the gold query fails (unscoreable). ``status``: match | mismatch | pred_error | blocked | timeout |
    over_cap | gold_error | refusal_correct | refusal_wrong | unexpected_refusal."""
    cfg = normalize_config(cfg)
    marker = cfg["refusal_marker"]
    pred, gold = clean_sql(pred_sql), clean_sql(gold_sql)
    out: dict[str, Any] = {
        "pass": False, "status": "mismatch", "ordered": None, "refusal": False,
        "blocked": False, "error": None, "per_db": [],
    }
    if gold == marker:
        out["refusal"] = True
        ok = pred == marker
        out.update({"pass": ok, "status": "refusal_correct" if ok else "refusal_wrong"})
        if not ok:
            out["error"] = "expected the refusal marker"
        return out
    if pred == marker:
        out.update({"status": "unexpected_refusal", "error": "refused, but the answer key has a query"})
        return out
    ordered = is_ordered(gold, cfg["order"])
    out["ordered"] = ordered
    kw = {"timeout_ms": cfg["timeout_ms"], "row_cap": cfg["row_cap"]}
    gold_results = []
    for path in db_paths:
        g = execute_query(path, gold, **kw)
        if not g["ok"]:
            out.update({"pass": None, "status": "gold_error",
                        "error": _short(f"answer-key query failed on {os.path.basename(path)}: {g['error']}")})
            out["per_db"].append({"db": os.path.basename(path), "pass": None, "gold_error": g["error"]})
            return out
        gold_results.append(g)
    passed = True
    for path, g in zip(db_paths, gold_results):
        name = os.path.basename(path)
        p = execute_query(path, pred, **kw)
        entry: dict[str, Any] = {"db": name, "pass": False, "gold_rows": len(g["rows"]), "pred_rows": None}
        if not p["ok"]:
            entry["pred_error"] = p["error"]
            out["per_db"].append(entry)
            out["blocked"] = bool(p["blocked"])
            status = (
                "blocked" if p["blocked"] else "timeout" if p["timed_out"]
                else "over_cap" if p["over_cap"] else "pred_error"
            )
            out.update({"pass": False, "status": status, "error": _short(f"{name}: {p['error']}")})
            return out
        entry["pred_rows"] = len(p["rows"])
        ok = compare_results(p["rows"], g["rows"], ordered=ordered, rel_tol=cfg["float_rel_tol"])
        entry["pass"] = ok
        out["per_db"].append(entry)
        if not ok:
            passed = False
            out["error"] = out["error"] or f"{name}: results differ"
            break  # all must match; no need to run the rest
    out["pass"] = passed
    out["status"] = "match" if passed else "mismatch"
    return out


def score_sql_block(
    predictions: list[dict[str, Any]],
    db_paths: list[str],
    cfg: dict[str, Any],
    *,
    pred_key: str = "prediction",
    gold_key: str = "reference",
) -> dict[str, Any]:
    """Score every prediction dict in place (``row_sql_exec`` = the ``score_row`` result,
    ``row_sql_exec_pass`` = its pass flag) and return the aggregate block. ``db_paths`` must already be
    resolved (``resolve_db_paths``). Blocking: run it in a worker thread."""
    cfg = normalize_config(cfg)
    counts: Counter[str] = Counter()
    errors_top: Counter[str] = Counter()
    n_rows = n_pass = n_scored = n_refusal = n_refusal_ok = 0
    for prediction in predictions:
        if not isinstance(prediction, dict) or prediction.get("row_skipped"):
            continue
        n_rows += 1
        res = score_row(prediction.get(pred_key), prediction.get(gold_key), db_paths, cfg)
        prediction["row_sql_exec"] = res
        prediction["row_sql_exec_pass"] = res["pass"]
        counts[res["status"]] += 1
        if res["blocked"]:
            counts["_blocked"] += 1  # counted before the row is failed, so it is reportable
        if res["refusal"]:
            n_refusal += 1
            n_refusal_ok += int(bool(res["pass"]))
        if res["pass"] is None:
            continue
        n_scored += 1
        n_pass += int(bool(res["pass"]))
        if res["error"]:
            errors_top[re.sub(r"\d+", "N", res["error"])[:120]] += 1
    rate = _rate(n_pass, n_scored)
    return {
        "status": "scored",
        "exec_match_rate": rate,
        "n_rows": n_rows,
        "n_scored": n_scored,
        "n_passed": n_pass,
        "n_unscoreable": counts["gold_error"],
        "n_blocked": counts["_blocked"],
        "n_timeout": counts["timeout"],
        "n_over_cap": counts["over_cap"],
        "n_pred_error": counts["pred_error"],
        "n_mismatch": counts["mismatch"],
        "n_refusal_rows": n_refusal,
        "refusal_correct_rate": _rate(n_refusal_ok, n_refusal),
        "databases": [os.path.basename(p) for p in db_paths],
        "order": cfg["order"],
        "timeout_ms": cfg["timeout_ms"],
        "row_cap": cfg["row_cap"],
        "float_rel_tol": cfg["float_rel_tol"],
        "refusal_marker": cfg["refusal_marker"],
        "errors_top": [{"message": m, "count": c} for m, c in errors_top.most_common(MAX_TOP_ERRORS)],
    }


def score_for_project(
    predictions: list[dict[str, Any]], runtime_config: Any, project_id: int
) -> tuple[dict[str, Any] | None, str | None]:
    """``(block, fingerprint)`` for an eval of ``project_id``; ``(None, None)`` when the scoring is off.
    A database path that does not resolve gives ``{"status": "config_error", "error": ...}`` and scores
    nothing. Blocking: run it in a worker thread."""
    cfg = effective_config(runtime_config)
    if cfg is None:
        return None, None
    try:
        paths = resolve_db_paths(cfg["db_paths"], project_data_dir(project_id))
    except SqlPathError as exc:
        return {"status": "config_error", "error": str(exc)}, f"{HANDLER_VERSION}+error"
    block = score_sql_block(predictions, paths, cfg)
    return block, db_fingerprint(cfg, paths)


def preview_fields(prediction: dict[str, Any]) -> dict[str, Any]:
    """The per-row keys a result preview carries (only for rows this scorer touched)."""
    if "row_sql_exec_pass" not in prediction:
        return {}
    return {"row_sql_exec": prediction.get("row_sql_exec"), "row_sql_exec_pass": prediction.get("row_sql_exec_pass")}
