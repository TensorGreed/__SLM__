"""``multi_option`` scoring: the model answers with a *menu* of alternative fixes -- N options, each
built on a different strategy and delivered in M concrete formats -- and every option is judged by
an operator-registered **option verifier** (``external_checks``), so BrewSLM stays generic: the
domain logic (does the fix apply, does it change the behaviour it should, is the rest preserved, do
the formats end in the same state) lives in the verifier.

Model output (JSON; option order is the model's ranking, option 0 = the top option)::

    {"explanation": "...",
     "options": [{"strategy": "guard-clause", "title": "...", "rationale": "...", "tradeoffs": "...",
                  "rollback": "...", "fixes": {"format_a": "...", "format_b": "..."}}]}

Project configuration (``runtime_config["multi_option"]``, overlaying the prepared manifest's
``output_schema["multi_option"]``; ``normalize_config``)::

    {"formats": ["format_a", "format_b"], "min_options": 2, "max_options": 3,
     "strategy_taxonomy": ["guard-clause", "validate-at-boundary", ...],
     "verifier": {"check_id": "<an operator-registered option verifier>"},
     "require_agreement": true, "row_pass": "top_option" | "any_option" | "all_options",
     "instruction": "optional task sentence for the prompt"}

Row context (copied onto the prediction as ``multi_option_row``): ``context`` (what the prompt
showed), ``verifier_payload`` (an opaque object handed to the verifier -- e.g. the starting artifact
and hidden assertions; BrewSLM never interprets it and never renders it into a prompt or shows it to
checked-decoding verifiers) and optional ``reference_options`` (display only).

An option passes when every configured format's verifier verdict passes, the formats agree (when
``require_agreement`` and >= 2 formats) and the verifier's own ``option_passes`` is not false. At
least one format is required (``normalize_config`` reports an enabled config without one); with no
format the option is decided only by an explicit ``option_passes`` boolean.

Verifier errors split in two (never a raised eval):

* **Infrastructure** -- the sandbox could not launch the verifier, or the verifier is not registered /
  not allow-listed / not installed (or no verifier ran). The option gets ``passes = None`` and
  ``error``; a row whose ``row_pass`` rule cannot be decided without it is excluded from every rate
  (``row_verifier_error``) and counted in ``verifier_error_rows``. A row with no ``verifier_payload``
  is the same case (the data is at fault, not the option): the verifier is not called and
  ``rows_missing_verifier_payload`` counts it. A verifier that needs nothing from the row takes an
  explicit empty payload (``{}``).
* **Option-level** -- a per-option timeout, a non-zero exit, unparseable output, output over the size
  cap, or a verdict that leaves a format / the agreement undecided. The model's option caused it, so
  THAT option fails (``passes = False``) with ``failure_reason`` (``verifier_timeout``,
  ``verifier_crash``, ``verifier_bad_output``, ``verifier_output_too_large``, ``verifier_undecided``);
  ``option_verifier_failures`` counts them.

Every ``row_pass`` rule also needs at least ``min_options`` options. Under ``top_option`` the row is
decided by option 0 alone, so an error on another option never excludes it.

The expensive part (one verifier call per option) is ``analyze_row`` (cached on the prediction as
``_multi_option_analysis``); ``precompute`` runs it for every row in a worker thread before scoring.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

SCORING_MODE = "multi_option"
CONFIG_KEY = "multi_option"
ROW_PASS_MODES = ("top_option", "any_option", "all_options")
MAX_OPTIONS_LIMIT = 10
NO_PAYLOAD_ERROR = "this row has no verifier_payload, so the option verifier was not run"
MAX_FORMATS = 6
MAX_TAXONOMY = 200
MAX_FIX_CHARS = 6000
MAX_TEXT_CHARS = 1500
_TEXT_FIELDS = ("title", "rationale", "tradeoffs", "rollback")

DEFAULTS: dict[str, Any] = {
    "formats": [],
    "min_options": 1,
    "max_options": 3,
    "strategy_taxonomy": [],
    "verifier": None,
    "require_agreement": False,
    "row_pass": "top_option",
    "instruction": None,
}

VerifyFn = Callable[[Any, Any, list[str]], dict[str, Any]]


# ── configuration ────────────────────────────────────────────────────────


def _ident_list(value: Any, name: str, limit: int, errors: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        errors.append(f"{name} must be a list of non-empty strings")
        return []
    out = list(dict.fromkeys(v.strip() for v in value))
    if len(out) > limit:
        errors.append(f"{name}: at most {limit} entries")
        out = out[:limit]
    return out


def normalize_config(raw: Any) -> tuple[dict[str, Any], list[str]]:
    """``(config, errors)``: the full config with defaults filled in; ``errors`` lists what was
    invalid (the invalid value is replaced by its default)."""
    errors: list[str] = []
    raw = raw if isinstance(raw, dict) else {}
    cfg = dict(DEFAULTS)
    cfg["formats"] = _ident_list(raw.get("formats"), "formats", MAX_FORMATS, errors)
    cfg["strategy_taxonomy"] = _ident_list(raw.get("strategy_taxonomy"), "strategy_taxonomy", MAX_TAXONOMY, errors)
    for key in ("min_options", "max_options"):
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_OPTIONS_LIMIT:
            errors.append(f"{key} must be an integer from 1 to {MAX_OPTIONS_LIMIT}")
        else:
            cfg[key] = value
    if cfg["min_options"] > cfg["max_options"]:
        errors.append("min_options must not exceed max_options")
        cfg["min_options"], cfg["max_options"] = DEFAULTS["min_options"], max(DEFAULTS["max_options"], cfg["max_options"])
    verifier = raw.get("verifier")
    if verifier is not None:
        cid = verifier.get("check_id") if isinstance(verifier, dict) else None
        if not isinstance(cid, str) or not cid.strip():
            errors.append("verifier must be {\"check_id\": \"<registered option verifier id>\"}")
        else:
            # Only an id: a project never supplies a command line.
            cfg["verifier"] = {"check_id": cid.strip()}
    if raw.get("require_agreement") is not None:
        if not isinstance(raw["require_agreement"], bool):
            errors.append("require_agreement must be true or false")
        else:
            cfg["require_agreement"] = raw["require_agreement"]
    if raw.get("row_pass") is not None:
        if raw["row_pass"] not in ROW_PASS_MODES:
            errors.append(f"row_pass must be one of: {', '.join(ROW_PASS_MODES)}")
        else:
            cfg["row_pass"] = raw["row_pass"]
    if raw.get("enabled") is True and not cfg["formats"]:
        # An option with no format to check would pass vacuously.
        errors.append("formats: an enabled multi_option config needs at least one format")
    instruction = raw.get("instruction")
    if instruction is not None:
        if not isinstance(instruction, str):
            errors.append("instruction must be a string")
        elif instruction.strip():
            cfg["instruction"] = instruction.strip()[:2000]
    return cfg, errors


def stored_config(runtime_config: Any) -> dict[str, Any]:
    rc = runtime_config if isinstance(runtime_config, dict) else {}
    block = rc.get(CONFIG_KEY)
    return dict(block) if isinstance(block, dict) else {}


def project_enabled(runtime_config: Any) -> bool:
    """The project opted in to ``multi_option`` scoring (``runtime_config.multi_option.enabled``)."""
    return stored_config(runtime_config).get("enabled") is True


def effective_config(runtime_config: Any, manifest_schema: Any = None) -> dict[str, Any]:
    """The manifest's stamped ``multi_option`` block overlaid by the project's live settings."""
    base = manifest_schema.get(CONFIG_KEY) if isinstance(manifest_schema, dict) else None
    merged = dict(base) if isinstance(base, dict) else {}
    merged.update({k: v for k, v in stored_config(runtime_config).items() if k in DEFAULTS})
    cfg, _errors = normalize_config(merged)
    return cfg


def uses_multi_option(runtime_config: Any, manifest: Any) -> bool:
    schema = (manifest or {}).get("output_schema") if isinstance(manifest, dict) else None
    mode = str((schema or {}).get("scoring_mode") or "").strip().lower() if isinstance(schema, dict) else ""
    return mode == SCORING_MODE or project_enabled(runtime_config)


# ── prompt / row context ─────────────────────────────────────────────────


def _context_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return ""


def output_shape_example(cfg: dict[str, Any]) -> str:
    formats = cfg.get("formats") or ["format_a"]
    option = {
        "strategy": "<strategy id>", "title": "...", "rationale": "...", "tradeoffs": "...", "rollback": "...",
        "fixes": {f: "..." for f in formats},
    }
    return json.dumps({"explanation": "...", "options": [option]}, ensure_ascii=False)


DEFAULT_INSTRUCTION = "Propose alternative solutions for the problem below."


def effective_instruction(cfg: dict[str, Any]) -> str:
    """The task sentence the prompt opens with: the project's ``instruction`` or the default."""
    return str(cfg.get("instruction") or DEFAULT_INSTRUCTION)


def build_multi_option_prompt(context: Any, cfg: dict[str, Any]) -> str:
    """The prompt for one row. Renders ONLY the row's ``context`` and the project's config (formats,
    option count, taxonomy) -- never the verifier payload or reference options."""
    lo, hi = int(cfg.get("min_options") or 1), int(cfg.get("max_options") or 1)
    count = f"{lo}" if lo == hi else f"between {lo} and {hi}"
    lines = [effective_instruction(cfg)]
    lines.append(f"Give {count} options, best first; each option must use a different strategy.")
    taxonomy = cfg.get("strategy_taxonomy") or []
    if taxonomy:
        lines.append("Strategy ids: " + ", ".join(taxonomy) + ".")
    formats = cfg.get("formats") or []
    if formats:
        lines.append("Give every option's fix in each of these formats: " + ", ".join(formats) + ".")
    lines.append("Reply with one JSON object shaped like: " + output_shape_example(cfg))
    lines.append("")
    lines.append("Context:")
    lines.append(_context_text(context))
    return "\n".join(lines).strip()


def reference_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("options_json", "reference_options_json"):
        value = row.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = None
        if isinstance(value, dict) and isinstance(value.get("options"), list):
            return value
    return None


def build_row_context(row: dict[str, Any]) -> dict[str, Any]:
    """``multi_option_row`` for a prediction: the context, the opaque verifier payload and the
    reference options (display only)."""
    ref = reference_payload(row)
    payload = row.get("verifier_payload")
    return {
        "context": row.get("context") if isinstance(row.get("context"), (str, dict, list)) else None,
        "verifier_payload": payload if isinstance(payload, (dict, list, str, int, float)) else None,
        "reference_options": list(ref["options"]) if ref else None,
    }


def row_context(prediction: dict[str, Any]) -> dict[str, Any] | None:
    row = prediction.get("multi_option_row")
    return row if isinstance(row, dict) else None


# ── per-row analysis (one verifier call per option) ──────────────────────


def _norm_strategy(value: Any) -> str:
    return " ".join(str(value).split()).strip().lower() if isinstance(value, str) else ""


def _clip_text(value: Any, cap: int) -> str | None:
    return value[:cap] if isinstance(value, str) else None


def _option_view(option: Any, rank: int, cfg: dict[str, Any], taxonomy: set[str]) -> dict[str, Any]:
    """Display + identity fields of one option (no verdict yet)."""
    if not isinstance(option, dict):
        return {"rank": rank, "strategy": None, "in_taxonomy": False if taxonomy else None, "malformed": True,
                "fixes": {}, "formats": {}, "agreement": None, "passes": False, "error": None,
                "note": "the option is not a JSON object"}
    strategy = option.get("strategy") if isinstance(option.get("strategy"), str) else None
    norm = _norm_strategy(strategy)
    fixes_raw = option.get("fixes") if isinstance(option.get("fixes"), dict) else {}
    view: dict[str, Any] = {
        "rank": rank,
        "strategy": strategy,
        "in_taxonomy": (norm in taxonomy) if taxonomy else (None if not norm else True),
        "malformed": False,
        "fixes": {f: _clip_text(fixes_raw.get(f), MAX_FIX_CHARS) for f in cfg.get("formats") or []},
        "formats": {}, "agreement": None, "passes": None, "error": None, "note": None,
    }
    for key in _TEXT_FIELDS:
        text = _clip_text(option.get(key), MAX_TEXT_CHARS)
        if text is not None:
            view[key] = text
    extra = sorted(k for k in option if k not in ("strategy", "fixes", *_TEXT_FIELDS))
    if extra:
        view["extra_fields"] = extra[:20]
    return view


def _fail_option(view: dict[str, Any], reason: str, note: str) -> None:
    """An option-level verifier failure: the option fails (never excluded) with the reason recorded."""
    view["passes"] = False
    view["failure_reason"] = reason
    view["note"] = note[:300]


def _decide_option(view: dict[str, Any], verdict: dict[str, Any], cfg: dict[str, Any], missing: list[str]) -> None:
    formats = list(cfg.get("formats") or [])
    per = dict(verdict.get("formats") or {})
    for fmt in missing:
        per[fmt] = {"checks": {"present": False}, "passes": False, "messages": ["no fix given in this format"]}
    view["formats"] = {f: per.get(f) or {"checks": {}, "passes": None, "messages": []} for f in formats}
    view["agreement"] = verdict.get("agreement")
    option_passes = verdict.get("option_passes")
    if not formats:
        # No format to check: only an explicit verdict decides (``all([])`` must never pass an option).
        if isinstance(option_passes, bool):
            view["passes"] = option_passes
        else:
            _fail_option(view, "verifier_undecided", "no formats are configured and the verifier gave no option_passes")
        return
    undecided = [f for f in formats if view["formats"][f].get("passes") is None]
    if undecided:
        _fail_option(view, "verifier_undecided", "the verifier did not decide: " + ", ".join(undecided))
        return
    passes = all(view["formats"][f]["passes"] is True for f in formats)
    if cfg.get("require_agreement") and len(formats) >= 2 and not missing:
        agree = (view["agreement"] or {}).get("agree")
        # Only an option the agreement would decide is undecided; a failed format or a veto already failed it.
        if agree is None and passes and option_passes is not False:
            _fail_option(view, "verifier_undecided", "the verifier reported no cross-format agreement")
            return
        passes = passes and agree is True
    if option_passes is False:
        passes = False
    view["passes"] = bool(passes)


def analyze_row(
    prediction: dict[str, Any],
    schema: dict[str, Any],
    parse_json: Callable[[Any], Any],
    verify: VerifyFn | None = None,
) -> dict[str, Any]:
    """Parse the prediction and verify each option (cached on the prediction)."""
    cached = prediction.get("_multi_option_analysis")
    if isinstance(cached, dict):
        return cached
    cfg = schema.get(CONFIG_KEY) if isinstance(schema.get(CONFIG_KEY), dict) else dict(DEFAULTS)
    row = row_context(prediction) or {}
    taxonomy = {_norm_strategy(t) for t in cfg.get("strategy_taxonomy") or []}
    parsed = parse_json(prediction.get("prediction") or "")
    options = parsed.get("options") if isinstance(parsed, dict) else None
    malformed = not isinstance(options, list)
    out: dict[str, Any] = {"parsed": parsed if isinstance(parsed, dict) else None, "malformed": malformed,
                           "option_count": 0, "options": [], "truncated": False}
    if malformed:
        prediction["_multi_option_analysis"] = out
        return out
    out["option_count"] = len(options)
    out["truncated"] = len(options) > MAX_OPTIONS_LIMIT
    formats = list(cfg.get("formats") or [])
    for rank, option in enumerate(options[:MAX_OPTIONS_LIMIT]):
        view = _option_view(option, rank, cfg, taxonomy)
        out["options"].append(view)
        if view["malformed"]:
            continue
        missing = [f for f in formats if not (view["fixes"].get(f) or "").strip()]
        if formats and len(missing) == len(formats):
            _decide_option(view, {"formats": {}, "agreement": None, "option_passes": None}, cfg, missing)
            continue
        if verify is None:
            view["error"] = "the option verifier did not run"
            continue
        if row.get("verifier_payload") is None:
            view["error"] = NO_PAYLOAD_ERROR
            continue
        try:
            verdict = verify(row.get("verifier_payload"), option, formats)
        except Exception as exc:  # noqa: BLE001 - a verifier never fails the eval (infrastructure)
            verdict = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        if not isinstance(verdict, dict):
            verdict = {"error": "no verdict"}
        if verdict.get("error"):
            message = str(verdict.get("error"))[:300]
            if verdict.get("error_kind") == "option":
                # Timeout / crash / bad or oversized output on THIS option: the option fails.
                _fail_option(view, str(verdict.get("reason") or "verifier_crash")[:60], message)
            else:
                # Infrastructure (could not launch, not registered / allowed / installed): unverified.
                view["error"] = message
                view["passes"] = None
            continue
        _decide_option(view, verdict, cfg, missing)
    prediction["_multi_option_analysis"] = out
    return out


def precompute(
    predictions: list[dict[str, Any]], schema: dict[str, Any], parse_json: Callable[[Any], Any], verify: VerifyFn | None
) -> None:
    """Fill the analysis cache for every multi-option row (blocking; run in a worker thread)."""
    for p in predictions:
        if isinstance(p, dict) and row_context(p) is not None:
            analyze_row(p, schema, parse_json, verify)


# ── scoring ──────────────────────────────────────────────────────────────


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _row_pass(mode: str, views: list[dict[str, Any]], n_options: int, min_options: int) -> bool | None:
    """The ``row_pass`` rule, three-valued: None only when an infrastructure-unverified option
    (``passes is None``) is needed to decide. Every rule needs at least ``min_options`` options."""
    if not views or n_options < min_options:
        return False
    flags = [v.get("passes") for v in views]
    if mode == "any_option":
        if any(f is True for f in flags):
            return True
        return None if any(f is None for f in flags) else False
    if mode == "all_options":
        if any(f is False for f in flags):
            return False
        return None if any(f is None for f in flags) else True
    # top_option: option 0 alone decides; other options' errors never exclude the row.
    return None if flags[0] is None else bool(flags[0])


def score_multi_option(
    predictions: list[dict[str, Any]],
    schema: dict[str, Any],
    parse_json: Callable[[Any], Any],
    verify: VerifyFn | None = None,
) -> dict[str, Any]:
    """Score ``predictions`` and enrich every row in place with the per-row keys the UI reads."""
    cfg = schema.get(CONFIG_KEY) if isinstance(schema.get(CONFIG_KEY), dict) else dict(DEFAULTS)
    formats = list(cfg.get("formats") or [])
    taxonomy_on = bool(cfg.get("strategy_taxonomy"))
    mode = cfg.get("row_pass") if cfg.get("row_pass") in ROW_PASS_MODES else "top_option"
    lo, hi = int(cfg.get("min_options") or 1), int(cfg.get("max_options") or 1)

    rows = skipped = decided = malformed_n = valid_json = 0
    error_rows = row_pass_n = top_pass = any_pass = all_pass = two_distinct = in_range = 0
    option_failures = missing_payload_rows = 0
    options_scored = options_passing = options_total = out_of_tax = 0
    agree_den = agree_num = 0
    distinct_sum = 0
    format_counts = {f: [0, 0] for f in formats}
    check_counts: dict[str, list[int]] = {}

    for prediction in predictions:
        row = row_context(prediction)
        if row is None:
            skipped += 1
            prediction["scoring_mode"] = SCORING_MODE
            prediction["row_skipped"] = True
            continue
        rows += 1
        a = analyze_row(prediction, schema, parse_json, verify)
        malformed = bool(a["malformed"])
        malformed_n += int(malformed)
        valid_json += int(a["parsed"] is not None)
        views = list(a["options"])
        n = int(a["option_count"])
        count_ok = lo <= n <= hi
        in_range += int(count_ok)
        errored = [v for v in views if v.get("error")]
        missing_payload_rows += int(any(v.get("error") == NO_PAYLOAD_ERROR for v in views))
        failed_by_verifier = [v for v in views if v.get("failure_reason")]
        row_flag: bool | None = False if malformed else _row_pass(mode, views, n, lo)
        row_error = row_flag is None
        passing = [v.get("passes") is True for v in views]
        distinct_passing = {
            _norm_strategy(v.get("strategy")) for v in views
            if v.get("passes") is True and _norm_strategy(v.get("strategy")) and v.get("in_taxonomy") is not False
        }
        distinct_all = {
            _norm_strategy(v.get("strategy")) for v in views
            if _norm_strategy(v.get("strategy")) and v.get("in_taxonomy") is not False
        }
        for v in views:
            options_total += 1
            if taxonomy_on and v.get("in_taxonomy") is False:
                out_of_tax += 1

        if row_error:
            error_rows += 1
        else:
            decided += 1
            option_failures += len(failed_by_verifier)
            row_pass_n += int(bool(row_flag))
            top_pass += int(bool(passing and passing[0]))
            any_pass += int(any(passing))
            all_pass += int(bool(passing) and all(passing))
            two_distinct += int(len(distinct_passing) >= 2)
            distinct_sum += len(distinct_passing)
            for v in views:
                if v.get("passes") is None:
                    continue
                options_scored += 1
                options_passing += int(v["passes"] is True)
                for fmt, fv in (v.get("formats") or {}).items():
                    if fmt in format_counts and isinstance(fv, dict) and isinstance(fv.get("passes"), bool):
                        format_counts[fmt][1] += 1
                        format_counts[fmt][0] += int(fv["passes"])
                    for check, value in ((fv or {}).get("checks") or {}).items():
                        if isinstance(value, bool):
                            tally = check_counts.setdefault(check, [0, 0])
                            tally[1] += 1
                            tally[0] += int(value)
                agreement = v.get("agreement")
                if len(formats) >= 2 and isinstance(agreement, dict) and isinstance(agreement.get("agree"), bool):
                    agree_den += 1
                    agree_num += int(agreement["agree"])

        parsed = a["parsed"]
        prediction["parsed_prediction"] = parsed
        prediction["parsed_reference"] = parse_json(prediction.get("reference") or "")
        prediction["is_valid_json"] = parsed is not None
        prediction["missing_required_fields"] = [] if not malformed else ["options"]
        prediction["scoring_mode"] = SCORING_MODE
        prediction["row_malformed"] = malformed
        prediction["row_option_count"] = n
        prediction["row_option_count_in_range"] = count_ok
        prediction["row_options"] = views
        prediction["row_options_passing"] = sum(passing)
        prediction["row_top_option_pass"] = (
            None if views and views[0].get("passes") is None else bool(passing and passing[0])
        )
        prediction["row_any_option_pass"] = None if row_error else any(passing)
        prediction["row_distinct_strategies"] = len(distinct_passing)
        prediction["row_distinct_strategies_all"] = len(distinct_all)
        # True only when the row is EXCLUDED (its rule needs an infrastructure-unverified option).
        prediction["row_verifier_error"] = row_error
        prediction["row_option_verifier_failures"] = len(failed_by_verifier)
        prediction["row_verifier_note"] = (
            f"{len(errored)} option(s) could not be verified: {errored[0].get('error')}" if errored else None
        )
        prediction["row_multi_option_pass"] = row_flag
        # None (not 0.0) for an excluded row, so per-row pairing skips it like the aggregate does.
        prediction["row_exact_match"] = None if row_flag is None else (1.0 if row_flag else 0.0)
        explanation = parsed.get("explanation") if isinstance(parsed, dict) else None
        prediction["row_explanation"] = explanation[:MAX_TEXT_CHARS] if isinstance(explanation, str) else None
        prediction["row_reference_options"] = row.get("reference_options")

    metrics: dict[str, Any] = {
        "scoring_mode": SCORING_MODE,
        "schema": {k: v for k, v in schema.items() if k != CONFIG_KEY},
        "multi_option": {
            "formats": formats, "min_options": lo, "max_options": hi, "row_pass": mode,
            "require_agreement": bool(cfg.get("require_agreement")),
            "strategy_taxonomy": list(cfg.get("strategy_taxonomy") or []),
            "verifier": schema.get("verifier_label"),
        },
        "skipped_non_multi_option_rows": skipped,
        "total": decided,
        "correct": row_pass_n,
        "verifier_error_rows": error_rows,
        "rows_missing_verifier_payload": missing_payload_rows,
        "option_verifier_failures": option_failures,
        "options_scored": options_scored,
    }
    if rows == 0:
        # Nothing scored: "not scored", never a 0% that reads like a result.
        metrics.update({"multi_option_pass_rate": None, "exact_match": None, "json_validity_rate": None,
                        "malformed_rate": None})
        return metrics
    metrics.update({
        # The row_pass rule's pass rate over decided rows (infrastructure-excluded rows left out): the
        # headline. None when every row was excluded.
        "multi_option_pass_rate": _rate(row_pass_n, decided),
        "exact_match": _rate(row_pass_n, decided),
        "option_pass_rate": _rate(options_passing, options_scored),
        "rows_with_passing_option_rate": _rate(any_pass, decided),
        "top_option_pass_rate": _rate(top_pass, decided),
        "all_options_pass_rate": _rate(all_pass, decided),
        "distinct_strategies_mean": round(distinct_sum / decided, 4) if decided else None,
        "rows_with_two_distinct_passing_rate": _rate(two_distinct, decided),
        "option_count_in_range_rate": _rate(in_range, rows),
        "json_validity_rate": _rate(valid_json, rows),
        "malformed_rate": _rate(malformed_n, rows),
        "format_pass_rate": {f: _rate(c[0], c[1]) for f, c in format_counts.items()},
        "check_pass_rate": {k: _rate(c[0], c[1]) for k, c in sorted(check_counts.items())},
    })
    if taxonomy_on:
        metrics["out_of_taxonomy_rate"] = _rate(out_of_tax, options_total)
    if agree_den:
        metrics["cross_format_agreement_rate"] = _rate(agree_num, agree_den)
    if error_rows:
        metrics["verifier_error"] = f"{error_rows} row(s) could not be verified and are excluded from the rates"
    if option_failures:
        metrics["option_verifier_failure_note"] = (
            f"{option_failures} option(s) failed in the verifier (timeout, crash or bad output) and count as failing"
        )
    if schema.get("verifier_unavailable"):
        metrics["verifier_unavailable"] = schema["verifier_unavailable"]
    return metrics


# ── fingerprint ──────────────────────────────────────────────────────────


def multi_option_fingerprint(cfg: dict[str, Any]) -> str:
    """Hash of everything that changes how an answer is scored: formats, option range, taxonomy,
    row_pass, require_agreement, the effective prompt instruction (sha256; changing it changes the
    answers) and the verifier's identity (id + protocol + revision). Stamped as
    ``multi_option_fingerprint``; a cached base-model result is reused / paired only when it matches."""
    from app.services.external_checks import verifier_identity

    parts = {
        "formats": list(cfg.get("formats") or []),
        "min_options": cfg.get("min_options"),
        "max_options": cfg.get("max_options"),
        "taxonomy": sorted(_norm_strategy(t) for t in cfg.get("strategy_taxonomy") or []),
        "row_pass": cfg.get("row_pass"),
        "require_agreement": bool(cfg.get("require_agreement")),
        "instruction": hashlib.sha256(effective_instruction(cfg).encode("utf-8", "replace")).hexdigest(),
        "verifier": verifier_identity((cfg.get("verifier") or {}).get("check_id")),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


# ── preview (EvalResult.details) ─────────────────────────────────────────

PREVIEW_FIX_CHARS = 1500
PREVIEW_MAX_OPTIONS = 5
PREVIEW_MAX_FORMATS = MAX_FORMATS


def _preview_fixes(fixes: Any) -> tuple[dict[str, Any], list[str]]:
    """Cap each format's fix text; returns ``(fixes, truncated_format_names)``."""
    out: dict[str, Any] = {}
    truncated: list[str] = []
    for fmt, text in list((fixes if isinstance(fixes, dict) else {}).items())[:PREVIEW_MAX_FORMATS]:
        if isinstance(text, str) and len(text) > PREVIEW_FIX_CHARS:
            out[fmt] = text[:PREVIEW_FIX_CHARS]
            truncated.append(str(fmt))
        else:
            out[fmt] = text
    return out, truncated


def _preview_option(option: Any) -> Any:
    if not isinstance(option, dict):
        return option
    view = dict(option)
    fixes, truncated = _preview_fixes(option.get("fixes"))
    view["fixes"] = fixes
    if isinstance(option.get("formats"), dict):
        view["formats"] = dict(list(option["formats"].items())[:PREVIEW_MAX_FORMATS])
    if truncated:
        view["fixes_truncated"] = truncated
    return view


def preview_options(options: Any) -> tuple[list[Any] | None, bool]:
    """Options for the stored preview: at most ``PREVIEW_MAX_OPTIONS`` options, ``PREVIEW_MAX_FORMATS``
    formats each and ``PREVIEW_FIX_CHARS`` per fix text (``fixes_truncated`` names the clipped
    formats). Returns ``(options, options_dropped)``; the scores always use the full analysis."""
    if not isinstance(options, list):
        return None, False
    return [_preview_option(o) for o in options[:PREVIEW_MAX_OPTIONS]], len(options) > PREVIEW_MAX_OPTIONS


def preview_fields(prediction: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """The per-row multi-option preview keys with the option lists capped (``preview_options``)."""
    out = {k: prediction.get(k) for k in keys}
    for key in ("row_options", "row_reference_options"):
        capped, dropped = preview_options(prediction.get(key))
        out[key] = capped
        if dropped:
            out[f"{key}_truncated"] = True
    return out
