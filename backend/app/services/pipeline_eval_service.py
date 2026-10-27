"""Pipeline evaluation runner (G10): per-stage scores, two views, first-wrong-stage attribution.

Runs a multi-stage pipeline over its held-out set, scores every stage with its
own project's task handler, and attributes each end-to-end failure to the
first stage that went wrong.

Two views per stage k>=2:

* ``end_to_end`` -- stage k reads stage k-1's PREDICTED output;
* ``isolated``   -- stage k reads stage k-1's GOLD output (rows without it, or
  without gold for stage k, are skipped for that view and counted in
  ``skipped_rows``).

Fan-out: one stage may declare ``fan_out: {pointer, item_key}``; it runs once
per item the pointer finds in the previous output, and every later stage runs
per item too. Gold for those stages is a list of objects carrying the item key
fields (the reference is the item's ``output`` field when present, else the
whole item). A predicted item matches a gold item when every key pointer's
value is present and equal after ``str`` + strip + lowercase (greedy,
one-to-one; the first of duplicate keys wins). An item missing a key value
matches nothing (``missing_item_key``); an output whose item matched no gold
item is ``spurious``; a gold item no predicted item matched is
``missed_upstream`` (``lost_to_error`` when the upstream unit errored).

Scoring = held-out scoring. Each scored unit becomes a row in the stage
project's own field names (the stage inputs + its gold under the prepared
rows' reference fields), built by that project's ``task_handler.build_prompts``
and carried with ``evaluation_service.copy_row_extras`` -- the same extras
held-out eval scores with. The judges held-out would run are run per stage
view. A unit is never counted right or wrong for want of scoring context,
a judge or a working scorer: it is UNSCORED with a reason
(``row_context_unavailable`` / ``needs_judge`` / ``scoring_error``) and kept
out of the end-to-end share and the attribution.

Model calls go through ``gateway_service.run_internal`` (no rate limit, no
gateway telemetry). Identical prompts to the same stage are generated once per
run (temperature 0).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services import gateway_service as gw
from app.services import gateway_tools_service as gts
from app.services import pipeline_service as ps
from app.services.json_pointer import resolve_json_pointer, split_json_pointer
from app.services.llm_budget_guard import LlmBudgetExceeded

_LOG = logging.getLogger(__name__)

DEFAULT_MAX_ROWS = 50
MAX_ROWS = 500
MAX_SYNC_ROWS = 5
MAX_MODEL_CALLS = 5000
ROWS_DETAIL_LIMIT = 200

# Sync (in-request) runs in flight in this process: a "running" run file without a Job is stale
# unless it is listed here.
ACTIVE_SYNC_RUNS: set[str] = set()

# Row fields the task handlers read a reference from (``_extract_prompt_and_reference``,
# ``StructuredExtractionHandler._read_reference_raw``, ``ClassificationHandler._extract_reference_label``,
# the recipe payload readers). A stage's gold is written under every one of them its project's prepared
# rows carry (``reference`` when none / no prepared rows).
REFERENCE_FIELDS = (
    "reference", "target_text", "expected", "output", "answer", "completion", "response", "label",
    "structured_output", "findings_json", "plan_json", "remediation_json", "code_example_json",
    "entities", "entities_json", "caption", "transcript", "class", "category",
)
PREPARED_SPLITS = ("test.jsonl", "val.jsonl", "validation.jsonl", "train.jsonl")


class _CallCapReached(Exception):
    pass


class _MissingInput(Exception):
    pass


@dataclass
class Unit:
    """One stage invocation (one row, or one fan-out item of a row)."""

    key: tuple[str, ...] | None = None
    key_values: list[Any] | None = None
    previous: Any = None
    item: Any = None
    values: dict[str, Any] | None = None
    prompt: str = ""
    raw: str | None = None
    output: Any = None
    ok: bool = False
    error: str | None = None
    has_gold: bool = False
    reference: Any = None
    spurious: bool = False
    missing_key: bool = False
    duplicate_key: bool = False
    correct: bool | None = None
    unscored_reason: str | None = None


@dataclass
class _Lost:
    """Upstream units that errored: every gold item (``all``) or those keys."""

    all: bool = False
    keys: set = field(default_factory=set)


@dataclass
class _StageCtx:
    index: int
    stage: dict[str, Any]
    resolved: Any
    project: Any
    adapter_config: dict[str, Any] | None
    experiment_id: int
    task_type: str | None


@dataclass
class _RunState:
    calls: int = 0
    cache: dict[tuple[int, str], str] = field(default_factory=dict)
    errors: Counter = field(default_factory=Counter)
    slots: dict[tuple[int, str], list[Unit]] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- small helpers


def _norm(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def _first(doc: Any, pointer: str) -> Any:
    tokens = split_json_pointer(pointer)
    if tokens is None:
        return None
    found = resolve_json_pointer(doc, tokens)
    return found[0] if found else None


def key_of(obj: Any, pointers: list[str]) -> tuple[tuple[str, ...] | None, list[Any]]:
    """``(normalized key, display values)`` of ``obj`` under the item-key pointers. The key is None
    (matches nothing) when any key value is missing or null."""
    raw = [_first(obj, p) for p in pointers]
    if any(v is None for v in raw):
        return None, raw
    return tuple(_norm(v) for v in raw), raw


def fan_items(doc: Any, pointer: str) -> list[Any]:
    tokens = split_json_pointer(pointer) or []
    found = resolve_json_pointer(doc, tokens)
    if "*" not in tokens and len(found) == 1 and isinstance(found[0], list):
        return list(found[0])
    return found


def gold_reference(item: Any) -> Any:
    """A fan-out gold item's reference: its ``output`` field when present, else the item."""
    if isinstance(item, dict) and "output" in item:
        return item["output"]
    return item


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _first_json_value(text: str) -> tuple[bool, Any]:
    """The first JSON object or array embedded in ``text`` (prose around it is ignored)."""
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text, idx)
            except ValueError:
                continue
            return True, value
    return False, None


def parse_output(text: str, kind: str) -> tuple[bool, Any]:
    """``(ok, value)``: text outputs are the stripped reply; JSON outputs are
    parsed after stripping a code fence (falling back to the first JSON object
    or array embedded in the reply)."""
    if kind == "text":
        return True, str(text or "").strip()
    from app.services.eval_task_handler_service import strip_code_fences

    stripped = strip_code_fences(text)
    try:
        return True, json.loads(stripped)
    except ValueError:
        pass
    return _first_json_value(stripped)


def _input_values(stage: dict[str, Any], row: dict[str, Any], unit: Unit) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for fld, spec in (stage.get("inputs") or {}).items():
        src = spec.get("from")
        doc = row.get("input") if src == "row" else unit.previous if src == "previous" else unit.item
        pointer = spec.get("pointer") or ""
        if pointer == "":
            if doc is None:
                raise _MissingInput(fld)
            values[fld] = doc
            continue
        tokens = split_json_pointer(pointer) or []
        found = resolve_json_pointer(doc, tokens)
        if not found:
            raise _MissingInput(fld)
        values[fld] = found if "*" in tokens else found[0]
    return values


def _canon(value: Any) -> Any:
    """Item comparison form: strings normalised like the exact-match scorer, numbers as floats."""
    from app.services.evaluation_service import _normalize_answer

    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canon(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return _normalize_answer(str(value))


def items_equal(predicted: Any, gold: Any) -> bool:
    return _canon(predicted) == _canon(gold)


# ---------------------------------------------------------------- execution


async def _prepare_stages(db: AsyncSession, pipeline: dict[str, Any]) -> list[_StageCtx]:
    slugs = await gw.all_slugs(db)
    out: list[_StageCtx] = []
    for i, stage in enumerate(pipeline.get("stages") or []):
        pid = int(stage["project_id"])
        slug = slugs.get(pid)
        if slug is None:
            raise ValueError(f"Stage '{stage['name']}': project {pid} no longer exists.")
        key = gw.internal_key([pid])
        try:
            resolved = await gw.resolve_model(db, key, f"{slug}{stage.get('model') or ''}")
        except gw.GatewayError as exc:
            raise ValueError(f"Stage '{stage['name']}' (project {pid}): {exc.message}") from exc
        adapter_config = None
        if (stage.get("prompt") or {}).get("mode") == "recipe":
            adapter_config = await gw._adapter_config(db, pid)
        exp = resolved.experiment
        task_type = None
        if exp is not None and isinstance(exp.config, dict):
            task_type = str(exp.config.get("task_type") or "").strip() or None
        out.append(
            _StageCtx(
                index=i, stage=stage, resolved=resolved, project=resolved.project, adapter_config=adapter_config,
                experiment_id=exp.id if exp is not None else 0, task_type=task_type,
            )
        )
    return out


async def _run_unit(db: AsyncSession, sc: _StageCtx, row: dict[str, Any], unit: Unit, st: _RunState) -> None:
    stage = sc.stage
    try:
        values = _input_values(stage, row, unit)
    except _MissingInput:
        unit.error = "missing_input"
        st.errors["missing_input"] += 1
        return
    unit.values = values
    prompt_cfg = stage.get("prompt") or {}
    messages: list[dict[str, str]] = []
    try:
        if prompt_cfg.get("mode") == "recipe":
            prompt = await asyncio.to_thread(gts.build_recipe_prompt, sc.project, values, sc.adapter_config or {})
        else:
            prompt = gts.render_template(str(prompt_cfg.get("template") or ""), values)
            if prompt_cfg.get("system"):
                messages.append({"role": "system", "content": str(prompt_cfg["system"])})
    except LlmBudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001 - a builder failure is this unit's error, never the run's
        _LOG.warning("pipeline stage %s prompt failed: %s", stage.get("name"), exc)
        unit.error = "prompt_error"
        st.errors["prompt_error"] += 1
        return
    unit.prompt = prompt
    if not prompt.strip():
        unit.error = "empty_prompt"
        st.errors["empty_prompt"] += 1
        return
    messages.append({"role": "user", "content": prompt})
    cache_key = (sc.index, json.dumps(messages, ensure_ascii=False))
    content = st.cache.get(cache_key)
    if content is None:
        if st.calls >= MAX_MODEL_CALLS:
            raise _CallCapReached()
        st.calls += 1
        try:
            result = await gw.run_internal(
                db, sc.resolved.requested, messages, project_ids=[sc.project.id],
                retrieval=stage.get("retrieval"), max_tokens=stage.get("max_tokens"), resolved=sc.resolved,
            )
        except LlmBudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - one failed call is counted, never crashes the run
            _LOG.warning("pipeline stage %s call failed: %s", stage.get("name"), exc)
            unit.error = "model_error"
            st.errors["model_error"] += 1
            return
        content = str(result["choices"][0]["message"]["content"] or "")
        st.cache[cache_key] = content
    unit.raw = content
    ok, value = parse_output(content, stage.get("output") or "json")
    if ok:
        unit.ok, unit.output = True, value
    else:
        unit.error = "invalid_json"
        st.errors["invalid_json"] += 1


def _units_from_previous(sc: _StageCtx, fan: int | None, previous_units: list[Unit]) -> tuple[list[Unit], _Lost]:
    out: list[Unit] = []
    lost = _Lost()
    pointers = (sc.stage.get("fan_out") or {}).get("item_key") or []
    for prev in previous_units:
        if not prev.ok:
            if fan is None or sc.index <= fan:
                lost.all = True
            else:
                lost.keys.add(prev.key)
            continue
        if sc.index == fan:
            for item in fan_items(prev.output, sc.stage["fan_out"]["pointer"]):
                k, vals = key_of(item, pointers)
                out.append(Unit(key=k, key_values=vals, previous=prev.output, item=item))
        else:
            out.append(Unit(key=prev.key, key_values=prev.key_values, previous=prev.output))
    return out, lost


def _isolated_units(
    sc: _StageCtx, fan: int | None, prev_name: str, gold: dict[str, Any], item_key: list[str]
) -> list[Unit] | None:
    """Inputs built from stage k-1's GOLD output; None when the row has none."""
    if prev_name not in gold:
        return None
    prev_gold = gold[prev_name]
    if fan is not None and sc.index - 1 >= fan:
        out = []
        for item in prev_gold if isinstance(prev_gold, list) else []:
            k, vals = key_of(item, item_key)
            out.append(Unit(key=k, key_values=vals, previous=gold_reference(item)))
        return out
    if sc.index == fan:
        out = []
        for item in fan_items(prev_gold, sc.stage["fan_out"]["pointer"]):
            k, vals = key_of(item, item_key)
            out.append(Unit(key=k, key_values=vals, previous=prev_gold, item=item))
        return out
    return [Unit(previous=prev_gold)]


def _attach_gold(
    sc: _StageCtx, fan: int | None, units: list[Unit], gold: dict[str, Any], item_key: list[str], lost: _Lost
) -> tuple[int, int]:
    """Mark each unit's reference (or spurious / missing key / duplicate key); returns
    ``(missed_upstream, lost_to_error)`` gold items."""
    name = sc.stage["name"]
    if name not in gold:
        return 0, 0
    if fan is None or sc.index < fan:
        for u in units:
            u.has_gold, u.reference = True, gold[name]
        return 0, (1 if not units and lost.all else 0)
    items = gold[name] if isinstance(gold[name], list) else []
    used = [False] * len(items)
    gold_keys = [key_of(it, item_key)[0] for it in items]
    seen: set = set()
    for u in units:
        if u.key is None:
            u.missing_key = True
            continue
        if u.key in seen:
            u.duplicate_key = True
        seen.add(u.key)
        match = next((j for j, gk in enumerate(gold_keys) if not used[j] and gk is not None and gk == u.key), None)
        if match is None:
            u.spurious = True
            continue
        used[match] = True
        u.has_gold, u.reference = True, gold_reference(items[match])
    missed = lost_n = 0
    for j, was_used in enumerate(used):
        if was_used:
            continue
        if lost.all or gold_keys[j] in lost.keys:
            lost_n += 1
        else:
            missed += 1
    return missed, lost_n


def _queue_for_scoring(sc: _StageCtx, view: str, units: list[Unit], st: _RunState) -> None:
    owners = st.slots.setdefault((sc.index, view), [])
    owners.extend(u for u in units if u.has_gold and not u.spurious)


@dataclass
class _RowResult:
    row: dict[str, Any]
    e2e: list[list[Unit]]
    iso: list[list[Unit] | None]
    e2e_missed: list[int]
    iso_missed: list[int]
    e2e_lost: list[int]
    iso_lost: list[int]


async def _run_units(
    db: AsyncSession, sc: _StageCtx, row: dict[str, Any], units: list[Unit], st: _RunState, handle: Any
) -> None:
    for u in units:
        if handle is not None and await handle.check_cancelled():
            raise asyncio.CancelledError("pipeline eval cancelled")
        await _run_unit(db, sc, row, u, st)


async def _run_row(
    db: AsyncSession, stages: list[_StageCtx], fan: int | None, item_key: list[str], row: dict[str, Any],
    st: _RunState, handle: Any = None,
) -> _RowResult:
    gold = row.get("gold") if isinstance(row.get("gold"), dict) else {}
    e2e: list[list[Unit]] = []
    iso: list[list[Unit] | None] = [None]
    e2e_missed: list[int] = []
    iso_missed: list[int] = [0]
    e2e_lost: list[int] = []
    iso_lost: list[int] = [0]
    # Queued for scoring only once the whole row ran (a row cut by the call cap leaves nothing behind).
    queued: list[tuple[_StageCtx, str, list[Unit]]] = []
    for sc in stages:
        if sc.index == 0:
            units, lost = [Unit()], _Lost()
        else:
            units, lost = _units_from_previous(sc, fan, e2e[-1])
        await _run_units(db, sc, row, units, st, handle)
        missed, lost_n = _attach_gold(sc, fan, units, gold, item_key, lost)
        e2e_missed.append(missed)
        e2e_lost.append(lost_n)
        queued.append((sc, "end_to_end", units))
        e2e.append(units)
        if sc.index == 0:
            continue
        # The isolated view only runs where it can be scored: gold for this stage AND for the one before.
        iso_units = None
        if sc.stage["name"] in gold:
            iso_units = _isolated_units(sc, fan, stages[sc.index - 1].stage["name"], gold, item_key)
        if iso_units is not None:
            await _run_units(db, sc, row, iso_units, st, handle)
            missed, lost_n = _attach_gold(sc, fan, iso_units, gold, item_key, _Lost())
            iso_missed.append(missed)
            iso_lost.append(lost_n)
            queued.append((sc, "isolated", iso_units))
        else:
            iso_missed.append(0)
            iso_lost.append(0)
        iso.append(iso_units)
    for sc, view, units in queued:
        _queue_for_scoring(sc, view, units, st)
    return _RowResult(row=row, e2e=e2e, iso=iso, e2e_missed=e2e_missed, iso_missed=iso_missed,
                      e2e_lost=e2e_lost, iso_lost=iso_lost)


# ---------------------------------------------------------------- scoring


@dataclass
class _Scorer:
    eval_ctx: Any
    handler: Any
    reference_fields: list[str]
    object_fields: set[str]


def sample_prepared_row(prepared_dir: Any) -> dict[str, Any] | None:
    """The first row of the project's prepared splits (test first) -- its field names are the ones the
    pseudo-rows use. None without prepared data."""
    if not prepared_dir:
        return None
    for name in PREPARED_SPLITS:
        path = Path(prepared_dir) / name
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        return row if isinstance(row, dict) else None
        except (OSError, ValueError):
            continue
    return None


def reference_fields_of(sample: dict[str, Any] | None) -> tuple[list[str], set[str]]:
    """``(fields the gold is written under, the subset that holds objects / lists)``."""
    if not isinstance(sample, dict):
        return ["reference"], set()
    fields = [k for k in REFERENCE_FIELDS if k in sample and sample[k] is not None]
    if not fields:
        return ["reference"], set()
    return fields, {k for k in fields if isinstance(sample[k], (dict, list))}


def _typed_gold(gold: Any, as_object: bool) -> Any:
    if as_object:
        if isinstance(gold, str):
            try:
                return json.loads(gold)
            except ValueError:
                return gold
        return gold
    return _as_text(gold)


def pseudo_row(unit: Unit, scorer: _Scorer) -> dict[str, Any]:
    """The unit as a row of its stage project: the stage input values under their field names, the
    stage prompt as ``source_text`` (unless an input is called that) and the gold under the prepared
    rows' reference fields."""
    row: dict[str, Any] = dict(unit.values or {})
    row.setdefault("source_text", unit.prompt)
    for fld in scorer.reference_fields:
        row[fld] = _typed_gold(unit.reference, fld in scorer.object_fields)
    return row


def _context_missing(bp: Any) -> bool:
    """A handler-built row that cannot be scored: no prompt / reference, or the recipe row lacks what
    its scorer needs (the code + finding of a remediation row, the scenario of a plan row)."""
    if not str(getattr(bp, "prompt", "") or "").strip() or not str(getattr(bp, "reference", "") or "").strip():
        return True
    extras = getattr(bp, "extras", None) or {}
    rem = extras.get("remediation_row")
    if isinstance(rem, dict) and (not str(rem.get("code") or "").strip() or not isinstance(rem.get("finding"), dict)):
        return True
    plan = extras.get("plan_row")
    if isinstance(plan, dict) and not str(plan.get("scenario") or "").strip():
        return True
    return False


def _needs_judge(judge: dict[str, Any] | None, prediction: dict[str, Any]) -> bool:
    """A row whose pass / fail would come from a judge that gave no verdict for it."""
    from app.services.evaluation_service import judge_decides_row

    if not judge:
        return False
    if judge["kind"] == "rubric":
        checks = prediction.get("row_structured_plan_checks_pass")
        # A plan failing its deterministic checks is wrong whatever the rubric says.
        return checks is True and not isinstance(prediction.get("row_rubric_passed"), bool)
    score = prediction.get("row_judge_score")
    return judge_decides_row(prediction) and not (isinstance(score, (int, float)) and not isinstance(score, bool))


async def _scorer_for(db: AsyncSession, sc: _StageCtx) -> _Scorer:
    from app.services.evaluation_service import heldout_scoring_context

    eval_ctx, handler = await heldout_scoring_context(
        db, sc.project.id, experiment_id=sc.experiment_id, experiment_task_type=sc.task_type,
    )
    fields, objects = reference_fields_of(sample_prepared_row(getattr(eval_ctx, "prepared_dir", None)))
    return _Scorer(eval_ctx=eval_ctx, handler=handler, reference_fields=fields, object_fields=objects)


async def _score_slot(db: AsyncSession, sc: _StageCtx, units: list[Unit], scorer: _Scorer) -> dict[str, Any]:
    """Score one stage view's gold-carrying units in place; returns ``{handler, scoring_mode, judge, error}``."""
    from app.services.evaluation_service import copy_row_extras, score_predictions_like_heldout

    out: dict[str, Any] = {"handler": getattr(scorer.handler, "profile_id", None), "scoring_mode": None,
                           "judge": None, "error": None}
    to_score = []
    for u in units:
        if u.error:
            u.correct = False  # the stage failed this unit (no output / unusable output)
        else:
            to_score.append(u)
    if not to_score:
        return out
    output_kind = sc.stage.get("output") or "json"
    try:
        built = await asyncio.to_thread(
            scorer.handler.build_prompts, [pseudo_row(u, scorer) for u in to_score], scorer.eval_ctx
        )
    except LlmBudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001 - the units are unscored, never counted
        _LOG.warning("pipeline stage %s row context failed: %s", sc.stage.get("name"), exc)
        for u in to_score:
            u.unscored_reason = "scoring_error"
        out["error"] = f"building scoring rows failed: {exc}"[:300]
        return out
    preds: list[dict[str, Any]] = []
    owners: list[Unit] = []
    for idx, u in enumerate(to_score):
        bp = built[idx] if idx < len(built) else None
        if bp is None or _context_missing(bp):
            u.unscored_reason = "row_context_unavailable"
            continue
        prediction = _as_text(u.output) if output_kind == "json" else str(u.output)
        pred = {"prompt": u.prompt, "input": u.prompt, "reference": bp.reference, "prediction": prediction}
        copy_row_extras(pred, bp.extras or {})
        preds.append(pred)
        owners.append(u)
    if not preds:
        return out
    try:
        scored = await score_predictions_like_heldout(
            db, sc.project.id, preds, eval_ctx=scorer.eval_ctx, task_handler=scorer.handler,
        )
    except LlmBudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001 - a scorer failure leaves the units unscored, never wrong
        _LOG.warning("pipeline stage %s scoring failed: %s", sc.stage.get("name"), exc)
        for u in owners:
            u.unscored_reason = "scoring_error"
        out["error"] = f"scorer failed: {exc}"[:300]
        return out
    out["handler"] = scored.get("handler")
    out["scoring_mode"] = scored.get("scoring_mode")
    out["judge"] = scored.get("judge")
    correct = scored.get("correct") or []
    for idx, (u, p) in enumerate(zip(owners, preds)):
        if p.get("row_skipped"):
            u.unscored_reason = "row_context_unavailable"
        elif _needs_judge(out["judge"], p):
            u.unscored_reason = "needs_judge"
        elif idx >= len(correct):
            u.unscored_reason = "scoring_error"
        else:
            u.correct = bool(correct[idx])
    return out


def _combine_judges(judges: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not judges:
        return None
    reason = next((j.get("reason") for j in judges if j.get("reason")), None)
    block = {
        "kind": judges[0].get("kind"),
        "label": next((j.get("label") for j in judges if j.get("label")), None),
        "ran": all(j.get("ran") for j in judges),
        "units": sum(int(j.get("units") or 0) for j in judges),
    }
    if reason:
        block["reason"] = reason
    return block


async def _score_all(db: AsyncSession, stages: list[_StageCtx], st: _RunState) -> dict[int, dict[str, Any]]:
    from app.services import llm_budget_guard

    info: dict[int, dict[str, Any]] = {}
    for sc in stages:
        stage_info: dict[str, Any] = {"handler": None, "scoring_mode": None, "scoring_error": None, "judge": None}
        info[sc.index] = stage_info
        slots = [(view, st.slots.get((sc.index, view)) or []) for view in ("end_to_end", "isolated")]
        if not any(units for _, units in slots):
            continue
        rc = getattr(sc.project, "runtime_config", None)
        override = rc.get("synth_command") if isinstance(rc, dict) else None
        judges: list[dict[str, Any]] = []
        # Judge spend is metered against the stage's project, as in its own held-out eval.
        with llm_budget_guard.project_context(sc.project.id, override if isinstance(override, dict) else None):
            try:
                scorer = await _scorer_for(db, sc)
            except LlmBudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("pipeline stage %s scorer unavailable: %s", sc.stage.get("name"), exc)
                stage_info["scoring_error"] = f"scorer unavailable: {exc}"[:300]
                for _, units in slots:
                    for u in units:
                        if u.error:
                            u.correct = False
                        else:
                            u.unscored_reason = "scoring_error"
                continue
            for _, units in slots:
                if not units:
                    continue
                res = await _score_slot(db, sc, units, scorer)
                stage_info["handler"] = stage_info["handler"] or res.get("handler")
                stage_info["scoring_mode"] = stage_info["scoring_mode"] or res.get("scoring_mode")
                stage_info["scoring_error"] = stage_info["scoring_error"] or res.get("error")
                if res.get("judge"):
                    judges.append(res["judge"])
        stage_info["judge"] = _combine_judges(judges)
    return info


# ---------------------------------------------------------------- metrics


def _view_block(units_per_row: list[list[Unit] | None], missed: list[int], lost: list[int]) -> dict[str, Any]:
    scored = correct = spurious = errors = skipped = missing_key = duplicate_key = 0
    unscored: Counter = Counter()
    for units in units_per_row:
        if units is None:
            skipped += 1
            continue
        for u in units:
            if u.error:
                errors += 1
            if u.duplicate_key:
                duplicate_key += 1
            if u.missing_key:
                missing_key += 1
            elif u.spurious:
                spurious += 1
            elif not u.has_gold:
                unscored["no_gold"] += 1
            elif u.unscored_reason:
                unscored[u.unscored_reason] += 1
            else:
                scored += 1
                correct += 1 if u.correct else 0
    return {
        "scored": scored, "correct": correct, "accuracy": round(correct / scored, 4) if scored else None,
        "unscored": sum(unscored.values()), "unscored_by_reason": dict(unscored), "spurious": spurious,
        "missed_upstream": sum(missed), "lost_to_error": sum(lost), "missing_item_key": missing_key,
        "duplicate_item_key": duplicate_key, "errors": errors, "skipped_rows": skipped,
    }


def _walk_final(
    stages: list[_StageCtx], fan: int | None, item_key: list[str], rr: _RowResult,
    key: tuple[str, ...] | None, nth: int, item_compare: Counter,
) -> tuple[str | None, bool, str | None]:
    """``(first_wrong_stage, missed_upstream, unscored_reason)`` for one final gold item;
    ``(None, False, None)`` when fully correct. A stage on the path that has gold but could not be
    scored makes the item unscored (it is neither correct nor attributed)."""
    gold = rr.row.get("gold") if isinstance(rr.row.get("gold"), dict) else {}
    for j, sc in enumerate(stages):
        name = sc.stage["name"]
        if fan is not None and j == fan - 1:
            # The stage that produces the fan-out items is judged PER ITEM: it must have produced an item
            # with this key that equals the gold upstream item (key match alone when no gold item exists).
            prod = rr.e2e[j][0] if rr.e2e[j] else None
            if prod is None:
                return stages[max(0, j - 1)].stage["name"], True, None
            if prod.error:
                return name, False, None
            cands = [u for u in rr.e2e[fan] if key is not None and u.key == key]
            if nth >= len(cands):
                return name, True, None
            pointer = stages[fan].stage["fan_out"]["pointer"]
            gold_items = (
                [it for it in fan_items(gold[name], pointer) if key_of(it, item_key)[0] == key] if name in gold else []
            )
            if nth < len(gold_items):
                item_compare["deep_equal"] += 1
                if not items_equal(cands[nth].item, gold_items[nth]):
                    return name, False, None
            else:
                item_compare["key_only"] += 1
            continue
        if fan is not None and j >= fan:
            cands = [u for u in rr.e2e[j] if key is not None and u.key == key]
            unit = cands[nth] if nth < len(cands) else None
        else:
            unit = rr.e2e[j][0] if rr.e2e[j] else None
        if unit is None:
            # Nothing on this path reached stage j: the stage before failed to produce it.
            return stages[max(0, j - 1)].stage["name"], True, None
        if unit.error:
            return name, False, None
        if unit.has_gold:
            if unit.unscored_reason:
                return None, False, unit.unscored_reason
            if not unit.correct:
                return name, False, None
    return None, False, None


def _finals_for_row(
    stages: list[_StageCtx], fan: int | None, item_key: list[str], rr: _RowResult, item_compare: Counter
) -> list[dict[str, Any]]:
    last = stages[-1].stage["name"]
    gold = rr.row.get("gold") if isinstance(rr.row.get("gold"), dict) else {}
    if last not in gold:
        return []

    def final(gold_key: Any, wrong: str | None, missed: bool, unscored: str | None) -> dict[str, Any]:
        out = {"gold_item_key": gold_key, "fully_correct": None if unscored else wrong is None,
               "first_wrong_stage": wrong, "_missed": missed}
        if unscored:
            out["unscored_reason"] = unscored
        return out

    if fan is None:
        return [final(None, *_walk_final(stages, fan, item_key, rr, None, 0, item_compare))]
    out: list[dict[str, Any]] = []
    seen: Counter = Counter()
    for item in gold[last] if isinstance(gold[last], list) else []:
        k, vals = key_of(item, item_key)
        nth = seen[k]
        seen[k] += 1
        out.append(final(vals, *_walk_final(stages, fan, item_key, rr, k, nth, item_compare)))
    return out


def build_run_payload(
    *, pipeline: dict[str, Any], run_id: str, started_at: str, stages: list[_StageCtx], fan: int | None,
    item_key: list[str], results: list[_RowResult], st: _RunState, truncated: bool, handler_info: dict[int, dict],
    truncated_at_row: Any = None, gold_mismatch: dict[str, list] | None = None,
) -> dict[str, Any]:
    first_wrong: Counter = Counter()
    missed_at: Counter = Counter()
    unscored_items: Counter = Counter()
    item_compare: Counter = Counter()
    gold_final = fully = 0
    rows_detail = []
    for n, rr in enumerate(results):
        finals = _finals_for_row(stages, fan, item_key, rr, item_compare)
        for f in finals:
            gold_final += 1
            if f.get("unscored_reason"):
                unscored_items[f["unscored_reason"]] += 1
            elif f["fully_correct"]:
                fully += 1
            else:
                first_wrong[f["first_wrong_stage"]] += 1
                if f["_missed"]:
                    missed_at[f["first_wrong_stage"]] += 1
        if n < ROWS_DETAIL_LIMIT:
            rows_detail.append(
                {
                    "id": rr.row.get("id"),
                    "stages": [
                        {
                            "name": sc.stage["name"], "inputs_count": len(rr.e2e[sc.index]),
                            "outputs": [
                                {"item_key_values": u.key_values,
                                 "correct": u.correct if (u.has_gold and not u.unscored_reason) else None,
                                 "unscored_reason": u.unscored_reason, "error": u.error, "spurious": u.spurious}
                                for u in rr.e2e[sc.index]
                            ],
                        }
                        for sc in stages
                    ],
                    "final": [{k: v for k, v in f.items() if not k.startswith("_")} for f in finals],
                }
            )
    stage_blocks = []
    for sc in stages:
        e2e = _view_block([rr.e2e[sc.index] for rr in results], [rr.e2e_missed[sc.index] for rr in results],
                          [rr.e2e_lost[sc.index] for rr in results])
        e2e.pop("skipped_rows", None)
        iso = None
        if sc.index > 0:
            iso = _view_block([rr.iso[sc.index] for rr in results], [rr.iso_missed[sc.index] for rr in results],
                              [rr.iso_lost[sc.index] for rr in results])
        exp = sc.resolved.experiment
        hi = handler_info.get(sc.index) or {}
        judge = hi.get("judge")
        block = {
            "name": sc.stage["name"], "project_id": sc.project.id, "project_name": sc.project.name,
            "model_id": sc.resolved.canonical, "run_id": exp.id if exp is not None else None,
            "handler": hi.get("handler"), "scoring_mode": hi.get("scoring_mode"),
            "scoring_error": hi.get("scoring_error"), "judge": judge,
            "judge_not_run": bool(judge and not judge.get("ran")),
            "end_to_end": e2e, "isolated": iso,
        }
        if fan is not None and sc.index == fan - 1:
            counts = {k: item_compare.get(k, 0) for k in ("deep_equal", "key_only")}
            block["item_compare"] = (
                "key_only" if counts["key_only"] and not counts["deep_equal"]
                else "mixed" if counts["key_only"] else "deep_equal"
            )
            block["item_compare_counts"] = counts
        stage_blocks.append(block)
    scored_items = gold_final - sum(unscored_items.values())
    payload = {
        "run_id": run_id, "pipeline_id": pipeline["id"], "status": "done", "started_at": started_at,
        "finished_at": _now(), "rows": len(results), "truncated": truncated, "calls": st.calls,
        "errors_by_reason": dict(st.errors), "stages": stage_blocks,
        "end_to_end": {
            "gold_final_items": gold_final, "unscored_items": sum(unscored_items.values()),
            "unscored_by_reason": dict(unscored_items), "fully_correct": fully,
            "share_fully_correct": round(fully / scored_items, 4) if scored_items else None,
        },
        "attribution": {"first_wrong_stage": dict(first_wrong), "correct": fully, "missed_upstream_at": dict(missed_at)},
        "rows_detail": rows_detail,
    }
    if truncated:
        payload["truncated_at_row"] = truncated_at_row
        payload["truncated_message"] = (
            f"Stopped at row {truncated_at_row!r}: the run reached the cap of {MAX_MODEL_CALLS} model calls; "
            f"{len(results)} rows before it were scored."
        )
    if gold_mismatch:
        payload["gold_mismatch"] = gold_mismatch
    return payload


# ---------------------------------------------------------------- entry points


def job_result(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline_id": run.get("pipeline_id"), "run_id": run.get("run_id"), "rows": run.get("rows"),
        "share_fully_correct": (run.get("end_to_end") or {}).get("share_fully_correct"),
        "truncated": bool(run.get("truncated")), "calls": run.get("calls"),
        "attribution": {"first_wrong_stage": dict((run.get("attribution") or {}).get("first_wrong_stage") or {})},
    }


async def run_pipeline_eval(
    db: AsyncSession, pipeline_id: str, run_id: str, *, max_rows: int = DEFAULT_MAX_ROWS, handle: Any = None,
    job_id: int | None = None,
) -> dict[str, Any]:
    """Run the pipeline over its held-out rows, write ``runs/<run_id>.json`` and
    return it. Raises (after recording the run as failed) when nothing could be
    scored -- never a silent empty success."""
    started_at = _now()
    base = {"run_id": run_id, "pipeline_id": pipeline_id, "started_at": started_at, "job_id": job_id}
    ps.save_run(pipeline_id, {**base, "status": "running"})
    if job_id is None:
        ACTIVE_SYNC_RUNS.add(run_id)
    try:
        run = await _run(db, pipeline_id, run_id, started_at, max_rows, handle)
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        ps.save_run(
            pipeline_id,
            {**base, "status": status, "finished_at": _now(), "error": str(exc)[:1000] or type(exc).__name__},
        )
        raise
    finally:
        ACTIVE_SYNC_RUNS.discard(run_id)
    run["job_id"] = job_id
    ps.save_run(pipeline_id, run)
    return run


async def _run(
    db: AsyncSession, pipeline_id: str, run_id: str, started_at: str, max_rows: int, handle: Any,
) -> dict[str, Any]:
    pipeline = ps.load_pipeline(pipeline_id)
    # Re-check the held-out set against the pipeline as it is NOW (stages may have been renamed or
    # re-keyed since the upload): rows whose gold no longer fits are left out and reported.
    stored = ps.load_heldout(pipeline_id)
    fitting, mismatch = ps.parse_heldout(pipeline, list(enumerate(stored, start=1)))
    gold_mismatch = {reason: [e.get("id") for e in entries] for reason, entries in mismatch.items()}
    rows = fitting[: max(1, min(MAX_ROWS, int(max_rows)))]
    if not rows:
        detail = f" ({sum(len(v) for v in gold_mismatch.values())} rows no longer fit the pipeline: " \
                 f"{', '.join(sorted(gold_mismatch))})" if gold_mismatch else ""
        raise ValueError(f"The pipeline has no usable held-out rows{detail}; upload a held-out set first.")
    stages = await _prepare_stages(db, pipeline)
    fan = ps.fan_out_index(pipeline)
    item_key = list((stages[fan].stage.get("fan_out") or {}).get("item_key") or []) if fan is not None else []
    st = _RunState()
    results: list[_RowResult] = []
    truncated = False
    truncated_at_row = None
    total = len(rows)
    for n, row in enumerate(rows):
        if handle is not None and await handle.check_cancelled():
            raise asyncio.CancelledError("pipeline eval cancelled")
        errors_before = Counter(st.errors)
        try:
            results.append(await _run_row(db, stages, fan, item_key, row, st, handle))
        except _CallCapReached:
            st.errors = errors_before  # the cut row is dropped whole
            truncated, truncated_at_row = True, row.get("id")
            break
        if handle is not None:
            await handle.set_progress(fraction=(n + 1) / total, message=f"{n + 1}/{total} rows · {st.calls} model calls")
    if handle is not None:
        queued = sum(len(units) for units in st.slots.values())
        await handle.set_progress(message=f"Scoring {queued} stage outputs…")
    handler_info = await _score_all(db, stages, st)
    run = build_run_payload(
        pipeline=pipeline, run_id=run_id, started_at=started_at, stages=stages, fan=fan, item_key=item_key,
        results=results, st=st, truncated=truncated, handler_info=handler_info,
        truncated_at_row=truncated_at_row, gold_mismatch=gold_mismatch,
    )
    outputs = sum(1 for rr in results for units in rr.e2e + [u for u in rr.iso if u] for x in units if x.raw is not None)
    scored = sum(
        (s["end_to_end"]["scored"] + ((s["isolated"] or {}).get("scored") or 0)) for s in run["stages"]
    )
    reasons = ", ".join(f"{k}: {v}" for k, v in sorted(st.errors.items())) or "none"
    if outputs == 0:
        raise ValueError(
            f"{len(results)} rows ran but no stage produced a model output (errors by reason: {reasons}). "
            "Check the stage inputs / pointers against the held-out rows."
        )
    if scored == 0:
        unscored: Counter = Counter()
        for s in run["stages"]:
            for view in (s["end_to_end"], s["isolated"] or {}):
                unscored.update({k: v for k, v in (view.get("unscored_by_reason") or {}).items() if k != "no_gold"})
        why = ", ".join(f"{k}: {v}" for k, v in sorted(unscored.items()))
        raise ValueError(
            f"{len(results)} rows ran ({outputs} model outputs) but nothing was scored: "
            + (f"every gold-carrying output was unscored ({why}). " if why else
               "no row carries gold for a stage that produced output. ")
            + f"Stages: {', '.join(s.stage['name'] for s in stages)}; errors by reason: {reasons}."
        )
    return run


async def start_pipeline_eval_job(
    db: AsyncSession, *, pipeline: dict[str, Any], max_rows: int, user_id: int | None = None
) -> Any:
    from app.database import async_session_factory
    from app.services.jobs_service import JobProgressHandle, start_job

    run_id = ps.new_run_id()
    pipeline_id = pipeline["id"]

    async def _runner(handle: JobProgressHandle) -> dict[str, Any]:
        async with async_session_factory() as runner_db:
            run = await run_pipeline_eval(
                runner_db, pipeline_id, run_id, max_rows=max_rows, handle=handle, job_id=handle.job_id,
            )
            await runner_db.commit()
        return job_result(run)

    return await start_job(
        db, kind="pipeline_eval", title=f"Pipeline eval · {pipeline.get('name') or pipeline_id}", runner=_runner,
        project_id=None, user_id=user_id, params={"pipeline_id": pipeline_id, "run_id": run_id, "max_rows": max_rows},
    )
