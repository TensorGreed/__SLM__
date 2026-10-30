"""Full per-row eval predictions, one JSONL file per EvalResult (Judge calibration V4, item 1).

``EvalResult.details`` keeps only small previews (``predictions_preview``, ``failures_preview``) and the
judged-rows side file for preference pairs. A held-out eval also writes EVERY scored row here:
``DATA_DIR/projects/<pid>/eval_predictions/<eval_result_id>.jsonl``, pointed to by
``details["eval_predictions_file"] = {path, count, truncated, judged}``. Text only, capped; written after the
EvalResult id exists; a write failure is logged and never fails the eval. Old results have no file and
read as "not stored" everywhere.

One record per row::

    {row_key, row_index, prompt | history, reference, prediction, correct, row_exact_match?, row_f1?,
     judge?: {scale: "verdict", task, label, verdict, score, reason, content_key, question?, reference?, prediction?},
     rubric?: {scale: "rubric", label, passed, weighted, critical_failed, rubric_sha, content_key, question?, passages?}}

``row_key`` is the row's key in ``details.row_scores.keys``. ``judge`` / ``rubric`` carry what that judge
graded: ``question`` / ``reference`` / ``prediction`` are stored only where the judge read something other
than the row's own prompt / reference / prediction (a chat history, a parsed patch, a plan scenario), so
nothing is duplicated. ``task`` names the answer judge's instruction (``answer`` = the plain prompt,
``remediation`` / ``code_example`` = that scorer's instruction); ``content_key`` is
``answer_judge_service.content_key`` of the judged texts and the task (the calibration label key; texts
cut at ``TEXT_CAP`` are left out of calibration). The file for a result id is replaced atomically on every
write, so an id reused after a delete never inherits stale rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterator

from app.config import settings

_LOG = logging.getLogger(__name__)

DIR_NAME = "eval_predictions"
ROWS_LIMIT = 10000
TEXT_CAP = 8000
HISTORY_CAP = 40
PASSAGE_CAP = 1600
PASSAGES_MAX = 20
PAGE_MAX = 500
TASKS: tuple[str, ...] = ("answer", "remediation", "code_example")


def predictions_dir(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(int(project_id)) / DIR_NAME


def predictions_path(project_id: int, eval_result_id: int) -> Path:
    return predictions_dir(project_id) / f"{int(eval_result_id)}.jsonl"


def delete_files(project_id: int, eval_result_ids: list[int]) -> int:
    """Remove the predictions files of deleted results (best-effort); returns how many were removed."""
    removed = 0
    for rid in eval_result_ids:
        try:
            predictions_path(project_id, rid).unlink()
            removed += 1
        except OSError:
            continue
    return removed


def row_key(prediction: dict[str, Any]) -> str:
    """The row's key in ``row_scores.keys`` (prompt + reference), so a stored row joins its scores."""
    raw = f"{prediction.get('prompt', '')}\x00{prediction.get('reference', '')}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:12]


def _cap(value: Any) -> str:
    return str(value or "")[:TEXT_CAP]


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _view_extras(row: dict[str, Any], question: str, reference: str, prediction: str) -> dict[str, str]:
    """The judged texts that differ from the row's own prompt / reference / prediction."""
    out: dict[str, str] = {}
    if question != str(row.get("prompt") or "") or row.get("history"):
        out["question"] = _cap(question)
    if reference != str(row.get("reference") or ""):
        out["reference"] = _cap(reference)
    if prediction != str(row.get("prediction") or ""):
        out["prediction"] = _cap(prediction)
    return out


def _answer_view(p: dict[str, Any], scoring_mode: str | None, parse_json: Callable[[Any], Any] | None) -> tuple[str, str, str] | None:
    """The (question, reference, prediction) the answer judge read for ``p`` (the same views
    ``evaluation_service._safe_judge_answers`` passes)."""
    if str(scoring_mode or "") == "remediation":
        from app.services import remediation_scoring as rs

        return rs.judge_view(p, parse_json) if parse_json is not None else None
    if p.get("history"):
        from app.services.dialogue_eval_service import history_text

        return history_text(p["history"]), str(p.get("reference") or ""), str(p.get("prediction") or "")
    return str(p.get("prompt") or ""), str(p.get("reference") or ""), str(p.get("prediction") or "")


def _judge_record(
    p: dict[str, Any], *, scoring_mode: str | None, answer_label: str | None, parse_json: Callable[[Any], Any] | None,
) -> dict[str, Any] | None:
    from app.services.answer_judge_service import VERDICT_SCORES, content_key

    ann = p.get("_code_example_judge")
    if isinstance(ann, dict) and isinstance(ann.get("verdict"), str):
        from app.services import code_example_scoring as ces

        view = ces.judge_view(p, parse_json) if parse_json is not None else None
        task, verdict, reason, label = "code_example", ann["verdict"], ann.get("reason"), ann.get("label")
        score = VERDICT_SCORES.get(verdict)
    elif isinstance(p.get("row_judge_verdict"), str):
        view = _answer_view(p, scoring_mode, parse_json)
        task = "remediation" if str(scoring_mode or "") == "remediation" else "answer"
        verdict, reason, label, score = p["row_judge_verdict"], p.get("row_judge_reason"), answer_label, _num(p.get("row_judge_score"))
    else:
        return None
    if verdict not in VERDICT_SCORES or view is None:
        return None
    question, reference, prediction = (str(v or "") for v in view)
    return {
        "scale": "verdict", "task": task, "label": str(label or ""), "verdict": verdict, "score": score,
        "reason": str(reason or "")[:500], "content_key": content_key(question, reference, prediction, task),
        **_view_extras(p, question, reference, prediction),
    }


def _rubric_record(p: dict[str, Any], *, rubric_label: str | None, rubric_sha: str | None) -> dict[str, Any] | None:
    if not isinstance(p.get("row_rubric_passed"), bool):
        return None
    from app.services import structured_plan_scoring as sps
    from app.services.answer_judge_service import content_key

    scenario, reference, prediction, passages = sps.rubric_view(p, lambda x: x)
    scenario, reference, prediction = str(scenario or ""), str(reference or ""), str(prediction or "")
    out: dict[str, Any] = {
        "scale": "rubric", "label": str(rubric_label or ""), "passed": bool(p["row_rubric_passed"]),
        "weighted": _num(p.get("row_rubric_weighted")),
        "critical_failed": list(p.get("row_rubric_critical_failed") or []),
        "rubric_sha": rubric_sha, "content_key": content_key(scenario, reference, prediction, "rubric"),
        **_view_extras(p, scenario, reference, prediction),
    }
    if passages:
        kept = []
        for i, item in enumerate(passages[:PASSAGES_MAX], 1):
            if isinstance(item, dict):
                kept.append({"id": str(item.get("id") or f"P{i}"), "text": str(item.get("text") or "")[:PASSAGE_CAP],
                             **({"source": str(item["source"])[:200]} if item.get("source") else {})})
            else:
                kept.append({"id": f"P{i}", "text": str(item)[:PASSAGE_CAP]})
        out["passages"] = kept
    return out


def build_records(
    predictions: list[dict[str, Any]],
    *,
    scoring_mode: str | None = None,
    answer_label: str | None = None,
    rubric_label: str | None = None,
    rubric_sha: str | None = None,
    parse_json: Callable[[Any], Any] | None = None,
    correct: list[int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """``(records, not_stored)``: one record per scored row (at most ``ROWS_LIMIT``). ``correct`` is the
    ``row_scores.correct`` column (same order as the dict rows)."""
    records: list[dict[str, Any]] = []
    not_stored = 0
    k = 0
    for position, p in enumerate(predictions):
        if not isinstance(p, dict):
            continue
        idx = k
        k += 1
        if len(records) >= ROWS_LIMIT:
            not_stored += 1
            continue
        rec: dict[str, Any] = {"row_key": row_key(p), "row_index": p.get("row_index", position)}
        history = [m for m in (p.get("history") or []) if isinstance(m, dict)]
        if history:
            if len(history) > HISTORY_CAP:
                rec["history_truncated"] = True
                history = history[-HISTORY_CAP:]
            rec["history"] = [{"role": str(m.get("role") or ""), "content": _cap(m.get("content"))} for m in history]
        else:
            rec["prompt"] = _cap(p.get("prompt"))
        rec["reference"] = _cap(p.get("reference"))
        rec["prediction"] = _cap(p.get("prediction"))
        if correct is not None and idx < len(correct):
            rec["correct"] = int(correct[idx])
        for key in ("row_exact_match", "row_f1"):
            if _num(p.get(key)) is not None:
                rec[key] = _num(p.get(key))
        try:
            judge = _judge_record(p, scoring_mode=scoring_mode, answer_label=answer_label, parse_json=parse_json)
        except Exception as exc:  # noqa: BLE001 - a row whose view can't be rebuilt is stored without it
            _LOG.debug("eval predictions: judge view skipped: %s", exc)
            judge = None
        if judge:
            rec["judge"] = judge
        try:
            rubric = _rubric_record(p, rubric_label=rubric_label, rubric_sha=rubric_sha)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("eval predictions: rubric view skipped: %s", exc)
            rubric = None
        if rubric:
            rec["rubric"] = rubric
        records.append(rec)
    return records, not_stored


def write_file(project_id: int, eval_result_id: int, records: list[dict[str, Any]], not_stored: int = 0) -> dict[str, Any] | None:
    """Write ``records`` (temp file + ``os.replace``: replaces any older file for the id) and return the
    details pointer; None for no rows (a stale file for a reused id is removed then)."""
    path = predictions_path(project_id, eval_result_id)
    if not records:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    os.replace(tmp, path)
    return {
        "path": f"{DIR_NAME}/{int(eval_result_id)}.jsonl",
        "count": len(records),
        "truncated": int(not_stored),
        "judged": sum(1 for r in records if r.get("judge") or r.get("rubric")),
    }


def safe_write(project_id: int, eval_result_id: int, predictions: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any] | None:
    """``build_records`` + ``write_file``; any failure is logged and returns None (never fails the eval)."""
    try:
        records, not_stored = build_records(predictions, **kwargs)
        return write_file(project_id, eval_result_id, records, not_stored)
    except Exception as exc:  # noqa: BLE001 - the eval result stands without its full predictions
        _LOG.warning("eval predictions file not written for result %s: %s", eval_result_id, exc)
        return None


def read_records(project_id: int, eval_result_id: int) -> list[dict[str, Any]] | None:
    """Every stored record, or None when the eval has no file (older results)."""
    path = predictions_path(project_id, eval_result_id)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def read_page(project_id: int, eval_result_id: int, *, offset: int = 0, limit: int = 50, judged_only: bool = False) -> dict[str, Any]:
    """One page of stored rows: ``{available, total, offset, limit, rows}`` (``available`` False and no rows
    when the eval has no file)."""
    offset = max(0, int(offset))
    limit = max(1, min(PAGE_MAX, int(limit)))
    records = read_records(project_id, eval_result_id)
    if records is None:
        return {"eval_result_id": int(eval_result_id), "available": False, "total": 0, "offset": offset, "limit": limit, "rows": []}
    if judged_only:
        records = [r for r in records if r.get("judge") or r.get("rubric")]
    return {
        "eval_result_id": int(eval_result_id), "available": True, "total": len(records),
        "offset": offset, "limit": limit, "rows": records[offset: offset + limit],
    }


_FILE_RE = re.compile(r"^(\d+)\.jsonl$")


def stored_result_ids(project_id: int) -> list[int]:
    """Eval result ids with a predictions file, ascending."""
    try:
        names = os.listdir(predictions_dir(project_id))
    except OSError:
        return []
    return sorted(int(m.group(1)) for m in (_FILE_RE.match(n) for n in names) if m)


def files_state(project_id: int) -> tuple[tuple[int, int, int], ...]:
    """``((result id, mtime_ns, size), ...)`` of every predictions file: a cache key for readers."""
    out = []
    base = predictions_dir(project_id)
    for rid in stored_result_ids(project_id):
        try:
            st = (base / f"{rid}.jsonl").stat()
        except OSError:
            continue
        out.append((rid, st.st_mtime_ns, st.st_size))
    return tuple(out)


def iter_project_records(project_id: int) -> Iterator[tuple[int, dict[str, Any]]]:
    """``(eval result id, record)`` over every stored file, oldest result first."""
    for rid in stored_result_ids(project_id):
        for rec in read_records(project_id, rid) or []:
            yield rid, rec
