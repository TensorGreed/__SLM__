"""Teacher answers from the same passages (passage-prompt distillation).

Passage-prompt training (``passage_prompts_service``) teaches a model to read
the passages it is served. Its targets used to be the generated pairs' own
answers — written by the generator from ONE source passage, and sometimes
not supported by what retrieval actually shows for the question (the answer
is paraphrased, or its passage isn't retrieved). The model then learns to
answer past its passages, or — with withheld-passage rows — to abstain while
the answer is in plain view (2026-10-19: a 1.5B run said "don't know" on 35%
of validation rows whose answer passage was retrieved).

This module makes the targets match the inputs exactly. For each training /
validation question:

1. retrieve with the project's SERVED retrieval
   (``runtime_config.auto_rag_retrieval``) over the document passages — the
   same index serving and the auto-RAG harness read;
2. render those passages with ``auto_rag_service.document_preamble`` — the
   byte-identical system message serving sends;
3. a TEACHER model (any synth backend; typically a cloud pin) answers the
   question from exactly that message. Its answer is the training target, and
   the training row's prompt is the same preamble + question, so training =
   serving. When the teacher says the passages don't cover it, that answer is
   kept as a natural abstention (no forced withheld rows).

Optional quality filter (``filter_splits``, training rows by default): the
answer judge grades each non-abstaining teacher answer against the pair's
original reference; answers graded ``wrong`` are dropped from training and
counted. Only a LOCAL judge runs the filter — it never spends cloud tokens.

Answers are cached per project in ``teacher_answers/answers.jsonl``, keyed by
the question + the retrieval setting, so a re-train reuses them at no cost and
a retrieval change asks the teacher again. Calls go through one Anthropic
Message Batch (half price) when the backend has ``use_batch`` set; failed
batch requests fall back to direct calls.

Nothing here reads the test split: the flow passes train / val rows only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.config import settings

_LOG = logging.getLogger("teacher_passage_answers")

CACHE_DIR = "teacher_answers"
CACHE_FILE = "answers.jsonl"
# Teacher-only style note, appended to the serving preamble the teacher sees
# (the training row's system message is the preamble without it): short
# answers fit the student's generation cap at serving / eval time.
TEACHER_STYLE_NOTE = "Answer in one to three sentences."
TEACHER_MAX_TOKENS = 600
SYNC_CONCURRENCY = 4
TEACHER_CONFIG_KEY = "passage_prompts_targets"
TEACHER_TARGETS = "teacher"

# "The passages don't contain / say / cover / mention …", "I don't know",
# "not in the passages": the preamble's own say-you-don't-know, in the
# phrasings models use for it.
_ABSTAIN_RE = re.compile(
    r"\b(i\s+(do\s+not|don['’]t)\s+know|(the\s+)?passages?\s+(do\s+not|don['’]t|does\s+not|doesn['’]t)\s+"
    r"(say|contain|include|mention|cover|state|provide|address|specify|answer|give)|"
    r"not\s+(in|covered\s+(in|by)|addressed\s+in|mentioned\s+in|stated\s+in)\s+the\s+(provided\s+)?passages|"
    r"(the\s+)?passages\s+(provided\s+)?(are\s+silent|have\s+no\s+information))",
    flags=re.IGNORECASE,
)

ProgressFn = Callable[[float, str], Awaitable[None]]
RetrieveFn = Callable[[str, dict[str, Any]], list[dict[str, Any]]]


_FULLWIDTH_CITE_RE = re.compile(r"[【［]\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*[】］]")


def normalize_teacher_answer(answer: str) -> str:
    """The teacher's answer as a training target in the serving format:
    full-width citation brackets (``【1】``, some models' habit) become the
    ``[1]`` the preamble asks for, and markdown bold markers are dropped."""
    text = _FULLWIDTH_CITE_RE.sub(lambda m: "[" + re.sub(r"\s*[,，、]\s*", "][", m.group(1)) + "]", str(answer or ""))
    text = text.replace("**", "")
    return text.strip()


def is_abstention(answer: str) -> bool:
    """The teacher said the passages don't hold the answer."""
    return bool(_ABSTAIN_RE.search(answer or ""))


def retrieval_cache_tag(retrieval: dict[str, Any] | None) -> str:
    from app.services.auto_rag_service import normalize_retrieval

    served = normalize_retrieval(retrieval or {})
    return json.dumps({k: served.get(k) for k in ("k", "reranker", "pool", "chunk_size")}, sort_keys=True)


def answer_key(question: str, retrieval: dict[str, Any] | None) -> str:
    """Cache key: the question + the retrieval that chose its passages."""
    raw = f"{retrieval_cache_tag(retrieval)}\x00{' '.join(str(question or '').split())}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def cache_path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / CACHE_DIR / CACHE_FILE


def load_answers(project_id: int) -> dict[str, dict[str, Any]]:
    """Every cached teacher answer by key (the last line for a key wins)."""
    path = cache_path(project_id)
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("key"):
                out[str(row["key"])] = row
    return out


def _write_answers(project_id: int, answers: dict[str, dict[str, Any]]) -> None:
    path = cache_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in answers.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _question_reference(row: dict[str, Any]) -> tuple[str, str]:
    from app.services.passage_prompts_service import _question_answer

    return _question_answer(row)


def teacher_request(question: str, preamble: str, *, max_tokens: int = TEACHER_MAX_TOKENS) -> dict[str, Any]:
    """The teacher's call: the serving preamble (+ the style note) as the
    system message, the question as the user turn."""
    return {
        "prompt": question,
        "system_prompt": f"{preamble}\n\n{TEACHER_STYLE_NOTE}",
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def _sources(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for hit in hits:
        payload = hit.get("payload") or {}
        out.append({"source_doc": payload.get("source_doc"), "chunk_id": payload.get("chunk_id")})
    return out


async def write_teacher_answers(
    project_id: int,
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    retrieval: dict[str, Any],
    llm: Any,
    retrieve_fn: RetrieveFn | None = None,
    report: ProgressFn | None = None,
    filter_judge: Any = None,
    filter_splits: tuple[str, ...] = ("train",),
    start: float = 0.0,
    span: float = 1.0,
    concurrency: int = SYNC_CONCURRENCY,
) -> dict[str, Any]:
    """Ask the teacher for every row's answer (cached ones are reused) and,
    with ``filter_judge`` (a ``ResolvedJudge``), grade the ``filter_splits``
    answers against each pair's reference. Returns the report the flow
    records (counts per split, usage). Raises ``ValueError`` when no row got
    a teacher answer — training on nothing would mislead."""
    from app.services.auto_rag_service import document_preamble, normalize_retrieval
    from app.services.span_relabel_service import batch_info, usage_block, uses_message_batch
    from app.services.synth_backends.base import SynthBackendError

    async def _report(frac: float, message: str) -> None:
        if report is not None:
            await report(start + span * max(0.0, min(1.0, frac)), message)

    served = normalize_retrieval(retrieval)
    fn = retrieve_fn
    if fn is None:
        from app.services.passage_prompts_service import served_retrieve_fn

        fn = served_retrieve_fn(project_id)
    teacher_label = llm.describe() if hasattr(llm, "describe") else getattr(llm, "name", "teacher")
    answers = load_answers(project_id)
    counts: dict[str, dict[str, int]] = {}
    todo: list[dict[str, Any]] = []  # {key, split, question, reference, preamble, sources}
    seen: set[str] = set()
    for split, rows in rows_by_split.items():
        c = counts.setdefault(split, {"rows": 0, "reused": 0, "asked": 0, "answered": 0, "failed": 0,
                                      "abstained": 0, "skipped": 0, "filter_judged": 0, "filter_dropped": 0,
                                      "filter_unjudged": 0})
        for row in rows:
            question, reference = _question_reference(row)
            if not question:
                c["skipped"] += 1
                continue
            c["rows"] += 1
            key = answer_key(question, served)
            cached = answers.get(key)
            if cached and cached.get("answer"):
                c["reused"] += 1
                cached.setdefault("split", split)
                continue
            if key in seen:
                continue
            seen.add(key)
            todo.append({"key": key, "split": split, "question": question, "reference": reference})

    await _report(0.02, f"Retrieving passages for {len(todo)} questions ({len(answers)} answers cached)")
    for index, item in enumerate(todo):
        hits = await asyncio.to_thread(fn, item["question"], served)
        item["preamble"] = document_preamble(list(hits[: served["k"]]))
        item["sources"] = _sources(list(hits[: served["k"]]))
        if index % 50 == 0:
            await _report(0.02 + 0.08 * index / max(1, len(todo)), f"Retrieving passages: {index + 1}/{len(todo)}")

    requests = [teacher_request(item["question"], item["preamble"]) for item in todo]
    raws: list[str | None] = [None] * len(todo)
    if todo and uses_message_batch(llm):
        async def _batch_progress(message: str, done: int, total: int) -> None:
            await _report(0.1 + 0.6 * done / max(1, total), f"Teacher ({teacher_label}) answering {total} questions: {message}")

        try:
            raws = await llm.complete_batch(requests, progress=_batch_progress)
        except SynthBackendError as exc:
            _LOG.warning("teacher answers: message batch failed, asking directly: %s", exc)
            raws = [None] * len(todo)

    pending = [i for i, raw in enumerate(raws) if raw is None]
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    done_count = {"n": 0}

    async def _ask(i: int) -> None:
        async with sem:
            req = dict(requests[i])
            prompt = req.pop("prompt")
            try:
                raws[i] = await llm.complete(prompt, **req)
            except SynthBackendError as exc:
                _LOG.warning("teacher answer failed for %r: %s", todo[i]["question"][:80], exc)
                raws[i] = None
            done_count["n"] += 1
            if done_count["n"] % 10 == 0 or done_count["n"] == len(pending):
                await _report(0.1 + 0.6 * done_count["n"] / max(1, len(pending)),
                              f"Teacher ({teacher_label}) answering: {done_count['n']}/{len(pending)}")

    if pending:
        await asyncio.gather(*(_ask(i) for i in pending))

    now = datetime.now(timezone.utc).isoformat()
    for item, raw in zip(todo, raws):
        c = counts[item["split"]]
        c["asked"] += 1
        answer = normalize_teacher_answer(str(raw or ""))
        if not answer:
            c["failed"] += 1
            continue
        c["answered"] += 1
        answers[item["key"]] = {
            "key": item["key"],
            "split": item["split"],
            "question": item["question"],
            "reference": item["reference"],
            "answer": answer,
            "abstained": is_abstention(answer),
            "system_prompt": item["preamble"],
            "sources": item["sources"],
            "retrieval": served,
            "teacher": teacher_label,
            "created_at": now,
        }
    _write_answers(project_id, answers)

    # Abstentions per split (cached + new) for the rows asked about.
    keys_by_split: dict[str, list[str]] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            question, _ = _question_reference(row)
            if question:
                keys_by_split.setdefault(split, []).append(answer_key(question, served))
    for split, keys in keys_by_split.items():
        counts[split]["abstained"] = sum(1 for k in keys if (answers.get(k) or {}).get("abstained"))

    # ── Quality filter (training rows only by default) ───────────────────
    filter_info: dict[str, Any] = {"applied": False}
    if filter_judge is not None and getattr(filter_judge, "local", True):
        from app.services.answer_judge_service import judge_predictions, project_cache

        filter_info = {"applied": True, "judge": filter_judge.label, "splits": list(filter_splits)}
        cache = project_cache(project_id)
        to_judge: list[tuple[str, dict[str, Any]]] = []
        for split in filter_splits:
            for k in dict.fromkeys(keys_by_split.get(split, [])):
                rec = answers.get(k)
                if not rec or rec.get("abstained"):
                    continue
                f = rec.get("filter") if isinstance(rec.get("filter"), dict) else None
                if f and f.get("judge") == filter_judge.label:
                    continue
                to_judge.append((split, rec))
        preds = [{"prompt": rec["question"], "reference": rec["reference"], "prediction": rec["answer"]} for _, rec in to_judge]

        async def _judge_progress(done: int, total: int) -> None:
            if done % 25 == 0 or done == total:
                await _report(0.72 + 0.26 * done / max(1, total), f"Checking teacher answers against the references ({done}/{total}, {filter_judge.label})")

        if preds:
            await judge_predictions(preds, filter_judge.judge, label=filter_judge.label, cache=cache, progress=_judge_progress)
        for (split, rec), pred in zip(to_judge, preds):
            verdict = pred.get("row_judge_verdict")
            if verdict is None:
                continue
            rec["filter"] = {"judge": filter_judge.label, "verdict": verdict, "reason": pred.get("row_judge_reason")}
            rec["dropped"] = verdict == "wrong"
        _write_answers(project_id, answers)
        for split in filter_splits:
            for k in dict.fromkeys(keys_by_split.get(split, [])):
                rec = answers.get(k) or {}
                if rec.get("abstained") or not rec.get("answer"):
                    continue
                f = rec.get("filter") if isinstance(rec.get("filter"), dict) else None
                if f and f.get("judge") == filter_judge.label:
                    counts[split]["filter_judged"] += 1
                    counts[split]["filter_dropped"] += int(bool(rec.get("dropped")))
                else:
                    counts[split]["filter_unjudged"] += 1
    elif filter_judge is not None:
        filter_info = {"applied": False, "reason": "the answer judge is a cloud model; the filter only runs on a local judge"}

    usable = sum(1 for keys in keys_by_split.values() for k in keys
                 if (answers.get(k) or {}).get("answer") and not (answers.get(k) or {}).get("dropped"))
    if not usable:
        raise ValueError("The teacher answered none of the questions; nothing to train on.")
    await _report(1.0, f"Teacher answers ready: {usable} usable")
    return {
        "teacher": teacher_label,
        "retrieval": served,
        "counts": counts,
        "filter": filter_info,
        "usable": usable,
        "usage": usage_block(llm),
        "batch": batch_info(llm),
        "cache_file": str(cache_path(project_id)),
    }


def teacher_rows(
    rows: list[dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    retrieval: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pure: training rows → passage-prompt rows whose prompt is the cached
    serving preamble and whose target is the teacher's answer. A row with no
    teacher answer or one the filter dropped is left out (counted)."""
    counts = {"rows_in": len(rows), "rows_out": 0, "teacher": 0, "abstained": 0, "no_teacher_answer": 0, "filtered": 0, "skipped": 0}
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        question, _ = _question_reference(row)
        if not question:
            counts["skipped"] += 1
            continue
        rec = answers.get(answer_key(question, retrieval))
        if not rec or not rec.get("answer") or not rec.get("system_prompt"):
            counts["no_teacher_answer"] += 1
            continue
        if rec.get("dropped"):
            counts["filtered"] += 1
            continue
        out = {key: value for key, value in row.items() if key not in {"text", "source_text", "target_text"}}
        out["question"] = question
        out["reference_answer"] = _question_reference(row)[1]
        out["answer"] = rec["answer"]
        out["system_prompt"] = rec["system_prompt"]
        out["passage_prompt"] = {"gold": "teacher", "abstained": bool(rec.get("abstained")), "teacher": rec.get("teacher")}
        counts["teacher"] += 1
        counts["abstained"] += int(bool(rec.get("abstained")))
        out_rows.append(out)
    counts["rows_out"] = len(out_rows)
    return out_rows, counts
