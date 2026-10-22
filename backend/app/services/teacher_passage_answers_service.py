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
counted (``filter_keep="not_wrong"``, the default), or only answers graded
``correct`` are kept (``filter_keep="correct"``: partial answers go too).
Only a LOCAL judge runs the filter — it never spends cloud tokens.

Optional critique-and-revise loop (``revise_rounds`` 1–2, off by default):
every answer the filter would drop gets a critique — the filter verdict's own
reason, or a CRITIC model's (``critic``: any synth backend / cloud pin) read
of the question, the reference, the answer and the passages — and the teacher
writes a corrected answer from the same preamble; the SAME filter judge
re-grades the revision and has the last word (``critique_and_revise``, a
helper any teacher-style flow can call). A kept revision becomes the training
target (the original stays as ``original_answer``); each round is appended to
the cache as it lands, so a re-run with the same critic and judge makes no new
calls.

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
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.config import settings

_LOG = logging.getLogger("teacher_passage_answers")

CACHE_DIR = "teacher_answers"
# What the quality filter keeps: every judged answer that is not ``wrong``
# (partial answers train), or only the ones judged ``correct``.
FILTER_KEEP_MODES = ("not_wrong", "correct")
DEFAULT_FILTER_KEEP = "not_wrong"


def filter_drops(verdict: str | None, keep: str = DEFAULT_FILTER_KEEP) -> bool:
    """Pure: does the filter drop an answer with this judge verdict under
    ``keep``? An unjudged answer (None) is never dropped."""
    if keep not in FILTER_KEEP_MODES:
        raise ValueError(f"filter_keep must be one of {FILTER_KEEP_MODES}, got {keep!r}")
    if verdict is None:
        return False
    return verdict != "correct" if keep == "correct" else verdict == "wrong"
CACHE_FILE = "answers.jsonl"
# Teacher-only style note, appended to the serving preamble the teacher sees
# (the training row's system message is the preamble without it): short
# answers fit the student's generation cap at serving / eval time.
TEACHER_STYLE_NOTE = "Answer in one to three sentences."
TEACHER_MAX_TOKENS = 600
# Critique-and-revise: how many rounds a dropped answer may be revised.
REVISE_MAX_ROUNDS = 2
CRITIQUE_MAX_TOKENS = 300
CRITIQUE_VERDICTS = ("correct", "partial", "wrong")
CRITIQUE_SYSTEM_PROMPT = (
    "You check an answer written from a set of passages against a reference answer. "
    "Reply with JSON only: {\"verdict\": \"correct|partial|wrong\", "
    "\"feedback\": \"what is wrong or missing, one or two sentences\"}."
)
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


# ``【1】`` / ``【2, 3】``, and the line-range form some reasoning models emit
# (``【1†L1-L4】``: passage 1, its lines 1-4); the passage number is what serves.
_FULLWIDTH_CITE_RE = re.compile(r"[【［]\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*(?:†[^】］]*)?[】］]")


def normalize_teacher_answer(answer: str) -> str:
    """The teacher's answer as a training target in the serving format:
    full-width citation brackets (``【1】``, ``【1†L1-L4】``, some models'
    habit) become the ``[1]`` the preamble asks for, and markdown bold markers
    are dropped. Idempotent."""
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


def _answer_record(
    item: dict[str, Any], answer: str, served: dict[str, Any], teacher_label: str, now: str | None = None
) -> dict[str, Any]:
    return {
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
        "created_at": now or datetime.now(timezone.utc).isoformat(),
    }


def _append_answer(project_id: int, record: dict[str, Any]) -> None:
    """Append one answer to the cache (``load_answers``: the last line for a
    key wins, a line cut off by a crash is skipped)."""
    path = cache_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def build_critique_prompt(question: str, reference: str, answer: str, preamble: str) -> str:
    """The critic's user turn: the passages the teacher read (the cached
    preamble), the question, the reference and the teacher's answer."""
    return (
        f"{preamble.strip()}\n\n"
        "QUESTION:\n"
        f"{question.strip()}\n\n"
        "REFERENCE ANSWER (treat as ground truth):\n"
        f"{reference.strip()}\n\n"
        "ANSWER TO CHECK:\n"
        f"{answer.strip() or '(empty)'}\n\n"
        "Say whether the answer is correct, partial or wrong against the reference, and what is "
        "wrong or missing in it — name the fact, not the style. Reply with JSON only: "
        "{\"verdict\": \"correct|partial|wrong\", \"feedback\": \"...\"}."
    )


def parse_critique(raw: str | None, fallback_feedback: str) -> dict[str, Any]:
    """``{"verdict", "feedback"}`` from the critic's reply. A reply that is
    not JSON, or carries no feedback, falls back to ``fallback_feedback``
    (the filter judge's reason) with ``verdict`` None."""
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    payload: Any = None
    if text:
        try:
            payload = json.loads(text)
        except Exception:  # noqa: BLE001
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except Exception:  # noqa: BLE001
                    payload = None
    if not isinstance(payload, dict):
        return {"verdict": None, "feedback": fallback_feedback, "parsed": False}
    verdict = str(payload.get("verdict") or "").strip().lower()
    feedback = " ".join(str(payload.get("feedback") or "").split())[:600]
    return {
        "verdict": verdict if verdict in CRITIQUE_VERDICTS else None,
        "feedback": feedback or fallback_feedback,
        "parsed": True,
    }


def revise_request(
    question: str, preamble: str, previous_answer: str, feedback: str, *, max_tokens: int = TEACHER_MAX_TOKENS
) -> dict[str, Any]:
    """The teacher's revision call: the SAME system message as its original
    answer (the serving preamble + the style note), and a user turn holding
    the question, its previous answer and the critique."""
    prompt = (
        "QUESTION:\n"
        f"{question.strip()}\n\n"
        "YOUR PREVIOUS ANSWER:\n"
        f"{previous_answer.strip() or '(empty)'}\n\n"
        "FEEDBACK ON IT:\n"
        f"{feedback.strip() or '(none)'}\n\n"
        "Write a corrected answer to the question from the passages above, in the same style: "
        "one to three sentences, citing the passages you use as [n]. Reply with the answer only."
    )
    request = teacher_request(question, preamble, max_tokens=max_tokens)
    request["prompt"] = prompt
    return request


def _fallback_feedback(record: dict[str, Any]) -> str:
    """The filter verdict's reason as the critique when there is no critic."""
    f = record.get("filter") if isinstance(record.get("filter"), dict) else {}
    reason = " ".join(str(f.get("reason") or "").split())
    if reason:
        return reason
    verdict = f.get("verdict") or "not correct"
    return f"The answer was judged {verdict} against the reference answer."


def _revise_rounds_done(record: dict[str, Any], critic_label: str, judge_label: str) -> int:
    """Rounds already revised under this critic + judge (resume)."""
    revisions = record.get("revisions") if isinstance(record.get("revisions"), list) else []
    return sum(1 for r in revisions if isinstance(r, dict) and r.get("critic") == critic_label and r.get("judge") == judge_label)


async def critique_and_revise(
    records: list[dict[str, Any]],
    *,
    llm: Any,
    judge: Any,
    keep: str = DEFAULT_FILTER_KEEP,
    rounds: int = 1,
    critic: Any = None,
    concurrency: int = SYNC_CONCURRENCY,
    judge_cache: Any = None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    report: Callable[[int, int], Awaitable[None]] | None = None,
) -> dict[str, int]:
    """Critique → revise → re-judge, up to ``rounds`` times, for every
    answer record the filter drops under ``keep``. Reusable by any flow whose
    records carry ``question`` / ``reference`` / ``answer`` /
    ``system_prompt`` (the preamble) and a ``filter`` verdict.

    ``critic``: a synth backend that writes the critique from the question,
    the reference, the answer and the passages; None → the filter verdict's
    reason, at no extra call. ``llm`` (the teacher) writes the revision from
    the same system message as its original answer; ``judge`` (a
    ``ResolvedJudge``, the final authority) re-grades it. A kept revision
    replaces the record's ``answer`` / ``filter`` (the original stays as
    ``original_answer``); every round lands in ``revisions`` and goes to
    ``on_record`` as it arrives (the cache append), so a re-run with the same
    critic and judge skips what is done. Abstentions are never revised —
    pass only non-abstaining records. Returns ``{attempted, fixed,
    still_dropped, abstained, calls}`` — ``abstained``: still-dropped records
    whose last revision is itself an abstention (a subset of still_dropped). Revisions go as direct calls even for a
    message-batch backend: each is one short call after a verdict, and a
    batch's turnaround would stall the loop between rounds."""
    from app.services.answer_judge_service import judge_predictions
    from app.services.synth_backends.base import SynthBackendError

    rounds = int(rounds)
    if rounds < 0 or rounds > REVISE_MAX_ROUNDS:
        raise ValueError(f"revise rounds must be between 0 and {REVISE_MAX_ROUNDS}, got {rounds}")
    counts = {"attempted": 0, "fixed": 0, "still_dropped": 0, "abstained": 0, "calls": 0}
    if rounds == 0 or judge is None:
        return counts
    judge_label = str(judge.label)
    critic_label = (critic.describe() if hasattr(critic, "describe") else getattr(critic, "name", "critic")) if critic is not None else judge_label
    todo = [
        rec for rec in records
        if rec.get("answer") and not rec.get("abstained")
        and filter_drops((rec.get("filter") or {}).get("verdict"), keep)
        and _revise_rounds_done(rec, critic_label, judge_label) < rounds
    ]
    counts["attempted"] = len(todo)
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    done = {"n": 0}

    async def _critique(rec: dict[str, Any]) -> dict[str, Any]:
        fallback = _fallback_feedback(rec)
        if critic is None:
            return {"verdict": None, "feedback": fallback, "parsed": False, "source": "judge_reason"}
        prompt = build_critique_prompt(rec["question"], rec.get("reference") or "", rec["answer"], rec.get("system_prompt") or "")
        try:
            raw = await critic.complete(prompt, system_prompt=CRITIQUE_SYSTEM_PROMPT, max_tokens=CRITIQUE_MAX_TOKENS, temperature=0.0)
        except SynthBackendError as exc:
            _LOG.warning("critic failed for %r: %s", rec["question"][:80], exc)
            raw = None
        counts["calls"] += 1
        out = parse_critique(raw, fallback)
        out["source"] = "critic" if out["parsed"] else "judge_reason"
        return out

    async def _revise_one(rec: dict[str, Any]) -> None:
        async with sem:
            start_round = _revise_rounds_done(rec, critic_label, judge_label)
            rec.setdefault("original_answer", rec["answer"])
            fixed = False
            for round_index in range(start_round, rounds):
                critique = await _critique(rec)
                req = revise_request(rec["question"], rec.get("system_prompt") or "", rec["answer"], critique["feedback"])
                prompt = req.pop("prompt")
                try:
                    raw = await llm.complete(prompt, **req)
                except SynthBackendError as exc:
                    _LOG.warning("teacher revision failed for %r: %s", rec["question"][:80], exc)
                    raw = None
                counts["calls"] += 1
                answer = normalize_teacher_answer(str(raw or ""))
                if not answer:
                    break
                pred = {"prompt": rec["question"], "reference": rec.get("reference") or "", "prediction": answer}
                await judge_predictions([pred], judge.judge, label=judge_label, cache=judge_cache)
                verdict = pred.get("row_judge_verdict")
                revision = {
                    "round": round_index + 1,
                    "answer": answer,
                    "feedback": critique["feedback"],
                    "critic": critic_label,
                    "critic_verdict": critique.get("verdict"),
                    "feedback_source": critique.get("source"),
                    "judge": judge_label,
                    "verdict": verdict,
                    "reason": pred.get("row_judge_reason"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                rec.setdefault("revisions", []).append(revision)
                rec["revised"] = True
                rec["answer"] = answer
                rec["abstained"] = is_abstention(answer)
                rec["filter"] = {"judge": judge_label, "verdict": verdict, "reason": pred.get("row_judge_reason"), "keep": keep, "revised": True}
                if verdict is None:
                    # The judge could not grade the revision: it stays dropped
                    # (an ungraded revision is not evidence), no further round.
                    rec["dropped"] = True
                    if on_record is not None:
                        on_record(rec)
                    break
                rec["dropped"] = filter_drops(verdict, keep)
                if on_record is not None:
                    on_record(rec)
                if not rec["dropped"]:
                    fixed = True
                    break
            counts["fixed" if fixed else "still_dropped"] += 1
            if not fixed and rec.get("abstained"):
                counts["abstained"] += 1
            done["n"] += 1
            if report is not None and (done["n"] % 10 == 0 or done["n"] == len(todo)):
                await report(done["n"], len(todo))

    if todo:
        await asyncio.gather(*(_revise_one(rec) for rec in todo))
    return counts


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
    filter_keep: str = DEFAULT_FILTER_KEEP,
    revise_rounds: int = 0,
    critic: Any = None,
    start: float = 0.0,
    span: float = 1.0,
    concurrency: int = SYNC_CONCURRENCY,
) -> dict[str, Any]:
    """Ask the teacher for every row's answer (cached ones are reused) and,
    with ``filter_judge`` (a ``ResolvedJudge``), grade the ``filter_splits``
    answers against each pair's reference; ``filter_keep`` says which
    verdicts train (``not_wrong`` / ``correct`` — a cached verdict is re-read
    under the current rule, so changing it costs no judge call).
    ``revise_rounds`` (0–``REVISE_MAX_ROUNDS``, 0 = off): answers the filter
    drops are critiqued (by ``critic``, a synth backend, else the verdict's
    reason) and revised by the teacher, then re-judged — see
    ``critique_and_revise``; needs the filter judge. Returns the report the flow
    records (counts per split, usage). Raises ``ValueError`` when no row got
    a teacher answer — training on nothing would mislead."""
    from app.services.auto_rag_service import document_preamble, normalize_retrieval
    from app.services.span_relabel_service import batch_info, usage_block, uses_message_batch
    from app.services.synth_backends.base import SynthBackendError

    async def _report(frac: float, message: str) -> None:
        if report is not None:
            await report(start + span * max(0.0, min(1.0, frac)), message)

    if filter_keep not in FILTER_KEEP_MODES:
        raise ValueError(f"filter_keep must be one of {FILTER_KEEP_MODES}, got {filter_keep!r}")
    revise_rounds = int(revise_rounds or 0)
    if revise_rounds < 0 or revise_rounds > REVISE_MAX_ROUNDS:
        raise ValueError(f"revise_rounds must be between 0 and {REVISE_MAX_ROUNDS}, got {revise_rounds}")
    if revise_rounds and (filter_judge is None or not getattr(filter_judge, "local", True)):
        raise ValueError(
            "Revising teacher answers needs the local filter judge (teacher_filter) — "
            "without its verdicts there is nothing to say which answers to revise."
        )
    revise_info: dict[str, Any] = {"rounds": revise_rounds}
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
                                      "filter_unjudged": 0, "filter_by_verdict": {}})
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
            answer = normalize_teacher_answer(str(raws[i] or ""))
            if answer:
                # On disk as it arrives: a run killed mid-way resumes from here.
                _append_answer(project_id, _answer_record(todo[i], answer, served, teacher_label))
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
        answers[item["key"]] = _answer_record(item, answer, served, teacher_label, now)
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

        filter_info = {"applied": True, "judge": filter_judge.label, "splits": list(filter_splits), "keep": filter_keep}
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
                cache.flush()  # verdicts survive a crash; a re-run reads them from the cache
                await _report(0.72 + 0.26 * done / max(1, total), f"Checking teacher answers against the references ({done}/{total}, {filter_judge.label})")

        if preds:
            await judge_predictions(preds, filter_judge.judge, label=filter_judge.label, cache=cache, progress=_judge_progress)
        for (split, rec), pred in zip(to_judge, preds):
            verdict = pred.get("row_judge_verdict")
            if verdict is None:
                continue
            rec["filter"] = {"judge": filter_judge.label, "verdict": verdict, "reason": pred.get("row_judge_reason")}
        # Every judged answer — new or cached — is kept or dropped under the
        # CURRENT keep rule (a cached ``dropped`` may come from another rule).
        judged_records: dict[str, list[dict[str, Any]]] = {}
        for split in filter_splits:
            for k in dict.fromkeys(keys_by_split.get(split, [])):
                rec = answers.get(k) or {}
                if rec.get("abstained") or not rec.get("answer"):
                    continue
                f = rec.get("filter") if isinstance(rec.get("filter"), dict) else None
                if f and f.get("judge") == filter_judge.label:
                    rec["dropped"] = filter_drops(f.get("verdict"), filter_keep)
                    rec["filter"]["keep"] = filter_keep
                    judged_records.setdefault(split, []).append(rec)
        _write_answers(project_id, answers)

        # ── Critique and revise what the filter drops ────────────────────
        if revise_rounds:
            critic_label = (critic.describe() if hasattr(critic, "describe") else getattr(critic, "name", "critic")) if critic is not None else filter_judge.label
            revise_info = {"rounds": int(revise_rounds), "critic": critic_label, "judge": filter_judge.label,
                           "critic_is_judge": critic is None}
            for split, recs in judged_records.items():
                async def _revise_progress(done: int, total: int, _split: str = split) -> None:
                    await _report(0.9 + 0.08 * done / max(1, total),
                                  f"Revising {_split} answers the filter dropped ({done}/{total}, critique by {critic_label})")

                counts[split]["revise"] = await critique_and_revise(
                    recs, llm=llm, judge=filter_judge, keep=filter_keep, rounds=int(revise_rounds), critic=critic,
                    concurrency=concurrency, judge_cache=cache,
                    on_record=lambda rec: _append_answer(project_id, rec), report=_revise_progress,
                )
            cache.flush()
            _write_answers(project_id, answers)

        # Counts over what trains: a kept revision's verdict counts, not the
        # original's (the verdict the record carries is the last judged one).
        # A judged answer whose revision came back as an abstention is still
        # judged and dropped here — skipping it as an abstention made it
        # vanish from every count (revise.abstained says how many).
        for split in filter_splits:
            judged_ids = {id(rec) for rec in judged_records.get(split, [])}
            for k in dict.fromkeys(keys_by_split.get(split, [])):
                rec = answers.get(k) or {}
                if id(rec) not in judged_ids and (rec.get("abstained") or not rec.get("answer")):
                    continue
                if id(rec) in judged_ids:
                    verdict = (rec.get("filter") or {}).get("verdict")
                    counts[split]["filter_judged"] += 1
                    counts[split]["filter_dropped"] += int(bool(rec.get("dropped")))
                    by_verdict = counts[split]["filter_by_verdict"]
                    by_verdict[str(verdict)] = by_verdict.get(str(verdict), 0) + 1
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
        "revise": revise_info,
        "usable": usable,
        "usage": usage_block(llm, critic) if critic is not None and critic is not llm else usage_block(llm),
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
        # Normalised again here so answers cached before a normaliser fix
        # train in the serving format without asking the teacher again.
        out["answer"] = normalize_teacher_answer(rec["answer"])
        out["system_prompt"] = rec["system_prompt"]
        out["passage_prompt"] = {"gold": "teacher", "abstained": bool(rec.get("abstained")), "teacher": rec.get("teacher")}
        counts["teacher"] += 1
        counts["abstained"] += int(bool(rec.get("abstained")))
        out_rows.append(out)
    counts["rows_out"] = len(out_rows)
    return out_rows, counts
