"""Train a model to answer FROM retrieved passages (passage-prompt training).

A documents project serves "base model + the passages retrieval finds"
(``auto_rag_service.build_preamble_from_query``): a system message listing
the passages, "answer using only these passages, cite [n], say you don't
know otherwise", then the question. A model fine-tuned on bare question →
answer pairs never saw that message — it learns to answer from memory, and
on a few hundred generated pairs it memorises badly (confident wrong facts).

This module rewrites each Q&A training row into the SERVED prompt shape, so
the fine-tune learns to read the passages it will be given:

* the question goes through the project's served retrieval
  (``runtime_config.auto_rag_retrieval`` — top-k, reranker, pool, chunk
  size) over the document passages — the same index serving and the
  auto-RAG harness read; never the Q&A pairs, the answer key or the
  val/test rows;
* the retrieved passages become the row's ``system_prompt``, rendered by
  ``auto_rag_service.document_preamble`` (byte-identical to serving);
  ``scripts/train.py`` renders it as the chat template's system message;
* RAFT-style mix (Zhang et al., 2024, "RAFT: Adapting Language Model to
  Domain Specific RAG"): for most rows the passage holding the answer is
  among the retrieved distractors — when retrieval missed it, it is put
  back at a random position in place of the lowest-ranked hit (counted as
  ``gold_inserted``) — and the target answer ends with a citation of its
  position, ``[n]``, as the preamble asks. For ``withheld_fraction`` of the
  rows (default 0.2) every passage holding the answer is removed and the
  target becomes ``NOT_IN_PASSAGES_ANSWER`` — the "say you don't know" the
  preamble asks for. ONE choice, applied to every withheld row; the
  original answer is never kept for a row whose passages don't support it
  (that would teach answering from memory, the failure this fixes).

"Holds the answer" is the answer-key gate's own test
(``documents_qa_flow_service.answer_support`` ≥ ``GOLD_SUPPORT_MIN``: the
share of the answer's content words present in the passage) — general,
no domain rules. Only the train and validation files are rewritten (into
the run's output dir); the prepared split, and above all its test rows,
are untouched.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable

# TrainingConfig fields (flat, like the other TrainingConfig switches).
CONFIG_FLAG = "passage_prompts"
CONFIG_WITHHELD = "passage_prompts_withheld_fraction"
# "reference" (default): the pair's own answer, RAFT-style gold insertion /
# withholding. "teacher": the cached teacher answer to exactly the served
# passages (teacher_passage_answers_service), prompt = the cached preamble.
CONFIG_TARGETS = "passage_prompts_targets"
TARGETS_REFERENCE = "reference"
TARGETS_TEACHER = "teacher"
DEFAULT_WITHHELD_FRACTION = 0.2
MAX_WITHHELD_FRACTION = 0.5
# The target for a row whose passages were withheld.
NOT_IN_PASSAGES_ANSWER = "The passages don't say, so I don't know."
# A passage "holds the answer" when this share of the answer's content
# words occur in it (the answer-key quotability threshold).
GOLD_SUPPORT_MIN = 0.7
# Extra hits retrieved beyond k so a withheld row still shows k passages.
_EXTRA_HITS = 3
_MAX_K = 10

RetrieveFn = Callable[[str, dict[str, Any]], list[dict[str, Any]]]


def is_enabled(config: Any) -> bool:
    return isinstance(config, dict) and bool(config.get(CONFIG_FLAG))


def targets(config: Any) -> str:
    raw = str((config or {}).get(CONFIG_TARGETS) or "").strip().lower() if isinstance(config, dict) else ""
    return TARGETS_TEACHER if raw == TARGETS_TEACHER else TARGETS_REFERENCE


def served_retrieve_fn(project_id: int) -> RetrieveFn:
    """``fn(question, retrieval) -> hits`` over the project's document index
    (the ``chunk_size`` variant when set) — the retrieval serving uses. The
    index is built / checked once, on the first call."""
    from app.services.auto_rag_service import (
        AutoRagUnavailable,
        document_index_dir,
        normalize_retrieval,
        retrieve_document_passages,
        retrieve_ranked,
    )

    checked: list[bool] = []

    def _retrieve(question: str, config: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            if not checked:
                retrieve_document_passages(project_id, question, config)
                checked.append(True)
            r = normalize_retrieval(config)
            return list(retrieve_ranked(
                question, index_dir=document_index_dir(project_id, r["chunk_size"]),
                k=r["k"], reranker=r["reranker"], pool=r["pool"],
            ))
        except AutoRagUnavailable as exc:
            raise ValueError(str(exc)) from exc

    return _retrieve


def withheld_fraction(config: Any) -> float:
    raw = config.get(CONFIG_WITHHELD) if isinstance(config, dict) else None
    try:
        value = float(raw) if raw is not None else DEFAULT_WITHHELD_FRACTION
    except (TypeError, ValueError):
        value = DEFAULT_WITHHELD_FRACTION
    return max(0.0, min(MAX_WITHHELD_FRACTION, value))


def _support(answer: str, text: str) -> float:
    from app.services.documents_qa_flow_service import answer_support

    value = answer_support(answer, text)
    return float(value) if value is not None else 0.0


def _hit_text(hit: dict[str, Any]) -> str:
    return str((hit.get("payload") or {}).get("text") or "")


def holds_answer(answer: str, hit: dict[str, Any]) -> bool:
    return _support(answer, _hit_text(hit)) >= GOLD_SUPPORT_MIN


def _row_rng(seed: int, question: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\x00{question}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _question_answer(row: dict[str, Any]) -> tuple[str, str]:
    question = str(row.get("question") or row.get("prompt") or row.get("instruction") or "").strip()
    answer = str(row.get("answer") or row.get("completion") or row.get("output") or row.get("response") or "").strip()
    return question, answer


def best_gold_passage(
    answer: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """The candidate passage (hit-shaped) that best holds ``answer``, or None
    when none reaches ``GOLD_SUPPORT_MIN``."""
    best, best_score = None, 0.0
    for candidate in candidates:
        score = _support(answer, _hit_text(candidate))
        if score > best_score:
            best, best_score = candidate, score
    return best if best is not None and best_score >= GOLD_SUPPORT_MIN else None


def passage_prompt_row(
    row: dict[str, Any],
    hits: list[dict[str, Any]],
    *,
    k: int,
    withhold: bool,
    gold_candidates: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[dict[str, Any] | None, str]:
    """Pure: one training row → its passage-prompt row, plus what happened
    to the passage holding the answer: ``retrieved`` (retrieval found it),
    ``inserted`` (retrieval missed it; put back), ``withheld`` (removed,
    target = ``NOT_IN_PASSAGES_ANSWER``) or ``not_found`` (no passage holds
    the answer — the row keeps its answer and the retrieved passages).
    Returns ``(None, "skipped")`` for a row without question + answer."""
    from app.services.auto_rag_service import document_preamble

    question, answer = _question_answer(row)
    if not question or not answer:
        return None, "skipped"
    gold_positions = [i for i, hit in enumerate(hits) if holds_answer(answer, hit)]
    if withhold:
        shown = [hit for i, hit in enumerate(hits) if i not in gold_positions][:k]
        outcome, target, cite = "withheld", NOT_IN_PASSAGES_ANSWER, None
    else:
        shown = list(hits[:k])
        first_gold = next((i for i in gold_positions if i < k), None)
        if first_gold is not None:
            outcome, cite = "retrieved", first_gold
        else:
            gold = best_gold_passage(answer, gold_candidates)
            if gold is None:
                outcome, cite = "not_found", None
            else:
                outcome = "inserted"
                if len(shown) >= k:
                    shown = shown[: k - 1]
                cite = rng.randint(0, len(shown))
                shown.insert(cite, gold)
        target = f"{answer} [{cite + 1}]" if cite is not None else answer
    if not shown:
        return None, "no_passages"
    out = {key: value for key, value in row.items() if key not in {"text", "source_text", "target_text"}}
    out["question"] = question
    out["answer"] = target
    out["system_prompt"] = document_preamble(shown)
    out["passage_prompt"] = {
        "gold": outcome,
        "cited": (cite + 1) if (not withhold and cite is not None) else None,
        "passages": len(shown),
    }
    return out, outcome


def _gold_candidates_for(
    row: dict[str, Any],
    by_document: dict[Any, list[dict[str, Any]]],
    source_passages: dict[tuple[Any, Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Where to look for the passage that holds a row's answer when
    retrieval missed it: the served corpus' passages of the row's source
    document (re-chunked variant when the retrieval uses one), then the
    row's own source passage (the cleaning's chunk it was generated from)."""
    doc_id = row.get("source_document_id")
    candidates = list(by_document.get(doc_id, []))
    own = source_passages.get((doc_id, row.get("chunk_id")))
    if own is not None:
        candidates.append(own)
    return candidates


def build_passage_prompt_rows(
    rows: list[dict[str, Any]],
    *,
    retrieval: dict[str, Any],
    retrieve_fn: RetrieveFn,
    corpus_passages: list[dict[str, Any]],
    source_passages: list[dict[str, Any]],
    withheld: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rewrite ``rows``. ``retrieve_fn(question, retrieval)`` returns hits
    (``RetrievedChunk``-shaped); ``corpus_passages`` are the served index's
    passages (hit payloads), ``source_passages`` the cleaning's passages the
    rows were generated from (``chunk_id`` provenance). Deterministic for a
    given ``seed``. Returns ``(rows, counts)``."""
    from app.services.auto_rag_service import normalize_retrieval

    served = normalize_retrieval(retrieval)
    k = served["k"]
    wide = {**served, "k": min(_MAX_K, k + _EXTRA_HITS)}
    by_document: dict[Any, list[dict[str, Any]]] = {}
    for passage in corpus_passages:
        by_document.setdefault(passage.get("source_document_id"), []).append({"payload": passage})
    own = {
        (p.get("source_document_id"), p.get("chunk_id")): {"payload": p} for p in source_passages
    }
    counts = {"rows_in": len(rows), "rows_out": 0, "retrieved": 0, "inserted": 0, "withheld": 0, "not_found": 0, "skipped": 0, "no_passages": 0}
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        question, _ = _question_answer(row)
        rng = _row_rng(seed, question)
        withhold = rng.random() < withheld
        hits = retrieve_fn(question, wide) if question else []
        new_row, outcome = passage_prompt_row(
            row,
            hits,
            k=k,
            withhold=withhold,
            gold_candidates=_gold_candidates_for(row, by_document, own),
            rng=rng,
        )
        counts[outcome] = counts.get(outcome, 0) + 1
        if new_row is not None:
            out_rows.append(new_row)
    counts["rows_out"] = len(out_rows)
    return out_rows, counts


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_passage_prompts(
    project_id: int,
    *,
    retrieval: dict[str, Any],
    train_file: Path,
    val_file: Path | None,
    output_dir: Path,
    withheld: float = DEFAULT_WITHHELD_FRACTION,
    seed: int = 42,
    retrieve_fn: RetrieveFn | None = None,
    target_mode: str = TARGETS_REFERENCE,
) -> dict[str, Any]:
    """Write the run's passage-prompt train / validation files under
    ``output_dir/passage_prompts/`` and return the block training_service
    records on the run (paths, retrieval, counts). Sync (retrieval runs the
    reranker); raises ``ValueError`` when the project has no document
    passages or no row could be rewritten — an opt-in that silently trained
    on bare prompts would mislead.

    ``target_mode="teacher"``: the targets are the cached teacher answers to
    exactly the served passages (``teacher_passage_answers_service``) and the
    prompt is the preamble the teacher read — no gold insertion, no withheld
    rows; rows without a usable teacher answer are left out (counted)."""
    from app.services.auto_rag_service import (
        load_document_passages,
        load_rechunked_passages,
        normalize_retrieval,
        retrieval_label,
    )

    served = normalize_retrieval(retrieval)
    if target_mode == TARGETS_TEACHER:
        return _apply_teacher_targets(
            project_id, retrieval=served, train_file=train_file, val_file=val_file, output_dir=output_dir,
        )
    source_passages = load_document_passages(project_id)
    if not source_passages:
        raise ValueError(
            "Training with retrieved passages needs the project's cleaned document passages; this project has none."
        )
    corpus_passages = (
        load_rechunked_passages(project_id, served["chunk_size"]) if served["chunk_size"] else source_passages
    )

    fn = retrieve_fn or served_retrieve_fn(project_id)
    target_dir = output_dir / "passage_prompts"
    files: dict[str, str | None] = {"train_file": None, "val_file": None}
    counts: dict[str, dict[str, int]] = {}
    for split, path in (("train", train_file), ("val", val_file)):
        if path is None or not Path(path).exists():
            continue
        rows, split_counts = build_passage_prompt_rows(
            _read_jsonl(Path(path)),
            retrieval=served,
            retrieve_fn=fn,
            corpus_passages=corpus_passages,
            source_passages=source_passages,
            withheld=withheld,
            seed=seed,
        )
        if split == "train" and not rows:
            raise ValueError("No training row could be rewritten with retrieved passages (rows need a question and an answer).")
        out = target_dir / f"{split}.jsonl"
        _write_jsonl(out, rows)
        files[f"{split}_file"] = str(out)
        counts[split] = split_counts
    train_counts = counts.get("train") or {}
    shown = max(1, train_counts.get("rows_out", 0) - train_counts.get("withheld", 0))
    block = {
        "applied": True,
        "retrieval": served,
        "retrieval_label": retrieval_label(served),
        "withheld_fraction": withheld,
        "withheld_target": NOT_IN_PASSAGES_ANSWER,
        "gold_support_min": GOLD_SUPPORT_MIN,
        "seed": seed,
        **files,
        "counts": counts,
        # Share of the non-withheld training rows whose answer passage
        # retrieval missed and the transform put back.
        "gold_inserted_share": round(train_counts.get("inserted", 0) / shown, 4),
    }
    (target_dir / "report.json").write_text(json.dumps(block, indent=2), encoding="utf-8")
    return block


def _apply_teacher_targets(
    project_id: int,
    *,
    retrieval: dict[str, Any],
    train_file: Path,
    val_file: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    from app.services.auto_rag_service import retrieval_label
    from app.services.teacher_passage_answers_service import (
        TEACHER_STYLE_NOTE,
        load_answers,
        teacher_provenance,
        teacher_rows,
    )

    answers = load_answers(project_id)
    if not answers:
        raise ValueError(
            "Training on teacher answers needs the project's cached teacher answers; run the documents → Q&A "
            "flow with answers_from_teacher first."
        )
    target_dir = output_dir / "passage_prompts"
    files: dict[str, str | None] = {"train_file": None, "val_file": None}
    counts: dict[str, dict[str, int]] = {}
    provenance: dict[str, Any] | None = None
    for split, path in (("train", train_file), ("val", val_file)):
        if path is None or not Path(path).exists():
            continue
        rows, split_counts = teacher_rows(_read_jsonl(Path(path)), answers, retrieval)
        if split == "train":
            # Who filtered / revised the targets, frozen with the run (the
            # cache is project-wide and later flows rewrite it).
            provenance = teacher_provenance(rows, answers, retrieval)
        if split == "train" and not rows:
            raise ValueError(
                "No training row has a teacher answer for the retrieval the project serves now "
                f"({retrieval_label(retrieval)}); re-run the teacher step."
            )
        out = target_dir / f"{split}.jsonl"
        _write_jsonl(out, rows)
        files[f"{split}_file"] = str(out)
        counts[split] = split_counts
    teachers = sorted({str(a.get("teacher")) for a in answers.values() if a.get("teacher")})
    block = {
        "applied": True,
        "targets": TARGETS_TEACHER,
        "teacher": teachers[0] if len(teachers) == 1 else teachers,
        "teacher_style_note": TEACHER_STYLE_NOTE,
        "retrieval": retrieval,
        "retrieval_label": retrieval_label(retrieval),
        "withheld_fraction": 0.0,
        **files,
        "counts": counts,
        "filter": (provenance or {}).get("filter"),
        "revise": (provenance or {}).get("revise"),
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "report.json").write_text(json.dumps(block, indent=2), encoding="utf-8")
    return block
