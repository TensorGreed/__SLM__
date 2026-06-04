"""Auto-RAG retrieval index + lookup (USER-SUCCESS Epic 9 Phase 9a).

For QA-SFT projects, the platform should be able to retrieve relevant
(question, answer) pairs from the training data at inference time so
the model can ground its answer in known-good context — rather than
relying on what it memorized during fine-tuning.

This module is the **index surface only**. Inference-time retrieve +
prepend lands in Phase 9b; the A/B harness in Phase 9c; the UI +
default-on heuristic in Phase 9d. Phase 9a ships:

  * ``build_bm25_index(rows, *, recipe_id, output_dir)`` — tokenize
    each row's (question, answer) text, persist a BM25 index to
    disk as JSON for inspectability.
  * ``retrieve(query, *, index_dir, k)`` — load the index, score
    the query against every row, return the top-K rows with their
    full payload + BM25 score.
  * ``recommended_text_keys_for_recipe(recipe_id)`` — recipe → the
    fields to concatenate as the retrieval corpus. QA-SFT ships in
    Phase 9a; other RAG-eligible recipes plug their own field
    tuples in later phases.

The public API is **recipe-agnostic** per the ``keep-brewslm-general``
design rule: the recipe lives in a small map, not in function names.

Tokenizer design: regex-based, lowercase + alphanumeric + apostrophes
(so "don't" / "user's" stay intact). Deliberately *not* using
``evaluation_service._normalize_answer`` — that strips punctuation
AND articles ("the", "a", "an"), which is the right call for
answer-overlap F1 but wrong for retrieval where rare tokens are the
signal. Also deliberately not adding ``nltk`` as a dep just to get
``word_tokenize`` — its quality bump over the regex doesn't justify
the install + data-bootstrap risk (the ``punkt`` tokenizer requires
a separate download step that's easy to miss in fresh environments).

BM25 parameters: standard ``k1=1.5``, ``b=0.75`` (Robertson & Walker
defaults from the original paper). Tuning these is Phase 9c's job
if the A/B shows the index is leaving lift on the table.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, TypedDict


# Public type — recipes that have a Phase 9a corpus shape. Add a new
# entry to the map below when a new recipe gains a RAG corpus shape.
# Arc R-1: rag-protocol indexes (context, question, answer) so the
# retrieval surface mirrors the protocol-aware fine-tune's training
# shape. Stage 2 customers (ecom-FAQ / legal / support) bolt their
# own data onto the same shape.
_RECIPE_TO_TEXT_KEYS: dict[str, tuple[str, ...]] = {
    "qa-sft": ("question", "answer"),
    "rag-protocol": ("context", "question", "answer"),
}


def recommended_text_keys_for_recipe(recipe_id: str) -> tuple[str, ...] | None:
    """Maps recipe_id → the row fields whose text should be
    concatenated to form the retrieval corpus, or None when no
    auto-RAG corpus shape ships for that recipe yet.

    QA-SFT indexes (question, answer) pairs so retrieval finds rows
    where either the asked question or the known answer matches the
    query semantically. Returning the answer field in the corpus is
    deliberate — at inference time the model needs the *answer* as
    grounded context, not just a similar question."""
    return _RECIPE_TO_TEXT_KEYS.get(recipe_id)


class RetrievedChunk(TypedDict):
    """One BM25 hit + the full payload of the source row.

    - ``row_id`` mirrors the row's own id field (or its 0-indexed
      position in the input list when no id was present).
    - ``score`` is the BM25 score; higher = better match. Not bounded.
    - ``payload`` is the full source row, so callers can extract
      whatever fields they need (Phase 9b will read both question
      and answer to build the prepend context).
    """

    row_id: int | str
    score: float
    payload: dict[str, Any]


class AutoRagIndexManifest(TypedDict):
    """Manifest the API + Phase 9b inference path consume.

    - ``index_path`` is the on-disk JSON the retrieve helper loads.
    - ``recipe_id`` + ``text_keys`` are stamped into the index file
      so a stale index from a recipe rename doesn't silently mis-
      retrieve.
    - ``doc_count`` + ``avg_doc_length`` are summary stats for
      observability (the preview endpoint surfaces them).
    """

    index_path: str
    recipe_id: str
    text_keys: list[str]
    doc_count: int
    avg_doc_length: float


class AutoRagUnavailable(RuntimeError):
    """Raised when the index can't be built or loaded — empty corpus,
    unsupported recipe, missing index file, corrupt index payload."""


# ─────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────


# Word-ish tokens: lowercase letters/digits + apostrophe-joined parts
# ("don't" → one token, "user's" → one token). Stays out of
# punctuation/whitespace; doesn't strip stopwords (BM25's idf handles
# common-word de-weighting naturally — explicit stopword lists tend
# to hurt retrieval quality on short query phrases).
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


# ─────────────────────────────────────────────────────────────────────
# BM25 (hand-rolled — no rank-bm25 dep)
# ─────────────────────────────────────────────────────────────────────


# Robertson & Walker (1994) defaults from the original BM25 paper.
# k1 controls term-frequency saturation; b controls length
# normalization. These are the values rank-bm25 uses out of the box
# too, and they generalize well across corpora.
_BM25_K1: float = 1.5
_BM25_B: float = 0.75


def _bm25_idf(doc_count: int, doc_freq: int) -> float:
    """BM25 idf — smoothed log-form so a term that appears in every
    doc still gets a non-zero (very small) weight, rather than the
    Robertson-Sparck-Jones form that goes negative for high doc-freq
    terms."""
    return math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    *,
    doc_count: int,
    avg_doc_length: float,
    doc_freq: dict[str, int],
) -> float:
    """BM25 score for one (query, doc) pair. Standard formulation."""
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_term_counts = Counter(doc_tokens)
    doc_length = len(doc_tokens)
    score = 0.0
    for term in query_tokens:
        tf = doc_term_counts.get(term, 0)
        if tf == 0:
            continue
        df = doc_freq.get(term, 0)
        if df == 0:
            continue
        idf = _bm25_idf(doc_count, df)
        # Standard BM25 term-frequency contribution.
        numerator = tf * (_BM25_K1 + 1)
        denominator = tf + _BM25_K1 * (
            1 - _BM25_B + _BM25_B * doc_length / (avg_doc_length or 1.0)
        )
        score += idf * numerator / denominator
    return score


# ─────────────────────────────────────────────────────────────────────
# Index build + persist
# ─────────────────────────────────────────────────────────────────────


def _extract_corpus_text(row: dict[str, Any], text_keys: tuple[str, ...]) -> str:
    """Concatenate the text_keys' values from the row into one corpus
    document. Handles nested ``input: {question: ...}`` /
    ``expected: {answer: ...}`` shapes used by template gold rows."""
    parts: list[str] = []
    for key in text_keys:
        # Top-level direct hit first.
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
            continue
        # Then the nested ``input.<key>`` and ``expected.<key>`` shapes.
        for parent_field in ("input", "expected"):
            parent = row.get(parent_field)
            if isinstance(parent, dict):
                nested = parent.get(key)
                if isinstance(nested, str) and nested.strip():
                    parts.append(nested.strip())
                    break
    return " ".join(parts)


def build_bm25_index(
    rows: list[dict[str, Any]],
    *,
    recipe_id: str,
    output_dir: Path,
) -> AutoRagIndexManifest:
    """Build a BM25 index over ``rows`` and persist it to
    ``output_dir/bm25_index.json``.

    Raises ``AutoRagUnavailable`` when:
      - ``rows`` is empty
      - ``recipe_id`` has no recommended text keys
      - every row's corpus text is empty after extraction (the
        retrieve helper would always return 0 results)

    Returns a manifest with the on-disk path + summary stats. The
    index file is JSON — bigger than a pickled scipy sparse matrix
    but inspectable with ``cat`` + survives Python version bumps.
    """
    if not rows:
        raise AutoRagUnavailable(
            "Cannot build BM25 index from an empty row list."
        )
    text_keys = recommended_text_keys_for_recipe(recipe_id)
    if text_keys is None:
        raise AutoRagUnavailable(
            f"Recipe {recipe_id!r} has no auto-RAG corpus shape. "
            f"Known RAG-eligible recipes: "
            f"{', '.join(sorted(_RECIPE_TO_TEXT_KEYS.keys())) or '(none)'}."
        )

    row_ids: list[int | str] = []
    doc_tokens_per_row: list[list[str]] = []
    for idx, row in enumerate(rows):
        rid = row.get("id") if isinstance(row.get("id"), (int, str)) else idx
        row_ids.append(rid)
        text = _extract_corpus_text(row, text_keys)
        doc_tokens_per_row.append(_tokenize(text))

    # Drop rows whose corpus text tokenized to nothing — they can't
    # be retrieved and would just inflate the average-doc-length
    # denominator. Track their ids in the manifest's "skipped" list
    # in case the caller cares (Phase 9a doesn't surface it).
    kept_indices = [i for i, toks in enumerate(doc_tokens_per_row) if toks]
    if not kept_indices:
        raise AutoRagUnavailable(
            f"Every row had an empty corpus text after extracting "
            f"fields {list(text_keys)!r}. Are these the right field "
            f"names for this recipe?"
        )

    doc_count = len(kept_indices)
    avg_doc_length = sum(
        len(doc_tokens_per_row[i]) for i in kept_indices
    ) / doc_count

    # Inverted doc-freq: how many docs contain each term?
    doc_freq: dict[str, int] = {}
    for i in kept_indices:
        for term in set(doc_tokens_per_row[i]):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    payload = {
        # Stamp recipe + tokenizer settings into the index file so a
        # future tokenizer / recipe change auto-invalidates retrieval
        # rather than silently mis-scoring against a stale index.
        "recipe_id": recipe_id,
        "text_keys": list(text_keys),
        "bm25_k1": _BM25_K1,
        "bm25_b": _BM25_B,
        "doc_count": doc_count,
        "avg_doc_length": avg_doc_length,
        "doc_freq": doc_freq,
        "rows": [
            {
                "row_id": row_ids[i],
                "doc_tokens": doc_tokens_per_row[i],
                "payload": rows[i],
            }
            for i in kept_indices
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "bm25_index.json"
    index_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "index_path": str(index_path),
        "recipe_id": recipe_id,
        "text_keys": list(text_keys),
        "doc_count": doc_count,
        "avg_doc_length": avg_doc_length,
    }


# ─────────────────────────────────────────────────────────────────────
# Retrieve
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Phase 9b — project-scoped helpers (training completion + inference)
# ─────────────────────────────────────────────────────────────────────


async def _load_rag_corpus_rows(db, project_id: int) -> list[dict[str, Any]]:
    """Load the rows the BM25 index should be built over.

    Reuses ``dataset_service._load_records_from_file`` (which excludes
    pending synth rows) so the corpus stays in sync with what the
    training pipeline trained on. Same loader the preview API uses.
    """
    from sqlalchemy import select

    from app.config import settings
    from app.models.dataset import Dataset, DatasetType
    from app.services.dataset_service import _load_records_from_file

    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                [
                    DatasetType.GOLD_DEV,
                    DatasetType.GOLD_TEST,
                    DatasetType.SYNTHETIC,
                ]
            ),
        )
    )
    rows: list[dict[str, Any]] = []
    for dataset in result.scalars():
        if not dataset.file_path:
            continue
        path = Path(dataset.file_path)
        if not path.exists():
            continue
        rows.extend(_load_records_from_file(path))
    # Also fall back to the prepared/train.jsonl file when the
    # Dataset rows don't surface anything (e.g. older projects that
    # never registered a Dataset row but did write the prepared
    # split). Phase 9b's training-completion hook always has the
    # prepared file written, so this fallback is the load-bearing
    # path for index rebuilds.
    if not rows:
        prepared = (
            settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "train.jsonl"
        )
        if prepared.exists():
            try:
                with prepared.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                pass
    return rows


async def build_index_for_project(db, project_id: int) -> dict[str, Any]:
    """Phase 9b training-completion hook. Builds the BM25 index for
    ``project_id`` if (recipe is RAG-eligible AND there are rows to
    index). **Never raises** — failures land in the returned dict so
    a broken build can't take down training completion.

    Returns ``{built: bool, reason: str, index_path: str|None,
    doc_count: int|None}`` for observability (the caller can stamp
    this onto the experiment's runtime_config).
    """
    from app.config import settings
    from app.models.project import Project

    project = await db.get(Project, project_id)
    if project is None:
        return {"built": False, "reason": "project_not_found"}
    selected_recipe = project.selected_recipe or {}
    recipe_id = selected_recipe.get("recipe_id")
    if not recipe_id:
        return {"built": False, "reason": "no_recipe_selected"}
    if recommended_text_keys_for_recipe(recipe_id) is None:
        return {
            "built": False,
            "reason": f"recipe_has_no_auto_rag:{recipe_id}",
        }
    rows = await _load_rag_corpus_rows(db, project_id)
    if not rows:
        return {"built": False, "reason": "no_corpus_rows"}
    index_dir = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag"
    try:
        manifest = build_bm25_index(
            rows,
            recipe_id=recipe_id,
            output_dir=index_dir,
        )
    except AutoRagUnavailable as e:
        return {"built": False, "reason": f"build_failed:{e}"}
    return {
        "built": True,
        "reason": "ok",
        "index_path": manifest["index_path"],
        "doc_count": manifest["doc_count"],
        "recipe_id": recipe_id,
    }


# Prepend preamble template — known shape so Phase 9c's A/B can
# attribute lift to this specific framing (a future bake-off
# variant would compare alternative phrasings against this baseline).
# Stays a short, neutral instruction that doesn't claim the
# retrieved pairs are authoritative — the model is being given
# *examples*, not *answers*.
_AUTO_RAG_PREAMBLE_TEMPLATE = (
    "Reference Q&A pairs from the knowledge base (use them to ground "
    "your answer; cite the matching pair number if you use one):\n"
    "{pairs}\n"
    "Now answer the user's next question."
)


async def build_preamble_from_query(
    db,
    project_id: int,
    query: str,
    *,
    k: int = 3,
) -> dict[str, Any] | None:
    """Phase 9b inference-time helper. Returns
    ``{preamble_text, retrieved}`` or ``None`` when auto-RAG should
    skip (recipe ineligible, no index, empty query, no hits). Never
    raises — broken-index errors return None so inference can fall
    back to no-RAG without blowing up.

    The preamble is a system-message-shaped string that the caller
    prepends to the chat messages. ``retrieved`` is the same shape
    the preview endpoint returns, so Phase 9d's interpretability
    panel can render it without a new contract.
    """
    from app.config import settings
    from app.models.project import Project

    project = await db.get(Project, project_id)
    if project is None:
        return None
    selected_recipe = project.selected_recipe or {}
    recipe_id = selected_recipe.get("recipe_id")
    if not recipe_id or recommended_text_keys_for_recipe(recipe_id) is None:
        return None  # recipe not eligible — silent skip
    index_dir = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag"
    if not (index_dir / "bm25_index.json").exists():
        return None  # no index yet — silent skip (the build hook runs at training completion)
    try:
        hits = retrieve(query, index_dir=index_dir, k=k)
    except AutoRagUnavailable:
        return None
    if not hits:
        return None
    pairs_text = "\n\n".join(_format_pair(idx, hit) for idx, hit in enumerate(hits, start=1))
    preamble_text = _AUTO_RAG_PREAMBLE_TEMPLATE.format(pairs=pairs_text)
    return {
        "preamble_text": preamble_text,
        "retrieved": hits,
    }


def _format_pair(idx: int, hit: RetrievedChunk) -> str:
    """Format one retrieved (Q, A) pair for the preamble. Handles
    both nested template shape and pre-flattened rows so the same
    formatter works regardless of which call path built the index."""
    payload = hit.get("payload") or {}
    question = ""
    answer = ""
    # Top-level first.
    if isinstance(payload.get("question"), str):
        question = payload["question"]
    if isinstance(payload.get("answer"), str):
        answer = payload["answer"]
    # Then nested input/expected.
    if not question:
        nested = payload.get("input")
        if isinstance(nested, dict) and isinstance(nested.get("question"), str):
            question = nested["question"]
    if not answer:
        nested = payload.get("expected")
        if isinstance(nested, dict) and isinstance(nested.get("answer"), str):
            answer = nested["answer"]
    return f"[{idx}] Q: {question}\n    A: {answer}"


def retrieve(
    query: str,
    *,
    index_dir: Path,
    k: int = 3,
) -> list[RetrievedChunk]:
    """Return the top-K rows by BM25 score for ``query``.

    Returns ``[]`` for an empty query (rather than raising) — the
    caller (Phase 9b's inference wrapper) can decide whether an empty
    query means "skip RAG" or "warn the user."

    Raises ``AutoRagUnavailable`` when the index file is missing or
    corrupt; the inference wrapper catches and falls back to no-RAG
    so a broken index can't take down inference."""
    if k < 1:
        return []

    index_path = index_dir / "bm25_index.json"
    if not index_path.exists():
        raise AutoRagUnavailable(
            f"BM25 index not found at {index_path}. Build it first via "
            f"``build_bm25_index(...)``."
        )
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise AutoRagUnavailable(
            f"BM25 index at {index_path} is unreadable: {e}"
        ) from e

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    doc_count = int(payload.get("doc_count") or 0)
    avg_doc_length = float(payload.get("avg_doc_length") or 0.0)
    doc_freq: dict[str, int] = payload.get("doc_freq") or {}
    rows = payload.get("rows") or []

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        score = _bm25_score(
            query_tokens=query_tokens,
            doc_tokens=list(row.get("doc_tokens") or []),
            doc_count=doc_count,
            avg_doc_length=avg_doc_length,
            doc_freq=doc_freq,
        )
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:k]
    return [
        {
            "row_id": row["row_id"],
            "score": float(score),
            "payload": dict(row.get("payload") or {}),
        }
        for score, row in top
    ]
