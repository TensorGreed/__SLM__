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
DOCUMENTS_CORPUS = "documents"

_RECIPE_TO_TEXT_KEYS: dict[str, tuple[str, ...]] = {
    "qa-sft": ("question", "answer"),
    "rag-protocol": ("context", "question", "answer"),
    # Not a recipe: the document-passage corpus (cleaned PDF / DOCX / HTML
    # passages), indexed separately under ``auto_rag/documents/``.
    DOCUMENTS_CORPUS: ("text",),
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


# Word-ish tokens: letters/digits/underscore + apostrophe-joined parts
# ("don't" → one token, "user's" → one token). Matched on the original
# case so camelCase / snake_case identifiers can be split. Stays out of
# punctuation/whitespace; doesn't strip stopwords (BM25's idf handles
# common-word de-weighting naturally — explicit stopword lists tend
# to hurt retrieval quality on short query phrases).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z]+)?")
# A lower→Upper step (``getUser``) or an acronym followed by a word (``HTTPServer``) is a
# boundary; a digit→Upper step is not (``5G``, ``v2API`` stay whole), and an all-caps
# plural (``IDs``, ``APIs``, ``URLs``) is one word.
_CASE_BOUNDARY_RE = re.compile(r"[a-z][A-Z]|[A-Z][A-Z][a-z]")
_IDENT_PART_RE = re.compile(r"[0-9]*[A-Z]+s(?![a-z])|[0-9]*[A-Z]+(?![a-z])|[0-9]*[A-Z]?[a-z]+|[0-9]+")
_CAPS_PLURAL_RE = re.compile(r"[A-Z]+s")

# Stamped into every index. v2: identifiers are split into their parts as
# well as kept whole (``getUserName`` → get user name + getusername), which
# is what lets a question about "user name" find code. An index built with
# another version is stale (``document_index_status`` / ``qa_index_is_current``).
BM25_TOKENIZER_VERSION = 3


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens. A token with an underscore or a case boundary
    (``getUserName``, ``HTTPServer``, ``user_id``) emits its lowercase parts
    (``get user name``) AND the whole lowercased identifier (``getusername``);
    plain lowercase / Capitalised / ALLCAPS words tokenize as before."""
    if not text:
        return []
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        if "'" in raw or ("_" not in raw and (_CAPS_PLURAL_RE.fullmatch(raw) or not _CASE_BOUNDARY_RE.search(raw))):
            tokens.append(raw.lower())
            continue
        parts = [
            part.lower()
            for piece in raw.split("_")
            if piece
            for part in ([piece] if _CAPS_PLURAL_RE.fullmatch(piece) else _IDENT_PART_RE.findall(piece))
        ]
        whole = raw.lower()
        if len(parts) != 1 or parts[0] != whole:
            tokens.extend(parts)
        tokens.append(whole)
    return tokens


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
    corpus_source: str | None = None,
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
        "tokenizer_version": BM25_TOKENIZER_VERSION,
        "corpus_source": corpus_source,
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


# Stamped into the Q&A index so an index built over an older corpus (gold +
# synthetic, before the answer key became eval-only) is rebuilt, not reused.
QA_CORPUS_SOURCE = "training_rows_v1"


def index_tokenizer_is_current(index_dir: Path) -> bool:
    """Whether the BM25 index under ``index_dir`` was built with this
    tokenizer (an index without the stamp predates it)."""
    try:
        payload = json.loads((index_dir / "bm25_index.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(payload, dict) and payload.get("tokenizer_version") == BM25_TOKENIZER_VERSION


def qa_index_is_current(index_path: Path) -> bool:
    """True when the Q&A index on disk was built over the training corpus."""
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("corpus_source") == QA_CORPUS_SOURCE
        and payload.get("tokenizer_version") == BM25_TOKENIZER_VERSION
    )


async def _load_rag_corpus_rows(db, project_id: int) -> list[dict[str, Any]]:
    """Load the rows the BM25 index should be built over: the project's
    training rows (``dataset_service.load_training_corpus_rows``).

    Never the answer key and never the val/test splits — the index used to
    hold GOLD_DEV + GOLD_TEST rows, so retrieval could return the exact
    question being scored together with its reference answer.
    """
    from app.services.dataset_service import load_training_corpus_rows

    return await load_training_corpus_rows(db, project_id)


def _drop_stale_qa_index(index_dir: Path) -> None:
    """Remove a Q&A index built over an older corpus when it can't be rebuilt
    (no training rows) — serving it would keep retrieving answer-key rows."""
    index_path = index_dir / "bm25_index.json"
    if index_path.exists() and not qa_index_is_current(index_path):
        try:
            index_path.unlink()
        except OSError:
            pass


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
    index_dir = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag"
    if not rows:
        _drop_stale_qa_index(index_dir)
        return {"built": False, "reason": "no_corpus_rows"}
    try:
        manifest = build_bm25_index(
            rows,
            recipe_id=recipe_id,
            output_dir=index_dir,
            corpus_source=QA_CORPUS_SOURCE,
        )
    except AutoRagUnavailable as e:
        _drop_stale_qa_index(index_dir)
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


# ─────────────────────────────────────────────────────────────────────
# Document passages (Wave 2c-2) — ground answers in the project's own
# documents, not only in Q&A pairs.
# ─────────────────────────────────────────────────────────────────────

_PASSAGE_PREVIEW_CHARS = 700

_DOCUMENT_PREAMBLE_TEMPLATE = (
    "Passages from this project's documents:\n"
    "{passages}\n"
    "Answer the user's next question using only these passages and cite the "
    "passage numbers you used, like [1]. If the passages don't contain the "
    "answer, say you don't know."
)


def document_index_dir(project_id: int, chunk_size: int | None = None) -> Path:
    """The passage index: the cleaning's own chunks at ``.../documents``, or
    a re-chunked variant (``chunk_size`` chars, built by
    ``ensure_document_index``) at ``.../documents/chunk-<n>``."""
    from app.config import settings

    base = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag" / DOCUMENTS_CORPUS
    return base / f"chunk-{int(chunk_size)}" if chunk_size else base


# Re-chunked passage variants (a retrieval sweep arm): the cleaning's
# per-document ``<file>.cleaned.txt`` (boilerplate removed, PII redacted) cut
# again with ``cleaning_service.chunk_text`` at a different size. Overlap
# keeps cleaning's default ratio (1/10 of the chunk).
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 4000


def _cleaned_text_files(project_id: int) -> list[Path]:
    from app.config import settings

    raw_dir = settings.DATA_DIR / "projects" / str(project_id) / "raw"
    return sorted(raw_dir.glob("*.cleaned.txt")) if raw_dir.exists() else []


def load_rechunked_passages(project_id: int, chunk_size: int) -> list[dict[str, Any]]:
    """The project's documents re-cut at ``chunk_size`` chars, in the same
    row shape as ``load_document_passages`` (document id + name from the
    cleaned.jsonl rows, chunk ids restarting at 0 per document)."""
    from app.services.cleaning_service import chunk_text
    from app.services.code_chunker import chunk_code

    chunk_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, int(chunk_size)))
    doc_ids: dict[str, Any] = {}
    code_docs: dict[str, dict[str, Any]] = {}
    for row in load_document_passages(project_id):
        name = row.get("source_doc")
        if isinstance(name, str) and name not in doc_ids:
            doc_ids[name] = row.get("source_document_id")
            if row.get("language") and row.get("path"):
                code_docs[name] = {"path": row["path"], "language": row["language"]}
    passages: list[dict[str, Any]] = []
    for path in _cleaned_text_files(project_id):
        source_doc = next(
            (name for name in doc_ids if Path(name).stem == path.name[: -len(".cleaned.txt")]),
            None,
        )
        if source_doc is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not text.strip():
            continue
        if source_doc in code_docs:
            # Source files are re-cut at declaration boundaries, never by sentence.
            info = code_docs[source_doc]
            for idx, chunk in enumerate(chunk_code(text, info["language"], max_chars=chunk_size)):
                passages.append({
                    "id": f"{doc_ids[source_doc]}:{idx}",
                    "text": chunk.text,
                    "source_doc": source_doc,
                    "source_document_id": doc_ids[source_doc],
                    "chunk_id": idx,
                    "path": info["path"],
                    "language": info["language"],
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "symbol": chunk.symbol,
                    "kind": chunk.kind,
                })
            continue
        for idx, chunk in enumerate(chunk_text(text, chunk_size, chunk_size // 10)):
            if chunk.strip():
                passages.append({
                    "id": f"{doc_ids[source_doc]}:{idx}",
                    "text": chunk.strip(),
                    "source_doc": source_doc,
                    "source_document_id": doc_ids[source_doc],
                    "chunk_id": idx,
                })
    return passages


def current_chunk_size_estimate(project_id: int) -> int | None:
    """The cleaning's chunk size, estimated as the longest cleaned passage
    (``chunk_text`` never exceeds it; most chunks reach it). None without
    passages."""
    lengths = [len(p["text"]) for p in load_document_passages(project_id)]
    return max(lengths) if lengths else None


def _cleaned_path(project_id: int) -> Path:
    from app.config import settings

    return settings.DATA_DIR / "projects" / str(project_id) / "cleaned" / "cleaned.jsonl"


_CODE_PASSAGE_KEYS = ("path", "language", "start_line", "end_line", "symbol", "kind")


def load_document_passages(project_id: int) -> list[dict[str, Any]]:
    """Cleaned document passages (chunks of PDF / DOCX / HTML / text).
    Rows from structured files (CSV/XLSX…, which cleaning keeps
    row-by-row with a ``row_index``) are Q&A / label data, not prose, and
    are left to the Q&A index."""
    path = _cleaned_path(project_id)
    if not path.exists():
        return []
    passages: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or "row_index" in row:
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            document_id = row.get("source_document_id")
            chunk_id = row.get("chunk_id")
            passage = {
                "id": f"{document_id}:{chunk_id}",
                "text": text.strip(),
                "source_doc": row.get("source_doc"),
                "source_document_id": document_id,
                "chunk_id": chunk_id,
            }
            # Code documents carry where the chunk lives (cited as path:start-end).
            for key in _CODE_PASSAGE_KEYS:
                if key in row:
                    passage[key] = row[key]
            passages.append(passage)
    return passages


def document_index_status(project_id: int, chunk_size: int | None = None) -> dict[str, Any]:
    index_path = document_index_dir(project_id, chunk_size) / "bm25_index.json"
    cleaned = _cleaned_path(project_id)
    exists = index_path.exists()
    stale = bool(
        exists and cleaned.exists() and cleaned.stat().st_mtime > index_path.stat().st_mtime
    )
    if exists and chunk_size:
        # A variant is cut from the per-document cleaned text.
        built_at = index_path.stat().st_mtime
        stale = stale or any(f.stat().st_mtime > built_at for f in _cleaned_text_files(project_id))
    passages = None
    if exists:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            passages = int(payload.get("doc_count") or 0)
            # An index built with another tokenizer scores differently: rebuild it.
            if payload.get("tokenizer_version") != BM25_TOKENIZER_VERSION:
                stale = True
        except (OSError, json.JSONDecodeError, AttributeError):
            passages = None
    return {"available": exists and not stale, "exists": exists, "stale": stale, "passages": passages}


def ensure_document_index(project_id: int, chunk_size: int | None = None) -> dict[str, Any]:
    """Build (or rebuild when cleaning ran since) the passage index — the
    cleaning's chunks, or the ``chunk_size`` variant
    (``load_rechunked_passages``). Never raises; returns a status dict."""
    status = document_index_status(project_id, chunk_size)
    if status["available"]:
        return {**status, "built": False, "reason": "up_to_date"}
    passages = load_rechunked_passages(project_id, chunk_size) if chunk_size else load_document_passages(project_id)
    if not passages:
        return {**status, "available": False, "built": False, "reason": "no_document_passages"}
    try:
        manifest = build_bm25_index(
            passages, recipe_id=DOCUMENTS_CORPUS, output_dir=document_index_dir(project_id, chunk_size)
        )
    except AutoRagUnavailable as exc:
        return {**status, "available": False, "built": False, "reason": f"build_failed:{exc}"}
    return {
        "available": True,
        "exists": True,
        "stale": False,
        "passages": manifest["doc_count"],
        "built": True,
        "reason": "ok",
    }


def _format_passage(idx: int, hit: RetrievedChunk) -> str:
    payload = hit.get("payload") or {}
    text = str(payload.get("text") or "")
    if payload.get("path") and isinstance(payload.get("start_line"), int) and isinstance(payload.get("end_line"), int):
        # Source code: cite the file + line range, keep the layout (indentation matters).
        where = f"{payload['path']}:{payload['start_line']}-{payload['end_line']}"
        if payload.get("symbol"):
            where += f" ({payload['symbol']})"
        if len(text) > _PASSAGE_PREVIEW_CHARS:
            text = text[:_PASSAGE_PREVIEW_CHARS].rsplit("\n", 1)[0] + "\n…"
        return f"[{idx}] {where}\n{text}"
    if len(text) > _PASSAGE_PREVIEW_CHARS:
        text = text[:_PASSAGE_PREVIEW_CHARS].rsplit(" ", 1)[0] + " …"
    source = payload.get("source_doc") or "document"
    chunk = payload.get("chunk_id")
    where = f"{source} · passage {int(chunk) + 1}" if isinstance(chunk, int) else str(source)
    return f"[{idx}] ({where}) {text}"


def document_preamble(hits: list[RetrievedChunk]) -> str:
    """The cite-or-say-you-don't-know system message for retrieved document
    passages — the one string serving (``build_preamble_from_query``), the
    auto-RAG harness and passage-prompt training
    (``passage_prompts_service``) all put in front of the question, so a
    model trained on it sees exactly what it is served."""
    passages_text = "\n\n".join(_format_passage(idx, hit) for idx, hit in enumerate(hits, start=1))
    return _DOCUMENT_PREAMBLE_TEMPLATE.format(passages=passages_text)


def retrieve_document_passages(project_id: int, query: str, retrieval: Any) -> list[RetrievedChunk]:
    """The passages a retrieval setting (``normalize_retrieval``) returns for
    ``query`` from the project's document index (the ``chunk_size`` variant
    when set, built on demand). Sync; raises ``AutoRagUnavailable`` when no
    index can be built."""
    r = normalize_retrieval(retrieval)
    chunk_size = r["chunk_size"]
    if not ensure_document_index(project_id, chunk_size).get("available"):
        raise AutoRagUnavailable(f"Project {project_id} has no document passages to retrieve from.")
    return retrieve_ranked(
        query, index_dir=document_index_dir(project_id, chunk_size), k=r["k"], reranker=r["reranker"], pool=r["pool"]
    )


def _qa_index_usable(project: Any, project_id: int) -> bool:
    from app.config import settings

    recipe_id = (project.selected_recipe or {}).get("recipe_id")
    if not recipe_id or recipe_id == DOCUMENTS_CORPUS or recommended_text_keys_for_recipe(recipe_id) is None:
        return False
    return (settings.DATA_DIR / "projects" / str(project_id) / "auto_rag" / "bm25_index.json").exists()


RETRIEVAL_SETTINGS_KEY = "auto_rag_retrieval"
DEFAULT_RETRIEVAL_K = 3
RETRIEVAL_FIELDS = ("k", "reranker", "pool", "chunk_size")


def _opt_int(value: Any, lo: int, hi: int) -> int | None:
    if value is None or value == "":
        return None
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return None


def normalize_retrieval(raw: Any) -> dict[str, Any]:
    """A retrieval setting ``{"k", "reranker", "pool", "chunk_size"}`` from
    any dict-ish input: ``k`` 1–10 (default 3), ``reranker`` a model id or
    None, ``pool`` the reranker's BM25 candidate count (None = the default
    rule; meaningless without a reranker, so dropped), ``chunk_size`` a
    re-chunked passage variant in chars (None = the cleaning's chunks)."""
    from app.services.retrieval_reranker import MAX_POOL, normalize_reranker

    raw = raw if isinstance(raw, dict) else {}
    reranker = normalize_reranker(raw.get("reranker"))
    return {
        "k": _opt_int(raw.get("k"), 1, 10) or DEFAULT_RETRIEVAL_K,
        "reranker": reranker,
        "pool": _opt_int(raw.get("pool"), 1, MAX_POOL) if reranker else None,
        "chunk_size": _opt_int(raw.get("chunk_size"), MIN_CHUNK_SIZE, MAX_CHUNK_SIZE),
    }


def same_retrieval(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    return normalize_retrieval(a) == normalize_retrieval(b)


def retrieval_label(config: Any) -> str:
    """``top-5 + reranker ms-marco-MiniLM-L-6-v2 · pool 50 · 600-char passages``."""
    r = normalize_retrieval(config)
    label = f"top-{r['k']}"
    if r["reranker"]:
        label += f" + reranker {r['reranker'].split('/')[-1]}"
        if r["pool"]:
            label += f" · pool {r['pool']}"
    if r["chunk_size"]:
        label += f" · {r['chunk_size']}-char passages"
    return label


def retrieval_settings(project: Any) -> dict[str, Any]:
    """The project's retrieval settings (``normalize_retrieval`` of
    ``runtime_config["auto_rag_retrieval"]``, set by the retrieval sweep or
    pinned by hand); defaults: top-3, no reranker, default pool, the
    cleaning's chunks."""
    cfg = getattr(project, "runtime_config", None)
    return normalize_retrieval(cfg.get(RETRIEVAL_SETTINGS_KEY) if isinstance(cfg, dict) else None)


def retrieve_ranked(
    query: str,
    *,
    index_dir: Path,
    k: int = DEFAULT_RETRIEVAL_K,
    reranker: str | None = None,
    pool: int | None = None,
) -> list[RetrievedChunk]:
    """BM25 top-k, or — with a reranker — ``pool`` BM25 candidates
    (``candidate_pool``) reranked by the cross-encoder and cut to k
    (``retrieval_reranker``)."""
    from app.services.retrieval_reranker import candidate_pool, rerank

    if not reranker:
        return retrieve(query, index_dir=index_dir, k=k)
    hits = retrieve(query, index_dir=index_dir, k=candidate_pool(k, pool))
    return rerank(query, hits, top_k=k, model_name=reranker)


async def build_preamble_from_query(
    db,
    project_id: int,
    query: str,
    *,
    k: int | None = None,
    corpus: str = "auto",
    retrieval_override: dict[str, Any] | None = None,
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
    # ``k`` unset → the project's retrieval settings (top-k + reranker,
    # chosen by the retrieval sweep); an explicit k keeps the reranker.
    settings_used = retrieval_settings(project)
    if k is None:
        k = settings_used["k"]
    # ``retrieval_override`` (the playground's "try the other retrieval")
    # replaces the given fields — reranker / pool / chunk_size — for this
    # call only.
    override = {f: v for f, v in (retrieval_override or {}).items() if f in RETRIEVAL_FIELDS and f != "k"}
    retrieval = normalize_retrieval({**settings_used, "k": k, **override})
    reranker, pool, chunk_size = retrieval["reranker"], retrieval["pool"], retrieval["chunk_size"]
    # ``corpus``: "qa" (Q&A-pair index, recipe-gated), "documents" (the
    # project's document passages) or "auto" — Q&A pairs when the recipe
    # has them, otherwise document passages.
    use_qa = corpus in {"qa", "auto"} and _qa_index_usable(project, project_id)
    if use_qa:
        index_dir = settings.DATA_DIR / "projects" / str(project_id) / "auto_rag"
        if not qa_index_is_current(index_dir / "bm25_index.json"):
            # Built before the corpus excluded the answer key — rebuild over
            # the training rows (drops the file when there are none).
            use_qa = bool((await build_index_for_project(db, project_id)).get("built"))
    if use_qa:
        try:
            hits = retrieve_ranked(query, index_dir=index_dir, k=k, reranker=reranker, pool=pool)
        except AutoRagUnavailable:
            hits = []
        if hits:
            pairs_text = "\n\n".join(_format_pair(idx, hit) for idx, hit in enumerate(hits, start=1))
            return {
                "preamble_text": _AUTO_RAG_PREAMBLE_TEMPLATE.format(pairs=pairs_text),
                "retrieved": hits,
                "corpus": "qa",
                # Q&A pairs are not re-chunked.
                "retrieval": {**retrieval, "chunk_size": None},
            }
    if corpus not in {"documents", "auto"}:
        return None
    if not ensure_document_index(project_id, chunk_size).get("available"):
        if not chunk_size or not ensure_document_index(project_id).get("available"):
            return None
        # The variant can't be built (no cleaned text files): serve the
        # cleaning's chunks and say so.
        chunk_size = None
        retrieval = {**retrieval, "chunk_size": None}
    try:
        hits = retrieve_ranked(
            query, index_dir=document_index_dir(project_id, chunk_size), k=k, reranker=reranker, pool=pool
        )
    except AutoRagUnavailable:
        return None
    if not hits:
        return None
    return {
        "preamble_text": document_preamble(hits),
        "retrieved": hits,
        "corpus": DOCUMENTS_CORPUS,
        "retrieval": retrieval,
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

    if not isinstance(payload, dict) or payload.get("tokenizer_version") != BM25_TOKENIZER_VERSION:
        # Query tokens from this tokenizer would be scored against differently
        # tokenized documents — silently wrong. Say so (callers rebuild the index).
        raise AutoRagUnavailable(
            f"BM25 index at {index_path} was built with tokenizer version "
            f"{payload.get('tokenizer_version') if isinstance(payload, dict) else None}, "
            f"not {BM25_TOKENIZER_VERSION}; rebuild it."
        )
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
