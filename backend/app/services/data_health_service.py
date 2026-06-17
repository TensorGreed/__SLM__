"""Data Health Report — D1 of the data-quality arc.

Aggregates every existing data-quality signal scattered across the
ingestion / cleaning / dataset-prep / trainability-forecast services
into a single payload the ``DataHealthReportPanel`` consumes. Each
signal carries a **plain-English** translation alongside the
technical headline, plus a "why this matters" line explaining the
training-time consequence — so a user staring at "entropy = 0.62"
gets "your classes are very uneven" + "the model will only learn
the dominant class" without having to look up vocabulary.

D1 is read-only — every signal's ``suggested_action`` is informational
("here's what you'd do"), not actionable yet. Auto-fix wiring lands
in D3/D4 of the arc.

Groups, in the order the panel renders them:

1. **Ingestion** — what came in. Document count, parse failures.
2. **Cleaning** — what got cleaned up. PII findings, duplicates,
   low-quality docs.
3. **Shape** — does the data match the recipe? Recipe selected?
   Prepared rows above the floor? (Schema mismatch detective lands
   in D5.)
4. **Balance** — for classification: class distribution + per-class
   minimums (delegated to trainability_forecast_service's existing
   classification signals so the data-health report and Coach Mode
   stay aligned — one source of truth for the threshold).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.models.project import Project


Severity = Literal["ok", "warn", "block"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────
# Thresholds — single source of truth for what "bad" means at each
# level. Where a threshold already exists in another service (e.g.
# Coach Mode's 15% class-imbalance floor), we re-use the same number
# so the panels stay aligned — no double vocabulary for the user.
# ─────────────────────────────────────────────────────────────────────

# Ingestion parse-failure rate above this surfaces a warning; above
# the block threshold the dataset is too damaged to train usefully.
# 10% / 25% are the empirical brackets we've seen between "a couple
# corrupt PDFs" and "you uploaded the wrong file type."
PARSE_FAILURE_WARN = 0.10
PARSE_FAILURE_BLOCK = 0.25

# Low-quality fraction in cleaning. quality_score < 0.4 = "this doc
# is mostly boilerplate / encoding garble / too short." Acceptable
# in small amounts (drift from one bad PDF); above 30% means the
# corpus itself is the problem.
LOW_QUALITY_THRESHOLD = 0.4
LOW_QUALITY_WARN_FRAC = 0.10
LOW_QUALITY_BLOCK_FRAC = 0.30

# Duplicate chunk fraction. text_hash collisions across the cleaned
# dataset. A few % is normal (chunk overlaps); above 20% means the
# user uploaded near-duplicate documents and the model will overfit
# on the duplicates.
DUPLICATE_CHUNK_WARN_FRAC = 0.10
DUPLICATE_CHUNK_BLOCK_FRAC = 0.30

# PII count buckets. A handful of emails in a corpus is normal; a
# hundred suggests the cleaning step didn't redact and the model
# will memorise PII at training time.
PII_WARN_COUNT = 5
PII_BLOCK_COUNT = 50

# ── Train ↔ gold leakage ──────────────────────────────────────────
# The single most dangerous data-health failure for a newbie: gold
# rows that also appear in the training set. When that happens the
# eval pass-rate measures memorisation, not generalisation — a green
# gate that is a lie the user cannot see. We catch both exact
# duplicates (the common bad-split / copy-paste case) and near-
# duplicates (synthetic paraphrases of gold rows bleeding into train)
# via token-set Jaccard.
#
# Jaccard ≥ this between a gold row and any training row = "the same
# example." 0.9 tolerates trivial edits (punctuation, a swapped word)
# while staying clear of merely topically-similar rows.
LEAKAGE_FUZZY_THRESHOLD = 0.9
# Leaked-gold fraction → severity. Any leakage at all warns (see
# _scan_leakage_rows); at or above this fraction it's a hard blocker
# because the pass-rate the gold set feeds is now untrustworthy.
LEAKAGE_BLOCK_FRAC = 0.10
# Bounded-compute caps so the data-health poll endpoint stays fast on
# big corpora. Training rows are indexed once; gold rows are matched
# against an inverted token index, so each gold row only compares
# against candidate training rows that share tokens, not the whole set.
LEAKAGE_GOLD_SCAN_CAP = 2000
LEAKAGE_TRAIN_SCAN_CAP = 5000
LEAKAGE_MAX_POSTINGS = 256       # cap an inverted-index posting list
LEAKAGE_MAX_CANDIDATES = 400     # cap fuzzy candidates per gold row
# Rows shorter than this (in tokens) are too short to judge fuzzy
# overlap reliably — one shared word swings Jaccard wildly — so they're
# matched on exact-normalised equality only.
LEAKAGE_MIN_TOKENS = 4
# How many leaked-row examples to surface for the drill-down (each
# carries the leaked excerpt + the source row it matched). Short
# excerpts, loaded on the data tab (not a 4s poll), so the budget can
# be generous enough to actually inspect the leak.
LEAKAGE_EXAMPLE_LIMIT = 25


# ─────────────────────────────────────────────────────────────────────
# Layman translation tables. Keyed on the technical signal id; each
# row carries the plain-English summary + "why this matters" line.
# Tables are central so updating one signal's wording doesn't
# require touching the signal-emitter code.
# ─────────────────────────────────────────────────────────────────────

# Phase 4 — sample budget for the text-scanning signals. Cleaned files
# are read up to this many docs × this many bytes each; signals
# extrapolate. Keeps the data-health poll endpoint fast (~50ms total
# even for projects with thousands of docs).
PHASE4_SAMPLE_DOC_LIMIT = 20
PHASE4_SAMPLE_BYTES_PER_DOC = 4096

# Thresholds for the phase-4 signals — projecting from the sample.
HTML_PRESENT_WARN_FRAC = 0.05      # ≥ 5% of sampled docs have tags → warn
WHITESPACE_NOISE_WARN_FRAC = 0.10  # ≥ 10% of sampled docs need normalize
NEAR_DUP_PRESENT_WARN_FRAC = 0.10
LENGTH_OVER_CAP_WARN_FRAC = 0.10   # mirrors truncation signal threshold


_LAYMAN: dict[str, dict[str, str]] = {
    "ingestion.no_documents": {
        "plain": "You haven't uploaded any source documents yet.",
        "why": "Training needs source text to learn from — the platform has nothing to work with until you upload at least one document.",
    },
    "ingestion.parse_failure_rate": {
        "plain": "Some of your uploaded files couldn't be read. The platform tried to extract text from them and got back nothing — usually a scanned PDF (image-only, no text layer) or a corrupted file.",
        "why": "Failed documents are silently excluded from training. With too many failing, you're training on less data than you think you are — and the model may not see whole categories of input.",
    },
    "cleaning.not_run": {
        "plain": "You've uploaded documents but haven't run the cleaning step yet.",
        "why": "Raw text isn't ready for training — cleaning removes boilerplate, redacts PII, and chunks documents into trainable-sized pieces. Skip this step and the model will train on cookie banners and email signatures alongside your real content.",
    },
    "cleaning.pii_unredacted": {
        "plain": "The cleaning step found personal information (emails, phone numbers, SSNs, credit cards) in your documents but didn't redact it.",
        "why": "An SLM trained on unredacted PII can memorise and emit it at inference time — a real compliance + privacy risk. Re-run cleaning with the redact-PII toggle on.",
    },
    "cleaning.low_quality_docs": {
        "plain": "A meaningful chunk of your documents scored low on the quality check — usually mostly-boilerplate text, encoding garble, or very short content.",
        "why": "Low-quality docs are noise: they dilute the training signal and the model spends its capacity learning patterns from text that doesn't represent your domain. Either curate them out or accept that training quality will suffer.",
    },
    "cleaning.duplicate_chunks": {
        "plain": "A significant share of your cleaned text chunks are duplicates of each other.",
        "why": "The model will see the same content multiple times during training and overfit on it — strong on the duplicated patterns, weak on everything else. Dedupe before training.",
    },
    # Phase 4 cleaning-side signals.
    "cleaning.html_tags_present": {
        "plain": "Your cleaned documents still contain HTML tags (<p>, <br>, etc.).",
        "why": "HTML tags are tokens the model has to learn to ignore — they steal capacity from learning your actual content. Strip them before training; the cleaned text reads the same to a human afterward.",
    },
    "cleaning.whitespace_artifacts": {
        "plain": "Your cleaned documents have excess whitespace — runs of spaces, blank lines, or weird line endings.",
        "why": "Whitespace artefacts inflate the cleaned text and burn through the trainer's max_seq_length window faster than the actual content does. Normalising shrinks the input without dropping any real signal.",
    },
    "cleaning.near_duplicate_docs": {
        "plain": "Several documents share substantially the same opening — likely near-duplicates the exact-hash dedup missed (paraphrases, shared boilerplate, etc.).",
        "why": "Near-duplicates over-represent specific phrasings in training and the model memorises them. The aggressive-normalisation dedup catches paraphrases the exact-hash one can't.",
    },
    "cleaning.length_over_cap": {
        "plain": "Some cleaned documents are longer than the trainer's max_seq_length will accept — they'll be silently truncated at training time.",
        "why": "Silent truncation drops the tail of these documents. The model never sees the end (answer, closing tag, rationale). Truncating now makes the truncation explicit and visible at the file level rather than hidden at training time.",
    },
    "leakage.gold_train_overlap": {
        "plain": "Some of your gold-set rows also appear in your training data — identical or near-identical copies.",
        "why": "The gold set is the ruler that decides whether your model works. If the model already saw those rows during training, it can recite the answers from memory and the eval pass-rate is inflated — a green score that doesn't mean the model generalises. Hold the gold set out: remove the leaked rows from training, or re-split so the gold rows are never trained on. A GOLD_TEST leak is the worst kind — that split is your final grade.",
    },
    "leakage.no_overlap": {
        "plain": "Your gold-set rows are held out — none of them appear in the training data.",
        "why": "",
    },
    "leakage.split_overlap": {
        "plain": "Some rows are shared across your prepared train / validation / test splits — identical or near-identical copies.",
        "why": "The three splits must be disjoint to mean anything. A validation row that's also in train makes your validation metric optimistic, so early-stopping and checkpoint selection pick the wrong model. A test row that's also in train (or in validation) inflates the final grade. Re-split with deduplication so every row lands in exactly one split.",
    },
    "leakage.splits_held_out": {
        "plain": "Your train / validation / test splits are disjoint — no rows are shared across them.",
        "why": "",
    },
    "shape.gold_field_variants": {
        "plain": "Your gold rows use non-canonical field names (`class` instead of `label`, `text` instead of `input`).",
        "why": "The trainer + eval pipeline both expect canonical names; non-canonical rows are either silently skipped or mis-mapped, which collapses your effective gold-set size without telling you. Renaming is a safe one-shot fix.",
    },
    "shape.no_recipe_selected": {
        "plain": "You haven't picked a recipe yet (classification, span-extraction, summarization, qa-sft, etc.).",
        "why": "The recipe is what tells the platform what shape your training data should look like — without it, the platform can't tell you whether your data will work, what fields are missing, or how to score the trained model.",
    },
    "shape.no_prepared_dataset": {
        "plain": "Your data hasn't been split into training / validation / test sets yet.",
        "why": "The trainer needs three separate, non-overlapping sets: one to learn from, one to monitor learning during training, and one to grade the final model. The dataset-prep step builds these.",
    },
    "shape.corpus_too_small": {
        "plain": "The labelled corpus is below the recipe's recommended minimum row count.",
        "why": "Small models trained on tiny datasets either memorise (perfect on training, terrible on new inputs) or fail to learn the task at all. The recipe's minimum is where past projects of the same shape started getting reliable results.",
    },
    # Delegated from trainability_forecast_service — same headlines,
    # plain-English versions here.
    "class_imbalance": {
        "plain": "Your classes are very uneven — some classes have many examples, others have few or none.",
        "why": "The model will get great at the popular classes and bad at the rare ones. If you ship as-is, the rare-class predictions will be unreliable. Fix by adding more examples of the under-represented classes (or by accepting that you can't predict them well).",
    },
    "per_class_minimum_unmet": {
        "plain": "Some of your classes have very few examples — fewer than 5 per class.",
        "why": "The model can't learn a class with only a handful of examples — there's not enough signal. F1 score for those classes will collapse to roughly guessing.",
    },
    "label_vocab_fragmented": {
        "plain": "The same label appears with different spellings or capitalisations (e.g. 'positive' and 'Positive' counted as two separate classes).",
        "why": "The trainer treats each spelling as its own class — your effective per-class count is smaller than it looks, and the model won't generalise across the variants. A quick rename to the canonical label fixes it.",
    },
    "single_class_dominance": {
        "plain": "One class makes up most of your gold set — everything else is a small slice.",
        "why": "The model can hit high accuracy just by always predicting the dominant class. The minority classes won't get learned and your evaluation number will be misleading.",
    },
}


def _layman_for(signal_id: str) -> dict[str, str]:
    """Return the plain/why translation for a signal id, with a
    sensible fallback when no translation is registered yet."""
    if signal_id in _LAYMAN:
        return _LAYMAN[signal_id]
    return {
        "plain": "",
        "why": "",
    }


def _make_signal(
    *,
    id: str,
    severity: Severity,
    headline: str,
    suggested_action: dict | None = None,
    context: dict | None = None,
    plain_english: str | None = None,
    why_it_matters: str | None = None,
    autofix_kind: str | None = None,
) -> dict[str, Any]:
    """Build a signal payload. Pulls plain-English / why-it-matters
    from the translation table when not explicitly passed.

    ``autofix_kind`` (D3) flags signals the safe auto-fix engine can
    resolve in one click. The frontend renders an "Auto-fix" button
    when this is set, calling ``POST /data-health/autofix`` with the
    kind as the payload. ``None`` = the signal is informational only
    (no safe transform exists yet for it).
    """
    layman = _layman_for(id)
    return {
        "id": id,
        "severity": severity,
        "headline": headline,
        "plain_english": plain_english if plain_english is not None else layman["plain"],
        "why_it_matters": why_it_matters if why_it_matters is not None else layman["why"],
        "suggested_action": suggested_action,
        "context": context or {},
        "autofix_kind": autofix_kind,
    }


def _resolve_task_profile(project: Project) -> str | None:
    """Resolve the project's recipe to a ``task_profile`` string (e.g.
    ``"classification"``, ``"structured_extraction"``, ``"qa_sft"``).

    Returns ``None`` when no recipe is selected or the recipe can't
    be loaded. Used by ``_cleaning_group`` to make the PII signal
    recipe-aware — span-extraction projects need PII in source data
    to teach the model what to detect, so the auto-redact path must
    not fire for those.
    """
    selected = project.selected_recipe or {}
    if not isinstance(selected, dict):
        return None
    recipe_id = selected.get("recipe_id")
    if not isinstance(recipe_id, str) or not recipe_id:
        return None
    try:
        from app.services.recipe_service import get_recipe
        recipe = get_recipe(recipe_id)
        if recipe is None:
            return None
        return getattr(recipe, "task_profile", None)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Ingestion group.
# ─────────────────────────────────────────────────────────────────────


async def _ingestion_group(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """Document-count + parse-failure signals.

    Two ingest paths feed a project:

    1. Document ingestion — unstructured PDFs/DOCs/text files land as
       ``RawDocument`` rows under a RAW Dataset. This is the path for
       qa-sft, span-extraction, summarization corpora.
    2. Dataset import — labelled CSV/JSONL files land directly as rows
       in a SYNTHETIC/CLEANED/TRAIN Dataset via ``/dataset-import/run``
       (skipping the RawDocument table entirely). This is the standard
       path for classification corpora, where each row is already a
       labelled training example, not a document to be cleaned + split.

    The ingestion signal is satisfied when *either* path has produced
    rows. Without this check, classification projects (which are
    expected to import labelled JSONL) get a permanent "No documents"
    block even after a successful import.
    """
    result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.RAW,
        )
    )
    docs = list(result.scalars())
    total = len(docs)
    signals: list[dict[str, Any]] = []

    if total == 0:
        # Before declaring "no documents", check whether the project
        # has any labelled-row corpus via dataset-import. Counting
        # CLEANED/SYNTHETIC/TRAIN here matches the trainability
        # forecast's "labeled corpus" definition — both signals
        # should agree on whether the project has data.
        labelled_count_result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type.in_([
                    DatasetType.CLEANED,
                    DatasetType.SYNTHETIC,
                    DatasetType.TRAIN,
                ]),
            )
        )
        labelled_rows = sum(
            int(ds.record_count or 0) for ds in labelled_count_result.scalars()
        )
        if labelled_rows == 0:
            signals.append(_make_signal(
                id="ingestion.no_documents",
                severity="block",
                headline="No documents or labelled rows ingested yet.",
                suggested_action={"kind": "navigate", "label": "Open Ingest tab", "target": "data"},
                context={"document_count": 0, "labelled_row_count": 0},
            ))
        else:
            signals.append(_make_signal(
                id="ingestion.no_documents",
                severity="ok",
                headline=(
                    f"{labelled_rows} labelled rows imported via dataset-import."
                ),
                context={
                    "document_count": 0,
                    "labelled_row_count": labelled_rows,
                    "ingest_path": "dataset_import",
                },
            ))
        return {
            "id": "ingestion",
            "title": "Ingestion",
            "subtitle": "Documents uploaded + parsed",
            "signals": signals,
        }

    errored = sum(1 for d in docs if d.status == DocumentStatus.ERROR)
    pending = sum(1 for d in docs if d.status == DocumentStatus.PENDING)
    accepted = sum(1 for d in docs if d.status == DocumentStatus.ACCEPTED)
    fail_rate = errored / total if total > 0 else 0.0

    if errored == 0:
        signals.append(_make_signal(
            id="ingestion.parse_failure_rate",
            severity="ok",
            headline=f"All {total} documents parsed cleanly.",
            context={"document_count": total, "errored": 0, "pending": pending, "accepted": accepted},
        ))
    else:
        severity: Severity = (
            "block" if fail_rate >= PARSE_FAILURE_BLOCK
            else "warn" if fail_rate >= PARSE_FAILURE_WARN
            else "warn"
        )
        signals.append(_make_signal(
            id="ingestion.parse_failure_rate",
            severity=severity,
            headline=(
                f"{errored} of {total} documents failed parsing ({fail_rate * 100:.0f}%)."
            ),
            suggested_action={
                "kind": "navigate",
                "label": "View failed docs",
                "target": "ingest-error-list",
            },
            context={
                "document_count": total,
                "errored": errored,
                "pending": pending,
                "accepted": accepted,
                "failure_rate": round(fail_rate, 4),
            },
            # D3 safe auto-fix — failed docs have no extracted text,
            # dropping them is non-destructive (they're already useless).
            autofix_kind="drop_failed_docs",
        ))

    return {
        "id": "ingestion",
        "title": "Ingestion",
        "subtitle": "Documents uploaded + parsed",
        "signals": signals,
    }


# ─────────────────────────────────────────────────────────────────────
# Cleaning group.
# ─────────────────────────────────────────────────────────────────────


async def _cleaning_group(
    db: AsyncSession,
    project_id: int,
    *,
    task_profile: str | None = None,
) -> dict[str, Any]:
    """PII, low-quality docs, duplicate chunks.

    ``task_profile`` (when known) lets the PII signal be recipe-aware.
    For ``structured_extraction`` recipes the user is almost certainly
    training a span-extraction model (PII detection, NER, entity
    extraction) where the source-document PII is the **training
    signal** — auto-redacting it would destroy the very pattern the
    model needs to learn. In that case we flip the signal to ``ok``
    and refuse the auto-fix.
    """
    result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.RAW,
        )
    )
    docs = list(result.scalars())
    total = len(docs)
    signals: list[dict[str, Any]] = []

    if total == 0:
        # Ingestion group already complained about this.
        return {
            "id": "cleaning",
            "title": "Cleaning",
            "subtitle": "PII redaction + quality + dedup",
            "signals": [],
        }

    # A doc has been "cleaned" iff its metadata carries a cleaned_path.
    # (Equivalent to: clean_document was called for it.)
    def _is_cleaned(d: RawDocument) -> bool:
        return bool((d.metadata_ or {}).get("cleaned_path"))

    cleaned_docs = [d for d in docs if _is_cleaned(d)]
    if not cleaned_docs:
        signals.append(_make_signal(
            id="cleaning.not_run",
            severity="warn",
            headline=f"No documents have been cleaned yet — {total} uploaded, 0 cleaned.",
            suggested_action={"kind": "navigate", "label": "Open Cleaning tab", "target": "cleaning"},
            context={"uploaded": total, "cleaned": 0},
        ))
        return {
            "id": "cleaning",
            "title": "Cleaning",
            "subtitle": "PII redaction + quality + dedup",
            "signals": signals,
        }

    # PII check: count findings across cleaned docs. The cleaning
    # service stores findings under metadata_.pii_findings (the
    # pre-redaction set). When redact=True ran, pii_findings stays
    # populated (an audit trail) — we look for an explicit
    # "pii_redacted" flag instead.
    pii_total = 0
    pii_findings_present = False
    for d in cleaned_docs:
        findings = (d.metadata_ or {}).get("pii_findings") or []
        if isinstance(findings, list):
            pii_total += len(findings)
            if findings:
                pii_findings_present = True
    # A doc that ran cleaning with redact=False keeps PII in the text;
    # a doc with redact=True has redact_pii=True in metadata.
    any_redacted = any(
        bool((d.metadata_ or {}).get("redact_pii")) for d in cleaned_docs
    )
    if pii_findings_present and not any_redacted:
        # Recipe-aware: span-extraction projects (PII detection, NER,
        # entity extraction, etc.) need the PII in source documents to
        # teach the model what to detect. Auto-redacting would destroy
        # the training signal. Flip to ok severity and remove the
        # auto-fix so the panel doesn't silently nuke the user's
        # training data.
        is_span_extraction = task_profile == "structured_extraction"
        if is_span_extraction:
            signals.append(_make_signal(
                id="cleaning.pii_unredacted",
                severity="ok",
                headline=(
                    f"{pii_total} PII finding(s) detected — kept in source "
                    f"(required for span-extraction training)."
                ),
                plain_english=(
                    "Your recipe is span-extraction (PII detection, NER, entity "
                    "extraction, etc.) — the model learns by seeing PII in the "
                    "source documents and the gold-set spans pointing at it. "
                    "Auto-redaction is intentionally disabled for this project "
                    "shape so the training signal isn't destroyed. If you need "
                    "redaction for a separate non-training use, do it manually "
                    "on a copy of the cleaned outputs."
                ),
                why_it_matters="",
                suggested_action=None,
                context={
                    "pii_findings": pii_total,
                    "cleaned_docs": len(cleaned_docs),
                    "task_profile": task_profile,
                    "autofix_blocked_reason": "span_extraction_needs_pii",
                },
                # Deliberately no autofix_kind — the panel will not
                # render the Drop/Redact button for this row.
            ))
        else:
            severity: Severity = (
                "block" if pii_total >= PII_BLOCK_COUNT else "warn"
            )
            signals.append(_make_signal(
                id="cleaning.pii_unredacted",
                severity=severity,
                headline=(
                    f"{pii_total} PII findings detected across {len(cleaned_docs)} cleaned doc(s) "
                    f"— redaction was not applied."
                ),
                suggested_action={
                    "kind": "navigate",
                    "label": "Re-clean with redact-PII on",
                    "target": "cleaning",
                },
                context={"pii_findings": pii_total, "cleaned_docs": len(cleaned_docs)},
                # D3 safe auto-fix — re-runs clean_document with
                # redact=True for every doc with PII findings +
                # redact_pii=False. The cleaning service is idempotent,
                # so this is a pure re-render of the cleaned text with
                # PII masked.
                autofix_kind="redact_pii",
            ))
    elif pii_findings_present and any_redacted:
        signals.append(_make_signal(
            id="cleaning.pii_unredacted",
            severity="ok",
            headline=f"PII detected and redacted ({pii_total} finding(s) redacted).",
            plain_english="The cleaning step found personal information and replaced it with [REDACTED] before the text reaches training.",
            why_it_matters="",
            context={"pii_findings": pii_total, "redacted": True},
        ))

    # Low-quality fraction.
    scored = [d for d in cleaned_docs if d.quality_score is not None]
    if scored:
        low_q = [d for d in scored if (d.quality_score or 0.0) < LOW_QUALITY_THRESHOLD]
        low_q_frac = len(low_q) / len(scored)
        if low_q_frac >= LOW_QUALITY_BLOCK_FRAC:
            severity = "block"
        elif low_q_frac >= LOW_QUALITY_WARN_FRAC:
            severity = "warn"
        else:
            severity = "ok"
        signals.append(_make_signal(
            id="cleaning.low_quality_docs",
            severity=severity,
            headline=(
                f"{len(low_q)} of {len(scored)} cleaned docs scored below quality threshold "
                f"({low_q_frac * 100:.0f}%)."
            ),
            suggested_action=(
                {"kind": "navigate", "label": "Review low-quality docs", "target": "cleaning"}
                if severity != "ok"
                else None
            ),
            context={
                "low_quality_count": len(low_q),
                "scored": len(scored),
                "fraction": round(low_q_frac, 4),
                "threshold": LOW_QUALITY_THRESHOLD,
            },
        ))

    # Duplicate chunks via text_hash.
    hashes: dict[str, int] = {}
    chunk_total = 0
    for d in cleaned_docs:
        h = (d.metadata_ or {}).get("text_hash")
        chunk_total += int(d.chunk_count or 0)
        if isinstance(h, str) and h:
            hashes[h] = hashes.get(h, 0) + 1
    dup_groups = [(h, c) for h, c in hashes.items() if c > 1]
    dup_total = sum(c - 1 for _, c in dup_groups)  # extras beyond the first occurrence
    dup_frac = dup_total / len(cleaned_docs) if cleaned_docs else 0.0
    if dup_total == 0:
        signals.append(_make_signal(
            id="cleaning.duplicate_chunks",
            severity="ok",
            headline="No duplicate documents detected (by content hash).",
            context={"duplicate_count": 0, "total_cleaned": len(cleaned_docs)},
        ))
    else:
        if dup_frac >= DUPLICATE_CHUNK_BLOCK_FRAC:
            severity = "block"
        elif dup_frac >= DUPLICATE_CHUNK_WARN_FRAC:
            severity = "warn"
        else:
            severity = "warn"
        signals.append(_make_signal(
            id="cleaning.duplicate_chunks",
            severity=severity,
            headline=(
                f"{dup_total} duplicate document(s) detected "
                f"({dup_frac * 100:.0f}% of cleaned set)."
            ),
            suggested_action={
                "kind": "navigate",
                "label": "Review duplicates",
                "target": "cleaning",
            },
            context={
                "duplicate_count": dup_total,
                "total_cleaned": len(cleaned_docs),
                "fraction": round(dup_frac, 4),
            },
            # D3 safe auto-fix — keeps the lowest-id occurrence of each
            # text_hash and drops the rest. Pure dedup, no data loss
            # beyond redundancy.
            autofix_kind="dedupe_duplicate_docs",
        ))

    # Phase 4 — text-scanning signals over the cleaned docs.
    phase4 = _phase4_cleaned_text_signals(cleaned_docs)
    signals.extend(phase4)

    return {
        "id": "cleaning",
        "title": "Cleaning",
        "subtitle": "PII redaction + quality + dedup",
        "signals": signals,
    }


def _phase4_cleaned_text_signals(
    cleaned_docs: list[RawDocument],
) -> list[dict[str, Any]]:
    """Phase 4 — sample the cleaned-text files and emit per-condition
    signals carrying the matching autofix_kind.

    Reads at most ``PHASE4_SAMPLE_DOC_LIMIT`` docs and
    ``PHASE4_SAMPLE_BYTES_PER_DOC`` bytes each so the data-health poll
    endpoint stays fast. The autofix preview, when clicked, scans the
    full set.
    """
    import re as _re
    from pathlib import Path

    signals: list[dict[str, Any]] = []
    sample = [
        d for d in cleaned_docs[:PHASE4_SAMPLE_DOC_LIMIT]
        if (d.metadata_ or {}).get("cleaned_path")
    ]
    if not sample:
        return signals

    tag_re = _re.compile(r"<[^>]+>")
    excess_whitespace_re = _re.compile(r"  +|\n{3,}|[ \t]+\n")
    html_hits = 0
    whitespace_hits = 0
    near_dup_prefixes: dict[str, int] = {}
    length_over_cap_hits = 0
    sampled = 0
    cap_chars = 8192  # default 2048 tokens × 4 — refined below if recipe is set
    for doc in sample:
        cleaned_path = (doc.metadata_ or {}).get("cleaned_path")
        if not isinstance(cleaned_path, str):
            continue
        path = Path(cleaned_path)
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read(PHASE4_SAMPLE_BYTES_PER_DOC)
        except (OSError, UnicodeDecodeError):
            continue
        sampled += 1
        if tag_re.search(text):
            html_hits += 1
        if excess_whitespace_re.search(text):
            whitespace_hits += 1
        # Length-over-cap uses the full file size (file might be larger
        # than what we read — stat is cheap).
        try:
            full_size = path.stat().st_size
        except OSError:
            full_size = len(text)
        if full_size > cap_chars:
            length_over_cap_hits += 1
        # Near-dup prefix bucketing: lowercase + alphanumeric + first
        # 200 chars (smaller window than the autofix uses since we're
        # only flagging, not deciding).
        normalised = _re.sub(r"[^\w\s]+", " ", text.lower())
        normalised = _re.sub(r"\s+", " ", normalised).strip()
        prefix = normalised[:200]
        if prefix:
            near_dup_prefixes[prefix] = near_dup_prefixes.get(prefix, 0) + 1

    if sampled == 0:
        return signals

    # HTML signal.
    if html_hits / sampled >= HTML_PRESENT_WARN_FRAC:
        signals.append(_make_signal(
            id="cleaning.html_tags_present",
            severity="warn",
            headline=(
                f"{html_hits} of {sampled} sampled cleaned documents still "
                f"contain HTML tags."
            ),
            suggested_action={
                "kind": "navigate",
                "label": "Review HTML hits",
                "target": "cleaning",
            },
            context={
                "sampled": sampled,
                "html_hits": html_hits,
                "sample_doc_limit": PHASE4_SAMPLE_DOC_LIMIT,
            },
            autofix_kind="strip_html",
        ))

    # Whitespace signal.
    if whitespace_hits / sampled >= WHITESPACE_NOISE_WARN_FRAC:
        signals.append(_make_signal(
            id="cleaning.whitespace_artifacts",
            severity="warn",
            headline=(
                f"{whitespace_hits} of {sampled} sampled cleaned documents "
                f"have excess whitespace runs or blank-line noise."
            ),
            suggested_action={
                "kind": "navigate",
                "label": "Review whitespace hits",
                "target": "cleaning",
            },
            context={
                "sampled": sampled,
                "whitespace_hits": whitespace_hits,
            },
            autofix_kind="normalize_whitespace",
        ))

    # Near-duplicate signal — any bucket with > 1 hit is a candidate.
    near_dup_groups = sum(1 for v in near_dup_prefixes.values() if v > 1)
    near_dup_extra = sum(v - 1 for v in near_dup_prefixes.values() if v > 1)
    if near_dup_extra / max(1, sampled) >= NEAR_DUP_PRESENT_WARN_FRAC:
        signals.append(_make_signal(
            id="cleaning.near_duplicate_docs",
            severity="warn",
            headline=(
                f"{near_dup_extra} near-duplicate document(s) in sample "
                f"across {near_dup_groups} group(s) — likely more across "
                f"the full corpus."
            ),
            suggested_action={
                "kind": "navigate",
                "label": "Preview near-dup groups",
                "target": "cleaning",
            },
            context={
                "sampled": sampled,
                "near_dup_extra": near_dup_extra,
                "near_dup_groups": near_dup_groups,
            },
            autofix_kind="near_duplicate_dedup",
        ))

    # Length-over-cap signal.
    if length_over_cap_hits / sampled >= LENGTH_OVER_CAP_WARN_FRAC:
        signals.append(_make_signal(
            id="cleaning.length_over_cap",
            severity="warn",
            headline=(
                f"{length_over_cap_hits} of {sampled} sampled cleaned "
                f"documents exceed the project's effective "
                f"max_seq_length ({cap_chars} chars)."
            ),
            suggested_action={
                "kind": "navigate",
                "label": "Review oversize docs",
                "target": "cleaning",
            },
            context={
                "sampled": sampled,
                "length_over_cap_hits": length_over_cap_hits,
                "cap_chars": cap_chars,
            },
            autofix_kind="length_cap",
        ))

    return signals


# ─────────────────────────────────────────────────────────────────────
# Shape group — does the data match the recipe?
# ─────────────────────────────────────────────────────────────────────


async def _shape_group(db: AsyncSession, project: Project) -> dict[str, Any]:
    signals: list[dict[str, Any]] = []
    selected = project.selected_recipe or {}
    recipe_id = selected.get("recipe_id") if isinstance(selected, dict) else None

    if not recipe_id:
        signals.append(_make_signal(
            id="shape.no_recipe_selected",
            severity="block",
            headline="No recipe selected for this project.",
            suggested_action={
                "kind": "navigate",
                "label": "Open recipe picker",
                "target": "recipe-picker",
            },
            context={},
        ))
        return {
            "id": "shape",
            "title": "Data shape vs recipe",
            "subtitle": "Does the data fit the recipe?",
            "signals": signals,
        }

    # We have a recipe; check whether train/val/test exist + size.
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project.id,
            Dataset.dataset_type.in_(
                [DatasetType.TRAIN, DatasetType.VALIDATION, DatasetType.TEST]
            ),
        )
    )
    splits = {ds.dataset_type.value: int(ds.record_count or 0) for ds in result.scalars()}
    train = splits.get("train", 0)
    val = splits.get("validation", 0)
    test = splits.get("test", 0)
    have_any_split = (train + val + test) > 0

    # Pull recipe min from the catalog when available.
    minimum_rows = 0
    try:
        from app.services.recipe_service import get_recipe
        recipe = get_recipe(recipe_id)
        if recipe is not None:
            minimum_rows = int(
                getattr(getattr(recipe, "gold_template", None), "min_rows_recommended", 0) or 0
            )
    except Exception:
        recipe = None

    if not have_any_split:
        signals.append(_make_signal(
            id="shape.no_prepared_dataset",
            severity="warn",
            headline="No prepared train/val/test splits yet.",
            suggested_action={
                "kind": "navigate",
                "label": "Open Dataset prep",
                "target": "dataprep",
            },
            context={"recipe_id": recipe_id},
        ))
    else:
        # Corpus-size check.
        if minimum_rows <= 0:
            signals.append(_make_signal(
                id="shape.corpus_too_small",
                severity="ok",
                headline=(
                    f"Train/val/test prepared: {train} / {val} / {test} rows "
                    f"(no recipe-level minimum to compare against)."
                ),
                plain_english="The recipe doesn't declare a minimum row count, so the platform can't gate on size — but the splits exist.",
                why_it_matters="",
                context={"train": train, "val": val, "test": test, "recipe_id": recipe_id},
            ))
        elif train < minimum_rows:
            severity: Severity = "block" if train == 0 else "warn"
            signals.append(_make_signal(
                id="shape.corpus_too_small",
                severity=severity,
                headline=(
                    f"Only {train} training rows — the {recipe_id} recipe recommends "
                    f"at least {minimum_rows}."
                ),
                suggested_action={
                    "kind": "navigate",
                    "label": "Run synth augmentation",
                    "target": "synthetic",
                },
                context={
                    "train": train,
                    "minimum_rows": minimum_rows,
                    "recipe_id": recipe_id,
                    "val": val,
                    "test": test,
                },
            ))
        else:
            signals.append(_make_signal(
                id="shape.corpus_too_small",
                severity="ok",
                headline=(
                    f"{train} training rows — above the {recipe_id} recipe minimum "
                    f"of {minimum_rows}."
                ),
                plain_english="",
                why_it_matters="",
                context={
                    "train": train,
                    "minimum_rows": minimum_rows,
                    "recipe_id": recipe_id,
                    "val": val,
                    "test": test,
                },
            ))

    # Phase 4 — gold-row field-variant detection. Sample-reads
    # GOLD_DEV/GOLD_TEST files looking for non-canonical field names;
    # emits a single rolled-up signal carrying autofix_kind=normalize_schema.
    gold_schema_signal = await _phase4_gold_schema_signal(db, project.id)
    if gold_schema_signal is not None:
        signals.append(gold_schema_signal)

    return {
        "id": "shape",
        "title": "Data shape vs recipe",
        "subtitle": "Does the data fit the recipe?",
        "signals": signals,
    }


async def _phase4_gold_schema_signal(
    db: AsyncSession, project_id: int,
) -> dict[str, Any] | None:
    """Sample-read up to ``PHASE4_SAMPLE_DOC_LIMIT`` rows per gold file
    and check for non-canonical field names. Returns None when nothing
    needs renaming so the signal list stays uncluttered.
    """
    import json
    from pathlib import Path
    from app.services.data_health_autofix_service import GOLD_FIELD_RENAMES

    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                [DatasetType.GOLD_DEV, DatasetType.GOLD_TEST]
            ),
        )
    )
    datasets = list(result.scalars())
    rename_counts: dict[str, int] = {}
    sampled_rows = 0
    for ds in datasets:
        if not ds.file_path:
            continue
        path = Path(ds.file_path)
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    sampled_rows += 1
                    for old_key, new_key in GOLD_FIELD_RENAMES.items():
                        if old_key in row and new_key not in row:
                            rename_counts[old_key] = (
                                rename_counts.get(old_key, 0) + 1
                            )
                    if sampled_rows >= PHASE4_SAMPLE_DOC_LIMIT * 5:
                        break  # cap the read so the poll stays fast
        except OSError:
            continue
    total_renames = sum(rename_counts.values())
    if total_renames == 0:
        return None

    return _make_signal(
        id="shape.gold_field_variants",
        severity="warn",
        headline=(
            f"{total_renames} gold row(s) in sample use non-canonical "
            f"field names (e.g. {', '.join(rename_counts.keys())})."
        ),
        suggested_action={
            "kind": "navigate",
            "label": "Review schema renames",
            "target": "dataprep",
        },
        context={
            "sampled_rows": sampled_rows,
            "rename_counts": rename_counts,
            "rename_map": GOLD_FIELD_RENAMES,
        },
        autofix_kind="normalize_schema",
    )


# ─────────────────────────────────────────────────────────────────────
# Balance group — classification only, delegated to the existing
# forecast helpers so thresholds stay aligned with Coach Mode.
# ─────────────────────────────────────────────────────────────────────


async def _balance_group(db: AsyncSession, project: Project) -> dict[str, Any]:
    """For classification recipes, surface class-balance / per-class-min
    signals from the trainability forecast — but reshaped with the
    layman translation layer."""
    selected = project.selected_recipe or {}
    recipe_id = selected.get("recipe_id") if isinstance(selected, dict) else None
    if not recipe_id:
        return {
            "id": "balance",
            "title": "Class balance",
            "subtitle": "Classification gold-set distribution",
            "signals": [],
        }

    # Only run for recipes with a classification task_profile.
    try:
        from app.services.recipe_service import get_recipe
        recipe = get_recipe(recipe_id)
        task_profile = getattr(recipe, "task_profile", None)
    except Exception:
        task_profile = None

    if task_profile != "classification":
        return {
            "id": "balance",
            "title": "Class balance",
            "subtitle": "Classification gold-set distribution",
            "signals": [],
        }

    # Reuse forecast helpers so thresholds stay aligned.
    from app.services.trainability_forecast_service import (
        _build_classification_signals,
        _load_gold_rows,
    )

    gold_rows = await _load_gold_rows(db, project.id)
    forecast_signals, _entropy = _build_classification_signals(gold_rows)

    signals: list[dict[str, Any]] = []
    for fs in forecast_signals:
        sid = fs.get("id", "")
        signals.append(_make_signal(
            id=sid,
            severity=fs.get("severity", "warn"),
            headline=fs.get("headline", ""),
            suggested_action=fs.get("suggested_action"),
            context={"detail": fs.get("detail", "")},
        ))

    if not signals:
        signals.append(_make_signal(
            id="balance.healthy",
            severity="ok",
            headline="Class balance looks healthy — no imbalance or per-class signals firing.",
            plain_english="Your gold set's classes are spread evenly enough that the trainer should learn each one without lopsidedness.",
            why_it_matters="",
            context={"gold_rows": len(gold_rows)},
        ))

    return {
        "id": "balance",
        "title": "Class balance",
        "subtitle": "Classification gold-set distribution",
        "signals": signals,
    }


# ─────────────────────────────────────────────────────────────────────
# Leakage group — train ↔ gold contamination.
# ─────────────────────────────────────────────────────────────────────


def _normalise_for_exact(text: str) -> str:
    """Lowercase + collapse whitespace — the key for exact-duplicate
    detection that ignores trivial formatting differences."""
    return " ".join((text or "").lower().split())


async def _load_training_corpus_rows(
    db: AsyncSession, project_id: int, *, cap: int
) -> tuple[list[dict[str, Any]], bool]:
    """Load the rows the trainer will actually see: TRAIN ∪ CLEANED ∪
    SYNTHETIC. Pending-review synthetic rows are excluded by the loader
    (they're gated out of training until reviewed), so they can't
    produce a phantom leak. Returns ``(rows, truncated)`` where
    ``truncated`` is True when ``cap`` was hit.
    """
    from pathlib import Path
    from app.services.dataset_service import _load_records_from_file

    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(
                [DatasetType.TRAIN, DatasetType.CLEANED, DatasetType.SYNTHETIC]
            ),
        )
    )
    rows: list[dict[str, Any]] = []
    truncated = False
    for ds in result.scalars():
        if not ds.file_path:
            continue
        path = Path(ds.file_path)
        if not path.exists():
            continue
        for row in _load_records_from_file(path):
            if isinstance(row, dict):
                rows.append(row)
            if len(rows) >= cap:
                truncated = True
                break
        if truncated:
            break
    return rows, truncated


async def _load_gold_rows_by_split(
    db: AsyncSession, project_id: int
) -> dict[str, list[dict[str, Any]]]:
    """Load GOLD_DEV + GOLD_TEST rows keyed by split so the leakage
    report can call out test contamination (the worst kind) separately
    from dev contamination."""
    from pathlib import Path
    from app.services.dataset_service import _load_records_from_file

    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_([DatasetType.GOLD_DEV, DatasetType.GOLD_TEST]),
        )
    )
    out: dict[str, list[dict[str, Any]]] = {"gold_dev": [], "gold_test": []}
    for ds in result.scalars():
        if not ds.file_path:
            continue
        path = Path(ds.file_path)
        if not path.exists():
            continue
        split = ds.dataset_type.value
        out.setdefault(split, []).extend(
            r for r in _load_records_from_file(path) if isinstance(r, dict)
        )
    return out


def _excerpt(text: str) -> str:
    """Whitespace-collapsed, length-capped row text for drill-down."""
    return " ".join((text or "").split())[:200]


def _build_leakage_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the haystack: exact-text map + inverted token index +
    per-row token sets + texts (for matched-row excerpts). Shared by
    the gold scan and the prepared-split scan."""
    from app.services.trainability_forecast_service import _row_to_text, _tokenize

    token_sets: list[frozenset[str]] = []
    texts: list[str] = []
    exact_map: dict[str, int] = {}
    inverted: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        text = _row_to_text(row)
        texts.append(text)
        norm = _normalise_for_exact(text)
        if norm not in exact_map:
            exact_map[norm] = idx
        toks = _tokenize(text)
        token_sets.append(toks)
        for tok in toks:
            postings = inverted.setdefault(tok, [])
            if len(postings) < LEAKAGE_MAX_POSTINGS:
                postings.append(idx)
    return {
        "token_sets": token_sets,
        "texts": texts,
        "exact_map": exact_map,
        "inverted": inverted,
    }


def _match_row_against_index(
    text: str, index: dict[str, Any]
) -> tuple[str | None, float, int]:
    """Match one row's text against a haystack index. Returns
    ``(match_kind, jaccard, matched_idx)`` — exact-normalised equality
    first, then fuzzy token-set Jaccard against candidates gathered
    from the rarest tokens. ``(None, 0.0, -1)`` when no match."""
    from app.services.trainability_forecast_service import _jaccard, _tokenize

    norm = _normalise_for_exact(text)
    hit = index["exact_map"].get(norm)
    if hit is not None:
        return "exact", 1.0, hit

    toks = _tokenize(text)
    if len(toks) < LEAKAGE_MIN_TOKENS:
        return None, 0.0, -1

    inverted = index["inverted"]
    token_sets = index["token_sets"]
    toks_by_rarity = sorted(toks, key=lambda t: len(inverted.get(t, ())))
    cand: set[int] = set()
    for tok in toks_by_rarity:
        for j in inverted.get(tok, ()):
            cand.add(j)
            if len(cand) >= LEAKAGE_MAX_CANDIDATES:
                break
        if len(cand) >= LEAKAGE_MAX_CANDIDATES:
            break

    best = 0.0
    best_j = -1
    for j in cand:
        s = _jaccard(toks, token_sets[j])
        if s > best:
            best = s
            best_j = j
            if best >= 0.999:
                break
    if best >= LEAKAGE_FUZZY_THRESHOLD:
        return "near_duplicate", round(best, 3), best_j
    return None, 0.0, -1


def _severity_for_frac(total_leaked: int, frac: float) -> "Severity":
    if total_leaked == 0:
        return "ok"
    if frac >= LEAKAGE_BLOCK_FRAC:
        return "block"
    return "warn"


def _scan_leakage_rows(
    train_rows: list[dict[str, Any]],
    gold_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Gold ↔ training-corpus leakage. Indexes the training rows once,
    then scans each gold split against it (sharing the gold scan cap).

    Returns ``total_scanned`` / ``total_leaked`` / ``frac`` /
    ``per_split`` / ``examples`` / ``test_leaked`` / ``severity``.
    """
    from app.services.trainability_forecast_service import _row_to_text

    index = _build_leakage_index(train_rows)
    examples: list[dict[str, Any]] = []
    per_split: dict[str, dict[str, int]] = {}
    total_scanned = 0
    total_leaked = 0

    for split, rows in gold_by_split.items():
        scanned = 0
        leaked = 0
        for row in rows:
            if total_scanned >= LEAKAGE_GOLD_SCAN_CAP:
                break
            text = _row_to_text(row)
            if not text.strip():
                continue
            scanned += 1
            total_scanned += 1
            kind, jacc, midx = _match_row_against_index(text, index)
            if kind:
                leaked += 1
                total_leaked += 1
                if len(examples) < LEAKAGE_EXAMPLE_LIMIT:
                    examples.append({
                        "source": "train",
                        "split": split,
                        "match_kind": kind,
                        "jaccard": jacc,
                        "excerpt": _excerpt(text),
                        "matched_excerpt": (
                            _excerpt(index["texts"][midx]) if midx >= 0 else ""
                        ),
                    })
        per_split[split] = {"scanned": scanned, "leaked": leaked}

    frac = (total_leaked / total_scanned) if total_scanned else 0.0
    return {
        "total_scanned": total_scanned,
        "total_leaked": total_leaked,
        "frac": round(frac, 4),
        "test_leaked": per_split.get("gold_test", {}).get("leaked", 0),
        "per_split": per_split,
        "examples": examples,
        "severity": _severity_for_frac(total_leaked, frac),
    }


# Prepared-split leakage pairs: each held-out split must be disjoint
# from the splits the model already saw. (source, needle): a ``needle``
# row found in ``source`` is contamination. We do NOT compare against
# CLEANED/SYNTHETIC here — the splits are *derived* from those pools, so
# a match there is the split's origin, not a leak.
_PREPARED_SPLIT_PAIRS: tuple[tuple[str, str], ...] = (
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
)


def _scan_prepared_split_pairs(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Train↔val / train↔test / val↔test leakage over the prepared
    splits. Returns per-pair leaked counts + examples + severity."""
    from app.services.trainability_forecast_service import _row_to_text

    indices: dict[str, dict[str, Any]] = {}
    per_pair: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    total_scanned = 0
    total_leaked = 0
    worst_frac = 0.0

    for source_name, needle_name in _PREPARED_SPLIT_PAIRS:
        source_rows = splits.get(source_name) or []
        needle_rows = splits.get(needle_name) or []
        if not source_rows or not needle_rows:
            continue
        if source_name not in indices:
            indices[source_name] = _build_leakage_index(
                source_rows[:LEAKAGE_TRAIN_SCAN_CAP]
            )
        index = indices[source_name]
        scanned = 0
        leaked = 0
        for row in needle_rows[:LEAKAGE_GOLD_SCAN_CAP]:
            text = _row_to_text(row)
            if not text.strip():
                continue
            scanned += 1
            total_scanned += 1
            kind, jacc, midx = _match_row_against_index(text, index)
            if kind:
                leaked += 1
                total_leaked += 1
                if len(examples) < LEAKAGE_EXAMPLE_LIMIT:
                    examples.append({
                        "source": source_name,
                        "split": needle_name,
                        "match_kind": kind,
                        "jaccard": jacc,
                        "excerpt": _excerpt(text),
                        "matched_excerpt": (
                            _excerpt(index["texts"][midx]) if midx >= 0 else ""
                        ),
                    })
        pair_frac = (leaked / scanned) if scanned else 0.0
        worst_frac = max(worst_frac, pair_frac)
        per_pair[f"{needle_name}_in_{source_name}"] = {
            "scanned": scanned,
            "leaked": leaked,
            "frac": round(pair_frac, 4),
        }

    return {
        "total_scanned": total_scanned,
        "total_leaked": total_leaked,
        "worst_frac": round(worst_frac, 4),
        "per_pair": per_pair,
        "examples": examples,
        "severity": _severity_for_frac(total_leaked, worst_frac),
    }


async def scan_train_gold_leakage(
    db: AsyncSession, project_id: int
) -> dict[str, Any] | None:
    """Public leakage analysis used by both the data-health group and
    Coach Mode. Returns ``None`` when the check isn't applicable yet
    (no gold set, no training rows, or no gold row had scannable text) —
    the absence of those is flagged by the shape/ingestion groups, so
    leakage stays silent rather than emitting a misleading "all clear."
    """
    gold_by_split = await _load_gold_rows_by_split(db, project_id)
    if sum(len(v) for v in gold_by_split.values()) == 0:
        return None
    train_rows, truncated = await _load_training_corpus_rows(
        db, project_id, cap=LEAKAGE_TRAIN_SCAN_CAP
    )
    if not train_rows:
        return None
    result = _scan_leakage_rows(train_rows, gold_by_split)
    if result["total_scanned"] == 0:
        return None
    result["train_rows_scanned"] = len(train_rows)
    result["train_truncated"] = truncated
    return result


async def _load_prepared_splits(
    db: AsyncSession, project_id: int
) -> dict[str, list[dict[str, Any]]]:
    """Load the prepared TRAIN / VALIDATION / TEST split rows keyed by
    short split name (``train`` / ``val`` / ``test``)."""
    from pathlib import Path
    from app.services.dataset_service import _load_records_from_file

    type_to_key = {
        DatasetType.TRAIN: "train",
        DatasetType.VALIDATION: "val",
        DatasetType.TEST: "test",
    }
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type.in_(list(type_to_key.keys())),
        )
    )
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for ds in result.scalars():
        if not ds.file_path:
            continue
        path = Path(ds.file_path)
        if not path.exists():
            continue
        key = type_to_key[ds.dataset_type]
        out[key].extend(
            r for r in _load_records_from_file(path) if isinstance(r, dict)
        )
    return out


async def scan_prepared_split_leakage(
    db: AsyncSession, project_id: int
) -> dict[str, Any] | None:
    """Public train↔val / train↔test / val↔test leakage analysis.
    Returns ``None`` until at least two of the three prepared splits
    exist (no pair to compare) or nothing scannable was found."""
    splits = await _load_prepared_splits(db, project_id)
    present = [k for k, v in splits.items() if v]
    if len(present) < 2:
        return None
    result = _scan_prepared_split_pairs(splits)
    if result["total_scanned"] == 0:
        return None
    return result


async def _leakage_group(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """Leakage group: gold↔training-corpus + prepared-split (train/val/
    test) contamination. Each signal self-hides until its inputs exist."""
    signals: list[dict[str, Any]] = []

    # ── Gold ↔ training corpus ─────────────────────────────────
    scan = await scan_train_gold_leakage(db, project_id)
    if scan is not None:
        scanned = scan["total_scanned"]
        leaked = scan["total_leaked"]
        test_leaked = scan["test_leaked"]
        context = {
            "scanned": scanned,
            "leaked": leaked,
            "fraction": scan["frac"],
            "per_split": scan["per_split"],
            "examples": scan["examples"],
            "train_rows_scanned": scan["train_rows_scanned"],
            "train_truncated": scan["train_truncated"],
            "fuzzy_threshold": LEAKAGE_FUZZY_THRESHOLD,
        }
        if scan["severity"] == "ok":
            signals.append(_make_signal(
                id="leakage.no_overlap",
                severity="ok",
                headline=(
                    f"No leakage — {scanned} gold rows checked against "
                    f"{scan['train_rows_scanned']} training rows, none overlap."
                ),
                context=context,
            ))
        else:
            test_note = (
                f" — including {test_leaked} GOLD_TEST row(s) (your final grade)"
                if test_leaked
                else ""
            )
            signals.append(_make_signal(
                id="leakage.gold_train_overlap",
                severity=scan["severity"],
                headline=(
                    f"{leaked} of {scanned} gold rows ({scan['frac'] * 100:.0f}%) "
                    f"also appear in training data{test_note}."
                ),
                suggested_action={
                    "kind": "navigate",
                    "label": "Re-split so gold is held out",
                    "target": "dataprep",
                },
                context=context,
                # No autofix_kind: removing rows from the gold set or the
                # training set is a judgement call (which copy is
                # canonical?), never a safe one-click delete.
            ))

    # ── Prepared splits (train / val / test) ───────────────────
    split_scan = await scan_prepared_split_leakage(db, project_id)
    if split_scan is not None:
        leaked = split_scan["total_leaked"]
        scanned = split_scan["total_scanned"]
        context = {
            "scanned": scanned,
            "leaked": leaked,
            "worst_fraction": split_scan["worst_frac"],
            "per_pair": split_scan["per_pair"],
            "examples": split_scan["examples"],
            "fuzzy_threshold": LEAKAGE_FUZZY_THRESHOLD,
        }
        if split_scan["severity"] == "ok":
            signals.append(_make_signal(
                id="leakage.splits_held_out",
                severity="ok",
                headline=(
                    f"Train / val / test splits are disjoint — "
                    f"{scanned} held-out rows checked, none shared."
                ),
                context=context,
            ))
        else:
            # Name the worst-leaking pair in the headline.
            worst_pair = max(
                split_scan["per_pair"].items(),
                key=lambda kv: kv[1]["leaked"],
            )[0]
            pretty = worst_pair.replace("_in_", " rows also in ")
            signals.append(_make_signal(
                id="leakage.split_overlap",
                severity=split_scan["severity"],
                headline=(
                    f"{leaked} row(s) shared across prepared splits "
                    f"(worst: {pretty})."
                ),
                suggested_action={
                    "kind": "navigate",
                    "label": "Re-split with dedup",
                    "target": "dataprep",
                },
                context=context,
            ))

    return {
        "id": "leakage",
        "title": "Leakage",
        "subtitle": "Are eval rows held out of training?",
        "signals": signals,
    }


# ─────────────────────────────────────────────────────────────────────
# Public entry point.
# ─────────────────────────────────────────────────────────────────────


async def compute_data_health_report(
    db: AsyncSession, project_id: int
) -> dict[str, Any]:
    """Aggregate every data-quality signal for a project into one
    panel-friendly payload. See module docstring for the shape.

    Raises ``ValueError`` if the project doesn't exist — the API layer
    translates that to 404.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    # Resolve task_profile once at the top so the per-group builders
    # don't each re-query the recipe service. None when no recipe is
    # selected, which is the state shape_group will flag separately.
    task_profile = _resolve_task_profile(project)

    groups = [
        await _ingestion_group(db, project_id),
        await _cleaning_group(db, project_id, task_profile=task_profile),
        await _shape_group(db, project),
        await _leakage_group(db, project_id),
        await _balance_group(db, project),
    ]

    all_signals = [s for g in groups for s in g["signals"]]
    severity_summary = {
        "ok": sum(1 for s in all_signals if s["severity"] == "ok"),
        "warn": sum(1 for s in all_signals if s["severity"] == "warn"),
        "block": sum(1 for s in all_signals if s["severity"] == "block"),
    }
    # Overall verdict: any block → red; any warn → amber; else green.
    if severity_summary["block"] > 0:
        overall: Severity = "block"
    elif severity_summary["warn"] > 0:
        overall = "warn"
    else:
        overall = "ok"

    return {
        "project_id": int(project_id),
        "computed_at": _utcnow().isoformat(),
        "overall": overall,
        "severity_summary": severity_summary,
        "total_signals": len(all_signals),
        "groups": groups,
    }
