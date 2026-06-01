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


# ─────────────────────────────────────────────────────────────────────
# Layman translation tables. Keyed on the technical signal id; each
# row carries the plain-English summary + "why this matters" line.
# Tables are central so updating one signal's wording doesn't
# require touching the signal-emitter code.
# ─────────────────────────────────────────────────────────────────────

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

    return {
        "id": "cleaning",
        "title": "Cleaning",
        "subtitle": "PII redaction + quality + dedup",
        "signals": signals,
    }


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

    return {
        "id": "shape",
        "title": "Data shape vs recipe",
        "subtitle": "Does the data fit the recipe?",
        "signals": signals,
    }


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
