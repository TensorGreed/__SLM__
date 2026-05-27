"""LLM-assisted gold-set generation across all supported recipes.

Crafts a project + recipe-aware prompt, calls a flagship cloud LLM
(OpenAI or Anthropic), parses the response into gold rows in the
recipe's canonical shape, and returns them to the caller for
**preview-then-save** review. This service does NOT persist anything
— the API endpoint shows the rows to the user, the user picks which
to keep, and the existing ``POST /api/projects/{id}/gold/import``
writes the accepted subset.

Supported recipes (each = its own prompt builder + parser):
  * ``qa-sft`` — `{question, answer}` rows
  * ``classification`` — `{text, label}` rows (labels seeded from
    existing gold rows if any)
  * ``span-extraction`` — `{text, entities: [{type, start, end, text}]}`
  * ``summarization`` — `{document, summary}` rows

Providers: OpenAI (incl. Deepseek + custom OpenAI-compatible
endpoints) and Anthropic. See ``cloud_llm_service``.

Prompt incorporates the active domain blueprint's ``problem_statement``
+ ``brief_text`` + ``domain_name`` when present so generated rows
reflect the actual project, not generic boilerplate.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Literal


_LOG = logging.getLogger("gold_llm")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.domain_blueprint import DomainBlueprintRevision
from app.models.project import Project
from app.services.cloud_llm_service import (
    CloudLlmError,
    call_anthropic_chat,
    call_openai_chat,
    extract_json_payload,
)
# Imported at module level (not inside the function) so tests can
# patch ``app.services.gold_llm_service._load_project_cleaned_chunks``.
# No import cycle — synthetic_service doesn't import from this module.
from app.services.synthetic_service import _load_project_cleaned_chunks


Provider = Literal["openai", "anthropic"]

# Recipes the LLM-assisted gold-gen path supports. Each one has its
# own prompt builder + parser; adding a fifth recipe is a 3-function
# extension, not a refactor. Touched in three places:
#   * ``_RECIPE_PROMPT_BUILDERS`` (prompt construction)
#   * ``_RECIPE_PARSERS`` (LLM-response parsing)
#   * ``_RECIPE_FOCUS_DEFAULTS`` (per-recipe focus-hint placeholders)
SUPPORTED_RECIPES: tuple[str, ...] = (
    "qa-sft",
    "classification",
    "span-extraction",
    "summarization",
)


@dataclass
class GeneratedQa:
    """Legacy qa-sft row dataclass. Kept for backward compat with the
    existing ``_parse_qa_payload`` tests and the qa-sft path. For
    non-qa-sft recipes the service returns plain dicts in the
    recipe's canonical shape — see ``_parse_classification_rows``
    et al."""
    question: str
    answer: str
    rationale: str = ""
    # When grounding is on, the LLM is asked to include the source
    # excerpt that backs each answer. Empty string when grounding
    # was off OR when the LLM omitted the field.
    source_excerpt: str = ""
    # Difficulty + hallucination-trap labels. The LLM is asked to
    # populate these on every row (even when no distribution was
    # specified) so the saved gold set captures the row mix. Defaults
    # apply when the LLM omits the field. Both round-trip through
    # ``/gold/import`` because that endpoint preserves arbitrary
    # caller-supplied fields.
    difficulty: str = "medium"
    is_hallucination_trap: bool = False


@dataclass
class ReferenceChunk:
    text: str
    source_label: str  # short tag like "doc-3/chunk-12" so the user can locate

    def to_prompt_block(self, idx: int) -> str:
        return f"[REF-{idx}] ({self.source_label})\n{self.text}"


@dataclass
class GenerationResult:
    # Recipe-shaped rows: list[dict] across all recipes. Keys per
    # recipe are documented at the parser level. qa-sft rows are also
    # surfaced as ``GeneratedQa`` dataclasses internally before being
    # converted to dicts at the API boundary — this preserves the
    # tests that import the parser directly.
    rows: list[dict[str, Any]]
    recipe_id: str
    provider: Provider
    model: str
    prompt_tokens: int
    completion_tokens: int
    prompt_preview: str
    # New fields for grounding + cost transparency.
    reference_chunk_count: int = 0
    estimated_cost_usd: float = 0.0


# ─────────────────────────────────────────────────────────────────────
# Pricing — approximate per-token prices (USD per 1M tokens) for the
# models we expose in the UI. Used to compute ``estimated_cost_usd``
# both for the pre-call estimate the UI shows next to the Generate
# button AND the post-call actual figure based on returned usage.
#
# Numbers tracked are *list prices as of late 2025*; they drift, but
# the order-of-magnitude is what matters for the user-facing "≈ $X.YY"
# label. A stale price by 20% doesn't change the "is this safe?"
# verdict for a $0.02 generation. Update when models / prices shift.
# ─────────────────────────────────────────────────────────────────────


# (input_per_1m_usd, output_per_1m_usd). Keys are normalized to lower
# and matched as a prefix (so "claude-haiku-4-5-20251001" matches
# "claude-haiku-4-5") so future date-stamped variants Just Work.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Anthropic
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    # Deepseek (V3 pricing as of late 2025; deepseek-reasoner is the
    # R1 family). Prefix-match means future deepseek-* variants
    # (including the unconfirmed "deepseek-v4-pro" some users have
    # asked about) resolve to the cheapest-tier fallback below if
    # they don't match a known prefix — never the wrong way.
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}

# Approximate-token heuristic: 1 token ≈ 4 chars for English. We use
# this for the *pre-call* estimate (where we have no actual token
# count yet); post-call uses the provider's returned usage figures.
_CHARS_PER_TOKEN = 4.0


def _lookup_pricing(model: str) -> tuple[float, float]:
    """Return (input_per_1m, output_per_1m) USD prices. Falls back
    to the cheapest tier when the model is unknown so the estimate
    is never wildly low (defensive — the user only cares about ceiling)."""
    token = (model or "").strip().lower()
    for prefix, prices in _MODEL_PRICING.items():
        if token.startswith(prefix):
            return prices
    return (0.15, 0.60)  # cheapest-tier fallback


def compute_estimated_cost_usd(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """USD cost for a single call given actual usage from the provider."""
    in_per_1m, out_per_1m = _lookup_pricing(model)
    return round(
        (prompt_tokens / 1_000_000) * in_per_1m
        + (completion_tokens / 1_000_000) * out_per_1m,
        6,
    )


def estimate_call_cost_usd(
    *,
    model: str,
    count: int,
    grounded: bool,
    reference_chunk_count: int,
    reference_total_chars: int,
) -> dict[str, float | int]:
    """Pre-call cost estimate the UI surfaces next to the Generate
    button. Deliberately *generous* on the prompt side so users
    aren't surprised by a higher post-call number.

    Token model:
      * Prompt fixed overhead (system + instructions + project ctx):
        ~600 tokens.
      * Per-row overhead in prompt (count appears in instructions):
        ~5 tokens × count.
      * Reference material (when grounded): chars/4 across all chunks.
      * Completion: ~120 tokens per row (Q + A + rationale + maybe
        source_excerpt). Grounded responses are longer because of
        the excerpt — bumps to ~160.
    """
    fixed_prompt_tokens = 600 + 5 * count
    reference_tokens = (
        int(reference_total_chars / _CHARS_PER_TOKEN) if grounded else 0
    )
    completion_per_row = 160 if grounded else 120
    completion_tokens = completion_per_row * count
    prompt_tokens = fixed_prompt_tokens + reference_tokens
    estimated = compute_estimated_cost_usd(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return {
        "estimated_cost_usd": estimated,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "reference_chunk_count": reference_chunk_count,
    }


class GoldGenerationError(ValueError):
    """Raised for caller-fixable problems (missing recipe, wrong
    recipe shape, missing API key, parse failure on the LLM
    response). The API layer translates these to 400 with a
    structured error code."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


# ─────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────


# Per-recipe system prompts. The qa-sft prompt is the original one
# — kept verbatim so existing snapshots / costs don't drift. The
# others are tuned for each shape: classification reminds the LLM
# that labels are categorical (not free-text), span-extraction is
# warned about offset accuracy, summarization is warned about
# faithfulness + length discipline.
_SYSTEM_PROMPT_QA = (
    "You generate gold-standard question/answer pairs for fine-tuning a "
    "small language model. The pairs you generate become the EVALUATION "
    "ground truth for the model — quality + correctness matter more than "
    "volume. Avoid trivia; favor questions a real user of this product "
    "would ask. Ground every answer in plausible domain knowledge."
)

_SYSTEM_PROMPT_CLASSIFICATION = (
    "You generate gold-standard classification examples for fine-tuning "
    "a small language model. Each example is a short text snippet paired "
    "with a SINGLE categorical label drawn from a fixed vocabulary. The "
    "rows you generate become EVALUATION ground truth — label correctness "
    "matters more than text variety. Stay strictly within the provided "
    "label set; if you find yourself wanting to coin a new label, you "
    "are off-task."
)

_SYSTEM_PROMPT_SPAN = (
    "You generate gold-standard span-extraction examples for fine-tuning "
    "a small language model. Each example is a text snippet paired with "
    "a JSON list of entity spans — every span has type, start offset, "
    "end offset, and the exact substring at those offsets. Offsets MUST "
    "be character-accurate against the text you provide; off-by-one "
    "errors render the row useless. The rows you generate become "
    "EVALUATION ground truth — span boundaries are load-bearing."
)

_SYSTEM_PROMPT_SUMMARIZATION = (
    "You generate gold-standard summarization examples for fine-tuning "
    "a small language model. Each example pairs a longer document with "
    "a SHORTER reference summary (1-5 sentences typically). Summaries "
    "must be FAITHFUL — every claim grounded in the source document, "
    "no fabricated facts. The rows you generate become EVALUATION "
    "ground truth — faithfulness + concision matter more than fluency."
)

# Kept as an alias so any external import of the old name still works.
_SYSTEM_PROMPT = _SYSTEM_PROMPT_QA


# ─────────────────────────────────────────────────────────────────────
# Reference-material sampling — strict char/total budget for cost
# control. Grounding sends excerpts of the project's actual source
# data (cleaned chunks) to the LLM so the answers are anchored to
# what the trained model could plausibly have learned. Without these
# caps a 500-chunk project could push the prompt to 100K+ tokens —
# a few cents per call on the cheap models, but a few dollars on
# Sonnet / gpt-4o for high counts. The caps keep a worst-case call
# well under 3¢ on the priciest supported model.
# ─────────────────────────────────────────────────────────────────────


# Hard caps (also surfaced to the UI via the cost estimator).
MAX_REFERENCE_CHUNKS = 6
MAX_CHARS_PER_REFERENCE_CHUNK = 1500
MAX_REFERENCE_TOTAL_CHARS = 8000


def _sample_reference_chunks(
    pool: list[str],
    *,
    max_chunks: int = MAX_REFERENCE_CHUNKS,
    max_chars_per_chunk: int = MAX_CHARS_PER_REFERENCE_CHUNK,
    max_total_chars: int = MAX_REFERENCE_TOTAL_CHARS,
    seed: int = 0,
) -> list[ReferenceChunk]:
    """Stratified sample of cleaned-chunk text for the prompt.

    Picks first + last + a few in between (deterministic via ``seed``)
    so the LLM sees representative material without the prompt blowing
    up. Each chunk is hard-truncated to ``max_chars_per_chunk``; the
    total across all chunks is capped at ``max_total_chars``.

    Returns ``[]`` when the pool is empty so the caller can gracefully
    fall back to ungrounded generation (e.g. a fresh project that
    hasn't imported data yet).
    """
    if not pool:
        return []

    n = len(pool)
    # Pick stratified indexes: first, last, plus evenly spaced fillers.
    # Cap at min(max_chunks, len(pool)).
    target = min(max_chunks, n)
    if target == 1:
        indexes = [0]
    elif target == 2:
        indexes = [0, n - 1]
    else:
        step = (n - 1) / (target - 1)
        indexes = sorted({round(i * step) for i in range(target)})
        # Round-collapse can drop us below target on very small pools;
        # backfill from a fresh shuffle until we hit it.
        if len(indexes) < target:
            rng = random.Random(seed)
            remaining = [i for i in range(n) if i not in indexes]
            rng.shuffle(remaining)
            for i in remaining:
                indexes.append(i)
                if len(indexes) >= target:
                    break
            indexes = sorted(set(indexes))

    chunks: list[ReferenceChunk] = []
    total_chars = 0
    for idx in indexes:
        if idx < 0 or idx >= n:
            continue
        raw = (pool[idx] or "").strip()
        if not raw:
            continue
        # Hard-truncate per-chunk before checking the global budget.
        text = raw[:max_chars_per_chunk]
        if total_chars + len(text) > max_total_chars:
            text = text[: max(0, max_total_chars - total_chars)]
            if not text:
                break
        chunks.append(
            ReferenceChunk(
                text=text,
                source_label=f"chunk-{idx + 1}-of-{n}",
            ),
        )
        total_chars += len(text)
        if total_chars >= max_total_chars:
            break
    return chunks


def _build_context_sections(
    *,
    project: Project,
    blueprint: DomainBlueprintRevision | None,
    focus_hint: str,
) -> list[str]:
    """Common project-context preamble shared across all recipe
    prompt builders. Kept as one source of truth so adding fields to
    the context preamble lands in all four recipe prompts at once."""
    domain = (getattr(blueprint, "domain_name", "") or "").strip()
    problem = (getattr(blueprint, "problem_statement", "") or "").strip()
    brief = (getattr(blueprint, "brief_text", "") or "").strip()
    project_name = (project.name or "").strip()
    project_desc = (project.description or "").strip()

    sections: list[str] = []
    sections.append(f"PROJECT: {project_name}")
    if project_desc:
        sections.append(f"DESCRIPTION: {project_desc}")
    if domain:
        sections.append(f"DOMAIN: {domain}")
    if problem:
        sections.append(f"PROBLEM STATEMENT: {problem}")
    if brief:
        sections.append(f"USER'S BRIEF: {brief}")
    if focus_hint.strip():
        sections.append(f"FOCUS REQUEST: {focus_hint.strip()}")
    return sections


def _build_reference_section(
    reference_chunks: list[ReferenceChunk] | None,
    *,
    label_pronoun: str = "answers",
) -> str | None:
    """When grounding is on + chunks exist, render the REFERENCE
    MATERIAL prompt section. Returns None when not grounded so the
    caller can skip the parts list cleanly."""
    if not reference_chunks:
        return None
    ref_blocks = "\n\n".join(
        chunk.to_prompt_block(i + 1)
        for i, chunk in enumerate(reference_chunks)
    )
    return (
        "REFERENCE MATERIAL (the model being trained will only learn "
        f"from material like this — {label_pronoun} MUST be grounded "
        "in these excerpts):\n\n" + ref_blocks
    )


def _build_user_prompt(
    *,
    project: Project,
    blueprint: DomainBlueprintRevision | None,
    count: int,
    focus_hint: str,
    reference_chunks: list[ReferenceChunk] | None = None,
) -> str:
    """Compose the user-facing prompt for the qa-sft path. Pulls
    project / blueprint context inline so the LLM produces
    domain-relevant rows.

    When ``reference_chunks`` is non-empty, the prompt instructs the
    LLM to GROUND each answer in one of the provided excerpts and
    include a ``source_excerpt`` field per row pointing at the
    supporting passage. Otherwise the prompt asks for general
    domain-relevant Q&A (legacy v1 behavior).

    Kept as a thin wrapper over ``_build_qa_prompt`` so external
    callers / tests that monkeypatched this symbol still work."""
    return _build_qa_prompt(
        project=project,
        blueprint=blueprint,
        count=count,
        focus_hint=focus_hint,
        reference_chunks=reference_chunks,
    )


def _build_qa_prompt(
    *,
    project: Project,
    blueprint: DomainBlueprintRevision | None,
    count: int,
    focus_hint: str,
    reference_chunks: list[ReferenceChunk] | None = None,
    distribution: tuple[int, int, int, int] | None = None,
) -> str:
    sections = _build_context_sections(
        project=project, blueprint=blueprint, focus_hint=focus_hint,
    )
    context_block = "\n".join(sections) if sections else "(no project context available)"
    grounded = bool(reference_chunks)
    parts: list[str] = [context_block]
    ref_section = _build_reference_section(reference_chunks, label_pronoun="answers")
    if ref_section:
        parts.append(ref_section)

    # When a distribution is provided, enumerate the mix explicitly so
    # the LLM crafts the right rows. Otherwise fall back to the
    # "Generate N + vary difficulty" wording.
    if distribution is not None:
        easy, medium, hard, traps = distribution
        bucket_lines: list[str] = []
        if easy:
            bucket_lines.append(
                f"  * {easy} EASY question{'s' if easy != 1 else ''} — "
                "direct lookups, single fact, the answer is clear from "
                "a single passage in the source."
            )
        if medium:
            bucket_lines.append(
                f"  * {medium} MEDIUM question{'s' if medium != 1 else ''} — "
                "require combining 2-3 facts, moderate inference, or "
                "domain-vocabulary disambiguation."
            )
        if hard:
            bucket_lines.append(
                f"  * {hard} HARD question{'s' if hard != 1 else ''} — "
                "multi-hop reasoning across passages, edge cases, "
                "ambiguous wording, or judgment calls a domain expert "
                "would have to mediate."
            )
        if traps:
            bucket_lines.append(
                f"  * {traps} HALLUCINATION TRAP{'s' if traps != 1 else ''} — "
                "questions whose answer is NOT in the project's domain "
                "knowledge (or this source material). The reference "
                "answer should explicitly say 'I don't know' / 'that "
                "isn't covered' / similar. These exist to test the "
                "trained model's ability to refuse rather than fabricate."
            )
        parts.append(
            f"Generate exactly {count} question/answer pairs total, "
            f"distributed as follows:\n" + "\n".join(bucket_lines)
        )
    else:
        parts.append(
            f"Generate exactly {count} question/answer pairs for the project above."
        )

    rules = [
        "Return ONLY valid JSON. No markdown, no code fences, no commentary.",
        'Shape: {"pairs": [{"question": "...", "answer": "...", '
        '"difficulty": "easy|medium|hard", "is_hallucination_trap": true|false}]}',
    ]
    if grounded:
        rules.extend([
            "Prefer answers grounded in the REFERENCE MATERIAL above. "
            "Use the project's domain knowledge where the material is "
            "thin — do NOT invent contradictions.",
            "Optionally include a ``source_excerpt`` field (≤200 chars) "
            "quoting the supporting reference passage when grounding is "
            "clear. Skip the field when uncertain rather than fabricating "
            "a citation.",
            "For HALLUCINATION TRAP rows specifically: do NOT supply a "
            "``source_excerpt`` (the point is the answer isn't in the "
            "source).",
        ])
    else:
        rules.append(
            "Each answer MUST be self-contained and factually defensible.",
        )

    rules.extend([
        "Each question MUST be specific to this project's domain.",
        # Per-row labeling: REQUIRED so the gold set captures the mix.
        # Even when no distribution was specified the LLM still needs to
        # tag each row so the user can filter / report by difficulty.
        '``difficulty`` field on each row is REQUIRED — exactly one of '
        '"easy", "medium", "hard".',
        '``is_hallucination_trap`` field on each row is REQUIRED — '
        'true|false. Use ``true`` ONLY for rows whose answer is "I don\'t '
        'know" or "not covered" style.',
        "Optionally include a ``rationale`` field — one short sentence on "
        "why this is the right answer (used by some evaluators).",
    ])
    if distribution is None:
        rules.insert(
            -1,
            "Vary difficulty: mix factual lookups, edge cases, and "
            "judgment calls (label each row with its ``difficulty`` field).",
        )
    rules.append(f"Return EXACTLY {count} pairs in the ``pairs`` array.")

    parts.append("Output rules:\n- " + "\n- ".join(rules))
    return "\n\n".join(parts)


def _build_classification_prompt(
    *,
    project: Project,
    blueprint: DomainBlueprintRevision | None,
    count: int,
    focus_hint: str,
    reference_chunks: list[ReferenceChunk] | None = None,
    known_labels: list[str] | None = None,
) -> str:
    """Classification-shape prompt. When ``known_labels`` is non-empty
    the LLM is locked to that vocabulary; otherwise it picks labels
    from focus_hint or infers them from the source material (the
    user can then edit + standardize during preview)."""
    sections = _build_context_sections(
        project=project, blueprint=blueprint, focus_hint=focus_hint,
    )
    context_block = "\n".join(sections) if sections else "(no project context available)"
    grounded = bool(reference_chunks)
    parts: list[str] = [context_block]

    if known_labels:
        # Stable order so the prompt cache + LLM behavior is
        # deterministic across calls with the same vocabulary.
        label_list = ", ".join(sorted({l.strip() for l in known_labels if l.strip()}))
        parts.append(
            "LABEL VOCABULARY (use ONLY these labels — coining new labels is "
            f"off-task):\n{label_list}"
        )

    ref_section = _build_reference_section(reference_chunks, label_pronoun="text snippets")
    if ref_section:
        parts.append(ref_section)

    parts.append(
        f"Generate exactly {count} classification examples for the project above."
    )

    rules = [
        "Return ONLY valid JSON. No markdown, no code fences, no commentary.",
        'Shape: {"rows": [{"text": "...", "label": "..."}]}',
    ]
    if known_labels:
        rules.append(
            "Each ``label`` MUST be one of the labels listed in the "
            "LABEL VOCABULARY above — verbatim, case-sensitive. Drop the "
            "row entirely rather than emit a label not in the vocabulary."
        )
    else:
        rules.append(
            "If the FOCUS REQUEST names a label set, use ONLY those labels. "
            "Otherwise pick a small consistent label set (3-6 labels) and "
            "reuse the same labels across rows — DO NOT invent a new label "
            "per row."
        )
    if grounded:
        rules.extend([
            "Prefer text snippets drawn from the REFERENCE MATERIAL above. "
            "Paraphrasing is fine; do NOT invent details that contradict "
            "the source.",
            "Optionally include a ``source_excerpt`` field (≤200 chars) "
            "quoting the supporting reference passage. Skip the field "
            "when uncertain rather than fabricating a citation.",
        ])
    else:
        rules.append(
            "Each text MUST be specific to this project's domain — short "
            "(1-3 sentences typical), realistic for the user being modeled."
        )
    rules.extend([
        "Cover a mix of labels — class balance matters for eval reliability.",
        "Include a few edge cases / ambiguous examples (clearly tagged via rationale).",
        "Optionally include a ``rationale`` field — one short sentence on "
        "why this label is correct.",
        f"Return EXACTLY {count} rows in the ``rows`` array.",
    ])
    parts.append("Output rules:\n- " + "\n- ".join(rules))
    return "\n\n".join(parts)


def _build_span_prompt(
    *,
    project: Project,
    blueprint: DomainBlueprintRevision | None,
    count: int,
    focus_hint: str,
    reference_chunks: list[ReferenceChunk] | None = None,
) -> str:
    """Span-extraction prompt. Each row is text + a JSON list of
    `{type, start, end, text}` entity spans. Strict on offset
    accuracy because off-by-one renders the row useless for eval."""
    sections = _build_context_sections(
        project=project, blueprint=blueprint, focus_hint=focus_hint,
    )
    context_block = "\n".join(sections) if sections else "(no project context available)"
    grounded = bool(reference_chunks)
    parts: list[str] = [context_block]

    ref_section = _build_reference_section(reference_chunks, label_pronoun="text snippets")
    if ref_section:
        parts.append(ref_section)

    parts.append(
        f"Generate exactly {count} span-extraction examples for the project above."
    )

    rules = [
        "Return ONLY valid JSON. No markdown, no code fences, no commentary.",
        'Shape: {"rows": [{"text": "...", "entities": [{"type": "...", '
        '"start": <int>, "end": <int>, "text": "..."}]}]}',
        "``start`` and ``end`` are 0-indexed character offsets into ``text``. "
        "``text[start:end]`` MUST equal the span's ``text`` field — verify "
        "by mentally slicing before emitting.",
        "``end`` is EXCLUSIVE (Python slice convention). A span of \"foo\" "
        "starting at offset 5 has end=8.",
        "Entity ``type`` values: use a small consistent vocabulary across rows. "
        "If the FOCUS REQUEST names span types, use ONLY those types.",
        "Rows can have multiple spans; an empty ``entities`` list is allowed "
        "(no-span examples are useful negative cases).",
    ]
    if grounded:
        rules.extend([
            "Prefer text snippets drawn from the REFERENCE MATERIAL above. "
            "Paraphrasing OK; do NOT invent details contradicting the source.",
            "Optionally include a ``source_excerpt`` field (≤200 chars) "
            "quoting the supporting reference passage. Skip the field "
            "when uncertain.",
        ])
    else:
        rules.append(
            "Each text MUST be realistic for this project's domain. Keep "
            "snippets short (1-3 sentences) so offsets stay verifiable."
        )
    rules.extend([
        "Mix span density: some rows with 1 entity, some with 2-4, some with 0.",
        "Optionally include a ``rationale`` field — one short sentence on "
        "what makes this row interesting (edge case, boundary, ambiguity).",
        f"Return EXACTLY {count} rows in the ``rows`` array.",
    ])
    parts.append("Output rules:\n- " + "\n- ".join(rules))
    return "\n\n".join(parts)


def _build_summarization_prompt(
    *,
    project: Project,
    blueprint: DomainBlueprintRevision | None,
    count: int,
    focus_hint: str,
    reference_chunks: list[ReferenceChunk] | None = None,
) -> str:
    """Summarization prompt. Each row is a longer document + a
    shorter faithful summary. Strict on faithfulness — fabricated
    facts are the failure mode this gold set is designed to catch."""
    sections = _build_context_sections(
        project=project, blueprint=blueprint, focus_hint=focus_hint,
    )
    context_block = "\n".join(sections) if sections else "(no project context available)"
    grounded = bool(reference_chunks)
    parts: list[str] = [context_block]

    ref_section = _build_reference_section(reference_chunks, label_pronoun="documents")
    if ref_section:
        parts.append(ref_section)

    parts.append(
        f"Generate exactly {count} (document, summary) pairs for the project above."
    )

    rules = [
        "Return ONLY valid JSON. No markdown, no code fences, no commentary.",
        'Shape: {"rows": [{"document": "...", "summary": "..."}]}',
        "Each ``summary`` MUST be SHORTER than its ``document`` — typically "
        "1-5 sentences. Avoid one-word summaries; avoid summaries that "
        "exceed half the document length.",
        "Every claim in ``summary`` MUST be supported by ``document``. "
        "If a fact isn't in the document, it doesn't belong in the summary. "
        "Faithfulness > fluency.",
    ]
    if grounded:
        rules.extend([
            "Prefer documents drawn from the REFERENCE MATERIAL above. "
            "You may stitch reference excerpts together, but the resulting "
            "``document`` is what the summary must be faithful to.",
            "Optionally include a ``source_excerpt`` field (≤200 chars) "
            "quoting the most relevant supporting passage. Skip the field "
            "when uncertain.",
        ])
    else:
        rules.append(
            "Each ``document`` MUST be realistic for this project's domain "
            "(meeting notes, article, ticket thread, etc. — match the "
            "FOCUS REQUEST style if provided)."
        )
    rules.extend([
        "Vary document length: mix short (1-2 paragraphs) and longer (3-6 "
        "paragraphs) sources so eval covers both ends.",
        "Optionally include a ``rationale`` field — one short sentence on "
        "what makes this summary correct (key points covered, what to omit).",
        f"Return EXACTLY {count} rows in the ``rows`` array.",
    ])
    parts.append("Output rules:\n- " + "\n- ".join(rules))
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────


# Field aliases the parser accepts in addition to the canonical
# question/answer names. Real LLM outputs across vendors use:
#   * q / a (compact form, common in code-targeted LLMs)
#   * prompt / response (instruction-tuned models default)
#   * input / output (training-data convention)
#   * query / reply (chat-flavored shorthand)
# Keys are checked in order; first non-empty match wins.
_QUESTION_FIELD_ALIASES = (
    "question", "q", "prompt", "input", "query", "user", "instruction",
)
_ANSWER_FIELD_ALIASES = (
    "answer", "a", "response", "output", "reply", "assistant", "completion",
)

# Container keys the parser searches for the items list. Extended
# beyond the canonical "pairs" because real LLM outputs often
# label the array as "questions", "examples", "samples", etc.
_ITEM_CONTAINER_KEYS = (
    "pairs", "qa_pairs", "questions", "examples", "samples", "items", "data", "rows", "output",
)


def _extract_field(item: dict[str, Any], aliases: tuple[str, ...]) -> str:
    """Return the first non-empty stripped string match against the
    alias list. Case-insensitive on keys so models that emit
    ``Question`` or ``ANSWER`` still parse."""
    lower_keys = {k.lower(): k for k in item.keys() if isinstance(k, str)}
    for alias in aliases:
        actual_key = lower_keys.get(alias)
        if actual_key is None:
            continue
        value = item.get(actual_key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_items(payload: Any) -> list[Any]:
    """Pull the list-of-pairs out of an arbitrarily-shaped JSON
    response. Walks the known container keys, falls back to a
    single-pair object, then to any top-level list-of-dicts."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    # Known container keys, case-insensitive.
    lower_keys = {k.lower(): k for k in payload.keys() if isinstance(k, str)}
    for container in _ITEM_CONTAINER_KEYS:
        actual_key = lower_keys.get(container)
        if actual_key is None:
            continue
        candidate = payload.get(actual_key)
        if isinstance(candidate, list):
            return candidate
        # One level of nesting tolerated: e.g. {"data": {"pairs": [...]}}.
        if isinstance(candidate, dict):
            nested = _extract_items(candidate)
            if nested:
                return nested

    # Single-pair object shape — wrap it.
    if _extract_field(payload, _QUESTION_FIELD_ALIASES) and _extract_field(
        payload, _ANSWER_FIELD_ALIASES,
    ):
        return [payload]

    # Last-ditch: a top-level dict with EXACTLY one value that's a
    # list (some models wrap with a single arbitrary key like
    # ``{"result": [...]}``).
    list_values = [v for v in payload.values() if isinstance(v, list)]
    if len(list_values) == 1:
        return list_values[0]

    return []


def _parse_qa_payload(content: str, expected_count: int) -> list[GeneratedQa]:
    """Pull ``GeneratedQa`` rows out of the LLM response. Tolerates:
      * top-level list OR ``{<container>: [...]}`` with container in
        ``_ITEM_CONTAINER_KEYS`` (case-insensitive)
      * one level of nesting (``{"data": {"pairs": [...]}}``)
      * field aliases on each item (``q``/``a``, ``prompt``/``response``,
        ``input``/``output``, etc. — see ``_QUESTION_FIELD_ALIASES``)
    Rows missing question OR answer are skipped silently. Doesn't
    enforce expected_count strictly (LLMs sometimes return N±1)."""
    try:
        payload = extract_json_payload(content)
    except ValueError as exc:
        raise GoldGenerationError(
            "LLM_RESPONSE_UNPARSEABLE",
            f"LLM response was not parseable JSON: {exc}",
        ) from exc

    items = _extract_items(payload)

    rows: list[GeneratedQa] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = _extract_field(item, _QUESTION_FIELD_ALIASES)
        a = _extract_field(item, _ANSWER_FIELD_ALIASES)
        r = _extract_field(item, ("rationale", "explanation", "reasoning", "why"))
        src = _extract_field(
            item,
            ("source_excerpt", "source", "evidence", "quote", "citation"),
        )
        if not q or not a:
            continue
        difficulty = _normalize_difficulty(
            _extract_field(item, ("difficulty", "level", "complexity")),
        )
        trap = _coerce_bool(
            _extract_any(item, ("is_hallucination_trap", "hallucination_trap", "trap")),
        )
        rows.append(
            GeneratedQa(
                question=q,
                answer=a,
                rationale=r,
                source_excerpt=src,
                difficulty=difficulty,
                is_hallucination_trap=trap,
            ),
        )

    if not rows:
        # Two distinct failure modes — distinguish so the user knows
        # what to change. Both surface as 400 from the API layer.
        snippet = (content or "").strip()[:300]
        # Log full response for backend debug so support can see the
        # actual model output without the user having to copy-paste.
        _LOG.warning(
            "gold_llm parse failed — payload kind=%s top_keys=%s items_seen=%d "
            "raw_response_preview=%r",
            type(payload).__name__,
            list(payload.keys())[:8] if isinstance(payload, dict) else "(not a dict)",
            len(items),
            (content or "")[:500],
        )

        if items and not rows:
            # Items list exists but every entry was rejected — field
            # name mismatch is overwhelmingly the cause.
            first_item_keys = (
                list(items[0].keys())[:8]
                if isinstance(items[0], dict)
                else "(first item not a dict)"
            )
            raise GoldGenerationError(
                "LLM_RESPONSE_UNPARSEABLE",
                f"Got {len(items)} item(s) back, but none had recognizable "
                f"question/answer fields. The first item's keys were: "
                f"{first_item_keys}. Try a different model, or rephrase "
                f"the focus hint to push the LLM toward standard "
                f"Q&A shape. Raw response (first 300 chars): {snippet!r}",
            )
        # Empty items list — model either refused, returned the
        # wrong outer shape, or (when grounded) couldn't anchor any
        # questions in the reference material.
        raise GoldGenerationError(
            "LLM_RESPONSE_UNPARSEABLE",
            "The LLM returned valid JSON but no question/answer pairs. "
            "When grounding is on, this often means the model couldn't "
            "anchor any questions to your reference material — try "
            "ungrounded mode OR add more focused source content. "
            f"Raw response (first 300 chars): {snippet!r}",
        )
    if abs(len(rows) - expected_count) > max(2, expected_count // 4):
        # Soft warning rather than hard fail — preview lets the user
        # discard. Reject only on egregious mismatch (off by >25%).
        raise GoldGenerationError(
            "LLM_RESPONSE_COUNT_MISMATCH",
            f"Asked for {expected_count} pairs but the LLM returned {len(rows)}. "
            "Try a different model or reduce the count.",
        )
    return rows


# Field-name aliases for the non-QA recipes. Same case-insensitive
# alias-walk as `_extract_field` for QA.
_TEXT_FIELD_ALIASES = ("text", "input", "content", "snippet", "passage", "source_text")
_LABEL_FIELD_ALIASES = ("label", "class", "category", "tag", "intent", "sentiment")
_DOCUMENT_FIELD_ALIASES = ("document", "article", "source", "body", "transcript", "text", "input")
_SUMMARY_FIELD_ALIASES = ("summary", "tldr", "abstract", "recap", "output", "answer")
_ENTITIES_FIELD_ALIASES = ("entities", "spans", "annotations", "labels_json", "extractions")
_RATIONALE_ALIASES = ("rationale", "explanation", "reasoning", "why", "notes")
_SOURCE_EXCERPT_ALIASES = (
    "source_excerpt", "source", "evidence", "quote", "citation",
)


def _extract_field_raw(item: dict[str, Any], aliases: tuple[str, ...]) -> str:
    """Variant of ``_extract_field`` that does NOT strip whitespace
    from the value. Used by the span-extraction parser where leading/
    trailing whitespace is load-bearing — character offsets are
    indexed against the original text, and stripping breaks them."""
    lower_keys = {k.lower(): k for k in item.keys() if isinstance(k, str)}
    for alias in aliases:
        actual_key = lower_keys.get(alias)
        if actual_key is None:
            continue
        value = item.get(actual_key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


_DIFFICULTY_VALUES = ("easy", "medium", "hard")


def _normalize_difficulty(raw: str) -> str:
    """Coerce LLM-emitted difficulty into one of {easy, medium, hard}.
    Falls back to ``medium`` for anything unrecognized (LLMs sometimes
    emit synonyms like 'simple', 'tough', or 'expert'). Best-effort
    canonicalization keeps the saved gold set's difficulty field
    homogeneous so the user can filter/group reliably."""
    token = (raw or "").strip().lower()
    if not token:
        return "medium"
    if token in _DIFFICULTY_VALUES:
        return token
    # Common synonyms the LLM may emit.
    aliases = {
        "easy": ("simple", "trivial", "basic", "low", "lvl1", "1"),
        "medium": ("med", "moderate", "intermediate", "normal", "lvl2", "2"),
        "hard": ("difficult", "tough", "advanced", "expert", "complex", "high", "lvl3", "3"),
    }
    for canonical, syns in aliases.items():
        if token in syns:
            return canonical
    return "medium"


def _coerce_bool(value: Any) -> bool:
    """Tolerant truthy coercion for the ``is_hallucination_trap``
    field. Accepts real bools, the strings true/false/yes/no/1/0,
    and falls back to ``bool(value)`` for anything else."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "y", "t")
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def _extract_any(item: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    """Variant of ``_extract_field`` that returns the raw value (not
    stripped/string-coerced). Used for boolean fields like
    ``is_hallucination_trap`` where the LLM might emit ``true`` (bool)
    or ``"true"`` (string) — both need to round-trip."""
    lower_keys = {k.lower(): k for k in item.keys() if isinstance(k, str)}
    for alias in aliases:
        actual_key = lower_keys.get(alias)
        if actual_key is None:
            continue
        value = item.get(actual_key)
        if value is not None:
            return value
    return None


def _extract_list(item: dict[str, Any], aliases: tuple[str, ...]) -> list[Any]:
    """Variant of ``_extract_field`` that returns the first list-valued
    match. Used for entity spans on the span-extraction parser."""
    lower_keys = {k.lower(): k for k in item.keys() if isinstance(k, str)}
    for alias in aliases:
        actual_key = lower_keys.get(alias)
        if actual_key is None:
            continue
        value = item.get(actual_key)
        if isinstance(value, list):
            return value
    return []


def _raise_parse_error(
    *,
    items: list[Any],
    payload: Any,
    content: str,
    recipe_id: str,
    required_field_summary: str,
) -> None:
    """Shared structured-error emission for the non-QA parsers. Mirrors
    the diagnostic detail in ``_parse_qa_payload`` so the frontend
    error UX is consistent across recipes."""
    snippet = (content or "").strip()[:300]
    _LOG.warning(
        "gold_llm parse failed (%s) — payload kind=%s top_keys=%s items_seen=%d "
        "raw_response_preview=%r",
        recipe_id,
        type(payload).__name__,
        list(payload.keys())[:8] if isinstance(payload, dict) else "(not a dict)",
        len(items),
        (content or "")[:500],
    )
    if items:
        first_item_keys = (
            list(items[0].keys())[:8]
            if isinstance(items[0], dict)
            else "(first item not a dict)"
        )
        raise GoldGenerationError(
            "LLM_RESPONSE_UNPARSEABLE",
            f"Got {len(items)} item(s) back, but none had the required "
            f"fields for {recipe_id} ({required_field_summary}). The "
            f"first item's keys were: {first_item_keys}. Try a different "
            f"model, or rephrase the focus hint. Raw response "
            f"(first 300 chars): {snippet!r}",
        )
    raise GoldGenerationError(
        "LLM_RESPONSE_UNPARSEABLE",
        f"The LLM returned valid JSON but no usable {recipe_id} rows. "
        "When grounding is on, this often means the model couldn't "
        "anchor any examples to your reference material — try "
        "ungrounded mode OR add more focused source content. "
        f"Raw response (first 300 chars): {snippet!r}",
    )


def _check_count_drift(rows: list[Any], expected_count: int) -> None:
    """Same count-drift rule as `_parse_qa_payload` — reject only on
    egregious off-by-more-than-25% mismatch."""
    if abs(len(rows) - expected_count) > max(2, expected_count // 4):
        raise GoldGenerationError(
            "LLM_RESPONSE_COUNT_MISMATCH",
            f"Asked for {expected_count} rows but the LLM returned {len(rows)}. "
            "Try a different model or reduce the count.",
        )


def _parse_classification_rows(
    content: str,
    expected_count: int,
    *,
    known_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse classification rows. Returns a list of dicts with shape
    `{text, label, rationale, source_excerpt}`. Rows missing text OR
    label are skipped. When ``known_labels`` is non-empty rows with
    out-of-vocabulary labels are silently dropped — the model is
    asked to stay in vocabulary, and drift is a row-level rejection
    rather than a hard fail."""
    try:
        payload = extract_json_payload(content)
    except ValueError as exc:
        raise GoldGenerationError(
            "LLM_RESPONSE_UNPARSEABLE",
            f"LLM response was not parseable JSON: {exc}",
        ) from exc

    items = _extract_items(payload)
    vocab = {l.strip().lower() for l in (known_labels or []) if l.strip()}

    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = _extract_field(item, _TEXT_FIELD_ALIASES)
        label = _extract_field(item, _LABEL_FIELD_ALIASES)
        if not text or not label:
            continue
        if vocab and label.lower() not in vocab:
            continue
        rows.append({
            "text": text,
            "label": label,
            "rationale": _extract_field(item, _RATIONALE_ALIASES),
            "source_excerpt": _extract_field(item, _SOURCE_EXCERPT_ALIASES),
        })

    if not rows:
        _raise_parse_error(
            items=items,
            payload=payload,
            content=content,
            recipe_id="classification",
            required_field_summary="text + label",
        )
    _check_count_drift(rows, expected_count)
    return rows


def _parse_span_rows(
    content: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Parse span-extraction rows. Returns dicts with shape
    `{text, entities, rationale, source_excerpt}` where ``entities``
    is a list of `{type, start, end, text}` dicts. Per-span offsets
    are validated against the row's text — spans where
    ``text[start:end]`` doesn't match the span's claimed text are
    dropped silently (off-by-one is the dominant LLM failure for
    this shape; we don't poison eval with broken offsets)."""
    try:
        payload = extract_json_payload(content)
    except ValueError as exc:
        raise GoldGenerationError(
            "LLM_RESPONSE_UNPARSEABLE",
            f"LLM response was not parseable JSON: {exc}",
        ) from exc

    items = _extract_items(payload)
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Preserve original whitespace: span offsets are indexed
        # against the raw text. Stripping would silently break valid
        # spans whose offsets land inside the leading/trailing space.
        text = _extract_field_raw(item, _TEXT_FIELD_ALIASES)
        if not text:
            continue
        raw_entities = _extract_list(item, _ENTITIES_FIELD_ALIASES)
        clean_entities: list[dict[str, Any]] = []
        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue
            etype = str(ent.get("type") or ent.get("label") or "").strip()
            try:
                start = int(ent.get("start", -1))
                end = int(ent.get("end", -1))
            except (TypeError, ValueError):
                continue
            if start < 0 or end <= start or end > len(text) or not etype:
                continue
            span_text = str(ent.get("text") or text[start:end]).strip()
            # Verify offsets actually match the claimed span text. We
            # allow whitespace-tolerant equality because some LLMs
            # emit the span text with surrounding spaces stripped.
            actual = text[start:end]
            if actual.strip() != span_text.strip():
                continue
            clean_entities.append({
                "type": etype,
                "start": start,
                "end": end,
                "text": actual,  # canonicalize to the offset-derived text
            })
        # Rows can legitimately have zero entities (negative examples).
        # Accept the row as long as the text is present + any provided
        # entities passed offset validation.
        if raw_entities and not clean_entities:
            # The model tried but all spans were broken — drop the row.
            continue
        rows.append({
            "text": text,
            "entities": clean_entities,
            "rationale": _extract_field(item, _RATIONALE_ALIASES),
            "source_excerpt": _extract_field(item, _SOURCE_EXCERPT_ALIASES),
        })

    if not rows:
        _raise_parse_error(
            items=items,
            payload=payload,
            content=content,
            recipe_id="span-extraction",
            required_field_summary="text + entities[]",
        )
    _check_count_drift(rows, expected_count)
    return rows


def _parse_summarization_rows(
    content: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Parse summarization rows. Returns dicts with shape
    `{document, summary, rationale, source_excerpt}`. Rows missing
    document or summary are skipped; rows where summary >= document
    in length are dropped (a "summary" longer than its document is
    a guaranteed bad eval row)."""
    try:
        payload = extract_json_payload(content)
    except ValueError as exc:
        raise GoldGenerationError(
            "LLM_RESPONSE_UNPARSEABLE",
            f"LLM response was not parseable JSON: {exc}",
        ) from exc

    items = _extract_items(payload)
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        document = _extract_field(item, _DOCUMENT_FIELD_ALIASES)
        summary = _extract_field(item, _SUMMARY_FIELD_ALIASES)
        if not document or not summary:
            continue
        if len(summary) >= len(document):
            continue
        rows.append({
            "document": document,
            "summary": summary,
            "rationale": _extract_field(item, _RATIONALE_ALIASES),
            "source_excerpt": _extract_field(item, _SOURCE_EXCERPT_ALIASES),
        })

    if not rows:
        _raise_parse_error(
            items=items,
            payload=payload,
            content=content,
            recipe_id="summarization",
            required_field_summary="document + summary (summary < document length)",
        )
    _check_count_drift(rows, expected_count)
    return rows


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


@dataclass
class PromptPreview:
    """Returned by ``build_prompt_preview`` — the prompts the service
    WOULD send to the LLM if the caller fired generation right now.
    Surfaced via ``/preview-prompt`` so advanced users can review +
    edit before committing to the actual API call."""
    recipe_id: str
    system_prompt: str
    user_prompt: str
    reference_chunk_count: int
    # Labels resolved from existing gold rows — informational so the
    # UI can show "we'll lock the LLM to: [billing, account, ...]"
    # alongside the prompt. Empty for non-classification recipes.
    known_labels: list[str]


async def build_prompt_preview(
    db: AsyncSession,
    *,
    project_id: int,
    count: int,
    focus_hint: str = "",
    ground_in_source: bool = True,
    distribution: tuple[int, int, int, int] | None = None,
) -> PromptPreview:
    """Build the user + system prompts the service would dispatch to
    the LLM, WITHOUT actually calling the LLM. Shares all prompt-
    construction logic with ``generate_gold_qa_via_llm`` — runs the
    same recipe gate, blueprint lookup, reference-chunk sampling, and
    classification label resolution — so the user reviews the exact
    string that would go out.

    Raises ``GoldGenerationError`` for the same caller-fixable cases
    as ``generate_gold_qa_via_llm`` (missing recipe, unsupported
    recipe, count out of range)."""
    if count < 1 or count > 50:
        raise GoldGenerationError(
            "COUNT_OUT_OF_RANGE",
            "Count must be between 1 and 50.",
        )
    project = await db.get(Project, project_id)
    if project is None:
        raise GoldGenerationError(
            "PROJECT_NOT_FOUND",
            f"Project {project_id} not found.",
        )
    selected = project.selected_recipe or {}
    recipe_id = str(selected.get("recipe_id") or "")
    if not recipe_id:
        raise GoldGenerationError(
            "RECIPE_REQUIRED",
            "Project has no selected recipe — pick a recipe before "
            "generating gold rows.",
        )
    if recipe_id not in SUPPORTED_RECIPES:
        raise GoldGenerationError(
            "RECIPE_NOT_SUPPORTED",
            f"LLM-assisted gold generation supports {list(SUPPORTED_RECIPES)} "
            f"(project is using '{recipe_id}').",
        )

    blueprint: DomainBlueprintRevision | None = None
    if project.active_domain_blueprint_version is not None:
        bp_result = await db.execute(
            select(DomainBlueprintRevision)
            .where(
                DomainBlueprintRevision.project_id == project_id,
                DomainBlueprintRevision.version == project.active_domain_blueprint_version,
            )
            .limit(1),
        )
        blueprint = bp_result.scalar_one_or_none()

    reference_chunks: list[ReferenceChunk] = []
    if ground_in_source:
        pool = await _load_project_cleaned_chunks(project_id)
        reference_chunks = _sample_reference_chunks(pool)

    known_labels: list[str] = []
    if recipe_id == "classification":
        known_labels = await _load_known_classification_labels(project_id)

    # Distribution applies only to qa-sft — other recipes' gold_template
    # has no difficulty / is_hallucination_trap field, so silently
    # ignoring on those recipes is the least-surprising behavior.
    effective_distribution: tuple[int, int, int, int] | None = (
        distribution if recipe_id == "qa-sft" else None
    )
    # The distribution's total replaces ``count`` for prompt-building
    # so the LLM gets the right "Generate exactly N rows" line.
    effective_count = (
        sum(effective_distribution)
        if effective_distribution is not None
        else count
    )
    user_prompt = _build_prompt_for_recipe(
        recipe_id=recipe_id,
        project=project,
        blueprint=blueprint,
        count=effective_count,
        focus_hint=focus_hint,
        reference_chunks=reference_chunks if reference_chunks else None,
        known_labels=known_labels,
        distribution=effective_distribution,
    )
    system_prompt = _SYSTEM_PROMPTS_BY_RECIPE.get(recipe_id, _SYSTEM_PROMPT_QA)
    return PromptPreview(
        recipe_id=recipe_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reference_chunk_count=len(reference_chunks),
        known_labels=known_labels,
    )


async def generate_gold_qa_via_llm(
    db: AsyncSession,
    *,
    project_id: int,
    provider: Provider,
    model: str,
    api_key: str,
    count: int,
    focus_hint: str = "",
    api_url: str | None = None,
    ground_in_source: bool = True,
    user_prompt_override: str | None = None,
    system_prompt_override: str | None = None,
    distribution: tuple[int, int, int, int] | None = None,
) -> GenerationResult:
    """Generate ``count`` Q&A pairs for the project using the named
    cloud provider. Returns the parsed rows for preview — does NOT
    persist. The caller saves accepted rows via the existing
    ``/gold/import`` endpoint.

    When ``ground_in_source`` is True (default), pulls a strict-budget
    sample of the project's cleaned chunks and asks the LLM to ground
    each answer in them. Gracefully falls back to ungrounded
    generation when no cleaned chunks exist (fresh project that hasn't
    imported data yet) — the response payload reflects the actual
    chunk count used.

    Raises ``GoldGenerationError`` for caller-fixable problems
    (recipe missing / wrong shape / API key missing / response
    unparseable). Raises ``CloudLlmError`` for upstream provider
    failures (bad credentials, rate-limit, model not found)."""
    if count < 1 or count > 50:
        raise GoldGenerationError(
            "COUNT_OUT_OF_RANGE",
            "Count must be between 1 and 50.",
        )

    project = await db.get(Project, project_id)
    if project is None:
        raise GoldGenerationError(
            "PROJECT_NOT_FOUND",
            f"Project {project_id} not found.",
        )

    selected = project.selected_recipe or {}
    recipe_id = str(selected.get("recipe_id") or "")
    if not recipe_id:
        raise GoldGenerationError(
            "RECIPE_REQUIRED",
            "Project has no selected recipe — pick a recipe before "
            "generating gold Q&A.",
        )
    if recipe_id not in SUPPORTED_RECIPES:
        raise GoldGenerationError(
            "RECIPE_NOT_SUPPORTED",
            f"LLM-assisted gold generation supports {list(SUPPORTED_RECIPES)} "
            f"(project is using '{recipe_id}').",
        )

    # Pull the active domain blueprint for richer prompting.
    blueprint: DomainBlueprintRevision | None = None
    if project.active_domain_blueprint_version is not None:
        bp_result = await db.execute(
            select(DomainBlueprintRevision)
            .where(
                DomainBlueprintRevision.project_id == project_id,
                DomainBlueprintRevision.version == project.active_domain_blueprint_version,
            )
            .limit(1),
        )
        blueprint = bp_result.scalar_one_or_none()

    # Grounding — pull a strict-budget sample of cleaned chunks when
    # requested. Graceful fallback to ungrounded when the project
    # hasn't imported any data yet (fresh project, empty chunk pool).
    reference_chunks: list[ReferenceChunk] = []
    if ground_in_source:
        pool = await _load_project_cleaned_chunks(project_id)
        reference_chunks = _sample_reference_chunks(pool)

    # Classification needs the project's label space so the LLM stays
    # in vocabulary. Pull from any existing gold rows on disk — empty
    # is fine, the prompt has a fallback path.
    known_labels: list[str] = []
    if recipe_id == "classification":
        known_labels = await _load_known_classification_labels(project_id)

    # Prompt resolution: caller-supplied override > service-built
    # prompt. When ``user_prompt_override`` is set, the user has gone
    # through the "review & edit" UX — we don't second-guess their
    # rewrite. Same for ``system_prompt_override``. The two can be
    # mixed independently (override one, default the other).
    used_user_override = bool(
        user_prompt_override and user_prompt_override.strip()
    )
    used_system_override = bool(
        system_prompt_override and system_prompt_override.strip()
    )
    # Distribution applies only to qa-sft — silently ignored elsewhere.
    effective_distribution: tuple[int, int, int, int] | None = (
        distribution if recipe_id == "qa-sft" else None
    )
    effective_count = (
        sum(effective_distribution)
        if effective_distribution is not None
        else count
    )
    if used_user_override:
        user_prompt = user_prompt_override  # type: ignore[assignment]
    else:
        user_prompt = _build_prompt_for_recipe(
            recipe_id=recipe_id,
            project=project,
            blueprint=blueprint,
            count=effective_count,
            focus_hint=focus_hint,
            reference_chunks=reference_chunks if reference_chunks else None,
            known_labels=known_labels,
            distribution=effective_distribution,
        )
    if used_system_override:
        system_prompt = system_prompt_override  # type: ignore[assignment]
    else:
        system_prompt = _SYSTEM_PROMPTS_BY_RECIPE.get(recipe_id, _SYSTEM_PROMPT_QA)

    # When the caller rewrote the user prompt, they took control of
    # the LLM contract — including the implied label vocabulary. Don't
    # apply the classification vocab filter on parse; the user can
    # introduce new labels in their edit and we'd silently drop the
    # rows otherwise, which would be very confusing UX for an
    # advanced user who just edited the prompt.
    parse_known_labels = [] if used_user_override else known_labels

    # Generous max_tokens for grounded calls: each row carries Q + A
    # + rationale + source_excerpt (4 fields × ~100 chars × 10 rows
    # ≈ 1000 tokens output minimum), AND reasoning-style models
    # (deepseek-reasoner / o-series / claude extended thinking) can
    # spend 3-8K tokens on <think> preambles before emitting any
    # user-facing JSON. The previous default (4096) silently
    # truncated grounded responses on those models, leaving the
    # parser to fail on broken JSON with no obvious cause.
    #
    # Cost is still bounded by chunk-char cap on the prompt side
    # (max ~2K reference tokens) + the per-call ceiling the cost
    # estimate badge surfaces. Worst-case 50-row grounded sonnet
    # call now ~25¢ instead of ~13¢ — still well within the "do
    # not shoot" guard for an opt-in user action.
    grounded_max_tokens = 8192 if reference_chunks else 4096

    if provider == "openai":
        response = await call_openai_chat(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_url=api_url,
            max_tokens=grounded_max_tokens,
        )
    elif provider == "anthropic":
        response = await call_anthropic_chat(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=grounded_max_tokens,
        )
    else:
        raise GoldGenerationError(
            "PROVIDER_NOT_SUPPORTED",
            f"Provider '{provider}' is not supported. Use 'openai' or 'anthropic'.",
        )

    rows = _parse_rows_for_recipe(
        recipe_id=recipe_id,
        content=response.content,
        expected_count=effective_count,
        known_labels=parse_known_labels,
    )

    return GenerationResult(
        rows=rows,
        recipe_id=recipe_id,
        provider=provider,
        model=response.model,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        prompt_preview=user_prompt[:500],
        reference_chunk_count=len(reference_chunks),
        estimated_cost_usd=compute_estimated_cost_usd(
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Per-recipe dispatch tables + helpers
# ─────────────────────────────────────────────────────────────────────


_SYSTEM_PROMPTS_BY_RECIPE: dict[str, str] = {
    "qa-sft": _SYSTEM_PROMPT_QA,
    "classification": _SYSTEM_PROMPT_CLASSIFICATION,
    "span-extraction": _SYSTEM_PROMPT_SPAN,
    "summarization": _SYSTEM_PROMPT_SUMMARIZATION,
}


def _build_prompt_for_recipe(
    *,
    recipe_id: str,
    project: Project,
    blueprint: DomainBlueprintRevision | None,
    count: int,
    focus_hint: str,
    reference_chunks: list[ReferenceChunk] | None,
    known_labels: list[str],
    distribution: tuple[int, int, int, int] | None = None,
) -> str:
    """Route to the per-recipe prompt builder. Caller guarantees the
    recipe is in ``SUPPORTED_RECIPES`` (validated upstream). The
    ``distribution`` tuple ``(easy, medium, hard, traps)`` is honored
    only by the qa-sft builder; other recipes ignore it."""
    if recipe_id == "qa-sft":
        return _build_qa_prompt(
            project=project,
            blueprint=blueprint,
            count=count,
            focus_hint=focus_hint,
            reference_chunks=reference_chunks,
            distribution=distribution,
        )
    if recipe_id == "classification":
        return _build_classification_prompt(
            project=project,
            blueprint=blueprint,
            count=count,
            focus_hint=focus_hint,
            reference_chunks=reference_chunks,
            known_labels=known_labels,
        )
    if recipe_id == "span-extraction":
        return _build_span_prompt(
            project=project,
            blueprint=blueprint,
            count=count,
            focus_hint=focus_hint,
            reference_chunks=reference_chunks,
        )
    if recipe_id == "summarization":
        return _build_summarization_prompt(
            project=project,
            blueprint=blueprint,
            count=count,
            focus_hint=focus_hint,
            reference_chunks=reference_chunks,
        )
    raise GoldGenerationError(
        "RECIPE_NOT_SUPPORTED",
        f"No prompt builder registered for recipe '{recipe_id}'.",
    )


def _parse_rows_for_recipe(
    *,
    recipe_id: str,
    content: str,
    expected_count: int,
    known_labels: list[str],
) -> list[dict[str, Any]]:
    """Route to the per-recipe parser. qa-sft returns ``GeneratedQa``
    dataclasses internally — we convert them to dicts here so the API
    surface is uniform across recipes."""
    if recipe_id == "qa-sft":
        qa_rows = _parse_qa_payload(content, expected_count)
        return [
            {
                "question": r.question,
                "answer": r.answer,
                "rationale": r.rationale,
                "source_excerpt": r.source_excerpt,
                "difficulty": r.difficulty,
                "is_hallucination_trap": r.is_hallucination_trap,
            }
            for r in qa_rows
        ]
    if recipe_id == "classification":
        return _parse_classification_rows(
            content, expected_count, known_labels=known_labels,
        )
    if recipe_id == "span-extraction":
        return _parse_span_rows(content, expected_count)
    if recipe_id == "summarization":
        return _parse_summarization_rows(content, expected_count)
    raise GoldGenerationError(
        "RECIPE_NOT_SUPPORTED",
        f"No parser registered for recipe '{recipe_id}'.",
    )


async def _load_known_classification_labels(project_id: int) -> list[str]:
    """Read any existing gold rows for the project + extract labels.
    Best-effort — returns ``[]`` when no gold rows exist yet. Done as
    a soft read so a fresh project doesn't fail; the prompt has a
    fallback path that asks the LLM to pick labels."""
    from app.services.trainability_forecast_service import (
        _extract_classification_labels,
    )

    # Both gold_dev + gold_test on disk. Use the JSONL files
    # directly (gold_service reads + writes the same shape so this
    # is cheap + side-effect-free).
    base = settings.DATA_DIR / "projects" / str(project_id) / "gold"
    rows: list[dict[str, Any]] = []
    for fname in ("gold_dev.jsonl", "gold_test.jsonl"):
        path = base / fname
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
        except OSError:
            continue
    labels = _extract_classification_labels(rows)
    # De-dup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        key = label.strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        out.append(key)
    return out
