"""LLM-assisted gold-set generation for qa-sft projects.

Crafts a project-aware prompt, calls a flagship cloud LLM (OpenAI or
Anthropic), parses the response into Q&A pairs, and returns them to
the caller for **preview-then-save** review. This service does NOT
persist anything — the API endpoint shows the rows to the user, the
user picks which to keep, and the existing
``POST /api/projects/{id}/gold/import`` writes the accepted subset.

Scope:
  * v1 supports qa-sft only. Recipe-shape check happens before the
    LLM call — projects with a non-qa-sft recipe are rejected with a
    structured error so the frontend can surface a useful message.
  * Providers: OpenAI (incl. Deepseek + custom OpenAI-compatible
    endpoints) and Anthropic. See ``cloud_llm_service``.
  * Prompt incorporates the active domain blueprint's
    ``problem_statement`` + ``brief_text`` + ``domain_name`` when
    present so generated Q&A reflects the actual project, not
    generic boilerplate.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Literal


_LOG = logging.getLogger("gold_llm")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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


@dataclass
class GeneratedQa:
    question: str
    answer: str
    rationale: str = ""
    # When grounding is on, the LLM is asked to include the source
    # excerpt that backs each answer. Empty string when grounding
    # was off OR when the LLM omitted the field.
    source_excerpt: str = ""


@dataclass
class ReferenceChunk:
    text: str
    source_label: str  # short tag like "doc-3/chunk-12" so the user can locate

    def to_prompt_block(self, idx: int) -> str:
        return f"[REF-{idx}] ({self.source_label})\n{self.text}"


@dataclass
class GenerationResult:
    rows: list[GeneratedQa]
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


_SYSTEM_PROMPT = (
    "You generate gold-standard question/answer pairs for fine-tuning a "
    "small language model. The pairs you generate become the EVALUATION "
    "ground truth for the model — quality + correctness matter more than "
    "volume. Avoid trivia; favor questions a real user of this product "
    "would ask. Ground every answer in plausible domain knowledge."
)


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


def _build_user_prompt(
    *,
    project: Project,
    blueprint: DomainBlueprintRevision | None,
    count: int,
    focus_hint: str,
    reference_chunks: list[ReferenceChunk] | None = None,
) -> str:
    """Compose the user-facing prompt. Pulls project / blueprint
    context inline so the LLM produces domain-relevant rows.

    When ``reference_chunks`` is non-empty, the prompt instructs the
    LLM to GROUND each answer in one of the provided excerpts and
    include a ``source_excerpt`` field per row pointing at the
    supporting passage. Otherwise the prompt asks for general
    domain-relevant Q&A (legacy v1 behavior)."""
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

    context_block = "\n".join(sections) if sections else "(no project context available)"

    grounded = bool(reference_chunks)
    parts: list[str] = [context_block]

    if grounded:
        ref_blocks = "\n\n".join(
            chunk.to_prompt_block(i + 1)
            for i, chunk in enumerate(reference_chunks or [])
        )
        parts.append(
            "REFERENCE MATERIAL (the model being trained will only learn "
            "from material like this — answers MUST be grounded in these "
            "excerpts):\n\n" + ref_blocks,
        )

    parts.append(f"Generate exactly {count} question/answer pairs for the project above.")

    rules = [
        "Return ONLY valid JSON. No markdown, no code fences, no commentary.",
        'Shape: {"pairs": [{"question": "...", "answer": "..."}]}',
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
        ])
    else:
        rules.append(
            "Each answer MUST be self-contained and factually defensible.",
        )

    rules.extend([
        "Each question MUST be specific to this project's domain.",
        "Vary difficulty: mix factual lookups, edge cases, and judgment calls.",
        "Optionally include a ``rationale`` field — one short sentence on "
        "why this is the right answer (used by some evaluators).",
        f"Return EXACTLY {count} pairs in the ``pairs`` array.",
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
        rows.append(
            GeneratedQa(question=q, answer=a, rationale=r, source_excerpt=src),
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


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


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
    if recipe_id != "qa-sft":
        raise GoldGenerationError(
            "RECIPE_NOT_SUPPORTED",
            f"LLM-assisted gold generation v1 only supports the "
            f"'qa-sft' recipe (project is using '{recipe_id}'). "
            "Classification + span-extraction will be added in a "
            "follow-up phase.",
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

    user_prompt = _build_user_prompt(
        project=project,
        blueprint=blueprint,
        count=count,
        focus_hint=focus_hint,
        reference_chunks=reference_chunks if reference_chunks else None,
    )

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
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            api_url=api_url,
            max_tokens=grounded_max_tokens,
        )
    elif provider == "anthropic":
        response = await call_anthropic_chat(
            api_key=api_key,
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=grounded_max_tokens,
        )
    else:
        raise GoldGenerationError(
            "PROVIDER_NOT_SUPPORTED",
            f"Provider '{provider}' is not supported. Use 'openai' or 'anthropic'.",
        )

    rows = _parse_qa_payload(response.content, count)

    return GenerationResult(
        rows=rows,
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
