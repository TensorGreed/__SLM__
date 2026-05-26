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

from dataclasses import dataclass
from typing import Literal

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


Provider = Literal["openai", "anthropic"]


@dataclass
class GeneratedQa:
    question: str
    answer: str
    rationale: str = ""


@dataclass
class GenerationResult:
    rows: list[GeneratedQa]
    provider: Provider
    model: str
    prompt_tokens: int
    completion_tokens: int
    prompt_preview: str


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


def _build_user_prompt(
    *,
    project: Project,
    blueprint: DomainBlueprintRevision | None,
    count: int,
    focus_hint: str,
) -> str:
    """Compose the user-facing prompt. Pulls project / blueprint
    context inline so the LLM produces domain-relevant rows."""
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

    return (
        f"{context_block}\n\n"
        f"Generate exactly {count} question/answer pairs for the project above.\n\n"
        "Output rules:\n"
        "- Return ONLY valid JSON. No markdown, no code fences, no commentary.\n"
        '- Shape: {"pairs": [{"question": "...", "answer": "...", '
        '"rationale": "..."}]}\n'
        "- Each question MUST be specific to this project's domain.\n"
        "- Each answer MUST be self-contained and factually defensible.\n"
        "- Vary difficulty: mix factual lookups, edge cases, and "
        "judgment calls.\n"
        "- ``rationale`` is optional but helpful — one short sentence on "
        "why this is the right answer (used by some evaluators).\n"
        f"- Return EXACTLY {count} pairs in the ``pairs`` array."
    )


# ─────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────


def _parse_qa_payload(content: str, expected_count: int) -> list[GeneratedQa]:
    """Pull ``GeneratedQa`` rows out of the LLM response. Tolerant
    of two top-level shapes:
      * ``{"pairs": [{"question": ..., "answer": ...}, ...]}``
      * ``[{"question": ..., "answer": ...}, ...]``
    Rejects rows missing question or answer; doesn't enforce the
    expected_count strictly (LLMs sometimes return N±1)."""
    try:
        payload = extract_json_payload(content)
    except ValueError as exc:
        # Re-raise as structured GoldGenerationError so the API
        # layer returns 400 with LLM_RESPONSE_UNPARSEABLE instead
        # of a generic 500.
        raise GoldGenerationError(
            "LLM_RESPONSE_UNPARSEABLE",
            f"LLM response was not parseable JSON: {exc}",
        ) from exc

    items: list = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("pairs", "qa_pairs", "items", "data"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        else:
            # Single-pair object shape — wrap it.
            if isinstance(payload.get("question"), str) and isinstance(
                payload.get("answer"), str,
            ):
                items = [payload]

    rows: list[GeneratedQa] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question") or "").strip()
        a = str(item.get("answer") or "").strip()
        r = str(item.get("rationale") or "").strip()
        if not q or not a:
            continue
        rows.append(GeneratedQa(question=q, answer=a, rationale=r))

    if not rows:
        raise GoldGenerationError(
            "LLM_RESPONSE_UNPARSEABLE",
            "The LLM response did not contain any parseable question/answer pairs. "
            "Try a different model or a clearer focus hint.",
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
) -> GenerationResult:
    """Generate ``count`` Q&A pairs for the project using the named
    cloud provider. Returns the parsed rows for preview — does NOT
    persist. The caller saves accepted rows via the existing
    ``/gold/import`` endpoint.

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

    user_prompt = _build_user_prompt(
        project=project,
        blueprint=blueprint,
        count=count,
        focus_hint=focus_hint,
    )

    if provider == "openai":
        response = await call_openai_chat(
            api_key=api_key,
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            api_url=api_url,
        )
    elif provider == "anthropic":
        response = await call_anthropic_chat(
            api_key=api_key,
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
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
    )
