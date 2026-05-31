"""Gold evaluation dataset API routes."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.dataset import DatasetType
from app.models.secret import ProjectSecret
from app.services.cloud_llm_service import CloudLlmError
from app.services.gold_llm_service import (
    GoldGenerationError,
    MAX_REFERENCE_CHUNKS,
    MAX_REFERENCE_TOTAL_CHARS,
    build_prompt_preview,
    estimate_call_cost_usd,
    generate_gold_qa_via_llm,
)
from app.services.synthetic_service import _load_project_cleaned_chunks
from app.services.gold_service import (
    add_gold_row,
    get_gold_entries,
    import_qa_pairs,
    lock_gold_dataset,
)
from app.services.gold_set_diagnostics_service import (
    compute_gold_set_diagnostics,
)
from app.services.secret_service import (
    delete_project_secret,
    get_project_secret_value,
    serialize_secret,
    upsert_project_secret,
)

router = APIRouter(prefix="/projects/{project_id}/gold", tags=["Gold Dataset"])


# Project-secret coordinates for the per-project API keys. Each
# panel-supported provider has its own (provider, key_name) tuple —
# Deepseek is treated as first-class even though the on-wire
# generate-via-llm payload still piggybacks on provider=openai +
# api_url=<deepseek host>. Keeping a separate secret coordinate lets
# users store all three keys side-by-side without one overwriting
# the others.
_OPENAI_SECRET = ("cloud_llm_openai", "api_key")
_ANTHROPIC_SECRET = ("cloud_llm_anthropic", "api_key")
_DEEPSEEK_SECRET = ("cloud_llm_deepseek", "api_key")

# UI-level provider tag → (secret_provider, key_name) for the saved-
# key endpoints + generate-via-llm fallback lookup.
_PROVIDER_SECRET_MAP: dict[str, tuple[str, str]] = {
    "openai": _OPENAI_SECRET,
    "anthropic": _ANTHROPIC_SECRET,
    "deepseek": _DEEPSEEK_SECRET,
}


def _resolve_generate_secret_coords(
    provider: str, api_url: str | None,
) -> tuple[str, str]:
    """Pick which stored secret to consult for a generate-via-llm fall-
    back. Deepseek's API is OpenAI-compatible so it arrives on the
    wire as provider=openai + api_url=<deepseek host>; the api_url
    is the only signal we have to route the lookup at the deepseek
    secret instead of the openai one."""
    if api_url and "deepseek" in api_url.lower():
        return _DEEPSEEK_SECRET
    if provider == "anthropic":
        return _ANTHROPIC_SECRET
    return _OPENAI_SECRET


async def _fetch_secret_obj(
    db: AsyncSession,
    project_id: int,
    provider: str,
    key_name: str,
) -> ProjectSecret | None:
    """Return the raw ProjectSecret row for hint extraction (NEVER
    decrypts the value)."""
    result = await db.execute(
        select(ProjectSecret).where(
            ProjectSecret.project_id == project_id,
            ProjectSecret.provider == provider,
            ProjectSecret.key_name == key_name,
        ).order_by(
            ProjectSecret.updated_at.desc(),
            ProjectSecret.id.desc(),
        ).limit(1)
    )
    return result.scalar_one_or_none()


class GoldRowCreate(BaseModel):
    """Generic per-recipe gold-row creator. Required fields differ
    by recipe and are validated here (one of qa-sft's question+answer,
    classification's text+label, span-extraction's text+entities,
    summarization's document+summary). Extra recipe-shaped fields
    are preserved verbatim into the JSONL via Pydantic's ``extra=allow``.

    The legacy ``{question, answer, ...}`` qa-sft wire shape still
    works — extras carry the recipe-specific keys when the panel
    sends classification / span / summarization rows."""
    model_config = ConfigDict(extra="allow")

    dataset_type: str = "gold_dev"
    difficulty: str = "medium"
    criticality: str = "normal"
    is_hallucination_trap: bool = False


class QAPairBatchImport(BaseModel):
    pairs: list[dict]
    dataset_type: str = "gold_dev"


def _validate_row_shape(payload: dict) -> None:
    """Reject obviously empty rows early. Service-level recipe-shape
    validation could go further (e.g. require question+answer for
    qa-sft) but the JSONL is recipe-agnostic on disk + the evaluator
    tolerates aliased field names, so a soft "at least one non-system
    field is non-empty" check is the right floor."""
    system_keys = {
        "dataset_type", "difficulty", "criticality",
        "is_hallucination_trap", "metadata",
    }
    has_content = any(
        k not in system_keys
        and v is not None
        and (not isinstance(v, str) or v.strip())
        for k, v in payload.items()
    )
    if not has_content:
        raise HTTPException(
            400,
            detail={
                "error_code": "EMPTY_GOLD_ROW",
                "message": (
                    "Gold row has no recipe-shaped content — provide at "
                    "least one of: question+answer (qa-sft), text+label "
                    "(classification), text+entities (span-extraction), "
                    "or document+summary (summarization)."
                ),
            },
        )


@router.post("/add", status_code=201)
async def add_pair(
    project_id: int,
    data: GoldRowCreate,
    db: AsyncSession = Depends(get_db),
):
    """Add a single gold row in any recipe shape (qa-sft Q+A,
    classification text+label, span-extraction text+entities,
    summarization document+summary). Recipe-shaped extras flow
    through ``extra=allow`` and round-trip into the JSONL."""
    payload = data.model_dump()
    dataset_type_str = payload.pop("dataset_type", "gold_dev")
    ds_type = (
        DatasetType.GOLD_DEV
        if dataset_type_str == "gold_dev"
        else DatasetType.GOLD_TEST
    )
    _validate_row_shape(payload)
    try:
        entry = await add_gold_row(db, project_id, payload, ds_type)
        return entry
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/import")
async def import_pairs(
    project_id: int,
    data: QAPairBatchImport,
    db: AsyncSession = Depends(get_db),
):
    """Import multiple Q&A pairs."""
    ds_type = DatasetType.GOLD_DEV if data.dataset_type == "gold_dev" else DatasetType.GOLD_TEST
    try:
        result = await import_qa_pairs(db, project_id, data.pairs, ds_type)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


# ── LLM-assisted generation ──────────────────────────────────────────


class RowDistribution(BaseModel):
    """Explicit mix of rows by difficulty + hallucination-trap. When
    set on a generate request, the four counts add up to the actual
    row count and override the top-level ``count`` field. Only
    honored for qa-sft (the gold-row schema's ``difficulty`` /
    ``is_hallucination_trap`` fields are QA-flavored)."""
    easy: int = Field(default=0, ge=0, le=50)
    medium: int = Field(default=0, ge=0, le=50)
    hard: int = Field(default=0, ge=0, le=50)
    hallucination_traps: int = Field(default=0, ge=0, le=50)

    @property
    def total(self) -> int:
        return self.easy + self.medium + self.hard + self.hallucination_traps


class GenerateViaLlmRequest(BaseModel):
    provider: Literal["openai", "anthropic"] = Field(
        ...,
        description="Cloud LLM provider. 'openai' also covers Deepseek + "
                    "any OpenAI-compatible endpoint via the optional api_url.",
    )
    model: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Model name (e.g. 'gpt-4o-mini', 'claude-haiku-4-5-20251001').",
    )
    count: int = Field(default=10, ge=1, le=50)
    focus_hint: str = Field(
        default="",
        max_length=1000,
        description="Optional user direction — 'focus on edge cases', "
                    "'cover refund policy questions', etc.",
    )
    api_key: str | None = Field(
        default=None,
        max_length=200,
        description="Inline API key (one-shot, not persisted). When omitted, "
                    "falls back to the stored project secret for this provider.",
    )
    api_url: str | None = Field(
        default=None,
        max_length=300,
        description="Override for OpenAI-compatible endpoints "
                    "(e.g. Deepseek). Ignored for Anthropic.",
    )
    ground_in_source: bool = Field(
        default=True,
        description="When True (recommended), include a strict-budget "
                    "sample of the project's cleaned chunks in the "
                    "prompt and ask the LLM to ground each answer in "
                    "them. Falls back to ungrounded when the project "
                    "hasn't imported any data yet.",
    )
    user_prompt_override: str | None = Field(
        default=None,
        max_length=200_000,
        description="Optional verbatim replacement for the user prompt. "
                    "When set, the service skips its own prompt "
                    "construction (project/blueprint/grounding/labels) "
                    "and sends this string instead. Powers the "
                    "panel's 'review & edit prompt before sending' UX. "
                    "Triggers a second behavior change: the classification "
                    "vocab filter on parse is suspended (the user took "
                    "control of the LLM contract, so we trust their "
                    "label space).",
    )
    system_prompt_override: str | None = Field(
        default=None,
        max_length=10_000,
        description="Optional verbatim replacement for the system prompt. "
                    "Independent of user_prompt_override — caller can "
                    "override one, both, or neither.",
    )
    distribution: RowDistribution | None = Field(
        default=None,
        description="Optional explicit row-count distribution by "
                    "difficulty + hallucination-trap. When set, the "
                    "sum replaces ``count`` and the prompt enumerates "
                    "the mix to the LLM. qa-sft only — silently ignored "
                    "for other recipes (their gold_template lacks "
                    "difficulty / trap fields).",
    )


class PreviewPromptRequest(BaseModel):
    """Body for ``/generate-via-llm/preview-prompt``. Mirrors the
    generate request's prompt-affecting fields only — provider,
    model, api_key, api_url don't influence the prompt string, so
    they're omitted here to keep the surface tight."""
    count: int = Field(default=10, ge=1, le=50)
    focus_hint: str = Field(default="", max_length=1000)
    ground_in_source: bool = Field(default=True)
    distribution: RowDistribution | None = None


class CostEstimateRequest(BaseModel):
    provider: Literal["openai", "anthropic"]
    model: str = Field(..., min_length=1, max_length=128)
    count: int = Field(default=10, ge=1, le=50)
    ground_in_source: bool = Field(default=True)


@router.post("/generate-via-llm")
async def generate_via_llm(
    project_id: int,
    req: GenerateViaLlmRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate Q&A pairs via a flagship cloud LLM (OpenAI or Anthropic)
    using the project's domain blueprint + selected recipe as context.

    Returns the generated rows for **preview** — does NOT persist.
    The caller renders the rows for user review and writes accepted
    rows via the existing ``POST /gold/import`` endpoint.

    Status codes:
      * 200 — rows generated + returned in the payload.
      * 400 — structured error (RECIPE_REQUIRED, RECIPE_NOT_SUPPORTED,
        API_KEY_REQUIRED, COUNT_OUT_OF_RANGE, LLM_RESPONSE_UNPARSEABLE,
        LLM_RESPONSE_COUNT_MISMATCH).
      * 502 — upstream provider failure (rate limit, bad key, model
        not found). Body has the provider's error text.
    """
    # Resolve the API key: inline > stored project secret. Deepseek
    # rides on provider=openai + api_url=<deepseek host>, so the
    # resolver inspects api_url to pick the right secret coordinate.
    api_key = (req.api_key or "").strip()
    if not api_key:
        secret_provider, key_name = _resolve_generate_secret_coords(
            req.provider, req.api_url,
        )
        stored = await get_project_secret_value(
            db, project_id, secret_provider, key_name,
        )
        api_key = stored or ""
    if not api_key:
        raise HTTPException(
            400,
            detail={
                "error_code": "API_KEY_REQUIRED",
                "message": (
                    f"No {req.provider} API key found. Either include "
                    "``api_key`` in the request or store it in this "
                    "project's secrets first."
                ),
            },
        )

    # When a distribution is supplied, its total replaces ``count``.
    # Validate that sum is in the same [1, 50] range we apply to count.
    if req.distribution is not None:
        dist_total = req.distribution.total
        if dist_total < 1 or dist_total > 50:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "DISTRIBUTION_OUT_OF_RANGE",
                    "message": (
                        "Distribution counts must sum to between 1 and 50 "
                        f"(got {dist_total}). Adjust easy/medium/hard/"
                        "hallucination_traps so the total stays in range."
                    ),
                },
            )
    distribution_tuple: tuple[int, int, int, int] | None = (
        None
        if req.distribution is None
        else (
            req.distribution.easy,
            req.distribution.medium,
            req.distribution.hard,
            req.distribution.hallucination_traps,
        )
    )

    try:
        result = await generate_gold_qa_via_llm(
            db,
            project_id=project_id,
            provider=req.provider,
            model=req.model,
            api_key=api_key,
            count=req.count,
            focus_hint=req.focus_hint,
            api_url=req.api_url,
            ground_in_source=req.ground_in_source,
            user_prompt_override=req.user_prompt_override,
            system_prompt_override=req.system_prompt_override,
            distribution=distribution_tuple,
        )
    except GoldGenerationError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": exc.error_code, "message": str(exc)},
        )
    except CloudLlmError as exc:
        # Upstream LLM failure — surface as 502 (bad gateway) so the
        # UI can distinguish "you broke" (400) from "OpenAI broke" (502).
        raise HTTPException(status_code=502, detail=str(exc))

    # Rows are already per-recipe dicts from the service. The frontend
    # branches its render on ``recipe_id`` — qa-sft rows carry
    # `question`/`answer`, classification carries `text`/`label`, etc.
    return {
        "rows": list(result.rows),
        "recipe_id": result.recipe_id,
        "provider": result.provider,
        "model": result.model,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        },
        "prompt_preview": result.prompt_preview,
        "reference_chunk_count": result.reference_chunk_count,
        "estimated_cost_usd": result.estimated_cost_usd,
    }


@router.post("/generate-via-llm/preview-prompt")
async def generate_via_llm_preview_prompt(
    project_id: int,
    req: PreviewPromptRequest,
    db: AsyncSession = Depends(get_db),
):
    """Return the user + system prompts the service WOULD send to the
    LLM for this project / recipe / grounding combo, without actually
    calling the LLM. Powers the "review & edit prompt before sending"
    UX for advanced users.

    Shape-identical to what ``/generate-via-llm`` constructs internally,
    so the user can copy the returned strings into the override fields
    on the generate call after editing.

    Status codes:
      * 200 — preview returned in the payload.
      * 400 — same caller-fixable cases as the generate endpoint
        (RECIPE_REQUIRED, RECIPE_NOT_SUPPORTED, COUNT_OUT_OF_RANGE,
        PROJECT_NOT_FOUND).
    """
    if req.distribution is not None:
        dist_total = req.distribution.total
        if dist_total < 1 or dist_total > 50:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "DISTRIBUTION_OUT_OF_RANGE",
                    "message": (
                        "Distribution counts must sum to between 1 and 50 "
                        f"(got {dist_total})."
                    ),
                },
            )
    distribution_tuple: tuple[int, int, int, int] | None = (
        None
        if req.distribution is None
        else (
            req.distribution.easy,
            req.distribution.medium,
            req.distribution.hard,
            req.distribution.hallucination_traps,
        )
    )
    try:
        preview = await build_prompt_preview(
            db,
            project_id=project_id,
            count=req.count,
            focus_hint=req.focus_hint,
            ground_in_source=req.ground_in_source,
            distribution=distribution_tuple,
        )
    except GoldGenerationError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": exc.error_code, "message": str(exc)},
        )
    return {
        "recipe_id": preview.recipe_id,
        "system_prompt": preview.system_prompt,
        "user_prompt": preview.user_prompt,
        "reference_chunk_count": preview.reference_chunk_count,
        "known_labels": preview.known_labels,
    }


@router.post("/generate-via-llm/cost-estimate")
async def generate_via_llm_cost_estimate(
    project_id: int,
    req: CostEstimateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Pre-call cost estimate for the LLM-assisted generate flow —
    powers the "≈ $X.YY" badge next to the Generate button so the
    user has price awareness before clicking.

    Cheap: no provider calls. Counts characters in the project's
    cleaned chunks (capped at the same budget the real call uses)
    and prices the resulting prompt + projected completion against
    the model's per-token price. Approximate (≈ ±25%) — actual cost
    surfaces in the response after the real call lands.
    """
    chars = 0
    chunk_count = 0
    if req.ground_in_source:
        pool = await _load_project_cleaned_chunks(project_id)
        # Mirror the sampler's per-chunk + total budgets exactly so
        # the estimate matches the actual call's reference-material
        # size, not the raw pool's.
        from app.services.gold_llm_service import (
            MAX_CHARS_PER_REFERENCE_CHUNK,
        )
        capped = min(len(pool), MAX_REFERENCE_CHUNKS)
        for text in pool[:capped]:
            chars_for_chunk = min(
                len(text or ""),
                MAX_CHARS_PER_REFERENCE_CHUNK,
            )
            if chars + chars_for_chunk > MAX_REFERENCE_TOTAL_CHARS:
                chars_for_chunk = max(0, MAX_REFERENCE_TOTAL_CHARS - chars)
                if chars_for_chunk == 0:
                    break
            chars += chars_for_chunk
            chunk_count += 1
            if chars >= MAX_REFERENCE_TOTAL_CHARS:
                break

    estimate = estimate_call_cost_usd(
        model=req.model,
        count=req.count,
        grounded=req.ground_in_source and chunk_count > 0,
        reference_chunk_count=chunk_count,
        reference_total_chars=chars,
    )
    return {
        "provider": req.provider,
        "model": req.model,
        "count": req.count,
        "ground_in_source_requested": req.ground_in_source,
        "ground_in_source_effective": req.ground_in_source and chunk_count > 0,
        **estimate,
    }


# ── Saved API key (panel-local UX) ────────────────────────────────────
#
# The LLM gold-generation panel asks for an API key on every Generate
# click today. These endpoints let the panel save / look up / clear
# the key per project + provider so the user only pastes once.
#
# Wire shape is deliberately minimal: only a masked hint round-trips
# back to the UI; the raw value never leaves the backend after write.


class SavedKeyResponse(BaseModel):
    has_stored_key: bool
    value_hint: str | None = None


class SavedKeyUpsert(BaseModel):
    provider: Literal["openai", "anthropic", "deepseek"]
    api_key: str = Field(
        ...,
        min_length=8,
        max_length=200,
        description="Raw API key. Validated as non-empty + reasonable "
                    "length here so the panel can't silently overwrite "
                    "a real key with a typo'd stub.",
    )


@router.get("/generate-via-llm/saved-key", response_model=SavedKeyResponse)
async def get_saved_llm_key(
    project_id: int,
    provider: Literal["openai", "anthropic", "deepseek"],
    db: AsyncSession = Depends(get_db),
):
    """Report whether a stored API key exists for this project +
    provider, returning only the masked hint (never the raw value)."""
    secret_provider, key_name = _PROVIDER_SECRET_MAP[provider]
    secret_obj = await _fetch_secret_obj(
        db, project_id, secret_provider, key_name,
    )
    if secret_obj is None:
        return SavedKeyResponse(has_stored_key=False, value_hint=None)
    serialized = serialize_secret(secret_obj)
    return SavedKeyResponse(
        has_stored_key=True, value_hint=serialized.get("value_hint"),
    )


@router.put("/generate-via-llm/saved-key", response_model=SavedKeyResponse)
async def put_saved_llm_key(
    project_id: int,
    req: SavedKeyUpsert,
    db: AsyncSession = Depends(get_db),
):
    """Store (or replace) the API key for this project + provider.
    Only the masked hint is returned — the raw value stays server-side."""
    secret_provider, key_name = _PROVIDER_SECRET_MAP[req.provider]
    try:
        secret_obj = await upsert_project_secret(
            db=db,
            project_id=project_id,
            provider=secret_provider,
            key_name=key_name,
            value=req.api_key,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    serialized = serialize_secret(secret_obj)
    return SavedKeyResponse(
        has_stored_key=True, value_hint=serialized.get("value_hint"),
    )


@router.delete("/generate-via-llm/saved-key", status_code=204)
async def delete_saved_llm_key(
    project_id: int,
    provider: Literal["openai", "anthropic", "deepseek"],
    db: AsyncSession = Depends(get_db),
):
    """Clear the stored API key for this project + provider. Idempotent
    — returns 204 whether or not a row existed, so the panel's
    "Remove" button never lies after a stale cache."""
    secret_provider, key_name = _PROVIDER_SECRET_MAP[provider]
    await delete_project_secret(db, project_id, secret_provider, key_name)
    return Response(status_code=204)


@router.get("/entries")
async def list_entries(
    project_id: int,
    dataset_type: str = "gold_dev",
    db: AsyncSession = Depends(get_db),
):
    """List all gold dataset entries."""
    ds_type = DatasetType.GOLD_DEV if dataset_type == "gold_dev" else DatasetType.GOLD_TEST
    entries = await get_gold_entries(db, project_id, ds_type)
    return {"entries": entries, "total": len(entries)}


@router.post("/lock")
async def lock_dataset(
    project_id: int,
    dataset_type: str = "gold_dev",
    db: AsyncSession = Depends(get_db),
):
    """Lock a gold dataset (make immutable)."""
    ds_type = DatasetType.GOLD_DEV if dataset_type == "gold_dev" else DatasetType.GOLD_TEST
    ds = await lock_gold_dataset(db, project_id, ds_type)
    return {"id": ds.id, "name": ds.name, "is_locked": ds.is_locked, "record_count": ds.record_count}


@router.get("/diagnostics")
async def gold_set_diagnostics(
    project_id: int,
    sample_per_class: int = 12,
    db: AsyncSession = Depends(get_db),
):
    """V4 ML-native viz — class-balance bars + class-similarity heatmap.

    Returns a single payload the ``GoldSetDiagnosticsPanel`` consumes:
    ``{class_balance: {labels, total, entropy_nats}, similarity:
    {labels, matrix, insufficient_labels}, total_rows,
    classification_eligible}``. ``classification_eligible=false`` for
    non-classification recipes (the UI renders an empty-state hint
    rather than a misleading single-class chart).
    """
    if sample_per_class < 2 or sample_per_class > 100:
        raise HTTPException(400, "sample_per_class must be in [2, 100]")
    return await compute_gold_set_diagnostics(
        db,
        project_id,
        sample_per_class=sample_per_class,
    )
