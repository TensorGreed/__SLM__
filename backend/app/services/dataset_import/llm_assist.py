"""Optional LLM-assist for the schema introspector (Phase H).

When the column-content sniffer can't form a high-confidence
hypothesis, the introspector can fall back to asking the project's
teacher model: "given these column names + sample values, which
mapper fits?". The LLM's output is treated as a *proposal* exactly
like the deterministic sniffer's — never a silent action. The CLI's
``--auto`` / ``--force`` flow and the UI wizard's confidence gate
apply to LLM proposals the same way they apply to sniffer proposals.

Disabled by default behind ``settings.DATASET_IMPORT_LLM_ASSIST_ENABLED``
+ requires ``TEACHER_MODEL_API_URL`` to be set. Both gates fail
gracefully (returning ``None`` from :func:`llm_assisted_proposal`) so
callers can just unconditionally try LLM-assist when the user opts in
without checking the env first.

This module is the *only* place that knows how to call the teacher;
the introspector + service.introspect_locator pass through a single
opt-in flag.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.services.dataset_import.protocols import ProposedMapping
from app.services.dataset_import.registry import list_registered_mappers


# Cap on how many sample rows we send to the teacher. Higher
# = better context, more tokens. Twenty rows of a ~5kB record =
# ~100kB prompt, which is fine for a 32k-context model. Plenty
# of signal for the LLM to spot patterns.
LLM_ASSIST_SAMPLE_CAP: int = 20

# Cap on raw-text length per sample value when serializing. Long
# free-text fields explode the prompt; the teacher just needs a
# representative snippet.
_VALUE_TRUNCATE: int = 240


_SYSTEM_PROMPT = (
    "You are a senior ML data engineer triaging an unknown dataset. "
    "Your job: read the column names + sample rows and pick the "
    "BrewSLM target mapper that best fits. Mappers and their "
    "expectations:\n\n"
    "- bio_to_spans: BIO-tagged NER (tokens + labels columns).\n"
    "- label_to_classification: {text, label} sentiment / intent.\n"
    "- text_only: single text column, no labels (LM pretraining).\n"
    "- qa_pair_passthrough: {question, answer}.\n"
    "- chat_messages_passthrough: list of {role, content} dicts.\n"
    "- preference_pair: {prompt, chosen, rejected} (DPO / ORPO).\n"
    "- rag_passthrough: {question, context, answer} (grounded QA).\n"
    "- kv_to_structured: flat key-value extractions (invoices / forms).\n\n"
    "Respond with a single JSON object. No prose, no markdown. Schema:\n"
    "{\n"
    "  \"mapper_id\": str,\n"
    "  \"field_map\": object,\n"
    "  \"confidence\": float in [0,1],\n"
    "  \"rationale\": str\n"
    "}\n"
    "Set mapper_id to '' if no listed mapper fits."
)


def _truncate(value: Any) -> Any:
    """Trim long string values so the prompt stays compact."""

    if isinstance(value, str) and len(value) > _VALUE_TRUNCATE:
        return value[:_VALUE_TRUNCATE] + "…"
    if isinstance(value, (list, dict)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return str(value)[:_VALUE_TRUNCATE]
        if len(text) > _VALUE_TRUNCATE:
            return text[:_VALUE_TRUNCATE] + "…"
        return text
    return value


def _build_prompt(
    *, columns: list[str], sample_rows: list[dict[str, Any]]
) -> str:
    """Materialize the user-content prompt sent to the teacher."""

    payload = {
        "columns": columns,
        "registered_mappers": list_registered_mappers(),
        "sample_rows": [
            {key: _truncate(val) for key, val in row.items()}
            for row in sample_rows[:LLM_ASSIST_SAMPLE_CAP]
        ],
    }
    return (
        "Pick the best-fit mapper for this dataset. Reply with the "
        "JSON schema described in the system prompt — nothing else.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_response(raw: str) -> dict[str, Any] | None:
    """Pull a JSON object out of the teacher's response.

    The teacher is asked for raw JSON but real-world models still wrap
    it in markdown fences or prose. We attempt full-document parse
    first, then fall back to extracting the largest ``{...}`` block.
    """

    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    # Strip Markdown code fences ('```json ... ```' or '``` ... ```').
    if text.startswith("```"):
        text = text.lstrip("`").lstrip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if text.endswith("```"):
            text = text[: -3].rstrip()
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(raw)
    if match:
        try:
            loaded = json.loads(match.group(0))
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _normalize_proposal(payload: dict[str, Any]) -> ProposedMapping | None:
    """Validate the teacher's response + wrap it in a ProposedMapping.

    Rejects responses that name a mapper not in the registry (the
    teacher hallucinated) and clamps confidence to [0, 1]. The
    introspector still gates on the confidence threshold downstream.
    """

    raw_mapper = payload.get("mapper_id")
    if not isinstance(raw_mapper, str):
        return None
    mapper_id = raw_mapper.strip()
    if not mapper_id:
        return None
    if mapper_id not in list_registered_mappers():
        return None
    field_map_raw = payload.get("field_map")
    if isinstance(field_map_raw, dict):
        field_map = dict(field_map_raw)
    else:
        field_map = {}
    raw_confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    rationale_raw = payload.get("rationale")
    rationale = (
        rationale_raw.strip()
        if isinstance(rationale_raw, str) and rationale_raw.strip()
        else "LLM-assisted proposal (no rationale supplied)"
    )

    # Resolve the target_task_profile from the chosen mapper rather
    # than trusting the LLM — keeps the field-map → handler mapping
    # consistent with the deterministic path.
    from app.services.dataset_import.registry import resolve_mapper

    try:
        target_profile = resolve_mapper(mapper_id).declared_target()
    except KeyError:
        return None

    return ProposedMapping(
        target_task_profile=target_profile,
        mapper_id=mapper_id,
        field_map=field_map,
        confidence=confidence,
        rationale=f"[llm-assist] {rationale}",
        warnings=["proposal-source: llm-assist"],
    )


async def llm_assisted_proposal(
    *,
    columns: list[str],
    sample_rows: list[dict[str, Any]],
) -> ProposedMapping | None:
    """Ask the teacher model for a mapping proposal.

    Returns ``None`` (never raises) when:
    - the LLM-assist setting is disabled,
    - no teacher API URL is configured,
    - the teacher call errors out,
    - or the response doesn't parse / names an unknown mapper.

    The deterministic sniffer's result is the source of truth; this
    function is the optional fallback.
    """

    from app.config import settings

    if not getattr(settings, "DATASET_IMPORT_LLM_ASSIST_ENABLED", False):
        return None
    if not getattr(settings, "TEACHER_MODEL_API_URL", ""):
        return None
    if not sample_rows:
        return None

    from app.services.synthetic_service import call_teacher_model

    try:
        response = await call_teacher_model(
            prompt=_build_prompt(columns=columns, sample_rows=sample_rows),
            system_prompt=_SYSTEM_PROMPT,
            force_json=True,
        )
    except Exception:  # noqa: BLE001
        # The introspector should still return whatever the sniffer
        # produced even when the teacher is unreachable. Swallowing
        # here keeps the deterministic path resilient.
        return None

    payload = _coerce_response(response.get("content", ""))
    if payload is None:
        return None
    return _normalize_proposal(payload)
