"""Natural Language to Pipeline (Magic Create) service."""

from __future__ import annotations

import json
import re
from typing import Any

from app.services.synthetic_service import call_teacher_model


def _extract_vram_gb(prompt: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*gb", str(prompt).lower())
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _fallback_magic_recommendation(prompt: str) -> dict[str, Any]:
    token = str(prompt or "").strip()
    lower = token.lower()
    vram_gb = _extract_vram_gb(lower)

    adapter_id = "default-canonical"
    task_profile = "instruction_sft"
    recipe_id = "recipe.pipeline.sft_default"
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    if any(kw in lower for kw in ("extract", "extraction", "entity", "json", "liabilit", "field")):
        adapter_id = "structured-extraction"
        task_profile = "structured_extraction"
    elif any(kw in lower for kw in ("rag", "grounded", "retrieval", "context")):
        adapter_id = "rag-grounded"
        task_profile = "rag_qa"
    elif any(kw in lower for kw in ("tool", "function call", "function-call")):
        adapter_id = "tool-call-json"
        task_profile = "tool_calling"
    elif any(kw in lower for kw in ("summarize", "summary", "summarization")):
        task_profile = "summarization"

    if vram_gb is not None:
        if vram_gb <= 8:
            base_model_name = "Qwen/Qwen1.5-1.8B-Chat"
            recipe_id = "recipe.pipeline.lora_fast"
        elif vram_gb <= 16:
            base_model_name = "microsoft/phi-2"
            recipe_id = "recipe.pipeline.lora_fast"
        else:
            base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            recipe_id = "recipe.pipeline.sft_default"
    elif any(kw in lower for kw in ("raspberry", "mobile", "iphone", "edge", "low memory")):
        base_model_name = "Qwen/Qwen1.5-1.8B-Chat"
        recipe_id = "recipe.pipeline.lora_fast"
    elif any(kw in lower for kw in ("3090", "4090", "24gb", "a100", "h100", "80gb", "server")):
        base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        recipe_id = "recipe.pipeline.sft_default"

    clean_words = [item for item in re.split(r"[^a-zA-Z0-9]+", token) if item]
    project_hint = " ".join(clean_words[:4]).strip() or "Magic Project"
    project_name = project_hint[:64]

    return {
        "project_name": project_name,
        "project_description": f"Generated from prompt: {token[:180]}",
        "domain_pack_id": "general-pack-v1",
        "adapter_id": adapter_id,
        "task_profile": task_profile,
        "base_model_name": base_model_name,
        "pipeline_recipe_id": recipe_id,
    }


async def magic_create_pipeline_recipe(
    prompt: str,
    *,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    """Parse a natural language prompt into a pipeline recipe recommendation."""
    system_prompt = (
        "You are an expert ML architect. The user will describe a dataset or a model they want to build. "
        "Your job is to recommend the best pipeline configuration for this task.\n\n"
        "Available Data Adapters: 'default-canonical' (chat/instruct), 'structured-extraction' (information extraction), "
        "'tool-call-json' (function calling), 'rag-grounded' (Q&A with context).\n"
        "Available Task Profiles: 'instruction_sft', 'chat_sft', 'qa', 'rag_qa', 'tool_calling', "
        "'structured_extraction', 'summarization', 'seq2seq', 'classification', 'preference'.\n"
        "Available Base Models: 'meta-llama/Meta-Llama-3-8B-Instruct', 'Qwen/Qwen1.5-1.8B-Chat', 'microsoft/phi-2'.\n\n"
        "Return ONLY a JSON object with exactly these fields:\n"
        "- 'project_name' (string: short, descriptive name)\n"
        "- 'project_description' (string: 1 sentence summary)\n"
        "- 'domain_pack_id' (string: always use 'general-pack-v1')\n"
        "- 'adapter_id' (string: from the available list)\n"
        "- 'task_profile' (string: from the available list)\n"
        "- 'base_model_name' (string: from the available list)\n"
        "- 'pipeline_recipe_id' (string: e.g. 'recipe.pipeline.lora_fast' or 'recipe.pipeline.sft_default')\n"
    )

    try:
        result = await call_teacher_model(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=600,
            temperature=0.2,
        )
    except Exception:
        if allow_fallback:
            return _fallback_magic_recommendation(prompt)
        raise
    content = str(result.get("content", "")).strip()

    if "```json" in content:
        content = content.split("```json")[-1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[-1].split("```")[0].strip()

    try:
        config = json.loads(content)
    except json.JSONDecodeError as exc:
        if allow_fallback:
            return _fallback_magic_recommendation(prompt)
        raise ValueError(f"Teacher model returned invalid JSON for Magic Create: {content}") from exc

    if not isinstance(config, dict):
        if allow_fallback:
            return _fallback_magic_recommendation(prompt)
        raise ValueError("Teacher model returned non-object JSON for Magic Create.")

    fallback = _fallback_magic_recommendation(prompt)
    for key, value in fallback.items():
        config.setdefault(key, value)
    return config
