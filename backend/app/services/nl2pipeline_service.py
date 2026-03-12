"""Natural Language to Pipeline (Magic Create) service."""

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.synthetic_service import call_teacher_model


async def magic_create_pipeline_recipe(prompt: str) -> dict[str, Any]:
    """Parse a natural language prompt into a Pipeline Recipe schema recommendation."""

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

    result = await call_teacher_model(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=600,
        temperature=0.2
    )

    content = result.get("content", "").strip()

    if "```json" in content:
        content = content.split("```json")[-1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[-1].split("```")[0].strip()

    try:
        config = json.loads(content)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Teacher model returned invalid JSON for Magic Create: {content}") from e
