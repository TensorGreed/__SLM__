"""Synthetic data generation service — teacher model integration."""

import asyncio
import json
import random
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory
from app.models.dataset import Dataset, DatasetType, RawDocument


def _coerce_completion_content(raw: Any) -> str:
    """Normalize OpenAI-compatible message content to plain text."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(raw, dict):
        text = raw.get("text")
        if isinstance(text, str):
            return text
    return str(raw or "")


def _synthetic_dir(project_id: int) -> Path:
    d = settings.DATA_DIR / "projects" / str(project_id) / "synthetic"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def get_or_create_synthetic_dataset(
    db: AsyncSession, project_id: int
) -> Dataset:
    """Get or create the synthetic dataset for a project."""
    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.SYNTHETIC,
        )
    )
    ds = result.scalar_one_or_none()
    if ds:
        return ds

    ds = Dataset(
        project_id=project_id,
        name="Synthetic Dataset",
        dataset_type=DatasetType.SYNTHETIC,
        description="Teacher-generated synthetic instruction data",
    )
    db.add(ds)
    await db.flush()
    await db.refresh(ds)
    return ds


DEFAULT_TEACHER_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates high-quality training data. "
    "Respond directly with the requested output. Do not include reasoning, "
    "planning, or preamble before the answer — produce only the final result."
)


async def call_teacher_model(
    prompt: str,
    system_prompt: str = DEFAULT_TEACHER_SYSTEM_PROMPT,
    api_url: str = "",
    api_key: str = "",
    model_name: str = "llama3",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    force_json: bool = False,
) -> dict[str, Any]:
    """Call external teacher LLM API (OpenAI-compatible format).

    Design choices that matter for heterogeneous teacher backends:

    * Default ``max_tokens`` is 4096 so the response has room for whatever
      preamble a reasoning model emits (Qwen3 ``<think>``, Claude structured
      thinking, DeepSeek-R1 reflections) *and* the JSON payload. 1024 was
      enough for llama3 + OpenAI but silently starved reasoning models.
    * The default system prompt tells the model to skip reasoning preamble.
      Model-agnostic and harmless to plain instruct models (they already
      respond directly).
    * ``settings.TEACHER_MODEL_NO_THINK_SUFFIX`` is appended to every user
      prompt when non-empty. This is opt-in and intentionally generic: Qwen3
      users set it to ``/no_think``; llama/OpenAI users leave it blank.
    * ``force_json=True`` adds OpenAI ``response_format`` + Ollama-native
      ``format=json`` to the payload. Either field is honored by models that
      support structured output; others ignore both without error.
    """
    url = api_url or settings.TEACHER_MODEL_API_URL
    key = api_key or settings.TEACHER_MODEL_API_KEY

    if not url:
        raise ValueError("Teacher model API URL not configured. Set TEACHER_MODEL_API_URL in .env")

    # Anthropic dispatch — their /v1/messages API isn't OpenAI-compatible
    # (uses x-api-key auth, system at top-level, different response shape),
    # so we route through call_anthropic_chat instead of building the
    # OpenAI-compat payload below. Detection is URL-based to keep the
    # legacy call sites (which only pass api_url + api_key) unchanged.
    if "anthropic.com" in (url or "").lower():
        if not key:
            raise ValueError(
                "Anthropic teacher model requires an API key. Save one "
                "under Project Settings → Secrets (provider="
                "'cloud_llm_anthropic', key_name='api_key')."
            )
        from app.services.cloud_llm_service import (
            CloudLlmError,
            call_anthropic_chat,
        )
        try:
            anth_resp = await call_anthropic_chat(
                api_key=key,
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except CloudLlmError as e:
            raise ValueError(f"Anthropic teacher call failed: {e}") from e
        return {
            "content": anth_resp.content or "",
            "tokens_used": (anth_resp.prompt_tokens or 0) + (anth_resp.completion_tokens or 0),
            "model": anth_resp.model or model_name,
        }

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    no_think_suffix = (getattr(settings, "TEACHER_MODEL_NO_THINK_SUFFIX", "") or "").strip()
    user_content = f"{prompt}\n\n{no_think_suffix}" if no_think_suffix else prompt

    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}
        # Ollama's OpenAI-compat shim also honors a top-level ``format`` field
        # when it's exactly "json" — harmless to servers that don't know it.
        payload["format"] = "json"

    # Generous timeout so larger batches on local Ollama / slower GPUs
    # don't get cut off while the GPU is still actively generating.
    # Configurable via TEACHER_MODEL_TIMEOUT_SECONDS — see backend/app/config.py.
    timeout_seconds = max(30.0, float(settings.TEACHER_MODEL_TIMEOUT_SECONDS or 600.0))
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    # Extract response (OpenAI format)
    content = _coerce_completion_content(
        data.get("choices", [{}])[0].get("message", {}).get("content", "")
    )
    usage = data.get("usage", {})

    return {
        "content": content,
        "tokens_used": usage.get("total_tokens", 0),
        "model": data.get("model", "unknown"),
    }


def _generate_demo_pairs(source_text: str, num_pairs: int = 5) -> list[dict]:
    """Heuristic QA extraction — works without any teacher API for demo/dev use."""
    import re

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', source_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    # Question starters keyed by detected pattern
    transformations = [
        (r'\b(is|are|was|were)\b', 'What {}?'),
        (r'\b(can|could|should|would|might)\b', 'How {}?'),
        (r'\b(because|since|therefore)\b', 'Why {}?'),
        (r'\b(when|after|before|during|until)\b', 'When {}?'),
        (r'\b(where|location|place|region)\b', 'Where {}?'),
    ]

    pairs: list[dict] = []
    used = set()
    for sentence in sentences:
        if len(pairs) >= num_pairs:
            break
        if sentence in used:
            continue
        used.add(sentence)

        question = None
        for pattern, template in transformations:
            if re.search(pattern, sentence, re.IGNORECASE):
                # Strip leading conjunctions/articles for cleaner questions
                cleaned = re.sub(r'^(the |a |an |this |that |these |those )', '', sentence, flags=re.IGNORECASE)
                question = template.format(cleaned.rstrip('.!?').lower())
                break

        if not question:
            # Default: "What is described by: <sentence>?"
            snippet = sentence[:80].rstrip('.!?')
            question = f"What can you tell me about: {snippet}?"

        pairs.append({
            "question": question,
            "answer": sentence,
            "confidence": round(min(0.7, 0.4 + len(sentence) / 500), 3),
            "source": "demo_heuristic",
            "model": "heuristic",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    return pairs


def _unwrap_pairs_payload(payload: Any) -> list[dict] | None:
    """Normalize known payload wrappers into a list of pair-like dicts."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("question"), str) and isinstance(payload.get("answer"), str):
            return [payload]
        for key in ("pairs", "qa_pairs", "questions", "items", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return None


_THINK_TAG_PATTERN = re.compile(
    r"<\s*(think|thinking|reasoning|reflection|scratchpad)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _strip_thinking_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` and similar reasoning wrappers.

    Qwen3, Claude and several other modern models wrap internal chain-of-thought
    output in ``<think>...</think>`` tags when instructed to show reasoning.
    That preamble breaks downstream JSON / Q&A parsing, so we scrub it before
    any structural extraction. Unterminated opening tags (streaming truncation)
    are also stripped — everything from the opening tag to end-of-text is
    treated as hidden reasoning.
    """
    if not text:
        return text
    cleaned = _THINK_TAG_PATTERN.sub("", text)
    cleaned = re.sub(
        r"<\s*(think|thinking|reasoning|reflection|scratchpad)\s*>.*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned.strip()


def _extract_json_blocks(text: str) -> list[str]:
    """Collect candidate JSON blocks from free-form model output."""
    text = _strip_thinking_blocks(text)
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        fragment = text[idx:]
        try:
            _, consumed = decoder.raw_decode(fragment)
        except json.JSONDecodeError:
            continue
        block = fragment[:consumed].strip()
        if block:
            candidates.append(block)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


_MARKDOWN_PREFIX = re.compile(r"^(?:[#>*\-\s\d\.\)]+|[*_]+)")
_MARKDOWN_WRAPPERS = re.compile(r"[*_`]+")

# Recognize question/answer markers tolerating common dressing like markdown
# bold, hash headings, numbered bullets and numbered suffixes (``Q1:``,
# ``**Q 1.**``, ``### Question 2:``, ``1. Q:``, etc.).
#
# Two forms per marker:
#   * ``_BODY``  — marker + separator + text on the same line
#                   (``Q: What is X?`` / ``**Answer 1:** The answer.``)
#   * ``_BARE``  — marker appears alone on a line as a heading; the question
#                  or answer body follows on subsequent lines
#                   (``### Q1\nWhat is X?``).
_Q_MARKER_BODY = re.compile(
    r"^\s*(?:q|question)\s*\d*\s*[:\-.]\s*(.+)$",
    re.IGNORECASE,
)
_A_MARKER_BODY = re.compile(
    r"^\s*(?:a|answer)\s*\d*\s*[:\-.]\s*(.+)$",
    re.IGNORECASE,
)
# Bare markers need a disambiguator so "Qualitative" doesn't trigger a new
# question block — we require either a number after the marker word or a
# trailing separator with no body after it.
_Q_MARKER_BARE = re.compile(
    r"^\s*(?:q|question)\s*(?:\d+\s*[:\-.]?|\s*[:\-.])\s*$",
    re.IGNORECASE,
)
_A_MARKER_BARE = re.compile(
    r"^\s*(?:a|answer)\s*(?:\d+\s*[:\-.]?|\s*[:\-.])\s*$",
    re.IGNORECASE,
)


def _decorate_line(raw_line: str) -> str:
    """Strip markdown dressing so the Q/A markers survive on the line."""
    stripped = raw_line.strip()
    if not stripped:
        return ""
    # Drop leading bullets / numbering / heading hashes: ``1.``, ``- ``, ``### ``.
    stripped = _MARKDOWN_PREFIX.sub("", stripped).strip()
    # Remove inline bold/italic wrappers like ``**Q:**`` → ``Q:``.
    stripped = _MARKDOWN_WRAPPERS.sub("", stripped).strip()
    return stripped


def _parse_plaintext_qa_pairs(text: str) -> list[dict]:
    """Fallback parser for Q/A-shaped model responses.

    Tolerates common dressing on top of the bare ``Q:`` / ``A:`` pattern:
    markdown bold (``**Q:**``), markdown headings (``### Q1``), numbered
    prefixes (``1. Question:``), and trailing numbering on the marker itself
    (``Question 1:``, ``Q1 -``).
    """
    text = _strip_thinking_blocks(text)
    pairs: list[dict] = []
    current_question = ""
    current_answer_lines: list[str] = []
    state = "idle"  # "idle" | "collect_q" | "collect_a"

    def _commit() -> None:
        nonlocal current_question, current_answer_lines
        if current_question and current_answer_lines:
            pairs.append({
                "question": current_question.strip(),
                "answer": " ".join(current_answer_lines).strip(),
            })
        current_question = ""
        current_answer_lines = []

    for raw_line in text.splitlines():
        line = _decorate_line(raw_line)
        if not line:
            continue

        q_body = _Q_MARKER_BODY.match(line)
        if q_body:
            _commit()
            current_question = q_body.group(1).strip()
            state = "collect_q"
            continue

        if _Q_MARKER_BARE.match(line):
            _commit()
            state = "collect_q"
            continue

        a_body = _A_MARKER_BODY.match(line)
        if a_body:
            if not current_question:
                continue
            current_answer_lines = [a_body.group(1).strip()]
            state = "collect_a"
            continue

        if _A_MARKER_BARE.match(line):
            if not current_question:
                continue
            current_answer_lines = []
            state = "collect_a"
            continue

        if state == "collect_q":
            current_question = f"{current_question} {line}".strip() if current_question else line
        elif state == "collect_a":
            current_answer_lines.append(line)

    _commit()
    return pairs


def _parse_teacher_pairs(content: str) -> list[dict]:
    """Parse model output into `{question, answer}` candidate rows."""
    for candidate in _extract_json_blocks(content):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        pairs = _unwrap_pairs_payload(payload)
        if pairs:
            return pairs
    return _parse_plaintext_qa_pairs(content)


def _preview_text(text: str, limit: int = 260) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _pick_text_value(pair: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = pair.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_role(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in {"user", "assistant", "system"}:
        return token
    if token in {"human"}:
        return "user"
    if token in {"ai", "model", "bot"}:
        return "assistant"
    return ""


def _normalize_message(item: Any) -> dict[str, str] | None:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return {"role": "assistant", "content": text}
    if not isinstance(item, dict):
        return None
    role = _normalize_role(item.get("role"))
    content = str(item.get("content") or item.get("text") or "").strip()
    if not content:
        return None
    if not role:
        role = "assistant"
    return {"role": role, "content": content}


def _messages_to_turn_count(messages: list[dict[str, str]]) -> int:
    count = 0
    pending_user = False
    for item in messages:
        role = str(item.get("role") or "").strip().lower()
        if role == "user":
            pending_user = True
            continue
        if role == "assistant" and pending_user:
            count += 1
            pending_user = False
    return count


def _normalize_conversation_payload(item: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    raw_messages = item.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = item.get("conversations")
    messages: list[dict[str, str]] = []
    if isinstance(raw_messages, list):
        for raw_msg in raw_messages:
            normalized = _normalize_message(raw_msg)
            if normalized:
                messages.append(normalized)

    if not messages:
        turns = item.get("turns")
        if isinstance(turns, list):
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                user_content = _pick_text_value(
                    turn,
                    ("user", "question", "prompt", "instruction", "input"),
                )
                assistant_content = _pick_text_value(
                    turn,
                    ("assistant", "answer", "response", "completion", "output"),
                )
                if user_content:
                    messages.append({"role": "user", "content": user_content})
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})

    if not messages:
        single_user = _pick_text_value(item, ("question", "prompt", "instruction", "input", "user"))
        single_assistant = _pick_text_value(item, ("answer", "response", "completion", "output", "assistant"))
        if single_user and single_assistant:
            messages = [
                {"role": "user", "content": single_user},
                {"role": "assistant", "content": single_assistant},
            ]

    turn_count = _messages_to_turn_count(messages)
    if turn_count <= 0:
        return None

    source = str(item.get("source") or "").strip() or "teacher_model"
    model = str(item.get("model") or "").strip() or "unknown"
    confidence = _compute_conversation_confidence(messages)
    conversation_id = str(item.get("conversation_id") or "").strip() or f"conv-{index + 1}-{uuid.uuid4().hex[:8]}"

    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "conversations": messages,
        "turn_count": turn_count,
        "confidence": confidence,
        "source": source,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _unwrap_conversations_payload(payload: Any) -> list[dict] | None:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        keys = (
            "conversations",
            "dialogues",
            "dialogs",
            "chats",
            "items",
            "results",
            "data",
        )
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if isinstance(payload.get("messages"), list):
            return [payload]
    return None


def _parse_teacher_conversations(content: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for candidate in _extract_json_blocks(content):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        rows = _unwrap_conversations_payload(payload)
        if not rows:
            continue
        for idx, item in enumerate(rows):
            normalized = _normalize_conversation_payload(item, idx)
            if normalized:
                parsed.append(normalized)
        if parsed:
            return parsed
    return []


def _build_demo_conversations(
    source_text: str,
    *,
    num_dialogues: int = 3,
    min_turns: int = 3,
    max_turns: int = 5,
) -> list[dict[str, Any]]:
    cleaned_source = re.sub(r"\s+", " ", source_text or "").strip()
    if not cleaned_source:
        return []
    safe_min_turns = max(1, int(min_turns))
    safe_max_turns = max(safe_min_turns, int(max_turns))
    sentence_candidates = re.split(r"(?<=[.!?])\s+", cleaned_source)
    sentence_candidates = [item.strip() for item in sentence_candidates if len(item.strip()) >= 20]
    if not sentence_candidates:
        sentence_candidates = [cleaned_source]

    dialogues: list[dict[str, Any]] = []
    for dialogue_idx in range(max(1, int(num_dialogues))):
        target_turns = safe_min_turns + (dialogue_idx % (safe_max_turns - safe_min_turns + 1))
        messages: list[dict[str, str]] = []
        for turn_idx in range(target_turns):
            sentence = sentence_candidates[(dialogue_idx + turn_idx) % len(sentence_candidates)]
            question_templates = [
                "Can you explain this in simple terms: {snippet}?",
                "What is the key point of: {snippet}?",
                "How does this relate to the rest of the document: {snippet}?",
                "What should a beginner remember from: {snippet}?",
                "Give a concise answer grounded in the source for: {snippet}.",
            ]
            question = question_templates[turn_idx % len(question_templates)].format(
                snippet=sentence[:140].rstrip(".!?")
            )
            assistant = sentence
            if turn_idx > 0:
                assistant = (
                    f"{sentence} This connects to the earlier context in the conversation."
                )
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": assistant})

        conversation_id = f"demo-conv-{dialogue_idx + 1}-{uuid.uuid4().hex[:8]}"
        confidence = _compute_conversation_confidence(messages)
        dialogues.append(
            {
                "conversation_id": conversation_id,
                "messages": messages,
                "conversations": messages,
                "turn_count": target_turns,
                "confidence": confidence,
                "source": "demo_heuristic",
                "model": "heuristic",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return dialogues


def _compute_conversation_confidence(messages: list[dict[str, str]]) -> float:
    pairs: list[dict[str, str]] = []
    pending_question: str | None = None
    for item in messages:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending_question = content
            continue
        if role == "assistant" and pending_question:
            pairs.append({"question": pending_question, "answer": content})
            pending_question = None
    if not pairs:
        return 0.0
    scores = [_compute_confidence(pair) for pair in pairs]
    average = sum(scores) / len(scores)
    turn_bonus = min(0.15, len(pairs) * 0.02)
    return round(min(1.0, max(0.0, average + turn_bonus)), 3)


async def generate_qa_pairs(
    db: AsyncSession | None,
    project_id: int,
    source_text: str,
    num_pairs: int = 5,
    api_url: str = "",
    api_key: str = "",
    model_name: str = "llama3",
) -> list[dict]:
    """Generate Q&A pairs from source text using teacher model, with demo fallback."""
    secret_url = None
    secret_key = None
    if db is not None:
        from app.services.secret_service import get_project_secret_value

        secret_url = await get_project_secret_value(db, project_id, "teacher_model", "api_url")
        secret_key = await get_project_secret_value(db, project_id, "teacher_model", "api_key")
    url = api_url or secret_url or settings.TEACHER_MODEL_API_URL
    resolved_api_key = api_key or secret_key or settings.TEACHER_MODEL_API_KEY

    # ── Demo mode: no teacher API configured ──────────────────
    if not url:
        if not settings.ALLOW_SYNTHETIC_DEMO_FALLBACK:
            raise ValueError(
                "Teacher model API URL is not configured. Set TEACHER_MODEL_API_URL "
                "or enable ALLOW_SYNTHETIC_DEMO_FALLBACK=true for demo-only mode."
            )
        pairs = _generate_demo_pairs(source_text, num_pairs)
        return pairs

    # ── Production mode: call teacher model ───────────────────
    prompt = (
        f"Based on the following text, generate {num_pairs} question-answer pairs "
        "suitable for fine-tuning a small language model.\n\n"
        "Output rules:\n"
        "- Return ONLY valid JSON (no markdown, no code fences, no commentary).\n"
        '- Preferred format: {"pairs":[{"question":"...","answer":"..."}]}.\n'
        '- Alternative accepted format: [{"question":"...","answer":"..."}].\n'
        "- Ground all answers in the source text.\n"
        "- Make questions specific and varied in difficulty.\n\n"
        f"Text:\n{source_text[:4000]}\n\n"
        f"Return exactly {num_pairs} Q&A pairs now."
    )

    result = await call_teacher_model(
        prompt,
        api_url=url,
        api_key=resolved_api_key,
        model_name=model_name,
        force_json=True,
    )

    content = result["content"]
    pairs = _parse_teacher_pairs(content)
    if not pairs:
        raw_content = content or ""
        had_thinking = "<think" in raw_content.lower()
        after_strip = _strip_thinking_blocks(raw_content)
        preview = _preview_text(raw_content)
        hints: list[str] = []
        if had_thinking and not after_strip:
            # Unterminated ``<think>`` → stripper consumed the whole response.
            # Root cause is almost always the token budget being spent inside
            # the reasoning block before any JSON started.
            hints.append(
                "response body was empty after stripping reasoning tags — the "
                "model likely ran out of tokens while still reasoning. Raise "
                "max_tokens and/or set TEACHER_MODEL_NO_THINK_SUFFIX (Qwen3: "
                "'/no_think') to skip the thinking step"
            )
        elif not raw_content.strip():
            hints.append(
                "model returned an empty response — check the teacher endpoint, "
                "model name, and any auth headers"
            )
        elif had_thinking:
            hints.append(
                "response contained <think> reasoning blocks; parser stripped "
                "them but found no JSON or Q/A structure after"
            )
        hints.append(
            "if the model ignores ``response_format=json_object``, try a "
            "stricter system prompt or switch to a model that honors JSON mode"
        )
        hint_text = "; ".join(hints)
        raise ValueError(
            "Teacher model response could not be parsed as Q&A pairs. "
            "Expected JSON `{\"pairs\":[{\"question\":\"...\",\"answer\":\"...\"}]}` "
            "or plaintext `Q:` / `A:` blocks. "
            f"Diagnostics: {hint_text}. "
            f"Response preview: {preview}"
        )

    # Score each pair
    scored_pairs = []
    for pair in pairs:
        question = _pick_text_value(pair, ("question", "q", "prompt", "instruction", "input"))
        answer = _pick_text_value(pair, ("answer", "a", "response", "output", "completion"))
        if not question or not answer:
            continue

        confidence = _compute_confidence(pair)
        scored_pairs.append({
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "source": "teacher_model",
            "model": result.get("model", "unknown"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    if not scored_pairs:
        raise ValueError("No valid synthetic Q&A pairs were returned by the teacher model")

    return scored_pairs


async def generate_conversation_dialogues(
    db: AsyncSession | None,
    project_id: int,
    source_text: str,
    num_dialogues: int = 3,
    min_turns: int = 3,
    max_turns: int = 5,
    api_url: str = "",
    api_key: str = "",
    model_name: str = "llama3",
) -> list[dict[str, Any]]:
    """Generate multi-turn chat dialogues grounded in source text."""
    if min_turns < 1:
        raise ValueError("min_turns must be >= 1")
    if max_turns < min_turns:
        raise ValueError("max_turns must be >= min_turns")

    secret_url = None
    secret_key = None
    if db is not None:
        from app.services.secret_service import get_project_secret_value

        secret_url = await get_project_secret_value(db, project_id, "teacher_model", "api_url")
        secret_key = await get_project_secret_value(db, project_id, "teacher_model", "api_key")

    url = api_url or secret_url or settings.TEACHER_MODEL_API_URL
    resolved_api_key = api_key or secret_key or settings.TEACHER_MODEL_API_KEY

    if not url:
        if not settings.ALLOW_SYNTHETIC_DEMO_FALLBACK:
            raise ValueError(
                "Teacher model API URL is not configured. Set TEACHER_MODEL_API_URL "
                "or enable ALLOW_SYNTHETIC_DEMO_FALLBACK=true for demo-only mode."
            )
        return _build_demo_conversations(
            source_text,
            num_dialogues=num_dialogues,
            min_turns=min_turns,
            max_turns=max_turns,
        )

    prompt = (
        "Generate multi-turn training dialogues grounded in the source text.\n\n"
        "Return ONLY valid JSON.\n"
        "JSON schema:\n"
        '{"conversations":[{"conversation_id":"...","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}]}\n'
        f"Create exactly {num_dialogues} dialogues.\n"
        f"Each dialogue must include between {min_turns} and {max_turns} user-assistant turns.\n"
        "Do not invent facts outside the source text.\n\n"
        f"Source text:\n{source_text[:5000]}"
    )

    result = await call_teacher_model(
        prompt,
        api_url=url,
        api_key=resolved_api_key,
        model_name=model_name,
        force_json=True,
    )
    content = result.get("content", "")
    conversations = _parse_teacher_conversations(str(content))
    if not conversations:
        preview = _preview_text(str(content))
        raise ValueError(
            "Teacher model response was not valid conversation JSON. "
            "Expected conversations/messages structure. "
            f"Response preview: {preview}"
        )

    filtered: list[dict[str, Any]] = []
    for item in conversations:
        turn_count = int(item.get("turn_count") or 0)
        if turn_count < min_turns or turn_count > max_turns:
            continue
        normalized = dict(item)
        normalized["source"] = "teacher_model"
        normalized["model"] = str(result.get("model", "unknown"))
        normalized["generated_at"] = datetime.now(timezone.utc).isoformat()
        filtered.append(normalized)
    if not filtered:
        raise ValueError(
            (
                "Teacher returned conversations, but none matched requested turn constraints "
                f"({min_turns}-{max_turns} turns)."
            )
        )
    return filtered[:num_dialogues]


def _compute_confidence(pair: dict) -> float:
    """Simple heuristic confidence scoring for generated pairs."""
    score = 0.5
    q = pair.get("question", "")
    a = pair.get("answer", "")

    # Length-based scoring
    if len(q) > 20:
        score += 0.1
    if len(a) > 50:
        score += 0.1
    if len(a) > 200:
        score += 0.1

    # Question quality
    if q.endswith("?"):
        score += 0.05
    if any(w in q.lower() for w in ["what", "how", "why", "when", "where", "which", "explain"]):
        score += 0.05

    # Penalize very short answers
    if len(a) < 10:
        score -= 0.2

    return round(min(1.0, max(0.0, score)), 3)


async def save_synthetic_batch(
    db: AsyncSession,
    project_id: int,
    pairs: list[dict],
    min_confidence: float = 0.4,
) -> dict:
    """Save approved synthetic pairs to the dataset, filtering by confidence."""
    ds = await get_or_create_synthetic_dataset(db, project_id)
    syn_dir = _synthetic_dir(project_id)
    file_path = syn_dir / "synthetic.jsonl"

    accepted = []
    rejected = []

    with open(file_path, "a", encoding="utf-8") as f:
        for pair in pairs:
            confidence = pair.get("confidence", 0)
            if confidence >= min_confidence:
                entry = {
                    "id": ds.record_count + len(accepted) + 1,
                    **pair,
                    "status": "accepted",
                }
                f.write(json.dumps(entry) + "\n")
                accepted.append(entry)
            else:
                rejected.append({**pair, "status": "rejected", "reason": "low_confidence"})

    ds.record_count += len(accepted)
    ds.file_path = str(file_path)
    await db.flush()

    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "total": ds.record_count,
        "rejected_pairs": rejected,
    }


async def save_synthetic_conversation_batch(
    db: AsyncSession,
    project_id: int,
    conversations: list[dict[str, Any]],
    min_confidence: float = 0.4,
) -> dict[str, Any]:
    """Save approved synthetic conversations to the synthetic dataset."""
    ds = await get_or_create_synthetic_dataset(db, project_id)
    syn_dir = _synthetic_dir(project_id)
    file_path = syn_dir / "synthetic.jsonl"

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    with open(file_path, "a", encoding="utf-8") as f:
        for idx, raw in enumerate(conversations):
            normalized = _normalize_conversation_payload(raw, idx)
            if not normalized:
                rejected.append(
                    {
                        "conversation_id": str(raw.get("conversation_id") or f"invalid-{idx+1}"),
                        "status": "rejected",
                        "reason": "invalid_conversation_payload",
                    }
                )
                continue

            confidence = float(raw.get("confidence") or normalized.get("confidence") or 0.0)
            normalized["confidence"] = round(min(1.0, max(0.0, confidence)), 3)
            if normalized["confidence"] < min_confidence:
                rejected.append(
                    {
                        "conversation_id": normalized.get("conversation_id"),
                        "status": "rejected",
                        "reason": "low_confidence",
                        "confidence": normalized["confidence"],
                    }
                )
                continue

            entry = {
                "id": ds.record_count + len(accepted) + 1,
                "conversation_id": normalized.get("conversation_id"),
                "conversations": list(normalized.get("messages") or []),
                "messages": list(normalized.get("messages") or []),
                "turn_count": int(normalized.get("turn_count") or 0),
                "confidence": normalized.get("confidence"),
                "source": normalized.get("source"),
                "model": normalized.get("model"),
                "generated_at": normalized.get("generated_at"),
                "status": "accepted",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            accepted.append(entry)

    ds.record_count += len(accepted)
    ds.file_path = str(file_path)
    await db.flush()

    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "total": ds.record_count,
        "accepted_turns": sum(int(item.get("turn_count") or 0) for item in accepted),
        "rejected_items": rejected,
    }


# ── Span-extraction synthesis (PII / NER / structured-extraction span_set) ──
#
# Generates rows shaped `{text, entities: [{type, start, end, text}, ...]}`
# — the same shape StructuredExtractionHandler's span_set scoring mode
# consumes. Triggered from the SyntheticPanel when the project's
# task_profile is structured_extraction and scoring_mode is span_set.


# Conservative built-in regexes for the demo fallback. These don't
# cover everything (person names + addresses + DOBs need an LLM or
# a real NER model), but they're enough to seed the format and give
# the user something useful when no teacher API is configured.
_DEFAULT_SPAN_PATTERNS: list[tuple[str, str]] = [
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # Matches both 10-digit (NNN-NNN-NNNN, (NNN) NNN-NNNN, +1-NNN-NNN-NNNN)
    # and 7-digit (NNN-NNNN) formats with optional country code +
    # optional area-code parens.
    (
        "phone",
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3,4}(?:[-.\s]?\d{4})?\b",
    ),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    # Catches 16-digit (Visa/MC/Discover) and 15-digit (Amex) PANs
    # with optional space/dash separators.
    ("credit_card", r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
    # Amex: 4-6-5 grouping with 15 digits total.
    ("credit_card", r"\b\d{4}[ -]?\d{6}[ -]?\d{5}\b"),
    (
        "ip_address",
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b",
    ),
    # Common API-key shapes: sk-..., sk_live_..., sk_test_..., ghp_...,
    # AKIA..., xoxb-..., xoxp-..., plus JWT prefixes (eyJ...). These
    # patterns are intentionally conservative — they match strings with
    # the right shape, not arbitrary tokens.
    (
        "api_key",
        r"\b(?:sk[-_](?:live|test)[-_][A-Za-z0-9]{16,}|sk-[A-Za-z0-9]{20,}|"
        r"ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xoxb-[A-Za-z0-9-]{20,}|"
        r"xoxp-[A-Za-z0-9-]{20,}|eyJ[A-Za-z0-9_.-]{20,})\b",
    ),
    # Dates of birth: ISO (YYYY-MM-DD) and US (MM/DD/YYYY) formats.
    # Restricted to plausible years (19XX or 20XX) to cut down on
    # false positives from invoice numbers etc.
    ("date_of_birth", r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"),
    ("date_of_birth", r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b"),
]


def _extract_entities_via_regex(
    text: str,
    entity_types: list[str] | None = None,
) -> list[dict]:
    """Cheap regex-based entity extraction for the demo fallback.

    Returns a list of `{type, start, end, text}` dicts. Only the
    well-defined patterns (email/phone/ssn/credit_card/ip_address)
    are detected — names / addresses / DOBs need an LLM. When
    ``entity_types`` is set, restrict detection to that subset.
    """

    import re as _re

    allowed = (
        {t.strip().lower() for t in entity_types if isinstance(t, str)}
        if entity_types
        else None
    )
    out: list[dict] = []
    for ent_type, pattern in _DEFAULT_SPAN_PATTERNS:
        if allowed is not None and ent_type not in allowed:
            continue
        for match in _re.finditer(pattern, text):
            span_text = match.group(0)
            out.append(
                {
                    "type": ent_type,
                    "start": match.start(),
                    "end": match.end(),
                    "text": span_text,
                }
            )
    # Sort by start so the entity list reads left-to-right.
    out.sort(key=lambda e: (e["start"], e["end"]))
    return out


def _merge_regex_entities(
    text: str,
    existing: list[dict],
    entity_types: list[str] | None,
) -> tuple[list[dict], bool]:
    """Augment a row's entity list with regex-detected entities the
    teacher missed.

    Small teacher models (llama3 / mistral 7B) reliably produce diverse
    text snippets but struggle to compute valid character offsets. The
    fix is hybrid: keep the teacher's text + entities, then run the
    regex extractor on the text and add any regex hits that DON'T
    overlap the teacher's. Result: high-quality text from the LLM,
    deterministic offsets from regex.

    Returns ``(merged_entities, augmented_flag)``. ``augmented_flag``
    is True when regex contributed at least one new entity (so callers
    can tag the row's source for transparency).
    """

    regex_hits = _extract_entities_via_regex(text, entity_types)
    if not regex_hits:
        return list(existing), False

    # Build occupied-range list from existing entities (multiset across
    # spans). A regex hit is added only when it doesn't overlap any
    # existing entity at all.
    occupied: list[tuple[int, int]] = [
        (int(e["start"]), int(e["end"])) for e in existing
    ]

    def _overlaps_any(span: tuple[int, int]) -> bool:
        s, e = span
        for os_, oe in occupied:
            if s < oe and os_ < e:
                return True
        return False

    merged = list(existing)
    augmented = False
    for hit in regex_hits:
        span = (int(hit["start"]), int(hit["end"]))
        if _overlaps_any(span):
            continue
        merged.append(hit)
        occupied.append(span)
        augmented = True
    merged.sort(key=lambda e: (e["start"], e["end"]))
    return merged, augmented


def _generate_demo_span_rows(
    source_text: str,
    num_rows: int,
    entity_types: list[str] | None,
) -> list[dict]:
    """Demo fallback. Splits source into sentences and emits up to
    ``num_rows`` rows; each carries the sentence plus any regex-
    detectable entities with correct offsets. Useful as a "show me
    what the format looks like" when no teacher API is configured."""

    import re as _re
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    sentences = [
        s.strip()
        for s in _re.split(r"(?<=[.!?])\s+", source_text.strip())
        if len(s.strip()) > 10
    ]
    if not sentences:
        sentences = [source_text.strip()]

    rows: list[dict] = []
    now = _dt.now(_tz.utc).isoformat()
    # Rotate through sentences; each row gets ONE sentence so the
    # offsets are simple. If num_rows > sentence count, cycle.
    for idx in range(min(num_rows, max(len(sentences), 1))):
        text = sentences[idx % len(sentences)]
        entities = _extract_entities_via_regex(text, entity_types)
        rows.append(
            {
                "text": text,
                "entities": entities,
                # Lower demo confidence so users can re-rank if they
                # set a teacher model later — these are heuristic
                # extractions, not labelled gold.
                "confidence": 0.55 if entities else 0.40,
                "source": "demo_heuristic",
                "model": "regex",
                "generated_at": now,
            }
        )
    return rows


def _validate_span_rows(
    raw: list[Any],
    entity_types: list[str] | None,
) -> list[dict]:
    """Normalize teacher output to canonical span rows.

    For each row:
      - text must be a non-empty string
      - entities must be a list of {type, start, end, text} where
        ``text[start:end] == entity.text`` (drop entities that fail
        the offset sanity check rather than poison the dataset)
      - if entity_types is supplied, drop entities of unknown types
    """

    allowed = (
        {t.strip().lower() for t in entity_types if isinstance(t, str)}
        if entity_types
        else None
    )
    out: list[dict] = []
    for raw_row in raw:
        if not isinstance(raw_row, dict):
            continue
        text = str(raw_row.get("text") or "").strip()
        if not text:
            continue
        entities_in = raw_row.get("entities")
        if not isinstance(entities_in, list):
            entities_in = []
        valid_entities: list[dict] = []
        for ent in entities_in:
            if not isinstance(ent, dict):
                continue
            ent_type = str(ent.get("type") or "").strip()
            if not ent_type:
                continue
            if allowed is not None and ent_type.lower() not in allowed:
                continue
            try:
                start = int(ent.get("start"))
                end = int(ent.get("end"))
            except (TypeError, ValueError):
                continue
            ent_text = str(ent.get("text") or "")
            if start < 0 or end <= start or end > len(text):
                continue
            # Drop entities whose claimed text doesn't match the
            # actual span — these are hallucinated offsets the model
            # often emits, and they poison training data.
            if text[start:end] != ent_text:
                continue
            valid_entities.append(
                {"type": ent_type, "start": start, "end": end, "text": ent_text}
            )
        valid_entities.sort(key=lambda e: (e["start"], e["end"]))
        out.append({"text": text, "entities": valid_entities})
    return out


def _parse_teacher_span_rows(content: str) -> list[dict]:
    """Best-effort parse of the teacher model's JSON output. Returns
    raw row dicts (validation + entity-offset checks are applied by
    ``_validate_span_rows``)."""

    if not content:
        return []
    stripped = _strip_thinking_blocks(content)
    for candidate in _extract_json_blocks(stripped):
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        # Accept either ``{"rows": [...]}`` or a bare ``[...]``.
        if isinstance(payload, dict):
            rows = payload.get("rows") or payload.get("data")
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
    return []


async def generate_span_extraction_rows(
    db: AsyncSession | None,
    project_id: int,
    source_text: str,
    num_rows: int = 5,
    entity_types: list[str] | None = None,
    api_url: str = "",
    api_key: str = "",
    model_name: str = "llama3",
) -> list[dict]:
    """Generate `{text, entities: [...]}` rows for PII / NER /
    structured-extraction span_set training, with demo fallback."""

    secret_url = None
    secret_key = None
    if db is not None:
        from app.services.secret_service import get_project_secret_value

        secret_url = await get_project_secret_value(
            db, project_id, "teacher_model", "api_url"
        )
        secret_key = await get_project_secret_value(
            db, project_id, "teacher_model", "api_key"
        )
    url = api_url or secret_url or settings.TEACHER_MODEL_API_URL
    resolved_api_key = api_key or secret_key or settings.TEACHER_MODEL_API_KEY

    if not url:
        if not settings.ALLOW_SYNTHETIC_DEMO_FALLBACK:
            raise ValueError(
                "Teacher model API URL is not configured. Set "
                "TEACHER_MODEL_API_URL or enable "
                "ALLOW_SYNTHETIC_DEMO_FALLBACK=true for demo-only mode."
            )
        rows = _generate_demo_span_rows(source_text, num_rows, entity_types)
        return rows

    types_clause = (
        f"Allowed entity types (use ONLY these): {', '.join(entity_types)}.\n"
        if entity_types
        else "Detect any common PII / PCI entity types.\n"
    )
    # Few-shot example. Small teacher models (llama3 7B) follow concrete
    # patterns much better than abstract instructions; one worked example
    # cuts the "teacher emits text but skips the entities" failure mode
    # by a lot.
    example = (
        'Example output for one row:\n'
        '{"text": "Email Jane Doe at jane@example.com or call 555-0173.", '
        '"entities": ['
        '{"type":"person_name","start":6,"end":14,"text":"Jane Doe"},'
        '{"type":"email","start":18,"end":34,"text":"jane@example.com"},'
        '{"type":"phone","start":43,"end":51,"text":"555-0173"}'
        ']}\n\n'
    )
    prompt = (
        f"You're generating training data for a PII / PCI span-detection "
        f"model. Produce {num_rows} new realistic snippets in the same "
        "style as the source paragraph but with DIFFERENT, SYNTHETIC PII "
        "values (use 555- phone numbers, 000- SSNs, test PANs like "
        "4242424242424242, @example.* emails). For each snippet, return "
        "the text AND a list of every PII entity with character offsets "
        "and type.\n\n"
        f"{types_clause}"
        f"{example}"
        "Output rules:\n"
        '- Return ONLY valid JSON of the form {"rows":[{"text":"…","entities":[{"type":"…","start":N,"end":M,"text":"…"}]}]}.\n'
        "- start/end are 0-indexed CHARACTER offsets in `text` (not "
        "  word offsets). text[start:end] in your snippet MUST equal "
        "  entity.text exactly. Count characters carefully.\n"
        "- If you're not sure about an entity's offsets, omit it from "
        "  the entities array rather than guess — a downstream regex "
        "  pass will catch the obvious ones (email/phone/SSN/credit_card/"
        "  ip_address/api_key/date_of_birth). Focus on getting the "
        "  TEXT right and only emit entities you're confident about.\n"
        "- Use only synthetic values — never real identifiers.\n"
        "- Vary entity coverage across rows so the dataset isn't repetitive.\n\n"
        f"Source paragraph:\n{source_text[:4000]}\n\n"
        f"Return exactly {num_rows} rows now."
    )

    result = await call_teacher_model(
        prompt,
        api_url=url,
        api_key=resolved_api_key,
        model_name=model_name,
        force_json=True,
    )
    content = result.get("content") or ""
    raw_rows = _parse_teacher_span_rows(content)
    validated = _validate_span_rows(raw_rows, entity_types)
    if not validated:
        # Teacher returned nothing parseable — fall back to demo
        # extraction so the user still sees the expected shape.
        return _generate_demo_span_rows(source_text, num_rows, entity_types)

    from datetime import datetime as _dt
    from datetime import timezone as _tz

    now = _dt.now(_tz.utc).isoformat()
    enriched: list[dict] = []
    for row in validated[:num_rows]:
        teacher_entities = row.get("entities") or []
        # Hybrid annotation: teacher's entities + regex-detected hits
        # the teacher missed. This is the load-bearing fix for small
        # models like llama3-7B that produce great text but unreliable
        # offsets — regex deterministically catches the well-defined
        # types (email/phone/ssn/credit_card/ip_address/api_key/
        # date_of_birth) so users see entities even when the teacher
        # skipped them.
        merged_entities, regex_augmented = _merge_regex_entities(
            row.get("text", ""), teacher_entities, entity_types
        )
        # Confidence reflects entity coverage AND whether annotations
        # came from the teacher only vs. teacher+regex hybrid.
        if merged_entities:
            base = 0.5 + 0.1 * len(merged_entities)
            if regex_augmented and not teacher_entities:
                # Teacher gave text, regex gave entities — text is still
                # high quality but cap confidence lower so the user
                # reviews these rows more carefully.
                base = min(base, 0.65)
            coverage = min(0.9, base)
        else:
            coverage = 0.4
        # Source label is transparent about hybrid origin so reviewers
        # know to spot-check regex-only entities (especially for the
        # types regex can't catch — person_name, street_address, etc.).
        if teacher_entities and regex_augmented:
            source_label = "teacher_llm+regex"
        elif teacher_entities:
            source_label = "teacher_llm"
        elif regex_augmented:
            source_label = "teacher_text+regex_entities"
        else:
            source_label = "teacher_llm"
        enriched.append(
            {
                "text": row.get("text", ""),
                "entities": merged_entities,
                "confidence": round(coverage, 3),
                "source": source_label,
                "model": model_name,
                "generated_at": now,
            }
        )
    return enriched


async def save_synthetic_span_batch(
    db: AsyncSession,
    project_id: int,
    rows: list[dict],
    min_confidence: float = 0.4,
) -> dict:
    """Persist approved span-extraction rows as JSONL alongside the
    other synthetic outputs. Each accepted entry stores `text` plus
    the entity list in the same shape `StructuredExtractionHandler`
    consumes during eval."""

    ds = await get_or_create_synthetic_dataset(db, project_id)
    syn_dir = _synthetic_dir(project_id)
    file_path = syn_dir / "synthetic.jsonl"

    accepted: list[dict] = []
    rejected: list[dict] = []
    with open(file_path, "a", encoding="utf-8") as f:
        for row in rows:
            confidence = row.get("confidence", 0)
            if confidence >= min_confidence:
                entry = {
                    "id": ds.record_count + len(accepted) + 1,
                    "text": row.get("text", ""),
                    "entities": row.get("entities") or [],
                    "confidence": confidence,
                    "source": row.get("source"),
                    "model": row.get("model"),
                    "generated_at": row.get("generated_at"),
                    "status": "accepted",
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                accepted.append(entry)
            else:
                rejected.append(
                    {**row, "status": "rejected", "reason": "low_confidence"}
                )

    ds.record_count += len(accepted)
    ds.file_path = str(file_path)
    await db.flush()
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "total": ds.record_count,
        "rejected_rows": rejected,
    }


# ─────────────────────────────────────────────────────────────────────
# Batched synthetic-span generation (long-running)
# ─────────────────────────────────────────────────────────────────────
#
# Why batched: a teacher LLM call asking for 50 structured-JSON rows is
# already at the upper end of what fits in a context window. Asking for
# 2000 in one shot doesn't work — either the prompt is truncated or
# the response is. The pattern below loops ``ceil(num_rows / 50)`` 50-row
# calls, optionally feeding each batch a fresh randomized sample of the
# project's cleaned chunks so the generated data spans the corpus
# instead of being 40 variations of the same 24KB of text.
#
# Lifecycle mirrors ``cleaning_service.CleaningTask`` exactly so the
# read-side polling endpoint can be a near-copy.

PER_BATCH_ROW_CAP: int = 50
MAX_TOTAL_ROWS: int = 5000
# Per-batch sampling target. 4–8k "tokens" using the standard ~4
# chars/token heuristic. Per-batch target picked uniformly in this
# range so different batches see different effective source sizes.
SAMPLE_MIN_CHARS: int = 4 * 4000
SAMPLE_MAX_CHARS: int = 4 * 8000


@dataclass
class SyntheticSpanTask:
    """In-memory record of a long-running batched span-generation job.
    Process-global, keyed by ``task_id`` in ``_SYNTHETIC_TASKS``."""

    task_id: str
    project_id: int
    target_rows: int
    entity_types: list[str]
    api_url: str
    api_key: str
    model_name: str
    use_all_chunks: bool
    source_text: str

    status: str = "pending"  # pending | running | completed | failed
    rows: list[dict] = field(default_factory=list)
    batches_done: int = 0
    batches_total: int = 0
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "status": self.status,
            "target_rows": self.target_rows,
            "rows_so_far": len(self.rows),
            "batches_done": self.batches_done,
            "batches_total": self.batches_total,
            "rows": list(self.rows),
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
        }


_SYNTHETIC_TASKS: dict[str, Any] = {}
_SYNTHETIC_TASKS_LOCK = threading.Lock()
_MAX_TRACKED_SYNTHETIC_TASKS: int = 64


def _trim_finished_synthetic_tasks() -> None:
    """Evict the oldest finished synthetic tasks when registry grows
    past the cap. Caller holds ``_SYNTHETIC_TASKS_LOCK``."""
    if len(_SYNTHETIC_TASKS) <= _MAX_TRACKED_SYNTHETIC_TASKS:
        return
    finished = sorted(
        (t for t in _SYNTHETIC_TASKS.values() if t.finished_at is not None),
        key=lambda t: t.finished_at,  # type: ignore[arg-type]
    )
    overflow = len(_SYNTHETIC_TASKS) - _MAX_TRACKED_SYNTHETIC_TASKS
    for task in finished[:overflow]:
        _SYNTHETIC_TASKS.pop(task.task_id, None)


async def _load_project_cleaned_chunks(project_id: int) -> list[str]:
    """Load every cleaned chunk text for a project. Mirrors the read
    path of :func:`app.api.cleaning.get_cleaned_chunks` but returns
    only the text payloads (sufficient for the sampler)."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(RawDocument)
            .join(Dataset, Dataset.id == RawDocument.dataset_id)
            .where(Dataset.project_id == project_id)
        )
        docs = list(result.scalars().all())

    texts: list[str] = []
    for doc in docs:
        if not doc.file_path:
            continue
        chunks_path = Path(doc.file_path).with_suffix(".chunks.jsonl")
        if not chunks_path.exists():
            continue
        try:
            content = chunks_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (row.get("text") or "").strip()
            if text:
                texts.append(text)
    return texts


def _sample_chunks_for_batch(
    pool: list[str], *, target_chars: int, rng: random.Random
) -> str:
    """Pick chunks from ``pool`` in random order, joined by ``---``,
    until accumulated char count meets ``target_chars`` (or pool is
    exhausted). Returns the joined source text for one batch."""
    if not pool:
        return ""
    order = list(range(len(pool)))
    rng.shuffle(order)
    accumulated: list[str] = []
    total = 0
    for idx in order:
        chunk = pool[idx]
        accumulated.append(chunk)
        total += len(chunk)
        if total >= target_chars:
            break
    return "\n\n---\n\n".join(accumulated)


async def _run_span_generation_task(task: SyntheticSpanTask) -> None:
    """Drive one batched span-generation job to completion.

    Each iteration generates up to ``PER_BATCH_ROW_CAP`` rows. When
    ``use_all_chunks`` is set, the source text for each batch is a
    fresh random sample of the project's cleaned chunks; otherwise
    every batch reuses ``task.source_text`` verbatim. Best-effort —
    a failed batch is logged into the task error trail but later
    batches still attempt.
    """
    task.status = "running"
    task.updated_at = datetime.now(timezone.utc)

    pool: list[str] = []
    if task.use_all_chunks:
        try:
            pool = await _load_project_cleaned_chunks(task.project_id)
        except Exception as exc:  # noqa: BLE001
            task.status = "failed"
            task.error = f"failed to load cleaned chunks: {exc}"
            task.finished_at = datetime.now(timezone.utc)
            task.updated_at = task.finished_at
            return
        if not pool:
            task.status = "failed"
            task.error = (
                "use_all_chunks=true but no cleaned chunks were found "
                "for this project. Run Data Cleaning first."
            )
            task.finished_at = datetime.now(timezone.utc)
            task.updated_at = task.finished_at
            return

    remaining = task.target_rows
    task.batches_total = (
        (task.target_rows + PER_BATCH_ROW_CAP - 1) // PER_BATCH_ROW_CAP
    )
    rng = random.Random()

    try:
        async with async_session_factory() as db:
            while remaining > 0:
                rows_this_batch = min(PER_BATCH_ROW_CAP, remaining)
                if task.use_all_chunks:
                    target_chars = rng.randint(
                        SAMPLE_MIN_CHARS, SAMPLE_MAX_CHARS
                    )
                    source_text = _sample_chunks_for_batch(
                        pool, target_chars=target_chars, rng=rng
                    )
                else:
                    source_text = task.source_text

                try:
                    batch_rows = await generate_span_extraction_rows(
                        db,
                        task.project_id,
                        source_text,
                        rows_this_batch,
                        task.entity_types or None,
                        task.api_url,
                        task.api_key,
                        task.model_name,
                    )
                    task.rows.extend(batch_rows)
                except Exception as exc:  # noqa: BLE001
                    # One bad batch shouldn't sink the job — record + continue.
                    task.error = (
                        f"batch {task.batches_done + 1}/{task.batches_total}: "
                        f"{exc}"
                    )
                finally:
                    task.batches_done += 1
                    remaining -= rows_this_batch
                    task.updated_at = datetime.now(timezone.utc)

        task.status = "completed"
    except Exception as exc:  # noqa: BLE001
        task.status = "failed"
        task.error = str(exc)
    finally:
        task.finished_at = datetime.now(timezone.utc)
        task.updated_at = task.finished_at


def start_span_generation_task(
    *,
    project_id: int,
    target_rows: int,
    entity_types: list[str] | None,
    api_url: str,
    api_key: str,
    model_name: str,
    use_all_chunks: bool,
    source_text: str,
) -> SyntheticSpanTask:
    """Register + launch a span-generation task. Returns the task
    record immediately; the actual work runs in the background.

    Raises ``ValueError`` for bad inputs (out-of-range ``target_rows``,
    missing source when not using all chunks)."""
    if target_rows < 1 or target_rows > MAX_TOTAL_ROWS:
        raise ValueError(
            f"target_rows must be between 1 and {MAX_TOTAL_ROWS} "
            f"(got {target_rows})"
        )
    if not use_all_chunks and not (source_text or "").strip():
        raise ValueError(
            "source_text is required when use_all_chunks is false"
        )

    task = SyntheticSpanTask(
        task_id=uuid.uuid4().hex,
        project_id=project_id,
        target_rows=target_rows,
        entity_types=list(entity_types or []),
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        use_all_chunks=use_all_chunks,
        source_text=source_text,
    )

    with _SYNTHETIC_TASKS_LOCK:
        _SYNTHETIC_TASKS[task.task_id] = task
        _trim_finished_synthetic_tasks()

    asyncio.create_task(_run_span_generation_task(task))
    return task


def get_span_task_status(task_id: str) -> SyntheticSpanTask | None:
    """Return the in-memory task record for a given id, or None if
    it's already been evicted or never existed."""
    with _SYNTHETIC_TASKS_LOCK:
        task = _SYNTHETIC_TASKS.get(task_id)
        if isinstance(task, SyntheticSpanTask):
            return task
        # Legacy callers may call this for any kind — we return None
        # for non-span tasks here to preserve the original contract.
        # New code should call `get_synth_task_status()` instead.
        if task is not None:
            return None
        return None


def get_synth_task_status(
    task_id: str,
) -> "SyntheticSpanTask | SyntheticQaTask | SyntheticConversationTask | None":
    """Generic task-status lookup that works for span, qa, and
    conversation kinds. Frontend should use this for the
    ``GET /synthetic/tasks/{task_id}`` poll endpoint."""
    with _SYNTHETIC_TASKS_LOCK:
        return _SYNTHETIC_TASKS.get(task_id)


# ─────────────────────────────────────────────────────────────────────
# QA-pair async batched generation (USER-SUCCESS Epic 2c parity).
#
# Mirrors the span-extraction async machinery: target rows up to
# MAX_TOTAL_ROWS, server batches into PER_BATCH_ROW_CAP-sized chunks,
# `use_all_chunks=True` samples a fresh 4-8k-char window from the
# project's cleaned corpus per batch.
# ─────────────────────────────────────────────────────────────────────


@dataclass
class SyntheticQaTask:
    """In-memory record of a batched QA-pair generation job."""

    task_id: str
    project_id: int
    target_rows: int
    api_url: str
    api_key: str
    model_name: str
    use_all_chunks: bool
    source_text: str

    status: str = "pending"
    rows: list[dict] = field(default_factory=list)
    batches_done: int = 0
    batches_total: int = 0
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_kind": "qa",
            "project_id": self.project_id,
            "status": self.status,
            "target_rows": self.target_rows,
            "rows_so_far": len(self.rows),
            "batches_done": self.batches_done,
            "batches_total": self.batches_total,
            "rows": list(self.rows),
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
        }


async def _run_qa_generation_task(task: SyntheticQaTask) -> None:
    """Drive one batched QA-pair generation job to completion.
    Per-batch behavior matches `_run_span_generation_task`: load
    chunks if needed, loop, swallow per-batch failures + continue."""
    task.status = "running"
    task.updated_at = datetime.now(timezone.utc)

    pool: list[str] = []
    if task.use_all_chunks:
        try:
            pool = await _load_project_cleaned_chunks(task.project_id)
        except Exception as exc:  # noqa: BLE001
            task.status = "failed"
            task.error = f"failed to load cleaned chunks: {exc}"
            task.finished_at = datetime.now(timezone.utc)
            task.updated_at = task.finished_at
            return
        if not pool:
            task.status = "failed"
            task.error = (
                "use_all_chunks=true but no cleaned chunks were found "
                "for this project. Run Data Cleaning first."
            )
            task.finished_at = datetime.now(timezone.utc)
            task.updated_at = task.finished_at
            return

    remaining = task.target_rows
    task.batches_total = (
        (task.target_rows + PER_BATCH_ROW_CAP - 1) // PER_BATCH_ROW_CAP
    )
    rng = random.Random()

    try:
        async with async_session_factory() as db:
            while remaining > 0:
                rows_this_batch = min(PER_BATCH_ROW_CAP, remaining)
                if task.use_all_chunks:
                    target_chars = rng.randint(SAMPLE_MIN_CHARS, SAMPLE_MAX_CHARS)
                    source_text = _sample_chunks_for_batch(
                        pool, target_chars=target_chars, rng=rng
                    )
                else:
                    source_text = task.source_text

                try:
                    batch_rows = await generate_qa_pairs(
                        db,
                        task.project_id,
                        source_text,
                        rows_this_batch,
                        task.api_url,
                        task.api_key,
                        task.model_name,
                    )
                    task.rows.extend(batch_rows)
                except Exception as exc:  # noqa: BLE001
                    task.error = (
                        f"batch {task.batches_done + 1}/{task.batches_total}: {exc}"
                    )
                finally:
                    task.batches_done += 1
                    remaining -= rows_this_batch
                    task.updated_at = datetime.now(timezone.utc)
        task.status = "completed"
    except Exception as exc:  # noqa: BLE001
        task.status = "failed"
        task.error = str(exc)
    finally:
        task.finished_at = datetime.now(timezone.utc)
        task.updated_at = task.finished_at


def start_qa_generation_task(
    *,
    project_id: int,
    target_rows: int,
    api_url: str,
    api_key: str,
    model_name: str,
    use_all_chunks: bool,
    source_text: str,
) -> SyntheticQaTask:
    """Register + launch a batched QA-pair generation task."""
    if target_rows < 1 or target_rows > MAX_TOTAL_ROWS:
        raise ValueError(
            f"target_rows must be between 1 and {MAX_TOTAL_ROWS} (got {target_rows})"
        )
    if not use_all_chunks and not (source_text or "").strip():
        raise ValueError(
            "source_text is required when use_all_chunks is false"
        )

    task = SyntheticQaTask(
        task_id=uuid.uuid4().hex,
        project_id=project_id,
        target_rows=target_rows,
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        use_all_chunks=use_all_chunks,
        source_text=source_text,
    )

    with _SYNTHETIC_TASKS_LOCK:
        _SYNTHETIC_TASKS[task.task_id] = task  # type: ignore[assignment]
        _trim_finished_synthetic_tasks()

    asyncio.create_task(_run_qa_generation_task(task))
    return task


# ─────────────────────────────────────────────────────────────────────
# Conversation async batched generation (USER-SUCCESS Epic 2c parity).
# ─────────────────────────────────────────────────────────────────────


@dataclass
class SyntheticConversationTask:
    """In-memory record of a batched multi-turn conversation
    generation job. ``target_rows`` carries the dialogue count
    (per the frontend's existing API field), but the runtime
    output is a list of conversation dicts rather than QA pairs."""

    task_id: str
    project_id: int
    target_rows: int
    min_turns: int
    max_turns: int
    api_url: str
    api_key: str
    model_name: str
    use_all_chunks: bool
    source_text: str

    status: str = "pending"
    rows: list[dict] = field(default_factory=list)
    batches_done: int = 0
    batches_total: int = 0
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_kind": "conversation",
            "project_id": self.project_id,
            "status": self.status,
            "target_rows": self.target_rows,
            "rows_so_far": len(self.rows),
            "batches_done": self.batches_done,
            "batches_total": self.batches_total,
            "rows": list(self.rows),
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
        }


# Conversations are heavier than single QA pairs — keep the batch
# size small so the LLM doesn't drown trying to emit 50 dialogues
# in a single call.
PER_BATCH_CONVERSATION_CAP: int = 5


async def _run_conversation_generation_task(task: SyntheticConversationTask) -> None:
    task.status = "running"
    task.updated_at = datetime.now(timezone.utc)

    pool: list[str] = []
    if task.use_all_chunks:
        try:
            pool = await _load_project_cleaned_chunks(task.project_id)
        except Exception as exc:  # noqa: BLE001
            task.status = "failed"
            task.error = f"failed to load cleaned chunks: {exc}"
            task.finished_at = datetime.now(timezone.utc)
            task.updated_at = task.finished_at
            return
        if not pool:
            task.status = "failed"
            task.error = (
                "use_all_chunks=true but no cleaned chunks were found "
                "for this project. Run Data Cleaning first."
            )
            task.finished_at = datetime.now(timezone.utc)
            task.updated_at = task.finished_at
            return

    remaining = task.target_rows
    task.batches_total = (
        (task.target_rows + PER_BATCH_CONVERSATION_CAP - 1) // PER_BATCH_CONVERSATION_CAP
    )
    rng = random.Random()

    try:
        async with async_session_factory() as db:
            while remaining > 0:
                rows_this_batch = min(PER_BATCH_CONVERSATION_CAP, remaining)
                if task.use_all_chunks:
                    target_chars = rng.randint(SAMPLE_MIN_CHARS, SAMPLE_MAX_CHARS)
                    source_text = _sample_chunks_for_batch(
                        pool, target_chars=target_chars, rng=rng
                    )
                else:
                    source_text = task.source_text

                try:
                    batch_rows = await generate_conversation_dialogues(
                        db,
                        task.project_id,
                        source_text,
                        rows_this_batch,
                        task.min_turns,
                        task.max_turns,
                        task.api_url,
                        task.api_key,
                        task.model_name,
                    )
                    task.rows.extend(batch_rows)
                except Exception as exc:  # noqa: BLE001
                    task.error = (
                        f"batch {task.batches_done + 1}/{task.batches_total}: {exc}"
                    )
                finally:
                    task.batches_done += 1
                    remaining -= rows_this_batch
                    task.updated_at = datetime.now(timezone.utc)
        task.status = "completed"
    except Exception as exc:  # noqa: BLE001
        task.status = "failed"
        task.error = str(exc)
    finally:
        task.finished_at = datetime.now(timezone.utc)
        task.updated_at = task.finished_at


def start_conversation_generation_task(
    *,
    project_id: int,
    target_rows: int,
    min_turns: int,
    max_turns: int,
    api_url: str,
    api_key: str,
    model_name: str,
    use_all_chunks: bool,
    source_text: str,
) -> SyntheticConversationTask:
    """Register + launch a batched multi-turn conversation generation task."""
    if target_rows < 1 or target_rows > MAX_TOTAL_ROWS:
        raise ValueError(
            f"target_rows must be between 1 and {MAX_TOTAL_ROWS} (got {target_rows})"
        )
    if not use_all_chunks and not (source_text or "").strip():
        raise ValueError("source_text is required when use_all_chunks is false")
    if min_turns < 1 or max_turns < min_turns:
        raise ValueError("min_turns >= 1 and max_turns >= min_turns required")

    task = SyntheticConversationTask(
        task_id=uuid.uuid4().hex,
        project_id=project_id,
        target_rows=target_rows,
        min_turns=min_turns,
        max_turns=max_turns,
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        use_all_chunks=use_all_chunks,
        source_text=source_text,
    )

    with _SYNTHETIC_TASKS_LOCK:
        _SYNTHETIC_TASKS[task.task_id] = task  # type: ignore[assignment]
        _trim_finished_synthetic_tasks()

    asyncio.create_task(_run_conversation_generation_task(task))
    return task
