"""Synthetic data generation service — teacher model integration."""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType


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


async def call_teacher_model(
    prompt: str,
    system_prompt: str = "You are a helpful assistant that generates high-quality training data.",
    api_url: str = "",
    api_key: str = "",
    model_name: str = "llama3",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Call external teacher LLM API (OpenAI-compatible format)."""
    url = api_url or settings.TEACHER_MODEL_API_URL
    key = api_key or settings.TEACHER_MODEL_API_KEY

    if not url:
        raise ValueError("Teacher model API URL not configured. Set TEACHER_MODEL_API_URL in .env")

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
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


def _extract_json_blocks(text: str) -> list[str]:
    """Collect candidate JSON blocks from free-form model output."""
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


def _parse_plaintext_qa_pairs(text: str) -> list[dict]:
    """Fallback parser for `Q:`/`A:` formatted model responses."""
    pairs: list[dict] = []
    current_question = ""
    current_answer_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        q_match = re.match(r"^(?:\d+[\).:\-]\s*)?(?:q|question)\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
        if q_match:
            if current_question and current_answer_lines:
                pairs.append({
                    "question": current_question.strip(),
                    "answer": " ".join(current_answer_lines).strip(),
                })
            current_question = q_match.group(1).strip()
            current_answer_lines = []
            continue

        a_match = re.match(r"^(?:a|answer)\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
        if a_match:
            if not current_question:
                continue
            current_answer_lines = [a_match.group(1).strip()]
            continue

        if current_question and current_answer_lines:
            current_answer_lines.append(line)
        elif current_question:
            current_question = f"{current_question} {line}".strip()

    if current_question and current_answer_lines:
        pairs.append({
            "question": current_question.strip(),
            "answer": " ".join(current_answer_lines).strip(),
        })
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
    )

    content = result["content"]
    pairs = _parse_teacher_pairs(content)
    if not pairs:
        preview = _preview_text(content)
        raise ValueError(
            "Teacher model response was not valid JSON for Q&A extraction. "
            "Expected JSON array/object with question+answer fields. "
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
