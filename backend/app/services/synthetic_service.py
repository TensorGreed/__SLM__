"""Synthetic data generation service — teacher model integration."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType


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
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
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


async def generate_qa_pairs(
    db: AsyncSession,
    project_id: int,
    source_text: str,
    num_pairs: int = 5,
    api_url: str = "",
    api_key: str = "",
    model_name: str = "llama3",
) -> list[dict]:
    """Generate Q&A pairs from source text using teacher model, with demo fallback."""
    url = api_url or settings.TEACHER_MODEL_API_URL

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
    prompt = f"""Based on the following text, generate {num_pairs} question-answer pairs suitable for fine-tuning a small language model.

Format your response as a JSON array of objects with "question" and "answer" keys.
Make questions specific, varied in difficulty, and grounded in the text.

Text:
{source_text[:4000]}

Generate {num_pairs} Q&A pairs as JSON array:"""

    result = await call_teacher_model(prompt, api_url=api_url, api_key=api_key, model_name=model_name)

    # Parse the JSON from the response
    content = result["content"]
    try:
        # Try to extract JSON array from response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            pairs = json.loads(content[start:end])
        else:
            pairs = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError("Teacher model response was not valid JSON for Q&A extraction")

    if not isinstance(pairs, list):
        raise ValueError("Teacher model response must be a JSON array of {question, answer}")

    # Score each pair
    scored_pairs = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        question = str(pair.get("question", "")).strip()
        answer = str(pair.get("answer", "")).strip()
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
