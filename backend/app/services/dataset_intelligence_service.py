"""Semantic dataset intelligence: embeddings, clustering, and diversity diagnostics."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import DatasetType
from app.services.dataset_service import combine_datasets


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_prepared_dir(project_id: int) -> Path:
    directory = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _normalize_split(value: str | None) -> str:
    token = str(value or "train").strip().lower()
    if token in {"train", "val", "validation", "test", "combined"}:
        return "val" if token == "validation" else token
    return "train"


def _safe_open_jsonl(path: Path, sample_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if not item:
                continue
            try:
                payload = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                rows.append({"value": payload})
            if len(rows) >= sample_size:
                break
    return rows


def _message_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    chunks: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or item.get("text") or "").strip()
        if not content:
            continue
        if role:
            chunks.append(f"{role}: {content}")
        else:
            chunks.append(content)
    return "\n".join(chunks).strip()


def _record_to_text(row: dict[str, Any]) -> str:
    direct_fields = (
        "text",
        "content",
        "source_text",
        "target_text",
        "prompt",
        "instruction",
        "question",
        "answer",
        "completion",
        "response",
    )
    for key in direct_fields:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("messages", "conversations"):
        merged = _message_text(row.get(key))
        if merged:
            return merged

    if "value" in row:
        value = row.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)

    chunks: list[str] = []
    for key, value in row.items():
        if key.startswith("_"):
            continue
        if isinstance(value, str) and value.strip():
            chunks.append(f"{key}: {value.strip()}")
        elif isinstance(value, (int, float, bool)):
            chunks.append(f"{key}: {value}")
    return "\n".join(chunks).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]{2,}", text.lower())


_TOXIC_TERMS = {
    "hate",
    "idiot",
    "stupid",
    "kill",
    "racist",
    "sexist",
    "threat",
    "abuse",
    "harass",
    "violence",
    "nazi",
    "terrorist",
}

_TOPIC_STOPWORDS = {
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "into",
    "your",
    "have",
    "will",
    "they",
    "their",
    "about",
    "there",
    "were",
    "been",
    "what",
    "when",
    "where",
    "which",
}


def _toxicity_score(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    toxic_hits = sum(1 for token in tokens if token in _TOXIC_TERMS)
    if toxic_hits <= 0:
        return 0.0
    return min(1.0, toxic_hits / max(1, len(tokens) * 0.08))


def _topic_label(texts: list[str]) -> str:
    counts: Counter[str] = Counter()
    for text in texts:
        for token in _tokenize(text):
            if token in _TOPIC_STOPWORDS or len(token) < 4:
                continue
            counts[token] += 1
    if not counts:
        return "mixed"
    top_tokens = [item for item, _ in counts.most_common(2)]
    return " / ".join(top_tokens) if top_tokens else "mixed"


def _hashing_embeddings(texts: list[str], dim: int = 256) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        for token in _tokenize(text):
            bucket = hash(token) % dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        vectors.append(vec)
    return vectors


def _sentence_transformer_embeddings(
    texts: list[str],
    embedding_model: str,
) -> tuple[list[list[float]], str, str]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return _hashing_embeddings(texts), "hashing_fallback", "hashing-256"

    model_name = str(embedding_model or "").strip() or "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    matrix = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors: list[list[float]] = []
    for row in matrix:
        if hasattr(row, "tolist"):
            values = row.tolist()
        else:
            values = list(row)
        vectors.append([float(item) for item in values])
    return vectors, "sentence_transformers", model_name


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    if size <= 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for idx in range(size):
        va = float(a[idx])
        vb = float(b[idx])
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _assign_clusters(
    embeddings: list[list[float]],
    requested_count: int,
) -> tuple[list[int], int, str]:
    n = len(embeddings)
    if n <= 1:
        return [0] * n, 1, "single_cluster"

    count = max(2, min(requested_count, n))
    try:
        from sklearn.cluster import KMeans
    except Exception:
        # Fallback: deterministic centroid assignment from first N rows.
        centroids = [list(embeddings[idx]) for idx in range(count)]
        labels: list[int] = []
        cluster_sizes = [0] * count
        for vector in embeddings:
            scored = [(_cosine(vector, centroids[idx]), idx) for idx in range(count)]
            _, cluster_id = max(scored, key=lambda item: item[0])
            labels.append(cluster_id)
            cluster_sizes[cluster_id] += 1
        return labels, count, "centroid_fallback"

    model = KMeans(
        n_clusters=count,
        n_init=10,
        random_state=42,
    )
    labels_raw = model.fit_predict(embeddings)
    labels = [int(item) for item in labels_raw]
    return labels, count, "kmeans"


def _nearest_neighbor_scores(embeddings: list[list[float]]) -> list[float]:
    n = len(embeddings)
    if n <= 1:
        return [0.0] * n
    scores = [0.0] * n
    for i in range(n):
        best = -1.0
        for j in range(n):
            if i == j:
                continue
            sim = _cosine(embeddings[i], embeddings[j])
            if sim > best:
                best = sim
        scores[i] = max(0.0, best)
    return scores


def _suggestions(
    redundancy_ratio: float,
    diversity_score: float,
    cluster_count: int,
    sample_size: int,
) -> list[str]:
    hints: list[str] = []
    if redundancy_ratio >= 0.4:
        hints.append(
            "High semantic redundancy detected. Consider deduplication or downsampling dense clusters."
        )
    if diversity_score < 0.5:
        hints.append(
            "Semantic diversity is low. Add examples from missing intents/styles to improve generalization."
        )
    if cluster_count <= 2 and sample_size >= 100:
        hints.append(
            "Dataset appears concentrated in very few clusters. Add broader task/domain coverage."
        )
    if not hints:
        hints.append("Semantic spread looks healthy for current sample size.")
    return hints


async def _load_rows(
    db: AsyncSession,
    project_id: int,
    *,
    target_split: str,
    sample_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = _normalize_split(target_split)
    if split == "combined":
        rows = await combine_datasets(
            db,
            project_id,
            include_types=[
                DatasetType.CLEANED,
                DatasetType.SYNTHETIC,
                DatasetType.GOLD_DEV,
            ],
            chat_template="llama3",
        )
        return rows[:sample_size], {"split": "combined", "source": "combined_datasets"}

    split_file = _project_prepared_dir(project_id) / f"{split}.jsonl"
    rows = _safe_open_jsonl(split_file, sample_size)
    return rows, {"split": split, "source": str(split_file)}


async def analyze_semantic_dataset_intelligence(
    db: AsyncSession,
    project_id: int,
    *,
    target_split: str = "train",
    sample_size: int = 400,
    cluster_count: int | None = None,
    similarity_threshold: float = 0.92,
    embedding_model: str = "",
) -> dict[str, Any]:
    """Analyze semantic diversity, redundancy, and cluster spread for project data."""
    safe_sample = max(20, min(int(sample_size), 2000))
    safe_threshold = max(0.5, min(float(similarity_threshold), 0.999))
    rows, source = await _load_rows(
        db,
        project_id,
        target_split=target_split,
        sample_size=safe_sample,
    )
    texts = []
    index_map = []
    for idx, row in enumerate(rows):
        text = _record_to_text(row)
        if not text:
            continue
        texts.append(text)
        index_map.append(idx)

    if not texts:
        raise ValueError(
            (
                f"No usable text rows found for semantic analysis in split '{_normalize_split(target_split)}'. "
                "Run dataset split/prep or choose a split with records."
            )
        )

    vectors, backend, resolved_model = _sentence_transformer_embeddings(texts, embedding_model)
    resolved_clusters = (
        max(2, min(int(cluster_count), max(2, len(vectors))))
        if cluster_count is not None
        else max(2, min(12, int(round(math.sqrt(len(vectors)))) or 2))
    )
    labels, resolved_clusters, cluster_backend = _assign_clusters(vectors, resolved_clusters)
    nearest_scores = _nearest_neighbor_scores(vectors)

    redundant_indices = [idx for idx, score in enumerate(nearest_scores) if score >= safe_threshold]
    redundancy_ratio = (len(redundant_indices) / len(vectors)) if vectors else 0.0
    avg_similarity = sum(nearest_scores) / len(nearest_scores)
    diversity_score = 1.0 - avg_similarity
    diversity_score = max(0.0, min(1.0, diversity_score))

    cluster_rows: dict[int, list[int]] = {}
    for local_idx, label in enumerate(labels):
        cluster_rows.setdefault(int(label), []).append(local_idx)

    clusters: list[dict[str, Any]] = []
    for cluster_id, member_indices in sorted(cluster_rows.items(), key=lambda item: len(item[1]), reverse=True):
        representative_indices = sorted(
            member_indices,
            key=lambda idx: nearest_scores[idx],
        )[:3]
        representatives = [
            {
                "sample_index": int(index_map[item]),
                "similarity_to_nearest": round(float(nearest_scores[item]), 4),
                "text_preview": texts[item][:240],
            }
            for item in representative_indices
        ]
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "size": len(member_indices),
                "share": round(len(member_indices) / len(vectors), 4),
                "representatives": representatives,
            }
        )

    report = {
        "project_id": project_id,
        "created_at": _utcnow_iso(),
        "source": source,
        "sample_size_requested": safe_sample,
        "sample_size_analyzed": len(vectors),
        "embedding_backend": backend,
        "embedding_model": resolved_model,
        "cluster_backend": cluster_backend,
        "cluster_count": resolved_clusters,
        "semantic_diversity_score": round(diversity_score, 4),
        "average_nearest_similarity": round(avg_similarity, 4),
        "similarity_threshold": safe_threshold,
        "redundant_count": len(redundant_indices),
        "redundancy_ratio": round(redundancy_ratio, 4),
        "clusters": clusters,
        "suggestions": _suggestions(
            redundancy_ratio=redundancy_ratio,
            diversity_score=diversity_score,
            cluster_count=resolved_clusters,
            sample_size=len(vectors),
        ),
    }

    prepared_dir = _project_prepared_dir(project_id)
    report_path = prepared_dir / "semantic_intelligence.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report

async def get_project_eda_stats(db: AsyncSession, project_id: int) -> dict[str, Any]:
    """Calculate onboarding-first EDA stats for raw ingestion data."""
    from sqlalchemy import select

    from app.models.dataset import Dataset, RawDocument

    docs_query = select(RawDocument).join(Dataset).where(
        Dataset.project_id == project_id,
        Dataset.dataset_type == DatasetType.RAW,
    )
    result = await db.execute(docs_query)
    docs = result.scalars().all()

    total_files = len(docs)
    total_size_bytes = sum(int(doc.file_size_bytes or 0) for doc in docs)

    sampled_texts: list[str] = []
    schema_keys: set[str] = set()
    total_rows = 0
    sample_limit_per_doc = 500
    sample_limit_total = 1200

    for doc in docs:
        path = Path(str(doc.file_path or ""))
        if not path.exists() or not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for row_idx, line in enumerate(handle):
                    total_rows += 1
                    if row_idx >= sample_limit_per_doc or len(sampled_texts) >= sample_limit_total:
                        continue
                    raw_line = line.strip()
                    if not raw_line:
                        continue
                    text_content = ""
                    try:
                        payload = json.loads(raw_line)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict):
                        schema_keys.update(str(key) for key in payload.keys())
                        text_content = _record_to_text(payload)
                    elif isinstance(payload, list):
                        text_content = _message_text(payload)
                    else:
                        text_content = raw_line
                    if text_content:
                        sampled_texts.append(text_content[:4000])
        except Exception:
            continue

    token_counts = [len(re.findall(r"\w+|[^\w\s]", text)) for text in sampled_texts]
    token_counts.sort()
    p50 = token_counts[len(token_counts) // 2] if token_counts else 0
    p90 = token_counts[int(len(token_counts) * 0.9)] if token_counts else 0
    p99 = token_counts[int(len(token_counts) * 0.99)] if token_counts else 0
    max_tokens = token_counts[-1] if token_counts else 0

    unique_hashes = {hash(text) for text in sampled_texts}
    duplicate_ratio = 1.0 - (len(unique_hashes) / len(sampled_texts)) if sampled_texts else 0.0

    toxicity_scores = [_toxicity_score(text) for text in sampled_texts]
    avg_toxicity = (sum(toxicity_scores) / len(toxicity_scores)) if toxicity_scores else 0.0
    flagged_toxicity = [idx for idx, score in enumerate(toxicity_scores) if score >= 0.35]

    topic_clusters: list[dict[str, Any]] = []
    if len(sampled_texts) >= 4:
        vectors = _hashing_embeddings(sampled_texts[: min(len(sampled_texts), 400)], dim=128)
        requested_clusters = max(2, min(8, int(round(math.sqrt(len(vectors))))))
        labels, resolved_clusters, cluster_backend = _assign_clusters(vectors, requested_clusters)
        by_cluster: dict[int, list[int]] = {}
        for idx, cluster_id in enumerate(labels):
            by_cluster.setdefault(int(cluster_id), []).append(idx)
        for cluster_id, members in sorted(by_cluster.items(), key=lambda item: len(item[1]), reverse=True):
            previews = [sampled_texts[idx][:120] for idx in members[:2]]
            label = _topic_label([sampled_texts[idx] for idx in members[: min(20, len(members))]])
            topic_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(members),
                    "share": round(len(members) / len(vectors), 4),
                    "label": label,
                    "sample_previews": previews,
                }
            )
        topic_cluster_backend = cluster_backend
        topic_cluster_count = resolved_clusters
    else:
        topic_cluster_backend = "insufficient_sample"
        topic_cluster_count = 0

    outlier_candidates: list[dict[str, Any]] = []
    p99_floor = max(p99, p90 + 50)
    min_token_floor = max(5, int(max(1, p50) * 0.1))
    for idx, text in enumerate(sampled_texts):
        token_count = len(re.findall(r"\w+|[^\w\s]", text))
        toxicity = toxicity_scores[idx] if idx < len(toxicity_scores) else 0.0
        reason: str | None = None
        if token_count > p99_floor and p99_floor > 0:
            reason = "very_long"
        elif token_count < min_token_floor:
            reason = "very_short"
        elif toxicity >= 0.35:
            reason = "toxic"
        if not reason:
            continue
        outlier_candidates.append(
            {
                "sample_index": idx,
                "token_count": token_count,
                "toxicity_score": round(toxicity, 4),
                "reason": reason,
                "text_preview": text[:160],
            }
        )
        if len(outlier_candidates) >= 30:
            break

    return {
        "project_id": project_id,
        "total_files": total_files,
        "total_size_bytes": total_size_bytes,
        "estimated_total_rows": total_rows,
        "sample_size": len(sampled_texts),
        "schema_keys_present": sorted(schema_keys)[:50],
        "token_distribution": {
            "p50": p50,
            "p90": p90,
            "p99": p99,
            "max": max_tokens,
        },
        "estimated_duplicate_ratio": round(duplicate_ratio, 4),
        "toxicity": {
            "average_score": round(avg_toxicity, 4),
            "flagged_ratio": round((len(flagged_toxicity) / len(sampled_texts)), 4) if sampled_texts else 0.0,
            "flagged_count": len(flagged_toxicity),
        },
        "topic_clusters": topic_clusters[:6],
        "topic_cluster_backend": topic_cluster_backend,
        "topic_cluster_count": topic_cluster_count,
        "outlier_candidates": outlier_candidates,
        "suggested_actions": {
            "remove_outliers_count": len(outlier_candidates),
        },
    }


async def remove_project_outliers(
    db: AsyncSession,
    project_id: int,
    *,
    min_tokens: int = 5,
    max_tokens: int = 4096,
    toxicity_threshold: float = 0.35,
    max_rows: int = 100000,
) -> dict[str, Any]:
    """Filter obvious outlier rows from prepared train split and persist a filtered file."""
    _ = db
    prepared_dir = _project_prepared_dir(project_id)
    train_path = prepared_dir / "train.jsonl"
    if not train_path.exists():
        raise ValueError("Prepared train split not found. Run dataset split before removing outliers.")

    rows = _safe_open_jsonl(train_path, sample_size=max(1, int(max_rows)))
    if not rows:
        raise ValueError("Prepared train split is empty. Nothing to filter.")

    safe_min_tokens = max(1, int(min_tokens))
    safe_max_tokens = max(safe_min_tokens, int(max_tokens))
    safe_toxicity = max(0.0, min(float(toxicity_threshold), 1.0))

    kept_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    for row in rows:
        text = _record_to_text(row)
        token_count = len(re.findall(r"\w+|[^\w\s]", text))
        toxicity = _toxicity_score(text)
        reason = None
        if token_count < safe_min_tokens:
            reason = "very_short"
        elif token_count > safe_max_tokens:
            reason = "very_long"
        elif toxicity >= safe_toxicity:
            reason = "toxic"

        if reason:
            removed_rows.append(
                {
                    "reason": reason,
                    "token_count": token_count,
                    "toxicity_score": round(toxicity, 4),
                    "text_preview": text[:160],
                }
            )
        else:
            kept_rows.append(row)

    output_path = prepared_dir / "train.no_outliers.jsonl"
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "project_id": project_id,
        "source_path": str(train_path),
        "output_path": str(output_path),
        "rows_in": len(rows),
        "rows_kept": len(kept_rows),
        "rows_removed": len(removed_rows),
        "removal_ratio": round(len(removed_rows) / len(rows), 4) if rows else 0.0,
        "removed_preview": removed_rows[:25],
        "thresholds": {
            "min_tokens": safe_min_tokens,
            "max_tokens": safe_max_tokens,
            "toxicity_threshold": safe_toxicity,
        },
        "created_at": _utcnow_iso(),
    }
    report_path = prepared_dir / "outlier_removal_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
