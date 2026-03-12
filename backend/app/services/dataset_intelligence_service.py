"""Semantic dataset intelligence: embeddings, clustering, and diversity diagnostics."""

from __future__ import annotations

import json
import math
import re
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
    """Calculate basic EDA stats (row counts, token distributions, duplicate estimate) for project data."""
    from app.models.dataset import Dataset, DatasetType, RawDocument
    from sqlalchemy import select
    
    # 1. Fetch raw documents to read samples
    docs_query = select(RawDocument).join(Dataset).where(
        Dataset.project_id == project_id,
        Dataset.dataset_type == DatasetType.RAW
    )
    result = await db.execute(docs_query)
    docs = result.scalars().all()
    
    total_files = len(docs)
    total_size_bytes = sum(doc.file_size_bytes for doc in docs)
    
    # 2. Sample records to compute token limits and schemas
    sampled_texts = []
    schema_keys: set[str] = set()
    total_rows = 0
    
    # Try reading as JSONL or plain text
    import json
    for doc in docs:
        path = Path(doc.file_path)
        if not path.exists():
            continue
            
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                total_rows += 1
                if i < 1000:  # Sample up to 1000 rows across raw docs
                    text_content = ""
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            for k in record.keys():
                                schema_keys.add(k)
                            text_content = _record_to_text(record)
                    except json.JSONDecodeError:
                        text_content = line.strip()
                        
                    if text_content:
                        sampled_texts.append(text_content)
                    
    # 3. Calculate distributions on sample
    token_counts = []
    for text in sampled_texts:
        # rough token estimate: words + punctuation
        tokens = len(re.findall(r"\w+|[^\w\s]", text))
        token_counts.append(tokens)
        
    token_counts.sort()
    p50 = token_counts[len(token_counts) // 2] if token_counts else 0
    p90 = token_counts[int(len(token_counts) * 0.9)] if token_counts else 0
    p99 = token_counts[int(len(token_counts) * 0.99)] if token_counts else 0
    
    # 4. Rough duplicate estimate using hashing on samples
    unique_hashes = set()
    for text in sampled_texts:
        import hashlib
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        unique_hashes.add(h)
    
    duplicate_ratio = 1.0 - (len(unique_hashes) / len(sampled_texts)) if sampled_texts else 0.0
    
    return {
        "project_id": project_id,
        "total_files": total_files,
        "total_size_bytes": total_size_bytes,
        "estimated_total_rows": total_rows,
        "sample_size": len(sampled_texts),
        "schema_keys_present": list(schema_keys)[:50],
        "token_distribution": {
            "p50": p50,
            "p90": p90,
            "p99": p99,
            "max": token_counts[-1] if token_counts else 0
        },
        "estimated_duplicate_ratio": round(duplicate_ratio, 4)
    }


