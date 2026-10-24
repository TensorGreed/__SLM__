"""The one predicate for "this source must never feed training".

A document the predefined-split import registered with ``metadata.split_role
== "test"`` is the user's own answer key. Nothing derived from it may reach a
train / val division, a synthetic seed pool or a passage index. Every place
that pools documents (or rows derived from documents) for training calls the
helpers here instead of re-deriving the rule:

- ``dataset_service.combine_datasets`` - RAW branch (``is_test_role_document``)
  and every non-RAW branch (rows carrying ``source_document_id`` of a test-role
  document, any status: ``test_role_document_ids`` + ``drop_test_role_rows``).
- ``cleaning_service.clean_document`` - refuses test-role documents, so no
  ``cleaned.jsonl`` row, ``.chunks.jsonl`` or ``.cleaned.txt`` (the inputs of
  the passage loaders, RAFT / teacher-answer prompts, documents-to-Q&A flow,
  span relabels and the synthetic seed pickers) is ever produced for one.
- ``synthetic_service._load_project_cleaned_chunks`` and
  ``GET /cleaning/chunks`` (the synthetic / gold seed pickers) - skip them.

Projects that never imported their own splits have no such document, so
every filter here is a no-op for them.
"""

from __future__ import annotations

from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, RawDocument

TEST_ROLE = "test"


def is_test_role_document(doc: Any) -> bool:
    """True for a document the predefined-split import registered as the test file."""
    return ((getattr(doc, "metadata_", None) or {}).get("split_role")) == TEST_ROLE


async def test_role_document_ids(db: AsyncSession, project_id: int) -> set[int]:
    """Ids of the project's test-role documents, ANY status: a superseded
    (REJECTED) earlier import's cleaned rows must stay out of training too."""
    result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(Dataset.project_id == project_id)
    )
    return {int(d.id) for d in result.scalars() if is_test_role_document(d)}


def drop_test_role_rows(rows: Iterable[Any], test_doc_ids: set[int]) -> list[Any]:
    """``rows`` without those whose ``source_document_id`` is a test-role document."""
    if not test_doc_ids:
        return list(rows)
    out = []
    for row in rows:
        doc_id = row.get("source_document_id") if isinstance(row, dict) else None
        try:
            if doc_id is not None and int(doc_id) in test_doc_ids:
                continue
        except (TypeError, ValueError):
            pass
        out.append(row)
    return out
