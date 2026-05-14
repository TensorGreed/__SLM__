"""Built-in target mappers.

Each mapper lives in its own module (one file per transform) and
registers itself at import time. Phase A ships two:

- ``bio_to_spans`` — generalizes the Phase 4.x kaggle PII converter.
  Handles any BIO-tagged dataset (PII, medical, legal, financial NER).
- ``label_to_classification`` — flat passthrough for ``{text, label}``-
  style classification data.

Phase C expands the catalog (preference_pair / rag_passthrough /
kv_to_structured / qa_pair_passthrough / chat_messages_passthrough /
text_only).
"""

from app.services.dataset_import.mappers import bio_to_spans as _bio_to_spans  # noqa: F401
from app.services.dataset_import.mappers import (  # noqa: F401
    label_to_classification as _label_to_classification,
)
