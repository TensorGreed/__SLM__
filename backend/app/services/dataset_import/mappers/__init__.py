"""Built-in target mappers.

Each mapper lives in its own module (one file per transform) and
registers itself at import time. Catalog:

- ``bio_to_spans`` — BIO-tagged tokens → entity spans
  (StructuredExtractionHandler / span_set scoring).
- ``label_to_classification`` — flat ``{text, label}`` passthrough
  (ClassificationHandler).
- ``text_only`` — single text column, no labels
  (language modeling via QAHandler).
- ``qa_pair_passthrough`` — ``{question, answer}`` rows
  (QAHandler, also seq2seq-compatible).
- ``chat_messages_passthrough`` — multi-turn chat ``messages`` list
  (chat SFT via QAHandler).
- ``preference_pair`` — ``{prompt, chosen, rejected}`` triples
  (AlignmentHandler — DPO / ORPO).
- ``rag_passthrough`` — ``{question, context, answer}`` triples
  (RAGHandler — grounded QA).
- ``kv_to_structured`` — flat key-value extractions
  (StructuredExtractionHandler / field_match scoring — invoices /
  forms / receipts).

- ``code_label_to_findings`` — labelled code → findings rows
  (a piece of code + the labelled findings in it; labels resolve through
  the project's label taxonomy).
- ``fix_pair`` — before / after code pairs → findings, patch or
  preference rows.
- ``contrast_functions`` — files with ``bad…`` / ``good…`` functions →
  findings rows (one per function).

Phase H will add custom plugin mappers; Phase F will surface this
catalog in the UI wizard.
"""

from app.services.dataset_import.mappers import bio_to_spans as _bio_to_spans  # noqa: F401
from app.services.dataset_import.mappers import (  # noqa: F401
    chat_messages_passthrough as _chat_messages_passthrough,
)
from app.services.dataset_import.mappers import (  # noqa: F401
    code_label_to_findings as _code_label_to_findings,
)
from app.services.dataset_import.mappers import (  # noqa: F401
    contrast_functions as _contrast_functions,
)
from app.services.dataset_import.mappers import fix_pair as _fix_pair  # noqa: F401
from app.services.dataset_import.mappers import (  # noqa: F401
    kv_to_structured as _kv_to_structured,
)
from app.services.dataset_import.mappers import (  # noqa: F401
    label_to_classification as _label_to_classification,
)
from app.services.dataset_import.mappers import (  # noqa: F401
    preference_pair as _preference_pair,
)
from app.services.dataset_import.mappers import (  # noqa: F401
    qa_pair_passthrough as _qa_pair_passthrough,
)
from app.services.dataset_import.mappers import (  # noqa: F401
    rag_passthrough as _rag_passthrough,
)
from app.services.dataset_import.mappers import text_only as _text_only  # noqa: F401
