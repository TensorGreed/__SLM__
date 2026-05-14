"""Built-in source connectors.

Importing this package registers every connector for the orchestrator.
Each connector lives in its own module so adding a new source = one
file (per the architectural rule pinned in DATASET_IMPORT_PLAN.md).
"""

from app.services.dataset_import.sources import csv as _csv  # noqa: F401
from app.services.dataset_import.sources import hf as _hf  # noqa: F401
from app.services.dataset_import.sources import jsonl as _jsonl  # noqa: F401
