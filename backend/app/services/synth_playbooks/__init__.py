"""Synthetic-data playbook registry (USER-SUCCESS Epic 2).

A **playbook** is a (recipe, mode) pair that knows how to generate
synthetic rows for one specific purpose — paraphrase positives,
balance class distribution, drill into a failure cluster, etc.

v1 ships only the `POSITIVES_PARAPHRASE` mode across all 6 recipes.
Hard-negatives, class-balance-fill, edge-cases, refusals, and
cluster-targeted modes land in Epic 2b.
"""

from __future__ import annotations

from .base import (
    Playbook,
    PlaybookContext,
    PlaybookResult,
    SynthMode,
    SynthRow,
    register_playbook,
    get_playbook,
    list_playbooks,
)
from . import classification_paraphrase  # noqa: F401 — register on import
from . import code_review_paraphrase  # noqa: F401
from . import generic_sft_paraphrase  # noqa: F401
from . import qa_sft_paraphrase  # noqa: F401
from . import span_extraction_paraphrase  # noqa: F401
from . import summarization_paraphrase  # noqa: F401


__all__ = [
    "Playbook",
    "PlaybookContext",
    "PlaybookResult",
    "SynthMode",
    "SynthRow",
    "get_playbook",
    "list_playbooks",
    "register_playbook",
]
