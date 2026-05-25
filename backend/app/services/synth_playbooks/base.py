"""Playbook contract + registry (USER-SUCCESS Epic 2).

A Playbook is a small object that knows:
  - which recipe it applies to,
  - which synth mode it implements (paraphrase / hard-negative / …),
  - how to build the LLM prompt,
  - how to parse the LLM response into recipe-shaped rows,
  - how to validate + score each parsed row.

The orchestrator (`synth_playbook_service.run_playbook`) wires a
Playbook to a SynthBackend and writes the accepted rows into the
project's synthetic dataset.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, TypedDict, runtime_checkable


class SynthMode(str, Enum):
    """Modes a playbook can implement.

    v1 ships POSITIVES_PARAPHRASE only. The remaining modes are wired
    by the framework but not yet implemented by any playbook —
    Epic 2b adds the rest.
    """

    POSITIVES_PARAPHRASE = "positives_paraphrase"
    HARD_NEGATIVES = "hard_negatives"
    CLASS_BALANCE_FILL = "class_balance_fill"
    EDGE_CASES = "edge_cases"
    REFUSALS = "refusals"
    FORMAT_ROBUSTNESS = "format_robustness"
    CLUSTER_TARGETED = "cluster_targeted"


class PlaybookContext(TypedDict, total=False):
    recipe_id: str
    project_id: int
    gold_rows: list[dict[str, Any]]
    raw_rows: list[dict[str, Any]] | None
    failure_cluster: dict[str, Any] | None
    target_class: str | None
    target_count: int


class SynthRow(TypedDict):
    payload: dict[str, Any]
    synth_confidence: float
    synth_source: str


class PlaybookResult(TypedDict):
    rows: list[SynthRow]
    backend_used: str
    elapsed_sec: float
    prompt_snippet: str  # first ~280 chars, for debugging


@runtime_checkable
class Playbook(Protocol):
    recipe_id: str
    mode: SynthMode

    def build_prompt(self, ctx: PlaybookContext) -> str: ...

    def parse_output(self, raw_llm_output: str, ctx: PlaybookContext) -> list[dict[str, Any]]: ...

    def validate(self, parsed_rows: list[dict[str, Any]], ctx: PlaybookContext) -> list[SynthRow]: ...


def get_response_schema(playbook: Playbook, ctx: PlaybookContext) -> dict | None:
    """Return the playbook's per-row JSON Schema, if it defines one.

    The schema is forwarded to schema-aware backends (NeMo / NIM) as
    ``response_format=json_schema``. Playbooks that don't need
    structured-output constraints simply don't define
    ``response_schema`` — backends that do support it then fall back
    to free-form generation + parser-only validation.
    """
    fn = getattr(playbook, "response_schema", None)
    if not callable(fn):
        return None
    schema = fn(ctx)
    if not isinstance(schema, dict) or not schema:
        return None
    return schema


# ─────────────────────────────────────────────────────────────────────
# Registry: (recipe_id, mode) → Playbook instance.
# ─────────────────────────────────────────────────────────────────────


_REGISTRY: dict[tuple[str, SynthMode], Playbook] = {}


def register_playbook(playbook: Playbook) -> None:
    """Register a playbook for its (recipe_id, mode) key."""
    key = (playbook.recipe_id, playbook.mode)
    _REGISTRY[key] = playbook


def get_playbook(recipe_id: str, mode: SynthMode) -> Playbook | None:
    """Look up a playbook by (recipe_id, mode). Returns None if not registered."""
    return _REGISTRY.get((recipe_id, mode))


def list_playbooks() -> list[dict[str, str]]:
    """Catalog of all registered playbooks. Used by the API for the
    'what modes are available for my recipe?' query."""
    return [
        {"recipe_id": pb.recipe_id, "mode": pb.mode.value}
        for pb in _REGISTRY.values()
    ]


# ─────────────────────────────────────────────────────────────────────
# Helpers shared across playbook implementations.
# ─────────────────────────────────────────────────────────────────────


def parse_jsonl_lines(raw: str) -> list[dict[str, Any]]:
    """Parse `{...}\n{...}` JSONL output. Robust to extra prose around
    the JSON, code-fence wrappers, and blank lines. Skips
    non-conforming lines silently."""
    import json
    import re

    out: list[dict[str, Any]] = []
    # Strip leading/trailing markdown code fences if the model wrapped
    # its response. We accept both fenced and unfenced output.
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    for line in cleaned.splitlines():
        line = line.strip().rstrip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def short_snippet(text: str, *, limit: int = 280) -> str:
    """First ~limit chars of a string, used for the prompt_snippet
    field of PlaybookResult."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def sample_gold_rows(gold_rows: list[dict[str, Any]], *, count: int, seed: int = 0) -> list[dict[str, Any]]:
    """Deterministic sample of up to `count` gold rows. Used by
    playbooks that include a few-shot example block in their prompt."""
    import random

    if not gold_rows:
        return []
    if len(gold_rows) <= count:
        return list(gold_rows)
    rng = random.Random(seed)
    return rng.sample(gold_rows, count)
