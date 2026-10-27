"""Orchestrator for the synthetic-data playbook framework
(USER-SUCCESS Epic 2).

The orchestrator's job is small and bounded:

    1. Look up the project's recipe.
    2. Look up the (recipe_id, mode) playbook from the registry.
    3. Load the project's gold rows.
    4. Pick a backend (caller's choice or auto).
    5. Build the prompt → call the backend → parse → validate.
    6. Persist the accepted rows into the project's synthetic dataset
       with provenance fields (`synth_source`, `synth_confidence`).
    7. Return a structured PlaybookResult to the caller.

The legacy `synthetic_service` machinery (call_teacher_model,
get_or_create_synthetic_dataset, save_synthetic_batch) stays alive
underneath — we reuse `get_or_create_synthetic_dataset` to land
rows in the right place.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetType
from app.models.project import Project
from app.services import llm_usage
from app.services.dataset_service import _load_records_from_file
from app.services.recipe_service import get_recipe
from app.services.synth_backends import (
    BACKEND_REGISTRY,
    SynthBackend,
    SynthBackendError,
    pick_backend,
)
from app.services.synth_playbooks import (
    Playbook,
    PlaybookContext,
    PlaybookResult,
    SynthMode,
    SynthRow,
    get_playbook,
    get_response_schema,
    list_playbooks,
)
from app.services.synth_playbooks.base import reject_report
from app.config import settings
from app.services.synthetic_service import get_or_create_synthetic_dataset

_LOG = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Helpers — reuse the Epic 1 gold-row loader pattern.
# ─────────────────────────────────────────────────────────────────────


async def _load_gold_rows(
    db: AsyncSession, project_id: int, *, dev_only: bool = False
) -> list[dict[str, Any]]:
    """The project's answer-key rows. ``dev_only`` loads the dev split alone (the code-findings
    playbooks seed from it; the held-out test split is never a seed)."""
    types = [DatasetType.GOLD_DEV] if dev_only else [DatasetType.GOLD_DEV, DatasetType.GOLD_TEST]
    result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id, Dataset.dataset_type.in_(types))
    )
    rows: list[dict[str, Any]] = []
    for dataset in result.scalars():
        if not dataset.file_path:
            continue
        path = Path(dataset.file_path)
        if not path.exists():
            continue
        rows.extend(_load_records_from_file(path))
    return rows


def _row_source_blocked(record: Any, usage: dict, provenance_service: Any) -> bool:
    """True when the row's ``data_source_id`` is an evaluation-only / retrieval-only source."""
    if not isinstance(record, dict):
        return False
    sid = record.get("data_source_id")
    try:
        return sid is not None and usage.get(int(sid)) in (provenance_service.USAGE_EVAL, provenance_service.USAGE_RAG)
    except (TypeError, ValueError):
        return False


DEFAULT_CONTEXT_SAMPLE_SIZE = 20
MAX_DOCUMENT_CHUNKS = 2000

# Reply budget per playbook row / per scenario / per structured plan. A backend that
# states ``max_output_tokens`` (cloud) gets max_tokens sized to what the call asks
# for, up to that ceiling; a fixed 2048 / 4096 truncated multi-row and multi-plan
# replies. Backends without it keep their fixed budgets.
PLAYBOOK_TOKENS_PER_ROW = 1024
GRID_TOKENS_PER_SCENARIO = 600
GRID_TOKENS_PER_PLAN = 6000


def _sized_max_tokens(backend: Any, units: int, per_unit: int, default: int) -> int:
    ceiling = getattr(backend, "max_output_tokens", None)
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling <= 0:
        return default
    return max(default, min(ceiling, max(1, units) * per_unit))


# ── Anthropic Message Batches (``use_batch`` on a cloud:anthropic pin) ──────────


def _batches(backend: Any) -> bool:
    """The backend can take a Message Batch (``use_batch`` on + ``complete_batch``; the
    predicate the relabel / documents flow share). Playbooks batch only when the run
    asks (``run_playbook(use_batch=True)``)."""
    from app.services.span_relabel_service import uses_message_batch

    return uses_message_batch(backend)


def _call(backend: Any, prompt: str, system_prompt: str, max_tokens: int) -> dict[str, Any]:
    """One playbook call, as ``complete`` (direct) and ``complete_batch`` take it."""
    return {"prompt": prompt, "system_prompt": system_prompt, "max_tokens": max_tokens, "temperature": 0.7}


async def _complete_direct(backend: Any, call: dict[str, Any], response_schema: dict | None) -> str:
    args = dict(call)
    prompt = args.pop("prompt")
    # The system prompt repeats across calls: a cloud backend marks it as a cached block (same text).
    return await backend.complete(prompt, **args, response_schema=response_schema, **llm_usage.cache_kwargs(backend))


def _batch_request(backend: Any, call: dict[str, Any]) -> dict[str, Any]:
    if getattr(backend, "supports_prompt_cache", False) and call.get("system_prompt"):
        # What ``complete(cache_system=True)`` sends: the system prompt as the cached block.
        return {**call, "system_prompt": "", "cache_prefix": call["system_prompt"]}
    return dict(call)


async def _complete_as_batch(
    backend: Any,
    calls: list[dict[str, Any]],
    progress: Callable[[float, str], Awaitable[None]] | None,
    *,
    start: float,
    width: float,
    what: str,
) -> tuple[list[str | None], str | None]:
    """Every call in ONE Message Batch: (reply per call, None where the batch didn't
    answer it; the batch's error or None). A failed batch answers nothing (the caller
    retries those calls directly); the hard monthly budget is checked by
    ``complete_batch`` before it submits and ``LlmBudgetExceeded`` propagates. A Job
    cancel noticed while waiting raises ``CancelledError`` (``complete_batch`` then
    cancels the batch)."""
    from app.services import jobs_service

    if not calls:
        return [], None
    handle = jobs_service.current_job_handle.get()
    if handle is not None and await handle.check_cancelled():
        raise asyncio.CancelledError()

    async def _on_status(_message: str, done: int, total: int) -> None:
        if handle is not None and await handle.check_cancelled():
            raise asyncio.CancelledError()
        if progress is not None:
            await progress(
                start + width * done / max(1, total),
                f"{what} · Batch: {done}/{total} requests done (half price; can take minutes to hours)",
            )

    try:
        raws = await backend.complete_batch([_batch_request(backend, c) for c in calls], progress=_on_status)
    except SynthBackendError as exc:
        _LOG.warning("playbook message batch failed, calling directly: %s", exc)
        return [None] * len(calls), str(exc)[:300]
    out = [r if isinstance(r, str) else None for r in list(raws)[: len(calls)]]
    return out + [None] * (len(calls) - len(out)), None


async def _replies(
    backend: Any,
    calls: list[dict[str, Any]],
    response_schema: dict | None,
    *,
    batch: bool,
    progress: Callable[[float, str], Awaitable[None]] | None,
    start: float,
    width: float,
    what: str,
) -> tuple[list[str], str | None]:
    """A reply per call: with ``batch``, ONE Message Batch first and the calls it didn't
    answer directly; otherwise every call directly, in order. A direct call's
    ``SynthBackendError`` propagates."""
    raws: list[str | None] = [None] * len(calls)
    error: str | None = None
    base = start
    if batch and calls:
        raws, error = await _complete_as_batch(backend, calls, progress, start=start, width=width * 0.9, what=what)
        base = start + width * 0.9
    pending = [i for i, raw in enumerate(raws) if raw is None]
    for n, i in enumerate(pending, start=1):
        raws[i] = await _complete_direct(backend, calls[i], response_schema)
        if progress is not None:
            note = " not answered by the batch" if batch else ""
            await progress(base + (start + width - base) * n / len(pending), f"{what}: {n}/{len(pending)} calls{note}")
    return [r or "" for r in raws], error


def batch_usage_fields(backend: Any) -> dict[str, Any]:
    """``usage`` / ``cost_usd`` / ``tokens`` / ``batch`` of a batched run: batch calls are
    priced at the Message Batch rate (half price) and are not seen by the Job's meter, so
    the run reports its own usage list (batch + any direct fallback calls)."""
    from app.services.span_relabel_service import batch_info, usage_block

    out: dict[str, Any] = {"batch": batch_info(backend)}
    usage = usage_block(backend)
    if usage:
        out["usage"] = usage
        out["cost_usd"] = round(sum(u.get("usd") or 0 for u in usage), 4)
        out["tokens"] = sum(int(u.get("total_tokens") or 0) for u in usage)
    return out


async def _prepare_code_context(
    db: AsyncSession, project_id: int, ctx: PlaybookContext, params: dict[str, Any]
) -> None:
    """Fill the seeds the code-findings playbooks draw from (``uses_code_seeds``):

    * ``document_chunks``: code chunks of the project's code documents (git repository /
      website sources; cleaned passages that carry ``path`` + ``start_line``). Documents whose
      source is ``eval``- or ``rag``-only in the licence ledger never seed training data and
      are skipped, the same gate as the documents -> Q&A flow. They are UNLABELLED, so the
      playbooks treat them as assumed-clean candidates.
    * ``taxonomy``: the project's label taxonomy index (None when it has none).
    * ``context_values`` / ``context_column``: up to ``context_sample_size`` distinct values of
      ``params.context_column`` in dataset ``params.context_dataset_id`` (a data-carrying context),
      skipping rows whose ledger source is eval- or rag-only.
    """
    from app.services import label_taxonomy_service, provenance_service
    from app.services.span_relabel_service import load_document_passages

    ctx["taxonomy"] = await label_taxonomy_service.load_taxonomy_index(db, project_id)

    # Answer-key (dev) rows from an evaluation-only / retrieval-only source never seed training data.
    usage_map = await provenance_service.source_usage_map(db, project_id)
    ctx["gold_rows"] = [
        r for r in ctx.get("gold_rows") or []
        if not _row_source_blocked(r, usage_map, provenance_service)
    ]

    excluded = await provenance_service.training_excluded_document_ids(db, project_id)
    passages = provenance_service.drop_excluded_passages(load_document_passages(project_id), excluded)
    chunks = [
        p for p in passages
        if isinstance(p.get("path"), str) and p.get("path")
        and isinstance(p.get("start_line"), int) and not isinstance(p.get("start_line"), bool)
    ]
    ctx["document_chunks"] = chunks[:MAX_DOCUMENT_CHUNKS]

    dataset_id = params.get("context_dataset_id")
    column = params.get("context_column")
    if dataset_id is None and column is None:
        return
    if dataset_id is None or not isinstance(column, str) or not column.strip():
        raise ValueError("params.context_dataset_id and params.context_column must be given together.")
    try:
        dataset_id = int(dataset_id)
    except (TypeError, ValueError):
        raise ValueError("params.context_dataset_id must be a dataset id.") from None
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None or dataset.project_id != project_id:
        raise ValueError(f"Context dataset {dataset_id} not found in this project.")
    if dataset.dataset_type in (DatasetType.GOLD_DEV, DatasetType.GOLD_TEST):
        raise ValueError(
            f"Context dataset {dataset_id} is an answer key (held out for evaluation); "
            "pick a raw, training or other data dataset."
        )
    if not dataset.file_path or not Path(dataset.file_path).exists():
        raise ValueError(f"Context dataset {dataset_id} has no data file.")
    column = column.strip()
    records = _load_records_from_file(Path(dataset.file_path))
    if not any(isinstance(r, dict) and column in r for r in records):
        raise ValueError(f"Column {column!r} is not in context dataset {dataset_id}.")
    usage = usage_map
    seen: dict[str, None] = {}
    for record in records:
        if not isinstance(record, dict) or _row_source_blocked(record, usage, provenance_service):
            continue
        value = record.get(column)
        if isinstance(value, (dict, list)):
            continue
        text = "" if value is None else str(value).strip()
        if text:
            seen.setdefault(text, None)
    if not seen:
        raise ValueError(
            f"Column {column!r} of dataset {dataset_id} has no usable values "
            "(empty, or every row comes from an evaluation-only / retrieval-only source)."
        )
    import random

    try:
        size = max(1, min(100, int(params.get("context_sample_size") or DEFAULT_CONTEXT_SAMPLE_SIZE)))
    except (TypeError, ValueError):
        size = DEFAULT_CONTEXT_SAMPLE_SIZE
    values = list(seen)
    ctx["context_values"] = values if len(values) <= size else sorted(random.Random(0).sample(values, size))
    ctx["context_column"] = column


async def _prepare_document_passages(db: AsyncSession, project_id: int, ctx: PlaybookContext) -> None:
    """Fill ``ctx["document_chunks"]`` with the project's cleaned document passages for playbooks
    that write from documents (``uses_document_passages``): documents whose source is eval- or
    rag-only in the licence ledger are dropped (the documents -> Q&A flow's gate). No passages is
    an error, not an empty success."""
    from app.services import provenance_service
    from app.services.span_relabel_service import load_document_passages

    excluded = await provenance_service.training_excluded_document_ids(db, project_id)
    passages = provenance_service.drop_excluded_passages(load_document_passages(project_id), excluded)
    if not passages:
        raise ValueError(
            "Project has no usable document passages. Add and clean documents first "
            "(documents whose source is evaluation-only or retrieval-only are not used for training data)."
        )
    ctx["document_chunks"] = passages[:MAX_DOCUMENT_CHUNKS]
    # The answer key's held-out rows: a dialogue that asks one of their questions would leak the test set.
    ctx["test_rows"] = await _load_test_rows(db, project_id)


async def _load_test_rows(db: AsyncSession, project_id: int) -> list[dict[str, Any]]:
    result = await db.execute(
        select(Dataset).where(Dataset.project_id == project_id, Dataset.dataset_type == DatasetType.GOLD_TEST)
    )
    rows: list[dict[str, Any]] = []
    for dataset in result.scalars():
        path = Path(dataset.file_path) if dataset.file_path else None
        if path is not None and path.exists():
            rows.extend(_load_records_from_file(path))
    return rows


async def _prepare_remediation_context(
    db: AsyncSession, project_id: int, ctx: PlaybookContext, params: dict[str, Any], *, target_count: int
) -> None:
    """Fill ``ctx['remediation_seeds']`` for ``remediation_from_findings``.

    Seeds come from (a) ``params.findings_dataset_id``, a dataset of THIS project (never an answer
    key; rows from eval- or rag-only ledger sources are skipped) holding findings rows, fix pairs or
    remediation rows, and (b) the GOLD_DEV remediation / findings rows already in ``ctx['gold_rows']``
    (ledger-filtered by ``_prepare_code_context``; GOLD_TEST is never loaded). One seed per finding.
    A seed with no passages of its own gets them from ``retrieve_passages_for_row`` when the project's
    ``runtime_config['remediation']['retrieve_passages']`` allows (``params.retrieve_passages`` overrides;
    default on) and the project has a document index; ``params.top_k`` (else the project's) sets how many.
    """
    from app.services import provenance_service
    from app.services.synth_playbooks._code_remediation_common import (
        MAX_REMEDIATION_SEEDS, build_seeds_from_rows, normalise_passages,
    )

    seed_rejects: list[tuple[str, str]] = []
    seeds = build_seeds_from_rows(ctx.get("gold_rows") or [], "answer_key", seed_rejects)
    raw_id = params.get("findings_dataset_id")
    if raw_id is not None:
        try:
            dataset_id = int(raw_id)
        except (TypeError, ValueError):
            raise ValueError("params.findings_dataset_id must be a dataset id.") from None
        dataset = await db.get(Dataset, dataset_id)
        if dataset is None or dataset.project_id != project_id:
            raise ValueError(f"Findings dataset {dataset_id} not found in this project.")
        if dataset.dataset_type in (DatasetType.GOLD_DEV, DatasetType.GOLD_TEST):
            raise ValueError(
                f"Findings dataset {dataset_id} is an answer key (held out for evaluation); pick a raw or "
                "training dataset. Answer-key dev rows are used as seeds automatically."
            )
        if not dataset.file_path or not Path(dataset.file_path).exists():
            raise ValueError(f"Findings dataset {dataset_id} has no data file.")
        usage = await provenance_service.source_usage_map(db, project_id)
        records = [
            r for r in _load_records_from_file(Path(dataset.file_path))
            if not _row_source_blocked(r, usage, provenance_service)
        ]
        seeds = build_seeds_from_rows(records, "findings_dataset", seed_rejects) + seeds
    from app.services.synth_playbooks.base import record_reject as _record_reject

    for _reason, _detail in seed_rejects:
        _record_reject(ctx, _reason, _detail)
    if not seeds:
        from app.services.synth_playbooks.code_remediation_from_findings import no_seed_message

        raise ValueError(no_seed_message())
    seeds = seeds[: max(40, min(MAX_REMEDIATION_SEEDS, target_count * 2))]

    from app.services import auto_rag_service, code_remediation_prompt

    project = await db.get(Project, project_id)
    recipe_snapshot = getattr(project, "selected_recipe", None)
    ctx["remediation_extra_fields"] = code_remediation_prompt.extra_fields_from_schema(
        recipe_snapshot.get("output_schema") if isinstance(recipe_snapshot, dict) else None
    )
    cfg = await asyncio.to_thread(
        code_remediation_prompt.effective_remediation_config, getattr(project, "runtime_config", None), project_id
    )
    retrieve = params.get("retrieve_passages", cfg["retrieve_passages"])
    top_k = int(params["top_k"]) if params.get("top_k") is not None else cfg["top_k"]
    if retrieve is not False:
        retrieval = auto_rag_service.retrieval_settings(project)
        taxonomy = ctx.get("taxonomy")

        def _fill() -> None:
            for seed in seeds:
                if seed.get("passages"):
                    continue
                entry = taxonomy.entries.get(seed["finding"]["label"]) if taxonomy is not None else None
                row = {k: seed[k] for k in ("code", "language", "path", "start_line", "end_line", "finding")}
                found = code_remediation_prompt.retrieve_passages_for_row(
                    project_id, row, retrieval, top_k, label_title=(entry or {}).get("title"),
                )
                seed["passages"] = normalise_passages(found) or None

        await asyncio.to_thread(_fill)
    ctx["remediation_seeds"] = seeds

_GRID_SYSTEM_PROMPT = (
    "You are generating training data for a model that writes structured plans (JSON documents) from "
    "scenarios. Reply with ONE valid JSON object and nothing else. Use DOUBLE QUOTES around every string. "
    "Never wrap your output in markdown code fences."
)


def structured_plan_bundle(project: Any) -> Any:
    """The project's saved schema bundle (``runtime_config['structured_plan']``) or None."""
    from app.services.json_schema_bundle import SchemaBundleError, bundle_from_config

    rc = getattr(project, "runtime_config", None)
    cfg = rc.get("structured_plan") if isinstance(rc, dict) else None
    try:
        return bundle_from_config(cfg if isinstance(cfg, dict) else None)
    except SchemaBundleError:
        return None


NO_SCHEMA_MESSAGE = (
    "scenario_grid needs the project's plan schema. Save a schema (and its root file) in the "
    "Structured plan settings on the Train page first: every generated plan is validated against it."
)


async def _prepare_grid_context(
    db: AsyncSession, project: Project, ctx: PlaybookContext, params: dict[str, Any], *, dry_run: bool
) -> None:
    """Fill ``grid_cells`` / ``grid_bundle`` / ``grid_schema`` for ``scenario_grid``. Raises
    ValueError for a bad matrix or a missing schema."""
    from app.services.synth_playbooks import _structured_plan_common as common

    problem = common.check_params(params)
    if problem is not None:
        raise ValueError(problem[1])
    bundle = structured_plan_bundle(project)
    if bundle is None:
        raise ValueError(NO_SCHEMA_MESSAGE)
    gp = common.grid_params(params)
    matrix = common.parse_matrix(gp["matrix_text"], gp["matrix_format"])
    max_cells = 1 if dry_run else gp["max_cells"]
    cells = common.build_grid(matrix["dimensions"], max_cells, gp["seed"])
    ctx["grid_cells"] = [{"index": i, "values": c} for i, c in enumerate(cells)]
    ctx["grid_bundle"] = bundle
    ctx["grid_schema"] = bundle.dereferenced()
    ctx["grid_seen"] = set()


async def _run_scenario_grid(
    project: Project, playbook: Any, ctx: PlaybookContext, selected_backend: SynthBackend,
    *, dry_run: bool, progress: Callable[[float, str], Awaitable[None]] | None, batch: bool = False,
) -> tuple[str, str, list[SynthRow], float]:
    """The two teacher stages of ``scenario_grid`` (scenarios per batch of cells, retrieval-grounding,
    then plans per batch of scenarios). Returns ``(first prompt, last raw reply, accepted rows, elapsed)``.
    With ``batch`` each stage's calls go as ONE Message Batch (calls it didn't answer go direct)."""
    from app.services.synth_playbooks import _structured_plan_common as common
    from app.services.synth_playbooks.structured_plan_scenario_grid import answer_batches

    started = time.monotonic()
    cells = ctx["grid_cells"]
    batches = [cells[i : i + common.CELLS_PER_CALL] for i in range(0, len(cells), common.CELLS_PER_CALL)]
    per_cell = common.grid_params(ctx.get("params"))["scenarios_per_cell"]

    async def _report(fraction: float, message: str) -> None:
        if progress is not None:
            await progress(fraction, message)

    # Stage 1: scenarios (every prompt depends on its cells alone, so all are built first)
    scenarios: list[dict[str, Any]] = []
    ctx["grid_stage"] = common.STAGE_SCENARIOS
    calls: list[dict[str, Any]] = []
    for cell_batch in batches:
        ctx["grid_batch"] = cell_batch
        calls.append(_call(
            selected_backend, playbook.build_prompt(ctx), _GRID_SYSTEM_PROMPT,
            _sized_max_tokens(selected_backend, len(cell_batch) * per_cell, GRID_TOKENS_PER_SCENARIO, 4096),
        ))
    first_prompt = calls[0]["prompt"] if calls else ""
    raws, _err = await _replies(
        selected_backend, calls, get_response_schema(playbook, ctx), batch=batch, progress=_report,
        start=0.0, width=0.4, what="Scenarios",
    )
    raw = raws[-1] if raws else ""
    for cell_batch, reply in zip(batches, raws):
        ctx["grid_batch"] = cell_batch
        scenarios.extend(playbook.accept_scenarios(playbook.parse_output(reply, ctx), ctx))
    ctx.pop("grid_batch", None)
    if not scenarios:
        return first_prompt, raw, [], time.monotonic() - started

    # Grounding: passages per scenario when retrieval is effective and the project has an index.
    from app.services import structured_plan_prompt as spp

    cfg = await asyncio.to_thread(
        spp.effective_structured_plan_config, getattr(project, "runtime_config", None), project.id
    )
    if cfg.get("retrieve_passages"):
        top_k = int(cfg.get("top_k") or 4)

        def _fill() -> None:
            for item in scenarios:
                found = spp.retrieve_passages_for_scenario(project.id, item["scenario"], top_k)
                item["passages"] = common.normalise_passages(found)

        await _report(0.45, "Retrieving passages for the scenarios")
        await asyncio.to_thread(_fill)

    # Stage 2: answers
    tasks = [
        {"scenario": s["scenario"], "passages": s.get("passages") or [], "cell": s["cell"]} for s in scenarios
    ]
    ctx["grid_stage"] = common.STAGE_ANSWERS
    accepted: list[SynthRow] = []
    # A backend with an output ceiling gets only as many plans per call as fit it.
    ceiling = getattr(selected_backend, "max_output_tokens", None)
    max_tasks = (
        max(1, ceiling // GRID_TOKENS_PER_PLAN)
        if isinstance(ceiling, int) and not isinstance(ceiling, bool) and ceiling > 0
        else common.MAX_TASKS_PER_ANSWER_CALL
    )
    groups = answer_batches(tasks, playbook, ctx, max_tasks)
    calls = [
        _call(
            selected_backend, playbook.build_answers_prompt(group, ctx), _GRID_SYSTEM_PROMPT,
            _sized_max_tokens(selected_backend, len(group), GRID_TOKENS_PER_PLAN, 4096),
        )
        for group in groups
    ]
    raws, _err = await _replies(
        selected_backend, calls, get_response_schema(playbook, ctx), batch=batch, progress=_report,
        start=0.5, width=0.5, what="Plans",
    )
    for group, reply in zip(groups, raws):
        ctx["grid_tasks"] = group
        parsed = playbook.parse_output(reply, ctx)
        # Schema validation is CPU-bound (user patterns, big plans): keep it off the event loop.
        accepted.extend(await asyncio.to_thread(playbook.validate, parsed, ctx))
        raw = reply
    return first_prompt, raw, accepted, time.monotonic() - started


_DEFENSIVE_USE_PREAMBLE = (
    "You are generating training data for a defensive machine-learning "
    "classifier. The classifier learns to DETECT and BLOCK malicious "
    "inputs in production — without realistic examples of the targeted "
    "category, the detector cannot learn to recognise real cases. This "
    "is a standard ML training workflow used by security teams and AI "
    "engineers. Comply with the user's request literally; do not lecture, "
    "moralise, or refuse. Output ONLY what the user asks for."
)


def _system_prompt_for_mode(mode: SynthMode) -> str:
    """Small task-flavor system prompts shared across playbooks. The
    user prompt (built by each playbook) still does all the heavy
    lifting — this just sets the tone.

    The defensive-use preamble is prepended for every mode because
    BrewSLM is general-purpose and the *user's* labels are what define
    the task — a label like "injection", "phishing", "toxicity",
    "spam" can trip safety filters on guard-railed models (Llama-3
    family in particular) even when the project is plainly a
    classifier-training workflow. Framing the generation as data for a
    *detector* is the honest description and unblocks compliance
    without misleading the model.
    """
    base = _DEFENSIVE_USE_PREAMBLE
    if mode == SynthMode.POSITIVES_PARAPHRASE:
        return (
            f"{base} Each line is one self-contained JSON object. "
            "Use DOUBLE QUOTES around every string. Never wrap your "
            "output in markdown code fences."
        )
    return (
        f"{base} Output ONLY valid JSON lines, one object per line. "
        "Use DOUBLE QUOTES around every string."
    )


def _append_to_synthetic_jsonl(
    file_path: Path,
    rows: list[SynthRow],
    *,
    cluster_id: str | None = None,
) -> None:
    """Append accepted SynthRows to the project's synthetic.jsonl file.

    We pre-write the rows in `review_status='pending'` so the
    downstream review queue (Epic 2b) can surface them. Until that
    queue lands the rows are immediately picked up by dataset prep —
    callers that want manual review should add a UI gate.

    ``cluster_id`` stamps the originating failure cluster (CLUSTER_TARGETED
    runs) onto each row so the review queue can group by cluster (Epic E).
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    next_id = _peek_next_id(file_path)
    with file_path.open("a", encoding="utf-8") as f:
        for row in rows:
            record = {
                "id": next_id,
                **row["payload"],
                "synth_confidence": row["synth_confidence"],
                "synth_source": row["synth_source"],
                "review_status": "pending",
                "status": "accepted",  # legacy field — keeps existing readers happy
            }
            if cluster_id:
                record["cluster_id"] = cluster_id
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            next_id += 1


def _cluster_id_from(failure_cluster: dict[str, Any] | None) -> str | None:
    """Pull the ``cluster_id`` off a failure-cluster dict (CLUSTER_TARGETED
    runs), normalised to a non-empty string or ``None``. The clusters emitted
    by ``failure_cluster_service`` carry ``cluster_id`` (e.g. ``"cluster-1"``)."""
    if not isinstance(failure_cluster, dict):
        return None
    raw = failure_cluster.get("cluster_id")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _peek_next_id(file_path: Path) -> int:
    """Read the last id in the JSONL file (if any) and return the next one."""
    if not file_path.exists():
        return 1
    last_id = 0
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("id"), int):
                last_id = max(last_id, obj["id"])
    return last_id + 1


# ─────────────────────────────────────────────────────────────────────
# Public entry point.
# ─────────────────────────────────────────────────────────────────────


async def run_playbook(
    db: AsyncSession,
    project_id: int,
    mode: SynthMode,
    *,
    target_count: int = 30,
    target_class: str | None = None,
    failure_cluster: dict[str, Any] | None = None,
    backend: str | None = None,
    backend_override: SynthBackend | None = None,
    dry_run: bool = False,
    progress: Callable[[float, str], Awaitable[None]] | None = None,
    params: dict[str, Any] | None = None,
    use_batch: bool = False,
) -> PlaybookResult:
    """Run the playbook matching (project's recipe, mode) and persist
    accepted rows into the project's synthetic dataset.

    Raises ValueError when the project / recipe / playbook isn't found
    (callers translate to 4xx at the API layer). Raises SynthBackendError
    when no backend is available.

    The `backend_override` arg lets tests inject a fake backend without
    going through the registry.

    A backend that declares ``playbook_rows_per_call`` (the command backend)
    is asked for at most that many rows per call: a larger ``target_count``
    runs as chunks, each checkpointed on disk so a re-run reuses finished
    chunks (see ``_run_chunked``). ``progress(fraction, message)`` is
    published per chunk. Other backends make ONE call for all rows.

    ``use_batch`` with a backend that can batch (``uses_message_batch``:
    Anthropic Message Batches, half price) sends every not-yet-checkpointed chunk — or both
    scenario_grid stages — as ONE batch each, then calls what the batch
    didn't answer directly; the result adds ``usage`` / ``cost_usd`` /
    ``tokens`` at the batch price and ``batch``. A dry run never batches.
    """
    batch_spend: dict[str, Any] = {}
    try:
        return await _run_playbook(
            db, project_id, mode, target_count=target_count, target_class=target_class,
            failure_cluster=failure_cluster, backend=backend, backend_override=backend_override,
            dry_run=dry_run, progress=progress, params=params, use_batch=use_batch, batch_spend=batch_spend,
        )
    except BaseException as exc:
        # A batched run's calls are paid for even when it fails or is cancelled: hang them
        # on the exception so the FAILED / CANCELLED Job still reports the spend.
        if batch_spend.get("backend") is not None:
            llm_usage.attach_result_usage(exc, batch_usage_fields(batch_spend["backend"]))
        raise


async def _run_playbook(
    db: AsyncSession,
    project_id: int,
    mode: SynthMode,
    *,
    target_count: int = 30,
    target_class: str | None = None,
    failure_cluster: dict[str, Any] | None = None,
    backend: str | None = None,
    backend_override: SynthBackend | None = None,
    dry_run: bool = False,
    progress: Callable[[float, str], Awaitable[None]] | None = None,
    params: dict[str, Any] | None = None,
    use_batch: bool = False,
    batch_spend: dict[str, Any] | None = None,
) -> PlaybookResult:
    """``run_playbook``'s body; a batched run records its backend in ``batch_spend``."""
    if target_count < 1 or target_count > 500:
        raise ValueError("target_count must be between 1 and 500")

    project = await db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    selected_recipe = project.selected_recipe or {}
    recipe_id = selected_recipe.get("recipe_id")
    if not recipe_id:
        raise ValueError("Project has no task type selected")
    if get_recipe(recipe_id) is None:
        raise ValueError(f"Task type '{recipe_id}' not found in the catalog")

    playbook = get_playbook(recipe_id, mode)
    if playbook is None:
        raise ValueError(
            f"No playbook registered for (task type={recipe_id}, mode={mode.value}). "
            f"v1 ships POSITIVES_PARAPHRASE only — Epic 2b adds the rest."
        )

    # code-findings playbooks can seed from code documents instead of an answer key, and from
    # the dev split only: the held-out test split never becomes a training prompt.
    uses_code_seeds = bool(getattr(playbook, "uses_code_seeds", False))
    uses_grid = bool(getattr(playbook, "uses_scenario_grid", False))
    uses_documents = bool(getattr(playbook, "uses_document_passages", False))
    gold_rows = [] if uses_documents else await _load_gold_rows(db, project_id, dev_only=uses_code_seeds or uses_grid)
    if not gold_rows and not uses_code_seeds and not uses_grid and not uses_documents:
        raise ValueError(
            "Project has no answer-key rows. Import an answer key or instantiate "
            "from a project template first."
        )

    ctx: PlaybookContext = {
        "recipe_id": recipe_id,
        "project_id": project_id,
        "gold_rows": gold_rows,
        "raw_rows": None,
        "failure_cluster": failure_cluster,
        "target_class": target_class,
        "target_count": target_count,
        "params": dict(params or {}),
        "rejects": [],
    }
    if uses_code_seeds:
        await _prepare_code_context(db, project_id, ctx, ctx["params"])
        if getattr(playbook, "uses_remediation_seeds", False):
            await _prepare_remediation_context(db, project_id, ctx, ctx["params"], target_count=target_count)
        # Varies the seed window from run to run (stable within a run, so a resumed run agrees).
        ctx["seed_nonce"] = _peek_next_id(settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / "synthetic.jsonl")

    if uses_documents:
        await _prepare_document_passages(db, project_id, ctx)
        ctx["seed_nonce"] = _peek_next_id(settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / "synthetic.jsonl")

    if uses_grid:
        # The grid is built from params.matrix_text alone: the answer key is never a seed.
        ctx["gold_rows"] = []
        await _prepare_grid_context(db, project, ctx, ctx["params"], dry_run=dry_run)

    selected_backend = backend_override or pick_backend(backend, registry=BACKEND_REGISTRY)

    # Message Batch (half price): only when the caller asked (the command backend
    # packs calls by default, which playbooks don't use), never for a dry run.
    if use_batch and not dry_run and not _batches(selected_backend):
        raise ValueError(
            f"use_batch needs a backend that takes Message Batches (cloud:anthropic:<model>); "
            f"{selected_backend.describe()} can't."
        )
    batching = use_batch and not dry_run
    if batching and batch_spend is not None:
        batch_spend["backend"] = selected_backend
    rows_per_call = getattr(selected_backend, "playbook_rows_per_call", None)
    if not (isinstance(rows_per_call, int) and not isinstance(rows_per_call, bool) and rows_per_call > 0):
        rows_per_call = target_count if batching else None
    # A batched run is always chunked (one batch request per chunk, checkpointed), even one chunk.
    chunked = (
        rows_per_call is not None and (target_count > rows_per_call or batching) and not dry_run
        and not uses_grid
    )
    chunk_meta: dict[str, Any] = {}
    checkpoint_path: Path | None = None
    if uses_grid:
        prompt, raw_llm_output, accepted_rows, elapsed = await _run_scenario_grid(
            project, playbook, ctx, selected_backend, dry_run=dry_run, progress=progress, batch=batching,
        )
        if not accepted_rows and not dry_run:
            summary = reject_report(ctx.get("rejects"))["reject_summary"]
            detail = ", ".join(f"{k}: {v}" for k, v in sorted(summary.items())) or "the teacher returned no usable output"
            raise ValueError(
                f"scenario_grid produced 0 accepted rows ({detail}). Check that the schema is satisfiable "
                "and that the teacher model follows the reply format."
            )
    elif chunked:
        prompt, raw_llm_output, accepted_rows, elapsed, chunk_meta, checkpoint_path = await _run_chunked(
            playbook, ctx, mode, selected_backend, int(rows_per_call), progress=progress,
            recipe_id=recipe_id, target_class=target_class, failure_cluster=failure_cluster, batch=batching,
        )
    else:
        prompt = playbook.build_prompt(ctx)
        # Per-row JSON Schema (Phase 5b): schema-aware backends (NeMo / NIM)
        # consume it via response_format; others silently ignore — the
        # playbook parser still validates structure either way.
        response_schema = get_response_schema(playbook, ctx)
        started_at = time.monotonic()
        raw_llm_output = await selected_backend.complete(
            prompt,
            system_prompt=_system_prompt_for_mode(mode),
            max_tokens=_sized_max_tokens(selected_backend, target_count, PLAYBOOK_TOKENS_PER_ROW, 2048),
            temperature=0.7,
            response_schema=response_schema,
            # The mode's system prompt repeats across playbook runs: a cloud
            # backend marks it as a cached block (same text).
            **llm_usage.cache_kwargs(selected_backend),
        )
        elapsed = time.monotonic() - started_at

        parsed_rows = playbook.parse_output(raw_llm_output, ctx)
        accepted_rows = playbook.validate(parsed_rows, ctx)

    if uses_documents and not accepted_rows and not dry_run:
        summary = reject_report(ctx.get("rejects"))["reject_summary"]
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(summary.items())) or "the teacher returned no usable output"
        raise ValueError(f"{mode.value} produced 0 accepted dialogues ({detail}).")

    # Persist into the project's synthetic dataset. We ensure the
    # Dataset row exists (legacy helper) and then append to the
    # synthetic.jsonl file directly so we control the row provenance.
    # The Dataset.file_path field is set lazily by the legacy save
    # flow, so we derive the canonical path ourselves to avoid a
    # circular dependency on whether that flow has fired yet.
    #
    # ``dry_run=True`` skips persistence — the caller wants to know
    # whether the (playbook, model) combo *would* produce usable
    # rows before committing to a full async job. See the
    # ``/synthetic/run-playbook/dry-run`` endpoint.
    if accepted_rows and not dry_run:
        synthetic_ds = await get_or_create_synthetic_dataset(db, project_id)
        synthetic_dir = settings.DATA_DIR / "projects" / str(project_id) / "synthetic"
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        file_path = synthetic_dir / "synthetic.jsonl"
        # Stamp the originating failure cluster so the review queue can group
        # cluster-targeted rows by cluster (Epic E).
        _append_to_synthetic_jsonl(
            file_path, accepted_rows, cluster_id=_cluster_id_from(failure_cluster),
        )
        # Bump the record_count + ensure file_path is stamped on the
        # Dataset row for downstream readers.
        synthetic_ds.record_count = (synthetic_ds.record_count or 0) + len(accepted_rows)
        if not synthetic_ds.file_path:
            synthetic_ds.file_path = str(file_path)
        await db.flush()
    if checkpoint_path is not None and not dry_run:
        # The rows are persisted (file appended, Dataset count committed
        # here): only now is the checkpoint's job done. Deleting it earlier
        # would lose finished chunks if the commit failed; it is reused for
        # up to 7 days otherwise (older ones are pruned).
        if accepted_rows:
            await db.commit()
        checkpoint_path.unlink(missing_ok=True)

    from .synth_playbooks.base import short_snippet  # local import — keeps top-level light

    result: dict[str, Any] = {
        "rows": accepted_rows,
        "backend_used": selected_backend.describe(),
        "elapsed_sec": round(elapsed, 3),
        "prompt_snippet": short_snippet(prompt),
        "raw_llm_snippet": short_snippet(raw_llm_output),
        "refusal_detected": _looks_like_refusal(raw_llm_output),
    }
    result.update(reject_report(ctx.get("rejects")))
    if chunk_meta:
        result.update(chunk_meta)
    if batching:
        result.update(batch_usage_fields(selected_backend))
    return result  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────
# Chunked + checkpointed generation (backends with playbook_rows_per_call)
# ─────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = "playbook_checkpoints"
CHECKPOINT_MAX_AGE_SECONDS = 7 * 24 * 3600
_DEDUP_FIELDS = ("text", "input", "prompt", "question", "source", "instruction", "message")


def chunk_sizes(target_count: int, rows_per_call: int) -> list[int]:
    """Split ``target_count`` into the fewest chunks of at most ``rows_per_call``, evenly."""
    n = max(1, math.ceil(target_count / max(1, rows_per_call)))
    base, extra = divmod(target_count, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


def checkpoint_path_for(
    project_id: int, recipe_id: str, mode: SynthMode, backend_describe: str,
    target_count: int, target_class: str | None, failure_cluster: dict[str, Any] | None,
    prompt_hash: str = "",
) -> Path:
    """Checkpoint file for one run's parameters. ``prompt_hash`` (sha256 of the
    first chunk's built prompt) ties it to the answer key / examples the prompt
    was built from: change them and an old checkpoint is no longer reused."""
    key = json.dumps(
        [recipe_id, mode.value, backend_describe, target_count, target_class, failure_cluster, prompt_hash],
        sort_keys=True, default=str, ensure_ascii=False,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / CHECKPOINT_DIR / f"{digest}.jsonl"


def _prune_stale_checkpoints(directory: Path) -> None:
    """Remove checkpoints older than 7 days (a stale one is never reused)."""
    try:
        cutoff = time.time() - CHECKPOINT_MAX_AGE_SECONDS
        for f in directory.glob("*.jsonl"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except OSError:
        pass


def _load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return done
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return done
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # a torn last line from a crash
        if isinstance(entry, dict) and isinstance(entry.get("chunk"), int) and isinstance(entry.get("rows"), list):
            done[entry["chunk"]] = entry
    return done


def _append_checkpoint(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _row_key(row: SynthRow) -> str:
    payload = row.get("payload") or {}
    if payload.get("row_kind") == "findings" and isinstance(payload.get("code"), str):
        # Same code with different findings is a different example.
        findings = json.dumps(payload.get("findings_json"), sort_keys=True, ensure_ascii=False)
        return re.sub(r"\s+", " ", payload["code"]).strip().lower() + "\n" + findings
    for field in _DEDUP_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return re.sub(r"\s+", " ", value).strip().lower()
    return re.sub(r"\s+", " ", json.dumps(payload, sort_keys=True, ensure_ascii=False)).strip().lower()


async def _run_chunked(
    playbook: Playbook,
    ctx: PlaybookContext,
    mode: SynthMode,
    backend: SynthBackend,
    rows_per_call: int,
    *,
    progress: Callable[[float, str], Awaitable[None]] | None,
    recipe_id: str,
    target_class: str | None,
    failure_cluster: dict[str, Any] | None,
    batch: bool = False,
) -> tuple[str, str, list[SynthRow], float, dict[str, Any], Path]:
    """Generate ``ctx['target_count']`` rows as chunks of ≤ ``rows_per_call``.

    Each chunk is its own backend call (prompt = the playbook's prompt for the
    chunk's row count + "Batch k of n: vary topics…"); its accepted rows are
    appended (fsynced) to a checkpoint file keyed by the run's parameters, and
    a re-run with the same parameters reuses finished chunks instead of
    calling them again. A chunk that fails (after the backend's own retries)
    or yields no rows is recorded in ``failed_chunks`` and the run goes on;
    zero rows overall is the caller's error to raise. Exact-duplicate rows
    across chunks are dropped. Returns (last prompt, last raw output,
    rows, elapsed seconds, result metadata, checkpoint path).

    With ``batch`` every chunk not in the checkpoint goes as ONE Message Batch
    first (one request per chunk); each reply is parsed, validated and
    checkpointed exactly as a direct reply, in chunk order, and the chunks the
    batch didn't answer are then called directly. The checkpoint key ignores
    ``batch``: a killed batched run resumes either way."""
    from app.services import jobs_service

    target_count = int(ctx["target_count"])
    sizes = chunk_sizes(target_count, rows_per_call)
    n = len(sizes)
    first_prompt = playbook.build_prompt({**ctx, "target_count": sizes[0], "chunk_index": 0}) + (  # type: ignore[typeddict-item]
        f"\n\nBatch 1 of {n}: vary topics and wording; do not repeat earlier batches."
    )
    path = checkpoint_path_for(
        int(ctx["project_id"]), recipe_id, mode, backend.describe(), target_count, target_class, failure_cluster,
        prompt_hash=hashlib.sha256(first_prompt.encode("utf-8")).hexdigest(),
    )
    _prune_stale_checkpoints(path.parent)
    saved = _load_checkpoint(path)

    rows: list[SynthRow] = []
    seen: set[str] = set()
    failed: list[dict[str, Any]] = []
    resumed = 0
    last_prompt = ""
    last_raw = ""
    started_at = time.monotonic()
    rejects = ctx["rejects"] if isinstance(ctx.get("rejects"), list) else None

    def _take(chunk_rows: list[SynthRow]) -> None:
        for row in chunk_rows:
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    def _build(k: int, size: int) -> tuple[PlaybookContext, dict[str, Any], list[dict[str, Any]]]:
        """(chunk ctx, call, rejects the prompt build recorded)."""
        chunk_ctx: PlaybookContext = {**ctx, "target_count": size, "chunk_index": k}  # type: ignore[typeddict-item]
        mark = len(rejects) if rejects is not None else 0
        prompt = playbook.build_prompt(chunk_ctx) + (
            f"\n\nBatch {k + 1} of {n}: vary topics and wording; do not repeat earlier batches."
        )
        call = _call(backend, prompt, _system_prompt_for_mode(mode), _sized_max_tokens(backend, size, PLAYBOOK_TOKENS_PER_ROW, 2048))
        return chunk_ctx, call, (list(rejects[mark:]) if rejects is not None else [])

    def _accept(k: int, chunk_ctx: PlaybookContext, raw: str, build_rejects: list[dict[str, Any]]) -> None:
        from .synth_playbooks.base import short_snippet

        mark = len(rejects) if rejects is not None else 0
        accepted = playbook.validate(playbook.parse_output(raw, chunk_ctx), chunk_ctx)
        if accepted:
            chunk_rejects = build_rejects + (list(rejects[mark:]) if rejects is not None else [])
            _append_checkpoint(path, {
                "chunk": k, "rows": accepted, "raw_snippet": short_snippet(raw), "rejects": chunk_rejects,
            })
            _take(list(accepted))
        else:
            failed.append({"chunk": k + 1, "reason": "no rows accepted from the reply"})

    built: dict[int, tuple[PlaybookContext, dict[str, Any], list[dict[str, Any]]]] = {}
    batch_raws: dict[int, str | None] = {}
    batch_meta: dict[str, Any] = {}
    if batch:
        todo = [k for k in range(n) if k not in saved]
        for k in todo:
            built[k] = _build(k, sizes[k])
        if todo:
            if progress is not None and len(todo) < n:
                await progress(0.0, f"Resuming: {n - len(todo)}/{n} chunks already written")
            raws, error = await _complete_as_batch(
                backend, [built[k][1] for k in todo], progress, start=0.0, width=0.9, what=f"{len(todo)} chunks",
            )
            batch_raws = dict(zip(todo, raws))
            batch_meta = {
                "batch_requests": len(todo),
                "batch_answered": sum(1 for r in raws if r is not None),
                **({"batch_error": error} if error else {}),
            }

    for k, size in enumerate(sizes):
        handle = jobs_service.current_job_handle.get()
        if handle is not None and await handle.check_cancelled():
            raise asyncio.CancelledError()
        if k in saved:
            _take(list(saved[k]["rows"]))
            saved_rejects = saved[k].get("rejects")
            if isinstance(saved_rejects, list) and rejects is not None:
                rejects.extend(r for r in saved_rejects if isinstance(r, dict))
            resumed += 1
            last_raw = str(saved[k].get("raw_snippet") or last_raw)
        else:
            chunk_ctx, call, build_rejects = built.get(k) or _build(k, size)
            last_prompt = call["prompt"]
            raw = batch_raws.get(k)
            if raw is None:
                try:
                    raw = await _complete_direct(backend, call, get_response_schema(playbook, chunk_ctx))
                except SynthBackendError as exc:
                    failed.append({"chunk": k + 1, "reason": str(exc)[:300]})
            if raw is not None:
                last_raw = raw
                _accept(k, chunk_ctx, raw, build_rejects)
        if progress is not None:
            fraction = (0.9 + 0.1 * (k + 1) / n) if batch_meta else (k + 1) / n
            await progress(fraction, f"chunk {k + 1}/{n} · {len(rows)} rows so far")
    meta: dict[str, Any] = {"chunks": n, "resumed_chunks": resumed, "failed_chunks": failed, **batch_meta}
    return last_prompt, last_raw, rows, time.monotonic() - started_at, meta, path


# Phrases that almost always indicate a guard-rail refusal rather than
# malformed-but-attempting-JSON output. Kept short + lowercase; the
# detector also requires the response to be short (< 600 chars) AND
# contain no '{' at all, so legitimate JSON output that happens to
# include the word "cannot" doesn't trip the detector.
_REFUSAL_PHRASES: tuple[str, ...] = (
    "i cannot",
    "i can't",
    "i'm not able",
    "i am not able",
    "i won't",
    "i will not",
    "as an ai",
    "as a language model",
    "i'm sorry",
    "i apologize",
    "violates",
    "against my",
    "i cannot generate",
    "i cannot provide",
    "i cannot help",
    "harmful or",
    "malicious or",
    "unable to assist",
    "unable to comply",
    "ethical guidelines",
)


def _looks_like_refusal(raw: str) -> bool:
    """Heuristic: the LLM refused on safety/guardrail grounds.

    Triggered when (a) the response is short, (b) contains no JSON
    object at all (no '{'), and (c) contains a known refusal phrase.
    Conservative on purpose — we'd rather miss a refusal than
    misclassify a valid (but malformed) generation as one.
    """
    if not raw:
        return False
    text = raw.strip()
    if len(text) > 600 or "{" in text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def available_playbooks_for_recipe(recipe_id: str) -> list[dict[str, str]]:
    """Catalog of playbooks registered for a specific recipe. Used by
    the UI to show only the modes that have a playbook implementation."""
    return [pb for pb in list_playbooks() if pb["recipe_id"] == recipe_id]
