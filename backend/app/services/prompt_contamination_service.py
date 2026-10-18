"""Prompts must never carry a benchmark's own values.

The seed-documents prompt (``kinds``, the genres, the domain hint) and the
span-synthesis prompt (the entity types, the focus types, the project's
labelling conventions) are written by the user and go to a generation
model that copies examples word for word (2026-10-17: the Gretel
benchmark's ids and plates were used as prompt examples; 32 of them landed
591 times in the training rows and the benchmark then measured
memorisation). ``benchmark_contamination_service`` finds that AFTER the
rows exist; this module stops it at the prompt.

``benchmark_gold_index`` collects every gold value of the project's
benchmarks (span entity texts, Q&A / summary answers; values of
``MIN_VALUE_CHARS``+). ``find_benchmark_values`` looks for them in the
prompt fields as whole words (so ``emp`` does not match ``employee``);
``check_prompt_fields`` is the dry run; ``refuse_if_contaminated`` is the
gate the two routes call before starting a Job — it raises
``PromptCarriesBenchmarkValues`` listing the offending values, unless the
request sets ``ignore_benchmark_values`` (a common name in a definition
is a judgement call, and the user makes it with the list in front of them).
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.benchmark_contamination_service import MIN_VALUE_CHARS, gold_values, normalise_value

ERROR_CODE = "PROMPT_CARRIES_BENCHMARK_VALUES"
OVERRIDE_FIELD = "ignore_benchmark_values"
MAX_REPORTED = 25


class PromptCarriesBenchmarkValues(ValueError):
    """Raised by ``refuse_if_contaminated``; ``.report`` is the dry-run block."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        values = ", ".join(f"{o['value']} ({o['type']}, {'/'.join(o['benchmarks'])})" for o in report["offending"][:8])
        more = "" if len(report["offending"]) <= 8 else f" and {len(report['offending']) - 8} more"
        super().__init__(
            f"The prompt carries {len(report['offending'])} value(s) from a benchmark's gold rows: {values}{more}. "
            "A generation model copies examples verbatim, so the benchmark would then measure memorisation. "
            "Describe the shape instead (or invent values), or send ignore_benchmark_values=true if these are not "
            "benchmark values in your text."
        )


def _normalise(text: Any) -> str:
    return normalise_value(str(text or ""))


def build_gold_index(benchmarks: list[tuple[str, str | None, list[dict[str, Any]]]]) -> dict[str, dict[str, Any]]:
    """``(slug, task_profile, rows)`` per benchmark → ``{normalised value:
    {value, type, benchmarks}}``, merged across benchmarks."""
    index: dict[str, dict[str, Any]] = {}
    for slug, profile, rows in benchmarks:
        for raw, kind in gold_values(rows, profile):
            key = normalise_value(raw)
            if len(key) < MIN_VALUE_CHARS:
                continue
            entry = index.get(key)
            if entry is None:
                index[key] = {"value": raw, "type": kind, "benchmarks": [slug]}
            elif slug not in entry["benchmarks"]:
                entry["benchmarks"].append(slug)
    return index


async def benchmark_gold_index(db: AsyncSession, project_id: int) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The project's benchmarks' gold values (and the benchmark slugs scanned)."""
    from app.services.external_benchmark_service import _read_jsonl, benchmark_dir, list_benchmarks, manifest_of

    benchmarks: list[tuple[str, str | None, list[dict[str, Any]]]] = []
    for ds in await list_benchmarks(db, project_id):
        m = manifest_of(ds)
        slug = str(m.get("slug") or ds.name)
        benchmarks.append((slug, m.get("task_profile"), _read_jsonl(benchmark_dir(project_id, slug) / "rows.jsonl")))
    return build_gold_index(benchmarks), [b[0] for b in benchmarks]


def find_benchmark_values(fields: dict[str, Any], index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Every gold value that occurs, as a whole word, in any prompt field
    (``{field name: text}``; a list is joined). Sorted by field then value."""
    texts = {name: _normalise(" ".join(map(str, v)) if isinstance(v, (list, tuple)) else v) for name, v in fields.items()}
    texts = {name: t for name, t in texts.items() if t}
    if not texts or not index:
        return []
    joined = "\n".join(texts.values())
    offending: list[dict[str, Any]] = []
    for key, entry in index.items():
        if key not in joined:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(key) + r"(?!\w)")
        in_fields = [name for name, t in texts.items() if pattern.search(t)]
        if in_fields:
            offending.append({"value": entry["value"], "type": entry["type"], "benchmarks": list(entry["benchmarks"]),
                              "fields": in_fields})
    offending.sort(key=lambda o: (o["fields"][0], o["value"].casefold()))
    return offending


def check_prompt_fields(fields: dict[str, Any], index: dict[str, dict[str, Any]], benchmarks: list[str]) -> dict[str, Any]:
    offending = find_benchmark_values(fields, index)
    return {
        "clean": not offending,
        "offending": offending[:MAX_REPORTED],
        "offending_total": len(offending),
        "values_checked": len(index),
        "benchmarks": benchmarks,
        "fields": sorted(name for name, v in fields.items() if _normalise(" ".join(map(str, v)) if isinstance(v, (list, tuple)) else v)),
    }


async def check_prompt(db: AsyncSession, project_id: int, fields: dict[str, Any]) -> dict[str, Any]:
    """The dry run: ``{clean, offending, values_checked, benchmarks, fields}``."""
    index, benchmarks = await benchmark_gold_index(db, project_id)
    return check_prompt_fields(fields, index, benchmarks)


async def refuse_if_contaminated(db: AsyncSession, project_id: int, fields: dict[str, Any], *, override: bool = False) -> dict[str, Any]:
    """The gate: raises ``PromptCarriesBenchmarkValues`` unless clean or overridden.
    Returns the report either way (callers record it on the Job params)."""
    report = await check_prompt(db, project_id, fields)
    if not report["clean"] and not override:
        raise PromptCarriesBenchmarkValues(report)
    return report


async def span_prompt_fields(db: AsyncSession, project_id: int, entity_types: list[str], focus_types: list[str]) -> dict[str, Any]:
    """What the span-synthesis prompt carries from the user: the types and
    the project's labelling conventions (definitions + custom-pattern
    descriptions), which every row's prompt includes."""
    from app.services.span_custom_patterns_service import load_patterns
    from app.services.span_type_definitions_service import load_definitions

    fields: dict[str, Any] = {"entity_types": list(entity_types or []), "focus_types": list(focus_types or [])}
    definitions = await load_definitions(db, project_id)
    for kind, definition in sorted((definitions or {}).items()):
        fields[f"definition:{kind}"] = definition
    for pattern in await load_patterns(db, project_id):
        if pattern.get("description"):
            fields[f"pattern:{pattern.get('type')}"] = pattern["description"]
    return fields


def http_detail(exc: PromptCarriesBenchmarkValues) -> dict[str, Any]:
    return {"error_code": ERROR_CODE, "message": str(exc),
            "metadata": {"offending": exc.report["offending"], "offending_total": exc.report["offending_total"],
                         "benchmarks": exc.report["benchmarks"], "override_field": OVERRIDE_FIELD}}
