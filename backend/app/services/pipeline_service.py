"""Multi-stage pipelines (G10): on-disk definitions, validation, held-out set and run files.

A pipeline chains projects: stage 1 reads a held-out row's ``input``; stage
k>=2 reads stage k-1's output (optionally fanning out over a list inside it).
Everything lives on disk -- no table, no migration::

    DATA_DIR/pipelines/<pipeline_id>/pipeline.json
    DATA_DIR/pipelines/<pipeline_id>/heldout.jsonl
    DATA_DIR/pipelines/<pipeline_id>/runs/<run_id>.json

Writes are atomic (temp file + ``os.replace``). The runner and the metrics are
in ``pipeline_eval_service``.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.experiment import Experiment
from app.models.project import Project
from app.services import gateway_tools_service as gts
from app.services import servable_run_service as srs
from app.services.json_pointer import split_json_pointer

MAX_STAGES = 8
MAX_HELDOUT_ROWS = 5000
MAX_HELDOUT_BYTES = 50 * 1024 * 1024
STAGE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
PIPELINE_ID_RE = re.compile(r"^pl-[0-9a-f]{8}$")
RUN_ID_RE = re.compile(r"^run-[0-9a-f]{10}$")
_MODEL_SUFFIX_RE = re.compile(r"^(|:base|:run-\d+)$")
INPUT_SOURCES = ("row", "previous", "item")
OUTPUT_KINDS = ("json", "text")
DEFAULT_MAX_TOKENS = 512


class PipelineValidationError(ValueError):
    def __init__(self, errors: list[dict[str, str]]) -> None:
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors) or "invalid pipeline")
        self.errors = errors


class PipelineNotFound(LookupError):
    pass


# ---------------------------------------------------------------- disk


def pipelines_root() -> Path:
    return Path(settings.DATA_DIR) / "pipelines"


def pipeline_dir(pipeline_id: str) -> Path:
    if not PIPELINE_ID_RE.fullmatch(str(pipeline_id or "")):
        raise PipelineNotFound(pipeline_id)
    return pipelines_root() / pipeline_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def new_pipeline_id() -> str:
    return "pl-" + secrets.token_hex(4)


def new_run_id() -> str:
    return "run-" + secrets.token_hex(5)


def load_pipeline(pipeline_id: str) -> dict[str, Any]:
    path = pipeline_dir(pipeline_id) / "pipeline.json"
    if not path.is_file():
        raise PipelineNotFound(pipeline_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineNotFound(pipeline_id) from exc
    if not isinstance(data, dict):
        raise PipelineNotFound(pipeline_id)
    return data


def list_pipeline_ids() -> list[str]:
    root = pipelines_root()
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and PIPELINE_ID_RE.fullmatch(p.name) and (p / "pipeline.json").is_file()
    )


def member_project_ids(pipeline: dict[str, Any]) -> list[int]:
    out: list[int] = []
    for stage in pipeline.get("stages") or []:
        pid = stage.get("project_id") if isinstance(stage, dict) else None
        if isinstance(pid, int) and not isinstance(pid, bool) and pid not in out:
            out.append(pid)
    return out


def delete_pipeline(pipeline_id: str) -> None:
    path = pipeline_dir(pipeline_id)
    if not (path / "pipeline.json").is_file():
        raise PipelineNotFound(pipeline_id)
    shutil.rmtree(path)


# ---------------------------------------------------------------- validation


def _pointer_ok(pointer: Any) -> bool:
    return pointer == "" or split_json_pointer(pointer) is not None


async def validate_definition(db: AsyncSession, raw: Any) -> dict[str, Any]:
    """The cleaned definition (no id / timestamps) or ``PipelineValidationError``
    with per-field ``{field, message}`` (stage fields as ``stages[i].<field>``)."""
    errors: list[dict[str, str]] = []

    def err(fld: str, msg: str) -> None:
        errors.append({"field": fld, "message": msg})

    if not isinstance(raw, dict):
        raise PipelineValidationError([{"field": "body", "message": "the pipeline must be a JSON object"}])
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        err("name", "name is required")
        name = ""
    elif len(name.strip()) > 200:
        err("name", "at most 200 characters")
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        err("description", "description must be a string")
        description = ""
    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list) or not 1 <= len(stages_raw) <= MAX_STAGES:
        err("stages", f"a pipeline has 1 to {MAX_STAGES} stages")
        stages_raw = stages_raw[:MAX_STAGES] if isinstance(stages_raw, list) else []

    ids = {
        s.get("project_id") for s in stages_raw
        if isinstance(s, dict) and isinstance(s.get("project_id"), int) and not isinstance(s.get("project_id"), bool)
    }
    projects: dict[int, Project] = {}
    if ids:
        rows = (await db.execute(select(Project).where(Project.id.in_(ids)))).scalars().all()
        projects = {p.id: p for p in rows}

    stages: list[dict[str, Any]] = []
    seen: set[str] = set()
    fan_out_stages: list[int] = []
    for i, st in enumerate(stages_raw):
        pre = f"stages[{i}]"
        if not isinstance(st, dict):
            err(pre, "each stage must be an object")
            continue
        sname = st.get("name")
        if not isinstance(sname, str) or not STAGE_NAME_RE.fullmatch(sname):
            err(f"{pre}.name", "name must match ^[a-z][a-z0-9_]{0,31}$")
            sname = ""
        elif sname in seen:
            err(f"{pre}.name", f"duplicate stage name '{sname}'")
        seen.add(sname)

        pid = st.get("project_id")
        project: Project | None = None
        if isinstance(pid, bool) or not isinstance(pid, int):
            err(f"{pre}.project_id", "project_id must be a project id")
            pid = None
        else:
            project = projects.get(pid)
            if project is None:
                err(f"{pre}.project_id", f"project {pid} not found")

        model = st.get("model") if st.get("model") is not None else ""
        if not isinstance(model, str) or not _MODEL_SUFFIX_RE.fullmatch(model):
            err(f"{pre}.model", "model must be '', ':base' or ':run-<id>'")
            model = ""
        elif project is not None and model == ":base" and not str(project.base_model_name or "").strip():
            err(f"{pre}.model", "this project has no base model")
        elif project is not None and model.startswith(":run-"):
            exp = await db.get(Experiment, int(model[5:]))
            if exp is None or exp.project_id != project.id:
                err(f"{pre}.model", f"run #{model[5:]} is not a run of project {project.id}")
            else:
                problem = srs.run_problem(exp)
                if problem:
                    err(f"{pre}.model", problem)

        prompt = st.get("prompt")
        clean_prompt: dict[str, Any] = {"mode": "template", "template": ""}
        if not isinstance(prompt, dict) or prompt.get("mode") not in ("recipe", "template"):
            err(f"{pre}.prompt", "prompt must be {mode: 'recipe'} or {mode: 'template', template, system?}")
        elif prompt["mode"] == "recipe":
            clean_prompt = {"mode": "recipe"}
            if project is not None and not gts.recipe_supported(project):
                recipe = gts.project_recipe_id(project) or "(none)"
                err(f"{pre}.prompt", f"project recipe '{recipe}' has no prompt builder; use a template")
        else:
            template = prompt.get("template")
            system = prompt.get("system")
            if not isinstance(template, str) or not template.strip():
                err(f"{pre}.prompt.template", "template is required")
                template = ""
            elif len(template) > gts.MAX_TEMPLATE_CHARS:
                err(f"{pre}.prompt.template", f"at most {gts.MAX_TEMPLATE_CHARS} characters")
            if system is not None and not isinstance(system, str):
                err(f"{pre}.prompt.system", "system must be a string")
                system = None
            clean_prompt = {"mode": "template", "template": template}
            if system:
                clean_prompt["system"] = system

        fan_raw = st.get("fan_out")
        fan_out: dict[str, Any] | None = None
        if fan_raw is not None:
            if i == 0:
                err(f"{pre}.fan_out", "stage 1 cannot fan out (it reads the held-out row)")
            elif not isinstance(fan_raw, dict):
                err(f"{pre}.fan_out", "fan_out must be {pointer, item_key}")
            else:
                ptr = fan_raw.get("pointer")
                keys = fan_raw.get("item_key")
                if not isinstance(ptr, str) or split_json_pointer(ptr) is None:
                    err(f"{pre}.fan_out.pointer", "pointer must be a JSON pointer such as /items/*")
                if not isinstance(keys, list) or not keys or not all(
                    isinstance(k, str) and split_json_pointer(k) is not None for k in keys
                ):
                    err(f"{pre}.fan_out.item_key", "item_key must list one or more JSON pointers")
                fan_out = {"pointer": ptr, "item_key": list(keys) if isinstance(keys, list) else []}
                fan_out_stages.append(i)
                if i > 0 and isinstance(stages_raw[i - 1], dict) and stages_raw[i - 1].get("output", "json") == "text":
                    err(f"{pre}.fan_out", "the previous stage outputs text; fan-out needs JSON")

        inputs_raw = st.get("inputs") if st.get("inputs") is not None else {}
        inputs: dict[str, dict[str, str]] = {}
        if not isinstance(inputs_raw, dict):
            err(f"{pre}.inputs", "inputs must map field names to {from, pointer}")
            inputs_raw = {}
        for fld, spec in inputs_raw.items():
            fpre = f"{pre}.inputs.{fld}"
            if not FIELD_NAME_RE.fullmatch(str(fld)):
                err(fpre, "field names must match ^[A-Za-z_][A-Za-z0-9_]{0,63}$")
                continue
            if not isinstance(spec, dict):
                err(fpre, "each input must be {from, pointer}")
                continue
            src = spec.get("from")
            ptr = spec.get("pointer", "")
            if src not in INPUT_SOURCES:
                err(f"{fpre}.from", "from must be 'row', 'previous' or 'item'")
            elif i == 0 and src != "row":
                err(f"{fpre}.from", "stage 1 can only read the held-out row")
            elif src == "item" and fan_raw is None:
                err(f"{fpre}.from", "'item' needs fan_out on this stage")
            elif (
                src == "previous" and ptr and i > 0 and isinstance(stages_raw[i - 1], dict)
                and stages_raw[i - 1].get("output", "json") == "text"
            ):
                err(f"{fpre}.pointer", "the previous stage outputs text; use the pointer \"\"")
            if not isinstance(ptr, str) or not _pointer_ok(ptr):
                err(f"{fpre}.pointer", "pointer must be \"\" or a JSON pointer such as /items/0")
                ptr = ""
            inputs[str(fld)] = {"from": str(src), "pointer": ptr}

        retrieval = st.get("retrieval")
        if retrieval is not None and not isinstance(retrieval, bool):
            err(f"{pre}.retrieval", "retrieval must be true, false or null")
            retrieval = None
        max_tokens = st.get("max_tokens")
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        elif isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
            err(f"{pre}.max_tokens", "max_tokens must be an integer 1..4096")
            max_tokens = DEFAULT_MAX_TOKENS
        output = st.get("output") if st.get("output") is not None else "json"
        if output not in OUTPUT_KINDS:
            err(f"{pre}.output", "output must be 'json' or 'text'")
            output = "json"

        if clean_prompt.get("mode") == "template" and isinstance(inputs_raw, dict):
            for part in ("template", "system"):
                text = clean_prompt.get(part)
                unknown = sorted(
                    {m for m in gts.template_placeholders(text) if m not in inputs_raw} if isinstance(text, str) else set()
                )
                if unknown:
                    err(
                        f"{pre}.prompt.{part}",
                        "unknown placeholder(s) " + ", ".join("{" + u + "}" for u in unknown)
                        + " -- every {name} must be one of this stage's inputs",
                    )

        stage: dict[str, Any] = {
            "name": sname, "project_id": pid, "model": model, "prompt": clean_prompt, "inputs": inputs,
            "retrieval": retrieval, "max_tokens": max_tokens, "output": output,
        }
        if fan_out is not None:
            stage["fan_out"] = fan_out
        stages.append(stage)

    if len(fan_out_stages) > 1:
        for i in fan_out_stages[1:]:
            err(f"stages[{i}].fan_out", "only one stage per pipeline can fan out")
    if errors:
        raise PipelineValidationError(errors)
    return {"name": name.strip(), "description": (description or "").strip(), "stages": stages}


async def create_pipeline(db: AsyncSession, raw: Any) -> dict[str, Any]:
    clean = await validate_definition(db, raw)
    pid = new_pipeline_id()
    while (pipelines_root() / pid).exists():
        pid = new_pipeline_id()
    now = _now()
    pipeline = {"id": pid, **clean, "created_at": now, "updated_at": now}
    atomic_write_json(pipeline_dir(pid) / "pipeline.json", pipeline)
    return pipeline


async def update_pipeline(db: AsyncSession, pipeline_id: str, raw: Any) -> dict[str, Any]:
    old = load_pipeline(pipeline_id)
    clean = await validate_definition(db, raw)
    pipeline = {"id": pipeline_id, **clean, "created_at": old.get("created_at") or _now(), "updated_at": _now()}
    atomic_write_json(pipeline_dir(pipeline_id) / "pipeline.json", pipeline)
    return pipeline


# ---------------------------------------------------------------- held-out


def fan_out_index(pipeline: dict[str, Any]) -> int | None:
    for i, st in enumerate(pipeline.get("stages") or []):
        if isinstance(st, dict) and st.get("fan_out"):
            return i
    return None


def parse_heldout(pipeline: dict[str, Any], lines: list[tuple[int, Any]]) -> tuple[list[dict], dict[str, list]]:
    """``(accepted, rejected_by_reason)`` for ``(line, raw)`` pairs, where raw is a
    JSONL line (str) or an already-parsed row. Every row is judged on its own:
    valid rows are kept even when others are rejected."""
    from app.services.pipeline_eval_service import key_of

    stage_names = [s["name"] for s in pipeline.get("stages") or []]
    fan = fan_out_index(pipeline)
    listed = set(stage_names[fan:]) if fan is not None else set()
    item_key = list((pipeline["stages"][fan].get("fan_out") or {}).get("item_key") or []) if fan is not None else []
    accepted: list[dict] = []
    rejected: dict[str, list] = {}
    seen: set[str] = set()

    def reject(reason: str, line: int, rid: Any = None) -> None:
        rejected.setdefault(reason, []).append({"line": line, "id": rid})

    for line, raw in lines:
        if isinstance(raw, str):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                reject("invalid_json", line)
                continue
        else:
            row = raw
        if not isinstance(row, dict):
            reject("not_an_object", line)
            continue
        rid = row.get("id")
        if isinstance(rid, bool) or not isinstance(rid, (str, int)) or not str(rid).strip():
            reject("missing_id", line)
            continue
        rid = str(rid).strip()
        if rid in seen:
            reject("duplicate_id", line, rid)
            continue
        if not isinstance(row.get("input"), dict):
            reject("input_not_object", line, rid)
            continue
        gold = row.get("gold") if row.get("gold") is not None else {}
        if not isinstance(gold, dict):
            reject("gold_not_object", line, rid)
            continue
        unknown = [k for k in gold if k not in stage_names]
        if unknown:
            reject("unknown_stage_in_gold", line, rid)
            continue
        if any(name in listed and not isinstance(gold[name], list) for name in gold):
            reject("fan_out_gold_not_list", line, rid)
            continue
        if any(name in listed and not all(isinstance(it, dict) for it in gold[name]) for name in gold):
            reject("fan_out_gold_item_not_object", line, rid)
            continue
        if any(name in listed and any(key_of(it, item_key)[0] is None for it in gold[name]) for name in gold):
            reject("fan_out_gold_item_missing_key", line, rid)
            continue
        if len(accepted) >= MAX_HELDOUT_ROWS:
            reject("over_row_limit", line, rid)
            continue
        seen.add(rid)
        accepted.append({"id": rid, "input": row["input"], "gold": gold})
    return accepted, rejected


def save_heldout(pipeline_id: str, rows: list[dict]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    atomic_write_text(pipeline_dir(pipeline_id) / "heldout.jsonl", text)
    atomic_write_json(pipeline_dir(pipeline_id) / "heldout.meta.json", {"rows": len(rows)})


def heldout_count(pipeline_id: str) -> int:
    """Held-out row count from the small sidecar (the list endpoints never parse the set)."""
    meta = pipeline_dir(pipeline_id) / "heldout.meta.json"
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("rows"), int):
            return data["rows"]
    except (OSError, ValueError):
        pass
    path = pipeline_dir(pipeline_id) / "heldout.jsonl"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def load_heldout(pipeline_id: str) -> list[dict]:
    path = pipeline_dir(pipeline_id) / "heldout.jsonl"
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


# ---------------------------------------------------------------- runs


def run_path(pipeline_id: str, run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(str(run_id or "")):
        raise PipelineNotFound(run_id)
    return pipeline_dir(pipeline_id) / "runs" / f"{run_id}.json"


def summary_path(pipeline_id: str, run_id: str) -> Path:
    return run_path(pipeline_id, run_id).with_suffix(".summary.json")


def save_run(pipeline_id: str, run: dict[str, Any]) -> None:
    """The run file plus its small summary sidecar (what the list endpoints read)."""
    atomic_write_json(run_path(pipeline_id, run["run_id"]), run)
    atomic_write_json(summary_path(pipeline_id, run["run_id"]), run_summary(run))


def load_run(pipeline_id: str, run_id: str) -> dict[str, Any]:
    path = run_path(pipeline_id, run_id)
    if not path.is_file():
        raise PipelineNotFound(run_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineNotFound(run_id) from exc
    return data if isinstance(data, dict) else {}


def run_summary(run: dict[str, Any]) -> dict[str, Any]:
    e2e = run.get("end_to_end") if isinstance(run.get("end_to_end"), dict) else {}
    attribution = run.get("attribution") if isinstance(run.get("attribution"), dict) else {}
    return {
        "id": run.get("run_id"),
        "run_id": run.get("run_id"),
        "status": run.get("status") or "done",
        "job_id": run.get("job_id"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "rows": run.get("rows"),
        "share_fully_correct": e2e.get("share_fully_correct"),
        "first_wrong_stage": attribution.get("first_wrong_stage") or {},
        "truncated": bool(run.get("truncated")),
        "error": run.get("error"),
    }


def list_runs(pipeline_id: str) -> list[dict[str, Any]]:
    """Run summaries, newest first, from the sidecars (a run without one -- written before
    sidecars existed -- is summarised once and gets one)."""
    runs_dir = pipeline_dir(pipeline_id) / "runs"
    if not runs_dir.is_dir():
        return []
    out = []
    for path in runs_dir.glob("run-*.json"):
        if path.name.endswith(".summary.json") or not RUN_ID_RE.fullmatch(path.stem):
            continue
        side = path.with_suffix(".summary.json")
        summary = None
        try:
            summary = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                summary = run_summary(data)
                try:
                    atomic_write_json(side, summary)
                except OSError:
                    pass
        if isinstance(summary, dict):
            out.append(summary)
    out.sort(key=lambda r: str(r.get("started_at") or ""), reverse=True)
    return out


async def mark_stale(db: AsyncSession, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A run still "running" on disk whose Job finished or vanished (or, for a sync run, that is not in
    flight in this process) is shown as ``stale`` -- its runner died before it could record an outcome."""
    from app.models.job import Job, JobStatus
    from app.services.pipeline_eval_service import ACTIVE_SYNC_RUNS

    for run in runs:
        if run.get("status") != "running":
            continue
        job_id = run.get("job_id")
        if isinstance(job_id, int) and not isinstance(job_id, bool):
            job = await db.get(Job, job_id)
            if job is not None and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                continue
        elif run.get("run_id") in ACTIVE_SYNC_RUNS:
            continue
        run["status"] = "stale"
        run["error"] = run.get("error") or "The run stopped without recording a result (its Job ended or is gone)."
    return runs


# ---------------------------------------------------------------- listing


async def list_summary(db: AsyncSession, pipeline: dict[str, Any]) -> dict[str, Any]:
    ids = member_project_ids(pipeline)
    names: dict[int, str] = {}
    if ids:
        rows = (await db.execute(select(Project.id, Project.name).where(Project.id.in_(ids)))).all()
        names = {int(pid): name for pid, name in rows}
    runs = await mark_stale(db, list_runs(pipeline["id"]))
    return {
        "id": pipeline["id"],
        "name": pipeline.get("name"),
        "description": pipeline.get("description") or "",
        "updated_at": pipeline.get("updated_at"),
        "stages": [
            {"name": s.get("name"), "project_id": s.get("project_id"), "project_name": names.get(s.get("project_id")),
             "project_missing": s.get("project_id") not in names}
            for s in pipeline.get("stages") or []
        ],
        "heldout_rows": heldout_count(pipeline["id"]),
        "last_run": runs[0] if runs else None,
    }
