"""Detector for ``issue_resolved``: does a finding still appear once the patch is applied?

``runtime_config["remediation"]["detector"]`` is ``None`` (no detector: ``issue_resolved`` is not
evaluated) or

* ``{"kind": "project", "project_id": int, "experiment_id": int | None}`` -- a code-findings
  project's completed run with weights (default: its latest completed non-baseline run). The
  shared ``code_findings_prompt`` builder renders each chunk with that project's labels, the run
  generates greedily through the eval inference path, and the G4 findings parser reads the output.
* ``{"kind": "command", "check_id": str}`` -- an allow-listed code check registered with
  ``kind="detector"`` in the code-check registry. It reads the code and reports
  ``{"findings": [{label, start_line, end_line}]}`` (its ``run`` result may carry a ``findings``
  list, or a JSON message holding that object); it runs wherever the check runs (sandboxed).

The detector runs on the ORIGINAL chunk and the PATCHED chunk, after the remediation model's own
predictions are done (a project detector loads its model only then and frees it right after).
Verdicts are cached per project by sha256 of code + chunk context + detector identity. Detector
errors never fail the eval: the affected rows get ``issue_resolved = None`` and the metrics a
``detector_error`` note.

``annotate_predictions`` attaches ``_detector = {label, original, patched, error}`` to each
prediction; ``remediation_scoring.detector_verdict`` turns it into the per-row flags.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import settings

_LOG = logging.getLogger(__name__)

MAX_DETECTOR_NEW_TOKENS = 512
CACHE_MAX_ENTRIES = 20000


class DetectorUnavailable(Exception):
    """The configured detector cannot run (missing project / run / weights / check)."""


@dataclass
class ResolvedDetector:
    kind: str
    label: str  # "<project name> · run <id>" / "<check id>"
    identity: str  # cache namespace: changes when the detector's weights or labels change
    detect: Callable[[list[dict[str, Any]]], "list[list[str] | None]"]  # chunks -> label lists (None = error)


# ── cache ────────────────────────────────────────────────────────────────


def cache_path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "remediation_detector_cache.json"


class DetectorCache:
    """File-backed ``{key: [labels]}``; greedy decoding makes detector output deterministic.
    Best-effort: load / write errors degrade to an in-memory cache."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._data: dict[str, list[str]] = {}
        self._dirty = False
        try:
            if path is not None and path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data = {k: v for k, v in loaded.items() if isinstance(v, list)}
        except Exception:  # noqa: BLE001
            self._data = {}

    def get(self, key: str) -> list[str] | None:
        value = self._data.get(key)
        return list(value) if isinstance(value, list) else None

    def set(self, key: str, labels: list[str]) -> None:
        self._data[key] = list(labels)
        self._dirty = True
        while len(self._data) > CACHE_MAX_ENTRIES:
            self._data.pop(next(iter(self._data)), None)

    def flush(self) -> None:
        if not self._dirty or self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._data), encoding="utf-8")
            tmp.replace(self._path)
            self._dirty = False
        except Exception:  # noqa: BLE001
            pass


def chunk_key(identity: str, chunk: dict[str, Any]) -> str:
    raw = json.dumps(
        [identity, chunk.get("code"), chunk.get("language"), chunk.get("path"), chunk.get("start_line")],
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


# ── detector identity (cache keys / scoring fingerprint) ─────────────────

_PROBE_ROW = {"code": "x = 1\nprint(x)\n", "language": "python", "path": "probe.py", "start_line": 1}


def detector_recipe_fingerprint(labels: list[str] | None) -> str:
    """Short hash of everything besides the weights that shapes a project detector's output: the
    code-findings prompt builder (rendered on a fixed probe row with the project's labels), the
    decoding settings and the generation budget. A prompt-builder change invalidates cached verdicts."""
    from app.services.code_findings_prompt import build_code_findings_prompt_from_row
    from app.services.decoding_settings import decoding_record, generation_kwargs

    try:
        prompt = build_code_findings_prompt_from_row(dict(_PROBE_ROW), list(labels) if labels else None)
    except Exception:  # noqa: BLE001
        prompt = ""
    try:
        decoding = decoding_record(generation_kwargs(max_new_tokens=MAX_DETECTOR_NEW_TOKENS, temperature=0.0))
    except Exception:  # noqa: BLE001
        decoding = {}
    raw = json.dumps([prompt, decoding, MAX_DETECTOR_NEW_TOKENS], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:10]


def binary_fingerprint(binary: str | None) -> str:
    """``realpath:mtime:size`` of a command detector's binary ("none" when it does not resolve)."""
    import os
    import shutil

    exe = shutil.which(binary) if binary else None
    if not exe:
        return "none"
    try:
        real = os.path.realpath(exe)
        st = os.stat(real)
        return hashlib.sha256(f"{real}:{int(st.st_mtime)}:{st.st_size}".encode()).hexdigest()[:10]
    except OSError:
        return "none"


# ── detector discovery / resolution ──────────────────────────────────────


def _is_code_findings_project(project: Any) -> bool:
    from app.services.code_findings_prompt import project_is_code_findings

    return project_is_code_findings(project)


async def latest_weighted_run(db: Any, project_id: int, experiment_id: int | None = None) -> Any | None:
    """The detector run: ``experiment_id`` when given (must be a completed non-baseline run with
    weights), else the project's latest completed non-baseline run with weights."""
    from sqlalchemy import select

    from app.models.experiment import Experiment, ExperimentStatus
    from app.services.post_training_eval_service import _model_dir_has_weights

    if experiment_id is not None:
        exp = await db.get(Experiment, experiment_id)
        candidates = [exp] if exp is not None and exp.project_id == project_id else []
    else:
        rows = await db.execute(
            select(Experiment)
            .where(Experiment.project_id == project_id, Experiment.status == ExperimentStatus.COMPLETED)
            .order_by(Experiment.completed_at.desc().nullslast(), Experiment.id.desc())
        )
        candidates = list(rows.scalars().all())
    for exp in candidates:
        cfg = exp.config if isinstance(exp.config, dict) else {}
        if cfg.get("is_baseline") is True or exp.status != ExperimentStatus.COMPLETED:
            continue
        if _model_dir_has_weights(exp.output_dir):
            return exp
    return None


def detector_label(project_name: str, experiment_id: int) -> str:
    return f"{project_name} · run {experiment_id}"


async def list_available_detectors(db: Any, accessible_project_ids: set[int] | None = None) -> list[dict[str, Any]]:
    """Code-findings projects with a completed run that has weights."""
    from sqlalchemy import select

    from app.models.project import Project

    rows = await db.execute(select(Project).order_by(Project.id))
    out: list[dict[str, Any]] = []
    for project in rows.scalars().all():
        if accessible_project_ids is not None and project.id not in accessible_project_ids:
            continue
        if not _is_code_findings_project(project):
            continue
        exp = await latest_weighted_run(db, project.id)
        if exp is None:
            continue
        out.append({
            "project_id": project.id,
            "project_name": project.name,
            "experiment_id": exp.id,
            "run_name": exp.name,
            "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        })
    return out


async def describe_detector(db: Any, cfg: dict[str, Any] | None) -> str | None:
    """``"<project name> · run <id>"`` / the check id for a configured detector, else None."""
    if not isinstance(cfg, dict):
        return None
    if cfg.get("kind") == "command":
        return str(cfg.get("check_id") or "") or None
    if cfg.get("kind") == "project":
        from app.models.project import Project

        project = await db.get(Project, int(cfg["project_id"])) if cfg.get("project_id") else None
        if project is None:
            return None
        exp = await latest_weighted_run(db, project.id, cfg.get("experiment_id"))
        return detector_label(project.name, exp.id) if exp is not None else None
    return None


def _infer_findings_text(model_ref: str, prompts: list[str]) -> list[str]:
    """Greedy generation of each prompt through the eval inference path (the model is loaded for
    this call and freed when it returns)."""
    from app.services.evaluation_service import NEUTRAL_DECODING, _run_local_inference

    pairs = [{"prompt": p, "reference": ""} for p in prompts]
    preds, _runtime = _run_local_inference(
        model_ref, pairs, MAX_DETECTOR_NEW_TOKENS, 0.0, [], apply_chat_template=True, decoding=NEUTRAL_DECODING,
    )
    return [str(p.get("prediction") or "") for p in preds]


def parse_finding_labels(text: str) -> list[str] | None:
    """Labels from a detector's raw output (G4 findings JSON); ``None`` when unparseable."""
    from app.services.eval_task_handler_service import StructuredExtractionHandler
    from app.services.finding_set_scoring import normalise_findings

    parsed = StructuredExtractionHandler()._parse_json_safely(text)
    findings, _invalid, has_list = normalise_findings(parsed)
    if not has_list:
        return None
    return [f["label"] for f in findings]


async def resolve_detector(db: Any, cfg: dict[str, Any] | None) -> ResolvedDetector | None:
    if not isinstance(cfg, dict):
        return None
    kind = cfg.get("kind")
    if kind == "project":
        return await _resolve_project_detector(db, cfg)
    if kind == "command":
        return _resolve_command_detector(cfg)
    raise DetectorUnavailable(f"unknown detector kind {kind!r}")


async def _resolve_project_detector(db: Any, cfg: dict[str, Any]) -> ResolvedDetector:
    from app.models.project import Project
    from app.services.code_findings_prompt import build_code_findings_prompt_from_row, taxonomy_label_ids
    from app.services.eval_task_handler_service import read_prepared_manifest
    from app.services import label_taxonomy_service as lts

    project = await db.get(Project, int(cfg.get("project_id") or 0))
    if project is None or not _is_code_findings_project(project):
        raise DetectorUnavailable("the detector project is not a code-findings project")
    exp = await latest_weighted_run(db, project.id, cfg.get("experiment_id"))
    if exp is None:
        raise DetectorUnavailable("the detector project has no completed run with weights")
    manifest_schema = (read_prepared_manifest(project.id).get("output_schema") or {})
    labels = list(manifest_schema.get("prompt_labels") or []) if isinstance(manifest_schema, dict) else []
    if not labels:
        labels = taxonomy_label_ids(lts.project_taxonomy(project))
    model_ref = str(Path(exp.output_dir).expanduser() / "model") if (Path(exp.output_dir).expanduser() / "model").is_dir() else exp.output_dir
    identity = (
        f"project:{project.id}:run:{exp.id}:{hashlib.sha256(json.dumps(labels).encode()).hexdigest()[:8]}"
        f":{detector_recipe_fingerprint(labels)}"
    )

    def detect(chunks: list[dict[str, Any]]) -> list[list[str] | None]:
        prompts = [build_code_findings_prompt_from_row(c, labels or None) for c in chunks]
        texts = _infer_findings_text(model_ref, prompts)
        return [parse_finding_labels(t) for t in texts]

    return ResolvedDetector("project", detector_label(project.name, exp.id), identity, detect)


def _resolve_command_detector(cfg: dict[str, Any]) -> ResolvedDetector:
    from app.services import code_check_service as ccs

    check_id = str(cfg.get("check_id") or "")
    check = ccs.get_code_check(check_id)
    if check is None or getattr(check, "kind", None) != "detector":
        raise DetectorUnavailable(f"'{check_id}' is not a registered detector check")
    if not ccs.is_allowed(check):
        raise DetectorUnavailable(f"detector '{check_id}' is not allow-listed (CODE_CHECK_ALLOWED)")
    if not check.available():
        raise DetectorUnavailable(f"detector '{check_id}' is not installed")

    def detect(chunks: list[dict[str, Any]]) -> list[list[str] | None]:
        out: list[list[str] | None] = []
        for chunk in chunks:
            try:
                res = check.run(
                    chunk.get("code") or "",
                    language=ccs.normalize_language(chunk.get("language")),
                    context={"path": chunk.get("path"), "start_line": chunk.get("start_line")},
                )
                out.append(_labels_from_check_result(res))
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("detector %s failed: %s", check_id, exc)
                out.append(None)
        return out

    return ResolvedDetector("command", check_id, f"command:{check_id}:{binary_fingerprint(getattr(check, 'binary', None))}", detect)


def _labels_from_check_result(res: Any) -> list[str] | None:
    if not isinstance(res, dict) or res.get("ok") is None and res.get("skipped"):
        return None
    findings = res.get("findings")
    if isinstance(findings, list):
        return [str(f.get("label")) for f in findings if isinstance(f, dict) and f.get("label")]
    for message in res.get("messages") or []:
        labels = parse_finding_labels(str(message))
        if labels is not None:
            return labels
    # A detector that ran and reported nothing found nothing.
    return [] if res.get("ok") is not None else None


# ── annotation ───────────────────────────────────────────────────────────


async def annotate_predictions(
    db: Any,
    project_id: int,
    predictions: list[dict[str, Any]],
    detector_cfg: dict[str, Any] | None,
    schema: dict[str, Any],
    parse_json: Callable[[Any], Any],
) -> dict[str, Any]:
    """Run the configured detector over the original and patched chunk of every row whose patch
    applies and attach ``_detector`` to the prediction. Returns a small snapshot
    (``{"detector", "chunks", "cached", "error"}``); never raises."""
    if not isinstance(detector_cfg, dict):
        return {}
    from app.services import remediation_scoring as rs

    try:
        resolved = await resolve_detector(db, detector_cfg)
    except Exception as exc:  # noqa: BLE001
        note = f"{type(exc).__name__}: {exc}"[:240]
        for p in predictions:
            if isinstance(p, dict) and rs.row_context(p) is not None:
                p["_detector"] = {"label": None, "original": None, "patched": None, "error": note}
        return {"error": note}
    if resolved is None:
        return {}

    rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for p in predictions:
        row = rs.row_context(p) if isinstance(p, dict) else None
        if row is None:
            continue
        a = await asyncio.to_thread(rs.analyze_row, p, schema, parse_json)
        if not a["applies"] or a["patched_code"] is None:
            continue
        original = {k: row.get(k) for k in ("code", "language", "path", "start_line")}
        patched = {**original, "code": a["patched_code"]}
        rows.append((p, original, patched))

    cache = DetectorCache(cache_path(project_id))
    pending: dict[str, dict[str, Any]] = {}
    for _p, original, patched in rows:
        for chunk in (original, patched):
            key = chunk_key(resolved.identity, chunk)
            if cache.get(key) is None:
                pending.setdefault(key, chunk)
    cached = len(rows) * 2 - len(pending)
    failed: set[str] = set()
    error_note: str | None = None
    if pending:
        keys = list(pending)
        try:
            results = await asyncio.to_thread(resolved.detect, [pending[k] for k in keys])
        except Exception as exc:  # noqa: BLE001 - the detector never fails the eval
            error_note = f"{type(exc).__name__}: {exc}"[:240]
            results = [None] * len(keys)
        for key, labels in zip(keys, results):
            if labels is None:
                failed.add(key)
            else:
                cache.set(key, labels)
        cache.flush()
        if failed and error_note is None:
            error_note = f"{len(failed)} chunk(s) could not be checked (unparseable detector output)"

    for p, original, patched in rows:
        ko, kp = chunk_key(resolved.identity, original), chunk_key(resolved.identity, patched)
        lo, lp = cache.get(ko), cache.get(kp)
        err = None if (lo is not None and lp is not None) else (error_note or "detector produced no result")
        p["_detector"] = {"label": resolved.label, "original": lo, "patched": lp, "error": err}
    return {"detector": resolved.label, "chunks": len(pending) + cached, "cached": cached, "error": error_note}


# ── scoring fingerprint ──────────────────────────────────────────────────


async def detector_identity(db: Any, cfg: dict[str, Any] | None) -> str:
    """Identity of the configured detector as scoring sees it (never raises): the resolved
    project run (an unpinned detector follows the project's latest run), or the command check
    with its binary. ``"none"`` without one."""
    if not isinstance(cfg, dict):
        return "none"
    try:
        resolved = await resolve_detector(db, cfg)
        return resolved.identity if resolved is not None else "none"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{type(exc).__name__}"


async def remediation_scoring_fingerprint(
    db: Any, project_id: int, runtime_config: Any, output_schema: dict[str, Any] | None
) -> str:
    """Hash of everything that changes how a remediation prediction is scored: the detector (identity
    incl. experiment id), the effective code-check ids and the allow-list, passage retrieval settings
    the extra-field schema and (when on) ``score_code_examples`` / ``format_before_check``. Stamped into the eval metrics as ``remediation_fingerprint``; a cached
    base-model result is reused only when it matches (like ``_baseline_judge_matches``)."""
    from app.services import code_check_service as ccs
    from app.services.code_remediation_prompt import effective_remediation_config

    cfg = effective_remediation_config(runtime_config, project_id)
    schema = output_schema if isinstance(output_schema, dict) else {}
    if cfg["checks"] is not None:
        check_ids = sorted(cfg["checks"])
    else:
        check_ids = sorted(c["id"] for c in ccs.list_code_checks() if c["available"] and c["allowed"] and c["kind"] != "detector")
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    parts = {
        "detector": await detector_identity(db, cfg["detector"]),
        "checks": check_ids,
        "allowed": sorted(ccs.allowed_check_ids()),
        "retrieve_passages": bool(cfg["retrieve_passages"]),
        "top_k": cfg["top_k"],
        "extra_fields": hashlib.sha256(
            json.dumps([props, sorted(schema.get("required") or [])], sort_keys=True, default=str).encode()
        ).hexdigest()[:10],
    }
    revisions = ccs.check_revisions(check_ids)
    if revisions:
        # Only checks whose semantics changed (yaml_parse 2: duplicate keys fail); results scored by the
        # older check get a different fingerprint, so they are neither reused as a baseline nor paired.
        parts["check_revisions"] = revisions
    if cfg.get("score_code_examples"):
        # Only when on, so every fingerprint stamped before the setting existed still matches.
        parts["score_code_examples"] = True
    if cfg.get("format_before_check") and cfg.get("score_code_examples"):
        # Same rule: on = a different scorer (W1's Compare and the base-result reuse see different fingerprints).
        # Only with code-example scoring on: it formats nothing otherwise.
        parts["format_before_check"] = True
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
