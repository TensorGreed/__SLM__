"""Export & runtime packaging service."""

import asyncio
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.export import Export, ExportFormat, ExportStatus
from app.models.experiment import Experiment
from app.models.project import Project
from app.schemas.export import (
    OptimizationCandidate,
    OptimizationMetric,
    OptimizationResponse,
    OptimizationRunEvidence,
)
from app.services import target_profile_service
from app.services.deployment_target_service import (
    build_deploy_target_plan,
    execute_deploy_target_plan,
    run_deployment_target_suite,
)

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_BENCHMARK_SCRIPT = _BACKEND_DIR / "scripts" / "benchmark.py"
_MODEL_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt"}
_OPTIMIZATION_PROMPT_SET_ID = "optimization.default.v1"
_OPTIMIZATION_DEFAULT_PROMPTS: list[str] = [
    "Summarize the key differences between supervised and unsupervised learning.",
    "Write a short troubleshooting guide for a web service returning HTTP 503 errors.",
    "Convert this requirement into a test case: user can reset password with email OTP.",
    "Explain how quantization reduces model size and what tradeoffs it introduces.",
    "Provide a concise answer: what is overfitting and how can we detect it?",
]


def _export_dir(project_id: int, export_id: int) -> Path:
    d = settings.DATA_DIR / "exports" / str(project_id) / str(export_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256_file(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _artifact_manifest_entry(file_path: Path, root_dir: Path) -> dict:
    return {
        "path": str(file_path.relative_to(root_dir)),
        "size_bytes": file_path.stat().st_size,
        "sha256": _sha256_file(file_path),
    }


def _project_dir(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id)


def _sanitize_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_json_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _optimization_dir(project_id: int) -> Path:
    directory = _project_dir(project_id) / "optimization"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _optimization_runs_dir(project_id: int) -> Path:
    directory = _optimization_dir(project_id) / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _prompt_set_hash(prompts: list[str]) -> str:
    normalized = [str(item or "").strip() for item in prompts if str(item or "").strip()]
    blob = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _ensure_optimization_prompt_set(project_id: int) -> dict[str, Any]:
    prompts_dir = _optimization_dir(project_id) / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompts_dir / f"{_OPTIMIZATION_PROMPT_SET_ID}.txt"
    if not prompt_file.exists():
        prompt_file.write_text(
            "\n".join(_OPTIMIZATION_DEFAULT_PROMPTS) + "\n",
            encoding="utf-8",
        )
    prompt_hash = _prompt_set_hash(_OPTIMIZATION_DEFAULT_PROMPTS)
    return {
        "prompt_set_id": _OPTIMIZATION_PROMPT_SET_ID,
        "prompt_set_hash": prompt_hash,
        "prompt_file_path": str(prompt_file),
        "prompt_count": len(_OPTIMIZATION_DEFAULT_PROMPTS),
    }


def _matches_quantization(file_name: str, quantization: str | None) -> bool:
    if not quantization:
        return True
    normalized = str(quantization).strip().lower()
    if normalized in {"none", "null", "false", "off", "no"}:
        return True
    quant_norm = _sanitize_token(quantization)
    name_norm = _sanitize_token(file_name)
    if not quant_norm:
        return True
    if quant_norm in name_norm:
        return True

    digits = "".join(ch for ch in quant_norm if ch.isdigit())
    if digits:
        for token in (f"{digits}bit", f"q{digits}", f"int{digits}"):
            if token in name_norm:
                return True
    return False


def _collect_compressed_files(
    project_id: int,
    export_format: ExportFormat,
    quantization: str | None,
) -> list[Path]:
    compressed_dir = _project_dir(project_id) / "compressed"
    if not compressed_dir.exists():
        return []

    all_files = [p for p in compressed_dir.rglob("*") if p.is_file()]

    if export_format == ExportFormat.ONNX:
        onnx_files = [
            p for p in all_files
            if p.suffix.lower() == ".onnx" and _matches_quantization(p.name, quantization)
        ]
        if not onnx_files:
            return []

        selected: set[Path] = set(onnx_files)
        metadata_names = {
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "added_tokens.json",
            "preprocessor_config.json",
        }
        for onnx_file in onnx_files:
            parent = onnx_file.parent
            for sibling in parent.iterdir():
                if not sibling.is_file():
                    continue
                if sibling.name in {"quantize_report.json", "quantize_result.json", "benchmark_report.json", "benchmark_results.json"}:
                    continue
                # Include sidecar data files and tokenizer/config metadata.
                if sibling.name in metadata_names or sibling.name.startswith(f"{onnx_file.name}."):
                    selected.add(sibling)
        return sorted(selected, key=lambda p: p.stat().st_mtime, reverse=True)

    extension_map = {
        ExportFormat.GGUF: {".gguf"},
        ExportFormat.TENSORRT: {".engine", ".plan"},
    }
    allowed_ext = extension_map.get(export_format)

    files = all_files
    if allowed_ext:
        files = [p for p in files if p.suffix.lower() in allowed_ext]
    else:
        files = [
            p for p in files
            if not p.name.endswith("_report.json")
            and p.name not in {"benchmark_results.json", "benchmark_report.json"}
        ]

    filtered = [p for p in files if _matches_quantization(p.name, quantization)]
    return sorted(filtered, key=lambda p: p.stat().st_mtime, reverse=True)


def _copy_files_preserve_structure(
    files: list[Path],
    dest_dir: Path,
    source_root: Path,
) -> list[Path]:
    copied: list[Path] = []
    for src in files:
        target = dest_dir / src.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(target)
    return copied


def _copy_files(files: list[Path], dest_dir: Path) -> list[Path]:
    copied: list[Path] = []
    used_names: set[str] = set()
    for src in files:
        base_name = src.name
        stem = src.stem
        suffix = src.suffix
        candidate = base_name
        idx = 1
        while candidate in used_names or (dest_dir / candidate).exists():
            candidate = f"{stem}_{idx}{suffix}"
            idx += 1
        target = dest_dir / candidate
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(target)
        used_names.add(candidate)
    return copied


def _resolve_source_model_files(
    project_id: int,
    experiment: Experiment,
    export_format: ExportFormat,
    quantization: str | None,
) -> tuple[str, list[Path], Path | None]:
    use_compressed = export_format in {ExportFormat.GGUF, ExportFormat.ONNX, ExportFormat.TENSORRT} or bool(quantization)
    if use_compressed:
        compressed_files = _collect_compressed_files(project_id, export_format, quantization)
        if compressed_files:
            return "compressed", compressed_files, None

    output_dir = Path(experiment.output_dir).expanduser() if experiment.output_dir else None
    if output_dir and output_dir.exists():
        model_dir = output_dir / "model"
        if model_dir.exists() and model_dir.is_dir():
            model_files = [p for p in model_dir.rglob("*") if p.is_file()]
            if model_files:
                return "experiment_model_dir", model_files, model_dir

        direct_model_files = [
            p for p in output_dir.rglob("*")
            if p.is_file() and p.name not in {"training_config.json", "training_report.json", "external_training.log"}
        ]
        if direct_model_files:
            return "experiment_output_dir", direct_model_files, output_dir

    return "none", [], None


def _resolve_export_run_dir_from_export(export: Export) -> Path | None:
    manifest = export.manifest if isinstance(export.manifest, dict) else {}
    run_dir = manifest.get("run_dir")
    if isinstance(run_dir, str) and run_dir.strip():
        candidate = Path(run_dir).expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate

    output_path = Path(export.output_path).expanduser() if export.output_path else None
    if output_path is None or not output_path.exists():
        return None

    latest_run = output_path / "LATEST_RUN"
    if latest_run.exists():
        token = latest_run.read_text(encoding="utf-8").strip()
        if token:
            candidate = output_path / f"run-{token}"
            if candidate.exists() and candidate.is_dir():
                return candidate
    return None


def _runtime_template_for_export_format(export_format: ExportFormat | str) -> str:
    token = str(export_format.value if isinstance(export_format, ExportFormat) else export_format).strip().lower()
    if token == "docker":
        return "huggingface"
    return token or "huggingface"


def _build_optimization_artifact_identifier(candidate: dict[str, Any]) -> str:
    artifact_path = Path(str(candidate.get("artifact_path") or candidate.get("model_ref") or "")).expanduser()
    try:
        artifact_token = artifact_path.resolve().as_posix()
    except Exception:
        artifact_token = artifact_path.as_posix()
    payload = {
        "artifact_path": artifact_token,
        "runtime_template": str(candidate.get("runtime_template") or ""),
        "quantization": str(candidate.get("quantization") or ""),
        "size_bytes": int(candidate.get("size_bytes") or 0),
        "file_count": int(candidate.get("file_count") or 0),
    }
    return _hash_json_payload(payload)[:24]


def _persist_optimization_run(
    *,
    project_id: int,
    target_id: str,
    target_device: str,
    preferred_formats: set[str],
    candidates: list[OptimizationCandidate],
    discovered_by_id: dict[str, dict[str, Any]],
    prompt_set: dict[str, Any],
) -> OptimizationRunEvidence:
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_path = _optimization_runs_dir(project_id) / f"{run_id}.json"
    created_at = datetime.now(timezone.utc).isoformat()
    recommended_candidate_id = next(
        (row.id for row in candidates if row.is_recommended),
        None,
    )

    candidate_rows: list[dict[str, Any]] = []
    measured_count = 0
    estimated_count = 0
    for row in candidates:
        source_candidate = dict(discovered_by_id.get(str(row.id)) or {})
        measurement = dict(row.measurement or {})
        artifact_identifier = _build_optimization_artifact_identifier(
            source_candidate if source_candidate else {
                "artifact_path": measurement.get("artifact_path"),
                "model_ref": measurement.get("model_ref"),
                "runtime_template": row.runtime_template,
                "quantization": row.quantization,
                "size_bytes": measurement.get("model_size_bytes") or 0,
                "file_count": measurement.get("file_count") or 0,
            }
        )
        metric_source = str(row.metric_source or "estimated").strip().lower() or "estimated"
        if metric_source == "measured":
            measured_count += 1
        else:
            estimated_count += 1

        candidate_rows.append(
            {
                "id": row.id,
                "name": row.name,
                "runtime_template": row.runtime_template,
                "quantization": row.quantization,
                "artifact_identifier": artifact_identifier,
                "artifact_path": measurement.get("artifact_path") or source_candidate.get("artifact_path"),
                "model_ref": measurement.get("model_ref") or source_candidate.get("model_ref"),
                "metrics": row.metrics.model_dump(),
                "metric_source": row.metric_source,
                "metric_sources": dict(row.metric_sources or {}),
                "measurement": measurement,
                "is_recommended": bool(row.is_recommended),
                "reasons": list(row.reasons or []),
            }
        )

    payload: dict[str, Any] = {
        "schema": "slm.optimization-evidence/v1",
        "run_id": run_id,
        "created_at": created_at,
        "project_id": project_id,
        "target_id": target_id,
        "target_device": target_device,
        "preferred_formats": sorted(preferred_formats),
        "prompt_set": dict(prompt_set),
        "candidate_count": len(candidate_rows),
        "measured_candidate_count": measured_count,
        "estimated_candidate_count": estimated_count,
        "recommended_candidate_id": recommended_candidate_id,
        "candidates": candidate_rows,
    }
    payload["run_hash"] = _hash_json_payload(payload)
    run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    latest_pointer = _optimization_dir(project_id) / "latest_run.json"
    latest_pointer.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_path": str(run_path),
                "run_hash": payload["run_hash"],
                "updated_at": created_at,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return OptimizationRunEvidence(
        run_id=run_id,
        created_at=created_at,
        run_path=str(run_path),
        run_hash=str(payload["run_hash"]),
        prompt_set_id=str(prompt_set.get("prompt_set_id") or _OPTIMIZATION_PROMPT_SET_ID),
        prompt_set_hash=str(prompt_set.get("prompt_set_hash") or ""),
        candidate_count=len(candidate_rows),
        measured_candidate_count=measured_count,
        estimated_candidate_count=estimated_count,
        recommended_candidate_id=recommended_candidate_id,
    )


def _load_latest_optimization_run(project_id: int) -> dict[str, Any] | None:
    latest_pointer = _optimization_dir(project_id) / "latest_run.json"
    if not latest_pointer.exists():
        return None
    try:
        pointer = json.loads(latest_pointer.read_text(encoding="utf-8"))
    except Exception:
        return None

    run_path_token = str(pointer.get("run_path") or "").strip()
    if not run_path_token:
        return None
    run_path = Path(run_path_token).expanduser()
    if not run_path.exists() or not run_path.is_file():
        return None
    try:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    payload["_run_path"] = str(run_path)
    return payload


def _build_export_optimization_evidence(
    *,
    project_id: int,
    export_format: ExportFormat,
    quantization: str | None,
) -> dict[str, Any] | None:
    run = _load_latest_optimization_run(project_id)
    if not isinstance(run, dict):
        return None

    candidates = [item for item in list(run.get("candidates") or []) if isinstance(item, dict)]
    if not candidates:
        return None

    runtime_template = _runtime_template_for_export_format(export_format)
    desired_quantization = _normalize_quantization_token(quantization)
    selected = next(
        (
            item for item in candidates
            if str(item.get("runtime_template") or "").strip().lower() == runtime_template
            and _normalize_quantization_token(item.get("quantization")) == desired_quantization
        ),
        None,
    )
    if selected is None:
        selected = next(
            (
                item for item in candidates
                if str(item.get("runtime_template") or "").strip().lower() == runtime_template
            ),
            None,
        )

    return {
        "schema": "slm.optimization-evidence.export/v1",
        "run_id": run.get("run_id"),
        "run_path": run.get("_run_path"),
        "run_hash": run.get("run_hash"),
        "created_at": run.get("created_at"),
        "prompt_set_id": (run.get("prompt_set") or {}).get("prompt_set_id"),
        "prompt_set_hash": (run.get("prompt_set") or {}).get("prompt_set_hash"),
        "recommended_candidate_id": run.get("recommended_candidate_id"),
        "candidate_count": run.get("candidate_count"),
        "measured_candidate_count": run.get("measured_candidate_count"),
        "estimated_candidate_count": run.get("estimated_candidate_count"),
        "selected_candidate": selected,
    }


def _normalize_gate_policy(payload: Any) -> dict[str, Any]:
    policy = {
        "must_pass": False,
        "min_score": 0.0,
        "blocked_if_missing": False,
    }
    if not isinstance(payload, dict):
        return policy

    if "must_pass" in payload:
        policy["must_pass"] = bool(payload.get("must_pass"))
    if "blocked_if_missing" in payload:
        policy["blocked_if_missing"] = bool(payload.get("blocked_if_missing"))

    try:
        score = float(payload.get("min_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    policy["min_score"] = max(0.0, min(1.0, score))
    return policy


def _gate_score_from_report(report: dict[str, Any]) -> float | None:
    checks = [item for item in list(report.get("checks") or []) if isinstance(item, dict)]
    scored = [bool(item.get("passed")) for item in checks if item.get("actual") is not None]
    if not scored:
        return None
    return round(sum(1 for ok in scored if ok) / float(len(scored)), 6)


async def create_export(
    db: AsyncSession,
    project_id: int,
    experiment_id: int,
    export_format: ExportFormat,
    quantization: str | None = None,
) -> Export:
    """Create a new export record."""
    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.project_id == project_id,
        )
    )
    if not exp_result.scalar_one_or_none():
        raise ValueError(f"Experiment {experiment_id} not found in project {project_id}")

    export = Export(
        project_id=project_id,
        experiment_id=experiment_id,
        export_format=export_format,
        quantization=quantization,
        status=ExportStatus.PENDING,
    )
    db.add(export)
    await db.flush()
    await db.refresh(export)

    output_dir = _export_dir(project_id, export.id)
    export.output_path = str(output_dir)
    await db.flush()

    return export


async def run_export(
    db: AsyncSession,
    project_id: int,
    export_id: int,
    eval_report: dict | None = None,
    safety_scorecard: dict | None = None,
    benchmark_report: dict | None = None,
    deployment_targets: list[str] | None = None,
    run_smoke_tests: bool = True,
) -> Export:
    """Execute the export process."""
    result = await db.execute(
        select(Export).where(
            Export.id == export_id,
            Export.project_id == project_id,
        )
    )
    export = result.scalar_one_or_none()
    if not export:
        raise ValueError(f"Export {export_id} not found in project {project_id}")

    exp_result = await db.execute(
        select(Experiment).where(
            Experiment.id == export.experiment_id,
            Experiment.project_id == project_id,
        )
    )
    experiment = exp_result.scalar_one_or_none()
    if not experiment:
        raise ValueError(f"Experiment {export.experiment_id} not found in project {project_id}")

    export.status = ExportStatus.IN_PROGRESS
    await db.flush()

    # Check mandatory quality gates
    from app.services.evaluation_pack_service import evaluate_experiment_auto_gates
    project_stmt = select(Project).where(Project.id == project_id)
    project_res = await db.execute(project_stmt)
    project = project_res.scalar_one_or_none()
    
    if project:
        policy = _normalize_gate_policy(project.gate_policy)
        must_pass = bool(policy.get("must_pass"))
        blocked_if_missing = bool(policy.get("blocked_if_missing"))
        min_score = float(policy.get("min_score") or 0.0)

        if must_pass or blocked_if_missing or min_score > 0.0:
            gate_report = await evaluate_experiment_auto_gates(
                db,
                project_id=project_id,
                experiment_id=experiment.id,
            )
            gate_score = _gate_score_from_report(gate_report)
            gate_report["aggregate_gate_score"] = gate_score

            is_blocked = False
            reasons: list[str] = []
            if must_pass and not gate_report.get("passed"):
                is_blocked = True
                reasons.append("Mandatory quality gates failed.")
            if blocked_if_missing and gate_report.get("missing_required_metrics"):
                is_blocked = True
                reasons.append(f"Missing required metrics: {', '.join(gate_report.get('missing_required_metrics', []))}")
            if min_score > 0.0 and (gate_score is None or gate_score < min_score):
                is_blocked = True
                if gate_score is None:
                    reasons.append("Gate score unavailable because no comparable metrics were evaluated.")
                else:
                    reasons.append(f"Gate score {gate_score:.3f} is below min_score {min_score:.3f}.")

            if is_blocked:
                export.status = ExportStatus.FAILED
                message = f"Export blocked by deployment gates: {'; '.join(reasons)}"
                export.error_message = message
                export.manifest = {
                    "error": message,
                    "gate_policy": policy,
                    "gate_report": gate_report,
                }
                await db.flush()
                return export

    deployment_report: dict | None = None
    run_dir: Path | None = None
    try:
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%SZ")

        output_dir = Path(export.output_path or _export_dir(project_id, export.id))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = output_dir / f"run-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[dict] = []

        # Copy actual model artifacts into export package.
        model_source, source_files, source_root = _resolve_source_model_files(
            project_id=project_id,
            experiment=experiment,
            export_format=export.export_format,
            quantization=export.quantization,
        )
        if not source_files:
            raise ValueError(
                "No model artifacts found for export. "
                "Complete training (and compression for quantized formats) before exporting."
            )

        model_dir = run_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        if source_root is not None:
            copied_model_files = _copy_files_preserve_structure(source_files, model_dir, source_root)
        else:
            copied_model_files = _copy_files(source_files, model_dir)
        for copied in copied_model_files:
            artifacts.append(_artifact_manifest_entry(copied, run_dir))
        model_size_bytes = sum(p.stat().st_size for p in copied_model_files)

        # Save eval report
        if eval_report:
            export.eval_report = eval_report
            eval_path = run_dir / "eval_report.json"
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_report, f, indent=2)
            artifacts.append(_artifact_manifest_entry(eval_path, run_dir))

        # Save safety scorecard
        if safety_scorecard:
            export.safety_scorecard = safety_scorecard
            safety_path = run_dir / "safety_scorecard.json"
            with open(safety_path, "w", encoding="utf-8") as f:
                json.dump(safety_scorecard, f, indent=2)
            artifacts.append(_artifact_manifest_entry(safety_path, run_dir))

        # Save benchmark report
        if benchmark_report:
            benchmark_path = run_dir / "benchmark_report.json"
            with open(benchmark_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_report, f, indent=2)
            artifacts.append(_artifact_manifest_entry(benchmark_path, run_dir))

        # Generate Dockerfile template
        dockerfile_content = _generate_dockerfile(export.export_format)
        dockerfile_path = run_dir / "Dockerfile"
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile_content)
        artifacts.append(_artifact_manifest_entry(dockerfile_path, run_dir))

        # Generate inference script template
        inference_script = _generate_inference_script(export.export_format)
        serve_path = run_dir / "serve.py"
        with open(serve_path, "w", encoding="utf-8") as f:
            f.write(inference_script)
        artifacts.append(_artifact_manifest_entry(serve_path, run_dir))

        release_notes_path = run_dir / "RELEASE_NOTES.md"
        release_notes_path.write_text(
            (
                f"# Export Release {run_id}\n\n"
                f"- Export ID: {export.id}\n"
                f"- Project ID: {project_id}\n"
                f"- Experiment ID: {experiment.id}\n"
                f"- Format: {export.export_format.value}\n"
                f"- Quantization: {export.quantization or 'none'}\n"
                f"- Model source: {model_source}\n"
                f"- Model artifacts: {len(copied_model_files)} files\n"
                f"- Model bytes: {model_size_bytes}\n"
                f"- Generated at: {now.isoformat()}\n"
                f"- Platform version: {settings.APP_VERSION}\n"
            ),
            encoding="utf-8",
        )
        artifacts.append(_artifact_manifest_entry(release_notes_path, run_dir))

        # Generate version manifest
        manifest = {
            "run_id": run_id,
            "export_id": export.id,
            "project_id": export.project_id,
            "experiment_id": export.experiment_id,
            "format": export.export_format.value,
            "quantization": export.quantization,
            "created_at": now.isoformat(),
            "platform_version": settings.APP_VERSION,
            "run_dir": str(run_dir),
            "experiment": {
                "name": experiment.name,
                "status": experiment.status.value,
                "training_mode": experiment.training_mode.value,
                "base_model": experiment.base_model,
                "final_train_loss": experiment.final_train_loss,
                "final_eval_loss": experiment.final_eval_loss,
            },
            "model_artifacts": {
                "source": model_source,
                "count": len(copied_model_files),
                "size_bytes": model_size_bytes,
                "path": "model",
            },
            "artifacts": artifacts,
        }
        optimization_evidence = _build_export_optimization_evidence(
            project_id=project_id,
            export_format=export.export_format,
            quantization=export.quantization,
        )
        if optimization_evidence is not None:
            manifest["optimization_evidence"] = optimization_evidence

        deployment_report = run_deployment_target_suite(
            run_dir=run_dir,
            export_format=export.export_format,
            deployment_targets=deployment_targets,
            run_smoke_tests=run_smoke_tests,
        )
        manifest["deployment"] = deployment_report
        if not bool((deployment_report.get("summary") or {}).get("deployable_artifact")):
            errors = list((deployment_report.get("artifact_validation") or {}).get("errors") or [])
            preview = "; ".join(str(item) for item in errors[:4]) or "artifact validation failed"
            raise ValueError(
                "Deployment validation failed for requested export format. "
                f"Details: {preview}"
            )

        run_manifest_path = run_dir / "manifest.json"
        with open(run_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Keep root manifest as the latest export run for backward compatibility.
        root_manifest_path = output_dir / "manifest.json"
        with open(root_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        (output_dir / "LATEST_RUN").write_text(run_id, encoding="utf-8")

        export.status = ExportStatus.COMPLETED
        export.manifest = manifest
        export.output_path = str(output_dir)
        export.file_size_bytes = model_size_bytes
        export.completed_at = now

    except Exception as e:
        export.status = ExportStatus.FAILED
        failed_manifest = export.manifest if isinstance(export.manifest, dict) else {}
        failed_manifest = dict(failed_manifest)
        if run_dir is not None:
            failed_manifest.setdefault("run_dir", str(run_dir))
        if isinstance(deployment_report, dict):
            failed_manifest["deployment"] = deployment_report
        failed_manifest["error"] = str(e)
        export.manifest = failed_manifest

    await db.flush()
    await db.refresh(export)
    return export


async def list_exports(
    db: AsyncSession, project_id: int
) -> list[Export]:
    """List all exports for a project."""
    result = await db.execute(
        select(Export)
        .where(Export.project_id == project_id)
        .order_by(Export.created_at.desc())
    )
    return list(result.scalars().all())


async def validate_export_deployment(
    db: AsyncSession,
    project_id: int,
    export_id: int,
    *,
    deployment_targets: list[str] | None = None,
    run_smoke_tests: bool = True,
) -> dict:
    """Validate deployability for an existing export run."""
    result = await db.execute(
        select(Export).where(
            Export.id == export_id,
            Export.project_id == project_id,
        )
    )
    export = result.scalar_one_or_none()
    if not export:
        raise ValueError(f"Export {export_id} not found in project {project_id}")

    run_dir = _resolve_export_run_dir_from_export(export)
    if run_dir is None:
        raise ValueError("Export run directory not found. Run export first.")

    report = run_deployment_target_suite(
        run_dir=run_dir,
        export_format=export.export_format,
        deployment_targets=deployment_targets,
        run_smoke_tests=run_smoke_tests,
    )

    manifest = export.manifest if isinstance(export.manifest, dict) else {}
    manifest["deployment"] = report
    export.manifest = manifest
    await db.flush()
    await db.refresh(export)
    return report


async def build_export_deploy_plan(
    db: AsyncSession,
    project_id: int,
    export_id: int,
    *,
    target_id: str,
    endpoint_name: str | None = None,
    region: str | None = None,
    instance_type: str | None = None,
) -> dict:
    """Build deployment/mobile SDK plan for an existing export run."""
    result = await db.execute(
        select(Export).where(
            Export.id == export_id,
            Export.project_id == project_id,
        )
    )
    export = result.scalar_one_or_none()
    if not export:
        raise ValueError(f"Export {export_id} not found in project {project_id}")

    run_dir = _resolve_export_run_dir_from_export(export)
    if run_dir is None:
        raise ValueError("Export run directory not found. Run export first.")

    model_name = str((export.manifest or {}).get("source_model") or f"project-{project_id}-export-{export_id}")
    plan = build_deploy_target_plan(
        run_dir=run_dir,
        export_format=export.export_format,
        target_id=target_id,
        model_name=model_name,
        endpoint_name=endpoint_name,
        region=region,
        instance_type=instance_type,
    )
    return {
        "project_id": project_id,
        "export_id": export_id,
        "export_format": export.export_format.value,
        "run_dir": str(run_dir),
        **plan,
    }


async def execute_export_deploy_plan(
    db: AsyncSession,
    project_id: int,
    export_id: int,
    *,
    target_id: str,
    endpoint_name: str | None = None,
    region: str | None = None,
    instance_type: str | None = None,
    dry_run: bool = True,
    hf_token: str | None = None,
    managed_api_url: str | None = None,
    managed_api_token: str | None = None,
    sagemaker_role_arn: str | None = None,
    sagemaker_image_uri: str | None = None,
    sagemaker_model_data_url: str | None = None,
) -> dict:
    """Execute managed deploy action (or dry-run) for an existing export run."""
    result = await db.execute(
        select(Export).where(
            Export.id == export_id,
            Export.project_id == project_id,
        )
    )
    export = result.scalar_one_or_none()
    if not export:
        raise ValueError(f"Export {export_id} not found in project {project_id}")

    run_dir = _resolve_export_run_dir_from_export(export)
    if run_dir is None:
        raise ValueError("Export run directory not found. Run export first.")

    model_name = str((export.manifest or {}).get("source_model") or f"project-{project_id}-export-{export_id}")
    result_payload = await execute_deploy_target_plan(
        run_dir=run_dir,
        export_format=export.export_format,
        target_id=target_id,
        model_name=model_name,
        endpoint_name=endpoint_name,
        region=region,
        instance_type=instance_type,
        dry_run=dry_run,
        hf_token=hf_token,
        managed_api_url=managed_api_url,
        managed_api_token=managed_api_token,
        sagemaker_role_arn=sagemaker_role_arn,
        sagemaker_image_uri=sagemaker_image_uri,
        sagemaker_model_data_url=sagemaker_model_data_url,
    )

    manifest = export.manifest if isinstance(export.manifest, dict) else {}
    history = list(manifest.get("deploy_execution_history") or [])
    execution = result_payload.get("execution") if isinstance(result_payload, dict) else {}
    history.append(
        {
            "target_id": str(result_payload.get("target_id") or target_id),
            "target_kind": str(result_payload.get("target_kind") or ""),
            "dry_run": bool((execution or {}).get("dry_run", dry_run)),
            "status": str((execution or {}).get("status") or ""),
            "started_at": (execution or {}).get("started_at"),
            "finished_at": (execution or {}).get("finished_at"),
        }
    )
    manifest["deploy_execution_history"] = history[-30:]
    manifest["last_deploy_execution"] = execution
    export.manifest = manifest
    await db.flush()
    await db.refresh(export)

    return {
        "project_id": project_id,
        "export_id": export_id,
        "export_format": export.export_format.value,
        "run_dir": str(run_dir),
        **result_payload,
    }


def _generate_dockerfile(export_format: ExportFormat) -> str:
    """Generate a Dockerfile for the inference server."""
    if export_format == ExportFormat.GGUF:
        return """FROM python:3.11-slim
WORKDIR /app
RUN pip install llama-cpp-python fastapi uvicorn
COPY . .
EXPOSE 8080
CMD ["python", "serve.py"]
"""
    elif export_format == ExportFormat.ONNX:
        return """FROM python:3.11-slim
WORKDIR /app
RUN pip install "optimum[onnxruntime]" transformers fastapi uvicorn
COPY . .
EXPOSE 8080
CMD ["python", "serve.py"]
"""
    elif export_format == ExportFormat.HUGGINGFACE:
        return """FROM python:3.11-slim
WORKDIR /app
RUN pip install torch transformers accelerate fastapi uvicorn
COPY . .
EXPOSE 8080
CMD ["python", "serve.py"]
"""
    else:
        return """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "serve.py"]
"""


def _generate_inference_script(export_format: ExportFormat) -> str:
    """Generate an inference server script tailored to export format."""
    if export_format in {ExportFormat.HUGGINGFACE, ExportFormat.DOCKER}:
        return '''"""Auto-generated inference server for exported SLM model."""
import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="SLM Inference Server")
MODEL_PATH = os.getenv("MODEL_PATH", "./model")
model = None
tokenizer = None
load_error = None


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class InferenceResponse(BaseModel):
    text: str
    tokens_used: int


@app.on_event("startup")
async def startup():
    global model, tokenizer, load_error
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
    except Exception as e:
        load_error = str(e)


@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {load_error}")
    import torch

    inputs = tokenizer(req.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
        )
    prompt_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = output_ids[0][prompt_tokens:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return InferenceResponse(text=text, tokens_used=int(generated_tokens.shape[-1]))


@app.get("/health")
async def health():
    with open("manifest.json") as f:
        manifest = json.load(f)
    return {
        "status": "ok" if model is not None else "degraded",
        "format": "huggingface",
        "model_path": MODEL_PATH,
        "load_error": load_error,
        "manifest": manifest,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

    if export_format == ExportFormat.GGUF:
        return '''"""Auto-generated GGUF inference server."""
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="SLM Inference Server")
MODEL_PATH = os.getenv("MODEL_PATH", "./model/model.gguf")
llm = None
load_error = None


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class InferenceResponse(BaseModel):
    text: str
    tokens_used: int


@app.on_event("startup")
async def startup():
    global llm, load_error
    try:
        from llama_cpp import Llama
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            candidates = sorted(Path("./model").glob("*.gguf"))
            if candidates:
                model_path = candidates[0]
        if not model_path.exists():
            raise RuntimeError("No .gguf model file found under ./model")
        llm = Llama(model_path=str(model_path), n_ctx=4096)
    except Exception as e:
        load_error = str(e)


@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {load_error}")
    out = llm(
        req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    choice = out.get("choices", [{}])[0]
    usage = out.get("usage", {})
    return InferenceResponse(
        text=choice.get("text", ""),
        tokens_used=int(usage.get("total_tokens", 0)),
    )


@app.get("/health")
async def health():
    with open("manifest.json") as f:
        manifest = json.load(f)
    return {
        "status": "ok" if llm is not None else "degraded",
        "format": "gguf",
        "model_path": MODEL_PATH,
        "load_error": load_error,
        "manifest": manifest,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

    if export_format == ExportFormat.ONNX:
        return '''"""Auto-generated ONNX inference server."""
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="SLM ONNX Inference Server")
MODEL_PATH = os.getenv("MODEL_PATH", "./model")
model = None
tokenizer = None
load_error = None
resolved_model_dir = MODEL_PATH


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


class InferenceResponse(BaseModel):
    text: str
    tokens_used: int


@app.on_event("startup")
async def startup():
    global model, tokenizer, load_error, resolved_model_dir
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        base_path = Path(MODEL_PATH)
        model_dir = base_path
        file_name = None
        if base_path.exists():
            candidates = sorted(base_path.rglob("*.onnx"))
            if candidates:
                file_name = candidates[0].name
                model_dir = candidates[0].parent
        resolved_model_dir = str(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(resolved_model_dir, trust_remote_code=True)
        model = ORTModelForCausalLM.from_pretrained(resolved_model_dir, file_name=file_name)
    except Exception as e:
        load_error = str(e)


@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {load_error}")
    inputs = tokenizer(req.prompt, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=req.temperature > 0,
    )
    prompt_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = output_ids[0][prompt_tokens:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return InferenceResponse(text=text, tokens_used=int(generated_tokens.shape[-1]))


@app.get("/health")
async def health():
    with open("manifest.json") as f:
        manifest = json.load(f)
    return {
        "status": "ok" if model is not None else "degraded",
        "format": "onnx",
        "model_path": MODEL_PATH,
        "resolved_model_dir": resolved_model_dir,
        "load_error": load_error,
        "manifest": manifest,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

    return '''"""Auto-generated inference server (manual runtime integration required)."""
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="SLM Inference Server")


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class InferenceResponse(BaseModel):
    text: str
    tokens_used: int


@app.post("/generate", response_model=InferenceResponse)
async def generate(req: InferenceRequest):
    raise HTTPException(
        status_code=501,
        detail="Inference runtime not auto-generated for this export format. Integrate your runtime engine in serve.py.",
    )


@app.get("/health")
async def health():
    with open("manifest.json") as f:
        manifest = json.load(f)
    return {"status": "degraded", "manifest": manifest}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''


async def optimize_for_target(
    db: AsyncSession, project_id: int, target_id: str
) -> OptimizationResponse:
    _ = db
    target = target_profile_service.get_target_by_id(target_id)
    if not target:
        raise ValueError(f"Target profile {target_id} not found")

    discovered = _discover_optimizer_candidates(project_id=project_id)
    if not discovered:
        raise ValueError(
            (
                "No model artifacts were found for optimization. "
                "Create at least one trained model (or compressed artifact) before running target optimization."
            )
        )
    discovered_by_id = {
        str(item["id"]): dict(item)
        for item in discovered
        if str(item.get("id") or "").strip()
    }

    preferred_formats = {
        str(item or "").strip().lower()
        for item in list(target.constraints.preferred_formats or [])
        if str(item or "").strip()
    }
    target_device = target_profile_service.resolve_target_device(target.id, fallback="laptop")
    prompt_set = _ensure_optimization_prompt_set(project_id)
    benchmark_probe_candidates = [
        item
        for item in sorted(
            discovered,
            key=lambda item: (
                0 if item["runtime_template"] in preferred_formats else 1,
                item["size_bytes"],
            ),
        )
        if bool(item.get("probe_supported"))
    ][:3]
    benchmark_probe_ids = {str(item["id"]) for item in benchmark_probe_candidates}

    candidates: list[OptimizationCandidate] = []
    target_memory_budget_gb = _coerce_float_positive(target.constraints.min_vram_gb)
    for candidate in discovered:
        fallback_reason = ""
        measured_report: dict[str, Any] | None = None
        if str(candidate["id"]) in benchmark_probe_ids:
            measured_report, fallback_reason = await _run_optimizer_benchmark_probe(
                project_id=project_id,
                candidate=candidate,
            )
        elif not bool(candidate.get("probe_supported")):
            fallback_reason = (
                f"Runtime probe is not available for format '{candidate['runtime_template']}'."
            )
        else:
            fallback_reason = "Candidate skipped for runtime probe budget; metrics estimated from artifact profile."

        if measured_report is not None:
            metric, metric_source, metric_sources, measurement = _build_measured_candidate_metrics(
                candidate=candidate,
                report=measured_report,
            )
        else:
            metric, metric_source, metric_sources, measurement = _build_estimated_candidate_metrics(
                candidate=candidate,
                target_device=target_device,
                fallback_reason=fallback_reason or "Runtime probe did not complete.",
            )

        reasons: list[str] = []
        if candidate["runtime_template"] in preferred_formats:
            reasons.append("Matches target preferred runtime format.")
        elif preferred_formats:
            preferred_label = ", ".join(sorted(preferred_formats))
            reasons.append(f"Not a preferred target format ({preferred_label}); kept for tradeoff comparison.")

        if metric_source != "measured":
            fallback_text = str(measurement.get("fallback_reason") or "").strip()
            if fallback_text:
                reasons.append(fallback_text)
        else:
            reasons.append("Latency and quality proxy measured from a live local benchmark probe.")

        if (
            target_memory_budget_gb is not None
            and float(metric.memory_gb) > float(target_memory_budget_gb)
        ):
            reasons.append(
                (
                    f"Estimated memory footprint {metric.memory_gb:.3f} GB exceeds target baseline "
                    f"{target_memory_budget_gb:.3f} GB."
                )
            )

        candidates.append(
            OptimizationCandidate(
                id=str(candidate["id"]),
                name=str(candidate["name"]),
                quantization=str(candidate["quantization"]),
                runtime_template=str(candidate["runtime_template"]),
                metrics=metric,
                reasons=reasons,
                metric_source=metric_source,
                metric_sources=metric_sources,
                measurement=measurement,
            )
        )

    def _rank_key(row: OptimizationCandidate) -> tuple[float, float, float, float, float]:
        format_penalty = 0.0 if row.runtime_template in preferred_formats else 1.0
        memory_penalty = 0.0
        if target_memory_budget_gb is not None and row.metrics.memory_gb > target_memory_budget_gb:
            memory_penalty = 1.0
        source_priority = {
            "measured": 0.0,
            "mixed": 0.5,
            "estimated": 1.0,
        }.get(str(row.metric_source).strip().lower(), 1.0)
        return (
            memory_penalty,
            format_penalty,
            source_priority,
            float(row.metrics.latency_ms),
            -float(row.metrics.quality_score),
        )

    candidates.sort(key=_rank_key)
    if candidates:
        candidates[0].is_recommended = True
        candidates[0].reasons.append("Best measured/estimated latency-quality tradeoff for this target.")

    optimization_run = _persist_optimization_run(
        project_id=project_id,
        target_id=target_id,
        target_device=target_device,
        preferred_formats=preferred_formats,
        candidates=candidates,
        discovered_by_id=discovered_by_id,
        prompt_set=prompt_set,
    )
    return OptimizationResponse(
        project_id=project_id,
        target_id=target_id,
        candidates=candidates,
        optimization_run=optimization_run,
    )


def _coerce_float_positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return float(parsed)


def _artifact_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if path.is_dir():
        return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))
    return 0


def _artifact_file_count(path: Path) -> int:
    if path.is_file():
        return 1
    if path.is_dir():
        return int(sum(1 for item in path.rglob("*") if item.is_file()))
    return 0


def _normalize_quantization_token(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "none"
    return token


def _detect_quantization_from_name(name: str) -> str:
    token = str(name or "").strip().lower()
    if not token:
        return "none"
    patterns = (
        (r"q(?:uantized[_-]?)?([248])(?:bit)?", "{bits}bit"),
        (r"int([248])", "{bits}bit"),
        (r"([248])bit", "{bits}bit"),
    )
    for pattern, template in patterns:
        match = re.search(pattern, token)
        if match:
            bits = match.group(1)
            return template.format(bits=bits)
    if "awq" in token:
        return "awq"
    if "gptq" in token:
        return "gptq"
    if "fp16" in token or "f16" in token:
        return "fp16"
    return "none"


def _candidate_id(*, format_token: str, model_ref: Path, quantization: str) -> str:
    digest = hashlib.sha256(
        f"{format_token}|{model_ref.resolve().as_posix()}|{quantization}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{format_token}-{digest}"


def _coerce_quality_hint_from_eval_loss(value: Any) -> float | None:
    try:
        eval_loss = float(value)
    except (TypeError, ValueError):
        return None
    if eval_loss <= 0:
        return None
    score = 1.0 / (1.0 + eval_loss)
    return round(max(0.0, min(score, 1.0)), 4)


def _load_experiment_quality_hint(experiment_dir: Path) -> float | None:
    report_path = experiment_dir / "training_report.json"
    if report_path.exists() and report_path.is_file():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        return _coerce_quality_hint_from_eval_loss(payload.get("final_eval_loss"))
    return None


def _discover_optimizer_candidates(project_id: int) -> list[dict[str, Any]]:
    project_root = _project_dir(project_id)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    experiments_root = project_root / "experiments"
    if experiments_root.exists():
        experiment_dirs = [
            item for item in experiments_root.iterdir() if item.is_dir()
        ]
        experiment_dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        for experiment_dir in experiment_dirs:
            model_dir = experiment_dir / "model"
            if not model_dir.exists() or not model_dir.is_dir():
                continue
            config_path = model_dir / "config.json"
            has_weights = any(
                file_path.suffix.lower() in _MODEL_WEIGHT_SUFFIXES
                for file_path in model_dir.rglob("*")
                if file_path.is_file()
            )
            if not config_path.exists() or not has_weights:
                continue
            quantization = _detect_quantization_from_name(model_dir.name)
            candidate_id = _candidate_id(
                format_token="huggingface",
                model_ref=model_dir,
                quantization=quantization,
            )
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            quality_hint = _load_experiment_quality_hint(experiment_dir)
            size_bytes = _artifact_size_bytes(model_dir)
            file_count = _artifact_file_count(model_dir)
            candidates.append(
                {
                    "id": candidate_id,
                    "name": f"HUGGINGFACE ({quantization})",
                    "runtime_template": "huggingface",
                    "quantization": quantization,
                    "model_ref": model_dir,
                    "artifact_path": model_dir,
                    "size_bytes": size_bytes,
                    "file_count": file_count,
                    "probe_supported": True,
                    "quality_hint": quality_hint,
                    "source": "experiment_model_dir",
                }
            )

    compressed_root = project_root / "compressed"
    if compressed_root.exists():
        compressed_files = [
            item
            for item in compressed_root.rglob("*")
            if item.is_file() and item.suffix.lower() in {".gguf", ".onnx", ".engine", ".plan"}
        ]
        compressed_files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        for artifact in compressed_files:
            suffix = artifact.suffix.lower()
            format_token = {
                ".gguf": "gguf",
                ".onnx": "onnx",
                ".engine": "tensorrt",
                ".plan": "tensorrt",
            }.get(suffix)
            if not format_token:
                continue
            quantization = _detect_quantization_from_name(artifact.name)
            model_ref = artifact if format_token == "gguf" else artifact.parent
            candidate_id = _candidate_id(
                format_token=format_token,
                model_ref=model_ref,
                quantization=quantization,
            )
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            size_target = artifact if artifact.is_file() else model_ref
            size_bytes = _artifact_size_bytes(size_target)
            file_count = _artifact_file_count(size_target)
            candidates.append(
                {
                    "id": candidate_id,
                    "name": f"{format_token.upper()} ({quantization})",
                    "runtime_template": format_token,
                    "quantization": quantization,
                    "model_ref": model_ref,
                    "artifact_path": artifact,
                    "size_bytes": size_bytes,
                    "file_count": file_count,
                    "probe_supported": format_token in {"gguf"},
                    "quality_hint": None,
                    "source": "compressed_artifact",
                }
            )

    return candidates


async def _run_optimizer_benchmark_probe(
    *,
    project_id: int,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    if not bool(candidate.get("probe_supported")):
        return None, (
            f"Runtime probe is not available for format '{candidate.get('runtime_template')}'."
        )
    if not _BENCHMARK_SCRIPT.exists():
        return None, "Benchmark script is not available on this runtime."

    model_ref = Path(str(candidate.get("model_ref") or "")).expanduser()
    if not model_ref.exists():
        return None, f"Model artifact path for benchmark probe does not exist: {model_ref}"

    optimization_dir = _project_dir(project_id) / "optimization" / "benchmarks"
    optimization_dir.mkdir(parents=True, exist_ok=True)
    report_path = optimization_dir / f"{candidate['id']}.json"
    prompt_set = _ensure_optimization_prompt_set(project_id)
    command = [
        str(Path(sys.executable).expanduser()),
        str(_BENCHMARK_SCRIPT),
        "--project",
        str(project_id),
        "--model",
        str(model_ref),
        "--samples",
        "2",
        "--max-new-tokens",
        "24",
        "--warmup",
        "0",
        "--runtime",
        "auto",
        "--prompt-file",
        str(prompt_set["prompt_file_path"]),
        "--out",
        str(report_path),
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        return None, f"Failed to start benchmark probe: {exc}"

    timeout_seconds = max(30, min(int(settings.EXTERNAL_COMMAND_TIMEOUT_SECONDS), 180))
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        return None, f"Benchmark probe timed out after {timeout_seconds} seconds."

    stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip() if stdout_bytes else ""
    stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip() if stderr_bytes else ""
    if process.returncode != 0:
        message = stderr_text or stdout_text or "benchmark probe failed"
        return None, f"Benchmark probe failed: {message}"

    if not report_path.exists():
        return None, "Benchmark probe completed without generating a report."

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Failed to parse benchmark report: {exc}"

    payload["report_path"] = str(report_path)
    payload["report_sha256"] = _sha256_file(report_path)
    payload.setdefault("prompt_set", {})
    if isinstance(payload.get("prompt_set"), dict):
        payload["prompt_set"].setdefault("prompt_set_id", prompt_set["prompt_set_id"])
        payload["prompt_set"].setdefault("prompt_set_hash", prompt_set["prompt_set_hash"])
        payload["prompt_set"].setdefault("prompt_file_path", prompt_set["prompt_file_path"])
        payload["prompt_set"].setdefault("prompt_count", prompt_set["prompt_count"])
    if stdout_text:
        payload["probe_stdout"] = stdout_text
    return payload, ""


def _derive_quality_proxy_from_probe(report: dict[str, Any]) -> float | None:
    sample_rows = [item for item in list(report.get("sample_outputs") or []) if isinstance(item, dict)]
    if not sample_rows:
        return None

    outputs = [str(item.get("output_preview") or "").strip() for item in sample_rows]
    non_empty = [text for text in outputs if text]
    non_empty_ratio = len(non_empty) / max(1, len(outputs))
    avg_chars = sum(len(text) for text in non_empty) / max(1, len(non_empty))
    length_score = min(1.0, avg_chars / 120.0)
    unique_ratio = len({text.lower() for text in non_empty}) / max(1, len(non_empty))

    score = (0.55 * non_empty_ratio) + (0.30 * length_score) + (0.15 * unique_ratio)
    return round(max(0.0, min(score, 1.0)), 4)


def _resolve_metric_source(metric_sources: dict[str, str]) -> str:
    normalized = {
        str(value or "").strip().lower()
        for value in metric_sources.values()
        if str(value or "").strip()
    }
    if normalized == {"measured"}:
        return "measured"
    if "measured" in normalized:
        return "mixed"
    return "estimated"


def _estimate_latency_from_artifact(
    *,
    size_bytes: int,
    runtime_template: str,
    target_device: str,
) -> float:
    size_gb = max(float(size_bytes) / float(1024 ** 3), 0.001)
    base_by_device = {
        "mobile": 220.0,
        "laptop": 130.0,
        "server": 80.0,
    }
    runtime_factor = {
        "huggingface": 1.0,
        "gguf": 0.85 if target_device == "mobile" else 1.1,
        "onnx": 0.9,
        "tensorrt": 0.7,
    }.get(runtime_template, 1.0)
    base = base_by_device.get(target_device, 130.0)
    latency = base * runtime_factor * (1.0 + min(size_gb, 24.0) * 0.35)
    return round(max(latency, 1.0), 3)


def _quantization_penalty(quantization: str) -> float:
    token = _normalize_quantization_token(quantization)
    penalty = {
        "none": 0.0,
        "fp16": 0.01,
        "8bit": 0.03,
        "4bit": 0.08,
        "awq": 0.06,
        "gptq": 0.07,
    }.get(token, 0.05)
    return float(penalty)


def _build_measured_candidate_metrics(
    *,
    candidate: dict[str, Any],
    report: dict[str, Any],
) -> tuple[OptimizationMetric, str, dict[str, str], dict[str, Any]]:
    latency_summary = dict((report.get("metrics") or {}).get("latency") or {})
    latency_ms = _coerce_float_positive(latency_summary.get("p50_ms"))
    if latency_ms is None:
        latency_ms = _coerce_float_positive(latency_summary.get("avg_ms"))
    if latency_ms is None:
        latency_ms = _estimate_latency_from_artifact(
            size_bytes=int(candidate["size_bytes"]),
            runtime_template=str(candidate["runtime_template"]),
            target_device="laptop",
        )
        latency_source = "estimated"
    else:
        latency_source = "measured"

    model_size_bytes = int(report.get("model_size_bytes") or candidate["size_bytes"])
    memory_gb = round(float(model_size_bytes) / float(1024 ** 3), 4)
    memory_source = "measured"

    quality_score = _derive_quality_proxy_from_probe(report)
    if quality_score is None:
        hint = candidate.get("quality_hint")
        if isinstance(hint, (int, float)):
            quality_score = round(max(0.0, min(float(hint), 1.0)), 4)
            quality_source = "estimated"
        else:
            quality_score = round(max(0.0, 0.72 - _quantization_penalty(str(candidate["quantization"]))), 4)
            quality_source = "estimated"
    else:
        quality_source = "measured"

    metric = OptimizationMetric(
        latency_ms=round(float(latency_ms), 3),
        memory_gb=round(float(memory_gb), 4),
        quality_score=round(float(quality_score), 4),
    )
    metric_sources = {
        "latency_ms": latency_source,
        "memory_gb": memory_source,
        "quality_score": quality_source,
    }
    measurement = {
        "mode": "measured",
        "source": str(candidate.get("source") or ""),
        "model_ref": str(candidate.get("model_ref") or ""),
        "artifact_path": str(candidate.get("artifact_path") or ""),
        "report_path": report.get("report_path"),
        "report_sha256": report.get("report_sha256"),
        "runtime": report.get("runtime"),
        "benchmark_samples": report.get("benchmark_samples"),
        "latency_summary": latency_summary,
        "prompt_set": report.get("prompt_set"),
    }
    return metric, _resolve_metric_source(metric_sources), metric_sources, measurement


def _build_estimated_candidate_metrics(
    *,
    candidate: dict[str, Any],
    target_device: str,
    fallback_reason: str,
) -> tuple[OptimizationMetric, str, dict[str, str], dict[str, Any]]:
    size_bytes = int(candidate["size_bytes"])
    memory_gb = round(float(size_bytes) / float(1024 ** 3), 4)
    latency_ms = _estimate_latency_from_artifact(
        size_bytes=size_bytes,
        runtime_template=str(candidate["runtime_template"]),
        target_device=target_device,
    )
    hint = candidate.get("quality_hint")
    if isinstance(hint, (int, float)):
        quality_score = round(max(0.0, min(float(hint), 1.0)) - _quantization_penalty(str(candidate["quantization"])), 4)
        quality_score = round(max(0.0, min(quality_score, 1.0)), 4)
        quality_source = "estimated"
    else:
        quality_score = round(max(0.0, 0.72 - _quantization_penalty(str(candidate["quantization"]))), 4)
        quality_source = "estimated"

    metric = OptimizationMetric(
        latency_ms=round(float(latency_ms), 3),
        memory_gb=round(float(memory_gb), 4),
        quality_score=round(float(quality_score), 4),
    )
    metric_sources = {
        "latency_ms": "estimated",
        "memory_gb": "measured",
        "quality_score": quality_source,
    }
    measurement = {
        "mode": "estimated",
        "source": str(candidate.get("source") or ""),
        "model_ref": str(candidate.get("model_ref") or ""),
        "artifact_path": str(candidate.get("artifact_path") or ""),
        "fallback_reason": str(fallback_reason or "").strip() or "Runtime probe unavailable.",
    }
    return metric, _resolve_metric_source(metric_sources), metric_sources, measurement
