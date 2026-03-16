"""Data Ingestion service — handles file uploads, parsing, and storage."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.exceptions import StrictExecutionError
from app.models.dataset import Dataset, DatasetType, DocumentStatus, RawDocument
from app.models.project import Project
from app.services.domain_hook_service import (
    apply_normalizer_hook,
    resolve_project_domain_hooks,
    run_validator_hook,
)
from app.services.data_adapter_service import (
    DEFAULT_ADAPTER_ID,
    map_record_with_adapter,
    resolve_task_profile_for_adapter,
    resolve_data_adapter_for_records,
)
from app.services.record_normalization import build_schema_profile
from app.utils.file_parsers import (
    SUPPORTED_EXTENSIONS,
    compute_file_hash,
    get_file_type,
    parse_file,
)


def _project_data_dir(project_id: int) -> Path:
    """Return the data directory for a specific project."""
    d = settings.DATA_DIR / "projects" / str(project_id) / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_filename(filename: str) -> str:
    """Normalize to a basename so uploads cannot escape the project directory."""
    safe = Path(filename).name.strip()
    if not safe or safe in {".", ".."}:
        raise ValueError("Invalid filename")
    return safe


async def _get_document_for_project(
    db: AsyncSession,
    project_id: int,
    document_id: int,
) -> RawDocument | None:
    """Fetch a document only if it belongs to the requested project."""
    result = await db.execute(
        select(RawDocument)
        .join(Dataset, Dataset.id == RawDocument.dataset_id)
        .where(
            RawDocument.id == document_id,
            Dataset.project_id == project_id,
        )
    )
    return result.scalar_one_or_none()


async def get_or_create_raw_dataset(
    db: AsyncSession, project_id: int
) -> Dataset:
    """Get or create the 'raw' dataset for a project."""
    project = await db.execute(select(Project.id).where(Project.id == project_id))
    if not project.scalar_one_or_none():
        raise ValueError(f"Project {project_id} not found")

    result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.RAW,
        ).order_by(Dataset.updated_at.desc(), Dataset.id.desc()).limit(1)
    )
    dataset = result.scalars().first()
    if dataset:
        return dataset

    dataset = Dataset(
        project_id=project_id,
        name="Raw Documents",
        dataset_type=DatasetType.RAW,
        description="Ingested raw documents",
    )
    db.add(dataset)
    await db.flush()
    await db.refresh(dataset)
    return dataset


async def ingest_file(
    db: AsyncSession,
    project_id: int,
    filename: str,
    file_content: bytes,
    source: str = "upload",
    sensitivity: str = "internal",
    license_info: str = "",
) -> RawDocument:
    """Ingest a single file: save to disk, parse, create DB record."""
    safe_filename = _safe_filename(filename)

    # Validate extension
    ext = Path(safe_filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Get or create raw dataset
    dataset = await get_or_create_raw_dataset(db, project_id)

    # Save file to disk
    data_dir = _project_data_dir(project_id)
    file_path = data_dir / safe_filename
    # Handle duplicate filenames
    counter = 1
    while file_path.exists():
        stem = Path(safe_filename).stem
        file_path = data_dir / f"{stem}_{counter}{ext}"
        counter += 1

    file_path.write_bytes(file_content)

    # Compute hash
    file_hash = compute_file_hash(file_path)

    # Create document record
    doc = RawDocument(
        dataset_id=dataset.id,
        filename=file_path.name,
        file_type=get_file_type(safe_filename),
        file_path=str(file_path),
        file_size_bytes=len(file_content),
        source=source,
        sensitivity=sensitivity,
        license_info=license_info,
        status=DocumentStatus.PENDING,
        metadata_={"hash": file_hash, "original_name": safe_filename},
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    # Update dataset record count
    dataset.record_count += 1
    await db.flush()

    return doc


async def process_document(
    db: AsyncSession,
    project_id: int,
    document_id: int,
) -> RawDocument:
    """Process (parse) an ingested document and extract text content."""
    doc = await _get_document_for_project(db, project_id, document_id)
    if not doc:
        raise ValueError(f"Document {document_id} not found in project {project_id}")

    try:
        doc.status = DocumentStatus.PROCESSING
        await db.flush()

        file_path = Path(doc.file_path)
        text = parse_file(file_path)

        # Save extracted text
        text_path = file_path.with_suffix(".extracted.txt")
        text_path.write_text(text, encoding="utf-8")

        doc.status = DocumentStatus.ACCEPTED
        doc.metadata_ = {
            **(doc.metadata_ or {}),
            "extracted_text_path": str(text_path),
            "char_count": len(text),
            "word_count": len(text.split()),
        }
        await db.flush()
        await db.refresh(doc)

    except Exception as e:
        doc.status = DocumentStatus.ERROR
        doc.metadata_ = {**(doc.metadata_ or {}), "error": str(e)}
        await db.flush()
        await db.refresh(doc)

    return doc


async def list_documents(
    db: AsyncSession, project_id: int, status: DocumentStatus | None = None
) -> list[RawDocument]:
    """List all documents for a project, optionally filtered by status."""
    dataset = await get_or_create_raw_dataset(db, project_id)
    query = select(RawDocument).where(RawDocument.dataset_id == dataset.id)
    if status:
        query = query.where(RawDocument.status == status)
    query = query.order_by(RawDocument.ingested_at.desc())
    result = await db.execute(query)
    return list(result.scalars().all())


async def delete_document(
    db: AsyncSession,
    project_id: int,
    document_id: int,
) -> None:
    """Delete a document from disk and database."""
    doc = await _get_document_for_project(db, project_id, document_id)
    if not doc:
        raise ValueError(f"Document {document_id} not found in project {project_id}")

    # Remove file from disk
    file_path = Path(doc.file_path)
    if file_path.exists():
        file_path.unlink()
    extracted = file_path.with_suffix(".extracted.txt")
    if extracted.exists():
        extracted.unlink()

    await db.delete(doc)


# ── Remote Dataset Connectors ─────────────────────────────────────────

def _import_job_dir(project_id: int) -> Path:
    d = _project_data_dir(project_id) / "import_jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_import_report_path(project_id: int, report_path: str) -> Path:
    base = _import_job_dir(project_id).resolve()
    requested = Path(report_path).expanduser()
    if not requested.is_absolute():
        requested = (base / requested).resolve()
    else:
        requested = requested.resolve()
    if requested != base and base not in requested.parents:
        raise ValueError("report_path must be inside the project ingestion import-jobs directory")
    return requested


def _render_remote_row_text(row: dict) -> str:
    """Render a normalized imported row to human-readable plain text."""
    text = str(row.get("text") or "").strip()
    if text:
        return text

    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if question and answer:
        return f"Q: {question}\nA: {answer}"
    if question:
        return question
    if answer:
        return answer

    prompt = str(row.get("prompt") or row.get("instruction") or "").strip()
    completion = str(row.get("completion") or row.get("output") or "").strip()
    if prompt and completion:
        return f"Prompt: {prompt}\nCompletion: {completion}"
    if prompt:
        return prompt
    if completion:
        return completion

    for key in ("input_text", "target_text", "document", "content", "value"):
        token = str(row.get(key) or "").strip()
        if token:
            return token
    return ""


async def queue_remote_import(
    project_id: int,
    source_type: str,
    identifier: str,
    split: str = "train",
    max_samples: int | None = None,
    config_name: str | None = None,
    field_mapping: dict[str, str] | None = None,
    adapter_id: str = DEFAULT_ADAPTER_ID,
    task_profile: str | None = None,
    adapter_config: dict[str, Any] | None = None,
    normalize_for_training: bool = True,
    hf_token: str | None = None,
    kaggle_username: str | None = None,
    kaggle_key: str | None = None,
    use_saved_secrets: bool = True,
) -> dict:
    report_dir = _import_job_dir(project_id)
    job_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = report_dir / f"remote_import_{job_stamp}.json"

    from app.worker import celery_app

    request_payload = {
        "source_type": source_type,
        "identifier": identifier,
        "split": split,
        "max_samples": max_samples,
        "config_name": config_name,
        "field_mapping": field_mapping or {},
        "adapter_id": adapter_id,
        "task_profile": task_profile or None,
        "adapter_config": dict(adapter_config or {}),
        "normalize_for_training": normalize_for_training,
        "hf_token": hf_token or "",
        "kaggle_username": kaggle_username or "",
        "kaggle_key": kaggle_key or "",
        "use_saved_secrets": bool(use_saved_secrets),
    }

    task = celery_app.send_task(
        "run_remote_import_job",
        kwargs={
            "project_id": project_id,
            "request_payload": request_payload,
            "report_path": str(report_path),
        },
    )
    return {
        "project_id": project_id,
        "status": "queued",
        "source_type": source_type,
        "identifier": identifier,
        "adapter_id": adapter_id,
        "task_profile": task_profile or None,
        "report_path": str(report_path),
        "task_id": task.id,
    }


def get_import_job_status(project_id: int, report_path: str) -> dict:
    safe_report_path = _resolve_import_report_path(project_id, report_path)
    if not safe_report_path.exists():
        return {
            "project_id": project_id,
            "status": "running",
            "report_path": str(safe_report_path),
        }

    import json

    try:
        payload = json.loads(safe_report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse ingestion report file {safe_report_path}: {e}")

    digest = sha256(str(payload).encode("utf-8")).hexdigest()
    return {
        "project_id": project_id,
        "status": payload.get("status", "running"),
        "report_path": str(safe_report_path),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "source_type": payload.get("source_type"),
        "identifier": payload.get("identifier"),
        "result": payload.get("result"),
        "error": payload.get("error"),
        "output_digest": digest,
    }

async def ingest_remote_dataset(
    db: AsyncSession,
    project_id: int,
    source_type: str,
    identifier: str,
    split: str = "train",
    max_samples: int | None = None,
    config_name: str | None = None,
    field_mapping: dict[str, str] | None = None,
    adapter_id: str = DEFAULT_ADAPTER_ID,
    task_profile: str | None = None,
    adapter_config: dict[str, Any] | None = None,
    normalize_for_training: bool = True,
    hf_token: str | None = None,
    kaggle_username: str | None = None,
    kaggle_key: str | None = None,
    use_saved_secrets: bool = True,
    progress_callback: Callable[[str], Awaitable[None]] | None = None,
) -> dict:
    """
    Ingest a dataset from a remote source.

    source_type: 'huggingface', 'kaggle', or 'url'
    identifier:
        - huggingface: dataset ID like 'squad' or 'tatsu-lab/alpaca'
        - kaggle: dataset slug like 'user/dataset-name'
        - url: direct link to a .csv, .json, or .jsonl file
    """
    import csv
    import json
    import os
    import random
    import re
    import tempfile

    import httpx
    from app.services.secret_service import get_project_secret_value

    async def _progress(message: str) -> None:
        if progress_callback is None:
            return
        try:
            await progress_callback(message)
        except Exception:
            pass

    raw_dataset = await get_or_create_raw_dataset(db, project_id)
    data_dir = _project_data_dir(project_id)
    raw_samples: list[dict] = []
    source_mode = "simulated"
    target_samples = max_samples or 200
    hook_state = await resolve_project_domain_hooks(db, project_id)
    normalizer_hook_spec = hook_state.get("normalizer")
    validator_hook_spec = hook_state.get("validator")
    await _progress(
        f"[import] source={source_type} identifier={identifier} split={split} target_samples={target_samples}"
    )
    await _progress(
        f"[import] hooks normalizer={(normalizer_hook_spec or {}).get('id')} "
        f"validator={(validator_hook_spec or {}).get('id')}"
    )
    await _progress(
        f"[import] adapter requested={adapter_id or DEFAULT_ADAPTER_ID}"
    )
    if task_profile:
        await _progress(f"[import] task_profile requested={task_profile}")

    if use_saved_secrets and source_type == "huggingface" and not hf_token:
        saved_hf_token = await get_project_secret_value(db, project_id, "huggingface", "token")
        if saved_hf_token:
            hf_token = saved_hf_token
            await _progress("[hf] using saved HuggingFace token from project secrets")
    if use_saved_secrets and source_type == "kaggle":
        if not kaggle_username:
            saved_kaggle_username = await get_project_secret_value(
                db,
                project_id,
                "kaggle",
                "username",
            )
            if saved_kaggle_username:
                kaggle_username = saved_kaggle_username
        if not kaggle_key:
            saved_kaggle_key = await get_project_secret_value(db, project_id, "kaggle", "key")
            if saved_kaggle_key:
                kaggle_key = saved_kaggle_key
        if kaggle_username and kaggle_key:
            await _progress("[kaggle] using saved Kaggle credentials from project secrets")

    def _safe_remote_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")[:80] or "dataset"

    def _append_records_from_text(raw_text: str, ext: str, target_count: int) -> None:
        ext = ext.lower()
        remaining = max(0, target_count - len(raw_samples))
        if remaining == 0:
            return

        if ext == ".jsonl":
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_samples.append(record if isinstance(record, dict) else {"value": record})
                if len(raw_samples) >= target_count:
                    break
            return

        if ext == ".json":
            data = json.loads(raw_text)
            if isinstance(data, list):
                for row in data[:remaining]:
                    raw_samples.append(row if isinstance(row, dict) else {"value": row})
            elif isinstance(data, dict):
                raw_samples.append(data)
            else:
                raw_samples.append({"value": data})
            return

        if ext == ".csv":
            reader = csv.DictReader(raw_text.splitlines())
            for row in reader:
                raw_samples.append(dict(row))
                if len(raw_samples) >= target_count:
                    break
            return

        if ext == ".tsv":
            reader = csv.DictReader(raw_text.splitlines(), delimiter="\t")
            for row in reader:
                raw_samples.append(dict(row))
                if len(raw_samples) >= target_count:
                    break
            return

        for i, line in enumerate(raw_text.splitlines()):
            if i >= remaining:
                break
            line = line.strip()
            if line:
                raw_samples.append({"content": line})

    async def _collect_hf_rows_from_repo_files() -> int:
        from huggingface_hub import HfApi, hf_hub_download

        token = hf_token or None
        api = HfApi(token=token)
        repo_files = list(api.list_repo_files(identifier, repo_type="dataset"))
        await _progress(f"[hf] repo file fallback discovered {len(repo_files)} files")

        allowed_exts = {".jsonl", ".json", ".csv", ".tsv", ".txt"}
        candidate_files = [
            item
            for item in repo_files
            if Path(str(item)).suffix.lower() in allowed_exts
        ]
        if config_name:
            config_token = f"/{str(config_name).strip().strip('/')}/"
            candidate_files = [
                item for item in candidate_files
                if config_token in f"/{item}/"
            ]
        if not candidate_files:
            raise ValueError("No parseable data files found in dataset repository")

        split_token = str(split or "").strip().lower()
        split_aliases = {split_token} if split_token else set()
        if split_token == "validation":
            split_aliases.update({"val", "dev"})
        if split_token == "test":
            split_aliases.update({"eval", "evaluation"})

        def _split_rank(path: str) -> int:
            lowered = str(path).lower()
            if not split_aliases:
                return 1
            for alias in split_aliases:
                if not alias:
                    continue
                if (
                    f"/{alias}." in lowered
                    or f"_{alias}." in lowered
                    or f"-{alias}." in lowered
                    or f"/{alias}/" in lowered
                    or lowered.endswith(f"{alias}.jsonl")
                    or lowered.endswith(f"{alias}.json")
                    or lowered.endswith(f"{alias}.csv")
                    or lowered.endswith(f"{alias}.tsv")
                    or lowered.endswith(f"{alias}.txt")
                ):
                    return 0
            return 1

        ordered_files = sorted(candidate_files, key=lambda item: (_split_rank(item), len(str(item))))
        for idx, repo_file in enumerate(ordered_files, start=1):
            if len(raw_samples) >= target_samples:
                break
            local_path = hf_hub_download(
                repo_id=identifier,
                filename=repo_file,
                repo_type="dataset",
                token=token,
            )
            ext = Path(repo_file).suffix.lower()
            raw_text = Path(local_path).read_text(encoding="utf-8", errors="replace")
            _append_records_from_text(raw_text, ext, target_samples)
            if idx <= 3 or idx % 20 == 0:
                await _progress(
                    f"[hf] repo fallback parsed {idx}/{len(ordered_files)} files, rows={len(raw_samples)}"
                )
        return len(raw_samples)

    if source_type == "huggingface":
        safe_name = _safe_remote_name(identifier.replace("/", "_"))
        filename = f"hf_{safe_name}_{split}.jsonl"
        try:
            from datasets import load_dataset

            dataset_kwargs = {"split": split}
            if config_name:
                dataset_kwargs["name"] = config_name
            if hf_token:
                dataset_kwargs["token"] = hf_token
            await _progress("[hf] loading dataset from HuggingFace Hub...")
            hf_dataset = load_dataset(identifier, **dataset_kwargs)
            await _progress("[hf] dataset loaded. Collecting rows...")
            for i, row in enumerate(hf_dataset):
                if i >= target_samples:
                    break
                if isinstance(row, dict):
                    raw_samples.append(row)
                else:
                    raw_samples.append({"value": row})
                if (i + 1) % 200 == 0:
                    await _progress(f"[hf] collected {i + 1} rows")
            source_mode = "live"
            await _progress(f"[hf] collected {len(raw_samples)} rows")
        except Exception as e:
            load_error = str(e)
            used_repo_fallback = False
            if "Dataset scripts are no longer supported" in load_error:
                await _progress(
                    "[hf] load_dataset() rejected legacy dataset script; trying repository file fallback..."
                )
                try:
                    parsed_rows = await _collect_hf_rows_from_repo_files()
                    if parsed_rows > 0:
                        source_mode = "live_hub_files"
                        used_repo_fallback = True
                        await _progress(f"[hf] repository file fallback collected {parsed_rows} rows")
                except Exception as repo_fallback_error:
                    await _progress(f"[hf] repository file fallback failed: {repo_fallback_error}")
                    load_error = f"{load_error}; fallback_error={repo_fallback_error}"

            if used_repo_fallback:
                pass
            elif settings.STRICT_EXECUTION_MODE:
                raise StrictExecutionError("ingestion", f"HuggingFace live import failed and STRICT_EXECUTION_MODE is enabled. Details: {load_error}")
            elif not settings.ALLOW_SIMULATED_INGESTION_FALLBACK:
                raise ValueError(
                    "HuggingFace import failed. Configure dependencies/network/auth and retry "
                    "(or set ALLOW_SIMULATED_INGESTION_FALLBACK=true for demo fallback). "
                    f"Details: {load_error}"
                )
            else:
                await _progress("[hf] live import failed; using simulated fallback rows")
                for i in range(target_samples):
                    raw_samples.append({
                        "instruction": f"Sample instruction #{i+1} from {identifier}",
                        "input": f"Context for sample {i+1}",
                        "output": f"Expected output for sample {i+1} from the {identifier} dataset.",
                        "source": f"huggingface/{identifier}",
                        "split": split,
                    })

    elif source_type == "kaggle":
        safe_name = _safe_remote_name(identifier.replace("/", "_"))
        filename = f"kaggle_{safe_name}.jsonl"
        prev_kaggle_user = os.environ.get("KAGGLE_USERNAME")
        prev_kaggle_key = os.environ.get("KAGGLE_KEY")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            if kaggle_username:
                os.environ["KAGGLE_USERNAME"] = kaggle_username
            if kaggle_key:
                os.environ["KAGGLE_KEY"] = kaggle_key

            with tempfile.TemporaryDirectory(prefix="slm_kaggle_") as tmp_dir:
                await _progress("[kaggle] authenticating and downloading dataset...")
                api = KaggleApi()
                api.authenticate()
                api.dataset_download_files(identifier, path=tmp_dir, unzip=True, quiet=True)

                candidate_files = [
                    p
                    for p in Path(tmp_dir).rglob("*")
                    if p.is_file() and p.suffix.lower() in {".jsonl", ".json", ".csv", ".txt", ".md"}
                ]
                if not candidate_files:
                    raise ValueError("No supported files found in downloaded Kaggle dataset")

                await _progress(f"[kaggle] parsing {len(candidate_files)} extracted files...")
                for fpath in candidate_files:
                    raw_text = fpath.read_text(encoding="utf-8", errors="replace")
                    _append_records_from_text(raw_text, fpath.suffix.lower(), target_samples)
                    if len(raw_samples) >= target_samples:
                        break

            if not raw_samples:
                raise ValueError("No records parsed from Kaggle dataset")
            source_mode = "live"
            await _progress(f"[kaggle] collected {len(raw_samples)} rows")
        except Exception as e:
            if settings.STRICT_EXECUTION_MODE:
                raise StrictExecutionError("ingestion", f"Kaggle live import failed and STRICT_EXECUTION_MODE is enabled. Details: {e}")
            if not settings.ALLOW_SIMULATED_INGESTION_FALLBACK:
                raise ValueError(
                    "Kaggle import failed. Configure kaggle API credentials/dependency and retry "
                    "(or set ALLOW_SIMULATED_INGESTION_FALLBACK=true for demo fallback). "
                    f"Details: {e}"
                )
            await _progress("[kaggle] live import failed; using simulated fallback rows")
            for i in range(target_samples):
                raw_samples.append({
                    "text": f"Row {i+1} from Kaggle dataset {identifier}.",
                    "label": random.choice(["positive", "negative", "neutral"]),
                    "source": f"kaggle/{identifier}",
                })
        finally:
            if prev_kaggle_user is None:
                os.environ.pop("KAGGLE_USERNAME", None)
            else:
                os.environ["KAGGLE_USERNAME"] = prev_kaggle_user
            if prev_kaggle_key is None:
                os.environ.pop("KAGGLE_KEY", None)
            else:
                os.environ["KAGGLE_KEY"] = prev_kaggle_key

    elif source_type == "url":
        safe_name = _safe_remote_name(Path(identifier).stem or "download")
        filename = f"url_{safe_name}.jsonl"
        try:
            await _progress("[url] downloading remote file...")
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                resp = await client.get(identifier)
                resp.raise_for_status()
                raw_text = resp.text

            ext = Path(identifier).suffix.lower()
            _append_records_from_text(raw_text, ext, target_samples)

            if not raw_samples:
                raise ValueError("No records parsed from URL response")
            source_mode = "live"
            await _progress(f"[url] parsed {len(raw_samples)} rows")
        except Exception as e:
            raise ValueError(f"URL import failed: {e}")

    else:
        raise ValueError(f"Unsupported source_type: {source_type}. Use 'huggingface', 'kaggle', or 'url'.")

    if not raw_samples:
        raise ValueError(f"No records available from source '{source_type}:{identifier}'")

    await _progress(f"[import] normalizing {len(raw_samples)} rows for training...")
    if normalize_for_training:
        source_rows = [sample for sample in raw_samples if isinstance(sample, dict)]
        dropped_records = len(raw_samples) - len(source_rows)
        resolved_adapter_id, detection_scores = resolve_data_adapter_for_records(
            source_rows,
            adapter_id=adapter_id,
            adapter_config=adapter_config,
            field_mapping=field_mapping,
            task_profile=task_profile,
        )
        resolved_task_profile = resolve_task_profile_for_adapter(
            resolved_adapter_id,
            requested_task_profile=task_profile,
        )
        await _progress(
            f"[import] adapter resolved={resolved_adapter_id} scores={detection_scores}"
        )

        rows_to_write: list[dict] = []
        for sample in source_rows:
            mapped = map_record_with_adapter(
                sample,
                adapter_id=resolved_adapter_id,
                adapter_config=adapter_config,
                field_mapping=field_mapping,
                task_profile=resolved_task_profile,
            )
            normalized = apply_normalizer_hook(sample, mapped, normalizer_hook_spec)
            if normalized:
                normalized["_adapter_id"] = resolved_adapter_id
                normalized["_task_profile"] = resolved_task_profile
                rows_to_write.append(normalized)
            else:
                dropped_records += 1
        if not rows_to_write:
            raise ValueError(
                (
                    "Unable to normalize imported records. Provide field_mapping or "
                    "switch adapter_id to match your source schema."
                )
            )
    else:
        rows_to_write = [s if isinstance(s, dict) else {"value": s} for s in raw_samples]
        dropped_records = 0
        resolved_adapter_id = "none"
        resolved_task_profile = None
        detection_scores: dict[str, float] = {}
    await _progress(f"[import] normalized rows={len(rows_to_write)} dropped={dropped_records}")

    schema_profile = build_schema_profile(
        raw_samples,
        field_mapping=field_mapping if normalize_for_training else None,
    )
    validator_report = run_validator_hook(
        rows_to_write if normalize_for_training else [],
        base_profile=schema_profile,
        hook_spec=validator_hook_spec,
    )

    # Write JSONL file
    file_path = data_dir / filename
    counter = 1
    while file_path.exists():
        stem = file_path.stem
        file_path = data_dir / f"{stem}_{counter}.jsonl"
        counter += 1

    await _progress(f"[import] writing dataset file: {file_path.name}")
    with open(file_path, "w", encoding="utf-8") as f:
        for sample in rows_to_write:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    extracted_path = file_path.with_suffix(".extracted.txt")
    extracted_fragments = [_render_remote_row_text(sample) for sample in rows_to_write]
    extracted_fragments = [frag for frag in extracted_fragments if frag]
    if extracted_fragments:
        extracted_text = "\n\n".join(extracted_fragments)
        extracted_path.write_text(extracted_text, encoding="utf-8")
        extracted_char_count = len(extracted_text)
        extracted_word_count = len(extracted_text.split())
    else:
        extracted_char_count = 0
        extracted_word_count = 0

    file_content = file_path.read_bytes()
    file_hash = compute_file_hash(file_path)

    doc = RawDocument(
        dataset_id=raw_dataset.id,
        filename=file_path.name,
        file_type="jsonl",
        file_path=str(file_path),
        file_size_bytes=len(file_content),
        source=f"{source_type}:{identifier}",
        sensitivity="public",
        license_info=f"Imported from {source_type}",
        status=DocumentStatus.ACCEPTED,
        metadata_={
            "hash": file_hash,
            "original_name": filename,
            "source_type": source_type,
            "identifier": identifier,
            "split": split,
            "raw_samples": len(raw_samples),
            "num_samples": len(rows_to_write),
            "dropped_records": dropped_records,
            "normalize_for_training": normalize_for_training,
            "field_mapping": field_mapping or {},
            "adapter_id_requested": adapter_id,
            "adapter_id_resolved": resolved_adapter_id,
            "task_profile_requested": task_profile or None,
            "task_profile_resolved": resolved_task_profile,
            "adapter_config": dict(adapter_config or {}),
            "adapter_detection_scores": detection_scores,
            "schema_profile": schema_profile,
            "domain_hooks": hook_state,
            "validator_report": validator_report,
            "config_name": config_name,
            "source_mode": source_mode,
            "extracted_text_path": str(extracted_path) if extracted_fragments else None,
            "char_count": extracted_char_count,
            "word_count": extracted_word_count,
        },
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    raw_dataset.record_count += 1
    await db.flush()
    await _progress(
        f"[import] completed document_id={doc.id} file={doc.filename} samples_ingested={len(rows_to_write)}"
    )

    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "source_type": source_type,
        "identifier": identifier,
        "raw_samples": len(raw_samples),
        "samples_ingested": len(rows_to_write),
        "dropped_records": dropped_records,
        "adapter_id_requested": adapter_id,
        "adapter_id_resolved": resolved_adapter_id,
        "task_profile_requested": task_profile or None,
        "task_profile_resolved": resolved_task_profile,
        "adapter_detection_scores": detection_scores,
        "file_size_bytes": len(file_content),
        "source_mode": source_mode,
        "schema_profile": schema_profile,
        "domain_hooks": hook_state,
        "validator_report": validator_report,
        "status": "accepted",
    }
