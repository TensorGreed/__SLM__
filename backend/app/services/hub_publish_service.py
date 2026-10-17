"""Publish a completed Hugging Face-format export to the Hugging Face Hub.

The export's ``model/`` directory (weights, tokenizer, ``README.md`` model
card, ``thresholds.json`` …) becomes the repo root; ``manifest.json`` and
``RELEASE_NOTES.md`` go under ``brewslm/`` so the provenance travels with
the model. The card's front matter gains ``license`` / ``language`` when it
has none (the Hub wants them; the card builder never guesses a licence).
The publication is recorded on the export's manifest (``hub_publish``:
repo id, URL, commit, time) — on disk and in the DB.

The token is never stored: explicit ``token`` > ``HF_TOKEN`` env > the
``huggingface-cli login`` cache. ``api`` is injectable so tests never
touch the network."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.export import Export, ExportFormat, ExportStatus

REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
DEFAULT_LICENSE = "mit"
PROVENANCE_DIR = "brewslm"
PROVENANCE_FILES = ("manifest.json", "RELEASE_NOTES.md")


class HubPublishError(ValueError):
    """A publish that cannot proceed (bad repo id, no token, export not ready)."""


def validate_repo_id(repo_id: str) -> str:
    token = str(repo_id or "").strip()
    if not REPO_ID_RE.match(token):
        raise HubPublishError("repo_id must be <owner>/<name> (letters, digits, '-', '_', '.').")
    return token


def resolve_hub_token(token: str | None = None) -> str:
    """Explicit token > HF_TOKEN env > the huggingface_hub login cache."""
    explicit = str(token or "").strip()
    if explicit:
        return explicit
    env = str(os.environ.get("HF_TOKEN") or "").strip()
    if env:
        return env
    try:
        from huggingface_hub import get_token

        cached = get_token()
    except Exception:  # noqa: BLE001 — no hub lib / no cache → the message below
        cached = None
    if cached:
        return str(cached)
    raise HubPublishError(
        "No Hugging Face token: pass one in the request, set HF_TOKEN, or run `huggingface-cli login`."
    )


def card_with_hub_metadata(card: str, *, license: str | None = DEFAULT_LICENSE, language: str | None = "en") -> str:
    """Add ``license`` / ``language`` to the card's YAML front matter when it
    has neither; a card without front matter gets a minimal block. Pure."""
    text = card or ""
    extra: list[str] = []
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end == -1:
            return text
        front = text[4:end]
        keys = {line.split(":", 1)[0].strip() for line in front.splitlines() if ":" in line and not line.startswith(" ") and not line.startswith("-")}
        if license and "license" not in keys:
            extra.append(f"license: {license}")
        if language and "language" not in keys:
            extra += ["language:", f"- {language}"]
        if not extra:
            return text
        return text[: end + 1] + "\n".join(extra) + text[end:]
    if license:
        extra.append(f"license: {license}")
    if language:
        extra += ["language:", f"- {language}"]
    return "---\n" + "\n".join(extra) + "\n---\n\n" + text if extra else text


def _run_dir(export: Export) -> Path:
    from app.services.export_service import _resolve_export_run_dir_from_export

    run_dir = _resolve_export_run_dir_from_export(export)
    if run_dir is None:
        raise HubPublishError(f"Export {export.id} has no run directory on disk.")
    return run_dir


def _ready_model_dir(export: Export) -> tuple[Path, Path]:
    if export.status != ExportStatus.COMPLETED:
        raise HubPublishError(f"Export {export.id} is {export.status.value}; only a completed export can be published.")
    if export.export_format != ExportFormat.HUGGINGFACE:
        raise HubPublishError(
            f"Export {export.id} is a {export.export_format.value} export; publish a Hugging Face-format export."
        )
    run_dir = _run_dir(export)
    model_dir = run_dir / "model"
    if not (model_dir / "config.json").exists():
        raise HubPublishError(f"Export {export.id} has no model/config.json under {run_dir}.")
    return run_dir, model_dir


def hub_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


async def publish_export_to_hub(
    db: AsyncSession,
    project_id: int,
    export_id: int,
    *,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    license: str | None = DEFAULT_LICENSE,
    commit_message: str | None = None,
    api: Any | None = None,
) -> dict[str, Any]:
    """Create (or reuse) the repo, upload the model folder + provenance,
    record ``hub_publish`` on the export's manifest. Returns the record."""
    repo_id = validate_repo_id(repo_id)
    export = await db.get(Export, int(export_id))
    if export is None or int(export.project_id) != int(project_id):
        raise ValueError(f"Export {export_id} not found")
    run_dir, model_dir = _ready_model_dir(export)

    card_path = model_dir / "README.md"
    if card_path.exists():
        updated = card_with_hub_metadata(card_path.read_text(encoding="utf-8"), license=license)
        if updated != card_path.read_text(encoding="utf-8"):
            card_path.write_text(updated, encoding="utf-8")

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=resolve_hub_token(token))

    message = commit_message or f"BrewSLM export {export.id} (run #{export.experiment_id})"
    api.create_repo(repo_id=repo_id, repo_type="model", private=bool(private), exist_ok=True)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=message,
        ignore_patterns=["*.lock", "__pycache__/*"],
    )
    uploaded = sorted(p.name for p in model_dir.iterdir() if p.is_file())
    provenance: list[str] = []
    for name in PROVENANCE_FILES:
        src = run_dir / name
        if src.exists():
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(src),
                path_in_repo=f"{PROVENANCE_DIR}/{name}",
                commit_message=f"{message}: {name}",
            )
            provenance.append(f"{PROVENANCE_DIR}/{name}")

    record = {
        "repo_id": repo_id,
        "url": hub_url(repo_id),
        "private": bool(private),
        "commit": getattr(info, "oid", None),
        "commit_url": getattr(info, "commit_url", None),
        "published_at": datetime.now(timezone.utc).isoformat(),
        "files": uploaded + provenance,
        "license": license,
    }
    manifest = dict(export.manifest or {})
    manifest["hub_publish"] = record
    for path in (run_dir / "manifest.json", Path(export.output_path or run_dir.parent) / "manifest.json"):
        try:
            if path.parent.exists():
                path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except OSError:
            pass
    export.manifest = manifest  # a new dict, so the JSON column is marked dirty
    await db.commit()
    return record
