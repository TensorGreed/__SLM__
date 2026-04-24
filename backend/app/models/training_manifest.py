"""Immutable training-run manifest (priority.md P14).

One manifest per training run (keyed 1:1 on ``experiment_id``, which is
this codebase's `run id` — see `app/models/experiment.py`). Captures at
launch time everything needed to **reproduce** a run deterministically:
dataset snapshot ids, adapter + blueprint + base-model registry
revisions, the fully-resolved recipe/tokenizer/runtime/seed, and the
environment the process was launched in (git sha, pip-freeze blob +
hash).

The table is append-only from the API's perspective: once written, the
``captured_at`` row is never mutated. Wave D's P15 (rerun-from-manifest)
reads these rows verbatim to replay a run.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrainingManifest(Base):
    __tablename__ = "training_manifests"
    __table_args__ = (
        UniqueConstraint("experiment_id", name="uq_training_manifests_experiment_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # The "run id" keyed on. Experiments are 1:1 with runs in this codebase,
    # so the foreign key is unique — one row per experiment, forever.
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"),
        nullable=False,
        index=True,
    )

    # Provenance of inputs — stored as ids into existing registry tables so
    # consumers can JOIN back to the live record. Revisions mirror the
    # `(id, version)` shape that blueprint / adapter tables already expose.
    base_model_registry_id: Mapped[int | None] = mapped_column(
        ForeignKey("base_model_registry_entries.id"),
        nullable=True,
        index=True,
    )
    base_model_cache_fingerprint: Mapped[str | None] = mapped_column(
        String(128), default=None
    )
    base_model_source_ref: Mapped[str | None] = mapped_column(String(512), default=None)

    dataset_adapter_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_adapter_definitions.id"),
        nullable=True,
        index=True,
    )
    dataset_adapter_version: Mapped[int | None] = mapped_column(Integer, default=None)

    blueprint_revision_id: Mapped[int | None] = mapped_column(
        ForeignKey("domain_blueprints.id"),
        nullable=True,
        index=True,
    )
    blueprint_version: Mapped[int | None] = mapped_column(Integer, default=None)

    # Dataset snapshot ids (train / val / test) — stored as a JSON list of
    # ``{dataset_id, dataset_version_id, role}`` so we can capture any number
    # of splits without stacking columns.
    dataset_snapshot_ids: Mapped[list] = mapped_column(JSON, default=list)

    # Fully-resolved launch config. Captured straight from `exp.config`
    # post-resolution so rerun can replay without re-running resolution.
    recipe_id: Mapped[str | None] = mapped_column(String(256), default=None)
    runtime_id: Mapped[str | None] = mapped_column(String(128), default=None, index=True)
    training_mode: Mapped[str | None] = mapped_column(String(64), default=None)
    tokenizer_name: Mapped[str | None] = mapped_column(String(256), default=None)
    tokenizer_config_hash: Mapped[str | None] = mapped_column(String(128), default=None)
    seed: Mapped[int | None] = mapped_column(Integer, default=None)
    resolved_config: Mapped[dict] = mapped_column(JSON, default=dict)

    # Environment of the launching process. `git_sha` is a best-effort
    # `git rev-parse HEAD`; `pip_freeze_blob` is the raw `pip freeze` output;
    # `env_digest` is the sha256 of (git_sha + pip_freeze_blob + runtime_id)
    # so tooling can quickly tell two manifests apart.
    git_sha: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    pip_freeze_blob: Mapped[str | None] = mapped_column(Text, default=None)
    pip_freeze_hash: Mapped[str | None] = mapped_column(String(128), default=None)
    env_digest: Mapped[str | None] = mapped_column(String(128), default=None, index=True)

    # Output artifacts referenced at capture time — usually just the
    # output_dir + config_path + prepared dir; other artifacts (checkpoints,
    # exports) accumulate post-run and are looked up by experiment_id on
    # demand instead of being copied in here.
    artifact_ids: Mapped[dict] = mapped_column(JSON, default=dict)

    # Any soft failures during capture land here (e.g. "git unavailable";
    # "pip freeze timed out") so we can diagnose partial manifests without
    # blocking the training launch. Empty list on clean captures.
    capture_warnings: Mapped[list] = mapped_column(JSON, default=list)

    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TrainingManifest exp={self.experiment_id} git={self.git_sha}>"
