"""Continued pretraining for projects whose data is plain documents.

A docs-only project (task shape ``language_modeling``: cleaned PDF / DOCX /
HTML passages, no answer column) should train with
``training_mode="domain_pretrain"`` — packed next-token prediction over the
documents — not SFT, which the data gate refuses for text-only rows. This
module decides the mode when the caller didn't, and fills CPT-appropriate
defaults for fields the caller didn't set:

* LoRA on every linear layer (domain knowledge lives largely in the MLPs,
  which the SFT default ``q_proj, v_proj`` never touches), higher rank;
* a gentler learning rate and a bit more warmup (less forgetting);
* sequence packing on (the trainer packs documents into full blocks).
"""

from __future__ import annotations

from typing import Any

from app.models.experiment import TrainingMode

CPT_DEFAULTS: dict[str, Any] = {
    "use_lora": True,
    "target_modules": "all-linear",
    "lora_r": 64,
    "lora_alpha": 128,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.05,
    "sequence_packing": True,
}


def project_is_documents_only(project: Any) -> bool:
    preset = getattr(project, "dataset_adapter_preset", None)
    if not isinstance(preset, dict):
        return False
    return str(preset.get("task_profile") or "").strip().lower() == "language_modeling"


def resolve_training_mode(
    project: Any,
    requested: TrainingMode | str | None,
    *,
    mode_was_provided: bool,
) -> TrainingMode:
    """The caller's mode when they chose one; otherwise continued pretraining
    for documents-only projects and SFT for everything else."""
    if mode_was_provided and requested:
        return TrainingMode(requested) if not isinstance(requested, TrainingMode) else requested
    if project_is_documents_only(project):
        return TrainingMode.DOMAIN_PRETRAIN
    if requested:
        return TrainingMode(requested) if not isinstance(requested, TrainingMode) else requested
    return TrainingMode.SFT


def apply_cpt_defaults(config: dict[str, Any], provided_fields: set[str]) -> list[str]:
    """Fill CPT defaults for keys the caller didn't set. Returns the keys applied."""
    applied: list[str] = []
    for key, value in CPT_DEFAULTS.items():
        if key in provided_fields:
            continue
        config[key] = value
        applied.append(key)
    return applied
