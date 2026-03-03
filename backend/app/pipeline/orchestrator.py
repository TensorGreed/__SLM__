"""Pipeline orchestrator — state machine for project pipeline flow."""

from app.models.project import PipelineStage

# Define valid stage transitions
STAGE_ORDER: list[PipelineStage] = [
    PipelineStage.INGESTION,
    PipelineStage.CLEANING,
    PipelineStage.GOLD_SET,
    PipelineStage.SYNTHETIC,
    PipelineStage.DATASET_PREP,
    PipelineStage.TOKENIZATION,
    PipelineStage.TRAINING,
    PipelineStage.EVALUATION,
    PipelineStage.COMPRESSION,
    PipelineStage.EXPORT,
    PipelineStage.COMPLETED,
]

STAGE_DISPLAY_NAMES: dict[PipelineStage, str] = {
    PipelineStage.INGESTION: "Data Ingestion",
    PipelineStage.CLEANING: "Data Cleaning",
    PipelineStage.GOLD_SET: "Gold Dataset",
    PipelineStage.SYNTHETIC: "Synthetic Generation",
    PipelineStage.DATASET_PREP: "Dataset Preparation",
    PipelineStage.TOKENIZATION: "Tokenization",
    PipelineStage.TRAINING: "Training",
    PipelineStage.EVALUATION: "Evaluation",
    PipelineStage.COMPRESSION: "Compression",
    PipelineStage.EXPORT: "Export",
    PipelineStage.COMPLETED: "Completed",
}


def get_stage_index(stage: PipelineStage) -> int:
    """Return 0-based index of a pipeline stage."""
    return STAGE_ORDER.index(stage)


def get_next_stage(current: PipelineStage) -> PipelineStage | None:
    """Return the next pipeline stage, or None if already completed."""
    idx = get_stage_index(current)
    if idx >= len(STAGE_ORDER) - 1:
        return None
    return STAGE_ORDER[idx + 1]


def get_prev_stage(current: PipelineStage) -> PipelineStage | None:
    """Return the previous pipeline stage, or None if at start."""
    idx = get_stage_index(current)
    if idx <= 0:
        return None
    return STAGE_ORDER[idx - 1]


def can_advance(current: PipelineStage, target: PipelineStage) -> bool:
    """Check if advancing from current to target is valid (forward only, one step)."""
    current_idx = get_stage_index(current)
    target_idx = get_stage_index(target)
    return target_idx == current_idx + 1


def can_rollback(current: PipelineStage, target: PipelineStage) -> bool:
    """Check if rolling back from current to target is valid (any prior stage)."""
    return get_stage_index(target) < get_stage_index(current)


def get_progress_percent(stage: PipelineStage) -> float:
    """Return pipeline completion percentage (0.0–100.0)."""
    idx = get_stage_index(stage)
    total = len(STAGE_ORDER) - 1  # exclude COMPLETED as a "step"
    return round((idx / total) * 100, 1)


def get_pipeline_status(current_stage: PipelineStage) -> list[dict]:
    """Return full pipeline status with each stage's completion state."""
    current_idx = get_stage_index(current_stage)
    result = []
    for i, stage in enumerate(STAGE_ORDER):
        if stage == PipelineStage.COMPLETED:
            continue
        status = "completed" if i < current_idx else ("active" if i == current_idx else "pending")
        result.append({
            "stage": stage.value,
            "display_name": STAGE_DISPLAY_NAMES[stage],
            "index": i,
            "status": status,
        })
    return result
