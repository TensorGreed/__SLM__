"""Preference alignment scaffolding: DPO/ORPO contracts and judge scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.evaluation_service import _heuristic_judge_score, f1_score


ALIGNMENT_RECIPES: list[dict[str, Any]] = [
    {
        "recipe_id": "recipe.alignment.dpo.fast",
        "display_name": "DPO Fast Start",
        "training_mode": "dpo",
        "description": "Direct preference optimization with conservative defaults for first run success.",
        "required_fields": ["prompt", "chosen", "rejected"],
        "config_patch": {
            "training_mode": "dpo",
            "task_type": "causal_lm",
            "trainer_backend": "hf_trainer",
            "chat_template": "chatml",
            "use_lora": True,
            "learning_rate": 1e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 2048,
            "num_epochs": 2,
        },
    },
    {
        "recipe_id": "recipe.alignment.orpo.safe",
        "display_name": "ORPO Safe Start",
        "training_mode": "orpo",
        "description": "Odds-ratio preference optimization tuned for stability on limited VRAM.",
        "required_fields": ["prompt", "chosen", "rejected"],
        "config_patch": {
            "training_mode": "orpo",
            "task_type": "causal_lm",
            "trainer_backend": "hf_trainer",
            "chat_template": "chatml",
            "use_lora": True,
            "learning_rate": 8e-6,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 2048,
            "num_epochs": 2,
        },
    },
]


_PROMPT_KEYS = ("prompt", "question", "instruction", "input", "query")
_CHOSEN_KEYS = ("chosen", "preferred", "accepted", "better", "response_chosen", "answer_chosen")
_REJECTED_KEYS = (
    "rejected",
    "dispreferred",
    "worse",
    "response_rejected",
    "answer_rejected",
)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _pick_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row:
            value = _coerce_text(row.get(key))
            if value:
                return value
    return ""


def canonicalize_preference_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = _pick_text(row, _PROMPT_KEYS)
    chosen = _pick_text(row, _CHOSEN_KEYS)
    rejected = _pick_text(row, _REJECTED_KEYS)
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def list_alignment_recipes(*, include_patch: bool = True) -> list[dict[str, Any]]:
    recipes: list[dict[str, Any]] = []
    for item in ALIGNMENT_RECIPES:
        row = dict(item)
        if not include_patch:
            row.pop("config_patch", None)
        recipes.append(row)
    return recipes


def resolve_alignment_recipe(
    recipe_id: str,
    *,
    base_config: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested = str(recipe_id or "").strip().lower()
    recipe = next(
        (item for item in ALIGNMENT_RECIPES if str(item.get("recipe_id", "")).strip().lower() == requested),
        None,
    )
    if recipe is None:
        supported = ", ".join(sorted(item["recipe_id"] for item in ALIGNMENT_RECIPES))
        raise ValueError(f"Unknown alignment recipe '{recipe_id}'. Supported: {supported}")

    resolved = dict(base_config or {})
    resolved.update(dict(recipe.get("config_patch") or {}))
    resolved.update(dict(overrides or {}))
    return {
        "recipe": dict(recipe),
        "resolved_config": resolved,
    }


def validate_preference_rows(
    rows: list[dict[str, Any]],
    *,
    min_coverage: float = 0.85,
    max_rows: int = 2000,
) -> dict[str, Any]:
    sample = [row for row in rows[: max(1, min(int(max_rows or 2000), 10000))] if isinstance(row, dict)]
    invalid_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, str]] = []

    for idx, row in enumerate(sample):
        canonical = canonicalize_preference_row(row)
        prompt = canonical.get("prompt", "")
        chosen = canonical.get("chosen", "")
        rejected = canonical.get("rejected", "")

        missing: list[str] = []
        if not prompt:
            missing.append("prompt")
        if not chosen:
            missing.append("chosen")
        if not rejected:
            missing.append("rejected")

        if missing:
            invalid_rows.append(
                {
                    "index": idx,
                    "reason": f"Missing required fields: {', '.join(missing)}.",
                    "missing_fields": missing,
                }
            )
            continue
        if chosen.strip() == rejected.strip():
            invalid_rows.append(
                {
                    "index": idx,
                    "reason": "chosen and rejected responses are identical.",
                    "missing_fields": [],
                }
            )
            continue
        valid_rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    total_rows = len(sample)
    valid_count = len(valid_rows)
    invalid_count = len(invalid_rows)
    coverage = float(valid_count / total_rows) if total_rows > 0 else 0.0
    threshold = max(0.0, min(float(min_coverage), 1.0))

    warnings: list[str] = []
    hints: list[str] = []
    errors: list[str] = []
    if total_rows == 0:
        errors.append("No readable preference rows were provided.")
    elif coverage < threshold:
        errors.append(
            (
                f"Preference pair coverage {coverage:.1%} is below required {threshold:.0%}. "
                "Ensure each row contains prompt/chosen/rejected and distinct responses."
            )
        )

    if invalid_count > 0 and total_rows > 0:
        warnings.append(f"{invalid_count}/{total_rows} rows do not satisfy preference pair contract.")
        hints.append(
            "Normalize fields to prompt/chosen/rejected before DPO/ORPO training."
        )
    if valid_count > 0:
        hints.append("Run judge scoring to filter low-quality pairs before alignment training.")

    return {
        "ok": len(errors) == 0,
        "required_fields": ["prompt", "chosen", "rejected"],
        "total_rows": total_rows,
        "valid_rows": valid_count,
        "invalid_rows": invalid_count,
        "coverage": round(coverage, 6),
        "min_coverage_required": round(threshold, 6),
        "errors": errors,
        "warnings": warnings,
        "hints": hints,
        "invalid_samples": invalid_rows[:50],
        "canonical_rows_preview": valid_rows[:50],
    }


def score_preference_rows(
    rows: list[dict[str, Any]],
    *,
    quality_threshold: float = 3.0,
    max_rows: int = 1000,
) -> dict[str, Any]:
    sample = [row for row in rows[: max(1, min(int(max_rows or 1000), 5000))] if isinstance(row, dict)]
    canonical_rows: list[dict[str, str]] = []
    for row in sample:
        canonical = canonicalize_preference_row(row)
        prompt = _coerce_text(canonical.get("prompt"))
        chosen = _coerce_text(canonical.get("chosen"))
        rejected = _coerce_text(canonical.get("rejected"))
        if not prompt or not chosen or not rejected:
            continue
        if chosen.strip() == rejected.strip():
            continue
        canonical_rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    contract = validate_preference_rows(sample, min_coverage=0.0, max_rows=len(sample))
    quality_cutoff = max(1.0, min(float(quality_threshold), 5.0))
    scored_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(canonical_rows):
        prompt = _coerce_text(row.get("prompt"))
        chosen = _coerce_text(row.get("chosen"))
        rejected = _coerce_text(row.get("rejected"))

        chosen_prompt_alignment = f1_score(chosen, prompt) if prompt else 0.0
        rejected_prompt_alignment = f1_score(rejected, prompt) if prompt else 0.0
        choice_gap = max(0.0, chosen_prompt_alignment - rejected_prompt_alignment)
        separation = 1.0 - f1_score(chosen, rejected)

        contrast_score, contrast_rationale = _heuristic_judge_score(
            reference=chosen,
            prediction=rejected,
        )
        contrast_bonus = 1.0 - ((float(contrast_score) - 1.0) / 4.0)
        composite = (
            0.50 * separation
            + 0.35 * choice_gap
            + 0.15 * max(0.0, chosen_prompt_alignment)
        )
        quality_score = round(1.0 + (4.0 * max(0.0, min(composite, 1.0))), 2)
        quality_score = round((0.8 * quality_score) + (0.2 * (1.0 + 4.0 * contrast_bonus)), 2)
        keep = quality_score >= quality_cutoff

        scored_rows.append(
            {
                "row_index": idx,
                "prompt_preview": prompt[:160],
                "quality_score": quality_score,
                "quality_threshold": quality_cutoff,
                "keep": keep,
                "separation_score": round(separation, 4),
                "chosen_prompt_alignment": round(chosen_prompt_alignment, 4),
                "rejected_prompt_alignment": round(rejected_prompt_alignment, 4),
                "contrast_score": int(contrast_score),
                "contrast_rationale": contrast_rationale[:240],
            }
        )

    score_counts = {
        "5": sum(1 for item in scored_rows if item["quality_score"] >= 4.5),
        "4": sum(1 for item in scored_rows if 3.5 <= item["quality_score"] < 4.5),
        "3": sum(1 for item in scored_rows if 2.5 <= item["quality_score"] < 3.5),
        "2": sum(1 for item in scored_rows if 1.5 <= item["quality_score"] < 2.5),
        "1": sum(1 for item in scored_rows if item["quality_score"] < 1.5),
    }
    kept = [item for item in scored_rows if item["keep"]]
    dropped = [item for item in scored_rows if not item["keep"]]
    avg = sum(float(item["quality_score"]) for item in scored_rows) / len(scored_rows) if scored_rows else 0.0

    return {
        "quality_threshold": quality_cutoff,
        "contract": contract,
        "scored_count": len(scored_rows),
        "keep_count": len(kept),
        "drop_count": len(dropped),
        "average_quality_score": round(avg, 4),
        "score_distribution": score_counts,
        "scored_rows": scored_rows,
        "kept_row_indices": [item["row_index"] for item in kept],
    }


def _load_jsonl_rows(path: Path, *, sample_size: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= sample_size:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def analyze_preference_dataset_contract(
    *,
    project_id: int,
    sample_size: int = 400,
    min_coverage: float = 0.85,
) -> dict[str, Any]:
    prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    train_file = prepared_dir / "train.jsonl"

    if not train_file.exists():
        return {
            "ok": False,
            "required_fields": ["prompt", "chosen", "rejected"],
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "coverage": 0.0,
            "min_coverage_required": min_coverage,
            "errors": [
                (
                    f"Prepared training dataset missing at {train_file}. "
                    "Run Dataset Prep split with preference pairs before DPO/ORPO training."
                )
            ],
            "warnings": [],
            "hints": [
                "Use adapter profile 'preference' and provide chosen/rejected response fields.",
            ],
            "invalid_samples": [],
            "canonical_rows_preview": [],
            "dataset_path": str(train_file),
        }

    rows = _load_jsonl_rows(train_file, sample_size=max(20, min(int(sample_size or 400), 5000)))
    report = validate_preference_rows(rows, min_coverage=min_coverage, max_rows=len(rows))
    report["dataset_path"] = str(train_file)
    return report


def analyze_preference_dataset_quality(
    *,
    project_id: int,
    sample_size: int = 400,
    quality_threshold: float = 3.0,
) -> dict[str, Any]:
    prepared_dir = settings.DATA_DIR / "projects" / str(project_id) / "prepared"
    train_file = prepared_dir / "train.jsonl"

    if not train_file.exists():
        return {
            "quality_threshold": max(1.0, min(float(quality_threshold), 5.0)),
            "contract": {
                "ok": False,
                "errors": [
                    (
                        f"Prepared training dataset missing at {train_file}. "
                        "Run Dataset Prep split with preference pairs before DPO/ORPO training."
                    )
                ],
            },
            "scored_count": 0,
            "keep_count": 0,
            "drop_count": 0,
            "average_quality_score": 0.0,
            "score_distribution": {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0},
            "scored_rows": [],
            "kept_row_indices": [],
            "dataset_path": str(train_file),
        }

    rows = _load_jsonl_rows(train_file, sample_size=max(20, min(int(sample_size or 400), 5000)))
    report = score_preference_rows(
        rows,
        quality_threshold=quality_threshold,
        max_rows=len(rows),
    )
    report["dataset_path"] = str(train_file)
    return report
