"""Newbie autopilot intent mapping for zero-knowledge training UX."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.config import settings


_INTENT_PRESETS: list[dict[str, Any]] = [
    {
        "preset_id": "autopilot.support_qa_safe",
        "label": "Support Q&A Assistant",
        "description": "Answer customer/support questions with grounded, concise replies.",
        "task_profile": "qa",
        "task_type": "causal_lm",
        "chat_template": "llama3",
        "keywords": (
            "support",
            "helpdesk",
            "ticket",
            "faq",
            "customer question",
            "answer questions",
            "qa",
        ),
    },
    {
        "preset_id": "autopilot.structured_extraction_safe",
        "label": "Structured Extraction",
        "description": "Extract key fields from contracts/forms/invoices into consistent outputs.",
        "task_profile": "structured_extraction",
        "task_type": "causal_lm",
        "chat_template": "chatml",
        "keywords": (
            "extract",
            "extraction",
            "invoice",
            "contract",
            "form",
            "fields",
            "json",
            "structured",
        ),
    },
    {
        "preset_id": "autopilot.summarization_safe",
        "label": "Summarization Assistant",
        "description": "Summarize long content into concise, useful outputs.",
        "task_profile": "summarization",
        "task_type": "causal_lm",
        "chat_template": "llama3",
        "keywords": (
            "summarize",
            "summary",
            "tldr",
            "brief",
            "digest",
            "notes",
            "recap",
        ),
    },
    {
        "preset_id": "autopilot.classification_safe",
        "label": "Text Classifier",
        "description": "Classify text into labels with conservative defaults.",
        "task_profile": "classification",
        "task_type": "classification",
        "chat_template": "chatml",
        "keywords": (
            "classify",
            "classification",
            "categorize",
            "category",
            "sentiment",
            "label",
            "routing",
        ),
    },
]

_DEFAULT_PRESET: dict[str, Any] = {
    "preset_id": "autopilot.general_chat_safe",
    "label": "General Assistant",
    "description": "General-purpose assistant behavior with safe starter defaults.",
    "task_profile": "instruction_sft",
    "task_type": "causal_lm",
    "chat_template": "llama3",
    "keywords": (),
}


def _normalize_intent(intent: str) -> str:
    token = str(intent or "").strip().lower()
    token = re.sub(r"\s+", " ", token)
    return token


def _normalize_target_device(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"mobile", "laptop", "server"}:
        return token
    return "laptop"


def _safe_defaults_for_device(target_device: str) -> dict[str, Any]:
    device = _normalize_target_device(target_device)
    if device == "mobile":
        return {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 1024,
            "num_epochs": 2,
        }
    if device == "server":
        return {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "num_epochs": 3,
        }
    return {
        "batch_size": 2,
        "gradient_accumulation_steps": 6,
        "max_seq_length": 1536,
        "num_epochs": 2,
    }


def _score_preset(intent: str, preset: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    matched: list[str] = []
    for raw_keyword in list(preset.get("keywords") or []):
        keyword = str(raw_keyword or "").strip().lower()
        if not keyword:
            continue
        if keyword in intent:
            score += 1
            matched.append(keyword)
    return score, matched


def _confidence_band(confidence: float) -> str:
    value = max(0.0, min(float(confidence), 1.0))
    if value >= 0.8:
        return "high"
    if value >= 0.6:
        return "medium"
    return "low"


def _intent_rewrite_suggestions_for_task_profile(
    *,
    task_profile: str | None,
) -> list[dict[str, Any]]:
    profile = str(task_profile or "").strip().lower() or "instruction_sft"
    if profile == "structured_extraction":
        return [
            {
                "id": "rewrite.structured_extraction.json",
                "label": "Extract key fields as JSON",
                "rewritten_intent": (
                    "Extract invoice number, vendor name, due date, and total amount from each "
                    "document and return strict JSON only."
                ),
                "reason": "Defines explicit fields and output format.",
                "recommended": True,
            },
            {
                "id": "rewrite.structured_extraction.qa",
                "label": "Contract clause extraction",
                "rewritten_intent": (
                    "Extract liability and payment clauses from contract text and return a concise "
                    "JSON object with clause type and evidence sentence."
                ),
                "reason": "Clarifies extraction target and expected structure.",
                "recommended": False,
            },
        ]
    if profile == "summarization":
        return [
            {
                "id": "rewrite.summarization.support",
                "label": "Ticket summary with actions",
                "rewritten_intent": (
                    "Summarize each support ticket into 3 bullet points plus next actions and urgency level."
                ),
                "reason": "Specifies summary format and decision-support fields.",
                "recommended": True,
            },
            {
                "id": "rewrite.summarization.exec",
                "label": "Executive brief summary",
                "rewritten_intent": (
                    "Generate a short executive summary of each thread with key risks and recommended actions."
                ),
                "reason": "Clarifies audience and desired output style.",
                "recommended": False,
            },
        ]
    if profile == "classification":
        return [
            {
                "id": "rewrite.classification.routing",
                "label": "Message routing labels",
                "rewritten_intent": (
                    "Classify each incoming message into exactly one label: billing, technical, or cancellation."
                ),
                "reason": "Defines fixed labels and single-label behavior.",
                "recommended": True,
            },
            {
                "id": "rewrite.classification.sentiment",
                "label": "Customer sentiment",
                "rewritten_intent": (
                    "Classify customer sentiment as positive, neutral, or negative and return only the label."
                ),
                "reason": "Specifies label set and concise output.",
                "recommended": False,
            },
        ]
    if profile == "qa":
        return [
            {
                "id": "rewrite.qa.support_answer",
                "label": "Support Q&A answer drafting",
                "rewritten_intent": (
                    "Answer customer support questions in concise plain language with one recommended next step."
                ),
                "reason": "Clarifies response objective and style.",
                "recommended": True,
            },
            {
                "id": "rewrite.qa.faq",
                "label": "FAQ-style response",
                "rewritten_intent": (
                    "Read a support question and draft a short FAQ-style answer with clear troubleshooting steps."
                ),
                "reason": "Specifies answer format for consistency.",
                "recommended": False,
            },
        ]
    return [
        {
            "id": "rewrite.general.task_output",
            "label": "Define output behavior clearly",
            "rewritten_intent": (
                "Given an input text, produce a concise and reliable output in a consistent format with examples."
            ),
            "reason": "Makes the task and output constraints explicit.",
            "recommended": True,
        },
        {
            "id": "rewrite.general.helpdesk",
            "label": "General assistant for support text",
            "rewritten_intent": (
                "Read user requests and generate safe, helpful responses that are short, accurate, and actionable."
            ),
            "reason": "Improves clarity around desired assistant behavior.",
            "recommended": False,
        },
    ]


def build_newbie_autopilot_intent_clarification(
    *,
    intent: str,
    confidence: float,
    task_profile: str | None = None,
    matched_keywords: list[str] | None = None,
) -> dict[str, Any]:
    normalized_intent = _normalize_intent(intent)
    matched = [str(item).strip() for item in list(matched_keywords or []) if str(item).strip()]
    word_count = len([item for item in normalized_intent.split(" ") if item])
    band = _confidence_band(confidence)

    requires_clarification = False
    reasons: list[str] = []
    if band == "low":
        requires_clarification = True
        reasons.append("intent confidence is low")
    if word_count < 6:
        requires_clarification = True
        reasons.append("intent text is very short")
    if not matched:
        requires_clarification = True
        reasons.append("no clear task keywords were detected")

    questions = [
        "What exact output should the model produce?",
        "What are 2-3 example inputs and expected answers?",
        "Should responses be concise summaries, labels, or structured JSON?",
    ]
    suggested_intent_examples = [
        "Summarize support tickets into 3 bullet points and next actions.",
        "Extract invoice number, due date, vendor, and total into JSON.",
        "Classify incoming messages into billing, technical, or cancellation.",
    ]
    rewrite_suggestions = _intent_rewrite_suggestions_for_task_profile(task_profile=task_profile)
    reason = "; ".join(reasons) if reasons else ""
    return {
        "required": requires_clarification,
        "confidence_band": band,
        "reason": reason or None,
        "matched_keywords": matched,
        "questions": questions if requires_clarification else [],
        "suggested_intent_examples": suggested_intent_examples if requires_clarification else [],
        "rewrite_suggestions": rewrite_suggestions if requires_clarification else [],
    }


def _project_prepared_train_path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "prepared" / "train.jsonl"


def evaluate_newbie_autopilot_dataset_readiness(
    *,
    project_id: int,
    min_rows: int = 20,
) -> dict[str, Any]:
    safe_min_rows = max(1, int(min_rows))
    train_path = _project_prepared_train_path(project_id)
    blockers: list[str] = []
    warnings: list[str] = []
    hints: list[str] = []
    auto_fixes: list[dict[str, str]] = []
    row_count = 0
    valid_json_rows = 0
    invalid_json_rows = 0

    if not train_path.exists():
        blockers.append("Prepared training split is missing (`prepared/train.jsonl`).")
        hints.append("Go to Data pipeline tab and run dataset split/prep before training.")
        auto_fixes.append(
            {
                "id": "open_data_pipeline",
                "label": "Open Data Prep",
                "description": "Open data prep tab and create prepared train/val splits.",
                "navigate_to": f"/project/{project_id}/pipeline/data",
            }
        )
    else:
        try:
            with open(train_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    token = str(line or "").strip()
                    if not token:
                        continue
                    row_count += 1
                    if token.startswith("{") and token.endswith("}"):
                        valid_json_rows += 1
                    else:
                        invalid_json_rows += 1
        except Exception:
            blockers.append("Prepared train split could not be read.")
            hints.append("Check file permissions and regenerate dataset split.")

    if row_count == 0 and not blockers:
        blockers.append("Prepared train split has zero usable rows.")
    elif 0 < row_count < safe_min_rows:
        warnings.append(
            f"Prepared train split is very small ({row_count} rows). Target at least {safe_min_rows} rows."
        )
        hints.append("Add more examples or generate synthetic data before one-click run.")
        auto_fixes.append(
            {
                "id": "open_synthetic_data",
                "label": "Generate Synthetic Data",
                "description": "Open synthetic data tools to expand training examples.",
                "navigate_to": f"/project/{project_id}/pipeline/synthetic",
            }
        )

    if invalid_json_rows > 0:
        warnings.append(
            f"Detected {invalid_json_rows} non-JSONL rows in prepared train split; parsing quality may be reduced."
        )

    ready = len(blockers) == 0
    return {
        "ready": ready,
        "prepared_train_exists": train_path.exists(),
        "prepared_train_path": str(train_path),
        "prepared_row_count": row_count,
        "valid_json_row_count": valid_json_rows,
        "invalid_json_row_count": invalid_json_rows,
        "min_recommended_rows": safe_min_rows,
        "blockers": blockers,
        "warnings": warnings,
        "hints": hints,
        "auto_fixes": auto_fixes,
    }


def build_newbie_autopilot_run_name(
    *,
    intent: str,
    preset_label: str,
) -> str:
    intent_short = re.sub(r"[^a-zA-Z0-9 ]+", " ", str(intent or "")).strip()
    intent_short = re.sub(r"\s+", " ", intent_short)
    if len(intent_short) > 40:
        intent_short = intent_short[:40].rstrip()
    base = str(preset_label or "autopilot").strip()
    if intent_short:
        return f"Autopilot - {base} - {intent_short}"
    return f"Autopilot - {base}"


def resolve_newbie_autopilot_intent(
    *,
    intent: str,
    target_device: str = "laptop",
    primary_language: str = "english",
    available_vram_gb: float | None = None,
) -> dict[str, Any]:
    normalized_intent = _normalize_intent(intent)
    if not normalized_intent:
        raise ValueError("intent is required")

    selected = dict(_DEFAULT_PRESET)
    selected_score = 0
    selected_keywords: list[str] = []
    for preset in _INTENT_PRESETS:
        score, matched = _score_preset(normalized_intent, preset)
        if score > selected_score:
            selected = dict(preset)
            selected_score = score
            selected_keywords = matched

    normalized_device = _normalize_target_device(target_device)
    device_defaults = _safe_defaults_for_device(normalized_device)
    safe_training_config = {
        "base_model": "microsoft/phi-2",
        "training_mode": "sft",
        "training_runtime_id": "auto",
        "trainer_backend": "auto",
        "task_type": str(selected.get("task_type") or "causal_lm"),
        "chat_template": str(selected.get("chat_template") or "llama3"),
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "save_steps": 100,
        "eval_steps": 100,
        "sequence_packing": True,
        "gradient_checkpointing": True,
        "auto_oom_retry": True,
        "max_oom_retries": 2,
        "oom_retry_seq_shrink": 0.75,
        **device_defaults,
    }

    return {
        "intent": normalized_intent,
        "target_device": normalized_device,
        "primary_language": str(primary_language or "english").strip().lower() or "english",
        "available_vram_gb": available_vram_gb,
        "preset_id": str(selected.get("preset_id") or _DEFAULT_PRESET["preset_id"]),
        "preset_label": str(selected.get("label") or _DEFAULT_PRESET["label"]),
        "preset_description": str(selected.get("description") or _DEFAULT_PRESET["description"]),
        "task_profile": str(selected.get("task_profile") or "instruction_sft"),
        "matched_keywords": selected_keywords,
        "confidence": min(1.0, 0.35 + (0.15 * selected_score)),
        "safe_training_config": safe_training_config,
        "run_name_suggestion": build_newbie_autopilot_run_name(
            intent=normalized_intent,
            preset_label=str(selected.get("label") or _DEFAULT_PRESET["label"]),
        ),
        "user_friendly_plan": [
            "We auto-picked a safe starter recipe for your goal.",
            "We keep memory settings conservative to reduce training failures.",
            "You can launch now with one click and refine later in Advanced Mode.",
        ],
    }
