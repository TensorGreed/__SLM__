"""Data adapter SDK registry and execution helpers."""

from __future__ import annotations

import copy
import importlib
import inspect
import threading
from typing import Any, Callable

from app.config import settings
from app.services.record_normalization import canonicalize_record

AdapterDetectFn = Callable[[dict[str, Any], dict[str, Any]], bool | float]
AdapterMapFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None]
AdapterValidateFn = Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any]]
AdapterSchemaHintFn = Callable[[], dict[str, Any]]

DEFAULT_ADAPTER_ID = "default-canonical"
AUTO_ADAPTER_ID = "auto"

ADAPTER_CONTRACT_VERSION = "slm.data_adapter/v2"
AUTO_TASK_PROFILE = "auto"
DEFAULT_TASK_PROFILE = "instruction_sft"

SUPPORTED_TASK_PROFILES: tuple[str, ...] = (
    "instruction_sft",
    "chat_sft",
    "qa",
    "seq2seq",
    "classification",
    "preference",
    "language_modeling",
)

_TASK_PROFILE_ALIASES: dict[str, str] = {
    "causal_lm": "instruction_sft",
    "causallm": "instruction_sft",
    "sft": "instruction_sft",
    "instruction": "instruction_sft",
    "chat": "chat_sft",
    "chat_messages": "chat_sft",
    "preference_pair": "preference",
    "ranking": "preference",
    "lm": "language_modeling",
    "language_modeling": "language_modeling",
    "language_model": "language_modeling",
}

_TRAINING_TASK_ALIASES: dict[str, str] = {
    "causal-lm": "causal_lm",
    "causal_lm": "causal_lm",
    "causallm": "causal_lm",
    "instruction_sft": "causal_lm",
    "chat_sft": "causal_lm",
    "sft": "causal_lm",
    "qa": "causal_lm",
    "language_modeling": "causal_lm",
    "seq2seq": "seq2seq",
    "classification": "classification",
    "dpo": "dpo",
    "orpo": "orpo",
    "preference": "dpo",
}

_TASK_PROFILE_TRAINING_TASKS: dict[str, tuple[str, ...]] = {
    "instruction_sft": ("causal_lm",),
    "chat_sft": ("causal_lm",),
    "qa": ("causal_lm", "seq2seq"),
    "seq2seq": ("seq2seq",),
    "classification": ("classification",),
    "preference": ("dpo", "orpo"),
    "language_modeling": ("causal_lm",),
}


def _normalize_adapter_id(value: str | None) -> str:
    token = str(value or "").strip().lower()
    return token.replace("_", "-").replace(" ", "-")


def normalize_task_profile(value: str | None, *, default: str = AUTO_TASK_PROFILE) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return default
    normalized = _TASK_PROFILE_ALIASES.get(token, token)
    if normalized in SUPPORTED_TASK_PROFILES:
        return normalized
    if normalized == AUTO_TASK_PROFILE:
        return AUTO_TASK_PROFILE
    return normalized


def normalize_training_task_type(value: str | None, *, default: str = "causal_lm") -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return default
    return _TRAINING_TASK_ALIASES.get(token, token)


def task_profile_training_tasks(task_profile: str | None) -> list[str]:
    normalized = normalize_task_profile(task_profile, default=DEFAULT_TASK_PROFILE)
    return list(_TASK_PROFILE_TRAINING_TASKS.get(normalized, ("causal_lm",)))


def is_training_task_compatible(task_profile: str | None, training_task_type: str | None) -> bool:
    profile = normalize_task_profile(task_profile, default=DEFAULT_TASK_PROFILE)
    task = normalize_training_task_type(training_task_type, default="causal_lm")
    compatible = set(task_profile_training_tasks(profile))
    return task in compatible


def _coerce_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    token = str(value).strip()
    return token


def _pick_text(record: dict[str, Any], aliases: list[str]) -> str:
    for key in aliases:
        if key in record:
            token = _coerce_text(record.get(key))
            if token:
                return token
    return ""


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if parsed < 0:
        return 0.0
    if parsed > 1:
        return 1.0
    return parsed


def _normalize_task_profiles(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        items = [normalize_task_profile(str(item), default="") for item in list(value)]
    elif isinstance(value, str):
        items = [normalize_task_profile(value, default="")]
    else:
        items = []
    out: list[str] = []
    for item in items:
        if not item or item == AUTO_TASK_PROFILE:
            continue
        if item not in out:
            out.append(item)
    if not out:
        out = [DEFAULT_TASK_PROFILE]
    return out


def _normalize_training_tasks(value: Any, *, fallback_profile: str) -> list[str]:
    tasks: list[str] = []
    if isinstance(value, (list, tuple, set)):
        candidates = [normalize_training_task_type(str(item), default="") for item in list(value)]
    elif isinstance(value, str):
        candidates = [normalize_training_task_type(value, default="")]
    else:
        candidates = []
    for item in candidates:
        if not item:
            continue
        if item not in tasks:
            tasks.append(item)
    if tasks:
        return tasks
    return task_profile_training_tasks(fallback_profile)


def _output_contract_from_schema_hint(schema_hint: dict[str, Any]) -> dict[str, Any]:
    output = schema_hint.get("output_shape")
    required_fields: list[str] = []
    optional_fields: list[str] = []
    if isinstance(output, dict):
        for key, raw_state in output.items():
            field = str(key).strip()
            state = str(raw_state or "").strip().lower()
            if not field:
                continue
            if state == "required":
                required_fields.append(field)
            else:
                optional_fields.append(field)
    if not required_fields and not optional_fields:
        required_fields = ["text"]
    if "text" not in required_fields and "text" not in optional_fields:
        optional_fields.append("text")
    return {
        "required_fields": required_fields,
        "optional_fields": optional_fields,
    }


def _normalize_output_contract(value: Any, *, schema_hint: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    base = _output_contract_from_schema_hint(schema_hint)
    required_fields = [str(item).strip() for item in list(value.get("required_fields") or []) if str(item).strip()]
    optional_fields = [str(item).strip() for item in list(value.get("optional_fields") or []) if str(item).strip()]
    if not required_fields:
        required_fields = list(base.get("required_fields") or ["text"])
    if not optional_fields:
        optional_fields = list(base.get("optional_fields") or [])
    optional_fields = [item for item in optional_fields if item not in required_fields]
    notes = [str(item).strip() for item in list(value.get("notes") or []) if str(item).strip()]
    return {
        "required_fields": required_fields,
        "optional_fields": optional_fields,
        "notes": notes,
    }


def _base_adapter_config(
    config: dict[str, Any] | None,
    field_mapping: dict[str, str] | None,
    task_profile: str | None = None,
) -> dict[str, Any]:
    payload = dict(config or {})
    if field_mapping and "field_mapping" not in payload:
        payload["field_mapping"] = dict(field_mapping)
    if task_profile not in (None, ""):
        normalized = normalize_task_profile(task_profile, default=AUTO_TASK_PROFILE)
        if normalized and normalized != AUTO_TASK_PROFILE:
            payload["task_profile"] = normalized
    return payload


def _messages_from_record(record: dict[str, Any]) -> list[dict[str, str]]:
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = record.get("conversations")
    if not isinstance(raw_messages, list):
        return []

    rows: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = _coerce_text(item.get("role") or item.get("from")).lower()
        content = _coerce_text(item.get("content") or item.get("value") or item.get("text"))
        if not role or not content:
            continue
        if role in {"human"}:
            role = "user"
        if role in {"gpt", "model"}:
            role = "assistant"
        rows.append({"role": role, "content": content})
    return rows


def _messages_as_text(record: dict[str, Any]) -> str:
    messages = _messages_from_record(record)
    if not messages:
        return ""
    has_user = any(item.get("role") == "user" for item in messages)
    has_assistant = any(item.get("role") == "assistant" for item in messages)
    if not (has_user and has_assistant):
        return ""

    chunks: list[str] = []
    for item in messages:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role:
            chunks.append(f"{role}: {content}")
        else:
            chunks.append(content)
    return "\n".join(chunks).strip()


def _validate_default(mapped_records: list[dict[str, Any]], _config: dict[str, Any]) -> dict[str, Any]:
    total = len(mapped_records)
    if total == 0:
        return {
            "status": "warning",
            "mapped_records": 0,
            "qa_pair_ratio": 0.0,
            "avg_text_chars": 0.0,
        }
    qa_pairs = 0
    total_chars = 0
    for row in mapped_records:
        question = _coerce_text(row.get("question"))
        answer = _coerce_text(row.get("answer"))
        text = _coerce_text(row.get("text"))
        if question and answer:
            qa_pairs += 1
        total_chars += len(text)
    return {
        "status": "ok",
        "mapped_records": total,
        "qa_pair_ratio": round(qa_pairs / total, 6),
        "avg_text_chars": round(total_chars / total, 2),
    }


def _schema_default() -> dict[str, Any]:
    return {
        "input_candidates": {
            "text": ["text", "content", "body", "document", "passage", "article"],
            "question": ["question", "prompt", "query", "instruction", "input"],
            "answer": ["answer", "response", "output", "target", "completion", "label"],
        },
        "output_shape": {
            "text": "required",
            "question": "optional",
            "answer": "optional",
            "source_text": "optional",
            "target_text": "optional",
        },
    }


def _detect_default(record: dict[str, Any], config: dict[str, Any]) -> bool | float:
    mapped = canonicalize_record(record, field_mapping=config.get("field_mapping"))
    return 1.0 if mapped else 0.0


def _map_default(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    canonical = canonicalize_record(record, field_mapping=config.get("field_mapping"))
    if not canonical:
        return None
    return canonical


def _schema_qa_pair() -> dict[str, Any]:
    return {
        "input_candidates": {
            "question": ["question", "prompt", "query", "instruction", "input"],
            "answer": ["answer", "response", "output", "target", "completion"],
        },
        "output_shape": {
            "text": "required",
            "question": "required",
            "answer": "required",
            "source_text": "required",
            "target_text": "required",
        },
    }


def _detect_qa_pair(record: dict[str, Any], config: dict[str, Any]) -> bool | float:
    mapped = _map_qa_pair(record, config)
    return 1.0 if mapped else 0.0


def _map_qa_pair(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    canonical = canonicalize_record(record, field_mapping=config.get("field_mapping"))
    question = _coerce_text((canonical or {}).get("question"))
    answer = _coerce_text((canonical or {}).get("answer"))

    if not question:
        question = _pick_text(record, ["question", "prompt", "query", "instruction", "input"])
    if not answer:
        answer = _pick_text(record, ["answer", "response", "output", "target", "completion"])
    if not question or not answer:
        return None

    text = f"Question: {question}\nAnswer: {answer}"
    return {
        "text": text,
        "question": question,
        "answer": answer,
        "source_text": question,
        "target_text": answer,
    }


def _validate_qa_pair(mapped_records: list[dict[str, Any]], _config: dict[str, Any]) -> dict[str, Any]:
    total = len(mapped_records)
    if total == 0:
        return {"status": "warning", "mapped_records": 0, "qa_pair_ratio": 0.0}
    qa_pairs = 0
    for row in mapped_records:
        if _coerce_text(row.get("question")) and _coerce_text(row.get("answer")):
            qa_pairs += 1
    ratio = qa_pairs / total
    return {
        "status": "ok" if ratio >= 0.9 else "warning",
        "mapped_records": total,
        "qa_pair_ratio": round(ratio, 6),
    }


def _schema_seq2seq() -> dict[str, Any]:
    return {
        "input_candidates": {
            "source": ["source", "input", "question", "prompt", "instruction", "text", "content"],
            "target": ["target", "answer", "output", "completion", "response"],
        },
        "output_shape": {
            "text": "required",
            "source_text": "required",
            "target_text": "required",
            "question": "optional",
            "answer": "optional",
        },
    }


def _detect_seq2seq(record: dict[str, Any], config: dict[str, Any]) -> bool | float:
    mapped = _map_seq2seq(record, config)
    return 1.0 if mapped else 0.0


def _map_seq2seq(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    source_fields = config.get("source_fields")
    target_fields = config.get("target_fields")
    source_aliases = list(source_fields) if isinstance(source_fields, list) and source_fields else [
        "source",
        "input",
        "question",
        "prompt",
        "instruction",
        "text",
        "content",
    ]
    target_aliases = list(target_fields) if isinstance(target_fields, list) and target_fields else [
        "target",
        "answer",
        "output",
        "completion",
        "response",
    ]

    source = _pick_text(record, source_aliases)
    target = _pick_text(record, target_aliases)
    if not source or not target:
        canonical = canonicalize_record(record, field_mapping=config.get("field_mapping"))
        source = source or _coerce_text((canonical or {}).get("question")) or _coerce_text((canonical or {}).get("text"))
        target = target or _coerce_text((canonical or {}).get("answer"))
    if not source or not target:
        return None

    return {
        "text": f"Input: {source}\nOutput: {target}",
        "source_text": source,
        "target_text": target,
        "question": source,
        "answer": target,
    }


def _validate_seq2seq(mapped_records: list[dict[str, Any]], _config: dict[str, Any]) -> dict[str, Any]:
    total = len(mapped_records)
    if total == 0:
        return {"status": "warning", "mapped_records": 0, "paired_ratio": 0.0}
    paired = 0
    for row in mapped_records:
        if _coerce_text(row.get("source_text")) and _coerce_text(row.get("target_text")):
            paired += 1
    ratio = paired / total
    return {
        "status": "ok" if ratio >= 0.95 else "warning",
        "mapped_records": total,
        "paired_ratio": round(ratio, 6),
    }


def _schema_classification() -> dict[str, Any]:
    return {
        "input_candidates": {
            "text": ["text", "content", "input", "question", "prompt", "instruction"],
            "label": ["label", "class", "category", "output_label", "target", "answer", "output"],
        },
        "output_shape": {
            "text": "required",
            "source_text": "required",
            "target_text": "required",
            "label": "required",
        },
    }


def _detect_classification(record: dict[str, Any], config: dict[str, Any]) -> bool | float:
    mapped = _map_classification(record, config)
    if not mapped:
        return 0.0
    has_label = bool(_coerce_text(mapped.get("label")))
    return 1.0 if has_label else 0.0


def _map_classification(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    text_fields = config.get("text_fields")
    label_fields = config.get("label_fields")
    text_aliases = list(text_fields) if isinstance(text_fields, list) and text_fields else [
        "text",
        "content",
        "input",
        "question",
        "prompt",
        "instruction",
    ]
    label_aliases = list(label_fields) if isinstance(label_fields, list) and label_fields else [
        "label",
        "class",
        "category",
        "output_label",
        "target",
        "answer",
        "output",
    ]

    text = _pick_text(record, text_aliases)
    label = _pick_text(record, label_aliases)
    if not text:
        canonical = canonicalize_record(record, field_mapping=config.get("field_mapping"))
        text = text or _coerce_text((canonical or {}).get("text")) or _coerce_text((canonical or {}).get("question"))
        label = label or _coerce_text((canonical or {}).get("answer"))
    if not text or not label:
        return None

    return {
        "text": text,
        "source_text": text,
        "target_text": label,
        "label": label,
        "answer": label,
    }


def _validate_classification(mapped_records: list[dict[str, Any]], _config: dict[str, Any]) -> dict[str, Any]:
    label_distribution: dict[str, int] = {}
    for row in mapped_records:
        label = _coerce_text(row.get("label") or row.get("target_text"))
        if not label:
            continue
        label_distribution[label] = label_distribution.get(label, 0) + 1
    total = len(mapped_records)
    unique_labels = len(label_distribution)
    top_labels = sorted(label_distribution.items(), key=lambda item: item[1], reverse=True)[:20]
    return {
        "status": "ok" if total > 0 and unique_labels > 1 else "warning",
        "mapped_records": total,
        "unique_labels": unique_labels,
        "label_distribution": {label: count for label, count in top_labels},
    }


def _schema_chat_messages() -> dict[str, Any]:
    return {
        "input_candidates": {
            "messages": ["messages", "conversations"],
        },
        "output_shape": {
            "text": "required",
            "messages": "required",
            "question": "optional",
            "answer": "optional",
            "source_text": "optional",
            "target_text": "optional",
        },
    }


def _detect_chat_messages(record: dict[str, Any], config: dict[str, Any]) -> bool | float:
    mapped = _map_chat_messages(record, config)
    return 1.0 if mapped else 0.0


def _map_chat_messages(record: dict[str, Any], _config: dict[str, Any]) -> dict[str, Any] | None:
    messages = _messages_from_record(record)
    if not messages:
        return None

    has_user = any(item.get("role") == "user" for item in messages)
    has_assistant = any(item.get("role") == "assistant" for item in messages)
    if not (has_user and has_assistant):
        return None

    rendered = _messages_as_text(record)
    if not rendered:
        return None

    question = ""
    answer = ""
    for item in messages:
        role = item.get("role")
        content = _coerce_text(item.get("content"))
        if role == "user" and content:
            question = content
        if role == "assistant" and content:
            answer = content

    return {
        "text": rendered,
        "messages": messages,
        "question": question,
        "answer": answer,
        "source_text": question,
        "target_text": answer,
    }


def _validate_chat_messages(mapped_records: list[dict[str, Any]], _config: dict[str, Any]) -> dict[str, Any]:
    total = len(mapped_records)
    if total == 0:
        return {"status": "warning", "mapped_records": 0, "chat_ratio": 0.0}
    complete = 0
    for row in mapped_records:
        messages = row.get("messages")
        if isinstance(messages, list) and len(messages) >= 2:
            complete += 1
    ratio = complete / total
    return {
        "status": "ok" if ratio >= 0.9 else "warning",
        "mapped_records": total,
        "chat_ratio": round(ratio, 6),
    }


def _schema_preference_pair() -> dict[str, Any]:
    return {
        "input_candidates": {
            "prompt": ["prompt", "question", "instruction", "input"],
            "chosen": ["chosen", "preferred", "answer", "completion", "response"],
            "rejected": ["rejected", "dispreferred", "rejected_response", "negative"],
        },
        "output_shape": {
            "text": "required",
            "prompt": "required",
            "chosen": "required",
            "rejected": "required",
            "source_text": "required",
            "target_text": "required",
        },
    }


def _detect_preference_pair(record: dict[str, Any], config: dict[str, Any]) -> bool | float:
    mapped = _map_preference_pair(record, config)
    return 1.0 if mapped else 0.0


def _map_preference_pair(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    prompt_fields = config.get("prompt_fields")
    chosen_fields = config.get("chosen_fields")
    rejected_fields = config.get("rejected_fields")

    prompt_aliases = list(prompt_fields) if isinstance(prompt_fields, list) and prompt_fields else [
        "prompt",
        "question",
        "instruction",
        "input",
    ]
    chosen_aliases = list(chosen_fields) if isinstance(chosen_fields, list) and chosen_fields else [
        "chosen",
        "preferred",
        "answer",
        "completion",
        "response",
    ]
    rejected_aliases = list(rejected_fields) if isinstance(rejected_fields, list) and rejected_fields else [
        "rejected",
        "dispreferred",
        "rejected_response",
        "negative",
    ]

    prompt = _pick_text(record, prompt_aliases)
    chosen = _pick_text(record, chosen_aliases)
    rejected = _pick_text(record, rejected_aliases)
    if not prompt or not chosen or not rejected:
        canonical = canonicalize_record(record, field_mapping=config.get("field_mapping"))
        prompt = prompt or _coerce_text((canonical or {}).get("question"))
        chosen = chosen or _coerce_text((canonical or {}).get("answer"))
    if not prompt or not chosen or not rejected:
        return None

    return {
        "text": f"Prompt: {prompt}\nChosen: {chosen}\nRejected: {rejected}",
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source_text": prompt,
        "target_text": chosen,
        "question": prompt,
        "answer": chosen,
    }


def _validate_preference_pair(mapped_records: list[dict[str, Any]], _config: dict[str, Any]) -> dict[str, Any]:
    total = len(mapped_records)
    if total == 0:
        return {"status": "warning", "mapped_records": 0, "pair_ratio": 0.0}
    complete = 0
    for row in mapped_records:
        if _coerce_text(row.get("prompt")) and _coerce_text(row.get("chosen")) and _coerce_text(row.get("rejected")):
            complete += 1
    ratio = complete / total
    return {
        "status": "ok" if ratio >= 0.95 else "warning",
        "mapped_records": total,
        "pair_ratio": round(ratio, 6),
    }


BUILTIN_ADAPTERS: dict[str, dict[str, Any]] = {
    DEFAULT_ADAPTER_ID: {
        "description": "General canonical adapter (text/question/answer inference with best-effort fallback).",
        "detect": _detect_default,
        "map_row": _map_default,
        "validate": _validate_default,
        "schema_hint": _schema_default,
        "task_profiles": ["instruction_sft", "qa", "language_modeling"],
        "preferred_training_tasks": ["causal_lm"],
        "output_contract": {
            "required_fields": ["text"],
            "optional_fields": ["question", "answer", "source_text", "target_text"],
            "notes": ["best-effort canonical mapping"],
        },
    },
    "qa-pair": {
        "description": "Strict QA adapter requiring both question and answer fields.",
        "detect": _detect_qa_pair,
        "map_row": _map_qa_pair,
        "validate": _validate_qa_pair,
        "schema_hint": _schema_qa_pair,
        "task_profiles": ["qa", "instruction_sft", "seq2seq"],
        "preferred_training_tasks": ["causal_lm", "seq2seq"],
        "output_contract": {
            "required_fields": ["text", "question", "answer", "source_text", "target_text"],
            "optional_fields": [],
            "notes": ["strict pair extraction"],
        },
    },
    "seq2seq-pair": {
        "description": "Seq2Seq adapter mapping source/target style rows.",
        "detect": _detect_seq2seq,
        "map_row": _map_seq2seq,
        "validate": _validate_seq2seq,
        "schema_hint": _schema_seq2seq,
        "task_profiles": ["seq2seq", "instruction_sft"],
        "preferred_training_tasks": ["seq2seq"],
        "output_contract": {
            "required_fields": ["text", "source_text", "target_text"],
            "optional_fields": ["question", "answer"],
            "notes": [],
        },
    },
    "classification-label": {
        "description": "Classification adapter mapping text + class/label fields.",
        "detect": _detect_classification,
        "map_row": _map_classification,
        "validate": _validate_classification,
        "schema_hint": _schema_classification,
        "task_profiles": ["classification"],
        "preferred_training_tasks": ["classification"],
        "output_contract": {
            "required_fields": ["text", "source_text", "target_text", "label"],
            "optional_fields": ["answer"],
            "notes": [],
        },
    },
    "chat-messages": {
        "description": "Chat transcript adapter for messages/conversations arrays.",
        "detect": _detect_chat_messages,
        "map_row": _map_chat_messages,
        "validate": _validate_chat_messages,
        "schema_hint": _schema_chat_messages,
        "task_profiles": ["chat_sft", "instruction_sft"],
        "preferred_training_tasks": ["causal_lm"],
        "output_contract": {
            "required_fields": ["text", "messages"],
            "optional_fields": ["question", "answer", "source_text", "target_text"],
            "notes": ["requires both user and assistant turns"],
        },
    },
    "preference-pair": {
        "description": "Preference-pair adapter for DPO/ORPO style chosen/rejected rows.",
        "detect": _detect_preference_pair,
        "map_row": _map_preference_pair,
        "validate": _validate_preference_pair,
        "schema_hint": _schema_preference_pair,
        "task_profiles": ["preference", "qa"],
        "preferred_training_tasks": ["dpo", "orpo"],
        "output_contract": {
            "required_fields": ["text", "prompt", "chosen", "rejected", "source_text", "target_text"],
            "optional_fields": ["question", "answer"],
            "notes": ["preference rows are not directly trainable by current SFT runtime"],
        },
    },
}

_PLUGIN_ADAPTERS: dict[str, dict[str, Any]] = {}
_PLUGIN_ADAPTER_SOURCE: dict[str, str] = {}
_LOADED_PLUGIN_MODULES: set[str] = set()
_PLUGIN_LOAD_ERRORS: dict[str, str] = {}
_REGISTRY_LOCK = threading.Lock()


def _adapter_registry() -> dict[str, dict[str, Any]]:
    merged = dict(BUILTIN_ADAPTERS)
    merged.update(_PLUGIN_ADAPTERS)
    return merged


def _adapter_contract(adapter_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    schema_hint_fn = payload.get("schema_hint")
    schema_hint: dict[str, Any] = {}
    if callable(schema_hint_fn):
        try:
            candidate = schema_hint_fn()
            if isinstance(candidate, dict):
                schema_hint = candidate
        except Exception:
            schema_hint = {}

    task_profiles = _normalize_task_profiles(payload.get("task_profiles"))
    preferred_training = _normalize_training_tasks(
        payload.get("preferred_training_tasks"),
        fallback_profile=task_profiles[0] if task_profiles else DEFAULT_TASK_PROFILE,
    )
    output_contract = _normalize_output_contract(
        payload.get("output_contract"),
        schema_hint=schema_hint,
    )

    return {
        "version": ADAPTER_CONTRACT_VERSION,
        "adapter_id": adapter_id,
        "task_profiles": task_profiles,
        "preferred_training_tasks": preferred_training,
        "output_contract": output_contract,
        "schema_hint": schema_hint,
    }


def resolve_data_adapter_contract(adapter_id: str | None) -> dict[str, Any]:
    normalized_id = _normalize_adapter_id(adapter_id) or DEFAULT_ADAPTER_ID
    registry = _adapter_registry()
    payload = registry.get(normalized_id) or registry.get(DEFAULT_ADAPTER_ID)
    if not payload:
        return {
            "version": ADAPTER_CONTRACT_VERSION,
            "adapter_id": DEFAULT_ADAPTER_ID,
            "task_profiles": [DEFAULT_TASK_PROFILE],
            "preferred_training_tasks": ["causal_lm"],
            "output_contract": {"required_fields": ["text"], "optional_fields": [], "notes": []},
            "schema_hint": {},
        }
    resolved_id = normalized_id if normalized_id in registry else DEFAULT_ADAPTER_ID
    return _adapter_contract(resolved_id, payload)


def resolve_task_profile_for_adapter(
    adapter_id: str | None,
    *,
    requested_task_profile: str | None = None,
) -> str:
    contract = resolve_data_adapter_contract(adapter_id)
    profiles = [str(item) for item in list(contract.get("task_profiles") or []) if str(item).strip()]
    requested = normalize_task_profile(requested_task_profile, default=AUTO_TASK_PROFILE)

    if requested not in {"", AUTO_TASK_PROFILE} and requested in profiles:
        return requested
    if profiles:
        return profiles[0]
    return DEFAULT_TASK_PROFILE


def _ensure_adapter_record(adapter_payload: dict[str, Any], *, adapter_id: str, source_module: str) -> dict[str, Any]:
    map_row = adapter_payload.get("map_row") or adapter_payload.get("map")
    if not callable(map_row):
        raise ValueError(f"Adapter '{adapter_id}' from {source_module} is missing callable map_row")

    detect = adapter_payload.get("detect")
    if detect is None:
        detect = lambda row, _cfg: 1.0 if map_row(row, _cfg) else 0.0  # noqa: E731
    if not callable(detect):
        raise ValueError(f"Adapter '{adapter_id}' from {source_module} has non-callable detect")

    validate = adapter_payload.get("validate") or _validate_default
    if not callable(validate):
        raise ValueError(f"Adapter '{adapter_id}' from {source_module} has non-callable validate")

    schema_hint = adapter_payload.get("schema_hint") or _schema_default
    if not callable(schema_hint):
        raise ValueError(f"Adapter '{adapter_id}' from {source_module} has non-callable schema_hint")

    task_profiles = _normalize_task_profiles(adapter_payload.get("task_profiles"))
    preferred_training_tasks = _normalize_training_tasks(
        adapter_payload.get("preferred_training_tasks"),
        fallback_profile=task_profiles[0],
    )

    schema_hint_payload: dict[str, Any] = {}
    try:
        candidate = schema_hint()
        if isinstance(candidate, dict):
            schema_hint_payload = candidate
    except Exception:
        schema_hint_payload = {}

    output_contract = _normalize_output_contract(
        adapter_payload.get("output_contract"),
        schema_hint=schema_hint_payload,
    )

    return {
        "description": str(adapter_payload.get("description") or f"Plugin adapter from {source_module}"),
        "detect": detect,
        "map_row": map_row,
        "validate": validate,
        "schema_hint": schema_hint,
        "task_profiles": task_profiles,
        "preferred_training_tasks": preferred_training_tasks,
        "output_contract": output_contract,
    }


def _register_plugin_adapter(
    adapter_id: str,
    adapter_payload: dict[str, Any],
    *,
    source_module: str,
) -> None:
    normalized_id = _normalize_adapter_id(adapter_id)
    if not normalized_id:
        raise ValueError("Adapter id cannot be empty")
    if normalized_id in {AUTO_ADAPTER_ID}:
        raise ValueError(f"Adapter id '{normalized_id}' is reserved")
    payload = _ensure_adapter_record(adapter_payload, adapter_id=normalized_id, source_module=source_module)
    _PLUGIN_ADAPTERS[normalized_id] = payload
    _PLUGIN_ADAPTER_SOURCE[normalized_id] = source_module


def _register_adapters_from_mapping(
    mapping: dict[str, Any],
    *,
    source_module: str,
) -> int:
    count = 0
    for adapter_id, payload in mapping.items():
        if callable(payload):
            _register_plugin_adapter(str(adapter_id), {"map_row": payload}, source_module=source_module)
            count += 1
            continue
        if not isinstance(payload, dict):
            continue
        _register_plugin_adapter(str(adapter_id), payload, source_module=source_module)
        count += 1
    return count


def clear_plugin_data_adapters() -> None:
    with _REGISTRY_LOCK:
        _PLUGIN_ADAPTERS.clear()
        _PLUGIN_ADAPTER_SOURCE.clear()
        _LOADED_PLUGIN_MODULES.clear()
        _PLUGIN_LOAD_ERRORS.clear()


def load_data_adapter_plugins(
    module_paths: list[str],
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    loaded: list[str] = []
    failed: dict[str, str] = {}
    registered = 0

    with _REGISTRY_LOCK:
        for module_path in module_paths:
            module_name = str(module_path).strip()
            if not module_name:
                continue
            if module_name in _LOADED_PLUGIN_MODULES and not force_reload:
                loaded.append(module_name)
                continue

            try:
                module = importlib.import_module(module_name)
                if force_reload:
                    module = importlib.reload(module)

                module_count = 0
                if hasattr(module, "register_data_adapters") and callable(module.register_data_adapters):
                    callback = module.register_data_adapters
                    params = inspect.signature(callback).parameters
                    if len(params) != 1:
                        raise ValueError("register_data_adapters(register) must accept exactly one argument")

                    def register(
                        adapter_id: str,
                        map_row: AdapterMapFn,
                        *,
                        detect: AdapterDetectFn | None = None,
                        validate: AdapterValidateFn | None = None,
                        schema_hint: AdapterSchemaHintFn | None = None,
                        description: str = "",
                        task_profiles: list[str] | tuple[str, ...] | None = None,
                        preferred_training_tasks: list[str] | tuple[str, ...] | None = None,
                        output_contract: dict[str, Any] | None = None,
                    ) -> None:
                        payload: dict[str, Any] = {"map_row": map_row, "description": description}
                        if detect is not None:
                            payload["detect"] = detect
                        if validate is not None:
                            payload["validate"] = validate
                        if schema_hint is not None:
                            payload["schema_hint"] = schema_hint
                        if task_profiles is not None:
                            payload["task_profiles"] = list(task_profiles)
                        if preferred_training_tasks is not None:
                            payload["preferred_training_tasks"] = list(preferred_training_tasks)
                        if output_contract is not None:
                            payload["output_contract"] = dict(output_contract)
                        _register_plugin_adapter(adapter_id, payload, source_module=module_name)

                    before = len(_PLUGIN_ADAPTERS)
                    callback(register)
                    module_count += max(0, len(_PLUGIN_ADAPTERS) - before)

                mapping: dict[str, Any] | None = None
                if hasattr(module, "get_data_adapters") and callable(module.get_data_adapters):
                    payload = module.get_data_adapters()
                    if isinstance(payload, dict):
                        mapping = payload
                elif isinstance(getattr(module, "DATA_ADAPTERS", None), dict):
                    mapping = getattr(module, "DATA_ADAPTERS")

                if mapping:
                    module_count += _register_adapters_from_mapping(mapping, source_module=module_name)

                _LOADED_PLUGIN_MODULES.add(module_name)
                if module_name in _PLUGIN_LOAD_ERRORS:
                    _PLUGIN_LOAD_ERRORS.pop(module_name, None)
                loaded.append(module_name)
                registered += module_count
            except Exception as exc:  # noqa: PERF203
                message = str(exc)
                _PLUGIN_LOAD_ERRORS[module_name] = message
                failed[module_name] = message

    return {
        "requested_modules": [str(m).strip() for m in module_paths if str(m).strip()],
        "loaded_modules": sorted(set(loaded)),
        "failed_modules": failed,
        "registered_adapters": registered,
    }


def load_data_adapter_plugins_from_settings(*, force_reload: bool = False) -> dict[str, Any]:
    modules = [str(item).strip() for item in (settings.DATA_ADAPTER_PLUGIN_MODULES or []) if str(item).strip()]
    if not modules:
        return {
            "requested_modules": [],
            "loaded_modules": sorted(_LOADED_PLUGIN_MODULES),
            "failed_modules": copy.deepcopy(_PLUGIN_LOAD_ERRORS),
            "registered_adapters": len(_PLUGIN_ADAPTERS),
            "status": "no_plugin_modules_configured",
        }
    return load_data_adapter_plugins(modules, force_reload=force_reload)


def list_data_adapter_catalog() -> dict[str, Any]:
    registry = _adapter_registry()
    adapters: dict[str, Any] = {
        AUTO_ADAPTER_ID: {
            "description": "Auto-detect best adapter from sampled rows. Falls back to default-canonical.",
            "source": "builtin",
            "schema_hint": {"output_shape": "auto"},
            "contract": {
                "version": ADAPTER_CONTRACT_VERSION,
                "adapter_id": AUTO_ADAPTER_ID,
                "task_profiles": list(SUPPORTED_TASK_PROFILES),
                "preferred_training_tasks": ["causal_lm", "seq2seq", "classification", "dpo", "orpo"],
                "output_contract": {
                    "required_fields": ["text"],
                    "optional_fields": ["question", "answer", "source_text", "target_text", "messages"],
                    "notes": ["auto picks a concrete adapter based on sampled rows"],
                },
            },
        }
    }
    for adapter_id, payload in sorted(registry.items()):
        contract = _adapter_contract(adapter_id, payload)
        adapters[adapter_id] = {
            "description": str(payload.get("description") or ""),
            "source": _PLUGIN_ADAPTER_SOURCE.get(adapter_id, "builtin"),
            "is_default": adapter_id == DEFAULT_ADAPTER_ID,
            "schema_hint": contract.get("schema_hint", {}),
            "contract": {
                "version": contract.get("version"),
                "adapter_id": adapter_id,
                "task_profiles": list(contract.get("task_profiles") or []),
                "preferred_training_tasks": list(contract.get("preferred_training_tasks") or []),
                "output_contract": dict(contract.get("output_contract") or {}),
            },
        }
    return {
        "default_adapter": DEFAULT_ADAPTER_ID,
        "contract_version": ADAPTER_CONTRACT_VERSION,
        "supported_task_profiles": list(SUPPORTED_TASK_PROFILES),
        "adapters": adapters,
        "loaded_plugin_modules": sorted(_LOADED_PLUGIN_MODULES),
        "plugin_load_errors": copy.deepcopy(_PLUGIN_LOAD_ERRORS),
    }


def _adapter_score(
    adapter_id: str,
    sample_records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    requested_task_profile: str | None = None,
) -> float:
    registry = _adapter_registry()
    payload = registry.get(adapter_id)
    if not payload:
        return 0.0
    detect_fn = payload.get("detect")
    if not callable(detect_fn):
        return 0.0
    if not sample_records:
        return 0.0

    scores: list[float] = []
    for row in sample_records:
        try:
            signal = detect_fn(row, config)
        except Exception:
            signal = 0.0
        if isinstance(signal, bool):
            scores.append(1.0 if signal else 0.0)
            continue
        scores.append(_safe_float(signal))
    if not scores:
        return 0.0

    score = sum(scores) / len(scores)

    requested = normalize_task_profile(requested_task_profile, default=AUTO_TASK_PROFILE)
    if requested != AUTO_TASK_PROFILE:
        contract = _adapter_contract(adapter_id, payload)
        profiles = set(str(item) for item in list(contract.get("task_profiles") or []))
        if requested in profiles:
            score += 0.12
        elif profiles:
            score -= 0.04

    return round(max(0.0, min(1.0, score)), 6)


def resolve_data_adapter_for_records(
    records: list[dict[str, Any]],
    *,
    adapter_id: str | None = None,
    adapter_config: dict[str, Any] | None = None,
    field_mapping: dict[str, str] | None = None,
    task_profile: str | None = None,
) -> tuple[str, dict[str, float]]:
    requested = _normalize_adapter_id(adapter_id or DEFAULT_ADAPTER_ID)
    registry = _adapter_registry()
    config = _base_adapter_config(adapter_config, field_mapping, task_profile=task_profile)

    if requested and requested != AUTO_ADAPTER_ID:
        if requested in registry:
            return requested, {requested: 1.0}
        return DEFAULT_ADAPTER_ID, {DEFAULT_ADAPTER_ID: 1.0}

    sample_records = [row for row in records[:200] if isinstance(row, dict)]
    candidate_ids = [key for key in registry.keys() if key != AUTO_ADAPTER_ID]
    if DEFAULT_ADAPTER_ID in candidate_ids:
        candidate_ids = [key for key in candidate_ids if key != DEFAULT_ADAPTER_ID] + [DEFAULT_ADAPTER_ID]

    detection_scores: dict[str, float] = {}
    best_id = DEFAULT_ADAPTER_ID
    best_score = -1.0
    for candidate in candidate_ids:
        score = _adapter_score(
            candidate,
            sample_records,
            config=config,
            requested_task_profile=task_profile,
        )
        detection_scores[candidate] = score
        if score > best_score:
            best_id = candidate
            best_score = score

    if best_score <= 0 and DEFAULT_ADAPTER_ID in registry:
        best_id = DEFAULT_ADAPTER_ID
    return best_id, detection_scores


def map_record_with_adapter(
    row: dict[str, Any],
    *,
    adapter_id: str,
    adapter_config: dict[str, Any] | None = None,
    field_mapping: dict[str, str] | None = None,
    task_profile: str | None = None,
) -> dict[str, Any] | None:
    normalized_id = _normalize_adapter_id(adapter_id) or DEFAULT_ADAPTER_ID
    registry = _adapter_registry()
    payload = registry.get(normalized_id) or registry.get(DEFAULT_ADAPTER_ID)
    if not payload:
        return None
    map_fn = payload.get("map_row")
    if not callable(map_fn):
        return None
    config = _base_adapter_config(adapter_config, field_mapping, task_profile=task_profile)
    result = map_fn(row, config)
    return result if isinstance(result, dict) else None


def validate_mapped_records(
    mapped_records: list[dict[str, Any]],
    *,
    adapter_id: str,
    adapter_config: dict[str, Any] | None = None,
    field_mapping: dict[str, str] | None = None,
    task_profile: str | None = None,
) -> dict[str, Any]:
    normalized_id = _normalize_adapter_id(adapter_id) or DEFAULT_ADAPTER_ID
    registry = _adapter_registry()
    payload = registry.get(normalized_id) or registry.get(DEFAULT_ADAPTER_ID)
    if not payload:
        return {"status": "warning", "message": "No adapter registered"}
    validate_fn = payload.get("validate")
    if not callable(validate_fn):
        return {"status": "warning", "message": "Adapter missing validate hook"}
    config = _base_adapter_config(adapter_config, field_mapping, task_profile=task_profile)
    try:
        report = validate_fn(mapped_records, config)
    except Exception as exc:  # noqa: PERF203
        return {"status": "warning", "message": f"Validation failed: {exc}"}
    if isinstance(report, dict):
        return report
    return {"status": "ok"}


def preview_data_adapter(
    raw_records: list[dict[str, Any]],
    *,
    adapter_id: str = AUTO_ADAPTER_ID,
    adapter_config: dict[str, Any] | None = None,
    field_mapping: dict[str, str] | None = None,
    task_profile: str | None = None,
    preview_limit: int = 20,
) -> dict[str, Any]:
    rows = [row for row in raw_records if isinstance(row, dict)]
    requested_task_profile = normalize_task_profile(task_profile, default=AUTO_TASK_PROFILE)
    resolved_adapter_id, detection_scores = resolve_data_adapter_for_records(
        rows,
        adapter_id=adapter_id,
        adapter_config=adapter_config,
        field_mapping=field_mapping,
        task_profile=requested_task_profile,
    )
    contract = resolve_data_adapter_contract(resolved_adapter_id)
    resolved_task_profile = resolve_task_profile_for_adapter(
        resolved_adapter_id,
        requested_task_profile=requested_task_profile,
    )

    mapped_rows: list[dict[str, Any]] = []
    dropped = 0
    errors: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            mapped = map_record_with_adapter(
                row,
                adapter_id=resolved_adapter_id,
                adapter_config=adapter_config,
                field_mapping=field_mapping,
                task_profile=resolved_task_profile,
            )
        except Exception as exc:  # noqa: PERF203
            errors.append({"index": idx, "error": str(exc)})
            dropped += 1
            continue

        if not mapped:
            dropped += 1
            if len(errors) < 20:
                errors.append({"index": idx, "error": "Row did not satisfy adapter contract"})
            continue

        mapped_rows.append(mapped)
        if len(preview_rows) < max(1, preview_limit):
            preview_rows.append(
                {
                    "index": idx,
                    "raw": row,
                    "mapped": mapped,
                }
            )

    validation = validate_mapped_records(
        mapped_rows,
        adapter_id=resolved_adapter_id,
        adapter_config=adapter_config,
        field_mapping=field_mapping,
        task_profile=resolved_task_profile,
    )

    profile_options = [str(item) for item in list(contract.get("task_profiles") or []) if str(item).strip()]
    task_profile_compatible = (
        requested_task_profile == AUTO_TASK_PROFILE
        or requested_task_profile in profile_options
    )
    compatibility_warnings: list[str] = []
    if not task_profile_compatible:
        compatibility_warnings.append(
            (
                f"Requested task_profile='{requested_task_profile}' is not declared by adapter "
                f"'{resolved_adapter_id}'. Declared profiles: {', '.join(profile_options) or 'none'}."
            )
        )

    return {
        "requested_adapter_id": _normalize_adapter_id(adapter_id) or DEFAULT_ADAPTER_ID,
        "resolved_adapter_id": resolved_adapter_id,
        "requested_task_profile": None if requested_task_profile == AUTO_TASK_PROFILE else requested_task_profile,
        "resolved_task_profile": resolved_task_profile,
        "task_profile_compatible": task_profile_compatible,
        "compatibility_warnings": compatibility_warnings,
        "adapter_contract": {
            "version": contract.get("version"),
            "adapter_id": contract.get("adapter_id"),
            "task_profiles": list(contract.get("task_profiles") or []),
            "preferred_training_tasks": list(contract.get("preferred_training_tasks") or []),
            "output_contract": dict(contract.get("output_contract") or {}),
        },
        "detection_scores": detection_scores,
        "sampled_records": len(rows),
        "mapped_records": len(mapped_rows),
        "dropped_records": dropped,
        "error_count": len(errors),
        "errors": errors[:20],
        "validation_report": validation,
        "preview_rows": preview_rows,
    }
