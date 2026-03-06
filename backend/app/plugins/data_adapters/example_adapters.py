"""Example data adapter plugin module."""

from __future__ import annotations

from typing import Any


def _map_instruction_response(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    instruction_key = str(config.get("instruction_field") or "instruction")
    response_key = str(config.get("response_field") or "response")
    instruction = str(row.get(instruction_key) or "").strip()
    response = str(row.get(response_key) or "").strip()
    if not instruction or not response:
        return None
    return {
        "text": f"Instruction: {instruction}\nResponse: {response}",
        "question": instruction,
        "answer": response,
        "source_text": instruction,
        "target_text": response,
    }


def _detect_instruction_response(row: dict[str, Any], config: dict[str, Any]) -> bool:
    mapped = _map_instruction_response(row, config)
    return mapped is not None


def _validate_instruction_response(mapped_rows: list[dict[str, Any]], _config: dict[str, Any]) -> dict[str, Any]:
    total = len(mapped_rows)
    return {
        "status": "ok" if total > 0 else "warning",
        "mapped_records": total,
    }


def _schema_instruction_response() -> dict[str, Any]:
    return {
        "input_candidates": {
            "instruction": ["instruction", "prompt", "question"],
            "response": ["response", "output", "answer", "completion"],
        },
        "output_shape": {
            "text": "required",
            "source_text": "required",
            "target_text": "required",
        },
    }


def get_data_adapters() -> dict[str, dict[str, Any]]:
    return {
        "instruction-response": {
            "description": "Example plugin adapter for instruction/response datasets.",
            "detect": _detect_instruction_response,
            "map_row": _map_instruction_response,
            "validate": _validate_instruction_response,
            "schema_hint": _schema_instruction_response,
        }
    }

