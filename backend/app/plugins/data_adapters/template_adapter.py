"""Template for third-party data adapter plugins.

Copy this file into your own module namespace, customize adapter id + mapping
logic, then add that module path to DATA_ADAPTER_PLUGIN_MODULES.
"""

from __future__ import annotations

from typing import Any


PLUGIN_VERSION = "1.0.0"


def register_data_adapters(register) -> None:
    """Register one or more data adapters."""

    def map_support_ticket(row: dict[str, Any], _config: dict[str, Any]) -> dict[str, Any] | None:
        question = str(row.get("customer_question") or "").strip()
        answer = str(row.get("agent_answer") or "").strip()
        if not question and not answer:
            return None
        return {
            "question": question,
            "answer": answer,
            "text": f"User: {question}\nAssistant: {answer}".strip(),
        }

    def detect_support_ticket(row: dict[str, Any], _config: dict[str, Any]) -> float:
        has_q = bool(str(row.get("customer_question") or "").strip())
        has_a = bool(str(row.get("agent_answer") or "").strip())
        if has_q and has_a:
            return 1.0
        if has_q or has_a:
            return 0.4
        return 0.0

    def schema_hint() -> dict[str, Any]:
        return {
            "version": PLUGIN_VERSION,
            "input_shape": {
                "customer_question": "required",
                "agent_answer": "required",
            },
            "output_shape": {
                "question": "required",
                "answer": "required",
                "text": "required",
            },
        }

    register(
        adapter_id="template-support-ticket-v1",
        map_row=map_support_ticket,
        detect=detect_support_ticket,
        schema_hint=schema_hint,
        description=(
            "Template adapter mapping support-ticket style rows into canonical "
            "question/answer/text fields."
        ),
        task_profiles=["instruction_sft", "chat_sft", "qa"],
        preferred_training_tasks=["causal_lm"],
        output_contract={
            "required_fields": ["question", "answer", "text"],
            "optional_fields": ["context", "metadata"],
            "notes": [
                "Update adapter_id before production use.",
                "Tune detect score for your source schema.",
            ],
        },
    )

