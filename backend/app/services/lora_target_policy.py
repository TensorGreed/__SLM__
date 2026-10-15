"""Which layers LoRA adapts when the config says ``target_modules="auto"``.

Adapting only the attention query/value projections (``q_proj``, ``v_proj``)
is the classic LoRA default, sized for large models. A small model has little
capacity there: on the Support FAQ sample (108 train rows, 216 steps, r=16)
the q/v adapter left train loss at 3.1 and the model kept the answer *style*
while inventing the facts. Adapting every linear layer (attention + MLP)
fixed most of that at the same rank.

Measured, held-out token-F1 (21 val / 21 test rows, same split and config):

    SmolLM2-135M (3 seeds)   q/v 0.179 / 0.153   all-linear 0.218 / 0.204
    Qwen2.5-1.5B (1 seed)    q/v 0.256 / 0.260   all-linear 0.362 / 0.312

Cost: ~1.6x training time and a ~5-8x larger adapter (20 MB / 74 MB) — small
in absolute terms at these sizes. Above ``AUTO_ALL_LINEAR_MAX_PARAMS`` nothing
was measured, so the classic q/v default stays; same for non-causal-LM tasks.
An explicit ``target_modules`` list (or ``"all-linear"``) is always used as-is.

Pure + dependency-free: imported by ``scripts/train.py`` at run time.
"""

from __future__ import annotations

from typing import Any

AUTO = "auto"
ALL_LINEAR = "all-linear"
CLASSIC_TARGET_MODULES: tuple[str, ...] = ("q_proj", "v_proj")
# The same attention query/value projections under other model families'
# names, tried in order when the model has no q_proj/v_proj (encoders:
# DeBERTa query_proj/value_proj, BERT/RoBERTa query/value, DistilBERT
# q_lin/v_lin, T5 q/v). Without this a classification run on an encoder
# failed: "No modules were targeted for adaptation".
QUERY_VALUE_ALIASES: tuple[tuple[str, str], ...] = (
    ("q_proj", "v_proj"),
    ("query_proj", "value_proj"),
    ("query", "value"),
    ("q_lin", "v_lin"),
    ("q", "v"),
)


def query_value_modules(linear_module_names: Any) -> list[str] | None:
    """The model's own query/value projection names (leaf names of its
    linear layers), or None when it has none of the known pairs."""
    names = {str(n).rsplit(".", 1)[-1] for n in (linear_module_names or [])}
    for q, v in QUERY_VALUE_ALIASES:
        if q in names and v in names:
            return [q, v]
    return None
# Largest model the all-linear default was measured on (Qwen2.5-1.5B), with
# headroom for the 2B class.
AUTO_ALL_LINEAR_MAX_PARAMS = 2_000_000_000


def resolve_lora_target_modules(
    requested: Any,
    *,
    model_params: int | None,
    task_type: str = "causal_lm",
    linear_module_names: Any = None,
) -> dict[str, Any]:
    """Resolve the configured ``target_modules`` to what PEFT should get.

    Returns ``{"target_modules", "requested", "auto", "reason"}``.
    """
    if requested != AUTO:
        return {
            "target_modules": requested,
            "requested": requested,
            "auto": False,
            "reason": "explicit target_modules",
        }
    params = int(model_params) if model_params else None
    normalized_task = str(task_type or "").strip().lower()
    if normalized_task != "causal_lm":
        own = query_value_modules(linear_module_names) if linear_module_names is not None else list(CLASSIC_TARGET_MODULES)
        if own:
            resolved: Any = own
            reason = f"task_type={normalized_task or 'unknown'}: the model's query/value projections ({own[0]}/{own[1]})"
        else:
            resolved = ALL_LINEAR
            reason = f"task_type={normalized_task or 'unknown'}: no known query/value projection names — every linear layer"
    elif params is None:
        resolved = list(CLASSIC_TARGET_MODULES)
        reason = "model size unknown"
    elif params <= AUTO_ALL_LINEAR_MAX_PARAMS:
        resolved = ALL_LINEAR
        reason = (
            f"{params / 1e6:.0f}M-parameter model: adapting every linear layer "
            f"(small models have too little capacity in q/v alone)"
        )
    else:
        resolved = list(CLASSIC_TARGET_MODULES)
        reason = (
            f"{params / 1e9:.1f}B-parameter model (> {AUTO_ALL_LINEAR_MAX_PARAMS / 1e9:.0f}B): "
            f"classic q_proj/v_proj"
        )
    return {
        "target_modules": resolved,
        "requested": AUTO,
        "auto": True,
        "model_params": params,
        "reason": reason,
    }
