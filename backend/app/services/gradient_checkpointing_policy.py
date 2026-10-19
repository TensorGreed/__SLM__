"""Whether to recompute activations in the backward pass
(``gradient_checkpointing="auto"``).

Gradient checkpointing trades ~1.6x training time for activation memory.
It was on for every run, including a 135M model on a 128 GB GPU where it
buys nothing: on the GB10, the default Support FAQ run took 30 s with it
and 19 s without, with identical loss (2.205 / 2.799 eval) — a silent 60%
tax on every small-model run. Models up to ``AUTO_CHECKPOINT_MAX_PARAMS``
train without it; larger models keep it (their activations are what runs a
GPU out of memory, and the OOM auto-retry only shrinks batch / sequence).
Sequence length matters as much as model size: activation memory grows
with tokens per row, so rows longer than ``AUTO_CHECKPOINT_MAX_SEQ_TOKENS``
(e.g. passage-prompt runs, ~1k-token prompts) turn it on whatever the model
size — a 1.5B full fine-tune on such rows sat near 90 GB on 2026-10-19.
``sequence_tokens`` is the longest tokenized training row, capped by
``max_seq_length`` (rows never exceed it).
An explicit True / False is always honoured.

Pure + dependency-free: imported by ``scripts/train.py`` at run time.
"""

from __future__ import annotations

from typing import Any

AUTO = "auto"
# Same size class as the all-linear LoRA default: measured up to 1.5B.
AUTO_CHECKPOINT_MAX_PARAMS = 2_000_000_000
# Rows longer than this many tokens checkpoint whatever the model size.
AUTO_CHECKPOINT_MAX_SEQ_TOKENS = 1024


def resolve_gradient_checkpointing(
    requested: Any, *, model_params: int | None, sequence_tokens: int | None = None
) -> dict[str, Any]:
    """Returns ``{"enabled", "requested", "auto", "reason"}`` (+ ``model_params``,
    ``sequence_tokens``)."""
    if requested != AUTO:
        enabled = bool(requested) if requested is not None else True
        return {"enabled": enabled, "requested": requested, "auto": False, "reason": "explicit setting"}
    seq = int(sequence_tokens) if sequence_tokens else None
    if seq is not None and seq > AUTO_CHECKPOINT_MAX_SEQ_TOKENS:
        return {
            "enabled": True, "requested": AUTO, "auto": True,
            "model_params": int(model_params) if model_params else None, "sequence_tokens": seq,
            "reason": (
                f"training rows up to {seq} tokens (> {AUTO_CHECKPOINT_MAX_SEQ_TOKENS}): "
                "recomputing activations to fit memory"
            ),
        }
    params = int(model_params) if model_params else None
    if params is None:
        return {
            "enabled": True, "requested": AUTO, "auto": True, "model_params": None,
            "reason": "model size unknown: keeping activation checkpointing on",
        }
    if params <= AUTO_CHECKPOINT_MAX_PARAMS:
        return {
            "enabled": False, "requested": AUTO, "auto": True, "model_params": params,
            "reason": (
                f"{params / 1e6:.0f}M-parameter model: activation memory is small, "
                f"so no recompute (~1.6x faster steps)"
            ),
        }
    return {
        "enabled": True, "requested": AUTO, "auto": True, "model_params": params,
        "reason": f"{params / 1e9:.1f}B-parameter model: recomputing activations to fit memory",
    }
