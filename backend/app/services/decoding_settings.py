"""Explicit decoding settings for every BrewSLM ``model.generate()`` call.

``generate()`` fills any knob the caller leaves out from the model's own
``generation_config.json``, and greedy decoding still applies the
logits processors among them. Qwen2.5-Instruct ships
``repetition_penalty: 1.1`` (plus ``top_k: 20`` / ``top_p: 0.8`` for
sampling), so an eval that only passed ``do_sample=False`` was decoded
with a penalty nobody chose: by the third JSON field the `` "`` token is
penalised and `` null`` wins close calls. On the NTSB extractor (project
39, seed 42, validation) that turned 232 of 468 stated aircraft models
into null; with the penalty off, 16.

So every call site passes the full set here, and evals record it
(``decoding_record``) so a result says how it was decoded. HF exports
write the same settings into ``generation_config.json``
(``neutral_generation_config``) so the shipped model decodes the way it
was evaluated.

A lift check also scores the base model under its OWN settings
(``decoding="model"``) when they differ, and the stronger of the two is
the bar: an Instruct base can need its maker's penalty (the Pub 17
reader's base + passages: 0.551 with it, 0.331 without, as it loops on
"[1]"), and a bar the base beats by decoding differently overstates the
fine-tune's lift. ``post_training_eval_service`` and
``passage_reader_eval_service`` apply it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Knobs that change output under greedy decoding too: pinned to "off" on
# every call, as explicit values so the intent reads the same in a recorded
# result or a config dump.
NEUTRAL_PROCESSORS: dict[str, Any] = {
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}

# Sampling-only warpers, pinned to values that filter nothing, so sampling
# is plain sampling at the given temperature. Not passed under greedy (no
# warper runs there; passing them only trips generate()'s config warnings).
# ``top_h`` adds a warper whenever set and has no neutral value; no model
# config we export sets it.
NEUTRAL_SAMPLING: dict[str, Any] = {
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "typical_p": 1.0,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
}

# llama-cpp-python's own sampler defaults (older releases: repeat_penalty 1.1,
# top_k 40, top_p 0.95, min_p 0.05) — same rule for the GGUF eval path.
LLAMA_CPP_NEUTRAL: dict[str, Any] = {
    "repeat_penalty": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
}


NEUTRAL_DECODING = "neutral"
MODEL_DECODING = "model"


def generation_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    pad_token_id: int | None = None,
    decoding: str = NEUTRAL_DECODING,
) -> dict[str, Any]:
    """Greedy when ``temperature <= 0``, plain sampling at ``temperature``
    otherwise; never a processor inherited from the model.
    ``decoding="model"``: greedy, every other knob from the model's own
    ``generation_config`` (the strongest-baseline rule's second bar)."""
    if decoding == MODEL_DECODING:
        kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
        if pad_token_id is not None:
            kwargs["pad_token_id"] = pad_token_id
        return kwargs
    sample = temperature > 0
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": sample,
        "temperature": float(temperature) if sample else 1.0,
        **NEUTRAL_PROCESSORS,
        **(NEUTRAL_SAMPLING if sample else {}),
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def decoding_record(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The decoding part of ``kwargs``, for eval metrics / details."""
    keys = ("do_sample", "temperature", *NEUTRAL_PROCESSORS, *NEUTRAL_SAMPLING)
    return {k: kwargs[k] for k in keys if k in kwargs}


def model_decoding_settings(config: Any) -> dict[str, Any]:
    """The greedy-relevant knobs a model's own ``generation_config`` (a
    ``GenerationConfig`` or its dict) applies, defaults filled in."""
    def get(key: str) -> Any:
        return config.get(key) if isinstance(config, dict) else getattr(config, key, None)

    return {k: (get(k) if get(k) is not None else neutral) for k, neutral in NEUTRAL_PROCESSORS.items()}


def model_decoding_record(config: Any) -> dict[str, Any]:
    """``decoding_record`` for a ``decoding="model"`` eval."""
    return {"source": MODEL_DECODING, "do_sample": False, **model_decoding_settings(config)}


def is_model_decoding(record: Any) -> bool:
    return isinstance(record, dict) and record.get("source") == MODEL_DECODING


def model_decoding_differs(model_ref: str) -> dict[str, Any] | None:
    """The model's own greedy settings when they differ from neutral (so
    scoring it under them is a second bar), else None — also when it ships
    no ``generation_config`` or it can't be read."""
    try:
        from transformers import GenerationConfig

        config = GenerationConfig.from_pretrained(model_ref)
    except Exception:  # noqa: BLE001 — no config: generate() falls back to neutral defaults
        return None
    settings = model_decoding_settings(config)
    return settings if settings != NEUTRAL_PROCESSORS else None


# Written into an exported generation_config.json; sampling warpers are
# removed instead (their defaults are off, and a non-default top_k under
# greedy fails GenerationConfig's strict save validation).
_EXPORT_SET: dict[str, Any] = {"do_sample": False, "temperature": 1.0, **NEUTRAL_PROCESSORS}
_EXPORT_DROP = ("top_k", "top_p", "min_p", "top_h", "typical_p", "epsilon_cutoff", "eta_cutoff")


def neutral_generation_config(existing: dict[str, Any] | None) -> dict[str, Any]:
    """``generation_config.json`` contents for an exported model: greedy with
    no inherited processors; every other key (token ids, forced / suppressed
    tokens, language maps, length limits) kept as the eval saw it."""
    out = {k: v for k, v in (existing or {}).items() if k not in _EXPORT_DROP}
    out.update(_EXPORT_SET)
    return out


def align_exported_generation_config(model_dir: Path) -> dict[str, Any] | None:
    """Rewrite ``model_dir/generation_config.json`` (if any) to the settings
    BrewSLM evaluated with. Returns ``{"replaced": {...}}`` with the old value
    of every key it changed or removed, or None when there was no file."""
    path = Path(model_dir) / "generation_config.json"
    if not path.is_file():
        return None
    existing = json.loads(path.read_text(encoding="utf-8"))
    aligned = neutral_generation_config(existing)
    replaced = {k: v for k, v in existing.items() if k not in aligned or aligned[k] != v}
    path.write_text(json.dumps(aligned, indent=2) + "\n", encoding="utf-8")
    return {"replaced": replaced, "written": aligned}
