"""Model introspection helpers for Hugging Face compatible model IDs.

Phase 2 objective:
- keep curated defaults, but enrich with on-demand model metadata
- support architecture/context/license/memory estimation for arbitrary models
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from urllib import parse, request


_HTTP_USER_AGENT = "slm-platform/model-introspection-v1"
_DEFAULT_TIMEOUT_SECONDS = 2.5
_INTROSPECTION_CACHE: dict[tuple[str, bool], dict[str, Any]] = {}

_CONTEXT_KEYS: tuple[str, ...] = (
    "max_position_embeddings",
    "n_positions",
    "n_ctx",
    "seq_length",
    "max_seq_len",
    "max_sequence_length",
)


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _coerce_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _read_json_url(url: str, *, timeout_seconds: float) -> dict[str, Any] | None:
    req = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": _HTTP_USER_AGENT,
        },
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=max(0.2, float(timeout_seconds))) as resp:
            data = resp.read()
    except Exception:
        return None
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _resolve_local_config(model_id: str) -> tuple[dict[str, Any] | None, str]:
    raw = str(model_id or "").strip()
    if not raw:
        return None, "none"

    path = Path(raw).expanduser()
    if not path.exists():
        return None, "none"
    if path.is_file():
        if path.name == "config.json":
            payload = _read_json_file(path)
            if payload is not None:
                return payload, "local_config"
        return None, "none"

    config_path = path / "config.json"
    if not config_path.exists():
        return None, "none"
    payload = _read_json_file(config_path)
    if payload is None:
        return None, "none"
    return payload, "local_config"


def _resolve_hf_remote_config(model_id: str, *, timeout_seconds: float) -> tuple[dict[str, Any] | None, str]:
    token = str(model_id or "").strip()
    if not token:
        return None, "none"
    escaped = parse.quote(token, safe="/")
    config_url = f"https://huggingface.co/{escaped}/raw/main/config.json"
    config_payload = _read_json_url(config_url, timeout_seconds=timeout_seconds)
    if config_payload is not None:
        return config_payload, "hf_config"
    return None, "none"


def _resolve_hf_model_info(model_id: str, *, timeout_seconds: float) -> dict[str, Any] | None:
    token = str(model_id or "").strip()
    if not token:
        return None
    escaped = parse.quote(token, safe="/")
    info_url = f"https://huggingface.co/api/models/{escaped}"
    return _read_json_url(info_url, timeout_seconds=timeout_seconds)


def _infer_architecture(config_payload: dict[str, Any]) -> tuple[str, str | None]:
    if not isinstance(config_payload, dict):
        return "unknown", None

    architectures = config_payload.get("architectures")
    architecture_hint = None
    if isinstance(architectures, list) and architectures:
        architecture_hint = str(architectures[0] or "").strip()
    if not architecture_hint:
        architecture_hint = str(config_payload.get("model_type") or "").strip() or None

    lowered = str(architecture_hint or "").lower()
    if "forcausallm" in lowered or "causallm" in lowered:
        return "causal_lm", architecture_hint
    if "forconditionalgeneration" in lowered or "seq2seq" in lowered:
        return "seq2seq", architecture_hint
    if "forsequenceclassification" in lowered or "forimageclassification" in lowered:
        return "classification", architecture_hint
    if bool(config_payload.get("is_encoder_decoder")):
        return "seq2seq", architecture_hint
    return "unknown", architecture_hint


def _infer_context_length(config_payload: dict[str, Any]) -> int | None:
    if not isinstance(config_payload, dict):
        return None
    for key in _CONTEXT_KEYS:
        value = _coerce_positive_int(config_payload.get(key))
        if value is not None:
            return value
    return None


def _estimate_params_b(config_payload: dict[str, Any], model_info_payload: dict[str, Any] | None) -> float | None:
    # Prefer explicit metadata from HF model info when present.
    info_payload = dict(model_info_payload or {})
    safetensors = info_payload.get("safetensors")
    if isinstance(safetensors, dict):
        parameters = safetensors.get("parameters")
        if isinstance(parameters, dict):
            explicit = _coerce_positive_float(parameters.get("total"))
            if explicit is not None:
                return round(float(explicit) / 1_000_000_000.0, 4)
            summed = 0.0
            for value in parameters.values():
                parsed = _coerce_positive_float(value)
                if parsed is not None:
                    summed += float(parsed)
            if summed > 0:
                return round(summed / 1_000_000_000.0, 4)

    hidden_size = _coerce_positive_float(config_payload.get("hidden_size"))
    layer_count = _coerce_positive_float(config_payload.get("num_hidden_layers"))
    vocab_size = _coerce_positive_float(config_payload.get("vocab_size"))
    intermediate_size = _coerce_positive_float(config_payload.get("intermediate_size"))
    if hidden_size is None or layer_count is None:
        return None

    if intermediate_size is None:
        intermediate_size = hidden_size * 4.0

    # Rough transformer estimate:
    # - attention projections + output projection ~ 4 * h^2
    # - MLP projections ~ 3 * h * i
    # - per-layer norms/biases are small relative to projections
    per_layer = (4.0 * hidden_size * hidden_size) + (3.0 * hidden_size * intermediate_size)
    total = float(layer_count) * float(per_layer)
    if vocab_size is not None:
        total += float(vocab_size) * float(hidden_size)
    if total <= 0:
        return None
    return round(total / 1_000_000_000.0, 4)


def _estimate_memory_profile(params_b: float | None) -> dict[str, float]:
    if params_b is None or params_b <= 0:
        return {}
    # Conservative, coarse heuristics for adapter-style fine-tuning.
    min_vram_gb = max(2.0, (params_b * 1.9) + 1.5)
    ideal_vram_gb = max(min_vram_gb + 2.0, (params_b * 2.8) + 3.0)
    return {
        "estimated_min_vram_gb": round(min_vram_gb, 2),
        "estimated_ideal_vram_gb": round(ideal_vram_gb, 2),
    }


def _extract_license(
    *,
    config_payload: dict[str, Any],
    model_info_payload: dict[str, Any] | None,
) -> str | None:
    raw = str(config_payload.get("license") or "").strip()
    if raw:
        return raw

    info_payload = dict(model_info_payload or {})
    card_data = info_payload.get("cardData")
    if isinstance(card_data, dict):
        value = str(card_data.get("license") or "").strip()
        if value:
            return value
    return None


def introspect_hf_model(
    model_id: str,
    *,
    allow_network: bool = True,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Resolve model metadata from local config or HF endpoints.

    Returns a stable payload even when introspection fails.
    """

    token = str(model_id or "").strip()
    cache_key = (token, bool(allow_network))
    cached = _INTROSPECTION_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)

    warnings: list[str] = []
    source = "none"
    config_payload: dict[str, Any] = {}
    model_info_payload: dict[str, Any] | None = None

    local_payload, local_source = _resolve_local_config(token)
    if local_payload is not None:
        config_payload = dict(local_payload)
        source = local_source
    elif bool(allow_network):
        remote_payload, remote_source = _resolve_hf_remote_config(
            token,
            timeout_seconds=timeout_seconds,
        )
        if remote_payload is not None:
            config_payload = dict(remote_payload)
            source = remote_source
        model_info_payload = _resolve_hf_model_info(
            token,
            timeout_seconds=timeout_seconds,
        )
    else:
        warnings.append("network introspection disabled")

    architecture, architecture_hint = _infer_architecture(config_payload)
    context_length = _infer_context_length(config_payload)
    params_b = _estimate_params_b(config_payload, model_info_payload)
    memory_profile = _estimate_memory_profile(params_b)
    license_value = _extract_license(
        config_payload=config_payload,
        model_info_payload=model_info_payload,
    )
    model_type = str(config_payload.get("model_type") or "").strip() or None

    payload = {
        "model_id": token,
        "resolved": bool(config_payload) or bool(model_info_payload),
        "source": source,
        "model_type": model_type,
        "architecture": architecture,
        "architecture_hint": architecture_hint,
        "context_length": context_length,
        "license": license_value,
        "params_estimate_b": params_b,
        "memory_profile": memory_profile,
        "warnings": warnings,
    }
    _INTROSPECTION_CACHE[cache_key] = copy.deepcopy(payload)
    return payload

