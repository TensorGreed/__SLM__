#!/usr/bin/env python3
"""External model merge runtime (TIES/DEX model soup).

This script is best-effort:
- Real merge when transformers+torch are available and model paths are readable.
- Deterministic simulated artifact output otherwise.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple model checkpoints")
    parser.add_argument("--project", type=int, required=True)
    parser.add_argument("--models-file", type=str, required=True)
    parser.add_argument("--method", type=str, default="ties", choices=["ties", "dex"])
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--ties-density", type=float, default=0.2)
    return parser.parse_args()


def _normalize_weights(raw: list[float], count: int) -> list[float]:
    if count <= 0:
        return []
    filtered = [float(item) for item in raw if float(item) > 0]
    if not filtered or len(filtered) != count:
        return [1.0 / count] * count
    total = sum(filtered)
    if total <= 0:
        return [1.0 / count] * count
    return [item / total for item in filtered]


def _read_models_payload(path: Path) -> tuple[list[str], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"models-file payload must be JSON object: {path}")
    model_paths = payload.get("model_paths")
    if not isinstance(model_paths, list):
        raise ValueError("models-file must include model_paths: list[str]")
    normalized_paths = [str(item or "").strip() for item in model_paths if str(item or "").strip()]
    weights_raw = payload.get("weights")
    weights: list[float] = []
    if isinstance(weights_raw, list):
        for item in weights_raw:
            try:
                weights.append(float(item))
            except (TypeError, ValueError):
                continue
    return normalized_paths, weights


def _parse_weights_csv(raw: str) -> list[float]:
    token = str(raw or "").strip()
    if not token:
        return []
    out: list[float] = []
    for part in token.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _simulate_output(
    out_dir: Path,
    *,
    method: str,
    model_paths: list[str],
    weights: list[float],
    ties_density: float,
    reason: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "simulated",
        "reason": reason,
        "method": method,
        "model_paths": model_paths,
        "weights": weights,
        "ties_density": ties_density,
        "created_at": utcnow(),
    }
    (out_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest


def _merge_dex(
    *,
    state_dicts: list[dict[str, Any]],
    weights: list[float],
):
    import torch

    base_keys = list(state_dicts[0].keys())
    merged: dict[str, Any] = {}
    for key in base_keys:
        tensors = []
        valid = True
        for state in state_dicts:
            tensor = state.get(key)
            if tensor is None or not torch.is_tensor(tensor):
                valid = False
                break
            tensors.append(tensor.detach().to(dtype=torch.float32))
        if not valid or not tensors:
            merged[key] = state_dicts[0][key]
            continue
        stacked = torch.stack(tensors, dim=0)
        weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).view(-1, *([1] * (stacked.ndim - 1)))
        combined = torch.sum(stacked * weight_tensor, dim=0)
        merged[key] = combined.to(dtype=tensors[0].dtype)
    return merged


def _merge_ties(
    *,
    state_dicts: list[dict[str, Any]],
    ties_density: float,
):
    import torch

    base = state_dicts[0]
    merged: dict[str, Any] = {}
    density = max(0.01, min(float(ties_density), 1.0))
    for key, base_value in base.items():
        if not torch.is_tensor(base_value):
            merged[key] = base_value
            continue

        base_tensor = base_value.detach().to(dtype=torch.float32)
        deltas = []
        for state in state_dicts[1:]:
            candidate = state.get(key)
            if candidate is None or not torch.is_tensor(candidate):
                continue
            deltas.append(candidate.detach().to(dtype=torch.float32) - base_tensor)

        if not deltas:
            merged[key] = base_value
            continue

        sparse_deltas = []
        for delta in deltas:
            flat = delta.abs().flatten()
            k = max(1, int(flat.numel() * density))
            if k >= flat.numel():
                sparse_deltas.append(delta)
                continue
            threshold = torch.topk(flat, k).values.min()
            mask = delta.abs() >= threshold
            sparse_deltas.append(delta * mask)

        stack = torch.stack(sparse_deltas, dim=0)
        sign_sum = torch.sign(stack).sum(dim=0)
        sign_mask = sign_sum != 0
        consensus_delta = stack.mean(dim=0) * sign_mask
        merged_tensor = base_tensor + consensus_delta
        merged[key] = merged_tensor.to(dtype=base_value.dtype)
    return merged


def run_merge(args: argparse.Namespace) -> dict[str, Any]:
    models_file = Path(args.models_file).expanduser().resolve()
    if not models_file.exists():
        raise FileNotFoundError(f"models-file not found: {models_file}")

    out_dir = Path(args.out).expanduser().resolve()
    model_paths, payload_weights = _read_models_payload(models_file)
    cli_weights = _parse_weights_csv(args.weights)
    selected_weights = cli_weights if cli_weights else payload_weights

    if len(model_paths) < 2:
        raise ValueError("Need at least two model paths for merge.")
    normalized_weights = _normalize_weights(selected_weights, len(model_paths))
    ties_density = max(0.01, min(float(args.ties_density), 1.0))

    resolved_paths = [Path(item).expanduser().resolve() for item in model_paths]
    if any(not path.exists() for path in resolved_paths):
        missing = [str(path) for path in resolved_paths if not path.exists()]
        return _simulate_output(
            out_dir,
            method=args.method,
            model_paths=[str(path) for path in resolved_paths],
            weights=normalized_weights,
            ties_density=ties_density,
            reason=f"missing model paths: {', '.join(missing)}",
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        return _simulate_output(
            out_dir,
            method=args.method,
            model_paths=[str(path) for path in resolved_paths],
            weights=normalized_weights,
            ties_density=ties_density,
            reason=f"missing merge runtime dependencies: {exc}",
        )

    models = []
    state_dicts = []
    for path in resolved_paths:
        model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32)
        models.append(model)
        state_dicts.append({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if args.method == "dex":
        merged_state = _merge_dex(
            state_dicts=state_dicts,
            weights=normalized_weights,
        )
    else:
        merged_state = _merge_ties(
            state_dicts=state_dicts,
            ties_density=ties_density,
        )

    merged_model = models[0]
    merged_model.load_state_dict(merged_state, strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(out_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(resolved_paths[0]), trust_remote_code=True)
    tokenizer.save_pretrained(str(out_dir))

    report = {
        "status": "completed",
        "method": args.method,
        "model_paths": [str(path) for path in resolved_paths],
        "weights": normalized_weights,
        "ties_density": ties_density,
        "output_model_path": str(out_dir),
        "created_at": utcnow(),
    }
    (out_dir / "merge_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run_merge(args)
        print(json.dumps({"status": report.get("status"), "report": report}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

