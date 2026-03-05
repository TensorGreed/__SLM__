#!/usr/bin/env python3
"""External benchmark runtime for real SLM serving metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


DEFAULT_PROMPTS = [
    "Summarize the key differences between supervised and unsupervised learning.",
    "Write a short troubleshooting guide for a web service returning HTTP 503 errors.",
    "Convert this requirement into a test case: user can reset password with email OTP.",
    "Explain how quantization reduces model size and what tradeoffs it introduces.",
    "Provide a concise answer: what is overfitting and how can we detect it?",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external SLM benchmark workflow")
    parser.add_argument("--project", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, help="Model file, model directory, or HF model id")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runtime", choices=["auto", "transformers", "llama_cpp"], default="auto")
    parser.add_argument("--prompt-file", type=str, default="", help="Optional .txt/.jsonl prompt file")
    parser.add_argument("--out", type=str, required=True, help="Benchmark output JSON path")
    return parser.parse_args()


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return [p for p in path.rglob("*") if p.is_file()]
    return []


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "count": 0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(latencies_ms),
        "avg_ms": round(sum(latencies_ms) / len(latencies_ms), 3),
        "p50_ms": round(_percentile(latencies_ms, 0.50), 3),
        "p95_ms": round(_percentile(latencies_ms, 0.95), 3),
        "p99_ms": round(_percentile(latencies_ms, 0.99), 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
    }


def _load_prompts(prompt_file: str, samples: int) -> list[str]:
    prompts: list[str] = []
    path = Path(prompt_file).expanduser().resolve() if prompt_file else None
    if path and path.exists():
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(row.get("prompt") or row.get("question") or row.get("instruction") or "").strip()
                if text:
                    prompts.append(text)
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if text:
                    prompts.append(text)

    if not prompts:
        prompts = DEFAULT_PROMPTS[:]

    sample_count = max(1, samples)
    return [prompts[i % len(prompts)] for i in range(sample_count)]


def _summarize_generation_metrics(
    latencies_ms: list[float],
    generated_tokens: list[int],
    total_wall_seconds: float,
) -> dict[str, Any]:
    total_tokens = int(sum(generated_tokens))
    token_rate = (total_tokens / total_wall_seconds) if total_wall_seconds > 0 else 0.0
    request_rate = (len(generated_tokens) / total_wall_seconds) if total_wall_seconds > 0 else 0.0
    return {
        "latency": _latency_summary(latencies_ms),
        "request_throughput_rps": round(request_rate, 3),
        "token_throughput_tps": round(token_rate, 3),
        "total_generated_tokens": total_tokens,
        "total_wall_time_seconds": round(total_wall_seconds, 3),
    }


def _benchmark_with_transformers(
    model_ref: str,
    prompts: list[str],
    warmup: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    use_cuda = torch.cuda.is_available()
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if use_cuda and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif use_cuda:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if use_cuda:
        model = model.to("cuda")
    model.eval()
    load_seconds = perf_counter() - load_started

    def _run_once(prompt: str) -> tuple[str, int, int, float]:
        inputs = tokenizer(prompt, return_tensors="pt")
        if use_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])

        t0 = perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = perf_counter() - t0

        generated = output_ids[0][input_tokens:]
        generated_tokens = int(generated.shape[-1])
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, input_tokens, generated_tokens, elapsed

    warmup_count = max(0, warmup)
    for i in range(warmup_count):
        _run_once(prompts[i % len(prompts)])

    latencies_ms: list[float] = []
    prompt_tokens: list[int] = []
    generated_tokens: list[int] = []
    samples: list[dict[str, Any]] = []
    wall_started = perf_counter()
    for prompt in prompts:
        output_text, p_tok, g_tok, elapsed = _run_once(prompt)
        elapsed_ms = elapsed * 1000
        latencies_ms.append(elapsed_ms)
        prompt_tokens.append(p_tok)
        generated_tokens.append(g_tok)
        if len(samples) < 5:
            samples.append(
                {
                    "prompt_preview": prompt[:160],
                    "output_preview": output_text[:220],
                    "prompt_tokens": p_tok,
                    "generated_tokens": g_tok,
                    "latency_ms": round(elapsed_ms, 3),
                }
            )

    wall_seconds = perf_counter() - wall_started
    metrics = _summarize_generation_metrics(latencies_ms, generated_tokens, wall_seconds)
    metrics["total_prompt_tokens"] = int(sum(prompt_tokens))
    metrics["average_generation_tokens"] = (
        round(sum(generated_tokens) / len(generated_tokens), 3) if generated_tokens else 0.0
    )

    return {
        "runtime": {
            "engine": "transformers",
            "device": "cuda" if use_cuda else "cpu",
            "dtype": str(getattr(model, "dtype", "unknown")),
            "model_load_seconds": round(load_seconds, 3),
        },
        "metrics": metrics,
        "samples": samples,
    }


def _benchmark_with_llama_cpp(
    model_ref: str,
    prompts: list[str],
    warmup: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    from llama_cpp import Llama

    load_started = perf_counter()
    llm = Llama(model_path=model_ref, n_ctx=4096)
    load_seconds = perf_counter() - load_started

    def _run_once(prompt: str) -> tuple[str, int, int, float]:
        t0 = perf_counter()
        out = llm(prompt, max_tokens=max_new_tokens, temperature=0.0)
        elapsed = perf_counter() - t0
        choice = out.get("choices", [{}])[0]
        usage = out.get("usage", {})
        prompt_tok = int(usage.get("prompt_tokens", 0))
        gen_tok = int(usage.get("completion_tokens", 0))
        text = str(choice.get("text", "")).strip()
        return text, prompt_tok, gen_tok, elapsed

    warmup_count = max(0, warmup)
    for i in range(warmup_count):
        _run_once(prompts[i % len(prompts)])

    latencies_ms: list[float] = []
    prompt_tokens: list[int] = []
    generated_tokens: list[int] = []
    samples: list[dict[str, Any]] = []
    wall_started = perf_counter()
    for prompt in prompts:
        output_text, p_tok, g_tok, elapsed = _run_once(prompt)
        elapsed_ms = elapsed * 1000
        latencies_ms.append(elapsed_ms)
        prompt_tokens.append(p_tok)
        generated_tokens.append(g_tok)
        if len(samples) < 5:
            samples.append(
                {
                    "prompt_preview": prompt[:160],
                    "output_preview": output_text[:220],
                    "prompt_tokens": p_tok,
                    "generated_tokens": g_tok,
                    "latency_ms": round(elapsed_ms, 3),
                }
            )

    wall_seconds = perf_counter() - wall_started
    metrics = _summarize_generation_metrics(latencies_ms, generated_tokens, wall_seconds)
    metrics["total_prompt_tokens"] = int(sum(prompt_tokens))
    metrics["average_generation_tokens"] = (
        round(sum(generated_tokens) / len(generated_tokens), 3) if generated_tokens else 0.0
    )

    return {
        "runtime": {
            "engine": "llama_cpp",
            "device": "cpu",
            "dtype": "gguf",
            "model_load_seconds": round(load_seconds, 3),
        },
        "metrics": metrics,
        "samples": samples,
    }


def _resolve_runtime(model_ref: str, runtime: str) -> str:
    if runtime != "auto":
        return runtime
    model_path = Path(model_ref).expanduser()
    if model_path.exists() and model_path.suffix.lower() == ".gguf":
        return "llama_cpp"
    return "transformers"


def main() -> int:
    args = parse_args()
    try:
        sample_count = max(1, min(int(args.samples), 1000))
        max_new_tokens = max(1, min(int(args.max_new_tokens), 1024))
        warmup = max(0, min(int(args.warmup), 20))
        model_ref = args.model.strip()
        if not model_ref:
            raise ValueError("--model cannot be empty")

        local_model_path = Path(model_ref).expanduser().resolve()
        local_files = collect_files(local_model_path) if local_model_path.exists() else []
        total_bytes = sum(f.stat().st_size for f in local_files)

        prompts = _load_prompts(args.prompt_file, sample_count)
        runtime = _resolve_runtime(model_ref, args.runtime)
        if runtime == "transformers":
            bench = _benchmark_with_transformers(model_ref, prompts, warmup, max_new_tokens)
        elif runtime == "llama_cpp":
            bench = _benchmark_with_llama_cpp(model_ref, prompts, warmup, max_new_tokens)
        else:
            raise ValueError(f"Unsupported runtime: {runtime}")

        result = {
            "project_id": args.project,
            "model_path": model_ref,
            "created_at": utcnow(),
            "benchmark_samples": sample_count,
            "warmup_runs": warmup,
            "max_new_tokens": max_new_tokens,
            "file_count": len(local_files),
            "model_size_bytes": total_bytes,
            "model_size_mb": round(total_bytes / (1024 * 1024), 3) if total_bytes else 0.0,
            "runtime": bench["runtime"],
            "metrics": bench["metrics"],
            "sample_outputs": bench["samples"],
        }

        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"status": "completed", "report_path": str(out_path), "runtime": runtime}))
        return 0
    except Exception as e:
        print(json.dumps({"status": "failed", "error": str(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
