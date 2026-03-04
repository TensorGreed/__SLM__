#!/usr/bin/env python3
"""Starter external benchmark runtime for SLM artifacts."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external SLM benchmark workflow")
    parser.add_argument("--project", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, help="Model file or directory")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--out", type=str, required=True, help="Benchmark output JSON path")
    return parser.parse_args()


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return [p for p in path.rglob("*") if p.is_file()]
    raise FileNotFoundError(f"Model path not found: {path}")


def run_io_benchmark(files: list[Path], iterations: int) -> dict:
    if not files:
        return {"iterations": 0, "avg_read_ms": 0.0, "max_read_ms": 0.0}

    iterations = max(1, iterations)
    latencies: list[float] = []
    for i in range(iterations):
        target = files[i % len(files)]
        started = time.perf_counter()
        _ = target.read_bytes()
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)

    avg_ms = round(sum(latencies) / len(latencies), 3)
    max_ms = round(max(latencies), 3)
    p95_ms = round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3)
    return {
        "iterations": len(latencies),
        "avg_read_ms": avg_ms,
        "max_read_ms": max_ms,
        "p95_read_ms": p95_ms,
    }


def main() -> int:
    args = parse_args()
    try:
        model_path = Path(args.model).expanduser().resolve()
        files = collect_files(model_path)
        total_bytes = sum(f.stat().st_size for f in files)

        benchmark = run_io_benchmark(files, iterations=min(max(args.samples, 1), 200))
        result = {
            "project_id": args.project,
            "model_path": str(model_path),
            "created_at": utcnow(),
            "file_count": len(files),
            "model_size_bytes": total_bytes,
            "model_size_mb": round(total_bytes / (1024 * 1024), 3),
            "benchmark_samples": args.samples,
            "io_benchmark": benchmark,
        }

        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"status": "completed", "report_path": str(out_path)}))
        return 0
    except Exception as e:
        print(json.dumps({"status": "failed", "error": str(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
