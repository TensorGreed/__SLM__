#!/usr/bin/env python3
"""Starter external quantization runtime.

This script validates input artifacts and emits a deterministic output artifact
plus a JSON report. It is intended to be wired through compression commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external SLM quantization workflow")
    parser.add_argument("--project", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, help="Source model file or directory")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--format", type=str, default="gguf")
    parser.add_argument("--out", type=str, required=True, help="Output artifact path")
    parser.add_argument("--adapter", type=str, default="", help="Optional adapter path for merge workflows")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


def summarize_source(path: Path) -> dict:
    if path.is_file():
        return {"type": "file", "files": 1, "bytes": path.stat().st_size}

    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        total = sum(f.stat().st_size for f in files)
        return {"type": "directory", "files": len(files), "bytes": total}

    raise FileNotFoundError(f"Model path not found: {path}")


def emit_artifact(out_path: Path, payload: dict) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix or out_path.name.endswith(".gguf"):
        content = json.dumps(payload, indent=2).encode("utf-8")
        out_path.write_bytes(content)
        return {
            "artifact_path": str(out_path),
            "artifact_bytes": len(content),
            "artifact_sha256": sha256_bytes(content),
        }

    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / "artifact.json"
    content = json.dumps(payload, indent=2).encode("utf-8")
    manifest_path.write_bytes(content)
    return {
        "artifact_path": str(manifest_path),
        "artifact_bytes": len(content),
        "artifact_sha256": sha256_bytes(content),
    }


def main() -> int:
    args = parse_args()
    try:
        model_path = Path(args.model).expanduser().resolve()
        out_path = Path(args.out).expanduser().resolve()
        adapter_path = Path(args.adapter).expanduser().resolve() if args.adapter else None

        source_info = summarize_source(model_path)
        if adapter_path and not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        payload = {
            "project_id": args.project,
            "source_model": str(model_path),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "quantization": f"{args.bits}-bit",
            "format": args.format,
            "created_at": utcnow(),
            "source_summary": source_info,
        }
        artifact_info = emit_artifact(out_path, payload)
        payload.update(artifact_info)

        report_path = out_path.parent / "quantize_result.json"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Copy small marker into output dir to mimic external pipeline side-effects.
        marker = out_path.parent / "COMPRESSION_OK"
        marker.write_text(f"ok:{utcnow()}", encoding="utf-8")

        print(json.dumps({"status": "completed", "report_path": str(report_path), "artifact": artifact_info}))
        return 0
    except Exception as e:
        print(json.dumps({"status": "failed", "error": str(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
