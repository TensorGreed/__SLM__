#!/usr/bin/env python3
"""External quantization runtime with real GGUF/ONNX support.

Supported modes:
- GGUF quantization (real): HF model dir/id -> FP16 GGUF -> quantized GGUF via llama.cpp
- ONNX export + INT8 quantization (real): HF model dir/id -> ONNX -> INT8 ONNX via onnxruntime
- Merge adapter (best-effort): base model + LoRA adapter -> merged HF directory

This script is executed by queued compression jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

# Common GGUF quantization presets in llama.cpp.
BITS_TO_GGUF_TYPE: dict[int, str] = {
    2: "Q2_K",
    3: "Q3_K_M",
    4: "Q4_K_M",
    5: "Q5_K_M",
    6: "Q6_K",
    8: "Q8_0",
    16: "F16",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external SLM quantization workflow")
    parser.add_argument("--project", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, help="Source model file/dir or HF model id")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bit width (2/3/4/5/6/8/16)")
    parser.add_argument("--format", type=str, default="gguf", help="Target format (gguf|onnx|merged)")
    parser.add_argument("--out", type=str, required=True, help="Output artifact path")
    parser.add_argument("--adapter", type=str, default="", help="Optional LoRA adapter path for merge workflows")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_source(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"type": "file", "files": 1, "bytes": path.stat().st_size}
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        return {"type": "directory", "files": len(files), "bytes": sum(p.stat().st_size for p in files)}
    raise FileNotFoundError(f"Model path not found: {path}")


def print_json_event(event: str, payload: dict[str, Any]) -> None:
    data = {"event": event}
    data.update(payload)
    print(json.dumps(data), flush=True)


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(command)
    print(f"[quantize] running: {printable}", flush=True)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip()
        if text:
            print(text, flush=True)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"Command failed with exit code {returncode}: {printable}")


def resolve_python_executable() -> str:
    explicit = os.getenv("PYTHON_EXECUTABLE", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists():
            return str(path)
    return str(Path(sys.executable).resolve())


def _resolve_explicit_path(env_key: str) -> Path | None:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    return path if path.exists() else None


def _resolve_llama_cpp_dir() -> Path | None:
    env_path = _resolve_explicit_path("LLAMA_CPP_DIR")
    if env_path is not None:
        return env_path
    candidates = [
        REPO_ROOT / "llama.cpp",
        REPO_ROOT.parent / "llama.cpp",
        REPO_ROOT / "third_party" / "llama.cpp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_convert_script() -> Path:
    explicit = _resolve_explicit_path("LLAMA_CPP_CONVERT_SCRIPT")
    if explicit is not None and explicit.is_file():
        return explicit

    llama_dir = _resolve_llama_cpp_dir()
    candidates: list[Path] = []
    if llama_dir is not None:
        candidates.extend(
            [
                llama_dir / "convert_hf_to_gguf.py",
                llama_dir / "scripts" / "convert_hf_to_gguf.py",
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate convert_hf_to_gguf.py. Set LLAMA_CPP_DIR or LLAMA_CPP_CONVERT_SCRIPT."
    )


def resolve_quantize_binary() -> Path:
    explicit = _resolve_explicit_path("LLAMA_CPP_QUANTIZE_BIN")
    if explicit is not None and explicit.is_file():
        return explicit

    llama_dir = _resolve_llama_cpp_dir()
    candidates: list[Path] = []
    if llama_dir is not None:
        candidates.extend(
            [
                llama_dir / "build" / "bin" / "llama-quantize",
                llama_dir / "build" / "bin" / "llama_quantize",
                llama_dir / "build" / "bin" / "quantize",
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    for name in ("llama-quantize", "llama_quantize", "quantize"):
        found = shutil.which(name)
        if found:
            return Path(found).resolve()

    raise FileNotFoundError(
        "Could not locate llama quantize binary. Set LLAMA_CPP_DIR or LLAMA_CPP_QUANTIZE_BIN."
    )


def resolve_model_source(model_arg: str) -> tuple[Path, str]:
    candidate = Path(model_arg).expanduser()
    if candidate.exists():
        return candidate.resolve(), "local_path"

    # Optional: allow Hugging Face repo id directly.
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise FileNotFoundError(
            f"Model path '{model_arg}' not found locally. Install huggingface_hub to use HF repo ids."
        ) from e

    print_json_event("download_start", {"model_id": model_arg})
    downloaded = snapshot_download(repo_id=model_arg)
    resolved = Path(downloaded).expanduser().resolve()
    print_json_event("download_complete", {"model_id": model_arg, "path": str(resolved)})
    return resolved, "huggingface_snapshot"


def resolve_onnx_export_task() -> str | None:
    task = os.getenv("ONNX_EXPORT_TASK", "").strip()
    if not task or task.lower() == "auto":
        return None
    return task


def _export_onnx_with_optimum(
    model_source_dir: Path,
    output_dir: Path,
    task: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    first_error: Exception | None = None

    try:
        from optimum.exporters.onnx import main_export

        kwargs: dict[str, Any] = {
            "model_name_or_path": str(model_source_dir),
            "output": output_dir,
        }
        if task:
            kwargs["task"] = task
        main_export(**kwargs)
        return
    except Exception as e:
        first_error = e

    optimum_cli = shutil.which("optimum-cli")
    if optimum_cli:
        cmd = [optimum_cli, "export", "onnx", "--model", str(model_source_dir), str(output_dir)]
        if task:
            cmd.extend(["--task", task])
        run_command(cmd)
        return

    raise RuntimeError(
        "ONNX export failed. Install optimum + onnx + onnxruntime, or provide optimum-cli."
    ) from first_error


def _pick_primary_onnx_file(onnx_root: Path) -> Path:
    candidates = [p for p in onnx_root.rglob("*.onnx") if p.is_file()]
    if not candidates:
        raise RuntimeError(f"No .onnx file produced in {onnx_root}")

    for preferred in ("model.onnx", "decoder_model.onnx"):
        for candidate in candidates:
            if candidate.name == preferred:
                return candidate

    return max(candidates, key=lambda p: p.stat().st_size)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def quantize_onnx(
    *,
    model_source: Path,
    out_path: Path,
    bits: int,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    if bits != 8:
        raise ValueError("ONNX path currently supports bits=8 only (dynamic INT8 quantization).")

    source_for_export = model_source
    out_path.parent.mkdir(parents=True, exist_ok=True)
    support_files: list[str] = []

    with tempfile.TemporaryDirectory(prefix="slm-onnx-") as tmp:
        tmp_dir = Path(tmp)

        if adapter_path is not None:
            merged_dir = tmp_dir / "merged-hf"
            merge_lora_adapter(model_source, adapter_path, merged_dir)
            source_for_export = merged_dir

        if source_for_export.is_file() and source_for_export.suffix.lower() == ".onnx":
            source_onnx = source_for_export
            print_json_event("onnx_export_skip", {"reason": "input_already_onnx", "path": str(source_onnx)})
        else:
            if not source_for_export.is_dir():
                raise ValueError(
                    f"ONNX export expects a model directory or .onnx file. Got: {source_for_export}"
                )
            task = resolve_onnx_export_task()
            raw_onnx_dir = tmp_dir / "raw-onnx"
            print_json_event(
                "onnx_export_start",
                {
                    "model_source": str(source_for_export),
                    "output_dir": str(raw_onnx_dir),
                    "task": task or "auto",
                },
            )
            _export_onnx_with_optimum(source_for_export, raw_onnx_dir, task)
            source_onnx = _pick_primary_onnx_file(raw_onnx_dir)
            print_json_event("onnx_export_complete", {"onnx_file": str(source_onnx)})

        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except Exception as e:
            raise RuntimeError("onnxruntime quantization is not available. Install onnxruntime.") from e

        print_json_event(
            "onnx_quantize_start",
            {
                "input_onnx": str(source_onnx),
                "output_onnx": str(out_path),
                "mode": "dynamic_int8",
            },
        )
        quantize_dynamic(
            model_input=str(source_onnx),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
        )
        print_json_event("onnx_quantize_complete", {"output_bytes": out_path.stat().st_size})

        # Copy ONNX external data and tokenizer/config sidecars if present.
        for sibling in source_onnx.parent.iterdir():
            if not sibling.is_file():
                continue
            if sibling.resolve() == source_onnx.resolve():
                continue
            dst = out_path.parent / sibling.name
            if _copy_if_exists(sibling, dst):
                support_files.append(str(dst))

        if source_for_export.is_dir():
            metadata_names = [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "vocab.txt",
                "merges.txt",
                "added_tokens.json",
                "preprocessor_config.json",
            ]
            for name in metadata_names:
                src = source_for_export / name
                dst = out_path.parent / name
                if _copy_if_exists(src, dst):
                    support_files.append(str(dst))

    if not out_path.exists():
        raise RuntimeError(f"Quantized ONNX not created at expected path: {out_path}")

    return {
        "artifact_path": str(out_path),
        "artifact_kind": "file",
        "artifact_bytes": out_path.stat().st_size,
        "artifact_sha256": sha256_file(out_path),
        "quant_type": "QInt8",
        "support_files": sorted(set(support_files)),
    }


def merge_lora_adapter(base_model_dir: Path, adapter_path: Path, out_dir: Path) -> dict[str, Any]:
    if not base_model_dir.exists() or not base_model_dir.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print_json_event(
        "merge_start",
        {"base_model": str(base_model_dir), "adapter": str(adapter_path), "output_dir": str(out_dir)},
    )
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError("LoRA merge requires transformers + peft in the runtime environment.") from e

    model = AutoModelForCausalLM.from_pretrained(str(base_model_dir), trust_remote_code=True)
    peft_model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(out_dir))

    files = [p for p in out_dir.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    print_json_event("merge_complete", {"files": len(files), "bytes": total_bytes})
    return {
        "artifact_path": str(out_dir),
        "artifact_kind": "directory",
        "artifact_files": len(files),
        "artifact_bytes": total_bytes,
    }


def quantize_gguf(
    *,
    model_source: Path,
    out_path: Path,
    bits: int,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    if bits not in BITS_TO_GGUF_TYPE:
        allowed = sorted(BITS_TO_GGUF_TYPE.keys())
        raise ValueError(f"Unsupported bit-width '{bits}'. Supported values: {allowed}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quant_type = BITS_TO_GGUF_TYPE[bits]
    source_for_conversion = model_source

    with tempfile.TemporaryDirectory(prefix="slm-gguf-") as tmp:
        tmp_dir = Path(tmp)

        if adapter_path is not None:
            merged_dir = tmp_dir / "merged-hf"
            merge_lora_adapter(model_source, adapter_path, merged_dir)
            source_for_conversion = merged_dir

        # If source is already GGUF, skip conversion.
        if source_for_conversion.is_file() and source_for_conversion.suffix.lower() == ".gguf":
            fp16_gguf = source_for_conversion
            print_json_event("convert_skip", {"reason": "input_already_gguf", "path": str(fp16_gguf)})
        else:
            if not source_for_conversion.is_dir():
                raise ValueError(
                    f"GGUF conversion expects a model directory or .gguf file. Got: {source_for_conversion}"
                )
            convert_script = resolve_convert_script()
            fp16_gguf = tmp_dir / "model-f16.gguf"
            print_json_event(
                "convert_start",
                {
                    "convert_script": str(convert_script),
                    "source_model_dir": str(source_for_conversion),
                    "fp16_output": str(fp16_gguf),
                },
            )
            run_command(
                [
                    resolve_python_executable(),
                    str(convert_script),
                    str(source_for_conversion),
                    "--outfile",
                    str(fp16_gguf),
                    "--outtype",
                    "f16",
                ]
            )
            if not fp16_gguf.exists():
                raise RuntimeError(f"Expected converted FP16 GGUF not found: {fp16_gguf}")
            print_json_event("convert_complete", {"fp16_gguf_bytes": fp16_gguf.stat().st_size})

        if bits == 16:
            if fp16_gguf.resolve() != out_path.resolve():
                shutil.copy2(fp16_gguf, out_path)
            print_json_event("quantize_skip", {"reason": "bits_16_copy", "output": str(out_path)})
        else:
            quantize_bin = resolve_quantize_binary()
            print_json_event(
                "quantize_start",
                {
                    "quantize_binary": str(quantize_bin),
                    "input_gguf": str(fp16_gguf),
                    "output_gguf": str(out_path),
                    "quant_type": quant_type,
                },
            )
            run_command([str(quantize_bin), str(fp16_gguf), str(out_path), quant_type])
            print_json_event("quantize_complete", {"output_bytes": out_path.stat().st_size})

    if not out_path.exists():
        raise RuntimeError(f"Quantized GGUF not created at expected path: {out_path}")

    return {
        "artifact_path": str(out_path),
        "artifact_kind": "file",
        "artifact_bytes": out_path.stat().st_size,
        "artifact_sha256": sha256_file(out_path),
        "quant_type": quant_type,
    }


def write_report(out_path: Path, payload: dict[str, Any]) -> Path:
    report_path = out_path.parent / "quantize_result.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    marker = out_path.parent / "COMPRESSION_OK"
    marker.write_text(f"ok:{utcnow()}", encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    try:
        format_name = str(args.format or "gguf").strip().lower()
        out_path = Path(args.out).expanduser().resolve()
        adapter_path = Path(args.adapter).expanduser().resolve() if args.adapter else None
        model_source, source_origin = resolve_model_source(args.model)
        source_summary = summarize_source(model_source)

        print_json_event(
            "runtime_preflight",
            {
                "project_id": args.project,
                "model_source": str(model_source),
                "source_origin": source_origin,
                "format": format_name,
                "bits": args.bits,
                "adapter": str(adapter_path) if adapter_path else None,
            },
        )

        if format_name == "merged":
            if adapter_path is None:
                raise ValueError("merge mode requires --adapter path")
            artifact_info = merge_lora_adapter(model_source, adapter_path, out_path)
        elif format_name == "gguf":
            artifact_info = quantize_gguf(
                model_source=model_source,
                out_path=out_path,
                bits=int(args.bits),
                adapter_path=adapter_path,
            )
        elif format_name == "onnx":
            artifact_info = quantize_onnx(
                model_source=model_source,
                out_path=out_path,
                bits=int(args.bits),
                adapter_path=adapter_path,
            )
        else:
            raise ValueError(
                f"Unsupported format '{args.format}'. Supported formats in this runtime: gguf, onnx, merged."
            )

        payload = {
            "project_id": args.project,
            "source_model": str(model_source),
            "source_origin": source_origin,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "quantization": f"{args.bits}-bit",
            "format": format_name,
            "created_at": utcnow(),
            "source_summary": source_summary,
        }
        payload.update(artifact_info)
        report_path = write_report(out_path if out_path.suffix else out_path / "artifact.bin", payload)

        print(json.dumps({"status": "completed", "report_path": str(report_path), "artifact": artifact_info}), flush=True)
        return 0
    except Exception as e:
        print(json.dumps({"status": "failed", "error": str(e)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
