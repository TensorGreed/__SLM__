"""Generate per-section narration WAV files for Video 14 — Architecture.

Companion slide video to V01. High-level platform architecture:
the components that make BrewSLM work and how they fit together.

Outputs:
  tts/audio/v14-section-<NN>-<slug>.wav    — per-section audio
  tts/audio/v14-durations.json             — per-section durations
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import requests

TTS_URL = "http://127.0.0.1:5005/v1/audio/speech"
VOICE = "leo"
OUT_DIR = Path(__file__).parent / "audio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTIONS: list[tuple[str, str, str]] = [
    (
        "01-title",
        "title",
        "Architecture. Five minutes on what makes this platform "
        "work under the hood, and why every piece runs on local "
        "hardware.",
    ),
    (
        "02-the-stack",
        "stack",
        "The stack. Five processes. A FastAPI backend handles the "
        "HTTP API and the database. A React frontend renders the "
        "workspace. A Celery worker runs the long jobs — training, "
        "compression, dataset import. Redis is the broker between "
        "FastAPI and Celery. Ollama serves three things: the "
        "teacher model for synthetic generation, the judge model "
        "for evaluation, and the trained models we ship.",
    ),
    (
        "03-data-flow",
        "data_flow",
        "Data flow. Raw rows come in through ingestion, get "
        "cleaned and chunked, get matched against a gold set, get "
        "expanded with synthetic samples if needed, get split into "
        "train, validation, and test, get tokenized, and then feed "
        "the training loop. Training produces a LoRA adapter. "
        "Evaluation scores it against the gold set. Compression "
        "merges and quantizes. Export packages. Ollama serves.",
    ),
    (
        "04-where-things-run",
        "where_things_run",
        "Where things run. The whole stack is local. No cloud "
        "dependencies. Training fires on the GPU through PyTorch. "
        "Inference for the teacher, judge, and final model all go "
        "through Ollama on the same GPU. Quantization shells out "
        "to llama dot c-p-p. Disk holds the artifacts. The "
        "frontend on port one one seventy three drives the whole "
        "thing through the backend's REST API on port eight "
        "thousand.",
    ),
    (
        "05-trust-boundaries",
        "trust_boundaries",
        "Trust boundaries. Two of them. The backend is the only "
        "thing that touches the database, the file system, and "
        "the model weights. Everything that goes through the "
        "frontend hits an authenticated REST endpoint. The Celery "
        "worker is inside the backend's trust boundary — it "
        "shares the same DB connection and the same data "
        "directory. Ollama runs in its own process; the backend "
        "treats it as an external service.",
    ),
    (
        "06-wrap",
        "wrap",
        "That's the architecture. One backend, one frontend, one "
        "worker, one broker, one inference runtime. Five "
        "processes, one machine, end to end. The other thirteen "
        "videos in this series demonstrate every piece of that "
        "loop running for real.",
    ),
]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    durations: dict[str, float] = {}
    total_inference = 0.0
    for slug, key, text in SECTIONS:
        out = OUT_DIR / f"v14-section-{slug}.wav"
        print(f"[{slug}] -> {len(text)} chars …", end="", flush=True)
        t0 = time.time()
        resp = requests.post(
            TTS_URL,
            json={
                "model": "orpheus",
                "input": text,
                "voice": VOICE,
                "response_format": "wav",
            },
            timeout=600,
        )
        elapsed = time.time() - t0
        total_inference += elapsed
        resp.raise_for_status()
        out.write_bytes(resp.content)
        dur = wav_duration_seconds(out)
        durations[key] = dur
        print(f" {dur:.2f}s audio in {elapsed:.1f}s wall")

    durations["__total_audio__"] = sum(
        v for k, v in durations.items() if not k.startswith("__")
    )
    durations["__total_inference_wall__"] = total_inference
    (OUT_DIR / "v14-durations.json").write_text(json.dumps(durations, indent=2))
    print()
    print(f"Total audio: {durations['__total_audio__']:.1f}s")
    print(f"Total inference: {total_inference:.1f}s")


if __name__ == "__main__":
    main()
