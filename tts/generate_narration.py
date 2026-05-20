"""Generate per-section narration WAV files for Video 02.

Sections map 1:1 to checkpoints in
tests/demo-recordings/02-brewslm-quickstart.spec.ts. Each section's
spoken text is sent to the local Orpheus-FastAPI server (port 5005,
backed by Ollama on 11434). Output:

  tts/audio/v02-section-<N>-<slug>.wav    — per-section audio
  tts/audio/v02-durations.json            — duration per section + total
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import requests

TTS_URL = "http://127.0.0.1:5005/v1/audio/speech"
VOICE = "leo"  # Male voice; switched from "dan" 2026-05-20 at user request
OUT_DIR = Path(__file__).parent / "audio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Each section's spoken text comes verbatim from
# docs-demo/scripts/narration/02-brewslm-quickstart.md so any edits
# to that file should be reflected here. Section names match the
# 5-section arc of the Playwright spec.
SECTIONS: list[tuple[str, str, str]] = [
    (
        "01-cold-open",
        "cold_open",
        "We're taking a fresh local BrewSLM install and getting from "
        "nothing to a fully-seeded demo project, in under five minutes. "
        "No training, no data prep — just login, click, and inspect.",
    ),
    (
        "02-login",
        "login",
        "BrewSLM auth is on by default. I'll log in as admin — the "
        "bootstrap user. The password is the local development token "
        "from the backend env file. And we're in.",
    ),
    (
        "03-tiles",
        "tiles",
        "This is the project list. Up top, three official demo tiles: "
        "Support FAQ, PII Detector, and Sentiment Classifier. Each is "
        "backed by a manifest, a source file, and a two-hundred-row "
        "gold set. Clicking a tile seeds a complete project — raw "
        "data imported, gold imported, and train, validation, and "
        "test splits already written.",
    ),
    (
        "04-seed",
        "seed",
        "I'll click Support FAQ. Behind the scenes, the backend copies "
        "the source file into the project's raw data, creates twenty "
        "raw documents, imports two hundred gold rows, and pre-writes "
        "a sixteen, two, two train, validation, test split. A few "
        "seconds later we land on the Data tab. Notice the pipeline "
        "status badge — already at training stage, sixty percent "
        "complete. The seed did the upstream work for us. We haven't "
        "trained anything yet, but we have a ready-to-train project.",
    ),
    (
        "05-cleaning",
        "cleaning",
        "Let's walk the pipeline tabs. Ten of them, left to right, in "
        "the order you'd touch them. Cleaning — chunk text, redact "
        "personal information, score quality. Nothing's running here. "
        "The support FAQ corpus is already small and clean.",
    ),
    (
        "06-goldset",
        "goldset",
        "Gold Set. Two hundred entries, locked. The evaluation ground "
        "truth. The manifest says six, but the file has two hundred. "
        "The file wins.",
    ),
    (
        "07-dataprep",
        "dataprep",
        "Dataset Prep. The adapter is applied — question and answer "
        "pair — turning each row into a question and a matching "
        "answer. Splits are already written: sixteen train, two "
        "validation, two test.",
    ),
    (
        "08-training",
        "training",
        "Training tab. No experiments yet. We haven't started "
        "anything. Launching a run is Video Nine.",
    ),
    (
        "09-expand-wrap",
        "expand_wrap",
        "Back to the Data tab. One more thing before we wrap. Each "
        "raw document is one support ticket from the source file. "
        "Expand a row and you see the question and the agent's "
        "answer. This is the data we'll fine-tune the model against. "
        "The model learns to write answers like this for questions "
        "it's never seen. Done. From a fresh install to a seeded "
        "project with twenty raw rows, two hundred gold rows, and "
        "ready-to-train splits. Next video walks the dataset "
        "lifecycle in detail.",
    ),
]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    durations: dict[str, float] = {}
    total_inference = 0.0
    for slug, key, text in SECTIONS:
        out = OUT_DIR / f"v02-section-{slug}.wav"
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

    (OUT_DIR / "v02-durations.json").write_text(
        json.dumps(durations, indent=2)
    )
    print()
    print(f"Total audio: {durations['__total_audio__']:.1f}s")
    print(f"Total inference: {total_inference:.1f}s")


if __name__ == "__main__":
    main()
