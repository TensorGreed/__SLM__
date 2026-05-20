"""Generate per-section narration WAV files for Video 06 — BYO Custom Samples.

Demonstrates the "bring your own data" path. Creates a fresh
non-demo project, uploads a tiny coffee-shop-FAQ CSV, and lands on
the data tab with the imported rows visible.

Outputs:
  tts/audio/v06-section-<NN>-<slug>.wav    — per-section audio
  tts/audio/v06-durations.json             — per-section durations
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
        "01-cold-open",
        "cold_open",
        "Bring your own data. The three sample projects are useful "
        "for learning the platform, but the real point is to use "
        "your own dataset. This video shows the shortest path "
        "from a CSV on your laptop to rows on the Data tab.",
    ),
    (
        "02-new-project",
        "new_project",
        "From the project list, click New Project. Give it a name "
        "and a one-line description. The starter pack is optional — "
        "the platform has reasonable defaults for the common task "
        "shapes. Click Create.",
    ),
    (
        "03-empty-data-tab",
        "empty_data_tab",
        "Project created. We land on the Data tab. Empty. No "
        "rows, no documents, no gold set. This is what a fresh "
        "project looks like before any data is imported.",
    ),
    (
        "04-upload-csv",
        "upload_csv",
        "The Data tab's upload zone accepts files directly from "
        "the local filesystem. Supported formats include CSV, "
        "JSON, JSONL, PDF, DOCX, plain text, and Markdown. For "
        "this video the file is a tiny six-row coffee-shop FAQ "
        "in CSV form. Drop it on the upload zone or pick it via "
        "the file picker.",
    ),
    (
        "05-rows-imported",
        "rows_imported",
        "And the rows are in. Six raw documents, one per CSV row, "
        "each one a question and an answer. The platform "
        "auto-detected the columns and matched them to the standard "
        "Q-and-A shape. From here the project goes through the "
        "same pipeline you saw on the seeded samples: clean, gold, "
        "synthetic, prep, train, eval, compress, serve.",
    ),
    (
        "06-wrap",
        "wrap",
        "That's the bring-your-own-data path. The rest of the "
        "lifecycle for a custom project is identical to what the "
        "demo videos walked. Same tabs, same artifacts, same "
        "training and serving pipeline. The only difference is "
        "the row at the start.",
    ),
]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    durations: dict[str, float] = {}
    total_inference = 0.0
    for slug, key, text in SECTIONS:
        out = OUT_DIR / f"v06-section-{slug}.wav"
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
    (OUT_DIR / "v06-durations.json").write_text(json.dumps(durations, indent=2))
    print()
    print(f"Total audio: {durations['__total_audio__']:.1f}s")
    print(f"Total inference: {total_inference:.1f}s")


if __name__ == "__main__":
    main()
