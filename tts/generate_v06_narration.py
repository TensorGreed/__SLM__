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
        "Bring your own data. The sample projects are useful for "
        "learning the platform, but the real point is to use your own "
        "dataset. This video shows the shortest path from a CSV on your "
        "laptop to rows listed under Data, Sources.",
    ),
    (
        "02-new-project",
        "new_project",
        "From the project list, click New Project. Give it a name, and "
        "say in a sentence or two what you want the model to do. The "
        "platform reads that and sets sensible defaults — base model, "
        "task shape, output schema — and you can change any of it "
        "later. Click Create project, and you land on the project's "
        "Overview.",
    ),
    (
        "03-empty-data-tab",
        "empty_data_tab",
        "Project created. Open Data. The first step is Sources, and "
        "it's empty. No documents, no rows, no answer key. This is what "
        "a fresh project looks like before any data is imported. The "
        "upload zone is right there.",
    ),
    (
        "04-upload-csv",
        "upload_csv",
        "The upload zone accepts files straight from your computer. "
        "Documents — PDF, Word, HTML, text, Markdown — and row files — "
        "CSV, TSV, Excel, JSON, JSONL, Parquet — in any encoding. For "
        "this video the file is a tiny six-row coffee-shop FAQ in CSV "
        "form. Drop it on the zone or choose it with the file picker.",
    ),
    (
        "05-rows-imported",
        "rows_imported",
        "And the rows are in. Under ingested documents, the file is "
        "listed, and expanding it shows its question and answer "
        "columns, kept as columns all the way to training. From here "
        "the project goes through the same steps you saw on the "
        "samples: prepare, labels, generate, split, then train, results "
        "and ship.",
    ),
    (
        "06-wrap",
        "wrap",
        "That's the bring-your-own-data path. The Overview now shows "
        "where this project is and what the next step is. The rest of "
        "the workflow for a custom project is identical to what the "
        "sample videos walked. Same workspace, same artifacts, same "
        "training and serving. The only difference is the file at the "
        "start.",
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
