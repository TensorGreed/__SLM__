"""Generate per-section narration WAV files for Video 05 — Sentiment.

Same Orpheus-FastAPI/Ollama setup as Videos 02/03/04. Two distinctive
beats for this sample:

1. Class balance — source is 10/10/10 across positive/neutral/
   negative; gold is 70/65/65. The balance is the teaching beat.
2. `mobile_cpu` target — hints at an ONNX-INT8 export path. The
   compression + export tabs get a light tour, but the ONNX story is
   explicitly marked as "natural target, not yet validated."

Outputs:
  tts/audio/v05-section-<NN>-<slug>.wav    — per-section audio
  tts/audio/v05-durations.json             — per-section durations
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
        "Welcome to the Sentiment Classifier walkthrough. This is the "
        "simplest of the three samples — three-way classification with "
        "the labels positive, neutral, and negative. Thirty source "
        "rows, perfectly balanced ten, ten, ten. Each row is text and a "
        "single label. We start on the Overview, then visit Data, "
        "Train, Results and Ship.",
    ),
    (
        "02-data",
        "data",
        "Data, Sources. Thirty source reviews. Each row has two "
        "columns: the text, and the label. Expand one. You see exactly "
        "the shape the model has to learn — read a review, emit one of "
        "three labels. The balance matters: ten of each class means the "
        "model never gets to cheat by always predicting the majority.",
    ),
    (
        "03-goldset",
        "goldset",
        "Labels. The answer key has two hundred entries, and the mix "
        "summary shows the distribution — seventy positive, sixty-five "
        "neutral, sixty-five negative. Slightly skewed positive, "
        "typical of real-world reviews. Scoring measures per-class "
        "precision and recall against this answer key, so "
        "under-represented classes still get measured.",
    ),
    (
        "04-dataprep",
        "dataprep",
        "Split. The task shape is classification with a text column and "
        "a label column, and every prepared row is canonicalised to "
        "that. The split gives twenty-two training rows, four for "
        "validation and four test examples. Small, but enough to verify "
        "the loop end to end.",
    ),
    (
        "05-tokenization",
        "tokenization",
        "On to Train. Under More options, Tokenization. The twist for "
        "this sample: the target profile is mobile CPU, so maximum "
        "sequence length matters more than usual. Short sequences mean "
        "a smaller model footprint and faster inference on-device.",
    ),
    (
        "06-training-config",
        "training_config",
        "The Train form. Base model, task shape, seeds, readiness, and "
        "one Train button. Under More options, LoRA and advanced "
        "training. The project's mobile CPU target hints at smaller "
        "batches, shorter sequences, and a tighter model footprint on "
        "export. The defaults are tuned for it.",
    ),
    (
        "07-eval",
        "eval",
        "Results. Empty until we have a finished run. For this sample "
        "the checks are the classification defaults — accuracy and "
        "macro-F1 in the headline, per-class precision and recall in "
        "the detail. The Verdict tab is where they land.",
    ),
    (
        "08-compression-export",
        "compression_export",
        "Ship, Package. Under Options, compress first, then the export "
        "form. The natural target for this sample is ONNX with "
        "eight-bit quantization, which would give us a fast on-device "
        "model. ONNX is in the export format list, but the end-to-end "
        "story for this sample isn't validated yet — that's Video "
        "Eleven. For now we're just confirming the shape of the export "
        "surface.",
    ),
    (
        "09-wrap",
        "wrap",
        "And that's the third sample. Three task shapes, three scoring "
        "contracts, one shared workspace. Quickstart, support FAQ, PII "
        "detector, sentiment classifier — that's the inspection arc "
        "complete. Next videos pick up the runtime-heavy side: actually "
        "launching a training run, scoring against the answer key, "
        "compressing, and serving.",
    ),
]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    durations: dict[str, float] = {}
    total_inference = 0.0
    for slug, key, text in SECTIONS:
        out = OUT_DIR / f"v05-section-{slug}.wav"
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
    (OUT_DIR / "v05-durations.json").write_text(json.dumps(durations, indent=2))
    print()
    print(f"Total audio: {durations['__total_audio__']:.1f}s")
    print(f"Total inference: {total_inference:.1f}s")


if __name__ == "__main__":
    main()
