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
        "Welcome to the Sentiment Classifier pipeline walkthrough. "
        "This is the simplest of the three samples — three-way "
        "classification with the labels positive, neutral, and "
        "negative. Thirty source rows, perfectly balanced ten, ten, "
        "ten. Each row is text and a single label.",
    ),
    (
        "02-data",
        "data",
        "Data tab. Thirty source reviews. Each row has two columns: "
        "the text, and the gold label. Expand one. You see exactly "
        "the shape the model has to learn — read a review, emit one "
        "of three labels. The balance matters: ten of each class "
        "means the model never gets to cheat by always predicting "
        "the majority.",
    ),
    (
        "03-goldset",
        "goldset",
        "Gold Set. Two hundred entries. The distribution is "
        "seventy positive, sixty-five neutral, sixty-five negative. "
        "Slightly skewed positive — typical of real-world reviews. "
        "The eval handler measures per-class precision and recall "
        "against this gold, so under-represented classes still get "
        "measured.",
    ),
    (
        "04-dataprep",
        "dataprep",
        "Dataset Prep. Schema Profile shows the three labels. The "
        "adapter is classification-label — it canonicalizes every "
        "prepared row to a text column and a label column. Splits "
        "are twenty-two train, four validation, four test. Small, "
        "but enough to verify the loop end to end.",
    ),
    (
        "05-tokenization",
        "tokenization",
        "Tokenization. Same idea as the previous samples. The "
        "twist for this sample: target profile is mobile CPU, so "
        "max sequence length matters more than usual. Short "
        "sequences mean a smaller model footprint and faster "
        "inference on-device.",
    ),
    (
        "06-training-config",
        "training_config",
        "Training tab — empty, expected. Into the Training Config "
        "page. Flip to Advanced. The Training Config picks up the "
        "mobile CPU target profile from the manifest — that hints "
        "at smaller batches, shorter sequences, and a tighter "
        "model footprint on export. Defaults are tuned for mobile.",
    ),
    (
        "07-eval",
        "eval",
        "Evaluation tab. Empty until we have an experiment. For "
        "this sample the eval pack is the classification default — "
        "accuracy and macro-F1 in the headline, per-class "
        "precision and recall in the detail panel.",
    ),
    (
        "08-compression-export",
        "compression_export",
        "Compression and Export. The natural target for this "
        "sample is ONNX with eight-bit quantization, which would "
        "give us a fast on-device model. ONNX is in the export "
        "format list, but the end-to-end story for this sample "
        "isn't validated yet — that's Video Eleven. For now we're "
        "just confirming the shape of the export surface.",
    ),
    (
        "09-wrap",
        "wrap",
        "And that's the third sample. Three task profiles, three "
        "scoring contracts, one shared pipeline. Quickstart, "
        "support FAQ, PII detector, sentiment classifier — that's "
        "the inspection arc complete. Next videos pick up the "
        "runtime-heavy side: actually launching a training run, "
        "scoring against gold, compressing, and serving.",
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
