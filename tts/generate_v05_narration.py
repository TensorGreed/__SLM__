"""Generate per-section narration WAV files for Video 05 — Sentiment.

Same Orpheus-FastAPI/Ollama setup as Videos 02/03/04. Two distinctive
beats for this sample:

1. Class balance — source is 10/10/10 across positive/neutral/
   negative; gold is 70/65/65. The balance is the teaching beat.
2. `mobile_cpu` target — hints at an ONNX-INT8 export path. The
   compression + export tabs get a light tour, but the ONNX story is
   explicitly marked as "natural target, not yet validated."

Outputs (tts/audio/):
  v05-section-<NN>-<slug>.wav   — per-section audio
  v05-durations.json            — duration per section + total (the spec reads it)
  concat-v05.txt                — ffmpeg concat list for the mux

Usage:
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v05_narration.py
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v05_narration.py --only <section-slug>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.narration_tts import VOICE, generate_video, parse_only  # noqa: E402

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


def main() -> None:
    generate_video("v05", SECTIONS, voice=VOICE, only=parse_only(sys.argv[1:]))


if __name__ == "__main__":
    main()
