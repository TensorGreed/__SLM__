"""Generate per-section narration WAV files for Video 03.

Sections map 1:1 to checkpoints in
tests/demo-recordings/03-support-faq-pipeline-narrated.spec.ts.
Same Orpheus-FastAPI server on port 5005 as Video 02; just a different
section list and different output file names.

Outputs:
  tts/audio/v03-section-<NN>-<slug>.wav    — per-section audio
  tts/audio/v03-durations.json             — per-section durations
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import requests

TTS_URL = "http://127.0.0.1:5005/v1/audio/speech"
VOICE = "leo"  # matches Video 02; consistent narrator across the series
OUT_DIR = Path(__file__).parent / "audio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Edit this list to change spoken narration. Keep the tone rules from
# tts/README.md: no literal API keys, no env var names, no REST paths,
# no file extensions, no adapter literal strings. The on-screen action
# shows those anyway.
SECTIONS: list[tuple[str, str, str]] = [
    (
        "01-cold-open",
        "cold_open",
        "Welcome to the Support FAQ pipeline walkthrough. We're "
        "taking the simplest of the three official samples — twenty "
        "customer tickets with hand-written answers — and walking it "
        "through every pipeline tab that does something useful on a "
        "seeded demo. No training, no synthetic generation, just "
        "inspection.",
    ),
    (
        "02-data",
        "data",
        "Data tab. Twenty raw documents — one per source ticket. The "
        "seeder turned each row into a raw document record. Expand "
        "one. You see the shape: a question and an answer. This is "
        "what the model has to learn — the agent's writing style for "
        "these specific questions. Imagine pasting thousands of "
        "resolved tickets here and you've got the dataset for a real "
        "support assistant.",
    ),
    (
        "03-cleaning",
        "cleaning",
        "Cleaning. Skip it for this sample — the corpus is already "
        "small and clean. For a messy real-world corpus, this is "
        "where you'd chunk long text, redact personal information, "
        "mask toxicity, and score quality. Same word as the next "
        "sample's PII Detector, but two completely different "
        "features — cleaning here is a regex pre-processing step, "
        "the detector is a trained model.",
    ),
    (
        "04-goldset",
        "goldset",
        "Gold Set. Two hundred entries. Locked. This is the "
        "evaluation ground truth — never trained against, only "
        "measured against. Each row has a question, an expected "
        "answer, and a rationale. The eval handler walks the entire "
        "two-hundred-row set after training and reports the fraction "
        "the model got right.",
    ),
    (
        "05-synthetic",
        "synthetic",
        "Synthetic. The lever that scales twenty source rows into "
        "two thousand training rows. It runs a teacher model — local "
        "Ollama on this machine — over your cleaned corpus, asking "
        "the teacher to generate matching question and answer pairs. "
        "Video Four is the full walkthrough; we're not running it "
        "here.",
    ),
    (
        "06-dataprep",
        "dataprep",
        "Dataset Prep. This is where the contract gets made. The "
        "adapter applied — question and answer pair — turns each "
        "row into the shape the trainer expects. Splits are already "
        "written: sixteen train, two validation, two test. That's "
        "the deterministic seventy-fifteen-fifteen split with a "
        "two-row floor on validation and test.",
    ),
    (
        "07-tokenization",
        "tokenization",
        "Tokenization. Runs a tokenizer over the prepared splits "
        "and reports the length distribution — how many tokens per "
        "row, what maximum sequence length you'd budget for. The "
        "actual analysis needs a tokenizer download, which is its "
        "own setup. Surface only for this video.",
    ),
    (
        "08-training-config",
        "training_config",
        "Training tab. No experiments yet — normal. Jumping into "
        "the Training Config page. Essentials view by default — "
        "base model, training mode, epochs, batch size, learning "
        "rate. Flip to Advanced and you unlock the parameter "
        "controls: low-rank adaptation rank, target modules, "
        "optimizer choice. The defaults work; the controls are "
        "there when you need them. Launching a run is Video Nine.",
    ),
    (
        "09-eval-wrap",
        "eval_wrap",
        "Evaluation tab. Empty until we have a finished experiment. "
        "This is where accuracy, F1, gate pass and fail, and "
        "side-by-side predictions would land. That's the Support "
        "FAQ tour. We touched ten tabs without running anything "
        "heavy. Next video walks the same shape for the PII "
        "Detector sample.",
    ),
]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    durations: dict[str, float] = {}
    total_inference = 0.0
    for slug, key, text in SECTIONS:
        out = OUT_DIR / f"v03-section-{slug}.wav"
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
    (OUT_DIR / "v03-durations.json").write_text(json.dumps(durations, indent=2))
    print()
    print(f"Total audio: {durations['__total_audio__']:.1f}s")
    print(f"Total inference: {total_inference:.1f}s")


if __name__ == "__main__":
    main()
