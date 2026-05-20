"""Generate per-section narration WAV files for Video 04 — PII Detector.

Same Orpheus-FastAPI/Ollama setup as Videos 02/03. The key beat is
Section 3 (Cleaning) — the on-screen disambiguation between
cleaning-time PII redaction (regex) and the PII Detector model task
(span extraction). Two product features share the name "PII"; the
narration has to make the contrast obvious.

Outputs:
  tts/audio/v04-section-<NN>-<slug>.wav    — per-section audio
  tts/audio/v04-durations.json             — per-section durations
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import requests

TTS_URL = "http://127.0.0.1:5005/v1/audio/speech"
VOICE = "leo"  # series consistency
OUT_DIR = Path(__file__).parent / "audio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTIONS: list[tuple[str, str, str]] = [
    (
        "01-cold-open",
        "cold_open",
        "Welcome to the PII Detector pipeline walkthrough. This "
        "sample is a span-level entity detector — you feed it a "
        "snippet of text, and it emits a structured list of every "
        "personal-information span it finds. Email, phone, social "
        "security, credit card — ten entity types in total. There's "
        "a confusion risk we'll clear up in a minute.",
    ),
    (
        "02-data",
        "data",
        "Data tab. Sixty-one source rows. Each row has two columns: "
        "the text, and a structured list of every entity in that "
        "text. The entity list is the ground truth — that's the "
        "shape the model has to learn to produce. Expand a row to "
        "see what one looks like.",
    ),
    (
        "03-cleaning",
        "cleaning",
        "Now the confusing part. The Cleaning tab has a "
        "personal-information redaction option. That's a regex "
        "pre-processing step — it can mask personal information in "
        "source text before training. The PII Detector model in "
        "this sample is the opposite — it finds personal "
        "information and emits a structured list. Same word in the "
        "product name, two completely different features. We're not "
        "running the redaction here.",
    ),
    (
        "04-goldset",
        "goldset",
        "Gold Set. Two hundred entries. Each one has a snippet, an "
        "expected entity list, and a rationale. The eval handler "
        "scores the model's predicted entities against the gold "
        "entities, per entity type.",
    ),
    (
        "05-synthetic",
        "synthetic",
        "Synthetic. The lever to grow sixty-one source rows into "
        "two thousand training rows. For this sample, the generator "
        "runs in span mode — you'd list the ten entity types you "
        "care about, and the teacher model generates new text with "
        "matching entity annotations. Local Ollama is wired up, but "
        "we're not running generation here. It's runtime-heavy and "
        "lives in its own walkthrough.",
    ),
    (
        "06-dataprep",
        "dataprep",
        "Dataset Prep. The schema profile shows the scoring mode — "
        "span set. That's the contract with eval: the model has to "
        "emit a structured output with an entities array, and eval "
        "scores per entity type. Splits are forty-five train, eight "
        "validation, eight test.",
    ),
    (
        "07-tokenization",
        "tokenization",
        "Tokenization. Same idea as the previous sample. Reports "
        "per-row token counts and the maximum sequence length "
        "you'd budget for. Surface only for this video.",
    ),
    (
        "08-training-config",
        "training_config",
        "Training tab — empty, expected. Into the Training Config "
        "page. Defaults to Essentials. Flip to Advanced. For span "
        "extraction tasks the docs recommend bumping low-rank "
        "adaptation from rank eight to rank sixteen, and targeting "
        "all four attention projections instead of two. The "
        "Advanced view exposes those controls. Defaults work if "
        "you're starting out.",
    ),
    (
        "09-eval-wrap",
        "eval_wrap",
        "Evaluation tab. Empty until we have a finished experiment. "
        "For this sample the eval handler scores per entity type — "
        "precision and recall for email, phone, social security, "
        "and the other seven. Next video: the sentiment classifier "
        "sample. Different task profile, different scoring mode.",
    ),
]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    durations: dict[str, float] = {}
    total_inference = 0.0
    for slug, key, text in SECTIONS:
        out = OUT_DIR / f"v04-section-{slug}.wav"
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
    (OUT_DIR / "v04-durations.json").write_text(json.dumps(durations, indent=2))
    print()
    print(f"Total audio: {durations['__total_audio__']:.1f}s")
    print(f"Total inference: {total_inference:.1f}s")


if __name__ == "__main__":
    main()
