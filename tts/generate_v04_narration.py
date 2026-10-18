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
        "Welcome to the PII Detector walkthrough. This sample is a "
        "span-level entity detector — you give it a snippet of text, "
        "and it returns a structured list of every personal-information "
        "span it finds. Email, phone, social security, credit card — "
        "ten entity types in total. This is the Overview: where you "
        "are, the next step, and the four places we'll visit — Data, "
        "Train, Results and Ship. There's a confusion risk we'll clear "
        "up in a minute.",
    ),
    (
        "02-data",
        "data",
        "Data, Sources. Sixty-one source rows. Each row has two "
        "columns: the text, and a structured list of every entity in "
        "that text. The entity list is the ground truth — that's the "
        "shape the model has to learn to produce. Open the ingested "
        "documents and expand a row to see what one looks like.",
    ),
    (
        "03-cleaning",
        "cleaning",
        "Now the confusing part. Prepare, the cleaning step, has a "
        "personal-information redaction option. That's a regex "
        "pre-processing step — it can mask personal information in "
        "source text before training. The PII Detector model in this "
        "sample is the opposite — it finds personal information and "
        "returns a structured list. Same word in the product name, two "
        "completely different features. We're not running the redaction "
        "here.",
    ),
    (
        "04-goldset",
        "goldset",
        "Labels. This is the answer key — two hundred entries. Each one "
        "has a snippet, an expected entity list, and a rationale. Above "
        "it, the labelling conventions: the entity types and a one-line "
        "definition of each, so every row is labelled the same way. "
        "Scoring compares the model's predicted entities against the "
        "answer key, per entity type.",
    ),
    (
        "05-synthetic",
        "synthetic",
        "Generate. The lever to grow sixty-one source rows into two "
        "thousand training rows. For this sample, the generator runs in "
        "span mode — you list the ten entity types you care about, and "
        "the generator model writes new text with matching entity "
        "annotations. Local Ollama is wired up, but we're not running "
        "generation here. It's runtime-heavy and lives in its own "
        "walkthrough.",
    ),
    (
        "06-dataprep",
        "dataprep",
        "Split. The task shape here is span extraction, and that's the "
        "contract with scoring: the model has to return an entities "
        "array, and the result is scored per entity type. The split "
        "gives forty-five training rows, eight for validation and eight "
        "test examples. There's also an option to keep the current test "
        "examples when you add data, so the answer never leaks into "
        "training.",
    ),
    (
        "07-tokenization",
        "tokenization",
        "Now Train. Under More options, Tokenization. It reports "
        "per-row token counts and the maximum sequence length you'd "
        "budget for, so you can see truncation before you spend time "
        "training. Surface only for this video.",
    ),
    (
        "08-training-config",
        "training_config",
        "The Train form itself. The base model, the task shape — "
        "read-only here, span extraction — the number of seeds, a "
        "readiness line, and one Train button. Under More options, LoRA "
        "and advanced training. For span extraction the docs recommend "
        "raising low-rank adaptation from rank eight to rank sixteen, "
        "and targeting all four attention projections instead of two. "
        "Defaults work if you're starting out.",
    ),
    (
        "09-eval-wrap",
        "eval_wrap",
        "Results. Nothing to judge until a run has finished. The page "
        "has four tabs — Verdict, Failures, Compare and Try it. On "
        "Failures you'll find the built-in checks, and for this sample "
        "the scoring is per entity type — precision and recall for "
        "email, phone, social security, and the other seven. Next "
        "video: the sentiment classifier sample. Different task shape, "
        "different scoring.",
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
