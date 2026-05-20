"""Generate per-section narration WAV files for Video 01 — SLM 101.

Slide-based conceptual intro. No product UI. Six slides, one per
narration section. The Playwright spec drives `window.showSlide(n)`
to advance, so the slide transitions stay deterministic.

Outputs:
  tts/audio/v01-section-<NN>-<slug>.wav    — per-section audio
  tts/audio/v01-durations.json             — per-section durations
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
        "Welcome to SLM one-oh-one. A short intro before we touch "
        "the product. Five minutes. By the end you'll know what a "
        "small language model is, why anyone bothers, and what "
        "shape the platform demo will take.",
    ),
    (
        "02-what-is-slm",
        "what_is_slm",
        "What's a small language model? It's a language model "
        "designed to be smaller, cheaper, faster, or easier to "
        "deploy than a big general-purpose one. Small is relative. "
        "It can mean fewer parameters, lower memory use, shorter "
        "context, a narrower task, or a smaller deployment target. "
        "There isn't a hard cutoff between LLM and SLM — there's a "
        "spectrum.",
    ),
    (
        "03-why-slms-matter",
        "why_matter",
        "Why smaller models matter. Five reasons. Lower serving "
        "cost — you don't need a fleet of H-one-hundreds to run "
        "inference. Lower latency — small models respond in "
        "milliseconds. Easier on-device or private deployment — "
        "the model fits on a laptop or a phone. Better control "
        "for narrow tasks — a model tuned for one job often beats "
        "a general-purpose one. And a smaller blast radius when "
        "something goes wrong.",
    ),
    (
        "04-lifecycle",
        "lifecycle",
        "The lifecycle. Define the task. Collect or import data. "
        "Clean it. Build a gold set — examples you trust to judge "
        "quality with. Generate synthetic examples if the source "
        "is small. Prepare train, validation, and test splits. "
        "Pick a base model. Configure and run training. Evaluate "
        "against the gold set. Compress and export. Then test "
        "the result in a real usage path. That's the loop the "
        "rest of these videos walk.",
    ),
    (
        "05-where-brewslm-fits",
        "brewslm_fits",
        "Where BrewSLM fits in this picture. It's a workspace for "
        "moving through that lifecycle on local hardware. Data, "
        "cleaning, gold sets, synthetic generation, dataset "
        "preparation, tokenization, training, evaluation, "
        "compression, export, registry, usage. Every stage has a "
        "tab. Every action leaves an artifact on disk. Nothing "
        "needs the cloud.",
    ),
    (
        "06-wrap",
        "wrap",
        "That's the intro. Next video opens the platform, seeds "
        "a demo project, and lands on the data tab in under five "
        "minutes. After that we walk three sample pipelines — "
        "support FAQ, PII detection, and sentiment classification "
        "— and then run a real training-to-serving loop on local "
        "hardware.",
    ),
]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    durations: dict[str, float] = {}
    total_inference = 0.0
    for slug, key, text in SECTIONS:
        out = OUT_DIR / f"v01-section-{slug}.wav"
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
    (OUT_DIR / "v01-durations.json").write_text(json.dumps(durations, indent=2))
    print()
    print(f"Total audio: {durations['__total_audio__']:.1f}s")
    print(f"Total inference: {total_inference:.1f}s")


if __name__ == "__main__":
    main()
