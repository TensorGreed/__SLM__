"""Generate per-section narration WAV files for Video 10 — Evaluation.

Second runtime-dependent video. Scores the V09 trained checkpoint
against the support-faq gold_dev dataset (200 rows hand-labelled,
never seen during training). For pacing the spec evaluates 20
samples, which lands the eval handler in ~18 seconds.

Outputs:
  tts/audio/v10-section-<NN>-<slug>.wav    — per-section audio
  tts/audio/v10-durations.json             — per-section durations
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
        "Now we score the model we just trained against the gold "
        "set. Two hundred rows, hand-labelled, never seen during "
        "training. The eval handler dispatches to question-and-"
        "answer mode because the task profile is instruction-"
        "following.",
    ),
    (
        "02-setup",
        "setup",
        "Quick recap. Eval against gold is how we actually measure "
        "quality. The model generates an answer for each question "
        "in the held-out set, and the handler scores it against the "
        "expected answer. For this sample we get two headline "
        "numbers: exact match for the strict score, and token-level "
        "F1 as a more forgiving secondary.",
    ),
    (
        "03-kickoff",
        "kickoff",
        "Launching an eval run via the API. Held-out dataset is "
        "gold dev — twenty samples for this recording to keep it "
        "short, but the same call works against all two hundred.",
    ),
    (
        "04-watching",
        "watching",
        "Eval pipeline: load the trained checkpoint, run generation "
        "for each sample, score per sample, aggregate. On this "
        "hardware that's about twenty seconds for twenty samples — "
        "model load is one-shot, then it's roughly half a second "
        "per sample.",
    ),
    (
        "05-results",
        "results",
        "Done. Exact match landed at zero — the model is too small "
        "to produce verbatim matches yet. Token-level F1 is in "
        "the low tens of percent, which says there's some overlap "
        "with the gold answers but the model is far from "
        "production quality. The point isn't the score. The point "
        "is the loop closed.",
    ),
    (
        "06-wrap",
        "wrap",
        "Eval result is now stored against the experiment. Next "
        "video compresses the trained adapter into a quantized "
        "artifact ready to serve.",
    ),
]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    durations: dict[str, float] = {}
    total_inference = 0.0
    for slug, key, text in SECTIONS:
        out = OUT_DIR / f"v10-section-{slug}.wav"
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
    (OUT_DIR / "v10-durations.json").write_text(json.dumps(durations, indent=2))
    print()
    print(f"Total audio: {durations['__total_audio__']:.1f}s")
    print(f"Total inference: {total_inference:.1f}s")


if __name__ == "__main__":
    main()
