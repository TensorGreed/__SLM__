"""Generate per-section narration WAV files for Video 09 — Training Run.

First runtime-dependent video in the series. Captures an actual
training experiment running on the support-faq sample: SmolLM2-135M,
two epochs, sixteen training steps. Runtime is real Celery via
`builtin.external_celery`; takes ~12 seconds on GB10.

Outputs:
  tts/audio/v09-section-<NN>-<slug>.wav    — per-section audio
  tts/audio/v09-durations.json             — per-section durations
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
        "Now we actually train. The first four videos walked the "
        "surfaces. This one launches a real training run on the "
        "support FAQ sample. Small model — a hundred and thirty-"
        "five million parameters — two epochs over sixteen prepared "
        "rows. Real Celery worker, real loss curve, real artifact "
        "on disk at the end.",
    ),
    (
        "02-config-recap",
        "config_recap",
        "Quick recap of the Training Config page. Essentials view "
        "covers what you'd touch first — base model, epochs, batch "
        "size, learning rate. Flip to Advanced and you get the "
        "parameter-efficient training controls: low-rank adaptation "
        "rank, target modules, optimizer. Defaults are tuned for "
        "this hardware.",
    ),
    (
        "03-kickoff",
        "kickoff",
        "Back to the Training tab. I'm creating a new experiment "
        "and starting it. The Playwright spec uses the API for the "
        "create-and-start sequence so the recording stays "
        "deterministic. Either way, the worker queues the job and "
        "the runtime takes over.",
    ),
    (
        "04-watching",
        "watching",
        "Status is running. Sixteen training steps total — each "
        "step does a forward pass, a backward pass, an optimizer "
        "step. The loss should drop across the run. On this "
        "hardware the whole thing finishes in about twelve seconds. "
        "Refresh the table.",
    ),
    (
        "05-results",
        "results",
        "Completed. Two epochs, sixteen steps, final evaluation "
        "loss around five. The loss number is high because the "
        "model is tiny and the dataset has sixteen rows. The point "
        "isn't the loss number — the point is the loop completed "
        "end to end, and we now have a checkpoint on disk ready "
        "for evaluation.",
    ),
    (
        "06-wrap",
        "wrap",
        "That's the training loop. Next video scores this "
        "experiment against the two-hundred-row gold set and "
        "tells us how often the model actually got the answer "
        "right.",
    ),
]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    durations: dict[str, float] = {}
    total_inference = 0.0
    for slug, key, text in SECTIONS:
        out = OUT_DIR / f"v09-section-{slug}.wav"
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
    (OUT_DIR / "v09-durations.json").write_text(json.dumps(durations, indent=2))
    print()
    print(f"Total audio: {durations['__total_audio__']:.1f}s")
    print(f"Total inference: {total_inference:.1f}s")


if __name__ == "__main__":
    main()
