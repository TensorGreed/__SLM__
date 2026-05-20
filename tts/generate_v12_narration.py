"""Generate per-section narration WAV files for Video 12 — Final Model Usage.

Closes the runtime arc. Takes the GGUF artifact from V11, registers
it with Ollama via `ollama create`, then sends a prompt through the
BrewSLM Playground UI to verify the model actually responds.

Outputs:
  tts/audio/v12-section-<NN>-<slug>.wav    — per-section audio
  tts/audio/v12-durations.json             — per-section durations
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
        "Last step. We trained, we evaluated, we compressed. The "
        "GGUF artifact is on disk. Now we serve. Ollama loads the "
        "model, the Playground sends a prompt, and the trained "
        "model actually responds. Loop closed.",
    ),
    (
        "02-ollama-register",
        "ollama_register",
        "First, register the artifact with Ollama. The Playwright "
        "spec runs the ollama create command in the background — "
        "it points at the GGUF file via a tiny Modelfile and "
        "publishes the model under a friendly alias. Takes a "
        "fraction of a second; Ollama is just indexing the bytes "
        "it already has on disk.",
    ),
    (
        "03-playground-setup",
        "playground_setup",
        "Open the Playground. Provider is OpenAI-Compatible, which "
        "covers Ollama's compatibility endpoint on port one one "
        "four three four. Model name is the alias we just created. "
        "Temperature low, max tokens enough for a short reply. "
        "That's all the configuration this needs.",
    ),
    (
        "04-send-prompt",
        "send_prompt",
        "Now the prompt. I'll ask the model the kind of question "
        "the training set covered — how to reset a password. The "
        "trained model has seen sixteen rows of customer support "
        "tickets. Not enough to be excellent, but enough to "
        "produce the right shape of answer.",
    ),
    (
        "05-response",
        "response",
        "And there it is. A coherent, numbered, support-ticket "
        "style answer. It's not factually grounded in this "
        "company's real password-reset flow — the model has "
        "never seen one. But the format is correct, the tone is "
        "correct, and the loop fired end to end on a hundred "
        "and five megabytes of quantized weights.",
    ),
    (
        "06-wrap",
        "wrap",
        "That's the full SLM platform demo. Eight videos. We "
        "started with raw customer tickets, walked the dataset "
        "pipeline, trained a tiny model with real Celery, scored "
        "it against gold, compressed the LoRA into GGUF, and "
        "served the result through Ollama. Same shape works on "
        "the PII detector and sentiment samples, scales up to "
        "real datasets, and runs entirely on local hardware.",
    ),
]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    durations: dict[str, float] = {}
    total_inference = 0.0
    for slug, key, text in SECTIONS:
        out = OUT_DIR / f"v12-section-{slug}.wav"
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
    (OUT_DIR / "v12-durations.json").write_text(json.dumps(durations, indent=2))
    print()
    print(f"Total audio: {durations['__total_audio__']:.1f}s")
    print(f"Total inference: {total_inference:.1f}s")


if __name__ == "__main__":
    main()
