"""Generate per-section narration WAV files for Video 11 — Compression + Export.

Third runtime-dependent video. Takes the LoRA adapter from V09's
trained experiment, merges it into the base model, quantizes the
merged weights to GGUF Q4_K_M using llama.cpp, and registers the
artifact via the Export pipeline.

Outputs:
  tts/audio/v11-section-<NN>-<slug>.wav    — per-section audio
  tts/audio/v11-durations.json             — per-section durations
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
        "Now we ship. The training run produced a LoRA adapter — "
        "a small set of weight deltas that ride on top of the base "
        "model. To actually deploy, we merge those deltas back into "
        "the base, then quantize the merged weights down to four "
        "bits so the model is small and fast enough to serve on "
        "modest hardware.",
    ),
    (
        "02-compression-setup",
        "compression_setup",
        "Compression tab. Two settings matter here: quantization "
        "bits — four for a tight artifact, eight for higher quality "
        "— and output format. GGUF is the one we want because "
        "Ollama loads it natively in the next video. The LoRA "
        "adapter path is filled in from the experiment.",
    ),
    (
        "03-compress-run",
        "compress_run",
        "I'm kicking off both steps. First, merge LoRA — the "
        "trained adapter folds into the base model, producing a "
        "full half-precision checkpoint. Then quantize — llama.cpp "
        "converts that checkpoint into GGUF and then crunches it "
        "down to four-bit. The whole pipeline takes about twenty "
        "seconds on this hardware.",
    ),
    (
        "04-compression-result",
        "compression_result",
        "Done. The merged half-precision model was around two "
        "hundred and fifty megabytes. After quantization the GGUF "
        "is roughly one hundred megabytes — under half the size, "
        "ready to load on a phone-class CPU. The file is on disk "
        "in the project's compressed directory.",
    ),
    (
        "05-export-create",
        "export_create",
        "Now the Export tab. This is where we register the artifact "
        "against the experiment and pick deployment targets. "
        "Format: GGUF. Quantization: four-bit. The recommended "
        "deployment target for this combination is Ollama — that's "
        "Video Twelve.",
    ),
    (
        "06-export-run",
        "export_run",
        "Running the export. It validates the GGUF artifact, "
        "writes a manifest with the model hash and the deployment "
        "plan, and registers everything against the experiment. The "
        "result is a packaged export that downstream serving can "
        "pick up.",
    ),
    (
        "07-wrap",
        "wrap",
        "That's compression and export. We started with a LoRA "
        "adapter from training, and we end with a quantized GGUF "
        "file and a registered export manifest. Next video loads "
        "this artifact in Ollama and actually serves a prediction.",
    ),
]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main() -> None:
    durations: dict[str, float] = {}
    total_inference = 0.0
    for slug, key, text in SECTIONS:
        out = OUT_DIR / f"v11-section-{slug}.wav"
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
    (OUT_DIR / "v11-durations.json").write_text(json.dumps(durations, indent=2))
    print()
    print(f"Total audio: {durations['__total_audio__']:.1f}s")
    print(f"Total inference: {total_inference:.1f}s")


if __name__ == "__main__":
    main()
