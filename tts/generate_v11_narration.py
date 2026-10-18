"""Generate per-section narration WAV files for Video 11 — Compression + Export.

Merges the Video 09 adapter into the base, quantizes it to GGUF and
registers the export — all on Ship, in the Package stage (the export
form on top, 'Compress first' under Options).

Sections map 1:1 to the sections of
tests/demo-recordings/11-compression-export-narrated.spec.ts (the spec reads
``v11-durations.json`` by section key). Audio comes straight from
Ollama + SNAC via ``tts/narration_tts.synthesize_checked``. Output:

  tts/audio/v11-section-<NN>-<slug>.wav   — per-section audio
  tts/audio/v11-durations.json            — duration per section + total
  tts/audio/concat-v11.txt                — ffmpeg concat list for the mux

Usage:
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v11_narration.py
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v11_narration.py --only 03-compress-run

The spoken text is mirrored (with stage directions) in
docs-demo/scripts/narration/11-compression-export.md — edit both in the same commit.
Rewritten 2026-10-18 for the Epic U workspace.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.narration_tts import VOICE, generate_video, parse_only  # noqa: E402

# (wav slug, durations key, spoken text). Tone rules from tts/README.md: no
# literal API keys, env var names, REST paths, file extensions or adapter ids.
SECTIONS: list[tuple[str, str, str]] = [
    (
        "01-cold-open",
        "cold_open",
        "Now we ship. The training run produced a LoRA adapter, a small set "
        "of weight changes that rides on top of the base model. To deploy "
        "it, we merge those changes back into the base, then quantize the "
        "merged weights down to four bits, so the model is small and fast "
        "enough for modest hardware. All of this happens on Ship, in the "
        "Package stage. The export form is on top, and compression lives "
        "under Options.",
    ),
    (
        "02-compression-setup",
        "compression_setup",
        "Open Options, then Compress first. You pick a completed training "
        "run, and the model and adapter paths fill in from it. Two settings "
        "matter. Quantization bits: four for a tight file, eight for higher "
        "quality. And the output format. GGUF is the one we want, because "
        "Ollama loads it directly in the next video.",
    ),
    (
        "03-compress-run",
        "compress_run",
        "Now the two steps. First, merge the adapter: the trained changes "
        "fold into the base model and produce a full half-precision "
        "checkpoint. Then quantize: llama dot c-p-p converts that "
        "checkpoint to GGUF and crunches it down to four bits. I fire both "
        "through the API to keep the recording repeatable. The whole "
        "pipeline takes about twenty seconds on this hardware.",
    ),
    (
        "04-compression-result",
        "compression_result",
        "Done. The merged half-precision model was around two hundred and "
        "fifty megabytes. After quantization the GGUF is roughly a hundred "
        "megabytes. That is under half the size, small enough for a phone- "
        "class processor. The files sit in the project's compressed folder "
        "on disk. The panel's own result card only fills for a run started "
        "from the panel itself, so here the numbers come from the disk.",
    ),
    (
        "05-export-create",
        "export_create",
        "Back on top, the New export form. This is where we register the "
        "artifact against the run. Pick the experiment, then the format: "
        "GGUF. The quantization is four-bit, and the deployment we are "
        "aiming for is Ollama. That is the next video.",
    ),
    (
        "06-export-run",
        "export_run",
        "Running the export validates the file, writes a manifest with the "
        "model hash and a deployment plan, and registers everything against "
        "the run. It lands under Releases, with the exports table below: "
        "format, status, size, and when. If an export fails validation, its "
        "row says so. Whatever the status says is the truth.",
    ),
    (
        "07-wrap",
        "wrap",
        "That is compression and export. We started with an adapter from "
        "training, and we end with a quantized GGUF file and a registered "
        "export. Publish and Deploy are the other two stages on Ship. The "
        "next video loads this file into Ollama and serves a real "
        "prediction.",
    ),
]


def main() -> None:
    generate_video("v11", SECTIONS, voice=VOICE, only=parse_only(sys.argv[1:]))


if __name__ == "__main__":
    main()
