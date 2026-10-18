"""Generate per-section narration WAV files for Video 12 — Final Model Usage.

Closes the runtime arc: registers Video 11's GGUF with Ollama, then asks
it a question in the chat playground on Results · Try it.

Sections map 1:1 to the sections of
tests/demo-recordings/12-final-model-usage-narrated.spec.ts (the spec reads
``v12-durations.json`` by section key). Audio comes straight from
Ollama + SNAC via ``tts/narration_tts.synthesize_checked``. Output:

  tts/audio/v12-section-<NN>-<slug>.wav   — per-section audio
  tts/audio/v12-durations.json            — duration per section + total
  tts/audio/concat-v12.txt                — ffmpeg concat list for the mux

Usage:
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v12_narration.py
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v12_narration.py --only 05-response

The spoken text is mirrored (with stage directions) in
docs-demo/scripts/narration/12-final-model-usage.md — edit both in the same commit.
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
        "Last step. We trained, we evaluated, we compressed, and the GGUF "
        "file is on disk. Now we serve it. Ollama loads the model, the chat "
        "playground sends a prompt, and the trained model actually "
        "responds. We are on Results, in the Try it tab. Loop closed. "
        "Everything you have seen in this series comes together in one "
        "reply.",
    ),
    (
        "02-ollama-register",
        "ollama_register",
        "First, register the file with Ollama. The recording script runs "
        "the ollama create command in the background. It points at the GGUF "
        "through a tiny Modelfile, and publishes the model under a friendly "
        "name. That takes a fraction of a second, because Ollama is only "
        "indexing bytes it already has on disk. The playground settings are "
        "here, ready for it.",
    ),
    (
        "03-playground-setup",
        "playground_setup",
        "Open the settings. The default provider is a trained run from this "
        "project, but to serve through Ollama we pick OpenAI-compatible, "
        "which covers Ollama's compatibility endpoint. The model name is "
        "the alias we just created. Temperature and max tokens can stay as "
        "they are. That is all the configuration this needs. Nothing here "
        "is specific to this one model. Anything Ollama can serve shows up "
        "the same way.",
    ),
    (
        "04-send-prompt",
        "send_prompt",
        "Now the prompt. I will ask the kind of question the training set "
        "covered: how to reset a password. The model has seen sixteen rows "
        "of customer support tickets. That is not enough to be excellent, "
        "but it is enough to produce the right shape of answer.",
    ),
    (
        "05-response",
        "response",
        "And there it is. A coherent, numbered, support-ticket style "
        "answer. It is not grounded in this company's real password reset "
        "flow, because the model has never seen one. But the format is "
        "right, the tone is right, and the loop fired end to end on a "
        "hundred megabytes of quantized weights.",
    ),
    (
        "06-wrap",
        "wrap",
        "That is the full runtime arc. We started with raw customer tickets "
        "and walked the workspace: Data, Train, Results, and Ship. We "
        "trained a tiny model with a real worker, scored it against an "
        "answer key, compressed the adapter into GGUF, and served it "
        "through Ollama. The same shape works for the PII detector and "
        "sentiment samples, scales up to real datasets, and runs entirely "
        "on local hardware.",
    ),
]


def main() -> None:
    generate_video("v12", SECTIONS, voice=VOICE, only=parse_only(sys.argv[1:]))


if __name__ == "__main__":
    main()
