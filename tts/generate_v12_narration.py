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
        "the alias we just created. Temperature goes to zero, so the model "
        "gives its single most likely answer, the same one on every run. "
        "That is all the configuration this needs. Nothing here "
        "is specific to this one model. Anything Ollama can serve shows up "
        "the same way.",
    ),
    (
        "04-send-prompt",
        "send_prompt",
        "Now the prompt. I will ask a question the training set "
        "covered: how to reset a password. The model has seen a hundred "
        "and eight customer support tickets, a few of them about "
        "exactly this. That is not enough to be excellent, but it is "
        "enough to see what it learned.",
    ),
    (
        "05-response",
        "response",
        "And there it is. Settings, then Security, then a Reset button: "
        "that much is exactly where the training tickets point. Then it "
        "drifts. The tickets say you get an emailed one-time link that "
        "expires after fifteen minutes. The model never mentions the email, "
        "invents a choice between twelve months and thirty days, and sends "
        "you to a keyring page that no ticket describes. Half right, "
        "confidently worded, which is what the evaluation's quarter F1 was "
        "telling us. And it all "
        "runs on a hundred megabytes of quantized weights.",
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
