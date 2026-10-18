"""Generate per-section narration WAV files for Video 02 (BrewSLM quickstart).

Sections map 1:1 to the sections of
tests/demo-recordings/02-brewslm-quickstart-narrated.spec.ts; the second
tuple field is the key the spec reads from the durations JSON. Audio comes
straight from Ollama + SNAC via ``tts/narration_tts.synthesize_checked``
(no FastAPI bridge). Output:

  tts/audio/v02-section-<NN>-<slug>.wav   — per-section audio
  tts/audio/v02-durations.json            — duration per section + total
  tts/audio/concat-v02.txt                — ffmpeg concat list for the mux

Usage:
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_narration.py
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_narration.py --only 04-overview

The spoken text is mirrored (with stage directions) in
docs-demo/scripts/narration/02-brewslm-quickstart.md — edit both in the
same commit. Rewritten 2026-10-18 for the Epic U workspace (Overview /
Data steps / Train form); no tabs, Coach bar or Training Config any more.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.narration_tts import VOICE, generate_video, parse_only  # noqa: E402

# (wav slug, durations key, spoken text). Keep the tone rules from
# tts/README.md: no literal API keys, env var names, REST paths or file
# extensions — the on-screen action shows those.
SECTIONS: list[tuple[str, str, str]] = [
    (
        "01-cold-open",
        "cold_open",
        "We're taking a fresh local BrewSLM install and getting from "
        "nothing to a fully seeded sample project, in under five minutes. "
        "No training, no data prep — just log in, click, and look around.",
    ),
    (
        "02-login",
        "login",
        "BrewSLM auth is on by default. I'll log in as admin — the "
        "bootstrap user. The password is the local development token "
        "from the backend env file. And we're in.",
    ),
    (
        "03-tiles",
        "tiles",
        "This is the project list. Up top, three starter projects: "
        "Support FAQ, PII Detector, and Sentiment Classifier. Each is "
        "backed by a manifest, a source file, and a two-hundred-row "
        "answer key. Clicking a tile seeds a complete project — raw data "
        "imported, the answer key imported, and train, validation, and "
        "test splits already written.",
    ),
    (
        "04-overview",
        "overview",
        "I'll click Support FAQ. The backend copies the source file into "
        "the project, creates twenty raw documents, imports two hundred "
        "answer-key rows, and pre-writes a sixteen, two, two split. We "
        "land on the project Overview: where you are, the next step, and "
        "under Full plan, the five stages — bring data in, prepare, "
        "train, check the results, ship. Nothing is trained yet, but the "
        "data work is done.",
    ),
    (
        "05-sources",
        "sources",
        "The sidebar has six pages. Data first. It's five steps: Sources, "
        "Prepare, Labels, Generate, Split. Sources holds the twenty raw "
        "documents — one per support ticket. Expand one and you see the "
        "shape: a question and the agent's answer. That's what the model "
        "will learn to write.",
    ),
    (
        "06-prepare",
        "prepare",
        "Prepare — chunk text, redact personal information, score "
        "quality. Nothing is running here; the support FAQ corpus is "
        "already small and clean. On a messy corpus this is where the "
        "work would happen.",
    ),
    (
        "07-labels",
        "labels",
        "Labels. The answer key: two hundred rows, locked. Every run gets "
        "scored against it — it is never trained on. This step also "
        "confirms the task shape, so training and scoring agree on what "
        "a row is.",
    ),
    (
        "08-split",
        "split",
        "Split. The adapter is applied — question and answer pair — so "
        "each row has the shape the trainer expects. The splits are "
        "already written: sixteen train, two validation, two test.",
    ),
    (
        "09-train",
        "train",
        "The Train page is one form. Base model, with Browse. The task "
        "shape, read-only, with Change. One run or three seeds. A run "
        "name. Below it, the readiness line tells you whether the project "
        "can train right now, and one Train button starts it. Everything "
        "else lives under More options.",
    ),
    (
        "10-wrap",
        "wrap",
        "We won't press Train in this video. From a fresh install to a "
        "seeded project with twenty raw documents, a two-hundred-row "
        "answer key, and ready-to-train splits. The next video walks "
        "every page of the workspace in detail.",
    ),
]


def main() -> None:
    generate_video("v02", SECTIONS, voice=VOICE, only=parse_only(sys.argv[1:]))


if __name__ == "__main__":
    main()
