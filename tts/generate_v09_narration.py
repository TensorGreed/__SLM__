"""Generate per-section narration WAV files for Video 09 — Training Run.

First runtime-dependent video: a real training run on the support-faq
sample (SmolLM2-135M, two epochs, sixteen steps, about twelve seconds on
the GB10) through the real worker, shown on the Train page.

Sections map 1:1 to the sections of
tests/demo-recordings/09-training-run-narrated.spec.ts (the spec reads
``v09-durations.json`` by section key). Audio comes straight from
Ollama + SNAC via ``tts/narration_tts.synthesize_checked``. Output:

  tts/audio/v09-section-<NN>-<slug>.wav   — per-section audio
  tts/audio/v09-durations.json            — duration per section + total
  tts/audio/concat-v09.txt                — ffmpeg concat list for the mux

Usage:
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v09_narration.py
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v09_narration.py --only 05-results

The spoken text is mirrored (with stage directions) in
docs-demo/scripts/narration/09-training-run.md — edit both in the same commit.
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
        "Now we actually train. The first videos walked the workspace. This "
        "one launches a real training run on the Support FAQ sample. This "
        "is the Train page, and it is one form. A base model — here a "
        "hundred and thirty-five million parameter model, small enough to "
        "train in seconds. A readiness line that says whether the project "
        "can train right now. A Train button. And below, the Runs list, "
        "where every run and its live metrics stay. A real worker, a real "
        "loss curve, and a real checkpoint on disk at the end.",
    ),
    (
        "02-config-recap",
        "config_recap",
        "Before we launch, a quick look behind More options. Model and "
        "hyperparameters holds what you would touch first: epochs, batch "
        "size, and learning rate. LoRA and advanced training holds the "
        "parameter-efficient controls — the adapter's rank, which layers it "
        "targets, the optimizer, and the numeric precision. The defaults "
        "are tuned for this hardware, and the form sizes the number of "
        "epochs to your row count by itself. For this recording I pin two "
        "epochs, so the run is short and repeatable.",
    ),
    (
        "03-kickoff",
        "kickoff",
        "Now the run. To keep the recording repeatable I create and start "
        "it through the API, with a fixed configuration: two epochs, batch "
        "size two. Pressing Train does the same thing. It asks you to "
        "confirm, creates the run, and hands it to the worker. Reload the "
        "page, and the new run sits at the top of the Runs list, queued and "
        "then running.",
    ),
    (
        "04-watching",
        "watching",
        "Status is running. A hundred and eight training rows, two "
        "epochs at a batch size of two: a hundred and eight optimizer "
        "steps. Each step is a forward pass, a backward pass, and a "
        "weight update, and the loss should come down across the run. "
        "On this hardware the whole thing takes about ten seconds. "
        "Reload the list, and the badge flips to completed.",
    ),
    (
        "05-results",
        "results",
        "Completed. Open the run's dashboard and you get the epoch "
        "counter, the training loss, and the evaluation loss. The final "
        "evaluation loss lands around three. That is still high: the "
        "model is tiny, the run was ten seconds long, and it is a loss, "
        "not an accuracy. It tells us nothing yet about the answers. "
        "What matters is that the loop completed end to end, and there "
        "is now a checkpoint on disk, ready to be scored on the Results "
        "page.",
    ),
    (
        "06-wrap",
        "wrap",
        "That is the training loop. One form, a run in the list, a "
        "dashboard with the metrics. The next video scores this run against "
        "the practice answer key, and tells us how often the model actually "
        "got the answer right.",
    ),
]


def main() -> None:
    generate_video("v09", SECTIONS, voice=VOICE, only=parse_only(sys.argv[1:]))


if __name__ == "__main__":
    main()
