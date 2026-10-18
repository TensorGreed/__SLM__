"""Generate per-section narration WAV files for Video 10 — Evaluation.

Scores the run Video 09 trained against the practice answer key on
Results: the Evaluate form, the Verdict card, the pass/fail rules and
scorecard, then the sample predictions under Failures.

Sections map 1:1 to the sections of
tests/demo-recordings/10-evaluation-narrated.spec.ts (the spec reads
``v10-durations.json`` by section key). Audio comes straight from
Ollama + SNAC via ``tts/narration_tts.synthesize_checked``. Output:

  tts/audio/v10-section-<NN>-<slug>.wav   — per-section audio
  tts/audio/v10-durations.json            — duration per section + total
  tts/audio/concat-v10.txt                — ffmpeg concat list for the mux

Usage:
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v10_narration.py
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v10_narration.py --only 05-results

The spoken text is mirrored (with stage directions) in
docs-demo/scripts/narration/10-evaluation.md — edit both in the same commit.
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
        "Now we score the model we just trained. We are on Results, the "
        "page that answers three questions: did it work, where does it "
        "fail, and how does it compare. Verdict is the first tab, and its "
        "summary card sits at the top. We will score the run against the "
        "practice answer key. That is two hundred rows with hand-written "
        "answers, never seen during training.",
    ),
    (
        "02-setup",
        "setup",
        "Under Verdict, the button Evaluate on test examples opens the "
        "form. Pick the dataset — the test examples, the validation split, "
        "or the practice or final answer key — and a metric: exact match, "
        "F1, or an LLM judge. The model generates an answer for each "
        "question, and the handler scores it against the expected one. Here "
        "it is the practice answer key, scored by exact match, with token- "
        "level F1 reported beside it as a more forgiving second number.",
    ),
    (
        "03-kickoff",
        "kickoff",
        "I launch the evaluation through the API, so the recording stays "
        "repeatable. Twenty samples keep it short, at most sixty-four new "
        "tokens each, and no sampling randomness. The same call works "
        "against all two hundred rows. The practice answer key is for "
        "iterating. The final answer key stays untouched until you are "
        "ready to report a number you can trust.",
    ),
    (
        "04-watching",
        "watching",
        "The evaluation loads the trained checkpoint once, generates an "
        "answer for each sample, scores each one, and aggregates. On "
        "this hardware that is about ten seconds for twenty samples. "
        "Reload Verdict, and the card reads the new result.",
    ),
    (
        "05-results",
        "results",
        "Done. Exact match is zero. The model is too small to reproduce "
        "an expected answer word for word. F1 is around a quarter: real "
        "word overlap with the expected answers, and nowhere near "
        "production quality. The pass/fail rules fail, honestly, and "
        "the scorecard shows which rule failed and by how much. Then "
        "Failures: the sample predictions, each question next to the "
        "expected answer and what the model actually said. That is "
        "where you would start reading.",
    ),
    (
        "06-wrap",
        "wrap",
        "The result is now stored against the run. The score is not the "
        "point. The loop closed, and we can see exactly where the model "
        "falls short, row by row. The next video compresses the trained "
        "adapter into a small file that is ready to serve.",
    ),
]


def main() -> None:
    generate_video("v10", SECTIONS, voice=VOICE, only=parse_only(sys.argv[1:]))


if __name__ == "__main__":
    main()
