"""Generate per-section narration WAV files for Video 03 (Support FAQ
workspace walkthrough — the video the in-app "Watch the walkthrough"
links point at).

Sections map 1:1 to the sections of
tests/demo-recordings/03-support-faq-pipeline-narrated.spec.ts and each
one is a YouTube chapter (the chapter list lives in
frontend/src/data/tabVideos.ts, v03). Audio comes straight from Ollama +
SNAC via ``tts/narration_tts.synthesize_checked``. Output:

  tts/audio/v03-section-<NN>-<slug>.wav   — per-section audio
  tts/audio/v03-durations.json            — duration per section + total
  tts/audio/concat-v03.txt                — ffmpeg concat list for the mux

Usage:
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v03_narration.py
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v03_narration.py --only 08-train-more

The spoken text is mirrored (with stage directions) in
docs-demo/scripts/narration/03-support-faq-pipeline.md — edit both in the
same commit. Rewritten 2026-10-18 for the Epic U workspace: Data (five
steps) → Train (the form, More options) → Results (four workflows) → Ship
(four stages).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.narration_tts import VOICE, generate_video, parse_only  # noqa: E402

# (wav slug, durations key, spoken text). Tone rules from tts/README.md: no
# literal API keys, env var names, REST paths, file extensions or adapter
# ids. Honest: the sample project has no trained run when the video starts,
# so the Results / Ship beats say what each page offers, never a result.
SECTIONS: list[tuple[str, str, str]] = [
    (
        "01-cold-open",
        "cold_open",
        "Welcome to the Support FAQ walkthrough. We're taking the simplest "
        "of the three starter projects — twenty customer tickets with "
        "hand-written answers — and walking every page of the workspace: "
        "Data, Train, Results, Ship. No training, no generation, just a "
        "tour of what each page offers.",
    ),
    (
        "02-data-sources",
        "data_sources",
        "Data is five steps. Sources: twenty raw documents, one per "
        "ticket. Expand one — a question and the agent's answer. Imagine "
        "pasting thousands of resolved tickets here and you have the "
        "dataset for a real support assistant.",
    ),
    (
        "03-data-prepare",
        "data_prepare",
        "Prepare. For a messy corpus this is where you chunk long text, "
        "redact personal information, and score quality. This sample is "
        "already small and clean, so there is nothing to run.",
    ),
    (
        "04-data-labels",
        "data_labels",
        "Labels. The answer key: two hundred rows, locked. Each has a "
        "question, an expected answer, and a rationale. After training, "
        "the evaluation walks the whole key and reports the fraction the "
        "model got right. The task shape is confirmed here too.",
    ),
    (
        "05-data-generate",
        "data_generate",
        "Generate. The lever that scales twenty source rows into thousands "
        "of training rows. A teacher model — local Ollama on this machine "
        "— reads your cleaned corpus and writes matching question and "
        "answer pairs. New rows land as pending until you review them. "
        "We're not running it here.",
    ),
    (
        "06-data-split",
        "data_split",
        "Split. This is where the contract gets made. The adapter — "
        "question and answer pair — turns each row into the shape the "
        "trainer expects. The splits are written: sixteen train, two "
        "validation, two test. You can keep the test examples fixed when "
        "you add data, and the page watches for leakage between the "
        "splits.",
    ),
    (
        "07-train-form",
        "train_form",
        "The Train page is one form. Base model, with Browse. The task "
        "shape, read-only, with Change. One run or three seeds — three "
        "picks the best by validation. A run name. The readiness line "
        "says whether the project can train right now, and the Train "
        "button starts it.",
    ),
    (
        "08-train-more",
        "train_more",
        "Everything else is under More options. Open it and the sections "
        "appear: model and hyperparameters, LoRA and advanced training, "
        "seeds, checks, cloud burst, and training presets. The defaults "
        "work; the controls are there when you need them. Launching a "
        "run is a later video.",
    ),
    (
        "09-results-verdict",
        "results_verdict",
        "Results has four workflows. Verdict answers one question — did "
        "it work? With no run yet it says so. After a run it shows the "
        "lift over the base model, with evidence, and a button to "
        "evaluate the run.",
    ),
    (
        "10-results-failures",
        "results_failures",
        "Failures — where does it fail? The failing rows, the built-in "
        "checks, your custom checks, the pass/fail rules, and the "
        "failure clusters all live here.",
    ),
    (
        "11-results-compare",
        "results_compare",
        "Compare — against other runs and other models: external "
        "benchmarks, a retrieval-only baseline, and a student-versus-"
        "teacher view. Pick two runs under Verdict and they appear side "
        "by side here.",
    ),
    (
        "12-results-try",
        "results_try",
        "Try it — a chat playground to talk to the trained model, with "
        "the option to answer from your own documents instead. It stays "
        "empty until a run completes.",
    ),
    (
        "13-ship-package",
        "ship_package",
        "Ship is four stages. Package: export the model — Hugging Face "
        "format, GGUF, ONNX — with a compress-first option and the "
        "project manifest. The releases list sits below it.",
    ),
    (
        "14-ship-publish",
        "ship_publish",
        "Publish: push a completed export to the Hugging Face Hub as a "
        "repository, model card included. Nothing to publish yet — no "
        "export exists.",
    ),
    (
        "15-ship-deploy",
        "ship_deploy",
        "Deploy: serve plans and deployments — where the packaged model "
        "will run, and with what resources. The plans are drafted from "
        "the export, so there is nothing to deploy before one exists.",
    ),
    (
        "16-ship-monitor",
        "ship_monitor",
        "Monitor: once a model is out, the timeline, production failures, "
        "and the bundles you shipped land here. What you learn feeds the "
        "next round of data — that closes the loop back to Data.",
    ),
    (
        "17-wrap",
        "wrap",
        "That's the Support FAQ tour — four pages, nothing heavy run. "
        "Overview tells you where you are, Data gets the rows ready, "
        "Train is one form, Results says whether it worked, Ship gets it "
        "out. Next video walks the same shape for the PII Detector "
        "sample.",
    ),
]


def main() -> None:
    generate_video("v03", SECTIONS, voice=VOICE, only=parse_only(sys.argv[1:]))


if __name__ == "__main__":
    main()
