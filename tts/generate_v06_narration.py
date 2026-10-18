"""Generate per-section narration WAV files for Video 06 — BYO Custom Samples.

Demonstrates the "bring your own data" path. Creates a fresh
non-demo project, uploads a tiny coffee-shop-FAQ CSV, and lands on
the data tab with the imported rows visible.

Outputs (tts/audio/):
  v06-section-<NN>-<slug>.wav   — per-section audio
  v06-durations.json            — duration per section + total (the spec reads it)
  concat-v06.txt                — ffmpeg concat list for the mux

Usage:
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v06_narration.py
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v06_narration.py --only <section-slug>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.narration_tts import VOICE, generate_video, parse_only  # noqa: E402

SECTIONS: list[tuple[str, str, str]] = [
    (
        "01-cold-open",
        "cold_open",
        "Bring your own data. The sample projects are useful for "
        "learning the platform, but the real point is to use your own "
        "dataset. This video shows the shortest path from a CSV on your "
        "laptop to rows listed under Data, Sources.",
    ),
    (
        "02-new-project",
        "new_project",
        "From the project list, click New Project. Give it a name, and "
        "say in a sentence or two what you want the model to do. The "
        "platform reads that and sets sensible defaults — base model, "
        "task shape, output schema — and you can change any of it "
        "later. Click Create project, and you land on the project's "
        "Overview.",
    ),
    (
        "03-empty-data-tab",
        "empty_data_tab",
        "Project created. Open Data. The first step is Sources, and "
        "it's empty. No documents, no rows, no answer key. This is what "
        "a fresh project looks like before any data is imported. Across "
        "the top are the five Data steps — Sources, Prepare, Labels, "
        "Generate and Split — each one gets the rows ready for the next. "
        "The upload zone is right there.",
    ),
    (
        "04-upload-csv",
        "upload_csv",
        "The upload zone accepts files straight from your computer. "
        "Documents — PDF, Word, HTML, text, Markdown — and row files — "
        "CSV, TSV, Excel, JSON, JSONL, Parquet — in any encoding. For "
        "this video the file is a tiny six-row coffee-shop FAQ in CSV "
        "form. Drop it on the zone or choose it with the file picker.",
    ),
    (
        "05-rows-imported",
        "rows_imported",
        "And the rows are in. Under ingested documents, the file is "
        "listed, and expanding it shows its question and answer "
        "columns, kept as columns all the way to training. The "
        "platform also reads the shape of the rows, and the Labels step "
        "is where you confirm the task it found. From here "
        "the project goes through the same steps you saw on the "
        "samples: prepare, labels, generate, split, then train, results "
        "and ship.",
    ),
    (
        "06-wrap",
        "wrap",
        "That's the bring-your-own-data path. The Overview now shows "
        "where this project is and what the next step is. The rest of "
        "the workflow for a custom project is identical to what the "
        "sample videos walked. Same workspace, same artifacts, same "
        "training and serving. When the split is done, Train is one "
        "form: a base model, a run name and one Train button, with "
        "everything else under More options. The only difference is "
        "the file at the start.",
    ),
]


def main() -> None:
    generate_video("v06", SECTIONS, voice=VOICE, only=parse_only(sys.argv[1:]))


if __name__ == "__main__":
    main()
