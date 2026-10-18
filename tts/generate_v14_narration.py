"""Generate per-section narration WAV files for Video 14 — Architecture.

Companion slide video to V01: the platform's components and how they fit
together (a static deck; no app screens).

Sections map 1:1 to the sections of
tests/demo-recordings/14-architecture-narrated.spec.ts (the spec reads
``v14-durations.json`` by section key). Audio comes straight from
Ollama + SNAC via ``tts/narration_tts.synthesize_checked``. Output:

  tts/audio/v14-section-<NN>-<slug>.wav   — per-section audio
  tts/audio/v14-durations.json            — duration per section + total
  tts/audio/concat-v14.txt                — ffmpeg concat list for the mux

Usage:
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v14_narration.py
  tts/Orpheus-FastAPI/.venv/bin/python tts/generate_v14_narration.py --only 02-the-stack

The spoken text is mirrored (with stage directions) in
docs-demo/scripts/narration/14-architecture.md — edit both in the same commit.
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
        "01-title",
        "title",
        "Architecture. A few minutes on what makes this platform work under "
        "the hood, and why every piece runs on local hardware. This is the "
        "closer for the series, so if you want to know why the other videos "
        "work the way they do, this is the one.",
    ),
    (
        "02-the-stack",
        "stack",
        "The stack. Five processes. A FastAPI backend handles the HTTP API "
        "and the database. A React frontend renders the workspace: "
        "Overview, Data, Train, Results, and Ship. A Celery worker runs the "
        "long jobs, such as training, compression, and dataset import. "
        "Redis is the broker between the backend and the worker. And Ollama "
        "serves three things: the teacher model for synthetic generation, "
        "the judge model for evaluation, and the trained models we ship.",
    ),
    (
        "03-data-flow",
        "data_flow",
        "Data flow. Raw rows come in through ingestion, get cleaned and "
        "chunked, get an answer key written, and get expanded with "
        "synthetic samples if needed. They are split into train, "
        "validation, and test examples, tokenized, and then feed the "
        "training loop. Training produces a LoRA adapter. Evaluation scores "
        "it against the answer key. Compression merges and quantizes. "
        "Export packages. Ollama serves.",
    ),
    (
        "04-where-things-run",
        "where_things_run",
        "Where things run. The whole stack is local. There are no cloud "
        "dependencies, though cloud models are an option when you choose "
        "one. Training fires on the GPU through PyTorch. Inference for the "
        "teacher, the judge, and the final model all goes through Ollama on "
        "the same GPU. Quantization shells out to llama dot c-p-p. Disk "
        "holds the artifacts. The frontend, on port five one seven three, "
        "drives the whole thing through the backend's REST API on port "
        "eight thousand.",
    ),
    (
        "05-trust-boundaries",
        "trust_boundaries",
        "Trust boundaries. Two of them. The backend is the only thing that "
        "touches the database, the file system, and the model weights. "
        "Everything that goes through the frontend hits an authenticated "
        "endpoint. The Celery worker sits inside the backend's trust "
        "boundary, because it shares the same database and the same data "
        "directory. Ollama runs in its own process, and the backend treats "
        "it as an external service.",
    ),
    (
        "06-wrap",
        "wrap",
        "That is the architecture. One backend, one frontend, one worker, "
        "one broker, one inference runtime. Five processes, one machine, "
        "from raw rows to a served model. The other videos in this series "
        "demonstrate every piece of that loop running for real.",
    ),
]


def main() -> None:
    generate_video("v14", SECTIONS, voice=VOICE, only=parse_only(sys.argv[1:]))


if __name__ == "__main__":
    main()
