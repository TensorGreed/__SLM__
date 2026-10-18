# TTS narration pipeline

Local-only setup that turns the recording-plan narration into per-section
WAV files, then re-records the Playwright video re-timed to match the
audio. The end artifact is a narrated MP4 of a demo video.

Only `generate_narration.py` is checked into the repo; the third-party
server (Orpheus-FastAPI), its venv, and the generated audio are all
gitignored — reproducible from the script + the Ollama model.

## One-time setup

Requires `uv` (https://docs.astral.sh/uv/) and a running local Ollama.

```bash
# 1. Python 3.11 via uv (Orpheus-FastAPI requires <3.12)
uv python install 3.11

# 2. Clone and configure Orpheus-FastAPI
cd tts
git clone --depth=1 https://github.com/Lex-au/Orpheus-FastAPI.git
cd Orpheus-FastAPI
uv venv --python 3.11 .venv

source .venv/bin/activate
# numpy 1.24 pin in their requirements.txt breaks on Python 3.11 aarch64;
# bump it to a recent compatible release.
sed -i 's/^numpy==1.24.0/numpy>=1.26,<3/' requirements.txt
uv pip install torch torchaudio
uv pip install -r requirements.txt
uv pip install imageio-ffmpeg   # static ffmpeg binary, no sudo apt needed

# 3. Patch sounddevice import to be optional (we don't need live playback,
#    only WAV file output). One try/except wrapper in
#    tts_engine/inference.py around `import sounddevice as sd`.

# 4. Configure server to point at local Ollama
cat > .env <<'EOF'
ORPHEUS_API_URL=http://127.0.0.1:11434/v1/completions
ORPHEUS_API_TIMEOUT=300
ORPHEUS_MAX_TOKENS=8192
ORPHEUS_TEMPERATURE=0.6
ORPHEUS_TOP_P=0.9
ORPHEUS_SAMPLE_RATE=24000
ORPHEUS_MODEL_NAME=legraphista/Orpheus:3b-ft-q4_k_m
ORPHEUS_PORT=5005
ORPHEUS_HOST=127.0.0.1
EOF

# 5. Pull the Orpheus model into Ollama
ollama pull legraphista/Orpheus:3b-ft-q4_k_m
```

## Per-recording workflow

The generators call Ollama + SNAC directly (`tts/orpheus_direct.py` via
`tts/narration_tts.py`) — the Orpheus-FastAPI server on :5005 is **not**
needed any more (its stream parser mangles the tokens anyway; see the
docstring in `orpheus_direct.py`). Only Ollama must be running. The venv's
python is `tts/Orpheus-FastAPI/.venv/bin/python` (torch + snac live there).

```bash
# 1. Generate per-section WAVs from the narration script (video 02 here;
#    video 03 is generate_v03_narration.py, same flags)
tts/Orpheus-FastAPI/.venv/bin/python tts/generate_narration.py
#    --only 04-overview 09-train   # regenerate just those sections
# Output:
#   tts/audio/v02-section-*.wav      (one per section)
#   tts/audio/v02-durations.json     (durations the spec reads, keyed by section)
#   tts/audio/concat-v02.txt         (the ffmpeg concat list for step 3)
# The generator refuses a section whose length is implausible for its word
# count (~2.6 words/s is normal; Orpheus occasionally truncates or loops —
# each section is retried once) and a total outside the 2–4 minute band.

# 2. Re-record the Playwright video, re-timed to the audio
npx playwright test 02-brewslm-quickstart-narrated.spec.ts --project chromium

# 3. Concatenate the WAVs + mux onto the video
FFMPEG=$(./tts/Orpheus-FastAPI/.venv/bin/python -c \
    "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
VIDEO="test-results/02-brewslm-quickstart-narrated-Video-02-—-narrated-take-chromium/video.webm"

$FFMPEG -y -f concat -safe 0 -i tts/audio/concat-v02.txt -c copy \
    tts/audio/v02-narration.wav

# mux video + audio
$FFMPEG -y -i "$VIDEO" -i tts/audio/v02-narration.wav \
    -c:v libx264 -preset slow -crf 20 \
    -c:a aac -b:a 192k \
    -map 0:v:0 -map 1:a:0 -shortest \
    docs-demo/recordings/raw/02-brewslm-quickstart-narrated.mp4
```

The older `generate_v04_narration.py` … `generate_v14_narration.py` still
post to the :5005 bridge; port them to `narration_tts.generate_video` when
they are next re-recorded.

## Editing narration

`generate_narration.py` is the authoritative source for the spoken
text. The narration MD at
`docs-demo/scripts/narration/02-brewslm-quickstart.md` is a
human-readable mirror with stage directions; keep the two in sync
when you edit.

Available voices (the `VOICE` constant in `tts/narration_tts.py`, shared
by every generator so the series keeps one narrator):
- Female: `tara`, `leah`, `jess`, `mia`, `zoe`
- Male: `leo`, `dan`, `zac`

Voices have different speech rates — leo runs ~16% faster than dan.
The Playwright spec auto-adjusts its waits to whatever `v02-durations.json`
reports, so the on-screen action always lines up with the audio.

## Why this stack

- **Local-first**: matches the project's no-cloud ethos. No API keys,
  no per-character billing, no rate limits.
- **Ollama re-use**: the same Ollama daemon used for the teacher /
  judge model serves the Orpheus inference too. One process, one
  GPU.
- **GB10 friendly**: NVIDIA's aarch64 CUDA PyTorch wheels work out
  of the box via the standard PyPI index — no special index URL,
  no JetPack, no Docker.

## Don't read these aloud

TTS engines mispronounce literal tech tokens. The narration script
should avoid:

- API keys (`sk-mock-admin-key` → "the local development token")
- Environment variable names (`AUTH_BOOTSTRAP_USERNAME` → "the bootstrap user")
- REST paths (`POST /api/demo-projects/support-faq` → "the backend")
- File extensions (`JSONL`, `CSV` → "gold set", "source file")
- Adapter names (`qa-pair` → "question and answer pair")

The on-screen action shows the literal value anyway.
