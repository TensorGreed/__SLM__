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

```bash
# 1. Start the Orpheus-FastAPI server (background)
cd tts/Orpheus-FastAPI
source .venv/bin/activate
nohup python app.py > /tmp/orpheus-server.log 2>&1 &
disown
# Wait until :5005 is listening

# 2. Generate per-section WAVs from the narration script
cd ..
python generate_narration.py
# Output:
#   tts/audio/v02-section-*.wav      (one per section)
#   tts/audio/v02-durations.json     (durations the spec reads)

# 3. Re-record the Playwright video, re-timed to the audio
cd ..
npx playwright test 02-brewslm-quickstart-narrated.spec.ts --project chromium

# 4. Concatenate the WAVs + mux onto the video
FFMPEG=$(./tts/Orpheus-FastAPI/.venv/bin/python -c \
    "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
VIDEO="test-results/02-brewslm-quickstart-narrated-Video-02-—-narrated-take-chromium/video.webm"

# concat list
> tts/audio/concat.txt
for n in 01-cold-open 02-login 03-tiles 04-seed 05-cleaning 06-goldset \
         07-dataprep 08-training 09-expand-wrap; do
    echo "file '$(pwd)/tts/audio/v02-section-${n}.wav'" >> tts/audio/concat.txt
done
$FFMPEG -y -f concat -safe 0 -i tts/audio/concat.txt -c copy \
    tts/audio/v02-narration.wav

# mux video + audio
$FFMPEG -y -i "$VIDEO" -i tts/audio/v02-narration.wav \
    -c:v libx264 -preset slow -crf 20 \
    -c:a aac -b:a 192k \
    -map 0:v:0 -map 1:a:0 -shortest \
    docs-demo/recordings/raw/02-brewslm-quickstart-narrated.mp4

# 5. Stop the Orpheus server (frees VRAM)
pkill -f "Orpheus-FastAPI.*app.py"
```

## Editing narration

`generate_narration.py` is the authoritative source for the spoken
text. The narration MD at
`docs-demo/scripts/narration/02-brewslm-quickstart.md` is a
human-readable mirror with stage directions; keep the two in sync
when you edit.

Available voices (set `VOICE` constant at the top of the script):
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
