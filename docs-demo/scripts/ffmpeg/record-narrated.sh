#!/usr/bin/env bash
# Record one narrated video and mux its narration.
# Usage: record-narrated.sh NN   (02, 03, 04, … — any tests/demo-recordings/NN-*-narrated.spec.ts)
#
# Records against DEMO_BASE_URL (default: the scratch stack from scratch-stack.sh,
# http://localhost:5176). Run `scratch-stack.sh down && scratch-stack.sh up` before
# each video so it starts from a fresh DB; never point this at the dev stack.
# Needs tts/audio/vNN-durations.json + concat-vNN.txt (tts/generate_vNN_narration.py).
set -euo pipefail
export PATH=$HOME/.nvm/versions/node/v22.23.2/bin:$PATH
cd /home/anuragj/Desktop/GitHub/__SLM__
N="${1:?usage: record-narrated.sh NN}"
SPEC=$(ls tests/demo-recordings/${N}-*-narrated.spec.ts 2>/dev/null | head -1)
[ -n "$SPEC" ] || { echo "no tests/demo-recordings/${N}-*-narrated.spec.ts" >&2; exit 2; }
SLUG=$(basename "$SPEC" -narrated.spec.ts)
export DEMO_BASE_URL="${DEMO_BASE_URL:-http://localhost:5176}"
export DEMO_DATA_DIR="${DEMO_DATA_DIR:-${DEMO_SCRATCH_DIR:-/tmp/brewslm-demo-stack}/data}"
curl -sf -o /dev/null "$DEMO_BASE_URL/" || { echo "$DEMO_BASE_URL is not up — run scratch-stack.sh up" >&2; exit 1; }
[ -f "tts/audio/concat-v${N}.txt" ] || { echo "no narration — run tts/generate_v${N}_narration.py" >&2; exit 1; }
rm -f "tts/audio/v${N}-preroll.json"
OUT=docs-demo/recordings/raw
mkdir -p "$OUT"
rm -rf test-results
echo "== recording $SLUG"
npx playwright test "$SPEC" --project chromium --reporter=line
VIDEO=$(find test-results -name 'video.webm' | head -1)
[ -n "$VIDEO" ] || { echo "no video produced" >&2; exit 1; }
echo "== video: $VIDEO ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")s)"
echo "== concatenating narration"
ffmpeg -y -loglevel error -f concat -safe 0 -i "tts/audio/concat-v${N}.txt" -c copy "tts/audio/v${N}-narration.wav"
echo "== narration: $(ffprobe -v error -show_entries format=duration -of csv=p=0 tts/audio/v${N}-narration.wav)s"
PREROLL=0
[ -f "tts/audio/v${N}-preroll.json" ] && PREROLL=$(python3 -c "import json;print(json.load(open('tts/audio/v${N}-preroll.json'))['preroll_seconds'])")
echo "== trimming pre-roll ${PREROLL}s"
ffmpeg -y -loglevel error -ss "$PREROLL" -i "$VIDEO" -i "tts/audio/v${N}-narration.wav" \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k \
  -map 0:v:0 -map 1:a:0 -shortest "$OUT/${SLUG}-narrated.mp4"
echo "== wrote $OUT/${SLUG}-narrated.mp4 ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/${SLUG}-narrated.mp4")s)"
