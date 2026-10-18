#!/usr/bin/env bash
# U7 lane F2 — record the two narrated navigation videos on the restyled UI and mux the narration.
# Usage: record.sh 02|03
set -euo pipefail
export PATH=$HOME/.nvm/versions/node/v22.23.2/bin:$PATH
cd /home/anuragj/Desktop/GitHub/__SLM__
N="$1"
case "$N" in
  02) SPEC=tests/demo-recordings/02-brewslm-quickstart-narrated.spec.ts; SLUG=02-brewslm-quickstart ;;
  03) SPEC=tests/demo-recordings/03-support-faq-pipeline-narrated.spec.ts; SLUG=03-support-faq-pipeline ;;
  *) echo "02 or 03" >&2; exit 2 ;;
esac
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
