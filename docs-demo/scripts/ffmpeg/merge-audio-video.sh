#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <video-path> <audio-path> <output-path>" >&2
  exit 2
fi

video_path="$1"
audio_path="$2"
output_path="$3"

if [ ! -f "$video_path" ]; then
  echo "Video file not found: $video_path" >&2
  exit 2
fi

if [ ! -f "$audio_path" ]; then
  echo "Audio file not found: $audio_path" >&2
  exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found on PATH." >&2
  exit 2
fi

mkdir -p "$(dirname "$output_path")"
ffmpeg -y -i "$video_path" -i "$audio_path" -c:v copy -c:a aac -shortest "$output_path"
echo "Merged video written to $output_path"

