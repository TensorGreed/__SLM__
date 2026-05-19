#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT="${*:-Hello from Android}"

if command -v kotlin >/dev/null 2>&1; then
  kotlin "$ROOT_DIR/scripts/smoke_inference.kts" "$PROMPT"
elif command -v kotlinc >/dev/null 2>&1; then
  kotlinc -script "$ROOT_DIR/scripts/smoke_inference.kts" -- "$PROMPT"
else
  echo "Kotlin runtime not found. Install kotlin or run inside Android Studio terminal."
  exit 1
fi
