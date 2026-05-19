#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT="${*:-Hello from iOS}"

swift "$ROOT_DIR/scripts/smoke_inference.swift" "$PROMPT"
