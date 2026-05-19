#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANDROID_DIR="$ROOT_DIR/android"

if [ -x "$ANDROID_DIR/gradlew" ]; then
  (cd "$ANDROID_DIR" && ./gradlew :app:assembleDebug)
elif command -v gradle >/dev/null 2>&1; then
  (cd "$ANDROID_DIR" && gradle :app:assembleDebug)
else
  echo "Gradle not found. Open '$ANDROID_DIR' in Android Studio and run Sync Project."
  exit 0
fi

echo "Build finished. Open '$ANDROID_DIR' in Android Studio to run on emulator/device."
