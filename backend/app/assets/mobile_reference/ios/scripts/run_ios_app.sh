#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IOS_DIR="$ROOT_DIR/ios"

if command -v xcodegen >/dev/null 2>&1; then
  (cd "$IOS_DIR" && xcodegen generate --quiet)
  echo "Generated: $IOS_DIR/SLMReference.xcodeproj"
  echo "Open the project in Xcode and run the SLMReference scheme on a simulator/device."
else
  echo "xcodegen not found."
  echo "Install xcodegen, or create an iOS project manually and add:"
  echo "  - ios/SLMReferenceApp.swift"
  echo "  - ios/SLMRuntime.swift"
  echo "  - ios/Info.plist"
fi
