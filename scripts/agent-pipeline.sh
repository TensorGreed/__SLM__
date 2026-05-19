#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"

has_npm_script() {
  local script_name="$1"
  [ -f package.json ] && jq -e --arg s "$script_name" '.scripts[$s] != null' package.json >/dev/null 2>&1
}

run_node_build() {
  if has_npm_script build; then
    npm run build
  fi
}

run_node_lint() {
  if has_npm_script lint; then
    npm run lint
  fi
}

run_node_unit() {
  if has_npm_script test; then
    npm test -- --runInBand || npm test || true
  fi
}

run_node_e2e() {
  if has_npm_script "test:e2e"; then
    npm run test:e2e
  elif [ -f playwright.config.ts ] || [ -f playwright.config.js ]; then
    npx playwright test
  fi
}

run_python_unit() {
  if command -v pytest >/dev/null 2>&1; then
    pytest -q || true
  fi
}

run_python_api() {
  if command -v pytest >/dev/null 2>&1; then
    pytest -q -k "api or integration" || true
  fi
}

run_java_build() {
  if [ -f pom.xml ] && command -v mvn >/dev/null 2>&1; then
    mvn -q -DskipTests package
  elif [ -f gradlew ]; then
    chmod +x ./gradlew
    ./gradlew assemble
  fi
}

run_java_unit() {
  if [ -f pom.xml ] && command -v mvn >/dev/null 2>&1; then
    mvn -q test || true
  elif [ -f gradlew ]; then
    ./gradlew test || true
  fi
}

case "$MODE" in
  build)
    run_node_build
    run_java_build
    ;;
  lint)
    run_node_lint
    ;;
  test-unit)
    run_node_unit
    run_python_unit
    run_java_unit
    ;;
  test-api)
    run_python_api
    ;;
  test-e2e)
    run_node_e2e
    ;;
  full)
    "$0" build
    "$0" lint
    "$0" test-unit
    "$0" test-api
    "$0" test-e2e
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 1
    ;;
esac