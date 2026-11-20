#!/usr/bin/env bash
# Regenerate frontend/src/components/evaluation/EvalRunForm.legacyBodies.json: the run-heldout bodies the LONG eval
# form (EvalPanel at e845d0a, before the short "Score this run" form) posts for each scenario in
# legacyEvalBodies.capture.tsx. EvalRunForm.test.tsx asserts the short form posts the same bodies.
#
# Usage (from anywhere in the repo):  bash frontend/scripts/legacy-eval-capture/capture.sh [OUT_JSON]
# Needs frontend/node_modules (npm ci) and Node >= 20.19. Checks e845d0a out into a temporary git worktree,
# links this tree's node_modules into it, runs the harness there with vitest, and removes the worktree.
set -euo pipefail

LEGACY_SHA="${LEGACY_SHA:-e845d0a}"
REPO="$(git rev-parse --show-toplevel)"
HERE="$REPO/frontend/scripts/legacy-eval-capture"
OUT="$(realpath -m "${1:-$REPO/frontend/src/components/evaluation/EvalRunForm.legacyBodies.json}")"
TMP="$(mktemp -d)"
TREE="$TMP/legacy"

cleanup() {
    git -C "$REPO" worktree remove --force "$TREE" >/dev/null 2>&1 || true
    rm -rf "$TMP"
}
trap cleanup EXIT

git -C "$REPO" worktree add --detach "$TREE" "$LEGACY_SHA" >/dev/null
ln -s "$(realpath "$REPO/frontend/node_modules")" "$TREE/frontend/node_modules"
cp "$HERE/legacyEvalBodies.capture.tsx" "$TREE/frontend/src/components/evaluation/legacyEvalBodies.capture.test.tsx"

(cd "$TREE/frontend" && LEGACY_CAPTURE_OUT="$OUT" npx vitest run src/components/evaluation/legacyEvalBodies.capture.test.tsx)
echo "wrote $OUT"
