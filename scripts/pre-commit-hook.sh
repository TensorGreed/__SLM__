#!/usr/bin/env bash
# Auto-regenerate INDEX.md when staged changes touch source files.
#
# Tracked here so the hook survives `git clean` / fresh clones.
# Install (or re-install) via:
#
#     bash scripts/install-git-hooks.sh
#
# That copies this file to .git/hooks/pre-commit + chmods it.
#
# Behavior:
#   * Only fires when at least one staged file is .py / .ts / .tsx / .sh
#     (skips pure CSS / markdown / config commits — no index churn).
#   * Regenerates INDEX.md via scripts/regenerate_index.py.
#   * Auto-stages the regenerated INDEX.md so the commit includes it.
#   * NEVER blocks a commit on its own failure — if the regen script
#     errors, we print a warning and let the commit through. The index
#     can drift; broken commits would be worse.

set -e

# Only run when source files are part of the commit.
if ! git diff --cached --name-only --diff-filter=ACMR \
        | grep -qE '\.(py|ts|tsx|sh)$'; then
    exit 0
fi

# Find python3 in PATH (alembic envs often use uv / venv).
if ! command -v python3 >/dev/null 2>&1; then
    echo "pre-commit: python3 not on PATH; skipping INDEX.md regen" >&2
    exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT="$REPO_ROOT/scripts/regenerate_index.py"

if [ ! -f "$SCRIPT" ]; then
    # Script removed — nothing to do.
    exit 0
fi

if ! python3 "$SCRIPT" >/dev/null 2>&1; then
    echo "pre-commit: regenerate_index.py failed; INDEX.md may be stale" >&2
    echo "pre-commit: re-run manually with 'python3 scripts/regenerate_index.py'" >&2
    exit 0
fi

# Stage the (possibly unchanged) INDEX.md. If nothing actually
# changed, `git add` is a no-op and no extra diff lands in the commit.
git add "$REPO_ROOT/INDEX.md"
