#!/usr/bin/env bash
# Install the project's git hooks into .git/hooks/.
#
# Run once after cloning, or any time you want to refresh the hooks:
#
#     bash scripts/install-git-hooks.sh
#
# Currently installs:
#   * pre-commit — regenerates INDEX.md when staged source files
#     change (see scripts/pre-commit-hook.sh for the body).

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_SRC="$REPO_ROOT/scripts/pre-commit-hook.sh"
HOOK_DST="$REPO_ROOT/.git/hooks/pre-commit"

if [ ! -f "$HOOK_SRC" ]; then
    echo "ERROR: $HOOK_SRC not found." >&2
    exit 1
fi

if [ -f "$HOOK_DST" ] && [ ! -L "$HOOK_DST" ]; then
    # Existing non-symlink hook — back it up so we don't clobber
    # something the user wrote manually.
    BACKUP="$HOOK_DST.bak.$(date +%Y%m%d%H%M%S)"
    cp "$HOOK_DST" "$BACKUP"
    echo "Backed up existing pre-commit hook to $BACKUP"
fi

cp "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_DST"

echo "Installed pre-commit hook → $HOOK_DST"
echo "  Source: $HOOK_SRC"
echo ""
echo "On next commit that touches .py/.ts/.tsx/.sh files, INDEX.md will"
echo "be regenerated automatically and added to the commit."
