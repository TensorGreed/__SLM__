#!/usr/bin/env bash
#
# Sync the local `website/` source-of-truth into the public GitHub Pages
# repo (TensorGreed/__SLM__website). See website/DEPLOY.md for the full
# workflow.
#
# Usage:
#   ./website/sync-to-public.sh /path/to/__SLM__website
#
# Safe to re-run; idempotent. Refuses to sync if the destination doesn't
# look like a clone of the website repo (presence of CNAME).

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <path-to-public-website-repo>" >&2
    echo "  e.g. $0 ../__SLM__website" >&2
    exit 64
fi

DEST_REPO="$1"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$DEST_REPO" ]; then
    echo "error: destination '$DEST_REPO' is not a directory" >&2
    exit 65
fi

if [ ! -f "$DEST_REPO/CNAME" ]; then
    echo "error: destination '$DEST_REPO' doesn't look like the website repo" >&2
    echo "       (missing CNAME). Aborting to prevent accidental sync." >&2
    exit 66
fi

if [ ! -d "$DEST_REPO/.git" ]; then
    echo "error: destination '$DEST_REPO' is not a git repo. Aborting." >&2
    exit 67
fi

echo "Syncing $SRC_DIR/  →  $DEST_REPO/"
echo

# --delete so files removed locally (the 10 SEO blog posts) also disappear
# from the public repo. Exclude repo-internal helpers + .git on both sides.
rsync -av --delete \
    --exclude='.git/' \
    --exclude='DEPLOY.md' \
    --exclude='sync-to-public.sh' \
    --exclude='CNAME' \
    --exclude='.nojekyll' \
    "$SRC_DIR/" "$DEST_REPO/"

echo
echo "Sync done. Status in $DEST_REPO:"
echo
cd "$DEST_REPO"
git status --short

cat <<'EOF'

Next steps:
  cd <path-to-public-website-repo>
  git diff               # review the changes
  git add -A
  git commit -m "Sync website from __SLM__ main"
  git push origin main

GitHub Pages will redeploy in ~60s. brewslm.com is served from main.
EOF
