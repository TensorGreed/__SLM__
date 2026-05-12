"""Dump the FastAPI OpenAPI spec to stdout (or --out path) as JSON.

Used by slm-docs/scripts/generate_api_ref.py to materialize per-tag
Markdown reference pages without needing a running backend.

Usage:

  python scripts/dump_openapi.py
  python scripts/dump_openapi.py --out openapi.json

Runs the FastAPI app's lifespan briefly to materialise routes; nothing
is written to the DB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=None,
        help="Output file path. Defaults to stdout.",
    )
    args = parser.parse_args()

    # Keep schema generation cheap + side-effect free: SQLite + no auth +
    # skip Alembic head requirement so we don't need a real DB.
    os.environ.setdefault("AUTH_ENABLED", "false")
    os.environ.setdefault("DEBUG", "false")
    os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
    os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    os.environ.setdefault("DATA_DIR", str(Path("/tmp/brewslm-openapi-tmp").resolve()))

    # Importing app.main wires every router; that's what we want.
    from app.main import app  # noqa: E402

    spec = app.openapi()
    payload = json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=False)

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"Wrote {len(payload):,} bytes to {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(payload)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
