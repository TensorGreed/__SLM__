#!/usr/bin/env python3
"""Export the metric vocabulary to frontend/src/generated/metricVocabulary.json.

``backend/app/services/metric_vocabulary.py`` is the one source for metric names, scales and the
score-of-record keys; the frontend reads this generated copy (UI simplify phase 7k).

Usage:
    python3 scripts/export_metric_vocabulary.py            # writes the JSON
    python3 scripts/export_metric_vocabulary.py --check    # exits 1 if stale

The module imports only the standard library, so it is loaded by path (no backend dependencies needed;
CI's ``repo-checks`` job runs ``--check``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "backend" / "app" / "services" / "metric_vocabulary.py"
TARGET = ROOT / "frontend" / "src" / "generated" / "metricVocabulary.json"


def render() -> str:
    spec = importlib.util.spec_from_file_location("_metric_vocabulary", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return json.dumps(module.export_payload(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when the JSON is stale")
    args = parser.parse_args()
    text = render()
    rel = TARGET.relative_to(ROOT)
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{rel} is stale: run python3 scripts/export_metric_vocabulary.py", file=sys.stderr)
            return 1
        print(f"{rel} is up to date")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
