#!/usr/bin/env python3
"""Fail when the ci-success gate's ``needs:`` list doesn't name every CI job.

GitHub Actions has no wildcard for ``needs``, so the aggregate gate lists
its jobs by hand — a job added to ``ci.yml`` but not to that list can fail
without failing the gate. This check makes the list the single source:
it must equal the set of every other job in the workflow.

Stdlib only (runs before any install): parses the workflow's two-space
job keys and the gate's ``needs`` block by line, so keep that layout.

Usage:
    python3 scripts/check_ci_gate.py [.github/workflows/ci.yml]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

GATE_JOB = "ci-success"
DEFAULT_WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"

_JOB_RE = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
_NEEDS_ITEM_RE = re.compile(r"^      - ([A-Za-z0-9_-]+)\s*$")


def jobs_and_gate_needs(text: str) -> tuple[list[str], list[str]]:
    jobs: list[str] = []
    needs: list[str] = []
    in_jobs = False
    current = None
    in_needs = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_jobs = line.rstrip() == "jobs:"
            continue
        if not in_jobs:
            continue
        match = _JOB_RE.match(line)
        if match:
            current = match.group(1)
            jobs.append(current)
            in_needs = False
            continue
        if current != GATE_JOB:
            continue
        if line.rstrip() == "    needs:":
            in_needs = True
            continue
        if in_needs:
            item = _NEEDS_ITEM_RE.match(line)
            if item:
                needs.append(item.group(1))
            else:
                in_needs = False
    return jobs, needs


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_WORKFLOW
    jobs, needs = jobs_and_gate_needs(path.read_text(encoding="utf-8"))
    if GATE_JOB not in jobs:
        print(f"{path}: no '{GATE_JOB}' job found", file=sys.stderr)
        return 1
    expected = set(jobs) - {GATE_JOB}
    missing = sorted(expected - set(needs))
    unknown = sorted(set(needs) - expected)
    if missing or unknown:
        if missing:
            print(f"{GATE_JOB}.needs is missing: {', '.join(missing)}", file=sys.stderr)
        if unknown:
            print(f"{GATE_JOB}.needs names no such job: {', '.join(unknown)}", file=sys.stderr)
        return 1
    print(f"{GATE_JOB} gates all {len(expected)} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
