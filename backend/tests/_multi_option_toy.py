"""A toy option verifier for the ``multi_option`` tests: a config fix delivered in two formats.

* ``config_file`` -- the whole edited file, ``key=value`` per line;
* ``commands``    -- ``set <key> <value>`` lines applied to the starting config.

The row's ``verifier_payload`` = ``{"start": {k: v}, "must_change": {k: v}, "must_keep": {k: v}}``
(the hidden assertions). Each format's checks: ``parses``, ``behaviour_changed`` (every must_change
key has its new value), ``behaviour_preserved`` (every must_keep key is untouched). The formats agree
when they end in the same state.

Strategies that misbehave on purpose: ``exit-nonzero`` (exit 3), ``bad-json`` (prints garbage),
``sleep`` (sleeps past any timeout).
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

VERIFIER_SOURCE = r'''
import json, sys, time

def parse_file(text):
    state = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            return None
        k, v = line.split("=", 1)
        state[k.strip()] = v.strip()
    return state

def apply_commands(start, text):
    state = dict(start)
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] != "set" or len(parts) != 3:
            return None
        state[parts[1]] = parts[2]
    return state

def checks_for(state, payload):
    if state is None:
        return {"parses": False, "behaviour_changed": None, "behaviour_preserved": None}
    changed = all(state.get(k) == v for k, v in payload.get("must_change", {}).items())
    kept = all(state.get(k) == v for k, v in payload.get("must_keep", {}).items())
    return {"parses": True, "behaviour_changed": changed, "behaviour_preserved": kept}

req = json.loads(sys.stdin.read())
option, payload = req["option"], req["payload"] or {}
strategy = option.get("strategy")
if strategy == "exit-nonzero":
    sys.stderr.write("boom\n")
    sys.exit(3)
if strategy == "bad-json":
    print("not json at all")
    sys.exit(0)
if strategy == "sleep":
    time.sleep(30)
fixes = option.get("fixes") or {}
states, out = {}, {}
for fmt in req["formats"]:
    text = fixes.get(fmt) or ""
    if fmt == "config_file":
        st = parse_file(text)
    elif fmt == "commands":
        st = apply_commands(payload.get("start", {}), text)
    else:
        st = None
    states[fmt] = st
    checks = checks_for(st, payload)
    passes = all(v is True for v in checks.values())
    out[fmt] = {"checks": checks, "passes": passes,
                "messages": [] if passes else [f"{fmt}: a check failed"]}
ends = [json.dumps(s, sort_keys=True) for s in states.values() if s is not None]
agree = len(ends) == len(states) and len(set(ends)) <= 1
diffs = [] if agree else ["the formats end in different states"]
print(json.dumps({"formats": out, "agreement": {"agree": agree, "differences": diffs},
                  "option_passes": all(o["passes"] for o in out.values()) and agree}))
'''

DETECTOR_SOURCE = r'''
import json, sys
code = sys.stdin.read()
findings = [{"label": "debug.enabled", "start_line": i + 1, "end_line": i + 1}
            for i, line in enumerate(code.splitlines()) if "debug=on" in line]
print(json.dumps({"findings": findings}))
'''

FORMATS = ["config_file", "commands"]
TAXONOMY = ["turn-off", "restrict", "require-auth"]
START = {"debug": "on", "port": "80", "mode": "open"}
PAYLOAD = {"start": START, "must_change": {"debug": "off"}, "must_keep": {"port": "80"}}


def write_script(directory: Path, name: str, source: str) -> Path:
    path = directory / name
    path.write_text(f"#!{sys.executable}\n" + source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def write_operator_file(directory: Path, *, revision: int = 1, timeout_s: float = 20) -> Path:
    verifier = write_script(directory, "toy_verify_option", VERIFIER_SOURCE)
    detector = write_script(directory, "toy_detect", DETECTOR_SOURCE)
    entries = [
        {"id": "toy_option_verifier", "kind": "option_verifier", "protocol": 1, "argv": [str(verifier)],
         "formats": FORMATS, "timeout_s": timeout_s, "cpu_s": 20, "mem_mb": 1024, "revision": revision},
        {"id": "toy_detector", "kind": "detector", "protocol": 1, "argv": [str(detector)], "languages": ["ini"],
         "timeout_s": 20, "cpu_s": 20, "mem_mb": 1024, "revision": 1},
    ]
    path = directory / "external_checks.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    os.utime(path, None)
    return path


def config_text(state: dict) -> str:
    return "".join(f"{k}={v}\n" for k, v in state.items())


def good_option(strategy: str = "turn-off") -> dict:
    end = {**START, "debug": "off"}
    return {"strategy": strategy, "title": "Turn the debug switch off", "rationale": "Debug output leaks internals.",
            "tradeoffs": "No debug output.", "rollback": "set debug on",
            "fixes": {"config_file": config_text(end), "commands": "set debug off\n"}}


def restrict_option() -> dict:
    end = {**START, "debug": "off", "mode": "restricted"}
    return {"strategy": "restrict", "title": "Restrict and turn off", "rationale": "x",
            "fixes": {"config_file": config_text(end), "commands": "set debug off\nset mode restricted\n"}}


def disagreeing_option() -> dict:
    end = {**START, "debug": "off"}
    return {"strategy": "require-auth", "title": "Formats disagree", "rationale": "x",
            "fixes": {"config_file": config_text(end), "commands": "set debug off\nset mode closed\n"}}


def breaking_option() -> dict:
    end = {**START, "debug": "off", "port": "8080"}
    return {"strategy": "restrict", "title": "Changes the port too", "rationale": "x",
            "fixes": {"config_file": config_text(end), "commands": "set debug off\nset port 8080\n"}}


def answer(*options: dict) -> str:
    return json.dumps({"explanation": "The debug switch is on.", "options": list(options)})
