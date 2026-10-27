#!/usr/bin/env python3
"""A tiny stdio MCP server for tests: a toy "deploy service" with four tools.

* ``list_services`` (readOnlyHint), ``get_service {name}`` (readOnlyHint),
  ``create_service {name, tier enum}`` (destructiveHint false), ``delete_service {name}`` (destructiveHint true)
* ``tools/list`` paginates in two pages (``nextCursor``)
* every ``tools/call`` appends one JSON line to the file named by env ``FAKE_MCP_CALL_LOG``
* env ``FAKE_MCP_NOISE=1`` prints a non-JSON line and a notification first; ``FAKE_MCP_PING=1`` sends a
  server ping before answering ``initialize`` and waits for the client's reply
* env ``FAKE_MCP_LEAK=<text>`` writes the text to stderr and exits at ``initialize`` (redaction tests)
* env ``FAKE_MCP_HANG=1`` never answers ``initialize`` (timeout tests)
"""

import json
import os
import sys
import time

TOOLS = [
    {
        "name": "list_services",
        "description": "List all services.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "get_service",
        "description": "Fetch one service.",
        "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "create_service",
        "description": "Create a service.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "tier": {"type": "string", "enum": ["free", "pro", "enterprise"]}},
            "required": ["name", "tier"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False},
    },
    {
        "name": "delete_service",
        "description": "Delete a service.",
        "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        "annotations": {"destructiveHint": True},
    },
]
PAGES = [TOOLS[:2], TOOLS[2:]]


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    log = os.environ.get("FAKE_MCP_CALL_LOG")
    if os.environ.get("FAKE_MCP_NOISE"):
        sys.stdout.write("starting up (not json)\n")
        sys.stdout.flush()
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "hello"}})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        method, rid = msg.get("method"), msg.get("id")
        if method == "initialize":
            if os.environ.get("FAKE_MCP_LEAK"):
                sys.stderr.write("fatal: bad config " + os.environ["FAKE_MCP_LEAK"] + "\n")
                sys.stderr.flush()
                sys.exit(3)
            if os.environ.get("FAKE_MCP_HANG"):
                time.sleep(60)
            if os.environ.get("FAKE_MCP_PING"):
                send({"jsonrpc": "2.0", "id": "srv-ping", "method": "ping"})
                sys.stdin.readline()  # the client's pong
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "deploy-service-fake", "version": "0.1"},
            }})
        elif method == "tools/list":
            cursor = (msg.get("params") or {}).get("cursor")
            page = 1 if cursor == "page2" else 0
            result = {"tools": PAGES[page]}
            if page == 0:
                result["nextCursor"] = "page2"
            send({"jsonrpc": "2.0", "id": rid, "result": result})
        elif method == "tools/call":
            params = msg.get("params") or {}
            if log:
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"name": params.get("name"), "arguments": params.get("arguments")}) + "\n")
            name = params.get("name")
            if name not in {t["name"] for t in TOOLS}:
                send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"unknown tool {name}"}})
                continue
            payload = {"ok": True, "tool": name, "arguments": params.get("arguments")}
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload,
                "isError": False,
            }})
        elif rid is not None and method:
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "method not found"}})


if __name__ == "__main__":
    main()
