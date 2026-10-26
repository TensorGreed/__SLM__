"""A small ``robots.txt`` matcher following RFC 9309 (no library).

- Groups are chosen by the crawler's product token (the part of the
  user-agent before ``/`` or a space), case-insensitively; with no matching
  group the ``*`` group applies; with neither everything is allowed. Several
  groups naming the same agent are merged.
- Within a group the longest matching rule wins; ``Allow`` wins a tie.
- ``*`` matches any run of characters, a trailing ``$`` anchors the end.
- Both the rule and the URL path are percent-encoding normalised before
  comparing (``%7e`` == ``~``, ``%2f`` == ``%2F``, raw non-ASCII == its UTF-8
  escape).

A file bigger than :data:`MAX_ROBOTS_BYTES` is read up to that size, as the
RFC allows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

MAX_ROBOTS_BYTES = 500 * 1024
_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_PERCENT = re.compile(r"%([0-9a-fA-F]{2})")


def product_token(user_agent: str) -> str:
    """``BrewSLM-crawler/1.0 (+https://x)`` → ``brewslm-crawler``."""
    return re.split(r"[/\s]", user_agent.strip(), maxsplit=1)[0].lower()


def normalize_path(value: str) -> str:
    """Canonical percent-encoding: unreserved characters decoded, other
    escapes upper-cased, raw non-ASCII / unsafe bytes escaped."""
    out: list[str] = []
    i = 0
    while i < len(value):
        ch = value[i]
        match = _PERCENT.match(value, i) if ch == "%" else None
        if match:
            decoded = chr(int(match.group(1), 16))
            out.append(decoded if decoded in _UNRESERVED else "%" + match.group(1).upper())
            i += 3
        elif ch.isascii() and (ch.isalnum() or ch in "-._~/:@!$&'()*+,;=?[]%"):
            out.append(ch)
            i += 1
        else:
            out.extend(f"%{byte:02X}" for byte in ch.encode("utf-8"))
            i += 1
    return "".join(out)


@dataclass
class _Rule:
    allow: bool
    pattern: str
    regex: re.Pattern[str]

    @property
    def specificity(self) -> int:
        return len(self.pattern)


def _compile(pattern: str) -> re.Pattern[str]:
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = ".*".join(re.escape(part) for part in body.split("*"))
    return re.compile(regex + (r"\Z" if anchored else ""))


@dataclass
class RobotsPolicy:
    rules: list[_Rule] = field(default_factory=list)
    crawl_delay: float | None = None

    def allows(self, url_or_path: str) -> bool:
        if "://" in url_or_path:
            parts = urlsplit(url_or_path)
            path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        else:
            path = url_or_path or "/"
        target = normalize_path(path)
        best: _Rule | None = None
        for rule in self.rules:
            if not rule.regex.match(target):
                continue
            if best is None or rule.specificity > best.specificity or (
                rule.specificity == best.specificity and rule.allow and not best.allow
            ):
                best = rule
        return True if best is None else best.allow


def parse_robots(text: str, user_agent: str) -> RobotsPolicy:
    """The policy ``user_agent`` must follow under ``text``."""
    token = product_token(user_agent)
    # groups: list of (agents, rules, crawl_delay)
    groups: list[tuple[list[str], list[tuple[bool, str]], float | None]] = []
    agents: list[str] = []
    rules: list[tuple[bool, str]] = []
    delay: float | None = None
    in_rules = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "user-agent":
            if in_rules:
                groups.append((agents, rules, delay))
                agents, rules, delay, in_rules = [], [], None, False
            agents.append(value.lower())
        elif key in {"allow", "disallow"}:
            in_rules = True
            if value:  # an empty Disallow allows everything; an empty Allow is a no-op
                rules.append((key == "allow", value))
        elif key == "crawl-delay":
            in_rules = True
            try:
                delay = float(value)
            except ValueError:
                pass
    if agents:
        groups.append((agents, rules, delay))

    def collect(name: str) -> RobotsPolicy | None:
        matched = [g for g in groups if name in g[0]]
        if not matched:
            return None
        policy = RobotsPolicy()
        for _, group_rules, group_delay in matched:
            for allow, pattern in group_rules:
                normalized = normalize_path(pattern.rstrip("$")) + ("$" if pattern.endswith("$") else "")
                policy.rules.append(_Rule(allow, normalized, _compile(normalized)))
            if group_delay is not None:
                policy.crawl_delay = group_delay
        return policy

    return collect(token) or collect("*") or RobotsPolicy()
