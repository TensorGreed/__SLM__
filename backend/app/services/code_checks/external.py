"""External (sandboxed) syntax checks. Each is gated by CODE_CHECK_ALLOWED + PATH.

Chunks are fragments, so only *syntax-class* diagnostics count as failures;
resolution errors ("cannot find symbol", "package does not exist", ...) are ignored.
The sandbox is called as ``sandbox_exec.run_sandboxed`` (module attribute) so tests
patch one place.
"""

from __future__ import annotations

import re
import shutil
from typing import Any

from app.services import sandbox_exec

from .base import BaseCheck, make_result

_JAVA_SYNTAX_RE = re.compile(
    r"expected|illegal start|not a statement|reached end of file|unclosed|orphaned|"
    r"class, interface, enum, or record expected|invalid method declaration|illegal character|"
    r"malformed|unterminated|without '?(if|try)'?|illegal escape|empty character literal|"
    r"variable declaration not allowed here|illegal (?:start|unicode)|'else' without|"
    r"modifier .* not allowed here|repeated modifier|text block",
    re.IGNORECASE,
)
_JAVA_DIAG_RE = re.compile(r"^(?:[^\s:]+\.java):(\d+):\s*error:\s*(.*)$")
_JAVA_TOPLEVEL_RE = re.compile(
    r"^(?:(?:@\w+(?:\([^)]*\))?\s+)*)(?:(?:public|protected|private|abstract|final|static|sealed|non-sealed|strictfp)\s+)*"
    r"(?:class|interface|enum|record|@interface)\s+\w+|^(?:package|import)\s",
)
_JAVA_SENTINEL = "__brewslm_end_sentinel"


def _java_first_code_line(code: str) -> str | None:
    """First line that is neither blank nor a comment (column 0 preserved)."""
    in_block = False
    for ln in code.splitlines():
        st = ln.strip()
        if in_block:
            if "*/" in st:
                in_block = False
            continue
        if not st or st.startswith("//"):
            continue
        if st.startswith("/*"):
            in_block = "*/" not in st
            continue
        return ln
    return None


def java_is_toplevel(code: str) -> bool:
    """A compilation unit: its first real line is a column-0 package / import / type declaration
    (leading statements, or an indented local class inside statements, do not count)."""
    first = _java_first_code_line(code)
    return bool(first and _JAVA_TOPLEVEL_RE.match(first))


def java_brace_balance(code: str, *, go: bool = False) -> tuple[int | None, int]:
    """``(first line where depth goes negative | None, final depth)`` counting ``{``/``}`` outside
    string / char literals, text blocks and comments (a small lexer). ``go=True`` swaps Java text
    blocks for Go's backtick raw strings."""
    depth = 0
    neg_line: int | None = None
    line = 1
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if c == "\n":
            line += 1
            i += 1
        elif code.startswith("//", i):
            while i < n and code[i] != "\n":
                i += 1
        elif code.startswith("/*", i):
            j = code.find("*/", i + 2)
            j = n if j < 0 else j + 2
            line += code.count("\n", i, j)
            i = j
        elif go and c == "`":
            j = code.find("`", i + 1)
            j = n if j < 0 else j + 1
            line += code.count("\n", i, j)
            i = j
        elif not go and code.startswith('"""', i):
            j = i + 3
            while j < n and not code.startswith('"""', j):
                j += 2 if code[j] == "\\" else 1
            j = min(n, j + 3)
            line += code.count("\n", i, j)
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and code[j] != c and code[j] != "\n":
                j += 2 if code[j] == "\\" else 1
            i = j + 1
        else:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth < 0 and neg_line is None:
                    neg_line = line
            i += 1
    return neg_line, depth


# C# error codes that are lexical / syntactic (best effort).
_CS_SYNTAX_CODES = frozenset(
    f"CS{n}" for n in (
        1001, 1002, 1003, 1004, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1021, 1022,
        1026, 1031, 1032, 1033, 1034, 1035, 1037, 1038, 1039, 1040, 1041, 1056, 1073, 1513, 1514, 1515, 1517,
        1518, 1519, 1520, 1525, 1528, 1551, 1552, 1553, 1576, 1585, 1586, 1587, 8124,
    )
)


def java_syntax_diagnostics(stderr: str) -> tuple[list[tuple[int, str]], list[str]]:
    """javac stderr -> ([(line, "line N: msg")] syntax diagnostics, ignored resolution diagnostics)."""
    syntax: list[tuple[int, str]] = []
    ignored: list[str] = []
    for line in stderr.splitlines():
        m = _JAVA_DIAG_RE.match(line.strip())
        if not m:
            continue
        text = f"line {m.group(1)}: {m.group(2)}"
        if _JAVA_SYNTAX_RE.search(m.group(2)):
            syntax.append((int(m.group(1)), text))
        else:
            ignored.append(text)
    return syntax, ignored


def java_syntax_messages(stderr: str) -> tuple[list[str], list[str]]:
    """Split javac stderr into (syntax diagnostics, ignored resolution diagnostics)."""
    syntax, ignored = java_syntax_diagnostics(stderr)
    return [t for _l, t in syntax], ignored


def _unavailable(check_id: str, err: Exception) -> dict[str, Any]:
    return make_result(check_id, None, [str(err)], skipped="unavailable")


class ExternalCheck(BaseCheck):
    kind = "external"
    timeout_s = 10.0
    cpu_s = 10
    mem_mb = 1024

    def _sandbox(self, argv, files, stdin=None, **kw):
        return sandbox_exec.run_sandboxed(
            argv, files=files, timeout_s=kw.pop("timeout_s", self.timeout_s), cpu_s=kw.pop("cpu_s", self.cpu_s),
            mem_mb=kw.pop("mem_mb", self.mem_mb), stdin=stdin, **kw,
        )

    def run(self, code, *, language, context):
        try:
            return self._run(code, language, context or {})
        except sandbox_exec.SandboxError as exc:
            return _unavailable(self.id, exc)

    def _run(self, code, language, context):  # pragma: no cover
        raise NotImplementedError

    def _timeout(self):
        return make_result(self.id, None, [f"timed out after {self.timeout_s:g}s"], skipped="timeout")


class JavaSyntax(ExternalCheck):
    id = "java_syntax"
    languages = frozenset({"java"})
    binary = "javac"
    timeout_s = 30.0
    cpu_s = 30
    mem_mb = 2048

    @staticmethod
    def candidates(code: str) -> list[str]:
        """Sources to try. Wrapped forms keep the code starting on line 1 and end with a sentinel
        member on the line after the code, so a fragment that closes more braces than it opened
        (or leaves some open) shows up as a syntax error at / after the sentinel line."""
        if java_is_toplevel(code):
            return [code]
        tail = f"\n int {_JAVA_SENTINEL}; }}"
        return [
            "class Snippet { " + code + tail,
            "class Snippet { void __snippet() { " + code + "\n} int " + _JAVA_SENTINEL + "; }",
        ]

    def _run(self, code, language, context):
        cands = self.candidates(code)
        wrapped = not (len(cands) == 1 and cands[0] == code)
        sentinel_line = code.count("\n") + 2  # the line after the code
        best: list[str] | None = None
        for src in cands:
            if wrapped:
                neg_line, final = java_brace_balance(code)
                if neg_line is not None:
                    msg = [f"line {neg_line}: unbalanced '}}' (closes more than the fragment opens)"]
                    best = msg if best is None or len(msg) < len(best) else best
                    continue
                if final != 0:
                    msg = [f"unbalanced braces: {final} unclosed '{{'"]
                    best = msg if best is None or len(msg) < len(best) else best
                    continue
            res = self._sandbox(
                ["javac", "-proc:none", "-Xlint:none", "-J-Xmx256m", "-d", ".", "Snippet.java"], {"Snippet.java": src}
            )
            if res.timed_out:
                return self._timeout()
            diags, _ignored = java_syntax_diagnostics(res.stderr)
            if wrapped:
                escaped = [t for ln, t in diags if ln >= sentinel_line]
                if escaped:
                    best = escaped if best is None or len(escaped) < len(best) else best
                    continue
            if res.returncode == 0 or not diags:
                return make_result(self.id, True)  # only resolution errors: syntactically fine
            msgs = [t for _l, t in diags]
            if best is None or len(msgs) < len(best):
                best = msgs
        return make_result(self.id, False, best or ["syntax error"])


_GO_PKG_RE = re.compile(r"^\s*package\s+\w+", re.MULTILINE)
_GO_PREFIX = "package snippet; "
_GO_FUNC_PREFIX = "package snippet; func _snippet() { "
_GO_DIAG_RE = re.compile(r"^<standard input>:(\d+):(\d+):\s*(.*)$")


def _remap_go_messages(stderr: str, prefix_len: int) -> list[str]:
    """gofmt diagnostics -> ``line L:C: msg`` against the ORIGINAL code (the wrapper prefix sits on
    line 1, so only line-1 columns shift)."""
    out: list[str] = []
    for ln in stderr.splitlines():
        if not ln.strip():
            continue
        m = _GO_DIAG_RE.match(ln.strip())
        if not m:
            out.append(ln.strip())
            continue
        line, col = int(m.group(1)), int(m.group(2))
        if line == 1:
            col = max(1, col - prefix_len)
        out.append(f"line {line}:{col}: {m.group(3)}")
    return out


class GoSyntax(ExternalCheck):
    id = "go_syntax"
    languages = frozenset({"go"})
    binary = "gofmt"

    @staticmethod
    def candidates(code: str) -> list[str]:
        if _GO_PKG_RE.search(code):
            return [code]
        return [_GO_PREFIX + code, _GO_FUNC_PREFIX + code + "\n}"]

    def _run(self, code, language, context):
        cands = self.candidates(code)
        prefixes = [0] if len(cands) == 1 else [len(_GO_PREFIX), len(_GO_FUNC_PREFIX)]
        failures: list[list[str]] = []
        if len(cands) > 1:
            # A wrapped fragment must not close the wrapper (``}\nfunc g() {`` would otherwise parse).
            neg_line, final = java_brace_balance(code, go=True)
            if neg_line is not None:
                return make_result(self.id, False, [f"line {neg_line}: unbalanced '}}' (closes more than the fragment opens)"])
            if final != 0:
                return make_result(self.id, False, [f"unbalanced braces: {final} unclosed '{{'"])
        for src, plen in zip(cands, prefixes):
            res = self._sandbox(["gofmt", "-e"], {}, stdin=src)
            if res.timed_out:
                return self._timeout()
            if res.returncode == 0:
                return make_result(self.id, True)
            failures.append(_remap_go_messages(res.stderr, plen))
        # A fragment without a package clause is most plausibly statements: report the
        # function-wrapped candidate's messages (the last one tried).
        return make_result(self.id, False, (failures[-1] if failures else None) or ["syntax error"])


class GoVet(ExternalCheck):
    id = "go_vet"
    languages = frozenset({"go"})
    binary = "go"
    timeout_s = 90.0
    cpu_s = 90
    mem_mb = 2048

    def _run(self, code, language, context):
        if not re.search(r"^\s*package\s+\w+", code, re.MULTILINE):
            return make_result(self.id, None, skipped="language_mismatch")  # whole files only
        # Per-call build cache / GOPATH inside the sandbox temp dir ("{tmp}"): nothing is shared
        # between calls (or with other tenants), and it is removed with the temp dir.
        res = self._sandbox(
            ["go", "vet", "./snippet.go"], {"snippet.go": code},
            env={"GOCACHE": "{tmp}/.gocache", "GOFLAGS": "-mod=mod", "GO111MODULE": "off", "GOPATH": "{tmp}/.gopath"},
        )
        if res.timed_out:
            return self._timeout()
        if res.returncode == 0:
            return make_result(self.id, True)
        return make_result(self.id, False, [ln for ln in res.stderr.splitlines() if ln.strip() and not ln.startswith("#")])


class TerraformFmt(ExternalCheck):
    id = "terraform_fmt"
    languages = frozenset({"terraform"})
    binary = "terraform"

    def _run(self, code, language, context):
        res = self._sandbox(["terraform", "fmt", "-check", "-no-color", "-"], {}, stdin=code)
        if res.timed_out:
            return self._timeout()
        if res.returncode == 0:
            return make_result(self.id, True)
        return make_result(self.id, False, [ln for ln in (res.stderr or res.stdout).splitlines() if ln.strip()] or ["not canonically formatted"])


class TerraformValidate(ExternalCheck):
    id = "terraform_validate"
    languages = frozenset({"terraform"})
    binary = "terraform"
    timeout_s = 30.0

    def _run(self, code, language, context):
        module_dir = context.get("module_dir")
        if not module_dir:
            return make_result(self.id, None, skipped="no_module_dir")
        # The module is read-only input; the candidate file is written into a copy by the caller's context.
        res = self._sandbox(["terraform", f"-chdir={module_dir}", "validate", "-no-color"], {})
        if res.timed_out:
            return self._timeout()
        if res.returncode == 0:
            return make_result(self.id, True)
        return make_result(self.id, False, [ln for ln in (res.stdout + "\n" + res.stderr).splitlines() if ln.strip()])


class Shellcheck(ExternalCheck):
    id = "shellcheck"
    languages = frozenset({"shell"})
    binary = "shellcheck"

    def _run(self, code, language, context):
        res = self._sandbox(["shellcheck", "-f", "gcc", "-"], {}, stdin=code)
        if res.timed_out:
            return self._timeout()
        if res.returncode == 0:
            return make_result(self.id, True)
        msgs = [ln.replace("-:", "line ", 1) for ln in res.stdout.splitlines() if ln.strip()]
        return make_result(self.id, False, msgs or [ln for ln in res.stderr.splitlines() if ln.strip()])


class NodeCheck(ExternalCheck):
    id = "node_check"
    languages = frozenset({"javascript"})
    binary = "node"

    def _run(self, code, language, context):
        res = self._sandbox(["node", "--check", "snippet.js"], {"snippet.js": code})
        if res.timed_out:
            return self._timeout()
        if res.returncode == 0:
            return make_result(self.id, True)
        lines = [ln for ln in res.stderr.splitlines() if ln.strip() and not ln.startswith("Node.js v")]
        return make_result(self.id, False, lines[:8] or ["syntax error"])


class CsharpSyntax(ExternalCheck):
    id = "csharp_syntax"
    languages = frozenset({"csharp"})
    binary = "csc"
    timeout_s = 30.0

    def available(self) -> bool:
        return shutil.which("csc") is not None or shutil.which("mcs") is not None

    def _run(self, code, language, context):
        exe = "csc" if shutil.which("csc") else "mcs"
        res = None
        for src in (code, "class Snippet { " + code + "\n}"):
            res = self._sandbox([exe, "-nologo", "-t:library", "-out:snippet.dll", "Snippet.cs"], {"Snippet.cs": src})
            if res.timed_out:
                return self._timeout()
            syntax = [ln for ln in (res.stdout + res.stderr).splitlines()
                      if "error" in ln and any(c in ln for c in _CS_SYNTAX_CODES)]
            if res.returncode == 0 or not syntax:
                return make_result(self.id, True)
        return make_result(self.id, False, syntax)


EXTERNALS: list[Any] = [
    JavaSyntax(), GoSyntax(), GoVet(), TerraformFmt(), TerraformValidate(), Shellcheck(), NodeCheck(), CsharpSyntax(),
]
