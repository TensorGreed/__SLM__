"""``context_grounding``: every external symbol the code uses is grounded in the prompt context.

Task-agnostic. A *term extractor* (registry below: id -> ``fn(code, language) -> ExtractedTerms``) pulls
the symbols the code depends on (``terms``) and the names the code declares itself (``defined``). A term
passes when any of its candidate names

(a) appears in ``context["grounding_text"]`` (the full prompt the model saw) as a whole word -- also
    matched case-, ``_``- and ``-``-insensitively, so ``widgetId`` matches ``widget_id``;
(b) is defined in the answer itself (functions, classes, variables, parameters, fields, HCL variables /
    locals / resources / data sources declared in the answer); or
(c) is in the language's builtin / standard-library / keyword allowlist (``grounding_allowlists``).

Precision matters more than recall: anything the extractor is unsure about is not extracted. Standard
library roots (``os.path.x``, ``fmt.X``, ``std::x``) are never checked past the root, string literals and
comments are blanked first, YAML template expressions are neutralised, and YAML keys under author-named
parents (labels, variables, environment...) are skipped.

The first extractor is ``code_symbols``: per-language-family regex tables (C-like: C / C++ / C# / Java /
JS / TS / Go / Rust; Python; shell; HCL / Terraform; YAML keys) covering imports, bare calls, dotted
attribute paths (every segment after a non-standard root) and type names. A later extractor (plain text,
say) registers with ``register_term_extractor`` and is picked with ``context["grounding_extractor"]``.

The check is opt-in (``default_enabled = False``): it runs when its id is requested in ``check_ids`` or
listed in ``CODE_CHECK_ALLOWED``. Without ``grounding_text`` it is ``ok None, skipped "no_context"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from . import grounding_allowlists as al
from .base import BaseCheck, make_result, normalize_language

MAX_CODE_CHARS = 200_000
MAX_LINE_CHARS = 1000
MAX_PATH_SEGMENTS = 8  # a dotted path is checked up to its first 8 segments (long chains stay linear)
MAX_TERMS = 2000  # distinct terms per check; past it the rest are not checked (precision over recall)
_LONG_LINE = re.compile(r"^[^\n]{%d,}$" % (MAX_LINE_CHARS + 1), re.MULTILINE)

_IDENT = r"[A-Za-z_$][\w$]*"
_WORD = re.compile(r"[A-Za-z_$][\w$]*")


@dataclass(frozen=True)
class Term:
    display: str  # what the message names, e.g. ``client.fetch_widget``
    names: tuple[str, ...]  # passes when ANY of these is grounded / defined / allowlisted
    kind: str = "symbol"  # import | call | attribute | type | command | key


@dataclass
class ExtractedTerms:
    terms: list[Term] = field(default_factory=list)
    defined: set[str] = field(default_factory=set)  # declared in the answer: satisfies any term but an import
    locals: set[str] = field(default_factory=set)  # variables / parameters: satisfy bare names only
    std_bound: set[str] = field(default_factory=set)  # names bound by standard-library imports
    allow: frozenset[str] = frozenset()
    # A wildcard import of a standard package whose members are not listed: bare PascalCase type names pass.
    wildcard_std: bool = False
    _seen: set[Term] = field(default_factory=set, repr=False)

    def add(self, display: str, *names: str, kind: str = "symbol") -> None:
        """Add a term once (duplicates are dropped); at most ``MAX_TERMS`` distinct terms."""
        cands = tuple(n for n in names if n)
        if not cands or len(self.terms) >= MAX_TERMS:
            return
        term = Term(display, cands, kind)
        if term in self._seen:
            return
        self._seen.add(term)
        self.terms.append(term)


Extractor = Callable[[str, str], ExtractedTerms]
TERM_EXTRACTORS: dict[str, Extractor] = {}


def register_term_extractor(extractor_id: str, fn: Extractor) -> None:
    if not isinstance(extractor_id, str) or not extractor_id.strip() or not callable(fn):
        raise ValueError("a term extractor needs a non-empty id and a callable")
    TERM_EXTRACTORS[extractor_id] = fn


# ── language families ────────────────────────────────────────────────────

_FAMILY = {
    "c": "c", "cpp": "c", "csharp": "csharp", "java": "java", "javascript": "js", "typescript": "js",
    "go": "go", "rust": "rust", "python": "python", "shell": "shell", "terraform": "hcl", "yaml": "yaml",
}
GROUNDING_LANGUAGES = frozenset(_FAMILY)


def family_of(language: str | None) -> str | None:
    return _FAMILY.get(normalize_language(language))


# ── comment / string blanking ────────────────────────────────────────────


def _blank(text: str) -> str:
    return re.sub(r"[^\n]", " ", text)


_CHAR_LITERAL = re.compile(r"'(?:\\.[^'\n]{0,8}|[^\\'\n])'")


def _scan(
    code: str, *, line_comments: tuple[str, ...] = (), block: tuple[str, str] | None = None,
    quotes: str = "", triple: tuple[str, ...] = (), char_literals: bool = False, hash_needs_space: bool = False,
    keep_interpolation: bool = False, blank_strings: bool = True,
) -> str:
    """``code`` with comments blanked and (``blank_strings``) string-literal contents blanked, newlines and
    offsets kept. ``keep_interpolation`` keeps ``${...}`` inside strings (HCL / shell / JS templates)."""
    out: list[str] = []
    i, n = 0, len(code)
    while i < n:
        if block and code.startswith(block[0], i):
            j = code.find(block[1], i + len(block[0]))
            j = n if j < 0 else j + len(block[1])
            out.append(_blank(code[i:j]))
            i = j
            continue
        lc = next((c for c in line_comments if code.startswith(c, i)), None)
        if lc is not None and (not hash_needs_space or lc != "#" or i == 0 or code[i - 1] in " \t\n;"):
            j = code.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
            continue
        tq = next((t for t in triple if code.startswith(t, i)), None)
        if tq is not None:
            j = code.find(tq, i + 3)
            j = n if j < 0 else j
            inner = code[i + 3:j]
            out.append(tq + (_blank(inner) if blank_strings else inner) + (tq if j < n else ""))
            i = min(n, j + 3)
            continue
        c = code[i]
        if char_literals and c == "'":
            m = _CHAR_LITERAL.match(code, i)
            if m:
                out.append("'" + " " * (len(m.group(0)) - 2) + "'")
                i = m.end()
            else:  # a Rust lifetime / label: not a literal
                out.append(c)
                i += 1
            continue
        if c in quotes:
            j = i + 1
            while j < n and code[j] != c:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == "\n" and c != "`":
                    break
                j += 1
            inner = code[i + 1:min(j, n)]
            if blank_strings:
                inner = _keep_interp(inner) if keep_interpolation else _blank(inner)
            closed = j < n and code[j] == c
            out.append(c + inner + (c if closed else ""))
            i = j + 1 if closed else j
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _keep_interp(inner: str) -> str:
    """Blank a string's text but keep the contents of ``${...}`` interpolations."""
    parts: list[str] = []
    i = 0
    while True:
        at = inner.find("${", i)
        if at < 0:
            parts.append(_blank(inner[i:]))
            break
        end = inner.find("}", at)
        if end < 0:
            parts.append(_blank(inner[i:]))
            break
        parts.append(_blank(inner[i:at]) + " " + inner[at + 1:end + 1])
        i = end + 1
    return "".join(parts)


def _clean(code: str, family: str, *, blank_strings: bool = True) -> str:
    if family == "python":
        return _scan(code, line_comments=("#",), quotes="\"'", triple=('"""', "'''"), blank_strings=blank_strings)
    if family == "js":
        return _scan(code, line_comments=("//",), block=("/*", "*/"), quotes="\"'`", keep_interpolation=True,
                     blank_strings=blank_strings)
    if family in ("c", "java", "csharp", "go", "rust"):
        return _scan(code, line_comments=("//",), block=("/*", "*/"), quotes="\"`" if family == "go" else "\"",
                     triple=('"""',) if family == "java" else (), char_literals=True, blank_strings=blank_strings)
    if family == "shell":
        return _scan(_blank_heredocs(code), line_comments=("#",), quotes="\"'", hash_needs_space=True,
                     blank_strings=blank_strings)
    if family == "hcl":
        return _scan(_blank_heredocs(code, keep_interpolation=True), line_comments=("#", "//"), block=("/*", "*/"),
                     quotes="\"", keep_interpolation=True, blank_strings=blank_strings)
    if family == "yaml":
        return _scan(code, line_comments=("#",), hash_needs_space=True, blank_strings=False)
    return code


_HEREDOC = re.compile(r"<<-?\s*['\"]?([A-Za-z_]\w*)['\"]?[^\n]*\n")


def _blank_heredocs(code: str, *, keep_interpolation: bool = False) -> str:
    out: list[str] = []
    i = 0
    for m in _HEREDOC.finditer(code):
        if m.start() < i:
            continue
        end = re.compile(r"^\s*" + re.escape(m.group(1)) + r"\s*$", re.MULTILINE).search(code, m.end())
        stop = end.start() if end else len(code)
        body = code[m.end():stop]
        out.append(code[i:m.end()])
        out.append(_keep_interp(body) if keep_interpolation else _blank(body))
        if end:  # the terminator line is not a command either
            out.append(_blank(code[end.start():end.end()]))
            stop = end.end()
        i = stop
    out.append(code[i:])
    return "".join(out)


# ── shared helpers ───────────────────────────────────────────────────────


def _idents(text: str) -> list[str]:
    return _WORD.findall(text)


def _split_top(text: str) -> list[str]:
    """Split on commas outside (), [], {}, <>."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p for p in (x.strip() for x in parts) if p]


def _param_names(params: str, *, last: bool) -> list[str]:
    names = []
    for p in _split_top(params):
        p = re.split(r"(?<![=!<>])=(?!=)", p, maxsplit=1)[0]  # drop a default value
        if not last:
            p = re.split(r":(?!:)", p, maxsplit=1)[0]
        words = [w for w in _idents(p) if w not in ("mut", "ref", "out", "in", "params", "this", "final", "const")]
        if words:
            names.append(words[-1] if last else words[0])
    return names


def _pascal(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9_]*", name)) and any(ch.islower() for ch in name) and len(name) > 1


# ── C-like + Python ──────────────────────────────────────────────────────

_DECL_KEYWORD = re.compile(
    r"\b(?:def|class|function|func|fn|struct|enum|union|interface|trait|type|record|namespace|module|static|impl|"
    r"typedef|object|protocol|macro_rules!)\s+(?:mut\s+)?(?:\*\s*)?(" + _IDENT + ")"
)
_LOCAL_KEYWORD = re.compile(r"\b(?:let|const|var|val|auto)\s+(?:mut\s+)?(" + _IDENT + ")")
_OBJECT_KEY = re.compile(r"[{,]\s*(" + _IDENT + r")\s*:(?!:)")
_ASSIGN = re.compile(r"(?<![.\w$])(" + _IDENT + r")\s*(?:\[[^\]\n]*\]\s*)?(?::=|(?<![=!<>+\-*/%&|^])=(?![=>~]))")
_AUG_ASSIGN = re.compile(r"(?<![.\w$])(" + _IDENT + r")\s*(?:\+|-|\*|/|%|&|\||\^|<<|>>|\?\?)=")
_TUPLE_ASSIGN = re.compile(r"((?:" + _IDENT + r"\s*,\s*)+" + _IDENT + r")\s*(?::=|(?<![=!<>])=(?![=>]))")
_DESTRUCTURE = re.compile(r"\b(?:const|let|var|val|auto)\s*[{\[(]([^}\])]*)[}\])]")
_SELF_ATTR = re.compile(r"\b(?:self|this|cls)\s*\.\s*(" + _IDENT + r")\s*(?::\s*[^=\n]+)?=(?!=)")
_TYPED_DECL = re.compile(
    r"(?<![\w$.])[A-Za-z_][\w:.]*(?:<[^;=(){}]*>)?(?:\[\])*[\s*&]+(" + _IDENT + r")\s*(?=[=;,)\[{:]|\bin\b|$)",
    re.MULTILINE,
)
_FIELD_COLON = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+|readonly\s+|private\s+|public\s+|protected\s+)*(" + _IDENT + r")\s*\??:(?!:)", re.MULTILINE)
_GO_FIELD = re.compile(r"^\s*([A-Za-z_]\w*)\s+[*\[\]]*[A-Za-z_][\w.]*(?:\s+`[^`]*`)?\s*$", re.MULTILINE)
_FOR_TARGET = re.compile(r"\bfor\s*\(?\s*(?:(?:const|let|var|auto|final|mut)\s+)?([^;:\n]*?)\s+(?:in|of)\b")
_FOR_COLON = re.compile(r"\bfor\s*\(\s*(?:final\s+)?[\w<>\[\].,?\s]+?\s+(" + _IDENT + r")\s*:")
_AS_NAME = re.compile(r"\bas\s+(" + _IDENT + ")")
_CATCH = re.compile(r"\bcatch\s*\(\s*(?:[\w.|\s]+\s+)?(" + _IDENT + r")\s*\)")
_LAMBDA = re.compile(r"\blambda\s+([^:\n]*):")
_ARROW_ONE = re.compile(r"(" + _IDENT + r")\s*=>")
_ARROW_MANY = re.compile(r"\(([^()]*)\)\s*(?::\s*[\w<>\[\]|]+\s*)?=>")
_CLOSURE = re.compile(r"(?:^|[=(,\s])(?:move\s+)?\|([^|\n]*)\|")
_FUNC_PARAMS = re.compile(
    r"\b(?:def|function|func|fn)\s*(?:\([^)]*\)\s*)?(" + _IDENT + r")?\s*(?:<[^>]*>|\[[^\]]*\])?\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)"
)
_METHOD_DECL = re.compile(
    r"(?<![\w$.])(" + _IDENT + r")\s*\(([^()]*)\)\s*(?:const\s*)?(?:throws\s+[\w.,\s]+)?(?::\s*[^{=;]+)?\{"
)
_GLOBAL = re.compile(r"\b(?:global|nonlocal)\s+([\w\s,]+)")
_GO_RECEIVER = re.compile(r"\bfunc\s*\(\s*(\w+)")
_GENERIC_PARAMS = re.compile(r"(?:\b(?:class|struct|interface|fn|func|trait|impl|type|record|enum)\s+\w*\s*|^\s*)<([^<>()]*)>", re.MULTILINE)
_LABEL = re.compile(r"(?:^|[{};])\s*(" + _IDENT + r")\s*:\s*(?:for|while|do|loop)\b")
# C++ direct initialisation: ``std::ifstream file(p);``, ``std::string s("x");``, ``Widget w(1, 2);``.
_DIRECT_INIT = re.compile(
    r"(?<![\w$.>])(" + _IDENT + r")\s*(?:::\s*" + _IDENT + r"\s*)*(?:<[^;=(){}]*>)?[\s*&]+(" + _IDENT
    + r")\s*\([^;{}]*\)\s*;"
)
_NOT_A_TYPE = frozenset({"return", "throw", "delete", "new", "else", "case", "goto", "co_return", "co_yield",
                         "co_await", "sizeof", "typedef", "using", "namespace", "operator", "not", "and", "or",
                         "if", "while", "switch", "for", "do", "throws", "default"})
# Go one-line struct types: ``type Acc struct{ total float64 }`` / ``struct{ x, y int; name string }``.
_GO_STRUCT_ONE_LINE = re.compile(r"\bstruct\s*\{([^{}\n]*)\}")
_GO_STRUCT_FIELD = re.compile(r"^\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s+[*\[\]]*[A-Za-z_][\w.]*")

_CALL = re.compile(r"(?<![\w$.@#>:!?])(" + _IDENT + r")\s*\(")
_NEW = re.compile(r"\bnew\s+(" + _IDENT + r")\s*[(<\[{]")
_PATH = re.compile(r"(?<![\w$.@:>])(" + _IDENT + r"(?:\s*(?:\?\.|\.|->|::)\s*" + _IDENT + r")+)")
_PATH_SEP = re.compile(r"\s*(\?\.|\.|->|::)\s*")
_TYPE_POSITIONS = (
    re.compile(r"\b(?:extends|implements|throws)\s+([\w$.,\s<>]+?)\s*[{(]"),
    re.compile(r"(?<![\w$.])(" + _IDENT + r")\s*(?:<[^;=(){}]*>)?(?:\[\])?\s*[*&]?\s+[a-z_]\w*\s*[=;,)]"),
    re.compile(r"\bcatch\s*\(\s*(" + _IDENT + r")"),
    re.compile(r"\bexcept\s+\(?\s*(" + _IDENT + r")"),
    re.compile(r"[<,]\s*(" + _IDENT + r")\s*(?=[>,])"),
)
_ANNOTATION_TYPES = (
    re.compile(r"(?<!:):\s*(?:&\s*(?:mut\s+)?)?(" + _IDENT + r")\s*(?=[=,)\]>{;\n]|$)", re.MULTILINE),
    re.compile(r"->\s*(?:&\s*)?(" + _IDENT + r")"),
)

_PY_IMPORT = re.compile(r"^\s*import\s+([\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*)", re.MULTILINE)
_PY_FROM = re.compile(r"^\s*from\s+(\.*[\w.]*)\s+import\s+(\([^)]*\)|[^\n]+)", re.MULTILINE)
_JS_IMPORT = re.compile(r"\bimport\s+(?:type\s+)?([\w$*\s{},]+?)\s+from\s+['\"]([^'\"\n]+)['\"]")
_JS_BARE_IMPORT = re.compile(r"\bimport\s*\(?\s*['\"]([^'\"\n]+)['\"]")
_JS_REQUIRE = re.compile(r"(?:\b(?:const|let|var)\s+([\w$]+|\{[^}]*\})\s*=\s*)?\brequire\s*\(\s*['\"]([^'\"\n]+)['\"]\s*\)")
_JAVA_IMPORT = re.compile(r"^\s*import\s+(static\s+)?([\w.]+(?:\.\*)?)\s*;", re.MULTILINE)
_CS_USING = re.compile(r"^\s*(?:global\s+)?using\s+(?:static\s+)?(?:(\w+)\s*=\s*)?([\w.]+)\s*;", re.MULTILINE)
_GO_IMPORT_ONE = re.compile(r"^\s*import\s+(?:([\w.]+)\s+)?\"([^\"\n]+)\"", re.MULTILINE)
_GO_IMPORT_BLOCK = re.compile(r"^\s*import\s*\(([^)]*)\)", re.MULTILINE)
_GO_IMPORT_LINE = re.compile(r"^\s*(?:([\w.]+)\s+)?\"([^\"\n]+)\"", re.MULTILINE)
_RUST_USE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?use\s+([^;]+);", re.MULTILINE)
_RUST_EXTERN = re.compile(r"^\s*extern\s+crate\s+(\w+)(?:\s+as\s+(\w+))?\s*;", re.MULTILINE)
_C_INCLUDE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"\n]+)[>\"]", re.MULTILINE)
_CPP_USING_NS = re.compile(r"\busing\s+namespace\s+([\w:]+)\s*;")
_C_DEFINE = re.compile(r"^\s*#\s*define\s+(\w+)(?:\(([^)]*)\))?", re.MULTILINE)

_LANG_ALLOW: dict[str, frozenset[str]] = {
    "python": al.PYTHON_BUILTINS | al.PYTHON_METHODS,
    "js": al.JS_KEYWORDS | al.JS_GLOBALS | al.JS_METHODS | al.COMMON_METHODS,
    "java": al.JAVA_KEYWORDS | al.JAVA_TYPES | al.JAVA_METHODS | al.COMMON_METHODS,
    "csharp": al.CSHARP_KEYWORDS | al.CSHARP_TYPES | al.CSHARP_METHODS | al.COMMON_METHODS,
    "go": al.GO_KEYWORDS | al.COMMON_METHODS,
    "rust": al.RUST_KEYWORDS | al.RUST_METHODS | al.COMMON_METHODS,
    "c": al.C_KEYWORDS | al.C_STDLIB_FUNCTIONS | al.COMMON_METHODS,
}
_STD_ROOTS: dict[str, frozenset[str]] = {
    "python": al.PYTHON_STDLIB_ROOTS,
    "js": frozenset({"console", "Math", "JSON", "Object", "Array", "Promise", "Reflect", "Intl", "process",
                     "window", "document", "navigator", "globalThis", "Number", "String", "Date", "Symbol"}),
    # Java's standard types are roots too: ``Map.Entry``, ``List.of`` are never checked past them.
    "java": al.JAVA_STDLIB_ROOTS | al.JAVA_TYPES | frozenset({"System", "Math", "Arrays", "Collections", "Objects", "Stream",
                                              "Collectors", "Optional", "Files", "Paths", "Thread", "String",
                                              "Integer", "Long", "Double", "Boolean", "Executors", "TimeUnit"}),
    "csharp": al.CSHARP_STDLIB_ROOTS | frozenset({"Console", "Math", "Convert", "String", "DateTime", "Task",
                                                  "Enumerable", "File", "Path", "Directory", "Environment", "Guid",
                                                  "Encoding", "TimeSpan", "Thread"}),
    "go": al.GO_STDLIB_ROOTS,
    "rust": al.RUST_STDLIB_ROOTS,
    "c": frozenset({"std"}),
}
# Values whose members are the standard library's: a local bound from one of these (``log = logging.getLogger()``,
# ``sb = new StringBuilder()``, ``with open(p) as fh``, ``std::vector<int> out``, Go ``w http.ResponseWriter``)
# is treated like a standard-library root, so ``log.warning`` / ``fh.readline`` are never checked.
_STD_VALUES: dict[str, frozenset[str]] = {
    "python": al.PYTHON_BUILTINS,
    "js": al.JS_GLOBALS,
    "java": al.JAVA_TYPES,
    "csharp": al.CSHARP_TYPES,
    "rust": al.RUST_KEYWORDS,
    "c": al.CPP_STD_NAMES | frozenset({"std"}),
    "go": frozenset(),
}
_BIND_ASSIGN = re.compile(
    r"(?<![.\w$])(" + _IDENT + r")\s*(?::\s*[^=\n]+?)?\s*(?::=|(?<![=!<>])=(?![=>]))\s*(?:await\s+|new\s+|&\s*|\*\s*)?(" + _IDENT + ")"
)
_BIND_WITH = re.compile(r"\bwith\s+(?:await\s+)?(" + _IDENT + r")[^:\n]*?\bas\s+(" + _IDENT + ")")
_BIND_TYPED = re.compile(
    r"(?<![\w$.])(" + _IDENT + r")\s*(?:(?:::|\.)\s*" + _IDENT + r"\s*)*(?:<[^;=(){}]*>)?[\s*&]+(" + _IDENT + r")\s*(?=[=;,)({])"
)
_BIND_GO_PARAM = re.compile(r"(?<![\w.])(" + _IDENT + r")\s+[*\[\]]*(" + _IDENT + r")\.\w+")


def _bind_std_locals(clean: str, family: str, out: "ExtractedTerms", std_roots: frozenset[str]) -> None:
    std_values = _STD_VALUES.get(family, frozenset())
    pairs: list[tuple[str, str]] = [(m.group(1), m.group(2)) for m in _BIND_ASSIGN.finditer(clean)]
    pairs += [(m.group(2), m.group(1)) for m in _BIND_WITH.finditer(clean)]
    if family in ("java", "csharp", "c"):
        pairs += [(m.group(2), m.group(1)) for m in _BIND_TYPED.finditer(clean)]
    if family == "go":
        pairs += [(m.group(1), m.group(2)) for m in _BIND_GO_PARAM.finditer(clean)]
    for _round in range(2):  # chains: ``a = os.x(); b = a.y()``
        for target, source in pairs:
            if source in std_roots or source in out.std_bound or source in std_values:
                out.std_bound.add(target)


_CONTROL = frozenset({"if", "for", "while", "switch", "catch", "foreach", "using", "lock", "fixed", "elif",
                      "return", "sizeof", "match", "with", "synchronized"})
_ROOT_SELF = frozenset({"self", "this", "cls", "super", "base", "Self", "me"})


def _defined_c_like(clean: str, family: str, language: str) -> tuple[set[str], set[str]]:
    """``(declarations, locals)``. Declarations (functions, classes, fields, ``self.x =`` attributes) also
    satisfy an attribute segment (``obj.x``); locals (variables, parameters, loop targets) only satisfy a
    bare name, so ``ids = w.ids`` does not ground ``w.ids``."""
    decls: set[str] = set()
    for rx in (_DECL_KEYWORD, _SELF_ATTR):
        decls.update(m.group(1) for m in rx.finditer(clean))
    if family == "js":
        decls.update(m.group(1) for m in _OBJECT_KEY.finditer(clean))
    d: set[str] = set()
    for rx in (_LOCAL_KEYWORD, _ASSIGN, _AUG_ASSIGN, _AS_NAME, _CATCH, _ARROW_ONE, _LABEL):
        d.update(m.group(1) for m in rx.finditer(clean))
    for m in _TUPLE_ASSIGN.finditer(clean):
        d.update(_idents(m.group(1)))
    for m in _DESTRUCTURE.finditer(clean):
        d.update(_idents(m.group(1)))
    for m in _FOR_TARGET.finditer(clean):
        d.update(w for w in _idents(m.group(1)) if w not in ("const", "let", "var", "auto"))
    for m in _FOR_COLON.finditer(clean):
        d.add(m.group(1))
    for m in _LAMBDA.finditer(clean):
        d.update(_idents(m.group(1)))
    for m in _ARROW_MANY.finditer(clean):
        d.update(_param_names(m.group(1), last=False))
    for m in _CLOSURE.finditer(clean):
        d.update(_param_names(m.group(1), last=False))
    for m in _GLOBAL.finditer(clean):
        d.update(_idents(m.group(1)))
    for m in _GENERIC_PARAMS.finditer(clean):
        d.update(_param_names(m.group(1), last=False))
    last = family in ("java", "csharp", "c")
    for m in _FUNC_PARAMS.finditer(clean):
        if m.group(1):
            decls.add(m.group(1))
        d.update(_param_names(m.group(2), last=last and family != "go"))
    for m in _METHOD_DECL.finditer(clean):
        if m.group(1) not in _CONTROL:
            decls.add(m.group(1))
            d.update(_param_names(m.group(2), last=last))
    if family in ("java", "csharp", "c", "js"):
        decls.update(m.group(1) for m in _TYPED_DECL.finditer(clean))  # fields or locals: both
    if family in ("python", "rust", "js"):
        decls.update(m.group(1) for m in _FIELD_COLON.finditer(clean))
    if family == "go":
        decls.update(m.group(1) for m in _GO_FIELD.finditer(clean))
        d.update(m.group(1) for m in _GO_RECEIVER.finditer(clean))
        for m in _GO_STRUCT_ONE_LINE.finditer(clean):
            for part in m.group(1).split(";"):
                f = _GO_STRUCT_FIELD.match(part)
                if f:
                    decls.update(_idents(f.group(1)))
    if family == "c":
        decls.update(m.group(2) for m in _DIRECT_INIT.finditer(clean) if m.group(1) not in _NOT_A_TYPE)
        for m in _C_DEFINE.finditer(clean):
            decls.add(m.group(1))
            if m.group(2):
                d.update(_idents(m.group(2)))
        decls.update(re.findall(r"\}\s*(" + _IDENT + r")\s*;", clean))  # typedef struct {...} name;
    return decls, d - decls


def _imports(raw: str, nocom: str, family: str, out: ExtractedTerms, std_roots: frozenset[str]) -> None:
    """Import statements: bind their names and add a term per non-standard module / imported name."""
    if family == "python":
        for m in _PY_IMPORT.finditer(nocom):
            for part in m.group(1).split(","):
                bits = part.split()
                mod = bits[0]
                alias = bits[2] if len(bits) >= 3 else None
                out.defined.add(alias or mod.split(".")[0])
                if mod.split(".")[0] in std_roots:
                    out.std_bound.add(alias or mod.split(".")[0])
                    continue
                segs = mod.split(".")
                out.add(mod, mod, segs[-1], kind="import")
        for m in _PY_FROM.finditer(nocom):
            mod = m.group(1)
            names = [x.strip() for x in m.group(2).strip("()").replace("\n", " ").split(",") if x.strip()]
            std = mod.startswith(".") or mod.split(".")[0] in std_roots
            if not std and mod:
                out.add(mod, mod, mod.split(".")[-1], kind="import")
            for nm in names:
                bits = nm.split()
                if not bits or bits[0] == "*":
                    continue
                out.defined.add(bits[2] if len(bits) >= 3 else bits[0])
                if std and not mod.startswith("."):
                    out.std_bound.add(bits[2] if len(bits) >= 3 else bits[0])
                if not std:
                    out.add(f"{mod}.{bits[0]}", bits[0], kind="import")
    elif family == "js":
        def module_term(spec: str) -> bool:
            """Add the term for a module spec; True when it is a standard / local module."""
            if spec.startswith((".", "/", "node:")) or "://" in spec:
                return True
            pkg = "/".join(spec.split("/")[:2]) if spec.startswith("@") else spec.split("/")[0]
            if pkg in al.NODE_BUILTIN_MODULES:
                return True
            out.add(spec, spec, pkg, spec.split("/")[-1], kind="import")
            return False

        for m in _JS_IMPORT.finditer(raw):
            std = module_term(m.group(2))
            spec = m.group(1)
            braces = re.search(r"\{([^}]*)\}", spec)
            for part in _split_top(braces.group(1)) if braces else []:
                bits = part.replace("type ", "").split()
                if not bits:
                    continue
                out.defined.add(bits[-1])
                if not std:
                    out.add(f"{m.group(2)}.{bits[0]}", bits[0], kind="import")
            rest = re.sub(r"\{[^}]*\}", " ", spec)
            bound = [w for w in _idents(rest) if w != "as"]
            out.defined.update(bound)
            if std and not m.group(2).startswith((".", "/")):
                out.std_bound.update(bound)
        for m in _JS_BARE_IMPORT.finditer(raw):
            module_term(m.group(1))
        for m in _JS_REQUIRE.finditer(raw):
            std = module_term(m.group(2))
            if m.group(1):
                out.defined.update(_idents(m.group(1)))
                if std and not m.group(2).startswith((".", "/")):
                    out.std_bound.update(_idents(m.group(1)))
    elif family == "java":
        for m in _JAVA_IMPORT.finditer(nocom):
            path = m.group(2)
            segs = path.split(".")
            wildcard = segs[-1] == "*"
            if wildcard:
                segs = segs[:-1]
            else:
                out.defined.add(segs[-1])
            if segs[0] in std_roots:
                if not wildcard:
                    out.std_bound.add(segs[-1])
                elif not m.group(1):  # ``import java.util.concurrent.*;``: its types by their simple names
                    members = al.JAVA_STDLIB_PACKAGE_TYPES.get(".".join(segs))
                    if members is None:
                        out.wildcard_std = True
                    else:
                        out.allow = out.allow | members
                        out.std_bound.update(members)
                continue
            out.add(".".join(segs), segs[-1], ".".join(segs), kind="import")
    elif family == "csharp":
        for m in _CS_USING.finditer(nocom):
            if m.group(1):
                out.defined.add(m.group(1))
            segs = m.group(2).split(".")
            if segs[0] in std_roots:
                continue
            out.add(m.group(2), segs[-1], m.group(2), kind="import")
    elif family == "go":
        specs = [(m.group(1), m.group(2)) for m in _GO_IMPORT_ONE.finditer(raw)]
        for block in _GO_IMPORT_BLOCK.finditer(raw):
            specs += [(m.group(1), m.group(2)) for m in _GO_IMPORT_LINE.finditer(block.group(1))]
        for alias, path in specs:
            segs = path.split("/")
            last = segs[-1]
            if re.fullmatch(r"v\d+", last) and len(segs) > 1:
                last = segs[-2]
            name = re.sub(r"\.v\d+$", "", last).replace("-", "_").split(".")[-1]
            out.defined.update(x for x in (alias, name, last) if x and x not in ("_", "."))
            if "." not in segs[0]:
                out.std_bound.update(x for x in (alias, name) if x and x not in ("_", "."))
                continue  # standard library
            out.add(path, path, segs[-1], last, name, kind="import")
    elif family == "rust":
        for m in _RUST_EXTERN.finditer(nocom):
            out.defined.add(m.group(2) or m.group(1))
            out.add(m.group(1), m.group(1), kind="import")
        for m in _RUST_USE.finditer(nocom):
            tree = m.group(1)
            root = tree.split("::")[0].strip().lstrip(":")
            leaves = re.findall(r"(\w+)\s*(?=,|\}|$|\s+as\b)", tree.strip())
            aliases = re.findall(r"\bas\s+(\w+)", tree)
            out.defined.update(aliases)
            out.defined.update(x for x in leaves if x != "self")
            if "self" in leaves or tree.strip() == root:
                out.defined.add(root)
            if root in std_roots:
                if root not in ("crate", "self", "super"):
                    out.std_bound.update(x for x in leaves + aliases if x != "self")
                continue
            out.add(root, root, kind="import")
            for leaf in leaves:
                if leaf not in ("self", root) and leaf not in aliases:
                    out.add(f"{root}::{leaf}", leaf, kind="import")
    elif family == "c":
        for m in _C_INCLUDE.finditer(nocom):
            header = m.group(1).strip()
            if header in al.C_STANDARD_HEADERS or header.startswith(al.C_STANDARD_HEADER_DIRS):
                continue
            base = header.rsplit("/", 1)[-1]
            out.add(header, header, base, base.rsplit(".", 1)[0], kind="import")


def _extract_c_like(code: str, language: str) -> ExtractedTerms:
    family = family_of(language) or ""
    nocom = _clean(code, family, blank_strings=False)
    clean = _clean(code, family)
    std_roots = _STD_ROOTS.get(family, frozenset())
    allow = _LANG_ALLOW.get(family, frozenset())
    if family == "c" and normalize_language(language) == "cpp":
        allow = allow | al.CPP_STD_NAMES if _CPP_USING_NS.search(clean) else allow
    out = ExtractedTerms(allow=allow)
    decls, local_names = _defined_c_like(clean, family, language)
    out.defined |= decls
    out.locals |= local_names
    _imports(code if family in ("js", "go") else nocom, nocom, family, out, std_roots)
    _bind_std_locals(clean, family, out, std_roots)
    # Preprocessor lines are handled by the include table only.
    body = re.sub(r"^\s*#[^\n]*", lambda m: _blank(m.group(0)), clean, flags=re.MULTILINE) if family == "c" else clean
    if family == "python":
        body = re.sub(r"^\s*(?:from|import)\s[^\n]*", lambda m: _blank(m.group(0)), body, flags=re.MULTILINE)
    elif family in ("java", "csharp"):
        body = re.sub(r"^\s*(?:import|package|using|namespace)\s[^\n;{]*", lambda m: _blank(m.group(0)), body,
                      flags=re.MULTILINE)
    elif family == "rust":
        body = re.sub(r"^\s*(?:pub\s+)?(?:use|mod|extern\s+crate)\s[^;]*;", lambda m: _blank(m.group(0)), body,
                      flags=re.MULTILINE)
    elif family == "go":
        body = re.sub(r"^\s*(?:package\s+\w+|import\s*\([^)]*\)|import\s[^\n]*)", lambda m: _blank(m.group(0)), body,
                      flags=re.MULTILINE)
    elif family == "js":
        body = re.sub(r"\bimport\s[^;\n]*", lambda m: _blank(m.group(0)), body)

    path_spans: list[tuple[int, int]] = []
    for m in _PATH.finditer(body):
        path_spans.append(m.span())
        raw_path = m.group(1)
        segs = _PATH_SEP.split(raw_path)  # [name, sep, name, sep, ...]
        names = segs[0::2]
        seps = segs[1::2]
        root = names[0]
        if root in std_roots or root in out.std_bound:
            continue
        if root not in _ROOT_SELF and root not in out.defined and root not in out.locals:
            out.add(root, root, kind="symbol")
        display = root
        for sep, name in list(zip(seps, names[1:]))[:MAX_PATH_SEGMENTS - 1]:
            display = f"{display}{sep}{name}"
            out.add(display, name, kind="member")
    in_path = _spans_lookup(path_spans)
    for m in _CALL.finditer(body):
        name = m.group(1)
        if in_path(m.start(1)):
            continue
        prev = body[max(0, m.start(1) - 4):m.start(1)]
        if re.search(r"\bnew\s*$", prev):
            continue  # handled as a type
        out.add(name, name, kind="call")
    for m in _NEW.finditer(body):
        if not in_path(m.start(1)):
            out.add(m.group(1), m.group(1), kind="type")
    type_rx = list(_TYPE_POSITIONS)
    if family in ("python", "js", "rust"):
        type_rx += list(_ANNOTATION_TYPES)
    for rx in type_rx:
        for m in rx.finditer(body):
            for name in _idents(m.group(1)):
                if _pascal(name) and not in_path(m.start(1)):
                    out.add(name, name, kind="type")
    if out.wildcard_std:
        out.terms = [t for t in out.terms if not (t.kind in ("type", "symbol") and _pascal(t.display))]
    return out


def _spans_lookup(spans: list[tuple[int, int]]) -> Callable[[int], bool]:
    import bisect

    starts = [s for s, _e in spans]

    def inside(pos: int) -> bool:
        i = bisect.bisect_right(starts, pos) - 1
        return i >= 0 and spans[i][0] <= pos < spans[i][1]

    return inside


# ── shell ────────────────────────────────────────────────────────────────

_SH_FUNC = re.compile(r"^\s*(?:function\s+)?([\w.:-]+)\s*\(\s*\)", re.MULTILINE)
_SH_FUNC_KW = re.compile(r"\bfunction\s+([\w.:-]+)")
_SH_VAR = re.compile(r"(?:^|[\s;&|(])(?:local\s+|export\s+|readonly\s+|declare\s+(?:-\w+\s+)*)?([A-Za-z_]\w*)(?:\[[^\]]*\])?\+?=")
_SH_READ = re.compile(r"\bread\s+(?:-\w+\s+(?:\S+\s+)?)*([A-Za-z_][\w\s]*)")
_SH_FOR = re.compile(r"\b(?:for|select)\s+([A-Za-z_]\w*)\s+in\b")
_SH_CASE_LABEL = re.compile(r"^[ \t]*\(?[^\s()]+(?:\s*\|\s*[^\s()]+)*\)(?=\s|$)", re.MULTILINE)
_SH_CMD = re.compile(
    r"(?:^|[|;&\n(]|\$\(|`|\b(?:then|do|else|elif|if|while|until|sudo|exec|time|nohup|xargs|env|nice|timeout|command|!)\s)"
    r"[ \t]*(?:[A-Za-z_]\w*=\S*[ \t]+)*(?:-\S+[ \t]+)*([A-Za-z_][\w.+-]*)(?=[\s;|&)]|$)",
    re.MULTILINE,
)


def _extract_shell(code: str, language: str) -> ExtractedTerms:
    clean = _clean(code, "shell")
    out = ExtractedTerms(allow=al.SHELL_BUILTINS | al.SHELL_COMMON_COMMANDS)
    for rx in (_SH_FUNC, _SH_FUNC_KW, _SH_VAR, _SH_FOR):
        out.defined.update(m.group(1) for m in rx.finditer(clean))
    for m in _SH_READ.finditer(clean):
        out.defined.update(_idents(m.group(1)))
    body = _SH_CASE_LABEL.sub(lambda m: _blank(m.group(0)), clean)
    body = _SH_FUNC.sub(lambda m: _blank(m.group(0)), body)
    for m in _SH_CMD.finditer(body):
        name = m.group(1)
        after = body[m.end(1):m.end(1) + 2]
        if after.startswith("="):
            continue
        out.add(name, name, kind="command")
    return out


# ── HCL / Terraform ──────────────────────────────────────────────────────

_HCL_BLOCK2 = re.compile(r"^\s*(resource|data)\s+\"([^\"\n]+)\"\s+\"([^\"\n]+)\"\s*\{", re.MULTILINE)
_HCL_BLOCK1 = re.compile(r"^\s*(variable|output|module|provider)\s+\"([^\"\n]+)\"", re.MULTILINE)
_HCL_LOCALS = re.compile(r"^\s*locals\s*\{", re.MULTILINE)
_HCL_ARG = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*(?:=(?!=)|\{)", re.MULTILINE)
_HCL_REF = re.compile(
    r"(?<![\w.\"-])(data\.)?([A-Za-z][\w-]*)\.([A-Za-z_][\w-]*)((?:\[[^\]\n]*\])?(?:\.[A-Za-z_][\w-]*(?:\[[^\]\n]*\])?)*)"
)
_HCL_FUNC = re.compile(r"(?<![\w.:-])([a-z][a-z0-9_]*)\s*\(")
_HCL_SKIP_ROOTS = frozenset({"each", "count", "self", "path", "terraform", "module", "var", "local"})


def _block_body(text: str, open_at: int) -> tuple[str, int]:
    """Text inside the brace that starts at ``open_at`` (index of ``{``)."""
    depth, i = 0, open_at
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i], i
        i += 1
    return text[open_at + 1:], len(text)


def _top_args(body: str) -> set[str]:
    """Argument and nested-block names at any depth of a block body."""
    return {m.group(1) for m in _HCL_ARG.finditer(body)}


def _extract_hcl(code: str, language: str) -> ExtractedTerms:
    nocom = _clean(code, "hcl", blank_strings=False)
    clean = _clean(code, "hcl")
    out = ExtractedTerms(allow=al.HCL_KEYWORDS | al.HCL_FUNCTIONS)
    block_args: dict[str, set[str]] = {}
    for m in _HCL_BLOCK2.finditer(nocom):
        kind, rtype, rname = m.groups()
        key = f"{'data.' if kind == 'data' else ''}{rtype}.{rname}"
        out.defined.add(key)
        body, _end = _block_body(nocom, m.end() - 1)
        block_args[key] = _top_args(body)
        out.add(rtype, rtype, kind="type")
    for m in _HCL_BLOCK1.finditer(nocom):
        kind, name = m.groups()
        out.defined.add({"variable": "var", "output": "output", "module": "module", "provider": "provider"}[kind] + "." + name)
        if kind == "provider":
            out.add(name, name, kind="type")
    for m in _HCL_LOCALS.finditer(nocom):
        body, _end = _block_body(nocom, m.end() - 1)
        out.defined.update(f"local.{n}" for n in re.findall(r"^\s{0,8}([A-Za-z_][\w-]*)\s*=(?!=)", body, re.MULTILINE))
    # References: only inside expressions (string contents blanked except ${...}).
    body = re.sub(r"^\s*(?:resource|data|variable|output|module|provider)\s+\"[^\n]*", lambda m: _blank(m.group(0)), clean,
                  flags=re.MULTILINE)
    for m in _HCL_REF.finditer(body):
        is_data, root, name, rest = m.groups()
        attrs = re.findall(r"\.([A-Za-z_][\w-]*)", re.sub(r"\[[^\]]*\]", "", rest or ""))
        if is_data:
            key = f"data.{root}.{name}"
            out.add(f"data.{root}", root, kind="type")
            if key not in out.defined:
                out.add(key, name, key, kind="attribute")
            elif attrs and attrs[0] not in block_args.get(key, set()) and attrs[0] not in al.HCL_COMMON_ATTRIBUTES:
                out.add(f"{key}.{attrs[0]}", attrs[0], kind="attribute")
            continue
        if root in ("var", "local"):
            ref = f"{root}.{name}"
            if ref not in out.defined:
                out.add(ref, name, ref, kind="attribute")
            continue
        if root in _HCL_SKIP_ROOTS or "_" not in root:
            continue  # module outputs, each/count/self/path, and anything that is not a resource type
        key = f"{root}.{name}"
        out.add(root, root, kind="type")
        if key in out.defined and attrs and attrs[0] not in block_args.get(key, set()) \
                and attrs[0] not in al.HCL_COMMON_ATTRIBUTES:
            out.add(f"{key}.{attrs[0]}", attrs[0], kind="attribute")
    for m in _HCL_FUNC.finditer(body):
        prev = body[max(0, m.start() - 2):m.start()]
        if prev.endswith("::"):
            continue
        out.add(m.group(1), m.group(1), kind="call")
    return out


# ── YAML keys ────────────────────────────────────────────────────────────

_YAML_KEY = re.compile(r"^(\s*)((?:-\s+)*)([A-Za-z_][\w.-]*)\s*:(?=\s|$)")


def _extract_yaml(code: str, language: str) -> ExtractedTerms:
    from .builtin import TEMPLATE_PLACEHOLDER, strip_yaml_templates

    text = _clean(strip_yaml_templates(code), "yaml")
    out = ExtractedTerms(allow=al.YAML_COMMON_KEYS)
    stack: list[tuple[int, str]] = []  # (indent, key) of the open parents
    block_scalar_indent: int | None = None
    for line in text.split("\n"):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if block_scalar_indent is not None:
            if indent > block_scalar_indent:
                continue  # inside a ``|`` / ``>`` block scalar: text, not keys
            block_scalar_indent = None
        m = _YAML_KEY.match(line)
        if not m:
            continue
        key_indent = indent + len(m.group(2))
        key = m.group(3)
        while stack and stack[-1][0] >= key_indent:
            stack.pop()
        parent = stack[-1][1] if stack else None
        rest = line[m.end():].strip()
        if rest[:1] in ("|", ">"):
            block_scalar_indent = key_indent
        stack.append((key_indent, key))
        if TEMPLATE_PLACEHOLDER in key or (parent is not None and parent in al.YAML_FREE_FORM_PARENTS):
            continue
        out.add(key, key, *key.split(".")[-1:], kind="key")
    return out


def _extract_code_symbols(code: str, language: str) -> ExtractedTerms:
    family = family_of(language)
    # Very long lines (minified code, inline data) are left out: nobody hand-writes them, and they would make
    # some of the regexes quadratic.
    code = _LONG_LINE.sub("", code)
    if family == "shell":
        return _extract_shell(code, language)
    if family == "hcl":
        return _extract_hcl(code, language)
    if family == "yaml":
        return _extract_yaml(code, language)
    if family is None:
        return ExtractedTerms()
    return _extract_c_like(code, language)


register_term_extractor("code_symbols", _extract_code_symbols)


# ── matching ─────────────────────────────────────────────────────────────


def _norm(name: str) -> str:
    return re.sub(r"[_\-$]", "", name).lower()


class GroundingIndex:
    """Whole-word lookups against the grounding text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.words = set(_WORD.findall(text))
        self.normed = {_norm(w) for w in self.words}

    def has(self, name: str) -> bool:
        if re.fullmatch(r"[A-Za-z_$][\w$]*", name):
            return name in self.words or _norm(name) in self.normed
        return re.search(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])", self.text) is not None


def unresolved_terms(code: str, language: str, grounding_text: str, *, extractor: str = "code_symbols") -> list[str]:
    """Displays of the terms that are not grounded, in order of first appearance (deduplicated)."""
    fn = TERM_EXTRACTORS.get(extractor)
    if fn is None:
        raise KeyError(f"unknown term extractor '{extractor}'")
    extracted = fn(code, normalize_language(language))
    index = GroundingIndex(grounding_text)
    seen: set[str] = set()
    out: list[str] = []
    every_name = extracted.defined | extracted.locals
    for term in extracted.terms:
        if term.display in seen:
            continue
        if term.kind == "import":
            known: set[str] = set()  # an import binds its own name: only the context or the allowlist ground it
        elif term.kind in ("member", "attribute"):
            known = extracted.defined  # ``ids = w.ids`` must not ground ``w.ids``
        else:
            known = every_name
        if any(n in known or n in extracted.allow or index.has(n) for n in term.names):
            continue
        seen.add(term.display)
        out.append(term.display)
    return out


class ContextGrounding(BaseCheck):
    id = "context_grounding"
    languages = GROUNDING_LANGUAGES
    requires_context = "grounding_text"
    default_enabled = False  # opt-in: only when requested in check_ids or listed in CODE_CHECK_ALLOWED
    # Scoring: a reference that fails this check makes the row not checkable for THIS check only.
    checkable_per_check = True

    def run(self, code, *, language, context):
        text = (context or {}).get("grounding_text")
        if not isinstance(text, str) or not text.strip():
            return make_result(self.id, None, skipped="no_context")
        extractor = (context or {}).get("grounding_extractor") or "code_symbols"
        if extractor not in TERM_EXTRACTORS:
            return make_result(self.id, None, [f"unknown term extractor '{extractor}'"], skipped="unavailable")
        if len(code) > MAX_CODE_CHARS:
            return make_result(self.id, None, ["code too long to check"], skipped="unavailable")
        missing = unresolved_terms(code, language, text, extractor=extractor)
        if not missing:
            return make_result(self.id, True)
        return make_result(self.id, False, [f"`{d}`: not in the context or defined in the answer" for d in missing])

