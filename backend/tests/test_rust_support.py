"""Rust as a supported code language: ingest (.rs -> code document, language
"rust"), declaration-aware chunking (tricky literals, attributes, impl blocks),
the ``rust_syntax`` parse check, and the empty-file rejection reason.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from app.config import settings
from app.services import code_check_service as svc
from app.services import sandbox_exec
from app.services.code_checks.base import normalize_language
from app.services.code_chunker import CODE_LANGUAGES, _mask_rust, chunk_code, detect_language
from app.services.document_sources.git_repo import GitRepoConfig, clone_repo
from app.utils.file_parsers import SUPPORTED_EXTENSIONS, parse_file

RUST = '''//! Crate docs.
#![allow(dead_code)]

use std::fmt;

/// A point.
#[derive(Debug, Clone)]
pub struct Point {
    x: i32,
    y: i32,
}

impl Point {
    /// Make one.
    pub const fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }

    pub(crate) async unsafe fn far(&self) -> i32 {
        self.x
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

pub enum Shape {
    Dot,
    Line(Point, Point),
}

pub trait Area {
    fn area(&self) -> f64;
    fn double(&self) -> f64 {
        self.area() * 2.0
    }
}

extern "C" fn callback(x: i32) -> i32 {
    x
}

macro_rules! square {
    ($x:expr) => {
        $x * $x
    };
}

mod util {
    pub fn helper() -> u8 {
        1
    }
}
'''


def _syms(chunks):
    return [(c.kind, c.symbol) for c in chunks]


def _by_symbol(chunks):
    """First chunk per symbol (``impl Point`` header chunks share the struct's name)."""
    out = {}
    for c in chunks:
        out.setdefault(c.symbol, c)
    return out


# -- registration / ingest ----------------------------------------------------


def test_rust_is_registered_everywhere():
    assert "rust" in CODE_LANGUAGES
    assert detect_language("src/lib.rs") == "rust"
    assert ".rs" in SUPPORTED_EXTENSIONS
    assert ".toml" not in SUPPORTED_EXTENSIONS
    assert normalize_language("rs") == "rust"


def test_rs_file_parses_as_code_text(tmp_path):
    path = tmp_path / "lib.rs"
    path.write_text("  fn a() {}\n")
    assert parse_file(path) == "  fn a() {}\n"  # indentation kept


def _run_git(repo: Path, *args: str) -> None:
    env = dict(os.environ, GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@example.test",
               GIT_COMMITTER_NAME="T", GIT_COMMITTER_EMAIL="t@example.test", GIT_CONFIG_GLOBAL="/dev/null")
    subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True)


def test_git_repo_accepts_rs_and_reports_empty_files():
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(settings, "DOCUMENT_SOURCES_ALLOW_LOCAL_GIT", True):
        repo = Path(tmp) / "r"
        (repo / "src").mkdir(parents=True)
        _run_git(repo, "init", "-q", "-b", "main")
        (repo / "src" / "main.rs").write_text("fn main() {}\n")
        (repo / "Cargo.toml").write_text('[package]\nname = "x"\n')
        (repo / "pkg").mkdir()
        (repo / "pkg" / "__init__.py").write_text("")
        (repo / "blank.py").write_text("  \n\n")
        (repo / "real.py").write_text("def f():\n    return 1\n")
        _run_git(repo, "add", "-A")
        _run_git(repo, "commit", "-q", "-m", "x")
        with clone_repo(GitRepoConfig(url=str(repo))) as snap:
            kept = {f.path: (f.content_kind, f.language) for f in snap.files}
            reasons = {r["path"]: r["reason"] for r in snap.rejected}
        assert kept["src/main.rs"] == ("code", "rust")
        assert kept["real.py"] == ("code", "python")
        assert reasons["Cargo.toml"] == "unsupported_type"
        assert reasons["pkg/__init__.py"] == "empty_file"
        assert reasons["blank.py"] == "empty_file"
        assert "pkg/__init__.py" not in kept


# -- chunker ------------------------------------------------------------------


def test_golden_declaration_chunks():
    chunks = chunk_code(RUST, "rust", max_chars=4000)
    got = _syms(chunks)
    for expected in [
        ("class", "Point"), ("method", "Point.new"), ("method", "Point.far"), ("method", "Point.fmt"),
        ("class", "Shape"), ("function", "callback"), ("function", "square"), ("function", "util.helper"),
        ("method", "Area.double"),
    ]:
        assert expected in got, (expected, got)
    by = _by_symbol(chunks)
    text = RUST.split("\n")
    point = by["Point"]
    assert text[point.start_line - 1] == "/// A point."
    assert text[point.start_line].startswith("#[derive")
    assert point.text.rstrip().splitlines()[-1] == "}"
    new = by["Point.new"]
    assert text[new.start_line - 1].strip() == "/// Make one."
    assert "Point { x, y }" in new.text
    assert text[by["Point.far"].start_line - 1].strip().startswith("pub(crate) async unsafe fn far")
    assert text[by["callback"].start_line - 1].startswith('extern "C" fn callback')
    assert by["square"].text.rstrip().endswith("}")
    nxt = 1
    for c in chunks:
        assert c.start_line == nxt
        nxt = c.end_line + 1


def test_trait_default_method_and_bodyless_signature():
    by = _by_symbol(chunk_code(RUST, "rust", max_chars=4000))
    assert by["Area.area"].kind == "method" and by["Area.area"].text.strip() == "fn area(&self) -> f64;"
    assert "self.area() * 2.0" in by["Area.double"].text


@pytest.mark.parametrize(
    "body",
    [
        'let s = "}}} { \\" }";',
        'let s = r"}}}";',
        'let s = r#"} " } "#;',
        'let s = r##"}"# }"##;',
        'let s = br#"}}"#;',
        "let c = '{';",
        "let c = '}';",
        "let c = '\\'';",
        "let c = '\\u{7d}';",
        "let r: &'static str = \"x\"; let _l: &'a () = &();",
        "/* { /* } */ } */ let x = 1;",
        "// } comment }\n    let x = 1;",
    ],
)
def test_tricky_literals_do_not_unbalance_braces(body):
    src = f"fn first() {{\n    {body}\n}}\n\nfn second() {{\n    1\n}}\n"
    chunks = chunk_code(src, "rust", max_chars=4000)
    assert [(c.symbol, c.start_line) for c in chunks] == [("first", 1), ("second", 5 + body.count("\n"))], body
    assert "second" not in chunks[0].text


def test_lifetime_is_not_a_char_literal():
    src = "fn longest<'a>(a: &'a str, b: &'a str) -> &'a str {\n    if a.len() > 1 { a } else { b }\n}\n\nfn other<'b>() {}\n"
    chunks = chunk_code(src, "rust")
    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [("longest", 1, 4), ("other", 5, 5)]
    masked = _mask_rust("fn f<'a>(x: &'a u8) { let c = 'x'; }")
    assert "'a" in masked and "'x'" not in masked


def test_nested_block_comment_hides_declarations():
    src = "/* outer /* inner */ fn ghost() { */\nfn real() {\n}\n"
    got = _syms(chunk_code(src, "rust"))
    assert ("function", "real") in got and ("function", "ghost") not in got


def test_multiline_signature_where_clause_and_test_module():
    src = (
        "pub fn run<T>(\n    a: T,\n    b: u8,\n) -> Vec<T>\nwhere\n    T: Clone,\n{\n    vec![a]\n}\n\n"
        "#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn it_works() {\n        assert!(true);\n    }\n}\n"
    )
    by = _by_symbol(chunk_code(src, "rust", max_chars=4000))
    assert (by["run"].start_line, by["run"].end_line) == (1, 10)
    assert by["tests.it_works"].text.lstrip().startswith("#[test]")
    assert by["tests.it_works"].kind == "function"


def test_multiline_attribute_attaches_to_item():
    src = "#[derive(\n    Debug,\n    Clone,\n)]\nstruct S {\n    a: u8,\n}\n"
    chunks = chunk_code(src, "rust")
    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [("S", 1, 7)]


# -- rust_syntax --------------------------------------------------------------

HAVE_RUSTFMT = shutil.which("rustfmt") is not None


def _check(code: str, allowed: str = "rust_syntax"):
    with mock.patch.object(settings, "CODE_CHECK_ALLOWED", allowed):
        return svc.run_code_checks(code, language="rust", check_ids=["rust_syntax"])


@pytest.mark.skipif(not HAVE_RUSTFMT, reason="rustfmt not installed")
class TestRustSyntaxReal:
    def test_valid_code_passes(self):
        assert _check('fn main() { println!("hi"); }')["ok"] is True

    def test_real_syntax_error_fails_with_message(self):
        out = _check("fn ok() {}\nfn broken( {\n")
        assert out["ok"] is False
        assert out["results"][0]["messages"]

    def test_formatting_only_differences_pass(self):
        assert _check("fn   main(){let   x=1 ;\n\n\n\nlet y   =  2;}")["ok"] is True

    def test_statement_fragment_is_wrapped(self):
        assert _check("let x = 1;\nlet y = x + 1;")["ok"] is True
        assert _check("let x = ;")["ok"] is False

    def test_extra_closing_brace_fails_even_when_wrapped(self):
        assert _check("}\nfn g() {")["ok"] is False

    def test_braces_in_literals_do_not_count(self):
        assert _check("let s = \"}\"; let t = '{';")["ok"] is True


def test_rust_syntax_skipped_when_rustfmt_missing():
    with mock.patch.object(shutil, "which", return_value=None):
        out = _check("fn a() {}")
    assert out["ok"] is None
    assert not out["results"] or out["results"][0]["skipped"]


def test_rust_syntax_not_allowed_is_not_run():
    with mock.patch.object(shutil, "which", return_value="/usr/bin/rustfmt"), \
            mock.patch.object(sandbox_exec, "run_sandboxed") as run:
        _check("fn a() {}", allowed="")
    run.assert_not_called()


def test_rust_syntax_uses_sandbox_parse_only_invocation():
    seen = []

    def fake(argv, *, files, stdin=None, **kw):
        seen.append((argv, stdin))
        return sandbox_exec.SandboxResult(0, "", "", False, False, False, 0.01)

    with mock.patch.object(shutil, "which", return_value="/usr/bin/rustfmt"), \
            mock.patch.object(sandbox_exec, "run_sandboxed", side_effect=fake):
        out = _check("fn a() {}")
    assert out["ok"] is True
    argv, stdin = seen[-1]  # [0] may be the availability probe
    assert argv[0] == "rustfmt" and "--emit" in argv and "stdout" in argv
    assert not {"rustc", "cargo"} & set(argv)
    assert stdin == "fn a() {}"


# -- review fixes -------------------------------------------------------------


@pytest.mark.skipif(not HAVE_RUSTFMT, reason="rustfmt not installed")
@pytest.mark.parametrize(
    "code",
    [
        'let s = "abc;',  # E0765 unterminated double quote string
        "fn f() { /* never closed }",  # E0758 unterminated block comment
        'fn f() { let s = r#"abc"; }',  # E0748 unterminated raw string
        "fn f() { let c = ''; }",  # empty char literal
        'fn f() { let s = "a\\qb"; }',  # unknown escape
        "fn f() { let c = 'ab'; }",  # char literal with two characters
    ],
)
def test_lexer_errors_with_error_codes_fail(code):
    out = _check(code)
    assert out["ok"] is False, out
    assert out["results"][0]["messages"]


def test_error_codes_are_parsed_from_stderr():
    from app.services.code_checks.external import _rust_messages

    msgs = _rust_messages("error[E0765]: unterminated double quote string\n --> <stdin>:1:9\n\nerror[internal]: x\n")
    assert msgs == ["line 1: unterminated double quote string"]


def test_nonzero_exit_without_diagnostics_is_a_failure_not_a_pass():
    def fake(argv, **kw):
        return sandbox_exec.SandboxResult(1, "", "something odd\n", False, False, False, 0.01)

    with mock.patch.object(shutil, "which", return_value="/usr/bin/rustfmt"), \
            mock.patch.object(sandbox_exec, "run_sandboxed", side_effect=fake):
        out = _check("fn a() {}")
    assert out["ok"] is False
    assert "something odd" in out["results"][0]["messages"][0]


def test_toolchain_error_is_unavailable_not_a_syntax_failure():
    from app.services.code_checks import external

    shim = "error: rustup could not choose a version of rustfmt to run, because one wasn't specified explicitly\n"

    def fake(argv, **kw):
        return sandbox_exec.SandboxResult(1, "", shim, False, False, False, 0.01)

    external._RUSTFMT_PROBE.clear()
    with mock.patch.object(shutil, "which", return_value="/usr/bin/rustfmt"), \
            mock.patch.object(external, "_rustup_env", return_value={}), \
            mock.patch.object(sandbox_exec, "run_sandboxed", side_effect=fake):
        assert external.RustSyntax().available() is False
        out = external.RustSyntax().run("fn a() {}", language="rust", context={})
    external._RUSTFMT_PROBE.clear()
    assert out["ok"] is None and out["skipped"] == "unavailable"


@pytest.mark.parametrize(
    "src, expected",
    [
        ("pub fn f(\n    a: i32,\n) -> Result<\n    Vec<u8>,\n    Error,\n> {\n    todo!()\n}\n", [("f", 1, 8)]),
        ("impl<T> From<Vec<T>>\n    for MyVec<T>\n{\n    fn from(v: Vec<T>) -> Self {\n        MyVec(v)\n    }\n}\n",
         [("MyVec", 1, 3), ("MyVec.from", 4, 7)]),
        ("impl<T, U> Convert<U>\n    for Wrapper<T>\nwhere\n    T: Into<U>,\n    U: Clone,\n{\n    fn go(&self) {}\n}\n",
         [("Wrapper", 1, 6), ("Wrapper.go", 7, 8)]),
    ],
)
def test_rustfmt_style_multiline_headers(src, expected):
    chunks = chunk_code(src, "rust", max_chars=4000)
    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == expected
