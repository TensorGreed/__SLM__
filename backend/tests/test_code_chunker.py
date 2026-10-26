"""Code chunker: declaration-level chunks with exact line ranges.

Pins, per language, where chunk boundaries fall on a small toy file, plus the
invariants every chunking must keep: with no context every source line lands
in exactly one chunk, in order; ``start_line`` / ``end_line`` describe the
chunk's exact text; oversize units split at line boundaries with ``#n``
symbol suffixes; shebang / filename language detection.
"""

from __future__ import annotations

import pytest

from app.services.code_chunker import CodeChunk, chunk_code, detect_language


def _summary(chunks: list[CodeChunk]) -> list[tuple]:
    return [(c.start_line, c.end_line, c.kind, c.symbol) for c in chunks]


def _assert_partition(text: str, chunks: list[CodeChunk]) -> None:
    """Every line in exactly one chunk, in order, and text == the lines it claims."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    expected = 1
    for chunk in chunks:
        assert chunk.start_line == expected, f"gap/overlap before {chunk}"
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.unit_start_line <= chunk.unit_end_line
        expected = chunk.end_line + 1
    assert expected == len(lines) + 1, "lines missing at the end"


JAVA = """package demo;

import java.util.List;

/** A greeter. */
public class Greeter {
    private final String name = "{";
    private int count;

    @Override
    public String greet(String who) {
        String s = "}";
        return "Hello " + who; // }
    }

    public int add(int a, int b) {
        Runnable r = new Runnable() {
            public void run() { }
        };
        return a + b;
    }

    public void last()
    {
        count++;
    }
}
"""


def test_java_one_chunk_per_method_plus_class_chunk():
    chunks = chunk_code(JAVA, "java")
    assert _summary(chunks) == [
        (1, 4, "module", None),
        (5, 9, "class", "Greeter"),
        (10, 15, "method", "Greeter.greet"),
        (16, 22, "method", "Greeter.add"),
        (23, 27, "method", "Greeter.last"),
    ]
    greet = chunks[2]
    assert (greet.unit_start_line, greet.unit_end_line) == (10, 14)
    assert greet.text.splitlines()[0].strip() == "@Override"
    # braces inside string / char literals and comments did not end the method early
    assert "return \"Hello \" + who;" in greet.text
    # the anonymous class stays inside its method
    assert "public void run()" in chunks[3].text
    # the class chunk carries the javadoc, header and fields
    assert chunks[1].text.splitlines()[0] == "/** A greeter. */"
    assert "private int count;" in chunks[1].text
    _assert_partition(JAVA, chunks)


def test_java_nested_class_methods_are_qualified():
    text = "class Outer {\n    static class Inner {\n        void ping() {\n        }\n    }\n}\n"
    chunks = chunk_code(text, "java")
    assert [c.symbol for c in chunks] == ["Outer", "Outer.Inner", "Outer.Inner.ping"]
    _assert_partition(text, chunks)


def test_csharp_namespace_class_method_and_expression_member():
    text = """namespace Demo
{
    public class Greeter
    {
        public int X { get; set; }
        public string Greet(string who)
        {
            return who;
        }
        public int Add(int a, int b) => a + b;
    }
}
"""
    chunks = chunk_code(text, "csharp")
    assert _summary(chunks) == [
        (1, 2, "module", None),
        (3, 5, "class", "Greeter"),
        (6, 9, "method", "Greeter.Greet"),
        (10, 12, "method", "Greeter.Add"),
    ]
    _assert_partition(text, chunks)


def test_go_type_receiver_method_and_function():
    text = """package main

import (
	"fmt"
)

type Greeter struct {
	Name string
}

// Greet says hi.
func (g *Greeter) Greet() string {
	return "hi"
}

func add(a, b int) int {
	return a + b
}
"""
    chunks = chunk_code(text, "go")
    assert _summary(chunks) == [
        (1, 6, "module", None),
        (7, 10, "class", "Greeter"),
        (11, 15, "method", "Greeter.Greet"),
        (16, 18, "function", "add"),
    ]
    assert chunks[2].text.startswith("// Greet says hi.")
    _assert_partition(text, chunks)


def test_javascript_functions_arrows_and_class_methods():
    text = """import a from 'a'
const x = 1;
function add(a, b) {
  return a + b;
}
export const mul = (a, b) => {
  return a * b;
};
class Greeter extends Base {
  constructor() { super(); }
  greet(name) {
    return `hi ${name}`;
  }
}
describe('x', () => {
  it('y', () => {});
});
"""
    chunks = chunk_code(text, "javascript")
    assert _summary(chunks) == [
        (1, 2, "module", None),
        (3, 5, "function", "add"),
        (6, 8, "function", "mul"),
        (9, 9, "class", "Greeter"),
        (10, 10, "method", "Greeter.constructor"),
        (11, 14, "method", "Greeter.greet"),
        (15, 17, "module", None),
    ]
    _assert_partition(text, chunks)


def test_typescript_interface_and_class():
    text = "export interface Shape {\n  area(): number;\n}\n\nexport class Square implements Shape {\n  area(): number {\n    return 4;\n  }\n}\n"
    chunks = chunk_code(text, "ts")
    assert [(c.kind, c.symbol) for c in chunks] == [
        ("class", "Shape"), ("class", "Square"), ("method", "Square.area"),
    ]
    _assert_partition(text, chunks)


def test_c_header_struct_and_function_with_allman_brace():
    text = """#include <stdio.h>
#define X 1
struct point { int x; int y; };
int add(int a, int b)
{
    return a + b;
}
typedef struct {
    int q;
} Foo;
"""
    chunks = chunk_code(text, "c")
    assert _summary(chunks) == [
        (1, 2, "module", None),
        (3, 3, "class", "point"),
        (4, 7, "function", "add"),
        (8, 10, "class", "Foo"),
    ]
    _assert_partition(text, chunks)


def test_cpp_class_methods():
    text = "class Greeter {\npublic:\n  int add(int a, int b) {\n    return a + b;\n  }\n};\n"
    chunks = chunk_code(text, "cpp")
    assert [(c.kind, c.symbol) for c in chunks] == [("class", "Greeter"), ("method", "Greeter.add")]
    _assert_partition(text, chunks)


def test_python_functions_classes_methods_decorators_docstring():
    text = '''import os

@decorator
def add(a, b):
    """doc
def fake():
    """
    return a + b

class Greeter:
    """Hi."""
    x = 1

    def greet(self):
        return 1

    @property
    def name(self):
        return "n"

if __name__ == "__main__":
    add(1, 2)
'''
    chunks = chunk_code(text, "python")
    assert _summary(chunks) == [
        (1, 2, "module", None),
        (3, 9, "function", "add"),
        (10, 13, "class", "Greeter"),
        (14, 16, "method", "Greeter.greet"),
        (17, 20, "method", "Greeter.name"),
        (21, 22, "module", None),
    ]
    assert chunks[1].text.startswith("@decorator")  # decorators belong to the unit
    assert "def fake():" in chunks[1].text  # a def inside a docstring is not a declaration
    assert chunks[4].text.startswith("    @property")
    _assert_partition(text, chunks)


def test_terraform_blocks_get_dotted_symbols():
    text = '# comment\nresource "toy_thing" "main" {\n  name = "x"\n}\n\nvariable "size" {\n  default = 1\n}\n'
    chunks = chunk_code(text, "tf")
    assert _summary(chunks) == [(1, 5, "block", "resource.toy_thing.main"), (6, 8, "block", "variable.size")]
    _assert_partition(text, chunks)


def test_yaml_top_level_keys_and_document_markers():
    text = "---\nname: demo\nitems:\n  - a\n  - b\n# about ports\nports:\n  - 80\n"
    chunks = chunk_code(text, "yaml")
    assert _summary(chunks) == [(1, 2, "block", "name"), (3, 5, "block", "items"), (6, 8, "block", "ports")]
    multi = "a: 1\n---\nb: 2\n"
    assert [c.symbol for c in chunk_code(multi, "yml")] == ["a", "b"]
    _assert_partition(text, chunks)


def test_json_top_level_keys():
    text = '{\n  "name": "demo",\n  "deps": {\n    "a": 1\n  },\n  "list": [1, 2]\n}\n'
    chunks = chunk_code(text, "json")
    assert _summary(chunks) == [(1, 2, "block", "name"), (3, 5, "block", "deps"), (6, 7, "block", "list")]
    _assert_partition(text, chunks)


def test_shell_functions_ignore_braces_in_strings():
    text = '#!/bin/bash\nset -e\ngreet() {\n  echo "hi {"\n}\nfunction add {\n  echo $(( $1 + $2 ))\n}\ngreet\n'
    chunks = chunk_code(text, "sh")
    assert _summary(chunks) == [
        (1, 2, "module", None), (3, 5, "function", "greet"), (6, 8, "function", "add"), (9, 9, "module", None),
    ]
    _assert_partition(text, chunks)


def test_sql_statements_with_symbols():
    text = "-- tables\nCREATE TABLE users (\n  id INT\n);\nINSERT INTO users VALUES (1);\nselect 1;\n"
    chunks = chunk_code(text, "sql")
    assert _summary(chunks) == [(1, 4, "block", "users"), (5, 5, "block", "users"), (6, 6, "block", None)]
    assert chunks[0].text.startswith("-- tables")
    _assert_partition(text, chunks)


def test_xml_depth_one_children_pom_style():
    text = """<?xml version="1.0"?>
<project>
  <modelVersion>4</modelVersion>
  <dependencies>
    <dependency>
      <id>a</id>
    </dependency>
  </dependencies>
</project>
"""
    chunks = chunk_code(text, "xml")
    assert _summary(chunks) == [
        (1, 2, "module", None), (3, 3, "block", "modelVersion"), (4, 9, "block", "dependencies"),
    ]
    _assert_partition(text, chunks)


def test_properties_groups_and_ini_sections():
    props = "# a\na.b=1\na.c=2\n\nx=3\n"
    chunks = chunk_code(props, "properties")
    assert _summary(chunks) == [(1, 4, "block", "a.b"), (5, 5, "block", "x")]
    ini = "top=1\n[one]\na=1\n; two\n[two]\nb=2\n"
    chunks = chunk_code(ini, "ini")
    assert _summary(chunks) == [(1, 1, "module", None), (2, 3, "block", "one"), (4, 6, "block", "two")]
    _assert_partition(props, chunk_code(props, "properties"))
    _assert_partition(ini, chunks)


def test_oversize_unit_is_split_on_line_boundaries_with_suffixes():
    body = "\n".join(f"    x{i} = {i}" for i in range(40))
    text = f"def big():\n{body}\n\ndef small():\n    return 1\n"
    chunks = chunk_code(text, "python", max_chars=120)
    assert all(len(c.text) <= 120 for c in chunks)
    big = [c for c in chunks if (c.symbol or "").startswith("big")]
    assert [c.symbol for c in big][:3] == ["big", "big#2", "big#3"]
    assert len({(c.unit_start_line, c.unit_end_line) for c in big}) == 1  # same unit for every piece
    assert chunks[-1].symbol == "small"
    _assert_partition(text, chunks)


def test_unknown_language_packs_blank_line_blocks():
    text = "alpha\nbeta\n\ngamma\ndelta\n\nepsilon\n"
    assert [c.text for c in chunk_code(text, None, max_chars=1000)] == [text.rstrip("\n")]
    small = chunk_code(text, "brainfuck", max_chars=14)
    assert all(len(c.text) <= 14 for c in small) and len(small) > 1
    _assert_partition(text, small)


def test_context_lines_extend_text_but_not_unit():
    text = "a = 1\n\ndef f():\n    return 1\n\nb = 2\n"
    chunks = chunk_code(text, "python", context_lines=1)
    f = next(c for c in chunks if c.symbol == "f")
    assert (f.unit_start_line, f.unit_end_line) == (3, 4)
    # the unit's own text (3-5, its trailing blank line included) plus one line each side
    assert (f.start_line, f.end_line) == (2, 6)
    assert f.text == "\n".join(text.split("\n")[1:6])


def test_empty_and_garbage_input_never_raise():
    assert chunk_code("", "python") == []
    assert chunk_code("   \n\n", "java") == []
    junk = "}}}{{{ ((( \n'unterminated\n\"also\n"
    for language in ("java", "python", "json", "xml", "yaml", "sql", "shell", "hcl", "c"):
        _assert_partition(junk, chunk_code(junk, language))
    with pytest.raises(ValueError):
        chunk_code("x", "python", max_chars=0)


def test_crlf_line_endings_keep_line_numbers():
    text = "def f():\r\n    return 1\r\n\r\ndef g():\r\n    return 2\r\n"
    chunks = chunk_code(text, "python")
    assert [(c.start_line, c.end_line, c.symbol) for c in chunks] == [(1, 3, "f"), (4, 5, "g")]


@pytest.mark.parametrize(
    "path,text,expected",
    [
        ("a/b/Main.java", "", "java"),
        ("x.py", "", "python"),
        ("pom.xml", "", "xml"),
        ("main.tf", "", "hcl"),
        ("Dockerfile", "", "dockerfile"),
        ("deploy.Dockerfile", "", "dockerfile"),
        ("app.tsx", "", "typescript"),
        ("noext", "#!/usr/bin/env python3\nprint(1)\n", "python"),
        ("run", "#!/bin/bash\necho hi\n", "shell"),
        ("tool", "#!/usr/bin/env node\nconsole.log(1)\n", "javascript"),
        ("tool", "#!/bin/sh\n", "shell"),
        ("notes.xyz", "plain words", None),
        ("config.yml", "", "yaml"),
    ],
)
def test_detect_language(path, text, expected):
    assert detect_language(path, text) == expected


# ── Python: continuation lines are never dedent boundaries ───────────


def test_python_triple_quoted_string_closing_at_column_zero_stays_in_its_method():
    text = (
        "class Greeter:\n"
        "    def run(self):\n"
        '        s = """\n'
        "def fake():\n"
        '"""\n'
        "        return s\n"
        "\n"
        "    def other(self):\n"
        "        return 1\n"
    )
    chunks = chunk_code(text, "python")
    _assert_partition(text, chunks)
    assert [(c.start_line, c.end_line, c.symbol) for c in chunks if c.kind == "method"] == [
        (2, 7, "Greeter.run"), (8, 9, "Greeter.other"),
    ]
    assert "fake" not in {c.symbol for c in chunks}


def test_python_parenthesised_continuation_at_column_zero_stays_in_its_function():
    text = "def f(x):\n    return (x +\n1)\n\ndef g():\n    pass\n"
    chunks = chunk_code(text, "python")
    _assert_partition(text, chunks)
    assert [(c.start_line, c.end_line, c.symbol) for c in chunks] == [(1, 4, "f"), (5, 6, "g")]


def test_python_bracket_and_dict_continuations_and_module_level_strings():
    text = (
        "def f():\n"
        "    data = {\n"
        "'a': 1,\n"
        "}\n"
        "    return data\n"
        "\n"
        'BANNER = """\n'
        "def not_a_function():\n"
        '"""\n'
        "\n"
        "def g():\n"
        "    pass\n"
    )
    chunks = chunk_code(text, "python")
    _assert_partition(text, chunks)
    assert [c.symbol for c in chunks if c.kind == "function"] == ["f", "g"]
    assert next(c for c in chunks if c.symbol == "f").end_line == 6


# ── JS / TS regex literals ───────────────────────────────────────────


def test_js_regex_literals_with_braces_do_not_unbalance_the_scan():
    text = (
        "function withRegex(s) {\n"
        "  return s.replace(/[{]/g, '');\n"
        "}\n"
        "\n"
        "function other(s) {\n"
        "  const re = /\\}+/;\n"
        "  const q = s.split(/[/}]/);\n"
        "  return a / b / c;\n"
        "}\n"
        "\n"
        "function last() {\n"
        "  return 1;\n"
        "}\n"
    )
    for lang in ("javascript", "typescript"):
        chunks = chunk_code(text, lang)
        _assert_partition(text, chunks)
        assert [(c.start_line, c.end_line, c.symbol) for c in chunks] == [
            (1, 4, "withRegex"), (5, 10, "other"), (11, 13, "last"),
        ]


def test_js_division_is_not_a_regex():
    text = "const half = (x) => {\n  return x / 2; // }\n};\n\nconst b = () => {\n  return a / b;\n};\n"
    chunks = chunk_code(text, "javascript")
    _assert_partition(text, chunks)
    assert [(c.start_line, c.end_line, c.symbol) for c in chunks] == [(1, 4, "half"), (5, 7, "b")]


# ── C: GNU style, HCL heredocs, long lines ───────────────────────────


def test_c_gnu_style_return_type_on_its_own_line():
    text = "static int\nlong_style(int x)\n{\n    return x;\n}\n\nint\nmain(void)\n{\n    return 0;\n}\n"
    chunks = chunk_code(text, "c")
    _assert_partition(text, chunks)
    assert [(c.start_line, c.end_line, c.symbol) for c in chunks] == [(1, 6, "long_style"), (7, 11, "main")]


def test_hcl_heredoc_bodies_are_masked():
    text = 'locals {\n  y = <<EOT\n}\n"{"\nEOT\n}\n\nvariable "x" {\n  default = 1\n}\n'
    chunks = chunk_code(text, "hcl")
    _assert_partition(text, chunks)
    assert [(c.start_line, c.end_line, c.symbol) for c in chunks] == [(1, 7, "locals"), (8, 10, "variable.x")]
    indented = 'locals {\n  y = <<-EOT\n    }\n    EOT\n}\n'
    assert [(c.start_line, c.end_line, c.symbol) for c in chunk_code(indented, "hcl")] == [(1, 5, "locals")]


def test_a_single_line_longer_than_max_chars_is_hard_split():
    text = "const data = '" + "x" * 2500 + "';\nconst a = 1;\n"
    chunks = chunk_code(text, "javascript", max_chars=1000)
    assert all(len(c.text) <= 1000 for c in chunks)
    long_slices = [c for c in chunks if c.start_line == 1 and c.end_line == 1]
    assert len(long_slices) == 3
    assert "".join(c.text for c in long_slices) == text.split("\n")[0]
    assert chunks[-1].start_line == chunks[-1].end_line == 2
    # prose of unknown type behaves the same
    blob = chunk_code("y" * 2500, None, max_chars=1000)
    assert [len(c.text) for c in blob] == [1000, 1000, 500]
