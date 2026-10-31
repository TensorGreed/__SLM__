"""context_grounding code check: term extractors per language family (true positives + false-positive
guards), opt-in selection (defaults unchanged), the API, and the code-example calibration metric."""

import json
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.config import settings
from app.services import code_check_service as svc
from app.services import code_example_scoring as ces
from app.services.code_checks import grounding
from app.services.code_checks.grounding import ExtractedTerms, register_term_extractor, unresolved_terms


class Allow:
    def __init__(self, ids):
        self.p = mock.patch.object(settings, "CODE_CHECK_ALLOWED", ids)

    def __enter__(self):
        self.p.start()

    def __exit__(self, *a):
        self.p.stop()


def missing(code: str, language: str, context: str) -> list[str]:
    return unresolved_terms(code, language, context)


# ── Python ───────────────────────────────────────────────────────────────

PY_CTX = ("The acme_client package: acme_client.Client(host) returns a client; client.fetch_widget(widget_id) "
          "returns a Widget with .name and .size.")


class PythonTests(unittest.TestCase):
    def test_true_positives(self):
        code = (
            "import acme_client\n"
            "from acme_client.models import Widget, Gadget\n"
            "import acme_extras\n"
            "c = acme_client.Client('h')\n"
            "w = c.fetch_widget(1)\n"
            "ids = w.ids\n"                      # invented attribute (``ids`` the local does not ground it)
            "g = GadgetFactory()\n"              # invented function / type
            "c.fetch_widgets_bulk([1])\n"        # invented method
        )
        got = missing(code, "python", PY_CTX)
        for want in ("acme_client.models.Gadget", "acme_extras", "w.ids", "GadgetFactory", "c.fetch_widgets_bulk"):
            self.assertIn(want, got)
        self.assertNotIn("c.fetch_widget", got)
        self.assertNotIn("acme_client.models.Widget", got)

    def test_false_positive_guards(self):
        code = (
            "import os, json, logging\n"
            "from pathlib import Path\n"
            "import acme_client\n"
            "log = logging.getLogger(__name__)\n"
            "class Store:\n"
            "    def __init__(self, host: str, retries: int = 3):\n"
            "        self.client = acme_client.Client(host)\n"
            "        self.retries = retries\n"
            "    def load(self, widget_id, *args, **kw):\n"
            "        '''acme_client.not_a_real_call in a docstring'''\n"
            "        w = self.client.fetch_widget(widget_id)  # acme_client.bogus in a comment\n"
            "        names = [x.strip() for x in w.name.split(',')]\n"
            "        log.warning('loaded %s', 'acme_client.also_bogus')\n"
            "        with open(Path(os.getcwd()) / 'f') as fh:\n"
            "            first = fh.readline()\n"
            "        helper = lambda v: v * self.retries\n"
            "        return json.dumps({'n': len(names), 's': w.size}), first, helper(2), self.load\n"
        )
        self.assertEqual(missing(code, "python", PY_CTX), [])


# ── C-like ───────────────────────────────────────────────────────────────


class CLikeTests(unittest.TestCase):
    def test_javascript(self):
        ctx = "Use @acme/widgets: fetchWidget(id) resolves to { name, size }."
        code = (
            "const fs = require('fs');\n"
            "import { fetchWidget, fetchGadget } from '@acme/widgets';\n"
            "import thing from 'acme-unknown-pkg';\n"
            "export async function main(id, opts = {}) {\n"
            "  const w = await fetchWidget(id);\n"
            "  const label = `name ${w.name} and acme.bogusInString`; // acme.bogusInComment\n"
            "  const list = [1, 2].map((x) => x * 2);\n"
            "  console.log(JSON.stringify(w), list.length, label, opts);\n"
            "  fs.writeFileSync('out.json', 'x');\n"
            "  w.colour;\n"
            "  return new WidgetCache(w);\n"
            "}\n"
        )
        got = missing(code, "javascript", ctx)
        self.assertEqual(set(got), {"@acme/widgets.fetchGadget", "acme-unknown-pkg", "w.colour", "WidgetCache"})

    def test_typescript_clean_code_passes(self):
        ctx = "acme_client exports connect(url) -> Session; Session.query(sql, params)."
        code = (
            'import * as fs from "fs";\n'
            'import { connect } from "acme_client";\n'
            "interface Row { id: number; name: string }\n"
            "export class Repo {\n"
            "  private cache = new Map<string, Row[]>();\n"
            "  constructor(private readonly url: string) {}\n"
            "  async rows(sql: string, params: unknown[] = []): Promise<Row[]> {\n"
            "    const session = await connect(this.url);\n"
            "    const out: Row[] = await session.query(sql, params);\n"
            "    this.cache.set(sql, out);\n"
            "    fs.writeFileSync('a.json', JSON.stringify(out.map((r) => r.name)));\n"
            "    return out;\n"
            "  }\n"
            "}\n"
        )
        self.assertEqual(missing(code, "typescript", ctx), [])

    def test_java(self):
        ctx = "com.acme.widgets.WidgetClient has fetchWidget(String id) returning Widget with getName()."
        code = (
            "import java.util.List;\n"
            "import com.acme.widgets.WidgetClient;\n"
            "import com.acme.widgets.Widget;\n"
            "import com.acme.widgets.GadgetStore;\n"
            "public class App {\n"
            "    private final WidgetClient client;\n"
            "    public App(WidgetClient client) { this.client = client; }\n"
            "    public String name(String id) {\n"
            "        Widget w = client.fetchWidget(id);\n"
            "        List<String> parts = List.of(w.getName());\n"
            "        w.getColour();\n"
            "        AcmeHelper h = null;\n"
            "        return String.join(\",\", parts);\n"
            "    }\n"
            "}\n"
        )
        got = missing(code, "java", ctx)
        self.assertEqual(set(got), {"com.acme.widgets.GadgetStore", "w.getColour", "AcmeHelper"})

    def test_go(self):
        ctx = "package acme: acme.NewClient(addr) *Client; (*Client).FetchWidget(ctx, id) (Widget, error). Widget has Name."
        code = (
            "package main\n\n"
            "import (\n    \"context\"\n    \"fmt\"\n    \"net/http\"\n    \"github.com/acme/acme-go/acme\"\n)\n\n"
            "type Holder struct {\n    Count int\n}\n\n"
            "func run(ctx context.Context, w http.ResponseWriter, addr string) error {\n"
            "    c := acme.NewClient(addr)\n"
            "    wd, err := c.FetchWidget(ctx, 3)\n"
            "    if err != nil {\n        return fmt.Errorf(\"fetch: %w\", err)\n    }\n"
            "    h := Holder{Count: 1}\n"
            "    w.Header().Set(\"X\", \"y\")\n"
            "    fmt.Println(wd.Name, wd.Colour, h.Count)\n"
            "    var g acme.Gadget\n    _ = g\n    return nil\n}\n"
        )
        self.assertEqual(set(missing(code, "go", ctx)), {"wd.Colour", "acme.Gadget"})

    def test_rust(self):
        ctx = "crate acme_widgets: Client::new(url), client.fetch_widget(id) -> Result<Widget>. Widget has name."
        code = (
            "use std::collections::HashMap;\n"
            "use acme_widgets::{Client, Widget, Gizmo};\n"
            "fn load<'a>(url: &'a str, id: u32) -> Result<String, Box<dyn std::error::Error>> {\n"
            "    let client = Client::new(url);\n"
            "    let w: Widget = client.fetch_widget(id)?;\n"
            "    let mut m: HashMap<u32, String> = HashMap::new();\n"
            "    m.insert(id, w.name.clone());\n"
            "    let c = 'x';\n"
            "    Ok(w.colour.to_string())\n"
            "}\n"
        )
        self.assertEqual(set(missing(code, "rust", ctx)), {"acme_widgets::Gizmo", "w.colour"})

    def test_c_and_cpp(self):
        ctx = "acme.h declares acme_widget_t *acme_widget_new(int size) and int acme_widget_size(acme_widget_t *w)."
        c_code = (
            "#include <stdio.h>\n#include \"acme.h\"\n#include \"acme_extra.h\"\n#define MAX 10\n"
            "typedef struct { int n; } pair_t;\n"
            "int main(int argc, char **argv) {\n"
            "    acme_widget_t *w = acme_widget_new(MAX);\n"
            "    pair_t p = { 1 };\n"
            "    printf(\"%d acme_bogus\\n\", acme_widget_size(w));\n"
            "    acme_widget_free(w);\n"
            "    return p.n;\n}\n"
        )
        self.assertEqual(set(missing(c_code, "c", ctx)), {"acme_extra.h", "acme_widget_free"})
        cpp_code = (
            "#include <vector>\n#include <iostream>\n"
            "class Repo {\n public:\n  std::vector<int> rows() {\n    std::vector<int> out;\n"
            "    out.push_back(acme_widget_size(nullptr));\n    std::cout << out.size() << std::endl;\n"
            "    return out;\n  }\n};\n"
        )
        self.assertEqual(missing(cpp_code, "c++", ctx), [])

    def test_csharp(self):
        ctx = "AcmeClient.Connect(url) returns a session with Query(sql)."
        code = (
            "using System;\nusing System.Linq;\nusing AcmeClient;\n"
            "public class Repo {\n"
            "    public int Count { get; private set; }\n"
            "    public int[] Rows(string sql) {\n"
            "        var session = AcmeClient.Connect(\"url\");\n"
            "        var rows = session.Query(sql).Select(r => r * 2).ToArray();\n"
            "        Console.WriteLine($\"{rows.Length} rows\");\n"
            "        session.Explode();\n"
            "        return rows;\n    }\n}\n"
        )
        self.assertEqual(missing(code, "csharp", ctx), ["session.Explode"])


# ── shell / HCL / YAML ───────────────────────────────────────────────────


class ReviewedFalsePositiveTests(unittest.TestCase):
    """Correct code the check used to flag (precision over recall)."""

    def test_cpp_direct_initialisation_declares_the_name(self):
        code = (
            "#include <fstream>\n#include <string>\n"
            "void read_it(const std::string& p) {\n"
            "    std::ifstream file(p);\n    file.close();\n"
            "    std::string s(\"x\");\n    s.size();\n"
            "    std::vector<int> v(3, 0);\n    v.size();\n}\n"
        )
        self.assertEqual(missing(code, "cpp", "read a config"), [])
        # A call statement is still not a declaration: ``return f(x);`` / ``else g(y);`` define nothing.
        got = missing("int h(int x) {\n    if (x) return acme_f(x);\n    else acme_g(x);\n    return 0;\n}\n", "cpp", "h")
        self.assertEqual(set(got), {"acme_f", "acme_g"})

    def test_java_nested_and_wildcard_imported_standard_types(self):
        code = (
            "import java.util.*;\n"
            "import java.util.concurrent.*;\n"
            "public class Totals {\n"
            "    int sum(Map<String, Integer> m, ExecutorService ex) throws Exception {\n"
            "        int t = 0;\n"
            "        for (Map.Entry<String, Integer> e : m.entrySet()) { t += e.getValue(); }\n"
            "        Future<Integer> f = ex.submit(() -> 1);\n"
            "        java.util.concurrent.Future<Integer> g = f;\n"
            "        AbstractMap.SimpleEntry<String, Integer> p = new AbstractMap.SimpleEntry<>(\"a\", 1);\n"
            "        ConcurrentHashMap<String, Integer> c = new ConcurrentHashMap<>();\n"
            "        return t + g.get() + c.size();\n"
            "    }\n}\n"
        )
        self.assertEqual(missing(code, "java", "sum the values"), [])
        # Without the wildcard import those types are not visible: still flagged.
        no_import = "class A {\n    void f() { Future<Integer> f = null; AcmeThing t = null; }\n}\n"
        self.assertEqual(set(missing(no_import, "java", "x")), {"Future", "AcmeThing"})
        # A standard package whose members are not listed: bare type names pass, invented members don't.
        other = "import java.util.logging.*;\nclass A {\n    void f(Widget w) { Level l = null; w.acmeRun(); }\n}\n"
        self.assertEqual(missing(other, "java", "Widget"), ["w.acmeRun"])

    def test_go_one_line_struct_fields(self):
        code = (
            "package main\n\n"
            "type Acc struct{ total float64 }\n"
            "type Pt struct{ x, y int; label string }\n\n"
            "func (a *Acc) Add(v float64) { a.total += v }\n"
            "func show(p Pt) int { _ = p.label; return p.x + p.y }\n"
            "func bad(a *Acc) float64 { return a.acmeRate }\n"
        )
        self.assertEqual(missing(code, "go", "accumulate"), ["a.acmeRate"])

    def test_long_dotted_chains_stay_fast_and_capped(self):
        chain = ".".join("abcdefghij"[i % 10] for i in range(500))  # 999 chars: under the long-line cut
        code = "\n".join(f"{chain}" for _ in range(200))
        t0 = time.monotonic()
        got = missing(code, "python", "ctx")
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertLessEqual(len(got), grounding.MAX_PATH_SEGMENTS)
        # Distinct roots: the per-check term cap bounds the work.
        code = "\n".join(f"q{k}.{chain[:600]}" for k in range(400))
        t0 = time.monotonic()
        extracted = grounding.TERM_EXTRACTORS["code_symbols"](code, "python")
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertLessEqual(len(extracted.terms), grounding.MAX_TERMS)
        self.assertEqual(len(extracted.terms), len(set(extracted.terms)), "terms are deduplicated")


class ShellHclYamlTests(unittest.TestCase):
    def test_shell(self):
        ctx = "Run acme-cli fetch --id N to fetch a widget."
        code = (
            "#!/bin/bash\nset -euo pipefail\n# acme-bogus in a comment\n"
            "fetch() {\n  local id=\"$1\"\n  acme-cli fetch --id \"$id\" | jq '.name'\n  echo \"acme-bogus in a string\"\n}\n"
            "for i in 1 2 3; do fetch \"$i\"; done\n"
            "count=$(wc -l < input.txt)\n"
            "case \"$1\" in\n  start) acme-cli start ;;\n  stop) acme-other stop ;;\nesac\n"
            "cat <<EOF > report\nacme-heredoc-tool should not count\nEOF\n"
        )
        self.assertEqual(missing(code, "bash", ctx), ["acme-other"])

    def test_hcl(self):
        ctx = 'data "acme_widget" "this" exports id, name and size. resource "acme_bucket" takes name and widget_id.'
        code = (
            'variable "region" {\n  type = string\n}\n'
            'locals {\n  prefix = "app-${var.region}"\n}\n'
            'data "acme_widget" "this" {\n  name = "main"\n}\n'
            'resource "acme_bucket" "b" {\n'
            '  name      = "${local.prefix}-bucket"  # data.acme_widget.this.in_comment\n'
            '  widget_id = data.acme_widget.this.id\n'
            '  size      = data.acme_widget.this.size\n'
            '  colour    = data.acme_widget.this.colour\n'
            '  label     = "data.acme_widget.this.in_string"\n'
            '  tags      = merge({ a = "b" }, { c = "d" })\n'
            '  owner     = upper(var.owner)\n'
            '  x         = local.missing\n'
            '  y         = acme_gizmo.g.id\n'
            '  z         = acme_bucket.b.name\n'
            '}\n'
        )
        got = missing(code, "terraform", ctx)
        self.assertEqual(set(got), {"data.acme_widget.this.colour", "var.owner", "local.missing", "acme_gizmo"})

    def test_yaml_keys(self):
        ctx = "The config file has keys widget and size under acme."
        code = (
            "acme:\n  widget: main\n  size: 3\n  colour: red\n"
            "  labels:\n    team: core\n"                       # author-named key under a free-form parent
            "  name: \"{{ .Values.acme.bogus }}\"\n"           # template expression in a value
            "  {{ .Values.keyName }}: 1\n"                    # template expression as a key
            "  script: |\n    notakey: here\n"                # block scalar text
            "# commented: out\n"
        )
        self.assertEqual(missing(code, "yaml", ctx), ["colour"])

    def test_names_match_across_case_and_separators(self):
        self.assertEqual(missing("w.widgetId\n", "javascript", "each row has widget_id and w"), [])


# ── the check, selection, API ────────────────────────────────────────────


class CheckTests(unittest.TestCase):
    def test_no_context_is_skipped(self):
        r = svc.run_code_checks("x = acme()\n", language="python", check_ids=["context_grounding"])
        self.assertEqual(r["results"][0]["skipped"], "no_context")
        self.assertIsNone(r["ok"])
        r = svc.run_code_checks("x = acme()\n", language="python", check_ids=["context_grounding"],
                                context={"grounding_text": "   "})
        self.assertEqual(r["results"][0]["skipped"], "no_context")

    def test_messages(self):
        r = svc.run_code_checks("w = acme.fetch_widget()\nw.ids\n", language="python", check_ids=["context_grounding"],
                                context={"path": None, "grounding_text": "acme.fetch_widget() returns a widget"})
        self.assertIs(r["ok"], False)
        self.assertEqual(r["results"][0]["messages"], ["`w.ids`: not in the context or defined in the answer"])
        many = "\n".join(f"acme.call_{i}()" for i in range(40))
        r = svc.run_code_checks(many, language="python", check_ids=["context_grounding"],
                                context={"grounding_text": "acme"})
        self.assertEqual(len(r["results"][0]["messages"]), 20)

    def test_language_without_a_family_is_a_mismatch(self):
        r = svc.run_code_checks("{}", language="json", check_ids=["context_grounding"], context={"grounding_text": "x"})
        self.assertEqual(r["results"][0]["skipped"], "language_mismatch")

    def test_pluggable_extractor(self):
        def words(code, language):
            out = ExtractedTerms()
            for w in code.split():
                out.add(w, w)
            return out

        register_term_extractor("zz_words", words)
        try:
            r = svc.run_code_checks("alpha beta", language="python", check_ids=["context_grounding"],
                                    context={"grounding_text": "alpha", "grounding_extractor": "zz_words"})
            self.assertEqual(r["results"][0]["messages"], ["`beta`: not in the context or defined in the answer"])
            r = svc.run_code_checks("x", language="python", check_ids=["context_grounding"],
                                    context={"grounding_text": "x", "grounding_extractor": "zz_nope"})
            self.assertEqual(r["results"][0]["skipped"], "unavailable")
        finally:
            grounding.TERM_EXTRACTORS.pop("zz_words", None)

    def test_large_input_is_fast(self):
        code = "\n".join(f"v{i} = acme.fetch_widget({i}).name.strip()  # c" for i in range(4000))
        t0 = time.monotonic()
        self.assertEqual(unresolved_terms(code, "python", "acme.fetch_widget name"), [])
        self.assertLess(time.monotonic() - t0, 10.0)


class OptInSelectionTests(unittest.TestCase):
    """Registering the check must not change any existing default selection or score."""

    def test_default_selection_is_unchanged(self):
        ctx = {"path": None, "grounding_text": "nothing relevant"}
        self.assertEqual([c.id for c in svc.checks_for_language("python")], ["python_ast"])
        r = svc.run_code_checks("x = acme.invented()\n", language="python", context=ctx)
        self.assertEqual([x["check_id"] for x in r["results"]], ["python_ast"])
        self.assertIs(r["ok"], True)
        for lang in ("yaml", "json", "terraform", "shell", "go", "java", "c", "cpp", "rust", "javascript"):
            self.assertNotIn("context_grounding", [c.id for c in svc.checks_for_language(lang)])
        self.assertNotIn("context_grounding", svc.default_check_ids())
        self.assertEqual(svc.default_check_ids(), ["json_parse", "json_schema", "python_ast", "yaml_parse"])

    def test_listed_in_allowed_joins_the_default(self):
        with Allow("context_grounding"):
            self.assertIn("context_grounding", [c.id for c in svc.checks_for_language("python")])
            self.assertIn("context_grounding", svc.default_check_ids())
            # still needs the grounding text to be selected by default
            r = svc.run_code_checks("x = 1\n", language="python")
            self.assertEqual([x["check_id"] for x in r["results"]], ["python_ast"])

    def test_listing_marks_it_opt_in(self):
        rows = {r["id"]: r for r in svc.list_code_checks()}
        self.assertTrue(rows["context_grounding"]["opt_in"])
        self.assertTrue(rows["context_grounding"]["allowed"])
        self.assertFalse(rows["context_grounding"]["default"])
        self.assertFalse(rows["python_ast"]["opt_in"])
        self.assertTrue(rows["python_ast"]["default"])


class ApiTests(unittest.TestCase):
    def test_run_accepts_grounding_text(self):
        from app.main import app

        with TestClient(app) as client:
            res = client.post("/api/code-checks/run", json={
                "language": "python", "code": "acme.fetch_gadget()\n", "check_ids": ["context_grounding"],
                "context": {"grounding_text": "acme.fetch_widget()", "module_dir": "/etc"},
            })
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["results"][0]["messages"],
                             ["`acme.fetch_gadget`: not in the context or defined in the answer"])
            listed = {c["id"]: c for c in client.get("/api/code-checks").json()["checks"]}
            self.assertTrue(listed["context_grounding"]["opt_in"])


# ── code-example scoring: the calibration metric ─────────────────────────


def _parse(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _row(ref_code: str, pred_code: str, prompt: str) -> dict:
    return {
        "prompt": prompt,
        "reference": json.dumps({"code": ref_code, "rationale": "r", "citations": []}),
        "prediction": json.dumps({"code": pred_code, "rationale": "r", "citations": []}),
        "code_example_row": {"task": "Fetch a widget.", "language": "python", "passages": []},
        "_code_example_judge": {"verdict": "correct", "label": "stub"},
    }


class ScoringCalibrationTests(unittest.TestCase):
    PROMPT = "Write Python. acme.fetch_widget(id) returns a widget."

    def rows(self):
        return [
            # reference grounded; prediction invents a call -> checks fail
            _row("import acme\nacme.fetch_widget(1)\n", "import acme\nacme.fetch_gadget(1)\n", self.PROMPT),
            # reference NOT grounded -> grounding not checkable on this row; python_ast still decides (passes)
            _row("import acme\nacme.undocumented(1)\n", "import acme\nacme.fetch_gadget(1)\n", self.PROMPT),
            # both grounded
            _row("import acme\nacme.fetch_widget(2)\n", "import acme\nw = acme.fetch_widget(2)\n", self.PROMPT),
        ]

    def schema(self, checks):
        return {"scoring_mode": "remediation", "remediation": {"checks": checks}}

    def test_reference_pass_rate_and_per_check_rule(self):
        preds = self.rows()
        m = ces.score_code_examples(preds, self.schema(["python_ast", "context_grounding"]), _parse)
        self.assertEqual(m["code_example_grounding_reference_checked_count"], 3)
        self.assertEqual(m["code_example_grounding_reference_pass_count"], 2)
        self.assertEqual(m["code_example_grounding_reference_pass_rate"], round(2 / 3, 4))
        self.assertEqual([p["row_code_example_grounding_reference_pass"] for p in preds], [True, False, True])
        self.assertEqual([p["row_code_example_checkable"] for p in preds], [True, True, True])
        self.assertEqual([p["row_code_example_checks_pass"] for p in preds], [False, True, True])
        self.assertIn("context_grounding: `acme.fetch_gadget`", preds[0]["row_code_example_check_messages"][0])

    def test_absent_when_not_selected(self):
        preds = self.rows()
        m = ces.score_code_examples(preds, self.schema(["python_ast"]), _parse)
        self.assertFalse(any(k.startswith("code_example_grounding") for k in m))
        self.assertFalse(any("grounding" in k for p in preds for k in p))
        self.assertEqual([p["row_code_example_checks_pass"] for p in preds], [True, True, True])

    def test_default_checks_none_are_unchanged(self):
        preds = self.rows()
        m = ces.score_code_examples(preds, self.schema(None), _parse)
        self.assertFalse(any(k.startswith("code_example_grounding") for k in m))
        self.assertEqual(m["code_example_code_checks_pass_rate"], 1.0)

    def test_falls_back_to_task_and_passages_without_a_prompt(self):
        row = _row("import acme\nacme.fetch_widget(1)\n", "import acme\nacme.fetch_widget(1)\n", "")
        row["code_example_row"]["passages"] = [{"id": "P1", "text": "acme.fetch_widget(id) returns a widget"}]
        m = ces.score_code_examples([row], self.schema(["context_grounding"]), _parse)
        self.assertEqual(m["code_example_grounding_reference_pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
