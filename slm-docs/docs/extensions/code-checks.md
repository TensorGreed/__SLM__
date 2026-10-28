---
sidebar_position: 7
title: Code checks
---

# Code checks

A **code check** answers one cheap, deterministic question about a piece of code, usually "does it still parse?". Remediation scoring runs them on the original chunk and on the patched chunk; you can also run them yourself over HTTP.

Every result has the same shape:

```json
{"check_id": "python_ast", "ok": true, "messages": [], "skipped": null}
```

`ok: null` means the check did not run; `skipped` says why: `unavailable` (binary missing, or the sandbox refused), `not_allowed`, `language_mismatch`, `timeout`, `no_schema`, `no_module_dir`, `unknown_check`. Messages are capped at 20, 300 characters each. A run's overall `ok` is true when every check that ran passed, and `null` when none ran.

Language ids are lower-case; aliases are normalised (`py` → `python`, `js` → `javascript`, `sh`/`bash` → `shell`, `tf`/`hcl` → `terraform`, `cs` → `csharp`, `golang` → `go`, `yml` → `yaml`).

## Built-in checks (always on, in-process)

| id | language | what it does |
|---|---|---|
| `python_ast` | python | `ast.parse`; an indented fragment is retried dedented |
| `json_parse` | json | `json.loads` |
| `yaml_parse` | yaml | safe YAML load, aliases refused; template expressions (`{{ … }}`, `{% … %}`, Helm / Jinja) are replaced by a placeholder first, so templated YAML is checked as YAML |
| `json_schema` | json | validates the JSON against `context.json_schema`; skipped (`no_schema`) without one |

## External checks (sandboxed, opt-in)

An external check runs only when its id is in `CODE_CHECK_ALLOWED` **and** its binary is on `PATH`. Chunks are fragments, so each check wraps a fragment when needed and counts only syntax-class diagnostics as failures; unresolved symbols or missing packages are ignored.

| id | binary | notes |
|---|---|---|
| `java_syntax` | `javac` | `-proc:none -Xlint:none`; a fragment without a top-level type is wrapped in a class, then in a method. Only syntax diagnostics (expected, illegal start, not a statement, reached end of file, unclosed, orphaned, class/interface/enum/record expected, ...) fail |
| `go_syntax` | `gofmt -e` | a fragment is prefixed with a package clause, then wrapped in a function |
| `go_vet` | `go vet` | whole files with a package clause only |
| `terraform_fmt` | `terraform fmt -check` | |
| `terraform_validate` | `terraform validate` | needs a module directory in the server-side context; otherwise skipped |
| `shellcheck` | `shellcheck -f gcc --severity=error` | Fails on errors only (warnings and style are not failures); a snippet without a shebang or `# shellcheck shell=` directive is checked as bash |
| `node_check` | `node --check` | |
| `csharp_syntax` | `csc` / `mcs` | best effort |

Settings (environment or `.env`):

| setting | default | meaning |
|---|---|---|
| `CODE_CHECK_ALLOWED` | empty | comma list of external check ids that may run |
| `CODE_CHECK_PLUGIN_MODULES` | empty | comma list of plugin modules (below) |
| `CODE_CHECK_NETWORK_ISOLATION` | `auto` | `auto` isolates the network when `unshare -rn` works on the host, `off` never, `require` refuses to run external checks without it |

## The sandbox

External commands run through `app/services/sandbox_exec.py`: a fresh temp directory per call (also `HOME` and `TMPDIR`), no shell, a minimal environment with no API keys and no proxy variables, its own process group (killed on timeout), `setrlimit` for CPU seconds, file size, data segment, process count and core dumps, and capped output. `GET /api/code-checks` reports whether network isolation is available; every sandbox result records honestly whether it was isolated. This is defence in depth for locally produced model output, not a hardened boundary: on a host where `unshare -rn` is unavailable (for example, unprivileged user namespaces disabled) `auto` silently falls back to **no isolation**: checks still run, with network access. Set `CODE_CHECK_NETWORK_ISOLATION=require` to refuse to run external checks there instead. The remediation settings card marks ready external checks "network not isolated" when isolation is unavailable. `go_vet` keeps its build cache and `GOPATH` inside each call's temp directory, so nothing is shared between calls.

## Plugins

Add your own checks (or a finder for a language the built-ins skip) by pointing `CODE_CHECK_PLUGIN_MODULES` at a module that exports either `register_code_checks(register)` or a `CODE_CHECKS` list:

```python
from app.services.code_checks.base import BaseCheck, make_result

class RubySyntax(BaseCheck):
    id = "ruby_syntax"
    languages = frozenset({"ruby"})
    kind = "external"      # builtin | external | detector
    binary = "ruby"

    def run(self, code, *, language, context):
        # call app.services.sandbox_exec.run_sandboxed(...) here
        return make_result(self.id, True)

def register_code_checks(register):
    register(RubySyntax())
```

Only `kind="builtin"` skips the allow-list; external and detector checks still need their id in `CODE_CHECK_ALLOWED`. A failing plugin import is recorded and does not stop the rest from loading.

## HTTP API

- `GET /api/code-checks` returns `{checks: [{id, languages, kind, binary, available, allowed}], network_isolation: {mode, available}}`.
- `POST /api/code-checks/run` with `{language, code, check_ids?}` (code up to 200,000 characters; admin or engineer when auth is on) returns `{ok, results}`. Without `check_ids` it runs every available and allowed check for the language; naming ids reports skipped ones too.
