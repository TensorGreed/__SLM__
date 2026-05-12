# slm-docs/scripts

Generator scripts that produce the auto-reference sections of the docs site. **Re-run these any time the backend API or the `brewslm` CLI changes.** The generated output lives under `docs/reference/api/` and `docs/reference/cli-full.md`.

| Script | Source of truth | Output |
|---|---|---|
| `generate_api_ref.py` | OpenAPI spec from FastAPI | `docs/reference/api/` (one file per tag + an index) |
| `generate_cli_ref.py` | `brewslm <cmd> --help` output | `docs/reference/cli-full.md` (single file) |

## Quickstart

From the repo root:

```sh
# 1. Dump the current OpenAPI spec to a file the docs script will read.
cd backend
PYTHONPATH=. python scripts/dump_openapi.py --out ../slm-docs/openapi.json

# 2. Generate the API reference Markdown.
cd ../slm-docs
python scripts/generate_api_ref.py --spec openapi.json

# 3. Generate the CLI reference Markdown.
python scripts/generate_cli_ref.py
```

Or use the shortcut script in `package.json`:

```sh
cd slm-docs
npm run docs:gen
```

`docs:gen` does both passes in order. Add it to your release / sprint-end checklist so the auto-ref stays current.

## When to regenerate

- After **any change to a FastAPI router** under `backend/app/api/` (new endpoint, removed endpoint, schema change).
- After **any change to `backend/scripts/brewslm.py`** (new subcommand, new flag, help text edit).
- Before every doc-site deploy if you want the auto-reference to match what's actually shipped.

## Why these are separate from the curated docs

The curated docs (`docs/reference/api-surface.md`, `docs/reference/cli.md`) are **narrative**: they show the most-used 30 endpoints + the CLI commands with example commands and link to UI/CLI/API trios on workflow pages. The auto-generated reference is **exhaustive**: every endpoint, every flag, no editorial decisions. Two tools, two audiences.

## Implementation notes

- `dump_openapi.py` (under `backend/scripts/`) imports the FastAPI app and calls `app.openapi()`. It uses an in-memory SQLite DB + a tmp DATA_DIR so the call is side-effect free — no real schema, no real artifacts created.
- `generate_api_ref.py` resolves `$ref` pointers up to depth 3 to keep recursive schemas tractable. It produces flat Markdown tables, not nested JSON dumps, so the result is greppable and small.
- `generate_cli_ref.py` runs `brewslm <cmd> --help` with `COLUMNS=120` and parses out nested subparsers one level deep. Top-level + subcommand help text gets fenced into code blocks.

Both scripts are idempotent: re-running with no API/CLI changes produces byte-identical output.

## Adding more generators

If you want to auto-generate from another source (e.g., reason-code taxonomy from `backend/app/models/reason_codes.py`), drop a new `generate_<thing>.py` here and add it to the `docs:gen` script in `package.json`. Keep the convention: emit Markdown, write to `docs/reference/<thing>/`, prepend the standard `auto-generated` HTML comment.
