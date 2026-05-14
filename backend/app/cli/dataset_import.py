"""CLI entry point for the dataset import pipeline (Phase A).

Runnable as ``python -m app.cli.dataset_import <subcommand> ...``.
When BrewSLM gets a top-level ``brewslm`` binary (separate project),
it can wrap this module via ``brewslm dataset import ...``.

Subcommands:

- ``preview`` — dry-run: runs source → mapper and prints transformed
  samples + rejection breakdown without touching the dataset. Use this
  to validate the mapping before committing a large import.
- ``run`` — same pipeline but persists accepted rows to the project's
  synthetic dataset. Requires a project ID.
- ``sources`` — list registered source connectors.
- ``mappers`` — list registered target mappers.

Common flags:

  --locator       Source-prefixed locator (jsonl:/path or csv:/path).
  --mapper        Target mapper id (e.g. label_to_classification).
  --map K=V       Repeatable per-field mapping. K=text_field,
                  V=review_text remaps the source's "review_text"
                  column to the mapper's "text_field" input.
  --map-json '<json>'
                  JSON object form of --map. Useful when the field map
                  needs nested values (e.g. entity_type_map for
                  bio_to_spans).
  --limit N       Stop after N source rows. Defaults to no limit on
                  ``run``; 100 on ``preview`` to keep the dry-run fast.
  --drop REASON   Bulk-drop rejected rows by reason code (repeatable).
                  Counts still surface in the rejection breakdown.
  --project ID    Project ID for ``run`` (required) and ``preview``
                  (optional — pulls task_profile from the project's
                  prepared manifest when supplied).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any


def _parse_map_pairs(pairs: list[str]) -> dict[str, Any]:
    """Turn ``["text_field=review_text", "label_field=sentiment"]`` into
    ``{"text_field": "review_text", "label_field": "sentiment"}``.
    Reject malformed entries with a clear message rather than
    swallowing them silently."""

    field_map: dict[str, Any] = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise SystemExit(
                f"--map entries must be KEY=VALUE; got '{raw}'"
            )
        key, _, value = raw.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            raise SystemExit(f"--map entry has empty key: '{raw}'")
        field_map[key] = value
    return field_map


def _merge_field_map(
    map_pairs: list[str], map_json: str | None
) -> dict[str, Any]:
    """Combine repeated --map flags and a --map-json object. Pair-form
    fields override JSON-form fields with the same key — small wins
    over the bulk JSON config."""

    base: dict[str, Any] = {}
    if map_json:
        try:
            payload = json.loads(map_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--map-json is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit("--map-json must be a JSON object")
        base.update(payload)
    base.update(_parse_map_pairs(map_pairs))
    return base


def _print_result(payload: dict[str, Any]) -> None:
    """Render a compact, human-readable summary instead of raw JSON.
    Users running the CLI almost always want to eyeball the result;
    those who need the JSON can pipe through ``--json``."""

    print(
        f"source={payload['source_id']}  "
        f"mapper={payload['mapper_id']}  "
        f"target_task_profile={payload['target_task_profile']}"
    )
    print(
        f"accepted={payload['accepted_count']}  "
        f"rejected={payload['rejected_count']}  "
        f"dry_run={payload['dry_run']}"
    )
    if payload.get("written_path"):
        print(f"written to: {payload['written_path']}")
    if payload.get("rejection_counts"):
        print("rejection breakdown:")
        for reason, count in sorted(
            payload["rejection_counts"].items(), key=lambda kv: -kv[1]
        ):
            print(f"  {reason:24} {count}")
    if payload.get("accepted_sample"):
        print(f"first {len(payload['accepted_sample'])} accepted rows:")
        for idx, row in enumerate(payload["accepted_sample"], 1):
            inline = json.dumps(row["payload"], ensure_ascii=False)
            if len(inline) > 200:
                inline = inline[:200] + "…"
            print(f"  [{idx}] {inline}")


# ── Subcommand handlers ───────────────────────────────────────────────


def _cmd_sources(_args: argparse.Namespace) -> int:
    from app.services.dataset_import import list_registered_sources

    for src in list_registered_sources():
        print(src)
    return 0


def _cmd_mappers(_args: argparse.Namespace) -> int:
    from app.services.dataset_import import (
        list_registered_mappers,
        resolve_mapper,
    )

    for mapper_id in list_registered_mappers():
        target = resolve_mapper(mapper_id).declared_target()
        print(f"{mapper_id:30}  → task_profile={target}")
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    from app.services.dataset_import.service import preview_import, result_to_dict

    field_map = _merge_field_map(args.map or [], args.map_json)
    result = preview_import(
        project_id=args.project or 0,
        project_task_profile=None,
        locator=args.locator,
        mapper_id=args.mapper,
        field_map=field_map,
        sample_cap=args.sample_cap,
        limit=args.limit,
        drop_reasons=set(args.drop or []),
    )
    payload = result_to_dict(result)
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        _print_result(payload)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.project is None:
        raise SystemExit("`run` requires --project <id>")

    from app.database import async_session_factory
    from app.services.dataset_import.service import result_to_dict, run_import

    field_map = _merge_field_map(args.map or [], args.map_json)

    async def _go():
        async with async_session_factory() as db:
            result = await run_import(
                db,
                project_id=args.project,
                project_task_profile=None,
                locator=args.locator,
                mapper_id=args.mapper,
                field_map=field_map,
                limit=args.limit,
                drop_reasons=set(args.drop or []),
            )
            await db.commit()
            return result

    result = asyncio.run(_go())
    payload = result_to_dict(result)
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        _print_result(payload)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.cli.dataset_import",
        description=(
            "Import external datasets into a BrewSLM project's synthetic "
            "dataset. See DATASET_IMPORT_PLAN.md for the full pipeline."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    # `sources` / `mappers` — catalog inspection.
    sub.add_parser("sources", help="List registered source connectors")
    sub.add_parser("mappers", help="List registered target mappers")

    # Shared flag set for `preview` + `run`.
    for cmd_name, help_text, is_run in (
        ("preview", "Dry-run a source × mapper combination", False),
        ("run", "Persist accepted rows to the project's synthetic dataset", True),
    ):
        cmd = sub.add_parser(cmd_name, help=help_text)
        cmd.add_argument(
            "--locator",
            required=True,
            help="Source-prefixed locator, e.g. 'jsonl:/tmp/data.jsonl'",
        )
        cmd.add_argument(
            "--mapper",
            required=True,
            help="Target mapper id (use `mappers` to list)",
        )
        cmd.add_argument(
            "--map",
            action="append",
            help="Per-field mapping KEY=VALUE (repeatable). "
            "e.g. --map text_field=review_text",
        )
        cmd.add_argument(
            "--map-json",
            help="JSON-object form of the field map. Use this for "
            "nested values like entity_type_map for bio_to_spans.",
        )
        cmd.add_argument(
            "--limit",
            type=int,
            help="Stop after N source rows. Defaults to no limit on "
            "`run`; sample-based for `preview`.",
        )
        cmd.add_argument(
            "--drop",
            action="append",
            help="Rejection reason code to bulk-drop (repeatable). "
            "Counts still surface in the breakdown.",
        )
        cmd.add_argument(
            "--project",
            type=int,
            required=is_run,
            help=(
                "Project ID — required for `run`, optional for `preview`"
            ),
        )
        cmd.add_argument(
            "--json",
            action="store_true",
            help="Emit raw JSON instead of the human-readable summary",
        )
        if not is_run:
            cmd.add_argument(
                "--sample-cap",
                type=int,
                default=5,
                help="Max accepted rows to keep in the preview "
                "(default: 5)",
            )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "sources": _cmd_sources,
        "mappers": _cmd_mappers,
        "preview": _cmd_preview,
        "run": _cmd_run,
    }
    handler = handlers[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
