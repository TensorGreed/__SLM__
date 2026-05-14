"""CLI entry point for the dataset import pipeline (Phase A + B).

Runnable as ``python -m app.cli.dataset_import <subcommand> ...``.
When BrewSLM gets a top-level ``brewslm`` binary (separate project),
it can wrap this module via ``brewslm dataset import ...``.

Subcommands:

- ``introspect`` — Phase B: sniff the source, propose a mapping.
  Prints the column signatures, ranked hypotheses, and the top
  proposal (mapper + field_map + confidence). Doesn't touch data.
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
                  Omit when using ``--auto`` — the introspector picks it.
  --map K=V       Repeatable per-field mapping. K=text_field,
                  V=review_text remaps the source's "review_text"
                  column to the mapper's "text_field" input.
  --map-json '<json>'
                  JSON object form of --map. Useful when the field map
                  needs nested values (e.g. entity_type_map for
                  bio_to_spans).
  --auto          Phase B: skip --mapper / --map, let the introspector
                  pick the highest-confidence mapping. Fails the
                  invocation if confidence < 0.8 unless ``--force``.
                  ``--map`` / ``--map-json`` passed alongside ``--auto``
                  override the introspector's suggested field_map keys.
  --force         Allow ``--auto`` to proceed below the confidence
                  threshold. Use only after eyeballing the
                  ``introspect`` output.
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


def _print_introspection(payload: dict[str, Any]) -> None:
    """Human-readable rendering of an introspection result."""

    print(f"source={payload['source_id']}  locator={payload['locator']}")
    if payload.get("approximate_total_rows") is not None:
        print(f"approximate rows: {payload['approximate_total_rows']}")
    print("columns:")
    for sig in payload.get("column_signatures") or []:
        notes = f"  ({sig['notes']})" if sig.get("notes") else ""
        unique = (
            f"  unique={sig['unique_values'][:5]}"
            if sig.get("unique_values")
            else ""
        )
        print(
            f"  {sig['name']:24} {sig['column_type']:20} "
            f"confidence={sig['confidence']:.2f}{unique}{notes}"
        )
    hypotheses = payload.get("hypotheses") or []
    if not hypotheses:
        print(
            "no mapping hypothesis matched the registered mappers — "
            "either the dataset has an unsupported shape, or you'll "
            "need to call `preview` with --mapper + --map directly."
        )
        return
    print("ranked hypotheses:")
    for idx, hyp in enumerate(hypotheses, 1):
        warn = (
            "  [WARN " + "; ".join(hyp["warnings"]) + "]"
            if hyp.get("warnings")
            else ""
        )
        print(
            f"  [{idx}] mapper={hyp['mapper_id']}  "
            f"task={hyp['target_task_profile']}  "
            f"confidence={hyp['confidence']:.2f}{warn}"
        )
        print(f"      field_map={json.dumps(hyp['field_map'], ensure_ascii=False)}")
        print(f"      rationale: {hyp['rationale']}")
    proposal = payload.get("proposal")
    if proposal:
        gate = (
            " — needs --force (below confidence threshold)"
            if proposal["needs_force"]
            else " — safe to --auto"
        )
        print(
            f"top proposal: {proposal['mapper_id']} "
            f"(confidence {proposal['confidence']:.2f}){gate}"
        )


def _cmd_introspect(args: argparse.Namespace) -> int:
    from app.services.dataset_import.service import introspect_locator

    payload = asyncio.run(
        introspect_locator(
            args.locator,
            sample_size=args.sample_size,
            llm_assist=bool(getattr(args, "llm_assist", False)),
        )
    )
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        _print_introspection(payload)
    return 0


def _resolve_via_auto(
    locator: str, force: bool, *, llm_assist: bool = False,
) -> tuple[str, dict[str, Any], str]:
    """Run the introspector and return ``(mapper_id, field_map, rationale)``.

    Enforces the confidence gate: below ``CONFIDENCE_HIGH`` and without
    ``--force`` we exit with a clear message naming the runner's escape
    hatch (run ``introspect`` first, then re-invoke with ``--force`` or
    pass ``--mapper`` explicitly).
    """

    from app.services.dataset_import.introspector import CONFIDENCE_HIGH
    from app.services.dataset_import.service import introspect_locator

    payload = asyncio.run(introspect_locator(locator, llm_assist=llm_assist))
    proposal = payload.get("proposal")
    if not proposal:
        raise SystemExit(
            "--auto: introspector could not match any registered mapper. "
            "Run `introspect` to see the column signatures, then pass "
            "--mapper / --map explicitly."
        )
    confidence = float(proposal["confidence"])
    if confidence < CONFIDENCE_HIGH and not force:
        raise SystemExit(
            f"--auto: proposal confidence {confidence:.2f} is below the "
            f"{CONFIDENCE_HIGH:.2f} threshold. Re-run `introspect` to "
            "inspect the rationale, then pass --force if the proposal is "
            "correct, or override with --mapper / --map."
        )
    return proposal["mapper_id"], dict(proposal["field_map"]), proposal["rationale"]


def _resolve_mapper_and_map(
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    """Combine --auto with --mapper / --map / --map-json overrides.

    Without --auto: --mapper is required (argparse enforces it via the
    callsite); field_map comes purely from --map and --map-json.
    With --auto: pull both from the introspector's proposal, then layer
    explicit --mapper / --map overrides on top.
    """

    explicit_field_map = _merge_field_map(args.map or [], args.map_json)
    if not args.auto:
        if not args.mapper:
            raise SystemExit(
                "--mapper is required (or pass --auto to let the "
                "introspector pick one)."
            )
        return args.mapper, explicit_field_map

    mapper_id, suggested_map, rationale = _resolve_via_auto(
        args.locator,
        args.force,
        llm_assist=bool(getattr(args, "llm_assist", False)),
    )
    print(f"--auto picked mapper '{mapper_id}': {rationale}")
    if args.mapper and args.mapper != mapper_id:
        print(f"--mapper override: using '{args.mapper}' instead")
        mapper_id = args.mapper
    field_map = dict(suggested_map)
    field_map.update(explicit_field_map)
    return mapper_id, field_map


def _cmd_preview(args: argparse.Namespace) -> int:
    from app.services.dataset_import.service import preview_import, result_to_dict

    mapper_id, field_map = _resolve_mapper_and_map(args)
    result = preview_import(
        project_id=args.project or 0,
        project_task_profile=None,
        locator=args.locator,
        mapper_id=mapper_id,
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

    mapper_id, field_map = _resolve_mapper_and_map(args)

    async def _go():
        async with async_session_factory() as db:
            result = await run_import(
                db,
                project_id=args.project,
                project_task_profile=None,
                locator=args.locator,
                mapper_id=mapper_id,
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

    # `introspect` — Phase B dry-introspect (no preview, no run).
    intro = sub.add_parser(
        "introspect",
        help="Sniff the source + propose a mapping (no data transform)",
    )
    intro.add_argument(
        "--locator",
        required=True,
        help="Source-prefixed locator, e.g. 'jsonl:/tmp/data.jsonl'",
    )
    intro.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of sample rows the sniffer reads (default: 20)",
    )
    intro.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of the human-readable summary",
    )
    intro.add_argument(
        "--llm-assist",
        dest="llm_assist",
        action="store_true",
        help="Phase H: also ask the teacher model for a mapping "
        "suggestion. Opt-in; falls through silently when "
        "DATASET_IMPORT_LLM_ASSIST_ENABLED is off or the teacher "
        "API URL isn't configured.",
    )

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
            required=False,
            help="Target mapper id (use `mappers` to list). Omit when "
            "--auto is set; the introspector picks one.",
        )
        cmd.add_argument(
            "--auto",
            action="store_true",
            help="Let the schema introspector pick the mapper + "
            "field_map. Requires confidence ≥ 0.8 unless --force.",
        )
        cmd.add_argument(
            "--force",
            action="store_true",
            help="Allow --auto to proceed even when the proposal's "
            "confidence is below the safe threshold.",
        )
        cmd.add_argument(
            "--llm-assist",
            dest="llm_assist",
            action="store_true",
            help="Phase H: also consult the teacher model during "
            "--auto introspection. Falls through silently when not "
            "configured; never overrides a higher-confidence "
            "deterministic proposal.",
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
        "introspect": _cmd_introspect,
        "preview": _cmd_preview,
        "run": _cmd_run,
    }
    handler = handlers[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
