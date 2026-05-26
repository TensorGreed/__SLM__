"""Backfill recipes onto demo projects that were seeded before the
demo seeder learned to assign one (the fix landing in this commit).

Scope: matches projects by ``project.name == manifest.name`` for each
of the 3 demo slugs that have bundles on disk. Skips projects that
already have ``selected_recipe`` set — won't clobber an explicit
user choice. Idempotent: safe to re-run.

Usage:
    python -m backend.scripts.backfill_demo_recipes [--dry-run] [--yes]

By default the script asks for confirmation before writing. Pass
``--yes`` to skip the prompt (CI / automation). Pass ``--dry-run``
to print the matches without writing anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select

# Allow ``python backend/scripts/backfill_demo_recipes.py`` (not just
# ``python -m``) by adding the backend dir to sys.path when called
# as a script.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.database import async_session_factory  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.services.demo_project_service import (  # noqa: E402
    DEMO_SAMPLES_DIR,
    DEMO_SLUG_TO_RECIPE_ID,
    _assign_recipe_from_slug_if_missing,
)


def _load_manifest_names() -> dict[str, str]:
    """Return ``{project_name: slug}`` for each demo slug that has a
    manifest on disk AND a recipe mapping. We match on project NAME
    rather than ``dataset_adapter_preset.demo_slug`` because some
    older seed paths didn't write the slug key — name matching catches
    those projects too."""
    out: dict[str, str] = {}
    for slug in DEMO_SLUG_TO_RECIPE_ID:
        manifest_path = DEMO_SAMPLES_DIR / slug / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        name = str(payload.get("name") or "").strip()
        if name:
            out[name] = slug
    return out


async def _find_demo_projects_needing_backfill(
    db, name_to_slug: dict[str, str]
) -> list[tuple[Project, str]]:
    """Return list of (project, slug) pairs for demo-named projects
    whose ``selected_recipe`` is empty/null."""
    if not name_to_slug:
        return []
    result = await db.execute(
        select(Project).where(Project.name.in_(list(name_to_slug.keys())))
    )
    rows: list[tuple[Project, str]] = []
    for project in result.scalars():
        slug = name_to_slug.get(project.name)
        if slug is None:
            continue
        existing = (project.selected_recipe or {}).get("recipe_id") if isinstance(
            project.selected_recipe, dict
        ) else None
        if existing:
            # Already has a recipe — skip silently. The user (or a
            # prior backfill run) set it.
            continue
        rows.append((project, slug))
    return rows


async def run(*, dry_run: bool = False) -> dict[str, Any]:
    """Execute the backfill. Returns a serializable report dict."""
    name_to_slug = _load_manifest_names()
    print(f"[backfill] {len(name_to_slug)} demo manifests resolved: {name_to_slug}")

    async with async_session_factory() as db:
        candidates = await _find_demo_projects_needing_backfill(db, name_to_slug)
        if not candidates:
            print("[backfill] no demo projects need backfill — done")
            return {"backfilled": 0, "skipped": 0, "candidates": []}
        print(f"[backfill] found {len(candidates)} demo project(s) without a recipe:")
        for project, slug in candidates:
            recipe_id = DEMO_SLUG_TO_RECIPE_ID[slug]
            print(f"  - project_id={project.id}  name={project.name!r}  slug={slug}  → recipe={recipe_id}")
        if dry_run:
            print("[backfill] --dry-run; no writes performed")
            return {
                "backfilled": 0,
                "skipped": 0,
                "candidates": [
                    {
                        "project_id": p.id,
                        "name": p.name,
                        "slug": slug,
                        "recipe_id": DEMO_SLUG_TO_RECIPE_ID[slug],
                    }
                    for p, slug in candidates
                ],
            }

        results: list[dict[str, Any]] = []
        backfilled = 0
        skipped = 0
        for project, slug in candidates:
            outcome = await _assign_recipe_from_slug_if_missing(
                db, project, slug, force=False,
            )
            if outcome.get("assigned"):
                backfilled += 1
            else:
                skipped += 1
            results.append({
                "project_id": project.id,
                "name": project.name,
                "slug": slug,
                **outcome,
            })
        await db.commit()
        print(f"[backfill] backfilled={backfilled} skipped={skipped}")
        return {"backfilled": backfilled, "skipped": skipped, "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill recipes onto demo projects.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print matches without writing.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt.")
    args = parser.parse_args(argv)

    if not args.dry_run and not args.yes:
        ok = input("Backfill demo project recipes? [y/N]: ").strip().lower()
        if ok not in {"y", "yes"}:
            print("[backfill] aborted")
            return 1

    asyncio.run(run(dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
