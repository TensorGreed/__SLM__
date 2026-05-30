"""First-class ``sweeps`` table + ``experiments.sweep_id`` FK + backfill.

Replaces the legacy ``config._sweep.sweep_id`` JSON-only breadcrumb with
a real ``sweeps`` table. The hyperparameter-sweep service now reads from
that table for the history sidebar, the inconclusive-verdict coach
nudge, and the pre-flight budget query — none of which need to scan
every experiment in the project anymore.

Backfill: for each existing experiment with a ``config._sweep.sweep_id``
breadcrumb, we group by token, materialise a ``Sweep`` row using the
earliest cell's metadata, and set every cell's ``sweep_id`` FK. Idempotent
on re-run because the existence checks short-circuit.

SQLite quirk: ALTER TABLE … ADD CONSTRAINT isn't supported. The FK on
``experiments.sweep_id`` is declared at the ORM level only; this
migration adds the column without the constraint (matches the pattern
in 0036_project_rag_first_columns).

Revision ID: 20260530_0043
Revises: 20260528_0042
Create Date: 2026-05-30 18:00:00
"""

from __future__ import annotations

import json

from alembic import op
import sqlalchemy as sa


revision = "20260530_0043"
down_revision = "20260528_0042"
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return name in inspector.get_table_names()


def _existing_columns(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    # 1) Create the sweeps table when missing.
    if not _table_exists("sweeps"):
        op.create_table(
            "sweeps",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "project_id",
                sa.Integer(),
                sa.ForeignKey("projects.id"),
                nullable=False,
            ),
            sa.Column("sweep_id", sa.String(32), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("base_model", sa.String(255), nullable=False),
            sa.Column("recipe_id", sa.String(128), nullable=True),
            sa.Column("axes", sa.JSON(), nullable=True),
            sa.Column("quality_target", sa.Float(), nullable=True),
            sa.Column(
                "requested_cells", sa.Integer(), nullable=False, server_default="0"
            ),
        )
        op.create_index("ix_sweeps_project_id", "sweeps", ["project_id"])
        op.create_index("ix_sweeps_sweep_id", "sweeps", ["sweep_id"])

    # 2) Add experiments.sweep_id (nullable, no DB-level FK on SQLite).
    if _table_exists("experiments"):
        existing = _existing_columns("experiments")
        if "sweep_id" not in existing:
            op.add_column(
                "experiments",
                sa.Column("sweep_id", sa.Integer(), nullable=True),
            )
            op.create_index(
                "ix_experiments_sweep_id", "experiments", ["sweep_id"]
            )

    # 3) Backfill: walk every experiment, group cells by their
    #    ``config._sweep.sweep_id`` breadcrumb, materialise a Sweep row
    #    per unique token, set the FK on each cell. Done in one pass so
    #    a re-run is cheap (the WHERE sweep_id IS NULL guard skips
    #    cells we've already linked).
    bind = op.get_bind()

    rows = list(
        bind.execute(
            sa.text(
                "SELECT id, project_id, base_model, config "
                "FROM experiments "
                "WHERE sweep_id IS NULL AND config IS NOT NULL"
            )
        )
    )
    # Aggregate cells by (project_id, sweep_token) so we can hold each
    # group together when materialising the Sweep row.
    grouped: dict[tuple[int, str], dict] = {}
    for row in rows:
        exp_id = int(row.id)
        project_id = int(row.project_id)
        base_model = row.base_model or ""
        config = row.config
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except (json.JSONDecodeError, ValueError):
                continue
        if not isinstance(config, dict):
            continue
        sweep_meta = config.get("_sweep") if isinstance(config, dict) else None
        if not isinstance(sweep_meta, dict):
            continue
        token = str(sweep_meta.get("sweep_id") or "").strip()
        if not token:
            continue
        bucket = grouped.setdefault(
            (project_id, token),
            {
                "project_id": project_id,
                "sweep_id": token,
                "base_model": base_model,
                "recipe_id": None,
                "axes": None,
                "quality_target": None,
                "cell_ids": [],
            },
        )
        bucket["cell_ids"].append(exp_id)
        # Prefer the first cell's base_model + axes; keep the first
        # quality_target we see (every cell carries the same one).
        if "quality_target" not in sweep_meta and bucket["quality_target"] is None:
            pass
        elif bucket["quality_target"] is None and sweep_meta.get("quality_target") is not None:
            try:
                bucket["quality_target"] = float(sweep_meta.get("quality_target"))
            except (TypeError, ValueError):
                pass
        if bucket["axes"] is None and isinstance(sweep_meta.get("axis_values"), dict):
            # The legacy breadcrumb stored per-cell axis values; we
            # collapse them into a coarse "axes" record (just for
            # display) — the sidebar shows what was swept, not the
            # exact grid (which is reconstructible from the cells).
            bucket["axes"] = {"per_cell": True}

    for key, bucket in grouped.items():
        project_id, token = key
        # Skip if a Sweep already exists for this token (re-running the
        # migration after a partial run, or after the app wrote one
        # itself between alembic upgrades).
        existing_sweep = bind.execute(
            sa.text(
                "SELECT id FROM sweeps WHERE project_id = :pid AND sweep_id = :sid"
            ),
            {"pid": project_id, "sid": token},
        ).first()
        if existing_sweep:
            sweep_pk = int(existing_sweep.id)
        else:
            result = bind.execute(
                sa.text(
                    "INSERT INTO sweeps "
                    "(project_id, sweep_id, base_model, recipe_id, axes, "
                    "quality_target, requested_cells, created_at) "
                    "VALUES (:pid, :sid, :bm, :rec, :ax, :qt, :rc, CURRENT_TIMESTAMP)"
                ),
                {
                    "pid": project_id,
                    "sid": token,
                    "bm": bucket["base_model"],
                    "rec": bucket["recipe_id"],
                    "ax": json.dumps(bucket["axes"]) if bucket["axes"] is not None else None,
                    "qt": bucket["quality_target"],
                    "rc": len(bucket["cell_ids"]),
                },
            )
            sweep_pk = int(result.lastrowid)
        # Wire the FK on every cell that doesn't have one yet.
        if bucket["cell_ids"]:
            placeholders = ",".join(str(int(i)) for i in bucket["cell_ids"])
            bind.execute(
                sa.text(
                    f"UPDATE experiments SET sweep_id = :pk "
                    f"WHERE id IN ({placeholders}) AND sweep_id IS NULL"
                ),
                {"pk": sweep_pk},
            )


def downgrade() -> None:
    if _table_exists("experiments"):
        existing = _existing_columns("experiments")
        if "sweep_id" in existing:
            op.drop_index("ix_experiments_sweep_id", table_name="experiments")
            op.drop_column("experiments", "sweep_id")
    if _table_exists("sweeps"):
        op.drop_index("ix_sweeps_sweep_id", table_name="sweeps")
        op.drop_index("ix_sweeps_project_id", table_name="sweeps")
        op.drop_table("sweeps")
