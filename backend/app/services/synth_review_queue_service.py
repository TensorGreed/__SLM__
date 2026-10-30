"""Synth review queue service (USER-SUCCESS Epic 2b).

Lists pending synthetic rows grouped by ``synth_source``, supports
bulk accept / reject, and rewrites the project's synthetic.jsonl
file so accepted rows enter training and rejected rows are
permanently removed.

The Epic 2a write path always stamps ``review_status="pending"`` so
no synth row enters dataset prep until the user (or an auto-accept
flow) flips it. The dataset_service's JSONL loader skips pending
rows by default — this service is what un-blocks them.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.dataset import Dataset, DatasetType


ReviewAction = Literal["accept", "reject"]


def _synthetic_jsonl_path(project_id: int) -> Path:
    return settings.DATA_DIR / "projects" / str(project_id) / "synthetic" / "synthetic.jsonl"


def _read_all_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _ensure_row_ids(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give every row without an integer ``id`` the next free one and save.
    The queue acts on rows by id, so a row written without one (any writer
    that forgot) could never be accepted — and pending rows never train —
    so it was silently left out of training for good."""
    used = [r["id"] for r in rows if isinstance(r.get("id"), int) and not isinstance(r.get("id"), bool)]
    next_id = (max(used) if used else 0) + 1
    changed = False
    for row in rows:
        rid = row.get("id")
        if not isinstance(rid, int) or isinstance(rid, bool):
            row["id"] = next_id
            next_id += 1
            changed = True
    if changed:
        _atomic_write_rows(path, rows)
    return rows


def _atomic_write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the full row list back to disk via a temp-file + rename
    so a crash mid-write can't truncate the synthetic dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.move(str(tmp), str(path))


# Where a row came from: the Job that wrote it, the backend pin, and whether it went through a
# Message Batch (stamped by the playbook append; absent on rows written before that).
PROVENANCE_KEYS = ("synth_job_id", "synth_backend", "synth_batch")
# ``review_assist`` (the LLM pre-screen verdict) is merged onto rows at read time from the
# sidecar (``synth_review_assist_service``); a legacy / hand-written row may carry it inline.
NO_ASSIST_LABEL = "(not pre-screened)"
ASSIST_KEYS = ("review_assist", "review_assist_judges", "assist_consensus")
# How and when a row's accept / reject was decided (stamped by ``bulk_update_review_queue``):
# ``decided_via`` is ``manual`` (hand-ticked rows), ``group:<key>`` (a group action in the UI),
# ``source:<synth_source>`` (by-source bulk) or ``api`` (a caller that did not say). Record keeping
# only; rows decided before this carry neither key.
DECISION_KEYS = ("decided_via", "decided_at")
DECIDED_VIA_MAX = 128
DECIDED_VIA_DEFAULT = "api"
_DECIDED_VIA_PREFIXES = ("group:", "source:")


def normalize_decided_via(value: Any) -> str:
    """``decided_via`` checked loosely: ``manual`` / ``api`` / ``group:<key>`` / ``source:<key>`` with a
    non-empty key, control characters dropped, cut to ``DECIDED_VIA_MAX`` chars; ``None`` / blank ->
    ``api``. Anything else raises ``ValueError``."""
    if value is None:
        return DECIDED_VIA_DEFAULT
    text = "".join(ch for ch in str(value) if ch.isprintable()).strip()[:DECIDED_VIA_MAX]
    if not text:
        return DECIDED_VIA_DEFAULT
    if text in ("manual", "api"):
        return text
    if text.startswith(_DECIDED_VIA_PREFIXES) and text.split(":", 1)[1].strip():
        return text
    raise ValueError("decided_via must be 'manual', 'api', 'group:<key>' or 'source:<key>'")


def _row_preview(row: dict[str, Any], *, limit: int = 280) -> str:
    """Single-line preview string used by the queue list. Drops the
    provenance fields and serializes only the payload-shaped keys."""
    skip = {
        "id", "synth_source", "synth_confidence", "review_status", "status", "generated_at", "source", "model",
        *ASSIST_KEYS, *PROVENANCE_KEYS, *DECISION_KEYS,
    }
    payload = {k: v for k, v in row.items() if k not in skip}
    if not payload:
        payload = row
    raw = json.dumps(payload, ensure_ascii=False, default=str)
    if len(raw) <= limit:
        return raw
    return raw[: limit - 1] + "…"


def _resolve_source_label(row: dict[str, Any]) -> str:
    """Pick a human-meaningful source for a synth row.

    Epic-2 playbook rows carry an explicit ``synth_source`` like
    ``playbook:classification:hard_negatives:vs=billing``. Legacy
    rows (from the pre-Epic-2 ``/synthetic/generate`` flow) don't
    have ``synth_source`` but do have a legacy ``source`` field
    (``teacher_model`` / ``demo_heuristic``). We surface that
    instead so the user sees ``legacy:teacher_model`` rather than
    ``playbook:unknown``.
    """
    explicit = row.get("synth_source")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    legacy = row.get("source")
    if isinstance(legacy, str) and legacy.strip():
        return f"legacy:{legacy}"
    return "legacy:manual"


# Hard cap on rows-per-group included in the response. A 1000-row
# legacy bucket would otherwise blow up the payload — and the UI
# can't usefully render 1000 rows inline anyway. The frontend gets a
# `truncated` flag + a `total_in_group` count so it can show
# "showing 25 of 1000" footers.
ACCEPTED_ROWS_PER_GROUP_CAP = 25


def _confidence_sort_key(row: dict[str, Any]) -> tuple[float, int]:
    """Sort key for confidence-ascending order: (confidence, id). Rows with a
    missing / non-numeric ``synth_confidence`` sort to the end (``inf``) — they
    carry no uncertainty signal to rank, so they shouldn't crowd out the
    genuinely-low-confidence rows a reviewer should see first."""
    conf = row.get("synth_confidence")
    conf_val = float(conf) if isinstance(conf, (int, float)) else float("inf")
    rid = row.get("id")
    return (conf_val, int(rid) if isinstance(rid, int) else 0)


def _resolve_class_label(row: dict[str, Any]) -> str:
    """Group key when grouping by class — the row's ``label`` (classification
    target). Unlabeled / non-classification rows fall into one bucket so the
    grouping never silently drops them."""
    label = row.get("label")
    if label is not None and str(label).strip():
        return str(label)
    return "(unlabeled)"


def _resolve_cluster_label(row: dict[str, Any]) -> str:
    """Group key when grouping by cluster — the originating failure cluster
    (CLUSTER_TARGETED rows carry ``cluster_id``). Rows generated outside a
    cluster fall into one bucket."""
    cluster = row.get("cluster_id")
    if cluster is not None and str(cluster).strip():
        return str(cluster)
    return "(no cluster)"


def _row_job_id(row: dict[str, Any]) -> int | None:
    jid = row.get("synth_job_id")
    return jid if isinstance(jid, int) and not isinstance(jid, bool) else None


def _resolve_job_label(row: dict[str, Any]) -> str:
    """Group key when grouping by Job — ``job:<id>`` (rows not written by a Job share one bucket)."""
    jid = _row_job_id(row)
    return f"job:{jid}" if jid is not None else "(no job)"


def _assist(row: dict[str, Any]) -> dict[str, Any] | None:
    a = row.get("review_assist")
    return a if isinstance(a, dict) and a.get("verdict") else None


def _resolve_verdict_label(row: dict[str, Any]) -> str:
    """Group key when grouping by pre-screen verdict (pass / fail / unsure)."""
    a = _assist(row)
    return str(a["verdict"]) if a else NO_ASSIST_LABEL


def _resolve_assist_reason_label(row: dict[str, Any]) -> str:
    """Group key when grouping by pre-screen reason: ``<verdict>:<reason_code>``."""
    a = _assist(row)
    return f"{a['verdict']}:{a.get('reason_code') or 'other'}" if a else NO_ASSIST_LABEL


def _judges(row: dict[str, Any]) -> list[dict[str, Any]]:
    j = row.get("review_assist_judges")
    return [e for e in j if isinstance(e, dict) and e.get("verdict")] if isinstance(j, list) else []


def _consensus(row: dict[str, Any]) -> str:
    c = row.get("assist_consensus")
    return c if isinstance(c, str) and c else "none"


def _resolve_consensus_label(row: dict[str, Any]) -> str:
    """Group key when grouping by judge consensus: none / single:<verdict> / agree_pass / agree_fail /
    disagree (a lone judge's pass is its own group so it can be bulk-accepted without mixing in fails)."""
    c = _consensus(row)
    if c == "single":
        return f"single:{_judges(row)[0]['verdict']}"
    return c


def _consensus_group_label(entries: list[dict[str, Any]]) -> str:
    key = _resolve_consensus_label(entries[0])
    names: list[str] = []
    for r in entries:
        for e in _judges(r):
            if e["judge"] not in names:
                names.append(e["judge"])
    head = key.replace("single:", "single · ").replace("_", " ")
    return f"{head} · {' + '.join(names)}" if names else head


def _resolve_backend_label(row: dict[str, Any]) -> str:
    """Group key when grouping by backend — the pin that wrote the row."""
    backend = row.get("synth_backend")
    if isinstance(backend, str) and backend.strip():
        return backend.strip()
    return "(unknown backend)"


def _mode_of(row: dict[str, Any]) -> str:
    """The generation mode of a row, from its ``synth_source`` (``playbook:<recipe>:<mode>[:...]``)."""
    source = _resolve_source_label(row)
    parts = source.split(":")
    if parts[0] == "playbook" and len(parts) >= 3:
        return parts[2]
    return parts[1] if parts[0] == "legacy" and len(parts) >= 2 else source


def _most_common(values: list[str]) -> str | None:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(sorted(counts), key=lambda k: counts[k]) if counts else None


def _group_label(grouping: str | None, entries: list[dict[str, Any]]) -> str | None:
    """A human label for a job / backend group so the reviewer can correlate: Job id + backend +
    mode + direct / batch (a group of mixed backends or modes lists the most common and says so)."""
    if grouping == "assist_consensus" and entries:
        return _consensus_group_label(entries)
    if grouping not in ("job", "backend"):
        return None
    backend_keys = [_resolve_backend_label(r) for r in entries]
    mode_keys = [_mode_of(r) for r in entries]
    backends = sorted(set(backend_keys))
    modes = sorted(set(mode_keys))
    batch = {bool(r.get("synth_batch")) for r in entries}
    jobs = sorted({_resolve_job_label(r) for r in entries})
    backend_text = backends[0] if len(backends) == 1 else f"{_most_common(backend_keys)} +{len(backends) - 1} more"
    mode_text = modes[0] if len(modes) == 1 else f"{_most_common(mode_keys)} +{len(modes) - 1} more"
    delivery = "batch" if batch == {True} else "direct" if batch == {False} else "batch + direct"
    if grouping == "job":
        return f"{jobs[0] if len(jobs) == 1 else str(len(jobs)) + ' jobs'} · {backend_text} · {mode_text} · {delivery}"
    return f"{backend_text} · {mode_text} · {delivery} · {len(jobs)} job{'s' if len(jobs) != 1 else ''}"


def _bucket_rows_by_source(
    rows: list[dict[str, Any]],
    *,
    rows_per_group_cap: int | None = None,
    key_fn: Any = None,
    conventions_current: str | None = None,
    grouping: str | None = None,
) -> list[dict[str, Any]]:
    """Group rows by a key (``synth_source`` by default, or the class label
    when ``key_fn`` is ``_resolve_class_label``), sort, project to the UI shape.

    The output group still carries its key in ``synth_source`` regardless of the
    grouping dimension, so the review UI renders either grouping transparently
    (it just shows the group key + its rows).

    When ``rows_per_group_cap`` is set, each group's ``rows`` array
    is truncated to that length + a ``truncated`` flag is emitted.
    The pending queue passes None (we want every row in the UI for
    accept/reject); the accepted view passes the cap because
    legacy buckets may have thousands of rows.
    """
    from app.services.span_type_definitions_service import conventions_status

    resolve = key_fn or _resolve_source_label
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = resolve(row)
        groups[source].append(row)
    grouped: list[dict[str, Any]] = []
    for source in sorted(groups):
        # Sort by confidence ASCENDING so the most uncertain rows — the ones a
        # reviewer's attention is worth most on — surface first. Rows with no
        # ``synth_confidence`` trail (we can't rank them as uncertain); ``id``
        # breaks ties for a stable order.
        entries = sorted(groups[source], key=_confidence_sort_key)
        total_in_group = len(entries)
        if rows_per_group_cap is not None and total_in_group > rows_per_group_cap:
            visible_entries = entries[:rows_per_group_cap]
            truncated = True
        else:
            visible_entries = entries
            truncated = False
        grouped.append({
            "synth_source": source,
            # job / backend groupings: "job:12 · cloud:anthropic:<model> · <mode> · batch".
            "label": _group_label(grouping, groups[source]),
            "count": total_in_group,
            "truncated": truncated,
            # Span rows labelled under older labelling conventions than the
            # project's current ones (None when nothing to compare).
            "conventions": conventions_status(groups[source], conventions_current),
            "rows": [
                {
                    "id": row.get("id"),
                    "synth_confidence": row.get("synth_confidence"),
                    "synth_job_id": _row_job_id(row),
                    "synth_backend": row.get("synth_backend") if isinstance(row.get("synth_backend"), str) else None,
                    "synth_batch": bool(row.get("synth_batch")),
                    "review_assist": _assist(row),
                    "review_assist_judges": _judges(row),
                    "assist_consensus": _consensus(row),
                    "decided_via": row.get("decided_via") if isinstance(row.get("decided_via"), str) else None,
                    "decided_at": row.get("decided_at") if isinstance(row.get("decided_at"), str) else None,
                    "preview": _row_preview(row),
                    "payload": {
                        k: v for k, v in row.items()
                        if k not in {"id", "synth_source", "synth_confidence", "review_status", "status", *ASSIST_KEYS, *PROVENANCE_KEYS}
                    },
                }
                for row in visible_entries
            ],
        })
    return grouped


def _provenance_facets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Which Jobs / backends wrote ``rows`` (with counts), for the queue's filter pickers."""
    jobs: dict[int, int] = {}
    backends: dict[str, int] = {}
    for row in rows:
        jid = _row_job_id(row)
        if jid is not None:
            jobs[jid] = jobs.get(jid, 0) + 1
        backend = _resolve_backend_label(row)
        backends[backend] = backends.get(backend, 0) + 1
    verdicts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for row in rows:
        a = _assist(row)
        if a:
            verdicts[str(a["verdict"])] = verdicts.get(str(a["verdict"]), 0) + 1
            reasons[str(a.get("reason_code") or "other")] = reasons.get(str(a.get("reason_code") or "other"), 0) + 1
    consensus: dict[str, int] = {}
    judge_counts: dict[str, int] = {}
    for row in rows:
        if _judges(row):
            consensus[_consensus(row)] = consensus.get(_consensus(row), 0) + 1
        for e in _judges(row):
            judge_counts[e["judge"]] = judge_counts.get(e["judge"], 0) + 1
    return {
        "consensus": [{"consensus": c, "count": consensus[c]} for c in sorted(consensus)],
        "judges": [{"judge": j, "count": judge_counts[j]} for j in sorted(judge_counts)],
        "jobs": [{"synth_job_id": j, "count": jobs[j]} for j in sorted(jobs)],
        "backends": [{"synth_backend": b, "count": backends[b]} for b in sorted(backends)],
        "verdicts": [{"verdict": v, "count": verdicts[v]} for v in sorted(verdicts)],
        "reason_codes": [{"reason_code": r, "count": reasons[r]} for r in sorted(reasons)],
    }


async def list_review_queue(
    db: AsyncSession,
    project_id: int,
    *,
    group_by: str = "source",
    synth_job_id: int | None = None,
    synth_backend: str | None = None,
    verdict: str | None = None,
    reason_code: str | None = None,
    assist_consensus: str | None = None,
    judge: str | None = None,
) -> dict[str, Any]:
    """Return synth rows for a project, split into pending + accepted
    groups (keyed by ``synth_source``, or by class label when
    ``group_by="class"``, or by originating failure cluster when
    ``group_by="cluster"`` — Epic E).

    The pending list is the active review queue. The accepted list
    shows the rows that already passed review and will enter the
    next dataset prep — answers the user's question of *where do
    approved synth rows show up?*"""
    # Confirm the Synthetic dataset exists for the project — if it
    # doesn't, the queue is empty, return an empty payload.
    ds_result = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project_id,
            Dataset.dataset_type == DatasetType.SYNTHETIC,
        )
    )
    dataset = ds_result.scalar_one_or_none()
    path = _synthetic_jsonl_path(project_id)
    all_rows = _ensure_row_ids(path, _read_all_rows(path))
    # Pre-screen verdicts live in a sidecar (never in the jsonl): merge them on for display.
    from app.services.synth_review_assist_service import attach_verdicts, judge_stats

    attach_verdicts(project_id, all_rows)
    # Facets describe every pending row; the filters narrow the three lists below
    # (the totals stay project-wide so the strip still adds up).
    facets = _provenance_facets([r for r in all_rows if r.get("review_status") == "pending"])
    # Per-judge pass rates (+ the rubber-stamp flag) over EVERY row a judge has a current verdict on —
    # accepting rows must not shrink a judge's record.
    facets["judge_stats"] = judge_stats(all_rows)
    total_rows_all = len(all_rows)
    if synth_job_id is not None:
        all_rows = [r for r in all_rows if _row_job_id(r) == synth_job_id]
    if synth_backend:
        all_rows = [r for r in all_rows if _resolve_backend_label(r) == synth_backend]
    if assist_consensus:
        all_rows = [r for r in all_rows if _consensus(r) == assist_consensus or _resolve_consensus_label(r) == assist_consensus]
    if judge:
        # View one judge's verdicts: rows it has not judged drop out; the verdict / reason filters and
        # group-bys below then read THAT judge's entry.
        kept = []
        for r in all_rows:
            mine = next((e for e in _judges(r) if e["judge"] == judge), None)
            if mine is not None:
                r["review_assist"] = {**{k: v for k, v in mine.items() if k != "judge"}, "backend": judge}
                kept.append(r)
        all_rows = kept
    if verdict:
        all_rows = [r for r in all_rows if _resolve_verdict_label(r) == verdict]
    if reason_code:
        all_rows = [r for r in all_rows if (_assist(r) or {}).get("reason_code") == reason_code]
    pending = [r for r in all_rows if r.get("review_status") == "pending"]
    # "Accepted" = explicitly accepted, OR no review_status field
    # (legacy rows from the pre-Epic-2a flow). Both will enter
    # training. Anything else (rejected etc.) is excluded.
    accepted = [
        r for r in all_rows
        if r.get("review_status") in (None, "accepted")
    ]
    # Arc 5 — soft-reject. Rejected rows used to be physically
    # deleted; now they stay on disk with ``review_status=rejected``
    # so the user can review what was rejected, recover an
    # accidentally-rejected row, and bulk-purge with reason
    # filtering. Matches the project preference: rejected rows
    # are selectable + bulk-droppable rather than vanishing
    # immediately.
    rejected = [r for r in all_rows if r.get("review_status") == "rejected"]

    _GROUP_RESOLVERS = {
        "class": _resolve_class_label,
        "cluster": _resolve_cluster_label,
        "source": _resolve_source_label,
        "job": _resolve_job_label,
        "backend": _resolve_backend_label,
        "verdict": _resolve_verdict_label,
        "assist_reason": _resolve_assist_reason_label,
        "assist_consensus": _resolve_consensus_label,
    }
    normalized_group_by = group_by if group_by in _GROUP_RESOLVERS else "source"
    key_fn = _GROUP_RESOLVERS[normalized_group_by]
    from app.services.span_type_definitions_service import load_conventions_version

    # Staleness is per synth_source group (what a relabel acts on), so only
    # computed for the source grouping.
    current = await load_conventions_version(db, project_id) if normalized_group_by == "source" else None
    from app.services.span_type_definitions_service import relabel_pin, row_labeller

    # Every model that labelled span rows here, as relabel pins — the
    # stale-conventions strip offers them as "Relabel with".
    labeller_pins = sorted({
        pin for pin in (relabel_pin(row_labeller(r)) for r in all_rows if isinstance(r.get("entities"), list)) if pin
    })

    return {
        "project_id": project_id,
        "dataset_id": dataset.id if dataset else None,
        "group_by": normalized_group_by,
        "total_rows": total_rows_all,
        "filter": {
            "synth_job_id": synth_job_id, "synth_backend": synth_backend,
            "verdict": verdict, "reason_code": reason_code,
            "assist_consensus": assist_consensus, "judge": judge,
        },
        "facets": facets,
        "total_pending": len(pending),
        "total_accepted": len(accepted),
        "total_rejected": len(rejected),
        "conventions_version": current,
        "labeller_pins": labeller_pins,
        "groups": _bucket_rows_by_source(pending, key_fn=key_fn, conventions_current=current, grouping=normalized_group_by),
        "accepted_groups": _bucket_rows_by_source(
            accepted, rows_per_group_cap=ACCEPTED_ROWS_PER_GROUP_CAP, key_fn=key_fn,
            conventions_current=current, grouping=normalized_group_by,
        ),
        "rejected_groups": _bucket_rows_by_source(
            rejected, rows_per_group_cap=ACCEPTED_ROWS_PER_GROUP_CAP, key_fn=key_fn, grouping=normalized_group_by,
        ),
    }


async def bulk_update_review_queue(
    db: AsyncSession,
    project_id: int,
    *,
    row_ids: list[int],
    action: ReviewAction,
    reject_reason: str | None = None,
    decided_via: str | None = None,
) -> dict[str, Any]:
    """Bulk apply an accept / reject to a set of pending rows.

    Every row accepted or rejected is stamped ``decided_via`` (see ``normalize_decided_via``;
    omitted = ``api``) and ``decided_at`` (UTC ISO) — record keeping, nothing reads them to decide.

    - ``accept`` flips ``review_status`` from ``"pending"`` to
      ``"accepted"`` for matching rows. Accepted rows are picked up
      by the next dataset prep run.
    - ``reject`` flips ``review_status`` to ``"rejected"`` AND
      stamps ``reject_reason`` (when supplied) on the row. Rows
      stay on disk so the user can:
        * review what was rejected (Rejected section in the UI);
        * recover an accidentally-rejected row by re-marking it
          ``pending`` (future feature);
        * bulk-purge a reason cohort via ``purge_rejected_rows``.
      Prior behavior (physical delete) ran against the project
      preference "rejected rows are selectable + bulk-droppable" —
      vanishing rows aren't selectable.

    Returns a summary dict with counts (`accepted` / `rejected` /
    `not_found` / `not_pending`) so the UI can show what landed.
    """
    if action not in ("accept", "reject"):
        raise ValueError("action must be 'accept' or 'reject'")
    if reject_reason is not None:
        reject_reason = str(reject_reason).strip() or None
    via = normalize_decided_via(decided_via)
    decided_at = datetime.now(timezone.utc).isoformat()
    if not row_ids:
        return {
            "accepted": 0,
            "rejected": 0,
            "not_found": 0,
            "not_pending": 0,
            "total_remaining_pending": 0,
        }

    target_ids = set(int(rid) for rid in row_ids)
    path = _synthetic_jsonl_path(project_id)
    rows = _ensure_row_ids(path, _read_all_rows(path))

    accepted_count = 0
    rejected_count = 0
    not_pending = 0
    new_rows: list[dict[str, Any]] = []
    matched_ids: set[int] = set()

    for row in rows:
        rid = row.get("id")
        if isinstance(rid, int) and rid in target_ids:
            matched_ids.add(rid)
            status = row.get("review_status")
            if status != "pending":
                # Already accepted (or rejected, in which case it
                # shouldn't be on disk; defensive). Keep as-is.
                not_pending += 1
                new_rows.append(row)
                continue
            row["decided_via"] = via
            row["decided_at"] = decided_at
            if action == "accept":
                row["review_status"] = "accepted"
                accepted_count += 1
                new_rows.append(row)
            else:  # reject — soft (mark, don't delete)
                row["review_status"] = "rejected"
                if reject_reason:
                    row["reject_reason"] = reject_reason
                rejected_count += 1
                new_rows.append(row)
        else:
            new_rows.append(row)

    not_found = len(target_ids - matched_ids)

    if accepted_count + rejected_count > 0:
        _atomic_write_rows(path, new_rows)
        # Update the Dataset row's record_count to match the file.
        ds_result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == DatasetType.SYNTHETIC,
            )
        )
        dataset = ds_result.scalar_one_or_none()
        if dataset is not None:
            dataset.record_count = len(new_rows)
            await db.flush()

    remaining_pending = sum(1 for r in new_rows if r.get("review_status") == "pending")

    return {
        "accepted": accepted_count,
        "rejected": rejected_count,
        "not_found": not_found,
        "not_pending": not_pending,
        "total_remaining_pending": remaining_pending,
    }


async def bulk_update_by_source(
    db: AsyncSession,
    project_id: int,
    *,
    source: str,
    action: ReviewAction,
    reject_reason: str | None = None,
) -> dict[str, Any]:
    """Accept / reject *every pending row* in one ``synth_source`` group by
    key — no row-id enumeration needed.

    This is what makes the Data Studio review-queue panel actionable: its
    cross-workflow summary surfaces pending groups by source + count, so a
    one-click "Accept all (N)" / "Reject all (N)" on a group is the natural
    bulk. We resolve the matching pending row ids (via the same
    ``_resolve_source_label`` the grouping uses, so a group's key always maps
    back to its rows) and delegate to the tested ``bulk_update_review_queue``
    — no second mutation path to keep in sync.

    Returns the same summary as ``bulk_update_review_queue`` plus the echoed
    ``source`` and ``matched`` (pending rows in the group at call time)."""
    if action not in ("accept", "reject"):
        raise ValueError("action must be 'accept' or 'reject'")
    source = str(source or "").strip()
    if not source:
        raise ValueError("source must be a non-empty synth_source key")

    path = _synthetic_jsonl_path(project_id)
    rows = _ensure_row_ids(path, _read_all_rows(path))
    matched_ids = [
        int(row["id"])
        for row in rows
        if row.get("review_status") == "pending"
        and isinstance(row.get("id"), int)
        and _resolve_source_label(row) == source
    ]

    summary = await bulk_update_review_queue(
        db,
        project_id,
        row_ids=matched_ids,
        action=action,
        reject_reason=reject_reason,
        decided_via=f"source:{source}",
    )
    if not matched_ids:
        # bulk_update_review_queue early-returns a zero summary (with a
        # hardcoded total_remaining_pending=0) when handed no ids — it never
        # reads the file. The group simply had no pending rows; report the
        # project's *actual* remaining pending so the panel doesn't falsely
        # show "0 left to review".
        summary["total_remaining_pending"] = sum(
            1 for row in rows if row.get("review_status") == "pending"
        )
    summary["source"] = source
    summary["matched"] = len(matched_ids)
    return summary


RETIRED_REASON = "retired"


async def set_group_retired(db: AsyncSession, project_id: int, *, source: str, retired: bool) -> dict[str, Any]:
    """Retire an accepted ``synth_source`` group (its accepted rows become
    rejected with ``reject_reason="retired"`` — out of training, still on
    disk) or restore a retired one (only rows rejected AS retired go back to
    accepted; rows rejected for another reason stay rejected)."""
    from datetime import datetime, timezone

    source = str(source or "").strip()
    if not source:
        raise ValueError("source must be a non-empty synth_source key")
    path = _synthetic_jsonl_path(project_id)
    rows = _ensure_row_ids(path, _read_all_rows(path))
    group = [r for r in rows if _resolve_source_label(r) == source]
    if not group:
        raise ValueError(f"No rows in source group {source!r}.")
    stamp = datetime.now(timezone.utc).isoformat()
    changed = 0
    for row in group:
        status = row.get("review_status")
        if retired and status in (None, "accepted"):
            row["review_status"] = "rejected"
            row["reject_reason"] = RETIRED_REASON
            row["retired_at"] = stamp
            row["decided_via"] = f"retire:{source}"[:DECIDED_VIA_MAX]
            row["decided_at"] = stamp
            changed += 1
        elif not retired and status == "rejected" and row.get("reject_reason") == RETIRED_REASON:
            row["review_status"] = "accepted"
            row.pop("reject_reason", None)
            row.pop("retired_at", None)
            row["decided_via"] = f"unretire:{source}"[:DECIDED_VIA_MAX]
            row["decided_at"] = stamp
            changed += 1
    if changed:
        _atomic_write_rows(path, rows)
    return {"source": source, "retired": retired, "rows_changed": changed, "group_rows": len(group)}


async def mark_conventions_current(db: AsyncSession, project_id: int, *, source: str) -> dict[str, Any]:
    """Stamp every span row of one ``synth_source`` group with the project's
    current conventions version — the user attests the group already
    follows them (e.g. it was labelled under the current wording before
    versions were recorded). Rows get ``conventions_marked_current_at`` so
    the attestation stays distinguishable from a real relabel."""
    from datetime import datetime, timezone

    from app.services.span_type_definitions_service import CONVENTIONS_VERSION_KEY, load_conventions_version

    if not source or not source.strip():
        raise ValueError("source must be a non-empty synth_source key")
    current = await load_conventions_version(db, project_id)
    if current is None:
        raise ValueError("This project has no labelling conventions to mark rows current with.")
    path = _synthetic_jsonl_path(project_id)
    rows = _read_all_rows(path)
    group = [r for r in rows if _resolve_source_label(r) == source and isinstance(r.get("entities"), list)]
    if not group:
        raise ValueError(f"No span rows in source group {source!r}.")
    stamp = datetime.now(timezone.utc).isoformat()
    marked = 0
    for row in group:
        if row.get(CONVENTIONS_VERSION_KEY) != current:
            row[CONVENTIONS_VERSION_KEY] = current
            row["conventions_marked_current_at"] = stamp
            marked += 1
    if marked:
        _atomic_write_rows(path, rows)
    return {"source": source, "span_rows": len(group), "rows_marked": marked, "conventions_version": current}


async def purge_rejected_rows(
    db: AsyncSession,
    project_id: int,
    *,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Physically remove rejected rows from synthetic.jsonl. The
    soft-reject in ``bulk_update_review_queue`` keeps rejected
    rows around so the user can review them; this is the explicit
    "I've reviewed the rejected pile, now delete it" step that
    drops them for good.

    ``reasons`` filters by ``reject_reason`` — when set, only
    rejected rows whose reason is in the list (or who have a
    matching empty-reason entry when ``""`` is passed) get
    purged. When omitted, every rejected row is removed.

    Returns a summary dict ``{purged: <count>, retained: <count>,
    total_rows: <count>}`` so the UI can confirm what landed.
    """
    path = _synthetic_jsonl_path(project_id)
    rows = _read_all_rows(path)
    if not rows:
        return {"purged": 0, "retained": 0, "total_rows": 0}

    reason_filter: set[str] | None = None
    if reasons is not None:
        reason_filter = {
            str(r).strip() for r in reasons if isinstance(r, str)
        }
        # Empty-set filter would purge nothing — treat as "no
        # filter" for caller convenience.
        if not reason_filter:
            reason_filter = None

    purged_count = 0
    retained_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("review_status") != "rejected":
            retained_rows.append(row)
            continue
        if reason_filter is not None:
            row_reason = str(row.get("reject_reason") or "").strip()
            if row_reason not in reason_filter:
                retained_rows.append(row)
                continue
        purged_count += 1
        # Don't append — row is physically dropped here.

    if purged_count > 0:
        _atomic_write_rows(path, retained_rows)
        # Keep the Dataset's record_count in sync with the file
        # the same way bulk_update_review_queue does — the
        # next dataset-prep read counts rows + would silently
        # mismatch otherwise.
        ds_result = await db.execute(
            select(Dataset).where(
                Dataset.project_id == project_id,
                Dataset.dataset_type == DatasetType.SYNTHETIC,
            )
        )
        dataset = ds_result.scalar_one_or_none()
        if dataset is not None:
            dataset.record_count = len(retained_rows)
            await db.flush()

    return {
        "purged": purged_count,
        "retained": len(retained_rows),
        "total_rows": len(retained_rows),
    }


LANGUAGE_MISMATCH = "language_mismatch"


async def recheck_code_format(
    db: AsyncSession, project_id: int, *, dry_run: bool = True, row_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Re-check the declared format of PENDING ``code_example`` rows: a row labelled json / yaml / toml / xml
    whose code does not parse as that format (``check_data_format``, the check generation applies) is a
    mislabel. ``dry_run`` only counts: ``mismatches`` plus ``groups`` (language + kind of failure, each with
    its count, row ids and a few examples). Otherwise the rows are marked rejected with reason
    ``language_mismatch`` (soft: they stay on disk, can be restored, never purged here) - all mismatches, or
    only ``row_ids`` (each id is re-validated: it must still be a pending row that fails the check).
    Accepted and already-rejected rows are not touched."""
    from app.services.synth_playbooks.code_examples import check_data_format

    path = _synthetic_jsonl_path(project_id)
    rows = _read_all_rows(path)
    wanted = {int(r) for r in row_ids} if row_ids is not None else None

    def problem(row: dict[str, Any]) -> str | None:
        if row.get("review_status") != "pending" or row.get("row_kind") != "code_example":
            return None
        payload = row.get("code_example_json")
        code = payload.get("code") if isinstance(payload, dict) else None
        return check_data_format(code, row.get("language")) if isinstance(code, str) else None

    checked = 0
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("review_status") == "pending" and row.get("row_kind") == "code_example" and isinstance(row.get("code_example_json"), dict):
            checked += 1
        why = problem(row)
        if why is None:
            continue
        lang = str(row.get("language") or "")
        key = (lang, why.split(":")[0])
        g = groups.setdefault(key, {"language": lang, "detail": key[1], "count": 0, "row_ids": [], "examples": []})
        g["count"] += 1
        if isinstance(row.get("id"), int):
            g["row_ids"].append(row["id"])
        if len(g["examples"]) < 3:
            g["examples"].append({"id": row.get("id"), "detail": why, "task": str(row.get("task") or "")[:120]})
    by_language: dict[str, int] = defaultdict(int)
    for g in groups.values():
        by_language[g["language"]] += g["count"]
    result: dict[str, Any] = {
        "dry_run": dry_run,
        "checked": checked,
        "mismatches": sum(by_language.values()),
        "by_language": dict(by_language),
        "groups": sorted(groups.values(), key=lambda g: -g["count"]),
        "examples": [e for g in groups.values() for e in g["examples"]][:5],
        "rejected": 0,
    }
    if dry_run or not groups:
        return result
    rows = _ensure_row_ids(path, _read_all_rows(path))
    rejected = 0
    for row in rows:
        if wanted is not None and row.get("id") not in wanted:
            continue
        if problem(row) is not None:
            row["review_status"] = "rejected"
            row["reject_reason"] = LANGUAGE_MISMATCH
            row["decided_via"] = "rule:language_mismatch"
            row["decided_at"] = datetime.now(timezone.utc).isoformat()
            rejected += 1
    if rejected:
        _atomic_write_rows(path, rows)
    result["rejected"] = rejected
    return result




RESTORE_MAX_ROW_IDS = 5000


def compute_plan_rules_check(
    project_id: int,
    rules_doc: Any,
    *,
    row_ids: list[int] | None = None,
    source: str | None = None,
    synth_job_id: int | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """The read-only half of the plan-rules check (CPU-bound; call it via ``asyncio.to_thread``).
    Returns the report plus ``_flagged``: ``{row_id: (reject rule ids, checked plan)}``.
    Raises ``plan_rules_service.PlanRulesError`` for bad rules."""
    from app.services import plan_rules_service as engine

    rules = engine.load_rules(rules_doc)
    path = _synthetic_jsonl_path(project_id)
    rows = _read_all_rows(path)
    selected = engine.select_pending_plan_rows(
        rows, row_ids=row_ids, source=source, synth_job_id=synth_job_id, max_rows=max_rows,
    )
    report = engine.check_rows(selected, rules)
    violations = report.pop("row_violations")
    by_id = {r.get("id"): r for r in selected}
    flagged: dict[int, tuple[list[str], Any]] = {}
    for row_id, found in violations.items():
        ids = engine.rule_ids([
            engine.Violation(v["rule_id"], v["severity"], tuple(v["steps"]), v["message"])
            for v in found if v["severity"] == "reject"
        ])
        if ids:
            flagged[row_id] = (ids, by_id[row_id].get("plan_json"))
    return {"dry_run": True, "rules": len(rules.rules), **report, "rejected": 0, "_flagged": flagged}


def apply_plan_rules_check(project_id: int, computed: dict[str, Any]) -> dict[str, Any]:
    """Soft-reject the rows ``compute_plan_rules_check`` flagged. Synchronous and await-free (so it is
    atomic against the other review-queue writers on the event loop); re-reads the file and only touches
    rows that are still pending with the very plan that was checked."""
    from app.services import plan_rules_service as engine

    flagged = computed.pop("_flagged", {})
    path = _synthetic_jsonl_path(project_id)
    rows = _ensure_row_ids(path, _read_all_rows(path))
    rejected = 0
    for row in rows:
        entry = flagged.get(row.get("id"))
        if entry is None or row.get("review_status") != "pending" or row.get("plan_json") != entry[1]:
            continue
        row["review_status"] = "rejected"
        row["reject_reason"] = engine.reject_reason_for(entry[0][0])
        row["plan_rule_ids"] = entry[0]
        row["decided_via"] = "rule:plan_rules"
        row["decided_at"] = datetime.now(timezone.utc).isoformat()
        rejected += 1
    if rejected:
        _atomic_write_rows(path, rows)
    return {**computed, "dry_run": False, "rejected": rejected}


async def check_plan_rules_for_project(
    project_id: int,
    rules_doc: Any,
    *,
    dry_run: bool = True,
    **filters: Any,
) -> dict[str, Any]:
    """Run a plan-rules document over the project's PENDING structured-plan rows.

    ``dry_run`` only counts (per rule: severity, count, sample row ids). Otherwise each row with a
    ``reject``-severity violation is soft-rejected as ``reject_reason="rule_violation:<first rule id>"``
    (+ ``plan_rule_ids``); the rejected view regroups by reason, and ``restore_rejected_rows`` puts them back.
    ``warn`` rules are counted, never acted on. The engine runs in a worker thread (regexes can be slow).
    """
    import asyncio

    computed = await asyncio.to_thread(compute_plan_rules_check, project_id, rules_doc, **filters)
    if dry_run:
        computed.pop("_flagged", None)
        return computed
    return apply_plan_rules_check(project_id, computed)


def restore_rejected_rows(
    project_id: int,
    *,
    reject_reason: str | None = None,
    reject_reason_prefix: str | None = None,
    row_ids: list[int] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Put rejected rows back to ``pending`` (the generic undo of any soft-reject).

    Selectors (at least one; all given must match): exact ``reject_reason`` (``""`` = rows with no
    reason), ``reject_reason_prefix`` (e.g. ``rule_violation:``), ``row_ids``. A restored row loses
    ``reject_reason``, the reason-specific fields (``plan_rule_ids``, ``retired_at``) and the decision
    stamps (``decided_via`` / ``decided_at``); everything
    else is kept. ``dry_run`` (default) only counts, by reason. Synchronous (atomic on the event loop).
    """
    if reject_reason is None and not reject_reason_prefix and not row_ids:
        raise ValueError("Give at least one of reject_reason, reject_reason_prefix, row_ids.")
    if row_ids is not None and len(row_ids) > RESTORE_MAX_ROW_IDS:
        raise ValueError(f"row_ids is capped at {RESTORE_MAX_ROW_IDS}.")
    wanted = set(row_ids) if row_ids else None
    path = _synthetic_jsonl_path(project_id)
    rows = _ensure_row_ids(path, _read_all_rows(path))
    by_reason: dict[str, int] = {}
    restored = 0
    for row in rows:
        if row.get("review_status") != "rejected":
            continue
        reason = str(row.get("reject_reason") or "").strip()
        if reject_reason is not None and reason != reject_reason.strip():
            continue
        if reject_reason_prefix and not reason.startswith(reject_reason_prefix):
            continue
        if wanted is not None and row.get("id") not in wanted:
            continue
        by_reason[reason or "(no reason)"] = by_reason.get(reason or "(no reason)", 0) + 1
        restored += 1
        if not dry_run:
            row["review_status"] = "pending"
            # Back to pending: the reject (and how it was decided) is undone.
            for key in ("reject_reason", "plan_rule_ids", "retired_at", *DECISION_KEYS):
                row.pop(key, None)
    if restored and not dry_run:
        _atomic_write_rows(path, rows)
    return {"dry_run": dry_run, "matched": restored, "restored": 0 if dry_run else restored, "by_reason": by_reason}
