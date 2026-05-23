"""Helpers shared by the 6 CLUSTER_TARGETED playbooks.

Each CLUSTER_TARGETED playbook reads ``ctx['failure_cluster']`` —
the dict emitted by `failure_cluster_service.cluster_eval_result_failures`
— and turns its exemplars + pattern signature into a prompt block
the recipe-specific playbook can splice into its own template.
"""

from __future__ import annotations

from typing import Any


def render_cluster_block(failure_cluster: dict[str, Any] | None, *, max_exemplars: int = 5) -> str:
    """Returns a human-readable description of a failure cluster,
    suitable for splicing into a playbook prompt.

    Format:
        Cluster reason: <reason_code>
        Output pattern: <output_pattern_signature>
        Affected rows: N (X% of all failures)
        Example failures:
          - Input: ...
            Expected: ...
            Model output: ...
          - ...
    """
    if not failure_cluster:
        return "(no cluster context — falling back to generic positive paraphrase)"

    reason = failure_cluster.get("reason_code") or failure_cluster.get("cluster_id") or "unknown"
    pattern = failure_cluster.get("output_pattern") or ""
    count = failure_cluster.get("failure_count")
    share = failure_cluster.get("share_of_total")
    classifier_reason = failure_cluster.get("classifier_reason") or ""

    lines: list[str] = []
    lines.append(f"Cluster reason: {reason}")
    if pattern:
        lines.append(f"Output pattern: {pattern}")
    if classifier_reason:
        lines.append(f"What's going wrong: {classifier_reason}")
    if isinstance(count, int) and count > 0:
        if isinstance(share, (int, float)) and share > 0:
            lines.append(f"Affected rows: {count} ({share:.0%} of failures)")
        else:
            lines.append(f"Affected rows: {count}")

    exemplars = failure_cluster.get("exemplars") or []
    if exemplars:
        lines.append("")
        lines.append("Example failures:")
        for ex in exemplars[:max_exemplars]:
            if not isinstance(ex, dict):
                continue
            inp = _coerce_str(ex.get("prompt") or ex.get("input") or ex.get("question") or "")
            expected = _coerce_str(ex.get("reference") or ex.get("expected") or ex.get("answer") or "")
            actual = _coerce_str(ex.get("prediction") or ex.get("model_output") or ex.get("output") or "")
            block: list[str] = []
            if inp:
                block.append(f"  - Input:    {inp[:240]!r}")
            if expected:
                block.append(f"    Expected: {expected[:240]!r}")
            if actual:
                block.append(f"    Got:      {actual[:240]!r}")
            if block:
                lines.append("\n".join(block))

    return "\n".join(lines)


def cluster_provenance_suffix(failure_cluster: dict[str, Any] | None) -> str:
    """Stable string id for the `synth_source` provenance field so the
    review queue can group rows by which cluster they targeted."""
    if not failure_cluster:
        return "noctx"
    cid = failure_cluster.get("cluster_id")
    if isinstance(cid, str) and cid:
        return cid
    reason = failure_cluster.get("reason_code")
    if isinstance(reason, str) and reason:
        return reason
    return "noctx"


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return ""
