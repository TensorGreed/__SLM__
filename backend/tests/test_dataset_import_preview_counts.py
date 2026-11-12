"""Dataset-import preview reports the true accepted count, separately from
the kept sample (``accepted_count`` used to be capped at ``sample_cap``)."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from app.cli.dataset_import import main as cli_main
from app.services.dataset_import.service import preview_import, result_to_dict


def _write(tmp_path: Path, n: int, bad: int = 0) -> Path:
    path = tmp_path / "rows.jsonl"
    rows = [{"text": f"row number {i} has plenty of words in it", "label": "a" if i % 2 else "b"} for i in range(n)]
    rows += [{"text": "", "label": "a"} for _ in range(bad)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _preview(path: Path, **kw):
    return preview_import(
        project_id=0, project_task_profile=None, locator=f"jsonl:{path}",
        mapper_id="label_to_classification", field_map={}, **kw,
    )


def test_accepted_count_is_not_capped_at_the_sample(tmp_path):
    result = _preview(_write(tmp_path, 12, bad=2), sample_cap=5)
    assert result.accepted_count == 12
    assert len(result.accepted_rows) == 5
    assert result.rejected_count == 2
    assert result.rows_scanned == 14
    assert result.scan_truncated is False
    payload = result_to_dict(result)
    assert payload["accepted_count"] == 12
    assert payload["accepted_sample_count"] == 5
    assert payload["rows_scanned"] == 14
    assert payload["scan_limit"] == 100
    assert payload["scan_truncated"] is False


def test_scan_limit_reached_is_reported(tmp_path):
    result = _preview(_write(tmp_path, 30), sample_cap=5, limit=10)
    assert result.accepted_count == 10
    assert result.rows_scanned == 10
    assert result.scan_limit == 10
    assert result.scan_truncated is True


def test_cli_preview_prints_the_true_count(tmp_path):
    path = _write(tmp_path, 12)
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli_main([
            "preview", "--locator", f"jsonl:{path}", "--mapper", "label_to_classification", "--sample-cap", "3",
        ])
    assert code == 0
    text = out.getvalue()
    assert "accepted=12" in text
    assert "scanned=12" in text
    assert "first 3 accepted rows" in text
