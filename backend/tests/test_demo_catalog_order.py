"""The demo catalog lists the README's starter (Support FAQ) first.

A fresh-clone test clicked the first "Open demo" tile, which was PII / PCI
(directory order), instead of the Support FAQ starter the README names.
"""

import json
from pathlib import Path
from unittest import mock

from app.services import demo_project_service
from app.services.demo_project_service import list_demo_archetypes


def test_support_faq_is_first():
    slugs = [a["slug"] for a in list_demo_archetypes()]
    assert slugs[0] == "support-faq"
    assert set(slugs) >= {"support-faq", "sentiment-classifier", "pii-detector"}


def test_order_key_not_exposed():
    for archetype in list_demo_archetypes():
        assert "_order" not in archetype
        assert "display_order" not in archetype


def _write(root: Path, slug: str, order=None):
    d = root / slug
    d.mkdir()
    manifest = {"slug": slug, "name": slug}
    if order is not None:
        manifest["display_order"] = order
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_ordered_first_then_unordered_by_name(tmp_path):
    _write(tmp_path, "aaa")
    _write(tmp_path, "zzz", order=20)
    _write(tmp_path, "mmm", order=10)
    _write(tmp_path, "bbb")
    with mock.patch.object(demo_project_service, "DEMO_SAMPLES_DIR", tmp_path):
        slugs = [a["slug"] for a in list_demo_archetypes()]
    assert slugs == ["mmm", "zzz", "aaa", "bbb"]
