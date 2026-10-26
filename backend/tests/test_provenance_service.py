"""provenance_service helpers that need no database: licence normalisation +
flags, usage rules, the hub / API licence lookups (stubbed, no network), the
per-row usage partition and the export text."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.services import provenance_service as prov


class LicenceTests(unittest.TestCase):
    def test_normalize_licence(self):
        for empty in (None, "", "   ", "unknown", "Unknown", "NOASSERTION", "Imported from huggingface", "imported from url"):
            self.assertIsNone(prov.normalize_licence(empty), empty)
        self.assertEqual(prov.normalize_licence("  MIT "), "MIT")
        self.assertEqual(prov.normalize_licence("cc-by-4.0\n"), "cc-by-4.0")
        self.assertEqual(prov.normalize_licence("other"), "other", "free text is kept")

    def test_flags(self):
        self.assertEqual(prov.licence_flags(None), ["unknown"])
        self.assertEqual(prov.licence_flags("NOASSERTION"), ["unknown"])
        self.assertEqual(prov.licence_flags("MIT"), [])
        self.assertEqual(prov.licence_flags("Apache-2.0"), [])
        for nc in ("CC-BY-NC-4.0", "cc-by-nc-sa-4.0", "CC BY-NC 4.0", "nc", "Non-Commercial use", "noncommercial",
                   "research use only", "Research only"):
            self.assertEqual(prov.licence_flags(nc), ["non_commercial"], nc)
        for nc in ("For research purposes only", "research purposes only", "Not for commercial use", "non commercial"):
            self.assertEqual(prov.licence_flags(nc), ["non_commercial"], nc)
        for fine in ("Unlicense", "BSD-3-Clause", "MIT", "Apache-2.0", "CC0-1.0"):
            self.assertEqual(prov.licence_flags(fine), [], fine)
        for unknown in ("none", "None", "n/a", "N/A", "unknown", "noassertion", ""):
            self.assertEqual(prov.licence_flags(unknown), ["unknown"], unknown)
        self.assertEqual(prov.normalize_licence("Unlicense"), "Unlicense", "a real licence, not 'none'")
        self.assertEqual(prov.licence_flags("ISC"), [], "'nc' inside another token is not NC")
        self.assertEqual(prov.licence_flags("CC-BY-SA-4.0"), [])

    def test_usage_rules(self):
        self.assertEqual(prov.effective_usage(None), "any")
        self.assertEqual(prov.effective_usage(SimpleNamespace(source_kind="upload", usage="eval")), "eval")
        self.assertEqual(prov.effective_usage(SimpleNamespace(source_kind="predefined_split", usage="rag")), "any")
        self.assertEqual(prov.effective_usage(SimpleNamespace(source_kind="upload", usage="bogus")), "any")
        self.assertEqual(prov.validate_usage(" RAG "), "rag")
        with self.assertRaises(prov.ProvenanceError):
            prov.validate_usage("nope")


class RemoteLicenceTests(unittest.TestCase):
    def test_hf_licence_from_dataset_infos_then_card(self):
        infos = {"default": SimpleNamespace(license="cc-by-4.0"), "other": SimpleNamespace(license=["mit", "apache-2.0"])}
        with mock.patch("datasets.get_dataset_infos", return_value=infos):
            self.assertEqual(prov.hf_dataset_licence("toy/dataset"), "cc-by-4.0")
            self.assertEqual(prov.hf_dataset_licence("toy/dataset", config="other"), "mit, apache-2.0")
        card = SimpleNamespace(card_data=SimpleNamespace(to_dict=lambda: {"license": "cc-by-nc-4.0"}))
        api = mock.MagicMock()
        api.dataset_info.return_value = card
        with mock.patch("datasets.get_dataset_infos", side_effect=RuntimeError("offline")), \
                mock.patch("huggingface_hub.HfApi", return_value=api):
            self.assertEqual(prov.hf_dataset_licence("toy/dataset", token="t"), "cc-by-nc-4.0")
        with mock.patch("datasets.get_dataset_infos", side_effect=RuntimeError("offline")), \
                mock.patch("huggingface_hub.HfApi", side_effect=RuntimeError("offline")):
            self.assertIsNone(prov.hf_dataset_licence("toy/dataset"))

    def test_kaggle_licence_shapes(self):
        # dataset_list(search=...) hits carry ref + license_name; the match is by ref.
        hits = [SimpleNamespace(ref="x/toy-extra", license_name="GPL-3.0"), SimpleNamespace(ref="owner/toy", license_name="CC0-1.0")]
        seen = {}

        def lister(search=None, **kw):
            seen["search"] = search
            return hits

        self.assertEqual(prov.kaggle_dataset_licence(SimpleNamespace(dataset_list=lister), "owner/toy"), "CC0-1.0")
        self.assertEqual(seen["search"], "toy")

        # dataset_metadata(dataset, path) writes dataset-metadata.json into path.
        def writer(slug, path):
            target = Path(path) / "dataset-metadata.json"
            target.write_text(json.dumps({"licenses": [{"name": "ODbL"}]}))
            return str(target)

        api = SimpleNamespace(dataset_list=lambda search=None, **kw: [], dataset_metadata=writer)
        self.assertEqual(prov.kaggle_dataset_licence(api, "owner/toy"), "ODbL")

        def boom(*a, **kw):
            raise RuntimeError("401")

        self.assertIsNone(prov.kaggle_dataset_licence(SimpleNamespace(dataset_list=boom, dataset_metadata=boom), "owner/toy"))
        self.assertIsNone(prov.kaggle_dataset_licence(object(), "owner/toy"))
        self.assertIsNone(prov.kaggle_dataset_licence(mock.MagicMock(), "owner/toy"), "a mock's attributes are not a licence")


class PartitionTests(unittest.TestCase):
    def _ctx(self):
        return prov.UsageContext(
            doc_source={1: 10, 2: 20},
            source_usage={10: "any", 20: "eval", 30: "rag"},
            sources={10: {"title": "A", "kind": "upload", "licence": "MIT", "usage": "any"},
                     20: {"title": "B", "kind": "upload", "licence": None, "usage": "eval"},
                     30: {"title": "C", "kind": "dataset_import", "licence": "CC-BY-NC-4.0", "usage": "rag"}},
        )

    def test_rows_are_routed_by_document_then_by_source_id(self):
        ctx = self._ctx()
        rows = [
            {"text": "a", "_source_document_id": 1},
            {"text": "b", "_source_document_id": 2},
            {"text": "c", "source_document_id": 2},
            {"text": "d", "data_source_id": 30},
            {"text": "e"},
            {"text": "f", "data_source_id": 999},
        ]
        state: dict = {}
        kept = ctx.partition(rows, state)
        self.assertEqual([r["text"] for r in kept], ["a", "e", "f"])
        self.assertEqual([r["text"] for r in state["eval_rows"]], ["b", "c"])
        self.assertEqual(state["excluded_rag"], 1)
        self.assertEqual(state["excluded_by_source"], {30: 1})

    def test_nothing_restricted_returns_the_same_list(self):
        ctx = prov.UsageContext(doc_source={1: 10}, source_usage={10: "any"}, sources={10: {}})
        rows = [{"text": "a", "_source_document_id": 1}]
        self.assertIs(ctx.partition(rows, {}), rows)

    def test_prepared_provenance_counts_and_warnings(self):
        ctx = self._ctx()
        splits = {
            "train": [{"text": "a", "_source_document_id": 1}, {"text": "x"}],
            "val": [{"text": "a2", "_source_document_id": 1}],
            "test": [{"text": "b", "_source_document_id": 2}],
        }
        out = prov.prepared_provenance(
            ctx, splits, {"excluded_rag": 2, "excluded_by_source": {30: 2}},
            eval_routed_to_test=1, eval_not_added_pinned_test=0,
        )
        self.assertEqual(out["by_source"]["10"]["train"], 1)
        self.assertEqual(out["by_source"]["10"]["val"], 1)
        self.assertEqual(out["by_source"]["20"]["test"], 1)
        self.assertEqual(out["by_source"]["30"]["excluded"], 2)
        self.assertEqual(out["unattributed"], {"train": 1, "val": 0, "test": 0})
        self.assertEqual(out["excluded_rag"], 2)
        self.assertEqual(out["eval_routed_to_test"], 1)
        # Only a source that fed train / val is warned about: A has a licence, B is test-only, C is excluded.
        self.assertEqual(out["licence_warnings"], [])

    def test_unknown_and_nc_sources_that_train_are_warned(self):
        ctx = prov.UsageContext(
            doc_source={1: 10, 2: 20}, source_usage={10: "any", 20: "any"},
            sources={10: {"title": "A", "kind": "upload", "licence": None, "usage": "any"},
                     20: {"title": "B", "kind": "hf", "licence": "cc-by-nc-4.0", "usage": "any"}},
        )
        splits = {"train": [{"_source_document_id": 1}, {"_source_document_id": 2}], "val": [], "test": []}
        out = prov.prepared_provenance(ctx, splits, {}, eval_routed_to_test=0, eval_not_added_pinned_test=0)
        self.assertEqual({(w["source_id"], w["flag"]) for w in out["licence_warnings"]},
                         {(10, "unknown"), (20, "non_commercial")})

    def test_drop_excluded_passages(self):
        passages = [{"source_document_id": 1}, {"source_document_id": "2"}, {"source_document_id": None}]
        self.assertEqual(prov.drop_excluded_passages(passages, {2}), [passages[0], passages[2]])
        self.assertEqual(prov.drop_excluded_passages(passages, set()), passages)

    def test_release_notes_section(self):
        self.assertEqual(prov.release_notes_section({"sources": []}), "")
        text = prov.release_notes_section({
            "sources": [{"title": "Docs", "kind": "git_repo", "locator": "https://github.com/org/repo", "ref": "a" * 40,
                         "licence": None, "licence_verified": False, "usage": "any", "train": 5, "val": 1, "test": 0}],
            "warnings": [{"message": "Docs: licence unknown."}],
        })
        self.assertIn("## Data sources & licences", text)
        self.assertIn("https://github.com/org/repo @ aaaaaaaaaaaa", text)
        self.assertIn("licence unknown (not verified)", text)
        self.assertIn("- Docs: licence unknown.", text)


if __name__ == "__main__":
    unittest.main()


class InferenceScriptTests(unittest.TestCase):
    def test_generated_inference_scripts_do_not_import_logging_they_never_use(self):
        from app.models.export import ExportFormat
        from app.services.export_service import _generate_inference_script

        for fmt in ExportFormat:
            script = _generate_inference_script(fmt)
            self.assertNotIn("import logging", script, fmt)
            compile(script, f"<{fmt}>", "exec")
