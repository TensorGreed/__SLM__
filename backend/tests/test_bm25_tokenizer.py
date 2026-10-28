"""BM25 tokenizer (identifier-aware for code) and the index tokenizer-version stamp.

Prose must tokenize exactly as it did before identifier splitting — PDF
artefacts and placeholders included; in code passages camelCase / snake_case
identifiers emit their parts AND the whole identifier; an index built by
another tokenizer version is stale and gets rebuilt.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")

from app.config import settings  # noqa: E402
from app.services import auto_rag_service as svc  # noqa: E402

_OLD_TOKEN_RE = __import__("re").compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _old_tokenize(text: str) -> list[str]:
    return _OLD_TOKEN_RE.findall(text.lower()) if text else []


class TokenizerGoldenTests(unittest.TestCase):
    def test_prose_tokenizes_exactly_as_before(self):
        samples = [
            "",
            "The quick brown fox jumps over the lazy dog.",
            "Don't panic: it's the user's call (2024-05-01)!",
            "Capitalised Words And ALLCAPS NASA FBI stay whole",
            "Section 12(b) applies; see R.S., 1985, c. 5.",
            "Mixed1 and a1b2 tokens 3rd 42",
            "Caf\u00e9 na\u00efve text with accents",
            # PDF artefacts, placeholders and identifiers in prose stay as they were
            "Married filing SepaRately or HeadOf household; call REDACTED_PHONE.",
            "The getUserName() call and user_id field in a manual",
        ]
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(svc._tokenize(text), _old_tokenize(text))

    def test_identifiers_emit_parts_and_the_whole(self):
        self.assertEqual(svc._tokenize("getUserName", identifiers=True), ["get", "user", "name", "getusername"])
        self.assertEqual(svc._tokenize("HTTPServer", identifiers=True), ["http", "server", "httpserver"])
        self.assertEqual(svc._tokenize("user_id", identifiers=True), ["user", "id", "user_id"])
        self.assertEqual(svc._tokenize("MAX_RETRY_COUNT", identifiers=True), ["max", "retry", "count", "max_retry_count"])
        self.assertEqual(svc._tokenize("parseXMLDoc2", identifiers=True), ["parse", "xml", "doc", "2", "parsexmldoc2"])
        self.assertEqual(svc._tokenize("_private", identifiers=True), ["private", "_private"])
        self.assertEqual(svc._tokenize("call getUserName() then user_id", identifiers=True), [
            "call", "get", "user", "name", "getusername", "then", "user", "id", "user_id",
        ])

    def test_all_caps_plurals_and_digit_capital_steps_stay_whole(self):
        self.assertEqual(svc._tokenize("IDs APIs URLs", identifiers=True), ["ids", "apis", "urls"])
        self.assertEqual(svc._tokenize("5G", identifiers=True), ["5g"])
        self.assertEqual(svc._tokenize("running 5G and v2API", identifiers=True), ["running", "5g", "and", "v2api"])
        # the camel split still applies around them
        self.assertEqual(svc._tokenize("getIDs", identifiers=True), ["get", "ids", "getids"])
        self.assertEqual(svc._tokenize("user_IDs", identifiers=True), ["user", "ids", "user_ids"])
        self.assertEqual(svc._tokenize("net_5G", identifiers=True), ["net", "5g", "net_5g"])

    def test_a_query_in_words_matches_an_identifier(self):
        doc = set(svc._tokenize("def getUserName(user_id):", identifiers=True))
        self.assertTrue({"user", "name"} <= doc)
        self.assertTrue(set(svc._tokenize("get user name", identifiers=True)) <= doc)


class IndexVersionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = settings.DATA_DIR
        settings.DATA_DIR = Path(self._tmp.name)
        self.project_id = 7
        cleaned = settings.DATA_DIR / "projects" / "7" / "cleaned"
        cleaned.mkdir(parents=True)
        rows = [
            {"source_document_id": 1, "source_doc": "a.py", "chunk_id": 0, "text": "def alpha(): return getUserName()",
             "path": "src/a.py", "language": "python", "start_line": 1, "end_line": 1},
            {"source_document_id": 1, "source_doc": "a.txt", "chunk_id": 1, "text": "beta passage about something else"},
        ]
        (cleaned / "cleaned.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def tearDown(self):
        settings.DATA_DIR = self._prev
        self._tmp.cleanup()

    def _index_path(self) -> Path:
        return svc.document_index_dir(self.project_id) / "bm25_index.json"

    def test_built_index_is_stamped_and_current(self):
        built = svc.ensure_document_index(self.project_id)
        self.assertTrue(built["built"], built)
        payload = json.loads(self._index_path().read_text())
        self.assertEqual(payload["tokenizer_version"], svc.BM25_TOKENIZER_VERSION)
        self.assertEqual(svc.ensure_document_index(self.project_id)["reason"], "up_to_date")

    def test_missing_or_different_version_is_stale_and_rebuilt(self):
        svc.ensure_document_index(self.project_id)
        path = self._index_path()
        for stamp in (None, svc.BM25_TOKENIZER_VERSION - 1, svc.BM25_TOKENIZER_VERSION + 1):
            with self.subTest(stamp=stamp):
                payload = json.loads(path.read_text())
                if stamp is None:
                    payload.pop("tokenizer_version", None)
                else:
                    payload["tokenizer_version"] = stamp
                path.write_text(json.dumps(payload))
                os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 1000))  # newer than cleaned.jsonl
                status = svc.document_index_status(self.project_id)
                self.assertTrue(status["exists"] and status["stale"] and not status["available"])
                rebuilt = svc.ensure_document_index(self.project_id)
                self.assertTrue(rebuilt["built"], rebuilt)
                self.assertEqual(json.loads(path.read_text())["tokenizer_version"], svc.BM25_TOKENIZER_VERSION)

    def test_qa_index_check_requires_the_current_version(self):
        path = Path(self._tmp.name) / "qa" / "bm25_index.json"
        path.parent.mkdir()
        base = {"corpus_source": svc.QA_CORPUS_SOURCE}
        path.write_text(json.dumps({**base, "tokenizer_version": svc.BM25_TOKENIZER_VERSION}))
        self.assertTrue(svc.qa_index_is_current(path))
        path.write_text(json.dumps(base))
        self.assertFalse(svc.qa_index_is_current(path))
        path.write_text(json.dumps({**base, "tokenizer_version": svc.BM25_TOKENIZER_VERSION - 1}))
        self.assertFalse(svc.qa_index_is_current(path))

    def test_retrieve_refuses_an_index_from_another_tokenizer_version(self):
        svc.ensure_document_index(self.project_id)
        path = self._index_path()
        for stamp in (None, svc.BM25_TOKENIZER_VERSION - 1):
            with self.subTest(stamp=stamp):
                payload = json.loads(path.read_text())
                payload.pop("tokenizer_version", None)
                if stamp is not None:
                    payload["tokenizer_version"] = stamp
                path.write_text(json.dumps(payload))
                self.assertFalse(svc.index_tokenizer_is_current(path.parent))
                with self.assertRaises(svc.AutoRagUnavailable) as ctx:
                    svc.retrieve("user name", index_dir=path.parent, k=1)
                self.assertIn("rebuild", str(ctx.exception))
                with self.assertRaises(svc.AutoRagUnavailable):
                    svc.retrieve_ranked("user name", index_dir=path.parent, k=1)

    def test_retrieval_finds_the_identifier_by_its_words(self):
        svc.ensure_document_index(self.project_id)
        hits = svc.retrieve(
            "which passage mentions user name", index_dir=svc.document_index_dir(self.project_id), k=1
        )
        self.assertEqual(hits[0]["payload"]["chunk_id"], 0)


class ProseIndexTests(unittest.TestCase):
    """An index of prose only tokenizes passages AND queries as before identifier splitting."""

    def test_prose_index_matches_the_v1_tokens_and_marks_no_code(self):
        rows = [{"id": 1, "text": "Married filing SepaRately status"}, {"id": 2, "text": "Call REDACTED_PHONE for help"}]
        with tempfile.TemporaryDirectory() as tmp:
            svc.build_bm25_index(rows, recipe_id=svc.DOCUMENTS_CORPUS, output_dir=Path(tmp))
            payload = json.loads((Path(tmp) / "bm25_index.json").read_text())
            self.assertIs(payload["code_rows"], False)
            terms = set(payload["doc_freq"])
            self.assertEqual(terms, set(_old_tokenize(rows[0]["text"])) | set(_old_tokenize(rows[1]["text"])))
            hits = svc.retrieve("filing SepaRately", index_dir=Path(tmp), k=1)
            self.assertEqual(hits[0]["payload"]["id"], 1)

    def test_code_rows_are_split_and_flag_the_index(self):
        rows = [{"id": 1, "text": "def getUserName(): pass", "path": "a.py", "start_line": 1},
                {"id": 2, "text": "plain prose about SepaRately"}]
        with tempfile.TemporaryDirectory() as tmp:
            svc.build_bm25_index(rows, recipe_id=svc.DOCUMENTS_CORPUS, output_dir=Path(tmp))
            payload = json.loads((Path(tmp) / "bm25_index.json").read_text())
            self.assertIs(payload["code_rows"], True)
            self.assertIn("getusername", payload["doc_freq"])
            self.assertNotIn("sepa", payload["doc_freq"])   # prose rows stay v1 even in a mixed index
            self.assertEqual(svc.retrieve("user name", index_dir=Path(tmp), k=1)[0]["payload"]["id"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
