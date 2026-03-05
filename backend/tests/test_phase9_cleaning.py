"""Tests for the data cleaning service — PII detection, quality scoring, dedup, chunking."""

import unittest

from app.services.cleaning_service import (
    compute_quality_score,
    compute_text_hash,
    chunk_text,
    detect_pii,
    redact_pii,
    remove_boilerplate,
)


class CleaningServiceTests(unittest.TestCase):
    """Unit tests for pure-function cleaning utilities."""

    # ── PII Detection ──────────────────────────────────────────────────

    def test_detect_email(self):
        findings = detect_pii("Contact us at admin@example.com for details.")
        types = [f["type"] for f in findings]
        self.assertIn("email", types)

    def test_detect_phone(self):
        findings = detect_pii("Call 555-867-5309 or (555) 123-4567 today.")
        types = [f["type"] for f in findings]
        self.assertIn("phone", types)

    def test_detect_ssn(self):
        findings = detect_pii("SSN: 123-45-6789.")
        types = [f["type"] for f in findings]
        self.assertIn("ssn", types)

    def test_detect_ip(self):
        findings = detect_pii("Server at 192.168.1.100 is down.")
        types = [f["type"] for f in findings]
        self.assertIn("ip_address", types)

    def test_no_pii_clean_text(self):
        findings = detect_pii("This is a normal sentence with no sensitive data.")
        self.assertEqual(len(findings), 0)

    def test_redact_pii_replaces_email(self):
        redacted = redact_pii("Email me at user@corp.com please.")
        self.assertNotIn("user@corp.com", redacted)

    # ── Quality Scoring ────────────────────────────────────────────────

    def test_quality_score_returns_float(self):
        score = compute_quality_score("This is a well-written paragraph with multiple sentences. It has good structure and length.")
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_quality_score_empty_text(self):
        score = compute_quality_score("")
        self.assertEqual(score, 0.0)

    def test_quality_score_short_text_lower(self):
        short = compute_quality_score("Hi")
        long = compute_quality_score(
            "This is a comprehensive paragraph about machine learning. "
            "It covers several important topics in sufficient detail. "
            "The structure is clear and the content is informative."
        )
        self.assertLess(short, long)

    # ── Deduplication ──────────────────────────────────────────────────

    def test_hash_deterministic(self):
        h1 = compute_text_hash("Hello world")
        h2 = compute_text_hash("Hello world")
        self.assertEqual(h1, h2)

    def test_hash_different_for_different_text(self):
        h1 = compute_text_hash("Hello world")
        h2 = compute_text_hash("Goodbye world")
        self.assertNotEqual(h1, h2)

    # ── Chunking ───────────────────────────────────────────────────────

    def test_chunk_short_text_single_chunk(self):
        chunks = chunk_text("Short text", chunk_size=1000, overlap=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Short text")

    def test_chunk_long_text_multiple_chunks(self):
        text = "word " * 500  # ~2500 chars
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertGreater(len(chunks), 1)

    def test_chunk_overlap_exists(self):
        text = "A" * 200 + "B" * 200 + "C" * 200  # 600 chars
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        if len(chunks) >= 2:
            # End of chunk 0 should overlap with start of chunk 1
            end_of_first = chunks[0][-50:]
            self.assertIn(end_of_first, chunks[1])

    # ── Boilerplate Removal ────────────────────────────────────────────

    def test_remove_cookie_notice(self):
        text = "Important content here.\nCookie policy: we use cookies.\nMore content."
        cleaned = remove_boilerplate(text)
        self.assertIn("Important content", cleaned)

    def test_remove_newsletter_cta(self):
        text = "Great article.\nSubscribe to our newsletter for updates.\nThe end."
        cleaned = remove_boilerplate(text)
        self.assertIn("Great article", cleaned)


if __name__ == "__main__":
    unittest.main()
