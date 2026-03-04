"""Phase 5 tests: strict runtime modes and fallback behavior."""

import unittest

from app.config import settings
from app.services.compression_service import quantize_model
from app.services.synthetic_service import generate_qa_pairs


class StrictModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthetic_generation_requires_teacher_when_demo_disabled(self):
        prev_allow = settings.ALLOW_SYNTHETIC_DEMO_FALLBACK
        prev_teacher_url = settings.TEACHER_MODEL_API_URL
        try:
            settings.ALLOW_SYNTHETIC_DEMO_FALLBACK = False
            settings.TEACHER_MODEL_API_URL = ""
            with self.assertRaises(ValueError):
                await generate_qa_pairs(
                    db=None,  # type: ignore[arg-type]
                    project_id=1,
                    source_text="This is a sufficiently long source text for testing synthetic generation.",
                    num_pairs=3,
                    api_url="",
                    api_key="",
                    model_name="llama3",
                )
        finally:
            settings.ALLOW_SYNTHETIC_DEMO_FALLBACK = prev_allow
            settings.TEACHER_MODEL_API_URL = prev_teacher_url

    async def test_compression_stub_requires_explicit_opt_in(self):
        prev_backend = settings.COMPRESSION_BACKEND
        prev_allow_stub = settings.ALLOW_STUB_COMPRESSION
        try:
            settings.COMPRESSION_BACKEND = "stub"
            settings.ALLOW_STUB_COMPRESSION = False
            with self.assertRaises(ValueError):
                await quantize_model(
                    project_id=99,
                    model_path="/tmp/nonexistent-model",
                    bits=4,
                    output_format="gguf",
                )
        finally:
            settings.COMPRESSION_BACKEND = prev_backend
            settings.ALLOW_STUB_COMPRESSION = prev_allow_stub


if __name__ == "__main__":
    unittest.main()
