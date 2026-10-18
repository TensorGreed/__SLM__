"""The API-reference generator escapes MDX-significant prose in docstrings.

``slm-docs/scripts/generate_api_ref.py`` writes endpoint descriptions as
Markdown prose; MDX 3 treats ``<reason>`` as a JSX tag and ``{x}`` as an
expression, and one such docstring fails the whole docs build.
"""

import importlib.util
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "slm-docs" / "scripts" / "generate_api_ref.py"


def _load():
    spec = importlib.util.spec_from_file_location("generate_api_ref", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProseSafeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = _load()

    def test_angle_brackets_and_braces_are_escaped_in_prose(self):
        out = self.gen._prose_safe('skipped because <reason>; fails in <10s; body {"k": 1}')
        self.assertEqual(out, 'skipped because &lt;reason&gt;; fails in &lt;10s; body \\{"k": 1\\}')

    def test_code_spans_are_left_as_written(self):
        out = self.gen._prose_safe("returns ``{a: <b>}`` when `x<y` and <z>")
        self.assertEqual(out, "returns ``{a: <b>}`` when `x<y` and &lt;z&gt;")

    def test_operation_description_goes_through_prose_escaping(self):
        op = {"summary": "Run", "description": "Skipped because <reason>.", "responses": {}}
        text = self.gen._render_endpoint("get", "/api/x", op, {})
        self.assertIn("Skipped because &lt;reason&gt;.", text)
        self.assertNotIn("<reason>", text)


if __name__ == "__main__":
    unittest.main()
