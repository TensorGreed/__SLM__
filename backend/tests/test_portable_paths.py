"""portable_paths: what an export records carries no machine-specific path."""

from __future__ import annotations

import unittest

from app.services.portable_paths import portable_paths, portable_text

RUN = "/home/ada/brewslm/data/exports/31/37/run-20261017T151456Z"
HOME = "/home/ada"


class PortableTextTests(unittest.TestCase):
    def test_paths_inside_the_run_dir_become_relative(self):
        self.assertEqual(portable_text(f"{RUN}/model/config.json", RUN, HOME), "model/config.json")
        self.assertEqual(portable_text(RUN, RUN, HOME), ".")
        self.assertEqual(
            portable_text(f"model directory must exist at '{RUN}/model'.", RUN, HOME),
            "model directory must exist at 'model'.",
        )

    def test_other_home_paths_become_tilde(self):
        self.assertEqual(portable_text("/home/ada/.cache/hf/x", RUN, HOME), "~/.cache/hf/x")
        self.assertEqual(portable_text("cd /home/ada", RUN, HOME), "cd ~")

    def test_unrelated_text_is_unchanged(self):
        for text in ("/opt/models/qwen", "run-20261017T151456Z", "Q4_K_M", ""):
            self.assertEqual(portable_text(text, RUN, HOME), text)

    def test_no_run_dir_still_hides_home(self):
        self.assertEqual(portable_text(f"{RUN}/model", None, HOME), "~/brewslm/data/exports/31/37/run-20261017T151456Z/model")


class PortablePathsTests(unittest.TestCase):
    def test_walks_nested_values_and_keeps_non_strings(self):
        manifest = {
            "run_dir": RUN,
            "deployment": {"run_dir": RUN, "checks": [{"message": f"ok at {RUN}/serve.py", "passed": True}]},
            "artifacts": [{"path": "model", "size_bytes": 12}],
            "count": 3,
            "none": None,
        }
        out = portable_paths(manifest, RUN, HOME)
        self.assertEqual(out["run_dir"], ".")
        self.assertEqual(out["deployment"]["run_dir"], ".")
        self.assertEqual(out["deployment"]["checks"][0], {"message": "ok at serve.py", "passed": True})
        self.assertEqual(out["artifacts"], [{"path": "model", "size_bytes": 12}])
        self.assertEqual((out["count"], out["none"]), (3, None))
        self.assertEqual(manifest["run_dir"], RUN, "the input is not mutated")


if __name__ == "__main__":
    unittest.main()
