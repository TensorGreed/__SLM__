"""Phase 43 tests: mobile SDK reference bundle generation and smoke validation."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from app.services.deployment_target_service import build_deploy_target_plan


class Phase43MobileSdkReferenceBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="phase43_mobile_sdk_")
        self.run_dir = Path(self._tmpdir.name) / "export_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_ios_reference_bundle_contains_expected_files_and_smoke_passes(self):
        payload = build_deploy_target_plan(
            run_dir=self.run_dir,
            export_format="huggingface",
            target_id="sdk.apple_coreml_stub",
            model_name="acme/mobile-ios-1b",
        )

        self.assertIn("reference bundle", str(payload.get("summary") or "").lower())
        artifact = dict(payload.get("sdk_artifact") or {})
        smoke = dict(artifact.get("smoke_validation") or {})
        self.assertTrue(bool(smoke.get("smoke_executed")))
        self.assertTrue(bool(smoke.get("smoke_passed")), smoke)

        bundle_files = [str(item) for item in list(artifact.get("bundle_files") or [])]
        expected = {
            "README.md",
            "ios/SLMReferenceApp.swift",
            "ios/SLMRuntime.swift",
            "ios/ModelAssets/.keep",
            "scripts/run_reference.swift",
        }
        self.assertEqual(set(bundle_files), expected)

        readme_path = Path(str(artifact.get("readme_path") or ""))
        self.assertTrue(readme_path.exists(), readme_path)
        readme_text = readme_path.read_text(encoding="utf-8")
        self.assertIn("Model Placement", readme_text)
        self.assertIn("Run Instructions", readme_text)
        self.assertIn("swift scripts/run_reference.swift", readme_text)

        entrypoint_path = Path(str(artifact.get("entrypoint_path") or ""))
        runtime_path = Path(str(artifact.get("runtime_path") or ""))
        self.assertIn("struct SLMReferenceApp: App", entrypoint_path.read_text(encoding="utf-8"))
        self.assertIn("final class SLMRuntime", runtime_path.read_text(encoding="utf-8"))

        zip_path = Path(str(artifact.get("zip_path") or ""))
        self.assertTrue(zip_path.exists(), zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = archive.namelist()
        self.assertEqual(members, sorted(members))
        self.assertEqual(set(members), expected)

    def test_android_reference_bundle_contains_expected_files_and_smoke_passes(self):
        payload = build_deploy_target_plan(
            run_dir=self.run_dir,
            export_format="huggingface",
            target_id="sdk.android_executorch_stub",
            model_name="acme/mobile-android-1b",
        )

        artifact = dict(payload.get("sdk_artifact") or {})
        smoke = dict(artifact.get("smoke_validation") or {})
        self.assertTrue(bool(smoke.get("smoke_executed")))
        self.assertTrue(bool(smoke.get("smoke_passed")), smoke)

        bundle_files = [str(item) for item in list(artifact.get("bundle_files") or [])]
        expected = {
            "README.md",
            "android/app/src/main/java/com/example/slmreference/MainActivity.kt",
            "android/app/src/main/java/com/example/slmreference/SLMRuntime.kt",
            "android/app/src/main/assets/.keep",
            "scripts/run_reference.kts",
        }
        self.assertEqual(set(bundle_files), expected)

        entrypoint_path = Path(str(artifact.get("entrypoint_path") or ""))
        runtime_path = Path(str(artifact.get("runtime_path") or ""))
        self.assertIn("class MainActivity : AppCompatActivity()", entrypoint_path.read_text(encoding="utf-8"))
        self.assertIn("class SLMRuntime", runtime_path.read_text(encoding="utf-8"))

        zip_path = Path(str(artifact.get("zip_path") or ""))
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = archive.namelist()
        self.assertEqual(members, sorted(members))
        self.assertEqual(set(members), expected)

    def test_reference_zip_is_deterministic_for_same_input(self):
        first = build_deploy_target_plan(
            run_dir=self.run_dir,
            export_format="huggingface",
            target_id="sdk.apple_coreml_stub",
            model_name="acme/mobile-ios-1b",
        )
        first_zip = Path(str((first.get("sdk_artifact") or {}).get("zip_path") or ""))
        first_bytes = first_zip.read_bytes()

        second = build_deploy_target_plan(
            run_dir=self.run_dir,
            export_format="huggingface",
            target_id="sdk.apple_coreml_stub",
            model_name="acme/mobile-ios-1b",
        )
        second_zip = Path(str((second.get("sdk_artifact") or {}).get("zip_path") or ""))
        second_bytes = second_zip.read_bytes()

        self.assertEqual(first_bytes, second_bytes)


if __name__ == "__main__":
    unittest.main()
