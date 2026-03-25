"""Phase 36 tests: target compatibility relies on model introspection metadata."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ["DEBUG"] = "false"

from app.services.target_profile_service import check_compatibility, estimate_metrics


class Phase36TargetCompatibilityIntrospectionTests(unittest.TestCase):
    @patch("app.services.target_profile_service.introspect_hf_model")
    def test_check_compatibility_blocks_large_model_from_introspection(self, mock_introspect):
        mock_introspect.return_value = {
            "model_id": "acme/legal-6b",
            "resolved": True,
            "source": "hf_config",
            "params_estimate_b": 6.2,
            "memory_profile": {"estimated_min_vram_gb": 10.1, "estimated_ideal_vram_gb": 14.0},
            "architecture": "causal_lm",
            "context_length": 8192,
            "license": "apache-2.0",
        }

        payload = check_compatibility("acme/legal-6b", "mobile_cpu")
        self.assertFalse(bool(payload.get("compatible")), payload)
        reasons = [str(item) for item in list(payload.get("reasons") or [])]
        self.assertTrue(any("exceeds target limit" in item for item in reasons), reasons)
        model_metadata = dict(payload.get("model_metadata") or {})
        self.assertEqual(float(model_metadata.get("parameters_billions") or 0.0), 6.2)
        self.assertEqual(str(model_metadata.get("parameters_source") or ""), "introspection")

    @patch("app.services.target_profile_service.introspect_hf_model")
    def test_check_compatibility_hard_blocks_clear_vram_over_target(self, mock_introspect):
        mock_introspect.return_value = {
            "model_id": "acme/edge-6b",
            "resolved": True,
            "source": "hf_config",
            "params_estimate_b": 6.0,
            "memory_profile": {"estimated_min_vram_gb": 7.4, "estimated_ideal_vram_gb": 10.0},
            "architecture": "causal_lm",
            "context_length": 4096,
            "license": "apache-2.0",
        }

        payload = check_compatibility("acme/edge-6b", "edge_gpu")
        self.assertFalse(bool(payload.get("compatible")), payload)
        reasons = [str(item) for item in list(payload.get("reasons") or [])]
        self.assertTrue(any("VRAM" in item and "exceeds target baseline" in item for item in reasons), reasons)
        vram_check = dict(payload.get("vram_check") or {})
        self.assertEqual(str(vram_check.get("status") or ""), "blocked", vram_check)

    @patch("app.services.target_profile_service.introspect_hf_model")
    def test_check_compatibility_warns_when_vram_estimate_is_unknown(self, mock_introspect):
        mock_introspect.return_value = {
            "model_id": "acme/custom-unknown",
            "resolved": False,
            "source": "none",
            "params_estimate_b": None,
            "memory_profile": {},
            "architecture": "unknown",
            "context_length": None,
            "license": None,
        }

        payload = check_compatibility("acme/custom-unknown", "edge_gpu")
        self.assertTrue(bool(payload.get("compatible")), payload)
        warnings = [str(item) for item in list(payload.get("warnings") or [])]
        self.assertTrue(any("Unable to estimate minimum VRAM" in item for item in warnings), warnings)
        vram_check = dict(payload.get("vram_check") or {})
        self.assertEqual(str(vram_check.get("status") or ""), "unknown", vram_check)

    @patch("app.services.target_profile_service.introspect_hf_model")
    def test_check_compatibility_falls_back_to_name_hint(self, mock_introspect):
        mock_introspect.return_value = {
            "model_id": "acme/support-3b",
            "resolved": False,
            "source": "none",
            "params_estimate_b": None,
            "memory_profile": {},
            "architecture": "unknown",
            "context_length": None,
            "license": None,
        }

        payload = check_compatibility("acme/support-3b", "mobile_cpu")
        self.assertTrue(bool(payload.get("compatible")), payload)
        model_metadata = dict(payload.get("model_metadata") or {})
        self.assertEqual(float(model_metadata.get("parameters_billions") or 0.0), 3.0)
        self.assertEqual(str(model_metadata.get("parameters_source") or ""), "name_hint")

    @patch("app.services.target_profile_service.introspect_hf_model")
    def test_check_compatibility_blocks_unknown_size_for_constrained_target(self, mock_introspect):
        mock_introspect.return_value = {
            "model_id": "acme/custom-unknown",
            "resolved": False,
            "source": "none",
            "params_estimate_b": None,
            "memory_profile": {},
            "architecture": "unknown",
            "context_length": None,
            "license": None,
        }

        payload = check_compatibility("acme/custom-unknown", "browser_webgpu")
        self.assertFalse(bool(payload.get("compatible")), payload)
        reasons = [str(item) for item in list(payload.get("reasons") or [])]
        self.assertTrue(any("could not be inferred" in item for item in reasons), reasons)

    @patch("app.services.target_profile_service.introspect_hf_model")
    def test_estimate_metrics_prefers_introspection_memory_profile(self, mock_introspect):
        mock_introspect.return_value = {
            "model_id": "acme/edge-2b",
            "resolved": True,
            "source": "hf_config",
            "params_estimate_b": 2.0,
            "memory_profile": {"estimated_min_vram_gb": 5.5, "estimated_ideal_vram_gb": 8.0},
            "architecture": "causal_lm",
            "context_length": 4096,
            "license": "apache-2.0",
        }

        payload = estimate_metrics("acme/edge-2b", "edge_gpu")
        self.assertEqual(float(payload.get("estimated_memory_gb") or 0.0), 5.5)
        self.assertGreater(float(payload.get("estimated_latency_tps") or 0.0), 0.0)
        metadata = dict(payload.get("model_metadata") or {})
        self.assertEqual(str(metadata.get("source") or ""), "hf_config")


if __name__ == "__main__":
    unittest.main()
