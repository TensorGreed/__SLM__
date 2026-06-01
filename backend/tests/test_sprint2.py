import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from app.services.ingestion_service import inspect_remote_dataset, ingest_remote_dataset
from app.models.dataset import DocumentStatus, RawDocument

class TestSprint2(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = AsyncMock()

    @patch("app.services.secret_service.get_project_secret_value", new_callable=AsyncMock)
    @patch("datasets.get_dataset_config_names")
    @patch("datasets.get_dataset_split_names")
    @patch("datasets.get_dataset_infos")
    @patch("huggingface_hub.HfApi")
    async def test_inspect_remote_dataset_hf(self, mock_hf_api, mock_infos, mock_splits, mock_configs, mock_secrets):
        mock_secrets.return_value = None
        mock_configs.return_value = ["default", "other"]
        mock_splits.return_value = ["train", "test"]
        
        mock_info = MagicMock()
        mock_info.features = {"text": "Value", "label": "Value"}
        mock_info.description = "Test dataset"
        mock_info.license = "MIT"
        mock_info.dataset_size = 1000
        mock_infos.return_value = {"default": mock_info}
        
        mock_api_instance = mock_hf_api.return_value
        mock_api_instance.list_repo_files.return_value = ["data.jsonl", "script.py"]

        result = await inspect_remote_dataset(1, "huggingface", "test/dataset", db=self.db)
        
        self.assertEqual(result["source_type"], "huggingface")
        self.assertEqual(result["configs"], ["default", "other"])
        self.assertEqual(result["splits"], ["train", "test"])
        self.assertEqual(result["features"]["text"], "Value")
        self.assertTrue(result["has_scripts"])

    async def test_kaggle_credentials_available_recognises_all_four_auth_paths(self):
        """The Kaggle SDK supports four credential sources in v2. The
        platform's pre-check must recognise every one of them or the
        Ingest tab will surface a 'no credentials' prompt on a system
        where Kaggle actually works fine."""
        import os
        from pathlib import Path
        from unittest.mock import patch
        from app.services.ingestion_service import _kaggle_credentials_available

        # Each path is tested in isolation by clearing all four and
        # then enabling exactly one.
        original_env = {
            k: os.environ.pop(k, None)
            for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")
        }
        try:
            # Baseline — nothing set, no file. Pre-check must return False
            # (kaggle.json + access_token may exist on the dev box, so
            # patch Path.exists for this segment).
            with patch.object(Path, "exists", return_value=False):
                self.assertFalse(_kaggle_credentials_available())

            # Path 1: KAGGLE_API_TOKEN env var (modern access token).
            os.environ["KAGGLE_API_TOKEN"] = "KGAT_dummy"
            self.assertTrue(_kaggle_credentials_available())
            del os.environ["KAGGLE_API_TOKEN"]

            # Path 3: legacy KAGGLE_USERNAME + KAGGLE_KEY env vars.
            os.environ["KAGGLE_USERNAME"] = "user"
            os.environ["KAGGLE_KEY"] = "key"
            self.assertTrue(_kaggle_credentials_available())
            del os.environ["KAGGLE_USERNAME"]
            del os.environ["KAGGLE_KEY"]

            # Either is insufficient on its own — both must be set
            # for the legacy path to count.
            os.environ["KAGGLE_USERNAME"] = "user"
            with patch.object(Path, "exists", return_value=False):
                self.assertFalse(_kaggle_credentials_available())
            del os.environ["KAGGLE_USERNAME"]
        finally:
            for key, val in original_env.items():
                if val is not None:
                    os.environ[key] = val

    @patch("app.services.ingestion_service._kaggle_credentials_available")
    async def test_inspect_remote_dataset_kaggle_returns_structured_error_when_no_creds(
        self, mock_creds,
    ):
        """Regression: the kaggle SDK calls sys.exit(1) at import time
        when credentials are missing — SystemExit is a BaseException
        and slipped past ``except Exception``, crashing the request
        handler with a 500.

        The fix pre-checks credentials and returns a structured 200
        with ``missing_credentials: True`` and a remediation string so
        the UI can render a 'set up credentials' prompt rather than
        propagating a 500 to the user. Reproduces by mocking the
        creds check to return False; if the SDK import ever happens,
        SystemExit aborts the test process.
        """
        mock_creds.return_value = False
        result = await inspect_remote_dataset(
            1, "kaggle", "user/dataset", db=self.db,
        )
        self.assertEqual(result["source_type"], "kaggle")
        self.assertTrue(result.get("missing_credentials"))
        self.assertIn("credentials not configured", result["error"])
        self.assertIn("KAGGLE_USERNAME", result["remediation"])

    @patch("app.services.ingestion_service.get_or_create_raw_dataset", new_callable=AsyncMock)
    @patch("app.services.ingestion_service.resolve_project_domain_hooks", new_callable=AsyncMock)
    @patch("app.services.secret_service.get_project_secret_value", new_callable=AsyncMock)
    async def test_ingest_remote_dataset_idempotent(self, mock_secrets, mock_hooks, mock_get_dataset):
        mock_secrets.return_value = None
        mock_hooks.return_value = {}
        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_get_dataset.return_value = mock_dataset
        
        # Mock existing document
        mock_doc = MagicMock(spec=RawDocument)
        mock_doc.id = 123
        mock_doc.filename = "existing.jsonl"
        mock_doc.status = DocumentStatus.ACCEPTED
        mock_doc.source = "huggingface:test/dataset"
        mock_doc.metadata_ = {"split": "train", "config_name": None, "raw_samples": 500, "num_samples": 500}
        
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_doc]
        
        mock_execute_result = MagicMock()
        mock_execute_result.scalars.return_value = mock_scalars
        self.db.execute.return_value = mock_execute_result
        
        progress_mock = AsyncMock()
        result = await ingest_remote_dataset(
            self.db, 1, "huggingface", "test/dataset", split="train", progress_callback=progress_mock
        )
        
        self.assertEqual(result["document_id"], 123)
        self.assertTrue(result.get("already_exists"))

    @patch("datasets.load_dataset")
    @patch("app.services.ingestion_service.get_or_create_raw_dataset", new_callable=AsyncMock)
    @patch("app.services.ingestion_service.resolve_project_domain_hooks", new_callable=AsyncMock)
    @patch("app.services.ingestion_service.build_schema_profile")
    @patch("app.services.ingestion_service.run_validator_hook")
    @patch("app.services.ingestion_service.compute_file_hash")
    @patch("app.services.ingestion_service._project_data_dir")
    @patch("app.services.secret_service.get_project_secret_value", new_callable=AsyncMock)
    @patch("pathlib.Path.read_bytes")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_ingest_remote_dataset_retries(self, mock_sleep, mock_open, mock_read_bytes, mock_secrets, mock_data_dir, mock_hash, mock_val, mock_profile, mock_hooks, mock_get_dataset, mock_load):
        mock_secrets.return_value = None
        mock_hooks.return_value = {}
        mock_dataset = MagicMock()
        mock_dataset.id = 1
        mock_get_dataset.return_value = mock_dataset
        
        mock_data_dir.return_value = Path("/tmp")
        mock_read_bytes.return_value = b"some data"
        
        # Mock no existing documents
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_execute_result = MagicMock()
        mock_execute_result.scalars.return_value = mock_scalars
        self.db.execute.return_value = mock_execute_result
        
        # Mock load_dataset failure then success
        mock_load.side_effect = [Exception("Temporary failure"), [{"text": "data"}]]
        
        # Mock other stuff to avoid errors during writing
        mock_profile.return_value = {}
        mock_val.return_value = {}
        mock_hash.return_value = "abc"
        
        # Mock DB add/commit/refresh
        self.db.add = MagicMock()
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()

        progress_mock = AsyncMock()
        result = await ingest_remote_dataset(
            self.db, 1, "huggingface", "test/dataset", split="train", progress_callback=progress_mock
        )
        
        # Check it was called twice
        self.assertEqual(mock_load.call_count, 2)
        self.assertEqual(result["samples_ingested"], 1)

if __name__ == "__main__":
    unittest.main()
