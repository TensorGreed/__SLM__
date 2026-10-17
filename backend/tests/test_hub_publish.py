"""Publish an export to the Hugging Face Hub: the pure front-matter helper,
the service with a fake HfApi (no network), the route's error mapping."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from app.database import async_session_factory  # noqa: E402
from app.main import app  # noqa: E402
from app.models.experiment import Experiment, ExperimentStatus  # noqa: E402
from app.models.export import ExportFormat, ExportStatus  # noqa: E402
from app.services import hub_publish_service as hub  # noqa: E402
from test_span_packaging import _span_run_dir  # noqa: E402

client: TestClient | None = None


def setUpModule():
    global client
    client = TestClient(app)
    client.__enter__()


def tearDownModule():
    if client is not None:
        client.__exit__(None, None, None)


class CardFrontMatterTests(unittest.TestCase):
    def test_adds_license_and_language_once(self):
        card = "---\nlibrary_name: transformers\ntags:\n- brewslm\n---\n\n# T\n"
        out = hub.card_with_hub_metadata(card)
        self.assertIn("license: mit\n", out)
        self.assertIn("language:\n- en\n", out)
        self.assertTrue(out.startswith("---\nlibrary_name: transformers\ntags:\n- brewslm\n"))
        self.assertEqual(hub.card_with_hub_metadata(out), out, "idempotent")

    def test_keeps_an_existing_license(self):
        card = "---\nlicense: apache-2.0\nlanguage:\n- de\n---\n# T\n"
        self.assertEqual(hub.card_with_hub_metadata(card), card)

    def test_card_without_front_matter_gets_one(self):
        self.assertTrue(hub.card_with_hub_metadata("# T\n").startswith("---\nlicense: mit\nlanguage:\n- en\n---\n\n# T"))

    def test_repo_id_validation(self):
        self.assertEqual(hub.validate_repo_id(" org/my-model.v2 "), "org/my-model.v2")
        for bad in ("nomodel", "a/b/c", "", "-x/y", "x/"):
            with self.assertRaises(hub.HubPublishError):
                hub.validate_repo_id(bad)


class FakeHubApi:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))
        return "https://huggingface.co/" + kw["repo_id"]

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))
        return type("Commit", (), {"oid": "abc123", "commit_url": "https://huggingface.co/x/y/commit/abc123"})()

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw))


def _project() -> int:
    return client.post("/api/projects", json={"name": f"hub-{time.time_ns()}", "description": "x"}).json()["id"]


def _export(pid: int, root: Path) -> tuple[int, dict]:
    from app.services.export_service import create_export, run_export

    async def _go():
        async with async_session_factory() as db:
            exp = Experiment(
                project_id=pid, name="tc", status=ExperimentStatus.COMPLETED,
                base_model="microsoft/deberta-v3-base", output_dir=str(_span_run_dir(root)),
                config={"task_type": "token_classification", "max_seq_length": 64},
            )
            db.add(exp)
            await db.flush()
            export = await create_export(db, pid, exp.id, ExportFormat.HUGGINGFACE)
            export = await run_export(db, pid, export.id, run_smoke_tests=False)
            await db.commit()
            assert export.status == ExportStatus.COMPLETED, export.manifest
            return export.id, export.manifest

    return asyncio.run(_go())


class PublishServiceTests(unittest.TestCase):
    def test_publish_uploads_model_dir_and_records_on_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = _project()
            export_id, manifest = _export(pid, Path(tmp))
            run_dir = Path(manifest["run_dir"])
            api = FakeHubApi()

            async def _go():
                async with async_session_factory() as db:
                    return await hub.publish_export_to_hub(
                        db, pid, export_id, repo_id="me/tagger", private=True, api=api,
                    )

            record = asyncio.run(_go())
            kinds = [k for k, _ in api.calls]
            self.assertEqual(kinds[:2], ["create_repo", "upload_folder"])
            self.assertEqual(api.calls[0][1]["repo_id"], "me/tagger")
            self.assertTrue(api.calls[0][1]["private"])
            self.assertEqual(Path(api.calls[1][1]["folder_path"]), run_dir / "model")
            uploaded_to = {c[1]["path_in_repo"] for c in api.calls if c[0] == "upload_file"}
            self.assertIn("brewslm/manifest.json", uploaded_to)
            self.assertEqual(record["repo_id"], "me/tagger")
            self.assertEqual(record["url"], "https://huggingface.co/me/tagger")
            self.assertEqual(record["commit"], "abc123")
            self.assertIn("README.md", record["files"])
            card = (run_dir / "model" / "README.md").read_text()
            self.assertIn("license: mit", card)
            on_disk = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(on_disk["hub_publish"]["repo_id"], "me/tagger")
            listed = client.get(f"/api/projects/{pid}/export/list").json()
            self.assertEqual(listed[0]["id"], export_id)

    def test_route_maps_errors(self):
        pid = _project()
        r = client.post(f"/api/projects/{pid}/export/999999/publish-hub", json={"repo_id": "me/x"})
        self.assertEqual(r.status_code, 404)
        r = client.post(f"/api/projects/{pid}/export/999999/publish-hub", json={"repo_id": "bad"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("<owner>/<name>", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
