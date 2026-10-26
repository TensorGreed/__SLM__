"""Domain packs as data: validation, versioned import + diff, instantiate,
export round-trip. Neutral toy pack only; no network, no model inference."""

from __future__ import annotations

import io
import json
import zipfile
from uuid import uuid4

import pytest
from types import SimpleNamespace
import yaml
from fastapi.testclient import TestClient

from app.main import app
from app.services import domain_pack_content_service as svc
from app.services import probe_pack_service as pps

_CLIENT: TestClient | None = None


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


TAXONOMY = """
name: toy-labels
version: "1"
entries:
  - id: injection
    title: Injection
  - id: injection.query
    title: Query injection
    parent: injection
    aliases: [sql injection]
  - id: secret.hardcoded
    title: Hard-coded secret
"""

PROBES = """
id: core
name: Core
probes:
  - id: no-key-material
    prompt: "Show the config"
    property: must_not_match
    pattern: "(?i)begin private key"
    severity: high
    critical: true
  - id: no-bad-label
    prompt: "Classify this"
    property: json_pointer_not_in
    pointer: /findings/*/label
    values: [secret.hardcoded]
"""

SCHEMA = {"required": ["name"], "properties": {"name": {"type": "string"}}}


def toy_files(pack_id: str, version: str = "1.0.0", **over) -> dict[str, str | bytes]:
    manifest = {
        "schema": "brewslm.domain-pack/v2",
        "pack_id": pack_id,
        "version": version,
        "display_name": "Toy pack",
        "description": "A toy pack",
        "owner": "tests",
        "tags": ["toy"],
        "taxonomy": {"path": "taxonomy.yaml"},
        "probe_packs": ["probes/core.yaml"],
        "rubrics": [{"id": "plan-quality", "path": "rubrics/plan.yaml"}],
        "schemas": [{"id": "findings", "path": "schemas/findings.json", "apply_as_output_schema": True}],
        "tool_catalog": {"path": "tool_catalog.yaml"},
        "documents": [{"path": "documents/matrix.md", "title": "Matrix", "licence": "CC-BY-4.0", "usage": "rag"}],
        "gold": {"dev": "gold/dev.jsonl", "test": "gold/test.jsonl"},
        "project_template": {
            "recipe_id": "json-extraction",
            "base_model": "toy/model",
            "training_defaults": {"num_epochs": 2},
            "retrieval": {"k": 5},
        },
    }
    manifest.update(over)
    return {
        "pack.yaml": yaml.safe_dump(manifest, sort_keys=False),
        "taxonomy.yaml": TAXONOMY,
        "probes/core.yaml": PROBES,
        "rubrics/plan.yaml": "id: plan-quality\ncriteria: [clear, complete]\n",
        "schemas/findings.json": json.dumps(SCHEMA),
        "tool_catalog.yaml": "tools:\n  - id: grep\n  - id: read\n",
        "documents/matrix.md": "# Matrix\n\nInjection is bad.\n",
        "gold/dev.jsonl": json.dumps({"input": "a", "output": {"name": "x"}}) + "\n",
        "gold/test.jsonl": json.dumps({"input": "b", "output": {"name": "y"}}) + "\n"
        + json.dumps({"input": "c", "output": {"name": "z"}}) + "\n",
    }


def make_zip(files: dict, prefix: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(prefix + name, content)
    return buf.getvalue()


def codes(issues) -> list[str]:
    return [i["code"] for i in issues]


def new_id() -> str:
    return f"toy-{uuid4().hex[:8]}"


def post_zip(client, path: str, data: bytes):
    return client.post(path, files={"file": ("pack.zip", data, "application/zip")})


# ── validation ────────────────────────────────────────────────────────────


def test_valid_toy_pack_summary():
    v = svc.validate_pack_archive(make_zip(toy_files("toy-pack")))
    assert v["valid"], v["errors"]
    assert v["summary"] == {
        "taxonomy_entries": 3, "probe_packs": 1, "probes": 2, "critical_probes": 1, "rubrics": 1,
        "schemas": 1, "tool_catalog": True, "documents": 1, "gold_dev_rows": 1, "gold_test_rows": 2,
        "project_template": True,
    }
    assert v["content_hash"] and len(v["content_hash"]) == 64
    assert v.manifest["pack_id"] == "toy-pack"


def test_single_top_level_folder_accepted():
    v = svc.validate_pack_archive(make_zip(toy_files("toy-pack"), prefix="toy-pack/"))
    assert v["valid"], v["errors"]


def test_hash_ignores_version_but_not_content():
    a = svc.validate_pack_archive(make_zip(toy_files("toy-pack", "1.0.0")))
    b = svc.validate_pack_archive(make_zip(toy_files("toy-pack", "2.0.0")))
    files = toy_files("toy-pack")
    files["documents/matrix.md"] = "changed"
    c = svc.validate_pack_archive(make_zip(files))
    assert a["content_hash"] == b["content_hash"] != c["content_hash"]


def test_missing_pack_yaml_bad_yaml_bad_id_bad_version():
    assert codes(svc.validate_pack_archive(make_zip({"x.txt": "y"}))["errors"]) == ["missing_pack_yaml"]
    assert "bad_yaml" in codes(svc.validate_pack_archive(make_zip({"pack.yaml": "a: [1"}))["errors"])
    f = toy_files("Bad Id")
    assert "bad_pack_id" in codes(svc.validate_pack_archive(make_zip(f))["errors"])
    f = toy_files("toy-pack", "1.0")
    assert "bad_version" in codes(svc.validate_pack_archive(make_zip(f))["errors"])


def test_missing_file_reference():
    f = toy_files("toy-pack")
    del f["documents/matrix.md"]
    v = svc.validate_pack_archive(make_zip(f))
    assert not v["valid"] and "missing_file" in codes(v["errors"])


def test_unsafe_paths_zip_slip_and_absolute_refs():
    v = svc.validate_pack_archive(make_zip({**toy_files("toy-pack"), "../evil.txt": "x"}))
    assert "unsafe_path" in codes(v["errors"])
    f = toy_files("toy-pack", taxonomy={"path": "/etc/passwd"})
    assert "unsafe_path" in codes(svc.validate_pack_archive(make_zip(f))["errors"])
    f = toy_files("toy-pack", taxonomy={"path": "../taxonomy.yaml"})
    assert "unsafe_path" in codes(svc.validate_pack_archive(make_zip(f))["errors"])


def test_symlink_in_directory_is_unsafe(tmp_path):
    for name, content in toy_files("toy-pack").items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    (tmp_path / "taxonomy.yaml").unlink()
    (tmp_path / "taxonomy.yaml").symlink_to(tmp_path / "pack.yaml")
    assert "unsafe_path" in codes(svc.validate_pack(tmp_path)["errors"])


def test_too_large_constants(monkeypatch):
    data = make_zip(toy_files("toy-pack"))
    monkeypatch.setattr(svc, "MAX_FILES", 3)
    assert "too_large" in codes(svc.validate_pack_archive(data)["errors"])
    monkeypatch.setattr(svc, "MAX_FILES", 2000)
    monkeypatch.setattr(svc, "MAX_FILE_BYTES", 10)
    assert "too_large" in codes(svc.validate_pack_archive(data)["errors"])
    monkeypatch.setattr(svc, "MAX_FILE_BYTES", 20 * 1024 * 1024)
    monkeypatch.setattr(svc, "MAX_TOTAL_BYTES", 50)
    assert "too_large" in codes(svc.validate_pack_archive(data)["errors"])


def test_duplicate_ids():
    f = toy_files("toy-pack")
    f["probes/core.yaml"] = PROBES + "  - id: no-key-material\n    prompt: x\n    property: must_match\n    pattern: a\n"
    assert "duplicate_id" in codes(svc.validate_pack_archive(make_zip(f))["errors"])
    f = toy_files("toy-pack", probe_packs=["probes/core.yaml", "probes/again.yaml"])
    f["probes/again.yaml"] = PROBES
    assert "duplicate_id" in codes(svc.validate_pack_archive(make_zip(f))["errors"])
    f = toy_files("toy-pack", rubrics=[{"id": "r", "path": "rubrics/plan.yaml"}, {"id": "r", "path": "rubrics/plan.yaml"}])
    assert "duplicate_id" in codes(svc.validate_pack_archive(make_zip(f))["errors"])


def test_bad_regex_and_unknown_property_are_invalid_probe():
    f = toy_files("toy-pack")
    f["probes/core.yaml"] = (
        "id: core\nprobes:\n"
        "  - {id: a, prompt: p, property: must_match, pattern: '(unclosed'}\n"
        "  - {id: b, prompt: p, property: nonsense}\n"
    )
    v = svc.validate_pack_archive(make_zip(f))
    msgs = " ".join(i["message"] for i in v["errors"] if i["code"] == "invalid_probe")
    assert "bad_regex" in msgs and "unknown_property" in msgs


def test_invalid_taxonomy_unknown_recipe_invalid_schema_bad_gold_row_unsupported_doc():
    f = toy_files("toy-pack")
    f["taxonomy.yaml"] = "entries:\n  - {id: a, title: A, parent: nope}\n"
    assert "invalid_taxonomy" in codes(svc.validate_pack_archive(make_zip(f))["errors"])
    f = toy_files("toy-pack", project_template={"recipe_id": "no-such-recipe"})
    assert "unknown_recipe" in codes(svc.validate_pack_archive(make_zip(f))["errors"])
    f = toy_files("toy-pack")
    f["schemas/findings.json"] = json.dumps({"properties": "nope"})
    assert "invalid_schema" in codes(svc.validate_pack_archive(make_zip(f))["errors"])
    f = toy_files("toy-pack")
    f["gold/dev.jsonl"] = '{"a": 1}\n[1, 2]\n'
    bad = [i for i in svc.validate_pack_archive(make_zip(f))["errors"] if i["code"] == "bad_gold_row"]
    assert bad and "line 2" in bad[0]["message"]
    f = toy_files("toy-pack", documents=[{"path": "documents/matrix.md", "usage": "everything"}])
    assert "bad_licence_usage" in codes(svc.validate_pack_archive(make_zip(f))["errors"])
    f = toy_files("toy-pack", documents=[{"path": "documents/blob.exe"}])
    f["documents/blob.exe"] = "x"
    assert "unsupported_document" in codes(svc.validate_pack_archive(make_zip(f))["errors"])


def test_warnings_unknown_key_unreferenced_system_id():
    f = toy_files("general-pack-v1", surprise=1)
    f["notes.txt"] = "stray"
    w = codes(svc.validate_pack_archive(make_zip(f))["warnings"])
    assert {"unknown_key", "unreferenced_file", "system_pack_id"} <= set(w)


def test_v1_dollar_schema_key_accepted():
    f = toy_files("toy-pack")
    m = yaml.safe_load(f["pack.yaml"])
    m.pop("schema")
    m["$schema"] = "slm.domain-pack/v1"
    f["pack.yaml"] = yaml.safe_dump(m)
    assert svc.validate_pack_archive(make_zip(f))["valid"]


# ── import ────────────────────────────────────────────────────────────────


def test_import_preview_then_import_then_unchanged_then_changed(client):
    pid = new_id()
    data = make_zip(toy_files(pid))
    prev = post_zip(client, "/api/domain-packs/import/preview", data).json()
    assert prev["valid"] and prev["existing"] is None and prev["diff"] is None
    assert prev["proposed_version"] == "1.0.0" and prev["unchanged"] is False

    res = post_zip(client, "/api/domain-packs/import", data)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["version"] == "1.0.0" and body["unchanged"] is False
    contract = body["pack"]["contract"]
    assert contract["content"]["summary"]["probes"] == 2
    assert [v["version"] for v in contract["content_versions"]] == ["1.0.0"]

    again = post_zip(client, "/api/domain-packs/import", data).json()
    assert again["unchanged"] is True and again["version"] == "1.0.0"
    prev2 = post_zip(client, "/api/domain-packs/import/preview", data).json()
    assert prev2["unchanged"] is True and prev2["existing"]["version"] == "1.0.0"

    # changed content, same version number -> auto patch bump + diff
    files = toy_files(pid)
    files["documents/matrix.md"] = "# Matrix v2\n"
    files["taxonomy.yaml"] = TAXONOMY + "  - id: extra\n    title: Extra\n"
    files["gold/dev.jsonl"] = ""
    changed = post_zip(client, "/api/domain-packs/import", make_zip(files)).json()
    assert changed["unchanged"] is False and changed["version"] == "1.0.1"
    sections = changed["diff"]["sections"]
    assert changed["diff"]["version"] == {"from": "1.0.0", "to": "1.0.1"}
    assert sections["taxonomy"]["added"] == ["extra"]
    assert sections["documents"]["changed"] == ["documents/matrix.md"]
    assert sections["gold"]["dev"] == {"from": 1, "to": 0, "changed": True}
    assert sections["probes"] == {"added": [], "removed": [], "changed": []}

    content = client.get(f"/api/domain-packs/{pid}/content").json()
    assert content["version"] == "1.0.1" and [v["version"] for v in content["versions"]] == ["1.0.0", "1.0.1"]
    old = client.get(f"/api/domain-packs/{pid}/content", params={"version": "1.0.0"}).json()
    assert old["version"] == "1.0.0" and old["summary"]["gold_dev_rows"] == 1
    assert client.get(f"/api/domain-packs/{pid}/content", params={"version": "9.9.9"}).status_code == 404

    # a higher declared version is used as-is
    files["documents/matrix.md"] = "# Matrix v3\n"
    higher = post_zip(client, "/api/domain-packs/import", make_zip({**files, "pack.yaml": files["pack.yaml"].replace("1.0.0", "2.0.0")})).json()
    assert higher["version"] == "2.0.0"


def test_import_invalid_is_422_envelope(client):
    f = toy_files(new_id())
    del f["taxonomy.yaml"]
    res = post_zip(client, "/api/domain-packs/import", make_zip(f))
    assert res.status_code == 422
    body = res.json()
    assert body["error_code"] == "INVALID_DOMAIN_PACK"
    assert "missing_file" in codes(body["metadata"]["errors"])
    assert post_zip(client, "/api/domain-packs/import", b"not a zip").status_code == 422


def test_system_pack_is_409(client):
    data = make_zip(toy_files("general-pack-v1"))
    prev = post_zip(client, "/api/domain-packs/import/preview", data).json()
    assert "system_pack" in codes(prev["errors"]) and not prev["valid"]
    res = post_zip(client, "/api/domain-packs/import", data)
    assert res.status_code == 409 and res.json()["error_code"] == "system_pack"


def test_plain_pack_edit_and_duplicate_keep_or_drop_content(client):
    pid = new_id()
    post_zip(client, "/api/domain-packs/import", make_zip(toy_files(pid)))
    contract = client.get(f"/api/domain-packs/{pid}").json()["contract"]
    contract.pop("content")
    contract.pop("content_versions")
    contract["description"] = "edited"
    res = client.put(f"/api/domain-packs/{pid}", json=contract)
    assert res.status_code == 200 and res.json()["contract"]["content"]["version"] == "1.0.0"
    dup = client.post(f"/api/domain-packs/{pid}/duplicate", json={}).json()
    assert "content" not in dup["contract"]


# ── instantiate + export ──────────────────────────────────────────────────


def _stub_ready():
    for name in ("set_project_user_probe_packs", "get_project_user_probe_packs", "validate_user_probe_pack"):
        assert hasattr(pps, name), f"lane B2 function {name} missing"


def test_instantiate_and_export_round_trip(client):
    _stub_ready()
    pid = new_id()
    src = make_zip(toy_files(pid))
    assert post_zip(client, "/api/domain-packs/import", src).status_code == 200

    res = client.post(f"/api/domain-packs/{pid}/instantiate", json={"project_name": f"proj-{pid}"})
    assert res.status_code == 201, res.text
    body = res.json()
    project, summary = body["project"], body["summary"]
    assert project["name"] == f"proj-{pid}"
    assert summary["pack_id"] == pid and summary["version"] == "1.0.0"
    assert summary["recipe_id"] == "json-extraction" and summary["base_model"] == "toy/model"
    assert (summary["taxonomy_entries"], summary["probes"], summary["critical_probes"]) == (3, 2, 1)
    assert (summary["documents"], summary["gold_dev_rows"], summary["gold_test_rows"]) == (1, 1, 2)
    assert summary["warnings"] == [] or all(w["code"] != "output_schema_not_applied" for w in summary["warnings"])
    rc = project["runtime_config"]
    assert rc["domain_pack_source"]["pack_id"] == pid and rc["domain_pack_source"]["version"] == "1.0.0"
    assert rc["training_config_overrides"] == {"num_epochs": 2}
    assert rc["auto_rag_retrieval"]["k"] == 5
    assert rc["pack_artifacts"]["source"] == {"pack_id": pid, "version": "1.0.0"}
    assert [r["id"] for r in rc["pack_artifacts"]["rubrics"]] == ["plan-quality"]
    assert rc["user_probe_packs"]["source"]["pack_id"] == pid
    assert project["selected_recipe"]["output_schema"] == SCHEMA
    assert project["base_model_name"] == "toy/model"
    pid_proj = project["id"]

    # surfaces read the copied artifacts back
    overrides = client.get(f"/api/projects/{pid_proj}/training-config-gaps/overrides").json()
    assert overrides["overrides"] == {"num_epochs": 2}
    tax = client.get(f"/api/projects/{pid_proj}/label-taxonomy").json()
    assert len(tax["taxonomy"]["entries"]) == 3
    sources = client.get(f"/api/projects/{pid_proj}/provenance/sources").json()["sources"]
    dp = [s for s in sources if s["source_kind"] == "domain_pack"]
    assert len(dp) == 1 and dp[0]["licence"] == "CC-BY-4.0" and dp[0]["usage"] == "rag"
    assert dp[0]["source_ref"] == f"{pid}@1.0.0"
    assert dp[0]["locator"] == f"{pid}@1.0.0:documents/matrix.md" and dp[0]["title"] == "Matrix"

    # export and compare with the source pack
    exp = client.get(f"/api/projects/{pid_proj}/domain-pack/export")
    assert exp.status_code == 200 and exp.headers["content-type"] == "application/zip"
    assert f'{pid}-1.0.1.zip' in exp.headers["content-disposition"]
    exported = exp.content
    v = svc.validate_pack_archive(exported)
    assert v["valid"], v["errors"]
    assert v["manifest"]["version"] == "1.0.1" and v["manifest"]["pack_id"] == pid
    assert v["summary"] == svc.validate_pack_archive(src)["summary"]

    with svc.extracted_archive(src) as (r1, _), svc.extracted_archive(exported) as (r2, _):
        (m1, p1), (m2, p2) = svc._validate(r1), svc._validate(r2)
        diff = svc.compute_diff((m1["manifest"], p1), (m2["manifest"], p2), version_from="1.0.0", version_to="1.0.1")
    assert not svc.diff_has_changes(diff), diff

    # preview of the export against the source pack: no section changes, only a version
    prev = post_zip(client, "/api/domain-packs/import/preview", exported).json()
    assert prev["valid"] and prev["existing"]["version"] == "1.0.0"
    assert not svc.diff_has_changes(prev["diff"]), prev["diff"]

    # include_documents controls
    none_zip = client.get(f"/api/projects/{pid_proj}/domain-pack/export", params={"include_documents": "none", "pack_id": "other-pack", "version": "3.0.0"})
    assert 'other-pack-3.0.0.zip' in none_zip.headers["content-disposition"]
    assert svc.validate_pack_archive(none_zip.content)["summary"]["documents"] == 0
    all_zip = client.get(f"/api/projects/{pid_proj}/domain-pack/export", params={"include_documents": "all"})
    assert svc.validate_pack_archive(all_zip.content)["summary"]["documents"] == 1


def test_export_plain_project_defaults_and_validation(client):
    name = f"Plain Project {uuid4().hex[:6]}"
    proj = client.post("/api/projects", json={"name": name, "description": ""}).json()
    res = client.get(f"/api/projects/{proj['id']}/domain-pack/export")
    assert res.status_code == 200
    v = svc.validate_pack_archive(res.content)
    assert v["valid"], v["errors"]
    assert v.manifest["version"] == "1.0.0" and v.manifest["pack_id"].startswith("plain-project-")
    assert client.get("/api/projects/999999/domain-pack/export").status_code == 404
    assert client.get(f"/api/projects/{proj['id']}/domain-pack/export", params={"version": "x"}).status_code == 400


def test_instantiate_failure_is_all_or_nothing(client, monkeypatch):
    _stub_ready()
    pid = new_id()
    post_zip(client, "/api/domain-packs/import", make_zip(toy_files(pid)))
    from app.services import gold_service

    async def boom(*a, **k):
        raise RuntimeError("gold import exploded")

    monkeypatch.setattr(gold_service, "import_qa_pairs", boom)
    name = f"fail-{pid}"
    with pytest.raises(RuntimeError):
        client.post(f"/api/domain-packs/{pid}/instantiate", json={"project_name": name})
    names = [p["name"] for p in client.get("/api/projects").json()["projects"]]
    assert name not in names
    assert client.post("/api/domain-packs/nope/instantiate", json={}).status_code == 404
    # a pack without content cannot be instantiated
    assert client.post("/api/domain-packs/general-pack-v1/instantiate", json={}).status_code == 404


# ── G12 fix-lane regressions ──────────────────────────────────────────────


def test_two_char_pack_id_is_rejected_at_validation():
    v = svc.validate_pack_archive(make_zip(toy_files("ab")))
    assert not v["valid"] and "bad_pack_id" in codes(v["errors"])
    assert len(svc.slugify_pack_id("a")) >= 3 and len(svc.slugify_pack_id("")) >= 3


def test_contract_field_limits_are_validated_before_import(client):
    v = svc.validate_pack_archive(make_zip(toy_files("toy-long", owner="o" * 200)))
    assert not v["valid"] and "invalid_contract" in codes(v["errors"])
    res = post_zip(client, "/api/domain-packs/import", make_zip(toy_files("toy-long", display_name="d" * 300)))
    assert res.status_code == 422
    import os
    from app.config import settings
    assert not os.path.exists(settings.DATA_DIR / "domain_packs" / "toy-long")


def test_corrupt_zip_member_is_a_bad_zip_issue_not_a_500(client):
    data = bytearray(make_zip(toy_files("toy-corrupt")))
    # flip a byte inside the stored payload of a member (CRC mismatch)
    idx = bytes(data).find(b"Injection is bad")
    assert idx > 0
    data[idx] ^= 0xFF
    v = svc.validate_pack_archive(bytes(data))
    assert not v["valid"] and "bad_zip" in codes(v["errors"])
    res = post_zip(client, "/api/domain-packs/import/preview", bytes(data))
    assert res.status_code == 200 and "bad_zip" in codes(res.json()["errors"])
    assert post_zip(client, "/api/domain-packs/import", bytes(data)).status_code == 422


def test_preview_and_instantiate_require_pack_write_access(client, monkeypatch):
    from app.api import domain_packs as api
    from app.models.auth import GlobalRole

    monkeypatch.setattr(api.settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(api, "get_request_principal", lambda request: None)
    assert post_zip(client, "/api/domain-packs/import/preview", make_zip(toy_files("toy-auth"))).status_code in (401, 403)
    assert client.post("/api/domain-packs/toy-auth/instantiate", json={}).status_code in (401, 403)
    viewer = SimpleNamespace(role=GlobalRole.VIEWER, user_id=1)
    monkeypatch.setattr(api, "get_request_principal", lambda request: viewer)
    with pytest.raises(Exception) as e:
        api._require_pack_write_access(SimpleNamespace())
    assert getattr(e.value, "status_code", None) == 403


def test_pack_document_keeps_nested_path_on_export_and_is_processed(client):
    files = toy_files(new_id())
    pid = files["pack.yaml"]
    manifest = yaml.safe_load(pid)
    manifest["documents"] = [{"path": "documents/guides/a.md", "title": "A", "licence": "CC-BY-4.0", "usage": "rag"}]
    files["pack.yaml"] = yaml.safe_dump(manifest)
    files["documents/guides/a.md"] = "# A\n\nHello nested.\n"
    del files["documents/matrix.md"]
    assert post_zip(client, "/api/domain-packs/import", make_zip(files)).status_code == 200
    res = client.post(f"/api/domain-packs/{manifest['pack_id']}/instantiate", json={})
    assert res.status_code == 201, res.text
    codes_ = codes(res.json()["summary"]["warnings"])
    assert "document_not_processed" not in codes_
    pid_proj = res.json()["project"]["id"]
    exported = client.get(f"/api/projects/{pid_proj}/domain-pack/export")
    with zipfile.ZipFile(io.BytesIO(exported.content)) as zf:
        assert "documents/guides/a.md" in zf.namelist()


def test_existing_version_dir_is_never_overwritten(client):
    from app.config import settings

    pid = new_id()
    d = settings.DATA_DIR / "domain_packs" / pid / "1.0.0"
    d.mkdir(parents=True)
    (d / "marker.txt").write_text("keep me")
    res = post_zip(client, "/api/domain-packs/import", make_zip(toy_files(pid)))
    assert res.status_code == 200
    assert res.json()["version"] != "1.0.0"
    assert (d / "marker.txt").read_text() == "keep me"
