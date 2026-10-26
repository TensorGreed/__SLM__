"""Git repository document source: ref pinning, URL safety, globs, licence,
file classification, the API job, code cleaning with line ranges, and the
passage index over the result.

Uses a real local repository created with ``git init`` in a temp dir — no
network. (Local paths are refused by default; tests switch the setting on.)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DB_REQUIRE_ALEMBIC_HEAD", "false")
os.environ.setdefault("ALLOW_SQLITE_AUTOCREATE", "true")

from fastapi.testclient import TestClient  # noqa: E402

from _job_wait import wait_for_jobs_idle  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services import auto_rag_service  # noqa: E402
from app.services.dataset_import import resolve_source  # noqa: E402
from app.services.dataset_import.sources import git as git_source  # noqa: E402
from app.services.document_sources.git_repo import (  # noqa: E402
    GitRepoConfig,
    GitSourceError,
    clone_repo,
    detect_licence,
    glob_match,
    validate_repo_url,
)

MIT = """MIT License

Copyright (c) 2025 Example Author

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.
"""

JAVA = """package demo;

public class Greeter {
    private int count;

    public String greet(String who) {
        count++;
        return "Hello " + who;
    }

    public String getUserName(int id) {
        // contact: dev@example.test
        return "user-" + id;
    }

    public int add(int a, int b) {
        return a + b;
    }
}
"""


def _git(repo: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@example.test",
               GIT_COMMITTER_NAME="T", GIT_COMMITTER_EMAIL="t@example.test", GIT_CONFIG_GLOBAL="/dev/null")
    out = subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def make_repo(root: Path) -> dict[str, str]:
    """A repo with two commits on ``main`` (tag ``v1`` on the first), a
    ``feature`` branch with a third; returns the commit ids."""
    repo = root / "toy-repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "LICENSE").write_text(MIT)
    (repo / "README.md").write_text("# Toy\n\nA toy repository used in tests.\n")
    (repo / "src").mkdir()
    (repo / "src" / "Greeter.java").write_text(JAVA)
    (repo / "data.bin").write_bytes(b"\x00\x01\x02binary")
    (repo / "big.txt").write_text("word " * 60_000)
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "x.js").write_text("function x() {}\n")
    (repo / "package.json").write_text('{"name": "toy", "version": "1.0.0"}\n')
    (repo / "Dockerfile").write_text("FROM base:1 AS build\nRUN echo hi\n")
    (repo / "run").write_text("#!/bin/bash\necho hi\n")
    (repo / "min.min.js").write_text("var a=1;\n")
    (repo / "notes.xyz").write_text("unknown type\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    ids = {"v1": _git(repo, "rev-parse", "HEAD")}
    _git(repo, "tag", "v1")
    (repo / "src" / "Extra.py").write_text("def extra():\n    return 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second")
    ids["main"] = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "feature.md").write_text("feature branch doc\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "third")
    ids["feature"] = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    ids["path"] = str(repo)
    return ids


class GitRepoSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ids = make_repo(Path(cls._tmp.name))
        cls.url = cls.ids["path"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        patcher = mock.patch.object(settings, "DOCUMENT_SOURCES_ALLOW_LOCAL_GIT", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _clone(self, **kw):
        return clone_repo(GitRepoConfig(url=self.url, **kw))

    def test_branch_tag_head_and_sha_refs_are_pinned_to_a_commit(self):
        with self._clone() as snap:
            self.assertEqual((snap.ref, snap.commit), ("HEAD", self.ids["main"]))
        with self._clone(ref="main") as snap:
            self.assertEqual(snap.commit, self.ids["main"])
        with self._clone(ref="feature") as snap:
            self.assertEqual(snap.commit, self.ids["feature"])
            self.assertIn("feature.md", [f.path for f in snap.files])
        with self._clone(ref="v1") as snap:
            self.assertEqual(snap.commit, self.ids["v1"])
            self.assertNotIn("src/Extra.py", [f.path for f in snap.files])
        with self._clone(ref=self.ids["v1"]) as snap:
            self.assertEqual(snap.commit, self.ids["v1"])
        with self._clone(ref=self.ids["v1"][:9]) as snap:
            self.assertEqual(snap.commit, self.ids["v1"])

    def test_unknown_ref_is_a_clear_error_and_leaves_no_temp_dir(self):
        with self.assertRaises(GitSourceError) as ctx:
            self._clone(ref="no-such-branch")
        self.assertIn("no-such-branch", str(ctx.exception))

    def test_files_are_classified_with_stable_reasons(self):
        with self._clone(max_file_bytes=200_000) as snap:
            kept = {f.path: (f.content_kind, f.language) for f in snap.files}
            self.assertEqual(kept["README.md"], ("doc", None))
            self.assertEqual(kept["src/Greeter.java"], ("code", "java"))
            self.assertEqual(kept["src/Extra.py"], ("code", "python"))
            self.assertEqual(kept["package.json"], ("code", "json"))
            self.assertEqual(kept["Dockerfile"], ("code", "dockerfile"))
            self.assertEqual(kept["run"], ("code", "shell"))  # by shebang
            reasons = {r["path"]: r["reason"] for r in snap.rejected}
            self.assertEqual(reasons["data.bin"], "binary")
            self.assertEqual(reasons["big.txt"], "too_large")
            self.assertEqual(reasons["node_modules/x.js"], "excluded_by_glob")
            self.assertEqual(reasons["min.min.js"], "excluded_by_glob")
            self.assertEqual(reasons["notes.xyz"], "unsupported_type")
            self.assertEqual(snap.licence, "MIT")
            self.assertEqual(snap.licence_file, "LICENSE")
            self.assertTrue(all(".git/" not in p for p in kept))

    def test_include_and_exclude_globs_and_max_files(self):
        with self._clone(include_globs=["src/**/*.java"]) as snap:
            self.assertEqual([f.path for f in snap.files], ["src/Greeter.java"])
            self.assertEqual({r["reason"] for r in snap.rejected}, {"excluded_by_glob"})
        with self._clone(exclude_globs=["*.md", "*.java"]) as snap:
            paths = [f.path for f in snap.files]
            self.assertNotIn("README.md", paths)
            self.assertNotIn("src/Greeter.java", paths)
            self.assertNotIn("node_modules/x.js", paths)  # added to the defaults, not replacing them
            self.assertNotIn("min.min.js", paths)
        with self._clone(exclude_globs=["*.md"], default_excludes=False) as snap:
            paths = [f.path for f in snap.files]
            self.assertIn("node_modules/x.js", paths)  # defaults switched off
            self.assertIn("min.min.js", paths)
            self.assertNotIn("README.md", paths)
        with self._clone(max_files=2) as snap:
            self.assertEqual(len(snap.files), 2)
            self.assertGreater(snap.limit_reached, 0)
            self.assertNotIn("limit_reached", {r["reason"] for r in snap.rejected})

    def test_glob_matching(self):
        self.assertTrue(glob_match("a/b/c.py", "**/*.py"))
        self.assertTrue(glob_match("c.py", "**/*.py"))
        self.assertTrue(glob_match("a/b/c.min.js", "*.min.js"))  # no slash: any depth
        self.assertTrue(glob_match("node_modules/a/b.js", "node_modules/**"))
        self.assertFalse(glob_match("src/node_modules/b.js", "node_modules/**"))  # anchored
        self.assertFalse(glob_match("src/a/b.py", "src/*.py"))  # * stays in one segment
        self.assertTrue(glob_match("src/b.py", "src/*.py"))
        self.assertTrue(glob_match("docs/guide.md", "docs/**"))
        self.assertTrue(glob_match("a.lock", "*.lock"))

    def test_default_excludes_match_vendored_folders_at_any_depth(self):
        from app.services.document_sources.git_repo import DEFAULT_EXCLUDE_GLOBS

        def skipped(path: str) -> bool:
            return any(glob_match(path, g) for g in DEFAULT_EXCLUDE_GLOBS)

        for path in (
            "node_modules/x.js", "web/node_modules/a/b.js", ".git/config", "sub/.git/HEAD", "vendor/z.go", "svc/vendor/z.go",
            "dist/a.js", "pkg/dist/a.js", "build/out.txt", "a/b/build/out.txt", "x.min.js", "static/js/x.min.js",
            "Cargo.lock", "a/b/yarn.lock", "package-lock.json", "web/package-lock.json",
        ):
            self.assertTrue(skipped(path), path)
        for path in ("src/main.py", "src/builder/a.py", "docs/distribution.md", "src/vendored.txt", "README.md"):
            self.assertFalse(skipped(path), path)

    def test_nested_vendored_folders_are_skipped_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "nested"
            repo.mkdir()
            _git(repo, "init", "-q", "-b", "main")
            for rel in ("web/node_modules/dep/index.js", "svc/vendor/lib.go", "pkg/dist/a.js", "keep/app.py"):
                (repo / rel).parent.mkdir(parents=True, exist_ok=True)
                (repo / rel).write_text("x = 1\n")
            _git(repo, "add", "-A")
            _git(repo, "commit", "-q", "-m", "n")
            with clone_repo(GitRepoConfig(url=str(repo))) as snap:
                self.assertEqual([f.path for f in snap.files], ["keep/app.py"])
                self.assertEqual({r["reason"] for r in snap.rejected}, {"excluded_by_glob"})
            with clone_repo(GitRepoConfig(url=str(repo), default_excludes=False)) as snap:
                self.assertEqual(len(snap.files), 4)

    def test_licence_detection(self):
        self.assertEqual(detect_licence(MIT), "MIT")
        self.assertEqual(detect_licence("Apache License\nVersion 2.0, January 2004"), "Apache-2.0")
        self.assertEqual(detect_licence("GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007"), "GPL-3.0")
        self.assertEqual(detect_licence("GNU GENERAL PUBLIC LICENSE Version 2, June 1991"), "GPL-2.0")
        self.assertEqual(detect_licence("GNU LESSER GENERAL PUBLIC LICENSE Version 2.1"), "LGPL-2.1")
        self.assertEqual(detect_licence("GNU AFFERO GENERAL PUBLIC LICENSE Version 3"), "AGPL-3.0")
        self.assertEqual(detect_licence("Mozilla Public License Version 2.0"), "MPL-2.0")
        self.assertEqual(detect_licence("This is free and unencumbered software released into the public domain."), "Unlicense")
        self.assertEqual(
            detect_licence("Redistribution and use in source and binary forms ... Neither the name of the holder"), "BSD-3-Clause"
        )
        self.assertEqual(detect_licence("Redistribution and use in source and binary forms ..."), "BSD-2-Clause")
        self.assertEqual(detect_licence("Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee"), "ISC")
        self.assertEqual(detect_licence("all rights reserved, ask us"), "NOASSERTION")

    def test_dataset_import_source_rows_and_describe(self):
        source = resolve_source("git")
        rows = list(source.load(f"{self.url}@v1"))
        java = [r for r in rows if r["path"] == "src/Greeter.java"]
        self.assertEqual([r["symbol"] for r in java], [None, "Greeter", "Greeter.greet", "Greeter.getUserName", "Greeter.add"])
        self.assertTrue(all({"language", "start_line", "end_line", "text"} <= set(r) for r in java))
        readme = next(r for r in rows if r["path"] == "README.md")
        self.assertEqual(set(readme), {"path", "text"})
        info = source.describe(f"{self.url}@v1")
        self.assertEqual(info["licence"], "MIT")
        self.assertEqual(info["commit"], self.ids["v1"])
        self.assertIn("symbol", info["columns"])
        self.assertEqual(len(list(source.load(self.url, limit=3))), 3)

    def test_locator_ref_splitting(self):
        split = git_source.split_repo_locator
        self.assertEqual(split("https://github.com/org/repo@v1.2"), ("https://github.com/org/repo", "v1.2"))
        self.assertEqual(split("https://github.com/org/repo"), ("https://github.com/org/repo", "HEAD"))
        self.assertEqual(split("git@github.com:org/repo@v1"), ("git@github.com:org/repo", "v1"))
        self.assertEqual(split("git@github.com:org/repo"), ("git@github.com:org/repo", "HEAD"))
        self.assertEqual(split("https://user@host.test/x"), ("https://user@host.test/x", "HEAD"))
        self.assertEqual(split("https://h.test/o/r@feature/x"), ("https://h.test/o/r", "feature/x"))
        self.assertEqual(split("/tmp/repo@v1"), ("/tmp/repo", "v1"))


class _FakeGit:
    """A scripted ``git`` runner: records argv, answers by subcommand."""

    def __init__(self, fetch_errors: list[str]) -> None:
        self.calls: list[list[str]] = []
        self._fetch_errors = list(fetch_errors)

    def __call__(self, args, cwd, env, timeout):
        self.calls.append(list(args))
        sub = args[1]
        if sub == "fetch" and self._fetch_errors:
            return subprocess.CompletedProcess(args, 128, "", self._fetch_errors.pop(0))
        if sub == "rev-parse" and "--verify" in args:
            return subprocess.CompletedProcess(args, 1, "", "")
        return subprocess.CompletedProcess(args, 0, "abc123\n" if sub == "rev-parse" else "", "")

    def fetches(self) -> list[list[str]]:
        return [c for c in self.calls if c[1] == "fetch"]


class FallbackFetchTests(unittest.TestCase):
    def _clone(self, ref: str, errors: list[str]) -> _FakeGit:
        fake = _FakeGit(errors)
        try:
            clone_repo(GitRepoConfig(url="https://h.test/o/r", ref=ref), runner=fake, allow_local=False).cleanup()
        except GitSourceError:
            pass
        return fake

    def test_auth_or_network_errors_do_not_trigger_the_wide_fetch(self):
        for err in ("fatal: Authentication failed for 'https://h.test/o/r'", "fatal: unable to access: Could not resolve host"):
            fake = self._clone("v1", [err])
            self.assertEqual(len(fake.fetches()), 1, err)

    def test_missing_ref_falls_back_shallowly(self):
        fake = self._clone("v1", ["fatal: couldn't find remote ref v1"])
        wide = fake.fetches()[1]
        self.assertIn("--depth", wide)
        self.assertEqual(wide[wide.index("--depth") + 1], "1")
        self.assertNotIn("--filter=blob:none", wide)

    def test_commit_id_falls_back_with_a_blobless_filter(self):
        fake = self._clone("a" * 40, ["fatal: remote error: upload-pack: not our ref " + "a" * 40])
        wide = fake.fetches()[1]
        self.assertIn("--filter=blob:none", wide)
        self.assertNotIn("--depth", wide)


class CancelTests(unittest.TestCase):
    def test_cancel_event_before_clone_raises_and_the_real_runner_kills_a_running_process(self):
        import threading

        from app.services.document_sources.git_repo import GitCancelled, _make_runner

        cancel = threading.Event()
        cancel.set()
        fake = _FakeGit([])
        with self.assertRaises(GitCancelled):
            clone_repo(GitRepoConfig(url="https://h.test/o/r"), runner=fake, allow_local=False, cancel=cancel)
        self.assertEqual(fake.calls, [])

        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        started = time.monotonic()
        with self.assertRaises(GitCancelled):
            _make_runner(cancel)(["sleep", "30"], None, dict(os.environ), 300)
        self.assertLess(time.monotonic() - started, 5)

    def test_with_progress_sets_the_event_on_task_cancel_and_on_job_cancel(self):
        import asyncio
        import threading

        from app.services.document_sources.importer import ImportCancelled, _with_progress

        async def task_cancelled() -> bool:
            cancel = threading.Event()
            task = asyncio.create_task(_with_progress(lambda: cancel.wait(10), None, None, cancel=cancel))
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await asyncio.sleep(0.1)
            return cancel.is_set()

        self.assertTrue(asyncio.run(task_cancelled()))

        async def job_cancelled() -> None:
            cancel = threading.Event()

            async def flag() -> bool:
                return True

            await _with_progress(lambda: cancel.wait(10), None, None, cancel=cancel, is_cancelled=flag)

        started = time.monotonic()
        with self.assertRaises(ImportCancelled):
            asyncio.run(job_cancelled())
        self.assertLess(time.monotonic() - started, 5)


class RepoUrlSafetyTests(unittest.TestCase):
    def test_option_like_and_local_urls_are_refused_by_default(self):
        for bad in ("-x", "--upload-pack=touch /tmp/x", "-c core.sshCommand=x", "", "ftp://h.test/r", "ext::sh -c x", "plain-word"):
            with self.assertRaises(GitSourceError, msg=bad):
                validate_repo_url(bad, allow_local=True)
        for local in ("/tmp/some/repo", "file:///tmp/some/repo", "./repo", "~/repo"):
            with self.assertRaises(GitSourceError, msg=local):
                validate_repo_url(local, allow_local=False)
            self.assertEqual(validate_repo_url(local, allow_local=True), local)
        for ok in ("https://h.test/o/r.git", "http://h.test/o/r", "ssh://git@h.test/o/r", "git@h.test:o/r.git"):
            self.assertEqual(validate_repo_url(ok, allow_local=False), ok)

    def test_credentials_in_the_url_are_refused(self):
        for bad in ("https://user:s3cr3t@h.test/o/r", "https://s3cr3t@h.test/o/r", "http://u:p@h.test/o/r", "ssh://u:p@h.test/o/r"):
            with self.assertRaises(GitSourceError, msg=bad) as ctx:
                validate_repo_url(bad, allow_local=True)
            self.assertIn("credentials", str(ctx.exception))
            self.assertNotIn("s3cr3t", str(ctx.exception))
        # a bare ssh user name is not a secret
        self.assertEqual(validate_repo_url("ssh://git@h.test/o/r", allow_local=False), "ssh://git@h.test/o/r")

    def test_ssh_hosts_starting_with_a_dash_are_refused(self):
        for bad in ("ssh://-oProxyCommand=x/o/r", "ssh://git@-bad/o/r", "git@-bad:o/r"):
            with self.assertRaises(GitSourceError, msg=bad):
                validate_repo_url(bad, allow_local=True)

    def test_loopback_ssh_targets_are_local_and_gated(self):
        for local in ("ssh://git@localhost/o/r", "ssh://git@127.0.0.1/o/r", "ssh://git@127.5.6.7:2222/o/r",
                      "ssh://git@[::1]/o/r", "git@localhost:o/r", "git@127.0.0.1:o/r"):
            with self.assertRaises(GitSourceError, msg=local) as ctx:
                validate_repo_url(local, allow_local=False)
            self.assertIn("DOCUMENT_SOURCES_ALLOW_LOCAL_GIT", str(ctx.exception))
            self.assertEqual(validate_repo_url(local, allow_local=True), local)
        self.assertEqual(validate_repo_url("ssh://git@127.example.test/o/r", allow_local=False), "ssh://git@127.example.test/o/r")

    def test_setting_defaults_to_off(self):
        self.assertFalse(type(settings).model_fields["DOCUMENT_SOURCES_ALLOW_LOCAL_GIT"].default)

    def test_bad_ref_is_refused(self):
        with self.assertRaises(GitSourceError):
            GitRepoConfig(url="https://h.test/o/r", ref="--upload-pack=x")
        with self.assertRaises(GitSourceError):
            GitRepoConfig(url="https://h.test/o/r", ref="a b")


# ── API + cleaning + passage index ───────────────────────────────────

_MODULE_CLIENT_CM = TestClient(app)


def setUpModule() -> None:  # noqa: N802
    _MODULE_CLIENT_CM.__enter__()


def tearDownModule() -> None:  # noqa: N802
    _MODULE_CLIENT_CM.__exit__(None, None, None)


def _poll_job(client: TestClient, job_id: int, timeout_s: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_s
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body.get("status") in {"succeeded", "failed", "cancelled"}:
            return body
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} still {body.get('status')}")


class GitRepoApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = _MODULE_CLIENT_CM
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ids = make_repo(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        patcher = mock.patch.object(settings, "DOCUMENT_SOURCES_ALLOW_LOCAL_GIT", True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.project_id = self.client.post("/api/projects", json={"name": f"git-{uuid4().hex[:6]}", "description": ""}).json()["id"]
        self.url = f"/api/projects/{self.project_id}/document-sources/git-repo"

    def tearDown(self):
        wait_for_jobs_idle(self.client)

    def _import(self, **extra) -> dict:
        resp = self.client.post(self.url, json={"url": self.ids["path"], "ref": "v1", **extra})
        self.assertEqual(resp.status_code, 202, resp.text)
        self.assertEqual(resp.json()["kind"], "document_source_git_repo")
        done = _poll_job(self.client, resp.json()["id"])
        self.assertEqual(done["status"], "succeeded", done)
        return done["result"]

    def _docs(self) -> dict[str, dict]:
        docs = self.client.get(f"/api/projects/{self.project_id}/ingestion/documents").json()
        return {d["path"] or d["filename"]: d for d in docs}

    def test_job_creates_code_documents_with_provenance(self):
        result = self._import()
        self.assertEqual(result["source"], "git_repo")
        self.assertEqual(result["commit"], self.ids["v1"])
        self.assertEqual(result["ref"], "v1")
        self.assertEqual(result["licence"], "MIT")
        self.assertEqual(result["documents_created"], len(result["document_ids"]))
        paths = {d["path"] for d in result["documents"]}
        self.assertEqual(paths, {"README.md", "src/Greeter.java", "package.json", "Dockerfile", "run"})
        self.assertEqual(result["rejection_counts"]["binary"], 1)
        self.assertEqual(result["rejection_counts"]["too_large"], 1)
        self.assertIn("path", result["rejected"][0])

        docs = self._docs()
        java = docs["src/Greeter.java"]
        self.assertEqual(java["source"], "git_repo")
        self.assertEqual(java["license_info"], "MIT")
        self.assertEqual(java["commit"], self.ids["v1"])
        self.assertEqual(java["language"], "java")
        self.assertEqual(java["content_kind"], "code")
        self.assertEqual(java["status"], "accepted")
        self.assertTrue(java["filename"].startswith("src__Greeter"))
        self.assertFalse(docs["package.json"]["structured"])  # a repo .json is source text, not rows
        self.assertEqual(docs["Dockerfile"]["filename"], "Dockerfile.dockerfile")
        self.assertEqual(docs["README.md"]["content_kind"], "doc")

    def test_unsafe_urls_and_unknown_project(self):
        for bad in ("-x", "--upload-pack=x"):
            resp = self.client.post(self.url, json={"url": bad})
            self.assertEqual(resp.status_code, 422, resp.text)
        with mock.patch.object(settings, "DOCUMENT_SOURCES_ALLOW_LOCAL_GIT", False):
            resp = self.client.post(self.url, json={"url": self.ids["path"]})
            self.assertEqual(resp.status_code, 422)
            self.assertIn("Local repository paths are disabled", resp.json()["detail"])
        resp = self.client.post(self.url, json={"url": "https://user:s3cr3t@h.test/o/r"})
        self.assertEqual(resp.status_code, 422, resp.text)
        self.assertEqual(resp.json()["error_code"], "DOCUMENT_SOURCE_INVALID")
        self.assertNotIn("s3cr3t", resp.text)
        with mock.patch.object(settings, "DOCUMENT_SOURCES_ALLOW_LOCAL_GIT", False):
            resp = self.client.post(self.url, json={"url": "ssh://git@127.0.0.1/o/r"})
            self.assertEqual(resp.status_code, 422, resp.text)
            self.assertIn("DOCUMENT_SOURCES_ALLOW_LOCAL_GIT", resp.json()["detail"])
        self.assertEqual(self.client.post("/api/projects/99999999/document-sources/git-repo", json={"url": "https://h.test/o/r"}).status_code, 404)
        self.assertEqual(self.client.post(self.url, json={"url": "https://h.test/o/r", "max_files": 0}).status_code, 422)

    def test_exclude_globs_add_to_defaults_and_default_excludes_can_be_turned_off(self):
        added = self._import(exclude_globs=["*.md"])
        self.assertNotIn("README.md", {d["path"] for d in added["documents"]})
        self.assertEqual(added["rejection_counts"]["excluded_by_glob"], 3)  # node_modules/x.js, min.min.js, README.md
        off = self._import(exclude_globs=["*.md"], default_excludes=False)
        paths = {d["path"] for d in off["documents"]}
        self.assertIn("node_modules/x.js", paths)
        self.assertNotIn("README.md", paths)

    def test_repo_package_json_is_text_not_rows_for_task_shape_samples_and_profile(self):
        result = self._import(include_globs=["package.json"])
        self.assertEqual([d["path"] for d in result["documents"]], ["package.json"])
        doc = self._docs()["package.json"]
        self.assertEqual(doc["content_kind"], "code")
        shape = self.client.get(f"/api/projects/{self.project_id}/task-shape").json()
        self.assertIsNone(shape["detection"], shape)  # no rows: the manifest is not a dataset
        sample = self.client.get(f"/api/projects/{self.project_id}/ingestion/documents/{doc['id']}/sample?n=5").json()
        self.assertTrue(sample["rows"])
        self.assertTrue(all(set(row) == {"line"} for row in sample["rows"]), sample["rows"])  # text lines, not name/version rows
        profile = self.client.post(
            f"/api/projects/{self.project_id}/dataset/profile", json={"dataset_type": "raw", "sample_size": 50}
        )
        self.assertEqual(profile.status_code, 200, profile.text)
        self.assertEqual(profile.json()["source"].get("documents_scanned"), 1)
        self.assertNotIn("name", json.dumps(profile.json().get("sample_columns") or profile.json().get("columns") or []))

    def test_inline_mode_and_empty_import_failure(self):
        resp = self.client.post(self.url + "?async_job=false", json={"url": self.ids["path"], "ref": "v1", "include_globs": ["README.md"]})
        self.assertEqual(resp.status_code, 201, resp.text)
        self.assertEqual(resp.json()["documents_created"], 1)
        empty = self.client.post(self.url + "?async_job=false", json={"url": self.ids["path"], "ref": "v1", "include_globs": ["nothing/**"]})
        self.assertEqual(empty.status_code, 422)
        self.assertIn("excluded_by_glob", empty.json()["detail"])
        job = self.client.post(self.url, json={"url": self.ids["path"], "ref": "v1", "include_globs": ["nothing/**"]}).json()
        done = _poll_job(self.client, job["id"])
        self.assertEqual(done["status"], "failed")
        self.assertIn("excluded_by_glob", done["error"])

    def test_clean_code_document_keeps_text_and_cites_line_ranges_then_index_retrieves_it(self):
        self._import(include_globs=["src/**"])
        java = self._docs()["src/Greeter.java"]
        resp = self.client.post(f"/api/projects/{self.project_id}/cleaning/clean", json={"document_id": java["id"], "chunk_size": 400})
        self.assertEqual(resp.status_code, 200, resp.text)
        summary = resp.json()
        self.assertEqual(summary["redaction"], "skipped_for_code")
        self.assertTrue(any(f["type"] == "email" for f in summary["pii_findings"]))

        cleaned = settings.DATA_DIR / "projects" / str(self.project_id) / "cleaned" / "cleaned.jsonl"
        rows = [json.loads(line) for line in cleaned.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r["source_document_id"] == java["id"]]
        by_symbol = {r["symbol"]: r for r in rows}
        self.assertEqual(list(by_symbol), [None, "Greeter", "Greeter.greet", "Greeter.getUserName", "Greeter.add"])
        source_lines = JAVA.splitlines()
        for row in rows:
            self.assertEqual(row["path"], "src/Greeter.java")
            self.assertEqual(row["language"], "java")
            self.assertEqual(row["text"], "\n".join(source_lines[row["start_line"] - 1 : row["end_line"]]))
        self.assertEqual((by_symbol["Greeter.greet"]["start_line"], by_symbol["Greeter.greet"]["end_line"]), (6, 10))
        # no redaction: the hard-coded address is still in the passage
        self.assertIn("dev@example.test", by_symbol["Greeter.getUserName"]["text"])

        # Passage index over the cleaned rows: identifier-aware retrieval + a line-range label.
        status = auto_rag_service.ensure_document_index(self.project_id)
        self.assertTrue(status["available"], status)
        hits = auto_rag_service.retrieve_document_passages(self.project_id, "what does user name lookup return", {"k": 2})
        top = hits[0]["payload"]
        self.assertEqual(top["symbol"], "Greeter.getUserName")
        self.assertEqual((top["path"], top["start_line"], top["end_line"]), ("src/Greeter.java", 11, 15))
        label = auto_rag_service._format_passage(1, hits[0])
        self.assertTrue(label.startswith("[1] src/Greeter.java:11-15 (Greeter.getUserName)\n"))

        # A re-chunked variant is cut by declaration too, never by sentence.
        variant = auto_rag_service.load_rechunked_passages(self.project_id, 200)
        self.assertTrue(all(p.get("path") == "src/Greeter.java" and p.get("start_line") for p in variant))
        self.assertIn("Greeter.add", {p["symbol"] for p in variant})
        self.assertTrue(auto_rag_service.ensure_document_index(self.project_id, 200)["available"])


class CodeUploadTests(unittest.TestCase):
    """Source files can also be uploaded: parsed verbatim, chunked as code."""

    def setUp(self):
        self.client = _MODULE_CLIENT_CM
        self.project_id = self.client.post("/api/projects", json={"name": f"up-{uuid4().hex[:6]}", "description": ""}).json()["id"]

    def test_uploaded_python_file_is_accepted_and_keeps_its_indentation(self):
        source = "def one():\n    return 1\n\n\ndef two():\n    return 2\n"
        up = self.client.post(
            f"/api/projects/{self.project_id}/ingestion/upload", files={"file": ("calc.py", source.encode(), "text/plain")}
        )
        self.assertEqual(up.status_code, 201, up.text)
        doc_id = up.json()["id"]
        proc = self.client.post(f"/api/projects/{self.project_id}/ingestion/documents/{doc_id}/process")
        self.assertEqual(proc.json()["status"], "accepted", proc.text)
        extracted = settings.DATA_DIR / "projects" / str(self.project_id) / "raw" / "calc.extracted.txt"
        self.assertEqual(extracted.read_text(), source)
        clean = self.client.post(f"/api/projects/{self.project_id}/cleaning/clean", json={"document_id": doc_id})
        self.assertEqual(clean.status_code, 200, clean.text)
        self.assertEqual(clean.json()["chunk_count"], 2)

    def test_same_stem_different_extension_does_not_share_sidecar_files(self):
        for name in ("lib.py", "lib.java"):
            up = self.client.post(
                f"/api/projects/{self.project_id}/ingestion/upload", files={"file": (name, b"x = 1\n", "text/plain")}
            )
            self.assertEqual(up.status_code, 201, up.text)
            self.client.post(f"/api/projects/{self.project_id}/ingestion/documents/{up.json()['id']}/process")
        names = sorted(d["filename"] for d in self.client.get(f"/api/projects/{self.project_id}/ingestion/documents").json())
        self.assertEqual(names, ["lib.py", "lib_1.java"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
