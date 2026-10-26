"""The provenance-ledger migration is additive and idempotent: it creates
``data_sources`` + the new ``raw_documents`` / ``training_manifests`` columns,
and running it again (or on a database that already has them) is a no-op."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "20261026_0050_data_sources_provenance.py"


def _load():
    spec = importlib.util.spec_from_file_location("migration_0050", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_schema(conn) -> None:
    conn.execute(sa.text("CREATE TABLE projects (id INTEGER PRIMARY KEY)"))
    conn.execute(sa.text("CREATE TABLE raw_documents (id INTEGER PRIMARY KEY, filename VARCHAR(512))"))
    conn.execute(sa.text("CREATE TABLE training_manifests (id INTEGER PRIMARY KEY)"))


def _run(conn, module) -> None:
    ctx = MigrationContext.configure(conn)
    with Operations.context(ctx):
        module.upgrade()


class OrmMatchesMigrationTests(unittest.TestCase):
    def test_nullable_columns_agree_with_the_migration(self):
        """The migration leaves these NULLable, so the ORM must not claim otherwise."""
        from app.models.data_source import DataSource
        from app.models.training_manifest import TrainingManifest

        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            engine = sa.create_engine(f"sqlite:///{tmp}/m.db")
            with engine.begin() as conn:
                _base_schema(conn)
                _run(conn, module)
                inspector = sa.inspect(conn)
                migrated = {c["name"]: c["nullable"] for c in inspector.get_columns("data_sources")}
                prov = {c["name"]: c["nullable"] for c in inspector.get_columns("training_manifests")}
        for name in ("created_at", "updated_at"):
            self.assertTrue(migrated[name])
            self.assertTrue(DataSource.__table__.c[name].nullable, name)
        self.assertTrue(prov["data_provenance"])
        self.assertTrue(TrainingManifest.__table__.c.data_provenance.nullable)


class ProvenanceMigrationTests(unittest.TestCase):
    def test_upgrade_creates_the_ledger_and_is_idempotent(self):
        module = _load()
        self.assertEqual(module.down_revision, "20261015_0049")
        with tempfile.TemporaryDirectory() as tmp:
            engine = sa.create_engine(f"sqlite:///{Path(tmp) / 'scratch.db'}")
            with engine.begin() as conn:
                _base_schema(conn)
                conn.execute(sa.text("INSERT INTO raw_documents (id, filename) VALUES (1, 'a.txt')"))
                _run(conn, module)
                _run(conn, module)  # second run: nothing to do, nothing raised
            insp = sa.inspect(engine)
            self.assertIn("data_sources", insp.get_table_names())
            source_cols = {c["name"] for c in insp.get_columns("data_sources")}
            self.assertTrue({
                "id", "project_id", "source_kind", "title", "locator", "source_ref", "licence", "licence_verified",
                "licence_verified_by", "licence_verified_at", "retrieved_at", "usage", "metadata", "created_at", "updated_at",
            } <= source_cols)
            doc_cols = {c["name"] for c in insp.get_columns("raw_documents")}
            self.assertTrue({"data_source_id", "source_url", "source_ref", "retrieved_at"} <= doc_cols)
            self.assertIn("data_provenance", {c["name"] for c in insp.get_columns("training_manifests")})
            index_names = {ix["name"] for ix in insp.get_indexes("raw_documents")}
            self.assertIn("ix_raw_documents_data_source_id", index_names)
            self.assertIn("ix_data_sources_project_id", {ix["name"] for ix in insp.get_indexes("data_sources")})
            with engine.connect() as conn:  # existing rows survive
                self.assertEqual(conn.execute(sa.text("SELECT filename FROM raw_documents")).scalar(), "a.txt")
            engine.dispose()

    def test_a_database_that_already_has_the_columns_is_left_alone(self):
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            engine = sa.create_engine(f"sqlite:///{Path(tmp) / 'scratch.db'}")
            with engine.begin() as conn:
                _base_schema(conn)
                conn.execute(sa.text("ALTER TABLE raw_documents ADD COLUMN data_source_id INTEGER"))
                conn.execute(sa.text("ALTER TABLE raw_documents ADD COLUMN source_url VARCHAR(1024)"))
                conn.execute(sa.text("ALTER TABLE training_manifests ADD COLUMN data_provenance JSON"))
                _run(conn, module)
            self.assertIn("retrieved_at", {c["name"] for c in sa.inspect(engine).get_columns("raw_documents")})
            engine.dispose()

    def test_orm_models_match_the_migration_columns(self):
        from app.models.data_source import DataSource
        from app.models.dataset import RawDocument
        from app.models.training_manifest import TrainingManifest

        self.assertEqual(DataSource.__table__.c.usage.default.arg, "any")
        self.assertTrue(RawDocument.__table__.c.data_source_id.nullable)
        self.assertIn("data_provenance", TrainingManifest.__table__.c)


if __name__ == "__main__":
    unittest.main()
