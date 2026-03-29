"""Phase 48 regression: SQLite legacy project schema repair for beginner-mode columns."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, text

from app.database import CRITICAL_COLUMN_REQUIREMENTS, _list_missing_columns, _repair_sqlite_schema_drift


class Phase48SchemaCompatibilityTests(unittest.TestCase):
    def test_sqlite_legacy_projects_table_is_repaired(self):
        with tempfile.TemporaryDirectory(prefix="phase48_schema_") as tmp_dir:
            db_path = Path(tmp_dir) / "legacy.db"
            engine = create_engine(f"sqlite:///{db_path.as_posix()}", future=True)
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        CREATE TABLE projects (
                            id INTEGER PRIMARY KEY,
                            name VARCHAR(255) NOT NULL,
                            description TEXT,
                            status VARCHAR(32) NOT NULL,
                            pipeline_stage VARCHAR(32) NOT NULL,
                            base_model_name VARCHAR(255),
                            domain_pack_id INTEGER,
                            domain_profile_id INTEGER,
                            training_preferred_plan_profile VARCHAR(64),
                            target_profile_id VARCHAR(64),
                            evaluation_preferred_pack_id VARCHAR(64),
                            dataset_adapter_preset VARCHAR(128),
                            gate_policy JSON,
                            budget_settings JSON,
                            created_at DATETIME NOT NULL,
                            updated_at DATETIME NOT NULL
                        )
                        """
                    )
                )

                missing_before = _list_missing_columns(conn, CRITICAL_COLUMN_REQUIREMENTS)
                self.assertEqual(
                    set(missing_before),
                    {"projects.beginner_mode", "projects.active_domain_blueprint_version"},
                )

                applied = _repair_sqlite_schema_drift(conn)
                self.assertEqual(
                    set(applied),
                    {"projects.beginner_mode", "projects.active_domain_blueprint_version"},
                )

                missing_after = _list_missing_columns(conn, CRITICAL_COLUMN_REQUIREMENTS)
                self.assertEqual(missing_after, [])


if __name__ == "__main__":
    unittest.main()
