"""Tests for SQLite catalog database."""

import sqlite3
from pathlib import Path

import pytest

from vam_tools.core.database import CatalogDatabase


class TestCatalogDatabase:
    """Tests for CatalogDatabase class."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        """Create temporary database path."""
        return tmp_path / "test_catalog"

    @pytest.fixture
    def db(self, db_path: Path) -> CatalogDatabase:
        """Create database instance."""
        return CatalogDatabase(db_path)

    def test_initialization(self, db: CatalogDatabase, db_path: Path) -> None:
        """Test database initialization creates directory."""
        assert db.catalog_path == db_path
        assert db.db_path == db_path / "catalog.db"
        assert db_path.exists()
        assert (db_path / "backups").exists()

    def test_connect(self, db: CatalogDatabase) -> None:
        """Test database connection."""
        db.connect()

        assert db.connection is not None
        assert db.db_path.exists()

        db.close()
        assert db.connection is None

    def test_context_manager(self, db: CatalogDatabase) -> None:
        """Test database context manager."""
        with db as database:
            assert database.connection is not None

        assert db.connection is None

    def test_initialize_schema(self, db: CatalogDatabase) -> None:
        """Test schema initialization."""
        db.connect()
        db.initialize()

        # Check that tables exist
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            "schema_version", "images", "tags", "image_tags",
            "duplicate_groups", "duplicate_group_images",
            "burst_groups", "burst_group_images",
            "review_queue", "problematic_files",
            "catalog_config", "statistics", "performance_snapshots"
        }

        assert expected_tables.issubset(tables)

    def test_schema_version(self, db: CatalogDatabase) -> None:
        """Test schema version tracking."""
        db.connect()
        db.initialize()

        version = db.get_schema_version()
        assert version == 1

    def test_transaction_commit(self, db: CatalogDatabase) -> None:
        """Test transaction commits on success."""
        db.connect()
        db.initialize()

        with db.transaction():
            db.execute(
                "INSERT INTO tags (name, category, created_at) "
                "VALUES (?, ?, datetime('now'))",
                ("test_tag", "subject")
            )

        # Verify committed
        cursor = db.execute("SELECT COUNT(*) FROM tags")
        count = cursor.fetchone()[0]
        assert count == 1

    def test_transaction_rollback(self, db: CatalogDatabase) -> None:
        """Test transaction rolls back on error."""
        db.connect()
        db.initialize()

        try:
            with db.transaction():
                db.execute(
                    "INSERT INTO tags (name, category, created_at) "
                    "VALUES (?, ?, datetime('now'))",
                    ("test_tag", "subject")
                )
                # Force error with duplicate
                db.execute(
                    "INSERT INTO tags (name, category, created_at) "
                    "VALUES (?, ?, datetime('now'))",
                    ("test_tag", "subject")
                )
        except sqlite3.IntegrityError:
            pass

        # Verify rolled back
        cursor = db.execute("SELECT COUNT(*) FROM tags")
        count = cursor.fetchone()[0]
        assert count == 0

    def test_execute_query(self, db: CatalogDatabase) -> None:
        """Test executing SQL queries."""
        db.connect()
        db.initialize()

        db.execute(
            "INSERT INTO tags (name, category, created_at) "
            "VALUES (?, ?, datetime('now'))",
            ("test_tag", "subject")
        )

        cursor = db.execute("SELECT * FROM tags WHERE name = ?", ("test_tag",))
        row = cursor.fetchone()

        assert row is not None
        assert row["name"] == "test_tag"
        assert row["category"] == "subject"

    def test_executemany(self, db: CatalogDatabase) -> None:
        """Test executing query with multiple parameters."""
        db.connect()
        db.initialize()

        tags = [
            ("dogs", "subject"),
            ("cats", "subject"),
            ("sunset", "lighting")
        ]

        db.executemany(
            "INSERT INTO tags (name, category, created_at) "
            "VALUES (?, ?, datetime('now'))",
            tags
        )

        cursor = db.execute("SELECT COUNT(*) FROM tags")
        count = cursor.fetchone()[0]
        assert count == 3

    def test_foreign_key_constraints(self, db: CatalogDatabase) -> None:
        """Test foreign key constraints are enforced."""
        db.connect()
        db.initialize()

        # Try to insert image_tag without valid tag_id
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO image_tags "
                "(image_id, tag_id, confidence, source, created_at) "
                "VALUES (?, ?, ?, ?, datetime('now'))",
                ("img1", 999, 0.5, "manual")
            )

    def test_check_constraints(self, db: CatalogDatabase) -> None:
        """Test CHECK constraints are enforced."""
        db.connect()
        db.initialize()

        # Insert valid tag first
        db.execute(
            "INSERT INTO tags (name, category, created_at) "
            "VALUES (?, ?, datetime('now'))",
            ("test_tag", "subject")
        )

        # Try invalid confidence (out of range)
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO image_tags "
                "(image_id, tag_id, confidence, source, created_at) "
                "VALUES (?, ?, ?, ?, datetime('now'))",
                ("img1", 1, 1.5, "manual")  # confidence > 1.0
            )

    def test_unique_constraints(self, db: CatalogDatabase) -> None:
        """Test UNIQUE constraints are enforced."""
        db.connect()
        db.initialize()

        # Insert tag
        db.execute(
            "INSERT INTO tags (name, category, created_at) "
            "VALUES (?, ?, datetime('now'))",
            ("test_tag", "subject")
        )

        # Try duplicate name
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO tags (name, category, created_at) "
                "VALUES (?, ?, datetime('now'))",
                ("test_tag", "lighting")  # Duplicate name
            )

    def test_create_backup(self, db: CatalogDatabase) -> None:
        """Test database backup creation."""
        db.connect()
        db.initialize()

        # Add some data
        db.execute(
            "INSERT INTO tags (name, category, created_at) "
            "VALUES (?, ?, datetime('now'))",
            ("test_tag", "subject")
        )

        # Create backup
        backup_path = db.create_backup()

        assert backup_path.exists()
        assert backup_path.parent == db.backup_dir
        assert "catalog_" in backup_path.name

    def test_cleanup_old_backups(self, db: CatalogDatabase) -> None:
        """Test old backup cleanup."""
        import time

        db.connect()
        db.initialize()

        # Create multiple backups with delay for distinct timestamps
        backups = []
        for i in range(5):
            backup = db.create_backup()
            backups.append(backup)
            time.sleep(1.1)  # Ensure distinct modification times (backup uses second precision)

        # Keep only 2
        deleted = db.cleanup_old_backups(keep_count=2)

        assert deleted == 3
        assert sum(1 for b in backups if b.exists()) == 2

    def test_vacuum(self, db: CatalogDatabase) -> None:
        """Test database vacuuming."""
        db.connect()
        db.initialize()

        # Add and delete data to create fragmentation
        for i in range(100):
            db.execute(
                "INSERT INTO tags (name, category, created_at) "
                "VALUES (?, ?, datetime('now'))",
                (f"tag_{i}", "subject")
            )

        db.execute("DELETE FROM tags")

        # Vacuum should not raise error
        db.vacuum()

    def test_get_stats(self, db: CatalogDatabase) -> None:
        """Test getting database statistics."""
        db.connect()
        db.initialize()

        # Add some data
        db.execute(
            "INSERT INTO tags (name, category, created_at) "
            "VALUES (?, ?, datetime('now'))",
            ("test_tag", "subject")
        )

        stats = db.get_stats()

        assert "tags_count" in stats
        assert stats["tags_count"] == 1
        assert "db_size_bytes" in stats
        assert "db_size_mb" in stats
        assert "schema_version" in stats
        assert stats["schema_version"] == 1

    def test_row_factory(self, db: CatalogDatabase) -> None:
        """Test rows can be accessed by column name."""
        db.connect()
        db.initialize()

        db.execute(
            "INSERT INTO tags (name, category, created_at) "
            "VALUES (?, ?, datetime('now'))",
            ("test_tag", "subject")
        )

        cursor = db.execute("SELECT * FROM tags")
        row = cursor.fetchone()

        # Should be able to access by name
        assert row["name"] == "test_tag"
        assert row["category"] == "subject"

    def test_views_created(self, db: CatalogDatabase) -> None:
        """Test that views are created."""
        db.connect()
        db.initialize()

        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        )
        views = {row[0] for row in cursor.fetchall()}

        expected_views = {
            "v_images_with_tags",
            "v_duplicate_images",
            "v_review_queue_detailed"
        }

        assert expected_views.issubset(views)

    def test_wal_mode_enabled(self, db: CatalogDatabase) -> None:
        """Test Write-Ahead Logging is enabled."""
        db.connect()

        cursor = db.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]

        assert mode.lower() == "wal"

    def test_repr(self, db: CatalogDatabase) -> None:
        """Test string representation."""
        repr_str = repr(db)
        assert "CatalogDatabase" in repr_str
        assert str(db.db_path) in repr_str
