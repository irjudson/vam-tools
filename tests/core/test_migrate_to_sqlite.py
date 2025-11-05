"""Tests for JSON to SQLite migration."""

import json
from pathlib import Path

import pytest

from vam_tools.core.database import CatalogDatabase
from vam_tools.core.migrate_to_sqlite import CatalogMigrator


class TestCatalogMigrator:
    """Tests for CatalogMigrator class."""

    @pytest.fixture
    def catalog_path(self, tmp_path: Path) -> Path:
        """Create temporary catalog path."""
        return tmp_path / "test_catalog"

    @pytest.fixture
    def sample_json_data(self) -> dict:
        """Create sample JSON catalog data."""
        return {
            "version": "2.0.0",
            "catalog_path": "/tmp/test_catalog",
            "catalog_id": "test-catalog-123",
            "created": "2025-01-01T00:00:00",
            "last_updated": "2025-01-02T00:00:00",
            "configuration": {
                "source_directories": ["/tmp/photos"],
                "naming_convention": "{year}/{month}/{day}/{filename}",
            },
            "state": {"phase": "complete", "progress": 100},
            "statistics": {
                "total_images": 10,
                "total_size_bytes": 1000000,
                "images_scanned": 10,
                "images_hashed": 10,
                "images_tagged": 5,
                "duplicate_groups": 2,
                "duplicate_images": 4,
                "processing_time_seconds": 120.5,
                "images_per_second": 0.083,
            },
            "images": {
                "img1": {
                    "id": "img1",
                    "source_path": "/tmp/photo1.jpg",
                    "organized_path": "/catalog/2025/01/01/photo1.jpg",
                    "file_size": 100000,
                    "file_hash": "abc123",
                    "format": "JPEG",
                    "width": 1920,
                    "height": 1080,
                    "created_at": "2025-01-01T10:00:00",
                    "modified_at": "2025-01-01T10:00:00",
                    "date_taken": "2025-01-01T09:00:00",
                    "camera_make": "Canon",
                    "camera_model": "EOS R5",
                    "quality_score": 85.5,
                    "perceptual_hash": "deadbeef",
                },
                "img2": {
                    "id": "img2",
                    "source_path": "/tmp/photo2.jpg",
                    "file_size": 90000,
                    "file_hash": "def456",
                    "format": "JPEG",
                    "created_at": "2025-01-01T11:00:00",
                    "modified_at": "2025-01-01T11:00:00",
                },
            },
            "duplicate_groups": {
                "group1": {
                    "hash_distance": 5,
                    "similarity_score": 0.95,
                    "reviewed": True,
                    "images": [
                        {
                            "id": "img1",
                            "is_primary": True,
                            "quality_score": 85.5,
                        },
                        {
                            "id": "img2",
                            "is_primary": False,
                            "quality_score": 80.0,
                        },
                    ],
                }
            },
            "burst_groups": {
                "burst1": {
                    "time_window_seconds": 5,
                    "images": [
                        {"id": "img1", "sequence_number": 0, "is_best": True},
                        {"id": "img2", "sequence_number": 1, "is_best": False},
                    ],
                }
            },
            "review_queue": [
                {
                    "image_id": "img2",
                    "reason": "duplicate",
                    "priority": 5,
                    "created_at": "2025-01-01T12:00:00",
                }
            ],
            "problematic_files": {
                "prob1": {
                    "file_path": "/tmp/corrupted.jpg",
                    "category": "corruption",
                    "error_message": "File corrupted",
                    "detected_at": "2025-01-01T13:00:00",
                }
            },
        }

    @pytest.fixture
    def json_catalog(self, catalog_path: Path, sample_json_data: dict) -> Path:
        """Create JSON catalog file."""
        catalog_path.mkdir(parents=True, exist_ok=True)
        json_file = catalog_path / "catalog.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_json_data, f, indent=2)
        return catalog_path

    def test_migrator_initialization(self, catalog_path: Path) -> None:
        """Test migrator initialization."""
        migrator = CatalogMigrator(catalog_path)

        assert migrator.catalog_path == catalog_path
        assert migrator.json_file == catalog_path / "catalog.json"
        assert isinstance(migrator.db, CatalogDatabase)

    def test_migrate_dry_run(self, json_catalog: Path, sample_json_data: dict) -> None:
        """Test dry run migration."""
        migrator = CatalogMigrator(json_catalog)
        stats = migrator.migrate(dry_run=True)

        assert stats["images"] == 2
        assert stats["duplicate_groups"] == 1
        assert stats["duplicate_images"] == 2
        assert stats["burst_groups"] == 1
        assert stats["burst_images"] == 2
        assert stats["review_queue"] == 1
        assert stats["problematic_files"] == 1

        # Verify database was not created
        assert not migrator.db.db_path.exists()

    def test_migrate_no_catalog(self, catalog_path: Path) -> None:
        """Test migration with missing catalog file."""
        migrator = CatalogMigrator(catalog_path)

        with pytest.raises(FileNotFoundError):
            migrator.migrate()

    def test_migrate_images(self, json_catalog: Path) -> None:
        """Test migrating image records."""
        migrator = CatalogMigrator(json_catalog)
        stats = migrator.migrate()

        assert stats["images"] == 2

        # Verify in database
        cursor = migrator.db.execute("SELECT * FROM images ORDER BY id")
        images = cursor.fetchall()

        assert len(images) == 2
        assert images[0]["id"] == "img1"
        assert images[0]["source_path"] == "/tmp/photo1.jpg"
        assert images[0]["file_size"] == 100000
        assert images[0]["file_hash"] == "abc123"
        assert images[0]["format"] == "JPEG"
        assert images[0]["width"] == 1920
        assert images[0]["height"] == 1080
        assert images[0]["camera_make"] == "Canon"
        assert images[0]["quality_score"] == 85.5

    def test_migrate_duplicate_groups(self, json_catalog: Path) -> None:
        """Test migrating duplicate groups."""
        migrator = CatalogMigrator(json_catalog)
        stats = migrator.migrate()

        assert stats["duplicate_groups"] == 1
        assert stats["duplicate_images"] == 2

        # Verify groups
        cursor = migrator.db.execute("SELECT * FROM duplicate_groups")
        groups = cursor.fetchall()
        assert len(groups) == 1
        assert groups[0]["hash_distance"] == 5
        assert groups[0]["similarity_score"] == 0.95
        assert groups[0]["reviewed"] == 1

        # Verify group images
        cursor = migrator.db.execute(
            "SELECT * FROM duplicate_group_images ORDER BY is_primary DESC"
        )
        images = cursor.fetchall()
        assert len(images) == 2
        assert images[0]["image_id"] == "img1"
        assert images[0]["is_primary"] == 1
        assert images[1]["image_id"] == "img2"
        assert images[1]["is_primary"] == 0

    def test_migrate_burst_groups(self, json_catalog: Path) -> None:
        """Test migrating burst groups."""
        migrator = CatalogMigrator(json_catalog)
        stats = migrator.migrate()

        assert stats["burst_groups"] == 1
        assert stats["burst_images"] == 2

        # Verify groups
        cursor = migrator.db.execute("SELECT * FROM burst_groups")
        groups = cursor.fetchall()
        assert len(groups) == 1
        assert groups[0]["time_window_seconds"] == 5

        # Verify group images
        cursor = migrator.db.execute(
            "SELECT * FROM burst_group_images ORDER BY sequence_number"
        )
        images = cursor.fetchall()
        assert len(images) == 2
        assert images[0]["image_id"] == "img1"
        assert images[0]["sequence_number"] == 0
        assert images[0]["is_best"] == 1
        assert images[1]["image_id"] == "img2"
        assert images[1]["sequence_number"] == 1

    def test_migrate_review_queue(self, json_catalog: Path) -> None:
        """Test migrating review queue."""
        migrator = CatalogMigrator(json_catalog)
        stats = migrator.migrate()

        assert stats["review_queue"] == 1

        # Verify queue
        cursor = migrator.db.execute("SELECT * FROM review_queue")
        items = cursor.fetchall()
        assert len(items) == 1
        assert items[0]["image_id"] == "img2"
        assert items[0]["reason"] == "duplicate"
        assert items[0]["priority"] == 5

    def test_migrate_problematic_files(self, json_catalog: Path) -> None:
        """Test migrating problematic files."""
        migrator = CatalogMigrator(json_catalog)
        stats = migrator.migrate()

        assert stats["problematic_files"] == 1

        # Verify files
        cursor = migrator.db.execute("SELECT * FROM problematic_files")
        files = cursor.fetchall()
        assert len(files) == 1
        assert files[0]["file_path"] == "/tmp/corrupted.jpg"
        assert files[0]["category"] == "corruption"
        assert files[0]["error_message"] == "File corrupted"

    def test_migrate_configuration(self, json_catalog: Path) -> None:
        """Test migrating configuration."""
        migrator = CatalogMigrator(json_catalog)
        stats = migrator.migrate()

        # Verify config
        cursor = migrator.db.execute("SELECT * FROM catalog_config")
        config = {row["key"]: json.loads(row["value"]) for row in cursor.fetchall()}

        assert "source_directories" in config
        assert config["source_directories"] == ["/tmp/photos"]
        assert config["naming_convention"] == "{year}/{month}/{day}/{filename}"

    def test_migrate_statistics(self, json_catalog: Path) -> None:
        """Test migrating statistics."""
        migrator = CatalogMigrator(json_catalog)
        stats = migrator.migrate()

        # Verify statistics
        cursor = migrator.db.execute("SELECT * FROM statistics")
        stat_row = cursor.fetchone()

        assert stat_row is not None
        assert stat_row["total_images"] == 10
        assert stat_row["total_size_bytes"] == 1000000
        assert stat_row["images_scanned"] == 10
        assert stat_row["duplicate_groups"] == 2
        assert stat_row["processing_time_seconds"] == 120.5

    def test_migrate_empty_catalog(self, catalog_path: Path) -> None:
        """Test migrating empty catalog."""
        catalog_path.mkdir(parents=True, exist_ok=True)
        json_file = catalog_path / "catalog.json"

        # Create minimal catalog
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": "2.0.0",
                    "images": {},
                    "duplicate_groups": {},
                    "burst_groups": {},
                    "review_queue": [],
                    "problematic_files": {},
                },
                f,
            )

        migrator = CatalogMigrator(catalog_path)
        stats = migrator.migrate()

        assert stats["images"] == 0
        assert stats["duplicate_groups"] == 0
        assert stats["review_queue"] == 0

        # Verify database was created and initialized
        assert migrator.db.db_path.exists()
        cursor = migrator.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "images" in tables

    def test_migrate_transaction_rollback(self, json_catalog: Path, monkeypatch) -> None:
        """Test migration rolls back on error."""
        migrator = CatalogMigrator(json_catalog)

        # Monkeypatch _migrate_images to raise an error mid-migration
        original_migrate_images = migrator._migrate_images

        def failing_migrate_images(images):
            # Migrate first image successfully
            count = 0
            for image_id, img_data in list(images.items())[:1]:
                migrator.db.execute(
                    """
                    INSERT OR REPLACE INTO images (
                        id, source_path, file_size, file_hash, format,
                        created_at, modified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        image_id,
                        img_data.get("source_path"),
                        img_data.get("file_size"),
                        img_data.get("file_hash"),
                        img_data.get("format"),
                        img_data.get("created_at"),
                        img_data.get("modified_at"),
                    ),
                )
                count += 1
            # Then raise an error
            raise ValueError("Simulated migration error")

        monkeypatch.setattr(migrator, "_migrate_images", failing_migrate_images)

        # Create and initialize database
        migrator.db.connect()
        migrator.db.initialize()

        # Migration should fail
        with pytest.raises(ValueError, match="Simulated migration error"):
            migrator.migrate()

        # Verify rollback - no images should be persisted
        cursor = migrator.db.execute("SELECT COUNT(*) FROM images")
        count = cursor.fetchone()[0]
        assert count == 0

    def test_migrate_full_workflow(self, json_catalog: Path) -> None:
        """Test complete migration workflow."""
        migrator = CatalogMigrator(json_catalog)

        # First do dry run
        dry_stats = migrator.migrate(dry_run=True)
        assert not migrator.db.db_path.exists()

        # Then do actual migration
        real_stats = migrator.migrate()
        assert migrator.db.db_path.exists()

        # Stats should match
        assert dry_stats["images"] == real_stats["images"]
        assert dry_stats["duplicate_groups"] == real_stats["duplicate_groups"]
        assert dry_stats["review_queue"] == real_stats["review_queue"]

        # Verify all data is accessible
        db_stats = migrator.db.get_stats()
        assert db_stats["images_count"] == 2
        assert db_stats["duplicate_groups_count"] == 1
        assert db_stats["review_queue_count"] == 1
