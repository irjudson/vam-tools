"""
Tests for Celery tasks.
"""

from unittest.mock import MagicMock, patch

import pytest

from vam_tools.jobs.tasks import (
    analyze_catalog_task,
    generate_thumbnails_task,
    organize_catalog_task,
)


class TestAnalyzeTask:
    """Test analyze_catalog_task."""

    def test_task_registration(self):
        """Test task is registered with Celery."""
        assert analyze_catalog_task.name == "analyze_catalog"

    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.analysis.scanner._process_file_worker")
    def test_analyze_task_success(
        self, mock_process_file_worker, mock_catalog_db, tmp_path
    ):
        """Test successful analysis returns expected result format."""
        # Setup mocks
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Mock db.execute for various queries
        # For catalog_exists check
        mock_db.execute.side_effect = [
            MagicMock(
                fetchone=MagicMock(return_value=None)
            ),  # catalog.db does not exist initially (this is the first call to execute)
            None,  # For INSERT OR REPLACE INTO catalog_config (source_directory_...)
            None,  # For INSERT OR REPLACE INTO catalog_config (created)
            None,  # For INSERT OR REPLACE INTO catalog_config (last_updated)
            None,  # For INSERT OR REPLACE INTO catalog_config (catalog_id)
            None,  # For INSERT OR REPLACE INTO catalog_config (version)
            None,  # For INSERT OR REPLACE INTO catalog_config (phase)
            None,  # For INSERT OR REPLACE INTO catalog_config (phase again)
            MagicMock(
                fetchone=MagicMock(
                    return_value={
                        "total_images": 0,
                        "total_videos": 0,
                        "total_size_bytes": 0,
                        "images_scanned": 0,
                        "images_hashed": 0,
                        "images_tagged": 0,
                        "no_date": 0,
                        "suspicious_dates": 0,
                        "corrupted_count": 0,
                        "unsupported_count": 0,
                        "duplicate_groups": 0,
                        "duplicates_total": 0,
                        "potential_savings_bytes": 0,
                        "high_quality_count": 0,
                        "medium_quality_count": 0,
                        "low_quality_count": 0,
                        "processing_time_seconds": 0,
                        "images_per_second": 0,
                        "problematic_files": 0,
                    }
                )
            ),  # For get latest statistics
            MagicMock(
                fetchone=MagicMock(return_value=None)
            ),  # For image existence check (image 1)
            None,  # For INSERT INTO images (image 1)
            MagicMock(
                fetchone=MagicMock(return_value=None)
            ),  # For image existence check (image 2)
            None,  # For INSERT INTO images (image 2)
            None,  # For INSERT INTO statistics (final snapshot)
        ]

        # Mock _process_file_worker to return dummy ImageRecord
        from datetime import datetime

        from vam_tools.core.types import (
            DateInfo,
            FileType,
            ImageMetadata,
            ImageRecord,
            ImageStatus,
        )

        mock_process_file_worker.side_effect = [
            (
                ImageRecord(
                    id="checksum1",
                    source_path=tmp_path / "photos" / "test1.jpg",
                    file_type=FileType.IMAGE,
                    checksum="checksum1",
                    status=ImageStatus.COMPLETE,
                    file_size=100,
                    dates=DateInfo(selected_date=datetime(2023, 1, 1)),
                    metadata=ImageMetadata(format="JPEG", width=100, height=100),
                ),
                100,
            ),
            (
                ImageRecord(
                    id="checksum2",
                    source_path=tmp_path / "photos" / "test2.jpg",
                    file_type=FileType.IMAGE,
                    checksum="checksum2",
                    status=ImageStatus.COMPLETE,
                    file_size=200,
                    dates=DateInfo(selected_date=datetime(2023, 1, 2)),
                    metadata=ImageMetadata(format="JPEG", width=100, height=100),
                ),
                200,
            ),
        ]

        # Create some test files
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        (photos_dir / "test1.jpg").touch()
        (photos_dir / "test2.jpg").touch()

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        # Create task instance
        task = analyze_catalog_task
        task.update_state = MagicMock()  # Mock Celery's update_state

        # Execute
        result = task(
            catalog_path=str(catalog_dir),
            source_directories=[str(photos_dir)],
            detect_duplicates=False,
        )

        # Verify
        assert result["status"] == "completed"
        assert result["processed"] == 2
        assert result["total_files"] == 2

        # Verify CatalogDatabase interactions
        mock_catalog_db.assert_called_once_with(catalog_dir)
        mock_db.initialize.assert_called_once()
        assert (
            mock_db.execute.call_count >= 5
        )  # At least for config, images, and statistics inserts

    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    def test_analyze_task_invalid_source_path(self, mock_catalog_db, tmp_path):
        """Test analysis with invalid source path raises FileNotFoundError."""
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db
        mock_db.execute.return_value.fetchone.return_value = None  # No existing catalog

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        task = analyze_catalog_task
        task.update_state = MagicMock()

        with pytest.raises(FileNotFoundError):
            task(
                catalog_path=str(catalog_dir),
                source_directories=["/nonexistent/photos"],
            )


class TestOrganizeTask:
    """Test organize_catalog_task."""

    def test_task_registration(self):
        """Test task is registered with Celery."""
        assert organize_catalog_task.name == "organize_catalog"

    @patch("vam_tools.jobs.tasks.FileOrganizer")
    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    def test_organize_dry_run(self, mock_catalog_db, mock_organizer, tmp_path):
        """Test dry-run organization doesn't modify files."""
        # Setup mocks
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db
        mock_db.execute.return_value.fetchone.return_value = (10,)  # Mock image count

        mock_org_instance = MagicMock()
        mock_organizer.return_value = mock_org_instance
        mock_org_instance.organize.return_value = MagicMock(
            dry_run=True,
            total_files=10,
            organized=0,
            skipped=0,
            failed=0,
            no_date=0,
            errors=[],
            transaction_id=None,
        )

        # Execute
        task = organize_catalog_task
        task.update_state = MagicMock()

        result = task(
            catalog_path=str(tmp_path / "catalog"),
            destination_path=str(tmp_path / "output"),
            strategy="date_based",  # Added a default strategy for the test
            simulate=True,
        )

        # Verify dry_run was passed
        mock_org_instance.organize.assert_called_once()
        call_kwargs = mock_org_instance.organize.call_args[1]
        assert call_kwargs["dry_run"] is True

        # Verify result
        assert result["dry_run"] is True
        assert result["organized"] == 0


class TestThumbnailTask:
    """Test generate_thumbnails_task."""

    def test_task_registration(self):
        """Test task is registered with Celery."""
        assert generate_thumbnails_task.name == "generate_thumbnails"

    @patch("vam_tools.jobs.tasks.generate_thumbnail")
    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    def test_thumbnail_generation(
        self, mock_catalog_db, mock_generate_thumbnail, tmp_path
    ):
        """Test thumbnail generation."""
        # Setup mocks
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Mock db.execute for image retrieval
        mock_db.execute.return_value.fetchall.return_value = [
            {
                "id": "img1",
                "source_path": str(tmp_path / "photos" / "img1.jpg"),
                "thumbnail_path": None,
            },
            {
                "id": "img2",
                "source_path": str(tmp_path / "photos" / "img2.jpg"),
                "thumbnail_path": None,
            },
        ]

        # Mock generate_thumbnail
        mock_generate_thumbnail.return_value = True

        # Create some dummy image files
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        (photos_dir / "img1.jpg").touch()
        (photos_dir / "img2.jpg").touch()

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        # Execute
        task = generate_thumbnails_task
        task.update_state = MagicMock()

        result = task(
            catalog_path=str(catalog_dir),
            sizes=[256],
            quality=85,
            force=False,
        )

        # Verify
        assert result["status"] == "completed"
        assert result["generated_count"] == 2
        assert result["skipped"] == 0
        assert result["failed"] == 0

        # Verify generate_thumbnail was called for each image
        assert mock_generate_thumbnail.call_count == 2

        # Verify thumbnail_path was updated in DB
        mock_db.execute.assert_any_call(
            "UPDATE images SET thumbnail_path = ? WHERE id = ?",
            (f"thumbnails/img1.jpg", "img1"),
        )
        mock_db.execute.assert_any_call(
            "UPDATE images SET thumbnail_path = ? WHERE id = ?",
            (f"thumbnails/img2.jpg", "img2"),
        )
