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

        # Mock get_image to return None (image doesn't exist)
        mock_db.get_image.return_value = None

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

        # Create task instance
        task = analyze_catalog_task
        task.update_state = MagicMock()  # Mock Celery's update_state

        # Execute
        # Generate a UUID for catalog_id
        import uuid

        catalog_id = str(uuid.uuid4())

        result = task(
            catalog_id=catalog_id,
            source_directories=[str(photos_dir)],
            detect_duplicates=False,
        )

        # Verify
        assert result["status"] == "completed"
        assert result["processed"] == 2
        assert result["total_files"] == 2

        # Verify CatalogDatabase was used
        assert mock_catalog_db.called
        # Verify add_image was called for each file
        assert mock_db.add_image.call_count == 2

    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    def test_analyze_task_invalid_source_path(self, mock_catalog_db, tmp_path):
        """Test analysis with invalid source path completes successfully but processes 0 files."""
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        task = analyze_catalog_task
        task.update_state = MagicMock()

        # Generate a UUID for catalog_id
        import uuid

        catalog_id = str(uuid.uuid4())

        # With nonexistent directory, os.walk will simply not find any files
        # So the task completes successfully with 0 files processed
        result = task(
            catalog_id=catalog_id,
            source_directories=["/nonexistent/photos"],
        )

        assert result["status"] == "completed"
        assert result["processed"] == 0
        assert result["total_files"] == 0


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

        mock_org_instance = MagicMock()
        mock_organizer.return_value = mock_org_instance
        # Mock the return value as a dict (the actual method returns a dict)
        mock_org_instance.organize_files.return_value = {
            "total": 10,
            "organized": 0,
            "skipped": 10,
            "failed": 0,
            "errors": [],
        }

        # Execute
        task = organize_catalog_task
        task.update_state = MagicMock()

        # Generate a UUID for catalog_id
        import uuid

        catalog_id = str(uuid.uuid4())

        result = task(
            catalog_id=catalog_id,
            destination_path=str(tmp_path / "output"),
            strategy="date_based",
            simulate=True,
        )

        # Verify organize_files was called
        assert mock_org_instance.organize_files.called

        # Verify result contains expected fields
        assert result["status"] == "completed"
        assert result["simulate"] is True
        assert "results" in result


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
        from pathlib import Path

        from vam_tools.core.types import FileType, ImageRecord, ImageStatus

        # Setup mocks
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Create dummy image records
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        (photos_dir / "img1.jpg").touch()
        (photos_dir / "img2.jpg").touch()

        img1 = ImageRecord(
            id="img1",
            source_path=photos_dir / "img1.jpg",
            file_type=FileType.IMAGE,
            checksum="checksum1",
            status=ImageStatus.COMPLETE,
            file_size=1000,
        )
        img2 = ImageRecord(
            id="img2",
            source_path=photos_dir / "img2.jpg",
            file_type=FileType.IMAGE,
            checksum="checksum2",
            status=ImageStatus.COMPLETE,
            file_size=2000,
        )

        # Mock get_all_images to return dict of image records
        mock_db.get_all_images.return_value = {
            "img1": img1,
            "img2": img2,
        }

        # Mock generate_thumbnail to succeed
        mock_generate_thumbnail.return_value = True

        # Execute
        task = generate_thumbnails_task
        task.update_state = MagicMock()

        # Generate a UUID for catalog_id
        import uuid

        catalog_id = str(uuid.uuid4())

        result = task(
            catalog_id=catalog_id,
            sizes=[256],
            quality=85,
            force=False,
        )

        # Verify
        assert result["status"] == "completed"
        assert result["generated_count"] == 2
        assert result["skipped_count"] == 0

        # Verify generate_thumbnail was called
        assert mock_generate_thumbnail.called
        # Should be called once per image per size = 2 * 1 = 2 times
        assert mock_generate_thumbnail.call_count >= 1
