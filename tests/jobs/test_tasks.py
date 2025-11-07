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
    def test_analyze_task_success(self, mock_catalog_db, tmp_path):
        """Test successful analysis returns expected result format."""
        # Setup mocks
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        mock_db.get_statistics.return_value = MagicMock(
            total_images=100,
            total_videos=10,
            total_size_bytes=1000000,
            no_date=5,
            suspicious_dates=2,
        )

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

    def test_analyze_task_invalid_path(self):
        """Test analysis with invalid path raises error."""
        task = analyze_catalog_task
        task.update_state = MagicMock()

        with pytest.raises(Exception):
            task(
                catalog_path="/nonexistent/path",
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
        mock_db.list_images.return_value = []

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
            output_directory=str(tmp_path / "output"),
            dry_run=True,
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

    @patch("vam_tools.jobs.tasks.ThumbnailGenerator")
    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    def test_thumbnail_generation(self, mock_catalog_db, mock_generator, tmp_path):
        """Test thumbnail generation."""
        # Setup mocks
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db
        mock_db.list_images.return_value = []

        mock_gen_instance = MagicMock()
        mock_generator.return_value = mock_gen_instance

        # Execute
        task = generate_thumbnails_task
        task.update_state = MagicMock()

        result = task(
            catalog_path=str(tmp_path / "catalog"),
            sizes=[256, 512],
            quality=85,
            force=False,
        )

        # Verify
        assert result["status"] == "completed"
        assert result["sizes"] == [256, 512]
