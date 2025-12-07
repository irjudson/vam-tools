"""Tests for burst detection Celery tasks."""

import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


class TestDetectBurstsTask:
    """Tests for detect_bursts_task."""

    def test_task_registration(self):
        """Test task is registered with Celery."""
        from vam_tools.jobs.tasks import detect_bursts_task

        assert detect_bursts_task.name == "detect_bursts"

    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.jobs.tasks.BurstDetector")
    def test_detect_bursts_task_creates_burst_records(
        self, mock_burst_detector, mock_catalog_db
    ):
        """Test that task creates burst records in database."""
        from vam_tools.jobs.tasks import detect_bursts_task

        # Setup mocks
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Mock images with burst pattern
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_images = [
            (
                f"img-{i:03d}",
                base_time + timedelta(seconds=i * 0.5),
                "Canon",
                "R5",
                0.8 + (i * 0.02),
            )
            for i in range(5)
        ]

        # Mock database query to return images
        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_images
        mock_db.session.execute.return_value = mock_result

        # Mock BurstDetector
        from vam_tools.analysis.burst_detector import BurstGroup, ImageInfo

        mock_detector = MagicMock()
        mock_burst_detector.return_value = mock_detector

        # Create a burst group for the mock
        images = [
            ImageInfo(
                image_id=img[0],
                timestamp=img[1],
                camera_make=img[2],
                camera_model=img[3],
                quality_score=img[4],
            )
            for img in mock_images
        ]
        burst_group = BurstGroup(images=images)
        burst_group.best_image_id = "img-002"  # Middle image
        mock_detector.detect_bursts.return_value = [burst_group]

        # Setup task
        task = detect_bursts_task
        task.update_state = MagicMock()

        # Execute
        catalog_id = str(uuid.uuid4())
        result = task(catalog_id)

        # Verify
        assert result["status"] == "completed"
        assert result["bursts_detected"] == 1
        assert result["images_processed"] == 5
        assert result["total_burst_images"] == 5

        # Verify BurstDetector was initialized correctly
        mock_burst_detector.assert_called_once_with(
            gap_threshold_seconds=2.0, min_burst_size=3
        )

        # Verify detector.detect_bursts was called
        assert mock_detector.detect_bursts.called

        # Verify database inserts were called
        assert mock_db.session.execute.called
        assert mock_db.session.commit.called

    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.jobs.tasks.BurstDetector")
    def test_detect_bursts_task_updates_image_burst_ids(
        self, mock_burst_detector, mock_catalog_db
    ):
        """Test that task updates images with burst_id and burst_sequence."""
        from vam_tools.jobs.tasks import detect_bursts_task

        # Setup mocks
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Mock images
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_images = [
            ("img-001", base_time, "Canon", "R5", 0.8),
            ("img-002", base_time + timedelta(seconds=0.5), "Canon", "R5", 0.9),
            ("img-003", base_time + timedelta(seconds=1.0), "Canon", "R5", 0.7),
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_images
        mock_db.session.execute.return_value = mock_result

        # Mock BurstDetector
        from vam_tools.analysis.burst_detector import BurstGroup, ImageInfo

        mock_detector = MagicMock()
        mock_burst_detector.return_value = mock_detector

        images = [
            ImageInfo(
                image_id=img[0],
                timestamp=img[1],
                camera_make=img[2],
                camera_model=img[3],
                quality_score=img[4],
            )
            for img in mock_images
        ]
        burst_group = BurstGroup(images=images)
        burst_group.best_image_id = "img-002"
        mock_detector.detect_bursts.return_value = [burst_group]

        # Setup task
        task = detect_bursts_task
        task.update_state = MagicMock()

        # Execute
        catalog_id = str(uuid.uuid4())
        result = task(catalog_id)

        # Verify result
        assert result["status"] == "completed"
        assert result["bursts_detected"] == 1

        # Count UPDATE calls for images (should be 3 - one per image in burst)
        execute_calls = mock_db.session.execute.call_args_list
        update_calls = [
            call for call in execute_calls if "UPDATE images" in str(call[0][0])
        ]
        assert len(update_calls) >= 3  # At least 3 update calls for the images

    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.jobs.tasks.BurstDetector")
    def test_detect_bursts_task_clears_existing_bursts(
        self, mock_burst_detector, mock_catalog_db
    ):
        """Test that task clears existing bursts before detection."""
        from vam_tools.jobs.tasks import detect_bursts_task

        # Setup mocks
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Mock empty result (no images)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db.session.execute.return_value = mock_result

        # Mock BurstDetector
        mock_detector = MagicMock()
        mock_burst_detector.return_value = mock_detector
        mock_detector.detect_bursts.return_value = []

        # Setup task
        task = detect_bursts_task
        task.update_state = MagicMock()

        # Execute
        catalog_id = str(uuid.uuid4())
        result = task(catalog_id)

        # Verify DELETE was called for existing bursts
        execute_calls = mock_db.session.execute.call_args_list
        delete_calls = [
            call for call in execute_calls if "DELETE FROM bursts" in str(call[0][0])
        ]
        assert len(delete_calls) >= 1

    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.jobs.tasks.BurstDetector")
    def test_detect_bursts_task_with_custom_parameters(
        self, mock_burst_detector, mock_catalog_db
    ):
        """Test that task respects custom gap_threshold and min_burst_size."""
        from vam_tools.jobs.tasks import detect_bursts_task

        # Setup mocks
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db.session.execute.return_value = mock_result

        mock_detector = MagicMock()
        mock_burst_detector.return_value = mock_detector
        mock_detector.detect_bursts.return_value = []

        # Setup task
        task = detect_bursts_task
        task.update_state = MagicMock()

        # Execute with custom parameters
        catalog_id = str(uuid.uuid4())
        result = task(catalog_id, gap_threshold=3.0, min_burst_size=5)

        # Verify BurstDetector was initialized with custom parameters
        mock_burst_detector.assert_called_once_with(
            gap_threshold_seconds=3.0, min_burst_size=5
        )

    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.jobs.tasks.BurstDetector")
    def test_detect_bursts_task_handles_no_bursts(
        self, mock_burst_detector, mock_catalog_db
    ):
        """Test that task handles case with no bursts detected."""
        from vam_tools.jobs.tasks import detect_bursts_task

        # Setup mocks
        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Mock images but no bursts
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_images = [
            ("img-001", base_time, "Canon", "R5", 0.8),
            ("img-002", base_time + timedelta(seconds=10), "Canon", "R5", 0.9),
        ]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_images
        mock_db.session.execute.return_value = mock_result

        # Mock BurstDetector returning no bursts
        mock_detector = MagicMock()
        mock_burst_detector.return_value = mock_detector
        mock_detector.detect_bursts.return_value = []

        # Setup task
        task = detect_bursts_task
        task.update_state = MagicMock()

        # Execute
        catalog_id = str(uuid.uuid4())
        result = task(catalog_id)

        # Verify
        assert result["status"] == "completed"
        assert result["bursts_detected"] == 0
        assert result["images_processed"] == 2
        assert result["total_burst_images"] == 0
