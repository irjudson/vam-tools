"""
Tests for Celery tasks.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from vam_tools.jobs.tasks import (
    analyze_catalog_task,
    auto_tag_task,
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
        assert result["total_processed"] == 2
        assert result["files_added"] == 2

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
        assert result["total_processed"] == 0
        assert result["files_added"] == 0


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


class TestAutoTagTask:
    """Test auto_tag_task."""

    def test_task_registration(self):
        """Test task is registered with Celery."""
        assert auto_tag_task.name == "auto_tag"

    @patch("vam_tools.jobs.job_metrics.get_gpu_info")
    @patch("vam_tools.jobs.job_metrics.check_gpu_available")
    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.analysis.image_tagger.ImageTagger")
    @patch("vam_tools.analysis.image_tagger.check_backends_available")
    def test_auto_tag_no_images_to_tag(
        self,
        mock_check_backends,
        mock_image_tagger,
        mock_catalog_db,
        mock_check_gpu,
        mock_get_gpu_info,
    ):
        """Test auto_tag when all images are already tagged."""
        # Setup mocks
        mock_check_gpu.return_value = False
        mock_get_gpu_info.return_value = None
        mock_check_backends.return_value = {"openclip": True, "ollama": False}

        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Mock session execute to return no images needing tagging
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.scalar.return_value = 10  # 10 images already tagged
        mock_db.session.execute.return_value = mock_result

        # Execute
        task = auto_tag_task
        task.update_state = MagicMock()

        catalog_id = str(uuid.uuid4())
        result = task(catalog_id=catalog_id, backend="openclip")

        # Verify
        assert result["status"] == "completed"
        assert result["images_tagged"] == 0
        assert result["images_skipped"] == 10

    @patch("vam_tools.jobs.job_metrics.get_gpu_info")
    @patch("vam_tools.jobs.job_metrics.check_gpu_available")
    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.analysis.image_tagger.ImageTagger")
    @patch("vam_tools.analysis.image_tagger.check_backends_available")
    def test_auto_tag_openclip_success(
        self,
        mock_check_backends,
        mock_image_tagger,
        mock_catalog_db,
        mock_check_gpu,
        mock_get_gpu_info,
        tmp_path,
    ):
        """Test successful auto-tagging with OpenCLIP backend."""
        from pathlib import Path

        from vam_tools.analysis.image_tagger import TagResult

        # Setup mocks
        mock_check_gpu.return_value = False
        mock_get_gpu_info.return_value = None
        mock_check_backends.return_value = {"openclip": True, "ollama": False}

        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Create test image files
        img1 = tmp_path / "img1.jpg"
        img2 = tmp_path / "img2.jpg"
        img1.touch()
        img2.touch()

        # Mock session execute - first call returns images to tag, subsequent calls for tag storage
        mock_result_images = MagicMock()
        mock_result_images.fetchall.return_value = [
            ("id1", str(img1)),
            ("id2", str(img2)),
        ]

        # Create a mock that returns proper tag_id values for INSERT INTO tags calls
        def mock_execute(query, params=None):
            mock_result = MagicMock()
            # Check if this is an INSERT INTO tags query (returns tag_id via scalar())
            query_str = str(query) if hasattr(query, "__str__") else ""
            if "INSERT INTO tags" in query_str or (
                params and "name" in params and "category" in params
            ):
                mock_result.scalar.return_value = 1  # Return a tag_id
            elif "INSERT INTO image_tags" in query_str:
                pass  # No return value needed
            return mock_result

        # First call returns images, subsequent calls use mock_execute
        call_count = [0]
        original_mock_execute = mock_execute

        def side_effect_execute(query, params=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_result_images
            return original_mock_execute(query, params)

        mock_db.session.execute.side_effect = side_effect_execute

        # Mock ImageTagger
        mock_tagger = MagicMock()
        mock_image_tagger.return_value = mock_tagger

        # Mock tag_batch to return results
        mock_tagger.tag_batch.return_value = {
            img1: [
                TagResult(
                    tag_name="dogs",
                    confidence=0.9,
                    category="subject",
                    source="openclip",
                ),
                TagResult(
                    tag_name="outdoor",
                    confidence=0.8,
                    category="scene",
                    source="openclip",
                ),
            ],
            img2: [
                TagResult(
                    tag_name="cats",
                    confidence=0.85,
                    category="subject",
                    source="openclip",
                ),
            ],
        }

        # Execute
        task = auto_tag_task
        task.update_state = MagicMock()

        catalog_id = str(uuid.uuid4())
        result = task(
            catalog_id=catalog_id,
            backend="openclip",
            threshold=0.25,
            max_tags=10,
            batch_size=32,
        )

        # Verify
        assert result["status"] == "completed"
        assert result["backend"] == "openclip"
        assert result["images_tagged"] == 2
        assert result["total_images"] == 2
        assert result["unique_tags_applied"] == 3  # dogs, outdoor, cats

        # Verify ImageTagger was initialized correctly
        mock_image_tagger.assert_called_once()

        # Verify tag_batch was called
        mock_tagger.tag_batch.assert_called()

    @patch("vam_tools.jobs.job_metrics.check_gpu_available")
    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.analysis.image_tagger.check_backends_available")
    def test_auto_tag_backend_not_available(
        self,
        mock_check_backends,
        mock_catalog_db,
        mock_check_gpu,
    ):
        """Test auto_tag fails gracefully when backend not available."""
        mock_check_gpu.return_value = False
        mock_check_backends.return_value = {"openclip": False, "ollama": False}

        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        task = auto_tag_task
        task.update_state = MagicMock()

        catalog_id = str(uuid.uuid4())

        with pytest.raises(RuntimeError, match="OpenCLIP backend not available"):
            task(catalog_id=catalog_id, backend="openclip")

    @patch("vam_tools.jobs.job_metrics.get_gpu_info")
    @patch("vam_tools.jobs.job_metrics.check_gpu_available")
    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.analysis.image_tagger.ImageTagger")
    @patch("vam_tools.analysis.image_tagger.check_backends_available")
    def test_auto_tag_ollama_success(
        self,
        mock_check_backends,
        mock_image_tagger,
        mock_catalog_db,
        mock_check_gpu,
        mock_get_gpu_info,
        tmp_path,
    ):
        """Test successful auto-tagging with Ollama backend."""
        from vam_tools.analysis.image_tagger import TagResult

        # Setup mocks
        mock_check_gpu.return_value = False
        mock_get_gpu_info.return_value = None
        mock_check_backends.return_value = {"openclip": False, "ollama": True}

        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Create test image
        img1 = tmp_path / "img1.jpg"
        img1.touch()

        # Mock session execute
        mock_result_images = MagicMock()
        mock_result_images.fetchall.return_value = [("id1", str(img1))]
        mock_db.session.execute.return_value = mock_result_images

        # Mock ImageTagger
        mock_tagger = MagicMock()
        mock_image_tagger.return_value = mock_tagger

        # Mock tag_image for Ollama (processes one at a time)
        mock_tagger.tag_image.return_value = [
            TagResult(
                tag_name="portrait",
                confidence=0.95,
                category="subject",
                source="ollama",
            ),
        ]

        # Execute
        task = auto_tag_task
        task.update_state = MagicMock()

        catalog_id = str(uuid.uuid4())
        result = task(
            catalog_id=catalog_id,
            backend="ollama",
            model="llava",
        )

        # Verify
        assert result["status"] == "completed"
        assert result["backend"] == "ollama"
        assert result["images_tagged"] == 1

        # Verify tag_image was called (not tag_batch for Ollama)
        mock_tagger.tag_image.assert_called()

    @patch("vam_tools.jobs.parallel_duplicates.duplicates_coordinator_task")
    @patch("vam_tools.jobs.job_metrics.get_gpu_info")
    @patch("vam_tools.jobs.job_metrics.check_gpu_available")
    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.analysis.image_tagger.ImageTagger")
    @patch("vam_tools.analysis.image_tagger.check_backends_available")
    def test_auto_tag_continue_pipeline(
        self,
        mock_check_backends,
        mock_image_tagger,
        mock_catalog_db,
        mock_check_gpu,
        mock_get_gpu_info,
        mock_duplicates_coordinator,
    ):
        """Test auto_tag triggers duplicate detection when continue_pipeline=True."""
        from pathlib import Path

        # Setup mocks
        mock_check_gpu.return_value = False
        mock_get_gpu_info.return_value = None
        mock_check_backends.return_value = {"openclip": True, "ollama": False}

        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Mock images to tag so we don't hit early return
        image_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(image_id, "/path/to/image.jpg")]
        mock_db.session.execute.return_value = mock_result

        # Mock the tagger
        mock_tagger = MagicMock()
        mock_image_tagger.return_value = mock_tagger
        # Return empty tags to complete quickly
        mock_tagger.tag_batch.return_value = {Path("/path/to/image.jpg"): []}

        # Execute
        task = auto_tag_task
        task.update_state = MagicMock()

        catalog_id = str(uuid.uuid4())
        result = task(
            catalog_id=catalog_id,
            backend="openclip",
            continue_pipeline=True,
        )

        # Verify
        assert result["status"] == "completed"
        assert result["next_job"] == "detect_duplicates"

        # Verify duplicates_coordinator_task.delay was called
        mock_duplicates_coordinator.delay.assert_called_once_with(
            catalog_id=catalog_id,
            similarity_threshold=5,
            recompute_hashes=False,
        )

    @patch("vam_tools.jobs.job_metrics.check_gpu_available")
    @patch("vam_tools.jobs.tasks.CatalogDatabase")
    @patch("vam_tools.analysis.image_tagger.check_backends_available")
    def test_auto_tag_empty_catalog(
        self,
        mock_check_backends,
        mock_catalog_db,
        mock_check_gpu,
    ):
        """Test auto_tag with empty catalog."""
        mock_check_gpu.return_value = False
        mock_check_backends.return_value = {"openclip": True, "ollama": False}

        mock_db = MagicMock()
        mock_catalog_db.return_value.__enter__.return_value = mock_db

        # Mock empty results
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.scalar.return_value = 0
        mock_db.session.execute.return_value = mock_result

        # Execute
        task = auto_tag_task
        task.update_state = MagicMock()

        catalog_id = str(uuid.uuid4())
        result = task(catalog_id=catalog_id, backend="openclip")

        # Verify
        assert result["status"] == "completed"
        assert result["message"] == "No images in catalog"
        assert result["images_tagged"] == 0
