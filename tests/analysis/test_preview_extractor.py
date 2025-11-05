"""Tests for preview extractor."""

import io
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from vam_tools.analysis.preview_extractor import (
    HEIC_TIFF_FORMATS,
    RAW_FORMATS,
    PreviewExtractor,
    _extract_preview_worker,
)
from vam_tools.core.catalog import CatalogDatabase
from vam_tools.core.types import FileType, ImageRecord


class TestExtractPreviewWorker:
    """Tests for _extract_preview_worker function."""

    @pytest.fixture
    def sample_raw_image(self) -> ImageRecord:
        """Create sample RAW image record."""
        return ImageRecord(
            id="img1",
            source_path="/tmp/test.cr2",
            file_size=1000000,
            file_hash="abc123",
            checksum="sha256:abc123",
            format="CR2",
            width=6000,
            height=4000,
            file_type=FileType.IMAGE,
        )

    @pytest.fixture
    def sample_heic_image(self) -> ImageRecord:
        """Create sample HEIC image record."""
        return ImageRecord(
            id="img2",
            source_path="/tmp/test.heic",
            file_size=500000,
            file_hash="def456",
            checksum="sha256:abc123",
            format="HEIC",
            width=4000,
            height=3000,
            file_type=FileType.IMAGE,
        )

    @patch("subprocess.run")
    def test_extract_raw_preview_success(
        self, mock_run, sample_raw_image: ImageRecord
    ) -> None:
        """Test successful RAW preview extraction."""
        mock_run.return_value = Mock(returncode=0, stdout=b"jpeg_data_here")

        image_id, preview_bytes, error = _extract_preview_worker(
            (sample_raw_image, Path("/tmp/catalog"))
        )

        assert image_id == "img1"
        assert preview_bytes == b"jpeg_data_here"
        assert error is None
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_extract_raw_fallback_to_jpgfromraw(
        self, mock_run, sample_raw_image: ImageRecord
    ) -> None:
        """Test RAW extraction falls back to JpgFromRaw."""
        # First call (PreviewImage) returns empty
        # Second call (JpgFromRaw) returns data
        mock_run.side_effect = [
            Mock(returncode=0, stdout=b""),
            Mock(returncode=0, stdout=b"raw_jpeg_data"),
        ]

        image_id, preview_bytes, error = _extract_preview_worker(
            (sample_raw_image, Path("/tmp/catalog"))
        )

        assert image_id == "img1"
        assert preview_bytes == b"raw_jpeg_data"
        assert error is None
        assert mock_run.call_count == 2

    @patch("subprocess.run")
    def test_extract_raw_no_preview(
        self, mock_run, sample_raw_image: ImageRecord
    ) -> None:
        """Test RAW file with no embedded preview."""
        mock_run.side_effect = [
            Mock(returncode=0, stdout=b""),
            Mock(returncode=0, stdout=b""),
        ]

        image_id, preview_bytes, error = _extract_preview_worker(
            (sample_raw_image, Path("/tmp/catalog"))
        )

        assert image_id == "img1"
        assert preview_bytes is None
        assert error == "No embedded preview found"

    @patch("subprocess.run")
    def test_extract_raw_timeout(
        self, mock_run, sample_raw_image: ImageRecord
    ) -> None:
        """Test RAW extraction timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("exiftool", 30)

        image_id, preview_bytes, error = _extract_preview_worker(
            (sample_raw_image, Path("/tmp/catalog"))
        )

        assert image_id == "img1"
        assert preview_bytes is None
        assert "Timeout" in error

    @patch("subprocess.run")
    def test_extract_raw_exiftool_not_found(
        self, mock_run, sample_raw_image: ImageRecord
    ) -> None:
        """Test RAW extraction when exiftool not installed."""
        mock_run.side_effect = FileNotFoundError()

        image_id, preview_bytes, error = _extract_preview_worker(
            (sample_raw_image, Path("/tmp/catalog"))
        )

        assert image_id == "img1"
        assert preview_bytes is None
        assert "ExifTool not installed" in error

    def test_extract_heic_success(
        self, sample_heic_image: ImageRecord, tmp_path: Path
    ) -> None:
        """Test successful HEIC conversion."""
        # Create a test HEIC file
        test_file = tmp_path / "test.heic"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(test_file, format="JPEG")  # Save as JPEG for testing

        sample_heic_image.source_path = str(test_file)

        image_id, preview_bytes, error = _extract_preview_worker(
            (sample_heic_image, Path("/tmp/catalog"))
        )

        assert image_id == "img2"
        assert preview_bytes is not None
        assert len(preview_bytes) > 0
        assert error is None

    def test_extract_heic_convert_mode(
        self, sample_heic_image: ImageRecord, tmp_path: Path
    ) -> None:
        """Test HEIC conversion with mode conversion."""
        # Create test file with RGBA mode
        test_file = tmp_path / "test.heic"
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img.save(test_file, format="PNG")

        sample_heic_image.source_path = str(test_file)

        image_id, preview_bytes, error = _extract_preview_worker(
            (sample_heic_image, Path("/tmp/catalog"))
        )

        assert image_id == "img2"
        assert preview_bytes is not None
        assert error is None

    def test_extract_unsupported_format(self, sample_raw_image: ImageRecord) -> None:
        """Test unsupported format."""
        sample_raw_image.source_path = "/tmp/test.xyz"

        image_id, preview_bytes, error = _extract_preview_worker(
            (sample_raw_image, Path("/tmp/catalog"))
        )

        assert image_id == "img1"
        assert preview_bytes is None
        assert "Unsupported format" in error


class TestPreviewExtractor:
    """Tests for PreviewExtractor class."""

    @pytest.fixture
    def catalog(self, tmp_path: Path) -> CatalogDatabase:
        """Create test catalog."""
        catalog_path = tmp_path / "catalog"
        catalog = CatalogDatabase(catalog_path)
        catalog.initialize([])
        return catalog

    @pytest.fixture
    def extractor(self, catalog: CatalogDatabase) -> PreviewExtractor:
        """Create preview extractor."""
        return PreviewExtractor(catalog, workers=2)

    def test_initialization(self, extractor: PreviewExtractor) -> None:
        """Test extractor initialization."""
        assert extractor.catalog is not None
        assert extractor.workers == 2
        assert extractor.preview_cache is not None

    def test_initialization_default_workers(
        self, catalog: CatalogDatabase
    ) -> None:
        """Test default worker count."""
        import multiprocessing

        extractor = PreviewExtractor(catalog)
        assert extractor.workers == multiprocessing.cpu_count()

    def test_extract_previews_no_images(self, extractor: PreviewExtractor) -> None:
        """Test extraction with no images."""
        extractor.extract_previews()
        # Should complete without error

    def test_extract_previews_no_raw_images(
        self, extractor: PreviewExtractor, tmp_path: Path
    ) -> None:
        """Test extraction with only JPEG images."""
        # Add regular JPEG image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        image = ImageRecord(
            id="img1",
            source_path=str(img_path),
            file_size=1000,
            file_hash="abc",
            checksum="sha256:abc123",
            format="JPEG",
            width=100,
            height=100,
            file_type=FileType.IMAGE,
        )
        extractor.catalog.add_image(image)

        extractor.extract_previews()
        # Should skip JPEG images

    @patch("vam_tools.analysis.preview_extractor._extract_preview_worker")
    @patch("multiprocessing.Pool")
    def test_extract_previews_success(
        self, mock_pool, mock_worker, extractor: PreviewExtractor, tmp_path: Path
    ) -> None:
        """Test successful preview extraction."""
        # Add RAW image
        image = ImageRecord(
            id="img1",
            source_path="/tmp/test.cr2",
            file_size=1000000,
            file_hash="abc",
            checksum="sha256:abc123",
            format="CR2",
            width=6000,
            height=4000,
            file_type=FileType.IMAGE,
        )
        extractor.catalog.add_image(image)

        # Mock worker to return preview data
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.imap_unordered.return_value = [
            ("img1", b"preview_data", None)
        ]

        extractor.extract_previews()

        # Verify cache was called
        assert extractor.preview_cache.has_preview("img1")

    @patch("multiprocessing.Pool")
    def test_extract_previews_force(
        self, mock_pool, extractor: PreviewExtractor
    ) -> None:
        """Test force re-extraction."""
        # Add RAW image
        image = ImageRecord(
            id="img1",
            source_path="/tmp/test.cr2",
            file_size=1000000,
            file_hash="abc",
            checksum="sha256:abc123",
            format="CR2",
            width=6000,
            height=4000,
            file_type=FileType.IMAGE,
        )
        extractor.catalog.add_image(image)

        # Pre-cache a preview
        extractor.preview_cache.store_preview("img1", b"old_data")

        # Mock pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.imap_unordered.return_value = [
            ("img1", b"new_data", None)
        ]

        # Force should re-extract even though cached
        extractor.extract_previews(force=True)

        mock_pool_instance.imap_unordered.assert_called_once()

    @patch("multiprocessing.Pool")
    def test_extract_previews_skip_cached(
        self, mock_pool, extractor: PreviewExtractor
    ) -> None:
        """Test skipping cached previews."""
        # Add RAW image
        image = ImageRecord(
            id="img1",
            source_path="/tmp/test.cr2",
            file_size=1000000,
            file_hash="abc",
            checksum="sha256:abc123",
            format="CR2",
            width=6000,
            height=4000,
            file_type=FileType.IMAGE,
        )
        extractor.catalog.add_image(image)

        # Pre-cache a preview
        extractor.preview_cache.store_preview("img1", b"cached_data")

        extractor.extract_previews(force=False)

        # Should not process anything since already cached
        mock_pool.assert_not_called()

    @patch("multiprocessing.Pool")
    def test_extract_previews_mixed_files(
        self, mock_pool, extractor: PreviewExtractor, tmp_path: Path
    ) -> None:
        """Test extraction with mixed file types."""
        # Add various file types
        images = [
            ImageRecord(
                id="img1",
                source_path="/tmp/test.cr2",
                file_size=1000000,
                file_hash="abc",
            checksum="sha256:abc123",
                format="CR2",
                width=6000,
                height=4000,
                file_type=FileType.IMAGE,
            ),
            ImageRecord(
                id="img2",
                source_path="/tmp/test.jpg",
                file_size=100000,
                file_hash="def",
            checksum="sha256:abc123",
                format="JPEG",
                width=1920,
                height=1080,
                file_type=FileType.IMAGE,
            ),
            ImageRecord(
                id="img3",
                source_path="/tmp/test.heic",
                file_size=200000,
                file_hash="ghi",
            checksum="sha256:abc123",
                format="HEIC",
                width=4000,
                height=3000,
                file_type=FileType.IMAGE,
            ),
            ImageRecord(
                id="vid1",
                source_path="/tmp/test.mp4",
                file_size=5000000,
                file_hash="jkl",
            checksum="sha256:abc123",
                format="MP4",
                file_type=FileType.VIDEO,
            ),
        ]

        for image in images:
            extractor.catalog.add_image(image)

        # Mock pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.imap_unordered.return_value = [
            ("img1", b"preview1", None),
            ("img3", b"preview3", None),
        ]

        extractor.extract_previews()

        # Should process only RAW and HEIC (img1, img3), not JPEG or video
        args_list = mock_pool_instance.imap_unordered.call_args[0][1]
        assert len(args_list) == 2
        processed_ids = {img.id for img, _ in args_list}
        assert "img1" in processed_ids
        assert "img3" in processed_ids

    @patch("multiprocessing.Pool")
    def test_extract_previews_handles_failures(
        self, mock_pool, extractor: PreviewExtractor
    ) -> None:
        """Test handling extraction failures."""
        # Add RAW images
        for i in range(3):
            image = ImageRecord(
                id=f"img{i}",
                source_path=f"/tmp/test{i}.cr2",
                file_size=1000000,
                file_hash=f"abc{i}",
            checksum="sha256:abc123",
                format="CR2",
                width=6000,
                height=4000,
                file_type=FileType.IMAGE,
            )
            extractor.catalog.add_image(image)

        # Mock mixed success/failure
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.imap_unordered.return_value = [
            ("img0", b"preview0", None),
            ("img1", None, "Extraction failed"),
            ("img2", b"preview2", None),
        ]

        extractor.extract_previews()

        # Should continue despite failures
        assert extractor.preview_cache.has_preview("img0")
        assert not extractor.preview_cache.has_preview("img1")
        assert extractor.preview_cache.has_preview("img2")


class TestConstants:
    """Test format constants."""

    def test_raw_formats(self) -> None:
        """Test RAW format list."""
        assert ".cr2" in RAW_FORMATS
        assert ".nef" in RAW_FORMATS
        assert ".arw" in RAW_FORMATS
        assert len(RAW_FORMATS) > 5

    def test_heic_tiff_formats(self) -> None:
        """Test HEIC/TIFF format list."""
        assert ".heic" in HEIC_TIFF_FORMATS
        assert ".heif" in HEIC_TIFF_FORMATS
        assert ".tif" in HEIC_TIFF_FORMATS
        assert ".tiff" in HEIC_TIFF_FORMATS
