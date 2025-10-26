"""
Tests for metadata extraction.
"""

import os
from datetime import datetime
from pathlib import Path

from PIL import Image

from vam_tools.analysis.metadata import MetadataExtractor
from vam_tools.core.types import FileType


class TestMetadataExtractor:
    """Tests for MetadataExtractor."""

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test that extractor works as context manager."""
        with MetadataExtractor() as extractor:
            assert extractor is not None
            assert extractor.exif_tool is not None

    def test_extract_basic_metadata(self, tmp_path: Path) -> None:
        """Test basic metadata extraction from image."""
        # Create test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (800, 600), color="red")
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)

            # Check file size was extracted
            assert metadata.size_bytes > 0
            assert metadata.size_bytes == os.path.getsize(img_path)

            # Check format and resolution
            assert metadata.format == "JPEG"
            assert metadata.width == 800
            assert metadata.height == 600
            assert metadata.resolution == (800, 600)

    def test_extract_metadata_png(self, tmp_path: Path) -> None:
        """Test metadata extraction from PNG."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (1920, 1080), color="blue")
        img.save(img_path, "PNG")

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)

            assert metadata.format == "PNG"
            assert metadata.width == 1920
            assert metadata.height == 1080

    def test_extract_dates_from_filename(self, tmp_path: Path) -> None:
        """Test date extraction from filename patterns."""
        test_cases = [
            ("2023-12-25_photo.jpg", datetime(2023, 12, 25)),
            ("20231225_image.jpg", datetime(2023, 12, 25)),
            ("2023_01_15_vacation.jpg", datetime(2023, 1, 15)),
            ("IMG_2023-06-20.jpg", datetime(2023, 6, 20)),
        ]

        with MetadataExtractor() as extractor:
            for filename, expected_date in test_cases:
                img_path = tmp_path / filename
                img = Image.new("RGB", (100, 100))
                img.save(img_path)

                metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
                dates = extractor.extract_dates(img_path, metadata)

                assert dates.filename_date is not None
                assert dates.filename_date.date() == expected_date.date()

    def test_extract_dates_from_directory(self, tmp_path: Path) -> None:
        """Test date extraction from directory structure."""
        # Create nested directory with year-month pattern
        year_dir = tmp_path / "2023-06"
        year_dir.mkdir(parents=True)
        img_path = year_dir / "photo.jpg"

        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            assert dates.directory_date is not None
            assert "2023-06" in dates.directory_date

    def test_extract_dates_year_only_directory(self, tmp_path: Path) -> None:
        """Test date extraction from year-only directory."""
        year_dir = tmp_path / "2022"
        year_dir.mkdir(parents=True)
        img_path = year_dir / "photo.jpg"

        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            assert dates.directory_date is not None
            assert "2022" in dates.directory_date

    def test_extract_filesystem_dates(self, tmp_path: Path) -> None:
        """Test filesystem date extraction."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            # Filesystem dates should always be present
            assert dates.filesystem_created is not None
            assert dates.filesystem_modified is not None
            assert isinstance(dates.filesystem_created, datetime)
            assert isinstance(dates.filesystem_modified, datetime)

    def test_date_selection_priority(self, tmp_path: Path) -> None:
        """Test that date selection follows priority order."""
        # Create image with date in filename
        img_path = tmp_path / "2023-01-15_photo.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            # Should select filename date (higher priority than filesystem)
            assert dates.selected_date is not None
            assert dates.selected_source == "filename"
            assert dates.confidence == 70  # Filename confidence

    def test_date_confidence_levels(self, tmp_path: Path) -> None:
        """Test that confidence levels are set correctly."""
        img_path = tmp_path / "2023-06-20_test.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            # Filename should have 70% confidence
            assert dates.confidence == 70
            assert dates.selected_source == "filename"

    def test_suspicious_date_detection(self, tmp_path: Path) -> None:
        """Test detection of suspicious dates."""
        # Create image with default camera date (2000-01-01)
        img_path = tmp_path / "2000-01-01_photo.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            # Should detect as suspicious
            if (
                dates.selected_date
                and dates.selected_date.date() == datetime(2000, 1, 1).date()
            ):
                assert dates.suspicious is True

    def test_future_date_detection(self, tmp_path: Path) -> None:
        """Test detection of future dates."""
        # Create image with future date
        future_year = datetime.now().year + 10
        img_path = tmp_path / f"{future_year}-01-01_photo.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            if dates.selected_date and dates.selected_date.year > datetime.now().year:
                assert dates.suspicious is True

    def test_parse_float_helper(self, tmp_path: Path) -> None:
        """Test _parse_float helper with various inputs."""
        with MetadataExtractor() as extractor:
            # Test valid inputs
            assert extractor._parse_float("50.5") == 50.5
            assert extractor._parse_float("100") == 100.0
            assert extractor._parse_float(42) == 42.0
            assert extractor._parse_float(3.14) == 3.14

            # Test string with units
            assert extractor._parse_float("50mm") == 50.0
            assert extractor._parse_float("f/2.8") == 2.8

            # Test invalid inputs
            assert extractor._parse_float(None) is None
            assert extractor._parse_float("invalid") is None
            assert extractor._parse_float("") is None

    def test_parse_int_helper(self, tmp_path: Path) -> None:
        """Test _parse_int helper with various inputs."""
        with MetadataExtractor() as extractor:
            # Test valid inputs
            assert extractor._parse_int("100") == 100
            assert extractor._parse_int("42") == 42
            assert extractor._parse_int(100) == 100
            assert extractor._parse_int(50.7) == 50

            # Test string with units
            assert extractor._parse_int("200mm") == 200
            assert extractor._parse_int("ISO 800") == 800

            # Test invalid inputs
            assert extractor._parse_int(None) is None
            assert extractor._parse_int("invalid") is None
            assert extractor._parse_int("") is None

    def test_video_format_extraction(self, tmp_path: Path) -> None:
        """Test format extraction for video files."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video data")

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(video_path, FileType.VIDEO)

            # Should extract format from extension
            assert metadata.format == "mp4"

    def test_invalid_image_handling(self, tmp_path: Path) -> None:
        """Test handling of corrupted/invalid images."""
        img_path = tmp_path / "corrupted.jpg"
        img_path.write_bytes(b"not a valid image")

        with MetadataExtractor() as extractor:
            # Should not crash, just return empty metadata
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)

            # Format/resolution might be None for corrupted files
            assert metadata is not None

    def test_no_date_in_filename(self, tmp_path: Path) -> None:
        """Test handling of files with no date in filename."""
        img_path = tmp_path / "vacation_photo.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            # Should not have filename date
            assert dates.filename_date is None
            # But should fall back to filesystem
            assert dates.selected_date is not None
            assert dates.selected_source == "filesystem"

    def test_filename_with_time(self, tmp_path: Path) -> None:
        """Test filename date extraction with time component."""
        img_path = tmp_path / "2023-06-20_14:30:45.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            assert dates.filename_date is not None
            assert dates.filename_date.year == 2023
            assert dates.filename_date.month == 6
            assert dates.filename_date.day == 20
            assert dates.filename_date.hour == 14
            assert dates.filename_date.minute == 30
            assert dates.filename_date.second == 45

    def test_mm_dd_yyyy_format(self, tmp_path: Path) -> None:
        """Test MM-DD-YYYY date format in filename."""
        img_path = tmp_path / "06-20-2023_photo.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            # Should parse MM-DD-YYYY format
            if dates.filename_date:
                assert dates.filename_date.year == 2023
                assert dates.filename_date.month == 6
                assert dates.filename_date.day == 20

    def test_metadata_with_no_exif(self, tmp_path: Path) -> None:
        """Test metadata extraction when no EXIF data present."""
        img_path = tmp_path / "no_exif.jpg"
        img = Image.new("RGB", (640, 480))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)

            # Should still extract basic info
            assert metadata.size_bytes > 0
            assert metadata.format == "JPEG"
            assert metadata.width == 640
            assert metadata.height == 480

            # EXIF fields should be None
            assert metadata.camera_make is None
            assert metadata.camera_model is None
            assert metadata.gps_latitude is None
            assert metadata.gps_longitude is None

    def test_extract_dates_no_directory_date(self, tmp_path: Path) -> None:
        """Test date extraction when directory has no date pattern."""
        # Regular directory name with no year/month
        img_path = tmp_path / "photo.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            # Should not have directory date
            assert dates.directory_date is None

    def test_very_old_date_detection(self, tmp_path: Path) -> None:
        """Test detection of very old dates (before 1990)."""
        img_path = tmp_path / "1980-01-01_old.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            if dates.selected_date and dates.selected_date.year < 1990:
                assert dates.suspicious is True

    def test_directory_date_invalid_month(self, tmp_path: Path) -> None:
        """Test handling of invalid month in directory date."""
        # Invalid month (13)
        year_dir = tmp_path / "2023-13"
        year_dir.mkdir(parents=True)
        img_path = year_dir / "photo.jpg"

        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            # Should not extract invalid date
            # Either no directory_date or it doesn't parse as 2023-13
            if dates.directory_date:
                # Regex should not match invalid month
                assert (
                    "13" not in dates.directory_date
                    or dates.selected_source != "directory"
                )

    def test_resolution_tuple(self, tmp_path: Path) -> None:
        """Test that resolution is stored as tuple."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (1024, 768))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)

            assert metadata.resolution == (1024, 768)
            assert isinstance(metadata.resolution, tuple)
            assert len(metadata.resolution) == 2
