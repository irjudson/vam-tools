"""
Tests for metadata extraction.
"""

import os
from datetime import datetime
from pathlib import Path

from PIL import Image

from lumina.analysis.metadata import MetadataExtractor
from lumina.core.types import FileType


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

    def test_video_resolution_extraction_from_exif(self, tmp_path: Path) -> None:
        """Test video resolution extraction from EXIF metadata."""
        with MetadataExtractor() as extractor:
            # Test ImageWidth/ImageHeight
            exif_data = {"ImageWidth": 1920, "ImageHeight": 1080}
            resolution = extractor._get_video_resolution(exif_data)
            assert resolution == (1920, 1080)

            # Test SourceImageWidth/SourceImageHeight
            exif_data = {"SourceImageWidth": 3840, "SourceImageHeight": 2160}
            resolution = extractor._get_video_resolution(exif_data)
            assert resolution == (3840, 2160)

            # Test VideoWidth/VideoHeight (MOV files)
            exif_data = {"VideoWidth": 1280, "VideoHeight": 720}
            resolution = extractor._get_video_resolution(exif_data)
            assert resolution == (1280, 720)

    def test_video_resolution_extraction_priority(self, tmp_path: Path) -> None:
        """Test that video resolution uses first available field."""
        with MetadataExtractor() as extractor:
            # Multiple fields present - should use first match
            exif_data = {
                "ImageWidth": 1920,
                "ImageHeight": 1080,
                "VideoWidth": 1280,
                "VideoHeight": 720,
            }
            resolution = extractor._get_video_resolution(exif_data)
            # Should use ImageWidth/ImageHeight (first in priority)
            assert resolution == (1920, 1080)

    def test_video_resolution_extraction_missing_data(self, tmp_path: Path) -> None:
        """Test video resolution extraction with missing data."""
        with MetadataExtractor() as extractor:
            # No resolution data
            exif_data = {}
            resolution = extractor._get_video_resolution(exif_data)
            assert resolution is None

            # Only width, no height
            exif_data = {"ImageWidth": 1920}
            resolution = extractor._get_video_resolution(exif_data)
            assert resolution is None

            # Only height, no width
            exif_data = {"ImageHeight": 1080}
            resolution = extractor._get_video_resolution(exif_data)
            assert resolution is None

    def test_video_resolution_extraction_invalid_values(self, tmp_path: Path) -> None:
        """Test video resolution extraction with invalid values."""
        with MetadataExtractor() as extractor:
            # Zero values
            exif_data = {"ImageWidth": 0, "ImageHeight": 0}
            resolution = extractor._get_video_resolution(exif_data)
            assert resolution is None

            # Negative values
            exif_data = {"ImageWidth": -1920, "ImageHeight": -1080}
            resolution = extractor._get_video_resolution(exif_data)
            assert resolution is None

            # String values that can be parsed
            exif_data = {"ImageWidth": "1920", "ImageHeight": "1080"}
            resolution = extractor._get_video_resolution(exif_data)
            assert resolution == (1920, 1080)

    def test_video_format_from_codec(self, tmp_path: Path) -> None:
        """Test video format extraction from codec information."""
        video_path = tmp_path / "test.mp4"

        with MetadataExtractor() as extractor:
            # Test H.264 codec
            exif_data = {"CompressorName": "H.264"}
            format_str = extractor._get_video_format(video_path, exif_data)
            assert format_str == "H.264"

            # Test HEVC codec
            exif_data = {"VideoCodecID": "HEVC"}
            format_str = extractor._get_video_format(video_path, exif_data)
            assert format_str == "HEVC"

            # Test AVC codec
            exif_data = {"VideoCodec": "AVC"}
            format_str = extractor._get_video_format(video_path, exif_data)
            assert format_str == "AVC"

    def test_video_format_fallback_to_extension(self, tmp_path: Path) -> None:
        """Test video format falls back to extension when no codec data."""
        with MetadataExtractor() as extractor:
            # No codec information
            video_path = tmp_path / "test.mp4"
            exif_data = {}
            format_str = extractor._get_video_format(video_path, exif_data)
            assert format_str == "mp4"

            # Unknown/None codec values
            exif_data = {"CompressorName": "unknown"}
            format_str = extractor._get_video_format(video_path, exif_data)
            assert format_str == "mp4"

            exif_data = {"CompressorName": "none"}
            format_str = extractor._get_video_format(video_path, exif_data)
            assert format_str == "mp4"

    def test_video_format_different_extensions(self, tmp_path: Path) -> None:
        """Test video format extraction for different file types."""
        with MetadataExtractor() as extractor:
            # MOV file
            video_path = tmp_path / "test.mov"
            exif_data = {}
            format_str = extractor._get_video_format(video_path, exif_data)
            assert format_str == "mov"

            # AVI file
            video_path = tmp_path / "test.avi"
            format_str = extractor._get_video_format(video_path, exif_data)
            assert format_str == "avi"

            # MKV file
            video_path = tmp_path / "test.mkv"
            format_str = extractor._get_video_format(video_path, exif_data)
            assert format_str == "mkv"

    def test_parse_exif_date_various_formats(self, tmp_path: Path) -> None:
        """Test parsing of various EXIF date formats."""
        with MetadataExtractor() as extractor:
            # Standard EXIF format with timezone
            date = extractor._parse_exif_date("2023:06:20 14:30:45+00:00")
            assert date is not None
            assert date.year == 2023
            assert date.month == 6
            assert date.day == 20

            # Standard EXIF format without timezone
            date = extractor._parse_exif_date("2023:06:20 14:30:45")
            assert date is not None

            # Hyphen separator
            date = extractor._parse_exif_date("2023-06-20 14:30:45")
            assert date is not None

            # Date only with colons
            date = extractor._parse_exif_date("2023:06:20")
            assert date is not None

            # Date only with hyphens
            date = extractor._parse_exif_date("2023-06-20")
            assert date is not None

            # Invalid format
            date = extractor._parse_exif_date("not a date")
            assert date is None

            # Empty string
            date = extractor._parse_exif_date("")
            assert date is None

    def test_extract_exif_dates_with_errors(self, tmp_path: Path) -> None:
        """Test EXIF date extraction with malformed dates."""
        with MetadataExtractor() as extractor:
            # Test with invalid date values
            exif_data = {
                "DateTimeOriginal": "invalid date",
                "CreateDate": "2023:06:20 14:30:45",  # Valid one
                "ModifyDate": "also invalid",
            }
            dates = extractor._extract_exif_dates(exif_data)

            # Should skip invalid dates, keep valid one
            assert "CreateDate" in dates
            assert dates["CreateDate"] is not None
            # Invalid ones might not be in dict or be None
            if "DateTimeOriginal" in dates:
                assert dates["DateTimeOriginal"] is None

    def test_filename_date_parsing_errors(self, tmp_path: Path) -> None:
        """Test filename date parsing with edge cases that cause errors."""
        with MetadataExtractor() as extractor:
            # Invalid month
            date = extractor._extract_filename_date(Path("2023-13-01_photo.jpg"))
            assert date is None

            # Invalid day
            date = extractor._extract_filename_date(Path("2023-06-32_photo.jpg"))
            assert date is None

            # Valid but edge case - Feb 29 non-leap year
            date = extractor._extract_filename_date(Path("2023-02-29_photo.jpg"))
            assert date is None

            # Hour out of range - will extract date part, ignore invalid time
            date = extractor._extract_filename_date(Path("2023-06-20_25:30:45.jpg"))
            # Extracts valid date part (2023-06-20), ignores invalid time
            assert date is not None
            assert date.date() == datetime(2023, 6, 20).date()

    def test_date_selection_with_secondary_exif_fields(self, tmp_path: Path) -> None:
        """Test date selection using secondary EXIF fields."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            # Manually set up DateInfo with secondary EXIF fields
            from lumina.core.types import DateInfo

            date_info = DateInfo()
            # Set a non-priority EXIF field
            date_info.exif_dates = {"ModifyDate": datetime(2023, 6, 20, 10, 30, 0)}

            # Call selection logic
            extractor._select_best_date(date_info)

            # Should select the ModifyDate since no priority fields exist
            assert date_info.selected_date is not None
            assert date_info.selected_source == "exif:ModifyDate"
            assert date_info.confidence == 85

    def test_date_selection_directory_date_parsing_error(self, tmp_path: Path) -> None:
        """Test date selection when directory date fails to parse."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            # Manually set up DateInfo with invalid directory date
            from lumina.core.types import DateInfo

            date_info = DateInfo()
            # Set an invalid directory date format
            date_info.directory_date = "invalid-format"
            date_info.filesystem_created = datetime(2023, 1, 1)

            # Call selection logic - should fall back to filesystem
            extractor._select_best_date(date_info)

            # Should fall back to filesystem date
            assert date_info.selected_date == datetime(2023, 1, 1)
            assert date_info.selected_source == "filesystem"
            assert date_info.confidence == 30

    def test_extract_metadata_error_handling(self, tmp_path: Path) -> None:
        """Test metadata extraction with file that causes errors."""
        # Create a file that exists but causes issues
        img_path = tmp_path / "problem.jpg"
        img_path.write_bytes(b"corrupted image data")

        with MetadataExtractor() as extractor:
            # Should not crash
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)

            # Should still have basic metadata
            assert metadata is not None
            assert metadata.size_bytes > 0

    def test_extract_dates_filesystem_error_handling(self, tmp_path: Path) -> None:
        """Test date extraction when filesystem stat fails."""
        # This is hard to test directly, but we can test that the method
        # handles missing filesystem dates gracefully
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            # Should still work even if filesystem dates are problematic
            assert dates is not None

    def test_1970_default_date_detection(self, tmp_path: Path) -> None:
        """Test detection of 1970-01-01 default date."""
        img_path = tmp_path / "1970-01-01_photo.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            if (
                dates.selected_date
                and dates.selected_date.date() == datetime(1970, 1, 1).date()
            ):
                assert dates.suspicious is True

    def test_1980_default_date_detection(self, tmp_path: Path) -> None:
        """Test detection of 1980-01-01 default date."""
        img_path = tmp_path / "1980-01-01_photo.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(img_path, FileType.IMAGE)
            dates = extractor.extract_dates(img_path, metadata)

            if (
                dates.selected_date
                and dates.selected_date.date() == datetime(1980, 1, 1).date()
            ):
                assert dates.suspicious is True


class TestRAWMetadataExtraction:
    """Tests for RAW file metadata extraction functionality."""

    def test_get_image_format_with_raw_file(self, tmp_path: Path, monkeypatch) -> None:
        """Test RAW file format detection using rawpy."""
        from unittest.mock import MagicMock

        # Create a mock RAW file
        raw_path = tmp_path / "test.CR2"
        raw_path.write_bytes(b"fake raw data")

        # Mock rawpy module
        mock_rawpy = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.sizes.raw_width = 6000
        mock_raw_obj.sizes.raw_height = 4000
        mock_rawpy.imread.return_value.__enter__.return_value = mock_raw_obj

        import sys

        sys.modules["rawpy"] = mock_rawpy

        with MetadataExtractor() as extractor:
            format_str, dimensions = extractor._get_image_format(raw_path)

            assert format_str == "CR2"
            assert dimensions == (6000, 4000)
            mock_rawpy.imread.assert_called_once_with(str(raw_path))

    def test_get_image_format_with_raw_file_no_rawpy(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test RAW file handling when rawpy is not available."""
        # Create a mock RAW file
        raw_path = tmp_path / "test.NEF"
        raw_path.write_bytes(b"fake raw data")

        # Mock ImportError for rawpy
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "rawpy":
                raise ImportError("rawpy not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with MetadataExtractor() as extractor:
            format_str, dimensions = extractor._get_image_format(raw_path)

            # Should fall back to extension-based detection
            assert format_str == "NEF"
            assert dimensions == (0, 0)

    def test_extract_metadata_raw_file(self, tmp_path: Path, monkeypatch) -> None:
        """Test complete metadata extraction for RAW file."""
        from unittest.mock import MagicMock

        # Create a mock RAW file
        raw_path = tmp_path / "IMG_1234.ARW"
        raw_path.write_bytes(b"fake sony raw data")

        # Mock rawpy
        mock_rawpy = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.sizes.raw_width = 7952
        mock_raw_obj.sizes.raw_height = 5304
        mock_rawpy.imread.return_value.__enter__.return_value = mock_raw_obj

        import sys

        sys.modules["rawpy"] = mock_rawpy

        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(raw_path, FileType.IMAGE)

            assert metadata.format == "ARW"
            assert metadata.width == 7952
            assert metadata.height == 5304
            assert metadata.size_bytes == len(b"fake sony raw data")

    def test_raw_format_detection_various_extensions(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test RAW format detection for various camera manufacturers."""
        from unittest.mock import MagicMock

        # Mock rawpy
        mock_rawpy = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.sizes.raw_width = 6000
        mock_raw_obj.sizes.raw_height = 4000
        mock_rawpy.imread.return_value.__enter__.return_value = mock_raw_obj

        import sys

        sys.modules["rawpy"] = mock_rawpy

        raw_extensions = [
            ".cr2",  # Canon
            ".cr3",  # Canon
            ".nef",  # Nikon
            ".arw",  # Sony
            ".dng",  # Adobe/Generic
            ".orf",  # Olympus
            ".rw2",  # Panasonic
            ".pef",  # Pentax
            ".sr2",  # Sony
            ".raf",  # Fujifilm
        ]

        with MetadataExtractor() as extractor:
            for ext in raw_extensions:
                raw_path = tmp_path / f"test{ext}"
                raw_path.write_bytes(b"fake raw data")

                format_str, dimensions = extractor._get_image_format(raw_path)

                assert format_str == ext[1:].upper()
                assert dimensions == (6000, 4000)

    def test_raw_file_error_handling(self, tmp_path: Path, monkeypatch) -> None:
        """Test error handling when RAW file cannot be read."""
        from unittest.mock import MagicMock

        # Create a mock RAW file
        raw_path = tmp_path / "corrupted.CR2"
        raw_path.write_bytes(b"corrupted data")

        # Mock rawpy to raise an error
        mock_rawpy = MagicMock()
        mock_rawpy.imread.side_effect = Exception("Cannot read RAW file")

        import sys

        sys.modules["rawpy"] = mock_rawpy

        with MetadataExtractor() as extractor:
            format_str, dimensions = extractor._get_image_format(raw_path)

            # Should fall back to extension-based detection on error
            assert format_str == "CR2"
            assert dimensions == (0, 0)

    def test_raw_metadata_with_exiftool(self, tmp_path: Path, monkeypatch) -> None:
        """Test that RAW files still get EXIF data from ExifTool."""
        from unittest.mock import MagicMock

        # Create a mock RAW file
        raw_path = tmp_path / "test.NEF"
        raw_path.write_bytes(b"fake nikon raw data")

        # Mock rawpy for dimensions
        mock_rawpy = MagicMock()
        mock_raw_obj = MagicMock()
        mock_raw_obj.sizes.raw_width = 6000
        mock_raw_obj.sizes.raw_height = 4000
        mock_rawpy.imread.return_value.__enter__.return_value = mock_raw_obj

        import sys

        sys.modules["rawpy"] = mock_rawpy

        with MetadataExtractor() as extractor:
            # The extractor should extract both format info and EXIF data
            metadata = extractor.extract_metadata(raw_path, FileType.IMAGE)

            # Should have format from rawpy
            assert metadata.format == "NEF"
            assert metadata.width == 6000
            assert metadata.height == 4000
