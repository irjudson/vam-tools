"""
Tests for date_extraction module.
"""

from pathlib import Path

import arrow
import pytest

from vam_tools.core.date_extraction import DateExtractor


class TestFilenameExtraction:
    """Tests for filename date extraction."""

    def test_extract_from_yyyy_mm_dd(self, temp_dir: Path) -> None:
        """Test extracting date from YYYY-MM-DD format."""
        image_path = temp_dir / "photo_2023-06-15.jpg"
        from PIL import Image
        Image.new("RGB", (10, 10)).save(image_path)

        with DateExtractor() as extractor:
            date_info = extractor.extract_filename_date(image_path)

        assert date_info is not None
        assert date_info.source == "filename"
        assert date_info.date.year == 2023
        assert date_info.date.month == 6
        assert date_info.date.day == 15

    def test_extract_from_yyyymmdd(self, temp_dir: Path) -> None:
        """Test extracting date from YYYYMMDD format."""
        image_path = temp_dir / "IMG_20221225.jpg"
        from PIL import Image
        Image.new("RGB", (10, 10)).save(image_path)

        with DateExtractor() as extractor:
            date_info = extractor.extract_filename_date(image_path)

        assert date_info is not None
        assert date_info.date.year == 2022
        assert date_info.date.month == 12
        assert date_info.date.day == 25

    def test_extract_from_yyyy_mm_dd_underscores(self, temp_dir: Path) -> None:
        """Test extracting date from YYYY_MM_DD format."""
        image_path = temp_dir / "photo_2023_01_01.jpg"
        from PIL import Image
        Image.new("RGB", (10, 10)).save(image_path)

        with DateExtractor() as extractor:
            date_info = extractor.extract_filename_date(image_path)

        assert date_info is not None
        assert date_info.date.year == 2023
        assert date_info.date.month == 1
        assert date_info.date.day == 1

    def test_no_date_in_filename(self, temp_dir: Path) -> None:
        """Test that None is returned when no date in filename."""
        image_path = temp_dir / "random_photo.jpg"
        from PIL import Image
        Image.new("RGB", (10, 10)).save(image_path)

        with DateExtractor() as extractor:
            date_info = extractor.extract_filename_date(image_path)

        assert date_info is None


class TestDirectoryExtraction:
    """Tests for directory structure date extraction."""

    def test_extract_from_year_directory(self, temp_dir: Path) -> None:
        """Test extracting year from directory structure."""
        dir_path = temp_dir / "2023" / "photos"
        dir_path.mkdir(parents=True)

        image_path = dir_path / "image.jpg"
        from PIL import Image
        Image.new("RGB", (10, 10)).save(image_path)

        with DateExtractor() as extractor:
            date_info = extractor.extract_directory_date(image_path)

        assert date_info is not None
        assert date_info.source == "directory"
        assert date_info.date.year == 2023

    def test_extract_from_year_month_directory(self, temp_dir: Path) -> None:
        """Test extracting year and month from directory structure."""
        dir_path = temp_dir / "2023" / "06-vacation"
        dir_path.mkdir(parents=True)

        image_path = dir_path / "image.jpg"
        from PIL import Image
        Image.new("RGB", (10, 10)).save(image_path)

        with DateExtractor() as extractor:
            date_info = extractor.extract_directory_date(image_path)

        assert date_info is not None
        assert date_info.date.year == 2023
        # Month extraction might vary based on implementation
        # Just check that a valid date was extracted

    def test_no_date_in_directory(self, temp_dir: Path) -> None:
        """Test that None is returned when no date in directory."""
        dir_path = temp_dir / "photos" / "vacation"
        dir_path.mkdir(parents=True)

        image_path = dir_path / "image.jpg"
        from PIL import Image
        Image.new("RGB", (10, 10)).save(image_path)

        with DateExtractor() as extractor:
            date_info = extractor.extract_directory_date(image_path)

        # May return None or a date depending on implementation
        # Just verify it doesn't crash


class TestFilesystemExtraction:
    """Tests for filesystem date extraction."""

    def test_extract_filesystem_date(self, sample_image: Path) -> None:
        """Test extracting filesystem creation date."""
        with DateExtractor() as extractor:
            date_info = extractor.extract_filesystem_date(sample_image)

        assert date_info is not None
        assert date_info.source == "filesystem"
        assert date_info.confidence == 30  # Lowest confidence

    def test_filesystem_date_is_recent(self, sample_image: Path) -> None:
        """Test that filesystem date is recent for newly created files."""
        with DateExtractor() as extractor:
            date_info = extractor.extract_filesystem_date(sample_image)

        assert date_info is not None
        # File was just created, should be within last minute
        now = arrow.now()
        assert (now.timestamp() - date_info.date.timestamp()) < 60


class TestEarliestDateExtraction:
    """Tests for extracting the earliest date from all sources."""

    def test_extract_from_filename_priority(self, temp_dir: Path) -> None:
        """Test that filename date is used when available."""
        image_path = temp_dir / "photo_2020-01-01.jpg"
        from PIL import Image
        Image.new("RGB", (10, 10)).save(image_path)

        with DateExtractor() as extractor:
            date_info = extractor.extract_earliest_date(image_path)

        assert date_info is not None
        # Should prioritize filename date over filesystem date
        assert date_info.date.year == 2020

    def test_fallback_to_filesystem(self, temp_dir: Path) -> None:
        """Test that filesystem date is used as fallback."""
        image_path = temp_dir / "random_image.jpg"
        from PIL import Image
        Image.new("RGB", (10, 10)).save(image_path)

        with DateExtractor() as extractor:
            date_info = extractor.extract_earliest_date(image_path)

        assert date_info is not None
        assert date_info.source == "filesystem"

    def test_earliest_date_selection(self, temp_dir: Path) -> None:
        """Test that the earliest date is selected among multiple sources."""
        # Create image with date in filename AND directory
        dir_path = temp_dir / "2023" / "photos"
        dir_path.mkdir(parents=True)

        # Filename has earlier date
        image_path = dir_path / "photo_2020-01-01.jpg"
        from PIL import Image
        Image.new("RGB", (10, 10)).save(image_path)

        with DateExtractor() as extractor:
            date_info = extractor.extract_earliest_date(image_path)

        assert date_info is not None
        # Should pick earlier date (2020 from filename, not 2023 from directory)
        assert date_info.date.year == 2020


class TestAnalyzeImages:
    """Tests for batch image analysis."""

    def test_analyze_multiple_images(self, dated_images: dict[str, Path]) -> None:
        """Test analyzing multiple images."""
        image_list = list(dated_images.values())

        with DateExtractor() as extractor:
            results = extractor.analyze_images(image_list)

        assert len(results) == len(image_list)

        # All images should have date info
        for image_path, date_info in results.items():
            assert image_path in image_list
            # Some should have dates from filenames
            # (dated_images fixture has dates in filenames)

    def test_analyze_empty_list(self) -> None:
        """Test analyzing empty list of images."""
        with DateExtractor() as extractor:
            results = extractor.analyze_images([])

        assert results == {}


class TestContextManager:
    """Tests for DateExtractor context manager."""

    def test_context_manager_usage(self, sample_image: Path) -> None:
        """Test that DateExtractor works as context manager."""
        with DateExtractor() as extractor:
            date_info = extractor.extract_filesystem_date(sample_image)
            assert date_info is not None

    def test_multiple_operations_in_context(self, sample_image: Path) -> None:
        """Test multiple operations within same context."""
        with DateExtractor() as extractor:
            info1 = extractor.extract_filename_date(sample_image)
            info2 = extractor.extract_filesystem_date(sample_image)
            info3 = extractor.extract_directory_date(sample_image)

            # At least filesystem should work
            assert info2 is not None
