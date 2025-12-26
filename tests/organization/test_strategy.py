"""Tests for organization strategy."""

from datetime import datetime
from pathlib import Path

import pytest

from vam_tools.core.types import DateInfo, FileType, ImageMetadata, ImageRecord
from vam_tools.organization.strategy import (
    DirectoryStructure,
    NamingStrategy,
    OrganizationStrategy,
)


@pytest.fixture
def sample_image_with_date():
    """Create a sample image with date information."""
    return ImageRecord(
        id="test123",
        source_path=Path("/source/IMG_1234.jpg"),
        file_type=FileType.IMAGE,
        checksum="abc123",
        metadata=ImageMetadata(
            size_bytes=1024,
            format="JPEG",
        ),
        dates=DateInfo(
            selected_date=datetime(2023, 6, 15, 14, 30, 22),
            exif_dates={"DateTimeOriginal": datetime(2023, 6, 15, 14, 30, 22)},
        ),
    )


@pytest.fixture
def sample_image_no_date():
    """Create a sample image without date information."""
    return ImageRecord(
        id="test456",
        source_path=Path("/source/IMG_5678.jpg"),
        file_type=FileType.IMAGE,
        checksum="def456",
        metadata=ImageMetadata(
            size_bytes=2048,
            format="JPEG",
        ),
    )


class TestDirectoryStructure:
    """Test directory structure generation."""

    def test_year_month_structure(self, sample_image_with_date):
        """Test YYYY-MM directory structure."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_MONTH,
            naming_strategy=NamingStrategy.ORIGINAL,
        )
        base_dir = Path("/output")

        target_dir = strategy.get_target_directory(base_dir, sample_image_with_date)

        assert target_dir == Path("/output/2023-06")

    def test_year_slash_month_structure(self, sample_image_with_date):
        """Test YYYY/MM directory structure."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_SLASH_MONTH,
            naming_strategy=NamingStrategy.ORIGINAL,
        )
        base_dir = Path("/output")

        target_dir = strategy.get_target_directory(base_dir, sample_image_with_date)

        assert target_dir == Path("/output/2023/06")

    def test_year_month_day_structure(self, sample_image_with_date):
        """Test YYYY-MM-DD directory structure."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_MONTH_DAY,
            naming_strategy=NamingStrategy.ORIGINAL,
        )
        base_dir = Path("/output")

        target_dir = strategy.get_target_directory(base_dir, sample_image_with_date)

        assert target_dir == Path("/output/2023-06-15")

    def test_year_only_structure(self, sample_image_with_date):
        """Test YYYY directory structure."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_ONLY,
            naming_strategy=NamingStrategy.ORIGINAL,
        )
        base_dir = Path("/output")

        target_dir = strategy.get_target_directory(base_dir, sample_image_with_date)

        assert target_dir == Path("/output/2023")

    def test_flat_structure(self, sample_image_with_date):
        """Test flat directory structure."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.ORIGINAL,
        )
        base_dir = Path("/output")

        target_dir = strategy.get_target_directory(base_dir, sample_image_with_date)

        assert target_dir == base_dir

    def test_year_slash_month_day_structure(self, sample_image_with_date):
        """Test YYYY/MM-DD directory structure."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_SLASH_MONTH_DAY,
            naming_strategy=NamingStrategy.ORIGINAL,
        )
        base_dir = Path("/output")

        target_dir = strategy.get_target_directory(base_dir, sample_image_with_date)

        assert target_dir == Path("/output/2023/06-15")

    def test_no_date_returns_none(self, sample_image_no_date):
        """Test that images without dates return None for directory."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_MONTH,
            naming_strategy=NamingStrategy.ORIGINAL,
        )
        base_dir = Path("/output")

        target_dir = strategy.get_target_directory(base_dir, sample_image_no_date)

        assert target_dir is None


class TestNamingStrategy:
    """Test file naming strategies."""

    def test_original_naming(self, sample_image_with_date):
        """Test keeping original filename."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.ORIGINAL,
        )

        filename = strategy.get_target_filename(sample_image_with_date)

        assert filename == "IMG_1234.jpg"

    def test_checksum_naming(self, sample_image_with_date):
        """Test checksum-based naming."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.CHECKSUM,
        )

        filename = strategy.get_target_filename(sample_image_with_date)

        assert filename == "abc123.jpg"

    def test_date_time_checksum_naming(self, sample_image_with_date):
        """Test date+time+checksum naming."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.DATE_TIME_CHECKSUM,
        )

        filename = strategy.get_target_filename(sample_image_with_date)

        assert filename == "2023-06-15_143022_abc123.jpg"

    def test_date_time_original_naming(self, sample_image_with_date):
        """Test date+time+original naming."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.DATE_TIME_ORIGINAL,
        )

        filename = strategy.get_target_filename(sample_image_with_date)

        assert filename == "2023-06-15_143022_IMG_1234.jpg"

    def test_time_checksum_naming(self, sample_image_with_date):
        """Test TIME_CHECKSUM naming strategy: HHMMSS_shortchecksum.ext"""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.TIME_CHECKSUM,
        )

        filename = strategy.get_target_filename(sample_image_with_date)

        # Expected: 143022_abc123.jpg (8 char checksum short)
        assert filename == "143022_abc123.jpg"

    def test_date_naming_without_date(self, sample_image_no_date):
        """Test date-based naming falls back when no date available."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.DATE_TIME_CHECKSUM,
        )

        filename = strategy.get_target_filename(sample_image_no_date)

        # Should fall back to checksum when no date
        assert filename == "def456.jpg"

    def test_preserves_extension(self, sample_image_with_date):
        """Test that file extension is preserved."""
        # Update the image to have a different extension
        image = ImageRecord(
            id="test",
            source_path=Path("/source/photo.HEIC"),
            file_type=FileType.IMAGE,
            checksum="abc123",
            metadata=ImageMetadata(
                size_bytes=1024,
                format="HEIC",
            ),
            dates=DateInfo(
                selected_date=datetime(2023, 6, 15, 14, 30, 22),
            ),
        )

        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.DATE_TIME_CHECKSUM,
        )

        filename = strategy.get_target_filename(image)

        assert filename.endswith(".HEIC")

    def test_handles_no_extension(self):
        """Test handling files without extensions."""
        image = ImageRecord(
            id="test",
            source_path=Path("/source/photo"),
            file_type=FileType.IMAGE,
            checksum="abc123",
            metadata=ImageMetadata(
                size_bytes=1024,
                format="JPEG",
            ),
            dates=DateInfo(
                selected_date=datetime(2023, 6, 15, 14, 30, 22),
            ),
        )

        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.ORIGINAL,
        )

        filename = strategy.get_target_filename(image)

        assert filename == "photo"


class TestOrganizationStrategy:
    """Test complete organization strategy."""

    def test_get_target_path(self, sample_image_with_date, tmp_path):
        """Test getting complete target path."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_MONTH,
            naming_strategy=NamingStrategy.DATE_TIME_CHECKSUM,
        )

        target_path = strategy.get_target_path(tmp_path, sample_image_with_date)

        expected = tmp_path / "2023-06" / "2023-06-15_143022_abc123.jpg"
        assert target_path == expected

    def test_get_target_path_no_date_returns_none(self, sample_image_no_date, tmp_path):
        """Test that images without dates return None."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_MONTH,
            naming_strategy=NamingStrategy.DATE_TIME_CHECKSUM,
        )

        target_path = strategy.get_target_path(tmp_path, sample_image_no_date)

        assert target_path is None

    def test_resolve_naming_conflict(self, sample_image_with_date, tmp_path):
        """Test resolving naming conflicts."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.ORIGINAL,
        )

        # Create the target file
        target_path = tmp_path / "IMG_1234.jpg"
        target_path.write_text("existing")

        # Resolve conflict
        new_path = strategy.resolve_naming_conflict(target_path, sample_image_with_date)

        # Should add suffix (format is _001)
        assert new_path == tmp_path / "IMG_1234_001.jpg"

    def test_resolve_multiple_conflicts(self, sample_image_with_date, tmp_path):
        """Test resolving multiple naming conflicts."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.ORIGINAL,
        )

        # Create multiple conflicting files
        (tmp_path / "IMG_1234.jpg").write_text("existing")
        (tmp_path / "IMG_1234_001.jpg").write_text("existing")
        (tmp_path / "IMG_1234_002.jpg").write_text("existing")

        # Resolve conflict
        target_path = tmp_path / "IMG_1234.jpg"
        new_path = strategy.resolve_naming_conflict(target_path, sample_image_with_date)

        # Should find next available suffix
        assert new_path == tmp_path / "IMG_1234_003.jpg"

    def test_handle_duplicates_disabled(self, sample_image_with_date, tmp_path):
        """Test that duplicate handling can be disabled."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.FLAT,
            naming_strategy=NamingStrategy.CHECKSUM,
            handle_duplicates=False,
        )

        # Create the target file
        target_path = tmp_path / "abc123.jpg"
        target_path.write_text("existing")

        # Should return None when duplicate handling is disabled
        new_path = strategy.resolve_naming_conflict(target_path, sample_image_with_date)

        assert new_path is None

    def test_status_based_routing_rejected(self, tmp_path):
        """Test that rejected images route to _rejected/ subdirectory."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_SLASH_MONTH_DAY,
            naming_strategy=NamingStrategy.TIME_CHECKSUM,
        )

        # Create rejected image
        image = ImageRecord(
            id="test123",
            source_path=Path("/source/IMG_1234.jpg"),
            file_type=FileType.IMAGE,
            checksum="abc123def456",
            status_id="rejected",
            metadata=ImageMetadata(size_bytes=1024, format="JPEG"),
            dates=DateInfo(selected_date=datetime(2023, 6, 15, 14, 30, 22)),
        )

        target_path = strategy.get_target_path(tmp_path, image)

        # Should route to _rejected/ subdirectory
        assert target_path == tmp_path / "_rejected" / "2023" / "06-15" / "143022_abc123de.jpg"

    def test_status_based_routing_active(self, tmp_path):
        """Test that active images route to main directory."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_SLASH_MONTH_DAY,
            naming_strategy=NamingStrategy.TIME_CHECKSUM,
        )

        # Create active image
        image = ImageRecord(
            id="test123",
            source_path=Path("/source/IMG_1234.jpg"),
            file_type=FileType.IMAGE,
            checksum="abc123def456",
            status_id="active",
            metadata=ImageMetadata(size_bytes=1024, format="JPEG"),
            dates=DateInfo(selected_date=datetime(2023, 6, 15, 14, 30, 22)),
        )

        target_path = strategy.get_target_path(tmp_path, image)

        # Should route to main directory (no _rejected/)
        assert target_path == tmp_path / "2023" / "06-15" / "143022_abc123de.jpg"

    def test_status_based_routing_no_status(self, tmp_path):
        """Test that images without status_id route to main directory."""
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_SLASH_MONTH_DAY,
            naming_strategy=NamingStrategy.TIME_CHECKSUM,
        )

        # Create image without status_id
        image = ImageRecord(
            id="test123",
            source_path=Path("/source/IMG_1234.jpg"),
            file_type=FileType.IMAGE,
            checksum="abc123def456",
            metadata=ImageMetadata(size_bytes=1024, format="JPEG"),
            dates=DateInfo(selected_date=datetime(2023, 6, 15, 14, 30, 22)),
        )

        target_path = strategy.get_target_path(tmp_path, image)

        # Should route to main directory (default)
        assert target_path == tmp_path / "2023" / "06-15" / "143022_abc123de.jpg"

    def test_mtime_fallback(self, tmp_path):
        """Test that mtime is used when EXIF date is missing."""
        import os

        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_SLASH_MONTH_DAY,
        )

        # Create temp file with known mtime
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"test content")

        # Set mtime to specific timestamp (June 15, 2023 14:30:22)
        target_time = datetime(2023, 6, 15, 14, 30, 22).timestamp()
        os.utime(test_file, (target_time, target_time))

        # Create image without dates (EXIF missing)
        image = ImageRecord(
            id="test123",
            source_path=test_file,
            file_type=FileType.IMAGE,
            checksum="abc123",
            metadata=ImageMetadata(size_bytes=1024, format="JPEG"),
            # dates not specified - will use default empty DateInfo
        )

        output_dir = tmp_path / "output"
        target_dir = strategy.get_target_directory(output_dir, image, use_mtime_fallback=True)

        # Should use mtime and create 2023/06-15 directory
        assert target_dir == output_dir / "2023" / "06-15"
