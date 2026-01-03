"""Tests for reorganization helper functions."""

from datetime import datetime
from pathlib import Path

import pytest

from lumina.core.types import DateInfo, FileType, ImageMetadata, ImageRecord
from lumina.organization.reorganizer import should_reorganize_image


class TestShouldReorganizeImage:
    """Test should_reorganize_image function."""

    def test_skip_already_organized(self):
        """Test that files already in organized structure are skipped."""
        output_dir = Path("/organized")

        # Image already in organized structure
        image = ImageRecord(
            id="test123",
            source_path=Path("/organized/2023/06-15/143022_abc123de.jpg"),
            file_type=FileType.IMAGE,
            checksum="abc123def456",
            metadata=ImageMetadata(size_bytes=1024, format="JPEG"),
        )

        assert should_reorganize_image(image, output_dir) is False

    def test_skip_matching_checksum(self, tmp_path):
        """Test that files with matching checksum at target are skipped."""
        output_dir = tmp_path

        # Create target file
        target_dir = output_dir / "2023" / "06-15"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "143022_abc123de.jpg"
        target_file.write_bytes(b"test content")

        # Calculate checksum of target
        from lumina.shared.media_utils import compute_checksum

        target_checksum = compute_checksum(target_file)

        # Image with same checksum
        image = ImageRecord(
            id="test123",
            source_path=Path("/source/IMG_1234.jpg"),
            file_type=FileType.IMAGE,
            checksum=target_checksum,
            metadata=ImageMetadata(size_bytes=1024, format="JPEG"),
        )

        # Should skip
        assert should_reorganize_image(image, output_dir, target_file) is False

    def test_reorganize_new_file(self):
        """Test that new files should be reorganized."""
        output_dir = Path("/organized")

        # Image not in organized structure
        image = ImageRecord(
            id="test123",
            source_path=Path("/source/IMG_1234.jpg"),
            file_type=FileType.IMAGE,
            checksum="abc123def456",
            metadata=ImageMetadata(size_bytes=1024, format="JPEG"),
        )

        assert should_reorganize_image(image, output_dir) is True
