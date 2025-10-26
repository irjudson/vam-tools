"""
Tests for core/utils.py (backward compatibility re-exports).
"""

from pathlib import Path

from vam_tools.core.utils import (
    compute_checksum,
    format_bytes,
    get_file_type,
    is_image_file,
    is_video_file,
    safe_filename,
    verify_checksum,
)


class TestUtilsReExports:
    """Tests that utils.py correctly re-exports shared utilities."""

    def test_reexports_exist(self) -> None:
        """Test that all functions are re-exported."""
        # Just verify imports work - actual functionality tested in test_media_utils
        assert callable(compute_checksum)
        assert callable(verify_checksum)
        assert callable(format_bytes)
        assert callable(safe_filename)
        assert callable(is_image_file)
        assert callable(is_video_file)
        assert callable(get_file_type)

    def test_compute_checksum_works(self, tmp_path) -> None:
        """Test compute_checksum via re-export."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        checksum = compute_checksum(test_file)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length

    def test_format_bytes_works(self) -> None:
        """Test format_bytes via re-export."""
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024 * 1024) == "1.00 MB"

    def test_file_type_detection_works(self) -> None:
        """Test file type detection via re-export."""
        assert is_image_file(Path("test.jpg"))
        assert is_image_file(Path("test.PNG"))
        assert is_video_file(Path("test.mp4"))
        assert is_video_file(Path("test.MOV"))
        assert not is_image_file(Path("test.mp4"))
        assert not is_video_file(Path("test.jpg"))
