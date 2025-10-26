"""
Tests for shared media utilities.

Consolidates tests from V1 and V2 to ensure shared module works correctly.
"""

from pathlib import Path

from PIL import Image

from vam_tools.shared import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    collect_image_files,
    compute_checksum,
    format_bytes,
    get_file_type,
    get_image_info,
    is_image_file,
    is_video_file,
    safe_filename,
    setup_logging,
    verify_checksum,
)


class TestFileTypeDetection:
    """Test file type detection functions."""

    def test_image_extensions_constant(self):
        """Test that IMAGE_EXTENSIONS contains expected values."""
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".heic" in IMAGE_EXTENSIONS
        assert ".raw" in IMAGE_EXTENSIONS

    def test_video_extensions_constant(self):
        """Test that VIDEO_EXTENSIONS contains expected values."""
        assert ".mp4" in VIDEO_EXTENSIONS
        assert ".mov" in VIDEO_EXTENSIONS
        assert ".avi" in VIDEO_EXTENSIONS
        assert ".mkv" in VIDEO_EXTENSIONS

    def test_is_image_file(self):
        """Test image file detection."""
        assert is_image_file(Path("photo.jpg"))
        assert is_image_file(Path("photo.JPEG"))
        assert is_image_file(Path("photo.png"))
        assert is_image_file(Path("photo.heic"))
        assert not is_image_file(Path("video.mp4"))
        assert not is_image_file(Path("document.txt"))

    def test_is_video_file(self):
        """Test video file detection."""
        assert is_video_file(Path("video.mp4"))
        assert is_video_file(Path("video.MOV"))
        assert is_video_file(Path("video.avi"))
        assert not is_video_file(Path("photo.jpg"))
        assert not is_video_file(Path("document.txt"))

    def test_get_file_type(self):
        """Test file type classification."""
        assert get_file_type(Path("photo.jpg")) == "image"
        assert get_file_type(Path("video.mp4")) == "video"
        assert get_file_type(Path("document.txt")) == "unknown"


class TestChecksumOperations:
    """Test checksum computation and verification."""

    def test_compute_checksum_sha256(self, tmp_path):
        """Test SHA256 checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = compute_checksum(test_file, algorithm="sha256")
        assert checksum is not None
        assert len(checksum) == 64  # SHA256 produces 64 hex characters

    def test_compute_checksum_md5(self, tmp_path):
        """Test MD5 checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = compute_checksum(test_file, algorithm="md5")
        assert checksum is not None
        assert len(checksum) == 32  # MD5 produces 32 hex characters

    def test_compute_checksum_large_file(self, tmp_path):
        """Test checksum computation with chunked reading."""
        test_file = tmp_path / "large.bin"
        # Create a file larger than chunk size (8192 bytes)
        test_file.write_bytes(b"x" * 100000)

        checksum = compute_checksum(test_file)
        assert checksum is not None

    def test_compute_checksum_nonexistent(self, tmp_path):
        """Test checksum of non-existent file returns None."""
        nonexistent = tmp_path / "nonexistent.txt"
        checksum = compute_checksum(nonexistent)
        assert checksum is None

    def test_verify_checksum(self, tmp_path):
        """Test checksum verification."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = compute_checksum(test_file)
        assert verify_checksum(test_file, checksum)
        assert not verify_checksum(test_file, "invalid_checksum")

    def test_verify_checksum_case_insensitive(self, tmp_path):
        """Test checksum verification is case-insensitive."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = compute_checksum(test_file)
        assert verify_checksum(test_file, checksum.upper())
        assert verify_checksum(test_file, checksum.lower())


class TestFormatting:
    """Test formatting utilities."""

    def test_format_bytes_zero(self):
        """Test formatting zero bytes."""
        assert format_bytes(0) == "0 B"

    def test_format_bytes_bytes(self):
        """Test formatting bytes."""
        assert format_bytes(500) == "500.00 B"

    def test_format_bytes_kilobytes(self):
        """Test formatting kilobytes."""
        result = format_bytes(1024)
        assert "1.00 KB" in result

    def test_format_bytes_megabytes(self):
        """Test formatting megabytes."""
        result = format_bytes(1024 * 1024)
        assert "1.00 MB" in result

    def test_format_bytes_gigabytes(self):
        """Test formatting gigabytes."""
        result = format_bytes(1024 * 1024 * 1024)
        assert "1.00 GB" in result

    def test_safe_filename_basic(self):
        """Test safe filename conversion."""
        assert safe_filename("normal_file.txt") == "normal_file.txt"

    def test_safe_filename_unsafe_chars(self):
        """Test safe filename removes unsafe characters."""
        result = safe_filename('file<>:"/\\|?*.txt')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result

    def test_safe_filename_trim_spaces(self):
        """Test safe filename trims spaces and dots."""
        assert safe_filename("  file.txt  ") == "file.txt"
        assert safe_filename("..file..") == "file"

    def test_safe_filename_long_name(self):
        """Test safe filename limits length."""
        long_name = "x" * 300
        result = safe_filename(long_name)
        assert len(result) <= 255

    def test_safe_filename_empty(self):
        """Test safe filename handles empty string."""
        assert safe_filename("") == "unnamed"
        assert safe_filename("   ") == "unnamed"


class TestImageOperations:
    """Test image-specific operations."""

    def test_get_image_info(self, tmp_path):
        """Test getting image information."""
        image_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 200), color="red")
        img.save(image_path)

        info = get_image_info(image_path)
        assert info is not None
        assert info["dimensions"] == (100, 200)
        assert info["format"] in ["JPEG", "JPG"]
        assert info["mode"] == "RGB"
        assert info["file_size"] > 0

    def test_get_image_info_invalid(self, tmp_path):
        """Test getting info from invalid image returns None."""
        text_file = tmp_path / "not_image.jpg"
        text_file.write_text("Not an image")

        info = get_image_info(text_file)
        assert info is None

    def test_collect_image_files_recursive(self, tmp_path):
        """Test collecting image files recursively."""
        # Create directory structure with images
        (tmp_path / "subdir").mkdir()
        Image.new("RGB", (10, 10)).save(tmp_path / "image1.jpg")
        Image.new("RGB", (10, 10)).save(tmp_path / "image2.png")
        Image.new("RGB", (10, 10)).save(tmp_path / "subdir" / "image3.jpg")
        (tmp_path / "text.txt").write_text("Not an image")

        files = collect_image_files(tmp_path, recursive=True)
        assert len(files) == 3
        assert all(f.suffix.lower() in IMAGE_EXTENSIONS for f in files)

    def test_collect_image_files_non_recursive(self, tmp_path):
        """Test collecting image files non-recursively."""
        (tmp_path / "subdir").mkdir()
        Image.new("RGB", (10, 10)).save(tmp_path / "image1.jpg")
        Image.new("RGB", (10, 10)).save(tmp_path / "subdir" / "image2.jpg")

        files = collect_image_files(tmp_path, recursive=False)
        assert len(files) == 1

    def test_collect_image_files_nonexistent(self, tmp_path):
        """Test collecting from non-existent directory."""
        nonexistent = tmp_path / "nonexistent"
        files = collect_image_files(nonexistent)
        assert files == []

    def test_collect_image_files_not_directory(self, tmp_path):
        """Test collecting from file (not directory)."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        files = collect_image_files(file_path)
        assert files == []


class TestLogging:
    """Test logging setup."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        # Should not raise an error
        setup_logging()

    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        setup_logging(verbose=True)

    def test_setup_logging_quiet(self):
        """Test quiet logging setup."""
        setup_logging(quiet=True)
