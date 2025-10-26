"""
Tests for image_utils module.
"""

from pathlib import Path

from vam_tools.core.image_utils import (
    collect_image_files,
    format_file_size,
    get_image_info,
    is_image_file,
)


class TestIsImageFile:
    """Tests for is_image_file function."""

    def test_valid_image_extensions(self) -> None:
        """Test that common image extensions are recognized."""
        valid_files = [
            Path("photo.jpg"),
            Path("picture.JPEG"),
            Path("image.png"),
            Path("graphic.gif"),
            Path("bitmap.bmp"),
            Path("document.tiff"),
            Path("scan.TIF"),
            Path("modern.webp"),
        ]

        for file_path in valid_files:
            assert is_image_file(
                file_path
            ), f"{file_path} should be recognized as image"

    def test_invalid_extensions(self) -> None:
        """Test that non-image files are not recognized."""
        invalid_files = [
            Path("document.txt"),
            Path("script.py"),
            Path("data.json"),
            Path("config.yaml"),
            Path("readme.md"),
        ]

        for file_path in invalid_files:
            assert not is_image_file(
                file_path
            ), f"{file_path} should not be recognized as image"

    def test_case_insensitive(self) -> None:
        """Test that extension matching is case-insensitive."""
        assert is_image_file(Path("photo.JPG"))
        assert is_image_file(Path("photo.jpg"))
        assert is_image_file(Path("photo.JpG"))


class TestGetImageInfo:
    """Tests for get_image_info function."""

    def test_get_info_from_valid_image(self, sample_image: Path) -> None:
        """Test extracting info from a valid image."""
        info = get_image_info(sample_image)

        assert info is not None
        assert "dimensions" in info
        assert "format" in info
        assert "mode" in info
        assert "file_size" in info

        assert info["dimensions"] == (100, 100)
        assert info["format"] == "JPEG"
        assert info["file_size"] > 0

    def test_get_info_from_nonexistent_file(self, temp_dir: Path) -> None:
        """Test that None is returned for nonexistent files."""
        nonexistent = temp_dir / "does_not_exist.jpg"
        info = get_image_info(nonexistent)

        assert info is None

    def test_get_info_from_invalid_image(self, temp_dir: Path) -> None:
        """Test that None is returned for invalid image files."""
        invalid = temp_dir / "not_an_image.jpg"
        invalid.write_text("This is not an image")

        info = get_image_info(invalid)
        assert info is None


class TestFormatFileSize:
    """Tests for format_file_size function."""

    def test_zero_bytes(self) -> None:
        """Test formatting zero bytes."""
        assert format_file_size(0) == "0 B"

    def test_bytes(self) -> None:
        """Test formatting bytes."""
        assert format_file_size(100) == "100.00 B"
        assert format_file_size(1023) == "1023.00 B"

    def test_kilobytes(self) -> None:
        """Test formatting kilobytes."""
        assert format_file_size(1024) == "1.00 KB"
        assert format_file_size(2048) == "2.00 KB"
        assert format_file_size(1536) == "1.50 KB"

    def test_megabytes(self) -> None:
        """Test formatting megabytes."""
        assert format_file_size(1024 * 1024) == "1.00 MB"
        assert format_file_size(5 * 1024 * 1024) == "5.00 MB"

    def test_gigabytes(self) -> None:
        """Test formatting gigabytes."""
        assert format_file_size(1024 * 1024 * 1024) == "1.00 GB"
        assert format_file_size(2 * 1024 * 1024 * 1024) == "2.00 GB"


class TestCollectImageFiles:
    """Tests for collect_image_files function."""

    def test_collect_from_flat_directory(
        self, sample_images: list[Path], temp_dir: Path
    ) -> None:
        """Test collecting images from a flat directory."""
        images = collect_image_files(temp_dir, recursive=False)

        assert len(images) == len(sample_images)
        for img in sample_images:
            assert img in images

    def test_collect_recursive(self, temp_dir: Path) -> None:
        """Test collecting images recursively."""
        # Create subdirectories with images
        sub1 = temp_dir / "sub1"
        sub1.mkdir()
        sub2 = temp_dir / "sub2"
        sub2.mkdir()

        from PIL import Image

        # Create images in subdirectories
        img1 = sub1 / "image1.jpg"
        Image.new("RGB", (10, 10)).save(img1)

        img2 = sub2 / "image2.jpg"
        Image.new("RGB", (10, 10)).save(img2)

        # Collect recursively
        images = collect_image_files(temp_dir, recursive=True)
        assert len(images) >= 2
        assert img1 in images
        assert img2 in images

    def test_collect_non_recursive(self, temp_dir: Path) -> None:
        """Test that non-recursive scan doesn't find subdirectory images."""
        from PIL import Image

        # Create image in subdirectory
        sub_dir = temp_dir / "subdir"
        sub_dir.mkdir()

        sub_img = sub_dir / "sub_image.jpg"
        Image.new("RGB", (10, 10)).save(sub_img)

        # Create image in root
        root_img = temp_dir / "root_image.jpg"
        Image.new("RGB", (10, 10)).save(root_img)

        # Non-recursive collection
        images = collect_image_files(temp_dir, recursive=False)

        assert root_img in images
        assert sub_img not in images

    def test_collect_excludes_non_images(self, mixed_directory: Path) -> None:
        """Test that non-image files are excluded."""
        images = collect_image_files(mixed_directory, recursive=False)

        # Should only contain image files
        for img in images:
            assert is_image_file(img)

    def test_nonexistent_directory(self, temp_dir: Path) -> None:
        """Test that empty list is returned for nonexistent directory."""
        nonexistent = temp_dir / "does_not_exist"
        images = collect_image_files(nonexistent)

        assert images == []

    def test_file_instead_of_directory(self, sample_image: Path) -> None:
        """Test that empty list is returned when path is a file."""
        images = collect_image_files(sample_image)

        assert images == []
