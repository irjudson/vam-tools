"""
Tests for thumbnail generation utilities.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from PIL import Image

from vam_tools.shared.thumbnail_utils import (
    DEFAULT_THUMBNAIL_SIZE,
    extract_video_thumbnail,
    generate_thumbnail,
    get_thumbnail_path,
    is_video_file,
    thumbnail_exists,
)


class TestGenerateThumbnail:
    """Tests for generate_thumbnail function."""

    def test_generate_thumbnail_source_not_found(self, tmp_path: Path) -> None:
        """Test thumbnail generation when source file doesn't exist."""
        source_path = tmp_path / "nonexistent.jpg"
        output_path = tmp_path / "thumb.jpg"

        result = generate_thumbnail(source_path, output_path)

        assert result is False
        assert not output_path.exists()

    def test_generate_thumbnail_success(self, tmp_path: Path) -> None:
        """Test successful thumbnail generation from image."""
        # Create a test image
        source_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (800, 600), color="red")
        img.save(source_path)

        output_path = tmp_path / "thumb.jpg"

        result = generate_thumbnail(source_path, output_path)

        assert result is True
        assert output_path.exists()

        # Verify thumbnail size
        thumb = Image.open(output_path)
        assert thumb.size[0] <= DEFAULT_THUMBNAIL_SIZE[0]
        assert thumb.size[1] <= DEFAULT_THUMBNAIL_SIZE[1]

    def test_generate_thumbnail_rgba_conversion(self, tmp_path: Path) -> None:
        """Test thumbnail generation with RGBA to RGB conversion."""
        # Create RGBA image
        source_path = tmp_path / "test_rgba.png"
        img = Image.new("RGBA", (800, 600), color=(255, 0, 0, 128))
        img.save(source_path)

        output_path = tmp_path / "thumb.jpg"

        result = generate_thumbnail(source_path, output_path)

        assert result is True
        assert output_path.exists()

        # Verify it's RGB (JPEG doesn't support transparency)
        thumb = Image.open(output_path)
        assert thumb.mode == "RGB"

    def test_generate_thumbnail_grayscale_conversion(self, tmp_path: Path) -> None:
        """Test thumbnail generation with grayscale to RGB conversion."""
        # Create grayscale image
        source_path = tmp_path / "test_gray.jpg"
        img = Image.new("L", (800, 600), color=128)
        img.save(source_path)

        output_path = tmp_path / "thumb.jpg"

        result = generate_thumbnail(source_path, output_path)

        assert result is True
        assert output_path.exists()

        # Verify it's RGB
        thumb = Image.open(output_path)
        assert thumb.mode == "RGB"

    def test_generate_thumbnail_custom_size_quality(self, tmp_path: Path) -> None:
        """Test thumbnail generation with custom size and quality."""
        source_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (800, 600), color="blue")
        img.save(source_path)

        output_path = tmp_path / "thumb.jpg"
        custom_size = (100, 100)
        custom_quality = 50

        result = generate_thumbnail(
            source_path, output_path, size=custom_size, quality=custom_quality
        )

        assert result is True
        assert output_path.exists()

        thumb = Image.open(output_path)
        assert thumb.size[0] <= custom_size[0]
        assert thumb.size[1] <= custom_size[1]

    def test_generate_thumbnail_video_file(self, tmp_path: Path) -> None:
        """Test thumbnail generation for video file."""
        source_path = tmp_path / "test.mp4"
        # Create a fake video file
        source_path.write_bytes(b"fake video content")

        output_path = tmp_path / "thumb.jpg"

        with patch("vam_tools.shared.thumbnail_utils.is_video_file", return_value=True):
            with patch(
                "vam_tools.shared.thumbnail_utils.extract_video_thumbnail",
                return_value=Image.new("RGB", (640, 480), color="green"),
            ):
                result = generate_thumbnail(source_path, output_path)

                assert result is True
                assert output_path.exists()

    def test_generate_thumbnail_video_extraction_fails(self, tmp_path: Path) -> None:
        """Test thumbnail generation when video extraction fails."""
        source_path = tmp_path / "test.mp4"
        source_path.write_bytes(b"fake video content")

        output_path = tmp_path / "thumb.jpg"

        with patch("vam_tools.shared.thumbnail_utils.is_video_file", return_value=True):
            with patch(
                "vam_tools.shared.thumbnail_utils.extract_video_thumbnail",
                return_value=None,
            ):
                result = generate_thumbnail(source_path, output_path)

                assert result is False
                assert not output_path.exists()

    def test_generate_thumbnail_general_exception(self, tmp_path: Path) -> None:
        """Test thumbnail generation handles general exceptions."""
        source_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (800, 600), color="red")
        img.save(source_path)

        output_path = tmp_path / "thumb.jpg"

        with patch("PIL.Image.open") as mock_open:
            mock_img = Mock()
            mock_img.mode = "RGB"
            mock_img.thumbnail.side_effect = Exception("Unexpected error")
            mock_open.return_value = mock_img

            result = generate_thumbnail(source_path, output_path)

            assert result is False


class TestExtractVideoThumbnail:
    """Tests for extract_video_thumbnail function."""

    def test_extract_video_thumbnail_ffmpeg_success(self, tmp_path: Path) -> None:
        """Test video thumbnail extraction with ffmpeg."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video")

        # Create a temporary image file that ffmpeg would create
        fake_frame_path = tmp_path / "fake_frame.jpg"
        test_img = Image.new("RGB", (640, 480), color="red")
        test_img.save(fake_frame_path)

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_file.name = str(fake_frame_path)
            mock_temp.return_value.__enter__.return_value = mock_file

            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0

                result = extract_video_thumbnail(video_path)

                assert result is not None
                assert isinstance(result, Image.Image)
                # Verify ffmpeg was called with correct arguments
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert call_args[0] == "ffmpeg"
                assert "-i" in call_args
                assert str(video_path) in call_args

    def test_extract_video_thumbnail_ffmpeg_fails(self, tmp_path: Path) -> None:
        """Test video thumbnail extraction when ffmpeg fails."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr.decode.return_value = "ffmpeg error"

            result = extract_video_thumbnail(video_path)

            assert result is None

    def test_extract_video_thumbnail_ffmpeg_exception(self, tmp_path: Path) -> None:
        """Test video thumbnail extraction when ffmpeg raises exception."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video")

        with patch(
            "subprocess.run",
            side_effect=Exception("ffmpeg not found"),
        ):
            result = extract_video_thumbnail(video_path)

            assert result is None


class TestIsVideoFile:
    """Tests for is_video_file function."""

    def test_is_video_file_mp4(self) -> None:
        """Test video detection for mp4 file."""
        assert is_video_file(Path("test.mp4")) is True
        assert is_video_file(Path("test.MP4")) is True

    def test_is_video_file_mov(self) -> None:
        """Test video detection for mov file."""
        assert is_video_file(Path("test.mov")) is True

    def test_is_video_file_avi(self) -> None:
        """Test video detection for avi file."""
        assert is_video_file(Path("test.avi")) is True

    def test_is_video_file_not_video(self) -> None:
        """Test video detection for non-video file."""
        assert is_video_file(Path("test.jpg")) is False
        assert is_video_file(Path("test.png")) is False
        assert is_video_file(Path("test.txt")) is False

    def test_is_video_file_various_formats(self) -> None:
        """Test video detection for various video formats."""
        video_formats = [
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".m4v",
            ".mpg",
            ".mpeg",
            ".wmv",
            ".flv",
            ".webm",
        ]
        for fmt in video_formats:
            assert is_video_file(Path(f"test{fmt}")) is True


class TestGetThumbnailPath:
    """Tests for get_thumbnail_path function."""

    def test_get_thumbnail_path_creates_dir(self, tmp_path: Path) -> None:
        """Test get_thumbnail_path creates directory."""
        thumbnails_dir = tmp_path / "thumbnails"
        image_id = "abc123"

        result = get_thumbnail_path(
            image_id, thumbnails_dir, size="medium", create_dir=True
        )

        # Now creates size subdirectory
        assert (thumbnails_dir / "medium").exists()
        assert result == thumbnails_dir / "medium" / f"{image_id}.jpg"

    def test_get_thumbnail_path_no_create_dir(self, tmp_path: Path) -> None:
        """Test get_thumbnail_path without creating directory."""
        thumbnails_dir = tmp_path / "thumbnails"
        image_id = "def456"

        result = get_thumbnail_path(
            image_id, thumbnails_dir, size="medium", create_dir=False
        )

        assert not thumbnails_dir.exists()
        assert result == thumbnails_dir / "medium" / f"{image_id}.jpg"

    def test_get_thumbnail_path_existing_dir(self, tmp_path: Path) -> None:
        """Test get_thumbnail_path with existing directory."""
        thumbnails_dir = tmp_path / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        image_id = "ghi789"

        result = get_thumbnail_path(
            image_id, thumbnails_dir, size="large", create_dir=True
        )

        assert (thumbnails_dir / "large").exists()
        assert result == thumbnails_dir / "large" / f"{image_id}.jpg"

    def test_get_thumbnail_path_different_sizes(self, tmp_path: Path) -> None:
        """Test get_thumbnail_path with different size options."""
        thumbnails_dir = tmp_path / "thumbnails"
        image_id = "test123"

        # Test all sizes
        for size in ["small", "medium", "large"]:
            result = get_thumbnail_path(
                image_id, thumbnails_dir, size=size, create_dir=True
            )
            assert result == thumbnails_dir / size / f"{image_id}.jpg"
            assert (thumbnails_dir / size).exists()


class TestThumbnailExists:
    """Tests for thumbnail_exists function."""

    def test_thumbnail_exists_true(self, tmp_path: Path) -> None:
        """Test thumbnail_exists returns True when thumbnail exists."""
        thumbnails_dir = tmp_path / "thumbnails"
        size_dir = thumbnails_dir / "medium"
        size_dir.mkdir(parents=True, exist_ok=True)

        image_id = "abc123"
        thumb_path = size_dir / f"{image_id}.jpg"
        thumb_path.write_bytes(b"fake thumbnail")

        result = thumbnail_exists(image_id, thumbnails_dir, size="medium")

        assert result is True

    def test_thumbnail_exists_false(self, tmp_path: Path) -> None:
        """Test thumbnail_exists returns False when thumbnail doesn't exist."""
        thumbnails_dir = tmp_path / "thumbnails"
        size_dir = thumbnails_dir / "medium"
        size_dir.mkdir(parents=True, exist_ok=True)

        image_id = "nonexistent"

        result = thumbnail_exists(image_id, thumbnails_dir, size="medium")

        assert result is False

    def test_thumbnail_exists_dir_not_exist(self, tmp_path: Path) -> None:
        """Test thumbnail_exists when directory doesn't exist."""
        thumbnails_dir = tmp_path / "nonexistent_thumbnails"
        image_id = "abc123"

        result = thumbnail_exists(image_id, thumbnails_dir, size="medium")

        assert result is False
