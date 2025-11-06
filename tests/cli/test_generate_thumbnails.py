"""Tests for generate thumbnails CLI."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from vam_tools.cli.generate_thumbnails import generate
from vam_tools.core.types import FileType, ImageRecord


class TestGenerateThumbnailsCLI:
    """Tests for generate thumbnails CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create Click test runner."""
        return CliRunner()

    @pytest.fixture
    def catalog_dir(self, tmp_path: Path) -> Path:
        """Create test catalog directory."""
        catalog = tmp_path / "catalog"
        catalog.mkdir()
        (catalog / "thumbnails").mkdir()
        return catalog

    @pytest.fixture
    def sample_images(self) -> list[ImageRecord]:
        """Create sample image records."""
        return [
            ImageRecord(
                id="img1",
                source_path="/tmp/photo1.jpg",
                file_size=100000,
                file_hash="abc123",
                checksum="sha256:abc123",
                format="JPEG",
                width=1920,
                height=1080,
                file_type=FileType.IMAGE,
            ),
            ImageRecord(
                id="img2",
                source_path="/tmp/photo2.jpg",
                file_size=200000,
                file_hash="def456",
                checksum="sha256:def456",
                format="JPEG",
                width=3840,
                height=2160,
                file_type=FileType.IMAGE,
            ),
        ]

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    @patch("vam_tools.cli.generate_thumbnails.get_thumbnail_path")
    def test_generate_basic(
        self,
        mock_get_path,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test basic thumbnail generation."""
        # Mock catalog
        mock_catalog = Mock()
        mock_catalog.list_images.return_value = sample_images
        mock_catalog_cls.return_value = mock_catalog

        # Mock thumbnail generation
        mock_thumb_path = catalog_dir / "thumbnails" / "img1.jpg"
        mock_get_path.return_value = mock_thumb_path
        mock_gen_thumb.return_value = True

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 0
        assert "Generated: 2" in result.output
        mock_catalog.save.assert_called_once()

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    def test_generate_no_images(
        self, mock_catalog_cls, runner: CliRunner, catalog_dir: Path
    ) -> None:
        """Test with no images in catalog."""
        mock_catalog = Mock()
        mock_catalog.list_images.return_value = []
        mock_catalog_cls.return_value = mock_catalog

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 0
        assert "No images found" in result.output

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    def test_generate_catalog_load_error(
        self, mock_catalog_cls, runner: CliRunner, catalog_dir: Path
    ) -> None:
        """Test catalog load error."""
        mock_catalog = Mock()
        mock_catalog.load.side_effect = Exception("Load failed")
        mock_catalog_cls.return_value = mock_catalog

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 1
        assert "Error loading catalog" in result.output

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    @patch("vam_tools.cli.generate_thumbnails.get_thumbnail_path")
    def test_generate_with_force(
        self,
        mock_get_path,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test force regeneration."""
        mock_catalog = Mock()
        mock_catalog.list_images.return_value = sample_images
        mock_catalog_cls.return_value = mock_catalog

        # Mock existing thumbnails
        mock_thumb_path = Mock()
        mock_thumb_path.exists.return_value = True
        mock_thumb_path.relative_to.return_value = Path("thumbnails/img1.jpg")
        mock_get_path.return_value = mock_thumb_path
        mock_gen_thumb.return_value = True

        result = runner.invoke(generate, [str(catalog_dir), "--force"])

        assert result.exit_code == 0
        assert "Force mode" in result.output
        assert "Generated: 2" in result.output

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    @patch("vam_tools.cli.generate_thumbnails.get_thumbnail_path")
    def test_generate_skip_existing(
        self,
        mock_get_path,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test skipping existing thumbnails."""
        mock_catalog = Mock()
        mock_catalog.list_images.return_value = sample_images
        mock_catalog_cls.return_value = mock_catalog

        # First image has thumbnail, second doesn't
        def mock_get_path_side_effect(image_id, thumbs_dir):
            mock_path = Mock()
            mock_path.exists.return_value = image_id == "img1"
            mock_path.relative_to.return_value = Path(f"thumbnails/{image_id}.jpg")
            return mock_path

        mock_get_path.side_effect = mock_get_path_side_effect
        mock_gen_thumb.return_value = True

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 0
        assert "Generated: 1" in result.output
        assert "Skipped: 1" in result.output

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    @patch("vam_tools.cli.generate_thumbnails.get_thumbnail_path")
    def test_generate_with_custom_size(
        self,
        mock_get_path,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test custom thumbnail size."""
        mock_catalog = Mock()
        mock_catalog.list_images.return_value = sample_images
        mock_catalog_cls.return_value = mock_catalog

        mock_thumb_path = Mock()
        mock_thumb_path.exists.return_value = False
        mock_thumb_path.relative_to.return_value = Path("thumbnails/img1.jpg")
        mock_get_path.return_value = mock_thumb_path
        mock_gen_thumb.return_value = True

        result = runner.invoke(generate, [str(catalog_dir), "--size", "300"])

        assert result.exit_code == 0
        assert "300x300px" in result.output
        # Verify size passed to generate_thumbnail
        calls = mock_gen_thumb.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["size"] == (300, 300)

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    @patch("vam_tools.cli.generate_thumbnails.get_thumbnail_path")
    def test_generate_with_custom_quality(
        self,
        mock_get_path,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test custom JPEG quality."""
        mock_catalog = Mock()
        mock_catalog.list_images.return_value = sample_images
        mock_catalog_cls.return_value = mock_catalog

        mock_thumb_path = Mock()
        mock_thumb_path.exists.return_value = False
        mock_thumb_path.relative_to.return_value = Path("thumbnails/img1.jpg")
        mock_get_path.return_value = mock_thumb_path
        mock_gen_thumb.return_value = True

        result = runner.invoke(generate, [str(catalog_dir), "--quality", "70"])

        assert result.exit_code == 0
        assert "quality: 70" in result.output
        # Verify quality passed to generate_thumbnail
        calls = mock_gen_thumb.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["quality"] == 70

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    @patch("vam_tools.cli.generate_thumbnails.get_thumbnail_path")
    def test_generate_with_errors(
        self,
        mock_get_path,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test handling generation errors."""
        mock_catalog = Mock()
        mock_catalog.list_images.return_value = sample_images
        mock_catalog_cls.return_value = mock_catalog

        mock_thumb_path = Mock()
        mock_thumb_path.exists.return_value = False
        mock_thumb_path.relative_to.return_value = Path("thumbnails/img.jpg")
        mock_get_path.return_value = mock_thumb_path

        # First succeeds, second fails
        mock_gen_thumb.side_effect = [True, False]

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 0
        assert "Generated: 1" in result.output
        assert "Errors: 1" in result.output

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    @patch("vam_tools.cli.generate_thumbnails.get_thumbnail_path")
    def test_generate_catalog_save_error(
        self,
        mock_get_path,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test catalog save error."""
        mock_catalog = Mock()
        mock_catalog.list_images.return_value = sample_images
        mock_catalog.save.side_effect = Exception("Save failed")
        mock_catalog_cls.return_value = mock_catalog

        mock_thumb_path = Mock()
        mock_thumb_path.exists.return_value = False
        mock_thumb_path.relative_to.return_value = Path("thumbnails/img.jpg")
        mock_get_path.return_value = mock_thumb_path
        mock_gen_thumb.return_value = True

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 1
        assert "Error saving catalog" in result.output

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    def test_generate_verbose(
        self, mock_catalog_cls, runner: CliRunner, catalog_dir: Path
    ) -> None:
        """Test verbose logging."""
        mock_catalog = Mock()
        mock_catalog.list_images.return_value = []
        mock_catalog_cls.return_value = mock_catalog

        result = runner.invoke(generate, [str(catalog_dir), "--verbose"])

        assert result.exit_code == 0
        # Verbose flag should enable debug logging

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    @patch("vam_tools.cli.generate_thumbnails.get_thumbnail_path")
    def test_generate_updates_catalog(
        self,
        mock_get_path,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test catalog is updated with thumbnail paths."""
        mock_catalog = Mock()
        mock_catalog.list_images.return_value = sample_images
        mock_catalog_cls.return_value = mock_catalog

        mock_thumb_path = Mock()
        mock_thumb_path.exists.return_value = False
        mock_thumb_path.relative_to.return_value = Path("thumbnails/img1.jpg")
        mock_get_path.return_value = mock_thumb_path
        mock_gen_thumb.return_value = True

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 0
        # Verify add_image was called to update records
        assert mock_catalog.add_image.call_count == 2
        # Verify thumbnail paths were set
        for call in mock_catalog.add_image.call_args_list:
            image = call[0][0]
            assert hasattr(image, "thumbnail_path")

    def test_generate_nonexistent_directory(self, runner: CliRunner) -> None:
        """Test with non-existent catalog directory."""
        result = runner.invoke(generate, ["/nonexistent/path"])

        assert result.exit_code != 0
        # Click will fail before our code runs

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    @patch("vam_tools.cli.generate_thumbnails.get_thumbnail_path")
    def test_generate_all_options(
        self,
        mock_get_path,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test with all CLI options."""
        mock_catalog = Mock()
        mock_catalog.list_images.return_value = sample_images
        mock_catalog_cls.return_value = mock_catalog

        mock_thumb_path = Mock()
        mock_thumb_path.exists.return_value = True  # Should regenerate with --force
        mock_thumb_path.relative_to.return_value = Path("thumbnails/img.jpg")
        mock_get_path.return_value = mock_thumb_path
        mock_gen_thumb.return_value = True

        result = runner.invoke(
            generate,
            [
                str(catalog_dir),
                "--force",
                "--size",
                "300",
                "--quality",
                "90",
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert "Force mode" in result.output
        assert "300x300px" in result.output
        assert "quality: 90" in result.output
