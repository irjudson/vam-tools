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
    def test_generate_basic(
        self,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test basic thumbnail generation."""
        # Mock catalog
        mock_db = Mock()
        mock_catalog_cls.return_value.__enter__.return_value = mock_db

        # Mock db.execute for image retrieval
        mock_db.execute.return_value.fetchall.return_value = [
            {"id": img.id, "source_path": str(img.source_path), "thumbnail_path": None}
            for img in sample_images
        ]

        # Mock generate_thumbnail
        mock_gen_thumb.return_value = True

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 0
        assert "Generated: 2" in result.output
        # Verify execute was called to update thumbnail_path
        assert mock_db.execute.call_count >= 2  # At least 2 for updates

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    def test_generate_no_images(
        self, mock_catalog_cls, runner: CliRunner, catalog_dir: Path
    ) -> None:
        """Test with no images in catalog."""
        mock_db = Mock()
        mock_catalog_cls.return_value.__enter__.return_value = mock_db
        mock_db.execute.return_value.fetchall.return_value = []  # No images

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 0
        assert "No images found" in result.output

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    def test_generate_catalog_connection_error(
        self, mock_catalog_cls, runner: CliRunner, catalog_dir: Path
    ) -> None:
        """Test catalog connection error."""
        mock_catalog_cls.side_effect = Exception("Connection failed")

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 1
        assert "Error loading catalog" in result.output

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    def test_generate_with_force(
        self,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test force regeneration."""
        mock_db = Mock()
        mock_catalog_cls.return_value.__enter__.return_value = mock_db

        # Mock db.execute for image retrieval, with existing thumbnail paths
        mock_db.execute.return_value.fetchall.return_value = [
            {
                "id": img.id,
                "source_path": str(img.source_path),
                "thumbnail_path": f"thumbnails/{img.id}.jpg",
            }
            for img in sample_images
        ]

        # Mock generate_thumbnail
        mock_gen_thumb.return_value = True

        result = runner.invoke(generate, [str(catalog_dir), "--force"])

        assert result.exit_code == 0
        assert "Force mode" in result.output
        assert "Generated: 2" in result.output
        assert mock_gen_thumb.call_count == 2  # Should call for all images

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    def test_generate_skip_existing(
        self,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test skipping existing thumbnails."""
        mock_db = Mock()
        mock_catalog_cls.return_value.__enter__.return_value = mock_db

        # Mock db.execute for image retrieval
        # First image has thumbnail, second doesn't
        mock_db.execute.return_value.fetchall.return_value = [
            {
                "id": sample_images[0].id,
                "source_path": str(sample_images[0].source_path),
                "thumbnail_path": f"thumbnails/{sample_images[0].id}.jpg",
            },
            {
                "id": sample_images[1].id,
                "source_path": str(sample_images[1].source_path),
                "thumbnail_path": None,
            },
        ]

        mock_gen_thumb.return_value = True

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 0
        assert "Generated: 1" in result.output
        assert "Skipped: 1" in result.output
        assert mock_gen_thumb.call_count == 1  # Only called for the second image

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    def test_generate_with_custom_size(
        self,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test custom thumbnail size."""
        mock_db = Mock()
        mock_catalog_cls.return_value.__enter__.return_value = mock_db

        # Mock db.execute for image retrieval
        mock_db.execute.return_value.fetchall.return_value = [
            {"id": img.id, "source_path": str(img.source_path), "thumbnail_path": None}
            for img in sample_images
        ]

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
    def test_generate_with_custom_quality(
        self,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test custom JPEG quality."""
        mock_db = Mock()
        mock_catalog_cls.return_value.__enter__.return_value = mock_db

        # Mock db.execute for image retrieval
        mock_db.execute.return_value.fetchall.return_value = [
            {"id": img.id, "source_path": str(img.source_path), "thumbnail_path": None}
            for img in sample_images
        ]

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
    def test_generate_with_errors(
        self,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test handling generation errors."""
        mock_db = Mock()
        mock_catalog_cls.return_value.__enter__.return_value = mock_db

        # Mock db.execute for image retrieval
        mock_db.execute.return_value.fetchall.return_value = [
            {"id": img.id, "source_path": str(img.source_path), "thumbnail_path": None}
            for img in sample_images
        ]

        # First succeeds, second fails
        mock_gen_thumb.side_effect = [True, False]

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 0
        assert "Generated: 1" in result.output
        assert "Errors: 1" in result.output

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    def test_generate_verbose(
        self, mock_catalog_cls, runner: CliRunner, catalog_dir: Path
    ) -> None:
        """Test verbose logging."""
        mock_db = Mock()
        mock_catalog_cls.return_value.__enter__.return_value = mock_db
        mock_db.execute.return_value.fetchall.return_value = (
            []
        )  # No images for simplicity

        result = runner.invoke(generate, [str(catalog_dir), "--verbose"])

        assert result.exit_code == 0
        # Verbose flag should enable debug logging

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    def test_generate_updates_catalog(
        self,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test catalog is updated with thumbnail paths."""
        mock_db = Mock()
        mock_catalog_cls.return_value.__enter__.return_value = mock_db

        # Mock db.execute for image retrieval
        mock_db.execute.return_value.fetchall.return_value = [
            {"id": img.id, "source_path": str(img.source_path), "thumbnail_path": None}
            for img in sample_images
        ]

        mock_gen_thumb.return_value = True

        result = runner.invoke(generate, [str(catalog_dir)])

        assert result.exit_code == 0
        # Verify execute was called to update thumbnail_path for each image
        mock_db.execute.assert_any_call(
            "UPDATE images SET thumbnail_path = ? WHERE id = ?",
            (f"thumbnails/{sample_images[0].id}.jpg", sample_images[0].id),
        )
        mock_db.execute.assert_any_call(
            "UPDATE images SET thumbnail_path = ? WHERE id = ?",
            (f"thumbnails/{sample_images[1].id}.jpg", sample_images[1].id),
        )

    def test_generate_nonexistent_directory(self, runner: CliRunner) -> None:
        """Test with non-existent catalog directory."""
        result = runner.invoke(generate, ["/nonexistent/path"])

        assert result.exit_code != 0
        # Click will fail before our code runs

    @patch("vam_tools.cli.generate_thumbnails.CatalogDatabase")
    @patch("vam_tools.cli.generate_thumbnails.generate_thumbnail")
    def test_generate_all_options(
        self,
        mock_gen_thumb,
        mock_catalog_cls,
        runner: CliRunner,
        catalog_dir: Path,
        sample_images: list[ImageRecord],
    ) -> None:
        """Test with all CLI options."""
        mock_db = Mock()
        mock_catalog_cls.return_value.__enter__.return_value = mock_db

        # Mock db.execute for image retrieval
        mock_db.execute.return_value.fetchall.return_value = [
            {
                "id": img.id,
                "source_path": str(img.source_path),
                "thumbnail_path": f"thumbnails/{img.id}.jpg",
            }
            for img in sample_images
        ]

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
