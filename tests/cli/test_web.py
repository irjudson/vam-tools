"""
Tests for web CLI.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from PIL import Image

from vam_tools.cli.analyze import analyze
from vam_tools.cli.web import web


class TestWebCLI:
    """Tests for vam-web CLI command."""

    def test_web_catalog_not_found(self, tmp_path: Path) -> None:
        """Test error when catalog doesn't exist."""
        runner = CliRunner()
        catalog_path = tmp_path / "nonexistent"

        result = runner.invoke(web, [str(catalog_path)])

        # Should fail with catalog not found message
        # Note: click.Path(exists=True) causes Click to fail early
        assert result.exit_code != 0

    def test_web_catalog_file_missing(self, tmp_path: Path) -> None:
        """Test error when catalog directory exists but .catalog.json missing."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir()

        result = runner.invoke(web, [str(catalog_path)])

        assert result.exit_code == 0  # Command runs
        assert "Error: Catalog not found" in result.output
        assert "vam-analyze" in result.output

    @patch("vam_tools.cli.web.uvicorn.run")
    @patch("vam_tools.cli.web.init_catalog")
    def test_web_basic_launch(
        self, mock_init: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Test basic web server launch."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create a catalog
        Image.new("RGB", (100, 100), color="red").save(photos_dir / "test.jpg")
        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )
        assert result.exit_code == 0

        # Launch web server
        result = runner.invoke(web, [str(catalog_path)])

        assert result.exit_code == 0
        assert "Catalog Viewer" in result.output
        assert "Starting web server" in result.output
        mock_init.assert_called_once_with(catalog_path)
        mock_run.assert_called_once()

    @patch("vam_tools.cli.web.uvicorn.run")
    @patch("vam_tools.cli.web.init_catalog")
    def test_web_custom_host_port(
        self, mock_init: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Test custom host and port."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create catalog
        Image.new("RGB", (100, 100), color="blue").save(photos_dir / "test.jpg")
        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )
        assert result.exit_code == 0

        # Launch with custom host and port
        result = runner.invoke(
            web, [str(catalog_path), "--host", "0.0.0.0", "--port", "9000"]
        )

        assert result.exit_code == 0
        assert "0.0.0.0:9000" in result.output
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 9000

    @patch("vam_tools.cli.web.uvicorn.run")
    @patch("vam_tools.cli.web.init_catalog")
    def test_web_reload_mode(
        self, mock_init: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Test reload mode for development."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create catalog
        Image.new("RGB", (100, 100), color="green").save(photos_dir / "test.jpg")
        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )
        assert result.exit_code == 0

        # Launch with reload
        result = runner.invoke(web, [str(catalog_path), "--reload"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["reload"] is True

    @patch("vam_tools.cli.web.uvicorn.run")
    @patch("vam_tools.cli.web.init_catalog")
    def test_web_default_settings(
        self, mock_init: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Test default host, port, and settings."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create catalog
        Image.new("RGB", (100, 100), color="yellow").save(photos_dir / "test.jpg")
        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )
        assert result.exit_code == 0

        # Launch with defaults
        result = runner.invoke(web, [str(catalog_path)])

        assert result.exit_code == 0
        assert "127.0.0.1:8765" in result.output
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 8765
        assert call_kwargs["reload"] is False
        assert call_kwargs["log_level"] == "info"
