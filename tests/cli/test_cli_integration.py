"""
Integration tests for CLI commands.

These tests verify that the CLIs work end-to-end.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner
from PIL import Image

from vam_tools.cli import catalog_cli, date_cli, duplicate_cli, main


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


class TestDateCLI:
    """Integration tests for date analysis CLI."""

    def test_date_cli_basic_usage(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test basic date CLI functionality."""
        # Create test images with dates in filenames
        for i in range(3):
            img = Image.new("RGB", (10, 10))
            img_path = temp_dir / f"photo_2023-0{i+1}-15.jpg"
            img.save(img_path)

        output_file = temp_dir / "dates.txt"

        result = cli_runner.invoke(
            date_cli.cli,
            [str(temp_dir), "-o", str(output_file), "-q"],
        )

        # Should succeed
        assert result.exit_code == 0

        # Output file should exist
        assert output_file.exists()

        # Should contain results
        content = output_file.read_text()
        assert "Image Date Analysis Results" in content

    def test_date_cli_no_images(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test date CLI with no images."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = cli_runner.invoke(
            date_cli.cli,
            [str(empty_dir), "-q"],
        )

        # Should exit successfully even with no images
        assert result.exit_code == 0

    def test_date_cli_verbose(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test date CLI with verbose flag."""
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "test.jpg"
        img.save(img_path)

        result = cli_runner.invoke(
            date_cli.cli,
            [str(temp_dir), "-v"],
        )

        assert result.exit_code == 0

    def test_date_cli_sort_options(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test date CLI with different sort options."""
        # Create test image
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "photo_2023-01-15.jpg"
        img.save(img_path)

        output_file = temp_dir / "dates.txt"

        # Test each sort option
        for sort_by in ["date", "path", "source"]:
            result = cli_runner.invoke(
                date_cli.cli,
                [str(temp_dir), "-o", str(output_file), "--sort-by", sort_by, "-q"],
            )

            assert result.exit_code == 0


class TestDuplicateCLI:
    """Integration tests for duplicate finder CLI."""

    def test_duplicate_cli_basic_usage(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test basic duplicate CLI functionality."""
        # Create two identical images
        img = Image.new("RGB", (10, 10), color="red")

        img1 = temp_dir / "image1.jpg"
        img2 = temp_dir / "image2.jpg"

        img.save(img1)
        img.save(img2)

        output_file = temp_dir / "duplicates.txt"

        result = cli_runner.invoke(
            duplicate_cli.cli,
            [str(temp_dir), "-o", str(output_file), "-q"],
        )

        # Should succeed
        assert result.exit_code == 0

        # Output file should exist
        assert output_file.exists()

        # Should contain results
        content = output_file.read_text()
        assert "DUPLICATE IMAGE ANALYSIS" in content

    def test_duplicate_cli_with_threshold(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test duplicate CLI with different thresholds."""
        # Create test images
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "test.jpg"
        img.save(img_path)

        output_file = temp_dir / "duplicates.txt"

        # Test with different thresholds
        for threshold in [0, 5, 15]:
            result = cli_runner.invoke(
                duplicate_cli.cli,
                [str(temp_dir), "-t", str(threshold), "-o", str(output_file), "-q"],
            )

            assert result.exit_code == 0

    def test_duplicate_cli_invalid_threshold(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test duplicate CLI with invalid threshold."""
        result = cli_runner.invoke(
            duplicate_cli.cli,
            [str(temp_dir), "-t", "100"],  # Out of range
        )

        # Should fail
        assert result.exit_code != 0

    def test_duplicate_cli_no_images(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test duplicate CLI with no images."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = cli_runner.invoke(
            duplicate_cli.cli,
            [str(empty_dir), "-q"],
        )

        # Should exit successfully
        assert result.exit_code == 0

    def test_duplicate_cli_single_image(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test duplicate CLI with single image."""
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "single.jpg"
        img.save(img_path)

        result = cli_runner.invoke(
            duplicate_cli.cli,
            [str(temp_dir), "-q"],
        )

        # Should exit successfully
        assert result.exit_code == 0


class TestCatalogCLI:
    """Integration tests for catalog reorganizer CLI."""

    def test_catalog_cli_dry_run(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test catalog CLI with dry-run."""
        # Create test image
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "photo_2023-01-15.jpg"
        img.save(img_path)

        output_dir = temp_dir / "output"

        result = cli_runner.invoke(
            catalog_cli.cli,
            [
                str(temp_dir),
                "-o", str(output_dir),
                "--dry-run",
                "--yes",
                "-q",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # Original file should still exist
        assert img_path.exists()

    def test_catalog_cli_copy_mode(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test catalog CLI with copy mode."""
        # Create test image
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "source" / "photo_2023-01-15.jpg"
        img_path.parent.mkdir()
        img.save(img_path)

        output_dir = temp_dir / "output"

        result = cli_runner.invoke(
            catalog_cli.cli,
            [
                str(img_path.parent),
                "-o", str(output_dir),
                "--copy",
                "--yes",
                "-q",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # Original should still exist (copy mode)
        assert img_path.exists()

    def test_catalog_cli_different_strategies(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test catalog CLI with different organization strategies."""
        # Create test image
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "source" / "photo_2023-01-15.jpg"
        img_path.parent.mkdir()
        img.save(img_path)

        strategies = ["year/month-day", "year/month", "year", "flat"]

        for strategy in strategies:
            output_dir = temp_dir / f"output_{strategy.replace('/', '_')}"

            result = cli_runner.invoke(
                catalog_cli.cli,
                [
                    str(img_path.parent),
                    "-o", str(output_dir),
                    "-s", strategy,
                    "--dry-run",
                    "--yes",
                    "-q",
                ],
            )

            assert result.exit_code == 0

    def test_catalog_cli_conflict_resolution(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test catalog CLI with different conflict resolutions."""
        # Create test image
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "source" / "photo_2023-01-15.jpg"
        img_path.parent.mkdir()
        img.save(img_path)

        output_dir = temp_dir / "output"

        for conflict in ["skip", "rename", "overwrite"]:
            result = cli_runner.invoke(
                catalog_cli.cli,
                [
                    str(img_path.parent),
                    "-o", str(output_dir),
                    "--conflict", conflict,
                    "--dry-run",
                    "--yes",
                    "-q",
                ],
            )

            assert result.exit_code == 0

    def test_catalog_cli_no_images(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test catalog CLI with no images."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        output_dir = temp_dir / "output"

        result = cli_runner.invoke(
            catalog_cli.cli,
            [str(empty_dir), "-o", str(output_dir), "--yes", "-q"],
        )

        # Should exit successfully
        assert result.exit_code == 0


class TestMainCLI:
    """Integration tests for main interactive CLI."""

    def test_main_cli_version(self, cli_runner: CliRunner) -> None:
        """Test main CLI --version flag."""
        result = cli_runner.invoke(main.cli, ["--version"])

        assert result.exit_code == 0
        assert "Lightroom Tools" in result.output

    def test_main_cli_help(self, cli_runner: CliRunner) -> None:
        """Test main CLI --help flag."""
        result = cli_runner.invoke(main.cli, ["--help"])

        assert result.exit_code == 0
        assert "Lightroom Tools" in result.output


class TestCLIErrorHandling:
    """Test error handling in CLIs."""

    def test_date_cli_nonexistent_directory(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test date CLI with nonexistent directory."""
        nonexistent = temp_dir / "does_not_exist"

        result = cli_runner.invoke(
            date_cli.cli,
            [str(nonexistent)],
        )

        # Should fail gracefully
        assert result.exit_code != 0

    def test_duplicate_cli_nonexistent_directory(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test duplicate CLI with nonexistent directory."""
        nonexistent = temp_dir / "does_not_exist"

        result = cli_runner.invoke(
            duplicate_cli.cli,
            [str(nonexistent)],
        )

        # Should fail gracefully
        assert result.exit_code != 0

    def test_catalog_cli_nonexistent_directory(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test catalog CLI with nonexistent directory."""
        nonexistent = temp_dir / "does_not_exist"
        output_dir = temp_dir / "output"

        result = cli_runner.invoke(
            catalog_cli.cli,
            [str(nonexistent), "-o", str(output_dir), "--yes"],
        )

        # Should fail gracefully
        assert result.exit_code != 0


class TestCLIOutputFiles:
    """Test that CLIs create proper output files."""

    def test_date_cli_creates_output_file(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test that date CLI creates output file."""
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "test.jpg"
        img.save(img_path)

        output_file = temp_dir / "custom_output.txt"

        result = cli_runner.invoke(
            date_cli.cli,
            [str(temp_dir), "-o", str(output_file), "-q"],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_duplicate_cli_creates_output_file(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test that duplicate CLI creates output file."""
        img = Image.new("RGB", (10, 10))
        img1 = temp_dir / "img1.jpg"
        img2 = temp_dir / "img2.jpg"
        img.save(img1)
        img.save(img2)

        output_file = temp_dir / "custom_duplicates.txt"

        result = cli_runner.invoke(
            duplicate_cli.cli,
            [str(temp_dir), "-o", str(output_file), "-q"],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestCLIFlags:
    """Test various CLI flags and options."""

    def test_quiet_flag_suppresses_output(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test that -q flag suppresses output."""
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "test.jpg"
        img.save(img_path)

        result = cli_runner.invoke(
            date_cli.cli,
            [str(temp_dir), "-q"],
        )

        assert result.exit_code == 0
        # Quiet mode should have minimal output
        assert len(result.output) < 100 or result.output == ""

    def test_recursive_flags(self, cli_runner: CliRunner, temp_dir: Path) -> None:
        """Test recursive and no-recursive flags."""
        # Create nested structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        img1 = Image.new("RGB", (10, 10))
        img2 = Image.new("RGB", (10, 10))

        (temp_dir / "root.jpg").touch()
        img1.save(temp_dir / "root.jpg")
        (subdir / "sub.jpg").touch()
        img2.save(subdir / "sub.jpg")

        # Test with recursive (default)
        result_recursive = cli_runner.invoke(
            date_cli.cli,
            [str(temp_dir), "-q"],
        )

        # Test with --no-recursive
        result_no_recursive = cli_runner.invoke(
            date_cli.cli,
            [str(temp_dir), "--no-recursive", "-q"],
        )

        assert result_recursive.exit_code == 0
        assert result_no_recursive.exit_code == 0
