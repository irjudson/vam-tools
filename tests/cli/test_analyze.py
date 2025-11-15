"""
Tests for analyze CLI.
"""

from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from vam_tools.cli.analyze import analyze, display_statistics, setup_logging
from vam_tools.core.types import Statistics


class TestAnalyzeCLI:
    """Tests for vam-analyze CLI command."""

    def test_analyze_no_source_error(self, tmp_path: Path) -> None:
        """Test that analyze requires --source when not in repair mode."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"

        result = runner.invoke(analyze, [str(catalog_path)])

        assert result.exit_code == 1
        assert "Error: --source/-s is required" in result.output

    def test_analyze_basic_workflow(self, tmp_path: Path) -> None:
        """Test basic analyze workflow."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create test images
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 80, 0, 0))
            img.save(photos_dir / f"photo{i}.jpg")

        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )

        assert result.exit_code == 0
        assert "Scan complete!" in result.output
        assert (catalog_path / "catalog.json").exists()

    def test_analyze_multiple_sources(self, tmp_path: Path) -> None:
        """Test analyzing multiple source directories."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos1 = tmp_path / "photos1"
        photos2 = tmp_path / "photos2"
        photos1.mkdir()
        photos2.mkdir()

        # Create images in both directories
        Image.new("RGB", (100, 100), color="red").save(photos1 / "red.jpg")
        Image.new("RGB", (100, 100), color="blue").save(photos2 / "blue.jpg")

        result = runner.invoke(
            analyze,
            [
                str(catalog_path),
                "-s",
                str(photos1),
                "-s",
                str(photos2),
                "-w",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert "Scan complete!" in result.output

    def test_analyze_verbose_mode(self, tmp_path: Path) -> None:
        """Test verbose logging mode."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        Image.new("RGB", (100, 100), color="green").save(photos_dir / "test.jpg")

        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-v", "-w", "1"]
        )

        assert result.exit_code == 0

    def test_analyze_clear_mode(self, tmp_path: Path) -> None:
        """Test clear mode creates backup and starts fresh."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        Image.new("RGB", (100, 100), color="orange").save(photos_dir / "test.jpg")

        # First analysis
        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )
        assert result.exit_code == 0

        # Clear and reanalyze
        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "--clear", "-w", "1"]
        )
        assert result.exit_code == 0
        assert "Clearing existing catalog" in result.output
        assert "Backup saved to:" in result.output

        # Check backup exists
        backups = list(catalog_path.glob(".catalog.backup.*.json"))
        assert len(backups) >= 1

    def test_analyze_repair_mode(self, tmp_path: Path) -> None:
        """Test repair mode for corrupted catalog."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir()

        # Create a valid catalog first
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        Image.new("RGB", (100, 100), color="purple").save(photos_dir / "test.jpg")

        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )
        assert result.exit_code == 0

        # Now repair it
        result = runner.invoke(analyze, [str(catalog_path), "--repair"])
        assert result.exit_code == 0
        assert "Repair mode enabled" in result.output
        assert "Catalog repaired successfully" in result.output

    def test_analyze_repair_no_catalog_error(self, tmp_path: Path) -> None:
        """Test repair mode fails when no catalog exists."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"

        result = runner.invoke(analyze, [str(catalog_path), "--repair"])

        assert result.exit_code == 1
        assert "Error: No catalog found to repair" in result.output

    def test_analyze_with_duplicates(self, tmp_path: Path) -> None:
        """Test analyze with duplicate detection."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create similar images
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 10, i * 10, i * 10))
            img.save(photos_dir / f"gray{i}.jpg")

        result = runner.invoke(
            analyze,
            [
                str(catalog_path),
                "-s",
                str(photos_dir),
                "--detect-duplicates",
                "-w",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert "Duplicate detection complete" in result.output
        assert "duplicate groups" in result.output

    def test_analyze_custom_similarity_threshold(self, tmp_path: Path) -> None:
        """Test custom similarity threshold."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create test images
        for i in range(2):
            img = Image.new("RGB", (100, 100), color=(i * 50, 0, 0))
            img.save(photos_dir / f"photo{i}.jpg")

        result = runner.invoke(
            analyze,
            [
                str(catalog_path),
                "-s",
                str(photos_dir),
                "--detect-duplicates",
                "--similarity-threshold",
                "10",
                "-w",
                "1",
            ],
        )

        assert result.exit_code == 0

    def test_analyze_incremental_scan(self, tmp_path: Path) -> None:
        """Test incremental scanning adds new files."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # First scan with 2 images
        Image.new("RGB", (100, 100), color="red").save(photos_dir / "photo1.jpg")
        Image.new("RGB", (100, 100), color="blue").save(photos_dir / "photo2.jpg")

        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )
        assert result.exit_code == 0

        # Add a third image
        Image.new("RGB", (100, 100), color="green").save(photos_dir / "photo3.jpg")

        # Rescan
        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )
        assert result.exit_code == 0
        assert "Loading existing catalog" in result.output
        assert "Files added:" in result.output
        assert "Files skipped:" in result.output

    def test_analyze_custom_workers(self, tmp_path: Path) -> None:
        """Test specifying custom number of workers."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        Image.new("RGB", (100, 100), color="cyan").save(photos_dir / "test.jpg")

        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "2"]
        )

        assert result.exit_code == 0
        assert "Starting scan with 2 worker processes" in result.output

    def test_analyze_auto_workers(self, tmp_path: Path) -> None:
        """Test auto-detection of CPU count."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        Image.new("RGB", (100, 100), color="magenta").save(photos_dir / "test.jpg")

        result = runner.invoke(analyze, [str(catalog_path), "-s", str(photos_dir)])

        assert result.exit_code == 0
        assert "worker processes (auto-detected)" in result.output

    def test_analyze_catalog_exists_shows_info(self, tmp_path: Path) -> None:
        """Test that existing catalog shows metadata."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        Image.new("RGB", (100, 100), color="yellow").save(photos_dir / "test.jpg")

        # First scan
        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )
        assert result.exit_code == 0

        # Second scan (should load existing)
        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )
        assert result.exit_code == 0
        assert "Loading existing catalog" in result.output
        assert "Catalog ID:" in result.output
        assert "Created:" in result.output
        assert "Last updated:" in result.output
        assert "Current phase:" in result.output

    def test_analyze_corrupt_catalog_error_handling(self, tmp_path: Path) -> None:
        """Test error handling for corrupted catalog."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir()

        # Create corrupted catalog
        catalog_file = catalog_path / "catalog.json"
        catalog_file.write_text("{ this is not valid json }")

        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        Image.new("RGB", (100, 100), color="brown").save(photos_dir / "test.jpg")

        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )

        # Catalog is now resilient and auto-repairs, so should succeed
        # but may show error messages
        assert result.exit_code == 0 or "Error loading catalog" in result.output


class TestDisplayStatistics:
    """Tests for display_statistics function."""

    def test_display_statistics(self) -> None:
        """Test statistics display."""
        stats = Statistics(
            total_images=100,
            total_videos=10,
            total_size_bytes=1024 * 1024 * 500,
            no_date=5,
        )

        # Should not raise
        display_statistics(stats)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self) -> None:
        """Test default logging setup."""
        # Should not raise
        setup_logging(verbose=False)

    def test_setup_logging_verbose(self) -> None:
        """Test verbose logging setup."""
        # Should not raise
        setup_logging(verbose=True)


class TestAnalyzeErrorHandling:
    """Tests for error handling in analyze CLI."""

    def test_analyze_with_duplicates_needing_review(self, tmp_path: Path) -> None:
        """Test analyze with duplicates that need review."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create duplicate images
        for i in range(2):
            img = Image.new("RGB", (100, 100), color="red")
            img.save(photos_dir / f"dup{i}.jpg")

        # Run with duplicate detection
        result = runner.invoke(
            analyze,
            [
                str(catalog_path),
                "-s",
                str(photos_dir),
                "-w",
                "1",
                "--detect-duplicates",
            ],
        )

        assert result.exit_code == 0
        # Should show duplicate statistics
        assert (
            "Duplicate Detection" in result.output
            or "duplicate" in result.output.lower()
        )

    def test_analyze_repair_with_error(self, tmp_path: Path, monkeypatch) -> None:
        """Test repair mode with error during repair."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir()

        # Create a catalog file that exists but will error on repair
        catalog_file = catalog_path / "catalog.json"
        catalog_file.write_text("{}")

        # Mock the repair method to raise an exception
        from vam_tools.db import CatalogDB as CatalogDatabase

        original_repair = CatalogDatabase.repair

        def failing_repair(self):
            raise RuntimeError("Simulated repair failure")

        monkeypatch.setattr(CatalogDatabase, "repair", failing_repair)

        result = runner.invoke(analyze, [str(catalog_path), "--repair", "-v"])

        # Should exit with error
        assert result.exit_code == 1
        assert "Error during repair" in result.output

        # Restore original method
        monkeypatch.setattr(CatalogDatabase, "repair", original_repair)

    def test_analyze_clear_with_permission_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test clear mode with permission error."""
        runner = CliRunner()
        catalog_path = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create initial catalog
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(photos_dir / "test.jpg")

        result = runner.invoke(
            analyze, [str(catalog_path), "-s", str(photos_dir), "-w", "1"]
        )
        assert result.exit_code == 0

        # Now mock shutil.copy2 to raise permission error
        import shutil

        def failing_copy(*args, **kwargs):
            raise PermissionError("Simulated permission error")

        monkeypatch.setattr(shutil, "copy2", failing_copy)

        # Try to clear - should fail
        result = runner.invoke(
            analyze,
            [str(catalog_path), "-s", str(photos_dir), "-w", "1", "--clear"],
        )

        assert result.exit_code == 1
        assert "Error clearing catalog" in result.output
