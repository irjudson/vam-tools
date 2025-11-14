"""Tests for organize CLI command."""

from datetime import datetime

import pytest
from click.testing import CliRunner

from vam_tools.cli.organize import organize
from vam_tools.core.database import CatalogDatabase
from vam_tools.core.types import DateInfo, FileType, ImageMetadata, ImageRecord


@pytest.fixture
def test_catalog_with_images(tmp_path):
    """Create a test catalog with images for organization."""
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()

    # Create source directory with test images
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    with CatalogDatabase(catalog_dir) as db:
        db.initialize()  # Initialize schema

        # Store source directories in catalog_config
        db.execute(
            "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            (f"source_directory_{source_dir.name}", str(source_dir)),
        )
        db.execute(
            "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("created", datetime.now().isoformat()),
        )
        db.execute(
            "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("last_updated", datetime.now().isoformat()),
        )
        db.execute(
            "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("catalog_id", "test-catalog-id"),
        )
        db.execute(
            "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("version", "2.0.0"),
        )

        # Image with date
        img1_path = source_dir / "photo1.jpg"
        img1_path.write_text("photo1 content")
        db.execute(
            """
            INSERT INTO images (
                id, source_path, file_size, file_hash, format,
                width, height, created_at, modified_at, indexed_at,
                date_taken, camera_make, camera_model, lens_model,
                focal_length, aperture, shutter_speed, iso,
                gps_latitude, gps_longitude, quality_score, is_corrupted,
                perceptual_hash, features_vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "img1",
                str(img1_path),
                100,
                "d8175d4fcc6b88ab5449aa424540d1bdf8dc3bf34139983913b4a3dd0ec9b481",
                "JPEG",
                None,
                None,  # width, height
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                datetime(2023, 6, 15, 14, 30, 22).isoformat(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,  # exif, gps
                None,  # quality_score
                0,  # is_corrupted
                None,
                None,  # perceptual_hash, features_vector
            ),
        )

        # Image with different date
        img2_path = source_dir / "photo2.jpg"
        img2_path.write_text("photo2 content")
        db.execute(
            """
            INSERT INTO images (
                id, source_path, file_size, file_hash, format,
                width, height, created_at, modified_at, indexed_at,
                date_taken, camera_make, camera_model, lens_model,
                focal_length, aperture, shutter_speed, iso,
                gps_latitude, gps_longitude, quality_score, is_corrupted,
                perceptual_hash, features_vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "img2",
                str(img2_path),
                200,
                "9e27cc6a030bb7b59fa7e05cbabf94a679262b3fe4e52a07a594c98fbf36e6da",
                "JPEG",
                None,
                None,  # width, height
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                datetime(2023, 7, 20, 10, 0, 0).isoformat(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,  # exif, gps
                None,  # quality_score
                0,  # is_corrupted
                None,
                None,  # perceptual_hash, features_vector
            ),
        )

        # Image without date
        img3_path = source_dir / "photo3.jpg"
        img3_path.write_text("photo3 content")
        db.execute(
            """
            INSERT INTO images (
                id, source_path, file_size, file_hash, format,
                width, height, created_at, modified_at, indexed_at,
                date_taken, camera_make, camera_model, lens_model,
                focal_length, aperture, shutter_speed, iso,
                gps_latitude, gps_longitude, quality_score, is_corrupted,
                perceptual_hash, features_vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "img3",
                str(img3_path),
                300,
                "583f1f6c00dbb8689f31c002e9a7be6aaceaddf3d892f573e630bc51b5abd34f",
                "JPEG",
                None,
                None,  # width, height
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                None,  # date_taken
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,  # exif, gps
                None,  # quality_score
                0,  # is_corrupted
                None,
                None,  # perceptual_hash, features_vector
            ),
        )

    return catalog_dir


class TestOrganizeCLI:
    """Test organize CLI command."""

    def test_organize_dry_run(self, test_catalog_with_images, tmp_path):
        """Test organize with dry-run mode."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--dry-run",
            ],
        )

        # Should succeed
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "Organization complete" in result.output

        # Output directory should not be created in dry-run
        assert not output_dir.exists()

    def test_organize_copy_operation(self, test_catalog_with_images, tmp_path):
        """Test organize with copy operation."""
        output_dir = tmp_path / "output"
        source_dir = tmp_path / "source"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--operation",
                "copy",
                "--no-verify",  # Skip checksum verification for speed
            ],
        )

        # Should succeed
        assert result.exit_code == 0
        assert "Organization complete" in result.output

        # Files should be copied (default structure is YYYY-MM)
        assert (output_dir / "2023-06").exists()
        assert (output_dir / "2023-07").exists()

        # Original files should still exist (copy, not move)
        assert (source_dir / "photo1.jpg").exists()
        assert (source_dir / "photo2.jpg").exists()

    def test_organize_move_operation_with_confirmation(
        self, test_catalog_with_images, tmp_path
    ):
        """Test organize with move operation (with user confirmation)."""
        output_dir = tmp_path / "output"
        source_dir = tmp_path / "source"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--operation",
                "move",
                "--no-verify",
            ],
            input="y\n",  # Confirm move operation
        )

        # Should succeed
        assert result.exit_code == 0
        assert "WARNING: MOVE operation" in result.output
        assert "Organization complete" in result.output

        # Files should be moved
        assert (output_dir / "2023-06").exists()
        assert (output_dir / "2023-07").exists()

        # Original files should NOT exist (moved)
        assert not (source_dir / "photo1.jpg").exists()
        assert not (source_dir / "photo2.jpg").exists()

    def test_organize_move_operation_cancelled(
        self, test_catalog_with_images, tmp_path
    ):
        """Test organize with move operation cancelled by user."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--operation",
                "move",
            ],
            input="n\n",  # Cancel move operation
        )

        # Should exit without error but cancelled
        assert result.exit_code == 0
        assert "Cancelled" in result.output

        # Output directory should not be created
        assert not output_dir.exists()

    def test_organize_custom_structure_year_month_day(
        self, test_catalog_with_images, tmp_path
    ):
        """Test organize with YYYY-MM-DD structure."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--structure",
                "YYYY-MM-DD",
                "--no-verify",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # Files should be in date-specific directories
        assert (output_dir / "2023-06-15").exists()
        assert (output_dir / "2023-07-20").exists()

    def test_organize_custom_structure_year_slash_month(
        self, test_catalog_with_images, tmp_path
    ):
        """Test organize with YYYY/MM structure."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--structure",
                "YYYY/MM",
                "--no-verify",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # Files should be in nested year/month directories
        assert (output_dir / "2023" / "06").exists()
        assert (output_dir / "2023" / "07").exists()

    def test_organize_custom_structure_year_only(
        self, test_catalog_with_images, tmp_path
    ):
        """Test organize with YYYY structure."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--structure",
                "YYYY",
                "--no-verify",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # Files should be in year directories
        assert (output_dir / "2023").exists()

    def test_organize_custom_structure_flat(self, test_catalog_with_images, tmp_path):
        """Test organize with FLAT structure."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--structure",
                "FLAT",
                "--no-verify",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # All files should be directly in output directory
        # (with date_time_checksum naming by default)
        files = list(output_dir.glob("*.jpg"))
        assert len(files) == 2  # 2 images with dates

    def test_organize_naming_strategy_original(
        self, test_catalog_with_images, tmp_path
    ):
        """Test organize with original naming strategy."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--naming",
                "original",
                "--no-verify",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # Files should keep original names
        assert (output_dir / "2023-06" / "photo1.jpg").exists()
        assert (output_dir / "2023-07" / "photo2.jpg").exists()

    def test_organize_naming_strategy_checksum(
        self, test_catalog_with_images, tmp_path
    ):
        """Test organize with checksum naming strategy."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--naming",
                "checksum",
                "--no-verify",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # Files should be named with checksums
        assert (
            output_dir
            / "2023-06"
            / "d8175d4fcc6b88ab5449aa424540d1bdf8dc3bf34139983913b4a3dd0ec9b481.jpg"
        ).exists()

    def test_organize_naming_strategy_date_time_checksum(
        self, test_catalog_with_images, tmp_path
    ):
        """Test organize with date_time_checksum naming strategy."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--naming",
                "date_time_checksum",
                "--no-verify",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # Files should be named with date, time, and truncated checksum
        assert (output_dir / "2023-06" / "2023-06-15_143022_d8175d4f.jpg").exists()
        assert (output_dir / "2023-07" / "2023-07-20_100000_9e27cc6a.jpg").exists()

    def test_organize_naming_strategy_date_time_original(
        self, test_catalog_with_images, tmp_path
    ):
        """Test organize with date_time_original naming strategy."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--naming",
                "date_time_original",
                "--no-verify",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # Files should be named with date, time, and original filename
        assert (output_dir / "2023-06" / "2023-06-15_143022_photo1.jpg").exists()
        assert (output_dir / "2023-07" / "2023-07-20_100000_photo2.jpg").exists()

    def test_organize_verbose_mode(self, test_catalog_with_images, tmp_path):
        """Test organize with verbose output."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--verbose",
                "--dry-run",
            ],
        )

        # Should succeed
        assert result.exit_code == 0
        assert "Organization complete" in result.output

    def test_organize_nonexistent_catalog(self, tmp_path):
        """Test organize with nonexistent catalog."""
        catalog_dir = tmp_path / "nonexistent"
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(catalog_dir),
                str(output_dir),
            ],
        )

        # Should fail with appropriate error
        assert result.exit_code != 0

    def test_organize_with_overwrite(self, test_catalog_with_images, tmp_path):
        """Test organize with overwrite flag."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "2023-06").mkdir()

        # Create existing file
        existing = output_dir / "2023-06" / "photo1.jpg"
        existing.write_text("existing content")

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--naming",
                "original",
                "--overwrite",
                "--no-verify",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # File should be renamed to avoid conflict
        assert (output_dir / "2023-06" / "photo1_001.jpg").exists()

    def test_organize_skip_existing(self, test_catalog_with_images, tmp_path):
        """Test organize skips existing files by default."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "2023-06").mkdir()

        # Create existing file
        existing = output_dir / "2023-06" / "photo1.jpg"
        existing.write_text("existing content")

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--naming",
                "original",
                "--no-verify",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # Existing file should be unchanged
        assert existing.read_text() == "existing content"

        # Results should show skipped files
        assert "Skipped" in result.output

    def test_organize_transaction_id_shown(self, test_catalog_with_images, tmp_path):
        """Test that transaction ID is shown after organization."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--no-verify",
            ],
        )

        # Should succeed and show transaction ID
        assert result.exit_code == 0
        assert "Transaction ID:" in result.output
        assert "rollback" in result.output.lower()

    def test_organize_shows_configuration(self, test_catalog_with_images, tmp_path):
        """Test that organize shows configuration before running."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--dry-run",
            ],
        )

        # Should show configuration
        assert "Organization Configuration:" in result.output
        assert "Catalog:" in result.output
        assert "Output:" in result.output
        assert "Operation:" in result.output
        assert "Structure:" in result.output
        assert "Naming:" in result.output

    def test_organize_shows_results_table(self, test_catalog_with_images, tmp_path):
        """Test that organize shows results in a table."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--dry-run",
            ],
        )

        # Should show results table
        assert "Results" in result.output
        assert "Total files" in result.output
        assert "Organized" in result.output
        assert "Skipped" in result.output
        assert "Failed" in result.output
        assert "No date" in result.output


class TestOrganizeErrorHandling:
    """Test error handling in organize CLI."""

    def test_organize_handles_empty_catalog(self, tmp_path):
        """Test organize handles empty catalog gracefully."""
        # Create an empty catalog directory
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(catalog_dir),
                str(output_dir),
                "--dry-run",
            ],
        )

        # Catalog system handles missing catalog gracefully with warning
        # Command should complete but with 0 files organized
        assert result.exit_code == 0
        assert "Organization complete" in result.output

    def test_organize_with_corrupt_catalog_file(self, tmp_path):
        """Test organize with corrupt catalog file."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        # Create corrupt catalog file
        catalog_file = catalog_dir / "catalog.json"
        catalog_file.write_text("{corrupt json")

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(catalog_dir),
                str(output_dir),
                "--dry-run",
            ],
        )

        # Catalog system logs error but continues gracefully
        # The command completes with 0 files organized
        assert result.exit_code == 0
        assert "Organization complete" in result.output


class TestDisplayResult:
    """Test _display_result helper function."""

    def test_display_shows_all_metrics(self, test_catalog_with_images, tmp_path):
        """Test that display shows all result metrics."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--dry-run",
            ],
        )

        # All metrics should be displayed
        assert "Total files" in result.output
        assert "Organized" in result.output
        assert "Skipped" in result.output
        assert "Failed" in result.output
        assert "No date" in result.output

    def test_display_shows_dry_run_warning(self, test_catalog_with_images, tmp_path):
        """Test that dry run shows appropriate warning."""
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                str(output_dir),
                "--dry-run",
            ],
        )

        # Should show dry run warning
        assert "This was a DRY RUN" in result.output
        assert "no files were modified" in result.output


class TestOrganizeTransactionFeatures:
    """Test transaction-related features (rollback, resume)."""

    def test_rollback_with_invalid_transaction_id(self, test_catalog_with_images):
        """Test rollback with invalid transaction ID."""
        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                "/tmp/output",
                "--rollback",
                "nonexistent-transaction-id",
            ],
        )

        # Should fail gracefully
        assert result.exit_code == 1
        assert "Rollback failed" in result.output or "failed" in result.output.lower()

    def test_resume_with_invalid_transaction_id(self, test_catalog_with_images):
        """Test resume with invalid transaction ID."""
        runner = CliRunner()
        result = runner.invoke(
            organize,
            [
                str(test_catalog_with_images),
                "/tmp/output",
                "--resume",
                "nonexistent-transaction-id",
            ],
        )

        # Should fail gracefully
        assert result.exit_code == 1
        assert "Resume failed" in result.output or "failed" in result.output.lower()
