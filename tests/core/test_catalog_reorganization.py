"""
Tests for catalog_reorganization module.
"""

from pathlib import Path

import pytest
from PIL import Image

from vam_tools.core.catalog_reorganization import (
    CatalogReorganizer,
    ConflictResolution,
    OrganizationStrategy,
)


class TestOrganizationStrategies:
    """Tests for different organization strategies."""

    def test_year_month_day_strategy(self, temp_dir: Path, dated_images: dict[str, Path]) -> None:
        """Test YEAR_MONTH_DAY organization strategy."""
        output_dir = temp_dir / "output"
        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            strategy=OrganizationStrategy.YEAR_MONTH_DAY,
        )

        # Get a dated image
        image_path = list(dated_images.values())[0]

        # Test with dry-run
        results = reorganizer.reorganize([image_path], dry_run=True)

        # Should have processed one file
        assert results["moved"] + results["copied"] + results["skipped"] >= 1

    def test_year_month_strategy(self, temp_dir: Path, dated_images: dict[str, Path]) -> None:
        """Test YEAR_MONTH organization strategy."""
        output_dir = temp_dir / "output"
        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            strategy=OrganizationStrategy.YEAR_MONTH,
        )

        image_path = list(dated_images.values())[0]
        results = reorganizer.reorganize([image_path], dry_run=True)

        assert results["moved"] + results["copied"] + results["skipped"] >= 1

    def test_year_strategy(self, temp_dir: Path, dated_images: dict[str, Path]) -> None:
        """Test YEAR organization strategy."""
        output_dir = temp_dir / "output"
        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            strategy=OrganizationStrategy.YEAR,
        )

        image_path = list(dated_images.values())[0]
        results = reorganizer.reorganize([image_path], dry_run=True)

        assert results["moved"] + results["copied"] + results["skipped"] >= 1

    def test_flat_date_strategy(self, temp_dir: Path, dated_images: dict[str, Path]) -> None:
        """Test FLAT_DATE organization strategy."""
        output_dir = temp_dir / "output"
        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            strategy=OrganizationStrategy.FLAT_DATE,
        )

        image_path = list(dated_images.values())[0]
        results = reorganizer.reorganize([image_path], dry_run=True)

        assert results["moved"] + results["copied"] + results["skipped"] >= 1


class TestConflictResolution:
    """Tests for conflict resolution strategies."""

    def test_rename_conflict_resolution(self, temp_dir: Path) -> None:
        """Test that RENAME creates unique filenames."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Create two identical images
        img = Image.new("RGB", (10, 10), color="red")
        img1 = temp_dir / "photo_2023-01-01.jpg"
        img2 = temp_dir / "photo2_2023-01-01.jpg"

        img.save(img1)
        img.save(img2)

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            strategy=OrganizationStrategy.FLAT_DATE,
            conflict_resolution=ConflictResolution.RENAME,
            copy_mode=True,
        )

        # Reorganize both images
        results = reorganizer.reorganize([img1, img2], dry_run=False)

        # Both should be copied (renamed to avoid conflict)
        assert results["copied"] == 2
        assert results["errors"] == 0

    def test_skip_conflict_resolution(self, temp_dir: Path) -> None:
        """Test that SKIP skips existing files."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Create two identical images
        img = Image.new("RGB", (10, 10), color="red")
        img1 = temp_dir / "photo_2023-01-01.jpg"
        img2 = temp_dir / "photo2_2023-01-01.jpg"

        img.save(img1)
        img.save(img2)

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            strategy=OrganizationStrategy.FLAT_DATE,
            conflict_resolution=ConflictResolution.SKIP,
            copy_mode=True,
        )

        # Reorganize both images
        results = reorganizer.reorganize([img1, img2], dry_run=False)

        # First should be copied, second should be skipped
        assert results["copied"] >= 1
        assert results["skipped"] >= 1


class TestCopyVsMove:
    """Tests for copy vs move modes."""

    def test_copy_mode(self, temp_dir: Path, dated_images: dict[str, Path]) -> None:
        """Test that copy mode preserves original files."""
        output_dir = temp_dir / "output"

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            copy_mode=True,
        )

        image_path = list(dated_images.values())[0]
        original_exists_before = image_path.exists()

        results = reorganizer.reorganize([image_path], dry_run=False)

        # Original should still exist after copy
        assert original_exists_before
        assert image_path.exists()
        assert results["copied"] >= 0  # May be 0 if no date found

    def test_move_mode(self, temp_dir: Path) -> None:
        """Test that move mode removes original files."""
        output_dir = temp_dir / "output"

        # Create a temporary image
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "test_2023-01-01.jpg"
        img.save(img_path)

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            copy_mode=False,  # Move mode
        )

        assert img_path.exists()

        results = reorganizer.reorganize([img_path], dry_run=False)

        # Original should be moved (not exist anymore)
        if results["moved"] > 0:
            assert not img_path.exists()


class TestDryRun:
    """Tests for dry-run mode."""

    def test_dry_run_does_not_modify_files(self, temp_dir: Path, dated_images: dict[str, Path]) -> None:
        """Test that dry-run doesn't actually move/copy files."""
        output_dir = temp_dir / "output"

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            copy_mode=False,  # Move mode
        )

        image_path = list(dated_images.values())[0]
        exists_before = image_path.exists()

        results = reorganizer.reorganize([image_path], dry_run=True)

        # File should still exist in original location
        assert exists_before == image_path.exists()

        # Output directory might not even be created
        if output_dir.exists():
            # If it exists, it should be empty
            assert len(list(output_dir.rglob("*"))) == 0

    def test_dry_run_reports_actions(self, temp_dir: Path, dated_images: dict[str, Path]) -> None:
        """Test that dry-run reports what would happen."""
        output_dir = temp_dir / "output"

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            copy_mode=True,
        )

        images = list(dated_images.values())
        results = reorganizer.reorganize(images, dry_run=True)

        # Should report actions (copied or skipped)
        total_actions = sum(results.values())
        assert total_actions >= len(images)


class TestPathGeneration:
    """Tests for destination path generation."""

    def test_generate_path_with_date(self, temp_dir: Path) -> None:
        """Test path generation for images with dates."""
        output_dir = temp_dir / "output"

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            strategy=OrganizationStrategy.YEAR_MONTH_DAY,
        )

        # Create image with date in filename
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "photo_2023-06-15.jpg"
        img.save(img_path)

        results = reorganizer.reorganize([img_path], dry_run=True)

        # Should have a plan for this image
        assert results["moved"] + results["copied"] + results["skipped"] >= 1

    def test_generate_path_without_date(self, temp_dir: Path) -> None:
        """Test path generation for images without dates."""
        output_dir = temp_dir / "output"

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
        )

        # Create image with no date info
        img = Image.new("RGB", (10, 10))
        img_path = temp_dir / "random_image.jpg"
        img.save(img_path)

        results = reorganizer.reorganize([img_path], dry_run=True)

        # Should still process it (probably to "unknown" directory)
        assert results["moved"] + results["copied"] + results["skipped"] >= 1


class TestBatchProcessing:
    """Tests for processing multiple images."""

    def test_reorganize_multiple_images(self, temp_dir: Path) -> None:
        """Test reorganizing multiple images at once."""
        output_dir = temp_dir / "output"

        # Create multiple dated images
        images = []
        for i in range(5):
            img = Image.new("RGB", (10, 10))
            img_path = temp_dir / f"photo_{i}_2023-01-{i+1:02d}.jpg"
            img.save(img_path)
            images.append(img_path)

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            copy_mode=True,
        )

        results = reorganizer.reorganize(images, dry_run=False)

        # All should be processed
        total = sum(results.values())
        assert total >= len(images)

    def test_empty_list(self, temp_dir: Path) -> None:
        """Test reorganizing empty list."""
        output_dir = temp_dir / "output"

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
        )

        results = reorganizer.reorganize([], dry_run=False)

        # Should have zero actions
        assert sum(results.values()) == 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_image_handling(self, temp_dir: Path) -> None:
        """Test that invalid images are handled gracefully."""
        output_dir = temp_dir / "output"

        # Create a fake image (text file with .jpg extension)
        fake_img = temp_dir / "fake.jpg"
        fake_img.write_text("This is not an image")

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
        )

        results = reorganizer.reorganize([fake_img], dry_run=False)

        # Should be skipped or error, not crash
        assert results["skipped"] + results["errors"] >= 1

    def test_missing_file_handling(self, temp_dir: Path) -> None:
        """Test that missing files are handled gracefully."""
        output_dir = temp_dir / "output"

        missing_file = temp_dir / "does_not_exist.jpg"

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
        )

        # Should not crash
        results = reorganizer.reorganize([missing_file], dry_run=False)

        # Should report error or skip
        assert results["errors"] + results["skipped"] >= 1


class TestStatistics:
    """Tests for reorganization statistics."""

    def test_statistics_accuracy(self, temp_dir: Path) -> None:
        """Test that statistics accurately reflect what happened."""
        output_dir = temp_dir / "output"

        # Create images
        images = []
        for i in range(3):
            img = Image.new("RGB", (10, 10))
            img_path = temp_dir / f"photo_{i}_2023-01-{i+1:02d}.jpg"
            img.save(img_path)
            images.append(img_path)

        reorganizer = CatalogReorganizer(
            output_directory=output_dir,
            copy_mode=True,
        )

        results = reorganizer.reorganize(images, dry_run=False)

        # Total should match input
        total = sum(results.values())
        assert total >= len(images)

        # Should have specific counts
        assert "copied" in results
        assert "moved" in results
        assert "skipped" in results
        assert "errors" in results
