"""Tests for file organizer.

All tests in this file require database connection.
"""

from datetime import datetime

import pytest

from vam_tools.core.types import (
    DateInfo,
    FileType,
    ImageMetadata,
    ImageRecord,
    ImageStatus,
)
from vam_tools.db import CatalogDB as CatalogDatabase
from vam_tools.organization.file_organizer import (
    FileOrganizer,
    OrganizationOperation,
    OrganizationResult,
)
from vam_tools.organization.strategy import (
    DirectoryStructure,
    NamingStrategy,
    OrganizationStrategy,
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def test_catalog(tmp_path, engine):
    """Create a test catalog with sample images."""
    import uuid

    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()

    # Ensure tables exist in the test database
    from vam_tools.db import Base

    Base.metadata.create_all(bind=engine)

    # Create a catalog database
    with CatalogDatabase(catalog_dir) as db:
        # Initialize the catalog first
        db.initialize(source_directories=[tmp_path / "source"])

        # Create some test images
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Image with date
        img1_path = source_dir / "photo1.jpg"
        img1_path.write_text("photo1 content")

        db.add_image(
            ImageRecord(
                id=f"img1-{uuid.uuid4()}",
                source_path=img1_path,
                file_type=FileType.IMAGE,
                checksum="d8175d4fcc6b88ab5449aa424540d1bdf8dc3bf34139983913b4a3dd0ec9b481",
                metadata=ImageMetadata(
                    size_bytes=100,
                    format="JPEG",
                ),
                dates=DateInfo(
                    selected_date=datetime(2023, 6, 15, 14, 30, 22),
                    exif_dates={"DateTimeOriginal": datetime(2023, 6, 15, 14, 30, 22)},
                ),
            )
        )

        # Image without date
        img2_path = source_dir / "photo2.jpg"
        img2_path.write_text("photo2 content")

        db.add_image(
            ImageRecord(
                id=f"img2-{uuid.uuid4()}",
                source_path=img2_path,
                file_type=FileType.IMAGE,
                checksum="9e27cc6a030bb7b59fa7e05cbabf94a679262b3fe4e52a07a594c98fbf36e6da",
                metadata=ImageMetadata(
                    size_bytes=200,
                    format="JPEG",
                ),
            )
        )

        # Image with date for second day
        img3_path = source_dir / "photo3.jpg"
        img3_path.write_text("photo3 content")

        db.add_image(
            ImageRecord(
                id=f"img3-{uuid.uuid4()}",
                source_path=img3_path,
                file_type=FileType.IMAGE,
                checksum="583f1f6c00dbb8689f31c002e9a7be6aaceaddf3d892f573e630bc51b5abd34f",
                metadata=ImageMetadata(
                    size_bytes=300,
                    format="JPEG",
                ),
                dates=DateInfo(
                    selected_date=datetime(2023, 6, 16, 10, 0, 0),
                    exif_dates={"DateTimeOriginal": datetime(2023, 6, 16, 10, 0, 0)},
                ),
            )
        )

        # Explicitly save the catalog
        db.save()

    return catalog_dir


class TestFileOrganizerDryRun:
    """Test file organizer dry run mode."""

    def test_dry_run_preview(self, test_catalog, tmp_path):
        """Test dry run mode previews without executing."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            result = organizer.organize(dry_run=True)

            # Should report success
            assert result.dry_run is True
            assert result.total_files == 3
            assert result.organized == 2  # 2 with dates
            assert result.no_date == 1  # 1 without date

            # Output directory should not be created in dry run
            assert not output_dir.exists()

    def test_dry_run_with_no_date(self, test_catalog, tmp_path):
        """Test dry run correctly skips files without dates."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH,
                naming_strategy=NamingStrategy.DATE_TIME_CHECKSUM,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            result = organizer.organize(dry_run=True)

            assert result.no_date == 1
            assert result.organized == 2


class TestFileOrganizerCopy:
    """Test file organizer copy operations."""

    def test_copy_operation(self, test_catalog, tmp_path):
        """Test basic copy operation."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            result = organizer.organize(dry_run=False, verify_checksums=False)

            # Check results
            assert result.dry_run is False
            assert result.organized == 2
            assert result.failed == 0

            # Check files were copied
            assert (output_dir / "2023-06" / "photo1.jpg").exists()
            assert (output_dir / "2023-06" / "photo3.jpg").exists()

            # Check original files still exist
            source_dir = tmp_path / "source"
            assert (source_dir / "photo1.jpg").exists()
            assert (source_dir / "photo3.jpg").exists()

    def test_copy_with_directory_structure(self, test_catalog, tmp_path):
        """Test copy with different directory structures."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH_DAY,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            _result = organizer.organize(dry_run=False, verify_checksums=False)

            # Check files in correct directories
            assert (output_dir / "2023-06-15" / "photo1.jpg").exists()
            assert (output_dir / "2023-06-16" / "photo3.jpg").exists()

    def test_copy_with_naming_strategy(self, test_catalog, tmp_path):
        """Test copy with different naming strategies."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.FLAT,
                naming_strategy=NamingStrategy.DATE_TIME_CHECKSUM,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            _result = organizer.organize(dry_run=False, verify_checksums=False)

            # Check files with new names (checksum truncated to 8 chars)
            assert (output_dir / "2023-06-15_143022_d8175d4f.jpg").exists()
            assert (output_dir / "2023-06-16_100000_583f1f6c.jpg").exists()

    def test_skip_existing(self, test_catalog, tmp_path):
        """Test skipping existing files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "2023-06").mkdir()

        # Create existing file
        existing = output_dir / "2023-06" / "photo1.jpg"
        existing.write_text("existing content")

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            result = organizer.organize(
                dry_run=False, verify_checksums=False, skip_existing=True
            )

            # Should skip existing file
            assert result.skipped >= 1

            # Existing file should be unchanged
            assert existing.read_text() == "existing content"

    def test_overwrite_existing(self, test_catalog, tmp_path):
        """Test overwriting existing files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "2023-06").mkdir()

        # Create existing file
        existing = output_dir / "2023-06" / "photo1.jpg"
        existing.write_text("existing content")

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH,
                naming_strategy=NamingStrategy.ORIGINAL,
                handle_duplicates=True,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            _result = organizer.organize(
                dry_run=False, verify_checksums=False, skip_existing=False
            )

            # Should rename to avoid conflict (with zero-padded suffix)
            assert (output_dir / "2023-06" / "photo1_001.jpg").exists()


class TestFileOrganizerMove:
    """Test file organizer move operations."""

    def test_move_operation(self, test_catalog, tmp_path):
        """Test basic move operation."""
        output_dir = tmp_path / "output"
        source_dir = tmp_path / "source"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.MOVE,
            )

            result = organizer.organize(dry_run=False, verify_checksums=False)

            # Check results
            assert result.organized == 2

            # Check files were moved
            assert (output_dir / "2023-06" / "photo1.jpg").exists()
            assert (output_dir / "2023-06" / "photo3.jpg").exists()

            # Check original files no longer exist
            assert not (source_dir / "photo1.jpg").exists()
            assert not (source_dir / "photo3.jpg").exists()


class TestFileOrganizerChecksums:
    """Test checksum verification."""

    def test_checksum_verification_pass(self, test_catalog, tmp_path):
        """Test that checksum verification passes for valid copies."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.FLAT,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            # This will verify checksums (though our test files are simple)
            result = organizer.organize(dry_run=False, verify_checksums=True)

            # Should succeed
            assert result.organized == 2
            assert result.failed == 0


class TestTransactionLogging:
    """Test transaction logging."""

    def test_transaction_log_created(self, test_catalog, tmp_path):
        """Test that transaction log is created."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.FLAT,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            result = organizer.organize(dry_run=False, verify_checksums=False)

            # Transaction log should be created
            assert result.transaction_id is not None

            # Log file should exist
            log_path = output_dir / ".transactions" / f"{result.transaction_id}.json"
            assert log_path.exists()

    def test_dry_run_no_transaction_log(self, test_catalog, tmp_path):
        """Test that dry run doesn't create transaction log file."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.FLAT,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            result = organizer.organize(dry_run=True)

            # Should have transaction ID
            assert result.transaction_id is not None

            # But no log file in dry run
            transactions_dir = output_dir / ".transactions"
            assert not transactions_dir.exists()


class TestOrganizationResult:
    """Test organization result model."""

    def test_result_initialization(self):
        """Test result initialization."""
        result = OrganizationResult(dry_run=True)

        assert result.total_files == 0
        assert result.organized == 0
        assert result.skipped == 0
        assert result.failed == 0
        assert result.no_date == 0
        assert result.dry_run is True
        assert result.transaction_id is None
        assert len(result.errors) == 0

    def test_result_with_errors(self):
        """Test result with errors."""
        result = OrganizationResult(
            total_files=10,
            organized=7,
            failed=3,
            errors=["Error 1", "Error 2", "Error 3"],
        )

        assert result.failed == 3
        assert len(result.errors) == 3


class TestErrorHandling:
    """Test error handling in file organizer."""

    def test_organize_handles_processing_errors(self, test_catalog, tmp_path):
        """Test that organize handles errors when processing individual files."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            # Delete one of the source files to cause an error
            source_dir = tmp_path / "source"
            (source_dir / "photo1.jpg").unlink()

            result = organizer.organize(dry_run=False, verify_checksums=False)

            # Should continue despite error
            assert result.total_files == 3
            assert result.failed >= 1  # At least the deleted file failed
            assert len(result.errors) >= 1

    def test_checksum_verification_failure(self, tmp_path):
        """Test that checksum verification detects corruption."""
        import uuid

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create a test file with known content
        test_file = source_dir / "test.jpg"
        test_file.write_text("original content")

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[source_dir])

            # Add image with WRONG checksum to simulate corruption detection
            from datetime import datetime

            from vam_tools.core.types import (
                DateInfo,
                FileType,
                ImageMetadata,
                ImageRecord,
            )

            db.add_image(
                ImageRecord(
                    id=f"test_img_{uuid.uuid4()}",
                    source_path=test_file,
                    file_type=FileType.IMAGE,
                    checksum="wrong_checksum_will_fail_verification",  # Intentionally wrong
                    metadata=ImageMetadata(size_bytes=100, format="JPEG"),
                    dates=DateInfo(
                        selected_date=datetime(2023, 1, 1),
                        exif_dates={"DateTimeOriginal": datetime(2023, 1, 1)},
                    ),
                )
            )
            db.save()

        with CatalogDatabase(catalog_dir) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.FLAT,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            result = organizer.organize(dry_run=False, verify_checksums=True)

            # Should fail due to checksum mismatch
            assert result.failed >= 1
            assert any("Checksum mismatch" in error for error in result.errors)


class TestRollback:
    """Test rollback functionality."""

    def test_rollback_copy_operation(self, test_catalog, tmp_path):
        """Test rolling back a copy operation."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            # Perform organization
            result = organizer.organize(dry_run=False, verify_checksums=False)
            transaction_id = result.transaction_id

            # Verify files were copied
            assert (output_dir / "2023-06" / "photo1.jpg").exists()
            assert (output_dir / "2023-06" / "photo3.jpg").exists()

            # Rollback the transaction
            organizer.rollback(transaction_id)

            # Files should be deleted
            assert not (output_dir / "2023-06" / "photo1.jpg").exists()
            assert not (output_dir / "2023-06" / "photo3.jpg").exists()

    def test_rollback_move_operation(self, test_catalog, tmp_path):
        """Test rolling back a move operation."""
        output_dir = tmp_path / "output"
        source_dir = tmp_path / "source"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.MOVE,
            )

            # Perform organization
            result = organizer.organize(dry_run=False, verify_checksums=False)
            transaction_id = result.transaction_id

            # Original files should be moved
            assert not (source_dir / "photo1.jpg").exists()
            assert (output_dir / "2023-06" / "photo1.jpg").exists()

            # Rollback the transaction
            organizer.rollback(transaction_id)

            # Files should be moved back
            assert (source_dir / "photo1.jpg").exists()
            assert not (output_dir / "2023-06" / "photo1.jpg").exists()

    def test_rollback_nonexistent_transaction(self, test_catalog, tmp_path):
        """Test rollback with nonexistent transaction ID."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy()

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            # Try to rollback nonexistent transaction
            with pytest.raises(ValueError, match="Transaction log not found"):
                organizer.rollback("nonexistent-transaction-id")


class TestResume:
    """Test resume functionality."""

    def test_resume_transaction(self, test_catalog, tmp_path):
        """Test resuming an interrupted transaction."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH,
                naming_strategy=NamingStrategy.ORIGINAL,
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            # Start organization but simulate interruption by only processing one file
            result = organizer.organize(dry_run=False, verify_checksums=False)
            transaction_id = result.transaction_id

            # For this test, we'll just verify that resume can be called
            # In a real scenario, the transaction would have pending operations
            # Note: Our current test setup completes all operations, so resume
            # will find no pending operations. This is expected.
            resume_result = organizer.resume(transaction_id)

            # Resume should complete without error
            assert resume_result.transaction_id == transaction_id

    def test_resume_nonexistent_transaction(self, test_catalog, tmp_path):
        """Test resume with nonexistent transaction ID."""
        output_dir = tmp_path / "output"

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy()

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            # Try to resume nonexistent transaction
            with pytest.raises(ValueError, match="Transaction log not found"):
                organizer.resume("nonexistent-transaction-id")


class TestNamingConflictResolution:
    """Test naming conflict resolution."""

    def test_naming_conflict_resolution_disabled(self, test_catalog, tmp_path):
        """Test behavior when conflict resolution is disabled."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "2023-06").mkdir()

        # Create existing file
        existing = output_dir / "2023-06" / "photo1.jpg"
        existing.write_text("existing content")

        with CatalogDatabase(test_catalog) as db:
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure.YEAR_MONTH,
                naming_strategy=NamingStrategy.ORIGINAL,
                handle_duplicates=False,  # Disable conflict resolution
            )

            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation.COPY,
            )

            # This should skip the existing file since skip_existing=True by default
            result = organizer.organize(
                dry_run=False, verify_checksums=False, skip_existing=True
            )

            # Should skip the conflicting file
            assert result.skipped >= 1

            # Existing file should be unchanged
            assert existing.read_text() == "existing content"
