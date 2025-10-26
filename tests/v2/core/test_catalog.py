"""
Tests for catalog database.
"""

import json
from pathlib import Path

from vam_tools.v2.core.catalog import CatalogDatabase
from vam_tools.v2.core.types import (
    CatalogConfiguration,
    CatalogPhase,
    DuplicateGroup,
    FileType,
    ImageMetadata,
    ImageRecord,
    ImageStatus,
    ReviewItem,
    ReviewPriority,
    ReviewType,
    Statistics,
)


class TestCatalogDatabase:
    """Tests for CatalogDatabase."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test catalog initialization."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[tmp_path])

            # Check catalog file was created
            assert (catalog_dir / ".catalog.json").exists()

            # Check state
            state = db.get_state()
            assert state.phase == CatalogPhase.ANALYZING
            assert state.catalog_id is not None

            # Check configuration
            config = db.get_configuration()
            assert tmp_path in config.source_directories

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test catalog works as context manager."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])
            assert db._lock_fd is not None

        # Lock should be released after exit
        # (Can't directly test this, but no exception is good)

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading catalog."""
        catalog_dir = tmp_path / "catalog"

        # Create and save
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[tmp_path])
            stats = db.get_statistics()
            stats.total_images = 42
            db.update_statistics(stats)
            db.save()  # Explicitly save

        # Load in new session
        with CatalogDatabase(catalog_dir) as db:
            stats = db.get_statistics()
            assert stats.total_images == 42

    def test_add_and_get_image(self, tmp_path: Path) -> None:
        """Test adding and retrieving images."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            # Create image record
            image = ImageRecord(
                id="test123",
                source_path=tmp_path / "test.jpg",
                file_type=FileType.IMAGE,
                checksum="abc123",
                metadata=ImageMetadata(format="JPEG", width=800, height=600),
                status=ImageStatus.ANALYZING,
            )

            # Add image
            db.add_image(image)

            # Retrieve image
            retrieved = db.get_image("test123")
            assert retrieved is not None
            assert retrieved.id == "test123"
            assert retrieved.checksum == "abc123"
            assert retrieved.metadata.format == "JPEG"

    def test_update_image(self, tmp_path: Path) -> None:
        """Test updating existing image."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            # Add image
            image = ImageRecord(
                id="test123",
                source_path=tmp_path / "test.jpg",
                file_type=FileType.IMAGE,
                checksum="abc123",
                metadata=ImageMetadata(),
                status=ImageStatus.ANALYZING,
            )
            db.add_image(image)

            # Update image
            image.metadata.width = 1920
            image.metadata.height = 1080
            db.update_image(image)

            # Verify update
            retrieved = db.get_image("test123")
            assert retrieved.metadata.width == 1920
            assert retrieved.metadata.height == 1080

    def test_list_images(self, tmp_path: Path) -> None:
        """Test listing all images."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            # Add multiple images
            for i in range(5):
                image = ImageRecord(
                    id=f"img{i}",
                    source_path=tmp_path / f"test{i}.jpg",
                    file_type=FileType.IMAGE,
                    checksum=f"sum{i}",
                    metadata=ImageMetadata(),
                    status=ImageStatus.ANALYZING,
                )
                db.add_image(image)

            # List images
            images = db.list_images()
            assert len(images) == 5
            assert all(isinstance(img, ImageRecord) for img in images)

    def test_get_all_images(self, tmp_path: Path) -> None:
        """Test getting all images as dict."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            # Add images
            for i in range(3):
                image = ImageRecord(
                    id=f"img{i}",
                    source_path=tmp_path / f"test{i}.jpg",
                    file_type=FileType.IMAGE,
                    checksum=f"sum{i}",
                    metadata=ImageMetadata(),
                    status=ImageStatus.ANALYZING,
                )
                db.add_image(image)

            # Get as dict
            images_dict = db.get_all_images()
            assert isinstance(images_dict, dict)
            assert len(images_dict) == 3
            assert "img0" in images_dict
            assert "img1" in images_dict
            assert "img2" in images_dict

    def test_has_image_by_path(self, tmp_path: Path) -> None:
        """Test checking if image exists by path."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            test_path = tmp_path / "test.jpg"
            image = ImageRecord(
                id="test123",
                source_path=test_path,
                file_type=FileType.IMAGE,
                checksum="abc123",
                metadata=ImageMetadata(),
                status=ImageStatus.ANALYZING,
            )
            db.add_image(image)

            # Check path exists
            assert db.has_image_by_path(test_path) is True
            assert db.has_image_by_path(tmp_path / "other.jpg") is False

    def test_update_state(self, tmp_path: Path) -> None:
        """Test updating catalog state."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            state = db.get_state()
            state.phase = CatalogPhase.ANALYZING
            state.images_processed = 100
            state.progress_percentage = 50.0
            db.update_state(state)

            # Verify update
            retrieved_state = db.get_state()
            assert retrieved_state.phase == CatalogPhase.ANALYZING
            assert retrieved_state.images_processed == 100
            assert retrieved_state.progress_percentage == 50.0

    def test_update_statistics(self, tmp_path: Path) -> None:
        """Test updating catalog statistics."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            stats = Statistics(
                total_images=100,
                total_videos=20,
                total_size_bytes=1024 * 1024 * 500,
                no_date=5,
            )
            db.update_statistics(stats)

            # Verify update
            retrieved_stats = db.get_statistics()
            assert retrieved_stats.total_images == 100
            assert retrieved_stats.total_videos == 20
            assert retrieved_stats.no_date == 5

    def test_checkpoint(self, tmp_path: Path) -> None:
        """Test checkpoint functionality."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            # Add some data
            stats = db.get_statistics()
            stats.total_images = 50
            db.update_statistics(stats)

            # Force checkpoint
            db.checkpoint(force=True)

            # Verify data was saved
            state = db.get_state()
            assert state.last_checkpoint is not None

    def test_backup_creation(self, tmp_path: Path) -> None:
        """Test that backups are created on save."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

        # Modify and save again
        with CatalogDatabase(catalog_dir) as db:
            stats = db.get_statistics()
            stats.total_images = 100
            db.update_statistics(stats)
            db.save()

        # Check backup exists
        backup_file = catalog_dir / ".catalog.backup.json"
        assert backup_file.exists()

    def test_add_duplicate_group(self, tmp_path: Path) -> None:
        """Test adding duplicate group."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            group = DuplicateGroup(
                id="dup1",
                images=["img1", "img2", "img3"],
                primary="img1",
            )
            db.add_duplicate_group(group)

            # Retrieve group
            retrieved = db.get_duplicate_group("dup1")
            assert retrieved is not None
            assert retrieved.id == "dup1"
            assert len(retrieved.images) == 3
            assert retrieved.primary == "img1"

    def test_save_multiple_duplicate_groups(self, tmp_path: Path) -> None:
        """Test saving multiple duplicate groups at once."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            groups = [
                DuplicateGroup(id=f"dup{i}", images=[f"img{i}a", f"img{i}b"])
                for i in range(5)
            ]
            db.save_duplicate_groups(groups)

            # Retrieve all groups
            all_groups = db.get_duplicate_groups()
            assert len(all_groups) == 5

    def test_get_duplicate_groups(self, tmp_path: Path) -> None:
        """Test getting all duplicate groups."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            # Add groups
            for i in range(3):
                group = DuplicateGroup(id=f"dup{i}", images=[f"a{i}", f"b{i}"])
                db.add_duplicate_group(group)

            # Get all groups
            groups = db.get_duplicate_groups()
            assert len(groups) == 3
            assert all(isinstance(g, DuplicateGroup) for g in groups)

    def test_add_review_item(self, tmp_path: Path) -> None:
        """Test adding item to review queue."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            review_item = ReviewItem(
                id="review1",
                type=ReviewType.DATE_CONFLICT,
                priority=ReviewPriority.HIGH,
                images=["img1"],
                description="Date conflict detected",
            )
            db.add_review_item(review_item)
            db.save()

            # Get review queue
            queue = db.get_review_queue()
            assert len(queue) == 1
            assert queue[0].id == "review1"
            assert queue[0].type == ReviewType.DATE_CONFLICT

    def test_get_review_queue(self, tmp_path: Path) -> None:
        """Test getting review queue."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            # Add multiple review items
            for i in range(3):
                item = ReviewItem(
                    id=f"review{i}",
                    type=ReviewType.NO_DATE,
                    priority=ReviewPriority.LOW,
                    images=[f"img{i}"],
                    description="Test item",
                )
                db.add_review_item(item)

            queue = db.get_review_queue()
            assert len(queue) == 3

    def test_load_nonexistent_catalog(self, tmp_path: Path) -> None:
        """Test loading catalog that doesn't exist."""
        catalog_dir = tmp_path / "nonexistent"

        with CatalogDatabase(catalog_dir) as db:
            # Should not crash
            assert db._data is None

    def test_get_image_nonexistent(self, tmp_path: Path) -> None:
        """Test getting image that doesn't exist."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            result = db.get_image("nonexistent")
            assert result is None

    def test_custom_configuration(self, tmp_path: Path) -> None:
        """Test initialization with custom configuration."""
        catalog_dir = tmp_path / "catalog"

        config = CatalogConfiguration(
            checkpoint_interval_seconds=60,
        )

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[tmp_path], config=config)

            retrieved_config = db.get_configuration()
            assert retrieved_config.checkpoint_interval_seconds == 60

    def test_empty_catalog_statistics(self, tmp_path: Path) -> None:
        """Test statistics on empty catalog."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            stats = db.get_statistics()
            assert stats.total_images == 0
            assert stats.total_videos == 0
            assert stats.total_size_bytes == 0

    def test_checkpoint_interval(self, tmp_path: Path) -> None:
        """Test checkpoint respects interval."""
        catalog_dir = tmp_path / "catalog"

        config = CatalogConfiguration(checkpoint_interval_seconds=3600)  # 1 hour

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[], config=config)

            # First checkpoint (forced)
            db.checkpoint(force=True)
            first_checkpoint = db._last_checkpoint

            # Try checkpoint immediately (should be skipped)
            db.checkpoint(force=False)
            # Last checkpoint should not have changed
            assert db._last_checkpoint == first_checkpoint

    def test_path_index_building(self, tmp_path: Path) -> None:
        """Test that path index is built correctly."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            # Add image
            test_path = tmp_path / "test.jpg"
            image = ImageRecord(
                id="test123",
                source_path=test_path,
                file_type=FileType.IMAGE,
                checksum="abc",
                metadata=ImageMetadata(),
                status=ImageStatus.ANALYZING,
            )
            db.add_image(image)
            db.save()  # Save before reloading

        # Reload catalog (rebuilds index)
        with CatalogDatabase(catalog_dir) as db:
            # Index should be rebuilt on load
            assert db.has_image_by_path(test_path) is True

    def test_duplicate_group_nonexistent(self, tmp_path: Path) -> None:
        """Test getting nonexistent duplicate group."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            result = db.get_duplicate_group("nonexistent")
            assert result is None

    def test_catalog_version(self, tmp_path: Path) -> None:
        """Test catalog version is stored."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

        # Check version in file
        catalog_file = catalog_dir / ".catalog.json"
        with open(catalog_file) as f:
            data = json.load(f)

        assert "version" in data
        assert data["version"] == "2.0.0"
