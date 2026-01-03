"""Tests for tag manager."""

from pathlib import Path

import pytest

from lumina.analysis.tag_manager import TagManager
from lumina.analysis.tag_taxonomy import TagCategory
from lumina.db import CatalogDB as CatalogDatabase


@pytest.mark.integration
class TestTagManager:
    """Tests for TagManager class."""

    @pytest.fixture
    def db(self, test_catalog_db, tmp_path: Path) -> CatalogDatabase:
        """Create and initialize database using transactional fixture."""
        catalog_path = tmp_path / "test_catalog"
        database = test_catalog_db(catalog_path)
        database.initialize()
        return database

    @pytest.fixture
    def manager(self, db: CatalogDatabase) -> TagManager:
        """Create tag manager instance."""
        return TagManager(db)

    def _create_test_image(self, db: CatalogDatabase, image_id: str) -> None:
        """Helper to create a dummy image in database."""
        db.execute(
            """
            INSERT INTO images (
                id, catalog_id, source_path, file_type, checksum, size_bytes,
                dates, metadata, status_id, processing_flags, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, '{}', '{}', ?, '{}', NOW(), NOW())
            ON CONFLICT (id) DO NOTHING
            """,
            (
                image_id,
                db.catalog_id,
                f"/tmp/{image_id}.jpg",
                "image",
                f"hash_{image_id}",
                1000,
                "active",
            ),
        )

    def test_initialization(self, manager: TagManager) -> None:
        """Test manager initialization."""
        assert manager.db is not None
        assert manager.taxonomy is not None
        assert len(manager.taxonomy.get_all_tags()) > 0

    def test_sync_taxonomy_to_db(self, manager: TagManager) -> None:
        """Test syncing taxonomy to database."""
        count = manager.sync_taxonomy_to_db()

        # Should sync all tags
        assert count >= 35

        # Verify tags in database
        cursor = manager.db.execute("SELECT COUNT(*) FROM tags")
        db_count = cursor.fetchone()[0]
        assert db_count == count

        # Syncing again should not add duplicates
        count2 = manager.sync_taxonomy_to_db()
        assert count2 == 0

    def test_add_image_tag(self, manager: TagManager) -> None:
        """Test adding tag to image."""
        manager.sync_taxonomy_to_db()
        self._create_test_image(manager.db, "img1")

        # Add valid tag
        success = manager.add_image_tag("img1", "dogs", 0.95, "clip")
        assert success

        # Verify in database
        cursor = manager.db.execute(
            """
            SELECT t.name, it.confidence, it.source
            FROM image_tags it
            JOIN tags t ON it.tag_id = t.id
            WHERE it.image_id = ?
            """,
            ("img1",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "dogs"  # name
        assert row[1] == 0.95  # confidence
        assert row[2] == "clip"  # source

    def test_add_image_tag_case_insensitive(self, manager: TagManager) -> None:
        """Test tag name is case-insensitive."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        self._create_test_image(manager.db, "img2")
        assert manager.add_image_tag("img1", "DOGS", 0.95, "clip")
        assert manager.add_image_tag("img2", "DoGs", 0.90, "clip")

        tags1 = manager.get_image_tags("img1")
        tags2 = manager.get_image_tags("img2")
        assert tags1[0][0] == "dogs"
        assert tags2[0][0] == "dogs"

    def test_add_image_tag_invalid(self, manager: TagManager) -> None:
        """Test adding invalid tag."""
        manager.sync_taxonomy_to_db()

        # Non-existent tag
        success = manager.add_image_tag("img1", "nonexistent_tag", 0.95, "clip")
        assert not success

    def test_add_image_tag_invalid_confidence(self, manager: TagManager) -> None:
        """Test adding tag with invalid confidence."""
        manager.sync_taxonomy_to_db()

        with pytest.raises(ValueError, match="Confidence must be"):
            manager.add_image_tag("img1", "dogs", 1.5, "clip")

        with pytest.raises(ValueError, match="Confidence must be"):
            manager.add_image_tag("img1", "dogs", -0.1, "clip")

    def test_add_image_tag_invalid_source(self, manager: TagManager) -> None:
        """Test adding tag with invalid source."""
        manager.sync_taxonomy_to_db()

        with pytest.raises(ValueError, match="Source must be"):
            manager.add_image_tag("img1", "dogs", 0.95, "invalid_source")

    def test_add_image_tag_replace(self, manager: TagManager) -> None:
        """Test replacing existing tag updates confidence."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        # Add tag
        manager.add_image_tag("img1", "dogs", 0.80, "clip")

        # Replace with higher confidence
        manager.add_image_tag("img1", "dogs", 0.95, "manual")

        # Verify updated
        tags = manager.get_image_tags("img1")
        assert len(tags) == 1
        assert tags[0] == ("dogs", 0.95, "manual")

    def test_remove_image_tag(self, manager: TagManager) -> None:
        """Test removing tag from image."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        # Add then remove
        manager.add_image_tag("img1", "dogs", 0.95, "clip")
        removed = manager.remove_image_tag("img1", "dogs")
        assert removed

        # Verify removed
        tags = manager.get_image_tags("img1")
        assert len(tags) == 0

        # Removing again should return False
        removed = manager.remove_image_tag("img1", "dogs")
        assert not removed

    def test_remove_image_tag_invalid(self, manager: TagManager) -> None:
        """Test removing non-existent tag."""
        manager.sync_taxonomy_to_db()

        removed = manager.remove_image_tag("img1", "nonexistent_tag")
        assert not removed

    def test_get_image_tags(self, manager: TagManager) -> None:
        """Test getting all tags for an image."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        # Add multiple tags
        manager.add_image_tag("img1", "dogs", 0.95, "clip")
        manager.add_image_tag("img1", "outdoor", 0.85, "clip")
        manager.add_image_tag("img1", "daylight", 0.90, "yolo")

        # Get all tags
        tags = manager.get_image_tags("img1")
        assert len(tags) == 3

        # Should be sorted by confidence descending
        assert tags[0][1] == 0.95  # dogs
        assert tags[1][1] == 0.90  # daylight
        assert tags[2][1] == 0.85  # outdoor

    def test_get_image_tags_min_confidence(self, manager: TagManager) -> None:
        """Test filtering tags by confidence."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        manager.add_image_tag("img1", "dogs", 0.95, "clip")
        manager.add_image_tag("img1", "outdoor", 0.70, "clip")
        manager.add_image_tag("img1", "daylight", 0.50, "clip")

        # Get high-confidence tags only
        tags = manager.get_image_tags("img1", min_confidence=0.80)
        assert len(tags) == 1
        assert tags[0][0] == "dogs"

    def test_get_image_tags_no_tags(self, manager: TagManager) -> None:
        """Test getting tags for untagged image."""
        manager.sync_taxonomy_to_db()

        tags = manager.get_image_tags("img_no_tags")
        assert tags == []

    def test_get_images_with_tag(self, manager: TagManager) -> None:
        """Test getting all images with a specific tag."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        self._create_test_image(manager.db, "img2")
        self._create_test_image(manager.db, "img3")
        manager.add_image_tag("img1", "dogs", 0.95, "clip")
        manager.add_image_tag("img2", "dogs", 0.85, "clip")
        manager.add_image_tag("img3", "cats", 0.90, "clip")

        images = manager.get_images_with_tag("dogs")
        assert len(images) == 2
        assert "img1" in images
        assert "img2" in images
        assert "img3" not in images

    def test_get_images_with_tag_min_confidence(self, manager: TagManager) -> None:
        """Test filtering images by tag confidence."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        self._create_test_image(manager.db, "img2")
        manager.add_image_tag("img1", "dogs", 0.95, "clip")
        manager.add_image_tag("img2", "dogs", 0.70, "clip")

        images = manager.get_images_with_tag("dogs", min_confidence=0.80)
        assert len(images) == 1
        assert "img1" in images

    def test_get_images_with_tag_invalid(self, manager: TagManager) -> None:
        """Test getting images with non-existent tag."""
        manager.sync_taxonomy_to_db()

        images = manager.get_images_with_tag("nonexistent")
        assert images == []

    def test_get_images_with_any_tag(self, manager: TagManager) -> None:
        """Test OR query for multiple tags."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        self._create_test_image(manager.db, "img2")
        self._create_test_image(manager.db, "img3")
        manager.add_image_tag("img1", "dogs", 0.95, "clip")
        manager.add_image_tag("img2", "cats", 0.90, "clip")
        manager.add_image_tag("img3", "birds", 0.85, "clip")

        # Get images with dogs OR cats
        images = manager.get_images_with_any_tag(["dogs", "cats"])
        assert len(images) == 2
        assert "img1" in images
        assert "img2" in images
        assert "img3" not in images

    def test_get_images_with_any_tag_empty(self, manager: TagManager) -> None:
        """Test OR query with no valid tags."""
        manager.sync_taxonomy_to_db()

        images = manager.get_images_with_any_tag(["nonexistent1", "nonexistent2"])
        assert len(images) == 0

    def test_get_images_with_all_tags(self, manager: TagManager) -> None:
        """Test AND query for multiple tags."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        self._create_test_image(manager.db, "img2")
        self._create_test_image(manager.db, "img3")
        manager.add_image_tag("img1", "dogs", 0.95, "clip")
        manager.add_image_tag("img1", "outdoor", 0.85, "clip")
        manager.add_image_tag("img2", "dogs", 0.90, "clip")
        manager.add_image_tag("img3", "outdoor", 0.80, "clip")

        # Get images with dogs AND outdoor
        images = manager.get_images_with_all_tags(["dogs", "outdoor"])
        assert len(images) == 1
        assert "img1" in images

    def test_get_images_with_all_tags_none_match(self, manager: TagManager) -> None:
        """Test AND query with no matches."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        self._create_test_image(manager.db, "img2")
        manager.add_image_tag("img1", "dogs", 0.95, "clip")
        manager.add_image_tag("img2", "cats", 0.90, "clip")

        # No image has both tags
        images = manager.get_images_with_all_tags(["dogs", "cats"])
        assert len(images) == 0

    def test_get_tag_statistics(self, manager: TagManager) -> None:
        """Test getting tag statistics."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        self._create_test_image(manager.db, "img2")
        manager.add_image_tag("img1", "dogs", 0.95, "clip")
        manager.add_image_tag("img1", "outdoor", 0.85, "manual")
        manager.add_image_tag("img2", "cats", 0.90, "clip")

        stats = manager.get_tag_statistics()

        assert stats["total_tags"] >= 35
        assert stats["total_image_tags"] == 3
        assert stats["tagged_images"] == 2
        assert stats["tags_from_clip"] == 2
        assert stats["tags_from_manual"] == 1

        # Check category counts
        assert stats["subject_tags"] > 0
        assert stats["scene_tags"] > 0
        assert stats["lighting_tags"] > 0
        assert stats["mood_tags"] > 0

    def test_get_most_common_tags(self, manager: TagManager) -> None:
        """Test getting most common tags."""
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        self._create_test_image(manager.db, "img2")
        self._create_test_image(manager.db, "img3")
        self._create_test_image(manager.db, "img4")
        self._create_test_image(manager.db, "img5")
        manager.add_image_tag("img1", "dogs", 0.95, "clip")
        manager.add_image_tag("img2", "dogs", 0.90, "clip")
        manager.add_image_tag("img3", "dogs", 0.85, "clip")
        manager.add_image_tag("img4", "cats", 0.90, "clip")
        manager.add_image_tag("img5", "cats", 0.85, "clip")

        top_tags = manager.get_most_common_tags(limit=5)

        assert len(top_tags) >= 2
        assert top_tags[0] == ("dogs", 3)
        assert top_tags[1] == ("cats", 2)

    def test_get_most_common_tags_empty(self, manager: TagManager) -> None:
        """Test getting common tags with no tagged images."""
        manager.sync_taxonomy_to_db()

        top_tags = manager.get_most_common_tags()
        assert top_tags == []

    def test_get_tag_by_name(self, manager: TagManager) -> None:
        """Test getting tag definition by name."""
        tag = manager.get_tag_by_name("dogs")

        assert tag is not None
        assert tag.name == "dogs"
        assert tag.category == TagCategory.SUBJECT

    def test_get_tag_by_name_invalid(self, manager: TagManager) -> None:
        """Test getting non-existent tag."""
        tag = manager.get_tag_by_name("nonexistent")
        assert tag is None

    def test_search_tags(self, manager: TagManager) -> None:
        """Test searching tags."""
        # Search by name
        tags = manager.search_tags("dogs")
        assert len(tags) > 0
        assert any(t.name == "dogs" for t in tags)

        # Search by synonym
        tags = manager.search_tags("puppy")
        assert len(tags) > 0
        assert any(t.name == "dogs" for t in tags)

    def test_search_tags_no_results(self, manager: TagManager) -> None:
        """Test searching for non-existent tag."""
        tags = manager.search_tags("nonexistent_query")
        assert tags == []

    def test_repr(self, manager: TagManager) -> None:
        """Test string representation."""
        repr_str = repr(manager)
        assert "TagManager" in repr_str
        assert "tags" in repr_str

    def test_workflow_complete(self, manager: TagManager) -> None:
        """Test complete tagging workflow."""
        # Sync taxonomy
        manager.sync_taxonomy_to_db()

        self._create_test_image(manager.db, "img1")
        self._create_test_image(manager.db, "img2")
        # Add tags to multiple images
        manager.add_image_tag("img1", "dogs", 0.95, "clip")
        manager.add_image_tag("img1", "outdoor", 0.85, "clip")
        manager.add_image_tag("img2", "cats", 0.90, "clip")
        manager.add_image_tag("img2", "indoor", 0.80, "clip")

        # Query tags
        img1_tags = manager.get_image_tags("img1")
        assert len(img1_tags) == 2

        # Query images
        dog_images = manager.get_images_with_tag("dogs")
        assert "img1" in dog_images

        # Complex queries
        outdoor_images = manager.get_images_with_any_tag(["outdoor", "indoor"])
        assert len(outdoor_images) == 2

        # Statistics
        stats = manager.get_tag_statistics()
        assert stats["tagged_images"] == 2
        assert stats["total_image_tags"] == 4

        # Remove tag
        manager.remove_image_tag("img1", "outdoor")
        img1_tags = manager.get_image_tags("img1")
        assert len(img1_tags) == 1
