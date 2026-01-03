"""Integration tests for duplicate detection with real-world scenarios."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from lumina.jobs.parallel_duplicates import (
    _build_duplicate_groups,
    _create_duplicate_group_tags,
)


def test_prevents_mega_group_from_hash_chain():
    """
    Test that hash chains don't create mega-groups.

    Simulates the real bug: many images with similar hashes that form a chain
    where each pair is similar but ends are not similar.
    """
    # Simulate real hash collision scenario from investigation:
    # cc4cccceced062cf (118 images) --[dist 3]--> cc4cccceced0704f (92 images)
    # cc4cccceced0704f --[dist 4]--> cc4ccececed062cf (54 images)
    # cc4ccececed062cf --[dist 2]--> cc4cccceced060cf (43 images)

    # Create pairs representing this chain (simplified)
    pairs = [
        # Group 1 internally similar (simulate 118 images with same hash)
        {"image_1": "A1", "image_2": "A2", "distance": 0},
        {"image_1": "A1", "image_2": "A3", "distance": 0},
        {"image_1": "A2", "image_2": "A3", "distance": 0},
        # Group 2 internally similar (simulate 92 images)
        {"image_1": "B1", "image_2": "B2", "distance": 0},
        {"image_1": "B1", "image_2": "B3", "distance": 0},
        {"image_1": "B2", "image_2": "B3", "distance": 0},
        # Group 3 internally similar (simulate 54 images)
        {"image_1": "C1", "image_2": "C2", "distance": 0},
        # Cross-group connections (these create the chain)
        {"image_1": "A1", "image_2": "B1", "distance": 3},  # Within threshold
        {"image_1": "B1", "image_2": "C1", "distance": 4},  # Within threshold
        # Note: A1 and C1 are NOT similar (distance would be > 5)
    ]

    groups = _build_duplicate_groups(pairs)

    # Old behavior: 1 mega-group with all images
    # New behavior: Should create separate groups or at most merge A's with B's
    #               but NOT create one group with all A, B, and C

    # Verify we don't have a mega-group
    max_group_size = max(len(g) for g in groups) if groups else 0
    assert max_group_size <= 5, f"Created mega-group with {max_group_size} images"

    # Verify A's and C's are not in same group (no transitive closure)
    group_sets = [set(g) for g in groups]
    for group_set in group_sets:
        # Should not have both A and C members in same group
        has_a = any(img.startswith("A") for img in group_set)
        has_c = any(img.startswith("C") for img in group_set)
        assert not (
            has_a and has_c
        ), "A and C images in same group (transitive closure)"


def test_exact_duplicates_still_group():
    """Test that exact duplicates (distance 0) still group correctly."""
    pairs = [
        {"image_1": "A", "image_2": "B", "distance": 0},
        {"image_1": "B", "image_2": "C", "distance": 0},
        {"image_1": "A", "image_2": "C", "distance": 0},
    ]

    groups = _build_duplicate_groups(pairs)

    # Exact duplicates should all be in one group
    assert len(groups) == 1
    assert set(groups[0]) == {"A", "B", "C"}


@pytest.mark.integration
def test_auto_create_tags_for_duplicate_groups(db_session):
    """Test that tags are automatically created for duplicate groups with dhash metadata."""
    import uuid

    # Create a test catalog
    catalog_id = str(uuid.uuid4())
    db_session.execute(
        text(
            """
            INSERT INTO catalogs (id, name, schema_name, source_directories, created_at, updated_at)
            VALUES (:id, :name, :schema_name, :source_directories, NOW(), NOW())
        """
        ),
        {
            "id": catalog_id,
            "name": "Test Catalog",
            "schema_name": "test_schema",
            "source_directories": ["/test/path"],
        },
    )

    # Create test images with dhash values
    test_images = [
        {
            "id": "img1",
            "dhash": "cc4cccceced062cf",
            "catalog_id": catalog_id,
            "source_path": "/test/img1.jpg",
            "file_type": "image",
            "checksum": "checksum1",
        },
        {
            "id": "img2",
            "dhash": "cc4cccceced062cf",
            "catalog_id": catalog_id,
            "source_path": "/test/img2.jpg",
            "file_type": "image",
            "checksum": "checksum2",
        },
        {
            "id": "img3",
            "dhash": "aa1bbbbbeced062cf",
            "catalog_id": catalog_id,
            "source_path": "/test/img3.jpg",
            "file_type": "image",
            "checksum": "checksum3",
        },
    ]

    for img in test_images:
        db_session.execute(
            text(
                """
                INSERT INTO images (id, catalog_id, source_path, file_type, dhash, checksum, dates, metadata, created_at)
                VALUES (:id, :catalog_id, :source_path, :file_type, :dhash, :checksum, '{}', '{}', NOW())
            """
            ),
            img,
        )

    # Create duplicate groups
    groups_data = [
        {
            "primary_id": "img1",
            "members": ["img1", "img2"],
            "dhash": "cc4cccceced062cf",
        },
        {
            "primary_id": "img3",
            "members": ["img3"],
            "dhash": "aa1bbbbbeced062cf",
        },
    ]

    db_session.commit()

    # Call the function to create tags
    from lumina.db import CatalogDB

    # Mock CatalogDB to use our test session
    with patch("lumina.jobs.parallel_duplicates.CatalogDatabase") as mock_db_class:
        mock_db = MagicMock()
        mock_db.session = db_session
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)
        mock_db_class.return_value = mock_db

        # Call the tag creation function
        _create_duplicate_group_tags(catalog_id, groups_data)

    # Verify tags were created with correct format
    result = db_session.execute(
        text("SELECT name, description FROM tags WHERE catalog_id = :catalog_id"),
        {"catalog_id": catalog_id},
    )
    tags = result.fetchall()

    # Should have 2 tags (one per group)
    assert len(tags) == 2

    # Verify tag names follow the format: dup-{first 8 chars of dhash}
    tag_names = {tag[0] for tag in tags}
    assert "dup-cc4cccce" in tag_names
    assert "dup-aa1bbbbb" in tag_names

    # Verify descriptions contain the full dhash
    tag_dict = {tag[0]: tag[1] for tag in tags}
    assert "cc4cccceced062cf" in tag_dict["dup-cc4cccce"]
    assert "aa1bbbbbeced062cf" in tag_dict["dup-aa1bbbbb"]
