"""Tests for database schema management.

All tests require database connection.
"""

import json
import uuid

import pytest
from sqlalchemy import text

from lumina.db.catalog_schema import (
    create_schema,
    delete_catalog_data,
    get_catalog_statistics,
    get_image_count,
    schema_exists,
)
from lumina.db.connection import SessionLocal
from lumina.db.models import Catalog

pytestmark = pytest.mark.integration


@pytest.fixture
def test_catalog_id():
    """Generate a test catalog ID and create catalog record."""
    # Ensure schema exists
    if not schema_exists():
        create_schema()

    catalog_id = uuid.uuid4()

    # Create catalog record in database
    db = SessionLocal()
    try:
        catalog = Catalog(
            id=catalog_id,
            name="Test Catalog",
            schema_name=f"deprecated_{catalog_id}",  # Unique to satisfy constraint
            source_directories=["/test"],
        )
        db.add(catalog)
        db.commit()
    finally:
        db.close()

    yield str(catalog_id)

    # Cleanup after test
    db = SessionLocal()
    try:
        # Delete catalog data
        delete_catalog_data(str(catalog_id))

        # Delete catalog record
        catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
        if catalog:
            db.delete(catalog)
            db.commit()
    except:
        pass
    finally:
        db.close()


def test_create_schema():
    """Test creating the main schema."""
    # Schema should exist (created by fixture)
    assert schema_exists()


def test_schema_has_all_tables():
    """Test that schema creation creates all required tables."""
    db = SessionLocal()
    try:
        # Check that all tables exist
        tables = [
            "images",
            "tags",
            "image_tags",
            "duplicate_groups",
            "duplicate_members",
            "jobs",
            "config",
        ]

        for table in tables:
            result = db.execute(
                text(
                    """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = :table
                )
            """
                ),
                {"table": table},
            ).scalar()
            assert result, f"Table {table} not found in schema"
    finally:
        db.close()


def test_get_image_count_empty(test_catalog_id):
    """Test getting image count from empty catalog."""
    count = get_image_count(test_catalog_id)
    assert count == 0


def test_get_image_count_with_images(test_catalog_id):
    """Test getting image count after inserting images."""
    # Insert some test images
    db = SessionLocal()
    try:
        for i in range(5):
            db.execute(
                text(
                    """
                INSERT INTO images
                (id, catalog_id, source_path, file_type, checksum, dates, metadata, created_at, updated_at)
                VALUES (:id, :catalog_id, :path, :type, :checksum, :dates, :metadata, NOW(), NOW())
            """
                ),
                {
                    "id": f"test_image_{i}",
                    "catalog_id": test_catalog_id,
                    "path": f"/test/image{i}.jpg",
                    "type": "image",
                    "checksum": f"checksum_{i}",
                    "dates": json.dumps({}),
                    "metadata": json.dumps({}),
                },
            )
        db.commit()
    finally:
        db.close()

    count = get_image_count(test_catalog_id)
    assert count == 5


def test_images_table_structure(test_catalog_id):
    """Test that images table has correct columns."""
    db = SessionLocal()
    try:
        # Insert a test image with all fields
        db.execute(
            text(
                """
            INSERT INTO images
            (id, catalog_id, source_path, file_type, checksum, size_bytes, dates, metadata,
             dhash, ahash, quality_score, status_id, created_at, updated_at)
            VALUES (
                'test_id',
                :catalog_id,
                '/test/path.jpg',
                'image',
                'abc123',
                1024,
                '{"selected_date": "2023-01-01"}'::jsonb,
                '{"width": 100}'::jsonb,
                'dhash123',
                'ahash123',
                85,
                'active',
                NOW(),
                NOW()
            )
        """
            ),
            {"catalog_id": test_catalog_id},
        )
        db.commit()

        # Verify we can read it back
        result = db.execute(
            text("SELECT * FROM images WHERE id = :id"),
            {"id": "test_id"},
        ).fetchone()

        assert result is not None
        assert result[0] == "test_id"  # id
        assert str(result[1]) == test_catalog_id  # catalog_id (UUID object)
        assert result[2] == "/test/path.jpg"  # source_path
        assert result[3] == "image"  # file_type
    finally:
        db.close()


def test_delete_catalog_data(test_catalog_id):
    """Test deleting all data for a catalog."""
    # Insert some test data
    db = SessionLocal()
    unique_prefix = str(uuid.uuid4())[:8]
    try:
        # Insert images
        for i in range(3):
            db.execute(
                text(
                    """
                INSERT INTO images
                (id, catalog_id, source_path, file_type, checksum, dates, metadata, created_at, updated_at)
                VALUES (:id, :catalog_id, :path, :type, :checksum, :dates, :metadata, NOW(), NOW())
            """
                ),
                {
                    "id": f"img_{unique_prefix}_{i}",
                    "catalog_id": test_catalog_id,
                    "path": f"/test/img{i}.jpg",
                    "type": "image",
                    "checksum": f"check_{i}",
                    "dates": json.dumps({}),
                    "metadata": json.dumps({}),
                },
            )

        # Insert tags
        db.execute(
            text(
                """
            INSERT INTO tags (catalog_id, name, created_at)
            VALUES (:catalog_id, 'test_tag', NOW())
        """
            ),
            {"catalog_id": test_catalog_id},
        )

        db.commit()

        # Verify data exists
        assert get_image_count(test_catalog_id) == 3

    finally:
        db.close()

    # Delete catalog data
    delete_catalog_data(test_catalog_id)

    # Verify data is gone
    assert get_image_count(test_catalog_id) == 0


def test_catalog_statistics(test_catalog_id):
    """Test getting catalog statistics."""
    # Insert test data
    db = SessionLocal()
    unique_prefix = str(uuid.uuid4())[:8]
    try:
        # Insert images with different sizes
        for i in range(5):
            db.execute(
                text(
                    """
                INSERT INTO images
                (id, catalog_id, source_path, file_type, checksum, size_bytes, dates, metadata, created_at, updated_at)
                VALUES (:id, :catalog_id, :path, :type, :checksum, :size, :dates, :metadata, NOW(), NOW())
            """
                ),
                {
                    "id": f"img_{unique_prefix}_{i}",
                    "catalog_id": test_catalog_id,
                    "path": f"/test/img{i}.jpg",
                    "type": "image",
                    "checksum": f"check_{i}",
                    "dates": json.dumps({}),
                    "metadata": json.dumps({}),
                    "size": 1000 * (i + 1),  # Different sizes
                },
            )

        # Insert tags
        for i in range(3):
            db.execute(
                text(
                    """
                INSERT INTO tags (catalog_id, name, created_at)
                VALUES (:catalog_id, :name, NOW())
            """
                ),
                {"catalog_id": test_catalog_id, "name": f"tag_{i}"},
            )

        db.commit()
    finally:
        db.close()

    # Get statistics
    stats = get_catalog_statistics(test_catalog_id)

    assert stats["images"] == 5
    assert stats["tags"] == 3
    assert stats["total_size_bytes"] == 15000  # 1000+2000+3000+4000+5000


def test_multiple_catalogs_isolated():
    """Test that multiple catalogs are properly isolated."""
    catalog1_id = uuid.uuid4()
    catalog2_id = uuid.uuid4()

    db = SessionLocal()
    try:
        # Create catalog records
        catalog1 = Catalog(
            id=catalog1_id,
            name="Catalog 1",
            schema_name=f"deprecated_{catalog1_id}",  # Unique to satisfy constraint
            source_directories=["/cat1"],
        )
        catalog2 = Catalog(
            id=catalog2_id,
            name="Catalog 2",
            schema_name=f"deprecated_{catalog2_id}",  # Unique to satisfy constraint
            source_directories=["/cat2"],
        )
        db.add(catalog1)
        db.add(catalog2)
        db.commit()

        # Add images to catalog 1
        for i in range(3):
            db.execute(
                text(
                    """
                INSERT INTO images
                (id, catalog_id, source_path, file_type, checksum, dates, metadata, created_at, updated_at)
                VALUES (:id, :catalog_id, :path, :type, :checksum, :dates, :metadata, NOW(), NOW())
            """
                ),
                {
                    "id": f"cat1_img_{i}",
                    "catalog_id": str(catalog1_id),
                    "path": f"/cat1/img{i}.jpg",
                    "type": "image",
                    "checksum": f"cat1_check_{i}",
                    "dates": json.dumps({}),
                    "metadata": json.dumps({}),
                },
            )

        # Add images to catalog 2
        for i in range(2):
            db.execute(
                text(
                    """
                INSERT INTO images
                (id, catalog_id, source_path, file_type, checksum, dates, metadata, created_at, updated_at)
                VALUES (:id, :catalog_id, :path, :type, :checksum, :dates, :metadata, NOW(), NOW())
            """
                ),
                {
                    "id": f"cat2_img_{i}",
                    "catalog_id": str(catalog2_id),
                    "path": f"/cat2/img{i}.jpg",
                    "type": "image",
                    "checksum": f"cat2_check_{i}",
                    "dates": json.dumps({}),
                    "metadata": json.dumps({}),
                },
            )

        db.commit()

        # Verify counts
        assert get_image_count(str(catalog1_id)) == 3
        assert get_image_count(str(catalog2_id)) == 2

        # Delete catalog 1 data
        delete_catalog_data(str(catalog1_id))

        # Verify catalog 1 is empty but catalog 2 is unchanged
        assert get_image_count(str(catalog1_id)) == 0
        assert get_image_count(str(catalog2_id)) == 2

    finally:
        # Cleanup
        try:
            delete_catalog_data(str(catalog1_id))
            delete_catalog_data(str(catalog2_id))

            # Delete catalog records
            cat1 = db.query(Catalog).filter(Catalog.id == catalog1_id).first()
            cat2 = db.query(Catalog).filter(Catalog.id == catalog2_id).first()
            if cat1:
                db.delete(cat1)
            if cat2:
                db.delete(cat2)
            db.commit()
        except:
            pass
        db.close()
