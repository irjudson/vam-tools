"""Tests for database layer.

All tests require database connection.
"""

import uuid

import pytest

from lumina.db.models import Catalog

pytestmark = pytest.mark.integration


def test_database_connection(db_session):
    """Test that we can connect to the database."""
    from sqlalchemy import text

    result = db_session.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_create_catalog(db_session):
    """Test creating a catalog."""
    catalog_id = uuid.uuid4()
    test_dir = "/tmp/test"
    catalog = Catalog(
        id=catalog_id,
        name="Test Catalog",
        schema_name=f"deprecated_{catalog_id}",
        source_directories=[test_dir],
    )

    db_session.add(catalog)
    db_session.commit()

    # Verify it was created
    loaded = db_session.query(Catalog).filter_by(name="Test Catalog").first()
    assert loaded is not None
    assert loaded.name == "Test Catalog"
    assert loaded.schema_name.startswith("deprecated_")
    # Just verify we have source directories, paths may be normalized
    assert len(loaded.source_directories) == 1
    assert loaded.source_directories[0].endswith("test")


def test_list_catalogs(db_session):
    """Test listing catalogs."""
    # Create multiple catalogs with unique names to filter later
    test_catalogs = []
    for i in range(3):
        catalog_id = uuid.uuid4()
        catalog = Catalog(
            id=catalog_id,
            name=f"Test List Catalog {i}",
            schema_name=f"deprecated_{catalog_id}",
            source_directories=[f"/tmp/test{i}"],
        )
        db_session.add(catalog)
        test_catalogs.append(catalog)

    db_session.commit()

    # List only the test catalogs we created
    created_ids = [c.id for c in test_catalogs]
    catalogs = db_session.query(Catalog).filter(Catalog.id.in_(created_ids)).all()
    assert len(catalogs) == 3


def test_update_catalog(db_session):
    """Test updating a catalog."""
    catalog_id = uuid.uuid4()
    catalog = Catalog(
        id=catalog_id,
        name="Original Name",
        schema_name=f"deprecated_{catalog_id}",
        source_directories=["/tmp/test"],
    )

    db_session.add(catalog)
    db_session.commit()

    # Update it
    catalog.name = "Updated Name"
    catalog.source_directories = ["/tmp/test", "/tmp/test2"]
    db_session.commit()

    # Verify update
    loaded = db_session.query(Catalog).filter_by(id=catalog.id).first()
    assert loaded.name == "Updated Name"
    assert len(loaded.source_directories) == 2


def test_delete_catalog(db_session):
    """Test deleting a catalog."""
    catalog_id = uuid.uuid4()
    catalog = Catalog(
        id=catalog_id,
        name="To Delete",
        schema_name=f"deprecated_{catalog_id}",
        source_directories=["/tmp/test"],
    )

    db_session.add(catalog)
    db_session.commit()

    catalog_id = catalog.id

    # Delete it
    db_session.delete(catalog)
    db_session.commit()

    # Verify deletion
    loaded = db_session.query(Catalog).filter_by(id=catalog_id).first()
    assert loaded is None
