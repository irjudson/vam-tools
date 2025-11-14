"""Tests for database layer."""

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from vam_tools.db.config import settings
from vam_tools.db.models import Base, Catalog


@pytest.fixture
def db_session():
    """Create a test database session."""
    # Use a test database
    test_db_url = settings.database_url.replace("vam-tools", "vam-tools-test")
    engine = create_engine(test_db_url)

    # Create tables
    Base.metadata.create_all(engine)

    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    # Cleanup
    session.close()
    Base.metadata.drop_all(engine)


def test_database_connection(db_session):
    """Test that we can connect to the database."""
    from sqlalchemy import text

    result = db_session.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_create_catalog(db_session):
    """Test creating a catalog."""
    catalog = Catalog(
        id=uuid.uuid4(),
        name="Test Catalog",
        schema_name="catalog_test",
        source_directories=["/tmp/test"],
    )

    db_session.add(catalog)
    db_session.commit()

    # Verify it was created
    loaded = db_session.query(Catalog).filter_by(name="Test Catalog").first()
    assert loaded is not None
    assert loaded.name == "Test Catalog"
    assert loaded.schema_name == "catalog_test"
    assert loaded.source_directories == ["/tmp/test"]


def test_list_catalogs(db_session):
    """Test listing catalogs."""
    # Create multiple catalogs
    for i in range(3):
        catalog = Catalog(
            id=uuid.uuid4(),
            name=f"Catalog {i}",
            schema_name=f"catalog_{i}",
            source_directories=[f"/tmp/test{i}"],
        )
        db_session.add(catalog)

    db_session.commit()

    # List them
    catalogs = db_session.query(Catalog).all()
    assert len(catalogs) == 3


def test_update_catalog(db_session):
    """Test updating a catalog."""
    catalog = Catalog(
        id=uuid.uuid4(),
        name="Original Name",
        schema_name="catalog_test",
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
    catalog = Catalog(
        id=uuid.uuid4(),
        name="To Delete",
        schema_name="catalog_delete",
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
