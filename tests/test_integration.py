"""Integration tests for end-to-end workflows."""

import tempfile
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from vam_tools.api.app import create_app
from vam_tools.db import get_db
from vam_tools.db.catalog_schema import (
    create_schema,
    delete_catalog_data,
    get_image_count,
    schema_exists,
)
from vam_tools.db.config import settings
from vam_tools.db.models import Base
from vam_tools.tasks.scanner import scan_directory


@pytest.fixture
def test_db():
    """Create a test database."""
    test_db_url = settings.database_url.replace("vam-tools", "vam-tools-test")
    engine = create_engine(test_db_url)
    Base.metadata.create_all(engine)

    TestSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    yield override_get_db

    Base.metadata.drop_all(engine)


@pytest.fixture
def client(test_db):
    """Create a test client."""
    app = create_app()
    app.dependency_overrides[get_db] = test_db
    return TestClient(app)


def test_complete_workflow(client):
    """
    Test complete workflow: create catalog, list, get, delete.

    This proves Phase 1 infrastructure is working end-to-end.
    NOTE: Job submission requires live Redis/Celery, tested separately.
    """
    # Step 1: Create a catalog
    response = client.post(
        "/api/catalogs/",
        json={
            "name": "Integration Test Catalog",
            "source_directories": ["/tmp/test"],
        },
    )
    assert response.status_code == 201
    catalog = response.json()
    catalog_id = catalog["id"]

    # Verify catalog was created
    assert catalog["name"] == "Integration Test Catalog"
    assert "schema_name" in catalog

    # Step 2: List all catalogs
    response = client.get("/api/catalogs/")
    assert response.status_code == 200
    catalogs = response.json()

    # Our catalog should be in the list
    catalog_ids = [c["id"] for c in catalogs]
    assert catalog_id in catalog_ids

    # Step 3: Get specific catalog
    response = client.get(f"/api/catalogs/{catalog_id}")
    assert response.status_code == 200
    fetched_catalog = response.json()
    assert fetched_catalog["id"] == catalog_id
    assert fetched_catalog["name"] == "Integration Test Catalog"

    # Step 4: Delete catalog
    response = client.delete(f"/api/catalogs/{catalog_id}")
    assert response.status_code == 204

    # Verify deletion
    response = client.get(f"/api/catalogs/{catalog_id}")
    assert response.status_code == 404


def test_multiple_catalogs(client):
    """Test creating and managing multiple catalogs."""
    catalog_ids = []

    # Create 5 catalogs
    for i in range(5):
        response = client.post(
            "/api/catalogs/",
            json={
                "name": f"Catalog {i}",
                "source_directories": [f"/tmp/test{i}"],
            },
        )
        assert response.status_code == 201
        catalog_ids.append(response.json()["id"])

    # List all catalogs
    response = client.get("/api/catalogs/")
    assert response.status_code == 200
    catalogs = response.json()
    assert len(catalogs) == 5

    # Delete all catalogs
    for catalog_id in catalog_ids:
        response = client.delete(f"/api/catalogs/{catalog_id}")
        assert response.status_code == 204

    # Verify all deleted
    response = client.get("/api/catalogs/")
    assert response.status_code == 200
    catalogs = response.json()
    assert len(catalogs) == 0


def test_health_and_api_structure(client):
    """Test that all expected endpoints exist and respond."""
    # Health check
    response = client.get("/health")
    assert response.status_code == 200

    # Catalog endpoints
    response = client.get("/api/catalogs/")
    assert response.status_code == 200

    # Jobs endpoint (should fail without body but prove it exists)
    response = client.post("/api/jobs/scan", json={})
    assert response.status_code in [422, 500]  # Validation error or internal error

    # API docs (FastAPI auto-generates these)
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200


def test_catalog_with_schema_creation(client):
    """Test that creating a catalog ensures main schema exists."""
    # Create catalog
    response = client.post(
        "/api/catalogs/",
        json={
            "name": "Schema Test Catalog",
            "source_directories": ["/tmp/test"],
        },
    )
    assert response.status_code == 201
    catalog = response.json()
    catalog_id = catalog["id"]

    # Verify main schema exists
    assert schema_exists()

    # Verify catalog has data isolation (no images yet)
    assert get_image_count(catalog_id) == 0

    # Delete catalog
    response = client.delete(f"/api/catalogs/{catalog_id}")
    assert response.status_code == 204

    # Verify catalog data was deleted
    assert get_image_count(catalog_id) == 0


def test_scan_workflow_end_to_end():
    """Test complete scan workflow: create catalog, scan directory, verify data."""
    from vam_tools.db.connection import SessionLocal
    from vam_tools.db.models import Catalog

    catalog_id = uuid.uuid4()

    # Ensure schema exists
    if not schema_exists():
        create_schema()

    # Create catalog
    db = SessionLocal()
    try:
        catalog = Catalog(
            id=catalog_id,
            name="E2E Test Catalog",
            schema_name=f"deprecated_{catalog_id}",
            source_directories=["/tmp/test"],
        )
        db.add(catalog)
        db.commit()
    finally:
        db.close()

    # Create temporary directory with test images
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test images with different colors (ensure distinct checksums)
        for i in range(10):
            color = (i * 22 + 10, i * 23 + 15, i * 24 + 20)
            img = Image.new("RGB", (100 + i, 100 + i), color=color)
            img.save(tmpdir / f"test_{i}.jpg")

        # Create nested directory with more images
        nested = tmpdir / "nested"
        nested.mkdir()
        for i in range(5):
            color = (i * 44 + 100, i * 45 + 110, i * 46 + 120)
            img = Image.new("RGB", (120 + i, 120 + i), color=color)
            img.save(nested / f"nested_{i}.jpg")

        try:
            # Scan directory
            stats = scan_directory(tmpdir, str(catalog_id))

            # Verify stats
            assert stats["files_found"] == 15
            assert stats["files_added"] == 15
            assert stats["files_skipped"] == 0
            assert stats["exact_duplicates"] == 0

            # Verify images were added to database
            assert get_image_count(str(catalog_id)) == 15

            # Verify image data in database
            db = SessionLocal()
            try:
                result = db.execute(
                    text(
                        "SELECT COUNT(*), MIN(size_bytes), MAX(size_bytes) FROM images WHERE catalog_id = :catalog_id"
                    ),
                    {"catalog_id": str(catalog_id)},
                ).fetchone()

                assert result[0] == 15  # count
                assert result[1] > 0  # min size
                assert result[2] > 0  # max size

                # Check that checksums are unique
                result = db.execute(
                    text(
                        "SELECT COUNT(DISTINCT checksum) FROM images WHERE catalog_id = :catalog_id"
                    ),
                    {"catalog_id": str(catalog_id)},
                ).scalar()
                assert result == 15

            finally:
                db.close()

            # Re-scan same directory (should detect duplicates)
            stats2 = scan_directory(tmpdir, str(catalog_id))
            assert stats2["files_found"] == 15
            assert stats2["files_added"] == 0
            assert stats2["exact_duplicates"] == 15

            # Image count should still be 15
            assert get_image_count(str(catalog_id)) == 15

        finally:
            # Cleanup
            db = SessionLocal()
            try:
                delete_catalog_data(str(catalog_id))
                cat = db.query(Catalog).filter(Catalog.id == catalog_id).first()
                if cat:
                    db.delete(cat)
                    db.commit()
            except:
                pass
            finally:
                db.close()
