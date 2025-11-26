"""Tests for FastAPI endpoints."""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from vam_tools.api.app import create_app
from vam_tools.db import get_db
from vam_tools.db.catalog_schema import schema_exists
from vam_tools.db.config import settings
from vam_tools.db.models import Base


@pytest.fixture
def test_db():
    """Create a test database."""
    # Use the already-patched settings and engine from conftest.py
    # conftest.py ensures settings.database_url points to test database
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)

    TestSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    yield override_get_db

    # Clean up catalogs created during tests
    from sqlalchemy import text

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM catalogs"))
        conn.commit()

    # Dispose the engine to close all connections
    engine.dispose()


@pytest.fixture
def client(test_db):
    """Create a test client."""
    app = create_app()
    app.dependency_overrides[get_db] = test_db
    with TestClient(app) as client:
        yield client


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_create_catalog(client):
    """Test creating a catalog via API."""
    response = client.post(
        "/api/catalogs/",
        json={
            "name": "Test Catalog",
            "source_directories": ["/tmp/test"],
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Catalog"
    assert data["source_directories"] == ["/tmp/test"]
    assert "id" in data
    assert "schema_name" in data

    # Verify main schema exists (single schema for all catalogs)
    assert schema_exists()


def test_list_catalogs(client):
    """Test listing catalogs."""
    # Create some catalogs
    for i in range(3):
        client.post(
            "/api/catalogs/",
            json={
                "name": f"Catalog {i}",
                "source_directories": [f"/tmp/test{i}"],
            },
        )

    # List them
    response = client.get("/api/catalogs/")
    assert response.status_code == 200
    catalogs = response.json()
    assert len(catalogs) == 3


def test_get_catalog(client):
    """Test getting a specific catalog."""
    # Create a catalog
    create_response = client.post(
        "/api/catalogs/",
        json={
            "name": "Get Test",
            "source_directories": ["/tmp/test"],
        },
    )
    catalog_id = create_response.json()["id"]

    # Get it
    response = client.get(f"/api/catalogs/{catalog_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == catalog_id
    assert data["name"] == "Get Test"


def test_get_nonexistent_catalog(client):
    """Test getting a catalog that doesn't exist."""
    fake_id = str(uuid.uuid4())
    response = client.get(f"/api/catalogs/{fake_id}")
    assert response.status_code == 404


def test_delete_catalog(client):
    """Test deleting a catalog."""
    # Create a catalog
    create_response = client.post(
        "/api/catalogs/",
        json={
            "name": "Delete Test",
            "source_directories": ["/tmp/test"],
        },
    )
    catalog_id = create_response.json()["id"]

    # Verify main schema exists
    assert schema_exists()

    # Delete it
    response = client.delete(f"/api/catalogs/{catalog_id}")
    assert response.status_code == 204

    # Verify it's gone
    response = client.get(f"/api/catalogs/{catalog_id}")
    assert response.status_code == 404

    # Note: Main schema persists (it's shared by all catalogs)
    # Only the catalog data is deleted
    assert schema_exists()


def test_create_catalog_validation(client):
    """Test catalog creation validation."""
    # Missing required fields
    response = client.post("/api/catalogs/", json={})
    assert response.status_code == 422

    # Empty name
    response = client.post(
        "/api/catalogs/",
        json={
            "name": "",
            "source_directories": ["/tmp/test"],
        },
    )
    assert response.status_code == 422

    # Empty source directories
    response = client.post(
        "/api/catalogs/",
        json={
            "name": "Test",
            "source_directories": [],
        },
    )
    assert response.status_code == 422
