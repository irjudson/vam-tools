"""Pytest configuration for API tests."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(db_session):
    """Create a test client for the FastAPI application."""
    from lumina.api.app import app
    from lumina.db import get_db

    # Override the get_db dependency to use our test database session
    def override_get_db():
        try:
            yield db_session
        finally:
            pass  # db_session cleanup is handled by the fixture

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    # Cleanup
    app.dependency_overrides.clear()
