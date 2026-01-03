"""Tests for semantic search API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lumina.analysis.semantic_search import SearchResult
from lumina.web.api import app

pytestmark = pytest.mark.integration


class TestSearchAPI:
    """Tests for /api/catalogs/{id}/search endpoint."""

    def test_search_endpoint_returns_results(self):
        """Test that search endpoint returns results."""
        client = TestClient(app)

        # Mock the SemanticSearchService
        with patch("lumina.web.api.get_search_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.search.return_value = [
                SearchResult(
                    image_id="img-001",
                    source_path="/photos/sunset.jpg",
                    similarity_score=0.85,
                )
            ]
            mock_get_service.return_value = mock_service

            # Mock get_catalog_db to avoid database connection
            with patch("lumina.web.api.get_catalog_db") as mock_get_db:
                mock_db = MagicMock()
                mock_db.session = MagicMock()
                mock_get_db.return_value = mock_db

                response = client.get(
                    "/api/catalogs/test-catalog/search", params={"q": "sunset"}
                )

                assert response.status_code == 200
                data = response.json()
                assert "results" in data
                assert "query" in data
                assert "count" in data
                assert data["query"] == "sunset"
                assert len(data["results"]) == 1
                assert data["count"] == 1
                assert data["results"][0]["similarity_score"] == 0.85
                assert data["results"][0]["image_id"] == "img-001"
                assert "thumbnail_url" in data["results"][0]

    def test_search_requires_query_parameter(self):
        """Test that search requires q parameter."""
        client = TestClient(app)

        response = client.get("/api/catalogs/test-catalog/search")
        assert response.status_code == 422  # Validation error

    def test_search_with_custom_limit_and_threshold(self):
        """Test that search accepts limit and threshold parameters."""
        client = TestClient(app)

        with patch("lumina.web.api.get_search_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.search.return_value = []
            mock_get_service.return_value = mock_service

            with patch("lumina.web.api.get_catalog_db") as mock_get_db:
                mock_db = MagicMock()
                mock_db.session = MagicMock()
                mock_get_db.return_value = mock_db

                response = client.get(
                    "/api/catalogs/test-catalog/search",
                    params={"q": "sunset", "limit": 100, "threshold": 0.3},
                )

                assert response.status_code == 200
                # Verify the service was called with correct parameters
                mock_service.search.assert_called_once()
                call_args = mock_service.search.call_args
                assert call_args.kwargs["query"] == "sunset"
                assert call_args.kwargs["limit"] == 100
                assert call_args.kwargs["threshold"] == 0.3

    def test_similar_endpoint_returns_results(self):
        """Test that similar endpoint returns results."""
        client = TestClient(app)

        with patch("lumina.web.api.get_search_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.find_similar.return_value = [
                SearchResult(
                    image_id="img-002",
                    source_path="/photos/similar.jpg",
                    similarity_score=0.92,
                )
            ]
            mock_get_service.return_value = mock_service

            with patch("lumina.web.api.get_catalog_db") as mock_get_db:
                mock_db = MagicMock()
                mock_db.session = MagicMock()
                mock_get_db.return_value = mock_db

                response = client.get("/api/catalogs/test-catalog/similar/img-001")

                assert response.status_code == 200
                data = response.json()
                assert "results" in data
                assert "source_image_id" in data
                assert "count" in data
                assert data["source_image_id"] == "img-001"
                assert len(data["results"]) == 1
                assert data["count"] == 1
                assert data["results"][0]["similarity_score"] == 0.92
                assert data["results"][0]["image_id"] == "img-002"

    def test_similar_with_custom_limit_and_threshold(self):
        """Test that similar endpoint accepts limit and threshold parameters."""
        client = TestClient(app)

        with patch("lumina.web.api.get_search_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.find_similar.return_value = []
            mock_get_service.return_value = mock_service

            with patch("lumina.web.api.get_catalog_db") as mock_get_db:
                mock_db = MagicMock()
                mock_db.session = MagicMock()
                mock_get_db.return_value = mock_db

                response = client.get(
                    "/api/catalogs/test-catalog/similar/img-001",
                    params={"limit": 30, "threshold": 0.6},
                )

                assert response.status_code == 200
                # Verify the service was called with correct parameters
                mock_service.find_similar.assert_called_once()
                call_args = mock_service.find_similar.call_args
                assert call_args.kwargs["image_id"] == "img-001"
                assert call_args.kwargs["limit"] == 30
                assert call_args.kwargs["threshold"] == 0.6

    def test_search_returns_empty_list_when_no_results(self):
        """Test that search returns empty list when no results found."""
        client = TestClient(app)

        with patch("lumina.web.api.get_search_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.search.return_value = []
            mock_get_service.return_value = mock_service

            with patch("lumina.web.api.get_catalog_db") as mock_get_db:
                mock_db = MagicMock()
                mock_db.session = MagicMock()
                mock_get_db.return_value = mock_db

                response = client.get(
                    "/api/catalogs/test-catalog/search", params={"q": "nonexistent"}
                )

                assert response.status_code == 200
                data = response.json()
                assert data["results"] == []
                assert data["count"] == 0

    def test_similar_returns_empty_list_when_no_results(self):
        """Test that similar returns empty list when no similar images found."""
        client = TestClient(app)

        with patch("lumina.web.api.get_search_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.find_similar.return_value = []
            mock_get_service.return_value = mock_service

            with patch("lumina.web.api.get_catalog_db") as mock_get_db:
                mock_db = MagicMock()
                mock_db.session = MagicMock()
                mock_get_db.return_value = mock_db

                response = client.get("/api/catalogs/test-catalog/similar/img-001")

                assert response.status_code == 200
                data = response.json()
                assert data["results"] == []
                assert data["count"] == 0
