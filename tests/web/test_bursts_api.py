"""Tests for burst API endpoints."""

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vam_tools.db import CatalogDB
from vam_tools.web.api import app

pytestmark = pytest.mark.integration


@pytest.fixture
def test_catalog_id():
    """Generate a unique catalog ID for testing."""
    return str(uuid.uuid4())


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def mock_catalog_db(db_session, test_catalog_id):
    """Create a mock catalog database for testing."""
    from sqlalchemy import text

    # First, create the catalog record in the catalogs table
    schema_name = f"catalog_{test_catalog_id.replace('-', '_')}"
    db_session.execute(
        text(
            """
            INSERT INTO catalogs (id, name, schema_name, source_directories, created_at, updated_at)
            VALUES (:id, :name, :schema_name, :source_dirs, NOW(), NOW())
            ON CONFLICT (id) DO NOTHING
        """
        ),
        {
            "id": test_catalog_id,
            "name": "Test Catalog",
            "schema_name": schema_name,
            "source_dirs": ["/test/path"],
        },
    )
    db_session.commit()

    with patch("vam_tools.web.api.get_catalog_db") as mock_get_db:
        # Create a real CatalogDB instance with the test session
        # CatalogDB accepts catalog_id_or_path as first argument
        catalog = CatalogDB(test_catalog_id, session=db_session)
        mock_get_db.return_value = catalog
        yield catalog


class TestListBurstsEndpoint:
    """Tests for GET /api/catalogs/{catalog_id}/bursts endpoint."""

    def test_list_bursts_returns_empty_when_no_bursts(
        self, client, test_catalog_id, mock_catalog_db
    ):
        """Test listing bursts returns empty list when no bursts exist."""
        response = client.get(f"/api/catalogs/{test_catalog_id}/bursts")

        assert response.status_code == 200
        data = response.json()
        assert "bursts" in data
        assert data["bursts"] == []
        assert data["total"] == 0
        assert data["limit"] == 100
        assert data["offset"] == 0

    def test_list_bursts_returns_bursts(
        self, client, test_catalog_id, mock_catalog_db, db_session
    ):
        """Test listing bursts returns burst data."""
        from sqlalchemy import text

        # Insert test burst
        burst_id = str(uuid.uuid4())
        db_session.execute(
            text(
                """
                INSERT INTO bursts (
                    id, catalog_id, image_count, start_time, end_time,
                    duration_seconds, camera_make, camera_model,
                    best_image_id, selection_method, created_at
                ) VALUES (
                    :id, :catalog_id, :image_count, :start_time, :end_time,
                    :duration, :camera_make, :camera_model,
                    :best_image_id, :selection_method, NOW()
                )
            """
            ),
            {
                "id": burst_id,
                "catalog_id": test_catalog_id,
                "image_count": 5,
                "start_time": datetime(2024, 1, 1, 12, 0, 0),
                "end_time": datetime(2024, 1, 1, 12, 0, 2),
                "duration": 2.0,
                "camera_make": "Canon",
                "camera_model": "R5",
                "best_image_id": str(uuid.uuid4()),
                "selection_method": "quality",
            },
        )
        db_session.commit()

        response = client.get(f"/api/catalogs/{test_catalog_id}/bursts")

        assert response.status_code == 200
        data = response.json()
        assert "bursts" in data
        assert len(data["bursts"]) == 1
        assert data["total"] == 1

        burst = data["bursts"][0]
        assert burst["id"] == burst_id
        assert burst["image_count"] == 5
        assert burst["duration_seconds"] == 2.0
        assert burst["camera_make"] == "Canon"
        assert burst["camera_model"] == "R5"
        assert burst["selection_method"] == "quality"

    def test_list_bursts_with_pagination(
        self, client, test_catalog_id, mock_catalog_db, db_session
    ):
        """Test listing bursts with pagination parameters."""
        from sqlalchemy import text

        # Insert multiple bursts
        for i in range(5):
            db_session.execute(
                text(
                    """
                    INSERT INTO bursts (
                        id, catalog_id, image_count, start_time, end_time,
                        duration_seconds, camera_make, camera_model,
                        best_image_id, selection_method, created_at
                    ) VALUES (
                        :id, :catalog_id, :image_count, :start_time, :end_time,
                        :duration, :camera_make, :camera_model,
                        :best_image_id, :selection_method, NOW()
                    )
                """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "catalog_id": test_catalog_id,
                    "image_count": 3,
                    "start_time": datetime(2024, 1, 1, 12, i, 0),
                    "end_time": datetime(2024, 1, 1, 12, i, 1),
                    "duration": 1.0,
                    "camera_make": "Canon",
                    "camera_model": "R5",
                    "best_image_id": str(uuid.uuid4()),
                    "selection_method": "quality",
                },
            )
        db_session.commit()

        # Test with limit
        response = client.get(f"/api/catalogs/{test_catalog_id}/bursts?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["bursts"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 2

        # Test with offset
        response = client.get(
            f"/api/catalogs/{test_catalog_id}/bursts?limit=2&offset=2"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["bursts"]) == 2
        assert data["offset"] == 2


class TestGetBurstEndpoint:
    """Tests for GET /api/catalogs/{catalog_id}/bursts/{burst_id} endpoint."""

    def test_get_burst_returns_404_when_not_found(
        self, client, test_catalog_id, mock_catalog_db
    ):
        """Test getting a non-existent burst returns 404."""
        burst_id = str(uuid.uuid4())
        response = client.get(f"/api/catalogs/{test_catalog_id}/bursts/{burst_id}")

        assert response.status_code == 404

    def test_get_burst_returns_burst_details(
        self, client, test_catalog_id, mock_catalog_db, db_session
    ):
        """Test getting burst details with images."""
        from sqlalchemy import text

        # Insert test burst
        burst_id = str(uuid.uuid4())
        best_image_id = str(uuid.uuid4())

        db_session.execute(
            text(
                """
                INSERT INTO bursts (
                    id, catalog_id, image_count, start_time, end_time,
                    duration_seconds, camera_make, camera_model,
                    best_image_id, selection_method, created_at
                ) VALUES (
                    :id, :catalog_id, :image_count, :start_time, :end_time,
                    :duration, :camera_make, :camera_model,
                    :best_image_id, :selection_method, NOW()
                )
            """
            ),
            {
                "id": burst_id,
                "catalog_id": test_catalog_id,
                "image_count": 3,
                "start_time": datetime(2024, 1, 1, 12, 0, 0),
                "end_time": datetime(2024, 1, 1, 12, 0, 2),
                "duration": 2.0,
                "camera_make": "Canon",
                "camera_model": "R5",
                "best_image_id": best_image_id,
                "selection_method": "quality",
            },
        )

        # Insert test images in burst
        for i in range(3):
            image_id = str(uuid.uuid4())
            db_session.execute(
                text(
                    """
                    INSERT INTO images (
                        id, catalog_id, source_path, file_type, checksum,
                        burst_id, burst_sequence, quality_score, dates, metadata,
                        created_at
                    ) VALUES (
                        :id, :catalog_id, :source_path, :file_type, :checksum,
                        :burst_id, :sequence, :quality_score, :dates, :metadata,
                        NOW()
                    )
                """
                ),
                {
                    "id": image_id if i != 1 else best_image_id,
                    "catalog_id": test_catalog_id,
                    "source_path": f"/path/to/image_{i}.jpg",
                    "file_type": "image",
                    "checksum": f"checksum_{i}",
                    "burst_id": burst_id,
                    "sequence": i,
                    "quality_score": 0.8 + (i * 0.05),
                    "dates": "{}",
                    "metadata": "{}",
                },
            )
        db_session.commit()

        response = client.get(f"/api/catalogs/{test_catalog_id}/bursts/{burst_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == burst_id
        assert data["image_count"] == 3
        assert data["camera_make"] == "Canon"
        assert data["camera_model"] == "R5"
        assert data["best_image_id"] == best_image_id
        assert "images" in data
        assert len(data["images"]) == 3

        # Check images are sorted by sequence
        for i, img in enumerate(data["images"]):
            assert img["sequence"] == i
            if i == 1:
                assert img["is_best"] is True
            else:
                assert img["is_best"] is False


class TestUpdateBurstEndpoint:
    """Tests for PUT /api/catalogs/{catalog_id}/bursts/{burst_id} endpoint."""

    def test_update_burst_best_image(
        self, client, test_catalog_id, mock_catalog_db, db_session
    ):
        """Test updating the best image for a burst."""
        from sqlalchemy import text

        # Insert test burst
        burst_id = str(uuid.uuid4())
        old_best_id = str(uuid.uuid4())
        new_best_id = str(uuid.uuid4())

        db_session.execute(
            text(
                """
                INSERT INTO bursts (
                    id, catalog_id, image_count, start_time, end_time,
                    duration_seconds, camera_make, camera_model,
                    best_image_id, selection_method, created_at
                ) VALUES (
                    :id, :catalog_id, :image_count, :start_time, :end_time,
                    :duration, :camera_make, :camera_model,
                    :best_image_id, :selection_method, NOW()
                )
            """
            ),
            {
                "id": burst_id,
                "catalog_id": test_catalog_id,
                "image_count": 2,
                "start_time": datetime(2024, 1, 1, 12, 0, 0),
                "end_time": datetime(2024, 1, 1, 12, 0, 1),
                "duration": 1.0,
                "camera_make": "Canon",
                "camera_model": "R5",
                "best_image_id": old_best_id,
                "selection_method": "quality",
            },
        )
        db_session.commit()

        # Update best image
        response = client.put(
            f"/api/catalogs/{test_catalog_id}/bursts/{burst_id}",
            json={"best_image_id": new_best_id},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"

        # Verify the update
        result = db_session.execute(
            text("SELECT best_image_id, selection_method FROM bursts WHERE id = :id"),
            {"id": burst_id},
        )
        row = result.fetchone()
        assert row[0] == new_best_id
        assert row[1] == "manual"


class TestDetectBurstsEndpoint:
    """Tests for POST /api/catalogs/{catalog_id}/detect-bursts endpoint."""

    def test_start_burst_detection_job(self, client, test_catalog_id, mock_catalog_db):
        """Test starting a burst detection job."""
        with patch("vam_tools.web.api.detect_bursts_task") as mock_task:
            mock_task.delay.return_value.id = "job-123"

            response = client.post(f"/api/catalogs/{test_catalog_id}/detect-bursts")

            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["job_id"] == "job-123"
            assert data["status"] == "queued"
            assert "message" in data

            # Verify task was called with correct parameters
            mock_task.delay.assert_called_once()
            call_kwargs = mock_task.delay.call_args.kwargs
            assert call_kwargs["catalog_id"] == test_catalog_id
            assert call_kwargs["gap_threshold"] == 2.0
            assert call_kwargs["min_burst_size"] == 3

    def test_start_burst_detection_with_custom_params(
        self, client, test_catalog_id, mock_catalog_db
    ):
        """Test starting burst detection with custom parameters."""
        with patch("vam_tools.web.api.detect_bursts_task") as mock_task:
            mock_task.delay.return_value.id = "job-456"

            response = client.post(
                f"/api/catalogs/{test_catalog_id}/detect-bursts?gap_threshold=1.5&min_burst_size=5"
            )

            assert response.status_code == 202

            # Verify task was called with custom parameters
            call_kwargs = mock_task.delay.call_args.kwargs
            assert call_kwargs["gap_threshold"] == 1.5
            assert call_kwargs["min_burst_size"] == 5
