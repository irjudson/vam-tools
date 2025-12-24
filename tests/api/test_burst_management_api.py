"""Tests for burst management API endpoints.

Tests for listing bursts with filtering by rejection status.
"""

import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import text

from vam_tools.db.models import Catalog

pytestmark = pytest.mark.integration


def _create_test_image(db_session, catalog_id, burst_id=None, sequence=None, status="active"):
    """Helper to create a test image with required fields."""
    img_id = str(uuid.uuid4())
    db_session.execute(
        text(
            """
            INSERT INTO images (
                id, catalog_id, source_path, file_type, checksum,
                burst_id, burst_sequence, quality_score, status_id,
                dates, metadata, created_at
            ) VALUES (
                :id, :catalog_id, :path, :file_type, :checksum,
                :burst_id, :seq, 0.8, :status,
                '{"selected_date": "2024-01-01T12:00:00"}'::jsonb,
                '{}'::jsonb, NOW()
            )
            """
        ),
        {
            "id": img_id,
            "catalog_id": str(catalog_id),
            "path": f"/tmp/img_{img_id}.jpg",
            "file_type": "image",
            "checksum": img_id,  # Use id as checksum for simplicity
            "burst_id": burst_id,
            "seq": sequence,
            "status": status,
        }
    )
    return img_id


class TestBurstManagementAPI:
    """Tests for GET /api/catalogs/{catalog_id}/bursts endpoint with rejection filtering."""

    def test_list_bursts_excludes_fully_rejected_by_default(self, client, db_session):
        """Test that bursts where all images are rejected are excluded by default."""
        # Create a catalog
        catalog_id = uuid.uuid4()
        catalog = Catalog(
            id=catalog_id,
            name="Test Catalog",
            schema_name=f"catalog_{str(catalog_id).replace('-', '_')}",
            source_directories=["/tmp/test"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Create a burst with all rejected images
        burst_id_1 = str(uuid.uuid4())
        db_session.execute(
            text(
                """
                INSERT INTO bursts (id, catalog_id, image_count, start_time, end_time,
                                   duration_seconds, camera_make, camera_model,
                                   best_image_id, selection_method, created_at)
                VALUES (:id, :catalog_id, 3, :start_time, :end_time, 2.0,
                       'Canon', 'R5', :best_id, 'quality', NOW())
                """
            ),
            {
                "id": burst_id_1,
                "catalog_id": str(catalog_id),
                "start_time": datetime.now(),
                "end_time": datetime.now() + timedelta(seconds=2),
                "best_id": str(uuid.uuid4()),
            }
        )

        # Add 3 rejected images for burst 1
        for i in range(3):
            _create_test_image(db_session, catalog_id, burst_id_1, i, "rejected")

        # Create a burst with mixed status images (some active, some rejected)
        burst_id_2 = str(uuid.uuid4())
        db_session.execute(
            text(
                """
                INSERT INTO bursts (id, catalog_id, image_count, start_time, end_time,
                                   duration_seconds, camera_make, camera_model,
                                   best_image_id, selection_method, created_at)
                VALUES (:id, :catalog_id, 3, :start_time, :end_time, 2.0,
                       'Canon', 'R5', :best_id, 'quality', NOW())
                """
            ),
            {
                "id": burst_id_2,
                "catalog_id": str(catalog_id),
                "start_time": datetime.now(),
                "end_time": datetime.now() + timedelta(seconds=2),
                "best_id": str(uuid.uuid4()),
            }
        )

        # Add 2 active and 1 rejected image for burst 2
        for i in range(3):
            status = "rejected" if i == 2 else "active"
            _create_test_image(db_session, catalog_id, burst_id_2, i, status)

        db_session.commit()

        # Make the request with default settings (show_rejected=false)
        response = client.get(f"/api/catalogs/{catalog_id}/bursts")

        assert response.status_code == 200
        data = response.json()
        assert "bursts" in data

        # Should only return burst_id_2 (mixed status), not burst_id_1 (all rejected)
        assert len(data["bursts"]) == 1
        assert data["bursts"][0]["id"] == burst_id_2

    def test_list_bursts_includes_fully_rejected_when_requested(self, client, db_session):
        """Test that bursts where all images are rejected are included when show_rejected=true."""
        # Create a catalog
        catalog_id = uuid.uuid4()
        catalog = Catalog(
            id=catalog_id,
            name="Test Catalog 2",
            schema_name=f"catalog_{str(catalog_id).replace('-', '_')}",
            source_directories=["/tmp/test"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Create a burst with all rejected images
        burst_id_1 = str(uuid.uuid4())
        db_session.execute(
            text(
                """
                INSERT INTO bursts (id, catalog_id, image_count, start_time, end_time,
                                   duration_seconds, camera_make, camera_model,
                                   best_image_id, selection_method, created_at)
                VALUES (:id, :catalog_id, 3, :start_time, :end_time, 2.0,
                       'Canon', 'R5', :best_id, 'quality', NOW())
                """
            ),
            {
                "id": burst_id_1,
                "catalog_id": str(catalog_id),
                "start_time": datetime.now(),
                "end_time": datetime.now() + timedelta(seconds=2),
                "best_id": str(uuid.uuid4()),
            }
        )

        # Add 3 rejected images for burst 1
        for i in range(3):
            _create_test_image(db_session, catalog_id, burst_id_1, i, "rejected")

        db_session.commit()

        # Make the request with show_rejected=true
        response = client.get(
            f"/api/catalogs/{catalog_id}/bursts",
            params={"show_rejected": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert "bursts" in data

        # Should return burst_id_1 (all rejected)
        assert len(data["bursts"]) == 1
        assert data["bursts"][0]["id"] == burst_id_1

    def test_list_bursts_supports_sorting(self, client, db_session):
        """Test that bursts can be sorted by newest, oldest, or largest."""
        # Create a catalog
        catalog_id = uuid.uuid4()
        catalog = Catalog(
            id=catalog_id,
            name="Test Catalog 3",
            schema_name=f"catalog_{str(catalog_id).replace('-', '_')}",
            source_directories=["/tmp/test"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Create three bursts with different times and sizes
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        burst_ids = []

        for i in range(3):
            burst_id = str(uuid.uuid4())
            burst_ids.append(burst_id)
            image_count = 3 + i  # 3, 4, 5 images
            db_session.execute(
                text(
                    """
                    INSERT INTO bursts (id, catalog_id, image_count, start_time, end_time,
                                       duration_seconds, camera_make, camera_model,
                                       best_image_id, selection_method, created_at)
                    VALUES (:id, :catalog_id, :count, :start_time, :end_time, 2.0,
                           'Canon', 'R5', :best_id, 'quality', NOW())
                    """
                ),
                {
                    "id": burst_id,
                    "catalog_id": str(catalog_id),
                    "count": image_count,
                    "start_time": base_time + timedelta(days=i),
                    "end_time": base_time + timedelta(days=i, seconds=2),
                    "best_id": str(uuid.uuid4()),
                }
            )

            # Add active images to each burst
            for j in range(image_count):
                _create_test_image(db_session, catalog_id, burst_id, j, "active")

        db_session.commit()

        # Test sorting by newest (default)
        response = client.get(
            f"/api/catalogs/{catalog_id}/bursts",
            params={"sort": "newest"}
        )
        assert response.status_code == 200
        data = response.json()
        # Newest first (burst with latest start_time)
        assert data["bursts"][0]["id"] == burst_ids[2]

        # Test sorting by oldest
        response = client.get(
            f"/api/catalogs/{catalog_id}/bursts",
            params={"sort": "oldest"}
        )
        assert response.status_code == 200
        data = response.json()
        # Oldest first (burst with earliest start_time)
        assert data["bursts"][0]["id"] == burst_ids[0]

        # Test sorting by largest
        response = client.get(
            f"/api/catalogs/{catalog_id}/bursts",
            params={"sort": "largest"}
        )
        assert response.status_code == 200
        data = response.json()
        # Largest first (burst with most images)
        assert data["bursts"][0]["id"] == burst_ids[2]  # 5 images

    def test_list_bursts_supports_pagination(self, client, db_session):
        """Test that bursts endpoint supports limit and offset parameters."""
        # Create a catalog
        catalog_id = uuid.uuid4()
        catalog = Catalog(
            id=catalog_id,
            name="Test Catalog 4",
            schema_name=f"catalog_{str(catalog_id).replace('-', '_')}",
            source_directories=["/tmp/test"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Create 5 bursts
        for i in range(5):
            burst_id = str(uuid.uuid4())
            db_session.execute(
                text(
                    """
                    INSERT INTO bursts (id, catalog_id, image_count, start_time, end_time,
                                       duration_seconds, camera_make, camera_model,
                                       best_image_id, selection_method, created_at)
                    VALUES (:id, :catalog_id, 3, :start_time, :end_time, 2.0,
                           'Canon', 'R5', :best_id, 'quality', NOW())
                    """
                ),
                {
                    "id": burst_id,
                    "catalog_id": str(catalog_id),
                    "start_time": datetime.now(),
                    "end_time": datetime.now() + timedelta(seconds=2),
                    "best_id": str(uuid.uuid4()),
                }
            )

            # Add active images to each burst
            for j in range(3):
                _create_test_image(db_session, catalog_id, burst_id, j, "active")

        db_session.commit()

        # Test limit
        response = client.get(
            f"/api/catalogs/{catalog_id}/bursts",
            params={"limit": 2}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["bursts"]) == 2
        assert data["limit"] == 2

        # Test offset
        response = client.get(
            f"/api/catalogs/{catalog_id}/bursts",
            params={"limit": 2, "offset": 2}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["bursts"]) == 2
        assert data["offset"] == 2
