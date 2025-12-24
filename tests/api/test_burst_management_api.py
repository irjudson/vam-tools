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


class TestGetBurstDetailEndpoint:
    """Tests for GET /api/catalogs/{catalog_id}/bursts/{burst_id} endpoint."""

    def test_get_burst_detail_returns_burst_with_images_sorted_by_quality(self, client, db_session):
        """Test that burst detail endpoint returns burst metadata and images sorted by quality_score DESC."""
        # Create a catalog
        catalog_id = uuid.uuid4()
        catalog = Catalog(
            id=catalog_id,
            name="Test Catalog Detail",
            schema_name=f"catalog_{str(catalog_id).replace('-', '_')}",
            source_directories=["/tmp/test"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Create a burst
        burst_id = str(uuid.uuid4())
        best_image_id = str(uuid.uuid4())
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 0, 5)

        db_session.execute(
            text(
                """
                INSERT INTO bursts (id, catalog_id, image_count, start_time, end_time,
                                   duration_seconds, camera_make, camera_model,
                                   best_image_id, selection_method, created_at)
                VALUES (:id, :catalog_id, 3, :start_time, :end_time, 5.0,
                       'Canon', 'EOS R5', :best_id, 'quality', NOW())
                """
            ),
            {
                "id": burst_id,
                "catalog_id": str(catalog_id),
                "start_time": start_time,
                "end_time": end_time,
                "best_id": best_image_id,
            }
        )

        # Create 3 images with different quality scores (integer 0-100)
        image_ids = []
        quality_scores = [60, 90, 70]  # Will be sorted as: 90, 70, 60

        for i, quality_score in enumerate(quality_scores):
            img_id = best_image_id if quality_score == 90 else str(uuid.uuid4())
            image_ids.append(img_id)
            db_session.execute(
                text(
                    """
                    INSERT INTO images (
                        id, catalog_id, source_path, file_type, checksum,
                        burst_id, burst_sequence, quality_score, status_id,
                        dates, metadata, created_at
                    ) VALUES (
                        :id, :catalog_id, :path, :file_type, :checksum,
                        :burst_id, :seq, :quality, 'active',
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
                    "checksum": img_id,
                    "burst_id": burst_id,
                    "seq": i,
                    "quality": quality_score,
                }
            )

        db_session.commit()

        # Make the request
        response = client.get(f"/api/catalogs/{catalog_id}/bursts/{burst_id}")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Check burst metadata
        assert data["id"] == burst_id
        assert data["catalog_id"] == str(catalog_id)
        assert data["image_count"] == 3
        assert data["start_time"] == start_time.isoformat()
        assert data["end_time"] == end_time.isoformat()
        assert data["duration_seconds"] == 5.0
        assert data["camera_make"] == "Canon"
        assert data["camera_model"] == "EOS R5"
        assert data["best_image_id"] == best_image_id

        # Check images are sorted by quality_score DESC
        assert len(data["images"]) == 3
        assert data["images"][0]["quality_score"] == 90
        assert data["images"][1]["quality_score"] == 70
        assert data["images"][2]["quality_score"] == 60

        # Check best image is marked correctly
        assert data["images"][0]["is_best"] is True
        assert data["images"][1]["is_best"] is False
        assert data["images"][2]["is_best"] is False

    def test_get_burst_detail_handles_null_quality_scores(self, client, db_session):
        """Test that images with NULL quality_score are sorted last by sequence."""
        # Create a catalog
        catalog_id = uuid.uuid4()
        catalog = Catalog(
            id=catalog_id,
            name="Test Catalog Null Quality",
            schema_name=f"catalog_{str(catalog_id).replace('-', '_')}",
            source_directories=["/tmp/test"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Create a burst
        burst_id = str(uuid.uuid4())
        best_image_id = str(uuid.uuid4())
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 0, 5)

        db_session.execute(
            text(
                """
                INSERT INTO bursts (id, catalog_id, image_count, start_time, end_time,
                                   duration_seconds, camera_make, camera_model,
                                   best_image_id, selection_method, created_at)
                VALUES (:id, :catalog_id, 3, :start_time, :end_time, 5.0,
                       'Canon', 'EOS R5', :best_id, 'quality', NOW())
                """
            ),
            {
                "id": burst_id,
                "catalog_id": str(catalog_id),
                "start_time": start_time,
                "end_time": end_time,
                "best_id": best_image_id,
            }
        )

        # Create 3 images: quality=90, quality=NULL (seq=1), quality=70 (seq=2)
        # Expected order: [90 (seq=0), 70 (seq=2), NULL (seq=1)]
        test_data = [
            {"quality": 90, "seq": 0, "is_best": True, "id": best_image_id},
            {"quality": None, "seq": 1, "is_best": False, "id": str(uuid.uuid4())},
            {"quality": 70, "seq": 2, "is_best": False, "id": str(uuid.uuid4())},
        ]

        for data_item in test_data:
            db_session.execute(
                text(
                    """
                    INSERT INTO images (
                        id, catalog_id, source_path, file_type, checksum,
                        burst_id, burst_sequence, quality_score, status_id,
                        dates, metadata, created_at
                    ) VALUES (
                        :id, :catalog_id, :path, :file_type, :checksum,
                        :burst_id, :seq, :quality, 'active',
                        '{"selected_date": "2024-01-01T12:00:00"}'::jsonb,
                        '{}'::jsonb, NOW()
                    )
                    """
                ),
                {
                    "id": data_item["id"],
                    "catalog_id": str(catalog_id),
                    "path": f"/tmp/img_{data_item['id']}.jpg",
                    "file_type": "image",
                    "checksum": data_item["id"],
                    "burst_id": burst_id,
                    "seq": data_item["seq"],
                    "quality": data_item["quality"],
                }
            )

        db_session.commit()

        # Make the request
        response = client.get(f"/api/catalogs/{catalog_id}/bursts/{burst_id}")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Check images are sorted: quality DESC NULLS LAST, then burst_sequence ASC
        # Expected order: 90 (seq=0), 70 (seq=2), NULL (seq=1)
        assert len(data["images"]) == 3
        assert data["images"][0]["quality_score"] == 90
        assert data["images"][0]["sequence"] == 0
        assert data["images"][0]["is_best"] is True

        assert data["images"][1]["quality_score"] == 70
        assert data["images"][1]["sequence"] == 2
        assert data["images"][1]["is_best"] is False

        assert data["images"][2]["quality_score"] is None
        assert data["images"][2]["sequence"] == 1
        assert data["images"][2]["is_best"] is False

    def test_get_burst_detail_returns_404_when_burst_not_found(self, client, db_session):
        """Test that endpoint returns 404 when burst does not exist."""
        # Create a catalog
        catalog_id = uuid.uuid4()
        catalog = Catalog(
            id=catalog_id,
            name="Test Catalog Not Found",
            schema_name=f"catalog_{str(catalog_id).replace('-', '_')}",
            source_directories=["/tmp/test"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Request a non-existent burst
        non_existent_burst_id = str(uuid.uuid4())
        response = client.get(f"/api/catalogs/{catalog_id}/bursts/{non_existent_burst_id}")

        # Verify 404 response
        assert response.status_code == 404
        assert "Burst not found" in response.json()["detail"]


class TestApplySelectionEndpoint:
    """Tests for POST /api/catalogs/{catalog_id}/bursts/{burst_id}/apply-selection endpoint."""

    def test_apply_selection_sets_selected_active_others_rejected(self, client, db_session):
        """Test that applying selection sets selected image to active and others to rejected."""
        # Create a catalog
        catalog_id = uuid.uuid4()
        catalog = Catalog(
            id=catalog_id,
            name="Test Catalog Apply Selection",
            schema_name=f"catalog_{str(catalog_id).replace('-', '_')}",
            source_directories=["/tmp/test"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Create a burst with 4 images all active
        burst_id = str(uuid.uuid4())
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 0, 3)

        db_session.execute(
            text(
                """
                INSERT INTO bursts (id, catalog_id, image_count, start_time, end_time,
                                   duration_seconds, camera_make, camera_model,
                                   best_image_id, selection_method, created_at)
                VALUES (:id, :catalog_id, 4, :start_time, :end_time, 3.0,
                       'Canon', 'EOS R5', :best_id, 'quality', NOW())
                """
            ),
            {
                "id": burst_id,
                "catalog_id": str(catalog_id),
                "start_time": start_time,
                "end_time": end_time,
                "best_id": str(uuid.uuid4()),
            }
        )

        # Create 4 active images in the burst
        image_ids = []
        for i in range(4):
            img_id = _create_test_image(db_session, catalog_id, burst_id, i, "active")
            image_ids.append(img_id)

        db_session.commit()

        # Select the second image (index 1)
        selected_image_id = image_ids[1]

        # Make the request
        response = client.post(
            f"/api/catalogs/{catalog_id}/bursts/{burst_id}/apply-selection",
            json={"selected_image_id": selected_image_id}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["selected_image_id"] == selected_image_id
        assert data["rejected_count"] == 3

        # Verify database state - selected image is active
        result = db_session.execute(
            text("SELECT status_id FROM images WHERE id = :image_id"),
            {"image_id": selected_image_id}
        ).fetchone()
        assert result[0] == "active"

        # Verify other images are rejected
        for img_id in image_ids:
            if img_id != selected_image_id:
                result = db_session.execute(
                    text("SELECT status_id FROM images WHERE id = :image_id"),
                    {"image_id": img_id}
                ).fetchone()
                assert result[0] == "rejected"

    def test_apply_selection_returns_404_when_burst_not_found(self, client, db_session):
        """Test that endpoint returns 404 when burst does not exist."""
        # Create a catalog
        catalog_id = uuid.uuid4()
        catalog = Catalog(
            id=catalog_id,
            name="Test Catalog Apply Not Found",
            schema_name=f"catalog_{str(catalog_id).replace('-', '_')}",
            source_directories=["/tmp/test"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Try to apply selection to non-existent burst
        non_existent_burst_id = str(uuid.uuid4())
        selected_image_id = str(uuid.uuid4())

        response = client.post(
            f"/api/catalogs/{catalog_id}/bursts/{non_existent_burst_id}/apply-selection",
            json={"selected_image_id": selected_image_id}
        )

        # Verify 404 response
        assert response.status_code == 404
        assert "Burst not found" in response.json()["detail"]

    def test_apply_selection_returns_400_when_image_not_in_burst(self, client, db_session):
        """Test that endpoint returns 400 when selected_image_id is not in the burst."""
        # Create a catalog
        catalog_id = uuid.uuid4()
        catalog = Catalog(
            id=catalog_id,
            name="Test Catalog Apply Bad Image",
            schema_name=f"catalog_{str(catalog_id).replace('-', '_')}",
            source_directories=["/tmp/test"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Create a burst with 3 images
        burst_id = str(uuid.uuid4())
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 0, 2)

        db_session.execute(
            text(
                """
                INSERT INTO bursts (id, catalog_id, image_count, start_time, end_time,
                                   duration_seconds, camera_make, camera_model,
                                   best_image_id, selection_method, created_at)
                VALUES (:id, :catalog_id, 3, :start_time, :end_time, 2.0,
                       'Canon', 'EOS R5', :best_id, 'quality', NOW())
                """
            ),
            {
                "id": burst_id,
                "catalog_id": str(catalog_id),
                "start_time": start_time,
                "end_time": end_time,
                "best_id": str(uuid.uuid4()),
            }
        )

        # Create 3 images in the burst
        for i in range(3):
            _create_test_image(db_session, catalog_id, burst_id, i, "active")

        db_session.commit()

        # Try to select an image that's not in the burst
        non_member_image_id = str(uuid.uuid4())

        response = client.post(
            f"/api/catalogs/{catalog_id}/bursts/{burst_id}/apply-selection",
            json={"selected_image_id": non_member_image_id}
        )

        # Verify 400 response
        assert response.status_code == 400
        assert "not in burst" in response.json()["detail"].lower()
