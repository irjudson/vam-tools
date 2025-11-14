"""
Tests for review queue API endpoints.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from vam_tools.core.database import CatalogDatabase
from vam_tools.web.api import app, init_catalog

client = TestClient(app)


@pytest.fixture
def populated_catalog(tmp_path: Path) -> Path:
    """Create a catalog with images and review queue items."""
    catalog_dir = tmp_path / "catalog"
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()

    with CatalogDatabase(catalog_dir) as db:
        db.initialize()

        # Insert images
        for i in range(5):
            img_path = photos_dir / f"photo{i}.jpg"
            img_path.touch()
            db.execute(
                """
                INSERT INTO images (
                    id, source_path, file_size, file_hash, format,
                    width, height, created_at, modified_at, indexed_at,
                    date_taken, quality_score, is_corrupted, thumbnail_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"img{i}",
                    str(img_path),
                    1000 + i,
                    f"hash{i}",
                    "JPEG",
                    100,
                    100,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    (
                        datetime(2023, 1, i + 1).isoformat() if i != 2 else None
                    ),  # img2 has no date
                    80,
                    0,
                    f"thumbnails/img{i}.jpg",
                ),
            )

        # Insert review queue items
        db.execute(
            """
            INSERT INTO review_queue (image_id, reason, priority, created_at)
            VALUES (?, ?, ?, ?)
            """,
            ("img2", "no_date", 1, datetime.now().isoformat()),
        )
        db.execute(
            """
            INSERT INTO review_queue (image_id, reason, priority, created_at)
            VALUES (?, ?, ?, ?)
            """,
            ("img0", "suspicious_date", 2, datetime.now().isoformat()),
        )
        db.execute(
            """
            INSERT INTO review_queue (image_id, reason, priority, created_at)
            VALUES (?, ?, ?, ?)
            """,
            ("img1", "date_conflict", 3, datetime.now().isoformat()),
        )

    return catalog_dir


class TestReviewQueueAPI:
    """Tests for review queue API endpoints."""

    def test_get_review_stats(self, populated_catalog: Path) -> None:
        """Test GET /api/review/stats endpoint."""
        init_catalog(populated_catalog)
        response = client.get("/api/review/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["total_needing_review"] == 3
        assert data["no_date"] == 1
        assert data["suspicious_dates"] == 1
        assert data["date_conflicts"] == 1
        assert data["low_confidence"] == 0  # Not implemented yet
        assert data["ready_to_organize"] == 0  # Not implemented yet

    def test_get_review_queue(self, populated_catalog: Path) -> None:
        """Test GET /api/review/queue endpoint."""
        init_catalog(populated_catalog)
        response = client.get("/api/review/queue")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 3
        assert len(data["items"]) == 3

        # Check structure of an item
        item = data["items"][0]
        assert "id" in item
        assert "type" in item
        assert "source_path" in item
        assert "current_date" in item or item["type"] == "no_date"
        assert "confidence" in item

    def test_get_review_queue_filter_by_type(self, populated_catalog: Path) -> None:
        """Test GET /api/review/queue with filter_type."""
        init_catalog(populated_catalog)
        response = client.get("/api/review/queue?filter_type=no_date")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["type"] == "no_date"
        assert data["items"][0]["id"] == "img2"

    def test_update_image_date(self, populated_catalog: Path) -> None:
        """Test PATCH /api/images/{image_id}/date endpoint."""
        init_catalog(populated_catalog)
        image_id = "img2"  # Image with no date

        new_date = "2024-03-01T10:00:00Z"
        response = client.patch(
            f"/api/images/{image_id}/date", params={"date_str": new_date}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["image_id"] == image_id
        assert data["new_date"] == new_date

        # Verify date updated in DB
        catalog = CatalogDatabase(populated_catalog)
        with catalog as db:
            row = db.execute(
                "SELECT date_taken FROM images WHERE id = ?", (image_id,)
            ).fetchone()
            assert row["date_taken"] == new_date

    def test_update_image_date_invalid_format(self, populated_catalog: Path) -> None:
        """Test PATCH /api/images/{image_id}/date with invalid date format."""
        init_catalog(populated_catalog)
        image_id = "img2"

        invalid_date = "not-a-date"
        response = client.patch(
            f"/api/images/{image_id}/date", params={"date_str": invalid_date}
        )

        assert response.status_code == 400
        assert "Invalid date format" in response.json()["detail"]

    def test_update_image_date_not_found(self, populated_catalog: Path) -> None:
        """Test PATCH /api/images/{image_id}/date for non-existent image."""
        init_catalog(populated_catalog)
        non_existent_id = "nonexistent-img"
        new_date = "2024-03-01T10:00:00Z"

        response = client.patch(
            f"/api/images/{non_existent_id}/date", params={"date_str": new_date}
        )

        assert response.status_code == 404
        assert "Image not found" in response.json()["detail"]
