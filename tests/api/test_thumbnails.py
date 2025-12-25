"""Tests for thumbnail generation API endpoints.

All tests require database connection.
"""

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

pytestmark = pytest.mark.integration


@pytest.fixture
def sample_catalog_with_images(db_session, temp_dir):
    """Create a catalog with sample images for testing."""
    from sqlalchemy import text

    from vam_tools.db.models import Catalog

    # Create catalog
    catalog_id = uuid.uuid4()
    catalog = Catalog(
        id=catalog_id,
        name="Test Thumbnail Catalog",
        schema_name=f"deprecated_{catalog_id}",
        source_directories=[str(temp_dir)],
    )
    db_session.add(catalog)
    db_session.commit()

    # Create sample image file
    image_path = temp_dir / "sample.jpg"
    img = Image.new("RGB", (800, 600), color="blue")
    img.save(image_path, "JPEG")

    # Insert image record directly
    import json
    from datetime import datetime

    image_id = str(uuid.uuid4())
    db_session.execute(
        text(
            """
            INSERT INTO images (id, catalog_id, source_path, file_type, checksum, size_bytes, dates, metadata, created_at, updated_at, status_id)
            VALUES (:id, :catalog_id, :source_path, :file_type, :checksum, :size_bytes, CAST(:dates AS jsonb), CAST(:metadata AS jsonb), :created_at, :updated_at, :status_id)
        """
        ),
        {
            "id": image_id,
            "catalog_id": str(catalog_id),
            "source_path": str(image_path),
            "file_type": "image",
            "checksum": "abc123",
            "size_bytes": image_path.stat().st_size,
            "dates": json.dumps({}),
            "metadata": json.dumps({}),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "status_id": "active",
        },
    )
    db_session.commit()

    return catalog_id, image_id, image_path


def test_thumbnail_generation_creates_file(
    client, sample_catalog_with_images, tmp_path
):
    """Test that requesting a thumbnail creates a thumbnail file."""
    catalog_id, image_id, source_path = sample_catalog_with_images

    # Create thumbnails directory for this test
    thumbnails_dir = tmp_path / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    # Patch the catalog directory path
    with patch("vam_tools.api.routers.catalogs.Path") as mock_path_class:

        def path_factory(path_str):
            if (
                isinstance(path_str, str)
                and "/app/catalogs/" in path_str
                and "/thumbnails" in path_str
            ):
                return thumbnails_dir
            elif isinstance(path_str, str):
                return Path(path_str)
            return path_str

        mock_path_class.side_effect = path_factory

        response = client.get(f"/api/catalogs/{catalog_id}/images/{image_id}/thumbnail")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 0

        # Verify thumbnail file was created (now in size subdirectory)
        thumbnail_path = thumbnails_dir / "medium" / f"{image_id}.jpg"
        assert thumbnail_path.exists()


def test_thumbnail_uses_cache_on_second_request(
    client, sample_catalog_with_images, tmp_path
):
    """Test that subsequent requests use cached thumbnail."""
    catalog_id, image_id, source_path = sample_catalog_with_images

    # Create thumbnails directory for this test
    thumbnails_dir = tmp_path / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    with patch("vam_tools.api.routers.catalogs.Path") as mock_path_class:

        def path_factory(path_str):
            if (
                isinstance(path_str, str)
                and "/app/catalogs/" in path_str
                and "/thumbnails" in path_str
            ):
                return thumbnails_dir
            elif isinstance(path_str, str):
                return Path(path_str)
            return path_str

        mock_path_class.side_effect = path_factory

        # First request - generates thumbnail
        response1 = client.get(
            f"/api/catalogs/{catalog_id}/images/{image_id}/thumbnail"
        )
        assert response1.status_code == 200

        # Get thumbnail modification time (now in size subdirectory)
        thumbnail_path = thumbnails_dir / "medium" / f"{image_id}.jpg"
        first_mtime = thumbnail_path.stat().st_mtime

        # Second request - should use cached thumbnail
        response2 = client.get(
            f"/api/catalogs/{catalog_id}/images/{image_id}/thumbnail"
        )
        assert response2.status_code == 200

        # Thumbnail should not have been regenerated
        second_mtime = thumbnail_path.stat().st_mtime
        assert first_mtime == second_mtime


def test_thumbnail_respects_quality_parameter(
    client, sample_catalog_with_images, tmp_path
):
    """Test that quality parameter is passed to thumbnail generation."""
    catalog_id, image_id, source_path = sample_catalog_with_images

    # Create thumbnails directory for this test
    thumbnails_dir = tmp_path / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    with patch("vam_tools.api.routers.catalogs.Path") as mock_path_class:

        def path_factory(path_str):
            if (
                isinstance(path_str, str)
                and "/app/catalogs/" in path_str
                and "/thumbnails" in path_str
            ):
                return thumbnails_dir
            elif isinstance(path_str, str):
                return Path(path_str)
            return path_str

        mock_path_class.side_effect = path_factory

        # Request with quality parameter
        response = client.get(
            f"/api/catalogs/{catalog_id}/images/{image_id}/thumbnail?quality=50"
        )
        assert response.status_code == 200

        # Verify thumbnail was created (now in size subdirectory)
        thumbnail_path = thumbnails_dir / "medium" / f"{image_id}.jpg"
        assert thumbnail_path.exists()


def test_thumbnail_returns_404_for_nonexistent_image(
    client, sample_catalog_with_images
):
    """Test that requesting thumbnail for nonexistent image returns 404."""
    catalog_id, _, _ = sample_catalog_with_images
    fake_image_id = str(uuid.uuid4())

    response = client.get(
        f"/api/catalogs/{catalog_id}/images/{fake_image_id}/thumbnail"
    )
    assert response.status_code == 404


def test_thumbnail_returns_404_for_nonexistent_catalog(client):
    """Test that requesting thumbnail for nonexistent catalog returns 404."""
    fake_catalog_id = str(uuid.uuid4())
    fake_image_id = str(uuid.uuid4())

    response = client.get(
        f"/api/catalogs/{fake_catalog_id}/images/{fake_image_id}/thumbnail"
    )
    assert response.status_code == 404


def test_thumbnail_handles_missing_source_file(
    client, sample_catalog_with_images, db_session, tmp_path
):
    """Test that thumbnail generation handles missing source file gracefully."""
    catalog_id, image_id, _ = sample_catalog_with_images

    # Update image record to point to non-existent file
    from sqlalchemy import text

    db_session.execute(
        text("UPDATE images SET source_path = :path WHERE id = :id"),
        {"path": "/nonexistent/path.jpg", "id": image_id},
    )
    db_session.commit()

    response = client.get(f"/api/catalogs/{catalog_id}/images/{image_id}/thumbnail")
    assert response.status_code == 404
