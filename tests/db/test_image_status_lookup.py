"""Tests for ImageStatus lookup table model."""

import pytest

from vam_tools.db.models import Image, ImageStatus


class TestImageStatusLookup:
    """Tests for ImageStatus lookup table."""

    def test_image_status_lookup_has_static_rows(self, db_session):
        """Test that ImageStatus lookup table has the 4 required status rows."""
        # Query all status rows
        statuses = db_session.query(ImageStatus).all()

        # Should have exactly 4 statuses
        assert len(statuses) >= 4

        # Get status IDs
        status_ids = {s.id for s in statuses}

        # Verify the required statuses exist
        assert "active" in status_ids
        assert "rejected" in status_ids
        assert "archived" in status_ids
        assert "flagged" in status_ids

    def test_image_has_status_relationship(self, db_session):
        """Test that Image has a status relationship to ImageStatus."""
        # Create a test catalog first
        import uuid

        from vam_tools.db.models import Catalog

        catalog = Catalog(
            id=uuid.uuid4(),
            name="Test Catalog",
            schema_name="test_schema",
            source_directories=["/test/path"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Create a test image with explicit status_id
        image = Image(
            id="test-image-001",
            catalog_id=catalog.id,
            source_path="/test/image.jpg",
            file_type="image",
            checksum="abc123",
            size_bytes=1024,
            status_id="active",
        )
        db_session.add(image)
        db_session.commit()

        # Verify the relationship works
        db_session.refresh(image)
        assert hasattr(image, "status")
        assert image.status is not None
        assert image.status.id == "active"
        assert isinstance(image.status, ImageStatus)

    def test_image_default_status_is_active(self, db_session):
        """Test that Image defaults to 'active' status."""
        # Create a test catalog
        import uuid

        from vam_tools.db.models import Catalog

        catalog = Catalog(
            id=uuid.uuid4(),
            name="Test Catalog",
            schema_name="test_schema",
            source_directories=["/test/path"],
        )
        db_session.add(catalog)
        db_session.commit()

        # Create image without specifying status_id (should default to 'active')
        image = Image(
            id="test-image-002",
            catalog_id=catalog.id,
            source_path="/test/image2.jpg",
            file_type="image",
            checksum="def456",
            size_bytes=2048,
        )
        db_session.add(image)
        db_session.commit()

        # Verify default status
        db_session.refresh(image)
        assert image.status_id == "active"
        assert image.status.id == "active"
