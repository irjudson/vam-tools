"""Tests for ImageStatus model."""

import pytest
from datetime import datetime
from vam_tools.db.models import Image, ImageStatus


class TestImageStatusModel:
    """Tests for ImageStatus model."""

    def test_image_status_creation(self, db_session):
        """Test that ImageStatus can be created with required fields."""
        # Create a test catalog first
        from vam_tools.db.models import Catalog
        import uuid

        catalog = Catalog(
            id=uuid.uuid4(),
            name="Test Catalog",
            schema_name="test_schema",
            source_directories=["/test/path"]
        )
        db_session.add(catalog)
        db_session.commit()

        # Create a test image
        image = Image(
            id="test-image-001",
            catalog_id=catalog.id,
            source_path="/test/image.jpg",
            file_type="image",
            checksum="abc123",
            size_bytes=1024
        )
        db_session.add(image)
        db_session.commit()

        # Create an ImageStatus
        status = ImageStatus(
            image_id=image.id,
            status="analyzing",
            created_at=datetime.utcnow()
        )
        db_session.add(status)
        db_session.commit()

        # Verify status was created
        assert status.id is not None
        assert status.image_id == "test-image-001"
        assert status.status == "analyzing"

    def test_image_status_relationship(self, db_session):
        """Test that Image has a relationship to ImageStatus."""
        # Create a test catalog
        from vam_tools.db.models import Catalog
        import uuid

        catalog = Catalog(
            id=uuid.uuid4(),
            name="Test Catalog",
            schema_name="test_schema",
            source_directories=["/test/path"]
        )
        db_session.add(catalog)
        db_session.commit()

        # Create a test image
        image = Image(
            id="test-image-002",
            catalog_id=catalog.id,
            source_path="/test/image2.jpg",
            file_type="image",
            checksum="def456",
            size_bytes=2048
        )
        db_session.add(image)
        db_session.commit()

        # Create multiple status records
        status1 = ImageStatus(
            image_id=image.id,
            status="pending",
            created_at=datetime.utcnow()
        )
        status2 = ImageStatus(
            image_id=image.id,
            status="analyzing",
            created_at=datetime.utcnow()
        )
        db_session.add_all([status1, status2])
        db_session.commit()

        # Verify relationship
        db_session.refresh(image)
        assert hasattr(image, 'status_history')
        assert len(image.status_history) == 2
        assert status1 in image.status_history
        assert status2 in image.status_history

    def test_image_status_cascade_delete(self, db_session):
        """Test that ImageStatus records are deleted when Image is deleted."""
        # Create a test catalog
        from vam_tools.db.models import Catalog
        import uuid

        catalog = Catalog(
            id=uuid.uuid4(),
            name="Test Catalog",
            schema_name="test_schema",
            source_directories=["/test/path"]
        )
        db_session.add(catalog)
        db_session.commit()

        # Create a test image
        image = Image(
            id="test-image-003",
            catalog_id=catalog.id,
            source_path="/test/image3.jpg",
            file_type="image",
            checksum="ghi789",
            size_bytes=3072
        )
        db_session.add(image)
        db_session.commit()

        # Create status records
        status = ImageStatus(
            image_id=image.id,
            status="complete",
            created_at=datetime.utcnow()
        )
        db_session.add(status)
        db_session.commit()

        status_id = status.id

        # Delete the image
        db_session.delete(image)
        db_session.commit()

        # Verify status was cascade deleted
        deleted_status = db_session.query(ImageStatus).filter_by(id=status_id).first()
        assert deleted_status is None
