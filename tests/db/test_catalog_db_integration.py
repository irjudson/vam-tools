"""Tests for CatalogDB integration with serializers.

All tests require database connection.
"""

from datetime import datetime
from pathlib import Path

import pytest

from lumina.core.types import (
    DateInfo,
    FileType,
    ImageMetadata,
    ImageRecord,
    ImageStatus,
)
from lumina.db import CatalogDB

pytestmark = pytest.mark.integration


def test_add_and_retrieve_image_record(tmp_path):
    """Test that ImageRecord can be added and retrieved with full fidelity."""
    import uuid

    unique_id = f"test_img_{uuid.uuid4().hex[:8]}"

    # Create a complete ImageRecord
    date_info = DateInfo(
        selected_date=datetime(2023, 6, 15, 14, 30, 22),
        selected_source="exif",
        confidence=90,
        suspicious=False,
    )

    metadata = ImageMetadata(
        exif={"Make": "Canon", "Model": "EOS R5"},
        camera_make="Canon",
        camera_model="EOS R5",
        size_bytes=1024000,
        resolution=(1920, 1080),
    )

    record = ImageRecord(
        id=unique_id,
        source_path=Path("/test/photos/image.jpg"),
        file_type=FileType.IMAGE,
        checksum=f"abc123def456_{unique_id}",
        status=ImageStatus.PENDING,
        dates=date_info,
        metadata=metadata,
    )

    # Add to database
    with CatalogDB(tmp_path) as db:
        db.add_image(record)

        # Retrieve by ID
        retrieved = db.get_image(unique_id)

        # Should return ImageRecord, not dict
        assert isinstance(retrieved, ImageRecord)
        assert retrieved.id == unique_id
        assert retrieved.file_type == FileType.IMAGE
        assert retrieved.status == ImageStatus.PENDING

        # Nested objects should be deserialized
        assert isinstance(retrieved.dates, DateInfo)
        assert retrieved.dates.selected_date == datetime(2023, 6, 15, 14, 30, 22)
        assert retrieved.dates.confidence == 90

        assert isinstance(retrieved.metadata, ImageMetadata)
        assert retrieved.metadata.camera_make == "Canon"
        assert retrieved.metadata.camera_model == "EOS R5"
        assert retrieved.metadata.size_bytes == 1024000
        assert retrieved.metadata.resolution == (1920, 1080)


def test_get_all_images_returns_image_records(tmp_path):
    """Test that get_all_images returns dict of ImageRecord objects."""
    import uuid

    unique_id1 = f"all_test_{uuid.uuid4().hex[:8]}"
    unique_id2 = f"all_test_{uuid.uuid4().hex[:8]}"

    record1 = ImageRecord(
        id=unique_id1,
        source_path=Path("/test/img1.jpg"),
        file_type=FileType.IMAGE,
        checksum=f"hash1_{unique_id1}",
        status=ImageStatus.PENDING,
    )

    record2 = ImageRecord(
        id=unique_id2,
        source_path=Path("/test/img2.png"),
        file_type=FileType.IMAGE,
        checksum=f"hash2_{unique_id2}",
        status=ImageStatus.COMPLETE,
    )

    with CatalogDB(tmp_path) as db:
        db.add_image(record1)
        db.add_image(record2)

        all_images = db.get_all_images()

        # Should return dict mapping id -> ImageRecord
        assert isinstance(all_images, dict)
        assert len(all_images) >= 2  # May have other images from other tests

        assert isinstance(all_images[unique_id1], ImageRecord)
        assert all_images[unique_id1].file_type == FileType.IMAGE

        assert isinstance(all_images[unique_id2], ImageRecord)
        assert all_images[unique_id2].file_type == FileType.IMAGE
        assert all_images[unique_id2].status == ImageStatus.COMPLETE


def test_list_images_returns_ids(tmp_path):
    """Test that list_images returns list of image IDs."""
    import uuid

    unique_id1 = f"list_test_{uuid.uuid4().hex[:8]}"
    unique_id2 = f"list_test_{uuid.uuid4().hex[:8]}"

    record1 = ImageRecord(
        id=unique_id1,
        source_path=Path("/test/img1.jpg"),
        file_type=FileType.IMAGE,
        checksum=f"h1_{unique_id1}",
        status=ImageStatus.PENDING,
    )
    record2 = ImageRecord(
        id=unique_id2,
        source_path=Path("/test/img2.jpg"),
        file_type=FileType.IMAGE,
        checksum=f"h2_{unique_id2}",
        status=ImageStatus.PENDING,
    )

    with CatalogDB(tmp_path) as db:
        db.add_image(record1)
        db.add_image(record2)

        ids = db.list_images()

        assert isinstance(ids, list)
        assert len(ids) >= 2  # May have other images from other tests
        assert unique_id1 in [r.id for r in ids]
        assert unique_id2 in [r.id for r in ids]
