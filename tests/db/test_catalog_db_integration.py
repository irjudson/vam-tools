"""Tests for CatalogDB integration with serializers."""

from datetime import datetime
from pathlib import Path

import pytest

from vam_tools.core.types import (
    DateInfo,
    FileType,
    ImageMetadata,
    ImageRecord,
    ImageStatus,
)
from vam_tools.db import CatalogDB


def test_add_and_retrieve_image_record(tmp_path):
    """Test that ImageRecord can be added and retrieved with full fidelity."""
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
        id="test_img_001",
        source_path=Path("/test/photos/image.jpg"),
        file_type=FileType.IMAGE,
        checksum="abc123def456",
        status=ImageStatus.PENDING,
        dates=date_info,
        metadata=metadata,
    )

    # Add to database
    with CatalogDB(tmp_path) as db:
        db.add_image(record)

        # Retrieve by ID
        retrieved = db.get_image("test_img_001")

        # Should return ImageRecord, not dict
        assert isinstance(retrieved, ImageRecord)
        assert retrieved.id == "test_img_001"
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
    record1 = ImageRecord(
        id="img1",
        source_path=Path("/test/img1.jpg"),
        file_type=FileType.IMAGE,
        checksum="hash1",
        status=ImageStatus.PENDING,
    )

    record2 = ImageRecord(
        id="img2",
        source_path=Path("/test/img2.png"),
        file_type=FileType.IMAGE,
        checksum="hash2",
        status=ImageStatus.COMPLETE,
    )

    with CatalogDB(tmp_path) as db:
        db.add_image(record1)
        db.add_image(record2)

        all_images = db.get_all_images()

        # Should return dict mapping id -> ImageRecord
        assert isinstance(all_images, dict)
        assert len(all_images) == 2

        assert isinstance(all_images["img1"], ImageRecord)
        assert all_images["img1"].file_type == FileType.IMAGE

        assert isinstance(all_images["img2"], ImageRecord)
        assert all_images["img2"].file_type == FileType.IMAGE
        assert all_images["img2"].status == ImageStatus.COMPLETE


def test_list_images_returns_ids(tmp_path):
    """Test that list_images returns list of image IDs."""
    record1 = ImageRecord(
        id="img1",
        source_path=Path("/test/img1.jpg"),
        file_type=FileType.IMAGE,
        checksum="h1",
        status=ImageStatus.PENDING,
    )
    record2 = ImageRecord(
        id="img2",
        source_path=Path("/test/img2.jpg"),
        file_type=FileType.IMAGE,
        checksum="h2",
        status=ImageStatus.PENDING,
    )

    with CatalogDB(tmp_path) as db:
        db.add_image(record1)
        db.add_image(record2)

        ids = db.list_images()

        assert isinstance(ids, list)
        assert len(ids) == 2
        assert "img1" in ids
        assert "img2" in ids
