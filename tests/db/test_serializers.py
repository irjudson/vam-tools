"""Tests for PostgreSQL ↔ Pydantic serializers."""

from datetime import datetime

import pytest

from pathlib import Path

from vam_tools.core.types import DateInfo, ImageMetadata, FileType, ImageStatus, ImageRecord
from vam_tools.db.serializers import (
    serialize_date_info,
    deserialize_date_info,
    serialize_image_metadata,
    deserialize_image_metadata,
    serialize_image_record,
    deserialize_image_record,
)


def test_serialize_date_info_with_all_fields():
    """Test serializing DateInfo with all fields populated."""
    date_info = DateInfo(
        exif_dates={"DateTimeOriginal": datetime(2023, 6, 15, 14, 30, 22)},
        filename_date=datetime(2023, 6, 15),
        directory_date="2023-06",
        filesystem_created=datetime(2023, 6, 15, 10, 0, 0),
        filesystem_modified=datetime(2023, 6, 15, 14, 30, 22),
        selected_date=datetime(2023, 6, 15, 14, 30, 22),
        selected_source="exif",
        confidence=90,
        suspicious=False,
        user_verified=True,
    )

    result = serialize_date_info(date_info)

    # Should be JSON-serializable dict
    assert isinstance(result, dict)
    assert result["selected_date"] == "2023-06-15T14:30:22"
    assert result["exif_dates"]["DateTimeOriginal"] == "2023-06-15T14:30:22"
    assert result["confidence"] == 90
    assert result["user_verified"] is True


def test_serialize_date_info_empty():
    """Test serializing empty DateInfo."""
    date_info = DateInfo()

    result = serialize_date_info(date_info)

    assert result == {
        "exif_dates": {},
        "filename_date": None,
        "directory_date": None,
        "filesystem_created": None,
        "filesystem_modified": None,
        "selected_date": None,
        "selected_source": None,
        "confidence": 0,
        "suspicious": False,
        "user_verified": False,
    }


def test_deserialize_date_info():
    """Test deserializing dict back to DateInfo."""
    data = {
        "exif_dates": {"DateTimeOriginal": "2023-06-15T14:30:22"},
        "selected_date": "2023-06-15T14:30:22",
        "confidence": 90,
        "user_verified": True,
    }

    result = deserialize_date_info(data)

    assert isinstance(result, DateInfo)
    assert result.selected_date == datetime(2023, 6, 15, 14, 30, 22)
    assert result.exif_dates["DateTimeOriginal"] == datetime(2023, 6, 15, 14, 30, 22)
    assert result.confidence == 90
    assert result.user_verified is True


def test_deserialize_date_info_empty():
    """Test deserializing empty dict to DateInfo."""
    result = deserialize_date_info({})

    assert isinstance(result, DateInfo)
    assert result.selected_date is None
    assert result.confidence == 0


def test_serialize_image_metadata_with_all_fields():
    """Test serializing ImageMetadata with all fields."""
    metadata = ImageMetadata(
        exif={"Make": "Canon", "Model": "EOS 5D"},
        format="JPEG",
        resolution=(1920, 1080),
        width=1920,
        height=1080,
        size_bytes=1024000,
        camera_make="Canon",
        camera_model="EOS 5D",
        lens_model="24-70mm",
        focal_length=50.0,
        aperture=2.8,
        shutter_speed="1/200",
        iso=400,
        gps_latitude=37.7749,
        gps_longitude=-122.4194,
        perceptual_hash_dhash="abc123",
        perceptual_hash_ahash="def456",
        perceptual_hash_whash="ghi789",
        merged_from=["img1", "img2"],
    )

    result = serialize_image_metadata(metadata)

    assert isinstance(result, dict)
    assert result["format"] == "JPEG"
    assert result["width"] == 1920
    assert result["height"] == 1080
    assert result["camera_make"] == "Canon"
    assert result["gps_latitude"] == 37.7749
    assert result["perceptual_hash_dhash"] == "abc123"
    assert result["merged_from"] == ["img1", "img2"]


def test_serialize_image_metadata_empty():
    """Test serializing empty ImageMetadata."""
    metadata = ImageMetadata()
    result = serialize_image_metadata(metadata)

    assert result["exif"] == {}
    assert result["format"] is None
    assert result["size_bytes"] is None


def test_deserialize_image_metadata():
    """Test deserializing dict to ImageMetadata."""
    data = {
        "format": "JPEG",
        "width": 1920,
        "height": 1080,
        "size_bytes": 1024000,
        "camera_make": "Canon",
        "perceptual_hash_dhash": "abc123",
    }

    result = deserialize_image_metadata(data)

    assert isinstance(result, ImageMetadata)
    assert result.format == "JPEG"
    assert result.width == 1920
    assert result.camera_make == "Canon"


def test_round_trip_image_metadata():
    """Test that serialize→deserialize is lossless."""
    original = ImageMetadata(
        format="PNG",
        width=3840,
        height=2160,
        size_bytes=2048000,
        iso=800,
    )

    serialized = serialize_image_metadata(original)
    deserialized = deserialize_image_metadata(serialized)

    assert deserialized.format == original.format
    assert deserialized.width == original.width
    assert deserialized.height == original.height
    assert deserialized.size_bytes == original.size_bytes
    assert deserialized.iso == original.iso


def test_serialize_image_record_complete():
    """Test serializing complete ImageRecord with all nested objects."""
    date_info = DateInfo(
        selected_date=datetime(2023, 6, 15, 14, 30, 22),
        selected_source="exif",
        confidence=90,
    )

    metadata = ImageMetadata(
        exif={"Make": "Canon"},
        camera_make="Canon",
        camera_model="EOS R5",
        size_bytes=1024000,
        resolution=(1920, 1080),
    )

    record = ImageRecord(
        id="test123",
        source_path=Path("/path/to/image.jpg"),
        file_type=FileType.IMAGE,
        checksum="abc123def456",
        status=ImageStatus.PENDING,
        dates=date_info,
        metadata=metadata,
    )

    result = serialize_image_record(record)

    assert isinstance(result, dict)
    assert result["id"] == "test123"
    assert result["source_path"] == "/path/to/image.jpg"
    assert result["file_type"] == "image"
    assert result["checksum"] == "abc123def456"
    assert result["status"] == "pending"

    # Nested objects should be serialized dicts
    assert isinstance(result["dates"], dict)
    assert result["dates"]["selected_date"] == "2023-06-15T14:30:22"
    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["camera_make"] == "Canon"


def test_serialize_image_record_minimal():
    """Test serializing minimal ImageRecord."""
    record = ImageRecord(
        id="minimal",
        source_path=Path("/test.jpg"),
        file_type=FileType.IMAGE,
        checksum="abc",
        status=ImageStatus.PENDING,
    )

    result = serialize_image_record(record)

    assert result["id"] == "minimal"
    # dates and metadata should be serialized empty DateInfo/ImageMetadata
    assert isinstance(result["dates"], dict)
    assert result["dates"]["confidence"] == 0
    assert result["dates"]["selected_date"] is None
    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["format"] is None


def test_deserialize_image_record():
    """Test deserializing ImageRecord from dict."""
    data = {
        "id": "test123",
        "source_path": "/path/to/image.jpg",
        "file_type": "image",
        "checksum": "abc123",
        "status": "pending",
        "dates": {
            "selected_date": "2023-06-15T14:30:22",
            "selected_source": "exif",
            "confidence": 90,
        },
        "metadata": {
            "camera_make": "Canon",
            "camera_model": "EOS R5",
            "size_bytes": 1024000,
            "resolution": [1920, 1080],
        },
    }

    result = deserialize_image_record(data)

    assert isinstance(result, ImageRecord)
    assert result.id == "test123"
    assert str(result.source_path) == "/path/to/image.jpg"
    assert result.file_type == FileType.IMAGE
    assert result.status == ImageStatus.PENDING

    # Nested objects should be deserialized
    assert isinstance(result.dates, DateInfo)
    assert result.dates.selected_date == datetime(2023, 6, 15, 14, 30, 22)
    assert isinstance(result.metadata, ImageMetadata)
    assert result.metadata.camera_make == "Canon"


def test_round_trip_image_record():
    """Test round-trip serialization preserves all data."""
    original = ImageRecord(
        id="roundtrip",
        source_path=Path("/test/image.jpg"),
        file_type=FileType.VIDEO,
        checksum="hash123",
        status=ImageStatus.COMPLETE,
        dates=DateInfo(
            selected_date=datetime(2023, 1, 1, 12, 0, 0),
            confidence=85,
        ),
        metadata=ImageMetadata(
            size_bytes=2048000,
            resolution=(3840, 2160),
        ),
    )

    serialized = serialize_image_record(original)
    deserialized = deserialize_image_record(serialized)

    assert deserialized.id == original.id
    assert deserialized.source_path == original.source_path
    assert deserialized.file_type == original.file_type
    assert deserialized.checksum == original.checksum
    assert deserialized.status == original.status
    assert deserialized.dates.selected_date == original.dates.selected_date
    assert deserialized.metadata.size_bytes == original.metadata.size_bytes
