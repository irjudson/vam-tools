# PostgreSQL ↔ Pydantic ORM Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build bidirectional serialization layer between PostgreSQL JSONB storage and Pydantic models (ImageRecord, DateInfo, ImageMetadata) to enable full PostgreSQL migration.

**Architecture:** Create serialize/deserialize functions in `vam_tools/db/serializers.py` to convert between Pydantic models and PostgreSQL-compatible dictionaries. Update CatalogDB methods to use serializers. Use TDD approach - write tests first, implement minimal code, iterate.

**Tech Stack:**
- PostgreSQL JSONB for flexible storage
- Pydantic for Python object validation
- SQLAlchemy for query execution
- pytest for TDD

---

## Task 1: Create Serializer Module Foundation

**Files:**
- Create: `vam_tools/db/serializers.py`
- Test: `tests/db/test_serializers.py`

**Step 1: Write failing test for DateInfo serialization**

Create test file `tests/db/test_serializers.py`:

```python
"""Tests for PostgreSQL ↔ Pydantic serializers."""

from datetime import datetime

import pytest

from vam_tools.core.types import DateInfo
from vam_tools.db.serializers import serialize_date_info, deserialize_date_info


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
```

**Step 2: Run test to verify it fails**

```bash
./venv/bin/pytest tests/db/test_serializers.py::test_serialize_date_info_with_all_fields -xvs
```

Expected: `ModuleNotFoundError: No module named 'vam_tools.db.serializers'`

**Step 3: Create minimal serializer implementation**

Create file `vam_tools/db/serializers.py`:

```python
"""
Serializers for converting between Pydantic models and PostgreSQL JSONB.

Handles bidirectional conversion:
- serialize_*: Pydantic model → JSON-serializable dict
- deserialize_*: Dict → Pydantic model
"""

from datetime import datetime
from typing import Any, Dict, Optional

from vam_tools.core.types import DateInfo, ImageMetadata, ImageRecord


def serialize_date_info(date_info: DateInfo) -> Dict[str, Any]:
    """
    Serialize DateInfo to JSON-serializable dict.

    Args:
        date_info: DateInfo object to serialize

    Returns:
        Dictionary suitable for JSONB storage
    """
    # Convert datetime objects to ISO format strings
    exif_dates = {}
    for key, value in date_info.exif_dates.items():
        exif_dates[key] = value.isoformat() if value else None

    return {
        "exif_dates": exif_dates,
        "filename_date": date_info.filename_date.isoformat() if date_info.filename_date else None,
        "directory_date": date_info.directory_date,
        "filesystem_created": date_info.filesystem_created.isoformat() if date_info.filesystem_created else None,
        "filesystem_modified": date_info.filesystem_modified.isoformat() if date_info.filesystem_modified else None,
        "selected_date": date_info.selected_date.isoformat() if date_info.selected_date else None,
        "selected_source": date_info.selected_source,
        "confidence": date_info.confidence,
        "suspicious": date_info.suspicious,
        "user_verified": date_info.user_verified,
    }


def deserialize_date_info(data: Dict[str, Any]) -> DateInfo:
    """
    Deserialize dict to DateInfo object.

    Args:
        data: Dictionary from JSONB storage

    Returns:
        DateInfo object
    """
    # Convert ISO format strings back to datetime objects
    exif_dates = {}
    for key, value in data.get("exif_dates", {}).items():
        exif_dates[key] = datetime.fromisoformat(value) if value else None

    return DateInfo(
        exif_dates=exif_dates,
        filename_date=datetime.fromisoformat(data["filename_date"]) if data.get("filename_date") else None,
        directory_date=data.get("directory_date"),
        filesystem_created=datetime.fromisoformat(data["filesystem_created"]) if data.get("filesystem_created") else None,
        filesystem_modified=datetime.fromisoformat(data["filesystem_modified"]) if data.get("filesystem_modified") else None,
        selected_date=datetime.fromisoformat(data["selected_date"]) if data.get("selected_date") else None,
        selected_source=data.get("selected_source"),
        confidence=data.get("confidence", 0),
        suspicious=data.get("suspicious", False),
        user_verified=data.get("user_verified", False),
    )
```

**Step 4: Run tests to verify they pass**

```bash
./venv/bin/pytest tests/db/test_serializers.py -xvs
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add vam_tools/db/serializers.py tests/db/test_serializers.py
git commit -m "feat: add DateInfo serializers for PostgreSQL JSONB"
```

---

## Task 2: ImageMetadata Serializers

**Files:**
- Modify: `vam_tools/db/serializers.py`
- Test: `tests/db/test_serializers.py`

**Step 1: Write failing tests for ImageMetadata**

Add to `tests/db/test_serializers.py`:

```python
from vam_tools.db.serializers import serialize_image_metadata, deserialize_image_metadata


def test_serialize_image_metadata_with_all_fields():
    """Test serializing ImageMetadata with all fields."""
    from vam_tools.core.types import ImageMetadata

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
    from vam_tools.core.types import ImageMetadata

    metadata = ImageMetadata()
    result = serialize_image_metadata(metadata)

    assert result["exif"] == {}
    assert result["format"] is None
    assert result["size_bytes"] is None


def test_deserialize_image_metadata():
    """Test deserializing dict to ImageMetadata."""
    from vam_tools.core.types import ImageMetadata

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
    from vam_tools.core.types import ImageMetadata

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
```

**Step 2: Run tests to verify they fail**

```bash
./venv/bin/pytest tests/db/test_serializers.py::test_serialize_image_metadata_with_all_fields -xvs
```

Expected: `AttributeError: module 'vam_tools.db.serializers' has no attribute 'serialize_image_metadata'`

**Step 3: Implement ImageMetadata serializers**

Add to `vam_tools/db/serializers.py`:

```python
def serialize_image_metadata(metadata: ImageMetadata) -> Dict[str, Any]:
    """
    Serialize ImageMetadata to JSON-serializable dict.

    Args:
        metadata: ImageMetadata object to serialize

    Returns:
        Dictionary suitable for JSONB storage
    """
    return {
        "exif": metadata.exif,
        "format": metadata.format,
        "resolution": list(metadata.resolution) if metadata.resolution else None,
        "width": metadata.width,
        "height": metadata.height,
        "size_bytes": metadata.size_bytes,
        "camera_make": metadata.camera_make,
        "camera_model": metadata.camera_model,
        "lens_model": metadata.lens_model,
        "focal_length": metadata.focal_length,
        "aperture": metadata.aperture,
        "shutter_speed": metadata.shutter_speed,
        "iso": metadata.iso,
        "gps_latitude": metadata.gps_latitude,
        "gps_longitude": metadata.gps_longitude,
        "perceptual_hash_dhash": metadata.perceptual_hash_dhash,
        "perceptual_hash_ahash": metadata.perceptual_hash_ahash,
        "perceptual_hash_whash": metadata.perceptual_hash_whash,
        "merged_from": metadata.merged_from,
    }


def deserialize_image_metadata(data: Dict[str, Any]) -> ImageMetadata:
    """
    Deserialize dict to ImageMetadata object.

    Args:
        data: Dictionary from JSONB storage

    Returns:
        ImageMetadata object
    """
    resolution = data.get("resolution")
    if resolution and isinstance(resolution, list):
        resolution = tuple(resolution)

    return ImageMetadata(
        exif=data.get("exif", {}),
        format=data.get("format"),
        resolution=resolution,
        width=data.get("width"),
        height=data.get("height"),
        size_bytes=data.get("size_bytes"),
        camera_make=data.get("camera_make"),
        camera_model=data.get("camera_model"),
        lens_model=data.get("lens_model"),
        focal_length=data.get("focal_length"),
        aperture=data.get("aperture"),
        shutter_speed=data.get("shutter_speed"),
        iso=data.get("iso"),
        gps_latitude=data.get("gps_latitude"),
        gps_longitude=data.get("gps_longitude"),
        perceptual_hash_dhash=data.get("perceptual_hash_dhash"),
        perceptual_hash_ahash=data.get("perceptual_hash_ahash"),
        perceptual_hash_whash=data.get("perceptual_hash_whash"),
        merged_from=data.get("merged_from", []),
    )
```

**Step 4: Run tests to verify they pass**

```bash
./venv/bin/pytest tests/db/test_serializers.py -k image_metadata -xvs
```

Expected: All ImageMetadata tests PASS

**Step 5: Commit**

```bash
git add vam_tools/db/serializers.py tests/db/test_serializers.py
git commit -m "feat: add ImageMetadata serializers for PostgreSQL JSONB"
```

---

## Task 3: ImageRecord Serializers

**Files:**
- Modify: `vam_tools/db/serializers.py`
- Test: `tests/db/test_serializers.py`

**Step 1: Write failing tests for ImageRecord**

Add to `tests/db/test_serializers.py`:

```python
from pathlib import Path
from vam_tools.core.types import FileType, ImageStatus
from vam_tools.db.serializers import serialize_image_record, deserialize_image_record


def test_serialize_image_record_complete():
    """Test serializing complete ImageRecord."""
    from vam_tools.core.types import ImageRecord, DateInfo, ImageMetadata

    record = ImageRecord(
        id="img123",
        source_path=Path("/photos/test.jpg"),
        file_type=FileType.IMAGE,
        checksum="abc123def456",
        dates=DateInfo(
            selected_date=datetime(2023, 6, 15, 14, 30, 22),
            confidence=90,
        ),
        metadata=ImageMetadata(
            format="JPEG",
            width=1920,
            height=1080,
            size_bytes=1024000,
        ),
        status=ImageStatus.COMPLETE,
    )

    result = serialize_image_record(record)

    assert result["id"] == "img123"
    assert result["source_path"] == "/photos/test.jpg"
    assert result["file_type"] == "image"
    assert result["checksum"] == "abc123def456"
    assert result["status"] == "complete"

    # Nested objects should be serialized
    assert isinstance(result["dates"], dict)
    assert result["dates"]["selected_date"] == "2023-06-15T14:30:22"
    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["format"] == "JPEG"


def test_deserialize_image_record_from_db_row():
    """Test deserializing database row to ImageRecord."""
    from vam_tools.core.types import ImageRecord

    # Simulate a row from PostgreSQL
    db_row = {
        "id": "img123",
        "source_path": "/photos/test.jpg",
        "file_type": "image",
        "checksum": "abc123def456",
        "size_bytes": 1024000,
        "dates": {
            "selected_date": "2023-06-15T14:30:22",
            "confidence": 90,
        },
        "metadata": {
            "format": "JPEG",
            "width": 1920,
            "height": 1080,
            "size_bytes": 1024000,
        },
        "quality_score": 85,
        "status": "complete",
        "created_at": "2023-06-15T10:00:00",
        "updated_at": "2023-06-15T14:30:22",
    }

    result = deserialize_image_record(db_row)

    assert isinstance(result, ImageRecord)
    assert result.id == "img123"
    assert str(result.source_path) == "/photos/test.jpg"
    assert result.file_type == FileType.IMAGE
    assert result.checksum == "abc123def456"
    assert result.status == ImageStatus.COMPLETE

    # Nested objects should be deserialized
    assert isinstance(result.dates, DateInfo)
    assert result.dates.selected_date == datetime(2023, 6, 15, 14, 30, 22)
    assert isinstance(result.metadata, ImageMetadata)
    assert result.metadata.format == "JPEG"


def test_round_trip_image_record():
    """Test that serialize→deserialize is lossless."""
    from vam_tools.core.types import ImageRecord, DateInfo, ImageMetadata

    original = ImageRecord(
        id="test123",
        source_path=Path("/test/photo.jpg"),
        file_type=FileType.IMAGE,
        checksum="checksum123",
        dates=DateInfo(
            selected_date=datetime(2023, 1, 1, 12, 0, 0),
            confidence=100,
        ),
        metadata=ImageMetadata(
            format="PNG",
            width=800,
            height=600,
        ),
        status=ImageStatus.PENDING,
    )

    # Serialize
    serialized = serialize_image_record(original)

    # Add mock DB fields
    serialized["created_at"] = "2023-01-01T12:00:00"
    serialized["updated_at"] = "2023-01-01T12:00:00"
    serialized["quality_score"] = 0

    # Deserialize
    deserialized = deserialize_image_record(serialized)

    assert deserialized.id == original.id
    assert deserialized.checksum == original.checksum
    assert deserialized.file_type == original.file_type
    assert deserialized.status == original.status
    assert deserialized.dates.selected_date == original.dates.selected_date
    assert deserialized.metadata.format == original.metadata.format
```

**Step 2: Run tests to verify they fail**

```bash
./venv/bin/pytest tests/db/test_serializers.py::test_serialize_image_record_complete -xvs
```

Expected: `AttributeError: module 'vam_tools.db.serializers' has no attribute 'serialize_image_record'`

**Step 3: Implement ImageRecord serializers**

Add to `vam_tools/db/serializers.py`:

```python
from pathlib import Path
from vam_tools.core.types import FileType, ImageStatus


def serialize_image_record(record: ImageRecord) -> Dict[str, Any]:
    """
    Serialize ImageRecord to dict for database storage.

    Args:
        record: ImageRecord object to serialize

    Returns:
        Dictionary with all fields ready for PostgreSQL
    """
    return {
        "id": record.id,
        "source_path": str(record.source_path),
        "file_type": record.file_type.value,
        "checksum": record.checksum,
        "dates": serialize_date_info(record.dates),
        "metadata": serialize_image_metadata(record.metadata),
        "status": record.status.value,
    }


def deserialize_image_record(data: Dict[str, Any]) -> ImageRecord:
    """
    Deserialize database row to ImageRecord object.

    Args:
        data: Dictionary from database row

    Returns:
        ImageRecord object
    """
    # Handle dates - could be dict or already DateInfo
    dates_data = data.get("dates", {})
    if isinstance(dates_data, dict):
        dates = deserialize_date_info(dates_data)
    else:
        dates = dates_data

    # Handle metadata - could be dict or already ImageMetadata
    metadata_data = data.get("metadata", {})
    if isinstance(metadata_data, dict):
        metadata = deserialize_image_metadata(metadata_data)
    else:
        metadata = metadata_data

    # Handle file_type enum
    file_type = data.get("file_type")
    if isinstance(file_type, str):
        file_type = FileType(file_type)

    # Handle status enum
    status = data.get("status", "pending")
    if isinstance(status, str):
        status = ImageStatus(status)

    return ImageRecord(
        id=data["id"],
        source_path=Path(data["source_path"]),
        file_type=file_type,
        checksum=data["checksum"],
        dates=dates,
        metadata=metadata,
        status=status,
    )
```

**Step 4: Run tests to verify they pass**

```bash
./venv/bin/pytest tests/db/test_serializers.py -k image_record -xvs
```

Expected: All ImageRecord tests PASS

**Step 5: Commit**

```bash
git add vam_tools/db/serializers.py tests/db/test_serializers.py
git commit -m "feat: add ImageRecord serializers for PostgreSQL JSONB"
```

---

## Task 4: Integrate Serializers into CatalogDB

**Files:**
- Modify: `vam_tools/db/catalog_db.py`
- Test: `tests/db/test_catalog_db.py`

**Step 1: Write failing test for add_image with serialization**

Create `tests/db/test_catalog_db.py`:

```python
"""Tests for CatalogDB with serialization."""

from datetime import datetime
from pathlib import Path
import uuid

import pytest

from vam_tools.core.types import DateInfo, FileType, ImageMetadata, ImageRecord, ImageStatus
from vam_tools.db import CatalogDB


@pytest.fixture
def test_catalog_db(tmp_path):
    """Create test catalog database."""
    catalog_db = CatalogDB(tmp_path / "test_catalog")
    yield catalog_db
    catalog_db.close()


def test_add_image_serializes_correctly(test_catalog_db):
    """Test that add_image correctly serializes ImageRecord to database."""
    record = ImageRecord(
        id="test_img_1",
        source_path=Path("/test/photo.jpg"),
        file_type=FileType.IMAGE,
        checksum="abc123",
        dates=DateInfo(
            selected_date=datetime(2023, 6, 15, 14, 30, 22),
            confidence=90,
        ),
        metadata=ImageMetadata(
            format="JPEG",
            width=1920,
            height=1080,
            size_bytes=1024000,
        ),
        status=ImageStatus.PENDING,
    )

    # Should not raise
    test_catalog_db.add_image(record)

    # Verify it was stored
    result = test_catalog_db.get_image("test_img_1")
    assert result is not None
    assert result["id"] == "test_img_1"


def test_get_image_deserializes_correctly(test_catalog_db):
    """Test that get_image returns properly deserialized ImageRecord."""
    # Add an image
    record = ImageRecord(
        id="test_img_2",
        source_path=Path("/test/photo2.jpg"),
        file_type=FileType.IMAGE,
        checksum="def456",
        dates=DateInfo(
            selected_date=datetime(2023, 7, 1, 10, 0, 0),
            confidence=95,
        ),
        metadata=ImageMetadata(
            format="PNG",
            width=800,
            height=600,
        ),
        status=ImageStatus.COMPLETE,
    )

    test_catalog_db.add_image(record)

    # Get it back as deserialized object
    result_dict = test_catalog_db.get_image("test_img_2")

    # Manually deserialize using our serializer
    from vam_tools.db.serializers import deserialize_image_record
    result_record = deserialize_image_record(result_dict)

    assert isinstance(result_record, ImageRecord)
    assert result_record.id == "test_img_2"
    assert result_record.file_type == FileType.IMAGE
    assert result_record.status == ImageStatus.COMPLETE
    assert isinstance(result_record.dates, DateInfo)
    assert result_record.dates.selected_date == datetime(2023, 7, 1, 10, 0, 0)
    assert isinstance(result_record.metadata, ImageMetadata)
    assert result_record.metadata.format == "PNG"


def test_get_all_images_deserializes_correctly(test_catalog_db):
    """Test that get_all_images returns deserializable records."""
    # Add multiple images
    for i in range(3):
        record = ImageRecord(
            id=f"img_{i}",
            source_path=Path(f"/test/photo{i}.jpg"),
            file_type=FileType.IMAGE,
            checksum=f"check{i}",
            dates=DateInfo(confidence=80 + i),
            metadata=ImageMetadata(format="JPEG"),
            status=ImageStatus.PENDING,
        )
        test_catalog_db.add_image(record)

    # Get all images
    all_images = test_catalog_db.get_all_images()

    assert len(all_images) == 3

    # Deserialize and validate
    from vam_tools.db.serializers import deserialize_image_record

    for img_id, img_data in all_images.items():
        record = deserialize_image_record(img_data)
        assert isinstance(record, ImageRecord)
        assert isinstance(record.dates, DateInfo)
        assert isinstance(record.metadata, ImageMetadata)
```

**Step 2: Run tests to verify they fail**

```bash
./venv/bin/pytest tests/db/test_catalog_db.py::test_add_image_serializes_correctly -xvs
```

Expected: Test might pass already or fail with serialization issues

**Step 3: Update CatalogDB.add_image to use serializers**

Modify `vam_tools/db/catalog_db.py`:

```python
from .serializers import serialize_date_info, serialize_image_metadata, serialize_image_record

# In add_image method, replace the manual dict creation with:

def add_image(self, image_record: Any) -> None:
    """
    Add an image to the database.

    Args:
        image_record: ImageRecord object to add
    """
    if self.session is None:
        self.connect()

    # Serialize the record
    serialized = serialize_image_record(image_record)

    # Insert image with JSONB metadata
    self.session.execute(
        text("""
            INSERT INTO images (
                id, catalog_id, source_path, file_type, checksum,
                size_bytes, dates, metadata, quality_score, status,
                created_at, updated_at
            ) VALUES (
                :id, :catalog_id, :source_path, :file_type, :checksum,
                :size_bytes, CAST(:dates AS jsonb), CAST(:metadata AS jsonb), :quality_score, :status,
                NOW(), NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                source_path = EXCLUDED.source_path,
                updated_at = NOW()
        """),
        {
            "id": serialized["id"],
            "catalog_id": self.catalog_id,
            "source_path": serialized["source_path"],
            "file_type": serialized["file_type"],
            "checksum": serialized["checksum"],
            "size_bytes": serialized["metadata"].get("size_bytes", 0),
            "dates": str(serialized["dates"]),  # Convert to JSON string
            "metadata": str(serialized["metadata"]),  # Convert to JSON string
            "quality_score": 0,
            "status": serialized["status"],
        }
    )
    self.session.commit()
```

**Wait!** The dates/metadata need to be JSON strings. Update the serialization:

In `catalog_db.py`:

```python
import json

# In add_image:
"dates": json.dumps(serialized["dates"]),
"metadata": json.dumps(serialized["metadata"]),
```

**Step 4: Run tests to verify they pass**

```bash
./venv/bin/pytest tests/db/test_catalog_db.py::test_add_image_serializes_correctly -xvs
```

Expected: PASS

**Step 5: Update get_image and get_all_images to return raw dicts**

Currently they return dicts correctly. The deserialization will happen in the calling code.

**Step 6: Run all catalog DB tests**

```bash
./venv/bin/pytest tests/db/test_catalog_db.py -xvs
```

Expected: All tests PASS

**Step 7: Commit**

```bash
git add vam_tools/db/catalog_db.py vam_tools/db/serializers.py tests/db/test_catalog_db.py
git commit -m "feat: integrate serializers into CatalogDB for JSONB storage"
```

---

## Task 5: Fix Organization Tests with Serialization

**Files:**
- Modify: `tests/organization/test_file_organizer.py`

**Step 1: Run organization test to see current failure**

```bash
./venv/bin/pytest tests/organization/test_file_organizer.py::TestFileOrganizerDryRun::test_dry_run_preview -xvs
```

Expected: Failure because ImageRecord.model_validate() receives raw dict

**Step 2: Update test to properly handle deserialization**

The issue is in `vam_tools/organization/file_organizer.py:112-114`:

```python
images_dict = self.catalog.get_all_images()
images = [
    ImageRecord.model_validate(img_data) for img_data in images_dict.values()
]
```

This needs to use our deserializer instead. Update `file_organizer.py`:

```python
from ..db.serializers import deserialize_image_record

# In organize() method:
images_dict = self.catalog.get_all_images()
images = [
    deserialize_image_record(img_data) for img_data in images_dict.values()
]
```

**Step 3: Run test to verify it passes**

```bash
./venv/bin/pytest tests/organization/test_file_organizer.py::TestFileOrganizerDryRun::test_dry_run_preview -xvs
```

Expected: PASS

**Step 4: Run all organization tests**

```bash
./venv/bin/pytest tests/organization/test_file_organizer.py -xvs
```

Expected: Multiple tests PASS

**Step 5: Commit**

```bash
git add vam_tools/organization/file_organizer.py
git commit -m "fix: use deserialize_image_record in file organizer"
```

---

## Task 6: Restore and Fix Web API Tests

**Files:**
- Restore: `tests/web/test_api.py` (from git)
- Modify: Update to use CatalogDB

**Step 1: Restore original test file**

```bash
git restore tests/web/test_api.py
```

**Step 2: Update imports in test file**

Replace old imports:

```python
from vam_tools.core.database import CatalogDatabase
```

With:

```python
from vam_tools.db import CatalogDB as CatalogDatabase
from vam_tools.db.serializers import deserialize_image_record
```

**Step 3: Update test fixtures to use CatalogDB properly**

The tests manually insert into database. They need to:
1. Use CatalogDB.add_image() instead of raw SQL
2. Or update their raw SQL to match PostgreSQL schema

Recommend: Update to use add_image() for cleaner tests.

Example for test_list_images (lines ~157-180):

```python
def test_list_images(self, tmp_path: Path) -> None:
    """Test listing images from catalog."""
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()

    with CatalogDatabase(catalog_dir) as db:
        db.initialize()

        # Add test images using add_image
        for i in range(5):
            record = ImageRecord(
                id=f"img{i}",
                source_path=Path(f"/photos/photo{i}.jpg"),
                file_type=FileType.IMAGE,
                checksum=f"checksum{i}",
                dates=DateInfo(selected_date=datetime(2023, 1, 1 + i)),
                metadata=ImageMetadata(
                    format="JPEG",
                    width=1920,
                    height=1080,
                    size_bytes=1024000,
                ),
                status=ImageStatus.COMPLETE,
            )
            db.add_image(record)
```

**Step 4: Run one test to verify approach**

```bash
./venv/bin/pytest tests/web/test_api.py::TestAPI::test_list_images -xvs
```

Expected: May still fail, but with different error (progress!)

**Step 5: This task is large - document what's needed**

The web API tests need systematic refactoring. Each test needs:
- Import updates
- Use CatalogDB instead of raw SQL inserts
- Use ImageRecord objects
- Update assertions for new schema

Create a checklist in the test file:

```python
"""
Tests for FastAPI web API.

TODO: Refactor for PostgreSQL migration
- [x] Update imports to use CatalogDB
- [ ] Update all test fixtures to use add_image()
- [ ] Fix assertions for JSONB schema
- [ ] Update date handling for PostgreSQL
"""
```

**Step 6: Commit what we have**

```bash
git add tests/web/test_api.py
git commit -m "wip: start refactoring web API tests for PostgreSQL"
```

---

## Task 7: Run Full Test Suite and Document Results

**Step 1: Run full test suite**

```bash
./venv/bin/pytest --tb=no -q 2>&1 | tail -20
```

**Step 2: Document results**

Create `docs/POSTGRES_MIGRATION_STATUS.md`:

```markdown
# PostgreSQL Migration Status

## Completed

- ✅ Created serializers for DateInfo, ImageMetadata, ImageRecord
- ✅ Integrated serializers into CatalogDB
- ✅ Fixed organization tests with proper deserialization
- ✅ CatalogDB supports get_all_images(), get_image(), add_image()

## Test Results

**Current:** XXX passing, YYY failed

**Working:**
- Serializer tests: All passing
- CatalogDB tests: All passing
- Organization tests: N passing

**Needs Work:**
- Web API tests: Need systematic refactoring for PostgreSQL
- Preview extractor tests: Need deserialization updates
- CLI tests: Need deserialization updates

## Next Steps

1. Systematically refactor web API tests file-by-file
2. Add deserialization to preview_extractor.py
3. Update CLI tools to use serializers
4. Add more serializer tests for edge cases
```

**Step 3: Commit**

```bash
git add docs/POSTGRES_MIGRATION_STATUS.md
git commit -m "docs: add PostgreSQL migration status"
```

---

## Summary

This plan creates a complete ORM layer between PostgreSQL JSONB and Pydantic models using TDD:

1. **Task 1-3:** Build and test serializers (DateInfo → ImageMetadata → ImageRecord)
2. **Task 4:** Integrate into CatalogDB with full round-trip tests
3. **Task 5:** Fix organization code to use deserializers
4. **Task 6:** Begin web API test refactoring (large task, document approach)
5. **Task 7:** Document status and next steps

**Key Principles:**
- TDD: Write test first, see it fail, implement, see it pass, commit
- DRY: Serializers handle all conversion logic
- YAGNI: Only serialize fields we actually use
- Frequent commits: After each passing test

**Testing Strategy:**
- Unit tests for each serializer function
- Round-trip tests (serialize → deserialize)
- Integration tests in CatalogDB
- Real-world tests via organization module
