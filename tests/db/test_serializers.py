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
