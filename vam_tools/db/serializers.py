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
