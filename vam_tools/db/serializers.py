"""
Serializers for converting between Pydantic models and PostgreSQL JSONB.

Handles bidirectional conversion:
- serialize_*: Pydantic model → JSON-serializable dict
- deserialize_*: Dict → Pydantic model
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from vam_tools.core.types import (
    DateInfo,
    FileType,
    ImageMetadata,
    ImageRecord,
    ImageStatus,
)


def _safe_deserialize_datetime(iso_string: Optional[str]) -> Optional[datetime]:
    """
    Safely deserialize ISO format datetime string.

    Args:
        iso_string: ISO format datetime string or None

    Returns:
        datetime object or None if string is invalid/None
    """
    if not iso_string:
        return None
    try:
        return datetime.fromisoformat(iso_string)
    except (ValueError, TypeError):
        return None


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
        "filename_date": (
            date_info.filename_date.isoformat() if date_info.filename_date else None
        ),
        "directory_date": date_info.directory_date,
        "filesystem_created": (
            date_info.filesystem_created.isoformat()
            if date_info.filesystem_created
            else None
        ),
        "filesystem_modified": (
            date_info.filesystem_modified.isoformat()
            if date_info.filesystem_modified
            else None
        ),
        "selected_date": (
            date_info.selected_date.isoformat() if date_info.selected_date else None
        ),
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
        exif_dates[key] = _safe_deserialize_datetime(value)

    return DateInfo(
        exif_dates=exif_dates,
        filename_date=_safe_deserialize_datetime(data.get("filename_date")),
        directory_date=data.get("directory_date"),
        filesystem_created=_safe_deserialize_datetime(data.get("filesystem_created")),
        filesystem_modified=_safe_deserialize_datetime(data.get("filesystem_modified")),
        selected_date=_safe_deserialize_datetime(data.get("selected_date")),
        selected_source=data.get("selected_source"),
        confidence=data.get("confidence", 0),
        suspicious=data.get("suspicious", False),
        user_verified=data.get("user_verified", False),
    )


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
        # Convert shutter_speed to string if it's a float (legacy data)
        shutter_speed=(
            str(data.get("shutter_speed"))
            if data.get("shutter_speed") is not None
            else None
        ),
        iso=data.get("iso"),
        gps_latitude=data.get("gps_latitude"),
        gps_longitude=data.get("gps_longitude"),
        # Support both old (perceptual_hash_dhash) and new (dhash) key names
        perceptual_hash_dhash=data.get("perceptual_hash_dhash") or data.get("dhash"),
        perceptual_hash_ahash=data.get("perceptual_hash_ahash") or data.get("ahash"),
        perceptual_hash_whash=data.get("perceptual_hash_whash") or data.get("whash"),
        merged_from=data.get("merged_from", []),
    )


def serialize_image_record(record: ImageRecord) -> Dict[str, Any]:
    """
    Serialize ImageRecord to JSON-serializable dict.

    Args:
        record: ImageRecord object to serialize

    Returns:
        Dictionary suitable for JSONB storage
    """
    # Prefer status_id (FK to image_statuses), fall back to status enum
    status_value = record.status_id if record.status_id else (record.status.value if record.status else None)

    return {
        "id": record.id,
        "source_path": str(record.source_path),
        "file_type": record.file_type.value if record.file_type else None,
        "checksum": record.checksum,
        "status": status_value,
        "dates": serialize_date_info(record.dates) if record.dates else {},
        "metadata": (
            serialize_image_metadata(record.metadata) if record.metadata else {}
        ),
    }


def deserialize_image_record(data: Dict[str, Any]) -> ImageRecord:
    """
    Deserialize dict to ImageRecord object.

    Args:
        data: Dictionary from JSONB storage

    Returns:
        ImageRecord object
    """
    # Deserialize nested objects
    dates = None
    if data.get("dates"):
        dates = deserialize_date_info(data["dates"])

    metadata = None
    if data.get("metadata"):
        metadata = deserialize_image_metadata(data["metadata"])

    # Deserialize enums
    file_type = None
    if data.get("file_type"):
        file_type = FileType(data["file_type"])

    # Handle status - can be either:
    # 1. status_id (FK to image_statuses: active, rejected, archived, flagged)
    # 2. Old ImageStatus enum value (pending, analyzing, etc.)
    status_str = data.get("status")
    status_id = None
    status = ImageStatus.PENDING  # Default

    if status_str:
        # Check if it's a valid status_id (database FK value)
        if status_str in ("active", "rejected", "archived", "flagged"):
            status_id = status_str
            # For backward compatibility, also set status enum
            status_mapping = {
                "active": ImageStatus.PENDING,
                "rejected": ImageStatus.PENDING,
                "archived": ImageStatus.COMPLETE,
                "flagged": ImageStatus.NEEDS_REVIEW,
            }
            status = status_mapping.get(status_str, ImageStatus.PENDING)
        else:
            # It's an old ImageStatus enum value
            try:
                status = ImageStatus(status_str)
            except ValueError:
                status = ImageStatus.PENDING

    return ImageRecord(
        id=data["id"],
        source_path=Path(data["source_path"]),
        file_type=file_type,
        checksum=data["checksum"],
        status_id=status_id,
        status=status,
        dates=dates,
        metadata=metadata,
    )
