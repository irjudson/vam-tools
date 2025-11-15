"""
Serializers for converting between Pydantic models and PostgreSQL JSONB.

Handles bidirectional conversion:
- serialize_*: Pydantic model → JSON-serializable dict
- deserialize_*: Dict → Pydantic model
"""

from datetime import datetime
from typing import Any, Dict, Optional

from vam_tools.core.types import DateInfo, ImageMetadata, ImageRecord


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
        shutter_speed=data.get("shutter_speed"),
        iso=data.get("iso"),
        gps_latitude=data.get("gps_latitude"),
        gps_longitude=data.get("gps_longitude"),
        perceptual_hash_dhash=data.get("perceptual_hash_dhash"),
        perceptual_hash_ahash=data.get("perceptual_hash_ahash"),
        perceptual_hash_whash=data.get("perceptual_hash_whash"),
        merged_from=data.get("merged_from", []),
    )
