"""
Quality scoring for selecting the best copy among duplicates.

Scores images based on multiple factors:
- Format (RAW > JPEG > compressed formats)
- Resolution (higher is better)
- File size (larger is generally better for same format)
- EXIF completeness (more metadata is better)
"""

import logging
from typing import Dict

from ..core.types import FileType, ImageMetadata, QualityScore

logger = logging.getLogger(__name__)

# Format quality scores (0-100)
FORMAT_SCORES = {
    # RAW formats (highest quality)
    ".cr2": 100,
    ".cr3": 100,
    ".nef": 100,
    ".arw": 100,
    ".dng": 100,
    ".raf": 100,
    ".orf": 100,
    ".rw2": 100,
    ".pef": 100,
    ".srw": 100,
    # Lossless formats
    ".tiff": 90,
    ".tif": 90,
    ".png": 85,
    # High quality lossy
    ".heic": 80,
    ".heif": 80,
    # Standard JPEG
    ".jpg": 70,
    ".jpeg": 70,
    # Lower quality
    ".webp": 60,
    ".bmp": 50,
    ".gif": 40,
}


def calculate_quality_score(
    metadata: ImageMetadata, file_type: FileType
) -> QualityScore:
    """
    Calculate comprehensive quality score for an image.

    Args:
        metadata: Image metadata
        file_type: File type (IMAGE or VIDEO)

    Returns:
        QualityScore with individual component scores and overall score
    """
    # Component scores (0-100 each)
    format_score = _score_format(metadata.format or "")
    resolution_score = _score_resolution(metadata.width, metadata.height)
    size_score = _score_file_size(metadata.size_bytes, metadata.format or "")
    metadata_score = _score_metadata_completeness(metadata)

    # Weights for overall score
    # Format is most important (raw vs jpeg matters a lot)
    # Resolution is second (4K vs 1080p is significant)
    # Size and metadata are tie-breakers
    weights = {
        "format": 0.40,
        "resolution": 0.35,
        "size": 0.15,
        "metadata": 0.10,
    }

    overall = (
        format_score * weights["format"]
        + resolution_score * weights["resolution"]
        + size_score * weights["size"]
        + metadata_score * weights["metadata"]
    )

    return QualityScore(
        overall=round(overall, 2),
        format_score=format_score,
        resolution_score=resolution_score,
        size_score=size_score,
        metadata_score=metadata_score,
    )


def _score_format(format_str: str) -> float:
    """
    Score based on image format.

    Args:
        format_str: File format (e.g., "JPEG", "PNG", "CR2")

    Returns:
        Format score (0-100)
    """
    if not format_str:
        return 50.0  # Unknown format

    # Normalize to lowercase with dot
    ext = f".{format_str.lower()}"

    return float(FORMAT_SCORES.get(ext, 50.0))


def _score_resolution(width: int | None, height: int | None) -> float:
    """
    Score based on image resolution.

    Uses megapixel count with diminishing returns.
    8K (33MP) = 100
    4K (8.3MP) = 90
    1080p (2.1MP) = 70
    720p (0.9MP) = 50

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Resolution score (0-100)
    """
    if width is None or height is None or width <= 0 or height <= 0:
        return 0.0

    megapixels = (width * height) / 1_000_000

    # Score based on megapixel thresholds with smooth scaling
    if megapixels >= 33:  # 8K+
        return 100.0
    elif megapixels >= 20:  # High res
        return 90 + (megapixels - 20) / 13 * 10  # Scale 90-100
    elif megapixels >= 8:  # 4K
        return 80 + (megapixels - 8) / 12 * 10  # Scale 80-90
    elif megapixels >= 2:  # 1080p
        return 60 + (megapixels - 2) / 6 * 20  # Scale 60-80
    elif megapixels >= 1:  # 720p
        return 40 + (megapixels - 1) * 20  # Scale 40-60
    else:  # Very low res
        return megapixels * 40  # Scale 0-40


def _score_file_size(size_bytes: int | None, format_str: str) -> float:
    """
    Score based on file size relative to format.

    Larger files are generally better quality for the same format,
    but we normalize by expected size for the format.

    Args:
        size_bytes: File size in bytes
        format_str: File format

    Returns:
        Size score (0-100)
    """
    if size_bytes is None or size_bytes <= 0:
        return 0.0

    size_mb = size_bytes / (1024 * 1024)

    # Normalize to lowercase with dot
    ext = f".{format_str.lower()}" if format_str else ""

    # Different expectations for different formats
    if ext in [".cr2", ".cr3", ".nef", ".arw", ".dng"]:  # RAW
        # RAW files: 20-50 MB is typical
        if size_mb >= 50:
            return 100.0
        elif size_mb >= 20:
            return 80 + (size_mb - 20) / 30 * 20
        else:
            return size_mb / 20 * 80
    elif ext in [".tiff", ".tif"]:  # TIFF
        # TIFF files: 10-30 MB is typical
        if size_mb >= 30:
            return 100.0
        elif size_mb >= 10:
            return 80 + (size_mb - 10) / 20 * 20
        else:
            return size_mb / 10 * 80
    else:  # JPEG, PNG, etc.
        # Compressed files: 2-10 MB is typical
        if size_mb >= 10:
            return 100.0
        elif size_mb >= 2:
            return 70 + (size_mb - 2) / 8 * 30
        elif size_mb >= 0.5:
            return 40 + (size_mb - 0.5) / 1.5 * 30
        else:
            return size_mb / 0.5 * 40


def _score_metadata_completeness(metadata: ImageMetadata) -> float:
    """
    Score based on completeness of EXIF metadata.

    More complete metadata indicates better preservation of camera data.

    Args:
        metadata: Image metadata

    Returns:
        Metadata completeness score (0-100)
    """
    score = 0.0
    total_possible = 0.0

    # Check for important EXIF fields
    checks = [
        (metadata.camera_make is not None, 15),
        (metadata.camera_model is not None, 15),
        (metadata.lens_model is not None, 10),
        (metadata.focal_length is not None, 10),
        (metadata.aperture is not None, 10),
        (metadata.shutter_speed is not None, 10),
        (metadata.iso is not None, 10),
        (metadata.gps_latitude is not None, 10),
        (metadata.gps_longitude is not None, 10),
    ]

    for has_field, weight in checks:
        total_possible += weight
        if has_field:
            score += weight

    if total_possible == 0:
        return 0.0

    return (score / total_possible) * 100


def compare_quality(
    metadata1: ImageMetadata,
    file_type1: FileType,
    metadata2: ImageMetadata,
    file_type2: FileType,
) -> int:
    """
    Compare quality of two images.

    Args:
        metadata1: First image metadata
        file_type1: First image file type
        metadata2: Second image metadata
        file_type2: Second image file type

    Returns:
        -1 if image1 is better, 1 if image2 is better, 0 if equal
    """
    score1 = calculate_quality_score(metadata1, file_type1)
    score2 = calculate_quality_score(metadata2, file_type2)

    if score1.overall > score2.overall:
        return -1
    elif score1.overall < score2.overall:
        return 1
    else:
        return 0


def select_best(
    images: Dict[str, tuple[ImageMetadata, FileType]],
) -> tuple[str, QualityScore]:
    """
    Select the best image from a group based on quality scores.

    Args:
        images: Dict mapping image_id to (metadata, file_type) tuple

    Returns:
        Tuple of (best_image_id, quality_score)
    """
    if not images:
        raise ValueError("Cannot select best from empty image list")

    best_id = None
    best_score = None

    for image_id, (metadata, file_type) in images.items():
        score = calculate_quality_score(metadata, file_type)

        if best_score is None or score.overall > best_score.overall:
            best_id = image_id
            best_score = score

    if best_id is None or best_score is None:
        raise ValueError("Failed to select best image")

    return (best_id, best_score)
