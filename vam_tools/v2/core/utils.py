"""
Utility functions for catalog operations.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compute_checksum(file_path: Path, algorithm: str = "sha256") -> Optional[str]:
    """
    Compute cryptographic checksum of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
        Hexadecimal checksum string, or None on error
    """
    try:
        hash_obj = hashlib.new(algorithm)

        with open(file_path, "rb") as f:
            # Read in chunks for memory efficiency
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error computing checksum for {file_path}: {e}")
        return None


def verify_checksum(
    file_path: Path, expected_checksum: str, algorithm: str = "sha256"
) -> bool:
    """
    Verify that a file's checksum matches expected value.

    Args:
        file_path: Path to the file
        expected_checksum: Expected checksum value
        algorithm: Hash algorithm

    Returns:
        True if checksums match, False otherwise
    """
    actual_checksum = compute_checksum(file_path, algorithm)
    if actual_checksum is None:
        return False

    return actual_checksum.lower() == expected_checksum.lower()


def format_bytes(size_bytes: int) -> str:
    """
    Format bytes in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "2.5 GB")
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


def safe_filename(name: str) -> str:
    """
    Convert a string to a safe filename.

    Args:
        name: Input string

    Returns:
        Safe filename string
    """
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        name = name.replace(char, "_")

    # Remove leading/trailing spaces and dots
    name = name.strip(". ")

    # Limit length
    if len(name) > 255:
        name = name[:255]

    return name or "unnamed"


def is_image_file(file_path: Path) -> bool:
    """
    Check if a file is an image based on extension.

    Args:
        file_path: Path to check

    Returns:
        True if image file, False otherwise
    """
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".heic",
        ".heif",
        ".raw",
        ".cr2",
        ".nef",
        ".arw",
        ".dng",
        ".orf",
        ".rw2",
        ".pef",
        ".sr2",
        ".raf",
    }
    return file_path.suffix.lower() in image_extensions


def is_video_file(file_path: Path) -> bool:
    """
    Check if a file is a video based on extension.

    Args:
        file_path: Path to check

    Returns:
        True if video file, False otherwise
    """
    video_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
        ".3gp",
        ".mts",
        ".m2ts",
    }
    return file_path.suffix.lower() in video_extensions


def get_file_type(file_path: Path) -> str:
    """
    Determine file type (image, video, or unknown).

    Args:
        file_path: Path to check

    Returns:
        "image", "video", or "unknown"
    """
    if is_image_file(file_path):
        return "image"
    elif is_video_file(file_path):
        return "video"
    else:
        return "unknown"
