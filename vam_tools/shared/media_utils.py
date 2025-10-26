"""
Media file utilities for VAM Tools.

Consolidated utilities from V1 and V2, providing a single source of truth
for file type detection, checksums, formatting, and image operations.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Supported file extensions (consolidated from V1 and V2)
IMAGE_EXTENSIONS: Set[str] = {
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

VIDEO_EXTENSIONS: Set[str] = {
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


# File type detection (from V2)
def is_image_file(file_path: Path) -> bool:
    """
    Check if a file is an image based on extension.

    Args:
        file_path: Path to check

    Returns:
        True if image file, False otherwise
    """
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(file_path: Path) -> bool:
    """
    Check if a file is a video based on extension.

    Args:
        file_path: Path to check

    Returns:
        True if video file, False otherwise
    """
    return file_path.suffix.lower() in VIDEO_EXTENSIONS


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


# Checksum operations (from V2 - better implementation with chunking)
def compute_checksum(file_path: Path, algorithm: str = "sha256") -> Optional[str]:
    """
    Compute cryptographic checksum of a file.

    Uses chunked reading for memory efficiency (better than V1's whole-file read).

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


# Formatting utilities (from V2)
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


# Image operations (from V1 - uses PIL)
def get_image_info(image_path: Path) -> Optional[Dict[str, any]]:
    """
    Extract basic information from an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing image information:
        - dimensions: Tuple of (width, height)
        - format: Image format (JPEG, PNG, etc.)
        - mode: Color mode (RGB, RGBA, L, etc.)
        - file_size: File size in bytes

        Returns None if the image cannot be opened or processed.
    """
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return {
                "dimensions": img.size,
                "format": img.format,
                "mode": img.mode,
                "file_size": os.path.getsize(image_path),
            }
    except Exception as e:
        logger.debug(f"Failed to get info for {image_path}: {e}")
        return None


def collect_image_files(
    directory: Path, recursive: bool = True, follow_symlinks: bool = False
) -> List[Path]:
    """
    Collect all image files from a directory.

    Args:
        directory: Directory to scan
        recursive: If True, scan subdirectories recursively
        follow_symlinks: If True, follow symbolic links

    Returns:
        List of Path objects pointing to image files
    """
    image_files: List[Path] = []

    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return image_files

    if not directory.is_dir():
        logger.error(f"Path is not a directory: {directory}")
        return image_files

    try:
        if recursive:
            for root, _, files in os.walk(directory, followlinks=follow_symlinks):
                root_path = Path(root)
                for file in files:
                    file_path = root_path / file
                    if is_image_file(file_path):
                        image_files.append(file_path)
        else:
            for file_path in directory.iterdir():
                if file_path.is_file() and is_image_file(file_path):
                    image_files.append(file_path)

        logger.info(f"Found {len(image_files)} image files in {directory}")
    except PermissionError as e:
        logger.error(f"Permission denied accessing {directory}: {e}")
    except Exception as e:
        logger.error(f"Error collecting image files from {directory}: {e}")

    return image_files


# Logging setup (from V1)
def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        verbose: If True, set logging level to DEBUG
        quiet: If True, set logging level to WARNING
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
