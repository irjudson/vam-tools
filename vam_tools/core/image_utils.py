"""
Shared utilities for image processing.

This module provides common functionality used across all lightroom tools,
including image file detection, metadata extraction, and utility functions.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from PIL import Image

# Supported image file extensions
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
}

logger = logging.getLogger(__name__)


def is_image_file(file_path: Path) -> bool:
    """
    Check if a file is an image based on its extension.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file has a supported image extension, False otherwise
    """
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


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


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string like "2.5 MB" or "1.2 GB"
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)

    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1

    return f"{size:.1f} {size_names[i]}"


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
            for root, _, files in os.walk(
                directory, followlinks=follow_symlinks
            ):
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
