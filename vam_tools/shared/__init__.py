"""
Shared utilities for VAM Tools.

This module provides common functionality used across V1 (legacy tools)
and V2 (catalog system), eliminating code duplication.
"""

from .media_utils import (
    # File type detection
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    is_image_file,
    is_video_file,
    get_file_type,
    # File operations
    compute_checksum,
    verify_checksum,
    format_bytes,
    safe_filename,
    # Image operations
    get_image_info,
    collect_image_files,
    # Logging
    setup_logging,
)

__all__ = [
    # Constants
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    # Functions
    "is_image_file",
    "is_video_file",
    "get_file_type",
    "compute_checksum",
    "verify_checksum",
    "format_bytes",
    "safe_filename",
    "get_image_info",
    "collect_image_files",
    "setup_logging",
]
