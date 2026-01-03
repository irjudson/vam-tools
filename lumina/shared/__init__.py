"""
Shared utilities for Lumina.

This module provides common functionality used across V1 (legacy tools)
and V2 (catalog system), eliminating code duplication.
"""

from .media_utils import (  # File type detection; File operations; Image operations; Logging
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    collect_image_files,
    compute_checksum,
    format_bytes,
    get_file_type,
    get_image_info,
    is_image_file,
    is_video_file,
    safe_filename,
    setup_logging,
    verify_checksum,
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
