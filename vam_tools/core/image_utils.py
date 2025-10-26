"""
Shared utilities for image processing.

NOTE: This module now re-exports functions from vam_tools.shared for backward
compatibility. New code should import directly from vam_tools.shared instead.
"""

# Re-export shared utilities for backward compatibility
from vam_tools.shared import (
    IMAGE_EXTENSIONS,
    collect_image_files,
    format_bytes as format_file_size,  # V1 used this name
    get_image_info,
    is_image_file,
    setup_logging,
)

__all__ = [
    "IMAGE_EXTENSIONS",
    "is_image_file",
    "get_image_info",
    "format_file_size",
    "collect_image_files",
    "setup_logging",
]
