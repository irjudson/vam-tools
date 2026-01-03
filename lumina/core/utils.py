"""
Utility functions for catalog operations.

NOTE: This module now re-exports functions from lumina.shared for backward
compatibility. New code should import directly from lumina.shared instead.
"""

# Re-export shared utilities for backward compatibility
from lumina.shared import (
    compute_checksum,
    format_bytes,
    get_file_type,
    is_image_file,
    is_video_file,
    safe_filename,
    verify_checksum,
)

__all__ = [
    "compute_checksum",
    "verify_checksum",
    "format_bytes",
    "safe_filename",
    "is_image_file",
    "is_video_file",
    "get_file_type",
]
