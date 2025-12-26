"""Helper functions for file reorganization."""

import logging
from pathlib import Path
from typing import Optional

from ..core.types import ImageRecord
from ..shared.media_utils import compute_checksum

logger = logging.getLogger(__name__)


def should_reorganize_image(
    image: ImageRecord, output_directory: Path, target_path: Optional[Path] = None
) -> bool:
    """Determine if an image should be reorganized.

    Args:
        image: Image record
        output_directory: Target organization directory
        target_path: Calculated target path (optional, for checksum check)

    Returns:
        True if image should be reorganized, False if should be skipped
    """
    # Skip if source_path already in organized structure
    if str(image.source_path).startswith(str(output_directory)):
        logger.debug(f"Skipping {image.id}: already in organized structure")
        return False

    # Skip if target exists with matching checksum
    if target_path and target_path.exists():
        target_checksum = compute_checksum(target_path)
        if target_checksum == image.checksum:
            logger.debug(f"Skipping {image.id}: already organized (checksum match)")
            return False

    return True
