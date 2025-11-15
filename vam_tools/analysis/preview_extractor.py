"""
Preview extraction for RAW and HEIC/TIFF files.

Extracts embedded previews from RAW files and converts HEIC/TIFF to JPEG,
storing them in a disk cache for fast web UI access.
"""

import io
import logging
import multiprocessing as mp
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from ..db import CatalogDB as CatalogDatabase
from ..core.types import FileType, ImageMetadata, ImageRecord, ImageStatus
from ..shared.preview_cache import PreviewCache

logger = logging.getLogger(__name__)

# Register HEIC support for Pillow
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    logger.debug("HEIC support registered for preview extraction")
except ImportError:
    logger.warning("pillow-heif not installed, HEIC preview extraction will fail")

# Formats that need preview extraction/conversion
RAW_FORMATS = {
    ".arw",
    ".cr2",
    ".cr3",
    ".nef",
    ".dng",
    ".orf",
    ".rw2",
    ".pef",
    ".sr2",
    ".raf",
    ".raw",
}

HEIC_TIFF_FORMATS = {".heic", ".heif", ".tif", ".tiff"}


def _extract_preview_worker(
    args: Tuple[ImageRecord, Path],
) -> Tuple[str, Optional[bytes], Optional[str]]:
    """
    Worker function for parallel preview extraction.

    Args:
        args: Tuple of (image_record, catalog_path)

    Returns:
        Tuple of (image_id, preview_bytes, error_message)
    """
    image, catalog_path = args
    image_id = image.id
    file_path = Path(image.source_path)
    file_ext = file_path.suffix.lower()

    try:
        # Handle RAW files with exiftool
        if file_ext in RAW_FORMATS:
            try:
                # Try PreviewImage first
                result = subprocess.run(
                    ["exiftool", "-b", "-PreviewImage", str(file_path)],
                    capture_output=True,
                    timeout=30,
                )

                if result.returncode == 0 and len(result.stdout) > 0:
                    return (image_id, result.stdout, None)

                # Fallback to JpgFromRaw
                result = subprocess.run(
                    ["exiftool", "-b", "-JpgFromRaw", str(file_path)],
                    capture_output=True,
                    timeout=30,
                )

                if result.returncode == 0 and len(result.stdout) > 0:
                    return (image_id, result.stdout, None)

                return (image_id, None, "No embedded preview found")

            except subprocess.TimeoutExpired:
                return (image_id, None, "Timeout extracting preview (30s)")
            except FileNotFoundError:
                return (image_id, None, "ExifTool not installed")
            except Exception as e:
                return (image_id, None, f"RAW extraction error: {e}")

        # Handle HEIC/TIFF with Pillow
        elif file_ext in HEIC_TIFF_FORMATS:
            try:
                with Image.open(file_path) as img:
                    # Convert to RGB if needed
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")

                    # Save to bytes buffer as JPEG
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=85)
                    preview_bytes = buffer.getvalue()

                    return (image_id, preview_bytes, None)

            except Exception as e:
                return (image_id, None, f"Pillow conversion error: {e}")

        else:
            # Unknown format
            return (image_id, None, f"Unsupported format: {file_ext}")

    except Exception as e:
        logger.error(f"Unexpected error extracting preview for {file_path}: {e}")
        return (image_id, None, f"Unexpected error: {e}")


class PreviewExtractor:
    """Extract and cache previews for RAW and HEIC/TIFF files."""

    def __init__(self, catalog: CatalogDatabase, workers: Optional[int] = None):
        """
        Initialize the preview extractor.

        Args:
            catalog: Catalog database
            workers: Number of worker processes (default: CPU count)
        """
        self.catalog = catalog
        self.workers = workers or mp.cpu_count()
        self.preview_cache = PreviewCache(catalog.catalog_path)

    def extract_previews(self, force: bool = False) -> None:
        """
        Extract and cache previews for all RAW/HEIC/TIFF files.

        Args:
            force: If True, re-extract even if already cached
        """
        # Get all images from catalog
        rows = self.catalog.execute("SELECT * FROM images").fetchall()
        all_images: List[ImageRecord] = []
        for row in rows:
            # Manually construct ImageRecord from row (simplified)
            image = ImageRecord(
                id=row["id"],
                source_path=Path(row["source_path"]),
                file_type=(
                    FileType.IMAGE
                    if row["format"]
                    in ["JPEG", "PNG", "GIF", "BMP", "WEBP", "TIFF", "HEIC"]
                    else FileType.VIDEO
                ),
                checksum=row["file_hash"],
                status=ImageStatus.COMPLETE,
                file_size=row["file_size"],
                metadata=ImageMetadata(
                    format=row["format"],
                ),
            )
            all_images.append(image)

        # Filter to only images that need preview extraction
        images_to_process: List[ImageRecord] = []
        for image in all_images:
            file_ext = Path(image.source_path).suffix.lower()

            # Only process RAW and HEIC/TIFF files
            if file_ext not in (RAW_FORMATS | HEIC_TIFF_FORMATS):
                continue

            # Skip if already cached (unless force=True)
            if not force and self.preview_cache.has_preview(image.id):
                continue

            # Only process images (not videos)
            if image.file_type == FileType.IMAGE:
                images_to_process.append(image)

        if not images_to_process:
            logger.info("No previews to extract (all already cached)")
            return

        logger.info(
            f"Extracting previews for {len(images_to_process)} files "
            f"with {self.workers} workers"
        )

        # Track statistics
        extracted_count = 0
        failed_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"Extracting previews ({self.workers} workers)...",
                total=len(images_to_process),
            )

            # Prepare worker arguments
            worker_args = [
                (image, self.catalog.catalog_path) for image in images_to_process
            ]

            # Process in parallel
            with mp.Pool(processes=self.workers) as pool:
                chunk_size = max(1, len(images_to_process) // (self.workers * 4))

                for image_id, preview_bytes, error_msg in pool.imap_unordered(
                    _extract_preview_worker, worker_args, chunk_size
                ):
                    if preview_bytes:
                        # Store in cache
                        if self.preview_cache.store_preview(image_id, preview_bytes):
                            extracted_count += 1
                        else:
                            failed_count += 1
                            logger.warning(f"Failed to cache preview for {image_id}")
                    else:
                        failed_count += 1
                        if error_msg:
                            logger.debug(
                                f"Failed to extract preview for {image_id}: {error_msg}"
                            )

                    progress.advance(task)

        # Log summary
        logger.info(
            f"Preview extraction complete: {extracted_count} extracted, "
            f"{failed_count} failed"
        )

        # Show cache stats
        cache_stats = self.preview_cache.get_cache_stats()
        logger.info(
            f"Preview cache: {cache_stats['num_previews']} previews, "
            f"{cache_stats['total_size_gb']:.2f} GB / {cache_stats['max_size_gb']:.2f} GB "
            f"({cache_stats['usage_percent']:.1f}% full)"
        )
