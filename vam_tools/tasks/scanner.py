"""Scanner implementation for analyzing directories."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import text

from ..db.catalog_schema import get_image_count
from ..db.connection import SessionLocal
from ..shared.media_utils import get_file_type, is_image_file, is_video_file

logger = logging.getLogger(__name__)


def compute_checksum(file_path: Path) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex string of SHA256 checksum
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def scan_directory(
    directory: Path, catalog_id: str, progress_callback: Optional[callable] = None
) -> Dict[str, int]:
    """
    Scan a directory and add images to catalog.

    Args:
        directory: Directory to scan
        catalog_id: Catalog UUID
        progress_callback: Optional callback(current, total, message)

    Returns:
        Statistics dictionary
    """
    stats = {
        "files_found": 0,
        "files_added": 0,
        "files_skipped": 0,
        "exact_duplicates": 0,
    }

    # Find all media files
    media_files = []
    for ext in [".jpg", ".jpeg", ".png", ".heic", ".mp4", ".mov"]:
        media_files.extend(directory.rglob(f"*{ext}"))
        media_files.extend(directory.rglob(f"*{ext.upper()}"))

    stats["files_found"] = len(media_files)
    logger.info(f"Found {len(media_files)} media files in {directory}")

    db = SessionLocal()
    try:
        for idx, file_path in enumerate(media_files):
            try:
                # Progress callback
                if progress_callback and idx % 10 == 0:
                    progress_callback(
                        idx, len(media_files), f"Processing {file_path.name}"
                    )

                # Get file info
                file_type = get_file_type(file_path)
                if file_type == "unknown":
                    stats["files_skipped"] += 1
                    continue

                # Compute checksum
                checksum = compute_checksum(file_path)

                # Check if already exists in this catalog
                existing = db.execute(
                    text(
                        "SELECT id FROM images WHERE catalog_id = :catalog_id AND checksum = :checksum"
                    ),
                    {"catalog_id": catalog_id, "checksum": checksum},
                ).fetchone()

                if existing:
                    stats["exact_duplicates"] += 1
                    continue

                # Get file size and dates
                stat = file_path.stat()
                size_bytes = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime)

                # Simple date extraction (just filesystem for now)
                dates = {
                    "filesystem_date": mtime.isoformat(),
                    "selected_date": mtime.isoformat(),
                    "source": "filesystem",
                    "confidence": 30,
                }

                # Basic metadata
                metadata = {
                    "size_bytes": size_bytes,
                    "filename": file_path.name,
                }

                # Insert into database
                db.execute(
                    text(
                        """
                        INSERT INTO images
                        (id, catalog_id, source_path, file_type, checksum, size_bytes, dates, metadata, status)
                        VALUES (:id, :catalog_id, :path, :type, :checksum, :size, :dates, :metadata, :status)
                    """
                    ),
                    {
                        "id": checksum,
                        "catalog_id": catalog_id,
                        "path": str(file_path.absolute()),
                        "type": file_type,
                        "checksum": checksum,
                        "size": size_bytes,
                        "dates": json.dumps(dates),
                        "metadata": json.dumps(metadata),
                        "status": "complete",
                    },
                )

                stats["files_added"] += 1

                # Commit every 100 files
                if stats["files_added"] % 100 == 0:
                    db.commit()
                    logger.info(f"Committed {stats['files_added']} files")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                stats["files_skipped"] += 1
                continue

        # Final commit
        db.commit()
        logger.info(f"Scan complete: {stats}")

    finally:
        db.close()

    return stats
