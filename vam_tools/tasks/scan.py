"""Scanning and metadata extraction tasks."""

import logging
import time
from pathlib import Path
from typing import Dict, List

from sqlalchemy import text

from ..celery_app import app
from ..db.connection import SessionLocal
from ..db.models import Catalog
from .base import ProgressTrackingTask
from .scanner import scan_directory

logger = logging.getLogger(__name__)


@app.task(base=ProgressTrackingTask, bind=True)
def scan_directories(self, catalog_id: str, directories: List[str]) -> Dict[str, int]:
    """
    Scan directories for images and videos.

    Args:
        catalog_id: Catalog UUID
        directories: List of directory paths to scan

    Returns:
        Dictionary with scan statistics
    """
    start_time = time.time()
    logger.info(f"Starting scan for catalog {catalog_id}")
    logger.info(f"Directories: {directories}")

    # Verify catalog exists
    db = SessionLocal()
    try:
        catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
        if not catalog:
            raise ValueError(f"Catalog {catalog_id} not found")
    finally:
        db.close()

    # Scan all directories
    total_stats = {
        "files_found": 0,
        "files_added": 0,
        "files_skipped": 0,
        "exact_duplicates": 0,
    }

    for idx, directory in enumerate(directories):
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            continue

        logger.info(f"Scanning directory {idx+1}/{len(directories)}: {directory}")

        # Progress callback
        def progress_cb(current, total, message):
            overall_progress = (idx * 100 + (current / total * 100)) / len(directories)
            self.update_progress(
                int(overall_progress),
                100,
                f"Dir {idx+1}/{len(directories)}: {message}",
            )

        # Scan directory
        stats = scan_directory(dir_path, catalog_id, progress_callback=progress_cb)

        # Aggregate stats
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    duration = time.time() - start_time
    total_stats["duration_seconds"] = round(duration, 2)

    logger.info(f"Scan complete: {total_stats}")
    return total_stats
