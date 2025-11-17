"""
Filesystem watcher service for automatic photo detection.

Monitors configured directories and automatically processes new/modified
image and video files, adding them to the catalog.
"""

import logging
import time
from pathlib import Path
from threading import Thread
from typing import Optional, Set

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class PhotoWatcherHandler(FileSystemEventHandler):
    """
    Handles filesystem events for photo directories.

    Automatically processes new and modified image/video files.
    """

    def __init__(self, catalog_path: Path, debounce_seconds: float = 2.0):
        """
        Initialize photo watcher handler.

        Args:
            catalog_path: Path to catalog directory
            debounce_seconds: Seconds to wait before processing (avoid rapid-fire events)
        """
        super().__init__()
        self.catalog_path = catalog_path
        self.debounce_seconds = debounce_seconds
        self.pending_files: Set[Path] = set()
        self.processing = False

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory:
            file_path = Path(event.src_path)
            self._queue_file(file_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory:
            file_path = Path(event.src_path)
            # Only process if it's a new file (not already in catalog)
            self._queue_file(file_path)

    def _queue_file(self, file_path: Path) -> None:
        """Queue a file for processing."""
        # Skip hidden files and temp files
        if file_path.name.startswith(".") or "@" in str(file_path):
            return

        # Check if it's a supported file type
        from ..shared.media_utils import get_file_type

        file_type = get_file_type(file_path)

        if file_type not in ("image", "video"):
            return

        logger.info(f"Detected new file: {file_path}")
        self.pending_files.add(file_path)

        # Start processing thread if not already running
        if not self.processing:
            thread = Thread(target=self._process_pending, daemon=True)
            thread.start()

    def _process_pending(self) -> None:
        """Process all pending files after debounce period."""
        self.processing = True

        try:
            # Wait for debounce period
            time.sleep(self.debounce_seconds)

            if not self.pending_files:
                return

            # Get files to process
            files_to_process = list(self.pending_files)
            self.pending_files.clear()

            logger.info(f"Processing {len(files_to_process)} new file(s)")

            # Process files
            from ..analysis.scanner import _process_file_worker
            from ..core.types import FileType, Statistics  # Import Statistics
            from ..db import CatalogDB as CatalogDatabase

            processed = 0
            added = 0

            with CatalogDatabase(self.catalog_path) as db:
                # Fetch latest statistics or initialize new
                latest_stats_row = db.execute(
                    "SELECT * FROM statistics ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if latest_stats_row:
                    stats = Statistics(**latest_stats_row)
                else:
                    stats = Statistics()  # Initialize with default values

                for file_path in files_to_process:
                    try:
                        # Check if file exists and is readable
                        if not file_path.exists():
                            logger.warning(f"File no longer exists: {file_path}")
                            continue

                        # Process file
                        result = _process_file_worker(file_path)

                        if result is not None:
                            image, file_size = result
                            processed += 1

                            # Check if already in catalog
                            existing_image = db.execute(
                                "SELECT id FROM images WHERE id = ?", (image.checksum,)
                            ).fetchone()

                            if not existing_image:
                                # Add to catalog (using SQL INSERT)
                                db.execute(
                                    """
                                    INSERT INTO images (
                                        id, source_path, file_size, file_hash, format,
                                        width, height, created_at, modified_at, indexed_at,
                                        date_taken, camera_make, camera_model, lens_model,
                                        focal_length, aperture, shutter_speed, iso,
                                        gps_latitude, gps_longitude, quality_score, is_corrupted,
                                        perceptual_hash, features_vector
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        image.id,
                                        str(image.source_path),
                                        image.file_size,
                                        image.checksum,
                                        image.metadata.format,
                                        image.metadata.width,
                                        image.metadata.height,
                                        (
                                            image.created_at.isoformat()
                                            if image.created_at
                                            else None
                                        ),
                                        (
                                            image.modified_at.isoformat()
                                            if image.modified_at
                                            else None
                                        ),
                                        datetime.now().isoformat(),  # indexed_at
                                        (
                                            image.dates.selected_date.isoformat()
                                            if image.dates and image.dates.selected_date
                                            else None
                                        ),
                                        image.metadata.exif.get("Make"),
                                        image.metadata.exif.get("Model"),
                                        image.metadata.exif.get("LensModel"),
                                        image.metadata.exif.get("FocalLength"),
                                        image.metadata.exif.get("ApertureValue"),
                                        image.metadata.exif.get("ExposureTime"),
                                        image.metadata.exif.get("ISO"),
                                        (
                                            image.metadata.gps.latitude
                                            if image.metadata.gps
                                            else None
                                        ),
                                        (
                                            image.metadata.gps.longitude
                                            if image.metadata.gps
                                            else None
                                        ),
                                        image.quality_score,
                                        1 if image.is_corrupted else 0,
                                        (
                                            image.hashes.perceptual_hash
                                            if image.hashes
                                            else None
                                        ),
                                        None,  # features_vector not handled yet
                                    ),
                                )
                                added += 1

                                # Update statistics in memory
                                if image.file_type == FileType.IMAGE:
                                    stats.total_images += 1
                                elif image.file_type == FileType.VIDEO:
                                    stats.total_videos += 1
                                stats.total_size_bytes += image.file_size
                                stats.images_scanned += 1
                                stats.images_hashed += (
                                    1  # Assuming hashing happens during scan
                                )

                                logger.info(f"Added to catalog: {file_path.name}")
                            else:
                                logger.debug(f"Already in catalog: {file_path.name}")

                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")

                # Save catalog with updated stats
                if added > 0:
                    db.execute(
                        """
                        INSERT INTO statistics (
                            timestamp, total_images, total_size_bytes, images_scanned, images_hashed
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            datetime.now().isoformat(),
                            stats.total_images,
                            stats.total_size_bytes,
                            stats.images_scanned,
                            stats.images_hashed,
                        ),
                    )
                    logger.info(f"Auto-scan complete: {added} file(s) added to catalog")

        finally:
            self.processing = False


class PhotoWatcherService:
    """
    Background service for watching photo directories.

    Monitors configured directories and automatically processes new photos.
    """

    def __init__(self, catalog_path: Path, watch_directories: list[Path]):
        """
        Initialize photo watcher service.

        Args:
            catalog_path: Path to catalog directory
            watch_directories: List of directories to watch
        """
        self.catalog_path = catalog_path
        self.watch_directories = watch_directories
        self.observer: Optional[Observer] = None
        self.handler: Optional[PhotoWatcherHandler] = None

    def start(self) -> None:
        """Start watching directories."""
        if self.observer is not None:
            logger.warning("Watcher already running")
            return

        if not self.watch_directories:
            logger.info("No directories configured for watching")
            return

        logger.info(
            f"Starting photo watcher for {len(self.watch_directories)} director(ies)"
        )

        self.observer = Observer()
        self.handler = PhotoWatcherHandler(self.catalog_path)

        # Add watches for each directory
        for directory in self.watch_directories:
            if directory.exists() and directory.is_dir():
                self.observer.schedule(self.handler, str(directory), recursive=True)
                logger.info(f"Watching: {directory}")
            else:
                logger.warning(f"Directory does not exist: {directory}")

        # Start observer
        self.observer.start()
        logger.info("Photo watcher started")

    def stop(self) -> None:
        """Stop watching directories."""
        if self.observer is None:
            return

        logger.info("Stopping photo watcher")
        self.observer.stop()
        self.observer.join()
        self.observer = None
        self.handler = None
        logger.info("Photo watcher stopped")

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self.observer is not None and self.observer.is_alive()
