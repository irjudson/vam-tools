"""
Image scanner using SQLAlchemy ORM (PostgreSQL).

This replaces the SQLite-based scanner with a proper ORM-based implementation.
"""

import hashlib
import logging
import multiprocessing as mp
import os
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from sqlalchemy.orm import Session

from vam_tools.shared import compute_checksum, get_file_type
from vam_tools.shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path

from ..core.performance_stats import PerformanceTracker
from ..core.types import CatalogPhase, FileType, ImageRecord, ImageStatus
from ..db.models import Config, Image, Statistics
from .metadata import MetadataExtractor

logger = logging.getLogger(__name__)


def _process_file_worker(file_path: Path) -> Optional[Tuple[ImageRecord, int]]:
    """
    Worker function for parallel file processing.

    Args:
        file_path: Path to the file to process

    Returns:
        Tuple of (ImageRecord, file_size) if successful, None if skipped/failed
    """
    try:
        # Determine file type
        file_type_str = get_file_type(file_path)
        if file_type_str == "image":
            file_type = FileType.IMAGE
        elif file_type_str == "video":
            file_type = FileType.VIDEO
        else:
            return None  # Skip unknown files

        # Compute checksum
        checksum = compute_checksum(file_path)
        if not checksum:
            logger.warning(f"Failed to compute checksum for {file_path}")
            return None

        # Extract metadata (each worker needs its own extractor)
        with MetadataExtractor() as extractor:
            metadata = extractor.extract_metadata(file_path, file_type)
            dates = extractor.extract_dates(file_path, metadata)

        # Create image record
        # Note: id will be set later by the scanner to include catalog_id
        image = ImageRecord(
            id=checksum,  # Temporary - will be updated by scanner
            source_path=file_path,
            file_type=file_type,
            checksum=checksum,
            dates=dates,
            metadata=metadata,
            status=ImageStatus.ANALYZING,
        )

        return (image, metadata.size_bytes or 0)

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


class ImageScannerORM:
    """
    Scans directories for images and videos using SQLAlchemy ORM.

    This replaces the raw SQL implementation with proper ORM usage.
    """

    def __init__(
        self,
        session: Session,
        catalog_id: str,
        catalog_path: Path,
        workers: int = 4,
        perf_tracker: Optional[PerformanceTracker] = None,
    ):
        """
        Initialize the scanner with SQLAlchemy session.

        Args:
            session: SQLAlchemy session
            catalog_id: Catalog UUID
            catalog_path: Path to catalog directory (for thumbnails)
            workers: Number of parallel workers for processing
            perf_tracker: Optional performance tracker
        """
        self.session = session
        self.catalog_id = catalog_id
        self.catalog_path = (
            Path(catalog_path) if not isinstance(catalog_path, Path) else catalog_path
        )
        self.workers = workers
        self.perf_tracker = perf_tracker

        # Track scanning statistics
        self.files_added = 0
        self.files_skipped = 0
        self.files_error = 0
        self.total_bytes = 0
        self.start_time = None
        self.end_time = None

    def scan_directories(self, directories: List[Path]) -> None:
        """
        Scan directories for images and videos using incremental discovery.

        Args:
            directories: List of directories to scan
        """
        logger.info(f"Scanning {len(directories)} directories (ORM mode)")
        print(f"DEBUG: scan_directories called with {directories}", flush=True)
        self.start_time = time.time()

        # Update catalog phase in config
        print("DEBUG: About to call _update_config", flush=True)
        self._update_config("phase", CatalogPhase.ANALYZING.value)
        print("DEBUG: _update_config completed", flush=True)

        # Track file collection
        ctx = (
            self.perf_tracker.track_operation(
                "scan_directories", items=len(directories)
            )
            if self.perf_tracker
            else nullcontext()
        )

        with ctx:
            # Process files incrementally as they're discovered
            batch_size = 100
            current_batch = []
            files_discovered = 0

            logger.info("Starting incremental file discovery and processing...")

            for directory in directories:
                logger.info(f"Scanning directory: {directory}")

                # Discover and process files incrementally
                for file_path in self._discover_files_incrementally(Path(directory)):
                    current_batch.append(file_path)
                    files_discovered += 1

                    # Process batch when it reaches batch_size
                    if len(current_batch) >= batch_size:
                        logger.info(
                            f"Processing batch of {len(current_batch)} files "
                            f"({files_discovered} discovered so far)"
                        )
                        self._process_files(current_batch)
                        current_batch = []

            # Process remaining files
            if current_batch:
                logger.info(f"Processing final batch of {len(current_batch)} files")
                self._process_files(current_batch)

        self.end_time = time.time()
        self._update_statistics()

        # Log summary
        logger.info(
            f"Scan complete: {self.files_added} added, "
            f"{self.files_skipped} skipped, {self.files_error} errors"
        )

    def _discover_files_incrementally(self, directory: Path) -> Iterator[Path]:
        """
        Discover files incrementally without blocking.

        Args:
            directory: Directory to scan

        Yields:
            Paths to discovered image/video files
        """
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return

        # Walk directory tree
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)

            # Skip Synology metadata directories
            dirs[:] = [d for d in dirs if d != "@eaDir"]

            for file in files:
                # Skip hidden files
                if file.startswith("."):
                    continue

                file_path = root_path / file

                # Quick check based on extension
                ext = file_path.suffix.lower()
                if ext in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".bmp",
                    ".tiff",
                    ".webp",
                    ".heic",
                    ".raw",
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".wmv",
                    ".mkv",
                }:
                    yield file_path

    def _process_files(self, file_paths: List[Path]) -> None:
        """
        Process a batch of files in parallel.

        Args:
            file_paths: List of file paths to process
        """
        if self.workers > 1:
            # Process files in parallel
            with mp.Pool(processes=self.workers) as pool:
                results = pool.map(_process_file_worker, file_paths)
        else:
            # Process files sequentially
            results = [_process_file_worker(f) for f in file_paths]

        # Track checksums in this batch to avoid duplicates within the batch
        batch_checksums = set()

        # Add results to database
        for result, file_path in zip(results, file_paths):
            if result is None:
                self.files_error += 1
                continue

            image_record, file_size = result
            self.total_bytes += file_size

            # Check if image already exists by checksum (in database or in this batch)
            if image_record.checksum in batch_checksums:
                logger.debug(f"Skipping duplicate in batch: {file_path}")
                self.files_skipped += 1
                continue

            existing = (
                self.session.query(Image)
                .filter_by(catalog_id=self.catalog_id, checksum=image_record.checksum)
                .first()
            )

            if existing:
                logger.debug(f"Skipping duplicate: {file_path}")
                self.files_skipped += 1
                continue

            # Track this checksum for the current batch
            batch_checksums.add(image_record.checksum)

            # Generate thumbnail if needed
            thumbnail_path = None
            if image_record.file_type == FileType.IMAGE:
                try:
                    thumbnail_full_path = get_thumbnail_path(
                        image_record.checksum,
                        self.catalog_path / "thumbnails",
                    )

                    if generate_thumbnail(file_path, thumbnail_full_path):
                        # Store relative path from catalog root
                        thumbnail_path = str(
                            thumbnail_full_path.relative_to(self.catalog_path)
                        )
                except Exception as e:
                    logger.warning(f"Failed to generate thumbnail for {file_path}: {e}")

            # Generate unique ID per catalog (catalog_id + checksum hash)
            unique_id = hashlib.sha256(
                f"{self.catalog_id}:{image_record.checksum}".encode()
            ).hexdigest()

            # Create ORM object
            image = Image(
                id=unique_id,
                catalog_id=self.catalog_id,
                source_path=str(image_record.source_path),
                file_type=image_record.file_type.value,
                checksum=image_record.checksum,
                size_bytes=(
                    image_record.metadata.size_bytes if image_record.metadata else None
                ),
                dates=(
                    image_record.dates.model_dump(mode="json")
                    if image_record.dates
                    else {}
                ),
                metadata_json=(
                    image_record.metadata.model_dump(mode="json")
                    if image_record.metadata
                    else {}
                ),
                thumbnail_path=thumbnail_path,
                status=image_record.status.value,
            )

            self.session.add(image)
            self.files_added += 1
            logger.debug(f"Added: {file_path}")

        # Commit batch
        try:
            self.session.commit()
        except Exception as e:
            logger.error(f"Failed to commit batch: {e}")
            self.session.rollback()
            self.files_error += len(file_paths)

    def _update_config(self, key: str, value: any) -> None:
        """
        Update or insert configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        print(f"DEBUG: _update_config called with key={key}, value={value}", flush=True)
        print(
            f"DEBUG: session={self.session}, catalog_id={self.catalog_id}", flush=True
        )

        # Check if there's a failed transaction and rollback if needed
        if self.session.in_transaction() and not self.session.is_active:
            print("DEBUG: Rolling back failed transaction", flush=True)
            self.session.rollback()

        # SQLAlchemy's JSONB type handles Python objects automatically
        # No need to json.dumps() - just pass the value directly
        config = (
            self.session.query(Config)
            .filter_by(catalog_id=self.catalog_id, key=key)
            .first()
        )
        print(f"DEBUG: Query completed, config={config}", flush=True)

        if config:
            config.value = value
            config.updated_at = datetime.utcnow()
        else:
            config = Config(
                catalog_id=self.catalog_id,
                key=key,
                value=value,
            )
            self.session.add(config)

        self.session.commit()

    def _update_statistics(self) -> None:
        """Update scan statistics in database."""
        # Get latest stats or create new
        stats = (
            self.session.query(Statistics)
            .filter_by(catalog_id=self.catalog_id)
            .order_by(Statistics.timestamp.desc())
            .first()
        )

        if not stats:
            stats = Statistics(catalog_id=self.catalog_id)
            self.session.add(stats)

        # Update counts
        stats.total_images = (
            self.session.query(Image)
            .filter_by(catalog_id=self.catalog_id, file_type="image")
            .count()
        )
        stats.total_videos = (
            self.session.query(Image)
            .filter_by(catalog_id=self.catalog_id, file_type="video")
            .count()
        )
        stats.images_scanned = stats.total_images + stats.total_videos
        stats.total_size_bytes = self.total_bytes

        # Update performance metrics
        if self.start_time and self.end_time:
            stats.processing_time_seconds = self.end_time - self.start_time
            if stats.processing_time_seconds > 0:
                stats.images_per_second = (
                    self.files_added / stats.processing_time_seconds
                )

        stats.timestamp = datetime.utcnow()
        self.session.commit()


# Compatibility wrapper for existing code
class ImageScanner:
    """
    Wrapper around ImageScannerORM for backward compatibility.

    This allows existing code to work while we migrate to ORM.
    """

    def __init__(self, catalog_db, workers: int = 4, perf_tracker=None):
        """
        Initialize scanner with CatalogDB instance.

        Args:
            catalog_db: CatalogDB instance
            workers: Number of parallel workers
            perf_tracker: Optional performance tracker
        """
        self.catalog = catalog_db
        self.workers = workers
        self.perf_tracker = perf_tracker

        # Create ORM scanner
        if hasattr(catalog_db, "session") and catalog_db.session:
            # Get catalog path (test_path is already a Path object if set)
            if catalog_db._test_path:
                catalog_path = (
                    catalog_db._test_path
                    if isinstance(catalog_db._test_path, Path)
                    else Path(catalog_db._test_path)
                )
            else:
                catalog_path = Path.cwd()

            self.scanner = ImageScannerORM(
                session=catalog_db.session,
                catalog_id=catalog_db.catalog_id,
                catalog_path=catalog_path,
                workers=workers,
                perf_tracker=perf_tracker,
            )
        else:
            raise ValueError("CatalogDB must have an active session for ImageScanner")

    def scan_directories(self, directories: List[Path]) -> None:
        """Scan directories for images."""
        self.scanner.scan_directories(directories)

        # Copy statistics to self for compatibility
        self.files_added = self.scanner.files_added
        self.files_skipped = self.scanner.files_skipped
        self.files_error = self.scanner.files_error
        self.total_bytes = self.scanner.total_bytes

    def _discover_files_incrementally(self, directory: Path):
        """Forward to ORM scanner for incremental file discovery."""
        return self.scanner._discover_files_incrementally(directory)
