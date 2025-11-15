"""
Image scanner and analyzer.

Scans directories, extracts metadata, computes checksums, and builds catalog.
"""

import logging
import multiprocessing as mp
import os
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from vam_tools.shared import compute_checksum, get_file_type
from vam_tools.shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path

from ..db import CatalogDB as CatalogDatabase
from ..core.performance_stats import PerformanceTracker
from ..core.types import CatalogPhase, FileType, ImageRecord, ImageStatus, Statistics
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
        image = ImageRecord(
            id=checksum,
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


class ImageScanner:
    """Scan directories and analyze images."""

    def __init__(
        self,
        catalog: CatalogDatabase,
        workers: Optional[int] = None,
        perf_tracker: Optional[PerformanceTracker] = None,
        generate_thumbnails: bool = True,
    ):
        """
        Initialize the scanner.

        Args:
            catalog: Catalog database to update
            workers: Number of worker processes (default: CPU count)
            perf_tracker: Optional performance tracker for collecting metrics
            generate_thumbnails: Whether to generate thumbnails during scan (default: True)
        """
        self.catalog = catalog
        self.generate_thumbnails = generate_thumbnails
        # Load existing statistics if catalog exists, otherwise start fresh
        latest_stats_row = self.catalog.execute(
            "SELECT * FROM statistics WHERE catalog_id = ? ORDER BY timestamp DESC LIMIT 1",
            (str(self.catalog.catalog_id),)
        ).fetchone()
        self.stats = (
            Statistics(**latest_stats_row) if latest_stats_row else Statistics()
        )
        self.files_added = 0
        self.files_skipped = 0
        self.files_processed = 0  # Track total files processed for performance stats
        self.workers = workers or mp.cpu_count()
        self.perf_tracker = perf_tracker

    def scan_directories(self, directories: List[Path]) -> None:
        """
        Scan directories for images and videos using incremental discovery.

        Files are discovered and processed in batches to avoid blocking on
        slow network filesystems.

        Args:
            directories: List of directories to scan
        """
        logger.info(f"Scanning {len(directories)} directories (incremental mode)")

        # Update state (phase) in catalog_config
        self.catalog.execute(
            "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("phase", CatalogPhase.ANALYZING.value),
        )

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
            batch_size = 100  # Process files in batches of 100
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

                        # Insert new statistics snapshot after each batch
                        self.catalog.execute(
                            """
                            INSERT INTO statistics (
                                catalog_id, timestamp, total_images, total_videos, total_size_bytes,
                                images_scanned, images_hashed, images_tagged,
                                duplicate_groups, duplicate_images, potential_savings_bytes,
                                high_quality_count, medium_quality_count, low_quality_count,
                                corrupted_count, unsupported_count,
                                processing_time_seconds, images_per_second,
                                no_date, suspicious_dates, problematic_files
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(self.catalog.catalog_id),
                                datetime.now().isoformat(),
                                self.stats.total_images,
                                self.stats.total_videos,
                                self.stats.total_size_bytes,
                                self.stats.images_scanned,
                                self.stats.images_hashed,
                                self.stats.images_tagged,
                                self.stats.duplicate_groups,
                                self.stats.duplicate_images,
                                self.stats.potential_savings_bytes,
                                self.stats.high_quality_count,
                                self.stats.medium_quality_count,
                                self.stats.low_quality_count,
                                self.stats.corrupted_count,
                                self.stats.unsupported_count,
                                self.stats.processing_time_seconds,
                                self.stats.images_per_second,
                                self.stats.no_date,
                                self.stats.suspicious_dates,
                                self.stats.problematic_files,
                            ),
                        )

            # Process any remaining files
            if current_batch:
                logger.info(f"Processing final batch of {len(current_batch)} files")
                self._process_files(current_batch)

            # Insert final statistics snapshot
            self.catalog.execute(
                """
                INSERT INTO statistics (
                    catalog_id, timestamp, total_images, total_videos, total_size_bytes,
                    images_scanned, images_hashed, images_tagged,
                    duplicate_groups, duplicate_images, potential_savings_bytes,
                    high_quality_count, medium_quality_count, low_quality_count,
                    corrupted_count, unsupported_count,
                    processing_time_seconds, images_per_second,
                    no_date, suspicious_dates, problematic_files
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(self.catalog.catalog_id),
                    datetime.now().isoformat(),
                    self.stats.total_images,
                    self.stats.total_videos,
                    self.stats.total_size_bytes,
                    self.stats.images_scanned,
                    self.stats.images_hashed,
                    self.stats.images_tagged,
                    self.stats.duplicate_groups,
                    self.stats.duplicate_images,
                    self.stats.potential_savings_bytes,
                    self.stats.high_quality_count,
                    self.stats.medium_quality_count,
                    self.stats.low_quality_count,
                    self.stats.corrupted_count,
                    self.stats.unsupported_count,
                    self.stats.processing_time_seconds,
                    self.stats.images_per_second,
                    self.stats.no_date,
                    self.stats.suspicious_dates,
                    self.stats.problematic_files,
                ),
            )

            # Track total files and bytes (use files_processed for accurate stats)
            if self.perf_tracker:
                self.perf_tracker.metrics.total_files_analyzed = self.files_processed
                self.perf_tracker.metrics.bytes_processed = self.stats.total_size_bytes

            logger.info(
                f"Scan complete: {files_discovered} files discovered, "
                f"{self.files_added} files added, {self.files_skipped} files skipped "
                f"({self.stats.total_images} images, {self.stats.total_videos} videos total)"
            )

    def _discover_files_incrementally(self, directory: Path) -> Iterator[Path]:
        """
        Discover image and video files incrementally using os.walk.

        This yields files as they're discovered instead of collecting all files first,
        which is much faster on network filesystems with large directory trees.

        Args:
            directory: Root directory to scan

        Yields:
            Path objects for image/video files
        """
        try:
            for root, dirs, files in os.walk(directory):
                # Skip synology metadata directories
                dirs[:] = [d for d in dirs if not d.startswith("@eaDir")]

                root_path = Path(root)

                for filename in files:
                    # Skip synology metadata files
                    if filename.startswith(".") or "@SynoResource" in filename:
                        continue

                    file_path = root_path / filename
                    file_type = get_file_type(file_path)

                    if file_type in ("image", "video"):
                        yield file_path

        except PermissionError as e:
            logger.warning(f"Permission denied: {directory}: {e}")
        except Exception as e:
            logger.error(f"Error scanning {directory}: {e}")

    def _collect_files(self, directory: Path) -> List[Path]:
        """
        Collect all image and video files from a directory.

        DEPRECATED: Use _discover_files_incrementally instead for better performance.
        """
        return list(self._discover_files_incrementally(directory))

    def _process_files(self, files: List[Path]) -> None:
        """Process all files in parallel and add to catalog."""
        # Track processing operation
        process_ctx = (
            self.perf_tracker.track_operation("process_files", items=len(files))
            if self.perf_tracker
            else nullcontext()
        )

        with process_ctx:
            # Filter out files already in catalog
            files_to_process = []
            for f in files:
                existing_image = self.catalog.execute(
                    "SELECT id FROM images WHERE catalog_id = ? AND source_path = ?",
                    (str(self.catalog.catalog_id), str(f))
                ).fetchone()
                if not existing_image:
                    files_to_process.append(f)
                else:
                    self.files_skipped += 1

            if not files_to_process:
                logger.info("All files already in catalog")
                return

            logger.info(
                f"Processing {len(files_to_process)} files with {self.workers} workers"
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    "Processing files...", total=len(files_to_process)
                )

                # Use multiprocessing pool to process files in parallel with health monitoring
                with mp.Pool(processes=self.workers) as pool:
                    # Process files in chunks for better progress updates
                    chunk_size = max(1, len(files_to_process) // (self.workers * 4))

                    # Track progress for stuck worker detection
                    last_progress_time = time.time()
                    stuck_timeout = 300  # 5 minutes without progress = stuck

                    result_iter = pool.imap_unordered(
                        _process_file_worker, files_to_process, chunk_size
                    )

                    for i, result in enumerate(result_iter):
                        # Check if workers are stuck (no progress recently)
                        current_time = time.time()
                        time_since_progress = current_time - last_progress_time

                        if time_since_progress > stuck_timeout:
                            logger.warning(
                                f"Workers appear stuck - no progress for {stuck_timeout}s. "
                                f"Processed {i}/{len(files_to_process)} files. "
                                f"This may indicate network/I/O issues with your file storage."
                            )
                            # Reset timer after warning
                            last_progress_time = current_time

                        # Update progress time for each successful result
                        last_progress_time = current_time

                        # Process the result (same logic as original)
                        if result is not None:
                            image, file_size = result
                            self.files_processed += 1

                            # Track file format
                            if self.perf_tracker and image.metadata:
                                format_str = image.metadata.format or "unknown"
                                # Track per-file processing time (approximate)
                                avg_time = 0.1  # Placeholder, can't track individual in multiprocessing
                                self.perf_tracker.record_file_format(
                                    format_str, file_size, avg_time
                                )

                            # Update real-time performance counters for ALL processed files
                            if self.perf_tracker:
                                self.perf_tracker.metrics.total_files_analyzed = (
                                    self.files_processed
                                )
                                self.perf_tracker.metrics.bytes_processed += file_size

                            # Check if already in catalog (by checksum - for duplicates)
                            existing_image_by_id = self.catalog.execute(
                                "SELECT id FROM images WHERE catalog_id = ? AND id = ?",
                                (str(self.catalog.catalog_id), image.checksum)
                            ).fetchone()

                            if not existing_image_by_id:
                                # Generate thumbnail if enabled (before adding to catalog)
                                if self.generate_thumbnails:
                                    thumb_path = get_thumbnail_path(
                                        image.id,
                                        self.catalog.catalog_path / "thumbnails",
                                    )
                                    if generate_thumbnail(
                                        source_path=image.source_path,
                                        output_path=thumb_path,
                                    ):
                                        # Set thumbnail path on image record (relative)
                                        image.thumbnail_path = thumb_path.relative_to(
                                            self.catalog.catalog_path
                                        )

                                # Add to catalog using CatalogDB's add_image method
                                self.catalog.add_image(image)
                                self.files_added += 1

                                # Update statistics
                                if image.file_type == FileType.IMAGE:
                                    self.stats.total_images += 1
                                elif image.file_type == FileType.VIDEO:
                                    self.stats.total_videos += 1

                                self.stats.total_size_bytes += file_size
                                self.stats.images_scanned += 1  # Assuming scanned here
                                self.stats.images_hashed += 1  # Assuming hashed here

                                if not image.dates.selected_date:
                                    self.stats.no_date += 1
                            else:
                                self.files_skipped += 1

                        # Checkpoint every 10 files for more frequent updates
                        if (i + 1) % 10 == 0:
                            # Insert new statistics snapshot
                            self.catalog.execute(
                                """
                                INSERT INTO statistics (
                                    catalog_id, timestamp, total_images, total_videos, total_size_bytes,
                                    images_scanned, images_hashed, images_tagged,
                                    duplicate_groups, duplicate_images, potential_savings_bytes,
                                    high_quality_count, medium_quality_count, low_quality_count,
                                    corrupted_count, unsupported_count,
                                    processing_time_seconds, images_per_second,
                                    no_date, suspicious_dates, problematic_files
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    str(self.catalog.catalog_id),
                                    datetime.now().isoformat(),
                                    self.stats.total_images,
                                    self.stats.total_videos,
                                    self.stats.total_size_bytes,
                                    self.stats.images_scanned,
                                    self.stats.images_hashed,
                                    self.stats.images_tagged,
                                    self.stats.duplicate_groups,
                                    self.stats.duplicate_images,
                                    self.stats.potential_savings_bytes,
                                    self.stats.high_quality_count,
                                    self.stats.medium_quality_count,
                                    self.stats.low_quality_count,
                                    self.stats.corrupted_count,
                                    self.stats.unsupported_count,
                                    self.stats.processing_time_seconds,
                                    self.stats.images_per_second,
                                    self.stats.no_date,
                                    self.stats.suspicious_dates,
                                    self.stats.problematic_files,
                                ),
                            )

                            # Update state (images_processed, progress_percentage) in catalog_config
                            self.catalog.execute(
                                "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                                ("images_processed", str(i + 1)),
                            )
                            self.catalog.execute(
                                "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                                (
                                    "progress_percentage",
                                    str(((i + 1) / len(files_to_process)) * 100),
                                ),
                            )

                        progress.advance(task)
