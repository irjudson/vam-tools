"""
Image scanner and analyzer.

Scans directories, extracts metadata, computes checksums, and builds catalog.
"""

import logging
import multiprocessing as mp
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from vam_tools.shared import compute_checksum, get_file_type
from vam_tools.shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path

from ..core.catalog import CatalogDatabase
from ..core.performance_stats import PerformanceTracker
from ..core.types import CatalogPhase, FileType, ImageRecord, ImageStatus
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
        self.stats = catalog.get_statistics()
        self.files_added = 0
        self.files_skipped = 0
        self.files_processed = 0  # Track total files processed for performance stats
        self.workers = workers or mp.cpu_count()
        self.perf_tracker = perf_tracker

    def scan_directories(self, directories: List[Path]) -> None:
        """
        Scan directories for images and videos.

        Args:
            directories: List of directories to scan
        """
        logger.info(f"Scanning {len(directories)} directories")

        # Update state
        state = self.catalog.get_state()
        state.phase = CatalogPhase.ANALYZING
        self.catalog.update_state(state)

        # Track file collection
        ctx = (
            self.perf_tracker.track_operation(
                "scan_directories", items=len(directories)
            )
            if self.perf_tracker
            else nullcontext()
        )

        with ctx:
            # Collect all files first
            all_files = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=None,
            ) as progress:
                task = progress.add_task("Collecting files...", total=None)

                for directory in directories:
                    files = self._collect_files(Path(directory))
                    all_files.extend(files)
                    progress.update(
                        task, description=f"Found {len(all_files)} files..."
                    )

            logger.info(f"Found {len(all_files)} image/video files")

            # Update total count
            state.images_total = len(all_files)
            self.catalog.update_state(state)

            # Process files
            self._process_files(all_files)

            # Update final statistics
            self.catalog.update_statistics(self.stats)

            # Track total files and bytes (use files_processed for accurate stats)
            if self.perf_tracker:
                self.perf_tracker.metrics.total_files_analyzed = self.files_processed
                self.perf_tracker.metrics.bytes_processed = self.stats.total_size_bytes

            # Save catalog with final statistics
            self.catalog.save()

            logger.info(
                f"Scan complete: {self.files_added} files added, {self.files_skipped} files skipped "
                f"({self.stats.total_images} images, {self.stats.total_videos} videos total)"
            )

    def _collect_files(self, directory: Path) -> List[Path]:
        """Collect all image and video files from a directory."""
        files = []

        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    file_type = get_file_type(item)
                    if file_type in ("image", "video"):
                        files.append(item)
        except PermissionError as e:
            logger.warning(f"Permission denied: {directory}: {e}")
        except Exception as e:
            logger.error(f"Error scanning {directory}: {e}")

        return files

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
            files_to_process = [
                f for f in files if not self.catalog.has_image_by_path(f)
            ]
            self.files_skipped = len(files) - len(files_to_process)

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

                # Use multiprocessing pool to process files in parallel
                with mp.Pool(processes=self.workers) as pool:
                    # Process files in chunks for better progress updates
                    chunk_size = max(1, len(files_to_process) // (self.workers * 4))

                    for i, result in enumerate(
                        pool.imap_unordered(
                            _process_file_worker, files_to_process, chunk_size
                        )
                    ):
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
                                self.perf_tracker.metrics.bytes_processed += (
                                    file_size
                                )

                            # Check if already in catalog (by checksum - for duplicates)
                            if not self.catalog.get_image(image.checksum):
                                # Add to catalog
                                self.catalog.add_image(image)
                                self.files_added += 1

                                # Generate thumbnail if enabled
                                if self.generate_thumbnails:
                                    thumb_path = get_thumbnail_path(
                                        image.id, self.catalog.thumbnails_dir
                                    )
                                    if generate_thumbnail(
                                        source_path=image.source_path,
                                        output_path=thumb_path,
                                    ):
                                        # Update image with thumbnail path (relative)
                                        image.thumbnail_path = thumb_path.relative_to(
                                            self.catalog.catalog_path
                                        )
                                        self.catalog.add_image(
                                            image
                                        )  # Update with thumbnail path

                                # Update statistics
                                if image.file_type == FileType.IMAGE:
                                    self.stats.total_images += 1
                                elif image.file_type == FileType.VIDEO:
                                    self.stats.total_videos += 1

                                self.stats.total_size_bytes += file_size

                                if not image.dates.selected_date:
                                    self.stats.no_date += 1
                            else:
                                self.files_skipped += 1

                        # Checkpoint every 10 files for more frequent updates
                        if (i + 1) % 10 == 0:
                            # Update statistics before checkpoint
                            self.catalog.update_statistics(self.stats)

                            # Update state
                            state = self.catalog.get_state()
                            state.images_processed = i + 1
                            state.progress_percentage = (
                                (i + 1) / len(files_to_process)
                            ) * 100
                            self.catalog.update_state(state)

                            # Track checkpoint operation
                            checkpoint_ctx = (
                                self.perf_tracker.track_operation("checkpoint")
                                if self.perf_tracker
                                else nullcontext()
                            )
                            with checkpoint_ctx:
                                self.catalog.checkpoint()

                        progress.advance(task)

            # Final checkpoint
            final_checkpoint_ctx = (
                self.perf_tracker.track_operation("final_checkpoint")
                if self.perf_tracker
                else nullcontext()
            )
            with final_checkpoint_ctx:
                self.catalog.checkpoint(force=True)
