"""
Image scanner and analyzer.

Scans directories, extracts metadata, computes checksums, and builds catalog.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from ..core.catalog import CatalogDatabase
from ..core.types import (
    CatalogPhase,
    FileType,
    ImageRecord,
    ImageStatus,
    Statistics,
)
from ..core.utils import compute_checksum, get_file_type
from .metadata import MetadataExtractor

logger = logging.getLogger(__name__)


class ImageScanner:
    """Scan directories and analyze images."""

    def __init__(self, catalog: CatalogDatabase):
        """
        Initialize the scanner.

        Args:
            catalog: Catalog database to update
        """
        self.catalog = catalog
        # Load existing statistics if catalog exists, otherwise start fresh
        self.stats = catalog.get_statistics()
        self.files_added = 0
        self.files_skipped = 0

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
                progress.update(task, description=f"Found {len(all_files)} files...")

        logger.info(f"Found {len(all_files)} image/video files")

        # Update total count
        state.images_total = len(all_files)
        self.catalog.update_state(state)

        # Process files
        self._process_files(all_files)

        # Update final statistics
        self.catalog.update_statistics(self.stats)

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
        """Process all files and add to catalog."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Processing files...", total=len(files))

            with MetadataExtractor() as extractor:
                for i, file_path in enumerate(files):
                    try:
                        self._process_file(file_path, extractor)

                        # Checkpoint every 100 files
                        if (i + 1) % 100 == 0:
                            # Update statistics before checkpoint
                            self.catalog.update_statistics(self.stats)

                            # Update state
                            state = self.catalog.get_state()
                            state.images_processed = i + 1
                            state.progress_percentage = ((i + 1) / len(files)) * 100
                            self.catalog.update_state(state)

                            # Save checkpoint
                            self.catalog.checkpoint()

                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")

                    progress.advance(task)

        # Final checkpoint
        self.catalog.checkpoint(force=True)

    def _process_file(self, file_path: Path, extractor: MetadataExtractor) -> None:
        """Process a single file."""
        # Quick check: if file path is already in catalog, skip it entirely
        # This avoids expensive checksum computation for already-processed files
        if self.catalog.has_image_by_path(file_path):
            logger.debug(f"File already in catalog (by path): {file_path}")
            self.files_skipped += 1
            return

        # Determine file type
        file_type_str = get_file_type(file_path)
        if file_type_str == "image":
            file_type = FileType.IMAGE
        elif file_type_str == "video":
            file_type = FileType.VIDEO
        else:
            return  # Skip unknown files

        # Compute checksum
        checksum = compute_checksum(file_path)
        if not checksum:
            logger.warning(f"Failed to compute checksum for {file_path}")
            return

        # Check if already in catalog (by checksum - for duplicates)
        existing = self.catalog.get_image(checksum)
        if existing:
            logger.debug(f"Image already in catalog (by checksum): {file_path}")
            self.files_skipped += 1
            return

        # This is a new file - update counts
        if file_type == FileType.IMAGE:
            self.stats.total_images += 1
        elif file_type == FileType.VIDEO:
            self.stats.total_videos += 1

        # Extract metadata
        metadata = extractor.extract_metadata(file_path, file_type)
        dates = extractor.extract_dates(file_path, metadata)

        # Update statistics
        self.stats.total_size_bytes += metadata.size_bytes or 0

        if not dates.selected_date:
            self.stats.no_date += 1

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

        # Add to catalog
        self.catalog.add_image(image)
        self.files_added += 1

        logger.debug(f"Processed: {file_path} -> {checksum[:8]}")
