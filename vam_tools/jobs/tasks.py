"""
Celery tasks for background processing of VAM Tools operations.

These tasks wrap the CLI operations and provide progress tracking
through Celery's state mechanism.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import Task

from ..core.catalog import CatalogDatabase
from ..organization import (
    FileOrganizer,
    OrganizationOperation,
    OrganizationStrategy,
)
from ..shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path
from .celery_app import app

logger = logging.getLogger(__name__)


class ProgressTask(Task):
    """
    Base task class with progress reporting.

    Provides utilities for reporting progress to the result backend.
    """

    def update_progress(
        self,
        current: int,
        total: int,
        message: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update task progress.

        Args:
            current: Current item number
            total: Total items
            message: Progress message
            extra: Additional metadata
        """
        meta = {
            "current": current,
            "total": total,
            "percent": int((current / total) * 100) if total > 0 else 0,
            "message": message,
        }
        if extra:
            meta.update(extra)

        self.update_state(
            state="PROGRESS",
            meta=meta,
        )


@app.task(bind=True, base=ProgressTask, name="analyze_catalog")
def analyze_catalog_task(
    self: ProgressTask,
    catalog_path: str,
    source_directories: List[str],
    detect_duplicates: bool = False,
    similarity_threshold: int = 5,
    workers: Optional[int] = None,
    checkpoint_interval: int = 100,
) -> Dict[str, Any]:
    """
    Analyze catalog in background.

    Args:
        catalog_path: Path to catalog directory
        source_directories: List of source directories to scan
        detect_duplicates: Whether to detect duplicates
        similarity_threshold: Hamming distance threshold for duplicates
        workers: Number of worker processes
        checkpoint_interval: Checkpoint every N files

    Returns:
        Analysis results dictionary
    """
    logger.info(f"Starting catalog analysis: {catalog_path}")

    try:
        catalog_dir = Path(catalog_path)
        source_dirs = [Path(d) for d in source_directories]

        # Update initial state
        self.update_progress(
            0, 1, "Initializing catalog analysis...", {"phase": "init"}
        )

        # NOTE: Full analysis with metadata extraction requires multiprocessing
        # which doesn't work in Celery workers (daemon processes).
        # For now, we'll do a simple file count and catalog initialization.
        # Use the CLI `vam-analyze` command for full analysis with multiprocessing.

        # Collect files
        from ..shared.media_utils import is_image_file, is_video_file

        all_files = []

        self.update_progress(0, 1, "Scanning directories...", {"phase": "scanning"})

        for source_dir in source_dirs:
            for file_path in Path(source_dir).rglob("*"):
                if file_path.is_file() and (
                    is_image_file(file_path) or is_video_file(file_path)
                ):
                    all_files.append(file_path)

        total_files = len(all_files)
        logger.info(f"Found {total_files} media files")

        self.update_progress(
            total_files,
            total_files,
            f"Found {total_files} media files",
            {"phase": "complete"},
        )

        # Initialize catalog directory
        catalog_dir.mkdir(parents=True, exist_ok=True)

        processed = total_files
        added = 0
        skipped = 0

        logger.info(f"File scan complete: {processed} files found")

        return {
            "status": "completed",
            "catalog_path": str(catalog_dir),
            "total_files": total_files,
            "processed": processed,
            "files_added": added,
            "files_skipped": skipped,
            "note": "This is a file count only. Use CLI 'vam-analyze' for full metadata extraction.",
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


@app.task(bind=True, base=ProgressTask, name="organize_catalog")
def organize_catalog_task(
    self: ProgressTask,
    catalog_path: str,
    output_directory: str,
    operation: str = "copy",
    directory_structure: str = "YYYY-MM",
    naming_strategy: str = "date_time_checksum",
    dry_run: bool = False,
    verify_checksums: bool = True,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Organize catalog files in background.

    Args:
        catalog_path: Path to catalog
        output_directory: Target output directory
        operation: "copy" or "move"
        directory_structure: Directory structure pattern
        naming_strategy: File naming strategy
        dry_run: Preview without executing
        verify_checksums: Verify checksums after operations
        skip_existing: Skip existing files

    Returns:
        Organization results dictionary
    """
    logger.info(f"Starting organization: {catalog_path} -> {output_directory}")

    try:
        catalog_dir = Path(catalog_path)
        output_dir = Path(output_directory)

        self.update_progress(0, 1, "Initializing organization...", {"phase": "init"})

        with CatalogDatabase(catalog_dir) as db:
            # Create strategy
            from ..organization import DirectoryStructure, NamingStrategy

            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure(directory_structure),
                naming_strategy=NamingStrategy(naming_strategy),
                handle_duplicates=True,
            )

            # Create organizer
            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation(operation),
            )

            # Get total file count
            images = db.list_images()
            total_files = len(images)

            self.update_progress(
                0,
                total_files,
                f"{'Previewing' if dry_run else 'Organizing'} files...",
                {"phase": "organizing", "dry_run": dry_run},
            )

            # Organize with progress tracking
            # Note: FileOrganizer doesn't have a progress callback yet
            # For now, we'll just run it and update state
            result = organizer.organize(
                dry_run=dry_run,
                verify_checksums=verify_checksums,
                skip_existing=skip_existing,
            )

            logger.info(f"Organization complete: {result.organized} files organized")

            return {
                "status": "completed",
                "catalog_path": str(catalog_dir),
                "output_directory": str(output_dir),
                "dry_run": dry_run,
                "total_files": result.total_files,
                "organized": result.organized,
                "skipped": result.skipped,
                "failed": result.failed,
                "no_date": result.no_date,
                "errors": result.errors,
                "transaction_id": result.transaction_id,
            }

    except Exception as e:
        logger.error(f"Organization failed: {e}", exc_info=True)
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


@app.task(bind=True, base=ProgressTask, name="generate_thumbnails")
def generate_thumbnails_task(
    self: ProgressTask,
    catalog_path: str,
    sizes: Optional[List[int]] = None,
    quality: int = 85,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Generate thumbnails for catalog images.

    Args:
        catalog_path: Path to catalog
        sizes: List of thumbnail sizes (default: [256, 512])
        quality: JPEG quality (1-100)
        force: Regenerate existing thumbnails

    Returns:
        Thumbnail generation results
    """
    logger.info(f"Starting thumbnail generation: {catalog_path}")

    if sizes is None:
        sizes = [256, 512]

    try:
        catalog_dir = Path(catalog_path)

        self.update_progress(
            0, 1, "Initializing thumbnail generation...", {"phase": "init"}
        )

        with CatalogDatabase(catalog_dir) as db:
            # Get images
            images = db.list_images()
            total_images = len(images)

            # Setup thumbnail directory
            thumbnail_dir = catalog_dir / "thumbnails"
            thumbnail_dir.mkdir(exist_ok=True)

            # Generate thumbnails with progress
            generated = 0
            skipped = 0
            failed = 0

            for i, image in enumerate(images):
                try:
                    # Update progress
                    if i % 10 == 0 or i == total_images:
                        self.update_progress(
                            i,
                            total_images,
                            f"Generating thumbnail for {image.source_path.name}...",
                            {
                                "phase": "generating",
                                "generated": generated,
                                "skipped": skipped,
                                "failed": failed,
                            },
                        )

                    # Check if thumbnail exists
                    if not force:
                        thumbnail_path = get_thumbnail_path(
                            image.id, thumbnail_dir, sizes[0] if sizes else 256
                        )
                        if thumbnail_path.exists():
                            skipped += 1
                            continue

                    # Generate thumbnails for all sizes
                    for size in sizes or [256, 512]:
                        generate_thumbnail(
                            image.source_path,
                            image.id,
                            thumbnail_dir,
                            size=size,
                            quality=quality,
                        )
                    generated += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to generate thumbnail for {image.source_path}: {e}"
                    )
                    failed += 1

            logger.info(
                f"Thumbnail generation complete: {generated} generated, "
                f"{skipped} skipped, {failed} failed"
            )

            return {
                "status": "completed",
                "catalog_path": str(catalog_dir),
                "total_images": total_images,
                "generated": generated,
                "skipped": skipped,
                "failed": failed,
                "sizes": sizes,
            }

    except Exception as e:
        logger.error(f"Thumbnail generation failed: {e}", exc_info=True)
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise
