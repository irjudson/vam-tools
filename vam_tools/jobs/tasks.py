"""
Celery tasks for background processing of VAM Tools operations.

These tasks wrap operations and provide progress tracking through Celery's state mechanism.
Updated for PostgreSQL backend.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from celery import Task
from sqlalchemy import text

from vam_tools.core.types import FileType, ImageRecord, ImageStatus

from ..analysis.duplicate_detector import DuplicateDetector
from ..analysis.scanner import _process_file_worker
from ..db import CatalogDB as CatalogDatabase
from ..organization import FileOrganizer, OrganizationStrategy
from ..shared.media_utils import get_file_type
from ..shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path

# Import app here so that @app.task decorators can find it
from .celery_app import app
from .scan_stats import ScanStatistics

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
    catalog_id: str,
    source_directories: List[str],
    detect_duplicates: bool = False,
    force_reanalyze: bool = False,
    similarity_threshold: int = 5,
) -> Dict[str, Any]:
    """
    Analyze catalog in background using PostgreSQL.

    Args:
        catalog_id: UUID of the catalog to work with
        source_directories: List of source directories to scan
        detect_duplicates: Whether to detect duplicates
        force_reanalyze: Reset catalog and analyze all files fresh
        similarity_threshold: Hamming distance threshold for duplicates

    Returns:
        Analysis results dictionary with detailed statistics
    """
    logger.info(f"Starting catalog analysis: {catalog_id}")

    # Initialize statistics tracking
    stats = ScanStatistics()

    try:
        source_dirs = [Path(d) for d in source_directories]

        # Update initial state
        self.update_progress(0, 1, "Initializing catalog...", {"phase": "init"})

        # PostgreSQL catalog - no file-based catalog anymore
        with CatalogDatabase(catalog_id) as db:
            # Update catalog config (value must be valid JSON for JSONB column)
            db.execute(
                """
                INSERT INTO config (catalog_id, key, value, updated_at)
                VALUES (?, ?, ?, NOW())
                ON CONFLICT (catalog_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
                """,
                (db.catalog_id, "phase", json.dumps("analyzing")),
            )
            db.execute(
                """
                INSERT INTO config (catalog_id, key, value, updated_at)
                VALUES (?, ?, ?, NOW())
                ON CONFLICT (catalog_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
                """,
                (db.catalog_id, "last_updated", json.dumps(datetime.now().isoformat())),
            )

            if force_reanalyze:
                logger.info("Force reanalyze: clearing existing images")
                self.update_progress(
                    0, 1, "Clearing existing images...", {"phase": "clearing"}
                )
                # Delete all images for this catalog
                from sqlalchemy import text as sql_text

                db.session.execute(
                    sql_text("DELETE FROM images WHERE catalog_id = :catalog_id"),
                    {"catalog_id": catalog_id},
                )
                db.session.commit()

        # Process files sequentially - simple and reliable
        self.update_progress(
            0, 1, "Discovering and processing files...", {"phase": "processing"}
        )

        files_processed = 0

        # Reopen database for processing
        with CatalogDatabase(catalog_id) as db:
            for directory in source_dirs:
                logger.info(f"Scanning directory: {directory}")

                for root, dirs, files in os.walk(directory):
                    stats.directories_scanned += 1

                    # Count and skip synology metadata directories
                    original_dir_count = len(dirs)
                    dirs[:] = [d for d in dirs if not d.startswith("@eaDir")]
                    stats.skipped_synology_metadata += original_dir_count - len(dirs)

                    root_path = Path(root)

                    for filename in files:
                        stats.files_discovered += 1

                        # Skip synology metadata files
                        if "@SynoResource" in filename:
                            stats.skipped_synology_metadata += 1
                            continue

                        # Skip hidden files
                        if filename.startswith("."):
                            stats.skipped_hidden_file += 1
                            continue

                        file_path = root_path / filename

                        # Skip if file doesn't exist or is not accessible
                        if not file_path.exists():
                            stats.skipped_file_not_accessible += 1
                            continue

                        file_type_str = get_file_type(file_path)

                        if file_type_str not in ("image", "video"):
                            stats.skipped_unsupported_format += 1
                            continue

                        # Update progress every 10 files
                        files_processed += 1
                        if files_processed % 10 == 0:
                            self.update_progress(
                                files_processed,
                                files_processed + 1,
                                f"Processing {file_path.name}...",
                                {
                                    "phase": "processing",
                                    "added": stats.files_added,
                                    "skipped": stats.total_skipped
                                    + stats.skipped_already_in_catalog,
                                    "errors": stats.total_errors,
                                },
                            )

                        # Process file
                        result = _process_file_worker(file_path)

                        if result is not None:
                            image_record, file_size = result

                            # Track file type
                            if file_type_str == "image":
                                stats.images_processed += 1
                            else:
                                stats.videos_processed += 1

                            # Track size
                            stats.total_bytes_processed += file_size
                            if file_size > stats.largest_file_bytes:
                                stats.largest_file_bytes = file_size
                                stats.largest_file_path = str(file_path)

                            # Check if already in catalog
                            existing_image = db.get_image(image_record.id)

                            if not existing_image:
                                # Add to catalog using CatalogDB.add_image()
                                try:
                                    db.add_image(image_record)
                                    stats.files_added += 1
                                except Exception as e:
                                    stats.errors_database += 1
                                    stats.record_error(file_path, "database", str(e))
                            else:
                                stats.skipped_already_in_catalog += 1
                        else:
                            stats.errors_metadata_extraction += 1
                            stats.record_error(
                                file_path, "metadata", "Failed to extract metadata"
                            )

            # Update catalog config with completion status
            db.execute(
                """
                INSERT INTO config (catalog_id, key, value, updated_at)
                VALUES (?, ?, ?, NOW())
                ON CONFLICT (catalog_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
                """,
                (db.catalog_id, "phase", json.dumps("complete")),
            )
            db.execute(
                """
                INSERT INTO config (catalog_id, key, value, updated_at)
                VALUES (?, ?, ?, NOW())
                ON CONFLICT (catalog_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
                """,
                (db.catalog_id, "last_updated", json.dumps(datetime.now().isoformat())),
            )
            db.save()

        # Finalize statistics
        stats.finish()

        logger.info(stats.to_summary())

        self.update_progress(
            100,
            100,
            f"Analysis complete: {stats.files_added} files added",
            {"phase": "complete"},
        )

        # Return full statistics
        result = stats.to_dict()
        result["catalog_id"] = catalog_id
        return result

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        stats.finish()
        stats.record_error("", "fatal", str(e))
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
                "statistics": stats.to_dict(),
            },
        )
        raise


@app.task(bind=True, base=ProgressTask, name="scan_catalog")
def scan_catalog_task(
    self: ProgressTask,
    catalog_id: str,
    source_directories: List[str],
    force_rescan: bool = False,
    generate_previews: bool = True,
) -> Dict[str, Any]:
    """
    Simple catalog scan: find files, extract metadata, create previews.

    This is a lightweight job that:
    - Scans directories for image/video files
    - Extracts metadata (EXIF, dates, etc.)
    - Generates thumbnail previews
    - Does NOT do duplicate detection or other analysis

    Args:
        catalog_id: UUID of the catalog to work with
        source_directories: List of source directories to scan
        force_rescan: Reset catalog and scan all files fresh
        generate_previews: Whether to generate thumbnails (default: True)

    Returns:
        Scan results dictionary with detailed statistics
    """
    logger.info(f"Starting catalog scan: {catalog_id}")

    # Initialize statistics tracking
    stats = ScanStatistics()

    try:
        source_dirs = [Path(d) for d in source_directories]

        # Update initial state
        self.update_progress(0, 1, "Initializing scan...", {"phase": "init"})

        # PostgreSQL catalog
        with CatalogDatabase(catalog_id) as db:
            # Update catalog config (value must be valid JSON for JSONB column)
            db.execute(
                """
                INSERT INTO config (catalog_id, key, value, updated_at)
                VALUES (?, ?, ?, NOW())
                ON CONFLICT (catalog_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
                """,
                (db.catalog_id, "phase", json.dumps("scanning")),
            )
            db.execute(
                """
                INSERT INTO config (catalog_id, key, value, updated_at)
                VALUES (?, ?, ?, NOW())
                ON CONFLICT (catalog_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
                """,
                (db.catalog_id, "last_updated", json.dumps(datetime.now().isoformat())),
            )

            if force_rescan:
                logger.info("Force rescan: clearing existing images")
                self.update_progress(
                    0, 1, "Clearing existing images...", {"phase": "clearing"}
                )
                # Delete all images for this catalog
                from sqlalchemy import text as sql_text

                db.session.execute(
                    sql_text("DELETE FROM images WHERE catalog_id = :catalog_id"),
                    {"catalog_id": catalog_id},
                )
                db.session.commit()

        # Helper function to process a single file
        def process_single_file(
            file_path: Path, catalog_id: str, generate_previews: bool
        ) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
            """
            Process a single file and return results.

            Returns:
                Tuple of (result_dict, outcome, error_message)
                outcome is one of: 'success', 'error_metadata', 'error_thumbnail', 'error_other'
            """
            try:
                # Process file (extract metadata)
                result = _process_file_worker(file_path)

                if result is None:
                    return None, "error_metadata", "Failed to extract metadata"

                image_record, file_size = result

                # Generate thumbnail if requested
                thumbnail_status = "not_requested"
                if generate_previews:
                    try:
                        # Compute thumbnails directory for this catalog
                        thumbnails_dir = Path(f"/app/catalogs/{catalog_id}/thumbnails")

                        thumbnail_path = get_thumbnail_path(
                            image_id=image_record.id, thumbnails_dir=thumbnails_dir
                        )
                        if thumbnail_path.exists():
                            thumbnail_status = "existing"
                        else:
                            success = generate_thumbnail(
                                source_path=file_path,
                                output_path=thumbnail_path,
                                size=(512, 512),
                                quality=85,
                            )
                            thumbnail_status = "generated" if success else "failed"
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate thumbnail for {file_path}: {e}"
                        )
                        thumbnail_status = "failed"

                return (
                    {
                        "image_record": image_record,
                        "file_size": file_size,
                        "file_path": file_path,
                        "thumbnail_status": thumbnail_status,
                    },
                    "success",
                    None,
                )
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                return None, "error_other", str(e)

        # Collect all media files first, tracking skips
        self.update_progress(0, 1, "Discovering files...", {"phase": "discovery"})

        all_files: List[Tuple[Path, str]] = []  # (path, file_type)
        for directory in source_dirs:
            logger.info(f"Discovering files in: {directory}")
            for root, dirs, files in os.walk(directory):
                stats.directories_scanned += 1

                # Count and skip synology metadata directories
                original_dir_count = len(dirs)
                dirs[:] = [d for d in dirs if not d.startswith("@eaDir")]
                stats.skipped_synology_metadata += original_dir_count - len(dirs)

                root_path = Path(root)
                for filename in files:
                    stats.files_discovered += 1

                    # Skip synology metadata files
                    if "@SynoResource" in filename or "@eaDir" in str(root_path):
                        stats.skipped_synology_metadata += 1
                        continue

                    # Skip hidden files
                    if filename.startswith("."):
                        stats.skipped_hidden_file += 1
                        continue

                    file_path = root_path / filename

                    # Check accessibility
                    if not file_path.exists():
                        stats.skipped_file_not_accessible += 1
                        continue

                    # Check file type
                    file_type_str = get_file_type(file_path)
                    if file_type_str not in ("image", "video"):
                        stats.skipped_unsupported_format += 1
                        continue

                    all_files.append((file_path, file_type_str))

        total_files = len(all_files)
        logger.info(
            f"Found {total_files} media files to process "
            f"(discovered {stats.files_discovered}, "
            f"skipped {stats.total_skipped} during discovery)"
        )

        # Process files in parallel
        self.update_progress(
            0, total_files, f"Processing {total_files} files...", {"phase": "scanning"}
        )

        progress_lock = Lock()
        files_processed = 0

        # Use ThreadPoolExecutor for parallel processing
        # Use 4x CPU count for I/O-bound thumbnail generation
        max_workers = min((os.cpu_count() or 4) * 4, 32)
        logger.info(f"Using {max_workers} workers for parallel processing")

        with CatalogDatabase(catalog_id) as db:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(
                        process_single_file, file_path, catalog_id, generate_previews
                    ): (file_path, file_type)
                    for file_path, file_type in all_files
                }

                # Process completed files as they finish
                for future in as_completed(future_to_file):
                    file_path, file_type = future_to_file[future]

                    try:
                        result, outcome, error_msg = future.result()

                        with progress_lock:
                            files_processed += 1

                            if result is not None:
                                image_record = result["image_record"]
                                file_size = result["file_size"]
                                thumbnail_status = result["thumbnail_status"]

                                # Track file type
                                if file_type == "image":
                                    stats.images_processed += 1
                                else:
                                    stats.videos_processed += 1

                                # Track size
                                stats.total_bytes_processed += file_size
                                if file_size > stats.largest_file_bytes:
                                    stats.largest_file_bytes = file_size
                                    stats.largest_file_path = str(file_path)

                                # Track thumbnail outcome
                                if thumbnail_status == "generated":
                                    stats.thumbnails_generated += 1
                                elif thumbnail_status == "existing":
                                    stats.thumbnails_skipped_existing += 1
                                elif thumbnail_status == "failed":
                                    stats.thumbnails_failed += 1

                                # Check if already in catalog
                                existing_image = db.get_image(image_record.id)

                                if not existing_image:
                                    # Add to catalog
                                    try:
                                        db.add_image(image_record)
                                        stats.files_added += 1
                                    except Exception as e:
                                        stats.errors_database += 1
                                        stats.record_error(
                                            file_path, "database", str(e)
                                        )
                                else:
                                    stats.skipped_already_in_catalog += 1
                            else:
                                # Processing failed
                                if outcome == "error_metadata":
                                    stats.errors_metadata_extraction += 1
                                elif outcome == "error_thumbnail":
                                    stats.errors_thumbnail_generation += 1
                                else:
                                    stats.errors_other += 1
                                if error_msg:
                                    stats.record_error(file_path, outcome, error_msg)

                            # Update progress every 10 files
                            if files_processed % 10 == 0:
                                self.update_progress(
                                    files_processed,
                                    total_files,
                                    f"Processed {files_processed}/{total_files} files...",
                                    {
                                        "phase": "scanning",
                                        "processed": files_processed,
                                        "added": stats.files_added,
                                        "skipped": stats.total_skipped
                                        + stats.skipped_already_in_catalog,
                                        "errors": stats.total_errors,
                                    },
                                )
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        with progress_lock:
                            files_processed += 1
                            stats.errors_other += 1
                            stats.record_error(file_path, "executor", str(e))

            # Update catalog config with completion status
            db.execute(
                """
                INSERT INTO config (catalog_id, key, value, updated_at)
                VALUES (?, ?, ?, NOW())
                ON CONFLICT (catalog_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
                """,
                (db.catalog_id, "phase", json.dumps("complete")),
            )
            db.execute(
                """
                INSERT INTO config (catalog_id, key, value, updated_at)
                VALUES (?, ?, ?, NOW())
                ON CONFLICT (catalog_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
                """,
                (db.catalog_id, "last_updated", json.dumps(datetime.now().isoformat())),
            )
            db.save()

        # Finalize statistics
        stats.finish()

        logger.info(stats.to_summary())

        self.update_progress(
            100,
            100,
            f"Scan complete: {stats.files_added} files added",
            {"phase": "complete"},
        )

        # Return full statistics
        result = stats.to_dict()
        result["catalog_id"] = catalog_id
        return result

    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)
        stats.finish()
        stats.record_error("", "fatal", str(e))
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
                "statistics": stats.to_dict(),
            },
        )
        raise


@app.task(name="process_file_batch")
def process_file_batch_task(
    catalog_id: str,
    file_paths: List[str],
    batch_num: int,
    total_batches: int,
) -> Dict[str, Any]:
    """
    Process a batch of files in parallel.

    This task can run in parallel with other batch tasks.

    Args:
        catalog_id: UUID of the catalog
        file_paths: List of file paths to process
        batch_num: Batch number (for logging)
        total_batches: Total number of batches

    Returns:
        Results dictionary with processed file records
    """
    logger.info(
        f"Processing batch {batch_num}/{total_batches} ({len(file_paths)} files)"
    )

    results = []
    processed = 0
    skipped = 0

    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        result = _process_file_worker(file_path)

        if result is not None:
            image_record, file_size = result
            processed += 1
            # Convert image to dict for JSON serialization
            from ..db.serializers import serialize_image_record

            results.append(
                {
                    "image_record": serialize_image_record(image_record),
                    "file_size": file_size,
                }
            )
        else:
            skipped += 1

    logger.info(
        f"Batch {batch_num}/{total_batches} complete: "
        f"{processed} processed, {skipped} skipped"
    )

    return {
        "batch_num": batch_num,
        "results": results,
        "processed": processed,
        "skipped": skipped,
    }


@app.task(bind=True, base=ProgressTask, name="finalize_analysis")
def finalize_analysis_task(
    self: ProgressTask,
    batch_results: List[Dict[str, Any]],
    catalog_id: str,
) -> Dict[str, Any]:
    """
    Finalize analysis by saving all results to catalog.

    This task runs after all batch processing tasks complete.

    Args:
        batch_results: List of results from all batch tasks
        catalog_id: UUID of the catalog

    Returns:
        Final analysis results
    """
    logger.info(f"Finalizing analysis for catalog {catalog_id}")

    try:
        from ..db.serializers import deserialize_image_record

        self.update_progress(0, 1, "Finalizing analysis...", {"phase": "finalizing"})

        files_added = 0
        files_skipped = 0
        total_processed = 0

        with CatalogDatabase(catalog_id) as db:
            # Process all batch results
            for batch_result in batch_results:
                total_processed += batch_result["processed"]
                files_skipped += batch_result["skipped"]

                for result_data in batch_result["results"]:
                    image_record = deserialize_image_record(result_data["image_record"])

                    # Check if already in catalog
                    existing_image = db.get_image(image_record.id)

                    if not existing_image:
                        # Add to catalog
                        db.add_image(image_record)
                        files_added += 1

            db.save()

        logger.info(
            f"Analysis finalized: {total_processed} processed, "
            f"{files_added} added, {files_skipped} skipped"
        )

        self.update_progress(
            100,
            100,
            f"Analysis complete: {files_added} files added",
            {"phase": "complete"},
        )

        return {
            "status": "completed",
            "catalog_id": catalog_id,
            "total_files": total_processed,
            "processed": total_processed,
            "files_added": files_added,
            "files_skipped": files_skipped,
        }

    except Exception as e:
        logger.error(f"Finalization failed: {e}", exc_info=True)
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
    catalog_id: str,
    destination_path: str,
    strategy: str,
    simulate: bool = True,
) -> Dict[str, Any]:
    """
    Organize files in catalog based on a given strategy.

    Args:
        catalog_id: UUID of the catalog
        destination_path: Path to organize files into
        strategy: Organization strategy (e.g., "date_based")
        simulate: If True, only simulate organization without moving files

    Returns:
        Organization results
    """
    logger.info(f"Starting catalog organization: {catalog_id} to {destination_path}")

    try:
        dest_dir = Path(destination_path)

        self.update_progress(0, 1, "Initializing organization...", {"phase": "init"})

        with CatalogDatabase(catalog_id) as db:
            organizer = FileOrganizer(db, dest_dir)
            # Create default OrganizationStrategy instance
            org_strategy = OrganizationStrategy()

            results = organizer.organize_files(org_strategy, simulate=simulate)

        self.update_progress(
            100, 100, "Organization complete", {"phase": "complete", "results": results}
        )

        return {
            "status": "completed",
            "catalog_id": catalog_id,
            "destination_path": str(dest_dir),
            "strategy": strategy,
            "simulate": simulate,
            "results": results,
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


@app.task(bind=True, base=ProgressTask, name="detect_duplicates")
def detect_duplicates_task(
    self: ProgressTask,
    catalog_id: str,
    similarity_threshold: int = 5,
    recompute_hashes: bool = False,
) -> Dict[str, Any]:
    """
    Detect duplicate images in a catalog using perceptual hashing.

    This task:
    - Automatically detects and uses GPU acceleration if available
    - Computes perceptual hashes for all images (if not already computed)
    - Finds exact duplicates (same checksum)
    - Finds similar images (similar perceptual hash within threshold)
    - Scores quality and selects primary (best) image in each group
    - Saves duplicate groups to database
    - Tracks timing metrics for adaptive batch sizing

    Args:
        catalog_id: UUID of the catalog to analyze
        similarity_threshold: Maximum Hamming distance for similar images (default: 5)
        recompute_hashes: Force recomputation of perceptual hashes

    Returns:
        Dictionary with duplicate detection statistics
    """
    from .job_metrics import JobMetricsTracker, check_gpu_available, get_gpu_info

    logger.info(f"Starting duplicate detection for catalog {catalog_id}")

    # Check GPU availability
    use_gpu = check_gpu_available()
    gpu_info = get_gpu_info() if use_gpu else None
    if use_gpu:
        logger.info(f"GPU acceleration enabled: {gpu_info['device_name']}")
    else:
        logger.info("GPU not available, using CPU processing")

    try:
        self.update_progress(
            0, 1, "Initializing duplicate detection...", {"phase": "init"}
        )

        with CatalogDatabase(catalog_id) as db:
            # Initialize metrics tracker
            metrics = JobMetricsTracker(db.session, catalog_id)

            # Get total image count for progress reporting
            result = db.session.execute(
                text("SELECT COUNT(*) FROM images WHERE catalog_id = :catalog_id"),
                {"catalog_id": catalog_id},
            )
            total_images = result.scalar() or 0

            # Plan batches based on historical timing data
            batch_plan = metrics.plan_batches(
                "hash_computation", total_images, use_gpu=use_gpu
            )

            self.update_progress(
                0,
                total_images,
                f"Analyzing {total_images} images "
                f"(est. {batch_plan.estimated_total_duration:.0f}s)...",
                {
                    "phase": "analyzing",
                    "total_images": total_images,
                    "use_gpu": use_gpu,
                    "estimated_duration": batch_plan.estimated_total_duration,
                },
            )

            # Create progress callback that updates Celery task state
            def progress_callback(current: int, total: int, message: str) -> None:
                self.update_progress(
                    current,
                    total,
                    message,
                    {"phase": "hashing", "use_gpu": use_gpu},
                )

            # Initialize duplicate detector with GPU support and progress callback
            # Note: num_workers=1 for CPU because Celery workers are daemon processes
            detector = DuplicateDetector(
                catalog=db,
                similarity_threshold=similarity_threshold,
                num_workers=1 if not use_gpu else None,
                use_gpu=use_gpu,
                progress_callback=progress_callback,
            )

            # Phase 1: Compute perceptual hashes (with timing)
            self.update_progress(
                0,
                total_images,
                "Computing perceptual hashes...",
                {"phase": "hashing", "use_gpu": use_gpu},
            )

            # Track timing for adaptive batch sizing in future runs
            with metrics.timed_operation("hash_computation", total_images, use_gpu):
                duplicate_groups = detector.detect_duplicates(
                    recompute_hashes=recompute_hashes
                )

            self.update_progress(
                total_images // 2,
                total_images,
                f"Found {len(duplicate_groups)} duplicate groups, saving...",
                {"phase": "saving", "groups_found": len(duplicate_groups)},
            )

            # Save duplicate groups to database
            detector.save_duplicate_groups()

            # Also save any problematic files encountered
            detector.save_problematic_files()

            # Commit changes
            db.save()

            # Get final statistics
            stats = detector.get_statistics()

            self.update_progress(
                total_images,
                total_images,
                f"Complete: {stats['total_groups']} groups, "
                f"{stats['total_redundant']} redundant images",
                {"phase": "complete", "statistics": stats},
            )

            logger.info(
                f"Duplicate detection complete for {catalog_id}: "
                f"{stats['total_groups']} groups, {stats['total_redundant']} redundant "
                f"(GPU: {use_gpu})"
            )

            return {
                "status": "completed",
                "catalog_id": catalog_id,
                "total_images": total_images,
                "similarity_threshold": similarity_threshold,
                "use_gpu": use_gpu,
                "gpu_info": gpu_info,
                **stats,
            }

    except Exception as e:
        logger.error(f"Duplicate detection failed: {e}", exc_info=True)
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
    catalog_id: str,
    sizes: Optional[List[int]] = None,
    quality: int = 85,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Generate thumbnails for catalog images.

    Args:
        catalog_id: UUID of the catalog
        sizes: List of thumbnail sizes (default: [256, 512])
        quality: JPEG quality (1-100)
        force: Regenerate existing thumbnails

    Returns:
        Thumbnail generation results
    """
    logger.info(f"Starting thumbnail generation for catalog {catalog_id}")

    if sizes is None:
        sizes = [256, 512]

    try:
        self.update_progress(
            0, 1, "Initializing thumbnail generation...", {"phase": "init"}
        )

        # For PostgreSQL, we need a thumbnail directory
        # Use /tmp/vam-thumbnails/{catalog_id}/ for now
        thumbnail_base = Path(f"/tmp/vam-thumbnails/{catalog_id}")
        thumbnail_base.mkdir(parents=True, exist_ok=True)

        with CatalogDatabase(catalog_id) as db:
            images_dict = db.get_all_images()
            image_records = list(images_dict.values())
            total_images = len(image_records)
            generated_count = 0
            skipped_count = 0

            for i, record in enumerate(image_records):
                self.update_progress(
                    i,
                    total_images,
                    f"Generating thumbnail for {record.source_path.name}...",
                    {"phase": "generating", "current_file": record.source_path.name},
                )

                for size in sizes:
                    thumbnail_path = thumbnail_base / f"{record.id}_{size}.jpg"

                    if thumbnail_path.exists() and not force:
                        skipped_count += 1
                        continue

                    try:
                        generate_thumbnail(
                            record.source_path, thumbnail_path, size, quality
                        )
                        generated_count += 1
                    except Exception as thumb_e:
                        logger.warning(
                            f"Failed to generate thumbnail for {record.source_path}: {thumb_e}"
                        )

            self.update_progress(
                total_images,
                total_images,
                "Thumbnail generation complete",
                {
                    "phase": "complete",
                    "generated": generated_count,
                    "skipped": skipped_count,
                },
            )

            return {
                "status": "completed",
                "catalog_id": catalog_id,
                "generated_count": generated_count,
                "skipped_count": skipped_count,
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


@app.task(bind=True, base=ProgressTask, name="auto_tag")
def auto_tag_task(
    self: ProgressTask,
    catalog_id: str,
    backend: str = "openclip",
    model: Optional[str] = None,
    threshold: float = 0.25,
    max_tags: int = 10,
    batch_size: int = 32,
    continue_pipeline: bool = False,
) -> Dict[str, Any]:
    """
    Automatically tag images in a catalog using AI models.

    This task:
    - Uses OpenCLIP (fast, batch processing) or Ollama (detailed, slower) backends
    - Tags images based on predefined taxonomy
    - Stores tags in the database
    - Optionally continues to the next pipeline step (duplicate detection)

    Args:
        catalog_id: UUID of the catalog to tag
        backend: "openclip" or "ollama"
        model: Model name (backend-specific, e.g., "ViT-B-32" or "llava")
        threshold: Minimum confidence threshold (0.0-1.0)
        max_tags: Maximum tags per image
        batch_size: Batch size for OpenCLIP processing
        continue_pipeline: Whether to trigger duplicate detection after completion

    Returns:
        Dictionary with tagging statistics
    """
    from .job_metrics import check_gpu_available, get_gpu_info

    logger.info(
        f"Starting auto-tagging for catalog {catalog_id} with {backend} backend"
    )

    # Check GPU availability for OpenCLIP
    use_gpu = check_gpu_available() if backend == "openclip" else False
    gpu_info = get_gpu_info() if use_gpu else None
    if use_gpu:
        logger.info(f"GPU acceleration enabled: {gpu_info['device_name']}")

    try:
        self.update_progress(
            0, 1, "Initializing auto-tagging...", {"phase": "init", "backend": backend}
        )

        # Check backend availability
        from ..analysis.image_tagger import ImageTagger, check_backends_available

        backends_status = check_backends_available()

        if backend == "openclip" and not backends_status.get("openclip"):
            raise RuntimeError(
                "OpenCLIP backend not available. Install with: pip install open-clip-torch"
            )
        if backend == "ollama" and not backends_status.get("ollama"):
            raise RuntimeError(
                "Ollama backend not available. Ensure Ollama is running with a vision model."
            )

        with CatalogDatabase(catalog_id) as db:
            # Get images that need tagging
            result = db.session.execute(
                text(
                    """
                    SELECT id, source_path FROM images
                    WHERE catalog_id = :catalog_id
                    AND (tags IS NULL OR tags = '[]' OR tags = '{}')
                """
                ),
                {"catalog_id": catalog_id},
            )
            images_to_tag = result.fetchall()
            total_images = len(images_to_tag)

            if total_images == 0:
                # Check if all images are already tagged
                result = db.session.execute(
                    text("SELECT COUNT(*) FROM images WHERE catalog_id = :catalog_id"),
                    {"catalog_id": catalog_id},
                )
                total_in_catalog = result.scalar() or 0

                if total_in_catalog > 0:
                    return {
                        "status": "completed",
                        "catalog_id": catalog_id,
                        "message": f"All {total_in_catalog} images already tagged",
                        "images_tagged": 0,
                        "images_skipped": total_in_catalog,
                    }
                else:
                    return {
                        "status": "completed",
                        "catalog_id": catalog_id,
                        "message": "No images in catalog",
                        "images_tagged": 0,
                        "images_skipped": 0,
                    }

            self.update_progress(
                0,
                total_images,
                f"Tagging {total_images} images with {backend}...",
                {
                    "phase": "tagging",
                    "backend": backend,
                    "total_images": total_images,
                    "use_gpu": use_gpu,
                },
            )

            # Initialize tagger
            device = "cuda" if use_gpu else "cpu"
            tagger = ImageTagger(
                backend=backend,
                model=model,
                device=device if backend == "openclip" else None,
            )

            # Process images
            tagged_count = 0
            failed_count = 0
            tag_stats: Dict[str, int] = {}

            # Process in batches for OpenCLIP
            if backend == "openclip":
                for batch_start in range(0, total_images, batch_size):
                    batch_end = min(batch_start + batch_size, total_images)
                    batch = images_to_tag[batch_start:batch_end]
                    batch_paths = [Path(row[1]) for row in batch]
                    batch_ids = [row[0] for row in batch]

                    self.update_progress(
                        batch_start,
                        total_images,
                        f"Tagging batch {batch_start // batch_size + 1}...",
                        {
                            "phase": "tagging",
                            "current_batch": batch_start // batch_size + 1,
                        },
                    )

                    try:
                        # Tag batch
                        results = tagger.tag_batch(
                            batch_paths,
                            threshold=threshold,
                            max_tags=max_tags,
                        )

                        # Update database
                        for img_id, img_path in zip(batch_ids, batch_paths):
                            tags = results.get(img_path, [])
                            if tags:
                                tag_data = [
                                    {
                                        "name": t.tag_name,
                                        "confidence": t.confidence,
                                        "category": t.category,
                                    }
                                    for t in tags
                                ]
                                db.session.execute(
                                    text(
                                        """
                                        UPDATE images SET tags = :tags
                                        WHERE id = :image_id AND catalog_id = :catalog_id
                                    """
                                    ),
                                    {
                                        "tags": json.dumps(tag_data),
                                        "image_id": str(img_id),
                                        "catalog_id": catalog_id,
                                    },
                                )
                                tagged_count += 1
                                for t in tags:
                                    tag_stats[t.tag_name] = (
                                        tag_stats.get(t.tag_name, 0) + 1
                                    )
                            else:
                                # Mark as processed but no tags found
                                db.session.execute(
                                    text(
                                        """
                                        UPDATE images SET tags = :tags
                                        WHERE id = :image_id AND catalog_id = :catalog_id
                                    """
                                    ),
                                    {
                                        "tags": json.dumps([]),
                                        "image_id": str(img_id),
                                        "catalog_id": catalog_id,
                                    },
                                )

                        db.session.commit()

                    except Exception as batch_e:
                        logger.warning(f"Batch tagging failed: {batch_e}")
                        failed_count += len(batch)

            else:
                # Process one at a time for Ollama
                for i, (img_id, source_path) in enumerate(images_to_tag):
                    self.update_progress(
                        i,
                        total_images,
                        f"Tagging {Path(source_path).name}...",
                        {"phase": "tagging", "current_file": Path(source_path).name},
                    )

                    try:
                        tags = tagger.tag_image(
                            source_path,
                            threshold=threshold,
                            max_tags=max_tags,
                        )

                        if tags:
                            tag_data = [
                                {
                                    "name": t.tag_name,
                                    "confidence": t.confidence,
                                    "category": t.category,
                                }
                                for t in tags
                            ]
                            db.session.execute(
                                text(
                                    """
                                    UPDATE images SET tags = :tags
                                    WHERE id = :image_id AND catalog_id = :catalog_id
                                """
                                ),
                                {
                                    "tags": json.dumps(tag_data),
                                    "image_id": str(img_id),
                                    "catalog_id": catalog_id,
                                },
                            )
                            tagged_count += 1
                            for t in tags:
                                tag_stats[t.tag_name] = tag_stats.get(t.tag_name, 0) + 1
                        else:
                            db.session.execute(
                                text(
                                    """
                                    UPDATE images SET tags = :tags
                                    WHERE id = :image_id AND catalog_id = :catalog_id
                                """
                                ),
                                {
                                    "tags": json.dumps([]),
                                    "image_id": str(img_id),
                                    "catalog_id": catalog_id,
                                },
                            )

                        if i % 10 == 0:
                            db.session.commit()

                    except Exception as img_e:
                        logger.warning(f"Failed to tag {source_path}: {img_e}")
                        failed_count += 1

                db.session.commit()

            # Get top tags
            top_tags = sorted(tag_stats.items(), key=lambda x: x[1], reverse=True)[:20]

            self.update_progress(
                total_images,
                total_images,
                f"Complete: tagged {tagged_count} images",
                {
                    "phase": "complete",
                    "tagged_count": tagged_count,
                    "failed_count": failed_count,
                },
            )

            logger.info(
                f"Auto-tagging complete for {catalog_id}: "
                f"{tagged_count} tagged, {failed_count} failed"
            )

            result = {
                "status": "completed",
                "catalog_id": catalog_id,
                "backend": backend,
                "use_gpu": use_gpu,
                "images_tagged": tagged_count,
                "images_failed": failed_count,
                "total_images": total_images,
                "unique_tags_applied": len(tag_stats),
                "top_tags": top_tags,
            }

            # Continue pipeline if requested
            if continue_pipeline:
                logger.info(
                    f"Continuing pipeline: starting duplicate detection for {catalog_id}"
                )
                detect_duplicates_task.delay(
                    catalog_id=catalog_id,
                    similarity_threshold=5,
                    recompute_hashes=False,
                )
                result["next_job"] = "detect_duplicates"

            return result

    except Exception as e:
        logger.error(f"Auto-tagging failed: {e}", exc_info=True)
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise
