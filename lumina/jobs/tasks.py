"""
Celery tasks for background processing of Lumina operations.

These tasks wrap operations and provide progress tracking through Celery's state mechanism.
Updated for PostgreSQL backend.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from celery import Task
from sqlalchemy import text

from lumina.core.types import FileType, ImageRecord, ImageStatus

from ..analysis.burst_detector import BurstDetector, BurstGroup, ImageInfo
from ..analysis.scanner import _process_file_worker
from ..db import CatalogDB as CatalogDatabase
from ..organization import FileOrganizer, OrganizationStrategy
from ..shared.media_utils import get_file_type
from ..shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path

# Import app here so that @app.task decorators can find it
from .celery_app import app
from .scan_stats import ScanStatistics

logger = logging.getLogger(__name__)


def _store_image_tags(
    db: CatalogDatabase,
    catalog_id: str,
    image_id: str,
    tags: List,
    source: str,
) -> int:
    """Store tags for an image in the proper relational schema.

    Creates Tag entries if they don't exist, then creates ImageTag entries
    linking the image to its tags.

    Args:
        db: CatalogDatabase session
        catalog_id: The catalog UUID
        image_id: The image ID
        tags: List of TagResult objects with tag_name, confidence, category, source,
              openclip_confidence, and ollama_confidence attributes
        source: The tagging source ('openclip', 'ollama', or 'combined')

    Returns:
        Number of tags stored
    """
    if not tags:
        return 0

    stored_count = 0

    for tag in tags:
        try:
            # Get category as string (handle enum or string)
            category = getattr(tag, "category", None)
            if category is not None and hasattr(category, "value"):
                category = category.value  # Convert enum to string

            # Get or create tag in the tags table
            result = db.session.execute(
                text(
                    """
                    INSERT INTO tags (catalog_id, name, category, created_at)
                    VALUES (:catalog_id, :name, :category, NOW())
                    ON CONFLICT (catalog_id, name) DO UPDATE SET catalog_id = tags.catalog_id
                    RETURNING id
                """
                ),
                {
                    "catalog_id": catalog_id,
                    "name": tag.tag_name,
                    "category": category,
                },
            )
            tag_id = result.scalar()

            # Insert or update image_tag relationship
            db.session.execute(
                text(
                    """
                    INSERT INTO image_tags (image_id, tag_id, confidence, source,
                                           openclip_confidence, ollama_confidence, created_at)
                    VALUES (:image_id, :tag_id, :confidence, :source,
                            :openclip_confidence, :ollama_confidence, NOW())
                    ON CONFLICT (image_id, tag_id) DO UPDATE SET
                        confidence = :confidence,
                        source = :source,
                        openclip_confidence = COALESCE(:openclip_confidence, image_tags.openclip_confidence),
                        ollama_confidence = COALESCE(:ollama_confidence, image_tags.ollama_confidence)
                """
                ),
                {
                    "image_id": image_id,
                    "tag_id": tag_id,
                    "confidence": tag.confidence,
                    "source": getattr(tag, "source", source),
                    "openclip_confidence": getattr(tag, "openclip_confidence", None),
                    "ollama_confidence": getattr(tag, "ollama_confidence", None),
                },
            )
            stored_count += 1
        except Exception as e:
            logger.warning(
                f"Failed to store tag {tag.tag_name} for image {image_id}: {e}"
            )
            # Rollback to clear the failed transaction state
            if db.session:
                db.session.rollback()

    return stored_count


class ProgressTask(Task):
    """
    Base task class with progress reporting.

    Provides utilities for reporting progress to the result backend
    and publishing to Redis for real-time updates.
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

        This updates both:
        1. Celery task state (for result backend)
        2. Redis pub/sub channel (for real-time frontend updates)

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

        # Update Celery state
        self.update_state(
            state="PROGRESS",
            meta=meta,
        )

        # Publish to Redis for real-time updates (non-blocking, fails silently)
        try:
            from .progress_publisher import publish_progress

            job_id = self.request.id if self.request else None
            if job_id:
                publish_progress(
                    job_id=job_id,
                    state="PROGRESS",
                    current=current,
                    total=total,
                    message=message,
                    extra=extra,
                )
        except Exception as e:
            # Don't let Redis failures affect task execution
            logger.debug(f"Failed to publish progress to Redis: {e}")


class CoordinatorTask(ProgressTask):
    """
    Base task class for coordinator tasks in the parallel processing pattern.

    Coordinator tasks orchestrate parallel work by:
    1. Creating batches
    2. Dispatching worker tasks via Celery chord
    3. Setting up finalizer callback

    These tasks should NOT update the parent Job status to SUCCESS when they
    complete, because workers are still running. Only the finalizer should
    update the Job to SUCCESS.

    The is_coordinator flag tells signal handlers in celery_app.py to skip
    updating Job.status for these tasks.
    """

    is_coordinator = True  # Flag for signal handlers to skip Job updates


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
                                    "skipped": stats.total_skipped,
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
                                    # Rollback to clear failed transaction state (PostgreSQL requirement)
                                    if db.session:
                                        db.session.rollback()
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
                                        # Rollback to clear failed transaction state (PostgreSQL requirement)
                                        if db.session:
                                            db.session.rollback()
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
                                        "skipped": stats.total_skipped,
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


def _get_checkpoint(db: CatalogDatabase, catalog_id: str, job_id: str) -> Optional[int]:
    """Get the last checkpoint (offset) for a tagging job.

    Args:
        db: Database connection
        catalog_id: The catalog ID
        job_id: The Celery task ID

    Returns:
        The offset to resume from, or None if no checkpoint exists
    """
    result = db.session.execute(
        text(
            """
            SELECT value FROM config
            WHERE catalog_id = :catalog_id AND key = :key
        """
        ),
        {"catalog_id": catalog_id, "key": f"auto_tag_checkpoint_{job_id}"},
    )
    row = result.fetchone()
    if row:
        import json

        # Handle both JSONB (returns dict) and TEXT (returns string) column types
        return row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return None


def _save_checkpoint(
    db: CatalogDatabase, catalog_id: str, job_id: str, offset: int
) -> None:
    """Save a checkpoint for resuming the tagging job.

    Args:
        db: Database connection
        catalog_id: The catalog ID
        job_id: The Celery task ID
        offset: The number of images processed so far
    """
    import json

    db.session.execute(
        text(
            """
            INSERT INTO config (catalog_id, key, value, updated_at)
            VALUES (:catalog_id, :key, :value, NOW())
            ON CONFLICT (catalog_id, key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = EXCLUDED.updated_at
        """
        ),
        {
            "catalog_id": catalog_id,
            "key": f"auto_tag_checkpoint_{job_id}",
            "value": json.dumps(offset),
        },
    )
    db.session.commit()


def _clear_checkpoint(db: CatalogDatabase, catalog_id: str, job_id: str) -> None:
    """Clear the checkpoint after job completion.

    Args:
        db: Database connection
        catalog_id: The catalog ID
        job_id: The Celery task ID
    """
    db.session.execute(
        text(
            """
            DELETE FROM config
            WHERE catalog_id = :catalog_id AND key = :key
        """
        ),
        {"catalog_id": catalog_id, "key": f"auto_tag_checkpoint_{job_id}"},
    )
    db.session.commit()


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
    max_images: Optional[int] = None,
    tag_mode: str = "untagged_only",
    resume_from_job: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Automatically tag images in a catalog using AI models.

    This task:
    - Uses OpenCLIP (fast, batch processing), Ollama (detailed, slower), or
      combined (both backends with weighted confidence) backends
    - Tags images based on predefined taxonomy
    - Stores tags in the database with source and per-backend confidence
    - Saves progress checkpoints for resumability
    - Optionally continues to the next pipeline step (duplicate detection)

    Args:
        catalog_id: UUID of the catalog to tag
        backend: "openclip", "ollama", or "combined"
        model: Model name (backend-specific, e.g., "ViT-B-32" or "llava")
        threshold: Minimum confidence threshold (0.0-1.0)
        max_tags: Maximum tags per image
        batch_size: Batch size for OpenCLIP/combined processing
        continue_pipeline: Whether to trigger duplicate detection after completion
        max_images: Maximum number of images to tag (for testing, None = all)
        tag_mode: "untagged_only" to skip already tagged, "all" to retag everything
        resume_from_job: Job ID to resume from (uses checkpoint from that job)

    Returns:
        Dictionary with tagging statistics
    """
    from .job_metrics import check_gpu_available, get_gpu_info

    logger.info(
        f"Starting auto-tagging for catalog {catalog_id} with {backend} backend "
        f"(mode={tag_mode})"
    )

    # Check GPU availability for OpenCLIP or combined backend
    use_gpu = check_gpu_available() if backend in ("openclip", "combined") else False
    gpu_info = get_gpu_info() if use_gpu else None
    if use_gpu:
        logger.info(f"GPU acceleration enabled: {gpu_info['device_name']}")

    try:
        self.update_progress(
            0, 1, "Initializing auto-tagging...", {"phase": "init", "backend": backend}
        )

        # Check backend availability
        from ..analysis.image_tagger import (
            CombinedTagger,
            ImageTagger,
            check_backends_available,
        )

        backends_status = check_backends_available()

        if backend == "openclip" and not backends_status.get("openclip"):
            raise RuntimeError(
                "OpenCLIP backend not available. Install with: pip install open-clip-torch"
            )
        if backend == "ollama" and not backends_status.get("ollama"):
            raise RuntimeError(
                "Ollama backend not available. Ensure Ollama is running with a vision model."
            )
        if backend == "combined":
            if not backends_status.get("openclip"):
                raise RuntimeError(
                    "Combined backend requires OpenCLIP. Install with: pip install open-clip-torch"
                )
            if not backends_status.get("ollama"):
                raise RuntimeError(
                    "Combined backend requires Ollama. Ensure Ollama is running with a vision model."
                )

        with CatalogDatabase(catalog_id) as db:
            # Get images based on tag_mode
            # - untagged_only: only images with no entries in image_tags table
            # - all: all images in the catalog (will replace existing AI tags)
            limit_clause = f"LIMIT {max_images}" if max_images else ""

            if tag_mode == "untagged_only":
                # Only images without any tags (exclude videos)
                result = db.session.execute(
                    text(
                        f"""
                        SELECT i.id, i.source_path FROM images i
                        WHERE i.catalog_id = :catalog_id
                        AND i.file_type = 'image'
                        AND NOT EXISTS (
                            SELECT 1 FROM image_tags it WHERE it.image_id = i.id
                        )
                        {limit_clause}
                    """
                    ),
                    {"catalog_id": catalog_id},
                )
            else:
                # All images - for retagging (exclude videos)
                result = db.session.execute(
                    text(
                        f"""
                        SELECT i.id, i.source_path FROM images i
                        WHERE i.catalog_id = :catalog_id
                        AND i.file_type = 'image'
                        {limit_clause}
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
            if backend == "combined":
                tagger = CombinedTagger(
                    openclip_model=model or "ViT-B-32",
                    ollama_model="llava",
                    device=device,
                    ollama_host=os.environ.get("OLLAMA_HOST"),
                )
            else:
                tagger = ImageTagger(
                    backend=backend,
                    model=model,
                    device=device if backend == "openclip" else None,
                )

            # Process images
            tagged_count = 0
            failed_count = 0
            tag_stats: Dict[str, int] = {}

            # Get job ID for checkpointing
            job_id = resume_from_job or self.request.id or "unknown"

            # Check for existing checkpoint to resume from
            start_offset = 0
            if resume_from_job:
                checkpoint = _get_checkpoint(db, catalog_id, resume_from_job)
                if checkpoint is not None:
                    start_offset = checkpoint
                    logger.info(
                        f"Resuming from checkpoint: {start_offset}/{total_images} images already processed"
                    )
                    self.update_progress(
                        start_offset,
                        total_images,
                        f"Resuming from checkpoint ({start_offset} already tagged)...",
                        {"phase": "resuming", "checkpoint": start_offset},
                    )

            # Process in batches for OpenCLIP or combined backend
            if backend in ("openclip", "combined"):
                for batch_start in range(start_offset, total_images, batch_size):
                    batch_end = min(batch_start + batch_size, total_images)
                    batch = images_to_tag[batch_start:batch_end]
                    batch_paths = [Path(row[1]) for row in batch]
                    batch_ids = [row[0] for row in batch]

                    phase_msg = (
                        "OpenCLIP+Ollama" if backend == "combined" else "OpenCLIP"
                    )
                    self.update_progress(
                        batch_start,
                        total_images,
                        f"Tagging batch {batch_start // batch_size + 1} ({phase_msg})...",
                        {
                            "phase": "tagging",
                            "current_batch": batch_start // batch_size + 1,
                            "backend": backend,
                        },
                    )

                    try:
                        # Tag batch - combined backend has progress_callback
                        if backend == "combined":
                            # Capture loop variables to avoid B023 closure issue
                            _batch_start = batch_start
                            _batch_size = batch_size

                            def progress_cb(
                                current: int,
                                total: int,
                                phase: str,
                                _bs: int = _batch_start,
                                _bsz: int = _batch_size,
                            ) -> None:
                                self.update_progress(
                                    _bs + current,
                                    total_images,
                                    f"Batch {_bs // _bsz + 1}: {phase} {current}/{total}...",
                                    {"phase": "tagging", "sub_phase": phase},
                                )

                            results = tagger.tag_batch(
                                batch_paths,
                                threshold=threshold,
                                max_tags=max_tags,
                                progress_callback=progress_cb,
                            )
                        else:
                            results = tagger.tag_batch(
                                batch_paths,
                                threshold=threshold,
                                max_tags=max_tags,
                            )

                        # Update database using proper relational schema
                        for img_id, img_path in zip(batch_ids, batch_paths):
                            tags = results.get(img_path, [])
                            if tags:
                                # Store tags in the relational image_tags table
                                stored = _store_image_tags(
                                    db, catalog_id, str(img_id), tags, backend
                                )
                                if stored > 0:
                                    tagged_count += 1
                                    for t in tags:
                                        tag_stats[t.tag_name] = (
                                            tag_stats.get(t.tag_name, 0) + 1
                                        )

                            # Save CLIP embedding for semantic search (OpenCLIP/combined only)
                            if backend in ("openclip", "combined") and hasattr(
                                tagger, "get_embedding"
                            ):
                                try:
                                    embedding = tagger.get_embedding(img_path)
                                    db.session.execute(
                                        text(
                                            """
                                            UPDATE images
                                            SET clip_embedding = :embedding
                                            WHERE id = :image_id
                                        """
                                        ),
                                        {
                                            "image_id": str(img_id),
                                            "embedding": embedding,
                                        },
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to save embedding for {img_id}: {e}"
                                    )

                        db.session.commit()

                        # Save checkpoint after each batch
                        _save_checkpoint(db, catalog_id, job_id, batch_end)
                        logger.debug(f"Checkpoint saved: {batch_end}/{total_images}")

                    except Exception as batch_e:
                        logger.warning(f"Batch tagging failed: {batch_e}")
                        failed_count += len(batch)
                        # Still save checkpoint so we don't reprocess failed batch
                        _save_checkpoint(db, catalog_id, job_id, batch_end)

            elif backend == "ollama":
                # Process one at a time for Ollama
                for i, (img_id, source_path) in enumerate(images_to_tag):
                    # Skip already processed images when resuming
                    if i < start_offset:
                        continue

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
                            # Store tags in the relational image_tags table
                            stored = _store_image_tags(
                                db, catalog_id, str(img_id), tags, "ollama"
                            )
                            if stored > 0:
                                tagged_count += 1
                                for t in tags:
                                    tag_stats[t.tag_name] = (
                                        tag_stats.get(t.tag_name, 0) + 1
                                    )

                        # Commit and checkpoint every 10 images
                        if (i + 1) % 10 == 0:
                            db.session.commit()
                            _save_checkpoint(db, catalog_id, job_id, i + 1)
                            logger.debug(f"Checkpoint saved: {i + 1}/{total_images}")

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

            # Clear checkpoint on successful completion
            _clear_checkpoint(db, catalog_id, job_id)

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
                from .parallel_duplicates import duplicates_coordinator_task

                logger.info(
                    f"Continuing pipeline: starting duplicate detection for {catalog_id}"
                )
                duplicates_coordinator_task.delay(
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


@app.task(bind=True, base=ProgressTask, name="detect_bursts")
def detect_bursts_task(
    self: ProgressTask,
    catalog_id: str,
    gap_threshold: float = 2.0,
    min_burst_size: int = 3,
) -> Dict[str, Any]:
    """Detect burst sequences in a catalog.

    Args:
        catalog_id: Catalog ID to process
        gap_threshold: Maximum seconds between burst images
        min_burst_size: Minimum images to form a burst

    Returns:
        Dict with detection results
    """
    job_id = self.request.id if hasattr(self.request, "id") else "unknown"
    logger.info(f"[{job_id}] Starting burst detection for catalog {catalog_id}")

    try:
        self.update_progress(0, 1, "Initializing burst detection...", {"phase": "init"})

        with CatalogDatabase(catalog_id) as db:
            # Clear existing bursts for this catalog
            db.session.execute(
                text("DELETE FROM bursts WHERE catalog_id = :catalog_id"),
                {"catalog_id": catalog_id},
            )
            db.session.commit()

            # Load images with timestamps - only those with proper date extraction
            # (confidence >= 70 means EXIF or filename date, not just directory fallback)
            result = db.session.execute(
                text(
                    """
                    SELECT id,
                           (dates->>'selected_date')::timestamp as date_taken,
                           metadata->>'camera_make' as camera_make,
                           metadata->>'camera_model' as camera_model,
                           quality_score
                    FROM images
                    WHERE catalog_id = :catalog_id
                    AND dates->>'selected_date' IS NOT NULL
                    AND (dates->>'confidence')::int >= 70
                    ORDER BY (dates->>'selected_date')::timestamp
                """
                ),
                {"catalog_id": catalog_id},
            )

            images = [
                ImageInfo(
                    image_id=str(row[0]),
                    timestamp=row[1],
                    camera_make=row[2],
                    camera_model=row[3],
                    quality_score=row[4] or 0.0,
                )
                for row in result.fetchall()
            ]

            logger.info(f"[{job_id}] Loaded {len(images)} images with timestamps")

            self.update_progress(
                1,
                2,
                f"Detecting bursts from {len(images)} images...",
                {"phase": "detecting", "images_loaded": len(images)},
            )

            # Detect bursts
            detector = BurstDetector(
                gap_threshold_seconds=gap_threshold,
                min_burst_size=min_burst_size,
            )
            bursts = detector.detect_bursts(images)

            logger.info(f"[{job_id}] Detected {len(bursts)} bursts")

            self.update_progress(
                1,
                2,
                f"Saving {len(bursts)} bursts to database...",
                {"phase": "saving", "bursts_detected": len(bursts)},
            )

            # Save bursts to database
            for burst in bursts:
                burst_id = str(uuid.uuid4())

                # Insert burst record
                db.session.execute(
                    text(
                        """
                        INSERT INTO bursts (
                            id, catalog_id, image_count, start_time, end_time,
                            duration_seconds, camera_make, camera_model,
                            best_image_id, selection_method, created_at
                        ) VALUES (
                            :id, :catalog_id, :image_count, :start_time, :end_time,
                            :duration, :camera_make, :camera_model,
                            :best_image_id, :selection_method, NOW()
                        )
                    """
                    ),
                    {
                        "id": burst_id,
                        "catalog_id": catalog_id,
                        "image_count": burst.image_count,
                        "start_time": burst.start_time,
                        "end_time": burst.end_time,
                        "duration": burst.duration_seconds,
                        "camera_make": burst.camera_make,
                        "camera_model": burst.camera_model,
                        "best_image_id": burst.best_image_id,
                        "selection_method": burst.selection_method,
                    },
                )

                # Update images with burst_id and sequence
                for seq, img in enumerate(burst.images):
                    db.session.execute(
                        text(
                            """
                            UPDATE images
                            SET burst_id = :burst_id, burst_sequence = :seq
                            WHERE id = :image_id
                        """
                        ),
                        {
                            "burst_id": burst_id,
                            "image_id": img.image_id,
                            "seq": seq,
                        },
                    )

            db.session.commit()

            self.update_progress(
                2,
                2,
                f"Complete: {len(bursts)} bursts detected",
                {"phase": "complete"},
            )

            return {
                "status": "completed",
                "catalog_id": catalog_id,
                "images_processed": len(images),
                "bursts_detected": len(bursts),
                "total_burst_images": sum(b.image_count for b in bursts),
            }

    except Exception as e:
        logger.error(f"[{job_id}] Burst detection failed: {e}", exc_info=True)
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


@app.task(bind=True, base=ProgressTask, name="chord_progress_monitor")
def chord_progress_monitor_task(
    self: ProgressTask,
    parent_job_id: str,
    catalog_id: str,
    job_type: str,
    expected_workers: Optional[int] = None,
    use_celery_backend: bool = False,
    comparison_start_time: Optional[str] = None,
) -> None:
    """
    Generic progress monitor for chord-based jobs.

    Periodically checks progress and publishes updates until all workers complete.
    Supports two monitoring modes:

    1. job_batches mode: Monitors BatchManager progress (default)
    2. celery_taskmeta mode: Monitors Celery result backend for worker completion

    Args:
        parent_job_id: The coordinator's task ID
        catalog_id: The catalog UUID
        job_type: Job type for BatchManager
        expected_workers: Expected number of workers (for celery_taskmeta mode)
        use_celery_backend: If True, monitor celery_taskmeta; if False, monitor job_batches
        comparison_start_time: ISO timestamp when workers started (for celery_taskmeta mode)
    """
    monitor_id = self.request.id or "unknown"

    try:
        if use_celery_backend:
            # Mode 2: Monitor celery_taskmeta for worker completion
            # Used by jobs that don't use job_batches (e.g., duplicate comparison)
            from ..db import get_db_context
            from .coordinator import publish_job_progress as publish_progress_func

            with get_db_context() as session:
                # Count workers that completed after the comparison phase started
                result = session.execute(
                    text(
                        """
                        SELECT
                            COUNT(*) FILTER (WHERE status = 'SUCCESS') as completed,
                            COUNT(*) FILTER (WHERE status = 'FAILURE') as failed
                        FROM celery_taskmeta
                        WHERE date_done >= :start_time
                    """
                    ),
                    {"start_time": comparison_start_time},
                )
                row = result.fetchone()
                completed = row[0] if row else 0
                failed = row[1] if row else 0

                remaining = expected_workers - completed - failed  # type: ignore[operator]
                progress_pct = (
                    int((completed * 100) / expected_workers)  # type: ignore[operator]
                    if expected_workers and expected_workers > 0
                    else 0
                )

                # Publish progress using publish_progress
                from .progress_publisher import publish_progress

                publish_progress(
                    parent_job_id,
                    "PROGRESS",
                    current=completed,
                    total=expected_workers or 0,
                    message=f"{job_type.replace('_', ' ').title()} workers: {completed}/{expected_workers} completed ({progress_pct}%), {failed} failed",
                    extra={
                        "phase": "processing",
                        "workers_completed": completed,
                        "workers_failed": failed,
                        "workers_remaining": remaining,
                        "workers_total": expected_workers,
                    },
                )

                logger.info(
                    f"[{monitor_id}] {job_type} progress: {completed}/{expected_workers} workers completed, "
                    f"{failed} failed, {remaining} remaining"
                )

                # If not all workers done, schedule another check
                if remaining > 0:
                    chord_progress_monitor_task.apply_async(
                        kwargs={
                            "parent_job_id": parent_job_id,
                            "catalog_id": catalog_id,
                            "job_type": job_type,
                            "expected_workers": expected_workers,
                            "use_celery_backend": use_celery_backend,
                            "comparison_start_time": comparison_start_time,
                        },
                        countdown=30,  # Check again in 30 seconds
                    )

        else:
            # Mode 1: Monitor job_batches table (default for most jobs)
            from .coordinator import BatchManager, publish_job_progress

            batch_manager = BatchManager(catalog_id, parent_job_id, job_type)
            progress = batch_manager.get_progress()

            # Publish progress update
            publish_job_progress(
                parent_job_id,
                progress,
                f"{job_type.replace('_', ' ').title()}: {progress.completed_batches}/{progress.total_batches} batches complete ({progress.percent_complete}%)",
                phase="processing",
            )

            logger.info(
                f"[{monitor_id}] {job_type} progress: {progress.completed_batches}/{progress.total_batches} batches completed, "
                f"{progress.running_batches} running, {progress.pending_batches} pending, {progress.failed_batches} failed"
            )

            # If not all batches done, schedule another check
            if not progress.is_complete:
                chord_progress_monitor_task.apply_async(
                    kwargs={
                        "parent_job_id": parent_job_id,
                        "catalog_id": catalog_id,
                        "job_type": job_type,
                        "expected_workers": expected_workers,
                        "use_celery_backend": use_celery_backend,
                        "comparison_start_time": comparison_start_time,
                    },
                    countdown=30,  # Check again in 30 seconds
                )

    except Exception as e:
        logger.warning(f"[{monitor_id}] Progress monitor error: {e}")
        # Don't fail - just skip this monitoring cycle
