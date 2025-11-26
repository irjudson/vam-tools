"""
Celery tasks for background processing of VAM Tools operations.

These tasks wrap operations and provide progress tracking through Celery's state mechanism.
Updated for PostgreSQL backend.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from celery import Task

from vam_tools.core.types import FileType, ImageRecord, ImageStatus

from ..analysis.scanner import _process_file_worker
from ..db import CatalogDB as CatalogDatabase
from ..organization import FileOrganizer, OrganizationStrategy
from ..shared.media_utils import get_file_type
from ..shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path

# Import app here so that @app.task decorators can find it
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
        Analysis results dictionary
    """
    logger.info(f"Starting catalog analysis: {catalog_id}")

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

        files_added = 0
        files_skipped = 0
        files_processed = 0

        # Reopen database for processing
        with CatalogDatabase(catalog_id) as db:
            for directory in source_dirs:
                logger.info(f"Scanning directory: {directory}")

                for root, dirs, files in os.walk(directory):
                    # Skip synology metadata directories
                    dirs[:] = [d for d in dirs if not d.startswith("@eaDir")]

                    root_path = Path(root)

                    for filename in files:
                        # Skip synology metadata files and hidden files
                        if filename.startswith(".") or "@SynoResource" in filename:
                            continue

                        file_path = root_path / filename

                        # Skip if file doesn't exist or is not accessible
                        if not file_path.exists():
                            continue

                        file_type_str = get_file_type(file_path)

                        if file_type_str not in ("image", "video"):
                            continue

                        # Update progress every 10 files
                        if files_processed % 10 == 0:
                            self.update_progress(
                                files_processed,
                                files_processed + 1,
                                f"Processing {file_path.name}...",
                                {
                                    "phase": "processing",
                                    "added": files_added,
                                    "skipped": files_skipped,
                                },
                            )

                        # Process file
                        result = _process_file_worker(file_path)

                        if result is not None:
                            image_record, file_size = result
                            files_processed += 1

                            # Check if already in catalog
                            existing_image = db.get_image(image_record.id)

                            if not existing_image:
                                # Add to catalog using CatalogDB.add_image()
                                db.add_image(image_record)
                                files_added += 1
                            else:
                                files_skipped += 1
                        else:
                            files_skipped += 1

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

        logger.info(
            f"Analysis complete: {files_processed} processed, "
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
            "total_files": files_processed,
            "processed": files_processed,
            "files_added": files_added,
            "files_skipped": files_skipped,
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
        Scan results dictionary
    """
    logger.info(f"Starting catalog scan: {catalog_id}")

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
        def process_single_file(file_path, catalog_id, generate_previews):
            """Process a single file and return results."""
            try:
                # Process file (extract metadata)
                result = _process_file_worker(file_path)

                if result is None:
                    return None

                image_record, file_size = result

                # Generate thumbnail if requested
                thumbnail_generated = False
                if generate_previews:
                    try:
                        # Compute thumbnails directory for this catalog
                        thumbnails_dir = Path(f"/app/catalogs/{catalog_id}/thumbnails")

                        thumbnail_path = get_thumbnail_path(
                            image_id=image_record.id, thumbnails_dir=thumbnails_dir
                        )
                        if not thumbnail_path.exists():
                            generate_thumbnail(
                                source_path=file_path,
                                output_path=thumbnail_path,
                                size=(512, 512),
                                quality=85,
                            )
                            thumbnail_generated = True
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate thumbnail for {file_path}: {e}"
                        )

                return {
                    "image_record": image_record,
                    "file_size": file_size,
                    "file_path": file_path,
                    "thumbnail_generated": thumbnail_generated,
                }
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                return None

        # Collect all media files first
        self.update_progress(0, 1, "Discovering files...", {"phase": "discovery"})

        all_files = []
        for directory in source_dirs:
            logger.info(f"Discovering files in: {directory}")
            for root, dirs, files in os.walk(directory):
                # Skip synology metadata directories
                dirs[:] = [d for d in dirs if not d.startswith("@eaDir")]

                root_path = Path(root)
                for filename in files:
                    # Skip synology metadata files and hidden files
                    if filename.startswith(".") or "@SynoResource" in filename:
                        continue

                    file_path = root_path / filename
                    if not file_path.exists():
                        continue

                    file_type_str = get_file_type(file_path)
                    if file_type_str in ("image", "video"):
                        all_files.append(file_path)

        total_files = len(all_files)
        logger.info(f"Found {total_files} media files to process")

        # Process files in parallel
        self.update_progress(
            0, total_files, f"Processing {total_files} files...", {"phase": "scanning"}
        )

        files_added = 0
        files_skipped = 0
        files_processed = 0
        progress_lock = Lock()

        # Use ThreadPoolExecutor for parallel processing
        # Use 4x CPU count for I/O-bound thumbnail generation
        max_workers = min(os.cpu_count() * 4, 32)
        logger.info(f"Using {max_workers} workers for parallel processing")

        with CatalogDatabase(catalog_id) as db:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(
                        process_single_file, file_path, catalog_id, generate_previews
                    ): file_path
                    for file_path in all_files
                }

                # Process completed files as they finish
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]

                    try:
                        result = future.result()

                        with progress_lock:
                            files_processed += 1

                            if result is not None:
                                image_record = result["image_record"]

                                # Check if already in catalog
                                existing_image = db.get_image(image_record.id)

                                if not existing_image:
                                    # Add to catalog
                                    db.add_image(image_record)
                                    files_added += 1
                                else:
                                    files_skipped += 1
                            else:
                                files_skipped += 1

                            # Update progress every 10 files
                            if files_processed % 10 == 0:
                                self.update_progress(
                                    files_processed,
                                    total_files,
                                    f"Processed {files_processed}/{total_files} files...",
                                    {
                                        "phase": "scanning",
                                        "processed": files_processed,
                                        "added": files_added,
                                        "skipped": files_skipped,
                                    },
                                )
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        with progress_lock:
                            files_processed += 1
                            files_skipped += 1

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

        logger.info(
            f"Scan complete: {files_processed} processed, "
            f"{files_added} added, {files_skipped} skipped"
        )

        self.update_progress(
            100,
            100,
            f"Scan complete: {files_added} files added",
            {"phase": "complete"},
        )

        return {
            "status": "completed",
            "catalog_id": catalog_id,
            "total_files": files_processed,
            "processed": files_processed,
            "files_added": files_added,
            "files_skipped": files_skipped,
        }

    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
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
