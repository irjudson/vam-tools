"""
Parallel Scan Tasks using the Coordinator Pattern.

This module implements parallel directory scanning across multiple Celery workers:

1. scan_coordinator_task: Discovers files, creates batches, spawns workers
2. scan_worker_task: Processes a batch of files
3. scan_finalizer_task: Aggregates results and updates catalog

Usage:
    # Start a parallel scan
    result = scan_coordinator_task.delay(
        catalog_id="...",
        source_directories=["/path/to/photos"],
        batch_size=1000,  # Files per worker
    )
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from celery import chord, group
from sqlalchemy import text

from ..analysis.scanner import _process_file_worker
from ..db import CatalogDB as CatalogDatabase
from ..db.models import Job
from ..shared.media_utils import get_file_type
from ..shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path
from .celery_app import app
from .coordinator import (
    CONSECUTIVE_FAILURE_THRESHOLD,
    BatchManager,
    BatchResult,
    cancel_and_requeue_job,
    publish_job_progress,
)
from .progress_publisher import publish_completion, publish_progress
from .scan_stats import ScanStatistics
from .tasks import ProgressTask

logger = logging.getLogger(__name__)


def _update_job_status(
    job_id: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """
    Update job status directly in the database.

    This bypasses Celery's automatic status updates, allowing us to keep
    the coordinator job in PROGRESS state while workers are running.

    Args:
        job_id: The job UUID
        status: New status (PENDING, PROGRESS, SUCCESS, FAILURE)
        result: Optional result dict (for SUCCESS)
        error: Optional error message (for FAILURE)
    """
    from ..db import get_db_context

    try:
        with get_db_context() as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = status
                if result is not None:
                    job.result = result
                if error is not None:
                    job.error = error
                session.commit()
                logger.debug(f"Updated job {job_id} status to {status}")
    except Exception as e:
        logger.warning(f"Failed to update job status for {job_id}: {e}")


def _discover_media_files(
    source_directories: List[str],
    stats: ScanStatistics,
) -> Tuple[List[str], ScanStatistics]:
    """
    Discover all media files in the source directories.

    This is the first phase - just discovery, no processing.

    Args:
        source_directories: List of directory paths to scan
        stats: ScanStatistics to update with discovery counts

    Returns:
        Tuple of (list of file paths, updated stats)
    """
    all_files: List[str] = []

    for directory in source_directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            continue

        logger.info(f"Discovering files in: {directory}")

        for root, dirs, files in os.walk(dir_path):
            stats.directories_scanned += 1

            # Skip Synology metadata directories
            original_dir_count = len(dirs)
            dirs[:] = [d for d in dirs if not d.startswith("@eaDir")]
            stats.skipped_synology_metadata += original_dir_count - len(dirs)

            root_path = Path(root)

            for filename in files:
                stats.files_discovered += 1

                # Skip Synology metadata files
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

                all_files.append(str(file_path))

    return all_files, stats


@app.task(bind=True, base=ProgressTask, name="scan_coordinator")
def scan_coordinator_task(
    self: ProgressTask,
    catalog_id: str,
    source_directories: List[str],
    force_rescan: bool = False,
    generate_previews: bool = True,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """
    Coordinator task for parallel catalog scanning.

    This task:
    1. Discovers all media files in source directories
    2. Creates batches in the database
    3. Spawns worker tasks via Celery chord
    4. The chord callback triggers the finalizer

    Args:
        catalog_id: UUID of the catalog
        source_directories: List of directories to scan
        force_rescan: Clear existing images first
        generate_previews: Whether workers should generate thumbnails
        batch_size: Number of files per batch/worker

    Returns:
        Dictionary with job setup info (actual results come from finalizer)
    """
    parent_job_id = self.request.id or "unknown"
    logger.info(f"[{parent_job_id}] Starting scan coordinator for catalog {catalog_id}")

    try:
        # Phase 1: Initialize
        self.update_progress(0, 1, "Initializing parallel scan...", {"phase": "init"})

        stats = ScanStatistics()

        with CatalogDatabase(catalog_id) as db:
            # Update catalog config
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

            if force_rescan:
                logger.info(f"[{parent_job_id}] Force rescan: clearing existing images")
                self.update_progress(
                    0, 1, "Clearing existing images...", {"phase": "clearing"}
                )
                assert db.session is not None
                db.session.execute(
                    text("DELETE FROM images WHERE catalog_id = :catalog_id"),
                    {"catalog_id": catalog_id},
                )
                assert db.session is not None
                db.session.commit()

        # Phase 2: Discover files
        self.update_progress(0, 1, "Discovering files...", {"phase": "discovery"})

        all_files, stats = _discover_media_files(source_directories, stats)
        total_files = len(all_files)

        logger.info(
            f"[{parent_job_id}] Discovered {total_files} media files "
            f"(scanned {stats.directories_scanned} dirs, "
            f"skipped {stats.total_skipped} files)"
        )

        if total_files == 0:
            # Nothing to process
            stats.finish()
            publish_completion(
                parent_job_id,
                "SUCCESS",
                result={
                    "status": "completed",
                    "message": "No media files found",
                    "statistics": stats.to_dict(),
                },
            )
            return {
                "status": "completed",
                "catalog_id": catalog_id,
                "message": "No media files found to process",
                "statistics": stats.to_dict(),
            }

        # Phase 3: Create batches
        self.update_progress(
            0,
            total_files,
            f"Creating batches for {total_files} files...",
            {"phase": "batching", "total_files": total_files},
        )

        batch_manager = BatchManager(catalog_id, parent_job_id, "scan")

        with CatalogDatabase(catalog_id) as db:
            batch_ids = batch_manager.create_batches(
                work_items=all_files,
                batch_size=batch_size,
                db=db,
            )

            # Store discovery stats for finalizer
            db.execute(
                """
                INSERT INTO config (catalog_id, key, value, updated_at)
                VALUES (?, ?, ?, NOW())
                ON CONFLICT (catalog_id, key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    db.catalog_id,
                    f"scan_discovery_stats_{parent_job_id}",
                    json.dumps(stats.to_dict()),
                ),
            )
            db.save()

        num_batches = len(batch_ids)
        logger.info(
            f"[{parent_job_id}] Created {num_batches} batches "
            f"({batch_size} files each)"
        )

        # Phase 4: Spawn sub-tasks via chord
        # chord(group(sub-tasks), finalizer) - finalizer runs after all sub-tasks complete
        self.update_progress(
            0,
            total_files,
            f"Spawning {num_batches} sub-tasks...",
            {"phase": "spawning", "num_batches": num_batches},
        )

        worker_tasks = group(
            scan_worker_task.s(
                catalog_id=catalog_id,
                batch_id=batch_id,
                parent_job_id=parent_job_id,
                generate_previews=generate_previews,
            )
            for batch_id in batch_ids
        )

        finalizer = scan_finalizer_task.s(
            catalog_id=catalog_id,
            parent_job_id=parent_job_id,
            source_directories=source_directories,
            generate_previews=generate_previews,
            batch_size=batch_size,
        )

        # Execute the chord (sub-tasks in parallel, then finalizer)
        chord(worker_tasks)(finalizer)

        logger.info(
            f"[{parent_job_id}] Chord dispatched: {num_batches} sub-tasks → finalizer"
        )

        # Set job to STARTED state - workers are now processing
        # The finalizer will update to SUCCESS when all batches complete
        _update_job_status(
            parent_job_id,
            "STARTED",
            result={
                "status": "processing",
                "total_files": total_files,
                "num_batches": num_batches,
                "message": f"Processing {total_files} files across {num_batches} batches",
            },
        )

        # Publish initial progress to Redis
        publish_progress(
            parent_job_id,
            "PROGRESS",
            current=0,
            total=total_files,
            message=f"Processing {total_files} files in {num_batches} batches",
            extra={
                "phase": "processing",
                "batches_total": num_batches,
                "batches_completed": 0,
                "batches_running": num_batches,
                "batches_pending": 0,
            },
        )

        # Return immediately - sub-tasks will process in background
        return {
            "status": "dispatched",
            "catalog_id": catalog_id,
            "parent_job_id": parent_job_id,
            "total_files": total_files,
            "num_batches": num_batches,
            "batch_size": batch_size,
            "message": f"Processing {total_files} files in {num_batches} batches",
        }

    except Exception as e:
        logger.error(f"[{parent_job_id}] Coordinator failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        raise


@app.task(bind=True, name="scan_worker")
def scan_worker_task(
    self: Any,
    catalog_id: str,
    batch_id: str,
    parent_job_id: str,
    generate_previews: bool = True,
) -> Dict[str, Any]:
    """
    Worker task that processes a batch of files.

    Each worker:
    1. Claims its batch (PENDING → RUNNING)
    2. Processes files (extract metadata, generate thumbnails)
    3. Saves results to database
    4. Marks batch complete (RUNNING → COMPLETED)

    Args:
        catalog_id: UUID of the catalog
        batch_id: UUID of the batch to process
        parent_job_id: Coordinator's task ID (for progress publishing)
        generate_previews: Whether to generate thumbnails

    Returns:
        BatchResult dictionary
    """
    worker_id = self.request.id or "unknown"
    logger.info(f"[{worker_id}] Starting scan worker for batch {batch_id}")

    batch_manager = BatchManager(catalog_id, parent_job_id, "scan")

    try:
        # Claim the batch
        with CatalogDatabase(catalog_id) as db:
            batch_data = batch_manager.claim_batch(batch_id, worker_id, db)

        if not batch_data:
            logger.warning(
                f"[{worker_id}] Batch {batch_id} already claimed or processed"
            )
            return {
                "batch_id": batch_id,
                "status": "skipped",
                "reason": "already_claimed",
            }

        batch_number = batch_data["batch_number"]
        total_batches = batch_data["total_batches"]
        file_paths = batch_data["work_items"]
        items_count = batch_data["items_count"]

        logger.info(
            f"[{worker_id}] Processing batch {batch_number + 1}/{total_batches} "
            f"({items_count} files)"
        )

        # Process files
        result = BatchResult(
            batch_id=batch_id,
            batch_number=batch_number,
        )

        thumbnails_dir = Path(f"/app/catalogs/{catalog_id}/thumbnails")

        with CatalogDatabase(catalog_id) as db:
            for i, file_path_str in enumerate(file_paths):
                file_path = Path(file_path_str)

                try:
                    # Extract metadata
                    process_result = _process_file_worker(file_path)

                    if process_result is None:
                        result.error_count += 1
                        result.errors.append(
                            {
                                "file": file_path_str,
                                "error": "Failed to extract metadata",
                            }
                        )
                        continue

                    image_record, file_size = process_result
                    result.processed_count += 1

                    # Generate thumbnail if requested
                    if generate_previews:
                        try:
                            thumbnail_path = get_thumbnail_path(
                                image_id=image_record.id,
                                thumbnails_dir=thumbnails_dir,
                            )
                            if not thumbnail_path.exists():
                                generate_thumbnail(
                                    source_path=file_path,
                                    output_path=thumbnail_path,
                                    size=(512, 512),
                                    quality=85,
                                )
                        except Exception as thumb_e:
                            logger.debug(f"Thumbnail failed for {file_path}: {thumb_e}")

                    # Check if already in catalog
                    existing = db.get_image(image_record.id)
                    if not existing:
                        db.add_image(image_record)
                        result.success_count += 1
                    else:
                        # Already exists - still a success, just skipped
                        result.success_count += 1

                except Exception as e:
                    result.error_count += 1
                    result.errors.append(
                        {
                            "file": file_path_str,
                            "error": str(e),
                        }
                    )
                    logger.debug(f"[{worker_id}] Error processing {file_path}: {e}")

                # Publish progress periodically
                if (i + 1) % 50 == 0:
                    progress = batch_manager.get_progress(db)
                    # Add this batch's partial progress
                    progress.processed_items += result.processed_count
                    publish_job_progress(
                        parent_job_id,
                        progress,
                        f"Batch {batch_number + 1}: {i + 1}/{items_count} files...",
                        phase="processing",
                    )

            db.save()

        # Mark batch complete
        with CatalogDatabase(catalog_id) as db:
            batch_manager.complete_batch(batch_id, result, db)

        # Publish final batch progress
        with CatalogDatabase(catalog_id) as db:
            progress = batch_manager.get_progress(db)
            publish_job_progress(
                parent_job_id,
                progress,
                f"Batch {batch_number + 1}/{total_batches} complete",
                phase="processing",
            )

        logger.info(
            f"[{worker_id}] Batch {batch_number + 1} complete: "
            f"{result.success_count} added, {result.error_count} errors"
        )

        return {
            "batch_id": batch_id,
            "batch_number": batch_number,
            "status": "completed",
            "processed_count": result.processed_count,
            "success_count": result.success_count,
            "error_count": result.error_count,
            "errors": result.errors[:10],  # Limit error details
        }

    except Exception as e:
        logger.error(f"[{worker_id}] Worker failed: {e}", exc_info=True)

        # Mark batch as failed
        try:
            batch_manager.fail_batch(batch_id, str(e))
        except Exception:
            pass

        return {
            "batch_id": batch_id,
            "status": "failed",
            "error": str(e),
        }


@app.task(bind=True, base=ProgressTask, name="scan_finalizer")
def scan_finalizer_task(
    self: ProgressTask,
    worker_results: List[Dict[str, Any]],
    catalog_id: str,
    parent_job_id: str,
    source_directories: Optional[List[str]] = None,
    generate_previews: bool = True,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """
    Finalizer task that aggregates results after all workers complete.

    This task:
    1. Collects results from all worker tasks
    2. Aggregates statistics
    3. Updates catalog state
    4. Publishes completion to Redis
    5. Cleans up batch records
    6. Auto-requeues if too many batches failed

    Args:
        worker_results: List of results from all worker tasks
        catalog_id: UUID of the catalog
        parent_job_id: Coordinator's task ID
        source_directories: Original source directories for requeue
        generate_previews: Whether workers generate thumbnails
        batch_size: Number of files per batch

    Returns:
        Final aggregated results
    """
    finalizer_id = self.request.id or "unknown"
    logger.info(
        f"[{finalizer_id}] Starting finalizer for job {parent_job_id} "
        f"({len(worker_results)} worker results)"
    )

    try:
        self.update_progress(0, 1, "Aggregating results...", {"phase": "finalizing"})

        batch_manager = BatchManager(catalog_id, parent_job_id, "scan")

        # Get final progress from database
        with CatalogDatabase(catalog_id) as db:
            progress = batch_manager.get_progress(db)

            # Get discovery stats saved by coordinator
            assert db.session is not None
            result = db.session.execute(
                text(
                    """
                    SELECT value FROM config
                    WHERE catalog_id = :catalog_id
                    AND key = :key
                """
                ),
                {
                    "catalog_id": catalog_id,
                    "key": f"scan_discovery_stats_{parent_job_id}",
                },
            )
            row = result.fetchone()
            # Handle both JSONB (returns dict) and TEXT (returns string) column types
            if row:
                discovery_stats = (
                    row[0] if isinstance(row[0], dict) else json.loads(row[0])
                )
            else:
                discovery_stats = {}

            # Update catalog config with completion
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

            # Clean up discovery stats
            assert db.session is not None
            db.session.execute(
                text(
                    """
                    DELETE FROM config
                    WHERE catalog_id = :catalog_id
                    AND key = :key
                """
                ),
                {
                    "catalog_id": catalog_id,
                    "key": f"scan_discovery_stats_{parent_job_id}",
                },
            )

            db.save()

        # Aggregate worker results
        total_processed = 0
        total_success = 0
        total_errors = 0
        all_errors: List[Dict[str, str]] = []
        failed_batches = 0
        completed_batches = 0

        for wr in worker_results:
            if wr.get("status") == "completed":
                completed_batches += 1
                total_processed += wr.get("processed_count", 0)
                total_success += wr.get("success_count", 0)
                total_errors += wr.get("error_count", 0)
                all_errors.extend(wr.get("errors", []))
            elif wr.get("status") == "failed":
                failed_batches += 1
            # skipped batches are ignored

        # If there were too many failed batches, auto-requeue to continue
        if failed_batches >= CONSECUTIVE_FAILURE_THRESHOLD and source_directories:
            logger.warning(
                f"[{finalizer_id}] {failed_batches} batches failed, auto-requeuing continuation"
            )

            cancel_and_requeue_job(
                parent_job_id=parent_job_id,
                catalog_id=catalog_id,
                job_type="scan",
                task_name="scan_coordinator",
                task_kwargs={
                    "catalog_id": catalog_id,
                    "source_directories": source_directories,
                    "force_rescan": False,  # Don't clear, continue from where we left off
                    "generate_previews": generate_previews,
                    "batch_size": batch_size,
                },
                reason=f"{failed_batches} batch failures",
                processed_so_far=total_success,
            )

            return {
                "status": "requeued",
                "catalog_id": catalog_id,
                "files_added": total_success,
                "errors": total_errors,
                "failed_batches": failed_batches,
                "message": f"Job requeued due to {failed_batches} batch failures",
            }

        # Build final result
        final_result = {
            "status": "completed" if failed_batches == 0 else "completed_with_errors",
            "catalog_id": catalog_id,
            "parent_job_id": parent_job_id,
            "batches": {
                "total": progress.total_batches,
                "completed": completed_batches,
                "failed": failed_batches,
            },
            "files": {
                "total": progress.total_items,
                "processed": total_processed,
                "added": total_success,
                "errors": total_errors,
            },
            "discovery": discovery_stats,
            "errors": all_errors[:50],  # Limit error list
        }

        # Publish completion to Redis
        publish_completion(
            parent_job_id,
            "SUCCESS",
            result=final_result,
        )

        # Update job status in database to SUCCESS
        # This completes the job lifecycle started by the coordinator
        _update_job_status(parent_job_id, "SUCCESS", result=final_result)

        self.update_progress(
            progress.total_items,
            progress.total_items,
            f"Scan complete: {total_success} files added",
            {"phase": "complete"},
        )

        logger.info(
            f"[{finalizer_id}] Scan complete for {catalog_id}: "
            f"{total_success} added, {total_errors} errors, "
            f"{completed_batches}/{progress.total_batches} batches"
        )

        # Clean up batch records (optional - keep for debugging)
        # batch_manager.cleanup_batches()

        return final_result

    except Exception as e:
        logger.error(f"[{finalizer_id}] Finalizer failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        # Update job status in database to FAILURE
        _update_job_status(parent_job_id, "FAILURE", error=str(e))
        raise


@app.task(name="scan_recovery")
def scan_recovery_task(
    catalog_id: str,
    parent_job_id: str,
    stale_minutes: int = 10,
) -> Dict[str, Any]:
    """
    Legacy recovery task - delegates to generic job_recovery.

    Kept for backward compatibility with existing API endpoints.
    """
    from .job_recovery import recover_job

    return recover_job(parent_job_id, stale_minutes)
