"""
Generic Parallel Job Framework for Lumina.

This module provides reusable coordinator, worker, and finalizer tasks that can
be used by any job type. Job types just need to register an item processor.

Usage:
    1. Register an item processor:
        @register_item_processor("thumbnails")
        def process_thumbnail_item(catalog_id, work_item, **kwargs):
            # Process single item
            return {"success": True}

    2. Start a parallel job:
        result = generic_coordinator_task.delay(
            catalog_id="...",
            job_type="thumbnails",
            work_items=[...],  # Or use work_items_query
            batch_size=500,
            processor_kwargs={...},
        )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from celery import chord, group
from sqlalchemy import text

from ..db import CatalogDB as CatalogDatabase
from ..db.models import Job
from ..shared.media_utils import get_file_type
from .celery_app import app
from .coordinator import (
    BatchManager,
    BatchResult,
    JobCancelledException,
    get_item_processor,
    publish_job_progress,
)
from .progress_publisher import publish_completion, publish_progress
from .tasks import CoordinatorTask, ProgressTask

logger = logging.getLogger(__name__)


def _discover_media_files(
    source_directories: List[str],
) -> Tuple[List[str], Dict[str, int]]:
    """
    Discover all media files in the source directories.

    Args:
        source_directories: List of directory paths to scan

    Returns:
        Tuple of (list of file paths, discovery stats dict)
    """
    all_files: List[str] = []
    stats = {
        "directories_scanned": 0,
        "files_discovered": 0,
        "skipped_synology_metadata": 0,
        "skipped_hidden_file": 0,
        "skipped_file_not_accessible": 0,
        "skipped_unsupported_format": 0,
    }

    for directory in source_directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            continue

        logger.info(f"Discovering files in: {directory}")

        for root, dirs, files in os.walk(dir_path):
            stats["directories_scanned"] += 1

            # Skip Synology metadata directories
            original_dir_count = len(dirs)
            dirs[:] = [d for d in dirs if not d.startswith("@eaDir")]
            stats["skipped_synology_metadata"] += original_dir_count - len(dirs)

            root_path = Path(root)

            for filename in files:
                stats["files_discovered"] += 1

                # Skip Synology metadata files
                if "@SynoResource" in filename or "@eaDir" in str(root_path):
                    stats["skipped_synology_metadata"] += 1
                    continue

                # Skip hidden files
                if filename.startswith("."):
                    stats["skipped_hidden_file"] += 1
                    continue

                file_path = root_path / filename

                # Check accessibility
                if not file_path.exists():
                    stats["skipped_file_not_accessible"] += 1
                    continue

                # Check file type
                file_type_str = get_file_type(file_path)
                if file_type_str not in ("image", "video"):
                    stats["skipped_unsupported_format"] += 1
                    continue

                all_files.append(str(file_path))

    return all_files, stats


def _update_job_status(
    job_id: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Update job status directly in the database."""
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


@app.task(bind=True, base=CoordinatorTask, name="generic_coordinator")
def generic_coordinator_task(
    self: CoordinatorTask,
    catalog_id: str,
    job_type: str,
    work_items: Optional[List[Any]] = None,
    work_items_query: Optional[str] = None,
    source_directories: Optional[List[str]] = None,
    batch_size: int = 500,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generic coordinator task for parallel job processing.

    This task:
    1. Gets work items (from list, query, or directory discovery)
    2. Creates batches in the database
    3. Spawns worker tasks via Celery chord
    4. The chord callback triggers the finalizer

    Args:
        catalog_id: UUID of the catalog
        job_type: Type of job (must have registered processor)
        work_items: List of items to process
        work_items_query: SQL query to get work items (returns list of values)
        source_directories: List of directories to scan for files (for analyze job)
        batch_size: Number of items per batch
        processor_kwargs: Extra kwargs to pass to item processor

    Returns:
        Dictionary with job setup info (actual results come from finalizer)
    """
    parent_job_id = self.request.id or "unknown"
    logger.info(
        f"[{parent_job_id}] Starting generic coordinator for {job_type} "
        f"on catalog {catalog_id}"
    )

    if processor_kwargs is None:
        processor_kwargs = {}

    discovery_stats: Optional[Dict[str, int]] = None

    try:
        # Verify processor exists
        get_item_processor(job_type)

        # Phase 1: Get work items
        self.update_progress(0, 1, "Querying work items...", {"phase": "init"})

        if work_items is None:
            if source_directories is not None:
                # Discover files from directories (for analyze job)
                self.update_progress(
                    0, 1, "Discovering files in directories...", {"phase": "discovery"}
                )
                work_items, discovery_stats = _discover_media_files(source_directories)
                logger.info(
                    f"[{parent_job_id}] Discovered {len(work_items)} files "
                    f"from {len(source_directories)} directories"
                )
            elif work_items_query is not None:
                # Query from database
                with CatalogDatabase(catalog_id) as db:
                    assert db.session is not None
                    result = db.session.execute(
                        text(work_items_query),
                        {"catalog_id": catalog_id},
                    )
                    work_items = [
                        row[0] if len(row) == 1 else list(row) for row in result
                    ]
            else:
                raise ValueError(
                    "Must provide work_items, work_items_query, or source_directories"
                )

        total_items = len(work_items)
        logger.info(f"[{parent_job_id}] Found {total_items} items for {job_type}")

        if total_items == 0:
            publish_completion(
                parent_job_id,
                "SUCCESS",
                result={"status": "completed", "message": "No items to process"},
            )
            return {"status": "completed", "message": "No items to process"}

        # Phase 2: Create batches
        self.update_progress(
            0,
            total_items,
            f"Creating batches for {total_items} items...",
            {"phase": "batching"},
        )

        batch_manager = BatchManager(catalog_id, parent_job_id, job_type)

        with CatalogDatabase(catalog_id) as db:
            batch_ids = batch_manager.create_batches(
                work_items=work_items,
                batch_size=batch_size,
                db=db,
            )
            db.save()

        num_batches = len(batch_ids)
        logger.info(f"[{parent_job_id}] Created {num_batches} batches")

        # Phase 3: Spawn workers via chord
        self.update_progress(
            0,
            total_items,
            f"Spawning {num_batches} workers...",
            {"phase": "spawning"},
        )

        worker_tasks = group(
            generic_worker_task.s(
                catalog_id=catalog_id,
                job_type=job_type,
                batch_id=batch_id,
                parent_job_id=parent_job_id,
                processor_kwargs=processor_kwargs,
            )
            for batch_id in batch_ids
        )

        finalizer = generic_finalizer_task.s(
            catalog_id=catalog_id,
            job_type=job_type,
            parent_job_id=parent_job_id,
        )

        chord(worker_tasks)(finalizer)

        logger.info(
            f"[{parent_job_id}] Chord dispatched: {num_batches} workers → finalizer"
        )

        # Set job to STARTED state
        _update_job_status(
            parent_job_id,
            "STARTED",
            result={
                "status": "processing",
                "total_items": total_items,
                "num_batches": num_batches,
                "message": f"Processing {total_items} items in {num_batches} batches",
            },
        )

        publish_progress(
            parent_job_id,
            "PROGRESS",
            current=0,
            total=total_items,
            message=f"Processing {total_items} items in {num_batches} batches",
            extra={"phase": "processing", "batches_total": num_batches},
        )

        return {
            "status": "dispatched",
            "catalog_id": catalog_id,
            "job_type": job_type,
            "total_items": total_items,
            "num_batches": num_batches,
        }

    except Exception as e:
        logger.error(f"[{parent_job_id}] Coordinator failed: {e}", exc_info=True)
        _update_job_status(parent_job_id, "FAILURE", error=str(e))
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        raise


@app.task(bind=True, name="generic_worker")
def generic_worker_task(
    self: Any,
    catalog_id: str,
    job_type: str,
    batch_id: str,
    parent_job_id: str,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generic worker task that processes a batch using the registered processor.

    Args:
        catalog_id: UUID of the catalog
        job_type: Type of job (determines which processor to use)
        batch_id: UUID of the batch to process
        parent_job_id: UUID of the parent coordinator job
        processor_kwargs: Extra kwargs to pass to item processor

    Returns:
        Batch result as dictionary
    """
    worker_id = self.request.id or "unknown"
    logger.info(f"[{worker_id}] Starting {job_type} worker for batch {batch_id}")

    if processor_kwargs is None:
        processor_kwargs = {}

    batch_manager = BatchManager(catalog_id, parent_job_id, job_type)

    try:
        # Get the item processor
        process_item = get_item_processor(job_type)

        with CatalogDatabase(catalog_id) as db:
            batch_data = batch_manager.claim_batch(batch_id, worker_id, db)

        if not batch_data:
            logger.warning(f"[{worker_id}] Batch {batch_id} already claimed")
            return {
                "batch_id": batch_id,
                "status": "skipped",
                "reason": "already_claimed",
            }

        batch_number = batch_data["batch_number"]
        total_batches = batch_data["total_batches"]
        work_items = batch_data["work_items"]
        items_count = batch_data["items_count"]

        logger.info(
            f"[{worker_id}] Processing batch {batch_number + 1}/{total_batches} "
            f"({items_count} items)"
        )

        result = BatchResult(batch_id=batch_id, batch_number=batch_number)

        # Process each item
        for idx, work_item in enumerate(work_items):
            # Check for cancellation every 10 items
            if idx % 10 == 0:
                if batch_manager.is_cancelled(batch_id):
                    logger.warning(
                        f"[{worker_id}] Batch {batch_id} cancelled, stopping after {idx} items"
                    )
                    raise JobCancelledException(
                        f"Job cancelled after processing {idx}/{len(work_items)} items"
                    )

            try:
                item_result = process_item(
                    catalog_id=catalog_id,
                    work_item=work_item,
                    db=None,  # Let processor create own DB session if needed
                    **processor_kwargs,
                )

                if item_result.get("success", False):
                    result.success_count += 1
                else:
                    result.error_count += 1
                    if "error" in item_result:
                        result.errors.append(
                            {
                                "item": str(work_item)[:100],
                                "error": item_result["error"],
                            }
                        )

                result.processed_count += 1

            except Exception as e:
                result.error_count += 1
                result.errors.append({"item": str(work_item)[:100], "error": str(e)})
                logger.warning(f"[{worker_id}] Failed to process item: {e}")

        # Mark batch complete
        with CatalogDatabase(catalog_id) as db:
            batch_manager.complete_batch(batch_id, result, db)
            progress = batch_manager.get_progress(db)
            publish_job_progress(
                parent_job_id,
                progress,
                f"Batch {batch_number + 1}/{total_batches} complete",
                phase="processing",
            )
            db.save()

        logger.info(
            f"[{worker_id}] Batch {batch_number + 1} complete: "
            f"{result.success_count} success, {result.error_count} errors"
        )

        return {
            "batch_id": batch_id,
            "batch_number": batch_number,
            "status": "completed",
            "success_count": result.success_count,
            "error_count": result.error_count,
            "processed_count": result.processed_count,
        }

    except JobCancelledException as e:
        logger.warning(f"[{worker_id}] Worker cancelled: {e}")
        # Don't mark as failed - job was intentionally cancelled
        return {
            "batch_id": batch_id,
            "batch_number": batch_number,
            "status": "cancelled",
            "success_count": result.success_count if "result" in locals() else 0,
            "error_count": result.error_count if "result" in locals() else 0,
            "processed_count": result.processed_count if "result" in locals() else 0,
            "message": str(e),
        }

    except Exception as e:
        logger.error(f"[{worker_id}] Worker failed: {e}", exc_info=True)
        try:
            with CatalogDatabase(catalog_id) as db:
                batch_manager.fail_batch(batch_id, str(e), db)
                db.save()
        except Exception:
            pass
        return {"batch_id": batch_id, "status": "failed", "error": str(e)}


@app.task(bind=True, base=ProgressTask, name="generic_finalizer")
def generic_finalizer_task(
    self: ProgressTask,
    worker_results: List[Dict[str, Any]],
    catalog_id: str,
    job_type: str,
    parent_job_id: str,
) -> Dict[str, Any]:
    """
    Generic finalizer that aggregates results from all workers.

    Args:
        worker_results: List of results from all worker tasks
        catalog_id: UUID of the catalog
        job_type: Type of job
        parent_job_id: UUID of the parent coordinator job

    Returns:
        Final aggregated result
    """
    finalizer_id = self.request.id or "unknown"
    logger.info(
        f"[{finalizer_id}] Starting {job_type} finalizer for job {parent_job_id}"
    )

    try:
        self.update_progress(0, 1, "Aggregating results...", {"phase": "finalizing"})

        batch_manager = BatchManager(catalog_id, parent_job_id, job_type)

        with CatalogDatabase(catalog_id) as db:
            progress = batch_manager.get_progress(db)

        total_success = sum(
            wr.get("success_count", 0)
            for wr in worker_results
            if wr.get("status") == "completed"
        )
        total_errors = sum(
            wr.get("error_count", 0)
            for wr in worker_results
            if wr.get("status") == "completed"
        )
        total_processed = sum(
            wr.get("processed_count", 0)
            for wr in worker_results
            if wr.get("status") == "completed"
        )
        failed_batches = sum(1 for wr in worker_results if wr.get("status") == "failed")

        final_result = {
            "status": "completed" if failed_batches == 0 else "completed_with_errors",
            "catalog_id": catalog_id,
            "job_type": job_type,
            "items_processed": total_processed,
            "items_success": total_success,
            "items_failed": total_errors,
            "total_batches": len(worker_results),
            "failed_batches": failed_batches,
        }

        # Determine overall status
        status = "SUCCESS"
        if failed_batches > 0 or total_errors > total_processed * 0.1:
            # More than 10% errors = partial success
            status = "SUCCESS"  # Still mark as success but with errors noted

        publish_completion(parent_job_id, status, result=final_result)
        _update_job_status(parent_job_id, status, result=final_result)

        self.update_progress(
            progress.total_items,
            progress.total_items,
            f"Complete: {total_success} processed, {total_errors} errors",
            {"phase": "complete"},
        )

        logger.info(
            f"[{finalizer_id}] {job_type} complete: "
            f"{total_success} success, {total_errors} errors"
        )

        return final_result

    except Exception as e:
        logger.error(f"[{finalizer_id}] Finalizer failed: {e}", exc_info=True)
        _update_job_status(parent_job_id, "FAILURE", error=str(e))
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        raise


def start_parallel_job(
    catalog_id: str,
    job_type: str,
    work_items: Optional[List[Any]] = None,
    work_items_query: Optional[str] = None,
    batch_size: int = 500,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Helper function to start a parallel job.

    Args:
        catalog_id: UUID of the catalog
        job_type: Type of job (must have registered processor)
        work_items: List of items to process
        work_items_query: SQL query to get work items
        batch_size: Number of items per batch
        processor_kwargs: Extra kwargs to pass to item processor

    Returns:
        Task ID of the coordinator job
    """
    task = generic_coordinator_task.delay(
        catalog_id=catalog_id,
        job_type=job_type,
        work_items=work_items,
        work_items_query=work_items_query,
        batch_size=batch_size,
        processor_kwargs=processor_kwargs,
    )
    return task.id
