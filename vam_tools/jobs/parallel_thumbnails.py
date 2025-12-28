"""
Parallel Thumbnail Generation using the Coordinator Pattern.

This module implements parallel thumbnail generation across multiple Celery workers:

1. thumbnail_coordinator_task: Queries images, creates batches, spawns workers
2. thumbnail_worker_task: Processes a batch of images
3. thumbnail_finalizer_task: Aggregates results
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import chord, group
from sqlalchemy import text

from ..db import CatalogDB as CatalogDatabase
from ..db.models import Job
from ..shared.thumbnail_utils import generate_thumbnail
from .celery_app import app
from .coordinator import (
    CONSECUTIVE_FAILURE_THRESHOLD,
    BatchManager,
    BatchResult,
    cancel_and_requeue_job,
    publish_job_progress,
)
from .progress_publisher import publish_completion, publish_progress
from .tasks import CoordinatorTask, ProgressTask

logger = logging.getLogger(__name__)


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


@app.task(bind=True, base=CoordinatorTask, name="thumbnail_coordinator")
def thumbnail_coordinator_task(
    self: CoordinatorTask,
    catalog_id: str,
    sizes: Optional[List[int]] = None,
    quality: int = 85,
    force: bool = False,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """
    Coordinator task for parallel thumbnail generation.

    Args:
        catalog_id: UUID of the catalog
        sizes: List of thumbnail sizes (default: [256, 512])
        quality: JPEG quality (1-100)
        force: Regenerate existing thumbnails
        batch_size: Number of images per batch
    """
    parent_job_id = self.request.id or "unknown"
    logger.info(
        f"[{parent_job_id}] Starting thumbnail coordinator for catalog {catalog_id}"
    )

    if sizes is None:
        sizes = [256, 512]

    try:
        self.update_progress(0, 1, "Querying images...", {"phase": "init"})

        # Get all image IDs that need thumbnails
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            result = db.session.execute(
                text(
                    "SELECT id, source_path FROM images WHERE catalog_id = :catalog_id"
                ),
                {"catalog_id": catalog_id},
            )
            image_data = [(str(row[0]), row[1]) for row in result.fetchall()]

        total_images = len(image_data)
        logger.info(
            f"[{parent_job_id}] Found {total_images} images for thumbnail generation"
        )

        if total_images == 0:
            publish_completion(
                parent_job_id,
                "SUCCESS",
                result={"status": "completed", "message": "No images found"},
            )
            return {"status": "completed", "message": "No images to process"}

        # Create batches
        self.update_progress(
            0,
            total_images,
            f"Creating batches for {total_images} images...",
            {"phase": "batching"},
        )

        batch_manager = BatchManager(catalog_id, parent_job_id, "thumbnails")

        with CatalogDatabase(catalog_id) as db:
            batch_ids = batch_manager.create_batches(
                work_items=image_data,
                batch_size=batch_size,
                db=db,
            )

        num_batches = len(batch_ids)
        logger.info(f"[{parent_job_id}] Created {num_batches} batches")

        # Spawn workers via chord
        self.update_progress(
            0,
            total_images,
            f"Spawning {num_batches} sub-tasks...",
            {"phase": "spawning"},
        )

        worker_tasks = group(
            thumbnail_worker_task.s(
                catalog_id=catalog_id,
                batch_id=batch_id,
                parent_job_id=parent_job_id,
                sizes=sizes,
                quality=quality,
                force=force,
            )
            for batch_id in batch_ids
        )

        finalizer = thumbnail_finalizer_task.s(
            catalog_id=catalog_id,
            parent_job_id=parent_job_id,
            sizes=sizes,
            quality=quality,
            force=force,
            batch_size=batch_size,
        )

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
                "total_images": total_images,
                "num_batches": num_batches,
                "message": f"Processing {total_images} images in {num_batches} batches",
            },
        )

        publish_progress(
            parent_job_id,
            "PROGRESS",
            current=0,
            total=total_images,
            message=f"Processing {total_images} images in {num_batches} batches",
            extra={"phase": "processing", "batches_total": num_batches},
        )

        return {
            "status": "dispatched",
            "catalog_id": catalog_id,
            "total_images": total_images,
            "num_batches": num_batches,
            "message": f"Processing {total_images} images in {num_batches} batches",
        }

    except Exception as e:
        logger.error(f"[{parent_job_id}] Coordinator failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        raise


@app.task(bind=True, name="thumbnail_worker")
def thumbnail_worker_task(
    self: Any,
    catalog_id: str,
    batch_id: str,
    parent_job_id: str,
    sizes: List[int],
    quality: int,
    force: bool,
) -> Dict[str, Any]:
    """Worker task that processes a batch of images for thumbnails."""
    worker_id = self.request.id or "unknown"
    logger.info(f"[{worker_id}] Starting thumbnail worker for batch {batch_id}")

    batch_manager = BatchManager(catalog_id, parent_job_id, "thumbnails")

    try:
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
        image_data = batch_data["work_items"]  # List of (image_id, source_path)
        items_count = batch_data["items_count"]

        logger.info(
            f"[{worker_id}] Processing batch {batch_number + 1}/{total_batches} ({items_count} images)"
        )

        result = BatchResult(batch_id=batch_id, batch_number=batch_number)

        thumbnail_base = Path(f"/app/catalogs/{catalog_id}/thumbnails")
        thumbnail_base.mkdir(parents=True, exist_ok=True)

        for image_id, source_path in image_data:
            source = Path(source_path)
            if not source.exists():
                result.error_count += 1
                result.errors.append(
                    {"image_id": image_id, "error": "Source file not found"}
                )
                continue

            try:
                for size in sizes:
                    thumbnail_path = thumbnail_base / f"{image_id}_{size}.jpg"
                    if thumbnail_path.exists() and not force:
                        continue
                    generate_thumbnail(source, thumbnail_path, (size, size), quality)

                result.success_count += 1
                result.processed_count += 1

            except Exception as e:
                result.error_count += 1
                result.errors.append({"image_id": image_id, "error": str(e)})

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

        logger.info(
            f"[{worker_id}] Batch {batch_number + 1} complete: {result.success_count} success, {result.error_count} errors"
        )

        return {
            "batch_id": batch_id,
            "batch_number": batch_number,
            "status": "completed",
            "success_count": result.success_count,
            "error_count": result.error_count,
        }

    except Exception as e:
        logger.error(f"[{worker_id}] Worker failed: {e}", exc_info=True)
        try:
            batch_manager.fail_batch(batch_id, str(e))
        except Exception:
            pass
        return {"batch_id": batch_id, "status": "failed", "error": str(e)}


@app.task(bind=True, base=ProgressTask, name="thumbnail_finalizer")
def thumbnail_finalizer_task(
    self: ProgressTask,
    worker_results: List[Dict[str, Any]],
    catalog_id: str,
    parent_job_id: str,
    sizes: Optional[List[int]] = None,
    quality: int = 85,
    force: bool = False,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """Finalizer that aggregates thumbnail generation results.

    If there are too many failed batches, automatically queues a continuation
    job to process remaining images without thumbnails.
    """
    finalizer_id = self.request.id or "unknown"
    logger.info(f"[{finalizer_id}] Starting finalizer for job {parent_job_id}")

    try:
        self.update_progress(0, 1, "Aggregating results...", {"phase": "finalizing"})

        batch_manager = BatchManager(catalog_id, parent_job_id, "thumbnails")

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
        failed_batches = sum(1 for wr in worker_results if wr.get("status") == "failed")

        # If there were too many failed batches, auto-requeue to continue
        if failed_batches >= CONSECUTIVE_FAILURE_THRESHOLD:
            logger.warning(
                f"[{finalizer_id}] {failed_batches} batches failed, auto-requeuing continuation"
            )

            cancel_and_requeue_job(
                parent_job_id=parent_job_id,
                catalog_id=catalog_id,
                job_type="thumbnails",
                task_name="thumbnail_coordinator",
                task_kwargs={
                    "catalog_id": catalog_id,
                    "sizes": sizes or [256, 512],
                    "quality": quality,
                    "force": False,  # Don't regenerate, only missing
                    "batch_size": batch_size,
                },
                reason=f"{failed_batches} batch failures",
                processed_so_far=total_success,
            )

            return {
                "status": "requeued",
                "catalog_id": catalog_id,
                "thumbnails_generated": total_success,
                "errors": total_errors,
                "failed_batches": failed_batches,
                "message": f"Job requeued due to {failed_batches} batch failures",
            }

        final_result = {
            "status": "completed" if failed_batches == 0 else "completed_with_errors",
            "catalog_id": catalog_id,
            "thumbnails_generated": total_success,
            "errors": total_errors,
            "failed_batches": failed_batches,
        }

        publish_completion(parent_job_id, "SUCCESS", result=final_result)
        _update_job_status(parent_job_id, "SUCCESS", result=final_result)

        self.update_progress(
            progress.total_items,
            progress.total_items,
            f"Complete: {total_success} thumbnails generated",
            {"phase": "complete"},
        )

        logger.info(
            f"[{finalizer_id}] Thumbnail generation complete: {total_success} success, {total_errors} errors"
        )

        return final_result

    except Exception as e:
        logger.error(f"[{finalizer_id}] Finalizer failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        _update_job_status(parent_job_id, "FAILURE", error=str(e))
        raise
