"""
Parallel Tagging using the Coordinator Pattern.

This module implements parallel-ready tagging across Celery workers:

1. tagging_coordinator_task: Queries images, creates batches, spawns workers
2. tagging_worker_task: Processes a batch of images with the tagger
3. tagging_finalizer_task: Aggregates results

Note: While the coordinator pattern enables parallel processing, tagging
benefits most from batch processing on a single GPU worker due to model
loading overhead. With multiple GPU workers, batches can be processed
in parallel. With a single worker, batches run sequentially.

Auto-Recovery: If a worker encounters consecutive failures (e.g., GPU OOM,
Ollama crashes), it will cancel the current job and automatically requeue
a new one to continue from where it left off (since tag_mode="untagged_only"
naturally resumes).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import chord, group
from sqlalchemy import text

from ..db import CatalogDB as CatalogDatabase
from ..db.models import Job
from .celery_app import app
from .coordinator import (
    CONSECUTIVE_FAILURE_THRESHOLD,
    BatchManager,
    BatchResult,
    cancel_and_requeue_job,
    publish_job_progress,
)
from .progress_publisher import publish_completion, publish_progress
from .tasks import CoordinatorTask, ProgressTask, _store_image_tags

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


@app.task(bind=True, base=CoordinatorTask, name="tagging_coordinator")
def tagging_coordinator_task(
    self: CoordinatorTask,
    catalog_id: str,
    backend: str = "openclip",
    model: Optional[str] = None,
    threshold: float = 0.25,
    max_tags: int = 10,
    batch_size: int = 500,
    tag_mode: str = "untagged_only",
) -> Dict[str, Any]:
    """
    Coordinator task for parallel image tagging.

    Args:
        catalog_id: UUID of the catalog
        backend: "openclip", "ollama", or "combined"
        model: Model name (backend-specific)
        threshold: Minimum confidence threshold (0.0-1.0)
        max_tags: Maximum tags per image
        batch_size: Number of images per batch (default 500 for GPU efficiency)
        tag_mode: "untagged_only" or "all"
    """
    parent_job_id = self.request.id or "unknown"
    logger.info(
        f"[{parent_job_id}] Starting tagging coordinator for catalog {catalog_id}"
    )

    try:
        self.update_progress(0, 1, "Querying images...", {"phase": "init"})

        # Get images based on tag_mode
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            if tag_mode == "untagged_only":
                result = db.session.execute(
                    text(
                        """
                        SELECT i.id, i.source_path FROM images i
                        WHERE i.catalog_id = :catalog_id
                        AND i.file_type = 'image'
                        AND NOT EXISTS (
                            SELECT 1 FROM image_tags it WHERE it.image_id = i.id
                        )
                    """
                    ),
                    {"catalog_id": catalog_id},
                )
            else:
                assert db.session is not None
                result = db.session.execute(
                    text(
                        """
                        SELECT i.id, i.source_path FROM images i
                        WHERE i.catalog_id = :catalog_id
                        AND i.file_type = 'image'
                    """
                    ),
                    {"catalog_id": catalog_id},
                )
            image_data = [(str(row[0]), row[1]) for row in result.fetchall()]

        total_images = len(image_data)
        logger.info(f"[{parent_job_id}] Found {total_images} images for tagging")

        if total_images == 0:
            publish_completion(
                parent_job_id,
                "SUCCESS",
                result={"status": "completed", "message": "No images to tag"},
            )
            return {"status": "completed", "message": "No images to tag"}

        # Create batches
        self.update_progress(
            0,
            total_images,
            f"Creating batches for {total_images} images...",
            {"phase": "batching"},
        )

        batch_manager = BatchManager(catalog_id, parent_job_id, "tagging")

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
            tagging_worker_task.s(
                catalog_id=catalog_id,
                batch_id=batch_id,
                parent_job_id=parent_job_id,
                backend=backend,
                model=model,
                threshold=threshold,
                max_tags=max_tags,
            )
            for batch_id in batch_ids
        )

        finalizer = tagging_finalizer_task.s(
            catalog_id=catalog_id,
            parent_job_id=parent_job_id,
            backend=backend,
            model=model,
            threshold=threshold,
            max_tags=max_tags,
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
            extra={
                "phase": "processing",
                "batches_total": num_batches,
                "backend": backend,
            },
        )

        return {
            "status": "dispatched",
            "catalog_id": catalog_id,
            "total_images": total_images,
            "num_batches": num_batches,
            "backend": backend,
            "message": f"Processing {total_images} images in {num_batches} batches",
        }

    except Exception as e:
        logger.error(f"[{parent_job_id}] Coordinator failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        raise


@app.task(bind=True, name="tagging_worker")
def tagging_worker_task(
    self: Any,
    catalog_id: str,
    batch_id: str,
    parent_job_id: str,
    backend: str,
    model: Optional[str],
    threshold: float,
    max_tags: int,
) -> Dict[str, Any]:
    """Worker task that processes a batch of images for tagging."""
    from .job_metrics import check_gpu_available

    worker_id = self.request.id or "unknown"
    logger.info(f"[{worker_id}] Starting tagging worker for batch {batch_id}")

    batch_manager = BatchManager(catalog_id, parent_job_id, "tagging")

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

        # Initialize tagger for this batch
        from typing import Union

        from ..analysis.image_tagger import CombinedTagger, ImageTagger

        use_gpu = (
            check_gpu_available() if backend in ("openclip", "combined") else False
        )
        device = "cuda" if use_gpu else "cpu"

        tagger: Union[ImageTagger, CombinedTagger]
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
        with CatalogDatabase(catalog_id) as db:
            if backend in ("openclip", "combined"):
                # Batch processing for OpenCLIP/combined
                batch_paths = [Path(item[1]) for item in image_data]
                batch_ids = [item[0] for item in image_data]

                try:
                    tag_results = tagger.tag_batch(
                        list(map(str, batch_paths)),
                        threshold=threshold,
                        max_tags=max_tags,
                    )

                    for img_id, img_path in zip(batch_ids, batch_paths):
                        tags = tag_results.get(img_path, [])
                        if tags:
                            stored = _store_image_tags(
                                db, catalog_id, str(img_id), tags, backend
                            )
                            if stored > 0:
                                result.success_count += 1
                        result.processed_count += 1

                    assert db.session is not None
                    db.session.commit()

                except Exception as e:
                    logger.warning(f"[{worker_id}] Batch tagging failed: {e}")
                    result.error_count = items_count
                    result.errors.append({"batch_id": batch_id, "error": str(e)})

            else:
                # Individual processing for Ollama
                for img_id, source_path in image_data:
                    try:
                        tags = tagger.tag_image(
                            source_path,
                            threshold=threshold,
                            max_tags=max_tags,
                        )
                        if tags:
                            stored = _store_image_tags(
                                db, catalog_id, str(img_id), tags, "ollama"
                            )
                            if stored > 0:
                                result.success_count += 1
                        result.processed_count += 1
                    except Exception as e:
                        result.error_count += 1
                        result.errors.append({"image_id": img_id, "error": str(e)})

                assert db.session is not None
                db.session.commit()

            # Mark batch complete
            batch_manager.complete_batch(batch_id, result, db)
            progress = batch_manager.get_progress(db)
            publish_job_progress(
                parent_job_id,
                progress,
                f"Batch {batch_number + 1}/{total_batches} complete",
                phase="processing",
            )

        # Release GPU resources after processing batch
        tagger.cleanup()
        logger.info(f"[{worker_id}] GPU resources released")

        logger.info(
            f"[{worker_id}] Batch {batch_number + 1} complete: {result.success_count} tagged, {result.error_count} errors"
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

        # Release GPU resources even on failure
        try:
            if "tagger" in locals():
                tagger.cleanup()
                logger.info(f"[{worker_id}] GPU resources released after failure")
        except Exception as cleanup_err:
            logger.warning(
                f"[{worker_id}] Failed to cleanup GPU resources: {cleanup_err}"
            )

        try:
            batch_manager.fail_batch(batch_id, str(e))
        except Exception:
            pass
        return {"batch_id": batch_id, "status": "failed", "error": str(e)}


@app.task(bind=True, base=ProgressTask, name="tagging_finalizer")
def tagging_finalizer_task(
    self: ProgressTask,
    worker_results: List[Dict[str, Any]],
    catalog_id: str,
    parent_job_id: str,
    backend: str = "openclip",
    model: Optional[str] = None,
    threshold: float = 0.25,
    max_tags: int = 10,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """Finalizer that aggregates tagging results.

    If there are failed batches, automatically queues a continuation job
    to process remaining untagged images.
    """
    finalizer_id = self.request.id or "unknown"
    logger.info(f"[{finalizer_id}] Starting finalizer for job {parent_job_id}")

    try:
        self.update_progress(0, 1, "Aggregating results...", {"phase": "finalizing"})

        batch_manager = BatchManager(catalog_id, parent_job_id, "tagging")

        with CatalogDatabase(catalog_id) as db:
            progress = batch_manager.get_progress(db)

        total_tagged = sum(
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

        # If there were failed batches, auto-requeue to continue
        if failed_batches >= CONSECUTIVE_FAILURE_THRESHOLD:
            logger.warning(
                f"[{finalizer_id}] {failed_batches} batches failed, auto-requeuing continuation"
            )

            # Requeue with the shared helper
            cancel_and_requeue_job(
                parent_job_id=parent_job_id,
                catalog_id=catalog_id,
                job_type="auto_tag",
                task_name="tagging_coordinator",
                task_kwargs={
                    "catalog_id": catalog_id,
                    "backend": backend,
                    "model": model,
                    "threshold": threshold,
                    "max_tags": max_tags,
                    "batch_size": batch_size,
                    "tag_mode": "untagged_only",  # Always resume with untagged
                },
                reason=f"{failed_batches} batch failures",
                processed_so_far=total_tagged,
            )

            return {
                "status": "requeued",
                "catalog_id": catalog_id,
                "images_tagged": total_tagged,
                "errors": total_errors,
                "failed_batches": failed_batches,
                "message": f"Job requeued due to {failed_batches} batch failures",
            }

        final_result = {
            "status": "completed" if failed_batches == 0 else "completed_with_errors",
            "catalog_id": catalog_id,
            "images_tagged": total_tagged,
            "errors": total_errors,
            "failed_batches": failed_batches,
        }

        publish_completion(parent_job_id, "SUCCESS", result=final_result)
        _update_job_status(parent_job_id, "SUCCESS", result=final_result)

        self.update_progress(
            progress.total_items,
            progress.total_items,
            f"Complete: {total_tagged} images tagged",
            {"phase": "complete"},
        )

        logger.info(
            f"[{finalizer_id}] Tagging complete: {total_tagged} tagged, {total_errors} errors"
        )

        return final_result

    except Exception as e:
        logger.error(f"[{finalizer_id}] Finalizer failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        _update_job_status(parent_job_id, "FAILURE", error=str(e))
        raise
