"""
Parallel Burst Detection using the Coordinator Pattern.

This module implements parallel burst detection across Celery workers:

1. burst_coordinator_task: Queries images, creates batches, spawns workers
2. burst_worker_task: Processes a batch of images for burst detection
3. burst_finalizer_task: Aggregates results and creates burst records

Note: Burst detection works on time-sequential images. Batches are divided
by time ranges so workers can detect bursts within their time window.
The finalizer merges any bursts that span batch boundaries.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from celery import chord, group
from sqlalchemy import text

from ..analysis.burst_detector import BurstDetector, ImageInfo
from ..db import CatalogDB as CatalogDatabase
from ..db.models import Job
from .celery_app import app
from .coordinator import BatchManager, BatchResult, publish_job_progress
from .progress_publisher import publish_completion, publish_progress
from .tasks import ProgressTask

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


@app.task(bind=True, base=ProgressTask, name="burst_coordinator")
def burst_coordinator_task(
    self: ProgressTask,
    catalog_id: str,
    gap_threshold: float = 2.0,
    min_burst_size: int = 3,
    batch_size: int = 5000,
) -> Dict[str, Any]:
    """
    Coordinator task for parallel burst detection.

    Args:
        catalog_id: UUID of the catalog
        gap_threshold: Maximum seconds between burst images
        min_burst_size: Minimum images to form a burst
        batch_size: Number of images per batch
    """
    parent_job_id = self.request.id or "unknown"
    logger.info(
        f"[{parent_job_id}] Starting burst coordinator for catalog {catalog_id}"
    )

    try:
        self.update_progress(0, 1, "Querying images...", {"phase": "init"})

        # Clear existing bursts for this catalog
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            db.session.execute(
                text("DELETE FROM bursts WHERE catalog_id = :catalog_id"),
                {"catalog_id": catalog_id},
            )
            assert db.session is not None
            db.session.commit()

            # Get all images with timestamps (sorted by time)
            assert db.session is not None
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

            # Build image data list: (id, timestamp_str, camera_make, camera_model, quality)
            image_data = []
            for row in result.fetchall():
                image_data.append(
                    (
                        str(row[0]),
                        row[1].isoformat() if row[1] else None,
                        row[2],
                        row[3],
                        row[4] or 0.0,
                    )
                )

        total_images = len(image_data)
        logger.info(f"[{parent_job_id}] Found {total_images} images with timestamps")

        if total_images == 0:
            publish_completion(
                parent_job_id,
                "SUCCESS",
                result={
                    "status": "completed",
                    "message": "No images with timestamps found",
                },
            )
            return {"status": "completed", "message": "No images with timestamps found"}

        # Create batches
        self.update_progress(
            0,
            total_images,
            f"Creating batches for {total_images} images...",
            {"phase": "batching"},
        )

        batch_manager = BatchManager(catalog_id, parent_job_id, "bursts")

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
            burst_worker_task.s(
                catalog_id=catalog_id,
                batch_id=batch_id,
                parent_job_id=parent_job_id,
                gap_threshold=gap_threshold,
                min_burst_size=min_burst_size,
            )
            for batch_id in batch_ids
        )

        finalizer = burst_finalizer_task.s(
            catalog_id=catalog_id,
            parent_job_id=parent_job_id,
            gap_threshold=gap_threshold,
            min_burst_size=min_burst_size,
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


@app.task(bind=True, name="burst_worker")
def burst_worker_task(
    self: Any,
    catalog_id: str,
    batch_id: str,
    parent_job_id: str,
    gap_threshold: float,
    min_burst_size: int,
) -> Dict[str, Any]:
    """Worker task that detects bursts within a batch of images."""
    worker_id = self.request.id or "unknown"
    logger.info(f"[{worker_id}] Starting burst worker for batch {batch_id}")

    batch_manager = BatchManager(catalog_id, parent_job_id, "bursts")

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
        image_data = batch_data[
            "work_items"
        ]  # List of (id, timestamp, camera_make, camera_model, quality)
        items_count = batch_data["items_count"]

        logger.info(
            f"[{worker_id}] Processing batch {batch_number + 1}/{total_batches} ({items_count} images)"
        )

        result = BatchResult(batch_id=batch_id, batch_number=batch_number)

        # Convert to ImageInfo objects
        images = []
        for item in image_data:
            img_id, timestamp_str, camera_make, camera_model, quality = item
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    images.append(
                        ImageInfo(
                            image_id=img_id,
                            timestamp=timestamp,
                            camera_make=camera_make,
                            camera_model=camera_model,
                            quality_score=quality,
                        )
                    )
                except ValueError:
                    logger.warning(
                        f"Invalid timestamp for image {img_id}: {timestamp_str}"
                    )

        # Detect bursts within this batch
        detector = BurstDetector(
            gap_threshold_seconds=gap_threshold,
            min_burst_size=min_burst_size,
        )
        bursts = detector.detect_bursts(images)

        logger.info(
            f"[{worker_id}] Detected {len(bursts)} bursts in batch {batch_number + 1}"
        )

        # Store burst data in results for the finalizer to aggregate
        burst_data = []
        for burst in bursts:
            burst_data.append(
                {
                    "image_ids": [img.image_id for img in burst.images],
                    "start_time": (
                        burst.start_time.isoformat() if burst.start_time else None
                    ),
                    "end_time": burst.end_time.isoformat() if burst.end_time else None,
                    "camera_make": burst.camera_make,
                    "camera_model": burst.camera_model,
                    "best_image_id": burst.best_image_id,
                    "selection_method": burst.selection_method,
                }
            )

        result.processed_count = len(images)
        result.success_count = len(bursts)
        result.results = {
            "bursts": burst_data,
            "first_timestamp": images[0].timestamp.isoformat() if images else None,
            "last_timestamp": images[-1].timestamp.isoformat() if images else None,
        }

        # Mark batch complete
        with CatalogDatabase(catalog_id) as db:
            batch_manager.complete_batch(batch_id, result, db)
            progress = batch_manager.get_progress(db)
            publish_job_progress(
                parent_job_id,
                progress,
                f"Batch {batch_number + 1}/{total_batches} complete ({len(bursts)} bursts)",
                phase="processing",
            )

        logger.info(
            f"[{worker_id}] Batch {batch_number + 1} complete: {len(bursts)} bursts detected"
        )

        return {
            "batch_id": batch_id,
            "batch_number": batch_number,
            "status": "completed",
            "bursts_count": len(bursts),
            "images_processed": len(images),
            "bursts": burst_data,
        }

    except Exception as e:
        logger.error(f"[{worker_id}] Worker failed: {e}", exc_info=True)
        try:
            batch_manager.fail_batch(batch_id, str(e))
        except Exception:
            pass
        return {"batch_id": batch_id, "status": "failed", "error": str(e)}


@app.task(bind=True, base=ProgressTask, name="burst_finalizer")
def burst_finalizer_task(
    self: ProgressTask,
    worker_results: List[Dict[str, Any]],
    catalog_id: str,
    parent_job_id: str,
    gap_threshold: float,
    min_burst_size: int,
) -> Dict[str, Any]:
    """Finalizer that aggregates burst detection results and saves to database."""
    finalizer_id = self.request.id or "unknown"
    logger.info(f"[{finalizer_id}] Starting finalizer for job {parent_job_id}")

    try:
        self.update_progress(
            0, 1, "Aggregating burst results...", {"phase": "finalizing"}
        )

        # Collect all bursts from workers, ordered by batch number
        all_bursts = []
        sorted_results = sorted(
            [wr for wr in worker_results if wr.get("status") == "completed"],
            key=lambda x: x.get("batch_number", 0),
        )

        for wr in sorted_results:
            bursts = wr.get("bursts", [])
            all_bursts.extend(bursts)

        logger.info(f"[{finalizer_id}] Collected {len(all_bursts)} bursts from workers")

        # Check for bursts that span batch boundaries and merge them
        # A burst at the end of batch N might continue into batch N+1
        merged_bursts = _merge_adjacent_bursts(
            all_bursts, gap_threshold, min_burst_size
        )

        logger.info(f"[{finalizer_id}] After merging: {len(merged_bursts)} bursts")

        # Save bursts to database
        self.update_progress(
            0,
            1,
            f"Saving {len(merged_bursts)} bursts to database...",
            {"phase": "saving"},
        )

        total_burst_images = 0
        with CatalogDatabase(catalog_id) as db:
            for burst in merged_bursts:
                burst_id = str(uuid.uuid4())
                image_ids = burst["image_ids"]
                total_burst_images += len(image_ids)

                # Calculate duration
                start_time = datetime.fromisoformat(burst["start_time"])
                end_time = datetime.fromisoformat(burst["end_time"])
                duration = (end_time - start_time).total_seconds()

                # Insert burst record
                assert db.session is not None
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
                        "image_count": len(image_ids),
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                        "camera_make": burst.get("camera_make"),
                        "camera_model": burst.get("camera_model"),
                        "best_image_id": burst.get("best_image_id"),
                        "selection_method": burst.get("selection_method"),
                    },
                )

                # Update images with burst_id and sequence
                for seq, img_id in enumerate(image_ids):
                    assert db.session is not None
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
                            "image_id": img_id,
                            "seq": seq,
                        },
                    )

            assert db.session is not None
            db.session.commit()

        batch_manager = BatchManager(catalog_id, parent_job_id, "bursts")
        with CatalogDatabase(catalog_id) as db:
            progress = batch_manager.get_progress(db)

        failed_batches = sum(1 for wr in worker_results if wr.get("status") == "failed")

        final_result = {
            "status": "completed" if failed_batches == 0 else "completed_with_errors",
            "catalog_id": catalog_id,
            "bursts_detected": len(merged_bursts),
            "total_burst_images": total_burst_images,
            "failed_batches": failed_batches,
        }

        publish_completion(parent_job_id, "SUCCESS", result=final_result)
        _update_job_status(parent_job_id, "SUCCESS", result=final_result)

        self.update_progress(
            progress.total_items,
            progress.total_items,
            f"Complete: {len(merged_bursts)} bursts detected",
            {"phase": "complete"},
        )

        logger.info(
            f"[{finalizer_id}] Burst detection complete: {len(merged_bursts)} bursts, {total_burst_images} images"
        )

        return final_result

    except Exception as e:
        logger.error(f"[{finalizer_id}] Finalizer failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        _update_job_status(parent_job_id, "FAILURE", error=str(e))
        raise


def _merge_adjacent_bursts(
    bursts: List[Dict[str, Any]],
    gap_threshold: float,
    min_burst_size: int,
) -> List[Dict[str, Any]]:
    """
    Merge bursts that span batch boundaries.

    If a burst at the end of batch N ends within gap_threshold seconds
    of when a burst at the start of batch N+1 begins, and they have
    the same camera, merge them.
    """
    if len(bursts) <= 1:
        return bursts

    merged = []
    current_burst = None

    for burst in bursts:
        if current_burst is None:
            current_burst = burst.copy()
            continue

        # Check if we should merge with current_burst
        current_end = datetime.fromisoformat(current_burst["end_time"])
        next_start = datetime.fromisoformat(burst["start_time"])
        gap = (next_start - current_end).total_seconds()

        # Same camera and within gap threshold?
        same_camera = current_burst.get("camera_make") == burst.get(
            "camera_make"
        ) and current_burst.get("camera_model") == burst.get("camera_model")

        if same_camera and gap <= gap_threshold:
            # Merge the bursts
            current_burst["image_ids"].extend(burst["image_ids"])
            current_burst["end_time"] = burst["end_time"]
            # Keep the best image with highest quality (would need quality info)
            # For now, keep the existing best_image_id from the first burst
        else:
            # Save current burst and start new one
            if len(current_burst["image_ids"]) >= min_burst_size:
                merged.append(current_burst)
            current_burst = burst.copy()

    # Don't forget the last burst
    if current_burst and len(current_burst["image_ids"]) >= min_burst_size:
        merged.append(current_burst)

    return merged
