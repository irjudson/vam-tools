"""
Parallel quality scoring job.

Computes quality scores for all images based on:
- Format (RAW > JPEG > compressed)
- Resolution (higher megapixels = better)
- File size (larger = better for same format)
- EXIF completeness (more metadata = better)

Updates the quality_score column in the database.
"""

import logging
from typing import Any, Dict, List

from sqlalchemy import text

from ..analysis.quality_scorer import calculate_quality_score
from ..core.types import FileType, ImageMetadata
from ..db import CatalogDB as CatalogDatabase
from .celery_app import app
from .coordinator import BatchManager, BatchResult, publish_job_progress
from .tasks import CoordinatorTask, ProgressTask

logger = logging.getLogger(__name__)


@app.task(bind=True, base=CoordinatorTask, name="quality_coordinator")
def quality_coordinator_task(
    self: CoordinatorTask,
    catalog_id: str,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Coordinator task for quality scoring.

    Args:
        catalog_id: Catalog UUID
        force: If True, recompute scores even if already present

    Returns:
        Job result summary
    """
    parent_job_id = self.request.id or "unknown"
    logger.info(f"[{parent_job_id}] Starting quality scoring coordinator")

    batch_manager = BatchManager(catalog_id, parent_job_id, "quality")

    try:
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            # Get all images that need quality scoring
            if force:
                query = text(
                    """
                    SELECT id, metadata, file_type
                    FROM images
                    ORDER BY id
                """
                )
            else:
                query = text(
                    """
                    SELECT id, metadata, file_type
                    FROM images
                    WHERE quality_score IS NULL
                    ORDER BY id
                """
                )

            assert db.session is not None
            result = db.session.execute(query)
            images = [
                {
                    "image_id": row[0],
                    "metadata": row[1],
                    "file_type": row[2],
                }
                for row in result
            ]

            total_images = len(images)
            logger.info(f"[{parent_job_id}] Found {total_images:,} images to score")

            if total_images == 0:
                logger.info(f"[{parent_job_id}] No images need quality scoring")
                return {
                    "status": "completed",
                    "job_type": "quality",
                    "catalog_id": catalog_id,
                    "items_processed": 0,
                    "items_success": 0,
                    "items_failed": 0,
                    "total_batches": 0,
                    "failed_batches": 0,
                }

            # Create batches (500 images per batch)
            batch_size = 500
            batch_ids = batch_manager.create_batches(
                work_items=images,
                batch_size=batch_size,
                db=db,
            )

            # Get total batches from progress
            total_batches = len(batch_ids)
            logger.info(f"[{parent_job_id}] Created {total_batches} batches")

            # Publish initial progress
            publish_job_progress(
                parent_job_id,
                batch_manager.get_progress(db),
                f"Starting quality scoring for {total_images:,} images",
                phase="starting",
            )

        # Dispatch worker tasks using chord pattern
        # Chord runs all workers in parallel, then calls finalizer when all complete
        from celery import chord

        worker_tasks = [
            quality_worker_task.s(
                catalog_id=catalog_id,
                batch_id=batch_id,
                parent_job_id=parent_job_id,
                force=force,
            )
            for batch_id in batch_ids
        ]

        # Execute workers in parallel, then run finalizer
        finalizer_callback = quality_finalizer_task.s(
            catalog_id=catalog_id,
            parent_job_id=parent_job_id,
        )

        chord(worker_tasks)(finalizer_callback)

        logger.info(
            f"[{parent_job_id}] Quality chord dispatched: {total_batches} batches"
        )

        # Return immediately - workers will run asynchronously
        return {
            "status": "processing",
            "job_type": "quality",
            "catalog_id": catalog_id,
            "total_batches": total_batches,
            "message": f"Processing {total_images:,} images",
        }

    except Exception as e:
        logger.error(f"[{parent_job_id}] Coordinator failed: {e}", exc_info=True)
        raise


@app.task(bind=True, base=ProgressTask, name="quality_worker")
def quality_worker_task(
    self: ProgressTask,
    catalog_id: str,
    batch_id: str,
    parent_job_id: str,
    force: bool,
) -> Dict[str, Any]:
    """Worker task that processes a batch of images for quality scoring."""
    worker_id = self.request.id or "unknown"
    logger.info(f"[{worker_id}] Starting quality worker for batch {batch_id}")

    batch_manager = BatchManager(catalog_id, parent_job_id, "quality")

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
        image_data = batch_data["work_items"]
        items_count = batch_data["items_count"]

        logger.info(
            f"[{worker_id}] Processing batch {batch_number + 1}/{total_batches} "
            f"({items_count} images)"
        )

        result = BatchResult(batch_id=batch_id, batch_number=batch_number)

        # Process each image
        for item in image_data:
            image_id = item["image_id"]
            metadata_dict = item["metadata"]
            file_type_str = item["file_type"]

            try:
                # Convert metadata dict to ImageMetadata object
                metadata = ImageMetadata(
                    format=metadata_dict.get("format"),
                    width=metadata_dict.get("width"),
                    height=metadata_dict.get("height"),
                    size_bytes=metadata_dict.get("size_bytes"),
                    camera_make=metadata_dict.get("camera_make"),
                    camera_model=metadata_dict.get("camera_model"),
                    lens_model=metadata_dict.get("lens_model"),
                    focal_length=metadata_dict.get("focal_length"),
                    aperture=metadata_dict.get("aperture"),
                    shutter_speed=metadata_dict.get("shutter_speed"),
                    iso=metadata_dict.get("iso"),
                    gps_latitude=metadata_dict.get("gps_latitude"),
                    gps_longitude=metadata_dict.get("gps_longitude"),
                    gps_altitude=metadata_dict.get("gps_altitude"),
                )

                # Convert file_type string to enum
                file_type = (
                    FileType.IMAGE if file_type_str == "image" else FileType.VIDEO
                )

                # Calculate quality score
                quality = calculate_quality_score(metadata, file_type)

                # Update database
                with CatalogDatabase(catalog_id) as db:
                    assert db.session is not None
                    db.session.execute(
                        text(
                            """
                            UPDATE images
                            SET quality_score = :score,
                                processing_flags = jsonb_set(
                                    COALESCE(processing_flags, '{}'::jsonb),
                                    '{quality_scored}',
                                    'true'
                                )
                            WHERE id = :image_id
                        """
                        ),
                        {"score": int(quality.overall), "image_id": image_id},
                    )
                    assert db.session is not None
                    db.session.commit()

                result.success_count += 1
                result.processed_count += 1

            except Exception as e:
                logger.error(
                    f"[{worker_id}] Failed to score {image_id}: {e}", exc_info=True
                )
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
            f"[{worker_id}] Batch {batch_number + 1} complete: "
            f"{result.success_count} success, {result.error_count} errors"
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


@app.task(bind=True, base=ProgressTask, name="quality_finalizer")
def quality_finalizer_task(
    self: ProgressTask,
    worker_results: List[Dict[str, Any]],
    catalog_id: str,
    parent_job_id: str,
) -> Dict[str, Any]:
    """Finalizer task that aggregates results and marks job complete."""
    logger.info(f"[{parent_job_id}] Running quality finalizer")

    # Log worker results summary
    total_workers = len(worker_results)
    successful_workers = sum(
        1 for r in worker_results if r.get("status") == "completed"
    )
    logger.info(
        f"[{parent_job_id}] Worker results: {successful_workers}/{total_workers} completed"
    )

    batch_manager = BatchManager(catalog_id, parent_job_id, "quality")

    try:
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            # Get final statistics
            progress = batch_manager.get_progress(db)

            # Calculate average quality score
            assert db.session is not None
            avg_score = db.session.execute(
                text(
                    """
                    SELECT AVG(quality_score)::int
                    FROM images
                    WHERE quality_score IS NOT NULL
                """
                )
            ).scalar()

            # Get score distribution
            assert db.session is not None
            distribution = db.session.execute(
                text(
                    """
                    SELECT
                        CASE
                            WHEN quality_score >= 90 THEN 'excellent'
                            WHEN quality_score >= 75 THEN 'good'
                            WHEN quality_score >= 60 THEN 'fair'
                            ELSE 'poor'
                        END as category,
                        COUNT(*) as count
                    FROM images
                    WHERE quality_score IS NOT NULL
                    GROUP BY category
                    ORDER BY
                        CASE
                            WHEN quality_score >= 90 THEN 1
                            WHEN quality_score >= 75 THEN 2
                            WHEN quality_score >= 60 THEN 3
                            ELSE 4
                        END
                """
                )
            ).fetchall()

            score_dist = {row[0]: row[1] for row in distribution}

            # Publish final progress
            publish_job_progress(
                parent_job_id,
                progress,
                f"Quality scoring complete: avg={avg_score}",
                phase="completed",
            )

            logger.info(
                f"[{parent_job_id}] Quality scoring complete: "
                f"{progress.completed_batches}/{progress.total_batches} batches, "
                f"{progress.success_items} success, {progress.error_items} failed"
            )

            return {
                "status": "completed",
                "job_type": "quality",
                "catalog_id": catalog_id,
                "items_processed": progress.success_items + progress.error_items,
                "items_success": progress.success_items,
                "items_failed": progress.error_items,
                "total_batches": progress.total_batches,
                "failed_batches": progress.failed_batches,
                "average_score": avg_score,
                "score_distribution": score_dist,
            }

    except Exception as e:
        logger.error(f"[{parent_job_id}] Finalizer failed: {e}", exc_info=True)
        raise
