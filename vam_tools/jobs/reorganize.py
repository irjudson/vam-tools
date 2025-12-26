"""
Parallel File Reorganization using the Coordinator Pattern.

This module implements parallel file reorganization across Celery workers:

1. reorganize_coordinator_task: Queries images, creates batches, spawns workers
2. reorganize_worker_task: Reorganizes a batch of images
3. reorganize_finalizer_task: Aggregates results and creates transaction log

Architecture:
    COORDINATOR → [WORKER_1, ..., WORKER_N] → FINALIZER
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from celery import chord, group
from sqlalchemy import text

from ..core.types import DateInfo, ImageRecord
from ..db import CatalogDB as CatalogDatabase
from ..db import get_db_context
from ..db.models import Job
from ..organization.reorganizer import should_reorganize_image
from ..organization.strategy import (
    DirectoryStructure,
    NamingStrategy,
    OrganizationStrategy,
)
from ..shared.media_utils import compute_checksum
from .celery_app import app
from .coordinator import BatchManager, BatchResult
from .progress_publisher import publish_completion, publish_progress
from .tasks import ProgressTask

logger = logging.getLogger(__name__)


@app.task(bind=True, base=ProgressTask, name="reorganize_coordinator")
def reorganize_coordinator_task(
    self: ProgressTask,
    catalog_id: str,
    output_directory: str,
    operation: str = "copy",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Coordinator task for file reorganization.

    Args:
        catalog_id: UUID of catalog to reorganize
        output_directory: Target directory for organized files
        operation: "copy" or "move"
        dry_run: If True, preview without executing

    Returns:
        Status and batch information
    """
    parent_job_id = self.request.id or "unknown"
    logger.info(f"[{parent_job_id}] Starting reorganization for catalog {catalog_id}")

    try:
        self.update_progress(0, 1, "Querying images...", {"phase": "init"})

        # Get all images from catalog
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            result = db.session.execute(
                text(
                    """
                    SELECT id FROM images
                    WHERE catalog_id = :catalog_id
                """
                ),
                {"catalog_id": catalog_id},
            )
            image_ids = [row[0] for row in result.fetchall()]

        total_images = len(image_ids)
        logger.info(f"[{parent_job_id}] Found {total_images} images to reorganize")

        if total_images == 0:
            publish_completion(
                parent_job_id,
                "SUCCESS",
                result={"status": "completed", "message": "No images in catalog"},
            )
            return {"status": "completed", "message": "No images in catalog"}

        # Create batches
        self.update_progress(
            0,
            total_images,
            f"Creating batches for {total_images} images...",
            {"phase": "batching"},
        )

        batch_manager = BatchManager(catalog_id, parent_job_id, "reorganize")
        batch_size = 500

        with CatalogDatabase(catalog_id) as db:
            batch_ids = batch_manager.create_batches(
                work_items=[(img_id,) for img_id in image_ids],
                batch_size=batch_size,
                db=db,
            )

        num_batches = len(batch_ids)
        logger.info(f"[{parent_job_id}] Created {num_batches} batches")

        # Spawn worker tasks
        self.update_progress(
            0,
            total_images,
            f"Spawning {num_batches} worker tasks...",
            {"phase": "spawning"},
        )

        worker_tasks = group(
            reorganize_worker_task.s(
                catalog_id=catalog_id,
                batch_id=batch_id,
                parent_job_id=parent_job_id,
                output_directory=output_directory,
                operation=operation,
                dry_run=dry_run,
            )
            for batch_id in batch_ids
        )

        # Finalizer collects results
        finalizer = reorganize_finalizer_task.s(
            catalog_id=catalog_id,
            parent_job_id=parent_job_id,
            output_directory=output_directory,
        )

        chord(worker_tasks)(finalizer)

        logger.info(f"[{parent_job_id}] Dispatched {num_batches} workers → finalizer")

        # Update job to STARTED
        with get_db_context() as session:
            job = session.query(Job).filter(Job.id == parent_job_id).first()
            if job:
                job.status = "STARTED"
                job.result = {
                    "status": "processing",
                    "total_images": total_images,
                    "message": f"Processing {total_images} images",
                }
                session.commit()

        publish_progress(
            parent_job_id,
            "PROGRESS",
            current=0,
            total=total_images,
            message=f"Processing {total_images} images",
            extra={"phase": "reorganizing"},
        )

        return {
            "status": "dispatched",
            "total_images": total_images,
            "total_batches": num_batches,
            "output_directory": output_directory,
        }

    except Exception as e:
        logger.error(f"[{parent_job_id}] Coordinator failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        raise


@app.task(bind=True, base=ProgressTask, name="reorganize_worker")
def reorganize_worker_task(
    self: ProgressTask,
    catalog_id: str,
    batch_id: str,
    parent_job_id: str,
    output_directory: str,
    operation: str,
    dry_run: bool,
) -> Dict[str, Any]:
    """Worker task - processes one batch of images.

    Args:
        catalog_id: UUID of catalog
        batch_id: Batch ID from batch manager
        parent_job_id: Parent job ID
        output_directory: Target directory
        operation: "copy" or "move"
        dry_run: If True, preview without executing

    Returns:
        Batch processing results
    """
    worker_id = self.request.id or "unknown"
    logger.info(f"[{worker_id}] Processing batch {batch_id}")

    batch_manager = BatchManager(catalog_id, parent_job_id, "reorganize")

    try:
        # Claim batch
        with CatalogDatabase(catalog_id) as db:
            batch_data = batch_manager.claim_batch(batch_id, worker_id, db)

        if not batch_data:
            logger.warning(f"[{worker_id}] Batch {batch_id} already claimed")
            return {"status": "skipped", "message": "Batch already claimed"}

        work_items = batch_data["work_items"]
        batch_number = batch_data["batch_number"]
        image_ids = [item[0] for item in work_items]

        # Create strategy
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_SLASH_MONTH_DAY,
            naming_strategy=NamingStrategy.TIME_CHECKSUM,
        )

        output_path = Path(output_directory)

        # Counters
        organized = 0
        skipped = 0
        failed = 0
        mtime_fallback_count = 0
        errors = []

        # Load images from database
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None

            for image_id in image_ids:
                try:
                    # Load image
                    result = db.session.execute(
                        text(
                            """
                            SELECT id, source_path, checksum, status_id, dates, metadata
                            FROM images
                            WHERE id = :image_id
                        """
                        ),
                        {"image_id": image_id},
                    )
                    row = result.fetchone()
                    if not row:
                        logger.warning(f"Image {image_id} not found")
                        skipped += 1
                        continue

                    # Build ImageRecord
                    from ..core.types import FileType, ImageMetadata

                    # Parse dates
                    dates_dict = row.dates if row.dates else {}
                    dates = DateInfo()
                    if dates_dict and "selected_date" in dates_dict:
                        from dateutil.parser import parse  # type: ignore

                        dates = DateInfo(
                            selected_date=parse(dates_dict["selected_date"])
                        )

                    image = ImageRecord(
                        id=row.id,
                        source_path=Path(row.source_path),
                        checksum=row.checksum,
                        status_id=row.status_id or "active",
                        dates=dates,
                        file_type=FileType.IMAGE,
                        metadata=ImageMetadata(),
                    )

                    # Check if should reorganize
                    if not should_reorganize_image(image, output_path):
                        skipped += 1
                        continue

                    # Get target path (with mtime fallback)
                    if not image.dates or not image.dates.selected_date:
                        mtime_fallback_count += 1

                    target_path = strategy.get_target_path(
                        output_path, image, use_mtime_fallback=True
                    )

                    if not target_path:
                        logger.warning(
                            f"Could not determine target path for {image_id}"
                        )
                        skipped += 1
                        continue

                    # Check for conflict
                    if target_path.exists():
                        target_checksum = compute_checksum(target_path)
                        if target_checksum == image.checksum:
                            # Already organized
                            skipped += 1
                            continue
                        else:
                            # Conflict - use full checksum
                            target_path = strategy.resolve_conflict_with_full_checksum(
                                output_path, image, target_path
                            )

                    # Execute operation
                    if dry_run:
                        logger.info(
                            f"[DRY RUN] Would {operation} {image.source_path} → {target_path}"
                        )
                        organized += 1
                    else:
                        # Create target directory
                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        # Copy or move
                        if operation == "copy":
                            shutil.copy2(image.source_path, target_path)
                        elif operation == "move":
                            shutil.move(str(image.source_path), str(target_path))

                        # Verify checksum
                        new_checksum = compute_checksum(target_path)
                        if new_checksum != image.checksum:
                            target_path.unlink()
                            raise ValueError(
                                f"Checksum mismatch: expected {image.checksum}, got {new_checksum}"
                            )

                        # Update database
                        db.session.execute(
                            text(
                                """
                                UPDATE images
                                SET source_path = :new_path
                                WHERE id = :image_id
                                  AND source_path != :new_path
                            """
                            ),
                            {"image_id": image_id, "new_path": str(target_path)},
                        )

                        organized += 1
                        logger.info(f"Reorganized {image.source_path} → {target_path}")

                except Exception as e:
                    logger.error(f"Error processing {image_id}: {e}")
                    failed += 1
                    errors.append(f"{image_id}: {str(e)}")

            # Commit database updates
            if not dry_run:
                db.session.commit()

        # Mark batch as complete
        batch_result = BatchResult(
            batch_id=batch_id,
            batch_number=batch_number,
            processed_count=organized + skipped + failed,
            success_count=organized,
            error_count=failed,
            results={
                "organized": organized,
                "skipped": skipped,
                "mtime_fallback_count": mtime_fallback_count,
            },
            errors=[{"message": e} for e in errors[:100]],
        )

        with CatalogDatabase(catalog_id) as db:
            batch_manager.complete_batch(batch_id, batch_result, db)

        logger.info(
            f"[{worker_id}] Batch {batch_id} complete: {organized} organized, {skipped} skipped, {failed} failed"
        )

        return {
            "status": "completed",
            "batch_id": batch_id,
            "organized": organized,
            "skipped": skipped,
            "failed": failed,
            "mtime_fallback_count": mtime_fallback_count,
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"[{worker_id}] Batch {batch_id} failed: {e}", exc_info=True)

        with CatalogDatabase(catalog_id) as db:
            batch_manager.fail_batch(batch_id, str(e), db)

        raise


@app.task(bind=True, base=ProgressTask, name="reorganize_finalizer")
def reorganize_finalizer_task(
    self: ProgressTask,
    worker_results: list,
    catalog_id: str,
    parent_job_id: str,
    output_directory: str,
) -> Dict[str, Any]:
    """Finalizer task - aggregates results from all workers.

    Args:
        worker_results: List of results from worker tasks
        catalog_id: UUID of catalog
        parent_job_id: Parent job ID
        output_directory: Target directory

    Returns:
        Aggregated results
    """
    finalizer_id = self.request.id or "unknown"
    logger.info(f"[{finalizer_id}] Starting finalizer for job {parent_job_id}")

    try:
        # Aggregate statistics
        total_organized = sum(r.get("organized", 0) for r in worker_results)
        total_skipped = sum(r.get("skipped", 0) for r in worker_results)
        total_failed = sum(r.get("failed", 0) for r in worker_results)
        mtime_fallback_count = sum(
            r.get("mtime_fallback_count", 0) for r in worker_results
        )
        all_errors = [e for r in worker_results for e in r.get("errors", [])]

        total_files = total_organized + total_skipped + total_failed

        logger.info(
            f"[{finalizer_id}] Aggregated: {total_organized} organized, "
            f"{total_skipped} skipped, {total_failed} failed"
        )

        # Build transaction log
        transaction_log = {
            "transaction_id": parent_job_id,
            "catalog_id": catalog_id,
            "completed_at": datetime.utcnow().isoformat(),
            "output_directory": output_directory,
            "statistics": {
                "total_files": total_files,
                "organized": total_organized,
                "skipped": total_skipped,
                "failed": total_failed,
                "mtime_fallback": mtime_fallback_count,
            },
            "errors": all_errors[:100],  # First 100 errors
        }

        # Save transaction log
        log_dir = Path(output_directory) / ".vam_transactions"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{parent_job_id}.json"

        with open(log_path, "w") as f:
            json.dump(transaction_log, f, indent=2)

        logger.info(f"[{finalizer_id}] Saved transaction log to {log_path}")

        # Determine final status
        if total_failed == 0:
            status = "SUCCESS"
        elif total_failed / total_files < 0.1:  # Less than 10% failed
            status = "SUCCESS"  # With warnings
        else:
            status = "FAILURE"

        # Update job in database
        with get_db_context() as session:
            job = session.query(Job).filter(Job.id == parent_job_id).first()
            if job:
                job.status = status
                job.result = {
                    "status": "completed",
                    "statistics": transaction_log["statistics"],
                    "transaction_log": str(log_path),
                    "errors": all_errors[:100],
                }
                session.commit()

        # Publish completion
        publish_completion(
            parent_job_id,
            status,
            result={
                "status": "completed",
                "total_organized": total_organized,
                "total_skipped": total_skipped,
                "total_failed": total_failed,
                "mtime_fallback_count": mtime_fallback_count,
                "transaction_log": str(log_path),
            },
        )

        logger.info(f"[{finalizer_id}] Finalizer complete")

        return {
            "status": "completed",
            "total_organized": total_organized,
            "total_skipped": total_skipped,
            "total_failed": total_failed,
            "mtime_fallback_count": mtime_fallback_count,
            "errors": all_errors,
            "transaction_log": str(log_path),
        }

    except Exception as e:
        logger.error(f"[{finalizer_id}] Finalizer failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        raise
