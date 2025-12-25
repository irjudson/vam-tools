"""
Parallel Job Coordinator Pattern for VAM Tools.

This module provides a standardized pattern for parallelizing large jobs
across multiple Celery workers with:
- Restartable batches tracked in database
- Aggregated progress reporting
- Fault tolerance (failed batches can be retried independently)
- No duplicate work (batches are claimed by batch_number)

Architecture:
    COORDINATOR → spawns → [WORKER_1, WORKER_2, ..., WORKER_N] → triggers → FINALIZER

Usage:
    1. Coordinator discovers all work items
    2. Coordinator creates job_batches records (PENDING)
    3. Coordinator spawns worker tasks via Celery chord
    4. Workers claim and process batches, update status
    5. Finalizer aggregates results when all workers complete
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

from sqlalchemy import text

from ..db import CatalogDB as CatalogDatabase
from .progress_publisher import publish_progress

logger = logging.getLogger(__name__)

# Type variable for work items (file paths, image IDs, etc.)
T = TypeVar("T")


class JobCancelledException(Exception):
    """Raised when a worker detects that its job has been cancelled."""

    pass


@dataclass
class BatchResult:
    """Result from processing a single batch."""

    batch_id: str
    batch_number: int
    processed_count: int = 0
    success_count: int = 0
    error_count: int = 0
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class JobProgress:
    """Aggregated progress across all batches."""

    parent_job_id: str
    total_batches: int
    completed_batches: int
    running_batches: int
    pending_batches: int
    failed_batches: int
    total_items: int
    processed_items: int
    success_items: int
    error_items: int

    @property
    def percent_complete(self) -> int:
        """Calculate overall completion percentage."""
        if self.total_items == 0:
            return 0
        return int((self.processed_items / self.total_items) * 100)

    @property
    def is_complete(self) -> bool:
        """Check if all batches are done (completed or failed)."""
        return (self.completed_batches + self.failed_batches) >= self.total_batches


class BatchManager:
    """
    Manages batch creation, claiming, and status updates.

    This class handles all database operations for the job_batches table.
    """

    def __init__(self, catalog_id: str, parent_job_id: str, job_type: str):
        """
        Initialize the batch manager.

        Args:
            catalog_id: The catalog UUID
            parent_job_id: The coordinator task's Celery ID
            job_type: Type of job (scan, tag, duplicates, etc.)
        """
        self.catalog_id = catalog_id
        self.parent_job_id = parent_job_id
        self.job_type = job_type

    def create_batches(
        self,
        work_items: List[Any],
        batch_size: int = 1000,
        db: Optional[CatalogDatabase] = None,
    ) -> List[str]:
        """
        Create batch records in the database.

        Args:
            work_items: List of items to process (file paths, image IDs, etc.)
            batch_size: Number of items per batch
            db: Optional database connection (will create one if not provided)

        Returns:
            List of batch IDs created
        """
        total_items = len(work_items)
        total_batches = math.ceil(total_items / batch_size) if total_items > 0 else 0

        if total_batches == 0:
            logger.info(f"No work items to batch for job {self.parent_job_id}")
            return []

        batch_ids = []

        def _create(session: Any) -> List[str]:
            ids = []
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_items)
                batch_items = work_items[start_idx:end_idx]

                batch_id = str(uuid.uuid4())

                session.execute(
                    text(
                        """
                        INSERT INTO job_batches (
                            id, parent_job_id, catalog_id, batch_number, total_batches,
                            job_type, work_items, items_count, status
                        ) VALUES (
                            :id, :parent_job_id, :catalog_id, :batch_number, :total_batches,
                            :job_type, :work_items, :items_count, 'PENDING'
                        )
                    """
                    ),
                    {
                        "id": batch_id,
                        "parent_job_id": self.parent_job_id,
                        "catalog_id": self.catalog_id,
                        "batch_number": batch_num,
                        "total_batches": total_batches,
                        "job_type": self.job_type,
                        "work_items": json.dumps(batch_items),
                        "items_count": len(batch_items),
                    },
                )
                ids.append(batch_id)

            session.commit()
            return ids

        if db:
            batch_ids = _create(db.session)
        else:
            with CatalogDatabase(self.catalog_id) as db_conn:
                batch_ids = _create(db_conn.session)

        logger.info(
            f"Created {len(batch_ids)} batches for job {self.parent_job_id} "
            f"({total_items} items, {batch_size} per batch)"
        )

        return batch_ids

    def claim_batch(
        self, batch_id: str, worker_id: str, db: Optional[CatalogDatabase] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Claim a batch for processing.

        Uses optimistic locking - only claims if status is PENDING.

        Args:
            batch_id: The batch UUID to claim
            worker_id: The Celery task ID of the worker
            db: Optional database connection

        Returns:
            Batch data if claimed, None if already claimed/processed
        """

        def _claim(session: Any) -> Optional[Dict[str, Any]]:
            # Try to claim the batch (only if PENDING)
            result = session.execute(
                text(
                    """
                    UPDATE job_batches
                    SET status = 'RUNNING',
                        worker_id = :worker_id,
                        started_at = NOW(),
                        updated_at = NOW()
                    WHERE id = :batch_id AND status = 'PENDING'
                    RETURNING id, batch_number, total_batches, work_items, items_count
                """
                ),
                {"batch_id": batch_id, "worker_id": worker_id},
            )
            row = result.fetchone()
            session.commit()

            if row:
                # work_items is stored as JSONB so SQLAlchemy auto-deserializes it
                work_items = row[3]
                if isinstance(work_items, str):
                    work_items = json.loads(work_items)
                return {
                    "id": str(row[0]),
                    "batch_number": row[1],
                    "total_batches": row[2],
                    "work_items": work_items,
                    "items_count": row[4],
                }
            return None

        if db:
            return _claim(db.session)
        else:
            with CatalogDatabase(self.catalog_id) as db_conn:
                return _claim(db_conn.session)

    def complete_batch(
        self,
        batch_id: str,
        result: BatchResult,
        db: Optional[CatalogDatabase] = None,
    ) -> None:
        """
        Mark a batch as completed with results.

        Args:
            batch_id: The batch UUID
            result: BatchResult with processing outcome
            db: Optional database connection
        """

        def _complete(session: Any) -> None:
            session.execute(
                text(
                    """
                    UPDATE job_batches
                    SET status = 'COMPLETED',
                        completed_at = NOW(),
                        updated_at = NOW(),
                        processed_count = :processed_count,
                        success_count = :success_count,
                        error_count = :error_count,
                        results = :results
                    WHERE id = :batch_id
                """
                ),
                {
                    "batch_id": batch_id,
                    "processed_count": result.processed_count,
                    "success_count": result.success_count,
                    "error_count": result.error_count,
                    "results": json.dumps(result.results),
                },
            )
            session.commit()

        if db:
            _complete(db.session)
        else:
            with CatalogDatabase(self.catalog_id) as db_conn:
                _complete(db_conn.session)

        logger.debug(
            f"Batch {batch_id} completed: {result.success_count} success, "
            f"{result.error_count} errors"
        )

    def fail_batch(
        self,
        batch_id: str,
        error_message: str,
        db: Optional[CatalogDatabase] = None,
    ) -> None:
        """
        Mark a batch as failed.

        Args:
            batch_id: The batch UUID
            error_message: Description of the failure
            db: Optional database connection
        """

        def _fail(session: Any) -> None:
            session.execute(
                text(
                    """
                    UPDATE job_batches
                    SET status = 'FAILED',
                        completed_at = NOW(),
                        updated_at = NOW(),
                        error_message = :error_message
                    WHERE id = :batch_id
                """
                ),
                {"batch_id": batch_id, "error_message": error_message},
            )
            session.commit()

        if db:
            _fail(db.session)
        else:
            with CatalogDatabase(self.catalog_id) as db_conn:
                _fail(db_conn.session)

        logger.warning(f"Batch {batch_id} failed: {error_message}")

    def is_cancelled(
        self, batch_id: Optional[str] = None, db: Optional[CatalogDatabase] = None
    ) -> bool:
        """
        Check if a batch or the parent job has been cancelled.

        Workers should call this periodically to check if they should stop processing.

        Args:
            batch_id: Optional batch UUID to check. If None, checks parent job.
            db: Optional database connection

        Returns:
            True if cancelled, False otherwise
        """

        def _check(session: Any) -> bool:
            if batch_id:
                # Check specific batch status
                result = session.execute(
                    text(
                        """
                        SELECT status FROM job_batches
                        WHERE id = :batch_id
                    """
                    ),
                    {"batch_id": batch_id},
                )
                row = result.fetchone()
                if row and row[0] == "CANCELLED":
                    return True

            # Check parent job status
            result = session.execute(
                text(
                    """
                    SELECT status FROM jobs
                    WHERE id = :parent_job_id
                """
                ),
                {"parent_job_id": self.parent_job_id},
            )
            row = result.fetchone()
            if row and row[0] in ("REVOKED", "CANCELLED"):
                return True

            return False

        if db:
            return _check(db.session)
        else:
            with CatalogDatabase(self.catalog_id) as db_conn:
                return _check(db_conn.session)

    def get_progress(self, db: Optional[CatalogDatabase] = None) -> JobProgress:
        """
        Get aggregated progress for the parent job.

        Args:
            db: Optional database connection

        Returns:
            JobProgress with aggregated statistics
        """

        def _get_progress(session: Any) -> JobProgress:
            result = session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) as total_batches,
                        COUNT(*) FILTER (WHERE status = 'COMPLETED') as completed,
                        COUNT(*) FILTER (WHERE status = 'RUNNING') as running,
                        COUNT(*) FILTER (WHERE status = 'PENDING') as pending,
                        COUNT(*) FILTER (WHERE status = 'FAILED') as failed,
                        COALESCE(SUM(items_count), 0) as total_items,
                        COALESCE(SUM(processed_count), 0) as processed_items,
                        COALESCE(SUM(success_count), 0) as success_items,
                        COALESCE(SUM(error_count), 0) as error_items
                    FROM job_batches
                    WHERE parent_job_id = :parent_job_id
                """
                ),
                {"parent_job_id": self.parent_job_id},
            )
            row = result.fetchone()

            return JobProgress(
                parent_job_id=self.parent_job_id,
                total_batches=row[0] or 0,
                completed_batches=row[1] or 0,
                running_batches=row[2] or 0,
                pending_batches=row[3] or 0,
                failed_batches=row[4] or 0,
                total_items=row[5] or 0,
                processed_items=row[6] or 0,
                success_items=row[7] or 0,
                error_items=row[8] or 0,
            )

        if db:
            return _get_progress(db.session)
        else:
            with CatalogDatabase(self.catalog_id) as db_conn:
                return _get_progress(db_conn.session)

    def get_batch_ids(self, db: Optional[CatalogDatabase] = None) -> List[str]:
        """
        Get all batch IDs for this job.

        Args:
            db: Optional database connection

        Returns:
            List of batch UUIDs
        """

        def _get_ids(session: Any) -> List[str]:
            result = session.execute(
                text(
                    """
                    SELECT id FROM job_batches
                    WHERE parent_job_id = :parent_job_id
                    ORDER BY batch_number
                """
                ),
                {"parent_job_id": self.parent_job_id},
            )
            return [str(row[0]) for row in result.fetchall()]

        if db:
            return _get_ids(db.session)
        else:
            with CatalogDatabase(self.catalog_id) as db_conn:
                return _get_ids(db_conn.session)

    def get_stale_batches(
        self, stale_minutes: int = 30, db: Optional[CatalogDatabase] = None
    ) -> List[str]:
        """
        Find batches that have been RUNNING for too long (likely dead workers).

        Args:
            stale_minutes: Minutes after which a RUNNING batch is considered stale
            db: Optional database connection

        Returns:
            List of stale batch IDs
        """

        def _get_stale(session: Any) -> List[str]:
            result = session.execute(
                text(
                    """
                    SELECT id FROM job_batches
                    WHERE parent_job_id = :parent_job_id
                    AND status = 'RUNNING'
                    AND started_at < NOW() - INTERVAL ':minutes minutes'
                """.replace(
                        ":minutes", str(stale_minutes)
                    )
                ),
                {"parent_job_id": self.parent_job_id},
            )
            return [str(row[0]) for row in result.fetchall()]

        if db:
            return _get_stale(db.session)
        else:
            with CatalogDatabase(self.catalog_id) as db_conn:
                return _get_stale(db_conn.session)

    def reset_batch(self, batch_id: str, db: Optional[CatalogDatabase] = None) -> None:
        """
        Reset a batch to PENDING state (for retry).

        Args:
            batch_id: The batch UUID to reset
            db: Optional database connection
        """

        def _reset(session: Any) -> None:
            session.execute(
                text(
                    """
                    UPDATE job_batches
                    SET status = 'PENDING',
                        worker_id = NULL,
                        started_at = NULL,
                        completed_at = NULL,
                        updated_at = NOW(),
                        processed_count = 0,
                        success_count = 0,
                        error_count = 0,
                        results = NULL,
                        error_message = NULL
                    WHERE id = :batch_id
                """
                ),
                {"batch_id": batch_id},
            )
            session.commit()

        if db:
            _reset(db.session)
        else:
            with CatalogDatabase(self.catalog_id) as db_conn:
                _reset(db_conn.session)

        logger.info(f"Reset batch {batch_id} to PENDING")

    def cleanup_batches(self, db: Optional[CatalogDatabase] = None) -> int:
        """
        Delete all batches for this job (cleanup after completion).

        Args:
            db: Optional database connection

        Returns:
            Number of batches deleted
        """

        def _cleanup(session: Any) -> int:
            result = session.execute(
                text(
                    """
                    DELETE FROM job_batches
                    WHERE parent_job_id = :parent_job_id
                    RETURNING id
                """
                ),
                {"parent_job_id": self.parent_job_id},
            )
            count = len(result.fetchall())
            session.commit()
            return count

        if db:
            return _cleanup(db.session)
        else:
            with CatalogDatabase(self.catalog_id) as db_conn:
                return _cleanup(db_conn.session)


def publish_job_progress(
    parent_job_id: str,
    progress: JobProgress,
    message: str = "",
    phase: str = "processing",
) -> None:
    """
    Publish aggregated job progress to Redis for UI updates.

    This function ensures monotonically increasing progress by checking the current
    progress in Redis before publishing. If the new progress shows fewer completed
    batches than what's already in Redis, the update is skipped to prevent the
    frontend from seeing progress go backwards due to out-of-order worker updates.

    Args:
        parent_job_id: The coordinator's task ID
        progress: Aggregated progress from BatchManager.get_progress()
        message: Human-readable status message (used as sub_message)
        phase: Current phase (discovery, processing, finalizing, complete)
    """
    from .progress_publisher import get_last_progress

    # Check current progress in Redis to prevent publishing stale updates
    # This prevents race conditions where Worker A completes batch 5 and publishes,
    # but then Worker B's delayed update for batch 3 overwrites it
    last_progress = get_last_progress(parent_job_id)
    if last_progress:
        last_completed = last_progress.get("progress", {}).get("batches_completed", 0)
        if progress.completed_batches < last_completed:
            # Skip this update - it's older than what's already published
            logger.debug(
                f"[{parent_job_id}] Skipping stale progress update: "
                f"current={last_completed} batches, new={progress.completed_batches} batches"
            )
            return

    # Build a consistent aggregate message
    if progress.total_items > 0:
        aggregate_message = (
            f"Processing {progress.total_items} items: "
            f"{progress.completed_batches}/{progress.total_batches} batches complete "
            f"({progress.percent_complete}%)"
        )
    else:
        aggregate_message = f"Processing: {progress.completed_batches}/{progress.total_batches} batches complete"

    publish_progress(
        job_id=parent_job_id,
        state="PROGRESS",
        current=progress.processed_items,
        total=progress.total_items,
        message=aggregate_message,
        extra={
            "phase": phase,
            "sub_message": message,  # The detailed batch message goes here
            "batches_total": progress.total_batches,
            "batches_completed": progress.completed_batches,
            "batches_running": progress.running_batches,
            "batches_pending": progress.pending_batches,
            "batches_failed": progress.failed_batches,
            "success_count": progress.success_items,
            "error_count": progress.error_items,
            "percent": progress.percent_complete,
        },
    )


# Registry of item processors for generic parallel worker
# Maps job_type -> callable(catalog_id, work_item, **kwargs) -> Dict[str, Any]
ITEM_PROCESSORS: Dict[str, Any] = {}


def register_item_processor(
    job_type: str,
) -> Callable[[Callable[..., Dict[str, Any]]], Callable[..., Dict[str, Any]]]:
    """
    Decorator to register an item processor for a job type.

    Usage:
        @register_item_processor("thumbnails")
        def process_thumbnail(catalog_id: str, work_item: Any, **kwargs) -> Dict[str, Any]:
            # Process single item
            return {"success": True, "result": {...}}
    """

    def decorator(
        func: Callable[..., Dict[str, Any]],
    ) -> Callable[..., Dict[str, Any]]:
        ITEM_PROCESSORS[job_type] = func
        return func

    return decorator


def get_item_processor(job_type: str) -> Callable[..., Dict[str, Any]]:
    """Get the item processor for a job type."""
    if job_type not in ITEM_PROCESSORS:
        raise ValueError(f"No item processor registered for job type: {job_type}")
    return ITEM_PROCESSORS[job_type]


# Configuration for auto-recovery across all parallel jobs
CONSECUTIVE_FAILURE_THRESHOLD = (
    3  # Cancel and requeue after this many consecutive failures
)
REQUEUE_DELAY_SECONDS = 30  # Wait before starting continuation job


def cancel_and_requeue_job(
    parent_job_id: str,
    catalog_id: str,
    job_type: str,
    task_name: str,
    task_kwargs: Dict[str, Any],
    reason: str,
    processed_so_far: int,
) -> str:
    """
    Cancel the current job and queue a continuation job.

    This is the shared auto-recovery mechanism for all parallel jobs.
    When a job encounters consecutive failures (GPU OOM, service crashes, etc.),
    it cancels itself and queues a new job that will pick up where it left off.

    Works because all parallel jobs query for "unprocessed" items, so restarting
    naturally continues from where it stopped.

    Args:
        parent_job_id: The current job's Celery task ID
        catalog_id: The catalog UUID
        job_type: Type of job for the Job record (e.g., "auto_tag", "scan")
        task_name: Celery task name to queue (e.g., "tagging_coordinator")
        task_kwargs: Keyword arguments for the continuation task
        reason: Human-readable reason for cancellation
        processed_so_far: Number of items successfully processed before cancel

    Returns:
        The new continuation job ID
    """
    import uuid as uuid_module
    from datetime import datetime

    from ..db import get_db_context
    from ..db.models import Job
    from .celery_app import app
    from .progress_publisher import publish_completion

    # Mark current job as cancelled with continuation info
    cancel_result = {
        "status": "cancelled_for_continuation",
        "reason": reason,
        "processed_before_cancel": processed_so_far,
        "message": f"Job cancelled after {reason}. Continuation job queued.",
    }

    try:
        with get_db_context() as session:
            job = session.query(Job).filter(Job.id == parent_job_id).first()
            if job:
                job.status = "CANCELLED"
                job.result = cancel_result
                session.commit()
    except Exception as e:
        logger.warning(f"Failed to update job status for {parent_job_id}: {e}")

    publish_completion(parent_job_id, "CANCELLED", result=cancel_result)

    logger.info(
        f"[{parent_job_id}] Cancelled job and queueing continuation after: {reason}"
    )

    # Create new job record
    new_job_id = str(uuid_module.uuid4())
    try:
        with get_db_context() as session:
            new_job = Job(
                id=new_job_id,
                catalog_id=catalog_id,
                job_type=job_type,
                status="PENDING",
                parameters={
                    **task_kwargs,
                    "continuation_of": parent_job_id,
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(new_job)
            session.commit()
    except Exception as e:
        logger.error(f"Failed to create continuation job record: {e}")

    # Queue the continuation task with delay
    task = app.tasks.get(task_name)
    if task:
        task.apply_async(
            kwargs=task_kwargs,
            task_id=new_job_id,
            countdown=REQUEUE_DELAY_SECONDS,
        )
        logger.info(
            f"[{parent_job_id}] Queued continuation job {new_job_id} "
            f"(starts in {REQUEUE_DELAY_SECONDS}s)"
        )
    else:
        logger.error(f"Task {task_name} not found, cannot queue continuation")

    return new_job_id
