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
from typing import Any, Dict, List, Optional, TypeVar

from sqlalchemy import text

from ..db import CatalogDB as CatalogDatabase
from .progress_publisher import publish_progress

logger = logging.getLogger(__name__)

# Type variable for work items (file paths, image IDs, etc.)
T = TypeVar("T")


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

    Args:
        parent_job_id: The coordinator's task ID
        progress: Aggregated progress from BatchManager.get_progress()
        message: Human-readable status message (used as sub_message)
        phase: Current phase (discovery, processing, finalizing, complete)
    """
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
