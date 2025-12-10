"""
Generic Job Recovery System for VAM Tools.

This module provides a centralized recovery mechanism for all parallel job types.
When workers die (container restart, crash, etc.), batches can be left in RUNNING
or PENDING state with no workers to process them. This module detects and recovers
such jobs.

Usage:
    # Recover a specific job
    from vam_tools.jobs.job_recovery import recover_job
    result = recover_job(job_id, stale_minutes=10)

    # Run periodic recovery check (all stale jobs)
    from vam_tools.jobs.job_recovery import job_recovery_check_task
    job_recovery_check_task.delay(stale_minutes=10)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from celery import chord, group
from sqlalchemy import text

from ..db import CatalogDB as CatalogDatabase
from ..db import get_db_context
from ..db.models import Job
from .celery_app import app
from .coordinator import BatchManager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Registry of job types and their worker/finalizer tasks
# Format: {job_type: (worker_task_name, finalizer_task_name, worker_kwargs_builder)}
JOB_TYPE_REGISTRY: Dict[str, Tuple[str, str, Optional[Callable]]] = {}


def register_job_type(
    job_type: str,
    worker_task_name: str,
    finalizer_task_name: str,
    worker_kwargs_builder: Optional[
        Callable[[Dict[str, Any], str], Dict[str, Any]]
    ] = None,
) -> None:
    """
    Register a job type for recovery.

    Args:
        job_type: The job type string (e.g., "scan", "auto_tag")
        worker_task_name: Celery task name for the worker
        finalizer_task_name: Celery task name for the finalizer
        worker_kwargs_builder: Optional function to build extra kwargs for worker
                              Signature: (job_params, batch_id) -> Dict[str, Any]
    """
    JOB_TYPE_REGISTRY[job_type] = (
        worker_task_name,
        finalizer_task_name,
        worker_kwargs_builder,
    )


def _get_pending_batch_ids(parent_job_id: str, db: CatalogDatabase) -> List[str]:
    """Get all PENDING batch IDs for a job."""
    assert db.session is not None
    result = db.session.execute(
        text(
            """
            SELECT id FROM job_batches
            WHERE parent_job_id = :parent_job_id
            AND status = 'PENDING'
            ORDER BY batch_number
        """
        ),
        {"parent_job_id": parent_job_id},
    )
    return [str(row[0]) for row in result.fetchall()]


def _update_job_status(
    job_id: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Update job status directly in the database."""
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


def recover_job(
    job_id: str,
    stale_minutes: int = 10,
) -> Dict[str, Any]:
    """
    Recover a stale/interrupted job.

    This function:
    1. Checks if the job has incomplete batches
    2. Resets stale RUNNING batches to PENDING
    3. Re-dispatches all incomplete batches with a chord to finalizer

    Args:
        job_id: The job ID to recover
        stale_minutes: Minutes after which RUNNING batches are considered stale

    Returns:
        Recovery result dict
    """
    logger.info(f"Attempting to recover job {job_id}")

    # Get job from database
    with get_db_context() as session:
        job = session.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"status": "error", "message": f"Job {job_id} not found"}

        if not job.catalog_id:
            return {"status": "error", "message": "Job has no catalog_id"}

        catalog_id = str(job.catalog_id)
        job_type = job.job_type
        job_params = job.parameters or {}

    # Check if job type is registered
    if job_type not in JOB_TYPE_REGISTRY:
        return {
            "status": "error",
            "message": f"Job type '{job_type}' not registered for recovery",
        }

    worker_task_name, finalizer_task_name, kwargs_builder = JOB_TYPE_REGISTRY[job_type]

    # Get batch manager
    batch_manager = BatchManager(catalog_id, job_id, job_type)

    with CatalogDatabase(catalog_id) as db:
        # Get batch progress
        progress = batch_manager.get_progress(db)

        if progress.total_batches == 0:
            return {
                "status": "error",
                "message": "Job has no batches (not a parallel job)",
            }

        if progress.is_complete:
            logger.info(f"Job {job_id} is already complete, triggering finalizer")
            # All batches done - just need to trigger finalizer
            finalizer_task = app.tasks.get(finalizer_task_name)
            if finalizer_task:
                finalizer_task.delay(
                    worker_results=[],
                    catalog_id=catalog_id,
                    parent_job_id=job_id,
                )
            return {
                "status": "finalizer_triggered",
                "message": "All batches complete, triggered finalizer",
                "completed_batches": progress.completed_batches,
                "failed_batches": progress.failed_batches,
            }

        # Find batches that need recovery
        stale_batch_ids = batch_manager.get_stale_batches(stale_minutes, db)
        pending_batch_ids = _get_pending_batch_ids(job_id, db)

        batches_to_recover = list(set(stale_batch_ids + pending_batch_ids))

        if not batches_to_recover:
            if progress.running_batches > 0:
                return {
                    "status": "batches_still_running",
                    "running_batches": progress.running_batches,
                    "message": f"{progress.running_batches} batches still running",
                }
            return {
                "status": "no_recovery_needed",
                "stale_batches": 0,
            }

        logger.info(
            f"Found {len(batches_to_recover)} batches to recover "
            f"({len(stale_batch_ids)} stale, {len(pending_batch_ids)} pending)"
        )

        # Reset stale batches to PENDING
        for batch_id in stale_batch_ids:
            batch_manager.reset_batch(batch_id, db)

    # Update job status to PROGRESS
    _update_job_status(
        job_id,
        "PROGRESS",
        result={
            "status": "recovering",
            "message": f"Recovering {len(batches_to_recover)} batches",
        },
    )

    # Get worker and finalizer tasks
    worker_task = app.tasks.get(worker_task_name)
    finalizer_task = app.tasks.get(finalizer_task_name)

    if not worker_task or not finalizer_task:
        return {
            "status": "error",
            "message": f"Tasks not found: worker={worker_task_name}, finalizer={finalizer_task_name}",
        }

    # Build worker task signatures
    worker_signatures = []
    for batch_id in batches_to_recover:
        # Base kwargs
        kwargs = {
            "catalog_id": catalog_id,
            "batch_id": batch_id,
            "parent_job_id": job_id,
        }
        # Add job-type-specific kwargs if builder provided
        if kwargs_builder:
            extra_kwargs = kwargs_builder(job_params, batch_id)
            kwargs.update(extra_kwargs)

        worker_signatures.append(worker_task.s(**kwargs))

    # Create chord: workers → finalizer
    worker_group = group(worker_signatures)
    finalizer_sig = finalizer_task.s(
        catalog_id=catalog_id,
        parent_job_id=job_id,
    )

    chord(worker_group)(finalizer_sig)

    logger.info(
        f"Recovery chord dispatched: {len(batches_to_recover)} workers → finalizer"
    )

    return {
        "status": "recovery_initiated",
        "job_id": job_id,
        "job_type": job_type,
        "stale_batches": len(stale_batch_ids),
        "pending_batches": len(pending_batch_ids),
        "total_recovered": len(batches_to_recover),
    }


@app.task(name="job_recovery")
def job_recovery_task(
    job_id: str,
    stale_minutes: int = 10,
) -> Dict[str, Any]:
    """
    Celery task wrapper for recover_job.

    This can be called via the API or directly:
        job_recovery_task.delay(job_id="...")
    """
    return recover_job(job_id, stale_minutes)


@app.task(name="job_recovery_check")
def job_recovery_check_task(
    stale_minutes: int = 10,
) -> Dict[str, Any]:
    """
    Periodic task to check for and recover all stale jobs.

    This task can be scheduled to run periodically (e.g., every 5 minutes)
    to automatically recover jobs that have stalled due to worker failures.

    Args:
        stale_minutes: Minutes after which jobs are considered stale

    Returns:
        Summary of recovery actions taken
    """
    logger.info("Running periodic job recovery check")

    recovered_jobs = []
    still_running = []
    errors = []
    jobs_checked = 0

    try:
        with get_db_context() as session:
            # Find jobs that might need recovery
            # - Status is STARTED/PROGRESS but haven't been updated recently
            stale_cutoff = datetime.utcnow() - timedelta(minutes=stale_minutes)

            # Find jobs in STARTED or PROGRESS state that haven't been updated
            stale_jobs = (
                session.query(Job)
                .filter(Job.status.in_(["STARTED", "PROGRESS"]))
                .filter(Job.updated_at < stale_cutoff)
                .all()
            )

            jobs_checked = len(stale_jobs)

            for job in stale_jobs:
                if not job.catalog_id:
                    continue

                # Check if this job type is recoverable
                if job.job_type not in JOB_TYPE_REGISTRY:
                    continue

                try:
                    # Check if this job has incomplete batches
                    batch_manager = BatchManager(
                        str(job.catalog_id), job.id, job.job_type
                    )

                    with CatalogDatabase(str(job.catalog_id)) as db:
                        progress = batch_manager.get_progress(db)

                    if progress.total_batches == 0:
                        # Not a parallel job
                        continue

                    if progress.is_complete:
                        # All batches done - just need to trigger finalizer
                        logger.info(f"Job {job.id} complete, triggering finalizer")
                        result = recover_job(job.id, stale_minutes)
                        recovered_jobs.append(
                            {
                                "job_id": job.id,
                                "action": "finalizer_triggered",
                                "result": result,
                            }
                        )
                    elif progress.running_batches > 0 or progress.pending_batches > 0:
                        # Has incomplete batches - trigger recovery
                        logger.info(f"Job {job.id} has stale batches, recovering")
                        # Run recovery asynchronously
                        job_recovery_task.delay(job.id, stale_minutes)
                        recovered_jobs.append(
                            {"job_id": job.id, "action": "recovery_triggered"}
                        )
                    else:
                        still_running.append(job.id)

                except Exception as e:
                    logger.warning(f"Failed to check job {job.id}: {e}")
                    errors.append({"job_id": job.id, "error": str(e)})

    except Exception as e:
        logger.error(f"Job recovery check failed: {e}")
        return {"status": "error", "error": str(e)}

    return {
        "status": "completed",
        "jobs_checked": jobs_checked,
        "recovered": recovered_jobs,
        "still_running": still_running,
        "errors": errors,
    }


# ============================================================================
# Job Type Registrations
# ============================================================================


def _scan_worker_kwargs(job_params: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    """Build extra kwargs for scan worker task."""
    return {
        "generate_previews": job_params.get("generate_previews", True),
    }


def _tagging_worker_kwargs(job_params: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    """Build extra kwargs for tagging worker task."""
    return {
        "backend": job_params.get("backend", "openclip"),
        "model": job_params.get("model"),
        "threshold": job_params.get("threshold", 0.25),
        "max_tags": job_params.get("max_tags", 10),
    }


def _duplicates_worker_kwargs(
    job_params: Dict[str, Any], batch_id: str
) -> Dict[str, Any]:
    """Build extra kwargs for duplicates worker task."""
    return {
        "similarity_threshold": job_params.get("similarity_threshold", 5),
    }


def _thumbnail_worker_kwargs(
    job_params: Dict[str, Any], batch_id: str
) -> Dict[str, Any]:
    """Build extra kwargs for thumbnail worker task."""
    return {
        "sizes": job_params.get("sizes"),
        "quality": job_params.get("quality", 85),
    }


def _burst_worker_kwargs(job_params: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
    """Build extra kwargs for burst worker task."""
    return {
        "gap_threshold": job_params.get("gap_threshold", 2.0),
        "min_burst_size": job_params.get("min_burst_size", 3),
    }


# Register all parallel job types
register_job_type(
    "scan",
    "scan_worker",
    "scan_finalizer",
    _scan_worker_kwargs,
)

register_job_type(
    "auto_tag",
    "tagging_worker",
    "tagging_finalizer",
    _tagging_worker_kwargs,
)

register_job_type(
    "detect_duplicates",
    "duplicates_worker",
    "duplicates_finalizer",
    _duplicates_worker_kwargs,
)

register_job_type(
    "generate_thumbnails",
    "thumbnail_worker",
    "thumbnail_finalizer",
    _thumbnail_worker_kwargs,
)

register_job_type(
    "detect_bursts",
    "burst_worker",
    "burst_finalizer",
    _burst_worker_kwargs,
)
