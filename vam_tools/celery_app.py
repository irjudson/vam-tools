"""Celery application for background task processing."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from celery import Celery
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
    task_revoked,
    task_success,
)

from .db.config import settings

logger = logging.getLogger(__name__)

# Create Celery app
app = Celery(
    "vam_tools",
    broker=settings.redis_url,
    backend=settings.redis_url,  # Store results in Redis
)

# Import tasks to register them
from .jobs import tasks  # noqa: E402, F401

# Celery configuration
app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task execution settings
    task_track_started=True,
    task_time_limit=3600 * 48,  # 48 hours max per task (for large catalogs)
    task_soft_time_limit=3600 * 47,  # 47 hours soft limit
    # Retry settings for transient failures
    task_acks_late=True,  # Acknowledge after task completes (enables retry on worker crash)
    task_reject_on_worker_lost=True,  # Reject and requeue if worker crashes
    # Result backend settings
    result_expires=3600 * 24,  # Keep results for 24 hours
    result_extended=True,  # Store more task metadata
    # Worker settings
    worker_prefetch_multiplier=1,  # Take one task at a time
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks (prevent memory leaks)
)


def _update_job_status(
    task_id: str,
    status: str,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Update job status in the database.

    This runs in the worker process, so we need our own database session.
    """
    try:
        from .db.connection import SessionLocal
        from .db.models import Job

        session = SessionLocal()
        try:
            job = session.query(Job).filter(Job.id == task_id).first()
            if job:
                job.status = status  # type: ignore[assignment]
                job.updated_at = datetime.utcnow()  # type: ignore[assignment]
                if error:
                    job.error = error  # type: ignore[assignment]
                if result:
                    job.result = result  # type: ignore[assignment]
                session.commit()
                logger.debug(f"Updated job {task_id} status to {status}")
            else:
                logger.debug(f"Job {task_id} not found in database (may be a sub-task)")
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"Failed to update job {task_id} status: {e}")


@task_prerun.connect
def task_prerun_handler(
    task_id: str = None,
    task: object = None,
    **kwargs: object,
) -> None:
    """Update job status when task starts running."""
    if task_id:
        task_name = getattr(task, "name", "unknown")
        logger.info(f"Task {task_name} [{task_id}] starting")
        _update_job_status(task_id, "STARTED")


@task_postrun.connect
def task_postrun_handler(
    task_id: str = None,
    task: object = None,
    state: str = None,
    retval: object = None,
    **kwargs: object,
) -> None:
    """Update job status when task completes (success or failure)."""
    if task_id:
        task_name = getattr(task, "name", "unknown")
        logger.info(f"Task {task_name} [{task_id}] finished with state: {state}")

        # Convert retval to dict if possible for storage
        result = None
        if retval is not None and isinstance(retval, dict):
            result = retval

        if state == "SUCCESS":
            _update_job_status(task_id, "SUCCESS", result=result)
        elif state == "FAILURE":
            error_msg = str(retval) if retval else "Task failed"
            _update_job_status(task_id, "FAILURE", error=error_msg)


@task_success.connect
def task_success_handler(
    sender: Any = None,
    result: Any = None,
    **kwargs: Any,
) -> None:
    """Log and persist successful task completion."""
    if sender is not None:
        task_id: Optional[str] = kwargs.get("task_id") or getattr(
            sender.request, "id", None
        )
        task_name = getattr(sender, "name", "unknown")
        logger.info(f"Task {task_name} [{task_id}] completed successfully")

        # Store result in database
        if task_id:
            result_dict = result if isinstance(result, dict) else None
            _update_job_status(task_id, "SUCCESS", result=result_dict)


@task_failure.connect
def task_failure_handler(
    sender: Any = None,
    exception: BaseException | None = None,
    einfo: Any = None,
    **kwargs: Any,
) -> None:
    """Log and persist task failures to database."""
    if sender is not None:
        task_id: Optional[str] = kwargs.get("task_id") or getattr(
            sender.request, "id", None
        )
        task_name = getattr(sender, "name", "unknown")

        # Build detailed error message
        error_type = type(exception).__name__ if exception else "UnknownError"
        error_msg = str(exception) if exception else "Unknown error occurred"
        full_error = f"{error_type}: {error_msg}"

        logger.error(f"Task {task_name} [{task_id}] failed: {full_error}")

        # Store failure in database
        if task_id:
            _update_job_status(
                task_id,
                "FAILURE",
                error=full_error,
                result={"error_type": error_type, "error": error_msg},
            )


@task_revoked.connect
def task_revoked_handler(
    request: Any = None,
    terminated: bool = False,
    signum: Any = None,
    expired: bool = False,
    **kwargs: Any,
) -> None:
    """Handle task revocation (cancellation or timeout)."""
    task_id: Optional[str] = getattr(request, "id", None) if request else None
    task_name = getattr(request, "name", "unknown") if request else "unknown"

    if expired:
        reason = "Task expired (timeout)"
        status = "TIMEOUT"
    elif terminated:
        reason = f"Task terminated (signal: {signum})"
        status = "TERMINATED"
    else:
        reason = "Task revoked"
        status = "REVOKED"

    logger.warning(f"Task {task_name} [{task_id}] revoked: {reason}")

    if task_id:
        _update_job_status(task_id, status, error=reason)


@task_retry.connect
def task_retry_handler(
    sender: Any = None,
    reason: Any = None,
    **kwargs: Any,
) -> None:
    """Handle task retry."""
    task_id: Optional[str] = None
    if sender is not None:
        task_id = kwargs.get("task_id") or getattr(sender.request, "id", None)
    task_name = getattr(sender, "name", "unknown") if sender else "unknown"
    retry_reason = str(reason) if reason else "Unknown reason"

    logger.info(f"Task {task_name} [{task_id}] retrying: {retry_reason}")

    if task_id:
        _update_job_status(
            task_id,
            "RETRY",
            error=f"Retrying: {retry_reason}",
        )


if __name__ == "__main__":
    app.start()
