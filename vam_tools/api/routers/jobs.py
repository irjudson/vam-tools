"""Job management endpoints."""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from celery.result import AsyncResult
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...celery_app import app as celery_app
from ...db import get_db
from ...db.catalog_schema import delete_catalog_data
from ...db.models import Job
from ...db.schemas import JobListResponse
from ...jobs.job_recovery import job_recovery_check_task, job_recovery_task
from ...jobs.parallel_bursts import burst_coordinator_task
from ...jobs.parallel_duplicates import duplicates_coordinator_task
from ...jobs.parallel_jobs import generic_coordinator_task
from ...jobs.parallel_scan import scan_coordinator_task, scan_recovery_task
from ...jobs.tasks import organize_catalog_task

logger = logging.getLogger(__name__)

router = APIRouter()

# Map job types to their Celery tasks for restart functionality
# All job types use the parallel coordinator pattern automatically
JOB_TYPE_TO_TASK = {
    "scan": scan_coordinator_task,  # Now uses coordinator pattern
    "scan_parallel": scan_coordinator_task,  # Legacy alias
    "analyze": generic_coordinator_task,  # Now uses generic parallel coordinator
    "detect_duplicates": duplicates_coordinator_task,  # Now uses coordinator pattern
    "auto_tag": generic_coordinator_task,  # Now uses generic parallel coordinator
    "detect_bursts": burst_coordinator_task,  # Now uses coordinator pattern
    "generate_thumbnails": generic_coordinator_task,  # Now uses generic parallel coordinator
    "organize": organize_catalog_task,  # TODO: Parallelize in future
}

# Job types that use the generic parallel coordinator
# These need special handling to pass job_type and work_items_query or source_directories
GENERIC_PARALLEL_JOB_TYPES = {
    "analyze": {
        "job_type": "analyze",
        # analyze uses source_directories, not work_items_query
        # source_directories will be passed from the request
        "needs_source_directories": True,
    },
    "auto_tag": {
        "job_type": "auto_tag",
        "work_items_query": "SELECT id FROM images WHERE catalog_id = :catalog_id",
    },
    "generate_thumbnails": {
        "job_type": "thumbnails",
        "work_items_query": "SELECT id FROM images WHERE catalog_id = :catalog_id",
    },
}

# Thread pool for blocking Celery operations (small pool to limit concurrent blocking calls)
_celery_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="celery_ops")

# Timeout for Celery operations (seconds)
CELERY_TIMEOUT = 2.0


def _run_with_timeout(func, timeout: float = CELERY_TIMEOUT, default=None):
    """Run a blocking function with a timeout, returning default on failure."""
    try:
        future = _celery_executor.submit(func)
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        logger.warning(f"Celery operation timed out after {timeout}s")
        return default
    except Exception as e:
        logger.warning(f"Celery operation failed: {e}")
        return default


@router.get("/", response_model=List[JobListResponse])
def list_jobs(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    exclude_status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all jobs with pagination and optional status filtering.

    Args:
        limit: Maximum number of jobs to return (default: 100)
        offset: Number of jobs to skip (default: 0)
        status: Comma-separated list of statuses to include (e.g., "PENDING,PROGRESS")
        exclude_status: Comma-separated list of statuses to exclude (e.g., "SUCCESS,FAILURE,REVOKED")

    Examples:
        - Active jobs only: ?exclude_status=SUCCESS,FAILURE,REVOKED
        - Completed jobs only: ?status=SUCCESS,FAILURE
        - Failed jobs: ?status=FAILURE
    """
    query = db.query(Job)

    # Filter by status if provided
    if status:
        status_list = [s.strip().upper() for s in status.split(",")]
        query = query.filter(Job.status.in_(status_list))

    # Exclude statuses if provided
    if exclude_status:
        exclude_list = [s.strip().upper() for s in exclude_status.split(",")]
        query = query.filter(~Job.status.in_(exclude_list))

    jobs = query.order_by(Job.created_at.desc()).limit(limit).offset(offset).all()
    return jobs


class ScanJobRequest(BaseModel):
    """Request to start a scan job."""

    catalog_id: uuid.UUID
    directories: List[str]
    reset_catalog: bool = False  # Clear ALL data (images, tags, duplicates) first


class ParallelScanJobRequest(BaseModel):
    """Request to start a parallel scan job."""

    catalog_id: uuid.UUID
    directories: List[str]
    batch_size: int = 1000  # Files per worker batch
    force_rescan: bool = False  # Rescan existing files (keeps other data)
    reset_catalog: bool = False  # Clear ALL data (images, tags, duplicates) first
    generate_previews: bool = True


class AnalyzeJobRequest(BaseModel):
    """Request to start an analyze job."""

    catalog_id: uuid.UUID
    source_directories: List[str]
    detect_duplicates: bool = False
    force_reanalyze: bool = False


class GenericJobRequest(BaseModel):
    """Generic request to start a job by type."""

    job_type: str
    catalog_id: uuid.UUID


class JobResponse(BaseModel):
    """Job status response."""

    job_id: str
    status: str
    progress: Dict[str, Any] = {}
    result: Dict[str, Any] = {}


@router.post("/start", response_model=JobResponse, status_code=202)
def start_job(request: GenericJobRequest, db: Session = Depends(get_db)):
    """Start a job by type.

    Supported job types:
    - generate_thumbnails: Generate thumbnails for all images (parallel)
    - detect_duplicates: Detect duplicate images (parallel)
    - auto_tag: Auto-tag images using AI (parallel)
    - detect_bursts: Detect burst photo sequences (parallel)
    """
    job_type = request.job_type
    catalog_id = str(request.catalog_id)

    # Look up the task for this job type
    task_func = JOB_TYPE_TO_TASK.get(job_type)
    if not task_func:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown job type: {job_type}. Supported types: {list(JOB_TYPE_TO_TASK.keys())}",
        )

    logger.info(f"Starting {job_type} job for catalog {catalog_id}")

    # Check if this is a generic parallel job type
    parallel_config = GENERIC_PARALLEL_JOB_TYPES.get(job_type)

    if parallel_config:
        # Use generic parallel coordinator
        task = generic_coordinator_task.delay(
            catalog_id=catalog_id,
            job_type=parallel_config["job_type"],
            work_items_query=parallel_config["work_items_query"],
            batch_size=500,
        )
        parameters = {
            "catalog_id": catalog_id,
            "parallel": True,
            "job_type": parallel_config["job_type"],
        }
    else:
        # Submit Celery task directly
        task = task_func.delay(catalog_id=catalog_id)
        parameters = {"catalog_id": catalog_id}

    # Save job to database
    job = Job(
        id=task.id,
        catalog_id=request.catalog_id,
        job_type=job_type,
        status="PENDING",
        parameters=parameters,
    )
    db.add(job)
    db.commit()

    return JobResponse(
        job_id=task.id,
        status="pending",
        progress={"parallel": parallel_config is not None},
        result={},
    )


@router.post("/scan", response_model=JobResponse, status_code=202)
def start_scan(request: ScanJobRequest, db: Session = Depends(get_db)):
    """Start a directory scan job using the coordinator pattern.

    This automatically breaks the work into batches that can be processed
    by available workers. With 1 worker, batches run sequentially.
    With N workers, they swarm the batches for faster processing.

    Args:
        reset_catalog: If True, clears ALL catalog metadata in the database (image records,
                      tags, duplicates, jobs) before starting the scan. Use for a fresh start.
                      NOTE: This only removes database records - original files on disk are
                      NEVER touched or modified.
    """
    logger.info(
        f"Starting scan for catalog {request.catalog_id} "
        f"with directories: {request.directories}"
        f" (reset_catalog={request.reset_catalog})"
    )

    # Reset catalog data if requested
    # This clears DATABASE RECORDS ONLY (image metadata, tags, duplicates, jobs, config)
    # Original files on disk are NEVER touched or modified
    if request.reset_catalog:
        logger.info(
            f"Resetting catalog {request.catalog_id} - clearing database records"
        )
        try:
            delete_catalog_data(str(request.catalog_id))
            logger.info(f"Catalog {request.catalog_id} reset complete")
        except Exception as e:
            logger.error(f"Failed to reset catalog {request.catalog_id}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to reset catalog: {str(e)}"
            )

    # Use coordinator pattern - works with 1 or N workers
    task = scan_coordinator_task.delay(
        catalog_id=str(request.catalog_id),
        source_directories=request.directories,
        force_rescan=False,
        generate_previews=True,
        batch_size=5000,  # Default batch size
    )

    # Save job to database
    job = Job(
        id=task.id,
        catalog_id=request.catalog_id,
        job_type="scan",
        status="PENDING",
        parameters={
            "directories": request.directories,
            "reset_catalog": request.reset_catalog,
            "force_rescan": False,
            "generate_previews": True,
            "parallel": True,
            "batch_size": 5000,
        },
    )
    db.add(job)
    db.commit()

    return JobResponse(
        job_id=task.id,
        status="pending",
        progress={
            "parallel": True,
            "batch_size": 5000,
            "reset_catalog": request.reset_catalog,
        },
        result={},
    )


@router.post("/scan/parallel", response_model=JobResponse, status_code=202)
def start_parallel_scan(request: ParallelScanJobRequest, db: Session = Depends(get_db)):
    """Start a parallel directory scan job that distributes work across multiple workers.

    This uses the Coordinator pattern:
    1. Coordinator discovers all files and creates batches in the database
    2. Worker tasks are spawned to process batches in parallel
    3. Finalizer aggregates results when all workers complete

    The batch_size parameter controls how many files each worker processes.
    With 6 workers and batch_size=1000, a 100k file scan would create 100 batches,
    with up to 6 batches processing simultaneously.

    Progress is tracked per-batch and aggregated for the parent job.
    Failed batches can be retried independently via /scan/recover/{job_id}.

    Args:
        force_rescan: If True, rescans files even if already in database (keeps tags, duplicates).
        reset_catalog: If True, clears ALL catalog metadata in the database (image records,
                      tags, duplicates, jobs) before starting the scan. Use for a fresh start.
                      NOTE: This only removes database records - original files on disk are
                      NEVER touched or modified.
    """
    logger.info(
        f"Starting parallel scan for catalog {request.catalog_id} "
        f"(batch_size={request.batch_size}, reset_catalog={request.reset_catalog})"
    )

    # Reset catalog data if requested
    # This clears DATABASE RECORDS ONLY (image metadata, tags, duplicates, jobs, config)
    # Original files on disk are NEVER touched or modified
    if request.reset_catalog:
        logger.info(
            f"Resetting catalog {request.catalog_id} - clearing database records"
        )
        try:
            delete_catalog_data(str(request.catalog_id))
            logger.info(f"Catalog {request.catalog_id} reset complete")
        except Exception as e:
            logger.error(f"Failed to reset catalog {request.catalog_id}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to reset catalog: {str(e)}"
            )

    # Submit coordinator task - it will spawn workers
    task = scan_coordinator_task.delay(
        catalog_id=str(request.catalog_id),
        source_directories=request.directories,
        force_rescan=request.force_rescan,
        generate_previews=request.generate_previews,
        batch_size=request.batch_size,
    )

    # Save job to database
    job = Job(
        id=task.id,
        catalog_id=request.catalog_id,
        job_type="scan",
        status="PENDING",
        parameters={
            "directories": request.directories,
            "reset_catalog": request.reset_catalog,
            "force_rescan": request.force_rescan,
            "generate_previews": request.generate_previews,
            "batch_size": request.batch_size,
            "parallel": True,
        },
    )
    db.add(job)
    db.commit()

    return JobResponse(
        job_id=task.id,
        status="pending",
        progress={
            "parallel": True,
            "batch_size": request.batch_size,
            "reset_catalog": request.reset_catalog,
        },
        result={},
    )


@router.post("/scan/recover/{job_id}", response_model=JobResponse)
def recover_parallel_scan(
    job_id: str,
    stale_minutes: int = 30,
    db: Session = Depends(get_db),
):
    """Recover a parallel scan job by retrying stale/stuck batches.

    This is useful when workers die mid-processing and batches are left
    in RUNNING state without completing.

    Args:
        job_id: The coordinator job ID to recover
        stale_minutes: Minutes after which RUNNING batches are considered stale
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.parameters or not job.parameters.get("parallel"):
        raise HTTPException(
            status_code=400,
            detail="Job is not a parallel scan job",
        )

    # Submit recovery task
    task = scan_recovery_task.delay(
        catalog_id=str(job.catalog_id),
        parent_job_id=job_id,
        stale_minutes=stale_minutes,
    )

    return JobResponse(
        job_id=task.id,
        status="pending",
        progress={"recovering_job": job_id},
        result={},
    )


@router.post("/{job_id}/recover", response_model=JobResponse)
def recover_job(
    job_id: str,
    stale_minutes: int = 10,
    db: Session = Depends(get_db),
):
    """Recover any stale/interrupted parallel job.

    This is a generic recovery endpoint that works for all parallel job types:
    - scan
    - auto_tag
    - detect_duplicates
    - generate_thumbnails
    - detect_bursts

    When workers die (container restart, crash, etc.), batches can be left
    in RUNNING or PENDING state. This endpoint detects such batches and
    re-dispatches them with a proper finalizer.

    Args:
        job_id: The job ID to recover
        stale_minutes: Minutes after which RUNNING batches are considered stale
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Submit recovery task
    task = job_recovery_task.delay(
        job_id=job_id,
        stale_minutes=stale_minutes,
    )

    return JobResponse(
        job_id=task.id,
        status="pending",
        progress={"recovering_job": job_id, "stale_minutes": stale_minutes},
        result={},
    )


@router.post("/recover-all")
def recover_all_stale_jobs(
    stale_minutes: int = 10,
):
    """Check for and recover all stale parallel jobs.

    This endpoint scans for jobs that:
    - Are in PROGRESS state
    - Haven't been updated recently (based on stale_minutes)
    - Have incomplete batches

    For each stale job found, it triggers recovery to re-dispatch
    incomplete batches with proper finalizers.

    Args:
        stale_minutes: Minutes after which jobs are considered stale
    """
    task = job_recovery_check_task.delay(stale_minutes=stale_minutes)

    return {
        "status": "recovery_check_started",
        "task_id": task.id,
        "stale_minutes": stale_minutes,
    }


@router.post("/analyze", response_model=JobResponse, status_code=202)
def start_analyze(request: AnalyzeJobRequest, db: Session = Depends(get_db)):
    """Start a parallel analyze job (scan directories for a catalog).

    This uses the generic parallel coordinator to distribute file processing
    across multiple workers for faster catalog building.
    """
    logger.info(
        f"Starting parallel analyze for catalog {request.catalog_id} "
        f"with directories: {request.source_directories}"
    )

    # Submit Celery task using generic parallel coordinator
    task = generic_coordinator_task.delay(
        catalog_id=str(request.catalog_id),
        job_type="analyze",
        source_directories=request.source_directories,
        batch_size=1000,
        processor_kwargs={
            "force_reanalyze": request.force_reanalyze,
        },
    )

    # Save job to database
    job = Job(
        id=task.id,
        catalog_id=request.catalog_id,
        job_type="analyze",
        status="PENDING",
        parameters={
            "source_directories": request.source_directories,
            "detect_duplicates": request.detect_duplicates,
            "force_reanalyze": request.force_reanalyze,
            "parallel": True,
            "batch_size": 1000,
        },
    )
    db.add(job)
    db.commit()

    return JobResponse(
        job_id=task.id,
        status="pending",
        progress={"parallel": True, "batch_size": 1000},
        result={},
    )


def _get_worker_stats():
    """Get worker stats - blocking call to be run in thread pool."""
    inspect = celery_app.control.inspect(timeout=1.0)
    return inspect.stats()


def _get_active_tasks():
    """Get active tasks - blocking call to be run in thread pool."""
    inspect = celery_app.control.inspect(timeout=1.0)
    return inspect.active()


@router.get("/health")
def get_worker_health():
    """Get Celery worker health status (non-blocking with timeout)."""
    # Run stats check with timeout
    stats = _run_with_timeout(_get_worker_stats, timeout=CELERY_TIMEOUT, default=None)

    if stats is None:
        return {
            "status": "unknown",
            "workers": 0,
            "active_tasks": 0,
            "message": "Worker status check timed out",
        }

    if stats:
        worker_count = len(stats)
        # Get active tasks with timeout (but don't fail if this times out)
        active = _run_with_timeout(
            _get_active_tasks, timeout=CELERY_TIMEOUT, default={}
        )
        active_count = sum(len(tasks) for tasks in (active or {}).values())
        return {
            "status": "healthy",
            "workers": worker_count,
            "active_tasks": active_count,
            "message": f"{worker_count} worker(s) online",
        }
    else:
        return {
            "status": "unhealthy",
            "workers": 0,
            "active_tasks": 0,
            "message": "No workers available",
        }


def _safe_get_task_state(task: AsyncResult) -> str:
    """Safely get task state, handling malformed exception info."""
    try:
        return task.state
    except ValueError as e:
        # Celery raises ValueError when exception info is malformed
        # (e.g., missing 'exc_type' key in result backend)
        logger.warning(f"Error getting task state for {task.id}: {e}")
        return "FAILURE"


def _safe_get_task_info(task: AsyncResult) -> Any:
    """Safely get task info, handling malformed data."""
    try:
        return task.info
    except Exception as e:
        logger.warning(f"Error getting task info for {task.id}: {e}")
        return {"error": f"Failed to retrieve task info: {e}"}


def _get_celery_task_state(job_id: str) -> Optional[str]:
    """Get Celery task state with timeout - returns None on failure/timeout."""

    def _fetch_state():
        task = AsyncResult(job_id, app=celery_app)
        return _safe_get_task_state(task)

    return _run_with_timeout(_fetch_state, timeout=CELERY_TIMEOUT, default=None)


def _get_celery_task_info(job_id: str) -> Optional[Dict[str, Any]]:
    """Get Celery task info with timeout - returns None on failure/timeout."""

    def _fetch_info():
        task = AsyncResult(job_id, app=celery_app)
        return _safe_get_task_info(task)

    return _run_with_timeout(_fetch_info, timeout=CELERY_TIMEOUT, default=None)


def _get_celery_task_result(job_id: str) -> Optional[Dict[str, Any]]:
    """Get Celery task result with timeout - returns None on failure/timeout."""

    def _fetch_result():
        task = AsyncResult(job_id, app=celery_app)
        return task.result

    return _run_with_timeout(_fetch_result, timeout=CELERY_TIMEOUT, default=None)


def _get_batch_progress(job_id: str) -> Optional[Dict[str, Any]]:
    """Get progress from job_batches table for coordinator-pattern jobs.

    This is the source of truth for parallel jobs since:
    - Redis progress may expire (1 hour TTL)
    - Celery coordinator returns SUCCESS after dispatching, not actual completion
    - Database job status may show FAILURE from a previous crash
    """
    from sqlalchemy import text

    from ...db import get_db_context

    try:
        with get_db_context() as session:
            # Get batch counts by status with aggregated progress
            result = session.execute(
                text(
                    """
                    SELECT
                        status,
                        COUNT(*) as count,
                        COALESCE(SUM(success_count), 0) as total_success,
                        COALESCE(SUM(processed_count), 0) as total_processed,
                        COALESCE(SUM(items_count), 0) as total_items
                    FROM job_batches
                    WHERE parent_job_id = :job_id
                    GROUP BY status
                """
                ),
                {"job_id": job_id},
            )
            rows = result.fetchall()

            if not rows:
                return None

            # Aggregate batch stats
            total_batches = 0
            completed_batches = 0
            running_batches = 0
            pending_batches = 0
            failed_batches = 0
            cancelled_batches = 0
            total_processed = 0
            total_success = 0
            total_items = 0

            for row in rows:
                status, count, success, processed, items = row
                total_batches += count
                total_processed += processed or 0
                total_success += success or 0
                total_items += items or 0
                if status == "COMPLETED":
                    completed_batches = count
                elif status == "RUNNING":
                    running_batches = count
                elif status == "PENDING":
                    pending_batches = count
                elif status == "FAILED":
                    failed_batches = count
                elif status == "CANCELLED":
                    cancelled_batches = count

            # Calculate percent complete
            percent = 0
            if total_items > 0:
                percent = int((total_processed / total_items) * 100)
            elif total_batches > 0:
                percent = int((completed_batches / total_batches) * 100)

            # Determine job status from batches
            if cancelled_batches > 0:
                # Job was revoked/cancelled
                batch_status = "REVOKED"
            elif completed_batches == total_batches and total_batches > 0:
                batch_status = "SUCCESS"
            elif failed_batches > 0 and running_batches == 0 and pending_batches == 0:
                batch_status = "FAILURE"
            elif running_batches > 0 or pending_batches > 0:
                batch_status = "PROGRESS"
            else:
                batch_status = "UNKNOWN"

            return {
                "status": batch_status,
                "progress": {
                    "current": total_processed,
                    "total": total_items,
                    "percent": percent,
                    "message": f"Processing: {completed_batches}/{total_batches} batches complete ({percent}%)",
                    "batches_total": total_batches,
                    "batches_completed": completed_batches,
                    "batches_running": running_batches,
                    "batches_pending": pending_batches,
                    "batches_failed": failed_batches,
                    "batches_cancelled": cancelled_batches,
                    "success_count": total_success,
                },
            }
    except Exception as e:
        logger.warning(f"Failed to get batch progress for job {job_id}: {e}")
        return None


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get job status and progress.

    This endpoint now prefers batch progress over database state for parallel jobs.
    The job_batches table is the source of truth for coordinator-pattern jobs.
    """
    # First, check database for job (fast, non-blocking)
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # For parallel jobs, ALWAYS check batch progress first (source of truth)
    # This handles cases where:
    # - Coordinator returned SUCCESS after dispatching but workers still running
    # - Job marked FAILURE from a previous crash but workers may have completed
    # - Redis progress expired but batch table has current state
    is_parallel_job = job.parameters and job.parameters.get("parallel", False)

    if is_parallel_job:
        batch_progress = _get_batch_progress(job_id)
        if batch_progress:
            batch_status = batch_progress["status"]
            progress_data = batch_progress["progress"]

            # If batches show work is still ongoing, report PROGRESS
            if batch_status == "PROGRESS":
                return JobResponse(
                    job_id=job_id,
                    status="PROGRESS",
                    progress=progress_data,
                    result={},
                )
            # If all batches completed successfully
            elif batch_status == "SUCCESS":
                return JobResponse(
                    job_id=job_id,
                    status="SUCCESS",
                    progress=progress_data,
                    result=job.result or {"status": "completed"},
                )
            # If batches show failure (all done, some failed)
            elif batch_status == "FAILURE":
                return JobResponse(
                    job_id=job_id,
                    status="FAILURE",
                    progress=progress_data,
                    result=job.result or {"error": job.error} if job.error else {},
                )
            # If job was revoked/cancelled
            elif batch_status == "REVOKED":
                return JobResponse(
                    job_id=job_id,
                    status="REVOKED",
                    progress=progress_data,
                    result=job.result or {"status": "cancelled"},
                )
            # Unknown state - fall through to database state

    # For non-parallel jobs or when batch progress unavailable,
    # use database/Celery state
    if job.status in ("SUCCESS", "FAILURE", "REVOKED"):
        return JobResponse(
            job_id=job_id,
            status=job.status,
            progress={},
            result=job.result or ({"error": job.error} if job.error else {}),
        )

    # For active jobs (PENDING, PROGRESS), try to get live status from Celery with timeout
    celery_state = _get_celery_task_state(job_id)

    # If Celery is unavailable or timed out, return database state
    if celery_state is None:
        logger.debug(f"Celery unavailable for job {job_id}, using database state")
        return JobResponse(
            job_id=job_id,
            status=job.status,
            progress={},
            result=job.result or {},
        )

    # Build response from Celery state
    response = JobResponse(
        job_id=job_id,
        status=celery_state,
        progress={},
        result={},
    )

    # Get progress/result based on Celery state
    if celery_state == "PROGRESS":
        task_info = _get_celery_task_info(job_id)
        response.progress = task_info or {}
        # Update job status in database
        if job.status != "PROGRESS":
            job.status = "PROGRESS"
            db.commit()
    elif celery_state == "SUCCESS":
        task_result = _get_celery_task_result(job_id)
        response.result = task_result or {}

        # For parallel scan jobs, the coordinator returns "dispatched" status
        # but workers are still processing. Don't mark as SUCCESS until finalizer completes.
        is_parallel_job = job.parameters and job.parameters.get("parallel", False)
        is_dispatched = (
            isinstance(task_result, dict) and task_result.get("status") == "dispatched"
        )

        if is_parallel_job and is_dispatched:
            # Coordinator finished dispatching, but workers are still running
            # Keep job in PROGRESS state (set by coordinator's _update_job_status)
            response.status = "PROGRESS"
            response.progress = {
                "phase": "processing",
                "message": task_result.get("message", "Processing..."),
                "total_files": task_result.get("total_files", 0),
                "num_batches": task_result.get("num_batches", 0),
            }
            # Ensure database reflects PROGRESS
            if job.status != "PROGRESS":
                job.status = "PROGRESS"
                db.commit()
        elif job.status != "SUCCESS":
            # Normal job completion
            job.status = "SUCCESS"
            job.result = response.result
            db.commit()
            logger.info(f"Saved job {job_id} result to database")
    elif celery_state == "FAILURE":
        task_info = _get_celery_task_info(job_id)
        response.result = {"error": str(task_info)}

        # Save error to database
        if job.status != "FAILURE":
            job.status = "FAILURE"
            job.error = str(task_info)
            # Try to get statistics from failure info if available
            if isinstance(task_info, dict) and "statistics" in task_info:
                job.result = task_info.get("statistics")
            db.commit()

    return response


@router.get("/{job_id}/progress")
def get_job_progress(job_id: str, db: Session = Depends(get_db)):
    """Get job progress from Redis (fast, never blocks).

    This endpoint is designed for frontend polling (1-2s intervals).
    It reads directly from Redis without any blocking Celery calls,
    so it will never hang even if Celery/workers are unresponsive.

    The response includes:
    - status: Current job state
    - progress: Current/total/percent/message
    - timestamp: When the progress was last updated

    For terminal states (SUCCESS/FAILURE), falls back to database.
    """
    # First check database for the job
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # For terminal states, return from database (definitive source)
    if job.status in ("SUCCESS", "FAILURE", "REVOKED"):
        return {
            "job_id": job_id,
            "status": job.status,
            "progress": {},
            "result": job.result or ({"error": job.error} if job.error else {}),
            "source": "database",
        }

    # Try to get progress from Redis (fast, non-blocking)
    try:
        from ...jobs.progress_publisher import get_last_progress

        redis_progress = get_last_progress(job_id)

        if redis_progress:
            return {
                **redis_progress,
                "source": "redis",
            }
    except Exception as e:
        logger.warning(f"Failed to get Redis progress for job {job_id}: {e}")

    # Fall back to database state if Redis unavailable
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": {},
        "result": {},
        "source": "database",
        "message": "No real-time progress available",
    }


def _cancel_job_batches(job_id: str):
    """Cancel all batches for a job by marking them as CANCELLED."""
    from sqlalchemy import text

    from ...db import get_db_context

    try:
        with get_db_context() as session:
            result = session.execute(
                text(
                    """
                    UPDATE job_batches
                    SET status = 'CANCELLED'
                    WHERE parent_job_id = :job_id
                    AND status IN ('PENDING', 'RUNNING')
                """
                ),
                {"job_id": job_id},
            )
            cancelled_count = result.rowcount
            session.commit()
            if cancelled_count > 0:
                logger.info(f"Cancelled {cancelled_count} batches for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to cancel batches for job {job_id}: {e}")


def _delete_job_batches(job_id: str):
    """Delete all batches for a job from the database."""
    from sqlalchemy import text

    from ...db import get_db_context

    try:
        with get_db_context() as session:
            result = session.execute(
                text(
                    """
                    DELETE FROM job_batches
                    WHERE parent_job_id = :job_id
                """
                ),
                {"job_id": job_id},
            )
            deleted_count = result.rowcount
            session.commit()
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} batches for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to delete batches for job {job_id}: {e}")


def _revoke_celery_task(job_id: str, terminate: bool):
    """Revoke a Celery task - blocking call to be run in thread pool or background."""
    try:
        celery_app.control.revoke(
            job_id,
            terminate=terminate,
            signal="SIGKILL" if terminate else "SIGTERM",
        )
        # Also try to forget from result backend
        task = AsyncResult(job_id, app=celery_app)
        if task:
            task.forget()
        logger.info(f"Celery task {job_id} revoked (terminate={terminate})")
    except Exception as e:
        logger.warning(f"Failed to revoke Celery task {job_id}: {e}")


@router.delete("/{job_id}")
def revoke_job(
    job_id: str,
    terminate: bool = False,
    force: bool = False,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    request: Request = None,
):
    """Revoke/cancel a job, or force delete it from the database.

    This is now non-blocking - Celery revocation happens in background.

    Args:
        job_id: The job ID to revoke/delete
        terminate: If True, forcefully terminate the task (default: False)
        force: If True, completely delete the job from the database instead of marking as REVOKED
    """
    action = "force deleting" if force else "revoking"

    # Log detailed request info for debugging job revocation sources
    client_ip = "unknown"
    user_agent = "unknown"
    referer = "none"
    if request:
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        referer = request.headers.get("referer", "none")

    logger.warning(
        f"JOB REVOKE REQUEST: {action} job {job_id} | "
        f"terminate={terminate} force={force} | "
        f"client={client_ip} | user-agent={user_agent[:80]} | referer={referer}"
    )

    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Cancel all batches for this job first (prevents workers from picking them up)
        _cancel_job_batches(job_id)

        if force:
            # Force delete: completely remove the job and its batches from the database
            # Delete batches first (they reference the job)
            _delete_job_batches(job_id)

            # Then delete the job record
            db.delete(job)
            db.commit()
            logger.info(f"Job {job_id} deleted from database")

            # Schedule Celery revoke in background (fire-and-forget)
            if background_tasks:
                background_tasks.add_task(_revoke_celery_task, job_id, terminate)
            else:
                # Fallback: run with short timeout
                _run_with_timeout(
                    lambda: _revoke_celery_task(job_id, terminate),
                    timeout=1.0,
                    default=None,
                )

            return {"status": "deleted", "job_id": job_id}
        else:
            # Soft delete: mark as REVOKED in database immediately (non-blocking)
            job.status = "REVOKED"
            job.error = "Job was manually cancelled"
            db.commit()

            # Schedule Celery revoke in background (fire-and-forget)
            if background_tasks:
                background_tasks.add_task(_revoke_celery_task, job_id, terminate)
            else:
                # Fallback: run with short timeout
                _run_with_timeout(
                    lambda: _revoke_celery_task(job_id, terminate),
                    timeout=1.0,
                    default=None,
                )

            return {"status": "revoked", "job_id": job_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error {action} job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to {action} job: {str(e)}")


@router.websocket("/{job_id}/stream")
async def stream_job_updates(websocket: WebSocket, job_id: str):
    """Stream real-time job updates via WebSocket.

    This endpoint uses Redis pub/sub for real-time updates, falling back
    to Redis key polling. This is much faster and more reliable than
    polling Celery directly.

    The connection will automatically close when:
    - The job completes (SUCCESS or FAILURE)
    - The client disconnects
    - The connection times out (30 minutes max)
    - The server is shutting down

    The frontend should also implement a fallback to REST polling
    (/api/jobs/{job_id}/progress) in case WebSocket connection fails.
    """
    await websocket.accept()

    # Maximum connection duration (30 minutes) to prevent orphaned connections
    max_duration = 30 * 60  # seconds
    start_time = asyncio.get_event_loop().time()

    try:
        from ...jobs.progress_publisher import get_last_progress

        last_progress_str = None
        loop = asyncio.get_event_loop()

        while True:
            # Check for timeout to prevent indefinite connections
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_duration:
                logger.info(f"WebSocket timeout for job {job_id} after {elapsed:.0f}s")
                await websocket.send_json(
                    {
                        "job_id": job_id,
                        "status": "TIMEOUT",
                        "message": "Connection timed out. Reconnect to continue monitoring.",
                    }
                )
                break

            # Check if WebSocket is still connected
            if websocket.client_state.name != "CONNECTED":
                logger.info(f"WebSocket no longer connected for job {job_id}")
                break

            # Get progress from Redis (fast, non-blocking)
            try:
                progress = await loop.run_in_executor(
                    _celery_executor, lambda: get_last_progress(job_id)
                )
            except Exception as e:
                logger.warning(f"Redis error for job {job_id}: {e}")
                progress = None

            if progress:
                # Send update if progress changed
                progress_str = str(progress)
                if progress_str != last_progress_str:
                    try:
                        await websocket.send_json(progress)
                    except Exception as send_error:
                        logger.warning(f"Failed to send WebSocket update: {send_error}")
                        break
                    last_progress_str = progress_str

                # Check for terminal states
                status = progress.get("status", "")
                if status in ("SUCCESS", "FAILURE"):
                    logger.info(f"Job {job_id} completed with status {status}")
                    break
            else:
                # No Redis progress yet - check if job exists and is pending
                # This happens for new jobs before first progress update
                pass

            # Poll Redis every 500ms (much faster than Celery polling)
            try:
                await asyncio.wait_for(asyncio.sleep(0.5), timeout=1.0)
            except asyncio.TimeoutError:
                pass  # Expected, continue loop
            except asyncio.CancelledError:
                logger.info(f"WebSocket cancelled for job {job_id}")
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except asyncio.CancelledError:
        logger.info(f"WebSocket task cancelled for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass  # Connection already closed


@router.post("/{job_id}/restart", response_model=JobResponse, status_code=202)
def restart_job(job_id: str, db: Session = Depends(get_db)):
    """
    Restart a failed, timed out, or cancelled job.

    This creates a new job with the same parameters as the original.
    The original job is kept in history with its original status.

    Args:
        job_id: The ID of the job to restart

    Returns:
        The new job's response
    """
    logger.info(f"Attempting to restart job {job_id}")

    # Find the original job
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if job is in a restartable state
    restartable_states = ("FAILURE", "TIMEOUT", "TERMINATED", "REVOKED", "STALE")
    if job.status not in restartable_states:
        raise HTTPException(
            status_code=400,
            detail=f"Job cannot be restarted. Status is '{job.status}', "
            f"must be one of: {', '.join(restartable_states)}",
        )

    # Find the appropriate task for this job type
    task_func = JOB_TYPE_TO_TASK.get(job.job_type)
    if not task_func:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown job type: {job.job_type}. Cannot restart.",
        )

    # Get parameters from original job
    params = job.parameters or {}

    # Submit new Celery task with the original parameters
    # All job types now use the coordinator pattern which handles parallelization automatically
    try:
        if job.job_type == "scan" or job.job_type == "scan_parallel":
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                source_directories=params.get("directories", []),
                force_rescan=params.get("force_rescan", False),
                generate_previews=params.get("generate_previews", True),
                batch_size=params.get("batch_size", 5000),
            )
        elif job.job_type == "analyze":
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                source_directories=params.get("source_directories", []),
                detect_duplicates=params.get("detect_duplicates", False),
                force_reanalyze=params.get("force_reanalyze", False),
            )
        elif job.job_type == "detect_duplicates":
            # Now uses duplicates_coordinator_task
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                similarity_threshold=params.get("similarity_threshold", 5),
                recompute_hashes=params.get("recompute_hashes", False),
                batch_size=params.get("batch_size", 1000),
            )
        elif job.job_type == "auto_tag":
            # Now uses tagging_coordinator_task
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                backend=params.get("backend", "openclip"),
                model=params.get("model"),
                threshold=params.get("threshold", 0.25),
                max_tags=params.get("max_tags", 10),
                batch_size=params.get("batch_size", 500),
                tag_mode=params.get("tag_mode", "untagged_only"),
            )
        elif job.job_type == "detect_bursts":
            # Now uses burst_coordinator_task
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                gap_threshold=params.get("gap_threshold", 2.0),
                min_burst_size=params.get("min_burst_size", 3),
                batch_size=params.get("batch_size", 5000),
            )
        elif job.job_type == "generate_thumbnails":
            # Now uses thumbnail_coordinator_task
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                sizes=params.get("sizes"),
                quality=params.get("quality", 85),
                force=params.get("force", False),
                batch_size=params.get("batch_size", 500),
            )
        elif job.job_type == "organize":
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                destination_path=params.get("destination_path", ""),
                strategy=params.get("strategy", "date_based"),
                simulate=params.get("simulate", True),
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Job type '{job.job_type}' restart not implemented",
            )
    except Exception as e:
        logger.error(f"Failed to submit restart task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restart job: {e}")

    # Create new job record
    new_job = Job(
        id=new_task.id,
        catalog_id=job.catalog_id,
        job_type=job.job_type,
        status="PENDING",
        parameters=params,
    )
    db.add(new_job)

    # Update original job to note that it was restarted
    job.error = (job.error or "") + f"\n[Restarted as job {new_task.id}]"
    db.commit()

    logger.info(f"Job {job_id} restarted as {new_task.id}")

    return JobResponse(
        job_id=new_task.id,
        status="pending",
        progress={"restarted_from": job_id},
        result={},
    )


# Stale job timeout (jobs that haven't updated in this time are considered stale)
STALE_JOB_TIMEOUT_MINUTES = 30


@router.get("/stale", response_model=List[JobListResponse])
def get_stale_jobs(
    timeout_minutes: int = STALE_JOB_TIMEOUT_MINUTES, db: Session = Depends(get_db)
):
    """
    Get jobs that appear to be stale (stuck in PENDING/PROGRESS/STARTED state).

    A job is considered stale if:
    - Its status is PENDING, PROGRESS, or STARTED
    - It hasn't been updated in the specified timeout period

    Args:
        timeout_minutes: Minutes without update to consider a job stale (default: 30)

    Returns:
        List of stale jobs
    """
    cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)

    stale_jobs = (
        db.query(Job)
        .filter(Job.status.in_(["PENDING", "PROGRESS", "STARTED"]))
        .filter(Job.updated_at < cutoff_time)
        .order_by(Job.updated_at.desc())
        .all()
    )

    return stale_jobs


@router.post("/recover-stale")
def recover_stale_jobs(
    timeout_minutes: int = STALE_JOB_TIMEOUT_MINUTES,
    auto_restart: bool = False,
    db: Session = Depends(get_db),
):
    """
    Detect and optionally recover stale jobs.

    This endpoint:
    1. Finds jobs stuck in PENDING/PROGRESS/STARTED state
    2. Checks if Celery has any record of the task
    3. Marks orphaned jobs as STALE
    4. Optionally restarts them

    Args:
        timeout_minutes: Minutes without update to consider a job stale
        auto_restart: If True, automatically restart stale jobs

    Returns:
        Summary of stale jobs found and actions taken
    """
    cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)

    stale_jobs = (
        db.query(Job)
        .filter(Job.status.in_(["PENDING", "PROGRESS", "STARTED"]))
        .filter(Job.updated_at < cutoff_time)
        .all()
    )

    recovered = []
    marked_stale = []
    still_running = []

    for job in stale_jobs:
        # Check Celery for the task state
        celery_state = _get_celery_task_state(job.id)

        if celery_state in ("PENDING", "STARTED", "PROGRESS"):
            # Task is actually still running in Celery
            still_running.append(job.id)
        elif celery_state in ("SUCCESS", "FAILURE"):
            # Task finished but database wasn't updated - sync it
            if celery_state == "SUCCESS":
                result = _get_celery_task_result(job.id)
                job.status = "SUCCESS"
                job.result = result if isinstance(result, dict) else None
            else:
                info = _get_celery_task_info(job.id)
                job.status = "FAILURE"
                job.error = str(info)
            db.commit()
            recovered.append(job.id)
        else:
            # Task is gone from Celery - mark as stale
            job.status = "STALE"
            job.error = (
                f"Task disappeared from queue after {timeout_minutes} minutes. "
                f"Last Celery state: {celery_state or 'UNKNOWN'}"
            )
            db.commit()
            marked_stale.append(job.id)

            # Auto-restart if requested
            if auto_restart:
                try:
                    # Use internal restart logic
                    task_func = JOB_TYPE_TO_TASK.get(job.job_type)
                    if task_func:
                        params = job.parameters or {}
                        # Dispatch based on job type (simplified version)
                        new_task = task_func.delay(
                            catalog_id=str(job.catalog_id), **params
                        )
                        new_job = Job(
                            id=new_task.id,
                            catalog_id=job.catalog_id,
                            job_type=job.job_type,
                            status="PENDING",
                            parameters=params,
                        )
                        db.add(new_job)
                        job.error += f"\n[Auto-restarted as job {new_task.id}]"
                        db.commit()
                        recovered.append(f"{job.id} -> {new_task.id}")
                except Exception as e:
                    logger.warning(f"Failed to auto-restart job {job.id}: {e}")

    return {
        "stale_jobs_found": len(stale_jobs),
        "still_running": still_running,
        "synced_with_celery": [j for j in recovered if "->" not in str(j)],
        "marked_stale": marked_stale,
        "auto_restarted": (
            [j for j in recovered if "->" in str(j)] if auto_restart else []
        ),
        "message": f"Found {len(stale_jobs)} potentially stale jobs",
    }
