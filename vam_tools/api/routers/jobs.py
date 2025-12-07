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
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...celery_app import app as celery_app
from ...db import get_db
from ...db.models import Job
from ...db.schemas import JobListResponse
from ...jobs.tasks import (
    analyze_catalog_task,
    auto_tag_task,
    detect_bursts_task,
    detect_duplicates_task,
    generate_thumbnails_task,
    organize_catalog_task,
    scan_catalog_task,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Map job types to their Celery tasks for restart functionality
JOB_TYPE_TO_TASK = {
    "scan": scan_catalog_task,
    "analyze": analyze_catalog_task,
    "detect_duplicates": detect_duplicates_task,
    "auto_tag": auto_tag_task,
    "detect_bursts": detect_bursts_task,
    "generate_thumbnails": generate_thumbnails_task,
    "organize": organize_catalog_task,
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
def list_jobs(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    """List all jobs with pagination."""
    jobs = (
        db.query(Job).order_by(Job.created_at.desc()).limit(limit).offset(offset).all()
    )
    return jobs


class ScanJobRequest(BaseModel):
    """Request to start a scan job."""

    catalog_id: uuid.UUID
    directories: List[str]


class AnalyzeJobRequest(BaseModel):
    """Request to start an analyze job."""

    catalog_id: uuid.UUID
    source_directories: List[str]
    detect_duplicates: bool = False
    force_reanalyze: bool = False


class JobResponse(BaseModel):
    """Job status response."""

    job_id: str
    status: str
    progress: Dict[str, Any] = {}
    result: Dict[str, Any] = {}


@router.post("/scan", response_model=JobResponse, status_code=202)
def start_scan(request: ScanJobRequest, db: Session = Depends(get_db)):
    """Start a simple directory scan job (metadata extraction + thumbnails only)."""
    logger.info(f"Starting scan for catalog {request.catalog_id}")

    # Submit Celery task - simple scan (no duplicate detection)
    task = scan_catalog_task.delay(
        catalog_id=str(request.catalog_id),
        source_directories=request.directories,
        force_rescan=False,
        generate_previews=True,
    )

    # Save job to database
    job = Job(
        id=task.id,
        catalog_id=request.catalog_id,
        job_type="scan",
        status="PENDING",
        parameters={
            "directories": request.directories,
            "force_rescan": False,
            "generate_previews": True,
        },
    )
    db.add(job)
    db.commit()

    return JobResponse(
        job_id=task.id,
        status="pending",
        progress={},
        result={},
    )


@router.post("/analyze", response_model=JobResponse, status_code=202)
def start_analyze(request: AnalyzeJobRequest, db: Session = Depends(get_db)):
    """Start an analyze job (scan directories for a catalog)."""
    logger.info(f"Starting analyze for catalog {request.catalog_id}")

    # Submit Celery task
    task = analyze_catalog_task.delay(
        catalog_id=str(request.catalog_id),
        source_directories=request.source_directories,
        detect_duplicates=request.detect_duplicates,
        force_reanalyze=request.force_reanalyze,
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
        },
    )
    db.add(job)
    db.commit()

    return JobResponse(
        job_id=task.id,
        status="pending",
        progress={},
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


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get job status and progress.

    This endpoint now prefers database state over Celery to avoid blocking.
    Celery is only queried (with timeout) for active jobs that may have progress updates.
    """
    # First, check database for job (fast, non-blocking)
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # For terminal states, return database data directly (no Celery query needed)
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

        # Save result to database for history
        if job.status != "SUCCESS":
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
):
    """Revoke/cancel a job, or force delete it from the database.

    This is now non-blocking - Celery revocation happens in background.

    Args:
        job_id: The job ID to revoke/delete
        terminate: If True, forcefully terminate the task (default: False)
        force: If True, completely delete the job from the database instead of marking as REVOKED
    """
    action = "force deleting" if force else "revoking"
    logger.info(f"{action.capitalize()} job {job_id} (terminate={terminate})")

    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if force:
            # Force delete: completely remove the job record from the database
            # DELETE the job record first (non-blocking)
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

    This endpoint streams job progress updates in real-time. The connection
    will automatically close when:
    - The job completes (SUCCESS or FAILURE)
    - The client disconnects
    - The connection times out (30 minutes max)
    - The server is shutting down

    NOTE: All Celery operations are run in a thread pool to avoid blocking
    the async event loop, which can cause the app and workers to stall.
    """
    await websocket.accept()

    # Maximum connection duration (30 minutes) to prevent orphaned connections
    max_duration = 30 * 60  # seconds
    start_time = asyncio.get_event_loop().time()

    try:
        last_state = None
        last_progress = None
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

            # Get task state using non-blocking thread pool (prevents async loop stall)
            current_state = await loop.run_in_executor(
                _celery_executor, lambda: _get_celery_task_state(job_id)
            )

            # If Celery is unavailable, report unknown state but keep connection open
            if current_state is None:
                current_state = "UNKNOWN"

            # Build update message
            update = {
                "job_id": job_id,
                "status": current_state,
                "progress": {},
                "result": {},
            }

            # Get progress/result based on state (all non-blocking)
            if current_state == "PROGRESS":
                task_info = await loop.run_in_executor(
                    _celery_executor, lambda: _get_celery_task_info(job_id)
                )
                update["progress"] = task_info or {}
            elif current_state == "SUCCESS":
                task_result = await loop.run_in_executor(
                    _celery_executor, lambda: _get_celery_task_result(job_id)
                )
                update["result"] = task_result or {}
                await websocket.send_json(update)
                break  # Job done, close connection
            elif current_state == "FAILURE":
                task_info = await loop.run_in_executor(
                    _celery_executor, lambda: _get_celery_task_info(job_id)
                )
                update["result"] = {"error": str(task_info)}
                await websocket.send_json(update)
                break  # Job failed, close connection

            # Send update if state changed OR progress data changed
            current_progress = str(update.get("progress", {}))
            if current_state != last_state or current_progress != last_progress:
                try:
                    await websocket.send_json(update)
                except Exception as send_error:
                    logger.warning(f"Failed to send WebSocket update: {send_error}")
                    break
                last_state = current_state
                last_progress = current_progress

            # Poll every 500ms using wait_for to allow cancellation
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
    try:
        if job.job_type == "scan":
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                source_directories=params.get("directories", []),
                force_rescan=params.get("force_rescan", False),
                generate_previews=params.get("generate_previews", True),
            )
        elif job.job_type == "analyze":
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                source_directories=params.get("source_directories", []),
                detect_duplicates=params.get("detect_duplicates", False),
                force_reanalyze=params.get("force_reanalyze", False),
            )
        elif job.job_type == "detect_duplicates":
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                similarity_threshold=params.get("similarity_threshold", 5),
                recompute_hashes=params.get("recompute_hashes", False),
            )
        elif job.job_type == "auto_tag":
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                backend=params.get("backend", "openclip"),
                model=params.get("model"),
                threshold=params.get("threshold", 0.25),
                max_tags=params.get("max_tags", 10),
                batch_size=params.get("batch_size", 32),
                continue_pipeline=params.get("continue_pipeline", False),
                max_images=params.get("max_images"),
                tag_mode=params.get("tag_mode", "untagged_only"),
                resume_from_job=job_id,  # Resume from the failed job's checkpoint
            )
        elif job.job_type == "detect_bursts":
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                gap_threshold=params.get("gap_threshold", 2.0),
                min_burst_size=params.get("min_burst_size", 3),
            )
        elif job.job_type == "generate_thumbnails":
            new_task = task_func.delay(
                catalog_id=str(job.catalog_id),
                sizes=params.get("sizes"),
                quality=params.get("quality", 85),
                force=params.get("force", False),
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
