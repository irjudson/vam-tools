"""Job management endpoints."""

import asyncio
import logging
import uuid
from typing import Any, Dict, List

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...celery_app import app as celery_app
from ...db import get_db
from ...db.models import Job
from ...db.schemas import JobListResponse
from ...jobs.tasks import analyze_catalog_task, scan_catalog_task

logger = logging.getLogger(__name__)

router = APIRouter()


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


@router.get("/health")
def get_worker_health():
    """Get Celery worker health status."""
    try:
        # Check worker stats
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        active = inspect.active()

        if stats:
            worker_count = len(stats)
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
    except Exception as e:
        logger.error(f"Error checking worker health: {e}")
        return {"status": "error", "workers": 0, "active_tasks": 0, "message": str(e)}


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


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get job status and progress."""
    task = AsyncResult(job_id, app=celery_app)

    if not task:
        raise HTTPException(status_code=404, detail="Job not found")

    # Safely get task state - can raise ValueError if exception info is malformed
    state = _safe_get_task_state(task)

    response = JobResponse(
        job_id=job_id,
        status=state,
        progress={},
        result={},
    )

    # Get progress if available
    if state == "PROGRESS":
        response.progress = _safe_get_task_info(task) or {}
        # Update job status in database
        job = db.query(Job).filter(Job.id == job_id).first()
        if job and job.status != "PROGRESS":
            job.status = "PROGRESS"
            db.commit()
    elif state == "SUCCESS":
        try:
            response.result = task.result or {}
        except Exception as e:
            logger.warning(f"Error getting task result for {job_id}: {e}")
            response.result = {"error": f"Failed to retrieve result: {e}"}

        # Save result to database for history
        job = db.query(Job).filter(Job.id == job_id).first()
        if job and job.status != "SUCCESS":
            job.status = "SUCCESS"
            job.result = response.result
            db.commit()
            logger.info(f"Saved job {job_id} result to database")
    elif state == "FAILURE":
        task_info = _safe_get_task_info(task)
        response.result = {"error": str(task_info)}

        # Save error to database
        job = db.query(Job).filter(Job.id == job_id).first()
        if job and job.status != "FAILURE":
            job.status = "FAILURE"
            job.error = str(task_info)
            # Try to get statistics from failure info if available
            if isinstance(task_info, dict) and "statistics" in task_info:
                job.result = task_info.get("statistics")
            db.commit()

    return response


@router.delete("/{job_id}", status_code=204)
def revoke_job(
    job_id: str,
    terminate: bool = False,
    force: bool = False,
    db: Session = Depends(get_db),
):
    """Revoke/cancel a job, or force delete it from the database.

    Args:
        job_id: The job ID to revoke/delete
        terminate: If True, forcefully terminate the task (default: False)
        force: If True, completely delete the job from the database instead of marking as REVOKED
    """
    action = "force deleting" if force else "revoking"
    logger.info(f"{action.capitalize()} job {job_id} (terminate={terminate})")

    try:
        if force:
            # Force delete: completely remove the job record from the database
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            # Still try to revoke in Celery if it's running
            celery_app.control.revoke(
                job_id,
                terminate=terminate,
                signal="SIGKILL" if terminate else "SIGTERM",
            )

            # Clean up result backend
            task = AsyncResult(job_id, app=celery_app)
            if task:
                task.forget()

            # DELETE the job record completely
            db.delete(job)
            db.commit()

            logger.info(f"Job {job_id} deleted from database")
            return {"status": "deleted", "job_id": job_id}
        else:
            # Soft delete: mark as REVOKED (preserves audit trail)
            # Revoke the task in Celery
            celery_app.control.revoke(
                job_id,
                terminate=terminate,
                signal="SIGKILL" if terminate else "SIGTERM",
            )

            # Also mark it as revoked in the result backend
            task = AsyncResult(job_id, app=celery_app)
            if task:
                # This doesn't actually revoke it but marks it for cleanup
                task.forget()

            # Update database record to reflect revocation
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = "REVOKED"
                job.error = "Job was manually cancelled"
                db.commit()

            return {"status": "revoked", "job_id": job_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error {action} job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to {action} job: {str(e)}")


@router.websocket("/{job_id}/stream")
async def stream_job_updates(websocket: WebSocket, job_id: str):
    """Stream real-time job updates via WebSocket."""
    await websocket.accept()

    try:
        task = AsyncResult(job_id, app=celery_app)
        last_state = None
        last_progress = None

        while True:
            # Use safe accessor for task state
            current_state = _safe_get_task_state(task)

            # Build update message
            update = {
                "job_id": job_id,
                "status": current_state,
                "progress": {},
                "result": {},
            }

            # Get progress/result based on state
            if current_state == "PROGRESS":
                update["progress"] = _safe_get_task_info(task) or {}
            elif current_state == "SUCCESS":
                try:
                    update["result"] = task.result or {}
                except Exception as e:
                    logger.warning(f"Error getting task result for {job_id}: {e}")
                    update["result"] = {"error": f"Failed to retrieve result: {e}"}
                await websocket.send_json(update)
                break  # Job done, close connection
            elif current_state == "FAILURE":
                task_info = _safe_get_task_info(task)
                update["result"] = {"error": str(task_info)}
                await websocket.send_json(update)
                break  # Job failed, close connection

            # Send update if state changed OR progress data changed
            current_progress = str(update.get("progress", {}))
            if current_state != last_state or current_progress != last_progress:
                await websocket.send_json(update)
                last_state = current_state
                last_progress = current_progress

            # Poll every 500ms
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: E722
            pass
