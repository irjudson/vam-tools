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


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str):
    """Get job status and progress."""
    task = AsyncResult(job_id, app=celery_app)

    if not task:
        raise HTTPException(status_code=404, detail="Job not found")

    response = JobResponse(
        job_id=job_id,
        status=task.state,
        progress={},
        result={},
    )

    # Get progress if available
    if task.state == "PROGRESS":
        response.progress = task.info or {}
    elif task.state == "SUCCESS":
        response.result = task.result or {}
    elif task.state == "FAILURE":
        response.result = {"error": str(task.info)}

    return response


@router.delete("/{job_id}", status_code=204)
def revoke_job(job_id: str, terminate: bool = False, db: Session = Depends(get_db)):
    """Revoke/cancel a job.

    Args:
        job_id: The job ID to revoke
        terminate: If True, forcefully terminate the task (default: False)
    """
    logger.info(f"Revoking job {job_id} (terminate={terminate})")

    try:
        # Revoke the task in Celery
        celery_app.control.revoke(
            job_id, terminate=terminate, signal="SIGKILL" if terminate else "SIGTERM"
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
    except Exception as e:
        logger.error(f"Error revoking job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to revoke job: {str(e)}")


@router.websocket("/{job_id}/stream")
async def stream_job_updates(websocket: WebSocket, job_id: str):
    """Stream real-time job updates via WebSocket."""
    await websocket.accept()

    try:
        task = AsyncResult(job_id, app=celery_app)
        last_state = None
        last_progress = None

        while True:
            current_state = task.state

            # Build update message
            update = {
                "job_id": job_id,
                "status": current_state,
                "progress": {},
                "result": {},
            }

            # Get progress/result based on state
            if current_state == "PROGRESS":
                update["progress"] = task.info or {}
            elif current_state == "SUCCESS":
                update["result"] = task.result or {}
                await websocket.send_json(update)
                break  # Job done, close connection
            elif current_state == "FAILURE":
                update["result"] = {"error": str(task.info)}
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
