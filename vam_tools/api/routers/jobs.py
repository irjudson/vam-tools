"""Job management endpoints."""

import asyncio
import logging
import uuid
from typing import Any, Dict, List

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ...celery_app import app as celery_app
from ...jobs.tasks import analyze_catalog_task

logger = logging.getLogger(__name__)

router = APIRouter()


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
def start_scan(request: ScanJobRequest):
    """Start a directory scan job."""
    logger.info(f"Starting scan for catalog {request.catalog_id}")

    # Submit Celery task - scan is now part of analyze
    task = analyze_catalog_task.delay(
        catalog_path=str(request.catalog_id),
        source_directories=request.directories,
        detect_duplicates=False,
        force_reanalyze=False,
    )

    return JobResponse(
        job_id=task.id,
        status="pending",
        progress={},
        result={},
    )


@router.post("/analyze", response_model=JobResponse, status_code=202)
def start_analyze(request: AnalyzeJobRequest):
    """Start an analyze job (scan directories for a catalog)."""
    logger.info(f"Starting analyze for catalog {request.catalog_id}")

    # Submit Celery task
    task = analyze_catalog_task.delay(
        catalog_path=str(request.catalog_id),
        source_directories=request.source_directories,
        detect_duplicates=request.detect_duplicates,
        force_reanalyze=request.force_reanalyze,
    )

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
                "message": f"{worker_count} worker(s) online"
            }
        else:
            return {
                "status": "unhealthy",
                "workers": 0,
                "active_tasks": 0,
                "message": "No workers available"
            }
    except Exception as e:
        logger.error(f"Error checking worker health: {e}")
        return {
            "status": "error",
            "workers": 0,
            "active_tasks": 0,
            "message": str(e)
        }


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


@router.websocket("/{job_id}/stream")
async def stream_job_updates(websocket: WebSocket, job_id: str):
    """Stream real-time job updates via WebSocket."""
    await websocket.accept()

    try:
        task = AsyncResult(job_id, app=celery_app)
        last_state = None

        while True:
            current_state = task.state

            # Build update message
            update = {
                "job_id": job_id,
                "status": current_state,
                "progress": {},
                "result": {}
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

            # Send update if state changed
            if current_state != last_state:
                await websocket.send_json(update)
                last_state = current_state

            # Poll every 500ms
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
