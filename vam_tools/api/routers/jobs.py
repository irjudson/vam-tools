"""Job management endpoints."""

import logging
import uuid
from typing import Any, Dict, List

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...celery_app import app as celery_app
from ...tasks.scan import scan_directories

logger = logging.getLogger(__name__)

router = APIRouter()


class ScanJobRequest(BaseModel):
    """Request to start a scan job."""

    catalog_id: uuid.UUID
    directories: List[str]


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

    # Submit Celery task
    task = scan_directories.delay(
        catalog_id=str(request.catalog_id),
        directories=request.directories,
    )

    return JobResponse(
        job_id=task.id,
        status="pending",
        progress={},
        result={},
    )


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
