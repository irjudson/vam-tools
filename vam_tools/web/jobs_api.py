"""
API endpoints for background job management.

Provides endpoints for submitting, monitoring, and controlling
background jobs (analysis, organization, thumbnail generation).
"""

import logging
from typing import Any, Dict, List, Optional

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..jobs.celery_app import app as celery_app
from ..jobs.tasks import (
    analyze_catalog_task,
    generate_thumbnails_task,
    organize_catalog_task,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ============================================================================
# Request/Response Models
# ============================================================================


class AnalyzeJobRequest(BaseModel):
    """Request to start an analysis job."""

    catalog_path: str = Field(description="Path to catalog directory")
    source_directories: List[str] = Field(
        description="List of source directories to scan"
    )
    detect_duplicates: bool = Field(
        default=False, description="Whether to detect duplicates"
    )
    similarity_threshold: int = Field(
        default=5, ge=0, le=64, description="Similarity threshold for duplicates"
    )
    workers: Optional[int] = Field(
        default=None, description="Number of worker processes"
    )
    checkpoint_interval: int = Field(
        default=100, ge=1, description="Checkpoint interval in number of files"
    )


class OrganizeJobRequest(BaseModel):
    """Request to start an organization job."""

    catalog_path: str = Field(description="Path to catalog")
    output_directory: str = Field(description="Output directory")
    operation: str = Field(default="copy", description="Operation type: copy or move")
    directory_structure: str = Field(
        default="YYYY-MM", description="Directory structure pattern"
    )
    naming_strategy: str = Field(
        default="date_time_checksum", description="File naming strategy"
    )
    dry_run: bool = Field(default=False, description="Preview without executing")
    verify_checksums: bool = Field(
        default=True, description="Verify checksums after operations"
    )
    skip_existing: bool = Field(default=True, description="Skip existing files")


class ThumbnailJobRequest(BaseModel):
    """Request to start a thumbnail generation job."""

    catalog_path: str = Field(description="Path to catalog")
    sizes: Optional[List[int]] = Field(
        default=None, description="Thumbnail sizes (default: [256, 512])"
    )
    quality: int = Field(default=85, ge=1, le=100, description="JPEG quality")
    force: bool = Field(default=False, description="Regenerate existing thumbnails")


class JobResponse(BaseModel):
    """Response after job submission."""

    job_id: str = Field(description="Unique job ID")
    status: str = Field(description="Job status")
    message: str = Field(description="Status message")


class JobStatus(BaseModel):
    """Job status and progress information."""

    job_id: str = Field(description="Job ID")
    status: str = Field(
        description="Job status: PENDING, STARTED, PROGRESS, SUCCESS, FAILURE, REVOKED"
    )
    progress: Optional[Dict[str, Any]] = Field(
        default=None, description="Progress information"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Job result (when complete)"
    )
    error: Optional[str] = Field(default=None, description="Error message (if failed)")


class JobList(BaseModel):
    """List of active jobs."""

    jobs: List[JobStatus] = Field(description="List of jobs")


# ============================================================================
# Job Submission Endpoints
# ============================================================================


@router.post("/analyze", response_model=JobResponse)
async def submit_analyze_job(request: AnalyzeJobRequest) -> JobResponse:
    """
    Submit a catalog analysis job.

    Starts background analysis of specified source directories,
    extracting metadata and optionally detecting duplicates.

    Example:
        ```bash
        curl -X POST http://localhost:8000/api/jobs/analyze \\
          -H "Content-Type: application/json" \\
          -d '{
            "catalog_path": "/app/catalogs/my-catalog",
            "source_directories": ["/app/photos"],
            "detect_duplicates": true
          }'
        ```
    """
    try:
        # Submit task to Celery
        task = analyze_catalog_task.delay(
            catalog_path=request.catalog_path,
            source_directories=request.source_directories,
            detect_duplicates=request.detect_duplicates,
            similarity_threshold=request.similarity_threshold,
            workers=request.workers,
            checkpoint_interval=request.checkpoint_interval,
        )

        logger.info(f"Submitted analysis job: {task.id}")

        return JobResponse(
            job_id=task.id,
            status="PENDING",
            message="Analysis job submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit analysis job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/organize", response_model=JobResponse)
async def submit_organize_job(request: OrganizeJobRequest) -> JobResponse:
    """
    Submit a catalog organization job.

    Organizes catalog files according to specified strategy,
    creating a clean chronological structure.

    Example:
        ```bash
        curl -X POST http://localhost:8000/api/jobs/organize \\
          -H "Content-Type: application/json" \\
          -d '{
            "catalog_path": "/app/catalogs/my-catalog",
            "output_directory": "/app/organized",
            "dry_run": true
          }'
        ```
    """
    try:
        task = organize_catalog_task.delay(
            catalog_path=request.catalog_path,
            output_directory=request.output_directory,
            operation=request.operation,
            directory_structure=request.directory_structure,
            naming_strategy=request.naming_strategy,
            dry_run=request.dry_run,
            verify_checksums=request.verify_checksums,
            skip_existing=request.skip_existing,
        )

        logger.info(f"Submitted organization job: {task.id}")

        return JobResponse(
            job_id=task.id,
            status="PENDING",
            message="Organization job submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit organization job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/thumbnails", response_model=JobResponse)
async def submit_thumbnail_job(request: ThumbnailJobRequest) -> JobResponse:
    """
    Submit a thumbnail generation job.

    Generates thumbnails for all images in the catalog.

    Example:
        ```bash
        curl -X POST http://localhost:8000/api/jobs/thumbnails \\
          -H "Content-Type: application/json" \\
          -d '{
            "catalog_path": "/app/catalogs/my-catalog",
            "sizes": [256, 512],
            "quality": 85
          }'
        ```
    """
    try:
        task = generate_thumbnails_task.delay(
            catalog_path=request.catalog_path,
            sizes=request.sizes,
            quality=request.quality,
            force=request.force,
        )

        logger.info(f"Submitted thumbnail job: {task.id}")

        return JobResponse(
            job_id=task.id,
            status="PENDING",
            message="Thumbnail generation job submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit thumbnail job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Job Status and Control Endpoints
# ============================================================================


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """
    Get status of a specific job.

    Returns current status, progress information, and results.

    Example:
        ```bash
        curl http://localhost:8000/api/jobs/{job_id}
        ```
    """
    try:
        result = AsyncResult(job_id, app=celery_app)

        status_response = JobStatus(
            job_id=job_id,
            status=result.state,
        )

        if result.state == "PENDING":
            status_response.progress = {"message": "Job is waiting to start..."}
        elif result.state == "PROGRESS":
            status_response.progress = result.info
        elif result.state == "SUCCESS":
            status_response.result = result.result
            status_response.progress = {
                "current": 100,
                "total": 100,
                "percent": 100,
                "message": "Job completed successfully",
            }
        elif result.state == "FAILURE":
            status_response.error = str(result.info)
        else:
            # STARTED, RETRY, REVOKED, etc.
            status_response.progress = {"message": f"Job is {result.state.lower()}..."}

        return status_response

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{job_id}")
async def cancel_job(job_id: str) -> Dict[str, str]:
    """
    Cancel a running job.

    Revokes the job and stops execution.

    Example:
        ```bash
        curl -X DELETE http://localhost:8000/api/jobs/{job_id}
        ```
    """
    try:
        result = AsyncResult(job_id, app=celery_app)
        result.revoke(terminate=True)

        logger.info(f"Cancelled job: {job_id}")

        return {
            "status": "cancelled",
            "message": f"Job {job_id} has been cancelled",
        }

    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/stream")
async def stream_job_progress(job_id: str) -> EventSourceResponse:
    """
    Stream job progress using Server-Sent Events (SSE).

    Provides real-time updates as the job progresses.

    Example:
        ```javascript
        const eventSource = new EventSource('/api/jobs/{job_id}/stream');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(`Progress: ${data.percent}%`);
        };
        ```
    """

    async def event_generator():
        """Generate SSE events for job progress."""
        result = AsyncResult(job_id, app=celery_app)
        last_state = None
        last_info = None

        while True:
            # Get current state
            current_state = result.state
            current_info = result.info

            # Only send update if something changed
            if current_state != last_state or current_info != last_info:
                data = {
                    "job_id": job_id,
                    "status": current_state,
                }

                if current_state == "PROGRESS":
                    data["progress"] = current_info
                elif current_state == "SUCCESS":
                    data["result"] = current_info
                    yield {"data": data}
                    break  # Job complete, close stream
                elif current_state == "FAILURE":
                    data["error"] = str(current_info)
                    yield {"data": data}
                    break  # Job failed, close stream
                elif current_state in ["REVOKED", "RETRY"]:
                    data["message"] = f"Job {current_state.lower()}"
                    yield {"data": data}
                    if current_state == "REVOKED":
                        break  # Job cancelled, close stream

                yield {"data": data}

                last_state = current_state
                last_info = current_info

            # Wait before checking again (1 second)
            import asyncio

            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


@router.get("", response_model=JobList)
async def list_active_jobs() -> JobList:
    """
    List all active jobs.

    Returns information about all currently running and recent jobs.

    Example:
        ```bash
        curl http://localhost:8000/api/jobs
        ```
    """
    try:
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()
        active = inspect.active()
        scheduled = inspect.scheduled()
        reserved = inspect.reserved()

        jobs: List[JobStatus] = []

        # Collect all task IDs
        task_ids = set()

        if active:
            for worker_tasks in active.values():
                task_ids.update(task["id"] for task in worker_tasks)

        if scheduled:
            for worker_tasks in scheduled.values():
                task_ids.update(task["id"] for task in worker_tasks)

        if reserved:
            for worker_tasks in reserved.values():
                task_ids.update(task["id"] for task in worker_tasks)

        # Get status for each task
        for task_id in task_ids:
            result = AsyncResult(task_id, app=celery_app)
            jobs.append(
                JobStatus(
                    job_id=task_id,
                    status=result.state,
                    progress=result.info if result.state == "PROGRESS" else None,
                    result=result.result if result.state == "SUCCESS" else None,
                    error=str(result.info) if result.state == "FAILURE" else None,
                )
            )

        return JobList(jobs=jobs)

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{job_id}/restart")
async def restart_job(job_id: str) -> Dict[str, Any]:
    """
    Restart a failed or cancelled job with the same parameters.

    This retrieves the job's original parameters and submits a new job.
    Useful for retrying failed jobs or restarting cancelled ones.
    """
    try:
        # Get original job result
        result = AsyncResult(job_id, app=celery_app)

        # Check if job exists
        if not result:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get job info (args and kwargs used)
        task_name = result.task_name if hasattr(result, "task_name") else None

        if not task_name:
            raise HTTPException(
                status_code=400, detail="Cannot determine job type for restart"
            )

        # Map task names to task functions
        task_map = {
            "analyze_catalog": analyze_catalog_task,
            "organize_catalog": organize_catalog_task,
            "generate_thumbnails": generate_thumbnails_task,
        }

        # Get the task function
        task_func = task_map.get(task_name)
        if not task_func:
            raise HTTPException(
                status_code=400, detail=f"Unknown task type: {task_name}"
            )

        # Submit new job (Note: we can't easily get original args/kwargs from Celery)
        # So we return instructions for the user to resubmit manually
        return {
            "status": "restart_required",
            "message": "To restart this job, please resubmit using the original parameters",
            "original_job_id": job_id,
            "task_type": task_name,
            "note": "Job parameters cannot be automatically retrieved. Please use the form to resubmit.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{job_id}/kill")
async def kill_job(job_id: str) -> Dict[str, str]:
    """
    Forcefully kill a stuck job.

    This is more aggressive than cancel - it sends SIGKILL to the worker process.
    Use this when a job is truly stuck and not responding to cancellation.
    """
    try:
        result = AsyncResult(job_id, app=celery_app)

        # Terminate with extreme prejudice
        result.revoke(terminate=True, signal="SIGKILL")

        logger.warning(f"Force-killed job: {job_id}")

        return {
            "status": "killed",
            "message": f"Job {job_id} has been force-killed",
            "note": "If the job was modifying files, they may be in an incomplete state. Check your catalog integrity.",
        }
    except Exception as e:
        logger.error(f"Failed to kill job: {e}")
        raise HTTPException(status_code=500, detail=str(e))
