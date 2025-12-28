"""
API endpoints for background job management.

Provides endpoints for submitting, monitoring, and controlling
background jobs (analysis, organization, thumbnail generation).
"""

import logging
import os
from typing import Any, Dict, List, Optional

import redis
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..jobs.celery_app import app as celery_app
from ..jobs.parallel_duplicates import duplicates_coordinator_task
from ..jobs.tasks import (
    analyze_catalog_task,
    generate_thumbnails_task,
    organize_catalog_task,
)

logger = logging.getLogger(__name__)

# Job history tracking
JOB_HISTORY_KEY = "vam:job_history"
MAX_HISTORY_SIZE = 100

# Redis client for job history
_redis_client = None


def get_redis_client() -> redis.Redis:
    """Get or create Redis client for job history tracking."""
    global _redis_client
    if _redis_client is None:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        _redis_client = redis.Redis(
            host=redis_host, port=redis_port, db=0, decode_responses=True
        )
    return _redis_client


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
    force_reanalyze: bool = Field(
        default=False, description="Reset catalog and analyze all files fresh"
    )
    similarity_threshold: int = Field(
        default=5, ge=0, le=64, description="Similarity threshold for duplicates"
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


class DuplicateDetectionJobRequest(BaseModel):
    """Request to start a duplicate detection job."""

    catalog_id: str = Field(description="Catalog UUID")
    similarity_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max Hamming distance for similar images (lower=stricter)",
    )
    recompute_hashes: bool = Field(
        default=False, description="Force recomputation of perceptual hashes"
    )


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


def _track_job(job_id: str, job_type: str, params: dict) -> None:
    """Track job submission in Redis for history."""
    try:
        redis_client = get_redis_client()
        import json
        import time

        job_data = {
            "job_id": job_id,
            "type": job_type,
            "params": params,
            "submitted_at": time.time(),
        }

        # Store job data with job_id as key
        redis_client.setex(
            f"vam:job:{job_id}", 3600 * 24, json.dumps(job_data)  # 24 hour TTL
        )

        # Add to job history list (most recent first)
        redis_client.lpush(JOB_HISTORY_KEY, job_id)
        redis_client.ltrim(JOB_HISTORY_KEY, 0, MAX_HISTORY_SIZE - 1)

    except Exception as e:
        logger.warning(f"Failed to track job {job_id}: {e}")


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
            force_reanalyze=request.force_reanalyze,
            similarity_threshold=request.similarity_threshold,
        )

        logger.info(f"Submitted analysis job: {task.id}")

        # Track job for history
        _track_job(
            task.id,
            "analyze_catalog",
            {
                "catalog_path": request.catalog_path,
                "source_directories": request.source_directories,
                "detect_duplicates": request.detect_duplicates,
                "force_reanalyze": request.force_reanalyze,
            },
        )

        return JobResponse(
            job_id=task.id,
            status="PENDING",
            message="Analysis job submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit analysis job: {e}")
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

        # Track job for history
        _track_job(
            task.id,
            "generate_thumbnails",
            {
                "catalog_path": request.catalog_path,
                "sizes": request.sizes,
                "quality": request.quality,
            },
        )

        return JobResponse(
            job_id=task.id,
            status="PENDING",
            message="Thumbnail generation job submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit thumbnail job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-duplicates", response_model=JobResponse)
async def submit_duplicate_detection_job(
    request: DuplicateDetectionJobRequest,
) -> JobResponse:
    """
    Submit a duplicate detection job.

    Analyzes images to find exact and perceptual duplicates using
    checksums and perceptual hashing (dHash, aHash, wHash).

    Example:
        ```bash
        curl -X POST http://localhost:8000/api/jobs/detect-duplicates \\
          -H "Content-Type: application/json" \\
          -d '{
            "catalog_id": "bd40ca52-c3f7-4877-9c97-1c227389c8c4",
            "similarity_threshold": 5
          }'
        ```
    """
    try:
        task = duplicates_coordinator_task.delay(
            catalog_id=request.catalog_id,
            similarity_threshold=request.similarity_threshold,
            recompute_hashes=request.recompute_hashes,
        )

        logger.info(f"Submitted duplicate detection job: {task.id}")

        # Track job for history
        _track_job(
            task.id,
            "detect_duplicates",
            {
                "catalog_id": request.catalog_id,
                "similarity_threshold": request.similarity_threshold,
                "recompute_hashes": request.recompute_hashes,
            },
        )

        return JobResponse(
            job_id=task.id,
            status="PENDING",
            message="Duplicate detection job submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit duplicate detection job: {e}")
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

        # Track job for history
        _track_job(
            task.id,
            "organize_catalog",
            {
                "catalog_path": request.catalog_path,
                "output_directory": request.output_directory,
                "operation": request.operation,
                "dry_run": request.dry_run,
            },
        )

        return JobResponse(
            job_id=task.id,
            status="PENDING",
            message="Organization job submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit organization job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Job Status and Control Endpoints
# ============================================================================


@router.get("/health")
async def get_worker_health() -> Dict[str, Any]:
    """
    Get Celery worker health status.

    Returns information about active workers and their status.

    Example:
        ```bash
        curl http://localhost:8000/api/jobs/health
        ```
    """
    try:
        inspect = celery_app.control.inspect()

        # Get active workers
        stats = inspect.stats()
        active = inspect.active()

        if not stats:
            return {
                "status": "unhealthy",
                "workers": 0,
                "message": "No workers are currently active",
            }

        worker_count = len(stats)
        active_tasks = sum(len(tasks) for tasks in (active.values() if active else []))

        return {
            "status": "healthy",
            "workers": worker_count,
            "active_tasks": active_tasks,
            "message": f"{worker_count} worker{'s' if worker_count != 1 else ''} active",
        }

    except Exception as e:
        logger.error(f"Failed to get worker health: {e}")
        return {
            "status": "error",
            "workers": 0,
            "message": "Failed to check worker health",
        }


def _get_batch_progress(job_id: str) -> Optional[Dict[str, Any]]:
    """Get progress from job_batches table for coordinator-pattern jobs."""
    from sqlalchemy import text

    from ..db import get_db_context

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

            # Calculate percent complete
            percent = 0
            if total_items > 0:
                percent = int((total_processed / total_items) * 100)
            elif total_batches > 0:
                percent = int((completed_batches / total_batches) * 100)

            # Determine job status from batches
            if completed_batches == total_batches and total_batches > 0:
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
                    "success_count": total_success,
                },
            }
    except Exception as e:
        logger.warning(f"Failed to get batch progress for job {job_id}: {e}")
        return None


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
    from ..jobs.progress_publisher import get_last_progress

    try:
        import sys

        result = AsyncResult(job_id, app=celery_app)
        print(
            f"[DEBUG] Job {job_id}: Celery state = {result.state}",
            file=sys.stderr,
            flush=True,
        )

        # Get progress from multiple sources (Redis, then batch table)
        redis_progress = get_last_progress(job_id)
        batch_progress = _get_batch_progress(job_id)
        logger.info(
            f"[DEBUG] Job {job_id}: batch_progress = {batch_progress}, redis_progress = {redis_progress is not None}"
        )

        # Use batch progress as primary source if available (more reliable than Redis)
        # Redis progress may expire (1 hour TTL) but batch table persists
        progress_source = batch_progress or redis_progress

        # Determine effective status
        # For coordinator-pattern jobs with batch tracking:
        #   - batch_progress is the source of truth (tracks actual worker completion)
        #   - With the CoordinatorTask fix, coordinators no longer update Job status
        #   - Only finalizers mark jobs as SUCCESS
        # For non-coordinator jobs (no batch tracking):
        #   - Trust Celery terminal states (SUCCESS, FAILURE, REVOKED, etc.)
        #   - Don't let stale Redis data override completed jobs
        TERMINAL_STATES = {
            "SUCCESS",
            "FAILURE",
            "REVOKED",
            "RETRY",
            "TIMEOUT",
            "TERMINATED",
        }
        effective_status = result.state

        if batch_progress:
            # Coordinator-pattern job: batch_progress is source of truth
            source_status = batch_progress.get("status")
            logger.info(
                f"[DEBUG] Job {job_id}: batch_status = {source_status}, celery_state = {effective_status}"
            )
            if source_status in ("PROGRESS", "STARTED", "SUCCESS", "FAILURE"):
                effective_status = source_status
        elif result.state not in TERMINAL_STATES and redis_progress:
            # Non-coordinator job: only use Redis if Celery state is non-terminal
            # This prevents stale Redis data from overriding completed jobs
            source_status = redis_progress.get("status")
            logger.info(
                f"[DEBUG] Job {job_id}: redis_status = {source_status}, celery_state = {effective_status}"
            )
            if source_status in ("PROGRESS", "STARTED"):
                effective_status = source_status

        logger.info(
            f"[DEBUG] Job {job_id}: Final effective_status = {effective_status}"
        )

        logger.info(f"[DEBUG] Job {job_id}: Final status = {effective_status}")
        status_response = JobStatus(
            job_id=job_id,
            status=effective_status,
        )

        if effective_status == "PENDING":
            status_response.progress = {"message": "Job is waiting to start..."}
        elif effective_status in ("PROGRESS", "STARTED"):
            # Use batch or Redis progress if available
            if progress_source and progress_source.get("progress"):
                status_response.progress = progress_source["progress"]
            elif result.info:
                status_response.progress = result.info
            else:
                status_response.progress = {"message": "Processing..."}
        elif effective_status == "SUCCESS":
            # Check if we have result from progress source
            if progress_source and progress_source.get("result"):
                status_response.result = progress_source.get("result")
            else:
                status_response.result = result.result
            status_response.progress = {
                "current": 100,
                "total": 100,
                "percent": 100,
                "message": "Job completed successfully",
            }
        elif effective_status == "FAILURE":
            status_response.error = str(result.info) if result.info else "Job failed"
        else:
            # RETRY, REVOKED, etc.
            if progress_source and progress_source.get("progress"):
                status_response.progress = progress_source["progress"]
            else:
                status_response.progress = {
                    "message": f"Job is {effective_status.lower()}..."
                }

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


@router.post("/{job_id}/rerun")
async def rerun_job(job_id: str) -> Dict[str, Any]:
    """
    Rerun a job with the same parameters.

    Fetches the original job parameters and resubmits the job.

    Example:
        ```bash
        curl -X POST http://localhost:8000/api/jobs/{job_id}/rerun
        ```
    """
    try:
        import json

        redis_client = get_redis_client()

        # Fetch original job data from Redis
        job_data_str = redis_client.get(f"vam:job:{job_id}")
        if not job_data_str:
            raise HTTPException(
                status_code=404,
                detail="Job parameters not found. Job may have expired.",
            )

        job_data = json.loads(job_data_str)
        job_type = job_data.get("type")
        params = job_data.get("params", {})

        # Resubmit based on job type
        if job_type == "analyze_catalog":
            task = analyze_catalog_task.delay(**params)
            _track_job(task.id, job_type, params)
            return {
                "job_id": task.id,
                "status": "PENDING",
                "message": "Analysis job resubmitted successfully",
            }
        elif job_type == "organize_catalog":
            task = organize_catalog_task.delay(**params)
            _track_job(task.id, job_type, params)
            return {
                "job_id": task.id,
                "status": "PENDING",
                "message": "Organization job resubmitted successfully",
            }
        elif job_type == "generate_thumbnails":
            task = generate_thumbnails_task.delay(**params)
            _track_job(task.id, job_type, params)
            return {
                "job_id": task.id,
                "status": "PENDING",
                "message": "Thumbnail generation job resubmitted successfully",
            }
        elif job_type == "detect_duplicates":
            task = duplicates_coordinator_task.delay(**params)
            _track_job(task.id, job_type, params)
            return {
                "job_id": task.id,
                "status": "PENDING",
                "message": "Duplicate detection job resubmitted successfully",
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown job type: {job_type}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rerun job: {e}")
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
    List all active and recent jobs.

    Returns information about all currently running jobs and recent completed jobs.

    Example:
        ```bash
        curl http://localhost:8000/api/jobs
        ```
    """
    from ..jobs.progress_publisher import get_last_progress

    try:
        jobs: List[JobStatus] = []

        # Get recent job IDs from history
        redis_client = get_redis_client()
        job_ids = redis_client.lrange(JOB_HISTORY_KEY, 0, 49)  # Last 50 jobs

        # Convert bytes to strings
        job_ids = [
            job_id.decode() if isinstance(job_id, bytes) else job_id
            for job_id in job_ids
        ]

        # Get status for each job
        for task_id in job_ids:
            result = AsyncResult(task_id, app=celery_app)
            redis_progress = get_last_progress(task_id)

            # Determine effective status - always trust Celery for terminal states
            # Terminal states (SUCCESS, FAILURE, etc.) from Celery take precedence
            # Only use Redis progress data if job is actually still running
            TERMINAL_STATES = {
                "SUCCESS",
                "FAILURE",
                "REVOKED",
                "RETRY",
                "TIMEOUT",
                "TERMINATED",
            }
            effective_status = result.state

            # Only use Redis status if Celery state is non-terminal
            # This prevents stale Redis data from showing completed jobs as running
            if result.state not in TERMINAL_STATES and redis_progress:
                redis_status = redis_progress.get("status")
                if redis_status in ("PROGRESS", "STARTED"):
                    effective_status = redis_status

            job_status = JobStatus(
                job_id=task_id,
                status=effective_status,
                progress=None,
                result=None,
                error=None,
            )

            if effective_status == "PENDING":
                job_status.progress = {"message": "Job is waiting to start..."}
            elif effective_status in ("PROGRESS", "STARTED"):
                if redis_progress and redis_progress.get("progress"):
                    job_status.progress = redis_progress["progress"]
                elif result.info:
                    job_status.progress = result.info
                else:
                    job_status.progress = {"message": "Processing..."}
            elif effective_status == "SUCCESS":
                if redis_progress and redis_progress.get("result"):
                    job_status.result = redis_progress.get("result")
                else:
                    job_status.result = result.result
                job_status.progress = {
                    "current": 100,
                    "total": 100,
                    "percent": 100,
                    "message": "Job completed successfully",
                }
            elif effective_status == "FAILURE":
                job_status.error = str(result.info)
            else:
                if redis_progress and redis_progress.get("progress"):
                    job_status.progress = redis_progress["progress"]

            jobs.append(job_status)

        return JobList(jobs=jobs)

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        # Fallback to old behavior if Redis fails
        try:
            inspect = celery_app.control.inspect()
            active = inspect.active()
            jobs = []

            if active:
                for worker_tasks in active.values():
                    for task in worker_tasks:
                        result = AsyncResult(task["id"], app=celery_app)
                        jobs.append(
                            JobStatus(
                                job_id=task["id"],
                                status=result.state,
                                progress=(
                                    result.info if result.state == "PROGRESS" else None
                                ),
                                result=(
                                    result.result if result.state == "SUCCESS" else None
                                ),
                                error=(
                                    str(result.info)
                                    if result.state == "FAILURE"
                                    else None
                                ),
                            )
                        )

            return JobList(jobs=jobs)
        except:
            return JobList(jobs=[])


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

        # Get original parameters from the job
        # Note: Celery doesn't store original kwargs easily, so we need to
        # extract them from the job tracking or result info
        params: Dict[str, Any] = {}
        if hasattr(result, "kwargs") and result.kwargs:
            params = result.kwargs
        elif hasattr(result, "args") and result.args:
            # Convert positional args to kwargs based on task signature
            params = {"catalog_id": result.args[0]} if result.args else {}

        # Resubmit the job
        task = task_func.delay(**params)
        _track_job(task.id, task_name, params)
        return {
            "job_id": task.id,
            "status": "PENDING",
            "message": f"{task_name} job resubmitted successfully",
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
