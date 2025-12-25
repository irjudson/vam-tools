# Task Status System Design

## Current Problem

The coordinator pattern creates a disconnect between Celery task status and actual job progress:

1. **Coordinator task** dispatches child tasks and returns immediately
2. **Celery** marks the coordinator as `SUCCESS` because it completed without error
3. **Child tasks** (workers, finalizers) continue running in the background
4. **UX** sees `SUCCESS` and thinks the job is done, when work is still in progress

This affects all parallel jobs: duplicate detection, thumbnails, tagging, scanning.

## Current Architecture

```
API Request
    ↓
Job Record Created (status: PENDING)
    ↓
Coordinator Task Starts (Celery sets: STARTED)
    ↓
Coordinator Dispatches Workers via chord()
    ↓
Coordinator Returns (Celery sets: SUCCESS)  ← PROBLEM: Job appears "done"
    ↓
Workers Process in Background
    ↓
Finalizer Runs
    ↓
Finalizer Updates Job Record (status: SUCCESS with final result)
```

## The Core Issue

There are TWO status sources that conflict:

1. **Celery Task State** - Managed by Celery, stored in Redis
   - States: PENDING, STARTED, SUCCESS, FAILURE, REVOKED
   - Automatically set when task completes

2. **Job Database Record** - Managed by our code, stored in PostgreSQL
   - `jobs.status` column
   - Can be manually updated by tasks

The API currently reads from the database, but Celery overwrites the status when tasks complete.

---

## Standardized Job Status Values

All jobs will use exactly **5 status values** for UX display:

| Status | Meaning | Typical Flow |
|--------|---------|--------------|
| `queued` | Job is waiting to start | Initial state after job creation |
| `running` | Job is actively processing | Coordinator dispatched, workers active |
| `completed` | Job finished successfully | All work done, no errors |
| `failed` | Job encountered errors | Unrecoverable error occurred |
| `killed` | Job was terminated | User or system stopped the job |

### Happy Path
```
queued → running → completed
```

### Failure Path
```
queued → running → failed
```

### Termination Path
```
queued → running → killed
```

---

## Status Mapping from Celery States

Map Celery's internal states to our 5 standardized values:

```python
def get_effective_status(job: Job) -> str:
    """
    Map Celery status + result.status to one of 5 standard values.

    Returns: "queued", "running", "completed", "failed", or "killed"
    """
    celery_status = job.status
    result = job.result or {}
    result_status = result.get("status")

    # Celery PENDING = job hasn't started yet
    if celery_status == "PENDING":
        return "queued"

    # Celery STARTED = coordinator is running
    if celery_status == "STARTED":
        return "running"

    # Celery SUCCESS with dispatched/processing = workers still running
    if celery_status == "SUCCESS" and result_status in ("dispatched", "processing"):
        return "running"

    # Celery SUCCESS with completed = job done
    if celery_status == "SUCCESS" and result_status in ("completed", "completed_with_errors"):
        return "completed"

    # Celery SUCCESS with requeued = continuation job, original is done
    if celery_status == "SUCCESS" and result_status == "requeued":
        return "completed"  # This job is done, new job handles remaining work

    # Celery FAILURE = job failed
    if celery_status == "FAILURE":
        return "failed"

    # Celery REVOKED = job was killed
    if celery_status == "REVOKED":
        return "killed"

    # Default fallback for unknown states
    if celery_status == "SUCCESS":
        return "completed"

    return "running"  # Safe default for in-progress states
```

---

## Attribution and Error Tracking

### Killed Jobs - Who Killed It?

Add `killed_by` and `killed_at` fields to track termination attribution:

```python
# In job.result when killed
{
    "status": "killed",
    "killed_by": "user",        # or "system", "timeout", "worker_crash"
    "killed_at": "2024-01-15T10:30:00Z",
    "killed_reason": "User requested cancellation via UI",
    "progress_at_kill": {
        "processed": 500,
        "total": 1000
    }
}
```

**Kill Sources:**
| `killed_by` | Meaning |
|-------------|---------|
| `user` | User clicked cancel in UI or called DELETE /api/jobs/{id} |
| `system` | System shutdown, deployment, or resource cleanup |
| `timeout` | Job exceeded maximum allowed runtime |
| `worker_crash` | Celery worker died unexpectedly |
| `oom` | Out of memory killed by OS |

### Failed Jobs - Why Did It Fail?

Add structured error information to track failure reasons:

```python
# In job.result when failed
{
    "status": "failed",
    "error": "Connection refused: PostgreSQL not available",
    "error_type": "DatabaseConnectionError",
    "error_code": "DB_CONN_REFUSED",
    "failed_at": "2024-01-15T10:30:00Z",
    "failed_phase": "processing",  # init, batching, processing, finalizing
    "failed_batch": 5,             # Which batch failed (if applicable)
    "progress_at_failure": {
        "processed": 500,
        "total": 1000,
        "batches_completed": 4,
        "batches_total": 10
    },
    "stack_trace": "..."  # Optional, for debugging
}
```

**Common Error Codes:**
| `error_code` | Meaning |
|--------------|---------|
| `DB_CONN_REFUSED` | Database connection failed |
| `REDIS_UNAVAILABLE` | Redis broker not available |
| `GPU_OOM` | GPU out of memory |
| `DISK_FULL` | No disk space for output |
| `CATALOG_NOT_FOUND` | Catalog ID doesn't exist |
| `WORKER_TIMEOUT` | Worker task exceeded time limit |
| `INVALID_INPUT` | Bad parameters passed to task |

---

## Database Schema Updates

### Option A: Add columns to Job table

```sql
ALTER TABLE jobs ADD COLUMN killed_by VARCHAR(50);
ALTER TABLE jobs ADD COLUMN killed_at TIMESTAMP;
ALTER TABLE jobs ADD COLUMN error_code VARCHAR(50);
ALTER TABLE jobs ADD COLUMN failed_phase VARCHAR(50);
```

### Option B: Store in result JSONB (Recommended)

No schema changes needed. Store all metadata in the existing `result` JSONB column.

---

## API Response Format

```python
class JobStatusResponse(BaseModel):
    job_id: str
    status: str              # One of: queued, running, completed, failed, killed
    job_type: str            # scan, thumbnails, duplicates, tagging
    created_at: datetime
    updated_at: datetime

    # Progress (when running)
    progress: Optional[JobProgress]

    # Completion info (when completed)
    result: Optional[dict]

    # Failure info (when failed)
    error: Optional[str]
    error_code: Optional[str]
    failed_at: Optional[datetime]
    failed_phase: Optional[str]

    # Kill info (when killed)
    killed_by: Optional[str]
    killed_at: Optional[datetime]
    killed_reason: Optional[str]

    # Debug info (optional, for troubleshooting)
    celery_status: Optional[str]  # Raw Celery state


class JobProgress(BaseModel):
    current: int
    total: int
    percent: float
    message: str
    phase: str  # init, batching, processing, finalizing
    batches_completed: Optional[int]
    batches_total: Optional[int]
```

---

## Implementation Steps

### Step 1: Add Status Helper Function

In `vam_tools/api/routers/jobs.py`:

```python
def get_effective_status(job: Job) -> str:
    """Map Celery status + result to standardized status."""
    celery_status = job.status
    result = job.result or {}
    result_status = result.get("status")

    if celery_status == "PENDING":
        return "queued"
    if celery_status == "REVOKED":
        return "killed"
    if celery_status == "FAILURE":
        return "failed"
    if celery_status == "SUCCESS":
        if result_status in ("dispatched", "processing"):
            return "running"
        return "completed"
    return "running"


def get_kill_info(job: Job) -> Optional[dict]:
    """Extract kill attribution from job result."""
    result = job.result or {}
    if result.get("killed_by"):
        return {
            "killed_by": result.get("killed_by"),
            "killed_at": result.get("killed_at"),
            "killed_reason": result.get("killed_reason"),
        }
    return None


def get_failure_info(job: Job) -> Optional[dict]:
    """Extract failure details from job."""
    if job.error or (job.result or {}).get("error"):
        return {
            "error": job.error or job.result.get("error"),
            "error_code": (job.result or {}).get("error_code"),
            "failed_at": (job.result or {}).get("failed_at"),
            "failed_phase": (job.result or {}).get("failed_phase"),
        }
    return None
```

### Step 2: Update Job Status Endpoint

```python
@router.get("/{job_id}")
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    status = get_effective_status(job)

    response = {
        "job_id": job.id,
        "status": status,
        "job_type": job.job_type,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "progress": get_progress_info(job),
        "result": job.result,
        "celery_status": job.status,  # For debugging
    }

    # Add context-specific info
    if status == "failed":
        response.update(get_failure_info(job) or {})
    elif status == "killed":
        response.update(get_kill_info(job) or {})

    return response
```

### Step 3: Update Kill Job Endpoint

```python
@router.delete("/{job_id}")
def cancel_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Revoke the Celery task
    app.control.revoke(job_id, terminate=True)

    # Update job with kill attribution
    job.status = "REVOKED"
    job.result = {
        **(job.result or {}),
        "status": "killed",
        "killed_by": "user",
        "killed_at": datetime.utcnow().isoformat(),
        "killed_reason": "User requested cancellation",
    }
    db.commit()

    return {"message": "Job cancelled", "status": "killed"}
```

### Step 4: Update Coordinator Tasks

All coordinators should set consistent result status:

```python
# When coordinator dispatches workers
return {
    "status": "dispatched",  # → effective status: "running"
    "catalog_id": catalog_id,
    "total_items": total,
    "message": f"Processing {total} items",
}

# When finalizer completes successfully
return {
    "status": "completed",  # → effective status: "completed"
    "catalog_id": catalog_id,
    "processed": count,
    "errors": error_count,
}

# When finalizer completes with errors
return {
    "status": "completed_with_errors",  # → effective status: "completed"
    "catalog_id": catalog_id,
    "processed": success_count,
    "errors": error_count,
    "error_details": [...],
}

# When task fails
return {
    "status": "failed",
    "error": str(e),
    "error_code": "PROCESSING_ERROR",
    "failed_at": datetime.utcnow().isoformat(),
    "failed_phase": "processing",
}
```

### Step 5: Update Jobs List Endpoint

Apply same status mapping to list view:

```python
@router.get("/")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(100).all()

    return [
        {
            "job_id": job.id,
            "status": get_effective_status(job),
            "job_type": job.job_type,
            "created_at": job.created_at,
            "celery_status": job.status,
        }
        for job in jobs
    ]
```

---

## Files to Modify

1. **`vam_tools/api/routers/jobs.py`**
   - Add `get_effective_status()`, `get_kill_info()`, `get_failure_info()`
   - Update `get_job_status()` endpoint
   - Update `list_jobs()` endpoint
   - Update `cancel_job()` endpoint with attribution

2. **All coordinator tasks** - Ensure consistent result format:
   - `vam_tools/jobs/parallel_duplicates.py`
   - `vam_tools/jobs/parallel_thumbnails.py`
   - `vam_tools/jobs/parallel_tagging.py`
   - `vam_tools/jobs/parallel_scan.py`

3. **Worker failure handlers** - Add error codes and phase info

---

## Testing

```bash
# Start a job
curl -X POST "http://localhost:8765/api/catalogs/{id}/detect-duplicates"

# Check status while running
curl "http://localhost:8765/api/jobs/{job_id}"
# Expected: {"status": "running", "celery_status": "SUCCESS", ...}

# After completion
curl "http://localhost:8765/api/jobs/{job_id}"
# Expected: {"status": "completed", ...}

# Kill a running job
curl -X DELETE "http://localhost:8765/api/jobs/{job_id}"
# Expected: {"status": "killed", "killed_by": "user", ...}

# Check killed job status
curl "http://localhost:8765/api/jobs/{job_id}"
# Expected: {"status": "killed", "killed_by": "user", "killed_reason": "...", ...}
```

---

## UX Display Guidelines

| Status | Icon | Color | Message |
|--------|------|-------|---------|
| `queued` | ⏳ | Gray | "Waiting to start..." |
| `running` | 🔄 | Blue | "Processing... {progress}%" |
| `completed` | ✅ | Green | "Completed successfully" |
| `failed` | ❌ | Red | "Failed: {error}" |
| `killed` | 🛑 | Orange | "Cancelled by {killed_by}" |

---

## Alternative Approaches Considered

### 1. Don't Return from Coordinator (Rejected)
Make coordinator wait for finalizer via `chord().get()`.
- Problem: Blocks a worker for entire job duration, defeats purpose of parallel processing.

### 2. Use Celery Groups with Parent Tracking (Complex)
- Problem: Requires significant refactoring of task infrastructure.

### 3. Separate "Job" from "Task" (Over-engineered)
Create separate Job model that tracks multiple tasks.
- Problem: Adds complexity, requires migrations, changes all endpoints.

---

## Conclusion

This design provides:
- **5 clear status values** that map to UX needs
- **Attribution tracking** for killed jobs (user vs system)
- **Structured error info** for failed jobs
- **Backwards compatibility** via result JSONB storage
- **Minimal code changes** (~100 lines in API layer)
- **No database migrations** required
