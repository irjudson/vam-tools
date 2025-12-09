"""
Background job processing with Celery.

This module provides Celery tasks for running CLI operations asynchronously
through the web interface with progress tracking.

Includes the Parallel Job Coordinator pattern for distributing large jobs
across multiple workers with restartable batches.
"""

from .celery_app import app as celery_app
from .coordinator import BatchManager, BatchResult, JobProgress, publish_job_progress
from .parallel_bursts import (
    burst_coordinator_task,
    burst_finalizer_task,
    burst_worker_task,
)
from .parallel_duplicates import (
    duplicates_compare_worker_task,
    duplicates_comparison_phase_task,
    duplicates_coordinator_task,
    duplicates_finalizer_task,
    duplicates_hash_worker_task,
)
from .parallel_scan import (
    scan_coordinator_task,
    scan_finalizer_task,
    scan_recovery_task,
    scan_worker_task,
)
from .parallel_tagging import (
    tagging_coordinator_task,
    tagging_finalizer_task,
    tagging_worker_task,
)
from .parallel_thumbnails import (
    thumbnail_coordinator_task,
    thumbnail_finalizer_task,
    thumbnail_worker_task,
)
from .tasks import (  # generate_thumbnails_task, # Commented out for debugging; organize_catalog_task, # Commented out for debugging
    analyze_catalog_task,
    scan_catalog_task,
)

__all__ = [
    "celery_app",
    "analyze_catalog_task",
    "scan_catalog_task",
    # Parallel scan tasks
    "scan_coordinator_task",
    "scan_worker_task",
    "scan_finalizer_task",
    "scan_recovery_task",
    # Parallel thumbnail tasks
    "thumbnail_coordinator_task",
    "thumbnail_worker_task",
    "thumbnail_finalizer_task",
    # Parallel tagging tasks
    "tagging_coordinator_task",
    "tagging_worker_task",
    "tagging_finalizer_task",
    # Parallel burst detection tasks
    "burst_coordinator_task",
    "burst_worker_task",
    "burst_finalizer_task",
    # Parallel duplicate detection tasks
    "duplicates_coordinator_task",
    "duplicates_hash_worker_task",
    "duplicates_compare_worker_task",
    "duplicates_comparison_phase_task",
    "duplicates_finalizer_task",
    # Coordinator utilities
    "BatchManager",
    "BatchResult",
    "JobProgress",
    "publish_job_progress",
]
