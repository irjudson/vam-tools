#!/usr/bin/env python3
"""Trigger duplicate detection finalizer task."""

import sys
import uuid
from lumina.celery_app import app
from lumina.jobs.parallel_duplicates import duplicates_finalizer_task

# Parameters from the previous run
CATALOG_ID = "bd40ca52-c3f7-4877-9c97-1c227389c8c4"
PARENT_JOB_ID = "fa8240b4-abe4-4b16-9a4c-1509a1159729"
TOTAL_IMAGES = 96264
SIMILARITY_THRESHOLD = 5

print(f"Triggering duplicate detection finalizer...")
print(f"  Catalog ID: {CATALOG_ID}")
print(f"  Parent Job ID: {PARENT_JOB_ID}")
print(f"  Total Images: {TOTAL_IMAGES}")
print(f"  Similarity Threshold: {SIMILARITY_THRESHOLD}")

# Generate a task ID
task_id = str(uuid.uuid4())

# Trigger the finalizer task without waiting for result
# Use apply_async with ignore_result=True since we have no result backend
result = duplicates_finalizer_task.apply_async(
    args=(CATALOG_ID, PARENT_JOB_ID, TOTAL_IMAGES, SIMILARITY_THRESHOLD),
    task_id=task_id,
    ignore_result=True
)

print(f"\nFinalizer task submitted!")
print(f"Task ID: {task_id}")
print(f"\nMonitor progress:")
print(f"  docker compose logs -f celery-worker | grep '{task_id[:8]}'")
print(f"\nCheck job status:")
print(f"  SELECT id, status, progress, updated_at FROM jobs WHERE id = '{task_id}'\\\\g")

sys.exit(0)
