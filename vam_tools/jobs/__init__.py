"""
Background job processing with Celery.

This module provides Celery tasks for running CLI operations asynchronously
through the web interface with progress tracking.
"""

from .celery_app import app as celery_app
from .tasks import (
    analyze_catalog_task,
    generate_thumbnails_task,
    organize_catalog_task,
)

__all__ = [
    "celery_app",
    "analyze_catalog_task",
    "organize_catalog_task",
    "generate_thumbnails_task",
]
