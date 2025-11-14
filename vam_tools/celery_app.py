"""Celery application for background task processing."""

import logging

from celery import Celery
from celery.signals import task_failure, task_success

from .db.config import settings

logger = logging.getLogger(__name__)

# Create Celery app
app = Celery(
    "vam_tools",
    broker=settings.redis_url,
    backend=settings.redis_url,  # Store results in Redis
)

# Import tasks to register them
from .tasks import duplicates, organize, scan  # noqa: E402, F401

# Celery configuration
app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task execution settings
    task_track_started=True,
    task_time_limit=3600 * 4,  # 4 hours max per task
    task_soft_time_limit=3600 * 3,  # 3 hours soft limit
    # Result backend settings
    result_expires=3600 * 24,  # Keep results for 24 hours
    result_extended=True,  # Store more task metadata
    # Worker settings
    worker_prefetch_multiplier=1,  # Take one task at a time
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks (prevent memory leaks)
    # Routing
    task_routes={
        "vam_tools.tasks.scan.*": {"queue": "scanner"},
        "vam_tools.tasks.duplicates.*": {"queue": "analyzer"},
        "vam_tools.tasks.organize.*": {"queue": "organizer"},
    },
)


@task_success.connect
def task_success_handler(sender=None, **kwargs):
    """Log successful task completion."""
    logger.info(f"Task {sender.name} completed successfully")


@task_failure.connect
def task_failure_handler(sender=None, exception=None, **kwargs):
    """Log task failures."""
    logger.error(f"Task {sender.name} failed: {exception}")


if __name__ == "__main__":
    app.start()
