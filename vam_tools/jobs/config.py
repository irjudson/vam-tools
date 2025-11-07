"""
Celery configuration settings.
"""

import os
from typing import Optional


class CeleryConfig:
    """Celery configuration."""

    # Broker settings
    broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

    # Result backend settings
    result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    # Task settings
    task_serializer: str = "json"
    accept_content: list = ["json"]
    result_serializer: str = "json"
    timezone: str = "UTC"
    enable_utc: bool = True

    # Task execution settings
    task_track_started: bool = True  # Track when tasks start
    task_time_limit: int = 3600 * 24  # 24 hours max per task
    task_soft_time_limit: int = 3600 * 23  # 23 hours soft limit
    task_acks_late: bool = True  # Acknowledge after task completion
    worker_prefetch_multiplier: int = 1  # One task at a time per worker

    # Result backend settings
    result_expires: int = 3600 * 24  # Results expire after 24 hours
    result_extended: bool = True  # Store additional metadata

    # Worker settings
    worker_max_tasks_per_child: Optional[int] = 10  # Restart worker after N tasks
    worker_disable_rate_limits: bool = True

    # Performance settings
    broker_connection_retry_on_startup: bool = True
    broker_connection_retry: bool = True

    # Task routing
    task_routes = {
        "vam_tools.jobs.tasks.analyze_catalog_task": {"queue": "analysis"},
        "vam_tools.jobs.tasks.organize_catalog_task": {"queue": "organization"},
        "vam_tools.jobs.tasks.generate_thumbnails_task": {"queue": "thumbnails"},
    }

    # Monitoring
    worker_send_task_events: bool = True
    task_send_sent_event: bool = True


def get_celery_config() -> CeleryConfig:
    """Get Celery configuration."""
    return CeleryConfig()
