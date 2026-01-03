"""
Celery configuration settings.
"""

import os


def get_celery_config() -> dict:
    """Get Celery configuration as a dictionary.

    Celery's config_from_object expects either:
    - A module path string (e.g., 'myapp.celeryconfig')
    - A dict
    - An object with UPPERCASE attributes

    Using a dict is clearest and most reliable.
    """
    return {
        # Broker settings
        "broker_url": os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/2"),
        # Result backend settings
        "result_backend": os.getenv(
            "CELERY_RESULT_BACKEND", "redis://localhost:6379/2"
        ),
        # Task settings
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        # Task execution settings - CRITICAL for fault tolerance
        "task_track_started": True,  # Track when tasks start
        "task_time_limit": 3600 * 24,  # 24 hours max per task
        "task_soft_time_limit": 3600 * 23,  # 23 hours soft limit
        "task_acks_late": True,  # Acknowledge AFTER completion (redelivery on crash)
        "task_reject_on_worker_lost": True,  # Reject task if worker dies (requeue)
        "worker_prefetch_multiplier": 1,  # One task at a time per worker
        # Result backend settings
        "result_expires": 3600 * 24,  # Results expire after 24 hours
        "result_extended": True,  # Store additional metadata
        # Worker settings
        "worker_max_tasks_per_child": 10,  # Restart worker after N tasks
        "worker_disable_rate_limits": True,
        # Performance settings
        "broker_connection_retry_on_startup": True,
        "broker_connection_retry": True,
        # Monitoring
        "worker_send_task_events": True,
        "task_send_sent_event": True,
    }
