"""Base task class with progress tracking."""

import logging
from typing import Any, Dict, Optional

from celery import Task

logger = logging.getLogger(__name__)


class ProgressTrackingTask(Task):
    """Base task class that supports progress tracking and database updates."""

    def update_progress(
        self,
        current: int,
        total: int,
        message: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update task progress.

        Args:
            current: Current progress count
            total: Total items to process
            message: Progress message
            extra: Additional progress data
        """
        percent = int((current / total * 100)) if total > 0 else 0

        progress = {
            "current": current,
            "total": total,
            "percent": percent,
            "message": message,
        }

        if extra:
            progress.update(extra)

        # Update Celery task state
        self.update_state(
            state="PROGRESS",
            meta=progress,
        )

        # Log progress at milestones
        if percent % 10 == 0 and current > 0:
            logger.info(f"{self.name}: {percent}% ({current}/{total}) - {message}")

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task completes successfully."""
        logger.info(f"Task {task_id} completed: {retval}")
        return super().on_success(retval, task_id, args, kwargs)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"Task {task_id} failed: {exc}")
        logger.error(f"Traceback: {einfo}")
        return super().on_failure(exc, task_id, args, kwargs, einfo)
