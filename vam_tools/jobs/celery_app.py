"""
Celery application instance and configuration.
"""

import logging

from celery import Celery
from celery.signals import after_setup_logger

from .config import get_celery_config

logger = logging.getLogger(__name__)

# Create Celery app
app = Celery("vam_tools")

# Load configuration
config = get_celery_config()
app.config_from_object(config)

# Auto-discover tasks in the jobs module
app.autodiscover_tasks(["vam_tools.jobs"])


@after_setup_logger.connect
def setup_celery_logging(logger: logging.Logger, **kwargs) -> None:  # type: ignore
    """Configure Celery logging to match application logging."""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Update all handlers
    for handler in logger.handlers:
        handler.setFormatter(formatter)


if __name__ == "__main__":
    app.start()
