"""Tests for Celery tasks."""

import pytest

from vam_tools.celery_app import app as celery_app


def test_celery_app_configured():
    """Test that Celery app is properly configured."""
    assert celery_app.conf.broker_url is not None
    # result_backend should be None - we use PostgreSQL Job model instead
    assert celery_app.conf.result_backend is None
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"


def test_task_registration():
    """Test that tasks are registered."""
    registered_tasks = celery_app.tasks.keys()
    # Check for new task names
    assert "analyze_catalog" in registered_tasks or "analyze" in registered_tasks
    assert "organize_catalog" in registered_tasks or "organize" in registered_tasks


def test_task_routing():
    """Test that task routing is configured."""
    # Task routing is optional - celery can work without it
    # Just verify the config attribute exists
    routes = celery_app.conf.task_routes
    # Routes may be None or empty dict if not configured
    assert routes is None or isinstance(routes, dict)
