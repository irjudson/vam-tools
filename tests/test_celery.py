"""Tests for Celery tasks."""

import pytest

from vam_tools.celery_app import app as celery_app
from vam_tools.tasks.scan import scan_directories


def test_celery_app_configured():
    """Test that Celery app is properly configured."""
    assert celery_app.conf.broker_url is not None
    assert celery_app.conf.result_backend is not None
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"


def test_task_registration():
    """Test that tasks are registered."""
    registered_tasks = celery_app.tasks.keys()
    assert "vam_tools.tasks.scan.scan_directories" in registered_tasks
    assert "vam_tools.tasks.duplicates.detect_duplicates" in registered_tasks
    assert "vam_tools.tasks.organize.execute_plan" in registered_tasks


def test_scan_task_signature():
    """Test scan task can be called."""
    # We can't actually run it without a worker, but we can test the signature
    task = scan_directories.s("test-catalog-id", ["/tmp/test"])
    assert task is not None
    assert task.name == "vam_tools.tasks.scan.scan_directories"


def test_task_routing():
    """Test that task routing is configured."""
    routes = celery_app.conf.task_routes
    assert routes is not None
    assert "vam_tools.tasks.scan.*" in routes
    assert routes["vam_tools.tasks.scan.*"]["queue"] == "scanner"
