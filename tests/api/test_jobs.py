"""Tests for jobs API router endpoints.

All tests require database connection.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from vam_tools.api.app import create_app
from vam_tools.api.routers.jobs import _safe_get_task_info, _safe_get_task_state

pytestmark = pytest.mark.integration


class TestSafeTaskAccessors:
    """Tests for safe Celery task accessor functions."""

    def test_safe_get_task_state_normal(self):
        """Test _safe_get_task_state with normal task."""
        mock_task = Mock()
        mock_task.state = "SUCCESS"

        result = _safe_get_task_state(mock_task)
        assert result == "SUCCESS"

    def test_safe_get_task_state_with_value_error(self):
        """Test _safe_get_task_state when task.state raises ValueError.

        This simulates the error that occurs when the Celery result backend
        has malformed exception info (missing 'exc_type' key).
        """
        mock_task = Mock()
        mock_task.id = "test-task-123"
        # Simulate the ValueError that Celery raises
        type(mock_task).state = property(
            lambda self: (_ for _ in ()).throw(
                ValueError("Exception information must include the exception type")
            )
        )

        result = _safe_get_task_state(mock_task)
        assert result == "FAILURE"

    def test_safe_get_task_info_normal(self):
        """Test _safe_get_task_info with normal task."""
        mock_task = Mock()
        mock_task.info = {"current": 50, "total": 100}

        result = _safe_get_task_info(mock_task)
        assert result == {"current": 50, "total": 100}

    def test_safe_get_task_info_with_value_error(self):
        """Test _safe_get_task_info when task.info raises ValueError."""
        mock_task = Mock()
        mock_task.id = "test-task-123"
        type(mock_task).info = property(
            lambda self: (_ for _ in ()).throw(
                ValueError("Exception information must include the exception type")
            )
        )

        result = _safe_get_task_info(mock_task)
        assert "error" in result
        assert "Failed to retrieve task info" in result["error"]


class TestJobStatusEndpoint:
    """Tests for GET /api/jobs/{job_id} endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI application."""
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client

    @patch("vam_tools.api.routers.jobs.AsyncResult")
    def test_get_job_status_success(self, mock_async_result, client):
        """Test get_job_status with a successful task."""
        mock_task = Mock()
        mock_task.state = "SUCCESS"
        mock_task.result = {"files_processed": 100}
        mock_async_result.return_value = mock_task

        response = client.get("/api/jobs/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "SUCCESS"
        assert data["result"]["files_processed"] == 100

    @patch("vam_tools.api.routers.jobs.AsyncResult")
    def test_get_job_status_progress(self, mock_async_result, client):
        """Test get_job_status with an in-progress task."""
        mock_task = Mock()
        mock_task.state = "PROGRESS"
        mock_task.info = {"current": 50, "total": 100}
        mock_async_result.return_value = mock_task

        response = client.get("/api/jobs/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PROGRESS"
        assert data["progress"]["current"] == 50

    @patch("vam_tools.api.routers.jobs.AsyncResult")
    def test_get_job_status_failure(self, mock_async_result, client):
        """Test get_job_status with a failed task."""
        mock_task = Mock()
        mock_task.state = "FAILURE"
        mock_task.info = Exception("Task failed due to error")
        mock_async_result.return_value = mock_task

        response = client.get("/api/jobs/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FAILURE"
        assert "Task failed" in data["result"]["error"]

    @patch("vam_tools.api.routers.jobs.AsyncResult")
    def test_get_job_status_malformed_exception_info(self, mock_async_result, client):
        """Test get_job_status handles malformed Celery exception info gracefully.

        This is a regression test for the ValueError that occurs when the Celery
        result backend stores exception info without the required 'exc_type' key.
        The endpoint should return a FAILURE status instead of a 500 error.
        """
        mock_task = Mock()
        mock_task.id = "test-job-123"

        # Simulate the error: accessing .state raises ValueError
        # because the exception info in the result backend is malformed
        type(mock_task).state = property(
            lambda self: (_ for _ in ()).throw(
                ValueError("Exception information must include the exception type")
            )
        )
        # Also make .info raise the same error
        type(mock_task).info = property(
            lambda self: (_ for _ in ()).throw(
                ValueError("Exception information must include the exception type")
            )
        )
        mock_async_result.return_value = mock_task

        response = client.get("/api/jobs/test-job-123")

        # Should NOT be a 500 error - should be handled gracefully
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "FAILURE"
        assert "error" in data["result"]
