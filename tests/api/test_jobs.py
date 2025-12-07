"""Tests for jobs API router endpoints.

All tests require database connection.
"""

import uuid
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from vam_tools.api.app import create_app
from vam_tools.api.routers.jobs import _safe_get_task_info, _safe_get_task_state
from vam_tools.db import get_db
from vam_tools.db.models import Job

pytestmark = pytest.mark.integration


def unique_job_id(prefix: str = "test-job") -> str:
    """Generate a unique job ID for test isolation."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


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

    @pytest.fixture
    def db_session(self, client):
        """Get a database session for creating test jobs."""
        # Use the same dependency override mechanism as the app
        db_gen = get_db()
        db = next(db_gen)
        try:
            yield db
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass

    def _create_test_job(self, db_session, job_id: str, status: str = "PENDING"):
        """Create a test job in the database."""
        job = Job(
            id=job_id,
            job_type="scan",
            status=status,
            parameters={},
        )
        db_session.add(job)
        db_session.commit()
        return job

    @patch("vam_tools.api.routers.jobs.AsyncResult")
    def test_get_job_status_success(self, mock_async_result, client, db_session):
        """Test get_job_status with a successful task."""
        # Create job in database first with unique ID for parallel test isolation
        job_id = unique_job_id("success")
        self._create_test_job(db_session, job_id, "PENDING")

        mock_task = Mock()
        mock_task.state = "SUCCESS"
        mock_task.result = {"files_processed": 100}
        mock_async_result.return_value = mock_task

        response = client.get(f"/api/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "SUCCESS"
        assert data["result"]["files_processed"] == 100

    @patch("vam_tools.api.routers.jobs.AsyncResult")
    def test_get_job_status_progress(self, mock_async_result, client, db_session):
        """Test get_job_status with an in-progress task."""
        # Create job in database first with unique ID for parallel test isolation
        job_id = unique_job_id("progress")
        self._create_test_job(db_session, job_id, "PENDING")

        mock_task = Mock()
        mock_task.state = "PROGRESS"
        mock_task.info = {"current": 50, "total": 100}
        mock_async_result.return_value = mock_task

        response = client.get(f"/api/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PROGRESS"
        assert data["progress"]["current"] == 50

    @patch("vam_tools.api.routers.jobs.AsyncResult")
    def test_get_job_status_failure(self, mock_async_result, client, db_session):
        """Test get_job_status with a failed task."""
        # Create job in database first with unique ID for parallel test isolation
        job_id = unique_job_id("failure")
        self._create_test_job(db_session, job_id, "PENDING")

        mock_task = Mock()
        mock_task.state = "FAILURE"
        mock_task.info = Exception("Task failed due to error")
        mock_async_result.return_value = mock_task

        response = client.get(f"/api/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FAILURE"
        assert "Task failed" in data["result"]["error"]

    @patch("vam_tools.api.routers.jobs.AsyncResult")
    def test_get_job_status_malformed_exception_info(
        self, mock_async_result, client, db_session
    ):
        """Test get_job_status handles malformed Celery exception info gracefully.

        This is a regression test for the ValueError that occurs when the Celery
        result backend stores exception info without the required 'exc_type' key.
        The endpoint should return a FAILURE status instead of a 500 error.
        """
        # Create job in database first with unique ID for parallel test isolation
        job_id = unique_job_id("malformed")
        self._create_test_job(db_session, job_id, "PENDING")

        mock_task = Mock()
        mock_task.id = job_id

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

        response = client.get(f"/api/jobs/{job_id}")

        # Should NOT be a 500 error - should be handled gracefully
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "FAILURE"
        assert "error" in data["result"]
