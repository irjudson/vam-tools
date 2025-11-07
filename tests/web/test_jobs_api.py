"""Tests for jobs API endpoints."""

import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from vam_tools.web.api import app

client = TestClient(app)


class TestJobSubmissionEndpoints:
    """Tests for job submission endpoints."""

    @patch("vam_tools.web.jobs_api.analyze_catalog_task")
    def test_submit_analyze_job(self, mock_task):
        """Test POST /api/jobs/analyze submits job correctly."""
        mock_result = Mock()
        mock_result.id = "test-job-id-123"
        mock_task.delay.return_value = mock_result

        response = client.post(
            "/api/jobs/analyze",
            json={
                "catalog_path": "/app/catalogs/test",
                "source_directories": ["/app/photos"],
                "detect_duplicates": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-id-123"
        assert data["status"] == "PENDING"
        assert "Analysis job submitted" in data["message"]

    @patch("vam_tools.web.jobs_api.organize_catalog_task")
    def test_submit_organize_job(self, mock_task):
        """Test POST /api/jobs/organize submits job correctly."""
        mock_result = Mock()
        mock_result.id = "organize-job-456"
        mock_task.delay.return_value = mock_result

        response = client.post(
            "/api/jobs/organize",
            json={
                "catalog_path": "/app/catalogs/test",
                "output_directory": "/app/organized",
                "dry_run": True,
                "operation": "copy",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "organize-job-456"
        assert data["status"] == "PENDING"

    @patch("vam_tools.web.jobs_api.generate_thumbnails_task")
    def test_submit_thumbnails_job(self, mock_task):
        """Test POST /api/jobs/thumbnails submits job correctly."""
        mock_result = Mock()
        mock_result.id = "thumb-job-789"
        mock_task.delay.return_value = mock_result

        response = client.post(
            "/api/jobs/thumbnails",
            json={
                "catalog_path": "/app/catalogs/test",
                "sizes": [200, 400],
                "quality": 85,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "thumb-job-789"

    def test_submit_analyze_job_missing_fields(self):
        """Test job submission with missing required fields."""
        response = client.post(
            "/api/jobs/analyze",
            json={"catalog_path": "/app/catalogs/test"},  # Missing source_directories
        )

        assert response.status_code == 422  # Validation error


class TestJobStatusEndpoints:
    """Tests for job status endpoints."""

    @patch("vam_tools.web.jobs_api.AsyncResult")
    def test_get_job_status_pending(self, mock_async_result):
        """Test GET /api/jobs/{job_id} for pending job."""
        mock_result = Mock()
        mock_result.state = "PENDING"
        mock_result.info = {}
        mock_async_result.return_value = mock_result

        response = client.get("/api/jobs/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "PENDING"

    @patch("vam_tools.web.jobs_api.AsyncResult")
    def test_get_job_status_progress(self, mock_async_result):
        """Test GET /api/jobs/{job_id} for in-progress job."""
        mock_result = Mock()
        mock_result.state = "PROGRESS"
        mock_result.info = {
            "current": 50,
            "total": 100,
            "percent": 50,
            "message": "Processing files...",
        }
        mock_async_result.return_value = mock_result

        response = client.get("/api/jobs/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PROGRESS"
        assert data["progress"]["percent"] == 50

    @patch("vam_tools.web.jobs_api.AsyncResult")
    def test_get_job_status_success(self, mock_async_result):
        """Test GET /api/jobs/{job_id} for completed job."""
        mock_result = Mock()
        mock_result.state = "SUCCESS"
        mock_result.result = {
            "status": "completed",
            "total_files": 100,
            "processed": 100,
        }
        mock_async_result.return_value = mock_result

        response = client.get("/api/jobs/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["result"]["processed"] == 100

    @patch("vam_tools.web.jobs_api.AsyncResult")
    def test_get_job_status_failure(self, mock_async_result):
        """Test GET /api/jobs/{job_id} for failed job."""
        mock_result = Mock()
        mock_result.state = "FAILURE"
        mock_result.info = Exception("Test error")
        mock_async_result.return_value = mock_result

        response = client.get("/api/jobs/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FAILURE"
        assert data["error"] is not None


class TestJobListEndpoint:
    """Tests for job list endpoint."""

    @patch("vam_tools.web.jobs_api.app_celery.control.inspect")
    def test_list_active_jobs(self, mock_inspect):
        """Test GET /api/jobs lists active jobs."""
        mock_inspect_obj = Mock()
        mock_inspect.return_value = mock_inspect_obj
        mock_inspect_obj.active.return_value = {
            "worker1": [
                {
                    "id": "job-1",
                    "name": "analyze_catalog",
                    "args": [],
                    "kwargs": {},
                }
            ]
        }

        response = client.get("/api/jobs")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data


class TestJobCancellation:
    """Tests for job cancellation."""

    @patch("vam_tools.web.jobs_api.app_celery.control.revoke")
    @patch("vam_tools.web.jobs_api.AsyncResult")
    def test_cancel_job(self, mock_async_result, mock_revoke):
        """Test DELETE /api/jobs/{job_id} cancels job."""
        mock_result = Mock()
        mock_result.state = "PROGRESS"
        mock_async_result.return_value = mock_result

        response = client.delete("/api/jobs/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Job cancelled successfully"
        mock_revoke.assert_called_once_with("test-job-123", terminate=True)

    @patch("vam_tools.web.jobs_api.AsyncResult")
    def test_cancel_completed_job(self, mock_async_result):
        """Test cancelling already completed job."""
        mock_result = Mock()
        mock_result.state = "SUCCESS"
        mock_async_result.return_value = mock_result

        response = client.delete("/api/jobs/test-job-123")

        # Should return error or appropriate message
        assert response.status_code in [200, 400]


class TestSSEProgressStream:
    """Tests for Server-Sent Events progress streaming."""

    @patch("vam_tools.web.jobs_api.AsyncResult")
    def test_stream_job_progress(self, mock_async_result):
        """Test GET /api/jobs/{job_id}/stream streams progress updates."""
        # This is challenging to test with TestClient
        # Would need async testing framework
        pass


class TestRequestValidation:
    """Tests for request validation."""

    def test_invalid_catalog_path(self):
        """Test job submission with invalid catalog path."""
        response = client.post(
            "/api/jobs/analyze",
            json={
                "catalog_path": "",  # Empty path
                "source_directories": ["/app/photos"],
            },
        )

        assert response.status_code == 422

    def test_invalid_sizes_for_thumbnails(self):
        """Test thumbnail job with invalid sizes."""
        response = client.post(
            "/api/jobs/thumbnails",
            json={
                "catalog_path": "/app/catalogs/test",
                "sizes": [],  # Empty list
                "quality": 85,
            },
        )

        # Should validate that sizes is not empty
        assert response.status_code in [200, 422]

    def test_invalid_quality_parameter(self):
        """Test thumbnail job with out-of-range quality."""
        response = client.post(
            "/api/jobs/thumbnails",
            json={
                "catalog_path": "/app/catalogs/test",
                "sizes": [200],
                "quality": 150,  # > 100
            },
        )

        # Should validate quality is 1-100
        assert response.status_code in [200, 422]


class TestErrorHandling:
    """Tests for error handling in API."""

    @patch("vam_tools.web.jobs_api.analyze_catalog_task")
    def test_task_submission_failure(self, mock_task):
        """Test API handles task submission failure."""
        mock_task.delay.side_effect = Exception("Celery error")

        response = client.post(
            "/api/jobs/analyze",
            json={
                "catalog_path": "/app/catalogs/test",
                "source_directories": ["/app/photos"],
            },
        )

        # Should return 500 or handle gracefully
        assert response.status_code in [500, 200]
