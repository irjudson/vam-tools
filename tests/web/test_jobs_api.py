"""Tests for jobs API endpoints."""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from lumina.web.api import app

client = TestClient(app)


class TestJobSubmissionEndpoints:
    """Tests for job submission endpoints."""

    @patch("lumina.web.jobs_api.analyze_catalog_task")
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

    @patch("lumina.web.jobs_api.organize_catalog_task")
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

    @patch("lumina.web.jobs_api.generate_thumbnails_task")
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

    @patch("lumina.web.jobs_api.AsyncResult")
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

    @patch("lumina.web.jobs_api.AsyncResult")
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

    @patch("lumina.web.jobs_api.AsyncResult")
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

    @patch("lumina.web.jobs_api.AsyncResult")
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

    @patch("lumina.web.jobs_api.get_redis_client")
    @patch("lumina.web.jobs_api.AsyncResult")
    def test_list_active_jobs(self, mock_async_result, mock_get_redis_client):
        """Test GET /api/jobs lists active jobs."""
        mock_redis = Mock()
        mock_get_redis_client.return_value = mock_redis
        mock_redis.lrange.return_value = ["job-1", "job-2"]

        mock_result_1 = Mock()
        mock_result_1.state = "PROGRESS"
        mock_result_1.info = {"percent": 50}
        mock_result_2 = Mock()
        mock_result_2.state = "SUCCESS"
        mock_result_2.result = {"status": "completed"}
        mock_async_result.side_effect = [mock_result_1, mock_result_2]

        response = client.get("/api/jobs")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert len(data["jobs"]) == 2
        assert data["jobs"][0]["job_id"] == "job-1"
        assert data["jobs"][0]["status"] == "PROGRESS"
        assert data["jobs"][1]["job_id"] == "job-2"
        assert data["jobs"][1]["status"] == "SUCCESS"


class TestJobCancellation:
    """Tests for job cancellation."""

    @patch("lumina.web.jobs_api.AsyncResult")
    def test_cancel_job(self, mock_async_result):
        """Test DELETE /api/jobs/{job_id} cancels job."""
        mock_result = Mock()
        mock_result.state = "PROGRESS"
        mock_async_result.return_value = mock_result

        response = client.delete("/api/jobs/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert "cancelled" in data["message"].lower()
        mock_result.revoke.assert_called_once_with(terminate=True)

    @patch("lumina.web.jobs_api.AsyncResult")
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

    @patch("lumina.web.jobs_api.AsyncResult")
    def test_stream_job_progress(self, mock_async_result):
        """Test GET /api/jobs/{job_id}/stream streams progress updates."""
        # This is challenging to test with TestClient
        # Would need async testing framework
        pass


class TestJobRerunEndpoint:
    """Tests for job rerun endpoint."""

    @patch("lumina.web.jobs_api.get_redis_client")
    @patch("lumina.web.jobs_api.analyze_catalog_task")
    def test_rerun_analyze_job(self, mock_analyze_task, mock_get_redis_client):
        """Test POST /api/jobs/{job_id}/rerun for analyze job."""
        mock_redis = Mock()
        mock_get_redis_client.return_value = mock_redis
        mock_redis.get.return_value = '{"job_id": "old-analyze-job", "type": "analyze_catalog", "params": {"catalog_path": "/app/test", "source_directories": ["/app/photos"]}}'

        mock_result = Mock()
        mock_result.id = "new-analyze-job"
        mock_analyze_task.delay.return_value = mock_result

        response = client.post("/api/jobs/old-analyze-job/rerun")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "new-analyze-job"
        mock_analyze_task.delay.assert_called_once_with(
            catalog_path="/app/test", source_directories=["/app/photos"]
        )

    @patch("lumina.web.jobs_api.get_redis_client")
    def test_rerun_job_not_found(self, mock_get_redis_client):
        """Test rerun job when original job parameters are not found."""
        mock_redis = Mock()
        mock_get_redis_client.return_value = mock_redis
        mock_redis.get.return_value = None

        response = client.post("/api/jobs/nonexistent-job/rerun")

        assert response.status_code == 404
        assert "Job parameters not found" in response.json()["detail"]


class TestJobKillEndpoint:
    """Tests for job kill endpoint."""

    @patch("lumina.web.jobs_api.AsyncResult")
    def test_kill_job(self, mock_async_result):
        """Test POST /api/jobs/{job_id}/kill kills job."""
        mock_result = Mock()
        mock_async_result.return_value = mock_result

        response = client.post("/api/jobs/test-job-to-kill/kill")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "killed"
        mock_result.revoke.assert_called_once_with(terminate=True, signal="SIGKILL")
