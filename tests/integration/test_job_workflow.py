"""Integration tests for complete job workflows.

All tests require running services (Redis, Celery, FastAPI).
"""

import time

import pytest

# Skip collection if requests not installed (not needed for unit tests)
requests = pytest.importorskip("requests")

pytestmark = pytest.mark.integration


class TestJobWorkflowIntegration:
    """End-to-end tests for job workflows."""

    BASE_URL = "http://localhost:8765"

    def test_analyze_job_end_to_end(self, tmp_path):
        """Test complete analysis workflow from submission to completion."""
        # Submit job
        response = requests.post(
            f"{self.BASE_URL}/api/jobs/analyze",
            json={
                "catalog_path": "/app/catalogs/test",
                "source_directories": ["/app/photos"],
                "detect_duplicates": False,
            },
        )

        assert response.status_code == 200
        job_data = response.json()
        job_id = job_data["job_id"]
        assert job_data["status"] == "PENDING"

        # Poll for completion
        max_wait = 30  # seconds
        start_time = time.time()
        final_status = None

        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{self.BASE_URL}/api/jobs/{job_id}")
            assert status_response.status_code == 200

            status_data = status_response.json()
            final_status = status_data["status"]

            if final_status in ["SUCCESS", "FAILURE"]:
                break

            time.sleep(0.5)

        # Verify completion
        assert final_status == "SUCCESS"
        assert status_data["result"] is not None
        assert "total_files" in status_data["result"]

    def test_thumbnail_job_workflow(self):
        """Test thumbnail generation workflow."""
        response = requests.post(
            f"{self.BASE_URL}/api/jobs/thumbnails",
            json={
                "catalog_path": "/app/catalogs/test",
                "sizes": [200, 400],
                "quality": 85,
            },
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Wait for completion
        time.sleep(2)

        status_response = requests.get(f"{self.BASE_URL}/api/jobs/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] in ["SUCCESS", "PROGRESS"]

    def test_organize_dry_run_workflow(self):
        """Test organization dry-run workflow."""
        response = requests.post(
            f"{self.BASE_URL}/api/jobs/organize",
            json={
                "catalog_path": "/app/catalogs/test",
                "output_directory": "/app/organized",
                "dry_run": True,
                "operation": "copy",
            },
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Wait for completion
        time.sleep(2)

        status_response = requests.get(f"{self.BASE_URL}/api/jobs/{job_id}")
        assert status_response.status_code == 200

        status_data = status_response.json()
        if status_data["status"] == "SUCCESS":
            assert status_data["result"]["dry_run"] is True

    def test_concurrent_jobs(self):
        """Test multiple jobs running concurrently."""
        job_ids = []

        # Submit multiple jobs
        for i in range(3):
            response = requests.post(
                f"{self.BASE_URL}/api/jobs/analyze",
                json={
                    "catalog_path": f"/app/catalogs/test{i}",
                    "source_directories": ["/app/photos"],
                    "detect_duplicates": False,
                },
            )
            assert response.status_code == 200
            job_ids.append(response.json()["job_id"])

        # Wait for all to complete
        time.sleep(5)

        # Check all jobs
        for job_id in job_ids:
            response = requests.get(f"{self.BASE_URL}/api/jobs/{job_id}")
            assert response.status_code == 200
            # Should eventually complete
            assert response.json()["status"] in ["SUCCESS", "FAILURE", "PROGRESS"]

    def test_job_cancellation(self):
        """Test job can be cancelled."""
        # Submit long-running job
        response = requests.post(
            f"{self.BASE_URL}/api/jobs/analyze",
            json={
                "catalog_path": "/app/catalogs/test",
                "source_directories": ["/app/photos"],
                "detect_duplicates": True,  # Makes it slower
            },
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Immediately try to cancel
        cancel_response = requests.delete(f"{self.BASE_URL}/api/jobs/{job_id}")
        assert cancel_response.status_code in [200, 404]

    def test_job_error_handling(self):
        """Test job failure is handled correctly."""
        # Submit job with invalid path
        response = requests.post(
            f"{self.BASE_URL}/api/jobs/analyze",
            json={
                "catalog_path": "/nonexistent/path",
                "source_directories": ["/nonexistent/photos"],
                "detect_duplicates": False,
            },
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Wait for processing
        time.sleep(3)

        status_response = requests.get(f"{self.BASE_URL}/api/jobs/{job_id}")
        assert status_response.status_code == 200

        status_data = status_response.json()
        # Should either fail or handle gracefully
        assert status_data["status"] in ["SUCCESS", "FAILURE"]

    def test_web_ui_accessibility(self):
        """Test web UI is accessible."""
        response = requests.get(f"{self.BASE_URL}/static/jobs.html")
        assert response.status_code == 200
        assert "VAM Tools" in response.text
        assert "Job Management" in response.text


@pytest.mark.integration
class TestServiceHealth:
    """Tests for service health checks."""

    BASE_URL = "http://localhost:8765"

    def test_api_health(self):
        """Test API is responding."""
        response = requests.get(f"{self.BASE_URL}/api")
        assert response.status_code == 200

    def test_redis_connection(self):
        """Test Redis is accessible through job submission."""
        # If job can be submitted, Redis is working
        response = requests.post(
            f"{self.BASE_URL}/api/jobs/analyze",
            json={
                "catalog_path": "/app/catalogs/test",
                "source_directories": ["/app/photos"],
            },
        )
        assert response.status_code == 200

    def test_celery_worker_active(self):
        """Test Celery worker is processing jobs."""
        # Submit simple job
        response = requests.post(
            f"{self.BASE_URL}/api/jobs/analyze",
            json={
                "catalog_path": "/app/catalogs/test",
                "source_directories": ["/app/photos"],
            },
        )
        job_id = response.json()["job_id"]

        # Wait and check it was processed
        time.sleep(3)

        status_response = requests.get(f"{self.BASE_URL}/api/jobs/{job_id}")
        status = status_response.json()["status"]

        # If status changed from PENDING, worker is active
        assert status in ["SUCCESS", "FAILURE", "PROGRESS"]
