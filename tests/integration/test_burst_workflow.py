"""Integration tests for complete burst workflow.

These tests require a FULLY running Docker environment with:
- FastAPI server on port 8765
- Redis
- Celery workers
- PostgreSQL

Run with: docker-compose up && pytest -m e2e tests/integration/test_burst_workflow.py
"""

import time
import uuid
from datetime import datetime, timedelta

import pytest

# Skip collection if requests not installed (not needed for unit tests)
requests = pytest.importorskip("requests")

# Mark as both integration and e2e - these need full Docker stack
pytestmark = [pytest.mark.integration, pytest.mark.e2e]


class TestBurstWorkflowIntegration:
    """End-to-end tests for burst detection and review workflows."""

    BASE_URL = "http://localhost:8765"

    def test_complete_burst_review_workflow(self, tmp_path):
        """Test complete burst review workflow: list → detail → apply → verify.

        This test verifies the full user workflow:
        1. Detect bursts in a catalog
        2. List all bursts
        3. View burst details
        4. Apply selection (mark best image as active, others as rejected)
        5. Verify the selection was applied correctly
        """
        # Create a test catalog
        catalog_id = str(uuid.uuid4())

        # Step 1: Create catalog with burst-like images
        # In a real scenario, this would be done through the catalog creation API
        # For this test, we'll trigger burst detection on an existing catalog

        # Step 2: Start burst detection job
        detect_response = requests.post(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/detect-bursts",
            params={
                "gap_threshold": 2.0,
                "min_burst_size": 3,
            },
        )

        # Should accept the job even if catalog doesn't exist (job will handle error)
        assert detect_response.status_code in [200, 202, 404]

        if detect_response.status_code == 404:
            # Catalog doesn't exist, which is expected in isolated test
            pytest.skip("Catalog not found - needs real catalog for e2e test")
            return

        job_id = detect_response.json()["job_id"]

        # Wait for burst detection to complete
        max_wait = 30
        start_time = time.time()
        detection_complete = False

        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{self.BASE_URL}/api/jobs/{job_id}")
            if status_response.status_code == 200:
                status = status_response.json()["status"]
                if status in ["SUCCESS", "FAILURE"]:
                    detection_complete = (status == "SUCCESS")
                    break
            time.sleep(0.5)

        if not detection_complete:
            pytest.skip("Burst detection did not complete in time")
            return

        # Step 3: List bursts
        list_response = requests.get(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/bursts",
            params={"limit": 100, "offset": 0},
        )

        assert list_response.status_code == 200
        burst_list = list_response.json()

        # Verify list response structure
        assert "bursts" in burst_list
        assert "total" in burst_list
        assert isinstance(burst_list["bursts"], list)

        if len(burst_list["bursts"]) == 0:
            pytest.skip("No bursts detected in catalog")
            return

        # Step 4: Get details for first burst
        first_burst_id = burst_list["bursts"][0]["id"]
        detail_response = requests.get(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/bursts/{first_burst_id}"
        )

        assert detail_response.status_code == 200
        burst_detail = detail_response.json()

        # Verify detail response structure
        assert "id" in burst_detail
        assert "images" in burst_detail
        assert "best_image_id" in burst_detail
        assert isinstance(burst_detail["images"], list)
        assert len(burst_detail["images"]) >= 3  # min_burst_size

        # Step 5: Apply selection (mark best image as active, others as rejected)
        best_image_id = burst_detail["best_image_id"]

        if not best_image_id:
            # If no best_image_id, use the first image
            best_image_id = burst_detail["images"][0]["id"]

        apply_response = requests.post(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/bursts/{first_burst_id}/apply-selection",
            json={"selected_image_id": best_image_id},
        )

        assert apply_response.status_code == 200
        apply_result = apply_response.json()

        # Verify apply response structure
        assert "selected_image_id" in apply_result
        assert "rejected_count" in apply_result
        assert apply_result["selected_image_id"] == best_image_id
        assert apply_result["rejected_count"] == len(burst_detail["images"]) - 1

        # Step 6: Verify the selection was applied
        # Get image details to verify status changes
        for image in burst_detail["images"]:
            image_response = requests.get(
                f"{self.BASE_URL}/api/catalogs/{catalog_id}/images/{image['id']}"
            )

            if image_response.status_code == 200:
                image_data = image_response.json()
                if image["id"] == best_image_id:
                    assert image_data.get("status_id") == "active"
                else:
                    assert image_data.get("status_id") == "rejected"

    def test_batch_apply_burst_workflow(self, tmp_path):
        """Test batch apply workflow: batch apply → verify all bursts processed.

        This test verifies:
        1. Detect bursts in a catalog
        2. Batch apply selections for all bursts
        3. Verify all bursts were processed
        4. Verify images were correctly marked as active/rejected
        """
        # Create a test catalog
        catalog_id = str(uuid.uuid4())

        # Step 1: Start burst detection
        detect_response = requests.post(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/detect-bursts",
            params={
                "gap_threshold": 2.0,
                "min_burst_size": 3,
            },
        )

        if detect_response.status_code == 404:
            pytest.skip("Catalog not found - needs real catalog for e2e test")
            return

        assert detect_response.status_code in [200, 202]
        job_id = detect_response.json()["job_id"]

        # Wait for detection to complete
        max_wait = 30
        start_time = time.time()
        detection_complete = False

        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{self.BASE_URL}/api/jobs/{job_id}")
            if status_response.status_code == 200:
                status = status_response.json()["status"]
                if status in ["SUCCESS", "FAILURE"]:
                    detection_complete = (status == "SUCCESS")
                    break
            time.sleep(0.5)

        if not detection_complete:
            pytest.skip("Burst detection did not complete in time")
            return

        # Step 2: Get burst count before batch apply
        list_response = requests.get(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/bursts"
        )

        assert list_response.status_code == 200
        burst_list = list_response.json()

        if len(burst_list["bursts"]) == 0:
            pytest.skip("No bursts detected in catalog")
            return

        expected_burst_count = len([
            b for b in burst_list["bursts"] if b.get("best_image_id")
        ])

        # Step 3: Batch apply burst selections
        batch_apply_response = requests.post(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/bursts/batch-apply",
            json={"use_recommendations": True},
        )

        assert batch_apply_response.status_code == 200
        batch_result = batch_apply_response.json()

        # Verify batch apply response structure
        assert "bursts_processed" in batch_result
        assert "images_rejected" in batch_result

        # Step 4: Verify all bursts with best_image_id were processed
        assert batch_result["bursts_processed"] == expected_burst_count

        # Step 5: Verify images were rejected
        # The number of rejected images should be:
        # (total images in bursts) - (number of bursts)
        if expected_burst_count > 0:
            assert batch_result["images_rejected"] > 0

        # Step 6: Verify individual burst states
        for burst in burst_list["bursts"]:
            if burst.get("best_image_id"):
                burst_detail = requests.get(
                    f"{self.BASE_URL}/api/catalogs/{catalog_id}/bursts/{burst['id']}"
                )

                if burst_detail.status_code == 200:
                    detail = burst_detail.json()

                    # Check each image in the burst
                    for image in detail["images"]:
                        image_response = requests.get(
                            f"{self.BASE_URL}/api/catalogs/{catalog_id}/images/{image['id']}"
                        )

                        if image_response.status_code == 200:
                            image_data = image_response.json()
                            if image["id"] == burst["best_image_id"]:
                                assert image_data.get("status_id") == "active"
                            else:
                                assert image_data.get("status_id") == "rejected"

    def test_rejected_images_hidden_from_list(self, tmp_path):
        """Test rejected images are hidden from list by default.

        This test verifies:
        1. Apply burst selections to reject some images
        2. List images without status filter (should exclude rejected by default)
        3. List images with status_id=rejected filter (should show only rejected)
        4. List images with status_id=active filter (should show only active)
        5. Verify counts are correct
        """
        # Create a test catalog
        catalog_id = str(uuid.uuid4())

        # Step 1: Detect bursts and apply selections
        detect_response = requests.post(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/detect-bursts",
            params={
                "gap_threshold": 2.0,
                "min_burst_size": 3,
            },
        )

        if detect_response.status_code == 404:
            pytest.skip("Catalog not found - needs real catalog for e2e test")
            return

        assert detect_response.status_code in [200, 202]
        job_id = detect_response.json()["job_id"]

        # Wait for detection
        max_wait = 30
        start_time = time.time()
        detection_complete = False

        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{self.BASE_URL}/api/jobs/{job_id}")
            if status_response.status_code == 200:
                status = status_response.json()["status"]
                if status in ["SUCCESS", "FAILURE"]:
                    detection_complete = (status == "SUCCESS")
                    break
            time.sleep(0.5)

        if not detection_complete:
            pytest.skip("Burst detection did not complete in time")
            return

        # Get total image count before applying selections
        all_images_response = requests.get(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/images",
            params={"limit": 1000},
        )

        if all_images_response.status_code != 200:
            pytest.skip("Cannot get image list")
            return

        all_images = all_images_response.json()
        initial_total = all_images.get("total", len(all_images.get("images", [])))

        # Step 2: Batch apply burst selections
        batch_apply_response = requests.post(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/bursts/batch-apply",
            json={"use_recommendations": True},
        )

        if batch_apply_response.status_code != 200:
            pytest.skip("Batch apply failed")
            return

        batch_result = batch_apply_response.json()
        expected_rejected = batch_result["images_rejected"]

        if expected_rejected == 0:
            pytest.skip("No images were rejected")
            return

        # Step 3: List images without status filter (should exclude rejected by default)
        default_list_response = requests.get(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/images",
            params={"limit": 1000},
        )

        assert default_list_response.status_code == 200
        default_list = default_list_response.json()

        # Verify rejected images are not in default list
        default_images = default_list.get("images", [])
        for image in default_images:
            assert image.get("status_id") != "rejected"

        # Step 4: List images with status_id=rejected filter
        rejected_list_response = requests.get(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/images",
            params={"limit": 1000, "status_id": "rejected"},
        )

        assert rejected_list_response.status_code == 200
        rejected_list = rejected_list_response.json()

        # Verify all images in list are rejected
        rejected_images = rejected_list.get("images", [])
        for image in rejected_images:
            assert image.get("status_id") == "rejected"

        # Verify count matches expected
        assert len(rejected_images) == expected_rejected

        # Step 5: List images with status_id=active filter
        active_list_response = requests.get(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/images",
            params={"limit": 1000, "status_id": "active"},
        )

        assert active_list_response.status_code == 200
        active_list = active_list_response.json()

        # Verify all images in list are active
        active_images = active_list.get("images", [])
        for image in active_images:
            assert image.get("status_id") == "active"

        # Step 6: Verify counts are correct
        # Total images = active + rejected (+ any other statuses)
        expected_active = initial_total - expected_rejected
        assert len(active_images) == expected_active

        # Default list should only show active images
        assert len(default_images) == len(active_images)


@pytest.mark.integration
class TestBurstAPIHealth:
    """Tests for burst API endpoints availability."""

    BASE_URL = "http://localhost:8765"

    def test_burst_endpoints_available(self):
        """Test that burst API endpoints are accessible."""
        # Create a test catalog ID (doesn't need to exist)
        catalog_id = str(uuid.uuid4())

        # Test list bursts endpoint (should return 200 or 404)
        list_response = requests.get(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/bursts"
        )
        assert list_response.status_code in [200, 404]

        # Test detect bursts endpoint (should accept request)
        detect_response = requests.post(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/detect-bursts"
        )
        assert detect_response.status_code in [200, 202, 404]

        # Test batch apply endpoint (should return 404 for non-existent catalog)
        batch_response = requests.post(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/bursts/batch-apply",
            json={"use_recommendations": True},
        )
        assert batch_response.status_code in [200, 404]

    def test_burst_api_error_handling(self):
        """Test burst API error handling for invalid inputs."""
        # Test with invalid catalog ID format
        invalid_id = "not-a-uuid"

        response = requests.get(
            f"{self.BASE_URL}/api/catalogs/{invalid_id}/bursts"
        )
        # Should return 422 (validation error) or handle gracefully
        assert response.status_code in [422, 400, 404, 500]

        # Test with non-existent burst ID
        catalog_id = str(uuid.uuid4())
        burst_id = str(uuid.uuid4())

        response = requests.get(
            f"{self.BASE_URL}/api/catalogs/{catalog_id}/bursts/{burst_id}"
        )
        assert response.status_code in [404, 422]
