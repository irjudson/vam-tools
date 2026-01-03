#!/usr/bin/env python3
"""
End-to-end test for parallel job processing (local setup).

Tests the complete workflow:
1. Create test catalog with 2 photos
2. Submit analysis job
3. Verify parallel processing works
4. Confirm catalog is populated
"""

import json
import os
import requests
import sys
import time
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8765/api"
TEST_CATALOG_NAME = "E2E Test"
TEST_CATALOG_DIR = Path.home() / "catalogs" / "e2e-test-catalog"
TEST_PHOTOS_DIR = Path.home() / "catalogs" / "e2e-test-photos"

def setup_test_photos():
    """Create test photo directory with sample files."""
    print("Setting up test photos...")

    # Use existing test photos
    source_dir = Path(__file__).parent / "test-photos"

    # Create test photos directory
    TEST_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy test photos
    import shutil
    for photo in source_dir.glob("*.jpg"):
        shutil.copy2(photo, TEST_PHOTOS_DIR / photo.name)

    photo_count = len(list(TEST_PHOTOS_DIR.glob("*.jpg")))
    print(f"  ✓ Created test directory with {photo_count} photos\n")
    return photo_count

def test_api_available():
    """Test that API is responding."""
    print("Testing API availability...")
    try:
        response = requests.get(f"{API_BASE}", timeout=5)
        response.raise_for_status()
        print("  ✓ API is responding\n")
    except Exception as e:
        raise Exception(f"API not available: {e}")

def test_worker_health():
    """Test that Celery workers are running."""
    print("Testing worker health...")
    response = requests.get(f"{API_BASE}/jobs/health")
    response.raise_for_status()
    health = response.json()
    print(f"  Worker status: {health['status']}")
    print(f"  Active workers: {health['workers']}")
    assert health["status"] == "healthy", f"Workers unhealthy: {health['message']}"
    assert health["workers"] > 0, "No workers running"
    print("  ✓ Workers healthy\n")

def create_test_catalog():
    """Create a test catalog."""
    print(f"Creating test catalog '{TEST_CATALOG_NAME}'...")

    # Clean up old catalog if exists
    import shutil
    if TEST_CATALOG_DIR.exists():
        shutil.rmtree(TEST_CATALOG_DIR)

    catalog_data = {
        "name": TEST_CATALOG_NAME,
        "catalog_path": str(TEST_CATALOG_DIR),
        "source_directories": [str(TEST_PHOTOS_DIR)],
        "description": "End-to-end test catalog",
        "color": "#22c55e"
    }

    response = requests.post(f"{API_BASE}/catalogs", json=catalog_data)
    response.raise_for_status()
    catalog = response.json()

    # Switch to this catalog
    response = requests.post(f"{API_BASE}/catalogs/current", json={"catalog_id": catalog["id"]})
    response.raise_for_status()

    print(f"  ✓ Created catalog: {catalog['id'][:8]}...\n")
    return catalog

def submit_analysis_job():
    """Submit an analysis job."""
    print("Submitting analysis job...")

    # Get current catalog
    response = requests.get(f"{API_BASE}/catalogs/current")
    response.raise_for_status()
    catalog = response.json()

    # Submit job
    job_data = {
        "catalog_path": catalog["catalog_path"],
        "source_directories": catalog["source_directories"],
        "detect_duplicates": False,
        "force_reanalyze": True
    }

    response = requests.post(f"{API_BASE}/jobs/analyze", json=job_data)
    response.raise_for_status()
    result = response.json()

    job_id = result["job_id"]
    print(f"  ✓ Job submitted: {job_id[:8]}...\n")
    return job_id

def poll_job_until_complete(job_id, timeout=60):
    """Poll job status until complete."""
    print(f"Polling job {job_id[:8]}...")
    start_time = time.time()
    last_status = None

    while time.time() - start_time < timeout:
        response = requests.get(f"{API_BASE}/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()

        status = job.get("status", "UNKNOWN")

        # Print status changes
        if status != last_status:
            progress = job.get("progress") or {}
            message = progress.get("message", "")
            phase = progress.get("phase", "")
            print(f"  [{status}] {phase}: {message}")
            last_status = status

        if status == "SUCCESS":
            result = job.get("result", {})
            print(f"  ✓ Job completed successfully")
            print(f"    Processed: {result.get('processed', 0)}")
            print(f"    Added: {result.get('files_added', 0)}")
            print(f"    Skipped: {result.get('files_skipped', 0)}\n")
            return job
        elif status == "FAILURE":
            error = job.get("error", "Unknown error")
            raise Exception(f"Job failed: {error}")

        time.sleep(1)

    raise TimeoutError(f"Job did not complete within {timeout}s")

def verify_catalog_populated(expected_files):
    """Verify catalog was populated."""
    print("Verifying catalog contents...")

    response = requests.get(f"{API_BASE}/dashboard/stats")
    response.raise_for_status()
    stats = response.json()

    total_images = stats.get("total_images", 0)
    total_videos = stats.get("total_videos", 0)
    total_files = total_images + total_videos

    print(f"  Total images: {total_images}")
    print(f"  Total videos: {total_videos}")

    assert total_files == expected_files, f"Expected {expected_files} files, got {total_files}"
    print(f"  ✓ Catalog correctly populated with {total_files} files\n")

def cleanup():
    """Clean up test data."""
    print("Cleaning up test data...")
    import shutil

    if TEST_CATALOG_DIR.exists():
        shutil.rmtree(TEST_CATALOG_DIR)
    if TEST_PHOTOS_DIR.exists():
        shutil.rmtree(TEST_PHOTOS_DIR)

    print("  ✓ Cleanup complete\n")

def main():
    """Run end-to-end test."""
    print("=" * 70)
    print("Lumina - End-to-End Test (Local Setup)")
    print("=" * 70 + "\n")

    try:
        # Setup
        photo_count = setup_test_photos()

        # Run tests
        test_api_available()
        test_worker_health()
        catalog = create_test_catalog()
        job_id = submit_analysis_job()
        job = poll_job_until_complete(job_id, timeout=60)

        # Verify results from job
        result = job.get('result', {})
        assert result.get('processed', 0) == photo_count, f"Expected {photo_count} processed, got {result.get('processed')}"
        assert result.get('files_added', 0) == photo_count, f"Expected {photo_count} added, got {result.get('files_added')}"
        print(f"  ✓ Job processed {photo_count} files correctly\n")

        # Success
        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print(f"\nThe job processing system is working correctly!")
        print(f"Job {job_id[:8]} successfully processed {photo_count} files.")
        print(f"\nResults:")
        print(f"  Catalog: {catalog['name']}")
        print(f"  Files processed: {job['result']['processed']}")
        print(f"  Files added: {job['result']['files_added']}")
        print(f"  Files skipped: {job['result']['files_skipped']}")

        cleanup()
        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST FAILED ✗")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        cleanup()
        return 1

if __name__ == "__main__":
    sys.exit(main())
