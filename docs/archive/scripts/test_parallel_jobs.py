#!/usr/bin/env python3
"""
End-to-end test for parallel job processing.

This test verifies that:
1. Jobs can be submitted via the API
2. Parallel batch processing works
3. Results are correctly saved to the catalog
"""

import json
import requests
import time
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8765/api"
TEST_CATALOG_ID = "test"

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

def test_catalog_exists():
    """Test that a catalog exists."""
    print("Checking for catalog...")
    response = requests.get(f"{API_BASE}/catalogs/current")
    response.raise_for_status()
    catalog = response.json()

    assert catalog is not None, "No current catalog configured"
    print(f"  ✓ Found catalog: {catalog['name']}")
    print(f"  Catalog path: {catalog['catalog_path']}")
    print(f"  Source dirs: {catalog['source_directories']}\n")
    return catalog

def submit_analysis_job(force_reanalyze=True):
    """Submit an analysis job."""
    print("Submitting analysis job...")

    # Get catalog info
    response = requests.get(f"{API_BASE}/catalogs/current")
    response.raise_for_status()
    catalog = response.json()

    # Submit job
    job_data = {
        "catalog_path": catalog["catalog_path"],
        "source_directories": catalog["source_directories"],
        "detect_duplicates": False,
        "force_reanalyze": force_reanalyze
    }

    response = requests.post(f"{API_BASE}/jobs/analyze", json=job_data)
    response.raise_for_status()
    result = response.json()

    job_id = result["job_id"]
    print(f"  ✓ Job submitted: {job_id}\n")
    return job_id

def poll_job_status(job_id, timeout=120):
    """Poll job status until complete or timeout."""
    print(f"Polling job {job_id[:8]}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        response = requests.get(f"{API_BASE}/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()

        status = job.get("status", "UNKNOWN")
        progress = job.get("progress", {})

        # Print progress
        if progress:
            percent = progress.get("percent", 0)
            message = progress.get("message", "")
            phase = progress.get("phase", "")
            print(f"  [{percent:3d}%] {phase}: {message}")

        if status == "SUCCESS":
            print(f"  ✓ Job completed successfully\n")
            return job
        elif status == "FAILURE":
            error = job.get("error", "Unknown error")
            raise Exception(f"Job failed: {error}")

        time.sleep(2)

    raise TimeoutError(f"Job did not complete within {timeout}s")

def verify_catalog_populated():
    """Verify that catalog was populated with images."""
    print("Verifying catalog contents...")

    response = requests.get(f"{API_BASE}/dashboard/stats")
    response.raise_for_status()
    stats = response.json()

    total_images = stats.get("total_images", 0)
    total_videos = stats.get("total_videos", 0)
    total_files = total_images + total_videos

    print(f"  Total images: {total_images}")
    print(f"  Total videos: {total_videos}")
    print(f"  Total files: {total_files}")

    assert total_files > 0, "Catalog is empty after analysis!"
    print(f"  ✓ Catalog populated with {total_files} files\n")
    return stats

def main():
    """Run end-to-end test."""
    print("=" * 60)
    print("Lumina - Parallel Job Processing Test")
    print("=" * 60 + "\n")

    try:
        # Run tests
        test_worker_health()
        test_catalog_exists()
        job_id = submit_analysis_job(force_reanalyze=True)
        job_result = poll_job_status(job_id)
        stats = verify_catalog_populated()

        # Print final results
        print("=" * 60)
        print("TEST PASSED")
        print("=" * 60)
        print(f"Job ID: {job_id}")
        print(f"Files processed: {job_result.get('result', {}).get('processed', 0)}")
        print(f"Files added: {job_result.get('result', {}).get('files_added', 0)}")
        print(f"Files skipped: {job_result.get('result', {}).get('files_skipped', 0)}")
        print(f"Catalog contains: {stats['total_images']} images, {stats['total_videos']} videos")

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
