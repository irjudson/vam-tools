#!/usr/bin/env python3
"""Manually trigger the duplicates finalizer for stuck job 0b568b28"""

from lumina.jobs.parallel_duplicates import duplicates_finalizer_task

# Parameters from logs
catalog_id = "bd40ca52-c3f7-4877-9c97-1c227389c8c4"
parent_job_id = "0b568b28-b8d8-47ae-a08b-8f9f34ce844d"
total_images = 96144  # From log: "Loaded 96144 images with valid hashes"
similarity_threshold = 10  # Default value

print(f"Manually triggering finalizer for job {parent_job_id}")
print(f"  catalog_id: {catalog_id}")
print(f"  total_images: {total_images}")
print(f"  similarity_threshold: {similarity_threshold}")
print()

# Call finalizer directly (not as async task)
result = duplicates_finalizer_task(
    catalog_id=catalog_id,
    parent_job_id=parent_job_id,
    total_images=total_images,
    similarity_threshold=similarity_threshold,
)

print("Finalizer completed!")
print(f"Result: {result}")
