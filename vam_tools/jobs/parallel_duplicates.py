"""
Parallel Duplicate Detection using the Coordinator Pattern.

This module implements parallel duplicate detection across Celery workers:

1. duplicates_coordinator_task: Queries images, creates batches, spawns workers
2. duplicates_hash_worker_task: Computes perceptual hashes for a batch
3. duplicates_compare_worker_task: Compares image hashes to find duplicates
4. duplicates_finalizer_task: Aggregates results and saves duplicate groups

Architecture:
    Phase 1 (Hashing): COORDINATOR → [HASH_WORKER_1, ..., HASH_WORKER_N] → callback
    Phase 2 (Comparison): callback → [COMPARE_WORKER_1, ..., COMPARE_WORKER_M] → FINALIZER

The comparison phase uses a block-based approach:
- All images are divided into blocks
- Each compare worker handles one block pair (i,j) comparison
- This distributes the O(n²) comparisons across workers
"""

from __future__ import annotations

import logging
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import chord, group
from sqlalchemy import text

from ..db import CatalogDB as CatalogDatabase
from ..db.models import Job
from .celery_app import app
from .coordinator import BatchManager, BatchResult, publish_job_progress
from .progress_publisher import publish_completion, publish_progress
from .tasks import ProgressTask

logger = logging.getLogger(__name__)


def _update_job_status(
    job_id: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Update job status directly in the database."""
    from ..db import get_db_context

    try:
        with get_db_context() as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = status
                if result is not None:
                    job.result = result
                if error is not None:
                    job.error = error
                session.commit()
                logger.debug(f"Updated job {job_id} status to {status}")
    except Exception as e:
        logger.warning(f"Failed to update job status for {job_id}: {e}")


@app.task(bind=True, base=ProgressTask, name="duplicates_coordinator")
def duplicates_coordinator_task(
    self: ProgressTask,
    catalog_id: str,
    similarity_threshold: int = 5,
    recompute_hashes: bool = False,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """
    Coordinator task for parallel duplicate detection.

    The process runs in two phases:
    1. Hash Phase: Compute perceptual hashes for all images in batches
    2. Compare Phase: Compare all image pairs to find duplicates

    Args:
        catalog_id: UUID of the catalog
        similarity_threshold: Maximum Hamming distance for similar images (default: 5)
        recompute_hashes: Force recomputation of perceptual hashes
        batch_size: Number of images per hash batch
    """
    parent_job_id = self.request.id or "unknown"
    logger.info(
        f"[{parent_job_id}] Starting duplicates coordinator for catalog {catalog_id}"
    )

    try:
        self.update_progress(0, 1, "Querying images...", {"phase": "init"})

        # Clear existing duplicate groups for this catalog
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            db.session.execute(
                text("DELETE FROM duplicate_groups WHERE catalog_id = :catalog_id"),
                {"catalog_id": catalog_id},
            )
            assert db.session is not None
            db.session.commit()

            # Get all images
            if recompute_hashes:
                # Get all images
                assert db.session is not None
                result = db.session.execute(
                    text(
                        """
                        SELECT id, source_path FROM images
                        WHERE catalog_id = :catalog_id
                        AND file_type = 'image'
                    """
                    ),
                    {"catalog_id": catalog_id},
                )
            else:
                # Only get images without perceptual hashes
                assert db.session is not None
                result = db.session.execute(
                    text(
                        """
                        SELECT id, source_path FROM images
                        WHERE catalog_id = :catalog_id
                        AND file_type = 'image'
                        AND (perceptual_hash IS NULL OR perceptual_hash = '')
                    """
                    ),
                    {"catalog_id": catalog_id},
                )

            images_to_hash = [(str(row[0]), row[1]) for row in result.fetchall()]

            # Count total images for comparison phase
            assert db.session is not None
            result = db.session.execute(
                text(
                    """
                    SELECT COUNT(*) FROM images
                    WHERE catalog_id = :catalog_id AND file_type = 'image'
                """
                ),
                {"catalog_id": catalog_id},
            )
            total_images = result.scalar() or 0

        images_needing_hash = len(images_to_hash)
        logger.info(
            f"[{parent_job_id}] {images_needing_hash} images need hashing, {total_images} total for comparison"
        )

        if total_images == 0:
            publish_completion(
                parent_job_id,
                "SUCCESS",
                result={"status": "completed", "message": "No images in catalog"},
            )
            return {"status": "completed", "message": "No images in catalog"}

        # Phase 1: Hash computation
        if images_needing_hash > 0:
            self.update_progress(
                0,
                images_needing_hash,
                f"Creating batches for {images_needing_hash} images...",
                {"phase": "batching"},
            )

            batch_manager = BatchManager(catalog_id, parent_job_id, "duplicates_hash")

            with CatalogDatabase(catalog_id) as db:
                batch_ids = batch_manager.create_batches(
                    work_items=images_to_hash,
                    batch_size=batch_size,
                    db=db,
                )

            num_batches = len(batch_ids)
            logger.info(f"[{parent_job_id}] Created {num_batches} hash batches")

            # Spawn hash workers, then comparison phase
            self.update_progress(
                0,
                images_needing_hash,
                f"Spawning {num_batches} sub-tasks for hashing...",
                {"phase": "spawning"},
            )

            hash_tasks = group(
                duplicates_hash_worker_task.s(
                    catalog_id=catalog_id,
                    batch_id=batch_id,
                    parent_job_id=parent_job_id,
                )
                for batch_id in batch_ids
            )

            # After hashing, run comparison phase via the callback
            comparison_callback = duplicates_comparison_phase_task.s(
                catalog_id=catalog_id,
                parent_job_id=parent_job_id,
                similarity_threshold=similarity_threshold,
                total_images=total_images,
            )

            chord(hash_tasks)(comparison_callback)

            logger.info(
                f"[{parent_job_id}] Hash chord dispatched: {num_batches} sub-tasks"
            )
        else:
            # All images already have hashes, go straight to comparison
            logger.info(
                f"[{parent_job_id}] All images have hashes, starting comparison phase"
            )
            duplicates_comparison_phase_task.delay(
                [],  # No hash results needed
                catalog_id=catalog_id,
                parent_job_id=parent_job_id,
                similarity_threshold=similarity_threshold,
                total_images=total_images,
            )

        _update_job_status(
            parent_job_id,
            "PROGRESS",
            result={
                "status": "processing",
                "total_images": total_images,
                "images_to_hash": images_needing_hash,
                "message": f"Processing {total_images} images",
            },
        )

        publish_progress(
            parent_job_id,
            "PROGRESS",
            current=0,
            total=total_images,
            message=f"Processing {total_images} images (hashing phase)",
            extra={"phase": "hashing", "images_to_hash": images_needing_hash},
        )

        return {
            "status": "dispatched",
            "catalog_id": catalog_id,
            "total_images": total_images,
            "images_to_hash": images_needing_hash,
            "message": f"Processing {total_images} images",
        }

    except Exception as e:
        logger.error(f"[{parent_job_id}] Coordinator failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        raise


@app.task(bind=True, name="duplicates_hash_worker")
def duplicates_hash_worker_task(
    self: Any,
    catalog_id: str,
    batch_id: str,
    parent_job_id: str,
) -> Dict[str, Any]:
    """Worker task that computes perceptual hashes for a batch of images."""
    from ..analysis.perceptual_hash import dhash

    worker_id = self.request.id or "unknown"
    logger.info(f"[{worker_id}] Starting hash worker for batch {batch_id}")

    batch_manager = BatchManager(catalog_id, parent_job_id, "duplicates_hash")

    try:
        with CatalogDatabase(catalog_id) as db:
            batch_data = batch_manager.claim_batch(batch_id, worker_id, db)

        if not batch_data:
            logger.warning(f"[{worker_id}] Batch {batch_id} already claimed")
            return {
                "batch_id": batch_id,
                "status": "skipped",
                "reason": "already_claimed",
            }

        batch_number = batch_data["batch_number"]
        total_batches = batch_data["total_batches"]
        image_data = batch_data["work_items"]  # List of (image_id, source_path)
        items_count = batch_data["items_count"]

        logger.info(
            f"[{worker_id}] Processing batch {batch_number + 1}/{total_batches} ({items_count} images)"
        )

        result = BatchResult(batch_id=batch_id, batch_number=batch_number)

        with CatalogDatabase(catalog_id) as db:
            for image_id, source_path in image_data:
                try:
                    path = Path(source_path)
                    if not path.exists():
                        result.error_count += 1
                        result.errors.append(
                            {"image_id": image_id, "error": "File not found"}
                        )
                        continue

                    # Compute perceptual hash
                    phash = dhash(path)

                    if phash:
                        # Save hash to database
                        assert db.session is not None
                        db.session.execute(
                            text(
                                """
                                UPDATE images
                                SET perceptual_hash = :hash
                                WHERE id = :image_id
                            """
                            ),
                            {"image_id": image_id, "hash": phash},
                        )
                        result.success_count += 1
                    else:
                        result.error_count += 1
                        result.errors.append(
                            {"image_id": image_id, "error": "Failed to compute hash"}
                        )

                    result.processed_count += 1

                except Exception as e:
                    result.error_count += 1
                    result.errors.append({"image_id": image_id, "error": str(e)})

            assert db.session is not None
            db.session.commit()

            # Mark batch complete
            batch_manager.complete_batch(batch_id, result, db)
            progress = batch_manager.get_progress(db)
            publish_job_progress(
                parent_job_id,
                progress,
                f"Hashing batch {batch_number + 1}/{total_batches} complete",
                phase="hashing",
            )

        logger.info(
            f"[{worker_id}] Batch {batch_number + 1} complete: {result.success_count} hashed, {result.error_count} errors"
        )

        return {
            "batch_id": batch_id,
            "batch_number": batch_number,
            "status": "completed",
            "success_count": result.success_count,
            "error_count": result.error_count,
        }

    except Exception as e:
        logger.error(f"[{worker_id}] Worker failed: {e}", exc_info=True)
        try:
            batch_manager.fail_batch(batch_id, str(e))
        except Exception:
            pass
        return {"batch_id": batch_id, "status": "failed", "error": str(e)}


@app.task(bind=True, base=ProgressTask, name="duplicates_comparison_phase")
def duplicates_comparison_phase_task(
    self: ProgressTask,
    hash_results: List[Dict[str, Any]],
    catalog_id: str,
    parent_job_id: str,
    similarity_threshold: int,
    total_images: int,
) -> Dict[str, Any]:
    """
    Intermediate task that launches the comparison phase after hashing.

    This loads all image hashes and divides comparison work into blocks.
    Each comparison worker handles a block pair.
    """
    task_id = self.request.id or "unknown"
    logger.info(f"[{task_id}] Starting comparison phase for job {parent_job_id}")

    try:
        # Report hash phase completion
        hash_success = sum(
            r.get("success_count", 0)
            for r in hash_results
            if r.get("status") == "completed"
        )
        hash_errors = sum(
            r.get("error_count", 0)
            for r in hash_results
            if r.get("status") == "completed"
        )
        logger.info(
            f"[{task_id}] Hash phase complete: {hash_success} success, {hash_errors} errors"
        )

        self.update_progress(
            0, 1, "Loading image hashes...", {"phase": "comparison_init"}
        )

        # Load all image hashes from database
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            result = db.session.execute(
                text(
                    """
                    SELECT id, perceptual_hash, checksum, source_path, quality_score
                    FROM images
                    WHERE catalog_id = :catalog_id
                    AND file_type = 'image'
                    AND perceptual_hash IS NOT NULL
                    AND perceptual_hash != ''
                """
                ),
                {"catalog_id": catalog_id},
            )

            images = []
            for row in result.fetchall():
                images.append(
                    {
                        "id": str(row[0]),
                        "perceptual_hash": row[1],
                        "checksum": row[2],
                        "source_path": row[3],
                        "quality_score": row[4] or 0.0,
                    }
                )

        num_images = len(images)
        logger.info(f"[{task_id}] Loaded {num_images} images with hashes")

        if num_images < 2:
            # Not enough images to compare
            final_result = {
                "status": "completed",
                "catalog_id": catalog_id,
                "total_images": num_images,
                "duplicate_groups": 0,
                "total_duplicates": 0,
                "message": "Not enough images for comparison",
            }
            publish_completion(parent_job_id, "SUCCESS", result=final_result)
            _update_job_status(parent_job_id, "SUCCESS", result=final_result)
            return final_result

        # Divide images into blocks for parallel comparison
        # Each block pair (i,j) will be compared by a worker
        # For N images divided into B blocks, we have B*(B+1)/2 comparisons
        block_size = 500  # Images per block
        num_blocks = math.ceil(num_images / block_size)

        # Create comparison tasks for each block pair
        # Block pairs: (0,0), (0,1), (0,2), ..., (1,1), (1,2), ..., (n-1,n-1)
        comparison_tasks = []
        for i in range(num_blocks):
            for j in range(i, num_blocks):
                comparison_tasks.append(
                    duplicates_compare_worker_task.s(
                        catalog_id=catalog_id,
                        parent_job_id=parent_job_id,
                        block_i=i,
                        block_j=j,
                        block_size=block_size,
                        similarity_threshold=similarity_threshold,
                        images=images,  # Pass all images - workers will slice
                    )
                )

        num_comparisons = len(comparison_tasks)
        logger.info(
            f"[{task_id}] Creating {num_comparisons} comparison tasks ({num_blocks} blocks)"
        )

        publish_progress(
            parent_job_id,
            "PROGRESS",
            current=hash_success,
            total=total_images + num_comparisons,
            message=f"Starting comparison phase ({num_comparisons} block pairs)",
            extra={"phase": "comparing", "comparison_tasks": num_comparisons},
        )

        # Launch comparison workers with finalizer
        finalizer = duplicates_finalizer_task.s(
            catalog_id=catalog_id,
            parent_job_id=parent_job_id,
            total_images=num_images,
        )

        chord(group(comparison_tasks))(finalizer)

        logger.info(
            f"[{task_id}] Comparison chord dispatched: {num_comparisons} tasks → finalizer"
        )

        return {
            "status": "comparison_dispatched",
            "hash_success": hash_success,
            "hash_errors": hash_errors,
            "images_with_hashes": num_images,
            "comparison_tasks": num_comparisons,
        }

    except Exception as e:
        logger.error(f"[{task_id}] Comparison phase failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        _update_job_status(parent_job_id, "FAILURE", error=str(e))
        raise


@app.task(bind=True, name="duplicates_compare_worker")
def duplicates_compare_worker_task(
    self: Any,
    catalog_id: str,
    parent_job_id: str,
    block_i: int,
    block_j: int,
    block_size: int,
    similarity_threshold: int,
    images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Worker task that compares images in two blocks.

    For block pair (i,j):
    - If i == j: Compare all pairs within block i
    - If i != j: Compare all images in block i against all in block j
    """
    worker_id = self.request.id or "unknown"
    logger.info(f"[{worker_id}] Starting comparison for blocks ({block_i}, {block_j})")

    try:
        # Get images for each block
        start_i = block_i * block_size
        end_i = min(start_i + block_size, len(images))
        block_i_images = images[start_i:end_i]

        if block_i == block_j:
            # Compare within same block
            block_j_images = block_i_images
        else:
            start_j = block_j * block_size
            end_j = min(start_j + block_size, len(images))
            block_j_images = images[start_j:end_j]

        # Find duplicate pairs
        duplicate_pairs = []

        for idx_i, img_i in enumerate(block_i_images):
            # For same block, only compare with images after this one
            start_idx = idx_i + 1 if block_i == block_j else 0

            for _idx_j, img_j in enumerate(block_j_images[start_idx:], start=start_idx):
                # Check exact duplicates (same checksum)
                if (
                    img_i["checksum"]
                    and img_j["checksum"]
                    and img_i["checksum"] == img_j["checksum"]
                ):
                    duplicate_pairs.append(
                        {
                            "image_1": img_i["id"],
                            "image_2": img_j["id"],
                            "type": "exact",
                            "distance": 0,
                        }
                    )
                    continue

                # Check perceptual similarity
                if img_i["perceptual_hash"] and img_j["perceptual_hash"]:
                    distance = _hamming_distance(
                        img_i["perceptual_hash"], img_j["perceptual_hash"]
                    )
                    if distance <= similarity_threshold:
                        duplicate_pairs.append(
                            {
                                "image_1": img_i["id"],
                                "image_2": img_j["id"],
                                "type": "similar",
                                "distance": distance,
                            }
                        )

        logger.info(
            f"[{worker_id}] Blocks ({block_i}, {block_j}): found {len(duplicate_pairs)} pairs"
        )

        return {
            "block_i": block_i,
            "block_j": block_j,
            "status": "completed",
            "pairs_found": len(duplicate_pairs),
            "duplicate_pairs": duplicate_pairs,
        }

    except Exception as e:
        logger.error(f"[{worker_id}] Comparison worker failed: {e}", exc_info=True)
        return {
            "block_i": block_i,
            "block_j": block_j,
            "status": "failed",
            "error": str(e),
        }


@app.task(bind=True, base=ProgressTask, name="duplicates_finalizer")
def duplicates_finalizer_task(
    self: ProgressTask,
    comparison_results: List[Dict[str, Any]],
    catalog_id: str,
    parent_job_id: str,
    total_images: int,
) -> Dict[str, Any]:
    """Finalizer that builds and saves duplicate groups from comparison pairs."""
    finalizer_id = self.request.id or "unknown"
    logger.info(f"[{finalizer_id}] Starting finalizer for job {parent_job_id}")

    try:
        self.update_progress(
            0, 1, "Building duplicate groups...", {"phase": "finalizing"}
        )

        # Collect all duplicate pairs
        all_pairs = []
        for result in comparison_results:
            if result.get("status") == "completed":
                all_pairs.extend(result.get("duplicate_pairs", []))

        logger.info(f"[{finalizer_id}] Collected {len(all_pairs)} duplicate pairs")

        if not all_pairs:
            final_result = {
                "status": "completed",
                "catalog_id": catalog_id,
                "total_images": total_images,
                "duplicate_groups": 0,
                "total_duplicates": 0,
            }
            publish_completion(parent_job_id, "SUCCESS", result=final_result)
            _update_job_status(parent_job_id, "SUCCESS", result=final_result)
            return final_result

        # Build groups using union-find
        groups = _build_duplicate_groups(all_pairs)
        logger.info(f"[{finalizer_id}] Built {len(groups)} duplicate groups")

        self.update_progress(
            0, 1, f"Saving {len(groups)} duplicate groups...", {"phase": "saving"}
        )

        # Load image quality scores for selecting primary
        image_quality = {}
        with CatalogDatabase(catalog_id) as db:
            for group_images in groups:
                for img_id in group_images:
                    if img_id not in image_quality:
                        assert db.session is not None
                        query_result = db.session.execute(
                            text("SELECT quality_score FROM images WHERE id = :id"),
                            {"id": img_id},
                        )
                        row = query_result.fetchone()
                        image_quality[img_id] = row[0] if row and row[0] else 0.0

        # Save groups to database
        total_duplicates = 0
        with CatalogDatabase(catalog_id) as db:
            for group_images in groups:
                if len(group_images) < 2:
                    continue

                group_id = str(uuid.uuid4())

                # Select primary (highest quality)
                primary_id = max(group_images, key=lambda x: image_quality.get(x, 0))

                # Insert group
                assert db.session is not None
                db.session.execute(
                    text(
                        """
                        INSERT INTO duplicate_groups (
                            id, catalog_id, primary_image_id, image_count, created_at
                        ) VALUES (
                            :id, :catalog_id, :primary_id, :count, NOW()
                        )
                    """
                    ),
                    {
                        "id": group_id,
                        "catalog_id": catalog_id,
                        "primary_id": primary_id,
                        "count": len(group_images),
                    },
                )

                # Update images with group membership
                for img_id in group_images:
                    is_primary = img_id == primary_id
                    assert db.session is not None
                    db.session.execute(
                        text(
                            """
                            UPDATE images
                            SET duplicate_group_id = :group_id,
                                is_duplicate_primary = :is_primary
                            WHERE id = :image_id
                        """
                        ),
                        {
                            "group_id": group_id,
                            "image_id": img_id,
                            "is_primary": is_primary,
                        },
                    )

                total_duplicates += (
                    len(group_images) - 1
                )  # Don't count primary as duplicate

            assert db.session is not None
            db.session.commit()

        failed_comparisons = sum(
            1 for r in comparison_results if r.get("status") == "failed"
        )

        final_result = {
            "status": (
                "completed" if failed_comparisons == 0 else "completed_with_errors"
            ),
            "catalog_id": catalog_id,
            "total_images": total_images,
            "duplicate_groups": len(groups),
            "total_duplicates": total_duplicates,
            "failed_comparisons": failed_comparisons,
        }

        publish_completion(parent_job_id, "SUCCESS", result=final_result)
        _update_job_status(parent_job_id, "SUCCESS", result=final_result)

        self.update_progress(
            total_images,
            total_images,
            f"Complete: {len(groups)} groups, {total_duplicates} duplicates",
            {"phase": "complete"},
        )

        logger.info(
            f"[{finalizer_id}] Duplicate detection complete: {len(groups)} groups, {total_duplicates} duplicates"
        )

        return final_result

    except Exception as e:
        logger.error(f"[{finalizer_id}] Finalizer failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        _update_job_status(parent_job_id, "FAILURE", error=str(e))
        raise


def _hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex-encoded hashes."""
    if len(hash1) != len(hash2):
        return 999  # Invalid comparison

    distance = 0
    for c1, c2 in zip(hash1, hash2):
        try:
            xor = int(c1, 16) ^ int(c2, 16)
            distance += bin(xor).count("1")
        except ValueError:
            return 999  # Invalid hex

    return distance


def _build_duplicate_groups(pairs: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Build duplicate groups from pairwise relationships using union-find.

    Args:
        pairs: List of {image_1, image_2, type, distance} dicts

    Returns:
        List of image ID lists, each representing a duplicate group
    """
    # Union-find data structure
    parent = {}
    rank = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
            rank[x] = 0
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px == py:
            return
        # Union by rank
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    # Build groups
    for pair in pairs:
        union(pair["image_1"], pair["image_2"])

    # Collect groups
    groups_dict: Dict[str, List[str]] = {}
    for img_id in parent.keys():
        root = find(img_id)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(img_id)

    # Only return groups with 2+ images
    return [group for group in groups_dict.values() if len(group) >= 2]
