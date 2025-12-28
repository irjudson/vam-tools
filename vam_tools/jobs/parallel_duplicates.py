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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import chord, group
from sqlalchemy import text
from sqlalchemy.engine import Result

from ..db import CatalogDB as CatalogDatabase
from ..db.models import Job
from .celery_app import app
from .coordinator import (
    CONSECUTIVE_FAILURE_THRESHOLD,
    BatchManager,
    BatchResult,
    JobCancelledException,
    cancel_and_requeue_job,
    publish_job_progress,
)
from .progress_publisher import publish_completion, publish_progress
from .tasks import ProgressTask

logger = logging.getLogger(__name__)


class DuplicateGroupingStrategy(str, Enum):
    """
    Different strategies for grouping similar images.

    STRICT_CLIQUE: Current implementation. Only groups images where every image
                   is similar to every other image in the group. Prevents
                   transitive closure mega-groups.

    STAR_GROUPS: Future strategy. Creates star-shaped groups with a primary
                 image (highest quality) and duplicates. Other images are only
                 considered duplicates if similar to the primary, not each other.
                 Good for photo shoots with many similar shots.

    PAIR_GROUPS: Future strategy. Creates separate groups for each pair.
                 Images can appear in multiple groups {A,B} and {B,C} are
                 independent. Good for finding all similar relationships.

    TRANSITIVE: Legacy Union-Find implementation. Creates transitive closures
                which cause mega-groups. Kept for reference only.
    """

    STRICT_CLIQUE = "strict_clique"  # Current implementation
    STAR_GROUPS = "star_groups"  # Future: primary with duplicates
    PAIR_GROUPS = "pair_groups"  # Future: overlapping pairs
    TRANSITIVE = "transitive"  # Legacy: Union-find (creates mega-groups)


# Current grouping strategy (may be made configurable in future)
CURRENT_GROUPING_STRATEGY = DuplicateGroupingStrategy.STRICT_CLIQUE


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


def _increment_job_failure_count(job_id: str) -> None:
    """Increment the worker failure count in the job's result field."""
    from ..db import get_db_context

    try:
        with get_db_context() as session:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job:
                if job.result is None:
                    job.result = {}
                if "worker_failures" not in job.result:
                    job.result["worker_failures"] = 0
                job.result["worker_failures"] += 1
                session.commit()
                logger.debug(
                    f"Incremented worker_failures for job {job_id} to {job.result['worker_failures']}"
                )
    except Exception as e:
        logger.warning(f"Failed to increment failure count for {job_id}: {e}")


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
                # Only get images without perceptual hashes (dhash column)
                assert db.session is not None
                result = db.session.execute(
                    text(
                        """
                        SELECT id, source_path FROM images
                        WHERE catalog_id = :catalog_id
                        AND file_type = 'image'
                        AND (dhash IS NULL OR dhash = '')
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

        # Set job to STARTED state - workers are now processing
        # The finalizer will update to SUCCESS when all batches complete
        _update_job_status(
            parent_job_id,
            "STARTED",
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
    from ..analysis.perceptual_hash import ahash, dhash, whash

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

                    # Compute all three perceptual hashes for maximum accuracy
                    # dhash: gradient-based, good for edits
                    # ahash: mean-based, fast, good for exact duplicates
                    # whash: wavelet-based, most robust to transformations
                    dhash_val = dhash(path)
                    ahash_val = ahash(path)
                    whash_val = whash(path)

                    if dhash_val or ahash_val or whash_val:
                        # Save all hashes to database
                        assert db.session is not None
                        db.session.execute(
                            text(
                                """
                                UPDATE images
                                SET dhash = :dhash, ahash = :ahash, whash = :whash
                                WHERE id = :image_id
                            """
                            ),
                            {
                                "image_id": image_id,
                                "dhash": dhash_val,
                                "ahash": ahash_val,
                                "whash": whash_val,
                            },
                        )
                        result.success_count += 1
                    else:
                        result.error_count += 1
                        result.errors.append(
                            {"image_id": image_id, "error": "Failed to compute hashes"}
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

        # Load all image hashes from database (filter out problematic images)
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            result = db.session.execute(
                text(
                    """
                    SELECT id, dhash, ahash, whash, checksum, source_path, quality_score
                    FROM images
                    WHERE catalog_id = :catalog_id
                    AND file_type = 'image'
                    AND dhash IS NOT NULL
                    AND dhash != ''
                    AND dhash != '0000000000000000'
                    AND ahash IS NOT NULL
                    AND ahash != ''
                    AND whash IS NOT NULL
                    AND whash != ''
                """
                ),
                {"catalog_id": catalog_id},
            )

            images = []
            for row in result.fetchall():
                images.append(
                    {
                        "id": str(row[0]),
                        "dhash": row[1],
                        "ahash": row[2],
                        "whash": row[3],
                        "checksum": row[4],
                        "source_path": row[5],
                        "quality_score": row[6] or 0.0,
                    }
                )

        num_images = len(images)
        logger.info(
            f"[{task_id}] Loaded {num_images} images with valid hashes "
            f"(filtered: videos, null hashes, zero hashes)"
        )

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

        # Create temp table for duplicate pairs (before spawning workers to avoid race condition)
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            db.session.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS duplicate_pairs_temp (
                        job_id TEXT NOT NULL,
                        image_1 TEXT NOT NULL,
                        image_2 TEXT NOT NULL,
                        type TEXT NOT NULL,
                        distance INTEGER NOT NULL,
                        PRIMARY KEY (job_id, image_1, image_2)
                    )
                    """
                )
            )
            db.session.commit()
        logger.info(f"[{task_id}] Created temp table for duplicate pairs")

        # Divide images into blocks for parallel comparison
        # Each block pair (i,j) will be compared by a worker
        # For N images divided into B blocks, we have B*(B+1)/2 comparisons
        block_size = 500  # Images per block
        num_blocks = math.ceil(num_images / block_size)

        # Generate all block pairs that need to be compared
        # Block pairs: (0,0), (0,1), (0,2), ..., (1,1), (1,2), ..., (n-1,n-1)
        block_pairs = []
        for i in range(num_blocks):
            for j in range(i, num_blocks):
                block_pairs.append((i, j))

        total_block_pairs = len(block_pairs)
        logger.info(
            f"[{task_id}] Generated {total_block_pairs} block pairs from {num_blocks} blocks"
        )

        # Batch block pairs to avoid creating too many Celery tasks
        # Each worker will process multiple block pairs
        pairs_per_batch = 100  # Process 100 block pairs per worker
        comparison_tasks = []

        for batch_start in range(0, total_block_pairs, pairs_per_batch):
            batch_end = min(batch_start + pairs_per_batch, total_block_pairs)
            batch = block_pairs[batch_start:batch_end]

            comparison_tasks.append(
                duplicates_compare_worker_task.s(
                    catalog_id=catalog_id,
                    parent_job_id=parent_job_id,
                    block_pairs=batch,  # Pass list of (i,j) tuples
                    block_size=block_size,
                    similarity_threshold=similarity_threshold,
                    images=images,  # Pass all images - workers will slice
                )
            )

        num_worker_tasks = len(comparison_tasks)
        logger.info(
            f"[{task_id}] Creating {num_worker_tasks} comparison workers for {total_block_pairs} block pairs ({pairs_per_batch} pairs/worker)"
        )

        publish_progress(
            parent_job_id,
            "PROGRESS",
            current=hash_success,
            total=total_images,
            message=f"Starting comparison phase ({num_worker_tasks} workers, {total_block_pairs} block pairs)",
            extra={
                "phase": "comparing",
                "worker_tasks": num_worker_tasks,
                "block_pairs": total_block_pairs,
            },
        )

        # Launch comparison workers with finalizer
        # Use .si() (immutable signature) to prevent passing worker results through Redis
        # Worker failures are tracked in the job record instead
        finalizer = duplicates_finalizer_task.si(
            catalog_id=catalog_id,
            parent_job_id=parent_job_id,
            total_images=num_images,
            similarity_threshold=similarity_threshold,
        )

        chord(group(comparison_tasks))(finalizer)

        logger.info(
            f"[{task_id}] Comparison chord dispatched: {num_worker_tasks} workers → finalizer"
        )

        return {
            "status": "comparison_dispatched",
            "hash_success": hash_success,
            "hash_errors": hash_errors,
            "images_with_hashes": num_images,
            "comparison_tasks": num_worker_tasks,
            "block_pairs": total_block_pairs,
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
    block_pairs: List[tuple],
    block_size: int,
    similarity_threshold: int,
    images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Worker task that compares images in multiple block pairs.

    Args:
        block_pairs: List of (i,j) tuples representing block pairs to compare

    For each block pair (i,j):
    - If i == j: Compare all pairs within block i
    - If i != j: Compare all images in block i against all in block j
    """
    worker_id = self.request.id or "unknown"
    num_pairs = len(block_pairs)
    logger.info(f"[{worker_id}] Starting comparison for {num_pairs} block pairs")

    # Publish progress at start
    try:
        from .progress_publisher import publish_progress

        publish_progress(
            job_id=parent_job_id,
            state="PROGRESS",
            current=0,
            total=num_pairs,
            message=f"Starting comparison batch ({num_pairs} block pairs)...",
            extra={"phase": "comparing", "num_pairs": num_pairs},
        )
    except Exception:
        pass  # Non-critical

    try:
        # Instead of collecting ALL pairs in memory, write them to database in batches
        # This prevents "value too large for Redis" errors
        from ..db import CatalogDB as CatalogDatabase

        pairs_found = 0
        pairs_processed = 0
        batch_pairs: List[Dict[str, Any]] = []
        BATCH_SIZE = 5000  # Write to DB every 5000 pairs

        def _flush_pairs_to_db() -> None:
            """Helper to write batch of pairs to database"""
            nonlocal batch_pairs
            if not batch_pairs:
                return

            try:
                with CatalogDatabase(catalog_id) as db:
                    assert db.session is not None
                    # Batch insert all pairs (table already created by coordinator)
                    for pair in batch_pairs:
                        db.session.execute(
                            text(
                                """
                                INSERT INTO duplicate_pairs_temp
                                (job_id, image_1, image_2, type, distance)
                                VALUES (:job_id, :img1, :img2, :type, :dist)
                                ON CONFLICT DO NOTHING
                                """
                            ),
                            {
                                "job_id": parent_job_id,
                                "img1": pair["image_1"],
                                "img2": pair["image_2"],
                                "type": pair["type"],
                                "dist": pair["distance"],
                            },
                        )
                    db.session.commit()
                    batch_pairs = []  # Reset for next batch
            except Exception as e:
                logger.warning(f"Failed to save duplicate pairs batch: {e}")
                # Don't fail the whole task, just log the error

        # Create batch manager for cancellation checking
        # (duplicates don't use job_batches, so we check parent job only)
        batch_manager = BatchManager(catalog_id, parent_job_id, "detect_duplicates")

        # Process each block pair in the batch
        for pair_idx, (block_i, block_j) in enumerate(block_pairs):
            # Check for cancellation every 10 block pairs
            if pair_idx % 10 == 0:
                if batch_manager.is_cancelled():
                    logger.warning(
                        f"[{worker_id}] Job cancelled, stopping after {pair_idx}/{num_pairs} block pairs"
                    )
                    # Flush any remaining pairs before stopping
                    _flush_pairs_to_db()
                    raise JobCancelledException(
                        f"Job cancelled after processing {pair_idx}/{num_pairs} block pairs"
                    )

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
            for idx_i, img_i in enumerate(block_i_images):
                # For same block, only compare with images after this one
                start_idx = idx_i + 1 if block_i == block_j else 0

                for _idx_j, img_j in enumerate(
                    block_j_images[start_idx:], start=start_idx
                ):
                    # Check exact duplicates (same checksum)
                    if (
                        img_i["checksum"]
                        and img_j["checksum"]
                        and img_i["checksum"] == img_j["checksum"]
                    ):
                        batch_pairs.append(
                            {
                                "image_1": img_i["id"],
                                "image_2": img_j["id"],
                                "type": "exact",
                                "distance": 0,
                            }
                        )
                        pairs_found += 1
                        if len(batch_pairs) >= BATCH_SIZE:
                            _flush_pairs_to_db()
                        continue

                    # Check perceptual similarity using all three hash types
                    dhash_similar = False
                    ahash_similar = False
                    whash_similar = False
                    best_distance = 999

                    if img_i["dhash"] and img_j["dhash"]:
                        dhash_distance = _hamming_distance(
                            img_i["dhash"], img_j["dhash"]
                        )
                        dhash_similar = dhash_distance <= similarity_threshold
                        best_distance = min(best_distance, dhash_distance)

                    if img_i["ahash"] and img_j["ahash"]:
                        ahash_distance = _hamming_distance(
                            img_i["ahash"], img_j["ahash"]
                        )
                        ahash_similar = ahash_distance <= similarity_threshold
                        best_distance = min(best_distance, ahash_distance)

                    # whash uses a slightly tighter threshold (more robust = stricter)
                    whash_threshold = max(1, similarity_threshold - 1)
                    if img_i["whash"] and img_j["whash"]:
                        whash_distance = _hamming_distance(
                            img_i["whash"], img_j["whash"]
                        )
                        whash_similar = whash_distance <= whash_threshold
                        best_distance = min(best_distance, whash_distance)

                    # Consider similar if any hash matches
                    if dhash_similar or ahash_similar or whash_similar:
                        batch_pairs.append(
                            {
                                "image_1": img_i["id"],
                                "image_2": img_j["id"],
                                "type": "similar",
                                "distance": best_distance,
                            }
                        )
                        pairs_found += 1
                        if len(batch_pairs) >= BATCH_SIZE:
                            _flush_pairs_to_db()

            pairs_processed += 1

            # Publish progress every 10 block pairs
            if pairs_processed % 10 == 0:
                try:
                    from .progress_publisher import publish_progress

                    publish_progress(
                        job_id=parent_job_id,
                        state="PROGRESS",
                        current=pairs_processed,
                        total=num_pairs,
                        message=f"Comparing batch: {pairs_processed}/{num_pairs} block pairs done, {pairs_found} duplicates found",
                        extra={
                            "phase": "comparing",
                            "pairs_processed": pairs_processed,
                            "pairs_found": pairs_found,
                        },
                    )
                except Exception:
                    pass

        # Flush any remaining pairs to database
        _flush_pairs_to_db()

        logger.info(
            f"[{worker_id}] Completed {num_pairs} block pairs: found {pairs_found} duplicate pairs"
        )

        # Publish final progress
        try:
            from .progress_publisher import publish_progress

            publish_progress(
                job_id=parent_job_id,
                state="PROGRESS",
                current=num_pairs,
                total=num_pairs,
                message=f"Batch complete: {num_pairs} block pairs, {pairs_found} duplicates found",
                extra={
                    "phase": "comparing",
                    "pairs_processed": num_pairs,
                    "pairs_found": pairs_found,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to publish comparison progress: {e}")

        # Return only counts, not the actual pairs (to avoid Redis size limits)
        return {
            "status": "completed",
            "block_pairs_processed": num_pairs,
            "pairs_found": pairs_found,
        }

    except JobCancelledException as e:
        logger.warning(f"[{worker_id}] Comparison worker cancelled: {e}")
        return {
            "status": "cancelled",
            "block_pairs_processed": (
                pairs_processed if "pairs_processed" in locals() else 0
            ),
            "pairs_found": pairs_found if "pairs_found" in locals() else 0,
            "message": str(e),
        }

    except Exception as e:
        logger.error(f"[{worker_id}] Comparison worker failed: {e}", exc_info=True)
        # Increment failure counter in job record
        try:
            _increment_job_failure_count(parent_job_id)
        except Exception as count_error:
            logger.warning(f"Failed to increment failure count: {count_error}")
        return {
            "status": "failed",
            "error": str(e),
            "block_pairs_processed": 0,
            "pairs_found": 0,
        }


def _create_duplicate_group_tags(
    catalog_id: str, groups_data: List[Dict[str, Any]]
) -> None:
    """
    Create tags for duplicate groups.

    Creates a tag for each duplicate group with format: dup-{first 8 chars of dhash}
    The full dhash is stored in the tag's description field.

    Args:
        catalog_id: UUID of the catalog
        groups_data: List of dicts with keys: primary_id, members, dhash
    """
    if not groups_data:
        return

    logger.info(
        f"Creating tags for {len(groups_data)} duplicate groups in catalog {catalog_id}"
    )

    with CatalogDatabase(catalog_id) as db:
        for group_data in groups_data:
            dhash = group_data.get("dhash")
            if not dhash or len(dhash) < 8:
                continue

            # Tag format: dup-{first 8 chars of dhash}
            tag_name = f"dup-{dhash[:8]}"

            # Description contains full dhash for reference
            description = f"Duplicate group with dhash: {dhash}"

            # Check if tag already exists (avoid duplicates)
            assert db.session is not None
            result = db.session.execute(
                text(
                    """
                    SELECT id FROM tags
                    WHERE catalog_id = :catalog_id AND name = :name
                """
                ),
                {"catalog_id": catalog_id, "name": tag_name},
            )
            existing_tag = result.fetchone()

            if existing_tag:
                tag_id = existing_tag[0]
                logger.debug(f"Tag {tag_name} already exists with id {tag_id}")
            else:
                # Create new tag
                assert db.session is not None
                insert_result = db.session.execute(
                    text(
                        """
                        INSERT INTO tags (catalog_id, name, category, description, created_at)
                        VALUES (:catalog_id, :name, :category, :description, NOW())
                        RETURNING id
                    """
                    ),
                    {
                        "catalog_id": catalog_id,
                        "name": tag_name,
                        "category": "duplicate",
                        "description": description,
                    },
                )
                tag_id = insert_result.scalar()
                logger.debug(f"Created tag {tag_name} with id {tag_id}")

            # Tag all members of the group
            for member_id in group_data.get("members", []):
                assert db.session is not None
                db.session.execute(
                    text(
                        """
                        INSERT INTO image_tags (image_id, tag_id, confidence, source, created_at)
                        VALUES (:image_id, :tag_id, :confidence, :source, NOW())
                        ON CONFLICT (image_id, tag_id) DO NOTHING
                    """
                    ),
                    {
                        "image_id": member_id,
                        "tag_id": tag_id,
                        "confidence": 1.0,
                        "source": "duplicate_detection",
                    },
                )

        assert db.session is not None
        db.session.commit()
        logger.info(
            f"Successfully created tags for {len(groups_data)} duplicate groups"
        )


@app.task(bind=True, base=ProgressTask, name="duplicates_finalizer")
def duplicates_finalizer_task(
    self: ProgressTask,
    catalog_id: str,
    parent_job_id: str,
    total_images: int,
    similarity_threshold: int = 5,
) -> Dict[str, Any]:
    """Finalizer that builds and saves duplicate groups from comparison pairs.

    If there are too many failed comparisons, automatically queues a continuation
    job to retry the duplicate detection.

    Note: Worker results are NOT passed to this task to avoid Redis memory issues.
    Failure counts are tracked in the job's result field instead.
    """
    finalizer_id = self.request.id or "unknown"
    logger.info(f"[{finalizer_id}] Starting finalizer for job {parent_job_id}")

    try:
        self.update_progress(
            0, 1, "Building duplicate groups...", {"phase": "finalizing"}
        )

        # Collect all duplicate pairs from database (not from results to avoid Redis size limits)
        all_pairs = []
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            result = db.session.execute(
                text(
                    """
                    SELECT image_1, image_2, type, distance
                    FROM duplicate_pairs_temp
                    WHERE job_id = :job_id
                    """
                ),
                {"job_id": parent_job_id},
            )
            for row in result:
                all_pairs.append(
                    {
                        "image_1": row[0],
                        "image_2": row[1],
                        "type": row[2],
                        "distance": row[3],
                    }
                )

        logger.info(
            f"[{finalizer_id}] Collected {len(all_pairs)} duplicate pairs from database"
        )

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
        group_sizes = [len(g) for g in groups]
        logger.info(
            f"[{finalizer_id}] Built {len(groups)} duplicate groups. "
            f"Sizes: min={min(group_sizes) if group_sizes else 0}, "
            f"max={max(group_sizes) if group_sizes else 0}, "
            f"avg={sum(group_sizes) / len(group_sizes) if group_sizes else 0:.1f}"
        )

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
                        row = query_result.fetchone()  # type: ignore
                        image_quality[img_id] = (
                            row[0] if row and row[0] is not None else 0.0
                        )

        # Save groups to database using correct schema
        # duplicate_groups: id (SERIAL), catalog_id, primary_image_id, similarity_type, confidence, reviewed
        # duplicate_members: group_id, image_id, similarity_score
        total_duplicates = 0

        # Build a mapping of image pairs to their best distance for similarity scores
        pair_distances: Dict[tuple, int] = {}
        for pair in all_pairs:
            key = tuple(sorted([pair["image_1"], pair["image_2"]]))
            dist = pair.get("distance", 0)
            if key not in pair_distances or dist < pair_distances[key]:
                pair_distances[key] = dist

        # Collect group data for tag creation
        groups_data_for_tags = []

        with CatalogDatabase(catalog_id) as db:
            groups_processed = 0
            for group_images in groups:
                if len(group_images) < 2:
                    continue

                groups_processed += 1
                # Log progress for large groups
                if len(group_images) > 1000:
                    logger.info(
                        f"[{finalizer_id}] Processing large group {groups_processed}/{len(groups)} "
                        f"with {len(group_images)} images"
                    )

                # Select primary (highest quality)
                primary_id = max(group_images, key=lambda x: image_quality.get(x, 0))

                # Get dhash from primary image for tag creation
                assert db.session is not None
                dhash_result = db.session.execute(
                    text("SELECT dhash FROM images WHERE id = :id"),
                    {"id": primary_id},
                )
                dhash_row = dhash_result.fetchone()
                primary_dhash = dhash_row[0] if dhash_row else None

                # Determine similarity type based on best distance in the group
                # Optimized: only check pairs that actually exist (O(m) instead of O(n²))
                group_distances = []
                group_images_set = set(group_images)
                for (img1, img2), distance in pair_distances.items():
                    if img1 in group_images_set and img2 in group_images_set:
                        group_distances.append(distance)

                best_distance = min(group_distances) if group_distances else 0
                # exact = distance 0, similar = distance > 0
                similarity_type = "exact" if best_distance == 0 else "perceptual"
                # Confidence: 100 for exact, scale down for higher distances
                confidence = max(0, 100 - best_distance * 10)

                # Insert group and get the auto-generated ID
                assert db.session is not None
                insert_result: Result[Any] = db.session.execute(
                    text(
                        """
                        INSERT INTO duplicate_groups (
                            catalog_id, primary_image_id, similarity_type, confidence, reviewed, created_at
                        ) VALUES (
                            :catalog_id, :primary_id, :similarity_type, :confidence, false, NOW()
                        )
                        RETURNING id
                    """
                    ),
                    {
                        "catalog_id": catalog_id,
                        "primary_id": primary_id,
                        "similarity_type": similarity_type,
                        "confidence": confidence,
                    },
                )
                group_id = insert_result.scalar()

                # Insert group members into duplicate_members table
                for img_id in group_images:
                    # Calculate similarity score for this member
                    key = tuple(sorted([primary_id, img_id]))
                    member_distance = pair_distances.get(key, 0)
                    # Convert distance to similarity score (0-100)
                    similarity_score = max(0, 100 - member_distance * 10)

                    assert db.session is not None
                    db.session.execute(
                        text(
                            """
                            INSERT INTO duplicate_members (group_id, image_id, similarity_score)
                            VALUES (:group_id, :image_id, :similarity_score)
                        """
                        ),
                        {
                            "group_id": group_id,
                            "image_id": img_id,
                            "similarity_score": similarity_score,
                        },
                    )

                total_duplicates += (
                    len(group_images) - 1
                )  # Don't count primary as duplicate

                # Collect data for tag creation
                if primary_dhash:
                    groups_data_for_tags.append(
                        {
                            "primary_id": primary_id,
                            "members": group_images,
                            "dhash": primary_dhash,
                        }
                    )

            assert db.session is not None
            db.session.commit()

        # Create tags for duplicate groups
        if groups_data_for_tags:
            try:
                _create_duplicate_group_tags(catalog_id, groups_data_for_tags)
            except Exception as e:
                logger.warning(
                    f"[{finalizer_id}] Failed to create duplicate group tags: {e}"
                )

        # Get worker failure count from job record (tracked in database, not passed through Redis)
        failed_comparisons = 0
        try:
            from ..db import get_db_context

            with get_db_context() as session:
                job = session.query(Job).filter(Job.id == parent_job_id).first()
                if job and job.result and "worker_failures" in job.result:
                    failed_comparisons = job.result["worker_failures"]
        except Exception as e:
            logger.warning(f"[{finalizer_id}] Failed to get worker failure count: {e}")

        # If there were too many failed comparisons, auto-requeue to continue
        if failed_comparisons >= CONSECUTIVE_FAILURE_THRESHOLD:
            logger.warning(
                f"[{finalizer_id}] {failed_comparisons} comparisons failed, auto-requeuing continuation"
            )

            cancel_and_requeue_job(
                parent_job_id=parent_job_id,
                catalog_id=catalog_id,
                job_type="duplicates",
                task_name="duplicates_coordinator",
                task_kwargs={
                    "catalog_id": catalog_id,
                    "similarity_threshold": similarity_threshold,
                    "recompute_hashes": False,  # Don't recompute, hashes are saved
                    "batch_size": 1000,  # default batch size
                },
                reason=f"{failed_comparisons} comparison failures",
                processed_so_far=len(groups),
            )

            return {
                "status": "requeued",
                "catalog_id": catalog_id,
                "total_images": total_images,
                "duplicate_groups": len(groups),
                "total_duplicates": total_duplicates,
                "failed_comparisons": failed_comparisons,
                "message": f"Job requeued due to {failed_comparisons} comparison failures",
            }

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

        # Clean up temporary table
        try:
            with CatalogDatabase(catalog_id) as db:
                assert db.session is not None
                db.session.execute(
                    text(
                        """
                        DELETE FROM duplicate_pairs_temp
                        WHERE job_id = :job_id
                        """
                    ),
                    {"job_id": parent_job_id},
                )
                db.session.commit()
                logger.info(f"[{finalizer_id}] Cleaned up temporary duplicate pairs")
        except Exception as e:
            logger.warning(f"[{finalizer_id}] Failed to clean up temp table: {e}")

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


def _can_add_to_group(img: str, group: set, graph: Dict[str, set]) -> bool:
    """
    Check if an image is similar to ALL members of a group.

    Args:
        img: Image ID to check
        group: Set of image IDs in the current group
        graph: Adjacency list mapping image_id -> set of similar image_ids

    Returns:
        True if img is similar to all group members, False otherwise
    """
    if img in group:
        return True

    neighbors = graph.get(img, set())
    return all(member in neighbors for member in group)


def _build_duplicate_groups(pairs: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Build duplicate groups using greedy maximal cliques.

    Only groups images where EVERY image is similar to EVERY other image.
    This prevents transitive closure mega-groups where A-B-C get grouped
    even when A and C are not similar to each other.

    Algorithm:
    1. Build similarity graph from pairs
    2. Sort pairs by distance (most similar first)
    3. Try to add each pair to existing groups where both images are similar to ALL members
    4. If no compatible group exists, create new group

    Args:
        pairs: List of {image_1, image_2, type, distance} dicts

    Returns:
        List of image ID lists, each representing a duplicate group
    """
    if not pairs:
        return []

    # Build adjacency list: image_id -> set of similar image_ids
    graph: Dict[str, set] = {}

    for pair in pairs:
        img1, img2 = pair["image_1"], pair["image_2"]

        # Add edges (bidirectional)
        if img1 not in graph:
            graph[img1] = set()
        if img2 not in graph:
            graph[img2] = set()

        graph[img1].add(img2)
        graph[img2].add(img1)

    # Sort pairs by distance (most similar first = lowest distance)
    sorted_pairs = sorted(pairs, key=lambda p: p["distance"])

    # Greedy group building
    groups: List[set] = []
    assigned: set = set()  # Track images already in groups

    for pair in sorted_pairs:
        img1, img2 = pair["image_1"], pair["image_2"]

        # Try to add this pair to an existing group
        added = False
        for dup_group in groups:
            # Check if both images are similar to ALL group members
            if _can_add_to_group(img1, dup_group, graph) and _can_add_to_group(
                img2, dup_group, graph
            ):
                dup_group.add(img1)
                dup_group.add(img2)
                assigned.add(img1)
                assigned.add(img2)
                added = True
                break

        # If couldn't add to existing group, create new group
        if not added:
            groups.append({img1, img2})
            assigned.add(img1)
            assigned.add(img2)

    # Convert sets to lists, filter groups with size >= 2
    return [list(g) for g in groups if len(g) >= 2]
