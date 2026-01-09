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

# flake8: noqa: B023

from __future__ import annotations

import logging
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from celery import chord, group
from sqlalchemy import text
from sqlalchemy.engine import Result

from ..db import CatalogDB as CatalogDatabase
from ..db import get_db_context
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
from .tasks import CoordinatorTask, ProgressTask

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


@app.task(bind=True, base=CoordinatorTask, name="duplicates_coordinator")
def duplicates_coordinator_task(
    self: CoordinatorTask,
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

            # Start progress monitoring using generic monitor (job_batches mode)
            from .coordinator import start_chord_progress_monitor

            start_chord_progress_monitor(
                parent_job_id=parent_job_id,
                catalog_id=catalog_id,
                job_type="duplicates_hash",
                use_celery_backend=False,  # Use job_batches table for tracking
                countdown=30,
            )

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

        # Load all image hashes from database (filter out problematic images and bursts)
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            result = db.session.execute(
                text(
                    """
                    SELECT id, dhash, ahash, whash, checksum, source_path, quality_score
                    FROM images
                    WHERE catalog_id = :catalog_id
                    AND file_type = 'image'
                    AND burst_id IS NULL
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

        # Permanent duplicate_pairs table exists (created by migration)
        # No longer need to create temp table
        logger.info(f"[{task_id}] Using permanent duplicate_pairs table")

        # Divide images into blocks for parallel comparison
        # Each block pair (i,j) will be compared by a worker
        # For N images divided into B blocks, we have B*(B+1)/2 comparisons
        block_size = (
            250  # Images per block (reduced from 500 to prevent worker timeouts)
        )
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
        pairs_per_batch = 50  # Process 50 block pairs per worker (reduced from 100 to prevent timeouts)
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

        # Store worker task IDs for progress tracking
        chord_result = chord(group(comparison_tasks))(finalizer)

        # Start progress monitoring task using generic monitor
        from .coordinator import start_chord_progress_monitor

        start_chord_progress_monitor(
            parent_job_id=parent_job_id,
            catalog_id=catalog_id,
            job_type="duplicate_comparison",
            expected_workers=num_worker_tasks,
            use_celery_backend=True,
            comparison_start_time=datetime.utcnow().isoformat(),
            countdown=30,
        )

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
                    # Batch insert all pairs into permanent table
                    for pair in batch_pairs:
                        # Ensure consistent ordering (image_1 < image_2)
                        img1 = pair["image_1"]
                        img2 = pair["image_2"]
                        if img1 > img2:
                            img1, img2 = img2, img1

                        db.session.execute(
                            text(
                                """
                                INSERT INTO duplicate_pairs
                                (catalog_id, image_1, image_2, hash_type, distance, job_id)
                                VALUES (:catalog_id, :img1, :img2, :hash_type, :dist, :job_id)
                                ON CONFLICT (catalog_id, image_1, image_2, hash_type) DO UPDATE
                                    SET distance = EXCLUDED.distance,
                                        compared_at = NOW(),
                                        job_id = EXCLUDED.job_id
                                """
                            ),
                            {
                                "catalog_id": catalog_id,
                                "img1": img1,
                                "img2": img2,
                                "hash_type": pair["type"],
                                "dist": pair["distance"],
                                "job_id": parent_job_id,
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
            0, 1, "Counting duplicate pairs...", {"phase": "finalizing"}
        )

        # First, count total pairs without loading into memory
        total_pairs = 0
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            count_result = db.session.execute(
                text(
                    """
                    SELECT COUNT(*) FROM duplicate_pairs
                    WHERE catalog_id = :catalog_id AND job_id = :job_id
                    """
                ),
                {"catalog_id": catalog_id, "job_id": parent_job_id},
            )
            total_pairs = count_result.scalar() or 0

        logger.info(f"[{finalizer_id}] Found {total_pairs} duplicate pairs to process")

        if total_pairs == 0:
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

        # ALWAYS use database-backed incremental grouping to avoid OOM
        # Never build full graph in memory regardless of dataset size
        logger.info(
            f"[{finalizer_id}] Using database-backed incremental grouping for {total_pairs:,} pairs"
        )
        groups = _build_groups_incrementally_from_db(
            catalog_id=catalog_id,
            parent_job_id=parent_job_id,
            total_pairs=total_pairs,
            finalizer_id=finalizer_id,
            progress_callback=self.update_progress,
        )
        # Convert sets to lists for consistency
        duplicate_groups = [list(group) for group in groups]

        logger.info(
            f"[{finalizer_id}] Built {len(duplicate_groups)} groups from {total_pairs:,} pairs"
        )

        # Assign back to groups variable for downstream processing
        groups = duplicate_groups

        # Now load pair_distances ONLY for images actually in groups (much smaller dataset)
        self.update_progress(
            0,
            1,
            "Loading distances for grouped images only...",
            {"phase": "loading_distances"},
        )

        # Get set of all images in groups
        images_in_groups = set()
        for img_group in groups:
            images_in_groups.update(img_group)

        logger.info(
            f"[{finalizer_id}] Loading distances for {len(images_in_groups)} images in {len(groups)} groups"
        )

        pair_distances: Dict[tuple, int] = {}
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            # Query only pairs where both images are in groups
            placeholders = ",".join([f":img{i}" for i in range(len(images_in_groups))])
            params = {f"img{i}": img_id for i, img_id in enumerate(images_in_groups)}
            params["catalog_id"] = catalog_id
            params["job_id"] = parent_job_id

            result = db.session.execute(
                text(
                    f"""
                    SELECT image_1, image_2, distance
                    FROM duplicate_pairs
                    WHERE catalog_id = :catalog_id
                      AND job_id = :job_id
                      AND image_1 IN ({placeholders})
                      AND image_2 IN ({placeholders})
                    """
                ),
                params,
            )

            for row in result:
                key = tuple(sorted([row[0], row[1]]))
                dist = row[2]
                if key not in pair_distances or dist < pair_distances[key]:
                    pair_distances[key] = dist

        logger.info(
            f"[{finalizer_id}] Loaded {len(pair_distances)} pair distances for similarity scoring"
        )
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

        # pair_distances was already built during streaming (line 1104-1106)
        # No need to rebuild it here

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

        # Pairs are now stored permanently - no cleanup needed
        logger.info(
            f"[{finalizer_id}] Duplicate pairs stored permanently in duplicate_pairs table"
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


def _build_groups_incrementally_from_db(
    catalog_id: str,
    parent_job_id: str,
    total_pairs: int,
    finalizer_id: str,
    progress_callback: Optional[Any],
) -> List[List[str]]:
    """
    Build duplicate groups efficiently using Union-Find algorithm.

    This is ~200-400x faster than the old incremental clique algorithm.
    Handles 10M+ pairs in 1-2 minutes instead of 6-8 hours.

    Algorithm:
    1. Load all pairs into memory as set (for O(1) lookup)
    2. Build connected components using Union-Find (O(n × α(n)))
    3. For each component, validate/decompose into maximal cliques

    Args:
        catalog_id: UUID of the catalog
        parent_job_id: Job ID for querying duplicate_pairs
        total_pairs: Total number of pairs (for progress tracking)
        finalizer_id: ID for logging
        progress_callback: Optional callback for progress updates

    Returns:
        List of image ID lists, each representing a duplicate group
    """
    logger.info(
        f"[{finalizer_id}] Using efficient Union-Find grouping for {total_pairs:,} pairs"
    )

    # Phase 1: Load all pairs WITH DISTANCES into memory
    # 10M pairs × ~110 bytes each ≈ 1.1GB (acceptable)
    logger.info(
        f"[{finalizer_id}] Phase 1: Loading pairs with distances from database..."
    )

    pairs_set = set()
    pairs_list = []
    pairs_distances = {}  # (img1, img2) -> distance

    with CatalogDatabase(catalog_id) as db:
        assert db.session is not None
        result = db.session.execute(
            text(
                """
                SELECT image_1, image_2, distance
                FROM duplicate_pairs
                WHERE catalog_id = :catalog_id AND job_id = :job_id
                ORDER BY distance ASC
                """
            ),
            {"catalog_id": catalog_id, "job_id": parent_job_id},
        )

        for row in result:
            img1, img2, distance = row[0], row[1], row[2]
            pair = (img1, img2)
            pairs_set.add(pair)
            pairs_list.append(pair)
            pairs_distances[pair] = distance
            # Also store reverse for lookup
            pairs_distances[(img2, img1)] = distance

    logger.info(f"[{finalizer_id}] Loaded {len(pairs_list):,} pairs into memory")

    if progress_callback:
        progress_callback(
            len(pairs_list),
            total_pairs,
            f"Loaded {len(pairs_list):,} pairs",
            {"phase": "union_find_grouping"},
        )

    # Phase 2: Continuity segmentation - find natural breaks in hash distances
    # Pairs are already sorted by distance (ASC) from database query
    CONTINUITY_GAP = 2  # Detect break if distance jumps by more than this

    logger.info(
        f"[{finalizer_id}] Phase 2: Segmenting by hash continuity (gap threshold: {CONTINUITY_GAP})..."
    )

    segments: List[List[Tuple[str, str]]] = (
        []
    )  # List of segments, each segment is a list of pairs
    current_segment = []
    last_distance = None

    for i, pair in enumerate(pairs_list):
        distance = pairs_distances[pair]

        # Detect discontinuity (significant jump in distance)
        if last_distance is not None and (distance - last_distance) > CONTINUITY_GAP:
            # Found a break - save current segment
            if current_segment:
                segments.append(current_segment)
                current_segment = []

        current_segment.append(pair)
        last_distance = distance

        # Progress every 1M pairs
        if (i + 1) % 1_000_000 == 0:
            logger.info(
                f"[{finalizer_id}] Continuity segmentation: {i + 1:,} / {total_pairs:,} pairs, "
                f"{len(segments)} segments found"
            )

    # Don't forget the last segment
    if current_segment:
        segments.append(current_segment)

    logger.info(
        f"[{finalizer_id}] Continuity segmentation complete: {len(segments)} segments from {total_pairs:,} pairs"
    )

    # Phase 3: Validate each segment with Union-Find
    logger.info(f"[{finalizer_id}] Phase 3: Validating segments with Union-Find...")

    components: Dict[str, List[str]] = {}  # Final components after validation
    total_validated = 0

    for seg_idx, segment_pairs in enumerate(segments):
        # Run Union-Find on this segment only
        parent_map = {}
        rank_map = {}

        def find(x: str) -> str:
            """Find root with path compression."""
            if x not in parent_map:
                parent_map[x] = x
                rank_map[x] = 0
            if parent_map[x] != x:
                parent_map[x] = find(parent_map[x])
            return parent_map[x]

        def union(x: str, y: str) -> None:
            """Union by rank."""
            px, py = find(x), find(y)
            if px == py:
                return
            if rank_map[px] < rank_map[py]:
                parent_map[px] = py
            elif rank_map[px] > rank_map[py]:
                parent_map[py] = px
            else:
                parent_map[py] = px
                rank_map[px] += 1

        # Union all pairs in this segment
        for img1, img2 in segment_pairs:
            union(img1, img2)

        # Extract components from this segment
        segment_components: Dict[str, List[str]] = {}
        for img in parent_map:
            root = find(img)
            if root not in segment_components:
                segment_components[root] = []
            segment_components[root].append(img)

        # Merge into global components (with unique keys)
        for component in segment_components.values():
            if len(component) >= 2:  # Only keep groups of 2+
                component_id = f"seg{seg_idx}_comp{len(components)}"
                components[component_id] = component
                total_validated += len(component)

        # Progress every 100 segments
        if (seg_idx + 1) % 100 == 0:
            logger.info(
                f"[{finalizer_id}] Validated {seg_idx + 1:,} / {len(segments)} segments, "
                f"{len(components)} components found"
            )

    logger.info(
        f"[{finalizer_id}] Validation complete: {len(components)} components from "
        f"{len(segments)} segments ({total_validated:,} images)"
    )

    if progress_callback:
        progress_callback(
            total_pairs,
            total_pairs,
            f"Found {len(components)} components",
            {"phase": "component_grouping"},
        )

    # Phase 4: Validate and decompose cliques
    logger.info(f"[{finalizer_id}] Phase 4: Validating/decomposing cliques...")

    cliques: List[List[str]] = []
    for i, (_root, images) in enumerate(components.items()):
        if len(images) < 2:
            continue

        # Progress every 1000 components
        if progress_callback and i % 1000 == 0 and i > 0:
            progress_callback(
                i,
                len(components),
                f"Processing component {i:,} / {len(components):,} ({len(cliques)} cliques so far)",
                {"phase": "clique_validation"},
            )
            logger.info(
                f"[{finalizer_id}] Clique validation: {i:,} / {len(components):,} components, "
                f"{len(cliques)} cliques found"
            )

        # Decompose component into constrained groups
        # Size limit prevents drift, diameter prevents loose groups
        sub_cliques = _constrained_grouping(
            images, pairs_set, pairs_distances, finalizer_id
        )
        cliques.extend(sub_cliques)

    logger.info(
        f"[{finalizer_id}] Clique decomposition complete: {len(cliques)} groups from "
        f"{total_pairs:,} pairs across {len(components)} components"
    )

    if progress_callback:
        progress_callback(
            len(components),
            len(components),
            f"Built {len(cliques)} duplicate groups",
            {"phase": "complete"},
        )

    return cliques


def _is_complete_clique(images: List[str], pairs_set: Set[Tuple[str, str]]) -> bool:
    """
    Check if images form a complete clique (all pairs exist).

    Args:
        images: List of image IDs
        pairs_set: Set of (image_1, image_2) pairs

    Returns:
        True if all pairs exist (complete graph), False otherwise
    """
    n = len(images)
    for i in range(n):
        for j in range(i + 1, n):
            pair = tuple(sorted([images[i], images[j]]))
            if pair not in pairs_set:
                return False
    return True


def _constrained_grouping(
    images: List[str],
    pairs_set: Set[Tuple[str, str]],
    pairs_distances: Dict[Tuple[str, str], int],
    finalizer_id: str = None,
) -> List[List[str]]:
    """
    Build constrained duplicate groups using hash distance windows.

    Constraints prevent drift and computation explosion:
    1. **Size limit**: Max 20 images per group (soft guideline)
    2. **Diameter constraint**: Max distance between ANY two images ≤ 8
    3. **Distance window**: Only add images within distance ≤ 3 from seed

    This balances:
    - **Local (transitivity)**: Allow A→B→C connections
    - **Global (complete linkage)**: Prevent drift (A and C must be similar)

    Args:
        images: List of image IDs in component
        pairs_set: Set of (image_1, image_2) pairs
        pairs_distances: Dict of (img1, img2) -> distance
        finalizer_id: Optional ID for logging

    Returns:
        List of constrained groups (each group is a list of image IDs)
    """
    MAX_GROUP_SIZE = 20  # Soft limit on group size
    MAX_DIAMETER = 8  # Max distance between ANY two images in group
    SEED_WINDOW = 3  # Only add images within this distance from seed

    total = len(images)

    if finalizer_id:
        logger.info(
            f"[{finalizer_id}] Building constrained groups for {total:,} nodes "
            f"(max_size={MAX_GROUP_SIZE}, max_diameter={MAX_DIAMETER}, seed_window={SEED_WINDOW})..."
        )

    # Pre-compute adjacency lists with distances
    adj: Dict[str, Dict[str, int]] = {
        img: {} for img in images
    }  # img -> {neighbor: distance}
    for img1, img2 in pairs_set:
        if img1 in adj and img2 in adj:
            dist = pairs_distances.get(
                (img1, img2), pairs_distances.get((img2, img1), 999)
            )
            adj[img1][img2] = dist
            adj[img2][img1] = dist

    if finalizer_id:
        total_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
        logger.info(f"[{finalizer_id}] Adjacency built: {total_edges:,} edges")

    remaining = set(images)
    groups = []
    processed = 0

    while remaining:
        # Find node with highest degree (most connections)
        degrees = {
            img: len([n for n, d in adj[img].items() if n in remaining])
            for img in remaining
        }
        seed = max(degrees, key=lambda x: degrees[x])

        # Start group with seed
        group = [seed]
        remaining.remove(seed)

        # Get candidates: neighbors of seed within SEED_WINDOW distance
        candidates = [
            (n, d) for n, d in adj[seed].items() if n in remaining and d <= SEED_WINDOW
        ]
        # Sort by distance (closest first)
        candidates.sort(key=lambda x: x[1])

        # Greedily add candidates
        for candidate, _seed_dist in candidates:
            if len(group) >= MAX_GROUP_SIZE:
                break  # Hit size limit

            # Check diameter constraint: candidate must be within MAX_DIAMETER of ALL group members
            max_dist = 0
            valid = True
            for member in group:
                sorted_pair = sorted([candidate, member])
                pair = (sorted_pair[0], sorted_pair[1])
                dist = pairs_distances.get(pair, 999)
                if dist > MAX_DIAMETER:
                    valid = False
                    break
                max_dist = max(max_dist, dist)

            if valid:
                group.append(candidate)
                remaining.discard(candidate)

        # Only keep groups with 2+ members
        if len(group) >= 2:
            groups.append(group)

        # Progress reporting
        processed += len(group)
        if finalizer_id and (processed % 1000 == 0 or len(remaining) == 0):
            pct = (processed / total) * 100
            logger.info(
                f"[{finalizer_id}] Constrained grouping: {processed:,} / {total:,} nodes "
                f"({pct:.1f}%), {len(groups)} groups, {len(remaining):,} remaining"
            )

    if finalizer_id:
        avg_size = sum(len(g) for g in groups) / len(groups) if groups else 0
        logger.info(
            f"[{finalizer_id}] Built {len(groups)} constrained groups, avg size: {avg_size:.1f}"
        )

    return groups


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


def _build_duplicate_groups_from_graph(
    graph: Dict[str, set], pair_distances: Optional[Dict[tuple, int]] = None
) -> List[List[str]]:
    """
    Build duplicate groups from a pre-built similarity graph.

    This is a memory-efficient version that works with a graph already built
    from streaming pairs, avoiding the need to load all pairs into memory.

    Args:
        graph: Adjacency list mapping image_id -> set of similar image_ids
        pair_distances: Optional mapping of (img1, img2) -> distance for sorting

    Returns:
        List of image ID lists, each representing a duplicate group
    """
    if not graph:
        return []

    # Extract unique pairs from graph (each edge appears twice, so deduplicate)
    pairs_set = set()
    for img1 in graph:
        for img2 in graph[img1]:
            # Use tuple with sorted IDs to ensure uniqueness
            pair = tuple(sorted([img1, img2]))
            pairs_set.add(pair)

    # Convert to list of pairs for sorting
    pairs_list = list(pairs_set)

    # Sort by distance if available (most similar first = lowest distance)
    if pair_distances:
        pairs_list.sort(key=lambda p: pair_distances.get(p, 999))

    # Greedy group building
    groups: List[set] = []
    assigned: set = set()  # Track images already in groups

    for img1, img2 in pairs_list:
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
