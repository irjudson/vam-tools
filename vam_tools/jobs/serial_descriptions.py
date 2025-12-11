"""
Serial Description Generation using Ollama.

This module implements serial (non-parallel) description generation using Ollama
vision models. Since Ollama can only handle one request at a time without
resource contention, this task processes images one at a time.

The task runs on a dedicated queue with concurrency=1 to ensure only one
Ollama request is processed at a time across all workers.

Failure Monitoring:
    The task monitors consecutive failures and will pause (enter PAUSED state)
    after hitting a configurable threshold. This allows the operator to restart
    Ollama before resuming the job. Resume by calling the task again with the
    same parameters - it will continue from where it left off (undescribed images).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import text

from ..db import CatalogDB as CatalogDatabase
from ..db.models import Job
from .celery_app import app
from .progress_publisher import publish_completion, publish_progress
from .tasks import ProgressTask

logger = logging.getLogger(__name__)

# Configuration for failure monitoring
CONSECUTIVE_FAILURE_THRESHOLD = 5  # Pause after this many consecutive failures
FAILURE_PAUSE_MESSAGE = (
    "Job paused due to {count} consecutive Ollama failures. "
    "Please restart Ollama (docker restart ollama) and run the job again to resume."
)


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


@app.task(
    bind=True,
    base=ProgressTask,
    name="generate_descriptions",
    # Route to a dedicated queue with concurrency=1
    queue="ollama",
    # Longer time limit since Ollama is slow
    soft_time_limit=3600,  # 1 hour soft limit
    time_limit=3900,  # 1 hour 5 min hard limit
)
def generate_descriptions_task(
    self: ProgressTask,
    catalog_id: str,
    model: str = "llava",
    mode: str = "undescribed_only",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate AI descriptions for images using Ollama vision model.

    This task processes images serially (one at a time) to avoid
    Ollama resource contention. For large catalogs, use the limit
    parameter to process in chunks.

    Args:
        catalog_id: UUID of the catalog
        model: Ollama vision model to use (llava, qwen3-vl, etc.)
        mode: "undescribed_only" or "all"
        limit: Maximum number of images to process (None = all)

    Returns:
        Dict with success count, failed count, and skipped count
    """
    job_id = self.request.id or "unknown"
    logger.info(
        f"[{job_id}] Starting description generation for catalog {catalog_id} "
        f"(model={model}, mode={mode}, limit={limit})"
    )

    try:
        self.update_progress(0, 1, "Initializing Ollama...", {"phase": "init"})

        # Initialize Ollama backend
        from ..analysis.image_tagger import OllamaBackend

        ollama_host = os.environ.get("OLLAMA_HOST")
        ollama = OllamaBackend(model=model, host=ollama_host)

        if not ollama.is_available():
            error_msg = f"Ollama not available or model {model} not found"
            logger.error(f"[{job_id}] {error_msg}")
            _update_job_status(job_id, "FAILURE", error=error_msg)
            return {"success": False, "error": error_msg}

        # Query images based on mode
        self.update_progress(0, 1, "Querying images...", {"phase": "query"})

        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None

            if mode == "undescribed_only":
                query = """
                    SELECT i.id, i.source_path FROM images i
                    WHERE i.catalog_id = :catalog_id
                    AND i.file_type = 'image'
                    AND (i.description IS NULL OR i.description = '')
                """
            else:
                query = """
                    SELECT i.id, i.source_path FROM images i
                    WHERE i.catalog_id = :catalog_id
                    AND i.file_type = 'image'
                """

            if limit:
                query += f" LIMIT {limit}"

            result = db.session.execute(text(query), {"catalog_id": catalog_id})
            image_data = [(str(row[0]), row[1]) for row in result.fetchall()]

        total_images = len(image_data)
        if total_images == 0:
            logger.info(f"[{job_id}] No images to process")
            result_data = {
                "success": True,
                "total": 0,
                "described": 0,
                "failed": 0,
                "skipped": 0,
            }
            _update_job_status(job_id, "SUCCESS", result=result_data)
            return result_data

        logger.info(f"[{job_id}] Processing {total_images} images")
        self.update_progress(
            0,
            total_images,
            f"Processing 0/{total_images} images...",
            {"phase": "process"},
        )

        # Process images serially
        described = 0
        failed = 0
        skipped = 0
        consecutive_failures = 0  # Track consecutive Ollama failures

        for idx, (image_id, source_path) in enumerate(image_data):
            try:
                source_file = Path(source_path)
                if not source_file.exists():
                    logger.warning(f"[{job_id}] Source file not found: {source_path}")
                    skipped += 1
                    # File not found doesn't count as Ollama failure
                    continue

                # Generate description
                description = ollama.describe_image(source_file)

                if not description:
                    logger.warning(f"[{job_id}] Empty description for {image_id}")
                    failed += 1
                    consecutive_failures += 1
                else:
                    # Save description to database
                    with CatalogDatabase(catalog_id) as db:
                        assert db.session is not None
                        db.session.execute(
                            text(
                                """
                                UPDATE images
                                SET description = :description,
                                    processing_flags = processing_flags || '{"description_generated": true}'::jsonb,
                                    updated_at = NOW()
                                WHERE id = :image_id
                            """
                            ),
                            {"description": description, "image_id": image_id},
                        )
                        db.session.commit()

                    described += 1
                    consecutive_failures = 0  # Reset on success

            except Exception as e:
                logger.warning(f"[{job_id}] Failed to describe {image_id}: {e}")
                failed += 1
                consecutive_failures += 1

            # Check if we've hit the consecutive failure threshold
            if consecutive_failures >= CONSECUTIVE_FAILURE_THRESHOLD:
                pause_message = FAILURE_PAUSE_MESSAGE.format(count=consecutive_failures)
                logger.error(f"[{job_id}] {pause_message}")

                # Update job status to PAUSED
                pause_result = {
                    "success": False,
                    "paused": True,
                    "total": total_images,
                    "processed": idx + 1,
                    "described": described,
                    "failed": failed,
                    "skipped": skipped,
                    "consecutive_failures": consecutive_failures,
                    "message": pause_message,
                }
                _update_job_status(job_id, "PAUSED", result=pause_result)
                publish_progress(
                    job_id=job_id,
                    state="PAUSED",
                    current=idx + 1,
                    total=total_images,
                    message=pause_message,
                    extra=pause_result,
                )

                # Return early - job will be resumed when user restarts Ollama
                # and triggers a new job (which will pick up undescribed images)
                return pause_result

            # Update progress
            progress = idx + 1
            self.update_progress(
                progress,
                total_images,
                f"Processing {progress}/{total_images} images...",
                {
                    "phase": "process",
                    "described": described,
                    "failed": failed,
                    "skipped": skipped,
                    "consecutive_failures": consecutive_failures,
                },
            )

            # Publish progress to Redis for SSE
            publish_progress(
                job_id=job_id,
                state="PROGRESS",
                current=progress,
                total=total_images,
                message=f"Described {described} images",
                extra={
                    "described": described,
                    "failed": failed,
                    "skipped": skipped,
                    "consecutive_failures": consecutive_failures,
                },
            )

        # Final result
        result_data = {
            "success": True,
            "total": total_images,
            "described": described,
            "failed": failed,
            "skipped": skipped,
        }

        logger.info(
            f"[{job_id}] Description generation complete: "
            f"{described} described, {failed} failed, {skipped} skipped"
        )

        _update_job_status(job_id, "SUCCESS", result=result_data)
        publish_completion(
            job_id=job_id,
            state="SUCCESS",
            result=result_data,
        )

        return result_data

    except Exception as e:
        logger.exception(f"[{job_id}] Description generation failed: {e}")
        _update_job_status(job_id, "FAILURE", error=str(e))
        publish_completion(job_id=job_id, state="FAILURE", error=str(e))
        raise
