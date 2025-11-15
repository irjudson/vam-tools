"""
Celery tasks for background processing of VAM Tools operations.

These tasks wrap the CLI operations and provide progress tracking
through Celery's state mechanism.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from celery import Task, chord, group

from vam_tools.core.types import (
    CatalogPhase,
    DateInfo,
    FileType,
    ImageMetadata,
    ImageRecord,
    ImageStatus,
    Statistics,
)

from ..db import CatalogDB as CatalogDatabase
from ..organization import (
    FileOrganizer,
    OrganizationOperation,
    OrganizationStrategy,
)
from ..shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path

# Import app here so that @app.task decorators can find it
from .celery_app import app

logger = logging.getLogger(__name__)


class ProgressTask(Task):
    """
    Base task class with progress reporting.

    Provides utilities for reporting progress to the result backend.
    """

    def update_progress(
        self,
        current: int,
        total: int,
        message: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update task progress.

        Args:
            current: Current item number
            total: Total items
            message: Progress message
            extra: Additional metadata
        """
        meta = {
            "current": current,
            "total": total,
            "percent": int((current / total) * 100) if total > 0 else 0,
            "message": message,
        }
        if extra:
            meta.update(extra)

        self.update_state(
            state="PROGRESS",
            meta=meta,
        )


@app.task(bind=True, base=ProgressTask, name="analyze_catalog")
def analyze_catalog_task(
    self: ProgressTask,
    catalog_path: str,
    source_directories: List[str],
    detect_duplicates: bool = False,
    force_reanalyze: bool = False,
    similarity_threshold: int = 5,
) -> Dict[str, Any]:
    """
    Analyze catalog in background.

    Args:
        catalog_path: Path to catalog directory
        source_directories: List of source directories to scan
        detect_duplicates: Whether to detect duplicates
        force_reanalyze: Reset catalog and analyze all files fresh
        similarity_threshold: Hamming distance threshold for duplicates

    Returns:
        Analysis results dictionary
    """
    logger.info(f"Starting catalog analysis: {catalog_path}")

    try:
        import os
        import shutil
        import uuid  # Added uuid import
        from datetime import datetime

        from ..analysis.scanner import _process_file_worker
        from ..shared.media_utils import get_file_type

        catalog_dir = Path(catalog_path)
        source_dirs = [Path(d) for d in source_directories]

        # Update initial state
        self.update_progress(0, 1, "Initializing catalog...", {"phase": "init"})

        # Initialize or load catalog
        with CatalogDatabase(catalog_dir) as db:
            catalog_exists = (
                catalog_dir / "catalog.db"
            ).exists() and not force_reanalyze

            if force_reanalyze and catalog_exists:
                # Clear existing catalog for fresh analysis
                logger.info("Force reanalyze: clearing existing catalog")
                self.update_progress(
                    0, 1, "Clearing existing catalog...", {"phase": "clearing"}
                )
                # Create backup
                backup_name = db.create_backup()
                logger.info(f"Backup saved to: {backup_name.name}")

                # Remove current catalog
                (catalog_dir / "catalog.db").unlink()
                catalog_exists = False

            if not catalog_exists:
                logger.info("Initializing new catalog")
                self.update_progress(
                    0, 1, "Initializing new catalog...", {"phase": "init"}
                )
                db.initialize()  # Initialize schema

                # Store source directories in catalog_config
                for src_dir in source_dirs:
                    db.execute(
                        "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                        (f"source_directory_{src_dir.name}", str(src_dir)),
                    )
                db.execute(
                    "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                    ("created", datetime.now().isoformat()),
                )
                db.execute(
                    "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                    ("last_updated", datetime.now().isoformat()),
                )
                db.execute(
                    "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                    ("catalog_id", str(uuid.uuid4())),
                )
                db.execute(
                    "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                    ("version", "2.0.0"),
                )
                db.execute(
                    "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                    ("phase", "analyzing"),
                )
            else:
                logger.info("Loading existing catalog")
                self.update_progress(
                    0, 1, "Loading existing catalog...", {"phase": "loading"}
                )

            # Update catalog state
            db.execute(
                "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ("phase", CatalogPhase.ANALYZING.value),
            )

        # Process files sequentially - simple and reliable
        self.update_progress(
            0, 1, "Discovering and processing files...", {"phase": "processing"}
        )

        files_added = 0
        files_skipped = 0
        files_processed = 0

        # Reopen database for processing
        with CatalogDatabase(catalog_dir) as db:
            # Get latest statistics or initialize new
            latest_stats_row = db.execute(
                "SELECT * FROM statistics ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            stats = Statistics(**latest_stats_row) if latest_stats_row else Statistics()

            for directory in source_dirs:
                logger.info(f"Scanning directory: {directory}")

                for root, dirs, files in os.walk(directory):
                    # Skip synology metadata directories
                    dirs[:] = [d for d in dirs if not d.startswith("@eaDir")]

                    root_path = Path(root)

                    for filename in files:
                        # Skip synology metadata files
                        if filename.startswith(".") or "@SynoResource" in filename:
                            continue

                        file_path = root_path / filename
                        file_type_str = get_file_type(file_path)

                        if file_type_str not in ("image", "video"):
                            continue

                        # Update progress every 10 files
                        if files_processed % 10 == 0:
                            self.update_progress(
                                files_processed,
                                files_processed + 1,
                                f"Processing {file_path.name}...",
                                {
                                    "phase": "processing",
                                    "added": files_added,
                                    "skipped": files_skipped,
                                },
                            )

                        # Process file
                        result = _process_file_worker(file_path)

                        if result is not None:
                            image, file_size = result
                            files_processed += 1

                            # Check if already in catalog
                            existing_image = db.execute(
                                "SELECT id FROM images WHERE id = ?", (image.checksum,)
                            ).fetchone()

                            if not existing_image:
                                # Add to catalog
                                db.execute(
                                    """
                                    INSERT INTO images (
                                        id, source_path, file_size, file_hash, format,
                                        width, height, created_at, modified_at, indexed_at,
                                        date_taken, camera_make, camera_model, lens_model,
                                        focal_length, aperture, shutter_speed, iso,
                                        gps_latitude, gps_longitude, quality_score, is_corrupted,
                                        perceptual_hash, features_vector
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        image.id,
                                        str(image.source_path),
                                        image.file_size,
                                        image.checksum,
                                        image.metadata.format,
                                        image.metadata.width,
                                        image.metadata.height,
                                        (
                                            image.created_at.isoformat()
                                            if image.created_at
                                            else None
                                        ),
                                        (
                                            image.modified_at.isoformat()
                                            if image.modified_at
                                            else None
                                        ),
                                        datetime.now().isoformat(),  # indexed_at
                                        (
                                            image.dates.selected_date.isoformat()
                                            if image.dates and image.dates.selected_date
                                            else None
                                        ),
                                        image.metadata.exif.get("Make"),
                                        image.metadata.exif.get("Model"),
                                        image.metadata.exif.get("LensModel"),
                                        image.metadata.exif.get("FocalLength"),
                                        image.metadata.exif.get("ApertureValue"),
                                        image.metadata.exif.get("ExposureTime"),
                                        image.metadata.exif.get("ISO"),
                                        (
                                            image.metadata.gps.latitude
                                            if image.metadata.gps
                                            else None
                                        ),
                                        (
                                            image.metadata.gps.longitude
                                            if image.metadata.gps
                                            else None
                                        ),
                                        image.quality_score,
                                        1 if image.is_corrupted else 0,
                                        (
                                            image.hashes.perceptual_hash
                                            if image.hashes
                                            else None
                                        ),
                                        None,  # features_vector not handled yet
                                    ),
                                )
                                files_added += 1

                                # Update statistics in memory
                                if image.file_type == FileType.IMAGE:
                                    stats.total_images += 1
                                elif image.file_type == FileType.VIDEO:
                                    stats.total_videos += 1

                                stats.total_size_bytes += file_size
                                stats.images_scanned += 1
                                stats.images_hashed += 1

                                if not image.dates.selected_date:
                                    stats.no_date += 1
                            else:
                                files_skipped += 1
                        else:
                            files_skipped += 1

            # Insert final statistics snapshot
            db.execute(
                """
                INSERT INTO statistics (
                    timestamp, total_images, total_videos, total_size_bytes,
                    images_scanned, images_hashed, images_tagged,
                    duplicate_groups, duplicate_images, potential_savings_bytes,
                    high_quality_count, medium_quality_count, low_quality_count,
                    corrupted_count, unsupported_count,
                    processing_time_seconds, images_per_second,
                    no_date, suspicious_dates, problematic_files
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    stats.total_images,
                    stats.total_videos,
                    stats.total_size_bytes,
                    stats.images_scanned,
                    stats.images_hashed,
                    stats.images_tagged,
                    stats.duplicate_groups,
                    stats.duplicate_images,
                    stats.potential_savings_bytes,
                    stats.high_quality_count,
                    stats.medium_quality_count,
                    stats.low_quality_count,
                    stats.corrupted_count,
                    stats.unsupported_count,
                    stats.processing_time_seconds,
                    stats.images_per_second,
                    stats.no_date,
                    stats.suspicious_dates,
                    stats.problematic_files,
                ),
            )

        logger.info(
            f"Analysis complete: {files_processed} processed, "
            f"{files_added} added, {files_skipped} skipped"
        )

        self.update_progress(
            100,
            100,
            f"Analysis complete: {files_added} files added",
            {"phase": "complete"},
        )

        return {
            "status": "completed",
            "catalog_path": str(catalog_dir),
            "total_files": files_processed,
            "processed": files_processed,
            "files_added": files_added,
            "files_skipped": files_skipped,
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


@app.task(name="process_file_batch")
def process_file_batch_task(
    catalog_path: str,
    file_paths: List[str],
    batch_num: int,
    total_batches: int,
) -> Dict[str, Any]:
    """
    Process a batch of files in parallel.

    This task can run in parallel with other batch tasks.

    Args:
        catalog_path: Path to catalog directory
        file_paths: List of file paths to process
        batch_num: Batch number (for logging)
        total_batches: Total number of batches

    Returns:
        Results dictionary with processed file records
    """
    logger.info(
        f"Processing batch {batch_num}/{total_batches} ({len(file_paths)} files)"
    )

    from ..analysis.scanner import _process_file_worker

    results = []
    processed = 0
    skipped = 0

    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        result = _process_file_worker(file_path)

        if result is not None:
            image, file_size = result
            processed += 1
            # Convert image to dict for JSON serialization
            results.append(
                {
                    "id": image.id,
                    "checksum": image.checksum,
                    "source_path": str(image.source_path),
                    "file_type": image.file_type.value,
                    "file_size": file_size,
                    "metadata": image.metadata.model_dump() if image.metadata else None,
                    "dates": image.dates.model_dump() if image.dates else None,
                }
            )
        else:
            skipped += 1

    logger.info(
        f"Batch {batch_num}/{total_batches} complete: "
        f"{processed} processed, {skipped} skipped"
    )

    return {
        "batch_num": batch_num,
        "results": results,
        "processed": processed,
        "skipped": skipped,
    }


@app.task(bind=True, base=ProgressTask, name="finalize_analysis")
def finalize_analysis_task(
    self: ProgressTask,
    batch_results: List[Dict[str, Any]],
    catalog_path: str,
) -> Dict[str, Any]:
    """
    Finalize analysis by saving all results to catalog.

    This task runs after all batch processing tasks complete.

    Args:
        batch_results: List of results from all batch tasks
        catalog_path: Path to catalog directory

    Returns:
        Final analysis results
    """
    logger.info(f"Finalizing analysis for {catalog_path}")

    try:
        # from ..core.database import CatalogDatabase # Removed redundant import
        # from ..core.types import DateInfo, FileType, ImageMetadata, ImageRecord, ImageStatus # This line is now at the top

        self.update_progress(0, 1, "Finalizing analysis...", {"phase": "finalizing"})

        catalog_dir = Path(catalog_path)
        files_added = 0
        files_skipped = 0
        total_processed = 0

        with CatalogDatabase(catalog_dir) as db:
            # Get latest statistics or initialize new
            latest_stats_row = db.execute(
                "SELECT * FROM statistics ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            stats = Statistics(**latest_stats_row) if latest_stats_row else Statistics()

            # Process all batch results
            for batch_result in batch_results:
                total_processed += batch_result["processed"]
                files_skipped += batch_result["skipped"]

                for image_data in batch_result["results"]:
                    # Check if already in catalog
                    existing_image = db.execute(
                        "SELECT id FROM images WHERE id = ?", (image_data["checksum"],)
                    ).fetchone()

                    if not existing_image:
                        # Reconstruct ImageRecord from dict
                        metadata = (
                            ImageMetadata(**image_data["metadata"])
                            if image_data["metadata"]
                            else None
                        )
                        dates = (
                            DateInfo(**image_data["dates"])
                            if image_data["dates"]
                            else None
                        )

                        image = ImageRecord(
                            id=image_data["id"],
                            source_path=Path(image_data["source_path"]),
                            file_type=FileType(image_data["file_type"]),
                            checksum=image_data["checksum"],
                            dates=dates,
                            metadata=metadata,
                            status=ImageStatus.ANALYZING,
                        )

                        # Add to catalog
                        db.execute(
                            """
                            INSERT INTO images (
                                id, source_path, file_size, file_hash, format,
                                width, height, created_at, modified_at, indexed_at,
                                date_taken, camera_make, camera_model, lens_model,
                                focal_length, aperture, shutter_speed, iso,
                                gps_latitude, gps_longitude, quality_score, is_corrupted,
                                perceptual_hash, features_vector
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                image.id,
                                str(image.source_path),
                                image.file_size,
                                image.checksum,
                                image.metadata.format,
                                image.metadata.width,
                                image.metadata.height,
                                (
                                    image.created_at.isoformat()
                                    if image.created_at
                                    else None
                                ),
                                (
                                    image.modified_at.isoformat()
                                    if image.modified_at
                                    else None
                                ),
                                datetime.now().isoformat(),  # indexed_at
                                (
                                    image.dates.selected_date.isoformat()
                                    if image.dates and image.dates.selected_date
                                    else None
                                ),
                                image.metadata.exif.get("Make"),
                                image.metadata.exif.get("Model"),
                                image.metadata.exif.get("LensModel"),
                                image.metadata.exif.get("FocalLength"),
                                image.metadata.exif.get("ApertureValue"),
                                image.metadata.exif.get("ExposureTime"),
                                image.metadata.exif.get("ISO"),
                                (
                                    image.metadata.gps.latitude
                                    if image.metadata.gps
                                    else None
                                ),
                                (
                                    image.metadata.gps.longitude
                                    if image.metadata.gps
                                    else None
                                ),
                                image.quality_score,
                                1 if image.is_corrupted else 0,
                                image.hashes.perceptual_hash if image.hashes else None,
                                None,  # features_vector not handled yet
                            ),
                        )
                        files_added += 1

                        # Update statistics in memory
                        if image.file_type == FileType.IMAGE:
                            stats.total_images += 1
                        elif image.file_type == FileType.VIDEO:
                            stats.total_videos += 1

                        stats.total_size_bytes += image_data["file_size"]

            # Insert final statistics snapshot
            db.execute(
                """
                INSERT INTO statistics (
                    timestamp, total_images, total_videos, total_size_bytes,
                    images_scanned, images_hashed, images_tagged,
                    duplicate_groups, duplicate_images, potential_savings_bytes,
                    high_quality_count, medium_quality_count, low_quality_count,
                    corrupted_count, unsupported_count,
                    processing_time_seconds, images_per_second,
                    no_date, suspicious_dates, problematic_files
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    stats.total_images,
                    stats.total_videos,
                    stats.total_size_bytes,
                    stats.images_scanned,
                    stats.images_hashed,
                    stats.images_tagged,
                    stats.duplicate_groups,
                    stats.duplicate_images,
                    stats.potential_savings_bytes,
                    stats.high_quality_count,
                    stats.medium_quality_count,
                    stats.low_quality_count,
                    stats.corrupted_count,
                    stats.unsupported_count,
                    stats.processing_time_seconds,
                    stats.images_per_second,
                    stats.no_date,
                    stats.suspicious_dates,
                    stats.problematic_files,
                ),
            )

        logger.info(
            f"Analysis finalized: {total_processed} processed, "
            f"{files_added} added, {files_skipped} skipped"
        )

        self.update_progress(
            100,
            100,
            f"Analysis complete: {files_added} files added",
            {"phase": "complete"},
        )

        return {
            "status": "completed",
            "catalog_path": str(catalog_dir),
            "total_files": total_processed,
            "processed": total_processed,
            "files_added": files_added,
            "files_skipped": files_skipped,
        }

    except Exception as e:
        logger.error(f"Finalization failed: {e}", exc_info=True)
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


@app.task(bind=True, base=ProgressTask, name="organize_catalog")
def organize_catalog_task(
    self: ProgressTask,
    catalog_path: str,
    destination_path: str,
    strategy: str,
    simulate: bool = True,
) -> Dict[str, Any]:
    """
    Organize files in catalog based on a given strategy.

    Args:
        catalog_path: Path to catalog
        destination_path: Path to organize files into
        strategy: Organization strategy (e.g., "date_based")
        simulate: If True, only simulate organization without moving files

    Returns:
        Organization results
    """
    logger.info(f"Starting catalog organization: {catalog_path} to {destination_path}")

    try:
        catalog_dir = Path(catalog_path)
        dest_dir = Path(destination_path)

        self.update_progress(0, 1, "Initializing organization...", {"phase": "init"})

        with CatalogDatabase(catalog_dir) as db:
            organizer = FileOrganizer(db, dest_dir)
            org_strategy = OrganizationStrategy[strategy.upper()]

            results = organizer.organize_files(org_strategy, simulate=simulate)

        self.update_progress(
            100, 100, "Organization complete", {"phase": "complete", "results": results}
        )

        return {
            "status": "completed",
            "catalog_path": str(catalog_dir),
            "destination_path": str(dest_dir),
            "strategy": strategy,
            "simulate": simulate,
            "results": results,
        }

    except Exception as e:
        logger.error(f"Organization failed: {e}", exc_info=True)
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


@app.task(bind=True, base=ProgressTask, name="generate_thumbnails")
def generate_thumbnails_task(
    self: ProgressTask,
    catalog_path: str,
    sizes: Optional[List[int]] = None,
    quality: int = 85,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Generate thumbnails for catalog images.

    Args:
        catalog_path: Path to catalog
        sizes: List of thumbnail sizes (default: [256, 512])
        quality: JPEG quality (1-100)
        force: Regenerate existing thumbnails

    Returns:
        Thumbnail generation results
    """
    logger.info(f"Starting thumbnail generation: {catalog_path}")

    if sizes is None:
        sizes = [256, 512]

    try:
        catalog_dir = Path(catalog_path)

        self.update_progress(
            0, 1, "Initializing thumbnail generation...", {"phase": "init"}
        )

        with CatalogDatabase(catalog_dir) as db:
            image_records = db.get_all_images()
            total_images = len(image_records)
            generated_count = 0
            skipped_count = 0

            for i, record in enumerate(image_records):
                self.update_progress(
                    i,
                    total_images,
                    f"Generating thumbnail for {record.source_path.name}...",
                    {"phase": "generating", "current_file": record.source_path.name},
                )

                for size in sizes:
                    thumbnail_path = get_thumbnail_path(catalog_dir, record.id, size)

                    if thumbnail_path.exists() and not force:
                        skipped_count += 1
                        continue

                    try:
                        generate_thumbnail(
                            record.source_path, thumbnail_path, size, quality
                        )
                        generated_count += 1
                    except Exception as thumb_e:
                        logger.warning(
                            f"Failed to generate thumbnail for {record.source_path}: {thumb_e}"
                        )

            self.update_progress(
                total_images,
                total_images,
                "Thumbnail generation complete",
                {
                    "phase": "complete",
                    "generated": generated_count,
                    "skipped": skipped_count,
                },
            )

            return {
                "status": "completed",
                "catalog_path": str(catalog_dir),
                "generated_count": generated_count,
                "skipped_count": skipped_count,
            }

    except Exception as e:
        logger.error(f"Thumbnail generation failed: {e}", exc_info=True)
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise
