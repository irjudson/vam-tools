"""
Item processors for the generic parallel job framework.

Each processor handles a single work item and returns a result dict.
Processors are registered with @register_item_processor decorator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from ..analysis.scanner import _process_file_worker
from ..db import CatalogDB as CatalogDatabase
from ..shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path
from .coordinator import register_item_processor

logger = logging.getLogger(__name__)


@register_item_processor("analyze")
def process_analyze_item(
    catalog_id: str,
    work_item: str,
    db: Optional[CatalogDatabase] = None,
    force_reanalyze: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Re-analyze a single image from the catalog (metadata extraction).

    Args:
        catalog_id: UUID of the catalog
        work_item: Image ID to re-analyze
        db: Optional database connection (will create one if not provided)
        force_reanalyze: Force re-processing even if already analyzed
        **kwargs: Additional arguments (unused)

    Returns:
        Dict with success status and result/error
    """
    image_id = work_item

    try:
        # Get image source path from database
        def _get_source_path(db_conn: CatalogDatabase) -> Optional[str]:
            assert db_conn.session is not None
            result = db_conn.session.execute(
                text("SELECT source_path FROM images WHERE id = :id"),
                {"id": image_id},
            )
            row = result.fetchone()
            return row[0] if row else None

        if db:
            source_path_str = _get_source_path(db)
        else:
            with CatalogDatabase(catalog_id) as db_conn:
                source_path_str = _get_source_path(db_conn)

        if not source_path_str:
            return {
                "success": False,
                "error": f"Image {image_id} not found in catalog",
            }

        file_path = Path(source_path_str)
        if not file_path.exists():
            return {
                "success": False,
                "error": f"Source file not found: {file_path}",
            }

        # Process file (extract metadata, compute hashes)
        result = _process_file_worker(file_path)

        if result is None:
            return {
                "success": False,
                "error": f"Failed to extract metadata from {file_path.name}",
            }

        image_record, file_size = result

        # Update the existing record in database
        def _update(db_conn: CatalogDatabase) -> Dict[str, Any]:
            try:
                # Update existing image with fresh metadata
                existing = db_conn.get_image(image_id)
                if existing:
                    # Update fields from re-analysis using proper field mappings
                    # DB model uses: size_bytes, metadata_json (JSONB), dates (JSONB)
                    existing.checksum = image_record.checksum
                    existing.size_bytes = (
                        image_record.metadata.size_bytes
                        if image_record.metadata
                        else None
                    )
                    # Update metadata as JSONB
                    existing.metadata_json = (
                        image_record.metadata.model_dump(mode="json")
                        if hasattr(image_record.metadata, "model_dump")
                        else {}
                    )
                    # Update dates as JSONB
                    existing.dates = (
                        image_record.dates.model_dump(mode="json")
                        if hasattr(image_record.dates, "model_dump")
                        else {}
                    )
                    db_conn.save()
                    return {
                        "success": True,
                        "result": "updated",
                        "image_id": image_id,
                        "file_size": file_size,
                    }
                else:
                    # Image was deleted, add it back
                    db_conn.add_image(image_record)
                    db_conn.save()
                    return {
                        "success": True,
                        "result": "added",
                        "image_id": image_record.id,
                        "file_size": file_size,
                    }
            except Exception as e:
                if db_conn.session:
                    db_conn.session.rollback()
                return {
                    "success": False,
                    "error": f"Database error: {e}",
                }

        if db:
            return _update(db)
        else:
            with CatalogDatabase(catalog_id) as db_conn:
                return _update(db_conn)

    except Exception as e:
        logger.warning(f"Failed to re-analyze image {image_id}: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@register_item_processor("thumbnails")
def process_thumbnail_item(
    catalog_id: str,
    work_item: str,
    db: Optional[CatalogDatabase] = None,
    sizes: Optional[List[str]] = None,
    quality: int = 85,
    force: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate thumbnails for a single image.

    Args:
        catalog_id: UUID of the catalog
        work_item: Image ID to generate thumbnail for
        db: Optional database connection
        sizes: List of thumbnail sizes (default: ["small", "medium", "large"])
        quality: JPEG quality (1-100)
        force: Regenerate existing thumbnails
        **kwargs: Additional arguments

    Returns:
        Dict with success status and thumbnail paths
    """
    if sizes is None:
        sizes = ["small", "medium", "large"]

    # Map size names to pixel values for generate_thumbnail
    SIZE_MAP = {"small": 128, "medium": 256, "large": 512}

    image_id = work_item

    try:
        # Get image source path from database
        def _get_source_path(db_conn: CatalogDatabase) -> Optional[str]:
            assert db_conn.session is not None
            result = db_conn.session.execute(
                text("SELECT source_path FROM images WHERE id = :id"),
                {"id": image_id},
            )
            row = result.fetchone()
            return row[0] if row else None

        if db:
            source_path_str = _get_source_path(db)
        else:
            with CatalogDatabase(catalog_id) as db_conn:
                source_path_str = _get_source_path(db_conn)

        if not source_path_str:
            return {
                "success": False,
                "error": f"Image {image_id} not found in catalog",
            }

        source_path = Path(source_path_str)
        if not source_path.exists():
            return {
                "success": False,
                "error": f"Source file not found: {source_path}",
            }

        # Get thumbnails directory
        thumbnails_dir = Path(f"/app/catalogs/{catalog_id}/thumbnails")
        thumbnails_dir.mkdir(parents=True, exist_ok=True)

        generated = []
        skipped = []
        failed = []

        for size_name in sizes:
            thumbnail_path = get_thumbnail_path(
                image_id=image_id,
                thumbnails_dir=thumbnails_dir,
                size=size_name,
            )

            if thumbnail_path.exists() and not force:
                skipped.append(size_name)
                continue

            try:
                # Get pixel size from size name
                pixel_size = SIZE_MAP.get(size_name, 256)
                success = generate_thumbnail(
                    source_path=source_path,
                    output_path=thumbnail_path,
                    size=(pixel_size, pixel_size),
                    quality=quality,
                )
                if success:
                    generated.append(size_name)
                else:
                    failed.append(size_name)
            except Exception as e:
                logger.warning(f"Thumbnail {size_name} failed for {image_id}: {e}")
                failed.append(size_name)

        return {
            "success": len(failed) == 0,
            "image_id": image_id,
            "generated": generated,
            "skipped": skipped,
            "failed": failed,
        }

    except Exception as e:
        logger.warning(f"Failed to generate thumbnails for {image_id}: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@register_item_processor("auto_tag")
def process_auto_tag_item(
    catalog_id: str,
    work_item: str,
    db: Optional[CatalogDatabase] = None,
    tagger: Any = None,
    backend: str = "openclip",
    threshold: float = 0.25,
    max_tags: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Auto-tag a single image using AI models.

    Args:
        catalog_id: UUID of the catalog
        work_item: Image ID to tag
        db: Optional database connection
        tagger: Pre-initialized ImageTagger instance
        backend: Tagging backend ('openclip', 'ollama', 'combined')
        threshold: Minimum confidence threshold
        max_tags: Maximum tags per image
        **kwargs: Additional arguments

    Returns:
        Dict with success status and tags applied
    """
    image_id = work_item

    try:
        # Get image source path from database
        def _get_source_path(db_conn: CatalogDatabase) -> Optional[str]:
            assert db_conn.session is not None
            result = db_conn.session.execute(
                text("SELECT source_path FROM images WHERE id = :id"),
                {"id": image_id},
            )
            row = result.fetchone()
            return row[0] if row else None

        if db:
            source_path_str = _get_source_path(db)
        else:
            with CatalogDatabase(catalog_id) as db_conn:
                source_path_str = _get_source_path(db_conn)

        if not source_path_str:
            return {
                "success": False,
                "error": f"Image {image_id} not found in catalog",
            }

        source_path = Path(source_path_str)
        if not source_path.exists():
            return {
                "success": False,
                "error": f"Source file not found: {source_path}",
            }

        # Initialize tagger if not provided
        if tagger is None:
            from ..analysis.image_tagger import ImageTagger

            tagger = ImageTagger(backend=backend)

        # Tag the image
        tags = tagger.tag_image(
            source_path,
            threshold=threshold,
            max_tags=max_tags,
        )

        if not tags:
            return {
                "success": True,
                "image_id": image_id,
                "tags_applied": 0,
                "message": "No tags met threshold",
            }

        # Store tags in database
        def _store_tags(db_conn: CatalogDatabase) -> int:
            assert db_conn.session is not None
            stored_count = 0
            for tag in tags:
                try:
                    # Get category as string (handle enum or string)
                    category = getattr(tag, "category", None)
                    if category is not None and hasattr(category, "value"):
                        category = category.value  # Convert enum to string

                    # Get or create tag
                    result = db_conn.session.execute(
                        text(
                            """
                            INSERT INTO tags (catalog_id, name, category, created_at)
                            VALUES (:catalog_id, :name, :category, NOW())
                            ON CONFLICT (catalog_id, name) DO UPDATE SET catalog_id = tags.catalog_id
                            RETURNING id
                        """
                        ),
                        {
                            "catalog_id": catalog_id,
                            "name": tag.tag_name,
                            "category": category,
                        },
                    )
                    tag_id = result.scalar()

                    # Create image_tag relationship
                    db_conn.session.execute(
                        text(
                            """
                            INSERT INTO image_tags (image_id, tag_id, confidence, source,
                                                   openclip_confidence, ollama_confidence, created_at)
                            VALUES (:image_id, :tag_id, :confidence, :source,
                                    :openclip_confidence, :ollama_confidence, NOW())
                            ON CONFLICT (image_id, tag_id) DO UPDATE SET
                                confidence = :confidence,
                                source = :source,
                                openclip_confidence = COALESCE(:openclip_confidence, image_tags.openclip_confidence),
                                ollama_confidence = COALESCE(:ollama_confidence, image_tags.ollama_confidence)
                        """
                        ),
                        {
                            "image_id": image_id,
                            "tag_id": tag_id,
                            "confidence": tag.confidence,
                            "source": getattr(tag, "source", backend),
                            "openclip_confidence": getattr(
                                tag, "openclip_confidence", None
                            ),
                            "ollama_confidence": getattr(
                                tag, "ollama_confidence", None
                            ),
                        },
                    )
                    stored_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to store tag {tag.tag_name} for {image_id}: {e}"
                    )
                    # Rollback to clear the failed transaction state
                    if db_conn.session:
                        db_conn.session.rollback()

            db_conn.session.commit()
            return stored_count

        if db:
            tags_stored = _store_tags(db)
        else:
            with CatalogDatabase(catalog_id) as db_conn:
                tags_stored = _store_tags(db_conn)

        return {
            "success": True,
            "image_id": image_id,
            "tags_applied": tags_stored,
            "tag_names": [t.tag_name for t in tags],
        }

    except Exception as e:
        logger.warning(f"Failed to tag image {image_id}: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# Helper to ensure all processors are loaded
def register_all_processors() -> None:
    """Import this module to register all item processors."""
    pass
