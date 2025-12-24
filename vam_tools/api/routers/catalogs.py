"""Catalog management endpoints."""

import csv
import io
import json
import logging
import uuid
from datetime import datetime
from math import cos, radians
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageOps
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from ...db import get_db
from ...db.catalog_schema import create_schema, delete_catalog_data, schema_exists
from ...db.models import Catalog
from ...db.schemas import CatalogCreate, CatalogResponse
from ...shared.thumbnail_utils import (
    HEIC_EXTENSIONS,
    RAW_EXTENSIONS,
    THUMBNAIL_SIZES,
    generate_thumbnail,
    get_thumbnail_path,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class BurstListResponse(BaseModel):
    bursts: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int


class BurstImageDetail(BaseModel):
    """Detail for a single image in a burst."""

    image_id: str
    source_path: str
    sequence: int
    quality_score: int | None
    size_bytes: int | None
    dates: Dict[str, Any]
    metadata: Dict[str, Any]
    is_best: bool


class BurstDetailResponse(BaseModel):
    """Detailed response for a single burst with all images."""

    id: str
    catalog_id: str
    image_count: int
    start_time: str | None
    end_time: str | None
    duration_seconds: float | None
    camera_make: str | None
    camera_model: str | None
    best_image_id: str | None
    selection_method: str | None
    created_at: str | None
    images: List[BurstImageDetail]


class ApplySelectionRequest(BaseModel):
    """Request for applying a burst selection."""

    selected_image_id: str


class ApplySelectionResponse(BaseModel):
    """Response from applying a burst selection."""

    selected_image_id: str
    rejected_count: int


class BatchApplyRequest(BaseModel):
    """Request for batch applying burst selections."""

    use_recommendations: bool = True


class BatchApplyResponse(BaseModel):
    """Response from batch applying burst selections."""

    bursts_processed: int
    images_rejected: int


@router.get("/", response_model=List[CatalogResponse])
def list_catalogs(db: Session = Depends(get_db)):
    """List all catalogs."""
    catalogs = db.query(Catalog).all()
    return catalogs


@router.post("/", response_model=CatalogResponse, status_code=201)
def create_catalog(catalog: CatalogCreate, db: Session = Depends(get_db)):
    """Create a new catalog."""
    # Ensure main schema exists
    if not schema_exists():
        try:
            create_schema()
        except Exception as e:
            logger.error(f"Failed to create main schema: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to create database schema: {str(e)}"
            )

    # Generate catalog ID
    catalog_id = uuid.uuid4()

    # Create catalog record
    db_catalog = Catalog(
        id=catalog_id,
        name=catalog.name,
        schema_name=f"deprecated_{catalog_id}",  # Unique to satisfy constraint
        source_directories=catalog.source_directories,
    )

    db.add(db_catalog)
    db.commit()
    db.refresh(db_catalog)

    logger.info(f"Created catalog: {catalog.name} (id: {catalog_id})")

    return db_catalog


@router.get("/{catalog_id}", response_model=CatalogResponse)
def get_catalog(catalog_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get catalog by ID."""
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")
    return catalog


@router.put("/{catalog_id}", response_model=CatalogResponse)
def update_catalog(
    catalog_id: uuid.UUID, catalog_update: CatalogCreate, db: Session = Depends(get_db)
):
    """Update an existing catalog's name and source directories."""
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Update catalog fields
    catalog.name = catalog_update.name
    catalog.source_directories = catalog_update.source_directories

    db.commit()
    db.refresh(catalog)

    logger.info(f"Updated catalog: {catalog.name} (id: {catalog_id})")

    return catalog


@router.delete("/{catalog_id}", status_code=204)
def delete_catalog(catalog_id: uuid.UUID, db: Session = Depends(get_db)):
    """Delete a catalog."""
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Delete all catalog data (CASCADE will handle images, tags, etc.)
    try:
        delete_catalog_data(str(catalog_id))
    except Exception as e:
        logger.error(f"Failed to delete catalog data for {catalog_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete catalog data: {str(e)}"
        )

    # Delete catalog record
    db.delete(catalog)
    db.commit()

    logger.info(f"Deleted catalog: {catalog.name}")


@router.get("/{catalog_id}/images")
def list_catalog_images(
    catalog_id: uuid.UUID,
    limit: int = 100,
    offset: int = 0,
    # Search
    search: str = None,
    # Filters
    file_type: str = None,  # "image" or "video"
    camera_make: str = None,
    camera_model: str = None,
    lens: str = None,
    focal_length: str = None,  # Focal length in mm
    f_stop: str = None,  # F-stop/aperture value
    has_gps: bool = None,
    date_from: str = None,  # ISO date string
    date_to: str = None,  # ISO date string
    min_width: int = None,
    min_height: int = None,
    # Tag filters
    tags: str = Query(None, description="Comma-separated tag names to filter by"),
    tag_match: str = Query(
        "any", pattern="^(any|all)$", description="Match any or all tags"
    ),
    has_tags: bool = Query(None, description="Filter: True=has tags, False=no tags"),
    # Include tags in response
    include_tags: bool = Query(
        False, description="Include tags array in each image response"
    ),
    # Sorting
    sort_by: str = "date",  # date, filename, size, created_at
    sort_order: str = "desc",  # asc or desc
    db: Session = Depends(get_db),
):
    """List images in a catalog with search, filtering, sorting, and pagination.

    Search:
        - search: Search in filename and path

    Filters:
        - file_type: Filter by "image" or "video"
        - camera_make: Filter by camera manufacturer
        - camera_model: Filter by camera model
        - lens: Filter by lens model
        - focal_length: Filter by focal length (mm)
        - f_stop: Filter by aperture/f-stop
        - has_gps: Filter images with GPS coordinates
        - date_from/date_to: Filter by date range
        - min_width/min_height: Filter by minimum resolution
        - tags: Comma-separated list of tag names to filter by
        - tag_match: "any" (default) or "all" - match any tag or all tags
        - has_tags: True to show only tagged images, False for untagged only

    Sorting:
        - sort_by: date, filename, size, created_at
        - sort_order: asc or desc
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Build dynamic query
    conditions = ["catalog_id = :catalog_id"]
    params = {"catalog_id": str(catalog_id), "limit": limit, "offset": offset}

    # Search filter
    if search:
        conditions.append("source_path ILIKE :search")
        params["search"] = f"%{search}%"

    # File type filter
    if file_type:
        conditions.append("file_type = :file_type")
        params["file_type"] = file_type

    # Camera make filter (from top-level metadata)
    if camera_make:
        conditions.append("metadata->>'camera_make' = :camera_make")
        params["camera_make"] = camera_make

    # Camera model filter (from top-level metadata)
    if camera_model:
        conditions.append("metadata->>'camera_model' = :camera_model")
        params["camera_model"] = camera_model

    # Lens filter (from top-level metadata)
    if lens:
        conditions.append("metadata->>'lens_model' = :lens")
        params["lens"] = lens

    # Focal length filter (from top-level metadata, rounded)
    if focal_length:
        conditions.append("ROUND((metadata->>'focal_length')::numeric) = :focal_length")
        params["focal_length"] = int(focal_length)

    # F-stop/aperture filter (from top-level metadata)
    if f_stop:
        conditions.append("(metadata->>'aperture')::float = :f_stop")
        params["f_stop"] = float(f_stop)

    # GPS filter (from top-level metadata)
    if has_gps is True:
        conditions.append(
            "metadata->>'gps_latitude' IS NOT NULL AND (metadata->>'gps_latitude')::float != 0"
        )
    elif has_gps is False:
        conditions.append(
            "(metadata->>'gps_latitude' IS NULL OR (metadata->>'gps_latitude')::float = 0)"
        )

    # Date range filters
    if date_from:
        conditions.append(
            "(dates->>'selected_date')::timestamp >= :date_from::timestamp"
        )
        params["date_from"] = date_from

    if date_to:
        conditions.append("(dates->>'selected_date')::timestamp <= :date_to::timestamp")
        params["date_to"] = date_to

    # Resolution filters
    if min_width:
        conditions.append("(metadata->>'width')::int >= :min_width")
        params["min_width"] = min_width

    if min_height:
        conditions.append("(metadata->>'height')::int >= :min_height")
        params["min_height"] = min_height

    # Tag filters
    if has_tags is True:
        # Only images with at least one tag
        conditions.append(
            "EXISTS (SELECT 1 FROM image_tags it WHERE it.image_id = images.id)"
        )
    elif has_tags is False:
        # Only images without any tags
        conditions.append(
            "NOT EXISTS (SELECT 1 FROM image_tags it WHERE it.image_id = images.id)"
        )

    if tags:
        # Parse comma-separated tag names
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        if tag_list:
            # Build tag filter based on match mode
            if tag_match == "all":
                # Image must have ALL specified tags
                conditions.append(
                    """
                    (SELECT COUNT(DISTINCT t.name) FROM image_tags it
                     JOIN tags t ON it.tag_id = t.id
                     WHERE it.image_id = images.id AND t.name = ANY(:tag_names)) = :tag_count
                """
                )
                params["tag_names"] = tag_list
                params["tag_count"] = len(tag_list)
            else:
                # Image must have ANY of the specified tags (default)
                conditions.append(
                    """
                    EXISTS (
                        SELECT 1 FROM image_tags it
                        JOIN tags t ON it.tag_id = t.id
                        WHERE it.image_id = images.id AND t.name = ANY(:tag_names)
                    )
                """
                )
                params["tag_names"] = tag_list

    # Build ORDER BY clause
    order_direction = "DESC" if sort_order.lower() == "desc" else "ASC"
    order_clauses = {
        "date": f"(dates->>'selected_date')::timestamp {order_direction} NULLS LAST",
        "filename": f"source_path {order_direction}",
        "size": f"size_bytes {order_direction}",
        "camera": f"metadata->>'camera_model' {order_direction} NULLS LAST",
        "created_at": f"created_at {order_direction}",
    }
    order_by = order_clauses.get(sort_by, order_clauses["date"])

    # Build and execute query
    where_clause = " AND ".join(conditions)
    query = text(
        f"""
        SELECT
            id,
            source_path,
            file_type,
            checksum,
            size_bytes,
            dates,
            metadata,
            thumbnail_path,
            created_at,
            updated_at
        FROM images
        WHERE {where_clause}
        ORDER BY {order_by}
        LIMIT :limit OFFSET :offset
    """
    )

    result = db.execute(query, params)

    images = []
    image_ids = []
    for row in result:
        row_dict = dict(row._mapping)
        image_data = {
            "id": row_dict["id"],
            "source_path": row_dict["source_path"],
            "file_type": row_dict["file_type"],
            "checksum": row_dict["checksum"],
            "size_bytes": row_dict["size_bytes"],
            "dates": row_dict["dates"],
            "metadata": row_dict["metadata"],
            "thumbnail_path": row_dict["thumbnail_path"],
            "created_at": (
                row_dict["created_at"].isoformat() if row_dict["created_at"] else None
            ),
            "updated_at": (
                row_dict["updated_at"].isoformat() if row_dict["updated_at"] else None
            ),
        }
        images.append(image_data)
        image_ids.append(row_dict["id"])

    # Fetch tags for all images in a single query if requested
    if include_tags and image_ids:
        tags_query = text(
            """
            SELECT
                it.image_id,
                t.name,
                t.category,
                it.confidence
            FROM image_tags it
            JOIN tags t ON it.tag_id = t.id
            WHERE it.image_id = ANY(:image_ids)
            ORDER BY it.confidence DESC
        """
        )
        tags_result = db.execute(tags_query, {"image_ids": image_ids})

        # Group tags by image_id
        image_tags_map = {}
        for tag_row in tags_result:
            tag_dict = dict(tag_row._mapping)
            img_id = tag_dict["image_id"]
            if img_id not in image_tags_map:
                image_tags_map[img_id] = []
            image_tags_map[img_id].append(
                {
                    "name": tag_dict["name"],
                    "category": tag_dict["category"],
                    "confidence": round(tag_dict["confidence"] or 0, 3),
                }
            )

        # Add tags to each image
        for image in images:
            image["tags"] = image_tags_map.get(image["id"], [])

    # Get total count with same filters
    count_query = text(f"SELECT COUNT(*) FROM images WHERE {where_clause}")
    # Remove limit/offset from params for count query
    count_params = {k: v for k, v in params.items() if k not in ("limit", "offset")}
    total = db.execute(count_query, count_params).scalar()

    return {
        "images": images,
        "total": total,
        "limit": limit,
        "offset": offset,
        "filters": {
            "search": search,
            "file_type": file_type,
            "camera_make": camera_make,
            "camera_model": camera_model,
            "lens": lens,
            "has_gps": has_gps,
            "date_from": date_from,
            "date_to": date_to,
            "tags": tags,
            "tag_match": tag_match,
            "has_tags": has_tags,
        },
        "sort": {"by": sort_by, "order": sort_order},
    }


@router.get("/{catalog_id}/filter-options")
def get_filter_options(catalog_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get available filter options for a catalog (distinct cameras, lenses, etc.)."""
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Get distinct camera makes (from top-level metadata)
    camera_makes_query = text(
        """
        SELECT DISTINCT metadata->>'camera_make' as camera_make
        FROM images
        WHERE catalog_id = :catalog_id
            AND metadata->>'camera_make' IS NOT NULL
            AND metadata->>'camera_make' != ''
        ORDER BY camera_make
        """
    )
    camera_makes = [
        row[0] for row in db.execute(camera_makes_query, {"catalog_id": catalog_id_str})
    ]

    # Get distinct camera models (from top-level metadata)
    camera_models_query = text(
        """
        SELECT DISTINCT metadata->>'camera_model' as camera_model
        FROM images
        WHERE catalog_id = :catalog_id
            AND metadata->>'camera_model' IS NOT NULL
            AND metadata->>'camera_model' != ''
        ORDER BY camera_model
        """
    )
    camera_models = [
        row[0]
        for row in db.execute(camera_models_query, {"catalog_id": catalog_id_str})
    ]

    # Get distinct lenses (from top-level metadata)
    lenses_query = text(
        """
        SELECT DISTINCT metadata->>'lens_model' as lens_model
        FROM images
        WHERE catalog_id = :catalog_id
            AND metadata->>'lens_model' IS NOT NULL
            AND metadata->>'lens_model' != ''
        ORDER BY lens_model
        """
    )
    lenses = [
        row[0] for row in db.execute(lenses_query, {"catalog_id": catalog_id_str})
    ]

    # Get distinct focal lengths (from top-level metadata, rounded to nearest mm)
    focal_lengths_query = text(
        """
        SELECT DISTINCT ROUND((metadata->>'focal_length')::numeric) as focal_length
        FROM images
        WHERE catalog_id = :catalog_id
            AND metadata->>'focal_length' IS NOT NULL
        ORDER BY focal_length
        """
    )
    focal_lengths = [
        str(int(row[0]))
        for row in db.execute(focal_lengths_query, {"catalog_id": catalog_id_str})
    ]

    # Get distinct f-stops/apertures (from top-level metadata)
    f_stops_query = text(
        """
        SELECT DISTINCT (metadata->>'aperture')::float as f_stop
        FROM images
        WHERE catalog_id = :catalog_id
            AND metadata->>'aperture' IS NOT NULL
        ORDER BY f_stop
        """
    )
    f_stops = [
        str(row[0]) for row in db.execute(f_stops_query, {"catalog_id": catalog_id_str})
    ]

    # Get date range
    date_range_query = text(
        """
        SELECT
            MIN((dates->>'selected_date')::timestamp) as min_date,
            MAX((dates->>'selected_date')::timestamp) as max_date
        FROM images
        WHERE catalog_id = :catalog_id
            AND dates->>'selected_date' IS NOT NULL
        """
    )
    date_result = db.execute(
        date_range_query, {"catalog_id": catalog_id_str}
    ).fetchone()

    # Get file type counts
    file_types_query = text(
        """
        SELECT file_type, COUNT(*) as count
        FROM images
        WHERE catalog_id = :catalog_id
        GROUP BY file_type
        ORDER BY count DESC
        """
    )
    file_types = {
        row[0]: row[1]
        for row in db.execute(file_types_query, {"catalog_id": catalog_id_str})
    }

    # Get GPS stats (from top-level metadata)
    gps_query = text(
        """
        SELECT
            COUNT(*) FILTER (WHERE metadata->>'gps_latitude' IS NOT NULL
                             AND (metadata->>'gps_latitude')::float != 0) as with_gps,
            COUNT(*) FILTER (WHERE metadata->>'gps_latitude' IS NULL
                             OR (metadata->>'gps_latitude')::float = 0) as without_gps
        FROM images
        WHERE catalog_id = :catalog_id
        """
    )
    gps_result = db.execute(gps_query, {"catalog_id": catalog_id_str}).fetchone()

    return {
        "camera_makes": camera_makes,
        "camera_models": camera_models,
        "lenses": lenses,
        "focal_lengths": focal_lengths,
        "f_stops": f_stops,
        "file_types": file_types,
        "date_range": {
            "min": (
                date_result[0].isoformat() if date_result and date_result[0] else None
            ),
            "max": (
                date_result[1].isoformat() if date_result and date_result[1] else None
            ),
        },
        "gps": {
            "with_gps": gps_result[0] if gps_result else 0,
            "without_gps": gps_result[1] if gps_result else 0,
        },
    }


@router.get("/{catalog_id}/stats")
def get_catalog_stats(catalog_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get catalog statistics and metadata."""
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Get image counts by file type
    count_query = text(
        """
        SELECT
            COUNT(*) as total_images,
            COUNT(CASE WHEN file_type = 'image' THEN 1 END) as images,
            COUNT(CASE WHEN file_type = 'video' THEN 1 END) as videos,
            COALESCE(SUM(size_bytes), 0) as total_bytes,
            MIN(created_at) as first_added,
            MAX(created_at) as last_added
        FROM images
        WHERE catalog_id = :catalog_id
        """
    )
    result = db.execute(count_query, {"catalog_id": catalog_id_str}).fetchone()

    # Get date range from image dates
    date_range_query = text(
        """
        SELECT
            MIN((dates->>'selected_date')::timestamp) as earliest_date,
            MAX((dates->>'selected_date')::timestamp) as latest_date
        FROM images
        WHERE catalog_id = :catalog_id
            AND dates->>'selected_date' IS NOT NULL
        """
    )
    date_result = db.execute(
        date_range_query, {"catalog_id": catalog_id_str}
    ).fetchone()

    # Get recent job history for this catalog
    from ...db.models import Job

    recent_jobs = (
        db.query(Job)
        .filter(Job.catalog_id == catalog_id)
        .order_by(Job.created_at.desc())
        .limit(5)
        .all()
    )

    return {
        "catalog_id": catalog_id_str,
        "name": catalog.name,
        "source_directories": catalog.source_directories,
        "statistics": {
            "total_files": result.total_images if result else 0,
            "images": result.images if result else 0,
            "videos": result.videos if result else 0,
            "total_bytes": result.total_bytes if result else 0,
            "total_gb": round((result.total_bytes or 0) / (1024**3), 2),
            "first_added": (
                result.first_added.isoformat()
                if result and result.first_added
                else None
            ),
            "last_added": (
                result.last_added.isoformat() if result and result.last_added else None
            ),
            "date_range": {
                "earliest": (
                    date_result.earliest_date.isoformat()
                    if date_result and date_result.earliest_date
                    else None
                ),
                "latest": (
                    date_result.latest_date.isoformat()
                    if date_result and date_result.latest_date
                    else None
                ),
            },
        },
        "recent_jobs": [
            {
                "id": job.id,
                "job_type": job.job_type,
                "status": job.status,
                "created_at": job.created_at.isoformat(),
                "result": job.result,
            }
            for job in recent_jobs
        ],
    }


# =============================================================================
# Tag Endpoints
# =============================================================================


@router.get("/{catalog_id}/tags")
def list_catalog_tags(
    catalog_id: uuid.UUID,
    min_count: int = Query(0, ge=0, description="Minimum image count for a tag"),
    category: str = Query(None, description="Filter by tag category"),
    search: str = Query(None, description="Search tag names"),
    sort_by: str = Query("count", pattern="^(count|name|confidence)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(100, le=500),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    List all tags in a catalog with usage counts and statistics.

    Returns tags with:
    - name: Tag name
    - category: Tag category (subject, scene, style, etc.)
    - count: Number of images with this tag
    - avg_confidence: Average confidence score across all images
    - sources: Breakdown of tagging sources (openclip, ollama, manual)

    Filters:
    - min_count: Only show tags with at least this many images
    - category: Filter by tag category
    - search: Search tag names

    Sorting:
    - sort_by: count, name, or confidence
    - sort_order: asc or desc
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Build query for tags with counts
    conditions = ["t.catalog_id = :catalog_id"]
    params = {"catalog_id": catalog_id_str, "limit": limit, "offset": offset}

    if category:
        conditions.append("t.category = :category")
        params["category"] = category

    if search:
        conditions.append("t.name ILIKE :search")
        params["search"] = f"%{search}%"

    # Determine sort column
    sort_map = {
        "count": "image_count",
        "name": "t.name",
        "confidence": "avg_confidence",
    }
    sort_col = sort_map.get(sort_by, "image_count")
    order = "DESC" if sort_order == "desc" else "ASC"

    query = text(
        f"""
        SELECT
            t.id,
            t.name,
            t.category,
            COUNT(it.image_id) as image_count,
            AVG(it.confidence) as avg_confidence,
            jsonb_object_agg(
                COALESCE(it.source, 'unknown'),
                source_counts.cnt
            ) FILTER (WHERE source_counts.cnt IS NOT NULL) as sources
        FROM tags t
        LEFT JOIN image_tags it ON t.id = it.tag_id
        LEFT JOIN LATERAL (
            SELECT it2.source, COUNT(*) as cnt
            FROM image_tags it2
            WHERE it2.tag_id = t.id
            GROUP BY it2.source
        ) source_counts ON true
        WHERE {" AND ".join(conditions)}
        GROUP BY t.id, t.name, t.category
        HAVING COUNT(it.image_id) >= :min_count
        ORDER BY {sort_col} {order}, t.name ASC
        LIMIT :limit OFFSET :offset
    """
    )

    params["min_count"] = min_count
    result = db.execute(query, params)

    tags = []
    for row in result:
        row_dict = dict(row._mapping)
        tags.append(
            {
                "id": row_dict["id"],
                "name": row_dict["name"],
                "category": row_dict["category"],
                "count": row_dict["image_count"],
                "avg_confidence": round(row_dict["avg_confidence"] or 0, 3),
                "sources": row_dict["sources"] or {},
            }
        )

    # Get total count for pagination
    count_query = text(
        f"""
        SELECT COUNT(DISTINCT t.id) as total
        FROM tags t
        LEFT JOIN image_tags it ON t.id = it.tag_id
        WHERE {" AND ".join(conditions)}
        GROUP BY t.id
        HAVING COUNT(it.image_id) >= :min_count
    """
    )
    count_result = db.execute(count_query, params).fetchall()
    total = len(count_result)

    # Get category breakdown
    category_query = text(
        """
        SELECT t.category, COUNT(DISTINCT t.id) as tag_count
        FROM tags t
        WHERE t.catalog_id = :catalog_id
        GROUP BY t.category
        ORDER BY tag_count DESC
    """
    )
    categories = [
        {"category": row[0] or "uncategorized", "count": row[1]}
        for row in db.execute(category_query, {"catalog_id": catalog_id_str})
    ]

    return {
        "tags": tags,
        "total": total,
        "limit": limit,
        "offset": offset,
        "categories": categories,
    }


@router.get("/{catalog_id}/images/{image_id}/tags")
def get_image_tags(
    catalog_id: uuid.UUID,
    image_id: str,
    db: Session = Depends(get_db),
):
    """
    Get all tags for a specific image.

    Returns tags with confidence scores and source information.
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Verify image exists
    image_query = text(
        "SELECT id FROM images WHERE id = :image_id AND catalog_id = :catalog_id"
    )
    image_result = db.execute(
        image_query, {"image_id": image_id, "catalog_id": str(catalog_id)}
    ).fetchone()

    if not image_result:
        raise HTTPException(status_code=404, detail="Image not found")

    # Get tags for the image
    query = text(
        """
        SELECT
            t.id,
            t.name,
            t.category,
            it.confidence,
            it.source,
            it.openclip_confidence,
            it.ollama_confidence,
            it.created_at
        FROM image_tags it
        JOIN tags t ON it.tag_id = t.id
        WHERE it.image_id = :image_id
        ORDER BY it.confidence DESC
    """
    )

    result = db.execute(query, {"image_id": image_id})

    tags = []
    for row in result:
        row_dict = dict(row._mapping)
        tags.append(
            {
                "id": row_dict["id"],
                "name": row_dict["name"],
                "category": row_dict["category"],
                "confidence": round(row_dict["confidence"] or 0, 3),
                "source": row_dict["source"],
                "openclip_confidence": (
                    round(row_dict["openclip_confidence"], 3)
                    if row_dict["openclip_confidence"]
                    else None
                ),
                "ollama_confidence": (
                    round(row_dict["ollama_confidence"], 3)
                    if row_dict["ollama_confidence"]
                    else None
                ),
                "created_at": (
                    row_dict["created_at"].isoformat()
                    if row_dict["created_at"]
                    else None
                ),
            }
        )

    return {"image_id": image_id, "tags": tags, "count": len(tags)}


@router.get("/{catalog_id}/images/{image_id}/thumbnail")
def get_image_thumbnail(
    catalog_id: uuid.UUID,
    image_id: str,
    size: str = "medium",
    quality: int = 80,
    db: Session = Depends(get_db),
):
    """Get or generate a thumbnail for an image.

    Args:
        size: Thumbnail size - "small" (100px), "medium" (200px), or "large" (400px)
        quality: JPEG quality (1-100)
    """
    # Validate size parameter
    if size not in THUMBNAIL_SIZES:
        size = "medium"

    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Get image record with edit_data
    query = text(
        "SELECT source_path, edit_data FROM images WHERE id = :image_id AND catalog_id = :catalog_id"
    )
    result = db.execute(
        query, {"image_id": image_id, "catalog_id": str(catalog_id)}
    ).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Image not found")

    source_path = Path(result[0])
    edit_data = result[1]

    # Check if source file exists
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source file not found")

    # Check if image has transforms
    has_transforms = False
    rotation = 0
    flip_h = False
    flip_v = False

    if edit_data:
        transforms = edit_data.get("transforms", {})
        rotation = transforms.get("rotation", 0)
        flip_h = transforms.get("flip_h", False)
        flip_v = transforms.get("flip_v", False)
        has_transforms = rotation != 0 or flip_h or flip_v

    # If no transforms, use cached thumbnail
    if not has_transforms:
        thumbnails_dir = Path(f"/app/catalogs/{catalog_id}/thumbnails")
        thumbnail_path = get_thumbnail_path(
            image_id=image_id, thumbnails_dir=thumbnails_dir, size=size
        )

        # Generate thumbnail if it doesn't exist
        if not thumbnail_path.exists():
            thumb_size = THUMBNAIL_SIZES[size]
            success = generate_thumbnail(
                source_path=source_path,
                output_path=thumbnail_path,
                size=thumb_size,
                quality=quality,
            )
            if not success:
                raise HTTPException(
                    status_code=500, detail="Failed to generate thumbnail"
                )

        # Return the cached thumbnail file
        return FileResponse(
            thumbnail_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000"},  # Cache for 1 year
        )

    # Image has transforms - generate transformed thumbnail on-the-fly
    try:
        thumb_size = THUMBNAIL_SIZES[size]

        # Load image using helper that supports all formats (JPEG, PNG, HEIC, RAW)
        # Use half-size for RAW files (faster and sufficient for thumbnails)
        img = load_image_any_format(source_path, full_size=False)

        # Apply EXIF orientation FIRST (respects camera metadata)
        img = ImageOps.exif_transpose(img)

        # Convert to RGB if needed
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Apply transforms BEFORE thumbnailing for better quality
        # Apply rotation (PIL rotates counter-clockwise, so negate for clockwise)
        if rotation != 0:
            img = img.rotate(-rotation, expand=True)

        # Apply flips
        if flip_h:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if flip_v:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # Now create thumbnail (thumb_size is already a tuple)
        img.thumbnail(thumb_size, Image.Resampling.LANCZOS)

        # Save to buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=300",  # Cache for 5 minutes (transforms may change)
            },
        )
    except Exception as e:
        logger.error(f"Error generating transformed thumbnail for {source_path}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to generate transformed thumbnail"
        )


# =============================================================================
# Map/Time View Endpoints
# =============================================================================


@router.get("/{catalog_id}/map/clusters")
def get_map_clusters(
    catalog_id: uuid.UUID,
    precision: int = Query(4, ge=2, le=8, description="Geohash precision (2-8)"),
    date_from: str = None,
    date_to: str = None,
    bounds_sw_lat: float = None,
    bounds_sw_lon: float = None,
    bounds_ne_lat: float = None,
    bounds_ne_lon: float = None,
    db: Session = Depends(get_db),
):
    """
    Get clustered image counts by geohash for map display.

    Returns clusters with count, center coordinates, and geohash.
    Precision levels:
    - 2: ~1250km (world view)
    - 4: ~39km (country view)
    - 6: ~1.2km (city view)
    - 8: ~40m (street view)
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Build date filter
    date_conditions = []
    params = {"catalog_id": catalog_id_str}

    if date_from:
        date_conditions.append(
            "(dates->>'selected_date')::timestamp >= (:date_from)::timestamp"
        )
        params["date_from"] = date_from

    if date_to:
        date_conditions.append(
            "(dates->>'selected_date')::timestamp <= (:date_to)::timestamp"
        )
        params["date_to"] = date_to

    date_filter = (" AND " + " AND ".join(date_conditions)) if date_conditions else ""

    # Build bounds filter
    bounds_filter = ""
    if all([bounds_sw_lat, bounds_sw_lon, bounds_ne_lat, bounds_ne_lon]):
        bounds_filter = """
            AND (metadata->>'gps_latitude')::float BETWEEN :sw_lat AND :ne_lat
            AND (metadata->>'gps_longitude')::float BETWEEN :sw_lon AND :ne_lon
        """
        params.update(
            {
                "sw_lat": bounds_sw_lat,
                "ne_lat": bounds_ne_lat,
                "sw_lon": bounds_sw_lon,
                "ne_lon": bounds_ne_lon,
            }
        )

    # Use pre-computed geohash if available for common precisions
    geohash_col = f"geohash_{precision}" if precision in [4, 6, 8] else None

    if geohash_col:
        # Use indexed geohash columns for efficient grouping
        query = text(
            f"""
            SELECT
                {geohash_col} as geohash,
                COUNT(*) as count,
                AVG((metadata->>'gps_latitude')::float) as center_lat,
                AVG((metadata->>'gps_longitude')::float) as center_lon
            FROM images
            WHERE catalog_id = :catalog_id
                AND {geohash_col} IS NOT NULL
                {date_filter}
                {bounds_filter}
            GROUP BY {geohash_col}
            ORDER BY count DESC
            LIMIT 1000
        """
        )
    else:
        # Fallback: grid-based approximation for non-indexed precisions
        cell_size = {
            2: 11.0,
            3: 3.0,
            4: 0.7,
            5: 0.18,
            6: 0.04,
            7: 0.01,
            8: 0.003,
        }.get(precision, 1.0)

        query = text(
            f"""
            WITH geo_images AS (
                SELECT
                    (metadata->>'gps_latitude')::float as lat,
                    (metadata->>'gps_longitude')::float as lon
                FROM images
                WHERE catalog_id = :catalog_id
                    AND metadata->>'gps_latitude' IS NOT NULL
                    AND (metadata->>'gps_latitude')::float != 0
                    {date_filter}
                    {bounds_filter}
            )
            SELECT
                CONCAT(FLOOR(lat / {cell_size})::text, '_', FLOOR(lon / {cell_size})::text) as geohash,
                COUNT(*) as count,
                AVG(lat) as center_lat,
                AVG(lon) as center_lon
            FROM geo_images
            GROUP BY FLOOR(lat / {cell_size}), FLOOR(lon / {cell_size})
            ORDER BY count DESC
            LIMIT 1000
        """
        )

    result = db.execute(query, params)

    clusters = []
    total_photos = 0
    for row in result:
        row_dict = dict(row._mapping)
        # Skip clusters with missing coordinates
        if row_dict["center_lat"] is None or row_dict["center_lon"] is None:
            continue
        clusters.append(
            {
                "geohash": row_dict["geohash"],
                "count": row_dict["count"],
                "center_lat": float(row_dict["center_lat"]),
                "center_lon": float(row_dict["center_lon"]),
            }
        )
        total_photos += row_dict["count"]

    return {
        "clusters": clusters,
        "total_with_gps": total_photos,
        "precision": precision,
    }


@router.get("/{catalog_id}/map/timeline")
def get_timeline_histogram(
    catalog_id: uuid.UUID,
    bucket_size: str = Query("month", pattern="^(day|week|month|year)$"),
    db: Session = Depends(get_db),
):
    """
    Get photo count histogram over time for the timeline slider.

    Returns buckets with both total count and count_with_gps for each time period.
    This allows the frontend to show all photos with GPS-enabled ones highlighted.
    Filters out unrealistic dates (before 1900 or after 2100).
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Filter for realistic dates (1900 to current year + 1) to exclude corrupted metadata
    max_year = datetime.now().year + 1
    date_filter = f"""
        AND (dates->>'selected_date')::timestamp >= '1900-01-01'::timestamp
        AND (dates->>'selected_date')::timestamp <= '{max_year}-12-31'::timestamp
    """

    # Get histogram buckets with both total and GPS counts
    query = text(
        f"""
        SELECT
            DATE_TRUNC(:bucket_size, (dates->>'selected_date')::timestamp) as bucket_start,
            COUNT(*) as count,
            COUNT(*) FILTER (
                WHERE metadata->>'gps_latitude' IS NOT NULL
                AND (metadata->>'gps_latitude')::float != 0
            ) as count_with_gps
        FROM images
        WHERE catalog_id = :catalog_id
            AND dates->>'selected_date' IS NOT NULL
            {date_filter}
        GROUP BY bucket_start
        ORDER BY bucket_start
    """
    )

    result = db.execute(
        query, {"catalog_id": catalog_id_str, "bucket_size": bucket_size}
    )

    buckets = []
    for row in result:
        row_dict = dict(row._mapping)
        buckets.append(
            {
                "date": (
                    row_dict["bucket_start"].isoformat()
                    if row_dict["bucket_start"]
                    else None
                ),
                "count": row_dict["count"],
                "count_with_gps": row_dict["count_with_gps"],
            }
        )

    # Get overall date range and totals (with same realistic date filter)
    range_query = text(
        f"""
        SELECT
            MIN((dates->>'selected_date')::timestamp) as min_date,
            MAX((dates->>'selected_date')::timestamp) as max_date,
            COUNT(*) as total,
            COUNT(*) FILTER (
                WHERE metadata->>'gps_latitude' IS NOT NULL
                AND (metadata->>'gps_latitude')::float != 0
            ) as total_with_gps
        FROM images
        WHERE catalog_id = :catalog_id
            AND dates->>'selected_date' IS NOT NULL
            {date_filter}
    """
    )
    range_result = db.execute(range_query, {"catalog_id": catalog_id_str}).fetchone()

    return {
        "buckets": buckets,
        "bucket_size": bucket_size,
        "date_range": {
            "min": (
                range_result.min_date.isoformat() if range_result.min_date else None
            ),
            "max": (
                range_result.max_date.isoformat() if range_result.max_date else None
            ),
        },
        "total_images": range_result.total if range_result else 0,
        "total_with_gps": range_result.total_with_gps if range_result else 0,
    }


@router.get("/{catalog_id}/map/images")
def get_images_in_cluster(
    catalog_id: uuid.UUID,
    geohash: str = None,
    lat: float = None,
    lon: float = None,
    radius_km: float = 1.0,
    date_from: str = None,
    date_to: str = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    Get images within a specific geohash cell or radius from a point.

    Either provide:
    - geohash: Get images matching this geohash prefix
    - lat/lon: Get images within radius_km of this point
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    conditions = [
        "catalog_id = :catalog_id",
        "metadata->>'gps_latitude' IS NOT NULL",
        "(metadata->>'gps_latitude')::float != 0",
    ]
    params = {"catalog_id": catalog_id_str, "limit": limit, "offset": offset}

    if geohash:
        # Match by geohash prefix using the appropriate indexed column
        precision = len(geohash)
        if precision <= 4:
            conditions.append("geohash_4 LIKE :geohash_prefix")
            params["geohash_prefix"] = geohash[:4] + "%"
        elif precision <= 6:
            conditions.append("geohash_6 LIKE :geohash_prefix")
            params["geohash_prefix"] = geohash[:6] + "%"
        else:
            conditions.append("geohash_8 LIKE :geohash_prefix")
            params["geohash_prefix"] = geohash[:8] + "%"
    elif lat is not None and lon is not None:
        # Bounding box approximation for radius search
        lat_delta = radius_km / 111.0  # ~111km per degree latitude
        lon_delta = radius_km / (
            111.0 * abs(cos(radians(lat))) + 0.001
        )  # Adjust for longitude
        conditions.append(
            """
            (metadata->>'gps_latitude')::float BETWEEN :lat_min AND :lat_max
            AND (metadata->>'gps_longitude')::float BETWEEN :lon_min AND :lon_max
        """
        )
        params.update(
            {
                "lat_min": lat - lat_delta,
                "lat_max": lat + lat_delta,
                "lon_min": lon - lon_delta,
                "lon_max": lon + lon_delta,
            }
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'geohash' or 'lat' and 'lon' parameters",
        )

    # Date filters
    if date_from:
        conditions.append(
            "(dates->>'selected_date')::timestamp >= (:date_from)::timestamp"
        )
        params["date_from"] = date_from

    if date_to:
        conditions.append(
            "(dates->>'selected_date')::timestamp <= (:date_to)::timestamp"
        )
        params["date_to"] = date_to

    where_clause = " AND ".join(conditions)

    # Get images
    query = text(
        f"""
        SELECT
            id,
            source_path,
            file_type,
            thumbnail_path,
            (metadata->>'gps_latitude')::float as lat,
            (metadata->>'gps_longitude')::float as lon,
            dates->>'selected_date' as photo_date
        FROM images
        WHERE {where_clause}
        ORDER BY (dates->>'selected_date')::timestamp DESC NULLS LAST
        LIMIT :limit OFFSET :offset
    """
    )

    result = db.execute(query, params)

    images = []
    for row in result:
        row_dict = dict(row._mapping)
        images.append(
            {
                "id": row_dict["id"],
                "source_path": row_dict["source_path"],
                "file_type": row_dict["file_type"],
                "thumbnail_path": row_dict["thumbnail_path"],
                "lat": row_dict["lat"],
                "lon": row_dict["lon"],
                "photo_date": row_dict["photo_date"],
            }
        )

    # Get total count
    count_query = text(f"SELECT COUNT(*) FROM images WHERE {where_clause}")
    count_params = {k: v for k, v in params.items() if k not in ("limit", "offset")}
    total = db.execute(count_query, count_params).scalar()

    return {
        "images": images,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


# =============================================================================
# Duplicate Detection Endpoints
# =============================================================================


@router.post("/{catalog_id}/detect-duplicates")
def start_duplicate_detection(
    catalog_id: uuid.UUID,
    similarity_threshold: int = Query(
        5, ge=1, le=20, description="Hamming distance threshold for similarity"
    ),
    recompute_hashes: bool = Query(
        False, description="Force recomputation of perceptual hashes"
    ),
    db: Session = Depends(get_db),
):
    """
    Start a duplicate detection job for a catalog.

    This creates a background Celery task that:
    1. Computes perceptual hashes for all images (if not already computed)
    2. Finds exact duplicates (same checksum)
    3. Finds similar images (similar perceptual hash within threshold)
    4. Scores quality and selects primary (best) image in each group
    5. Saves duplicate groups to database

    Args:
        similarity_threshold: Maximum Hamming distance for images to be considered similar (1-20).
                              Lower values are more strict. Default 5 means ~92% similar.
        recompute_hashes: Force recomputation of perceptual hashes even if already computed.

    Returns:
        Job information with task ID to track progress.
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Import Celery task - use parallel coordinator for better performance
    from ...db.models import Job
    from ...jobs.parallel_duplicates import duplicates_coordinator_task

    # Generate job ID upfront (used as both DB id and Celery task id)
    job_id = str(uuid.uuid4())

    # Create job record
    job = Job(
        id=job_id,
        catalog_id=catalog_id,
        job_type="detect_duplicates",
        status="pending",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start Celery task with same ID - uses parallel coordinator pattern
    task = duplicates_coordinator_task.apply_async(
        kwargs={
            "catalog_id": str(catalog_id),
            "similarity_threshold": similarity_threshold,
            "recompute_hashes": recompute_hashes,
        },
        task_id=job_id,
    )

    logger.info(
        f"Started duplicate detection job {job.id} for catalog {catalog_id} "
        f"(threshold={similarity_threshold}, recompute={recompute_hashes})"
    )

    return {
        "job_id": str(job.id),
        "task_id": task.id,
        "status": "pending",
        "message": f"Duplicate detection started for catalog {catalog.name}",
    }


@router.post("/{catalog_id}/auto-tag")
def start_auto_tagging(
    catalog_id: uuid.UUID,
    backend: str = Query(
        "openclip", description="AI backend: 'openclip', 'ollama', or 'combined'"
    ),
    model: str = Query(
        None,
        description="Model name (e.g., 'ViT-B-32' for OpenCLIP, 'llava' for Ollama)",
    ),
    threshold: float = Query(
        0.25, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
    max_tags: int = Query(10, ge=1, le=50, description="Maximum tags per image"),
    max_images: int = Query(
        None, ge=1, description="Maximum images to tag (for testing, None = all)"
    ),
    tag_mode: str = Query(
        "untagged_only",
        description="Tagging mode: 'untagged_only' (skip already tagged) or 'all' (retag everything)",
    ),
    continue_pipeline: bool = Query(
        False, description="Continue to duplicate detection after tagging"
    ),
    db: Session = Depends(get_db),
):
    """
    Start an auto-tagging job for a catalog.

    This creates a background Celery task that:
    1. Uses AI models (OpenCLIP or Ollama) to analyze images
    2. Assigns tags from a predefined taxonomy based on image content
    3. Stores tags in the database for each image
    4. Optionally continues to duplicate detection

    Backends:
    - openclip: Fast batch processing using CLIP zero-shot classification (GPU accelerated)
    - ollama: Vision language models (LLaVA) for detailed understanding (slower, more accurate)
    - combined: Both backends with weighted confidence (40% OpenCLIP + 60% Ollama)

    Tag Modes:
    - untagged_only: Only process images that have no tags yet (default, faster)
    - all: Retag all images, replacing existing AI-generated tags

    Args:
        backend: AI backend to use ('openclip', 'ollama', or 'combined')
        model: Model name (optional, uses defaults if not specified)
        threshold: Minimum confidence threshold for tags (0.0-1.0)
        max_tags: Maximum number of tags per image
        tag_mode: 'untagged_only' to skip tagged images, 'all' to retag everything
        continue_pipeline: If True, starts duplicate detection after tagging completes

    Returns:
        Job information with task ID to track progress.
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Validate backend
    if backend not in ("openclip", "ollama", "combined"):
        raise HTTPException(
            status_code=400,
            detail="Invalid backend. Use 'openclip', 'ollama', or 'combined'",
        )

    # Validate tag_mode
    if tag_mode not in ("untagged_only", "all"):
        raise HTTPException(
            status_code=400,
            detail="Invalid tag_mode. Use 'untagged_only' or 'all'",
        )

    # Import Celery task
    from ...db.models import Job
    from ...jobs.tasks import auto_tag_task

    # Generate job ID upfront (used as both DB id and Celery task id)
    job_id = str(uuid.uuid4())

    # Create job record
    job = Job(
        id=job_id,
        catalog_id=catalog_id,
        job_type="auto_tag",
        status="pending",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start Celery task with same ID
    task = auto_tag_task.apply_async(
        kwargs={
            "catalog_id": str(catalog_id),
            "backend": backend,
            "model": model,
            "threshold": threshold,
            "max_tags": max_tags,
            "max_images": max_images,
            "tag_mode": tag_mode,
            "continue_pipeline": continue_pipeline,
        },
        task_id=job_id,
    )

    logger.info(
        f"Started auto-tagging job {job.id} for catalog {catalog_id} "
        f"(backend={backend}, model={model}, tag_mode={tag_mode}, continue_pipeline={continue_pipeline})"
    )

    return {
        "job_id": str(job.id),
        "task_id": task.id,
        "status": "pending",
        "message": f"Auto-tagging started for catalog {catalog.name} with {backend} backend",
    }


@router.post("/{catalog_id}/generate-descriptions")
def start_description_generation(
    catalog_id: uuid.UUID,
    model: str = Query(
        "llava", description="Ollama vision model to use (llava, qwen3-vl, etc.)"
    ),
    mode: str = Query(
        "undescribed_only",
        description="Processing mode: 'undescribed_only' (skip already described) or 'all'",
    ),
    limit: int = Query(
        None, ge=1, description="Maximum images to process (None = all)"
    ),
    db: Session = Depends(get_db),
):
    """
    Start AI description generation for catalog images using Ollama.

    This creates a background Celery task that:
    1. Uses Ollama vision models (LLaVA) to analyze images
    2. Generates natural language descriptions of image content
    3. Stores descriptions in the database for each image

    IMPORTANT: This task runs serially (one image at a time) on a dedicated
    queue because Ollama cannot handle concurrent requests without resource
    contention. For large catalogs, use the limit parameter to process in
    chunks.

    Modes:
    - undescribed_only: Only process images without descriptions (default, faster)
    - all: Regenerate descriptions for all images

    Args:
        model: Ollama vision model name (default: llava)
        mode: 'undescribed_only' to skip described images, 'all' to regenerate
        limit: Maximum images to process (for chunked processing)

    Returns:
        Job information with task ID to track progress.
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Validate mode
    if mode not in ("undescribed_only", "all"):
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Use 'undescribed_only' or 'all'",
        )

    # Import Celery task
    from ...db.models import Job
    from ...jobs.serial_descriptions import generate_descriptions_task

    # Generate job ID upfront (used as both DB id and Celery task id)
    job_id = str(uuid.uuid4())

    # Create job record
    job = Job(
        id=job_id,
        catalog_id=catalog_id,
        job_type="generate_descriptions",
        status="pending",
        parameters={
            "model": model,
            "mode": mode,
            "limit": limit,
        },
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start Celery task with same ID on the dedicated ollama queue
    task = generate_descriptions_task.apply_async(
        kwargs={
            "catalog_id": str(catalog_id),
            "model": model,
            "mode": mode,
            "limit": limit,
        },
        task_id=job_id,
    )

    logger.info(
        f"Started description generation job {job.id} for catalog {catalog_id} "
        f"(model={model}, mode={mode}, limit={limit})"
    )

    return {
        "job_id": str(job.id),
        "task_id": task.id,
        "status": "pending",
        "message": f"Description generation started for catalog {catalog.name} with {model} model",
    }


@router.get("/{catalog_id}/duplicates")
def list_duplicate_groups(
    catalog_id: uuid.UUID,
    limit: int = Query(50, le=200),
    offset: int = 0,
    reviewed: bool = None,
    similarity_type: str = None,  # "exact" or "perceptual"
    db: Session = Depends(get_db),
):
    """
    List duplicate groups in a catalog.

    Returns groups with their member images, similarity scores, and review status.
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Build query conditions
    conditions = ["dg.catalog_id = :catalog_id"]
    params = {"catalog_id": catalog_id_str, "limit": limit, "offset": offset}

    if reviewed is not None:
        conditions.append("dg.reviewed = :reviewed")
        params["reviewed"] = reviewed

    if similarity_type:
        conditions.append("dg.similarity_type = :similarity_type")
        params["similarity_type"] = similarity_type

    where_clause = " AND ".join(conditions)

    # Get duplicate groups with member count
    query = text(
        f"""
        SELECT
            dg.id,
            dg.primary_image_id,
            dg.similarity_type,
            dg.confidence,
            dg.reviewed,
            dg.created_at,
            COUNT(dm.image_id) as member_count
        FROM duplicate_groups dg
        LEFT JOIN duplicate_members dm ON dm.group_id = dg.id
        WHERE {where_clause}
        GROUP BY dg.id
        ORDER BY member_count DESC, dg.created_at DESC
        LIMIT :limit OFFSET :offset
    """
    )

    result = db.execute(query, params)

    groups = []
    for row in result:
        row_dict = dict(row._mapping)
        group_id = row_dict["id"]

        # Get members for this group
        members_query = text(
            """
            SELECT
                dm.image_id,
                dm.similarity_score,
                i.source_path,
                i.file_type,
                i.size_bytes,
                i.metadata,
                i.dates
            FROM duplicate_members dm
            JOIN images i ON i.id = dm.image_id AND i.catalog_id = :catalog_id
            WHERE dm.group_id = :group_id
            ORDER BY dm.similarity_score DESC
        """
        )
        members_result = db.execute(
            members_query, {"group_id": group_id, "catalog_id": catalog_id_str}
        )

        members = []
        for member_row in members_result:
            member_dict = dict(member_row._mapping)
            members.append(
                {
                    "image_id": member_dict["image_id"],
                    "similarity_score": member_dict["similarity_score"],
                    "source_path": member_dict["source_path"],
                    "file_type": member_dict["file_type"],
                    "size_bytes": member_dict["size_bytes"],
                    "is_primary": member_dict["image_id"]
                    == row_dict["primary_image_id"],
                    "metadata": member_dict["metadata"],
                    "dates": member_dict["dates"],
                }
            )

        groups.append(
            {
                "id": group_id,
                "primary_image_id": row_dict["primary_image_id"],
                "similarity_type": row_dict["similarity_type"],
                "confidence": row_dict["confidence"],
                "reviewed": row_dict["reviewed"],
                "created_at": (
                    row_dict["created_at"].isoformat()
                    if row_dict["created_at"]
                    else None
                ),
                "member_count": row_dict["member_count"],
                "members": members,
            }
        )

    # Get total count
    count_query = text(f"SELECT COUNT(*) FROM duplicate_groups dg WHERE {where_clause}")
    count_params = {k: v for k, v in params.items() if k not in ("limit", "offset")}
    total = db.execute(count_query, count_params).scalar()

    # Get summary stats
    stats_query = text(
        """
        SELECT
            COUNT(*) as total_groups,
            COUNT(*) FILTER (WHERE reviewed = true) as reviewed_groups,
            COUNT(*) FILTER (WHERE similarity_type = 'exact') as exact_groups,
            COUNT(*) FILTER (WHERE similarity_type = 'perceptual') as perceptual_groups
        FROM duplicate_groups
        WHERE catalog_id = :catalog_id
    """
    )
    stats_result = db.execute(stats_query, {"catalog_id": catalog_id_str}).fetchone()

    return {
        "groups": groups,
        "total": total,
        "offset": offset,
        "limit": limit,
        "statistics": {
            "total_groups": stats_result.total_groups if stats_result else 0,
            "reviewed_groups": stats_result.reviewed_groups if stats_result else 0,
            "exact_groups": stats_result.exact_groups if stats_result else 0,
            "perceptual_groups": stats_result.perceptual_groups if stats_result else 0,
        },
    }


@router.get("/{catalog_id}/duplicates/stats")
def get_duplicate_stats(
    catalog_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """
    Get duplicate detection statistics for a catalog.
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Get comprehensive stats
    query = text(
        """
        SELECT
            COUNT(DISTINCT dg.id) as total_groups,
            COUNT(dm.image_id) as total_duplicate_images,
            COUNT(*) FILTER (WHERE dg.reviewed = true) as reviewed_groups,
            COUNT(DISTINCT dg.id) FILTER (WHERE dg.similarity_type = 'exact') as exact_groups,
            COUNT(DISTINCT dg.id) FILTER (WHERE dg.similarity_type = 'perceptual') as perceptual_groups,
            COALESCE(SUM(i.size_bytes), 0) as duplicate_bytes
        FROM duplicate_groups dg
        LEFT JOIN duplicate_members dm ON dm.group_id = dg.id
        LEFT JOIN images i ON i.id = dm.image_id AND i.catalog_id = :catalog_id
            AND i.id != dg.primary_image_id  -- Only count non-primary images for space savings
        WHERE dg.catalog_id = :catalog_id
    """
    )
    result = db.execute(query, {"catalog_id": catalog_id_str}).fetchone()

    return {
        "catalog_id": catalog_id_str,
        "total_groups": result.total_groups if result else 0,
        "total_duplicate_images": result.total_duplicate_images if result else 0,
        "reviewed_groups": result.reviewed_groups if result else 0,
        "exact_groups": result.exact_groups if result else 0,
        "perceptual_groups": result.perceptual_groups if result else 0,
        "potential_space_savings_bytes": result.duplicate_bytes if result else 0,
        "potential_space_savings_gb": round(
            (result.duplicate_bytes or 0) / (1024**3), 2
        ),
    }


# =============================================================================
# Export Endpoints
# =============================================================================


@router.get("/{catalog_id}/duplicates/export")
def export_duplicates(
    catalog_id: uuid.UUID,
    format: str = Query("json", pattern="^(json|csv)$"),
    include_metadata: bool = Query(True, description="Include image metadata"),
    db: Session = Depends(get_db),
):
    """
    Export duplicate report as JSON or CSV.

    Args:
        format: Export format - "json" or "csv"
        include_metadata: Whether to include full image metadata (default: true)

    Returns:
        StreamingResponse with the export file
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Get all duplicate groups with members
    groups_query = text(
        """
        SELECT
            dg.id as group_id,
            dg.primary_image_id,
            dg.similarity_type,
            dg.confidence,
            dg.reviewed,
            dg.created_at as group_created_at
        FROM duplicate_groups dg
        WHERE dg.catalog_id = :catalog_id
        ORDER BY dg.created_at DESC
    """
    )
    groups_result = db.execute(groups_query, {"catalog_id": catalog_id_str})

    export_data = {
        "catalog_id": catalog_id_str,
        "catalog_name": catalog.name,
        "export_date": datetime.now().isoformat(),
        "groups": [],
        "summary": {},
    }

    total_groups = 0
    total_duplicates = 0
    total_savings = 0

    for group_row in groups_result:
        group_dict = dict(group_row._mapping)
        group_id = group_dict["group_id"]
        total_groups += 1

        # Get members for this group
        members_query = text(
            """
            SELECT
                dm.image_id,
                dm.similarity_score,
                i.source_path,
                i.file_type,
                i.size_bytes,
                i.checksum,
                i.metadata,
                i.dates
            FROM duplicate_members dm
            JOIN images i ON i.id = dm.image_id AND i.catalog_id = :catalog_id
            WHERE dm.group_id = :group_id
            ORDER BY dm.similarity_score DESC
        """
        )
        members_result = db.execute(
            members_query, {"group_id": group_id, "catalog_id": catalog_id_str}
        )

        members = []
        for member_row in members_result:
            member_dict = dict(member_row._mapping)
            is_primary = member_dict["image_id"] == group_dict["primary_image_id"]

            member_data = {
                "image_id": member_dict["image_id"],
                "source_path": member_dict["source_path"],
                "file_type": member_dict["file_type"],
                "size_bytes": member_dict["size_bytes"],
                "checksum": member_dict["checksum"],
                "similarity_score": member_dict["similarity_score"],
                "is_primary": is_primary,
                "recommended_action": "keep" if is_primary else "delete",
            }

            if include_metadata:
                member_data["metadata"] = member_dict["metadata"]
                member_data["dates"] = member_dict["dates"]

            members.append(member_data)

            # Count non-primary images for savings
            if not is_primary:
                total_duplicates += 1
                total_savings += member_dict["size_bytes"] or 0

        export_data["groups"].append(
            {
                "group_id": group_id,
                "similarity_type": group_dict["similarity_type"],
                "confidence": group_dict["confidence"],
                "reviewed": group_dict["reviewed"],
                "created_at": (
                    group_dict["group_created_at"].isoformat()
                    if group_dict["group_created_at"]
                    else None
                ),
                "member_count": len(members),
                "members": members,
            }
        )

    export_data["summary"] = {
        "total_groups": total_groups,
        "total_duplicate_images": total_duplicates,
        "potential_savings_bytes": total_savings,
        "potential_savings_gb": round(total_savings / (1024**3), 2),
    }

    if format == "json":
        # Return JSON
        json_str = json.dumps(export_data, indent=2, default=str)
        return StreamingResponse(
            io.StringIO(json_str),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=duplicates_{catalog_id_str[:8]}.json"
            },
        )
    else:
        # Return CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        headers = [
            "group_id",
            "similarity_type",
            "confidence",
            "reviewed",
            "image_id",
            "source_path",
            "file_type",
            "size_bytes",
            "checksum",
            "similarity_score",
            "is_primary",
            "recommended_action",
        ]
        if include_metadata:
            headers.extend(["photo_date", "camera_model", "dimensions"])
        writer.writerow(headers)

        # Write data rows
        for group in export_data["groups"]:
            for member in group["members"]:
                row = [
                    group["group_id"],
                    group["similarity_type"],
                    group["confidence"],
                    group["reviewed"],
                    member["image_id"],
                    member["source_path"],
                    member["file_type"],
                    member["size_bytes"],
                    member["checksum"],
                    member["similarity_score"],
                    member["is_primary"],
                    member["recommended_action"],
                ]
                if include_metadata:
                    metadata = member.get("metadata", {}) or {}
                    dates = member.get("dates", {}) or {}
                    row.extend(
                        [
                            dates.get("selected_date", ""),
                            metadata.get("camera_model", ""),
                            (
                                f"{metadata.get('width', '')}x{metadata.get('height', '')}"
                                if metadata.get("width")
                                else ""
                            ),
                        ]
                    )
                writer.writerow(row)

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=duplicates_{catalog_id_str[:8]}.csv"
            },
        )


@router.get("/{catalog_id}/images/export")
def export_images(
    catalog_id: uuid.UUID,
    format: str = Query("json", pattern="^(json|csv)$"),
    include_metadata: bool = Query(True, description="Include full metadata"),
    db: Session = Depends(get_db),
):
    """
    Export all images in a catalog as JSON or CSV.

    Args:
        format: Export format - "json" or "csv"
        include_metadata: Whether to include full image metadata

    Returns:
        StreamingResponse with the export file
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Get all images
    query = text(
        """
        SELECT
            id,
            source_path,
            file_type,
            checksum,
            size_bytes,
            dates,
            metadata,
            thumbnail_path,
            dhash,
            ahash,
            quality_score,
            status,
            created_at
        FROM images
        WHERE catalog_id = :catalog_id
        ORDER BY (dates->>'selected_date')::timestamp DESC NULLS LAST
    """
    )
    result = db.execute(query, {"catalog_id": catalog_id_str})

    if format == "json":
        images = []
        for row in result:
            row_dict = dict(row._mapping)
            image_data = {
                "id": row_dict["id"],
                "source_path": row_dict["source_path"],
                "file_type": row_dict["file_type"],
                "checksum": row_dict["checksum"],
                "size_bytes": row_dict["size_bytes"],
                "quality_score": row_dict["quality_score"],
                "status": row_dict["status"],
                "created_at": (
                    row_dict["created_at"].isoformat()
                    if row_dict["created_at"]
                    else None
                ),
            }
            if include_metadata:
                image_data["dates"] = row_dict["dates"]
                image_data["metadata"] = row_dict["metadata"]
            images.append(image_data)

        export_data = {
            "catalog_id": catalog_id_str,
            "catalog_name": catalog.name,
            "export_date": datetime.now().isoformat(),
            "total_images": len(images),
            "images": images,
        }

        json_str = json.dumps(export_data, indent=2, default=str)
        return StreamingResponse(
            io.StringIO(json_str),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=images_{catalog_id_str[:8]}.json"
            },
        )
    else:
        # CSV export
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        headers = [
            "id",
            "source_path",
            "file_type",
            "checksum",
            "size_bytes",
            "quality_score",
            "status",
            "created_at",
        ]
        if include_metadata:
            headers.extend(
                [
                    "photo_date",
                    "date_source",
                    "camera_make",
                    "camera_model",
                    "lens_model",
                    "width",
                    "height",
                    "gps_latitude",
                    "gps_longitude",
                ]
            )
        writer.writerow(headers)

        # Re-execute query for CSV (iterator was consumed)
        result = db.execute(query, {"catalog_id": catalog_id_str})

        for row in result:
            row_dict = dict(row._mapping)
            csv_row = [
                row_dict["id"],
                row_dict["source_path"],
                row_dict["file_type"],
                row_dict["checksum"],
                row_dict["size_bytes"],
                row_dict["quality_score"],
                row_dict["status"],
                (row_dict["created_at"].isoformat() if row_dict["created_at"] else ""),
            ]
            if include_metadata:
                metadata = row_dict["metadata"] or {}
                dates = row_dict["dates"] or {}
                csv_row.extend(
                    [
                        dates.get("selected_date", ""),
                        dates.get("source", ""),
                        metadata.get("camera_make", ""),
                        metadata.get("camera_model", ""),
                        metadata.get("lens_model", ""),
                        metadata.get("width", ""),
                        metadata.get("height", ""),
                        metadata.get("gps_latitude", ""),
                        metadata.get("gps_longitude", ""),
                    ]
                )
            writer.writerow(csv_row)

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=images_{catalog_id_str[:8]}.csv"
            },
        )


# =============================================================================
# Burst Detection Endpoints
# =============================================================================


@router.post("/{catalog_id}/detect-bursts")
def start_burst_detection(
    catalog_id: uuid.UUID,
    gap_threshold: float = Query(
        2.0, ge=0.1, le=30.0, description="Max seconds between burst images"
    ),
    min_burst_size: int = Query(
        3, ge=2, le=20, description="Minimum images to form a burst"
    ),
    db: Session = Depends(get_db),
):
    """
    Start a burst detection job for a catalog.

    This creates a background Celery task that:
    1. Analyzes image timestamps to detect rapid sequences (bursts)
    2. Groups images taken within the gap threshold
    3. Identifies the best image in each burst based on quality metrics
    4. Saves burst groups to database

    A "burst" is a sequence of images taken rapidly (e.g., holding shutter button).
    Common in sports, wildlife, and event photography.

    Args:
        gap_threshold: Maximum seconds between consecutive images in a burst (0.1-30).
                       Default 2.0 seconds works well for most burst photography.
        min_burst_size: Minimum number of images to form a burst (2-20).
                        Default 3 ensures only meaningful sequences are detected.

    Returns:
        Job information with task ID to track progress.
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Import Celery task
    from ...db.models import Job
    from ...jobs.tasks import detect_bursts_task

    # Generate job ID upfront (used as both DB id and Celery task id)
    job_id = str(uuid.uuid4())

    # Create job record
    job = Job(
        id=job_id,
        catalog_id=catalog_id,
        job_type="detect_bursts",
        status="pending",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start Celery task with same ID
    task = detect_bursts_task.apply_async(
        kwargs={
            "catalog_id": str(catalog_id),
            "gap_threshold": gap_threshold,
            "min_burst_size": min_burst_size,
        },
        task_id=job_id,
    )

    logger.info(
        f"Started burst detection job {job.id} for catalog {catalog_id} "
        f"(gap_threshold={gap_threshold}, min_burst_size={min_burst_size})"
    )

    return {
        "job_id": str(job.id),
        "task_id": task.id,
        "status": "pending",
        "message": f"Burst detection started for catalog {catalog.name}",
    }


@router.get("/{catalog_id}/bursts", response_model=BurstListResponse)
def list_bursts(
    catalog_id: uuid.UUID,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    camera_make: str = None,
    camera_model: str = None,
    min_images: int = Query(None, ge=2, description="Minimum images in burst"),
    show_rejected: bool = Query(False, description="Include bursts where all images are rejected"),
    sort: str = Query("newest", regex="^(newest|oldest|largest)$", description="Sort order: newest, oldest, or largest"),
    db: Session = Depends(get_db),
):
    """
    List burst groups in a catalog.

    Returns burst sequences with their member images, timing info, and best selection.
    By default, excludes bursts where all images are rejected.

    Args:
        catalog_id: Catalog UUID
        limit: Maximum bursts to return (default 50, max 200)
        offset: Pagination offset
        camera_make: Filter by camera make
        camera_model: Filter by camera model
        min_images: Minimum images in burst
        show_rejected: Include bursts where all images are rejected (default: False)
        sort: Sort order - newest (default), oldest, or largest
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Build query conditions
    conditions = ["b.catalog_id = :catalog_id"]
    params = {"catalog_id": catalog_id_str, "limit": limit, "offset": offset}

    if camera_make:
        conditions.append("b.camera_make = :camera_make")
        params["camera_make"] = camera_make

    if camera_model:
        conditions.append("b.camera_model = :camera_model")
        params["camera_model"] = camera_model

    if min_images:
        conditions.append("b.image_count >= :min_images")
        params["min_images"] = min_images

    # Exclude bursts where ALL images are rejected (unless show_rejected=True)
    if not show_rejected:
        conditions.append("""
            EXISTS (
                SELECT 1 FROM images i
                WHERE i.burst_id = b.id
                AND (i.status_id IS NULL OR i.status_id != 'rejected')
            )
        """)

    where_clause = " AND ".join(conditions)

    # Determine sort order
    if sort == "oldest":
        order_by = "b.start_time ASC"
    elif sort == "largest":
        order_by = "b.image_count DESC"
    else:  # newest (default)
        order_by = "b.start_time DESC"

    # Get bursts with member count
    query = text(
        f"""
        SELECT
            b.id,
            b.image_count,
            b.start_time,
            b.end_time,
            b.duration_seconds,
            b.camera_make,
            b.camera_model,
            b.best_image_id,
            b.selection_method,
            b.created_at
        FROM bursts b
        WHERE {where_clause}
        ORDER BY {order_by}
        LIMIT :limit OFFSET :offset
    """
    )

    result = db.execute(query, params)

    bursts = []
    for row in result:
        row_dict = dict(row._mapping)
        burst_id = row_dict["id"]

        # Get member images for this burst
        # Convert burst_id to string since images.burst_id is varchar and bursts.id is UUID
        members_query = text(
            """
            SELECT
                i.id,
                i.source_path,
                i.burst_sequence,
                i.quality_score,
                i.dates->>'selected_date' as photo_date,
                i.metadata
            FROM images i
            WHERE i.burst_id = :burst_id
            ORDER BY i.burst_sequence
        """
        )
        members_result = db.execute(members_query, {"burst_id": str(burst_id)})

        members = []
        for member_row in members_result:
            member_dict = dict(member_row._mapping)
            members.append(
                {
                    "image_id": member_dict["id"],
                    "source_path": member_dict["source_path"],
                    "sequence": member_dict["burst_sequence"],
                    "quality_score": member_dict["quality_score"],
                    "photo_date": member_dict["photo_date"],
                    "is_best": member_dict["id"] == row_dict["best_image_id"],
                }
            )

        bursts.append(
            {
                "id": burst_id,
                "image_count": row_dict["image_count"],
                "start_time": (
                    row_dict["start_time"].isoformat()
                    if row_dict["start_time"]
                    else None
                ),
                "end_time": (
                    row_dict["end_time"].isoformat() if row_dict["end_time"] else None
                ),
                "duration_seconds": row_dict["duration_seconds"],
                "camera_make": row_dict["camera_make"],
                "camera_model": row_dict["camera_model"],
                "best_image_id": row_dict["best_image_id"],
                "selection_method": row_dict["selection_method"],
                "created_at": (
                    row_dict["created_at"].isoformat()
                    if row_dict["created_at"]
                    else None
                ),
                "images": members,
            }
        )

    # Get total count
    count_query = text(f"SELECT COUNT(*) FROM bursts b WHERE {where_clause}")
    count_params = {k: v for k, v in params.items() if k not in ("limit", "offset")}
    total = db.execute(count_query, count_params).scalar()

    # Get summary stats
    stats_query = text(
        """
        SELECT
            COUNT(*) as total_bursts,
            COALESCE(SUM(image_count), 0) as total_burst_images,
            COALESCE(AVG(image_count), 0) as avg_burst_size,
            COALESCE(AVG(duration_seconds), 0) as avg_duration
        FROM bursts
        WHERE catalog_id = :catalog_id
    """
    )
    stats_result = db.execute(stats_query, {"catalog_id": catalog_id_str}).fetchone()

    return {
        "bursts": bursts,
        "total": total,
        "offset": offset,
        "limit": limit,
        "statistics": {
            "total_bursts": stats_result.total_bursts if stats_result else 0,
            "total_burst_images": (
                stats_result.total_burst_images if stats_result else 0
            ),
            "avg_burst_size": (
                round(stats_result.avg_burst_size, 1) if stats_result else 0
            ),
            "avg_duration_seconds": (
                round(stats_result.avg_duration, 2) if stats_result else 0
            ),
        },
    }


@router.get("/{catalog_id}/bursts/{burst_id}", response_model=BurstDetailResponse)
def get_burst(
    catalog_id: uuid.UUID,
    burst_id: str,
    db: Session = Depends(get_db),
):
    """
    Get details for a specific burst.

    Returns full burst information with all member images sorted by quality score.
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Get burst
    query = text(
        """
        SELECT
            b.id,
            b.image_count,
            b.start_time,
            b.end_time,
            b.duration_seconds,
            b.camera_make,
            b.camera_model,
            b.best_image_id,
            b.selection_method,
            b.created_at
        FROM bursts b
        WHERE b.id = :burst_id AND b.catalog_id = :catalog_id
    """
    )
    result = db.execute(
        query, {"burst_id": burst_id, "catalog_id": catalog_id_str}
    ).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Burst not found")

    row_dict = dict(result._mapping)

    # Get member images
    members_query = text(
        """
        SELECT
            i.id,
            i.source_path,
            i.burst_sequence,
            i.quality_score,
            i.size_bytes,
            i.dates,
            i.metadata
        FROM images i
        WHERE i.burst_id = :burst_id
        ORDER BY i.quality_score DESC NULLS LAST, i.burst_sequence ASC
    """
    )
    members_result = db.execute(members_query, {"burst_id": burst_id})

    members = []
    for member_row in members_result:
        member_dict = dict(member_row._mapping)
        members.append(
            {
                "image_id": str(member_dict["id"]) if member_dict["id"] else None,
                "source_path": member_dict["source_path"],
                "sequence": member_dict["burst_sequence"],
                "quality_score": member_dict["quality_score"],
                "size_bytes": member_dict["size_bytes"],
                "dates": member_dict["dates"],
                "metadata": member_dict["metadata"],
                "is_best": member_dict["id"] == row_dict["best_image_id"],
            }
        )

    return {
        "id": str(row_dict["id"]) if row_dict["id"] else None,
        "catalog_id": catalog_id_str,
        "image_count": row_dict["image_count"],
        "start_time": (
            row_dict["start_time"].isoformat() if row_dict["start_time"] else None
        ),
        "end_time": (
            row_dict["end_time"].isoformat() if row_dict["end_time"] else None
        ),
        "duration_seconds": row_dict["duration_seconds"],
        "camera_make": row_dict["camera_make"],
        "camera_model": row_dict["camera_model"],
        "best_image_id": str(row_dict["best_image_id"]) if row_dict["best_image_id"] else None,
        "selection_method": row_dict["selection_method"],
        "created_at": (
            row_dict["created_at"].isoformat() if row_dict["created_at"] else None
        ),
        "images": members,
    }


@router.post(
    "/{catalog_id}/bursts/{burst_id}/apply-selection",
    response_model=ApplySelectionResponse
)
def apply_burst_selection(
    catalog_id: uuid.UUID,
    burst_id: str,
    request: ApplySelectionRequest,
    db: Session = Depends(get_db),
):
    """
    Apply a burst selection by setting the selected image to active and all others to rejected.

    Args:
        catalog_id: The catalog ID
        burst_id: The burst ID
        request: Request containing the selected_image_id
        db: Database session

    Returns:
        ApplySelectionResponse with selected_image_id and rejected_count

    Raises:
        HTTPException: 404 if burst not found, 400 if selected_image_id not in burst
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # Verify burst exists
    burst_query = text(
        """
        SELECT id
        FROM bursts
        WHERE id = :burst_id AND catalog_id = :catalog_id
        """
    )
    burst_result = db.execute(
        burst_query, {"burst_id": burst_id, "catalog_id": catalog_id_str}
    ).fetchone()

    if not burst_result:
        raise HTTPException(status_code=404, detail="Burst not found")

    # Verify selected image is in the burst
    image_check_query = text(
        """
        SELECT id
        FROM images
        WHERE id = :image_id AND burst_id = :burst_id
        """
    )
    image_result = db.execute(
        image_check_query,
        {"image_id": request.selected_image_id, "burst_id": burst_id}
    ).fetchone()

    if not image_result:
        raise HTTPException(
            status_code=400,
            detail=f"Image {request.selected_image_id} is not in burst {burst_id}"
        )

    # Set selected image to active
    update_selected_query = text(
        """
        UPDATE images
        SET status_id = 'active'
        WHERE id = :image_id
        """
    )
    db.execute(update_selected_query, {"image_id": request.selected_image_id})

    # Set all other images in burst to rejected
    update_others_query = text(
        """
        UPDATE images
        SET status_id = 'rejected'
        WHERE burst_id = :burst_id AND id != :selected_id
        """
    )
    result = db.execute(
        update_others_query,
        {"burst_id": burst_id, "selected_id": request.selected_image_id}
    )
    rejected_count = result.rowcount

    db.commit()

    return ApplySelectionResponse(
        selected_image_id=request.selected_image_id,
        rejected_count=rejected_count
    )


@router.post(
    "/{catalog_id}/bursts/batch-apply",
    response_model=BatchApplyResponse
)
def batch_apply_burst_selections(
    catalog_id: uuid.UUID,
    request: BatchApplyRequest = BatchApplyRequest(),
    db: Session = Depends(get_db),
):
    """
    Batch apply burst selections for all bursts with best_image_id set.

    For each burst in the catalog that has best_image_id set:
    - Set the best image to active
    - Set all other images in the burst to rejected

    Args:
        catalog_id: The catalog ID
        request: Request containing use_recommendations flag (default: true)
        db: Database session

    Returns:
        BatchApplyResponse with bursts_processed and images_rejected counts

    Raises:
        HTTPException: 404 if catalog not found
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    catalog_id_str = str(catalog_id)

    # If use_recommendations is False, return immediately without processing
    if not request.use_recommendations:
        return BatchApplyResponse(
            bursts_processed=0,
            images_rejected=0
        )

    # Find all bursts in this catalog that have best_image_id set
    bursts_query = text(
        """
        SELECT id, best_image_id
        FROM bursts
        WHERE catalog_id = :catalog_id AND best_image_id IS NOT NULL
        """
    )
    bursts_result = db.execute(bursts_query, {"catalog_id": catalog_id_str}).fetchall()

    bursts_processed = 0
    images_rejected = 0

    # Process each burst
    for burst_row in bursts_result:
        burst_id = burst_row[0]
        best_image_id = burst_row[1]

        # Set best image to active
        update_best_query = text(
            """
            UPDATE images
            SET status_id = 'active'
            WHERE id = :image_id
            """
        )
        db.execute(update_best_query, {"image_id": best_image_id})

        # Set all other images in burst to rejected
        update_others_query = text(
            """
            UPDATE images
            SET status_id = 'rejected'
            WHERE burst_id = :burst_id AND id != :best_id
            """
        )
        result = db.execute(
            update_others_query,
            {"burst_id": burst_id, "best_id": best_image_id}
        )
        images_rejected += result.rowcount
        bursts_processed += 1

    db.commit()

    return BatchApplyResponse(
        bursts_processed=bursts_processed,
        images_rejected=images_rejected
    )


# ============================================================================
# Edit Mode Endpoints
# ============================================================================


def load_image_any_format(source_path: Path, full_size: bool = False) -> Image.Image:
    """
    Load an image from any supported format (JPEG, PNG, HEIC, RAW, etc.).

    Args:
        source_path: Path to the image file
        full_size: For RAW files, use full resolution (slower) vs half size (faster)

    Returns:
        PIL Image object

    Raises:
        HTTPException: If the file cannot be loaded
    """
    suffix = source_path.suffix.lower()

    # Handle RAW files
    if suffix in RAW_EXTENSIONS:
        try:
            import rawpy

            with rawpy.imread(str(source_path)) as raw:
                # Use full size for histograms/viewing, half size for thumbnails
                if full_size:
                    rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False)
                else:
                    rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                return Image.fromarray(rgb)
        except ImportError:
            raise HTTPException(
                status_code=422,
                detail=f"RAW file support not available. Install rawpy to view {suffix} files.",
            )
        except Exception as e:
            logger.error(f"Error loading RAW file {source_path}: {e}")
            raise HTTPException(
                status_code=422, detail=f"Cannot load RAW file: {str(e)}"
            )

    # Handle HEIC files (pillow_heif should be registered globally)
    elif suffix in HEIC_EXTENSIONS:
        try:
            img = Image.open(source_path)
            # Make a copy to ensure we don't hold file handles
            img_copy = img.copy()
            img.close()
            return img_copy
        except Exception as e:
            logger.error(f"Error loading HEIC file {source_path}: {e}")
            raise HTTPException(
                status_code=422,
                detail="Cannot load HEIC file. Install pillow-heif for HEIC support.",
            )

    # Handle standard formats (JPEG, PNG, etc.)
    else:
        try:
            img = Image.open(source_path)
            img_copy = img.copy()
            img.close()
            return img_copy
        except Exception as e:
            logger.error(f"Error loading image {source_path}: {e}")
            raise HTTPException(
                status_code=422, detail=f"Cannot load image file: {str(e)}"
            )


class EditData(BaseModel):
    """Edit data for non-destructive transforms."""

    version: int = 1
    transforms: Dict[str, Any] = {"rotation": 0, "flip_h": False, "flip_v": False}


class HistogramResponse(BaseModel):
    """RGB histogram data."""

    red: List[int]
    green: List[int]
    blue: List[int]
    luminance: List[int]


@router.get("/{catalog_id}/images/{image_id}/histogram")
def get_image_histogram(
    catalog_id: uuid.UUID,
    image_id: str,
    db: Session = Depends(get_db),
) -> HistogramResponse:
    """Generate RGB histogram for an image.

    Args:
        catalog_id: Catalog ID
        image_id: Image ID
        db: Database session

    Returns:
        Histogram data with red, green, blue, and luminance channels (256 bins each)
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Get image record
    result = db.execute(
        text(
            "SELECT source_path FROM images WHERE id = :id AND catalog_id = :catalog_id"
        ),
        {"id": image_id, "catalog_id": str(catalog_id)},
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    source_path = Path(row[0])
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    # Load image and compute histogram
    try:
        # Load image using helper that supports all formats (JPEG, PNG, HEIC, RAW)
        # Use half-size for RAW files (faster and sufficient for histogram)
        img = load_image_any_format(source_path, full_size=False)

        # Apply EXIF orientation FIRST (respects camera metadata)
        img = ImageOps.exif_transpose(img)

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Get image data as numpy array
        img_array = np.array(img)

        # Compute histograms for each channel
        red_hist = np.histogram(img_array[:, :, 0], bins=256, range=(0, 256))[0]
        green_hist = np.histogram(img_array[:, :, 1], bins=256, range=(0, 256))[0]
        blue_hist = np.histogram(img_array[:, :, 2], bins=256, range=(0, 256))[0]

        # Compute luminance histogram (using standard weights)
        luminance = (
            0.299 * img_array[:, :, 0]
            + 0.587 * img_array[:, :, 1]
            + 0.114 * img_array[:, :, 2]
        )
        lum_hist = np.histogram(luminance, bins=256, range=(0, 256))[0]

        return HistogramResponse(
            red=red_hist.tolist(),
            green=green_hist.tolist(),
            blue=blue_hist.tolist(),
            luminance=lum_hist.tolist(),
        )
    except Exception as e:
        logger.error(f"Error computing histogram for {source_path}: {e}")
        raise HTTPException(
            status_code=422,
            detail="Cannot generate histogram: unsupported or corrupted file format",
        )


@router.get("/{catalog_id}/images/{image_id}/edit")
def get_image_edit_data(
    catalog_id: uuid.UUID,
    image_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get current edit data for an image.

    Args:
        catalog_id: Catalog ID
        image_id: Image ID
        db: Database session

    Returns:
        Edit data or default empty structure
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    result = db.execute(
        text(
            "SELECT edit_data FROM images WHERE id = :id AND catalog_id = :catalog_id"
        ),
        {"id": image_id, "catalog_id": str(catalog_id)},
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    edit_data = row[0]
    if edit_data is None:
        # Return default structure
        return {
            "version": 1,
            "transforms": {"rotation": 0, "flip_h": False, "flip_v": False},
        }

    return edit_data


@router.put("/{catalog_id}/images/{image_id}/edit")
def update_image_edit_data(
    catalog_id: uuid.UUID,
    image_id: str,
    edit_data: EditData,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Update edit data for an image.

    Args:
        catalog_id: Catalog ID
        image_id: Image ID
        edit_data: New edit data
        db: Database session

    Returns:
        Success status and updated edit data
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Verify image exists
    result = db.execute(
        text("SELECT id FROM images WHERE id = :id AND catalog_id = :catalog_id"),
        {"id": image_id, "catalog_id": str(catalog_id)},
    )
    if not result.fetchone():
        raise HTTPException(status_code=404, detail="Image not found")

    # Update edit_data
    edit_dict = edit_data.model_dump()
    db.execute(
        text(
            "UPDATE images SET edit_data = :edit_data, updated_at = NOW() "
            "WHERE id = :id AND catalog_id = :catalog_id"
        ),
        {
            "edit_data": json.dumps(edit_dict),
            "id": image_id,
            "catalog_id": str(catalog_id),
        },
    )
    db.commit()

    return {"success": True, "edit_data": edit_dict}


@router.delete("/{catalog_id}/images/{image_id}/edit")
def reset_image_edit_data(
    catalog_id: uuid.UUID,
    image_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Reset edit data for an image (clear all edits).

    Args:
        catalog_id: Catalog ID
        image_id: Image ID
        db: Database session

    Returns:
        Success status
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Verify image exists
    result = db.execute(
        text("SELECT id FROM images WHERE id = :id AND catalog_id = :catalog_id"),
        {"id": image_id, "catalog_id": str(catalog_id)},
    )
    if not result.fetchone():
        raise HTTPException(status_code=404, detail="Image not found")

    # Clear edit_data
    db.execute(
        text(
            "UPDATE images SET edit_data = NULL, updated_at = NOW() "
            "WHERE id = :id AND catalog_id = :catalog_id"
        ),
        {"id": image_id, "catalog_id": str(catalog_id)},
    )
    db.commit()

    return {"success": True, "message": "Edit data reset to original"}


@router.get("/{catalog_id}/images/{image_id}/full", response_model=None)
def get_full_image(
    catalog_id: uuid.UUID,
    image_id: str,
    apply_transforms: bool = Query(
        False, description="Apply stored transforms to image"
    ),
    db: Session = Depends(get_db),
):
    """Serve full-size image, optionally with transforms applied.

    Args:
        catalog_id: Catalog ID
        image_id: Image ID
        apply_transforms: Whether to apply stored rotation/flip transforms
        db: Database session

    Returns:
        Image file (JPEG for processed images, original format otherwise)
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    result = db.execute(
        text(
            "SELECT source_path, edit_data FROM images "
            "WHERE id = :id AND catalog_id = :catalog_id"
        ),
        {"id": image_id, "catalog_id": str(catalog_id)},
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    source_path = Path(row[0])
    edit_data = row[1]

    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    # Check if file needs conversion (RAW/HEIC can't be served directly to browsers)
    suffix = source_path.suffix.lower()
    needs_conversion = suffix in RAW_EXTENSIONS or suffix in HEIC_EXTENSIONS

    # If no transforms requested or no edit_data, serve original (unless it needs conversion)
    if (not apply_transforms or edit_data is None) and not needs_conversion:
        return FileResponse(
            source_path,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # Apply transforms
    transforms = edit_data.get("transforms", {}) if edit_data else {}
    rotation = transforms.get("rotation", 0)
    flip_h = transforms.get("flip_h", False)
    flip_v = transforms.get("flip_v", False)

    # If no actual transforms and no conversion needed, serve original
    if rotation == 0 and not flip_h and not flip_v and not needs_conversion:
        return FileResponse(
            source_path,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # Load and transform image
    try:
        # Load image using helper that supports all formats (JPEG, PNG, HEIC, RAW)
        # Use full-size for RAW files to get best quality
        img = load_image_any_format(source_path, full_size=True)

        # Apply EXIF orientation FIRST (respects camera metadata)
        img = ImageOps.exif_transpose(img)

        # Convert to RGB if needed
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Apply rotation (PIL rotates counter-clockwise, so negate for clockwise)
        if rotation != 0:
            img = img.rotate(-rotation, expand=True)

        # Apply flips
        if flip_h:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if flip_v:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # Save to buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename={source_path.stem}_edited.jpg",
                "Cache-Control": "no-cache",  # Don't cache transformed images
            },
        )
    except Exception as e:
        logger.error(f"Error applying transforms to {source_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error applying transforms: {e}")


@router.post("/{catalog_id}/images/{image_id}/export-xmp")
def export_xmp_sidecar(
    catalog_id: uuid.UUID,
    image_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Export edit data as XMP sidecar file for Darktable compatibility.

    The XMP file will be created in the same directory as the source image
    with the same filename but .xmp extension.

    Args:
        catalog_id: Catalog ID
        image_id: Image ID
        db: Database session

    Returns:
        Success status and path to created XMP file
    """
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    result = db.execute(
        text(
            "SELECT source_path, edit_data FROM images "
            "WHERE id = :id AND catalog_id = :catalog_id"
        ),
        {"id": image_id, "catalog_id": str(catalog_id)},
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    source_path = Path(row[0])
    edit_data = row[1]

    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    # Calculate XMP file path
    xmp_path = source_path.with_suffix(".xmp")

    # Get transforms
    transforms = edit_data.get("transforms", {}) if edit_data else {}
    rotation = transforms.get("rotation", 0)
    flip_h = transforms.get("flip_h", False)
    # Note: flip_v not currently used in EXIF orientation mapping

    # Map transforms to EXIF orientation value
    # Standard EXIF orientation mapping:
    # 1 = normal, 2 = flip H, 3 = 180, 4 = 180 + flip H
    # 5 = 90 CW + flip H, 6 = 90 CW, 7 = 90 CCW + flip H, 8 = 90 CCW
    orientation = 1  # Default: normal

    if rotation == 0:
        orientation = 2 if flip_h else 1
    elif rotation == 90:
        orientation = 5 if flip_h else 6
    elif rotation == 180:
        orientation = 4 if flip_h else 3
    elif rotation == 270 or rotation == -90:
        orientation = 7 if flip_h else 8

    # Generate XMP content
    xmp_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
      <tiff:Orientation>{orientation}</tiff:Orientation>
      <xmp:CreatorTool>VAM Tools</xmp:CreatorTool>
      <xmp:ModifyDate>{datetime.now().isoformat()}</xmp:ModifyDate>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""

    # Write XMP file
    try:
        xmp_path.write_text(xmp_content, encoding="utf-8")
        logger.info(f"Exported XMP sidecar: {xmp_path}")

        return {
            "success": True,
            "xmp_path": str(xmp_path),
            "orientation": orientation,
        }
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail=f"Permission denied: Cannot write to {xmp_path.parent}",
        )
    except Exception as e:
        logger.error(f"Error writing XMP file {xmp_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error writing XMP file: {e}")
