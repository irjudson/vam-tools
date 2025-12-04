"""Catalog management endpoints."""

import logging
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from ...db import get_db
from ...db.catalog_schema import create_schema, delete_catalog_data, schema_exists
from ...db.models import Catalog
from ...db.schemas import CatalogCreate, CatalogResponse
from ...shared.thumbnail_utils import (
    THUMBNAIL_SIZES,
    generate_thumbnail,
    get_thumbnail_path,
)

logger = logging.getLogger(__name__)

router = APIRouter()


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

    # Camera make filter
    if camera_make:
        conditions.append("metadata->>'camera_make' ILIKE :camera_make")
        params["camera_make"] = f"%{camera_make}%"

    # Camera model filter
    if camera_model:
        conditions.append("metadata->>'camera_model' ILIKE :camera_model")
        params["camera_model"] = f"%{camera_model}%"

    # Lens filter
    if lens:
        conditions.append("metadata->>'lens_model' ILIKE :lens")
        params["lens"] = f"%{lens}%"

    # Focal length filter
    if focal_length:
        conditions.append("metadata->>'focal_length' = :focal_length")
        params["focal_length"] = focal_length

    # F-stop/aperture filter
    if f_stop:
        conditions.append("metadata->>'f_stop' = :f_stop")
        params["f_stop"] = f_stop

    # GPS filter
    if has_gps is True:
        conditions.append(
            "metadata->>'gps_latitude' IS NOT NULL AND metadata->>'gps_longitude' IS NOT NULL"
        )
    elif has_gps is False:
        conditions.append(
            "(metadata->>'gps_latitude' IS NULL OR metadata->>'gps_longitude' IS NULL)"
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
            created_at
        FROM images
        WHERE {where_clause}
        ORDER BY {order_by}
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
                "checksum": row_dict["checksum"],
                "size_bytes": row_dict["size_bytes"],
                "dates": row_dict["dates"],
                "metadata": row_dict["metadata"],
                "created_at": (
                    row_dict["created_at"].isoformat()
                    if row_dict["created_at"]
                    else None
                ),
            }
        )

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

    # Get distinct camera makes
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

    # Get distinct camera models
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

    # Get distinct lenses
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

    # Get distinct focal lengths
    focal_lengths_query = text(
        """
        SELECT DISTINCT metadata->>'focal_length' as focal_length
        FROM images
        WHERE catalog_id = :catalog_id
            AND metadata->>'focal_length' IS NOT NULL
            AND metadata->>'focal_length' != ''
        ORDER BY (metadata->>'focal_length')::float
        """
    )
    focal_lengths = [
        row[0]
        for row in db.execute(focal_lengths_query, {"catalog_id": catalog_id_str})
    ]

    # Get distinct f-stops/apertures
    f_stops_query = text(
        """
        SELECT DISTINCT metadata->>'f_stop' as f_stop
        FROM images
        WHERE catalog_id = :catalog_id
            AND metadata->>'f_stop' IS NOT NULL
            AND metadata->>'f_stop' != ''
        ORDER BY (metadata->>'f_stop')::float
        """
    )
    f_stops = [
        row[0] for row in db.execute(f_stops_query, {"catalog_id": catalog_id_str})
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

    # Get GPS stats
    gps_query = text(
        """
        SELECT
            COUNT(*) FILTER (WHERE metadata->>'gps_latitude' IS NOT NULL) as with_gps,
            COUNT(*) FILTER (WHERE metadata->>'gps_latitude' IS NULL) as without_gps
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

    # Get image record
    query = text(
        "SELECT source_path FROM images WHERE id = :image_id AND catalog_id = :catalog_id"
    )
    result = db.execute(
        query, {"image_id": image_id, "catalog_id": str(catalog_id)}
    ).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Image not found")

    source_path = Path(result[0])

    # Check if source file exists
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source file not found")

    # Get or create thumbnail for the requested size
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
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")

    # Return the thumbnail file
    return FileResponse(
        thumbnail_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000"},  # Cache for 1 year
    )
