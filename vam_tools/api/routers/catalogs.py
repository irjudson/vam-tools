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
from ...shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path

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
    db: Session = Depends(get_db),
):
    """List images in a catalog with pagination."""
    # Verify catalog exists
    catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    # Query images directly using SQL (since images table doesn't have ORM model yet)
    from sqlalchemy import text

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
            created_at
        FROM images
        WHERE catalog_id = :catalog_id
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
    """
    )

    result = db.execute(
        query, {"catalog_id": str(catalog_id), "limit": limit, "offset": offset}
    )

    images = []
    for row in result:
        # Convert row to dict for easier access
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

    # Get total count
    count_query = text("SELECT COUNT(*) FROM images WHERE catalog_id = :catalog_id")
    total = db.execute(count_query, {"catalog_id": str(catalog_id)}).scalar()

    return {"images": images, "total": total, "limit": limit, "offset": offset}


@router.get("/{catalog_id}/images/{image_id}/thumbnail")
def get_image_thumbnail(
    catalog_id: uuid.UUID,
    image_id: str,
    size: str = "medium",
    quality: int = 80,
    db: Session = Depends(get_db),
):
    """Get or generate a thumbnail for an image."""
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

    # Get or create thumbnail
    thumbnails_dir = Path(f"/app/catalogs/{catalog_id}/thumbnails")
    thumbnail_path = get_thumbnail_path(
        image_id=image_id, thumbnails_dir=thumbnails_dir
    )

    # Generate thumbnail if it doesn't exist
    if not thumbnail_path.exists():
        success = generate_thumbnail(
            source_path=source_path, output_path=thumbnail_path, quality=quality
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")

    # Return the thumbnail file
    return FileResponse(
        thumbnail_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000"},  # Cache for 1 year
    )
