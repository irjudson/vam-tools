"""Catalog management endpoints."""

import logging
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...db import get_db
from ...db.catalog_schema import create_schema, delete_catalog_data, schema_exists
from ...db.models import Catalog
from ...db.schemas import CatalogCreate, CatalogResponse

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
