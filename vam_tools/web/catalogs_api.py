"""
API endpoints for catalog configuration management.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.catalog_config import get_catalog_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/catalogs", tags=["catalogs"])


def normalize_path(path: str) -> str:
    """
    Normalize user-friendly paths to container paths.

    Converts:
    - ~/catalogs/my-photos -> /app/catalogs/my-photos
    - ~/photos/2024 -> /app/photos/2024

    This allows users to use familiar home directory syntax
    while the backend uses Docker container paths.
    """
    if path.startswith("~/"):
        return "/app/" + path[2:]
    return path


class CatalogCreateRequest(BaseModel):
    """Request model for creating a catalog."""

    name: str
    catalog_path: str
    source_directories: List[str]
    description: str = ""
    color: str = "#60a5fa"


class CatalogUpdateRequest(BaseModel):
    """Request model for updating a catalog."""

    name: Optional[str] = None
    catalog_path: Optional[str] = None
    source_directories: Optional[List[str]] = None
    description: Optional[str] = None
    color: Optional[str] = None


class CatalogResponse(BaseModel):
    """Response model for catalog data."""

    id: str
    name: str
    catalog_path: str
    source_directories: List[str]
    description: str
    created_at: str
    last_accessed: str
    color: str


class CurrentCatalogRequest(BaseModel):
    """Request model for setting current catalog."""

    catalog_id: str


@router.get("", response_model=List[CatalogResponse])
async def list_catalogs():
    """Get all configured catalogs."""
    manager = get_catalog_manager()
    catalogs = manager.list_catalogs()
    return [
        CatalogResponse(
            id=c.id,
            name=c.name,
            catalog_path=c.catalog_path,
            source_directories=c.source_directories,
            description=c.description,
            created_at=c.created_at,
            last_accessed=c.last_accessed,
            color=c.color,
        )
        for c in catalogs
    ]


@router.post("", response_model=CatalogResponse)
async def create_catalog(request: CatalogCreateRequest):
    """Create a new catalog configuration."""
    manager = get_catalog_manager()

    try:
        # Normalize paths from user-friendly (~/) to container paths (/app/)
        catalog_path = normalize_path(request.catalog_path)
        source_directories = [normalize_path(d) for d in request.source_directories]

        catalog = manager.add_catalog(
            name=request.name,
            catalog_path=catalog_path,
            source_directories=source_directories,
            description=request.description,
            color=request.color,
        )

        return CatalogResponse(
            id=catalog.id,
            name=catalog.name,
            catalog_path=catalog.catalog_path,
            source_directories=catalog.source_directories,
            description=catalog.description,
            created_at=catalog.created_at,
            last_accessed=catalog.last_accessed,
            color=catalog.color,
        )
    except Exception as e:
        logger.error(f"Error creating catalog: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current", response_model=Optional[CatalogResponse])
async def get_current_catalog():
    """Get the currently active catalog."""
    manager = get_catalog_manager()
    catalog = manager.get_current_catalog()

    if not catalog:
        return None

    return CatalogResponse(
        id=catalog.id,
        name=catalog.name,
        catalog_path=catalog.catalog_path,
        source_directories=catalog.source_directories,
        description=catalog.description,
        created_at=catalog.created_at,
        last_accessed=catalog.last_accessed,
        color=catalog.color,
    )


@router.post("/current")
async def set_current_catalog(request: CurrentCatalogRequest):
    """Set the currently active catalog."""
    manager = get_catalog_manager()

    if not manager.set_current_catalog(request.catalog_id):
        raise HTTPException(status_code=404, detail="Catalog not found")

    return {"message": "Current catalog updated", "catalog_id": request.catalog_id}


@router.get("/{catalog_id}", response_model=CatalogResponse)
async def get_catalog(catalog_id: str):
    """Get a specific catalog by ID."""
    manager = get_catalog_manager()
    catalog = manager.get_catalog(catalog_id)

    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    return CatalogResponse(
        id=catalog.id,
        name=catalog.name,
        catalog_path=catalog.catalog_path,
        source_directories=catalog.source_directories,
        description=catalog.description,
        created_at=catalog.created_at,
        last_accessed=catalog.last_accessed,
        color=catalog.color,
    )


@router.put("/{catalog_id}", response_model=CatalogResponse)
async def update_catalog(catalog_id: str, request: CatalogUpdateRequest):
    """Update a catalog's configuration."""
    manager = get_catalog_manager()

    # Build update dict with only provided fields
    updates = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.catalog_path is not None:
        updates["catalog_path"] = request.catalog_path
    if request.source_directories is not None:
        updates["source_directories"] = request.source_directories
    if request.description is not None:
        updates["description"] = request.description
    if request.color is not None:
        updates["color"] = request.color

    catalog = manager.update_catalog(catalog_id, **updates)

    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")

    return CatalogResponse(
        id=catalog.id,
        name=catalog.name,
        catalog_path=catalog.catalog_path,
        source_directories=catalog.source_directories,
        description=catalog.description,
        created_at=catalog.created_at,
        last_accessed=catalog.last_accessed,
        color=catalog.color,
    )


@router.delete("/{catalog_id}")
async def delete_catalog(catalog_id: str):
    """Delete a catalog configuration."""
    manager = get_catalog_manager()

    if not manager.delete_catalog(catalog_id):
        raise HTTPException(status_code=404, detail="Catalog not found")

    return {"message": "Catalog deleted", "catalog_id": catalog_id}
