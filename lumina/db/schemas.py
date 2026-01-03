"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CatalogCreate(BaseModel):
    """Schema for creating a new catalog."""

    name: str = Field(..., min_length=1, max_length=255, description="Catalog name")
    source_directories: List[str] = Field(
        ..., min_length=1, description="Source directories to scan"
    )
    organized_directory: Optional[str] = Field(
        None, description="Default output directory for file reorganization"
    )


class CatalogResponse(BaseModel):
    """Schema for catalog API responses."""

    id: UUID
    name: str
    schema_name: str
    source_directories: List[str]
    organized_directory: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)  # Pydantic v2: allow ORM mode


class JobListResponse(BaseModel):
    """Schema for job list item in API responses."""

    id: str
    catalog_id: Optional[UUID]
    job_type: str
    status: str
    parameters: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
