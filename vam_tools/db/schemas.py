"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import List
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CatalogCreate(BaseModel):
    """Schema for creating a new catalog."""

    name: str = Field(..., min_length=1, max_length=255, description="Catalog name")
    source_directories: List[str] = Field(
        ..., min_length=1, description="Source directories to scan"
    )


class CatalogResponse(BaseModel):
    """Schema for catalog API responses."""

    id: UUID
    name: str
    schema_name: str
    source_directories: List[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)  # Pydantic v2: allow ORM mode
