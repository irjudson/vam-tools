"""SQLAlchemy ORM models for global schema."""

import uuid
from datetime import datetime
from typing import List

from sqlalchemy import ARRAY, JSON, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Job(Base):
    """Job history in the global (public) schema."""

    __tablename__ = "jobs"

    id = Column(String(255), primary_key=True)  # Celery task ID
    catalog_id = Column(UUID(as_uuid=True), nullable=True)  # Optional catalog reference
    job_type = Column(String(50), nullable=False)  # 'scan' or 'analyze'
    status = Column(
        String(50), nullable=False
    )  # PENDING, PROGRESS, SUCCESS, FAILURE, etc.
    parameters = Column(
        JSON, nullable=True
    )  # Job parameters (directories, options, etc.)
    result = Column(JSON, nullable=True)  # Final result when complete
    error = Column(Text, nullable=True)  # Error message if failed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, type={self.job_type}, status={self.status})>"


class Catalog(Base):
    """Catalog registry in the global (public) schema."""

    __tablename__ = "catalogs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    schema_name = Column(String(255), nullable=False, unique=True)
    source_directories = Column(ARRAY(Text), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<Catalog(id={self.id}, name={self.name}, schema={self.schema_name})>"
