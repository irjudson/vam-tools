"""SQLAlchemy ORM models for global schema."""

import uuid
from datetime import datetime
from typing import List

from sqlalchemy import ARRAY, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


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
