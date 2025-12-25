"""SQLAlchemy ORM models for global schema."""

import uuid as uuid_module
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship

Base = declarative_base()


class Job(Base):
    """Job history in the global (public) schema."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # Celery task ID
    catalog_id: Mapped[Optional[uuid_module.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )  # Optional catalog reference
    job_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # 'scan' or 'analyze'
    status: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # PENDING, PROGRESS, SUCCESS, FAILURE, etc.
    parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )  # Job parameters (directories, options, etc.)
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )  # Final result when complete
    error: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Error message if failed
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, type={self.job_type}, status={self.status})>"


class Catalog(Base):
    """Catalog registry in the global (public) schema."""

    __tablename__ = "catalogs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_module.uuid4)
    name = Column(String(255), nullable=False)
    schema_name = Column(String(255), nullable=False, unique=True)
    source_directories = Column(ARRAY(Text), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<Catalog(id={self.id}, name={self.name}, schema={self.schema_name})>"


class ImageStatus(Base):
    """Status lookup table for images."""

    __tablename__ = "image_statuses"

    id = Column(
        String(50), primary_key=True
    )  # 'active', 'rejected', 'archived', 'flagged'
    name = Column(String(100), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<ImageStatus(id={self.id}, name={self.name})>"


class Image(Base):
    """Image/video records in catalogs."""

    __tablename__ = "images"

    id = Column(String, primary_key=True)  # Unique ID (checksum or UUID)
    catalog_id = Column(
        UUID(as_uuid=True),
        ForeignKey("catalogs.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_path = Column(Text, nullable=False)
    file_type = Column(String, nullable=False)  # 'image' or 'video'
    checksum = Column(Text, nullable=False)
    size_bytes = Column(BigInteger)

    # Dates and metadata stored as JSONB
    dates = Column(JSONB, nullable=False, default={})
    metadata_json = Column("metadata", JSONB, nullable=False, default={})

    # Thumbnail
    thumbnail_path = Column(Text)

    # Perceptual hashes for duplicate detection
    dhash = Column(Text)
    ahash = Column(Text)
    whash = Column(Text)  # Wavelet hash - most robust to transformations

    # Geohash columns for spatial queries (populated for images with GPS)
    geohash_4 = Column(String(4))  # ~39km precision (country view)
    geohash_6 = Column(String(6))  # ~1.2km precision (city view)
    geohash_8 = Column(String(8))  # ~40m precision (street view)

    # Analysis results
    quality_score = Column(Integer)

    # Status (references lookup table)
    status_id = Column(
        String(50),
        ForeignKey("image_statuses.id", ondelete="RESTRICT"),
        nullable=False,
        default="active",
        server_default="active",
    )

    # Processing flags - tracks which processing steps are complete
    # Structure: {
    #   "metadata_extracted": bool,  # EXIF/metadata extracted
    #   "dates_extracted": bool,     # Dates parsed with confidence
    #   "thumbnail_generated": bool, # Thumbnail created
    #   "hashes_computed": bool,     # Perceptual hashes computed
    #   "quality_scored": bool,      # Quality analysis complete
    #   "embedding_generated": bool, # CLIP embedding generated
    #   "tags_applied": bool,        # Auto-tagging complete
    #   "description_generated": bool, # Ollama description generated
    #   "ready_for_analysis": bool,  # All required fields for analysis tasks
    # }
    processing_flags = Column(JSONB, nullable=False, default={}, server_default="{}")

    # Burst detection
    burst_id = Column(UUID(as_uuid=True), ForeignKey("bursts.id", ondelete="SET NULL"))
    burst_sequence = Column(Integer)

    # Semantic search
    clip_embedding = Column(Vector(768))  # CLIP embedding for semantic search

    # AI-generated description from Ollama vision model
    description = Column(Text)

    # Non-destructive edit data (transforms, crop, adjustments)
    # Structure: {
    #   "version": 1,
    #   "transforms": {"rotation": 0, "flip_h": false, "flip_v": false},
    #   "crop": null,  # Future: {x, y, width, height}
    #   "adjustments": null,  # Future: {exposure, contrast, saturation}
    # }
    edit_data = Column(JSONB, nullable=True, default=None)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    catalog = relationship("Catalog", backref="images")
    status = relationship("ImageStatus")
    tags = relationship(
        "ImageTag", back_populates="image", cascade="all, delete-orphan"
    )
    duplicate_memberships = relationship(
        "DuplicateMember", back_populates="image", cascade="all, delete-orphan"
    )
    burst = relationship("Burst", back_populates="images")

    __table_args__ = (
        UniqueConstraint("catalog_id", "checksum", name="unique_catalog_checksum"),
    )

    def __repr__(self) -> str:
        return f"<Image(id={self.id}, path={self.source_path})>"


class Burst(Base):
    """Burst groups of images taken in rapid succession."""

    __tablename__ = "bursts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_module.uuid4)
    catalog_id = Column(
        UUID(as_uuid=True),
        ForeignKey("catalogs.id", ondelete="CASCADE"),
        nullable=False,
    )
    image_count = Column(Integer, nullable=False)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    camera_make = Column(String(255))
    camera_model = Column(String(255))
    best_image_id = Column(String)
    selection_method = Column(String(50), default="quality")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    catalog = relationship("Catalog", backref="bursts")
    images = relationship("Image", back_populates="burst")

    def __repr__(self) -> str:
        return f"<Burst(id={self.id}, image_count={self.image_count}, camera={self.camera_make} {self.camera_model})>"


class Tag(Base):
    """Tags for categorizing images."""

    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    catalog_id = Column(
        UUID(as_uuid=True),
        ForeignKey("catalogs.id", ondelete="CASCADE"),
        nullable=False,
    )
    name = Column(Text, nullable=False)
    category = Column(Text)  # Optional category
    parent_id = Column(Integer, ForeignKey("tags.id", ondelete="SET NULL"))
    synonyms = Column(ARRAY(Text))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    catalog = relationship("Catalog", backref="tags")
    parent = relationship("Tag", remote_side=[id], backref="children")
    images = relationship(
        "ImageTag", back_populates="tag", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("catalog_id", "name", name="unique_catalog_tag"),
    )

    def __repr__(self) -> str:
        return f"<Tag(id={self.id}, name={self.name})>"


class ImageTag(Base):
    """Many-to-many relationship between images and tags."""

    __tablename__ = "image_tags"

    image_id = Column(
        String, ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id = Column(
        Integer, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
    )
    confidence = Column(Float, default=1.0)  # Combined/final confidence
    source = Column(String, default="manual")  # manual, openclip, ollama, combined
    openclip_confidence = Column(Float, nullable=True)  # Confidence from OpenCLIP
    ollama_confidence = Column(Float, nullable=True)  # Confidence from Ollama
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    image = relationship("Image", back_populates="tags")
    tag = relationship("Tag", back_populates="images")

    def __repr__(self) -> str:
        return f"<ImageTag(image={self.image_id}, tag={self.tag_id}, confidence={self.confidence}, source={self.source})>"


class DuplicateGroup(Base):
    """Groups of duplicate or similar images."""

    __tablename__ = "duplicate_groups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    catalog_id = Column(
        UUID(as_uuid=True),
        ForeignKey("catalogs.id", ondelete="CASCADE"),
        nullable=False,
    )
    primary_image_id = Column(
        String, ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    similarity_type = Column(String, nullable=False)  # 'exact' or 'perceptual'
    confidence = Column(Integer, nullable=False)  # 0-100
    reviewed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    catalog = relationship("Catalog", backref="duplicate_groups")
    primary_image = relationship("Image", foreign_keys=[primary_image_id])
    members = relationship(
        "DuplicateMember", back_populates="group", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DuplicateGroup(id={self.id}, type={self.similarity_type}, confidence={self.confidence})>"


class DuplicateMember(Base):
    """Members of duplicate groups."""

    __tablename__ = "duplicate_members"

    group_id = Column(
        Integer, ForeignKey("duplicate_groups.id", ondelete="CASCADE"), primary_key=True
    )
    image_id = Column(
        String, ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    similarity_score = Column(Integer, nullable=False)  # 0-100

    # Relationships
    group = relationship("DuplicateGroup", back_populates="members")
    image = relationship("Image", back_populates="duplicate_memberships")

    def __repr__(self) -> str:
        return f"<DuplicateMember(group={self.group_id}, image={self.image_id}, score={self.similarity_score})>"


class Config(Base):
    """Per-catalog configuration settings."""

    __tablename__ = "config"

    catalog_id = Column(
        UUID(as_uuid=True),
        ForeignKey("catalogs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    key = Column(String, primary_key=True)
    value = Column(JSONB, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    catalog = relationship("Catalog", backref="config_entries")

    def __repr__(self) -> str:
        return f"<Config(catalog={self.catalog_id}, key={self.key})>"


class Statistics(Base):
    """Per-catalog statistics tracking."""

    __tablename__ = "statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    catalog_id = Column(
        UUID(as_uuid=True),
        ForeignKey("catalogs.id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Image counts
    total_images = Column(Integer, default=0)
    total_videos = Column(Integer, default=0)
    total_size_bytes = Column(BigInteger, default=0)
    images_scanned = Column(Integer, default=0)
    images_hashed = Column(Integer, default=0)
    images_tagged = Column(Integer, default=0)

    # Duplicate stats
    duplicate_groups = Column(Integer, default=0)
    duplicate_images = Column(Integer, default=0)
    potential_savings_bytes = Column(BigInteger, default=0)

    # Quality stats
    high_quality_count = Column(Integer, default=0)
    medium_quality_count = Column(Integer, default=0)
    low_quality_count = Column(Integer, default=0)
    corrupted_count = Column(Integer, default=0)
    unsupported_count = Column(Integer, default=0)

    # Performance metrics
    processing_time_seconds = Column(Float, default=0.0)
    images_per_second = Column(Float, default=0.0)

    # Date analysis
    no_date = Column(Integer, default=0)
    suspicious_dates = Column(Integer, default=0)
    problematic_files = Column(Integer, default=0)

    # Relationships
    catalog = relationship("Catalog", backref="statistics")

    def __repr__(self) -> str:
        return f"<Statistics(id={self.id}, catalog={self.catalog_id}, timestamp={self.timestamp})>"


class PerformanceSnapshot(Base):
    """Real-time performance tracking snapshots."""

    __tablename__ = "performance_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    catalog_id = Column(
        UUID(as_uuid=True),
        ForeignKey("catalogs.id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    phase = Column(Text, nullable=False)  # scanning, hashing, tagging, etc.

    # Progress tracking
    files_processed = Column(Integer, default=0)
    files_total = Column(Integer, default=0)
    bytes_processed = Column(BigInteger, default=0)

    # System metrics
    cpu_percent = Column(Float)
    memory_mb = Column(Float)
    disk_read_mb = Column(Float)
    disk_write_mb = Column(Float)

    # Performance metrics
    elapsed_seconds = Column(Float)
    rate_files_per_sec = Column(Float)
    rate_mb_per_sec = Column(Float)

    # GPU metrics
    gpu_utilization = Column(Float)
    gpu_memory_mb = Column(Float)

    # Relationships
    catalog = relationship("Catalog", backref="performance_snapshots")

    def __repr__(self) -> str:
        return f"<PerformanceSnapshot(id={self.id}, phase={self.phase}, timestamp={self.timestamp})>"
