"""
Type definitions for the catalog system.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class FileType(Enum):
    """Type of media file."""

    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


class CatalogPhase(Enum):
    """Current phase of catalog processing."""

    ANALYZING = "analyzing"
    REVIEWING = "reviewing"
    VERIFIED = "verified"
    EXECUTING = "executing"
    COMPLETE = "complete"


class ImageStatus(Enum):
    """Status of an individual image."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    EXECUTED = "executed"


class DuplicateRole(Enum):
    """Role of an image in a duplicate group."""

    PRIMARY = "primary"
    DUPLICATE = "duplicate"


class BurstRole(Enum):
    """Role of an image in a burst group."""

    PRIMARY = "primary"
    BURST_IMAGE = "burst_image"


class OperationType(Enum):
    """Type of file operation."""

    MOVE = "move"
    DELETE = "delete"
    SKIP = "skip"
    EXIF_MERGE = "exif_merge"


class OperationStatus(Enum):
    """Status of an operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ReviewType(Enum):
    """Type of review item."""

    DATE_CONFLICT = "date_conflict"
    SUSPICIOUS_DATE = "suspicious_date"
    NO_DATE = "no_date"
    BURST_REVIEW = "burst_review"
    NAME_COLLISION = "name_collision"
    MANUAL_SELECTION = "manual_selection"


class ReviewPriority(Enum):
    """Priority level for review items."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReviewStatus(Enum):
    """Status of a review item."""

    PENDING = "pending"
    REVIEWING = "reviewing"
    RESOLVED = "resolved"


class DateInfo(BaseModel):
    """Date information extracted from various sources."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    exif_dates: Dict[str, Optional[datetime]] = Field(default_factory=dict)
    filename_date: Optional[datetime] = None
    directory_date: Optional[str] = None
    filesystem_created: Optional[datetime] = None
    filesystem_modified: Optional[datetime] = None
    selected_date: Optional[datetime] = None
    selected_source: Optional[str] = None
    confidence: int = 0
    suspicious: bool = False
    user_verified: bool = False


class ImageMetadata(BaseModel):
    """Complete metadata for an image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    exif: Dict[str, any] = Field(default_factory=dict)
    format: Optional[str] = None
    resolution: Optional[tuple[int, int]] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None

    # Camera information
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None

    # Camera settings
    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    shutter_speed: Optional[str] = None
    iso: Optional[int] = None

    # GPS information
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None

    # Perceptual hashes for duplicate detection
    perceptual_hash_dhash: Optional[str] = None
    perceptual_hash_ahash: Optional[str] = None

    merged_from: List[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """Plan for executing operations on an image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: OperationType
    target_path: Optional[Path] = None
    target_exists: bool = False
    target_checksum: Optional[str] = None
    burst_folder: Optional[Path] = None
    reason: Optional[str] = None


class ExecutionInfo(BaseModel):
    """Information about execution status."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    executed: bool = False
    executed_at: Optional[datetime] = None
    verified: bool = False
    rollback_info: Dict[str, any] = Field(default_factory=dict)


class QualityScore(BaseModel):
    """Quality scoring for an image."""

    overall: float = 0.0
    format_score: float = 0.0
    resolution_score: float = 0.0
    size_score: float = 0.0
    metadata_score: float = 0.0
    ai_score: Optional[float] = None


class ImageRecord(BaseModel):
    """Complete record for a single image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str  # SHA256 checksum
    source_path: Path
    file_type: FileType
    checksum: str
    dates: DateInfo = Field(default_factory=DateInfo)
    metadata: ImageMetadata = Field(default_factory=ImageMetadata)
    duplicate_group_id: Optional[str] = None
    duplicate_role: Optional[DuplicateRole] = None
    burst_group_id: Optional[str] = None
    burst_role: Optional[BurstRole] = None
    status: ImageStatus = ImageStatus.PENDING
    issues: List[str] = Field(default_factory=list)
    plan: Optional[ExecutionPlan] = None
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)


class DuplicateGroup(BaseModel):
    """Group of duplicate images."""

    id: str
    images: List[str]  # Image IDs (checksums)
    primary: Optional[str] = None  # Primary image ID
    perceptual_hash: Optional[str] = None
    quality_scores: Dict[str, QualityScore] = Field(default_factory=dict)
    date_conflict: bool = False
    needs_review: bool = False
    user_override: Optional[str] = None


class BurstGroup(BaseModel):
    """Group of burst images."""

    id: str
    images: List[str]  # Image IDs (checksums)
    primary: Optional[str] = None  # Primary image ID
    time_span_seconds: float = 0.0
    ai_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    needs_review: bool = False
    user_override: Optional[str] = None


class ReviewItem(BaseModel):
    """Item in the review queue."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    type: ReviewType
    priority: ReviewPriority
    images: List[str]  # Image IDs
    description: str
    details: Dict[str, any] = Field(default_factory=dict)
    status: ReviewStatus = ReviewStatus.PENDING
    resolution: Optional[Dict[str, any]] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


class Operation(BaseModel):
    """A single file operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seq: int
    type: OperationType
    source: Path
    target: Optional[Path] = None
    checksum_before: Optional[str] = None
    checksum_after: Optional[str] = None
    status: OperationStatus = OperationStatus.PENDING
    timestamp: Optional[datetime] = None
    error: Optional[str] = None


class Transaction(BaseModel):
    """Transaction log for file operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    started: datetime
    completed: Optional[datetime] = None
    phase: str = ""
    operations: List[Operation] = Field(default_factory=list)
    rollback_available: bool = True
    status: str = "in_progress"


class Statistics(BaseModel):
    """Catalog statistics."""

    total_images: int = 0
    total_videos: int = 0
    total_size_bytes: int = 0
    organized: int = 0
    needs_review: int = 0
    no_date: int = 0
    duplicate_groups: int = 0
    duplicates_total: int = 0
    burst_groups: int = 0
    burst_images: int = 0
    unique_images: int = 0


class CatalogState(BaseModel):
    """Current state of catalog processing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    phase: CatalogPhase = CatalogPhase.ANALYZING
    last_checkpoint: Optional[datetime] = None
    checkpoint_interval_seconds: int = 300  # 5 minutes
    images_processed: int = 0
    images_total: int = 0
    progress_percentage: float = 0.0

    # Catalog-level properties (populated by get_state())
    version: str = "2.0.0"
    catalog_id: str = ""
    created: Optional[datetime] = None
    last_updated: Optional[datetime] = None


class CatalogConfiguration(BaseModel):
    """Configuration for catalog processing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_directories: List[Path] = Field(default_factory=list)
    import_directory: Optional[Path] = None
    date_format: str = "YYYY-MM"
    file_naming: str = "{date}_{time}_{checksum}.{ext}"
    burst_threshold_seconds: float = 10.0
    burst_min_images: int = 3
    ai_model: str = "hybrid"
    video_support: bool = True
    checkpoint_interval_seconds: int = 300
