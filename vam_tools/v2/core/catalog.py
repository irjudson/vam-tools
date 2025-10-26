"""
Catalog database manager.

Handles reading, writing, locking, and checkpointing the catalog database.
"""

import fcntl
import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .types import (
    BurstGroup,
    CatalogConfiguration,
    CatalogPhase,
    CatalogState,
    DuplicateGroup,
    ImageRecord,
    ReviewItem,
    Statistics,
    Transaction,
)

logger = logging.getLogger(__name__)


class CatalogDatabase:
    """
    Manages the catalog database with locking and checkpointing.

    Provides safe concurrent access to the catalog data with:
    - File locking for concurrent access
    - Automatic checkpointing
    - Backup and recovery
    - Transaction support
    """

    def __init__(self, catalog_path: Path):
        """
        Initialize catalog database.

        Args:
            catalog_path: Path to the organized catalog directory
        """
        self.catalog_path = Path(catalog_path)
        self.db_file = self.catalog_path / ".catalog.json"
        self.backup_file = self.catalog_path / ".catalog.backup.json"
        self.lock_file = self.catalog_path / ".catalog.lock"
        self.transactions_dir = self.catalog_path / ".transactions"

        self._lock_fd: Optional[int] = None
        self._data: Optional[Dict] = None
        self._last_checkpoint: Optional[datetime] = None
        self._path_index: Optional[Dict[str, str]] = (
            None  # Maps source_path -> image_id for fast lookup
        )

        # Ensure directories exist
        self.catalog_path.mkdir(parents=True, exist_ok=True)
        self.transactions_dir.mkdir(exist_ok=True)

    def __enter__(self) -> "CatalogDatabase":
        """Context manager entry - acquire lock."""
        self.acquire_lock()
        self.load()
        return self

    def __exit__(self, *args: any) -> None:
        """Context manager exit - release lock."""
        self.release_lock()

    def acquire_lock(self, timeout: int = 30) -> None:
        """
        Acquire exclusive lock on the catalog.

        Args:
            timeout: Maximum seconds to wait for lock

        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        try:
            self._lock_fd = open(self.lock_file, "w")
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.debug("Acquired catalog lock")
        except BlockingIOError:
            logger.warning(f"Waiting for catalog lock (timeout: {timeout}s)")
            # TODO: Implement timeout
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX)
        except Exception as e:
            logger.error(f"Error acquiring lock: {e}")
            raise

    def release_lock(self) -> None:
        """Release catalog lock."""
        if self._lock_fd:
            try:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
                self._lock_fd.close()
                self._lock_fd = None
                logger.debug("Released catalog lock")
            except Exception as e:
                logger.error(f"Error releasing lock: {e}")

    def initialize(
        self,
        source_directories: List[Path],
        config: Optional[CatalogConfiguration] = None,
    ) -> None:
        """
        Initialize a new catalog database.

        Args:
            source_directories: List of source directories to catalog
            config: Optional catalog configuration
        """
        if config is None:
            config = CatalogConfiguration()

        config.source_directories = source_directories

        self._data = {
            "version": "2.0.0",
            "catalog_path": str(self.catalog_path),
            "catalog_id": str(uuid.uuid4()),
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "configuration": self._serialize_config(config),
            "state": self._serialize_state(CatalogState()),
            "statistics": self._serialize_stats(Statistics()),
            "images": {},
            "duplicate_groups": {},
            "burst_groups": {},
            "review_queue": [],
            "transactions": {"current": None, "history": []},
        }

        # Initialize empty path index
        self._path_index = {}

        self.save()
        logger.info(f"Initialized new catalog at {self.catalog_path}")

    def load(self) -> None:
        """Load catalog from disk."""
        if not self.db_file.exists():
            logger.warning(f"Catalog does not exist: {self.db_file}")
            self._data = None
            self._path_index = {}
            return

        try:
            with open(self.db_file, "r", encoding="utf-8") as f:
                self._data = json.load(f)

            # Build path index for fast lookups
            self._build_path_index()

            logger.info(f"Loaded catalog from {self.db_file}")
        except Exception as e:
            logger.error(f"Error loading catalog: {e}")

            # Try to load from backup
            if self.backup_file.exists():
                logger.warning("Attempting to load from backup")
                try:
                    with open(self.backup_file, "r", encoding="utf-8") as f:
                        self._data = json.load(f)
                    self._build_path_index()
                    logger.info("Successfully loaded from backup")
                except Exception as backup_error:
                    logger.error(f"Failed to load backup: {backup_error}")
                    raise

    def _build_path_index(self) -> None:
        """Build path index for fast path-based lookups."""
        self._path_index = {}
        if self._data:
            for image_id, image_data in self._data.get("images", {}).items():
                source_path = image_data.get("source_path")
                if source_path:
                    self._path_index[source_path] = image_id
        logger.debug(f"Built path index with {len(self._path_index)} entries")

    def save(self, create_backup: bool = True) -> None:
        """
        Save catalog to disk.

        Args:
            create_backup: Whether to create backup before saving
        """
        if self._data is None:
            logger.warning("No data to save")
            return

        # Update last_updated timestamp
        self._data["last_updated"] = datetime.now().isoformat()

        # Create backup of existing catalog
        if create_backup and self.db_file.exists():
            try:
                shutil.copy2(self.db_file, self.backup_file)
                logger.debug("Created backup")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

        # Write to temp file first, then atomic rename
        temp_file = self.db_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)

            # Atomic rename
            temp_file.replace(self.db_file)
            logger.debug(f"Saved catalog to {self.db_file}")

        except Exception as e:
            logger.error(f"Error saving catalog: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise

    def checkpoint(self, force: bool = False) -> None:
        """
        Create a checkpoint (save current state).

        Args:
            force: Force checkpoint even if interval hasn't elapsed
        """
        if not force:
            # Check if checkpoint interval has elapsed
            if self._last_checkpoint:
                config = self.get_configuration()
                elapsed = (datetime.now() - self._last_checkpoint).total_seconds()
                if elapsed < config.checkpoint_interval_seconds:
                    return

        self.save(create_backup=False)
        self._last_checkpoint = datetime.now()

        # Update checkpoint time in state
        state = self.get_state()
        state.last_checkpoint = self._last_checkpoint
        self.update_state(state)

        logger.info("Created checkpoint")

    def repair(self) -> None:
        """
        Repair a corrupted catalog database.

        This method attempts to:
        1. Load catalog from backup if main file is corrupted
        2. Validate and fix data structure
        3. Rebuild indexes
        4. Recalculate statistics
        5. Remove invalid entries
        """
        logger.info("Starting catalog repair...")

        # Try to load catalog, fall back to backup if needed
        if not self._data:
            logger.info("Attempting to load catalog...")
            try:
                if self.db_file.exists():
                    with open(self.db_file, "r", encoding="utf-8") as f:
                        self._data = json.load(f)
                    logger.info("Loaded main catalog file")
            except Exception as e:
                logger.warning(f"Main catalog file corrupted: {e}")
                if self.backup_file.exists():
                    logger.info("Attempting to load from backup...")
                    with open(self.backup_file, "r", encoding="utf-8") as f:
                        self._data = json.load(f)
                    logger.info("Loaded backup catalog file")
                else:
                    raise RuntimeError("No valid catalog or backup file found")

        if not self._data:
            raise RuntimeError("No catalog data to repair")

        # Validate and fix data structure
        logger.info("Validating data structure...")
        required_keys = [
            "version",
            "catalog_path",
            "catalog_id",
            "created",
            "last_updated",
            "configuration",
            "state",
            "statistics",
            "images",
        ]

        for key in required_keys:
            if key not in self._data:
                logger.warning(f"Missing key '{key}', adding default value")
                if key == "images":
                    self._data[key] = {}
                elif key == "statistics":
                    self._data[key] = self._serialize_stats(Statistics())
                elif key == "state":
                    self._data[key] = self._serialize_state(CatalogState())
                elif key == "configuration":
                    self._data[key] = self._serialize_config(CatalogConfiguration())
                elif key == "version":
                    self._data[key] = "2.0.0"
                elif key == "catalog_id":
                    self._data[key] = str(uuid.uuid4())
                elif key == "catalog_path":
                    self._data[key] = str(self.catalog_path)
                elif key in ["created", "last_updated"]:
                    self._data[key] = datetime.now().isoformat()

        # Ensure optional keys exist
        if "duplicate_groups" not in self._data:
            self._data["duplicate_groups"] = {}
        if "burst_groups" not in self._data:
            self._data["burst_groups"] = {}
        if "review_queue" not in self._data:
            self._data["review_queue"] = []
        if "transactions" not in self._data:
            self._data["transactions"] = {"current": None, "history": []}

        # Validate and clean image records
        logger.info("Validating image records...")
        valid_images = {}
        invalid_count = 0

        for image_id, image_data in self._data.get("images", {}).items():
            try:
                # Check required fields
                if not isinstance(image_data, dict):
                    raise ValueError("Image data is not a dictionary")

                required_image_keys = [
                    "id",
                    "source_path",
                    "file_type",
                    "checksum",
                    "status",
                ]
                for key in required_image_keys:
                    if key not in image_data:
                        raise ValueError(f"Missing required field: {key}")

                # Validate source path exists (if possible)
                source_path = Path(image_data["source_path"])

                # Try to deserialize to validate format
                self._deserialize_image(image_data)

                # Image is valid
                valid_images[image_id] = image_data

            except Exception as e:
                logger.warning(f"Invalid image record {image_id}: {e}")
                invalid_count += 1

        logger.info(
            f"Found {len(valid_images)} valid images, removed {invalid_count} invalid records"
        )
        self._data["images"] = valid_images

        # Rebuild path index
        logger.info("Rebuilding path index...")
        self._build_path_index()

        # Recalculate statistics
        logger.info("Recalculating statistics...")
        stats = Statistics()

        for image_data in self._data["images"].values():
            file_type = image_data.get("file_type")
            if file_type == "image":
                stats.total_images += 1
            elif file_type == "video":
                stats.total_videos += 1

            if "metadata" in image_data:
                stats.total_size_bytes += image_data["metadata"].get("size_bytes", 0)

            if "dates" in image_data:
                if not image_data["dates"].get("selected_date"):
                    stats.no_date += 1

        self._data["statistics"] = self._serialize_stats(stats)

        # Update last_updated timestamp
        self._data["last_updated"] = datetime.now().isoformat()

        # Save repaired catalog
        logger.info("Saving repaired catalog...")
        self.save(create_backup=True)

        logger.info("Catalog repair complete")

    # Getters and setters for catalog components

    def get_configuration(self) -> CatalogConfiguration:
        """Get catalog configuration."""
        if not self._data:
            return CatalogConfiguration()

        config_data = self._data.get("configuration", {})
        return CatalogConfiguration(
            source_directories=[
                Path(p) for p in config_data.get("source_directories", [])
            ],
            import_directory=(
                Path(config_data["import_directory"])
                if config_data.get("import_directory")
                else None
            ),
            date_format=config_data.get("date_format", "YYYY-MM"),
            file_naming=config_data.get(
                "file_naming", "{date}_{time}_{checksum}.{ext}"
            ),
            burst_threshold_seconds=config_data.get("burst_threshold_seconds", 10.0),
            burst_min_images=config_data.get("burst_min_images", 3),
            ai_model=config_data.get("ai_model", "hybrid"),
            video_support=config_data.get("video_support", True),
            checkpoint_interval_seconds=config_data.get(
                "checkpoint_interval_seconds", 300
            ),
        )

    def update_configuration(self, config: CatalogConfiguration) -> None:
        """Update catalog configuration."""
        if self._data:
            self._data["configuration"] = self._serialize_config(config)

    def get_state(self) -> CatalogState:
        """Get current catalog state."""
        if not self._data:
            return CatalogState()

        state_data = self._data.get("state", {})
        state = CatalogState(
            phase=CatalogPhase(state_data.get("phase", "analyzing")),
            last_checkpoint=(
                datetime.fromisoformat(state_data["last_checkpoint"])
                if state_data.get("last_checkpoint")
                else None
            ),
            checkpoint_interval_seconds=state_data.get(
                "checkpoint_interval_seconds", 300
            ),
            images_processed=state_data.get("images_processed", 0),
            images_total=state_data.get("images_total", 0),
            progress_percentage=state_data.get("progress_percentage", 0.0),
        )

        # Add catalog-level properties to state for convenience
        state.version = self._data.get("version", "2.0.0")
        state.catalog_id = self._data.get("catalog_id", "")
        state.created = (
            datetime.fromisoformat(self._data["created"])
            if self._data.get("created")
            else None
        )
        state.last_updated = (
            datetime.fromisoformat(self._data["last_updated"])
            if self._data.get("last_updated")
            else None
        )

        return state

    def update_state(self, state: CatalogState) -> None:
        """Update catalog state."""
        if self._data:
            self._data["state"] = self._serialize_state(state)

    def get_statistics(self) -> Statistics:
        """Get catalog statistics."""
        if not self._data:
            return Statistics()

        stats_data = self._data.get("statistics", {})
        return Statistics(**stats_data)

    def update_statistics(self, stats: Statistics) -> None:
        """Update catalog statistics."""
        if self._data:
            self._data["statistics"] = self._serialize_stats(stats)

    def add_image(self, image: ImageRecord) -> None:
        """Add an image record to the catalog."""
        if self._data:
            self._data["images"][image.id] = self._serialize_image(image)
            # Update path index
            if self._path_index is not None:
                self._path_index[str(image.source_path)] = image.id

    def get_image(self, image_id: str) -> Optional[ImageRecord]:
        """Get an image record by ID."""
        if not self._data:
            return None

        image_data = self._data.get("images", {}).get(image_id)
        if not image_data:
            return None

        return self._deserialize_image(image_data)

    def has_image_by_path(self, source_path: Path) -> bool:
        """Check if an image with the given source path exists in the catalog."""
        if not self._path_index:
            return False

        return str(source_path) in self._path_index

    def get_all_images(self) -> Dict[str, ImageRecord]:
        """Get all image records."""
        if not self._data:
            return {}

        images = {}
        for image_id, image_data in self._data.get("images", {}).items():
            images[image_id] = self._deserialize_image(image_data)

        return images

    def list_images(self) -> List[ImageRecord]:
        """Get list of all image records."""
        return list(self.get_all_images().values())

    def add_duplicate_group(self, group: DuplicateGroup) -> None:
        """Add a duplicate group."""
        if self._data:
            self._data["duplicate_groups"][group.id] = self._serialize_duplicate_group(
                group
            )

    def add_burst_group(self, group: BurstGroup) -> None:
        """Add a burst group."""
        if self._data:
            self._data["burst_groups"][group.id] = self._serialize_burst_group(group)

    def add_review_item(self, item: ReviewItem) -> None:
        """Add an item to the review queue."""
        if self._data:
            self._data["review_queue"].append(self._serialize_review_item(item))

    def get_review_queue(self) -> List[ReviewItem]:
        """Get all review queue items."""
        if not self._data:
            return []

        items = []
        for item_data in self._data.get("review_queue", []):
            items.append(self._deserialize_review_item(item_data))

        return items

    # Serialization helpers

    def _serialize_config(self, config: CatalogConfiguration) -> Dict:
        """Serialize configuration to dict."""
        return {
            "source_directories": [str(p) for p in config.source_directories],
            "import_directory": (
                str(config.import_directory) if config.import_directory else None
            ),
            "date_format": config.date_format,
            "file_naming": config.file_naming,
            "burst_threshold_seconds": config.burst_threshold_seconds,
            "burst_min_images": config.burst_min_images,
            "ai_model": config.ai_model,
            "video_support": config.video_support,
            "checkpoint_interval_seconds": config.checkpoint_interval_seconds,
        }

    def _serialize_state(self, state: CatalogState) -> Dict:
        """Serialize state to dict."""
        return {
            "phase": state.phase.value,
            "last_checkpoint": (
                state.last_checkpoint.isoformat() if state.last_checkpoint else None
            ),
            "checkpoint_interval_seconds": state.checkpoint_interval_seconds,
            "images_processed": state.images_processed,
            "images_total": state.images_total,
            "progress_percentage": state.progress_percentage,
        }

    def _serialize_stats(self, stats: Statistics) -> Dict:
        """Serialize statistics to dict."""
        return {
            "total_images": stats.total_images,
            "total_videos": stats.total_videos,
            "total_size_bytes": stats.total_size_bytes,
            "organized": stats.organized,
            "needs_review": stats.needs_review,
            "no_date": stats.no_date,
            "duplicate_groups": stats.duplicate_groups,
            "duplicates_total": stats.duplicates_total,
            "burst_groups": stats.burst_groups,
            "burst_images": stats.burst_images,
            "unique_images": stats.unique_images,
        }

    def _serialize_image(self, image: ImageRecord) -> Dict:
        """Serialize image record to dict."""
        data = {
            "id": image.id,
            "source_path": str(image.source_path),
            "file_type": image.file_type.value,
            "checksum": image.checksum,
            "status": image.status.value,
        }

        # Add dates if present
        if image.dates:
            data["dates"] = {
                "exif_dates": {
                    k: v.isoformat() if v else None
                    for k, v in image.dates.exif_dates.items()
                },
                "filename_date": (
                    image.dates.filename_date.isoformat()
                    if image.dates.filename_date
                    else None
                ),
                "directory_date": image.dates.directory_date,
                "filesystem_created": (
                    image.dates.filesystem_created.isoformat()
                    if image.dates.filesystem_created
                    else None
                ),
                "filesystem_modified": (
                    image.dates.filesystem_modified.isoformat()
                    if image.dates.filesystem_modified
                    else None
                ),
                "selected_date": (
                    image.dates.selected_date.isoformat()
                    if image.dates.selected_date
                    else None
                ),
                "selected_source": image.dates.selected_source,
                "confidence": image.dates.confidence,
                "suspicious": image.dates.suspicious,
                "user_verified": image.dates.user_verified,
            }

        # Add metadata if present
        if image.metadata:
            data["metadata"] = {
                "format": image.metadata.format,
                "resolution": image.metadata.resolution,
                "size_bytes": image.metadata.size_bytes,
                "exif": image.metadata.exif,
            }

        return data

    def _deserialize_image(self, data: Dict) -> ImageRecord:
        """Deserialize image record from dict."""
        from .types import DateInfo, FileType, ImageMetadata, ImageStatus

        # Deserialize dates
        dates = None
        if "dates" in data:
            date_data = data["dates"]
            dates = DateInfo(
                exif_dates={
                    k: datetime.fromisoformat(v) if v else None
                    for k, v in date_data.get("exif_dates", {}).items()
                },
                filename_date=(
                    datetime.fromisoformat(date_data["filename_date"])
                    if date_data.get("filename_date")
                    else None
                ),
                directory_date=date_data.get("directory_date"),
                filesystem_created=(
                    datetime.fromisoformat(date_data["filesystem_created"])
                    if date_data.get("filesystem_created")
                    else None
                ),
                filesystem_modified=(
                    datetime.fromisoformat(date_data["filesystem_modified"])
                    if date_data.get("filesystem_modified")
                    else None
                ),
                selected_date=(
                    datetime.fromisoformat(date_data["selected_date"])
                    if date_data.get("selected_date")
                    else None
                ),
                selected_source=date_data.get("selected_source"),
                confidence=date_data.get("confidence", 0),
                suspicious=date_data.get("suspicious", False),
                user_verified=date_data.get("user_verified", False),
            )

        # Deserialize metadata
        metadata = None
        if "metadata" in data:
            meta_data = data["metadata"]
            metadata = ImageMetadata(
                format=meta_data.get("format"),
                resolution=(
                    tuple(meta_data["resolution"])
                    if meta_data.get("resolution")
                    else None
                ),
                size_bytes=meta_data.get("size_bytes"),
                exif=meta_data.get("exif"),
            )

        return ImageRecord(
            id=data["id"],
            source_path=Path(data["source_path"]),
            file_type=FileType(data["file_type"]),
            checksum=data["checksum"],
            status=ImageStatus(data["status"]),
            dates=dates,
            metadata=metadata,
        )

    def _serialize_duplicate_group(self, group: DuplicateGroup) -> Dict:
        """Serialize duplicate group to dict."""
        return {
            "id": group.id,
            "images": group.images,
            "primary": group.primary,
            "needs_review": group.needs_review,
        }

    def _serialize_burst_group(self, group: BurstGroup) -> Dict:
        """Serialize burst group to dict."""
        return {
            "id": group.id,
            "images": group.images,
            "primary": group.primary,
            "time_span_seconds": group.time_span_seconds,
            "needs_review": group.needs_review,
        }

    def _serialize_review_item(self, item: ReviewItem) -> Dict:
        """Serialize review item to dict."""
        return {
            "id": item.id,
            "type": item.type.value,
            "priority": item.priority.value,
            "images": item.images,
            "description": item.description,
            "status": item.status.value,
        }

    def _deserialize_review_item(self, data: Dict) -> ReviewItem:
        """Deserialize review item from dict."""
        from .types import ReviewPriority, ReviewStatus, ReviewType

        return ReviewItem(
            id=data["id"],
            type=ReviewType(data["type"]),
            priority=ReviewPriority(data["priority"]),
            images=data["images"],
            description=data["description"],
            status=ReviewStatus(data["status"]),
        )
