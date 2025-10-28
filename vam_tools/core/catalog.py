"""
Catalog database manager.

Handles reading, writing, locking, and checkpointing the catalog database.
"""

import fcntl
import json
import logging
import shutil
import signal
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from .types import (
    BurstGroup,
    CatalogConfiguration,
    CatalogPhase,
    CatalogState,
    DuplicateGroup,
    ImageRecord,
    ReviewItem,
    Statistics,
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
        self.db_file = self.catalog_path / "catalog.json"  # Visible file
        self.backup_file = self.catalog_path / ".backup.json"  # Hidden backup
        self.lock_file = self.catalog_path / ".lock"  # Hidden lock
        self.transactions_dir = (
            self.catalog_path / ".transactions"
        )  # Hidden transactions
        self.thumbnails_dir = self.catalog_path / "thumbnails"  # Visible thumbnails

        self._lock_fd: Optional[TextIO] = None
        self._data: Optional[Dict] = None
        self._last_checkpoint: Optional[datetime] = None
        self._path_index: Optional[Dict[str, str]] = (
            None  # Maps source_path -> image_id for fast lookup
        )

        # Ensure directories exist
        self.catalog_path.mkdir(parents=True, exist_ok=True)
        self.transactions_dir.mkdir(exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)

    def __enter__(self) -> "CatalogDatabase":
        """Context manager entry - acquire lock."""
        self.acquire_lock()
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
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

        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Could not acquire catalog lock within {timeout}s")

        try:
            self._lock_fd = open(self.lock_file, "w")
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.debug("Acquired catalog lock")
        except BlockingIOError:
            logger.warning(f"Waiting for catalog lock (timeout: {timeout}s)")
            # Set up timeout using signal alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            try:
                assert self._lock_fd is not None, "Lock file should be open"
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX)
                logger.debug("Acquired catalog lock after waiting")
            finally:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restore handler
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
            "configuration": config.model_dump(mode="json"),
            "state": CatalogState().model_dump(mode="json"),
            "statistics": Statistics().model_dump(mode="json"),
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
                    self._data[key] = Statistics().model_dump(mode="json")
                elif key == "state":
                    self._data[key] = CatalogState().model_dump(mode="json")
                elif key == "configuration":
                    self._data[key] = CatalogConfiguration().model_dump(mode="json")
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
                _ = Path(image_data["source_path"])

                # Try to deserialize to validate format
                ImageRecord.model_validate(image_data)

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

        self._data["statistics"] = stats.model_dump(mode="json")

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
            self._data["configuration"] = config.model_dump(mode="json")

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
            self._data["state"] = state.model_dump(mode="json")

    def get_statistics(self) -> Statistics:
        """Get catalog statistics."""
        if not self._data:
            return Statistics()

        stats_data = self._data.get("statistics", {})
        return Statistics(**stats_data)

    def update_statistics(self, stats: Statistics) -> None:
        """Update catalog statistics."""
        if self._data:
            self._data["statistics"] = stats.model_dump(mode="json")

    def add_image(self, image: ImageRecord) -> None:
        """Add an image record to the catalog."""
        if self._data:
            self._data["images"][image.id] = image.model_dump(mode="json")
            # Update path index
            if self._path_index is not None:
                self._path_index[str(image.source_path)] = image.id

    def update_image(self, image: ImageRecord) -> None:
        """Update an existing image record in the catalog."""
        if self._data:
            self._data["images"][image.id] = image.model_dump(mode="json")
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

        return ImageRecord.model_validate(image_data)

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
            images[image_id] = ImageRecord.model_validate(image_data)

        return images

    def list_images(self) -> List[ImageRecord]:
        """Get list of all image records."""
        return list(self.get_all_images().values())

    def add_duplicate_group(self, group: DuplicateGroup) -> None:
        """Add a duplicate group."""
        if self._data:
            self._data["duplicate_groups"][group.id] = group.model_dump(mode="json")

    def save_duplicate_groups(self, groups: List[DuplicateGroup]) -> None:
        """Save multiple duplicate groups at once."""
        if self._data:
            for group in groups:
                self._data["duplicate_groups"][group.id] = group.model_dump(mode="json")

    def get_duplicate_groups(self) -> List[DuplicateGroup]:
        """Get all duplicate groups."""
        if not self._data:
            return []

        groups = []
        for group_data in self._data.get("duplicate_groups", {}).values():
            groups.append(DuplicateGroup.model_validate(group_data))

        return groups

    def get_duplicate_group(self, group_id: str) -> Optional[DuplicateGroup]:
        """Get a specific duplicate group by ID."""
        if not self._data:
            return None

        group_data = self._data.get("duplicate_groups", {}).get(group_id)
        if group_data:
            return DuplicateGroup.model_validate(group_data)
        return None

    def add_burst_group(self, group: BurstGroup) -> None:
        """Add a burst group."""
        if self._data:
            self._data["burst_groups"][group.id] = group.model_dump(mode="json")

    def add_review_item(self, item: ReviewItem) -> None:
        """Add an item to the review queue."""
        if self._data:
            self._data["review_queue"].append(item.model_dump(mode="json"))

    def get_review_queue(self) -> List[ReviewItem]:
        """Get all review queue items."""
        if not self._data:
            return []

        items = []
        for item_data in self._data.get("review_queue", []):
            items.append(ReviewItem.model_validate(item_data))

        return items

    def store_performance_statistics(self, stats_data: Dict) -> None:
        """
        Store performance statistics in the catalog.

        Args:
            stats_data: Performance statistics data (from AnalysisStatistics.model_dump())
        """
        if self._data is None:
            logger.warning("Cannot store performance statistics: catalog not loaded")
            return

        # Ensure performance_statistics section exists
        if "performance_statistics" not in self._data:
            self._data["performance_statistics"] = {
                "last_run": None,
                "history": [],
                "total_runs": 0,
                "total_files_analyzed": 0,
                "total_time_seconds": 0.0,
                "average_throughput": 0.0,
            }

        # Update with new statistics
        perf_section = self._data["performance_statistics"]
        perf_section["last_run"] = stats_data.get("last_run")
        perf_section["history"] = stats_data.get("history", [])
        perf_section["total_runs"] = stats_data.get("total_runs", 0)
        perf_section["total_files_analyzed"] = stats_data.get("total_files_analyzed", 0)
        perf_section["total_time_seconds"] = stats_data.get("total_time_seconds", 0.0)
        perf_section["average_throughput"] = stats_data.get("average_throughput", 0.0)

        logger.info("Stored performance statistics in catalog")

    def get_performance_statistics(self) -> Optional[Dict]:
        """
        Get performance statistics from the catalog.

        Returns:
            Performance statistics data or None if not available
        """
        if self._data is None:
            logger.warning("Cannot get performance statistics: catalog not loaded")
            return None

        return self._data.get("performance_statistics")
