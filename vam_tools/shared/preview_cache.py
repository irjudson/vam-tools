"""
Preview cache manager for extracted RAW and converted image previews.

Manages a disk-based cache of extracted previews with LRU eviction based on
configurable size limits. This dramatically improves web UI performance by
avoiding repeated extraction of RAW file previews.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default cache configuration
DEFAULT_CACHE_SIZE_GB = 10
DEFAULT_CACHE_SIZE_BYTES = DEFAULT_CACHE_SIZE_GB * 1024 * 1024 * 1024


class PreviewCache:
    """
    Manages a disk-based cache of extracted image previews with LRU eviction.

    The cache stores extracted RAW previews and converted images (HEIC, TIFF)
    to avoid repeated extraction, which can take 30+ seconds on network storage.

    Features:
    - LRU (Least Recently Used) eviction when size limit exceeded
    - Configurable cache size limit (default: 10 GB)
    - Automatic cache cleanup on startup
    - Fast lookups using metadata index

    Example:
        >>> cache = PreviewCache(catalog_path, max_size_bytes=10*1024**3)
        >>> # Check if preview exists
        >>> if cache.has_preview("abc123"):
        ...     preview_path = cache.get_preview_path("abc123")
        >>> else:
        ...     # Extract and store preview
        ...     cache.store_preview("abc123", extracted_jpeg_bytes)
    """

    def __init__(
        self,
        catalog_path: Path,
        max_size_bytes: int = DEFAULT_CACHE_SIZE_BYTES,
    ):
        """
        Initialize the preview cache.

        Args:
            catalog_path: Path to the catalog directory
            max_size_bytes: Maximum cache size in bytes (default: 10 GB)
        """
        self.catalog_path = Path(catalog_path)
        self.cache_dir = self.catalog_path / "previews"
        self.metadata_file = self.cache_dir / ".cache_metadata.json"
        self.max_size_bytes = max_size_bytes

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize metadata
        self.metadata: Dict[str, Dict[str, Any]] = self._load_metadata()

        # Verify cache integrity on startup
        self._verify_cache_integrity()

        # Perform cleanup if over size limit
        self._enforce_size_limit()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data: Dict[str, Dict[str, Any]] = json.load(f)
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}, starting fresh")
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _verify_cache_integrity(self) -> None:
        """
        Verify cache integrity and remove orphaned entries.

        Removes metadata entries for files that no longer exist and
        updates file sizes if they've changed.
        """
        orphaned = []
        updated = False

        for image_id, meta in list(self.metadata.items()):
            preview_path = self.cache_dir / f"{image_id}.jpg"

            if not preview_path.exists():
                # File was deleted, remove from metadata
                orphaned.append(image_id)
                updated = True
            else:
                # Update size if changed
                actual_size = preview_path.stat().st_size
                if meta.get("size_bytes") != actual_size:
                    meta["size_bytes"] = actual_size
                    updated = True

        # Remove orphaned entries
        for image_id in orphaned:
            del self.metadata[image_id]
            logger.debug(f"Removed orphaned cache entry: {image_id}")

        if updated:
            self._save_metadata()
            logger.info(
                f"Cache integrity verified: {len(orphaned)} orphaned entries removed"
            )

    def _enforce_size_limit(self) -> None:
        """
        Enforce cache size limit using LRU eviction.

        Removes least recently used previews until cache is under size limit.
        """
        total_size = self.get_cache_size()

        if total_size <= self.max_size_bytes:
            return

        logger.info(
            f"Cache size ({total_size / 1024**3:.2f} GB) exceeds limit "
            f"({self.max_size_bytes / 1024**3:.2f} GB), performing LRU eviction"
        )

        # Sort by last_accessed (oldest first)
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1].get("last_accessed", 0),
        )

        evicted_count = 0
        freed_bytes = 0

        for image_id, meta in sorted_entries:
            if total_size <= self.max_size_bytes * 0.9:  # Leave 10% buffer
                break

            preview_path = self.cache_dir / f"{image_id}.jpg"
            file_size = meta.get("size_bytes", 0)

            try:
                if preview_path.exists():
                    preview_path.unlink()
                del self.metadata[image_id]

                total_size -= file_size
                freed_bytes += file_size
                evicted_count += 1

            except Exception as e:
                logger.error(f"Failed to evict {image_id}: {e}")

        self._save_metadata()

        logger.info(
            f"LRU eviction complete: {evicted_count} previews removed, "
            f"{freed_bytes / 1024**2:.2f} MB freed"
        )

    def has_preview(self, image_id: str) -> bool:
        """
        Check if a preview exists in the cache.

        Args:
            image_id: Image ID (checksum)

        Returns:
            True if preview is cached
        """
        preview_path = self.cache_dir / f"{image_id}.jpg"
        return image_id in self.metadata and preview_path.exists()

    def get_preview_path(
        self, image_id: str, update_access_time: bool = True
    ) -> Optional[Path]:
        """
        Get the path to a cached preview.

        Args:
            image_id: Image ID (checksum)
            update_access_time: Whether to update last access time (default: True)

        Returns:
            Path to the cached preview, or None if not found
        """
        if not self.has_preview(image_id):
            return None

        preview_path = self.cache_dir / f"{image_id}.jpg"

        # Update access time for LRU
        if update_access_time and image_id in self.metadata:
            self.metadata[image_id]["last_accessed"] = time.time()
            self._save_metadata()

        return preview_path

    def store_preview(self, image_id: str, preview_bytes: bytes) -> bool:
        """
        Store a preview in the cache.

        Args:
            image_id: Image ID (checksum)
            preview_bytes: JPEG preview data

        Returns:
            True if stored successfully
        """
        try:
            preview_path = self.cache_dir / f"{image_id}.jpg"

            # Write preview to disk
            with open(preview_path, "wb") as f:
                f.write(preview_bytes)

            # Update metadata
            file_size = len(preview_bytes)
            current_time = time.time()

            self.metadata[image_id] = {
                "size_bytes": file_size,
                "created_at": current_time,
                "last_accessed": current_time,
            }

            self._save_metadata()

            # Check if we need to evict
            if self.get_cache_size() > self.max_size_bytes:
                self._enforce_size_limit()

            logger.debug(f"Stored preview for {image_id} ({file_size / 1024:.2f} KB)")
            return True

        except Exception as e:
            logger.error(f"Failed to store preview for {image_id}: {e}")
            return False

    def remove_preview(self, image_id: str) -> bool:
        """
        Remove a preview from the cache.

        Args:
            image_id: Image ID (checksum)

        Returns:
            True if removed successfully
        """
        try:
            preview_path = self.cache_dir / f"{image_id}.jpg"

            if preview_path.exists():
                preview_path.unlink()

            if image_id in self.metadata:
                del self.metadata[image_id]
                self._save_metadata()

            return True

        except Exception as e:
            logger.error(f"Failed to remove preview for {image_id}: {e}")
            return False

    def get_cache_size(self) -> int:
        """
        Get the current cache size in bytes.

        Returns:
            Total size of all cached previews in bytes
        """
        return sum(meta.get("size_bytes", 0) for meta in self.metadata.values())

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_size = self.get_cache_size()
        num_previews = len(self.metadata)

        return {
            "num_previews": num_previews,
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "max_size_bytes": self.max_size_bytes,
            "max_size_gb": self.max_size_bytes / (1024**3),
            "usage_percent": (
                (total_size / self.max_size_bytes * 100)
                if self.max_size_bytes > 0
                else 0
            ),
        }

    def clear_cache(self) -> int:
        """
        Clear the entire preview cache.

        Returns:
            Number of previews removed
        """
        count = 0

        for image_id in list(self.metadata.keys()):
            if self.remove_preview(image_id):
                count += 1

        logger.info(f"Cache cleared: {count} previews removed")
        return count
