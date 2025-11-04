"""Tests for preview cache management."""

import time
from pathlib import Path

import pytest

from vam_tools.shared.preview_cache import PreviewCache


class TestPreviewCache:
    """Tests for PreviewCache class."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create a temporary cache directory."""
        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir()
        return catalog_path

    @pytest.fixture
    def cache(self, cache_dir: Path) -> PreviewCache:
        """Create a PreviewCache instance."""
        return PreviewCache(cache_dir, max_size_bytes=1024 * 1024)  # 1 MB limit

    def test_initialization(self, cache_dir: Path) -> None:
        """Test cache initialization."""
        cache = PreviewCache(cache_dir)

        assert cache.catalog_path == cache_dir
        assert cache.cache_dir == cache_dir / "previews"
        assert cache.cache_dir.exists()
        assert cache.metadata_file == cache_dir / "previews" / ".cache_metadata.json"

    def test_store_and_retrieve_preview(self, cache: PreviewCache) -> None:
        """Test storing and retrieving a preview."""
        image_id = "abc123"
        preview_data = b"fake jpeg data" * 100  # ~1.4 KB

        # Store preview
        result = cache.store_preview(image_id, preview_data)
        assert result is True

        # Check if preview exists
        assert cache.has_preview(image_id)

        # Get preview path
        path = cache.get_preview_path(image_id)
        assert path is not None
        assert path.exists()
        assert path.read_bytes() == preview_data

    def test_store_preview_updates_metadata(self, cache: PreviewCache) -> None:
        """Test that storing a preview updates metadata."""
        image_id = "def456"
        preview_data = b"test data"

        cache.store_preview(image_id, preview_data)

        assert image_id in cache.metadata
        assert "size_bytes" in cache.metadata[image_id]
        assert "created_at" in cache.metadata[image_id]
        assert "last_accessed" in cache.metadata[image_id]
        assert cache.metadata[image_id]["size_bytes"] == len(preview_data)

    def test_get_preview_path_updates_access_time(self, cache: PreviewCache) -> None:
        """Test that getting a preview updates the access time."""
        image_id = "ghi789"
        preview_data = b"access time test"

        cache.store_preview(image_id, preview_data)
        initial_access_time = cache.metadata[image_id]["last_accessed"]

        # Wait a bit to ensure time difference
        time.sleep(0.1)

        # Get preview with access time update
        cache.get_preview_path(image_id, update_access_time=True)
        new_access_time = cache.metadata[image_id]["last_accessed"]

        assert new_access_time > initial_access_time

    def test_get_preview_path_without_access_time_update(
        self, cache: PreviewCache
    ) -> None:
        """Test getting preview without updating access time."""
        image_id = "jkl012"
        preview_data = b"no update test"

        cache.store_preview(image_id, preview_data)
        initial_access_time = cache.metadata[image_id]["last_accessed"]

        time.sleep(0.1)

        # Get preview without access time update
        cache.get_preview_path(image_id, update_access_time=False)
        new_access_time = cache.metadata[image_id]["last_accessed"]

        assert new_access_time == initial_access_time

    def test_has_preview_false_for_missing(self, cache: PreviewCache) -> None:
        """Test has_preview returns False for non-existent preview."""
        assert cache.has_preview("nonexistent") is False

    def test_get_preview_path_returns_none_for_missing(
        self, cache: PreviewCache
    ) -> None:
        """Test get_preview_path returns None for non-existent preview."""
        assert cache.get_preview_path("nonexistent") is None

    def test_remove_preview(self, cache: PreviewCache) -> None:
        """Test removing a preview."""
        image_id = "mno345"
        preview_data = b"remove test"

        cache.store_preview(image_id, preview_data)
        assert cache.has_preview(image_id)

        # Remove preview
        result = cache.remove_preview(image_id)
        assert result is True
        assert not cache.has_preview(image_id)
        assert image_id not in cache.metadata

    def test_get_cache_size(self, cache: PreviewCache) -> None:
        """Test getting cache size."""
        # Initially empty
        assert cache.get_cache_size() == 0

        # Add some previews
        cache.store_preview("img1", b"x" * 100)
        cache.store_preview("img2", b"y" * 200)
        cache.store_preview("img3", b"z" * 300)

        # Size should be sum of all
        assert cache.get_cache_size() == 600

    def test_get_cache_stats(self, cache: PreviewCache) -> None:
        """Test getting cache statistics."""
        cache.store_preview("img1", b"a" * 1000)
        cache.store_preview("img2", b"b" * 2000)

        stats = cache.get_cache_stats()

        assert stats["num_previews"] == 2
        assert stats["total_size_bytes"] == 3000
        assert stats["max_size_bytes"] == 1024 * 1024
        assert "total_size_gb" in stats
        assert "usage_percent" in stats
        assert stats["usage_percent"] > 0

    def test_clear_cache(self, cache: PreviewCache) -> None:
        """Test clearing the entire cache."""
        # Add some previews
        cache.store_preview("img1", b"data1")
        cache.store_preview("img2", b"data2")
        cache.store_preview("img3", b"data3")

        assert len(cache.metadata) == 3

        # Clear cache
        count = cache.clear_cache()

        assert count == 3
        assert len(cache.metadata) == 0
        assert cache.get_cache_size() == 0

    def test_lru_eviction(self, cache: PreviewCache) -> None:
        """Test LRU eviction when cache exceeds size limit."""
        # Cache has 1 MB limit
        # Add files that together exceed the limit
        data_size = 300 * 1024  # 300 KB each

        # Add 4 files = 1.2 MB total
        cache.store_preview("img1", b"x" * data_size)
        time.sleep(0.01)
        cache.store_preview("img2", b"y" * data_size)
        time.sleep(0.01)
        cache.store_preview("img3", b"z" * data_size)
        time.sleep(0.01)
        cache.store_preview("img4", b"w" * data_size)

        # Should have triggered eviction
        # Oldest files should be evicted
        assert not cache.has_preview("img1")  # Oldest, should be evicted
        assert cache.has_preview("img4")  # Newest, should remain

    def test_metadata_persistence(self, cache_dir: Path) -> None:
        """Test that metadata persists across cache instances."""
        # Create first cache and add preview
        cache1 = PreviewCache(cache_dir)
        cache1.store_preview("persistent", b"test data")

        # Create second cache (simulates restart)
        cache2 = PreviewCache(cache_dir)

        # Should still have the preview
        assert cache2.has_preview("persistent")
        assert "persistent" in cache2.metadata

    def test_verify_cache_integrity_removes_orphaned_entries(
        self, cache: PreviewCache
    ) -> None:
        """Test cache integrity verification removes orphaned metadata."""
        # Add preview normally
        cache.store_preview("img1", b"data")

        # Manually delete the file but keep metadata
        preview_path = cache.cache_dir / "img1.jpg"
        preview_path.unlink()

        # Create new cache instance (triggers integrity check)
        cache2 = PreviewCache(cache.catalog_path)

        # Orphaned entry should be removed
        assert "img1" not in cache2.metadata

    def test_verify_cache_integrity_updates_sizes(self, cache: PreviewCache) -> None:
        """Test cache integrity updates file sizes if changed."""
        # Add preview
        cache.store_preview("img1", b"original")
        original_size = cache.metadata["img1"]["size_bytes"]

        # Manually modify the file
        preview_path = cache.cache_dir / "img1.jpg"
        preview_path.write_bytes(b"modified data with different size")

        # Create new cache instance (triggers integrity check)
        cache2 = PreviewCache(cache.catalog_path)

        # Size should be updated
        new_size = cache2.metadata["img1"]["size_bytes"]
        assert new_size != original_size
        assert new_size == len(b"modified data with different size")

    def test_metadata_file_corruption_recovery(self, cache_dir: Path) -> None:
        """Test recovery from corrupted metadata file."""
        # Create cache
        cache1 = PreviewCache(cache_dir)
        cache1.store_preview("img1", b"data")

        # Corrupt metadata file
        metadata_file = cache_dir / "previews" / ".cache_metadata.json"
        metadata_file.write_text("{ invalid json ")

        # Create new cache (should recover gracefully)
        cache2 = PreviewCache(cache_dir)

        # Should start fresh with empty metadata
        assert len(cache2.metadata) == 0

    def test_store_preview_failure_handling(self, cache: PreviewCache) -> None:
        """Test handling of storage failures."""
        # Try to store with invalid data (simulated by making dir read-only won't work)
        # Instead, test with a very long filename that might fail
        very_long_id = "x" * 300  # Most filesystems have 255 char limit
        result = cache.store_preview(very_long_id, b"data")

        # Should handle gracefully (might succeed or fail depending on OS)
        assert isinstance(result, bool)

    def test_empty_preview_data(self, cache: PreviewCache) -> None:
        """Test storing empty preview data."""
        result = cache.store_preview("empty", b"")
        assert result is True
        assert cache.has_preview("empty")
        assert cache.get_cache_size() == 0  # Empty file contributes 0 bytes

    def test_multiple_store_overwrites(self, cache: PreviewCache) -> None:
        """Test that storing the same preview multiple times overwrites."""
        image_id = "overwrite_test"

        cache.store_preview(image_id, b"first")
        assert len(cache.metadata) == 1

        cache.store_preview(image_id, b"second version")
        assert len(cache.metadata) == 1  # Still only one entry

        path = cache.get_preview_path(image_id)
        assert path is not None
        assert path.read_bytes() == b"second version"

    def test_save_metadata_error_handling(
        self, cache: PreviewCache, monkeypatch
    ) -> None:
        """Test that save metadata handles errors gracefully."""
        import builtins

        # Mock open to raise an exception when saving metadata
        original_open = builtins.open

        def mock_open(*args, **kwargs):
            # Allow reading, but fail on writes to metadata file
            if (
                len(args) > 0
                and str(args[0]).endswith(".cache_metadata.json")
                and "w" in args[1]
            ):
                raise OSError("Mock write error")
            return original_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)

        # This should not crash, just log an error
        cache.store_preview("test", b"data")
        # Metadata save failed, but preview should still be stored
        assert cache.has_preview("test")

    def test_eviction_error_handling(self, cache: PreviewCache, monkeypatch) -> None:
        """Test that eviction handles errors gracefully."""
        # Add a preview normally
        cache.store_preview("test1", b"x" * 100)

        # Mock Path.unlink to raise an exception
        from pathlib import Path

        original_unlink = Path.unlink

        def mock_unlink(self, *args, **kwargs):
            if str(self).endswith("test1.jpg"):
                raise OSError("Mock delete error")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", mock_unlink)

        # Try to trigger eviction by filling cache beyond limit
        # Cache has 1 MB limit, add files to exceed it
        data_size = 300 * 1024  # 300 KB
        cache.store_preview("test2", b"y" * data_size)
        cache.store_preview("test3", b"z" * data_size)
        cache.store_preview("test4", b"w" * data_size)

        # Eviction should handle the error and continue

    def test_remove_preview_error_handling(
        self, cache: PreviewCache, monkeypatch
    ) -> None:
        """Test that remove_preview handles errors gracefully."""
        cache.store_preview("test", b"data")

        # Mock Path.unlink to raise an exception
        from pathlib import Path

        original_unlink = Path.unlink

        def mock_unlink(self, *args, **kwargs):
            raise PermissionError("Mock permission error")

        monkeypatch.setattr(Path, "unlink", mock_unlink)

        # Should return False but not crash
        result = cache.remove_preview("test")
        assert result is False
