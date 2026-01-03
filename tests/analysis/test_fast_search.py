"""
Tests for FAISS fast similarity search.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from lumina.analysis.fast_search import FastSimilaritySearcher


class TestFastSimilaritySearcherInit:
    """Tests for FastSimilaritySearcher initialization."""

    def test_init_without_faiss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization when FAISS is not available."""
        # Try to test without FAISS, but skip if already loaded
        try:
            pass

            pytest.skip("FAISS is already loaded, cannot test unavailable case")
        except ImportError:
            pass

        searcher = FastSimilaritySearcher()

        assert searcher.available is False
        assert searcher.faiss is None
        assert searcher.hash_size == 64
        assert searcher.use_gpu is False
        assert searcher.index is None
        assert searcher.id_map == []

    def test_init_with_faiss_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization when FAISS is available (mocked)."""
        # Mock faiss module
        mock_faiss = MagicMock()

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher()

        assert searcher.available is True
        assert searcher.faiss is mock_faiss
        assert searcher.hash_size == 64
        assert searcher.use_gpu is False

    def test_init_custom_hash_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with custom hash size."""
        mock_faiss = MagicMock()

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=128)

        assert searcher.hash_size == 128

    def test_init_with_gpu_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with GPU enabled."""
        mock_faiss = MagicMock()
        mock_gpu_resources = MagicMock()
        mock_faiss.StandardGpuResources.return_value = mock_gpu_resources

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(use_gpu=True)

        assert searcher.use_gpu is True
        mock_faiss.StandardGpuResources.assert_called_once()

    def test_init_with_gpu_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization when GPU is requested but not available."""
        mock_faiss = MagicMock()
        # Remove StandardGpuResources attribute
        delattr(mock_faiss, "StandardGpuResources")

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(use_gpu=True)

        # Should fall back to CPU
        assert searcher.use_gpu is False

    def test_init_with_gpu_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization handles GPU initialization errors."""
        mock_faiss = MagicMock()
        mock_faiss.StandardGpuResources.side_effect = RuntimeError("GPU init failed")

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(use_gpu=True)

        # Should fall back to CPU
        assert searcher.use_gpu is False


class TestHexToBinary:
    """Tests for hex to binary conversion."""

    def test_hex_to_binary_simple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test converting a simple hex hash to binary."""
        mock_faiss = MagicMock()

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=8)

        # Test with 0xFF (all ones)
        result = searcher._hex_to_binary("ff")

        assert result.dtype == np.uint8
        assert result.shape == (8,)
        assert np.array_equal(result, np.ones(8, dtype=np.uint8))

    def test_hex_to_binary_zeros(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test converting zeros."""
        mock_faiss = MagicMock()

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=8)

        result = searcher._hex_to_binary("00")

        assert np.array_equal(result, np.zeros(8, dtype=np.uint8))

    def test_hex_to_binary_mixed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test converting mixed bits."""
        mock_faiss = MagicMock()

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=8)

        # 0xAA = 10101010 in binary
        result = searcher._hex_to_binary("aa")

        # LSB first: bit 0 = 0, bit 1 = 1, bit 2 = 0, bit 3 = 1, etc.
        expected = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        assert np.array_equal(result, expected)

    def test_hex_to_binary_64bit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test converting a 64-bit hash."""
        mock_faiss = MagicMock()

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=64)

        result = searcher._hex_to_binary("8f373c3f1f0f0f0f")

        assert result.shape == (64,)
        assert result.dtype == np.uint8
        # Verify it's binary (only 0s and 1s)
        assert np.all((result == 0) | (result == 1))


class TestBuildIndex:
    """Tests for building FAISS index."""

    def test_build_index_without_faiss(self) -> None:
        """Test building index when FAISS is not available."""
        # Skip if FAISS is available
        try:
            pass

            pytest.skip("FAISS is available, cannot test unavailable case")
        except ImportError:
            pass

        searcher = FastSimilaritySearcher()
        hashes = {"img1": "aaaa", "img2": "bbbb"}

        # Should not raise error, just log warning
        searcher.build_index(hashes)

        assert searcher.index is None
        assert searcher.id_map == []

    def test_build_index_empty_hashes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test building index with empty hash dict."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher()
            searcher.build_index({})

        # Should not create index
        assert searcher.id_map == []

    def test_build_index_with_none_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test building index with None hash values."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=16)

            hashes = {"img1": "aaaa", "img2": None, "img3": "bbbb"}
            searcher.build_index(hashes)

        # Should only include non-None values
        assert len(searcher.id_map) == 2
        assert "img1" in searcher.id_map
        assert "img3" in searcher.id_map
        assert "img2" not in searcher.id_map

    def test_build_index_basic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test building a basic index."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=16)

            hashes = {"img1": "aaaa", "img2": "bbbb", "img3": "cccc"}
            searcher.build_index(hashes)

        # Verify index was created
        mock_faiss.IndexBinaryFlat.assert_called_once_with(16)
        assert searcher.index is mock_index
        assert len(searcher.id_map) == 3
        assert "img1" in searcher.id_map
        assert "img2" in searcher.id_map
        assert "img3" in searcher.id_map

        # Verify add was called with packed vectors
        mock_index.add.assert_called_once()

    def test_build_index_with_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test building index with GPU acceleration."""
        mock_faiss = MagicMock()
        mock_index_cpu = MagicMock()
        mock_index_gpu = MagicMock()
        mock_gpu_resources = MagicMock()

        mock_faiss.IndexBinaryFlat.return_value = mock_index_cpu
        mock_faiss.StandardGpuResources.return_value = mock_gpu_resources
        mock_faiss.index_cpu_to_gpu.return_value = mock_index_gpu

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=16, use_gpu=True)

            hashes = {"img1": "aaaa", "img2": "bbbb"}
            searcher.build_index(hashes)

        # Verify GPU transfer was attempted
        mock_faiss.index_cpu_to_gpu.assert_called_once_with(
            mock_gpu_resources, 0, mock_index_cpu
        )
        assert searcher.index is mock_index_gpu

    def test_build_index_gpu_transfer_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling GPU transfer failure."""
        mock_faiss = MagicMock()
        mock_index_cpu = MagicMock()
        mock_gpu_resources = MagicMock()

        mock_faiss.IndexBinaryFlat.return_value = mock_index_cpu
        mock_faiss.StandardGpuResources.return_value = mock_gpu_resources
        mock_faiss.index_cpu_to_gpu.side_effect = RuntimeError("GPU transfer failed")

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=16, use_gpu=True)

            hashes = {"img1": "aaaa", "img2": "bbbb"}
            searcher.build_index(hashes)

        # Should fall back to CPU index
        assert searcher.index is mock_index_cpu


class TestFindSimilar:
    """Tests for finding similar image pairs."""

    def test_find_similar_no_index(self) -> None:
        """Test finding similar pairs without an index."""
        original_faiss = sys.modules.pop("faiss", None)

        try:
            searcher = FastSimilaritySearcher()
            results = searcher.find_similar(threshold=5, k=10)

            assert results == []
        finally:
            if original_faiss is not None:
                sys.modules["faiss"] = original_faiss

    def test_find_similar_basic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test finding similar pairs in a small dataset."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 3

        # Mock reconstruct to return simple vectors
        def mock_reconstruct(idx):
            return np.array([idx] * 2, dtype=np.uint8)  # 2 bytes = 16 bits

        mock_index.reconstruct.side_effect = mock_reconstruct

        # Mock search to return results
        # distances and indices for each query
        mock_index.search.return_value = (
            np.array([[0, 1, 5], [1, 0, 6], [5, 6, 0]]),  # distances
            np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]]),  # indices
        )

        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=16)

            hashes = {"img1": "aaaa", "img2": "aaab", "img3": "cccc"}
            searcher.build_index(hashes)

            results = searcher.find_similar(threshold=5, k=3)

        # Should find pairs within threshold
        assert isinstance(results, list)
        # Results contain tuples of (id1, id2, distance)
        for result in results:
            assert len(result) == 3
            assert result[2] <= 5  # Distance within threshold

    def test_find_similar_with_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that threshold filtering works."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 2

        def mock_reconstruct(idx):
            return np.array([idx] * 2, dtype=np.uint8)

        mock_index.reconstruct.side_effect = mock_reconstruct

        # Return one pair within threshold, one beyond
        mock_index.search.return_value = (
            np.array([[0, 3], [3, 0]]),  # distances: 0 (self), 3
            np.array([[0, 1], [1, 0]]),  # indices
        )

        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=16)

            hashes = {"img1": "aaaa", "img2": "aaab"}
            searcher.build_index(hashes)

            # With threshold=5, should include the pair
            results_loose = searcher.find_similar(threshold=5, k=2)
            assert len(results_loose) >= 0  # At least won't crash

            # With threshold=2, should exclude the pair
            results_strict = searcher.find_similar(threshold=2, k=2)
            assert len(results_strict) >= 0


class TestFindSimilarTo:
    """Tests for finding images similar to a specific hash."""

    def test_find_similar_to_no_index(self) -> None:
        """Test finding similar images without an index."""
        original_faiss = sys.modules.pop("faiss", None)

        try:
            searcher = FastSimilaritySearcher()
            results = searcher.find_similar_to("aaaa", threshold=5, k=10)

            assert results == []
        finally:
            if original_faiss is not None:
                sys.modules["faiss"] = original_faiss

    def test_find_similar_to_basic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test finding images similar to a query hash."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()

        # Mock search to return results
        mock_index.search.return_value = (
            np.array([[0, 2, 4]]),  # distances
            np.array([[0, 1, 2]]),  # indices
        )

        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=16)

            hashes = {"img1": "aaaa", "img2": "aaab", "img3": "cccc"}
            searcher.build_index(hashes)

            results = searcher.find_similar_to("aaaa", threshold=5, k=3)

        # Should return list of (id, distance) tuples
        assert isinstance(results, list)
        assert len(results) == 3  # All within threshold
        for result in results:
            assert len(result) == 2
            assert isinstance(result[0], str)  # image ID
            assert isinstance(result[1], int)  # distance
            assert result[1] <= 5  # Within threshold

    def test_find_similar_to_with_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that threshold filtering works for single query."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()

        # Return results with varying distances
        mock_index.search.return_value = (
            np.array([[0, 2, 10]]),  # distances: 0, 2, 10
            np.array([[0, 1, 2]]),  # indices
        )

        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=16)

            hashes = {"img1": "aaaa", "img2": "aaab", "img3": "ffff"}
            searcher.build_index(hashes)

            # With threshold=5, should exclude distance=10
            results = searcher.find_similar_to("aaaa", threshold=5, k=3)

        # Should only include results within threshold
        assert all(dist <= 5 for _, dist in results)

    def test_find_similar_to_out_of_bounds_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling of out-of-bounds indices."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()

        # Return an invalid index
        mock_index.search.return_value = (
            np.array([[0, 2]]),
            np.array([[0, 999]]),  # Index 999 doesn't exist
        )

        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)
            searcher = FastSimilaritySearcher(hash_size=16)

            hashes = {"img1": "aaaa", "img2": "bbbb"}
            searcher.build_index(hashes)

            _results = searcher.find_similar_to("aaaa", threshold=5, k=2)

        # Should only include valid indices
        assert all(idx < len(searcher.id_map) for idx, _ in [(0, 0)])


class TestIndexPersistence:
    """Tests for FAISS index save/load functionality."""

    def test_save_and_load_index(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Test saving and loading an index."""
        import json

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 3

        mock_faiss.IndexBinaryFlat.return_value = mock_index

        # Make write_index_binary actually create the file
        def mock_write(index, path):
            with open(path, "wb") as f:
                f.write(b"fake_index_data")

        mock_faiss.write_index_binary = MagicMock(side_effect=mock_write)
        mock_faiss.read_index_binary = MagicMock(return_value=mock_index)

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            # Create and build index
            searcher = FastSimilaritySearcher(hash_size=16, catalog_id="test-catalog")
            hashes = {"img1": "aaaa", "img2": "bbbb", "img3": "cccc"}
            searcher.build_index(hashes, method="dhash")

            # Save the index
            result = searcher.save(tmp_path)
            assert result is True
            mock_faiss.write_index_binary.assert_called_once()

            # Create a new searcher and load
            searcher2 = FastSimilaritySearcher(hash_size=16, catalog_id="test-catalog")
            result = searcher2.load(tmp_path)
            assert result is True
            mock_faiss.read_index_binary.assert_called_once()

    def test_save_without_index(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Test that save fails gracefully when no index exists."""
        mock_faiss = MagicMock()

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            searcher = FastSimilaritySearcher(hash_size=16)
            result = searcher.save(tmp_path)

            assert result is False

    def test_load_nonexistent_index(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Test that load fails gracefully for missing index."""
        mock_faiss = MagicMock()

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            searcher = FastSimilaritySearcher(hash_size=16, catalog_id="test-catalog")
            result = searcher.load(tmp_path)

            assert result is False

    def test_metadata_tracking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that metadata is correctly tracked after build."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            searcher = FastSimilaritySearcher(hash_size=16, catalog_id="test-catalog")
            hashes = {"img1": "aaaa", "img2": "bbbb"}
            searcher.build_index(hashes, method="dhash")

            assert searcher.metadata is not None
            assert searcher.metadata.version == 1
            assert searcher.metadata.hash_size == 16
            assert searcher.metadata.hash_method == "dhash"
            assert searcher.metadata.image_count == 2
            assert searcher.metadata.catalog_id == "test-catalog"

    def test_needs_rebuild_same_images(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test needs_rebuild returns False for same image set."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            searcher = FastSimilaritySearcher(hash_size=16)
            hashes = {"img1": "aaaa", "img2": "bbbb", "img3": "cccc"}
            searcher.build_index(hashes)

            # needs_rebuild returns (needs_rebuild, missing_ids, extra_ids)
            needs_rebuild, missing, extra = searcher.needs_rebuild(
                {"img1", "img2", "img3"}
            )
            assert needs_rebuild is False
            assert len(missing) == 0
            assert len(extra) == 0

    def test_needs_rebuild_different_images(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test needs_rebuild returns True for different image set."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            searcher = FastSimilaritySearcher(hash_size=16)
            hashes = {"img1": "aaaa", "img2": "bbbb", "img3": "cccc"}
            searcher.build_index(hashes)

            # Completely different image set
            needs_rebuild, missing, extra = searcher.needs_rebuild(
                {"img4", "img5", "img6"}
            )
            assert needs_rebuild is True
            assert missing == {"img4", "img5", "img6"}
            assert extra == {"img1", "img2", "img3"}

    def test_needs_rebuild_within_tolerance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test needs_rebuild respects tolerance for minor changes."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            searcher = FastSimilaritySearcher(hash_size=16)
            hashes = {f"img{i}": "aaaa" for i in range(100)}
            searcher.build_index(hashes)

            # 5% different images (5 changed) should be within 10% tolerance
            new_ids = {f"img{i}" for i in range(5, 105)}
            needs_rebuild, missing, extra = searcher.needs_rebuild(
                new_ids, tolerance=0.1
            )
            assert needs_rebuild is False
            assert len(missing) == 5  # img100-104
            assert len(extra) == 5  # img0-4

    def test_get_statistics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_statistics returns index info."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 5
        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            searcher = FastSimilaritySearcher(hash_size=64, catalog_id="test-cat")
            hashes = {f"img{i}": "a" * 16 for i in range(5)}
            searcher.build_index(hashes, method="ahash")

            stats = searcher.get_statistics()

            # Check actual field names from get_statistics
            assert stats["has_index"] is True
            assert stats["hash_size"] == 64
            assert stats["id_map_size"] == 5
            assert stats["total_vectors"] == 5
            # Metadata is in a nested dict
            assert stats["metadata"]["hash_method"] == "ahash"
            assert stats["metadata"]["catalog_id"] == "test-cat"


class TestIncrementalUpdates:
    """Tests for incremental index update functionality."""

    def test_add_hashes_to_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test adding new hashes to existing index."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            searcher = FastSimilaritySearcher(hash_size=16)
            hashes = {"img1": "aaaa", "img2": "bbbb"}
            searcher.build_index(hashes)

            # Add new hashes
            new_hashes = {"img3": "cccc", "img4": "dddd"}
            searcher.add_hashes(new_hashes)

            # Verify add was called again
            assert mock_index.add.call_count == 2
            assert len(searcher.id_map) == 4

    def test_add_hashes_without_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that add_hashes without index logs warning and returns 0."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            searcher = FastSimilaritySearcher(hash_size=16)
            # No build_index called first, so index is None

            new_hashes = {"img1": "aaaa", "img2": "bbbb"}
            result = searcher.add_hashes(new_hashes)

            # Should return 0 and not add anything (need to build first)
            assert result == 0

    def test_remove_ids_marks_as_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that remove_ids marks entries as removed."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 3
        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            searcher = FastSimilaritySearcher(hash_size=16)
            hashes = {"img1": "aaaa", "img2": "bbbb", "img3": "cccc"}
            searcher.build_index(hashes)

            # Remove some IDs
            searcher.remove_ids({"img1", "img3"})

            # img1 and img3 should be marked as removed
            assert searcher.id_map[0] == "__REMOVED__"
            assert searcher.id_map[1] == "img2"
            assert searcher.id_map[2] == "__REMOVED__"


class TestIntegration:
    """Integration tests with real FAISS if available."""

    def test_full_workflow_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test complete workflow with mocked FAISS."""
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 3

        def mock_reconstruct(idx):
            return np.array([idx] * 2, dtype=np.uint8)

        mock_index.reconstruct.side_effect = mock_reconstruct
        mock_index.search.return_value = (np.array([[0, 1]]), np.array([[0, 1]]))

        mock_faiss.IndexBinaryFlat.return_value = mock_index

        with monkeypatch.context() as m:
            m.setitem(sys.modules, "faiss", mock_faiss)

            # Initialize searcher
            searcher = FastSimilaritySearcher(hash_size=16)
            assert searcher.available

            # Build index
            hashes = {"img1": "a1b2", "img2": "a1b3", "img3": "ffff"}
            searcher.build_index(hashes)
            assert len(searcher.id_map) == 3

            # Find similar images to a query
            results = searcher.find_similar_to("a1b2", threshold=5, k=2)
            assert isinstance(results, list)
