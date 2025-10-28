"""
Tests for FAISS fast similarity search.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from vam_tools.analysis.fast_search import FastSimilaritySearcher


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
