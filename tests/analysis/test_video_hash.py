"""
Tests for video perceptual hashing module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from lumina.analysis.video_hash import (
    are_videos_similar,
    compute_video_hash,
    compute_video_hashes,
    hamming_distance,
)


class TestComputeVideoHash:
    """Tests for compute_video_hash function."""

    def test_compute_video_hash_success(self) -> None:
        """Test successful video hash computation."""
        # Create a mock videohash module
        mock_videohash_module = MagicMock()

        with patch.dict("sys.modules", {"videohash": mock_videohash_module}):
            # Mock VideoHash to return a hash
            mock_vh_instance = MagicMock()
            mock_vh_instance.hash_hex = "1a2b3c4d5e6f7890"
            mock_videohash_module.VideoHash.return_value = mock_vh_instance

            video_path = Path("/fake/video.mp4")
            result = compute_video_hash(video_path)

            assert result == "1a2b3c4d5e6f7890"
            mock_videohash_module.VideoHash.assert_called_once_with(
                path=str(video_path)
            )

    def test_compute_video_hash_import_error(self) -> None:
        """Test handling of missing videohash library."""
        # Mock the import to fail
        import sys

        videohash_backup = sys.modules.get("videohash")
        if "videohash" in sys.modules:
            del sys.modules["videohash"]

        try:
            # Force reimport
            import importlib

            import lumina.analysis.video_hash as vh_module

            importlib.reload(vh_module)

            # Now patch the import to raise ImportError
            with patch.dict("sys.modules", {"videohash": None}):
                video_path = Path("/fake/video.mp4")
                result = compute_video_hash(video_path)

                # Should handle gracefully and return None
                assert result is None
        finally:
            # Restore videohash
            if videohash_backup:
                sys.modules["videohash"] = videohash_backup

    def test_compute_video_hash_computation_error(self) -> None:
        """Test handling of video hash computation errors."""
        # Create a mock videohash module
        mock_videohash_module = MagicMock()

        with patch.dict("sys.modules", {"videohash": mock_videohash_module}):
            mock_videohash_module.VideoHash.side_effect = Exception("FFmpeg not found")

            video_path = Path("/fake/video.mp4")
            result = compute_video_hash(video_path)

            assert result is None

    def test_compute_video_hash_3gp_format(self) -> None:
        """Test video hash computation for .3gp files."""
        # Create a mock videohash module
        mock_videohash_module = MagicMock()

        with patch.dict("sys.modules", {"videohash": mock_videohash_module}):
            mock_vh_instance = MagicMock()
            mock_vh_instance.hash_hex = "abcdef0123456789"
            mock_videohash_module.VideoHash.return_value = mock_vh_instance

            video_path = Path("/fake/video.3gp")
            result = compute_video_hash(video_path)

            assert result == "abcdef0123456789"
            mock_videohash_module.VideoHash.assert_called_once_with(
                path=str(video_path)
            )


class TestComputeVideoHashes:
    """Tests for compute_video_hashes function."""

    @patch("lumina.analysis.video_hash.compute_video_hash")
    def test_compute_video_hashes_success(self, mock_compute: MagicMock) -> None:
        """Test successful video hashes computation."""
        mock_compute.return_value = "1a2b3c4d5e6f7890"

        video_path = Path("/fake/video.mp4")
        result = compute_video_hashes(video_path)

        assert result == {
            "dhash": "1a2b3c4d5e6f7890",
            "ahash": None,
            "whash": None,
        }
        mock_compute.assert_called_once_with(video_path)

    @patch("lumina.analysis.video_hash.compute_video_hash")
    def test_compute_video_hashes_failure(self, mock_compute: MagicMock) -> None:
        """Test video hashes computation when hash fails."""
        mock_compute.return_value = None

        video_path = Path("/fake/video.mp4")
        result = compute_video_hashes(video_path)

        assert result == {
            "dhash": None,
            "ahash": None,
            "whash": None,
        }

    @patch("lumina.analysis.video_hash.compute_video_hash")
    def test_compute_video_hashes_dict_format(self, mock_compute: MagicMock) -> None:
        """Test that returned dict is compatible with image hash format."""
        mock_compute.return_value = "fedcba9876543210"

        video_path = Path("/fake/video.mkv")
        result = compute_video_hashes(video_path)

        # Verify dict has all expected keys
        assert "dhash" in result
        assert "ahash" in result
        assert "whash" in result

        # Verify only dhash has a value (videos use single hash)
        assert result["dhash"] is not None
        assert result["ahash"] is None
        assert result["whash"] is None


class TestHammingDistance:
    """Tests for hamming_distance function."""

    def test_hamming_distance_identical_hashes(self) -> None:
        """Test Hamming distance for identical hashes."""
        hash1 = "1a2b3c4d5e6f7890"
        hash2 = "1a2b3c4d5e6f7890"

        distance = hamming_distance(hash1, hash2)

        assert distance == 0

    def test_hamming_distance_different_hashes(self) -> None:
        """Test Hamming distance for different hashes."""
        # These hashes differ in the last digit: 0 vs 1
        hash1 = "1a2b3c4d5e6f7890"
        hash2 = "1a2b3c4d5e6f7891"

        distance = hamming_distance(hash1, hash2)

        # 0 = 0000, 1 = 0001, XOR = 0001 = 1 bit different
        assert distance == 1

    def test_hamming_distance_all_bits_different(self) -> None:
        """Test Hamming distance when all bits differ."""
        hash1 = "0000000000000000"
        hash2 = "ffffffffffffffff"

        distance = hamming_distance(hash1, hash2)

        # All 64 bits are different
        assert distance == 64

    def test_hamming_distance_none_hash1(self) -> None:
        """Test Hamming distance with None as first hash."""
        distance = hamming_distance(None, "1a2b3c4d5e6f7890")  # type: ignore

        assert distance is None

    def test_hamming_distance_none_hash2(self) -> None:
        """Test Hamming distance with None as second hash."""
        distance = hamming_distance("1a2b3c4d5e6f7890", None)  # type: ignore

        assert distance is None

    def test_hamming_distance_both_none(self) -> None:
        """Test Hamming distance with both hashes None."""
        distance = hamming_distance(None, None)  # type: ignore

        assert distance is None

    def test_hamming_distance_length_mismatch(self) -> None:
        """Test Hamming distance with mismatched hash lengths."""
        hash1 = "1a2b3c4d"  # 8 hex digits
        hash2 = "1a2b3c4d5e6f7890"  # 16 hex digits

        distance = hamming_distance(hash1, hash2)

        assert distance is None

    def test_hamming_distance_invalid_hex(self) -> None:
        """Test Hamming distance with invalid hex string."""
        hash1 = "1a2b3c4d5e6f7890"
        hash2 = "not_a_hex_string"

        distance = hamming_distance(hash1, hash2)

        assert distance is None

    def test_hamming_distance_multiple_bits_different(self) -> None:
        """Test Hamming distance with known bit differences."""
        # f = 1111, 0 = 0000
        hash1 = "f000000000000000"
        hash2 = "0000000000000000"

        distance = hamming_distance(hash1, hash2)

        # 4 bits different (1111 vs 0000)
        assert distance == 4


class TestAreVideosSimilar:
    """Tests for are_videos_similar function."""

    @patch("lumina.analysis.video_hash.hamming_distance")
    def test_are_videos_similar_identical(self, mock_distance: MagicMock) -> None:
        """Test similarity check for identical videos."""
        mock_distance.return_value = 0

        hash1 = "1a2b3c4d5e6f7890"
        hash2 = "1a2b3c4d5e6f7890"

        is_similar, distance = are_videos_similar(hash1, hash2)

        assert is_similar is True
        assert distance == 0

    @patch("lumina.analysis.video_hash.hamming_distance")
    def test_are_videos_similar_within_threshold(
        self, mock_distance: MagicMock
    ) -> None:
        """Test similarity check for videos within threshold."""
        mock_distance.return_value = 5

        hash1 = "1a2b3c4d5e6f7890"
        hash2 = "1a2b3c4d5e6f7891"

        is_similar, distance = are_videos_similar(hash1, hash2, threshold=10)

        assert is_similar is True
        assert distance == 5

    @patch("lumina.analysis.video_hash.hamming_distance")
    def test_are_videos_similar_exceeds_threshold(
        self, mock_distance: MagicMock
    ) -> None:
        """Test similarity check for videos exceeding threshold."""
        mock_distance.return_value = 15

        hash1 = "1a2b3c4d5e6f7890"
        hash2 = "fedcba9876543210"

        is_similar, distance = are_videos_similar(hash1, hash2, threshold=10)

        assert is_similar is False
        assert distance == 15

    @patch("lumina.analysis.video_hash.hamming_distance")
    def test_are_videos_similar_at_threshold_boundary(
        self, mock_distance: MagicMock
    ) -> None:
        """Test similarity check exactly at threshold boundary."""
        mock_distance.return_value = 10

        hash1 = "1a2b3c4d5e6f7890"
        hash2 = "1a2b3c4d5e6f7890"

        is_similar, distance = are_videos_similar(hash1, hash2, threshold=10)

        assert is_similar is True  # <= threshold
        assert distance == 10

    @patch("lumina.analysis.video_hash.hamming_distance")
    def test_are_videos_similar_custom_threshold(
        self, mock_distance: MagicMock
    ) -> None:
        """Test similarity check with custom threshold."""
        mock_distance.return_value = 3

        hash1 = "1a2b3c4d5e6f7890"
        hash2 = "1a2b3c4d5e6f7893"

        # With default threshold (10), would be similar
        is_similar_default, _ = are_videos_similar(hash1, hash2)
        assert is_similar_default is True

        # With strict threshold (2), would not be similar
        is_similar_strict, distance = are_videos_similar(hash1, hash2, threshold=2)
        assert is_similar_strict is False
        assert distance == 3

    @patch("lumina.analysis.video_hash.hamming_distance")
    def test_are_videos_similar_distance_none(self, mock_distance: MagicMock) -> None:
        """Test similarity check when distance calculation fails."""
        mock_distance.return_value = None

        hash1 = "1a2b3c4d5e6f7890"
        hash2 = "invalid"

        is_similar, distance = are_videos_similar(hash1, hash2)

        assert is_similar is False
        assert distance is None


class TestVideoHashIntegration:
    """Integration tests for video hash module."""

    def test_full_workflow_similar_videos(self) -> None:
        """Test complete workflow for detecting similar videos."""
        # Create a mock videohash module
        mock_videohash_module = MagicMock()

        with patch.dict("sys.modules", {"videohash": mock_videohash_module}):
            # Mock VideoHash for two similar videos
            mock_vh1 = MagicMock()
            mock_vh1.hash_hex = "1a2b3c4d5e6f7890"

            mock_vh2 = MagicMock()
            mock_vh2.hash_hex = "1a2b3c4d5e6f7891"  # Differs by 1 bit

            mock_videohash_module.VideoHash.side_effect = [mock_vh1, mock_vh2]

            # Compute hashes
            video1_path = Path("/fake/video1.mp4")
            video2_path = Path("/fake/video2.mp4")

            hash1 = compute_video_hash(video1_path)
            hash2 = compute_video_hash(video2_path)

            # Check similarity
            is_similar, distance = are_videos_similar(hash1, hash2, threshold=5)

            assert hash1 is not None
            assert hash2 is not None
            assert is_similar is True
            assert distance == 1

    def test_full_workflow_different_videos(self) -> None:
        """Test complete workflow for detecting different videos."""
        # Create a mock videohash module
        mock_videohash_module = MagicMock()

        with patch.dict("sys.modules", {"videohash": mock_videohash_module}):
            # Mock VideoHash for two very different videos
            mock_vh1 = MagicMock()
            mock_vh1.hash_hex = "0000000000000000"

            mock_vh2 = MagicMock()
            mock_vh2.hash_hex = "ffffffffffffffff"

            mock_videohash_module.VideoHash.side_effect = [mock_vh1, mock_vh2]

            # Compute hashes
            video1_path = Path("/fake/video1.mp4")
            video2_path = Path("/fake/video2.mp4")

            hash1 = compute_video_hash(video1_path)
            hash2 = compute_video_hash(video2_path)

            # Check similarity
            is_similar, distance = are_videos_similar(hash1, hash2, threshold=10)

            assert hash1 is not None
            assert hash2 is not None
            assert is_similar is False
            assert distance == 64  # All bits different

    def test_various_video_formats(self) -> None:
        """Test hash computation for various video formats."""
        # Create a mock videohash module
        mock_videohash_module = MagicMock()

        with patch.dict("sys.modules", {"videohash": mock_videohash_module}):
            mock_vh = MagicMock()
            mock_vh.hash_hex = "1a2b3c4d5e6f7890"
            mock_videohash_module.VideoHash.return_value = mock_vh

            video_formats = [
                ".mp4",
                ".mov",
                ".avi",
                ".mkv",
                ".3gp",
                ".webm",
                ".flv",
                ".m4v",
            ]

            for fmt in video_formats:
                video_path = Path(f"/fake/video{fmt}")
                result = compute_video_hash(video_path)

                assert result is not None
                assert result == "1a2b3c4d5e6f7890"
