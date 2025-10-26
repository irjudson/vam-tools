"""
Tests for perceptual hashing module.
"""

from pathlib import Path

import pytest
from PIL import Image

from vam_tools.analysis.perceptual_hash import (
    ahash,
    are_similar,
    combined_hash,
    dhash,
    hamming_distance,
    similarity_score,
)


class TestDHash:
    """Tests for dHash (difference hash) function."""

    def test_dhash_basic(self, tmp_path: Path) -> None:
        """Test basic dHash computation."""
        # Create a simple test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)

        # Compute hash
        hash_val = dhash(img_path)

        assert hash_val is not None
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16  # 64 bits = 16 hex digits

    def test_dhash_identical_images(self, tmp_path: Path) -> None:
        """Test that identical images produce same hash."""
        # Create two identical images
        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"

        img = Image.new("RGB", (100, 100), color="blue")
        img.save(img1_path)
        img.save(img2_path)

        hash1 = dhash(img1_path)
        hash2 = dhash(img2_path)

        assert hash1 == hash2

    def test_dhash_different_images(self, tmp_path: Path) -> None:
        """Test that different images produce different hashes."""
        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"

        # Create images with gradients (more interesting than solid colors)
        img1 = Image.new("L", (100, 100))
        img2 = Image.new("L", (100, 100))

        # Horizontal gradient
        for x in range(100):
            for y in range(100):
                img1.putpixel((x, y), x * 2)

        # Vertical gradient
        for x in range(100):
            for y in range(100):
                img2.putpixel((x, y), y * 2)

        img1.save(img1_path)
        img2.save(img2_path)

        hash1 = dhash(img1_path)
        hash2 = dhash(img2_path)

        assert hash1 != hash2

    def test_dhash_different_sizes_same_content(self, tmp_path: Path) -> None:
        """Test that images with same content but different sizes produce similar hashes."""
        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"

        # Create gradient images of different sizes
        img1 = Image.new("L", (100, 100))
        img2 = Image.new("L", (200, 200))

        # Add gradient
        for x in range(100):
            for y in range(100):
                img1.putpixel((x, y), (x + y) % 256)

        for x in range(200):
            for y in range(200):
                img2.putpixel((x, y), (x // 2 + y // 2) % 256)

        img1.save(img1_path)
        img2.save(img2_path)

        hash1 = dhash(img1_path)
        hash2 = dhash(img2_path)

        # Should be similar (low Hamming distance)
        distance = hamming_distance(hash1, hash2)
        assert distance < 10  # Allow some variation

    def test_dhash_nonexistent_file(self, tmp_path: Path) -> None:
        """Test that non-existent file returns None."""
        fake_path = tmp_path / "nonexistent.jpg"
        hash_val = dhash(fake_path)
        assert hash_val is None

    def test_dhash_custom_hash_size(self, tmp_path: Path) -> None:
        """Test dHash with custom hash size."""
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100), color="green").save(img_path)

        # Test different hash sizes
        hash4 = dhash(img_path, hash_size=4)
        hash16 = dhash(img_path, hash_size=16)

        assert len(hash4) == 4  # 16 bits = 4 hex digits
        assert len(hash16) == 64  # 256 bits = 64 hex digits


class TestAHash:
    """Tests for aHash (average hash) function."""

    def test_ahash_basic(self, tmp_path: Path) -> None:
        """Test basic aHash computation."""
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100), color="red").save(img_path)

        hash_val = ahash(img_path)

        assert hash_val is not None
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16  # 64 bits = 16 hex digits

    def test_ahash_identical_images(self, tmp_path: Path) -> None:
        """Test that identical images produce same hash."""
        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"

        img = Image.new("RGB", (100, 100), color="yellow")
        img.save(img1_path)
        img.save(img2_path)

        hash1 = ahash(img1_path)
        hash2 = ahash(img2_path)

        assert hash1 == hash2

    def test_ahash_different_images(self, tmp_path: Path) -> None:
        """Test that different images produce different hashes."""
        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"

        # Create images with different patterns
        img1 = Image.new("L", (100, 100))
        img2 = Image.new("L", (100, 100))

        # Checkerboard pattern
        for x in range(100):
            for y in range(100):
                img1.putpixel((x, y), 255 if (x + y) % 2 == 0 else 0)

        # Diagonal stripes
        for x in range(100):
            for y in range(100):
                img2.putpixel((x, y), 255 if (x - y) % 10 < 5 else 0)

        img1.save(img1_path)
        img2.save(img2_path)

        hash1 = ahash(img1_path)
        hash2 = ahash(img2_path)

        assert hash1 != hash2

    def test_ahash_nonexistent_file(self, tmp_path: Path) -> None:
        """Test that non-existent file returns None."""
        fake_path = tmp_path / "nonexistent.jpg"
        hash_val = ahash(fake_path)
        assert hash_val is None


class TestCombinedHash:
    """Tests for combined_hash function."""

    def test_combined_hash_returns_both(self, tmp_path: Path) -> None:
        """Test that combined_hash returns both hashes."""
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100), color="purple").save(img_path)

        result = combined_hash(img_path)

        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

        dhash_val, ahash_val = result
        assert isinstance(dhash_val, str)
        assert isinstance(ahash_val, str)
        assert len(dhash_val) == 16
        assert len(ahash_val) == 16

    def test_combined_hash_nonexistent_file(self, tmp_path: Path) -> None:
        """Test that non-existent file returns None."""
        fake_path = tmp_path / "nonexistent.jpg"
        result = combined_hash(fake_path)
        assert result is None


class TestHammingDistance:
    """Tests for hamming_distance function."""

    def test_hamming_distance_identical(self) -> None:
        """Test Hamming distance of identical hashes is 0."""
        hash1 = "a1b2c3d4e5f6a7b8"
        hash2 = "a1b2c3d4e5f6a7b8"

        distance = hamming_distance(hash1, hash2)
        assert distance == 0

    def test_hamming_distance_one_bit_different(self) -> None:
        """Test Hamming distance when one bit differs."""
        hash1 = "0000000000000000"  # All zeros
        hash2 = "0000000000000001"  # Last bit is 1

        distance = hamming_distance(hash1, hash2)
        assert distance == 1

    def test_hamming_distance_all_bits_different(self) -> None:
        """Test Hamming distance when all bits differ."""
        hash1 = "0000000000000000"  # All zeros
        hash2 = "ffffffffffffffff"  # All ones

        distance = hamming_distance(hash1, hash2)
        assert distance == 64  # All 64 bits are different

    def test_hamming_distance_different_lengths(self) -> None:
        """Test that different length hashes raise ValueError."""
        hash1 = "a1b2c3d4"
        hash2 = "a1b2c3d4e5f6a7b8"

        with pytest.raises(ValueError):
            hamming_distance(hash1, hash2)

    def test_hamming_distance_symmetric(self) -> None:
        """Test that Hamming distance is symmetric."""
        hash1 = "a1b2c3d4e5f6a7b8"
        hash2 = "b2c3d4e5f6a7b8c9"

        dist1 = hamming_distance(hash1, hash2)
        dist2 = hamming_distance(hash2, hash1)

        assert dist1 == dist2


class TestAreSimilar:
    """Tests for are_similar function."""

    def test_are_similar_identical(self) -> None:
        """Test that identical hashes are similar."""
        hash1 = "a1b2c3d4e5f6a7b8"
        hash2 = "a1b2c3d4e5f6a7b8"

        assert are_similar(hash1, hash2, threshold=5)

    def test_are_similar_within_threshold(self) -> None:
        """Test that hashes within threshold are similar."""
        hash1 = "0000000000000000"
        hash2 = "0000000000000003"  # 2 bits different

        assert are_similar(hash1, hash2, threshold=5)

    def test_are_similar_outside_threshold(self) -> None:
        """Test that hashes outside threshold are not similar."""
        hash1 = "0000000000000000"
        hash2 = "00000000000000ff"  # 8 bits different

        assert not are_similar(hash1, hash2, threshold=5)

    def test_are_similar_custom_threshold(self) -> None:
        """Test with custom threshold."""
        hash1 = "0000000000000000"
        hash2 = "000000000000000f"  # 4 bits different

        assert not are_similar(hash1, hash2, threshold=3)
        assert are_similar(hash1, hash2, threshold=10)


class TestSimilarityScore:
    """Tests for similarity_score function."""

    def test_similarity_score_identical(self) -> None:
        """Test that identical hashes have 100% similarity."""
        hash1 = "a1b2c3d4e5f6a7b8"
        hash2 = "a1b2c3d4e5f6a7b8"

        score = similarity_score(hash1, hash2)
        assert score == 100.0

    def test_similarity_score_completely_different(self) -> None:
        """Test that completely different hashes have 0% similarity."""
        hash1 = "0000000000000000"
        hash2 = "ffffffffffffffff"

        score = similarity_score(hash1, hash2)
        assert score == 0.0

    def test_similarity_score_half_different(self) -> None:
        """Test similarity when half the bits are different."""
        hash1 = "0000000000000000"
        hash2 = "00000000ffffffff"  # Half the bits different

        score = similarity_score(hash1, hash2)
        assert 45 < score < 55  # Should be around 50%

    def test_similarity_score_range(self) -> None:
        """Test that similarity score is always between 0 and 100."""
        hash1 = "a1b2c3d4e5f6a7b8"
        hash2 = "b2c3d4e5f6a7b8c9"

        score = similarity_score(hash1, hash2)
        assert 0 <= score <= 100
