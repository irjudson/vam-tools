"""
Tests for duplicate_detection module.
"""

from pathlib import Path

from PIL import Image

from vam_tools.core.duplicate_detection import DuplicateDetector


class TestHashCalculation:
    """Tests for hash calculation functions."""

    def test_calculate_file_hash(self, sample_image: Path) -> None:
        """Test MD5 file hash calculation."""
        detector = DuplicateDetector()
        file_hash = detector.calculate_file_hash(sample_image)

        assert file_hash is not None
        assert len(file_hash) == 32  # MD5 is 32 hex characters
        assert all(c in "0123456789abcdef" for c in file_hash)

    def test_identical_files_same_hash(self, temp_dir: Path) -> None:
        """Test that identical files have the same hash."""
        # Create two identical files
        img = Image.new("RGB", (50, 50), color="blue")

        path1 = temp_dir / "image1.jpg"
        path2 = temp_dir / "image2.jpg"

        img.save(path1, "JPEG")
        img.save(path2, "JPEG")

        detector = DuplicateDetector()
        hash1 = detector.calculate_file_hash(path1)
        hash2 = detector.calculate_file_hash(path2)

        assert hash1 == hash2

    def test_calculate_dhash(self, sample_image: Path) -> None:
        """Test dHash calculation."""
        detector = DuplicateDetector(hash_size=8)
        dhash = detector.calculate_dhash(sample_image)

        assert dhash is not None
        assert len(dhash) == 64  # 8x8 = 64 bits
        assert all(c in "01" for c in dhash)

    def test_calculate_ahash(self, sample_image: Path) -> None:
        """Test aHash calculation."""
        detector = DuplicateDetector(hash_size=8)
        ahash = detector.calculate_ahash(sample_image)

        assert ahash is not None
        assert len(ahash) == 64  # 8x8 = 64 bits
        assert all(c in "01" for c in ahash)

    def test_similar_images_similar_hashes(self, temp_dir: Path) -> None:
        """Test that similar images have similar perceptual hashes."""
        # Create two similar images (same color, different sizes)
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (200, 200), color="red")

        path1 = temp_dir / "small.jpg"
        path2 = temp_dir / "large.jpg"

        img1.save(path1, "JPEG")
        img2.save(path2, "JPEG")

        detector = DuplicateDetector()
        dhash1 = detector.calculate_dhash(path1)
        dhash2 = detector.calculate_dhash(path2)

        # Calculate Hamming distance
        distance = detector.hamming_distance(dhash1, dhash2)

        # Should be very similar (low Hamming distance)
        # Solid colors should have nearly identical hashes
        assert distance < 10


class TestHammingDistance:
    """Tests for Hamming distance calculation."""

    def test_identical_hashes(self) -> None:
        """Test that identical hashes have distance 0."""
        hash1 = "1010101010101010"
        hash2 = "1010101010101010"

        distance = DuplicateDetector.hamming_distance(hash1, hash2)
        assert distance == 0

    def test_completely_different_hashes(self) -> None:
        """Test distance for completely different hashes."""
        hash1 = "1111111111111111"
        hash2 = "0000000000000000"

        distance = DuplicateDetector.hamming_distance(hash1, hash2)
        assert distance == 16

    def test_partially_different_hashes(self) -> None:
        """Test distance for partially different hashes."""
        hash1 = "11110000"
        hash2 = "11111111"

        distance = DuplicateDetector.hamming_distance(hash1, hash2)
        assert distance == 4

    def test_none_hash_returns_large_distance(self) -> None:
        """Test that None hashes return large distance."""
        distance = DuplicateDetector.hamming_distance(None, "1010")
        assert distance == 999999

        distance = DuplicateDetector.hamming_distance("1010", None)
        assert distance == 999999

    def test_different_length_hashes(self) -> None:
        """Test that hashes of different lengths return large distance."""
        distance = DuplicateDetector.hamming_distance("1010", "101010")
        assert distance == 999999


class TestExactDuplicates:
    """Tests for exact duplicate detection."""

    def test_find_exact_duplicates(
        self, duplicate_images: dict[str, list[Path]]
    ) -> None:
        """Test finding exact file duplicates."""
        exact_duplicates = duplicate_images["exact"]

        detector = DuplicateDetector()
        detector.process_images(exact_duplicates)

        groups = detector.find_exact_duplicates()

        assert len(groups) == 1
        assert len(groups[0].images) == 2
        assert groups[0].similarity_type == "exact"
        assert groups[0].hash_distance == 0

    def test_no_exact_duplicates(self, sample_images: list[Path]) -> None:
        """Test that different images don't show as exact duplicates."""
        detector = DuplicateDetector()
        detector.process_images(sample_images)

        groups = detector.find_exact_duplicates()

        # sample_images are different colors, so no exact duplicates
        assert len(groups) == 0


class TestPerceptualDuplicates:
    """Tests for perceptual duplicate detection."""

    def test_find_perceptual_duplicates(
        self, duplicate_images: dict[str, list[Path]]
    ) -> None:
        """Test finding perceptually similar images."""
        similar_images = duplicate_images["similar"]

        detector = DuplicateDetector()
        detector.process_images(similar_images)

        # Use a reasonable threshold
        groups = detector.find_perceptual_duplicates(threshold=15)

        # Should find at least one group of similar images
        assert len(groups) >= 1

    def test_strict_threshold(self, temp_dir: Path) -> None:
        """Test that strict threshold finds fewer matches."""
        # Create slightly different images
        img1 = Image.new("RGB", (100, 100), color="blue")
        img2 = Image.new("RGB", (100, 100), color="lightblue")

        path1 = temp_dir / "blue1.jpg"
        path2 = temp_dir / "blue2.jpg"

        img1.save(path1, "JPEG")
        img2.save(path2, "JPEG")

        detector = DuplicateDetector()
        detector.process_images([path1, path2])

        # Very strict threshold should not match these
        strict_groups = detector.find_perceptual_duplicates(threshold=0)

        # Loose threshold might match these
        loose_groups = detector.find_perceptual_duplicates(threshold=30)

        # Strict should have fewer (or same) matches than loose
        assert len(strict_groups) <= len(loose_groups)


class TestBatchProcessing:
    """Tests for batch image processing."""

    def test_process_multiple_images(self, sample_images: list[Path]) -> None:
        """Test processing multiple images."""
        detector = DuplicateDetector()
        detector.process_images(sample_images)

        assert len(detector.image_hashes) == len(sample_images)

        for image_path in sample_images:
            assert image_path in detector.image_hashes
            hashes = detector.image_hashes[image_path]
            assert hashes.file_hash is not None
            assert hashes.dhash is not None
            assert hashes.ahash is not None

    def test_find_all_duplicates(self, duplicate_images: dict[str, list[Path]]) -> None:
        """Test finding both exact and perceptual duplicates."""
        all_images = duplicate_images["exact"] + duplicate_images["similar"]

        detector = DuplicateDetector()
        groups = detector.find_all_duplicates(all_images, threshold=15)

        # Should find at least exact duplicates
        assert len(groups) >= 1

        # Check that exact duplicates are marked correctly
        exact_groups = [g for g in groups if g.similarity_type == "exact"]
        assert len(exact_groups) >= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_image(self, sample_image: Path) -> None:
        """Test processing a single image."""
        detector = DuplicateDetector()
        groups = detector.find_all_duplicates([sample_image])

        # Single image cannot have duplicates
        assert len(groups) == 0

    def test_empty_list(self) -> None:
        """Test processing empty list."""
        detector = DuplicateDetector()
        groups = detector.find_all_duplicates([])

        assert len(groups) == 0

    def test_corrupted_image_handling(self, temp_dir: Path) -> None:
        """Test that corrupted images are handled gracefully."""
        # Create a "corrupted" image (just text file with .jpg extension)
        corrupted = temp_dir / "corrupted.jpg"
        corrupted.write_text("This is not an image")

        detector = DuplicateDetector()
        file_hash = detector.calculate_file_hash(corrupted)
        dhash = detector.calculate_dhash(corrupted)
        ahash = detector.calculate_ahash(corrupted)

        # File hash should work (it's just bytes)
        assert file_hash is not None

        # Perceptual hashes should return None
        assert dhash is None
        assert ahash is None
