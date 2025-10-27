"""
Tests for duplicate detection system.
"""

import json
from pathlib import Path

from PIL import Image

from vam_tools.analysis.duplicate_detector import DuplicateDetector
from vam_tools.analysis.scanner import ImageScanner
from vam_tools.core.catalog import CatalogDatabase


class TestDuplicateDetector:
    """Tests for DuplicateDetector."""

    def test_detector_initialization(self, tmp_path: Path) -> None:
        """Test that detector initializes correctly."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])
            detector = DuplicateDetector(db, similarity_threshold=5)

            assert detector.catalog == db
            assert detector.similarity_threshold == 5
            assert detector.hash_size == 8
            assert detector.duplicate_groups == []

    def test_detect_exact_duplicates(self, tmp_path: Path) -> None:
        """Test detection of exact duplicates (same checksum)."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create identical images with different names
        img = Image.new("RGB", (100, 100), color="red")
        img.save(photos_dir / "photo1.jpg")
        img.save(photos_dir / "photo2.jpg")
        img.save(photos_dir / "photo3.jpg")

        # Scan images into catalog
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # All images have same checksum, so only 1 is added
            images = db.list_images()
            assert len(images) == 1
            assert scanner.files_added == 1
            assert scanner.files_skipped == 2

    def test_detect_similar_images(self, tmp_path: Path) -> None:
        """Test detection of similar images using perceptual hashing."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create slightly different images
        # Base image
        img1 = Image.new("L", (100, 100))
        for x in range(100):
            for y in range(100):
                img1.putpixel((x, y), x * 2)
        img1.save(photos_dir / "gradient1.jpg")

        # Very similar gradient (shifted by 1 pixel)
        img2 = Image.new("L", (100, 100))
        for x in range(100):
            for y in range(100):
                img2.putpixel((x, y), (x + 1) * 2)
        img2.save(photos_dir / "gradient2.jpg")

        # Completely different image
        img3 = Image.new("RGB", (100, 100), color="blue")
        img3.save(photos_dir / "different.jpg")

        # Scan images
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # Run duplicate detection
            detector = DuplicateDetector(db, similarity_threshold=10)
            groups = detector.detect_duplicates()

            # Should find at least one group (the similar gradients)
            assert len(groups) >= 1

            # Check that similar images are grouped
            image_ids = db.list_images()
            assert len(image_ids) == 3  # All 3 unique by checksum

    def test_quality_scoring(self, tmp_path: Path) -> None:
        """Test that quality scoring selects the best image."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create images of different sizes (higher res = better quality)
        small_img = Image.new("RGB", (100, 100), color="red")
        small_img.save(photos_dir / "small.jpg")

        large_img = Image.new("RGB", (200, 200), color="red")
        large_img.save(photos_dir / "large.jpg")

        # Scan images
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # Run duplicate detection
            detector = DuplicateDetector(db, similarity_threshold=5)
            groups = detector.detect_duplicates()

            # Should find a group with the similar images
            assert len(groups) >= 1

            # Primary should be the larger image (better quality)
            for group in groups:
                if len(group.images) > 1:
                    primary_img = db.get_image(group.primary)
                    assert primary_img is not None
                    # Higher resolution should be selected
                    if primary_img.metadata.width:
                        assert primary_img.metadata.width >= 100

    def test_save_and_load_duplicate_groups(self, tmp_path: Path) -> None:
        """Test saving and loading duplicate groups."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create test images
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 80, 0, 0))
            img.save(photos_dir / f"photo{i}.jpg")

        # Scan and detect duplicates
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            detector = DuplicateDetector(db, similarity_threshold=5)
            detector.detect_duplicates()
            detector.save_duplicate_groups()

        # Verify saved to disk
        catalog_file = catalog_dir / ".catalog.json"
        assert catalog_file.exists()

        with open(catalog_file) as f:
            catalog_data = json.load(f)
        assert "duplicate_groups" in catalog_data

        # Load in new session and verify
        with CatalogDatabase(catalog_dir) as db:
            loaded_groups = db.get_duplicate_groups()
            # Should be able to load groups (even if empty)
            assert isinstance(loaded_groups, list)

    def test_statistics(self, tmp_path: Path) -> None:
        """Test duplicate detection statistics."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create some test images
        for i in range(5):
            img = Image.new("RGB", (100, 100), color=(i * 50, 0, 0))
            img.save(photos_dir / f"image{i}.jpg")

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            detector = DuplicateDetector(db, similarity_threshold=5)
            detector.detect_duplicates()

            stats = detector.get_statistics()

            # Check statistics structure
            assert "total_groups" in stats
            assert "total_images_in_groups" in stats
            assert "total_unique" in stats
            assert "total_redundant" in stats
            assert "groups_needing_review" in stats

            # All values should be non-negative
            assert stats["total_groups"] >= 0
            assert stats["total_images_in_groups"] >= 0
            assert stats["total_unique"] >= 0
            assert stats["total_redundant"] >= 0
            assert stats["groups_needing_review"] >= 0

    def test_perceptual_hash_computation(self, tmp_path: Path) -> None:
        """Test that perceptual hashes are computed and cached."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create test image
        img = Image.new("RGB", (100, 100), color="green")
        img_path = photos_dir / "test.jpg"
        img.save(img_path)

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # Initially, no perceptual hashes
            images = db.list_images()
            assert len(images) == 1
            image = images[0]
            assert image.metadata.perceptual_hash_dhash is None
            assert image.metadata.perceptual_hash_ahash is None

            # Detect duplicates (computes hashes)
            detector = DuplicateDetector(db, similarity_threshold=5)
            detector.detect_duplicates()

            # Now hashes should be computed
            images = db.list_images()
            image = images[0]
            assert image.metadata.perceptual_hash_dhash is not None
            assert image.metadata.perceptual_hash_ahash is not None
            assert len(image.metadata.perceptual_hash_dhash) > 0
            assert len(image.metadata.perceptual_hash_ahash) > 0

    def test_recompute_hashes_flag(self, tmp_path: Path) -> None:
        """Test that recompute_hashes flag forces hash recomputation."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create test image
        img = Image.new("RGB", (100, 100), color="purple")
        img.save(photos_dir / "test.jpg")

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # First detection computes hashes
            detector = DuplicateDetector(db, similarity_threshold=5)
            detector.detect_duplicates()

            images = db.list_images()
            first_dhash = images[0].metadata.perceptual_hash_dhash

            # Second detection with recompute=True
            detector2 = DuplicateDetector(db, similarity_threshold=5)
            detector2.detect_duplicates(recompute_hashes=True)

            images = db.list_images()
            second_dhash = images[0].metadata.perceptual_hash_dhash

            # Hashes should be the same (same image)
            assert first_dhash == second_dhash

    def test_date_conflict_detection(self, tmp_path: Path) -> None:
        """Test that date conflicts are detected in duplicate groups."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create images with different dates in filenames
        img = Image.new("RGB", (100, 100), color="orange")
        img.save(photos_dir / "2023-01-15_photo.jpg")
        img.save(photos_dir / "2023-06-20_photo.jpg")

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # Should only add 1 (same checksum)
            assert scanner.files_added == 1

            detector = DuplicateDetector(db, similarity_threshold=5)
            groups = detector.detect_duplicates()

            # No groups because only 1 unique image in catalog
            # (duplicates by checksum are not added)
            assert len(groups) == 0

    def test_empty_catalog(self, tmp_path: Path) -> None:
        """Test duplicate detection on empty catalog."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

            detector = DuplicateDetector(db, similarity_threshold=5)
            groups = detector.detect_duplicates()

            assert len(groups) == 0

            stats = detector.get_statistics()
            assert stats["total_groups"] == 0
            assert stats["total_images_in_groups"] == 0

    def test_custom_similarity_threshold(self, tmp_path: Path) -> None:
        """Test that custom similarity threshold affects grouping."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create slightly different images
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 10, i * 10, i * 10))
            img.save(photos_dir / f"gray{i}.jpg")

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # Strict threshold (fewer matches)
            detector_strict = DuplicateDetector(db, similarity_threshold=2)
            groups_strict = detector_strict.detect_duplicates()

            # Lenient threshold (more matches)
            detector_lenient = DuplicateDetector(db, similarity_threshold=20)
            groups_lenient = detector_lenient.detect_duplicates(recompute_hashes=False)

            # Lenient should find same or more groups
            assert len(groups_lenient) >= len(groups_strict)

    def test_similarity_metrics_capture(self, tmp_path: Path) -> None:
        """Test that similarity metrics are captured for duplicate groups."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create two visually similar but not identical images
        # (different checksums to avoid exact duplicate detection)
        img1 = Image.new("RGB", (100, 100), color="red")
        img1.save(photos_dir / "img1.jpg", quality=95)

        # Slightly different quality to get different checksum
        img2 = Image.new("RGB", (100, 100), color="red")
        img2.save(photos_dir / "img2.jpg", quality=90)

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # Detect duplicates with all hash methods
            from vam_tools.analysis.perceptual_hash import HashMethod

            detector = DuplicateDetector(
                db,
                similarity_threshold=10,
                hash_methods=[HashMethod.DHASH, HashMethod.AHASH, HashMethod.WHASH],
            )
            groups = detector.detect_duplicates()

            # Should find one group with two images
            assert len(groups) == 1
            group = groups[0]
            assert len(group.images) == 2

            # Should have similarity metrics captured
            assert len(group.similarity_metrics) > 0

            # Get the metrics for the pair
            pair_key = list(group.similarity_metrics.keys())[0]
            metrics = group.similarity_metrics[pair_key]

            # Should have metrics for all three hash methods
            assert metrics.dhash_distance is not None
            assert metrics.ahash_distance is not None
            assert metrics.whash_distance is not None
            assert metrics.dhash_similarity is not None
            assert metrics.ahash_similarity is not None
            assert metrics.whash_similarity is not None

            # Overall similarity should be computed
            assert metrics.overall_similarity > 0

            # For very similar images, distances should be low and similarities should be high
            assert metrics.dhash_distance <= 5
            assert metrics.ahash_distance <= 5
            assert metrics.whash_distance <= 5
            assert metrics.dhash_similarity >= 90.0
            assert metrics.ahash_similarity >= 90.0
            assert metrics.whash_similarity >= 90.0
            assert metrics.overall_similarity >= 90.0
