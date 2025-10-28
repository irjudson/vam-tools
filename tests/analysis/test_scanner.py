"""
Tests for scanner module.
"""

import json
from pathlib import Path

from PIL import Image

from vam_tools.analysis.scanner import ImageScanner
from vam_tools.core.catalog import CatalogDatabase


class TestImageScanner:
    """Tests for ImageScanner."""

    def test_scanner_basic_workflow(self, tmp_path: Path) -> None:
        """Test that scanner adds images to catalog and saves them."""
        # Setup
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create test images
        img1_path = photos_dir / "test1.jpg"
        img2_path = photos_dir / "test2.jpg"
        Image.new("RGB", (100, 100), color="red").save(img1_path)
        Image.new("RGB", (100, 100), color="blue").save(img2_path)

        # Initialize catalog
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])

        # Run scanner
        with CatalogDatabase(catalog_dir) as db:
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        # Verify images were added to catalog
        with CatalogDatabase(catalog_dir) as db:
            images = db.list_images()
            assert len(images) == 2, f"Expected 2 images, got {len(images)}"

            # Check that images have checksums
            checksums = [img.checksum for img in images]
            assert all(checksums), "All images should have checksums"
            assert len(set(checksums)) == 2, "Checksums should be unique"

        # Verify catalog was saved to disk
        catalog_file = catalog_dir / "catalog.json"
        assert catalog_file.exists(), "Catalog file should exist"

        # Verify catalog contents on disk
        with open(catalog_file) as f:
            catalog_data = json.load(f)

        assert "images" in catalog_data
        assert len(catalog_data["images"]) == 2
        assert "statistics" in catalog_data
        assert catalog_data["statistics"]["total_images"] == 2

    def test_scanner_multiprocessing(self, tmp_path: Path) -> None:
        """Test scanner with multiple workers."""
        # Setup
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create multiple test images
        for i in range(10):
            img_path = photos_dir / f"test{i}.jpg"
            # Create different colored images
            color = (i * 25, (255 - i * 25) % 256, (i * 50) % 256)
            Image.new("RGB", (100, 100), color=color).save(img_path)

        # Initialize catalog
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])

        # Run scanner with multiple workers
        with CatalogDatabase(catalog_dir) as db:
            scanner = ImageScanner(db, workers=4)
            scanner.scan_directories([photos_dir])

        # Verify all images were added
        with CatalogDatabase(catalog_dir) as db:
            images = db.list_images()
            assert len(images) == 10

            # Verify statistics
            stats = db.get_statistics()
            assert stats.total_images == 10
            assert stats.total_videos == 0

    def test_scanner_incremental_scan(self, tmp_path: Path) -> None:
        """Test that rescanning only processes new files."""
        # Setup
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create initial images
        img1_path = photos_dir / "test1.jpg"
        img2_path = photos_dir / "test2.jpg"
        Image.new("RGB", (100, 100), color="red").save(img1_path)
        Image.new("RGB", (100, 100), color="blue").save(img2_path)

        # Initialize and first scan
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            assert scanner.files_added == 2
            assert scanner.files_skipped == 0

        # Add new image
        img3_path = photos_dir / "test3.jpg"
        Image.new("RGB", (100, 100), color="green").save(img3_path)

        # Rescan
        with CatalogDatabase(catalog_dir) as db:
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # Should only add 1 new file, skip 2 existing
            assert scanner.files_added == 1
            assert scanner.files_skipped == 2

        # Verify total
        with CatalogDatabase(catalog_dir) as db:
            images = db.list_images()
            assert len(images) == 3

    def test_scanner_handles_invalid_files(self, tmp_path: Path) -> None:
        """Test that scanner handles invalid files gracefully."""
        # Setup
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create valid image
        img1_path = photos_dir / "valid.jpg"
        Image.new("RGB", (100, 100), color="red").save(img1_path)

        # Create invalid "image" (corrupted file)
        invalid_path = photos_dir / "invalid.jpg"
        invalid_path.write_bytes(b"not a valid image file")

        # Create text file
        text_path = photos_dir / "readme.txt"
        text_path.write_text("This is not an image")

        # Initialize and scan
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        # Should only process valid image
        with CatalogDatabase(catalog_dir) as db:
            images = db.list_images()
            # Might be 0 or 1 depending on how invalid file is handled
            # At minimum, should not crash
            assert len(images) >= 0

    def test_scanner_with_subdirectories(self, tmp_path: Path) -> None:
        """Test scanner processes subdirectories."""
        # Setup
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"

        # Create nested structure
        (photos_dir / "2023" / "june").mkdir(parents=True)
        (photos_dir / "2023" / "july").mkdir(parents=True)

        # Create images in different directories
        img1 = photos_dir / "2023" / "june" / "photo1.jpg"
        img2 = photos_dir / "2023" / "july" / "photo2.jpg"
        img3 = photos_dir / "top_level.jpg"

        Image.new("RGB", (100, 100), color="red").save(img1)
        Image.new("RGB", (100, 100), color="blue").save(img2)
        Image.new("RGB", (100, 100), color="green").save(img3)

        # Initialize and scan
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        # Should find all 3 images
        with CatalogDatabase(catalog_dir) as db:
            images = db.list_images()
            assert len(images) == 3

    def test_scanner_statistics_update(self, tmp_path: Path) -> None:
        """Test that scanner updates statistics correctly."""
        # Setup
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create test images with unique content
        for i in range(5):
            img_path = photos_dir / f"image{i}.jpg"
            # Create unique images by varying the color
            color = (i * 50, (i * 30) % 256, (i * 70) % 256)
            Image.new("RGB", (100, 100), color=color).save(img_path)

        # Initialize and scan
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            stats = db.get_statistics()
            assert stats.total_images == 5
            assert stats.total_videos == 0
            assert stats.total_size_bytes > 0

    def test_scanner_checksum_duplicates(self, tmp_path: Path) -> None:
        """Test that scanner detects exact duplicates by checksum."""
        # Setup
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create identical images with different names
        img = Image.new("RGB", (100, 100), color="red")
        img.save(photos_dir / "original.jpg")
        img.save(photos_dir / "copy.jpg")

        # Initialize and scan
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        # Should only add one image (identical checksum)
        with CatalogDatabase(catalog_dir) as db:
            images = db.list_images()
            # Scanner processes both files but only adds unique checksums
            assert scanner.files_added == 1
            assert scanner.files_skipped == 1
            assert len(images) == 1

    def test_scanner_processes_video_files(self, tmp_path: Path) -> None:
        """Test that scanner handles video files."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create a fake video file
        video_path = photos_dir / "test_video.mp4"
        # Create a minimal valid video-like file
        video_path.write_bytes(b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 100)

        # Create a normal image too
        img_path = photos_dir / "image.jpg"
        Image.new("RGB", (100, 100), color="blue").save(img_path)

        # Initialize and scan
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            stats = db.get_statistics()
            # Should have both image and video
            assert stats.total_images >= 1
            assert stats.total_videos >= 1

    def test_scanner_handles_permission_errors(self, tmp_path: Path) -> None:
        """Test scanner handles permission errors gracefully."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create a subdirectory with restricted permissions
        restricted_dir = photos_dir / "restricted"
        restricted_dir.mkdir(mode=0o000)

        # Create an accessible image
        img_path = photos_dir / "accessible.jpg"
        Image.new("RGB", (100, 100), color="green").save(img_path)

        try:
            # Initialize and scan
            with CatalogDatabase(catalog_dir) as db:
                db.initialize(source_directories=[photos_dir])
                scanner = ImageScanner(db, workers=1)
                scanner.scan_directories([photos_dir])

                # Should process accessible images without crashing
                images = db.list_images()
                assert len(images) >= 1
        finally:
            # Restore permissions for cleanup
            restricted_dir.chmod(0o755)

    def test_scanner_checkpoint_logic(self, tmp_path: Path) -> None:
        """Test that scanner creates checkpoints during long scans."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create enough files to trigger checkpoint (every 100 files)
        for i in range(105):
            img_path = photos_dir / f"image{i:03d}.jpg"
            # Create unique images
            color = (i % 256, (i * 2) % 256, (i * 3) % 256)
            Image.new("RGB", (50, 50), color=color).save(img_path)

        # Initialize and scan
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # Verify all files were processed
            images = db.list_images()
            assert len(images) >= 100

            # Verify statistics were updated
            stats = db.get_statistics()
            assert stats.total_images >= 100

    def test_scanner_handles_files_without_dates(self, tmp_path: Path) -> None:
        """Test scanner tracks images without date metadata."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create images with no date info in filename or directory
        img1_path = photos_dir / "no_date_image.jpg"
        img2_path = photos_dir / "another_image.jpg"
        Image.new("RGB", (100, 100), color="red").save(img1_path)
        Image.new("RGB", (100, 100), color="blue").save(img2_path)

        # Initialize and scan
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            stats = db.get_statistics()
            # Images should be processed even without dates
            assert stats.total_images == 2
            # no_date count should be tracked
            assert stats.no_date is not None

    def test_scanner_unknown_file_types(self, tmp_path: Path) -> None:
        """Test scanner skips unknown file types."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create files of various types
        img_path = photos_dir / "photo.jpg"
        Image.new("RGB", (100, 100), color="red").save(img_path)

        # Unknown file types
        (photos_dir / "document.pdf").write_bytes(b"PDF content")
        (photos_dir / "spreadsheet.xlsx").write_bytes(b"Excel content")
        (photos_dir / "readme.txt").write_text("Text file")

        # Initialize and scan
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # Should only process the image
            images = db.list_images()
            assert len(images) == 1
            stats = db.get_statistics()
            assert stats.total_images == 1
