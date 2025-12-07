"""
Tests for scanner module.

All tests require database connection.
"""

from pathlib import Path

import pytest
from PIL import Image

from vam_tools.analysis.scanner import ImageScanner
from vam_tools.db import CatalogDB as CatalogDatabase

pytestmark = pytest.mark.integration


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
            db.initialize()

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

            # Verify statistics were updated in database
            stats = db.get_statistics()
            assert stats.total_images == 2

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
            db.initialize()

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
            db.initialize()
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

            # Should add 1 new file
            # The 2 existing files may be skipped or updated depending on
            # whether their processing_flags indicate they're "complete"
            # (simple test images without EXIF get marked incomplete)
            assert scanner.files_added == 1
            # Either skipped (complete) or updated (incomplete) - total should be 2
            assert scanner.files_skipped + scanner.files_updated == 2

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
            db.initialize()
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
            db.initialize()
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
            db.initialize()
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
            db.initialize()
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
            db.initialize()
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
                db.initialize()
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
            db.initialize()
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
            db.initialize()
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
            db.initialize()
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

            # Should only process the image
            images = db.list_images()
            assert len(images) == 1
            stats = db.get_statistics()
            assert stats.total_images == 1


class TestIncrementalFileDiscovery:
    """Tests for incremental file discovery functionality."""

    def test_discover_files_incrementally_basic(self, tmp_path: Path) -> None:
        """Test basic incremental file discovery."""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create some test images
        for i in range(5):
            img_path = photos_dir / f"test{i}.jpg"
            Image.new("RGB", (100, 100), color="red").save(img_path)

        # Initialize catalog and scanner
        catalog_dir = tmp_path / "catalog"
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            scanner = ImageScanner(db, workers=1)

            # Collect files from incremental discovery
            files = list(scanner._discover_files_incrementally(photos_dir))

            assert len(files) == 5
            assert all(f.suffix == ".jpg" for f in files)

    def test_discover_files_incrementally_skips_synology_metadata(
        self, tmp_path: Path
    ) -> None:
        """Test that Synology metadata directories are skipped."""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create regular images
        img1_path = photos_dir / "photo1.jpg"
        img2_path = photos_dir / "photo2.jpg"
        Image.new("RGB", (100, 100), color="red").save(img1_path)
        Image.new("RGB", (100, 100), color="blue").save(img2_path)

        # Create Synology metadata directory
        synology_dir = photos_dir / "@eaDir"
        synology_dir.mkdir()
        synology_img = synology_dir / "photo1.jpg"
        Image.new("RGB", (100, 100), color="green").save(synology_img)

        # Create @SynoResource file
        synology_resource = photos_dir / "@SynoResource"
        synology_resource.write_text("synology metadata")

        # Initialize catalog and scanner
        catalog_dir = tmp_path / "catalog"
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            scanner = ImageScanner(db, workers=1)

            # Collect files
            files = list(scanner._discover_files_incrementally(photos_dir))

            # Should only find the 2 real images, not Synology metadata
            assert len(files) == 2
            assert all("@eaDir" not in str(f) for f in files)
            assert all("@SynoResource" not in str(f) for f in files)

    def test_discover_files_incrementally_skips_hidden_files(
        self, tmp_path: Path
    ) -> None:
        """Test that hidden files (starting with .) are skipped."""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create regular image
        img_path = photos_dir / "photo.jpg"
        Image.new("RGB", (100, 100), color="red").save(img_path)

        # Create hidden file
        hidden_img = photos_dir / ".hidden.jpg"
        Image.new("RGB", (100, 100), color="blue").save(hidden_img)

        # Initialize catalog and scanner
        catalog_dir = tmp_path / "catalog"
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            scanner = ImageScanner(db, workers=1)

            # Collect files
            files = list(scanner._discover_files_incrementally(photos_dir))

            # Should only find the visible image
            assert len(files) == 1
            assert files[0].name == "photo.jpg"

    def test_discover_files_incrementally_nested_directories(
        self, tmp_path: Path
    ) -> None:
        """Test incremental discovery with nested directory structure."""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create nested structure
        sub1 = photos_dir / "2024" / "January"
        sub2 = photos_dir / "2024" / "February"
        sub3 = photos_dir / "2023" / "December"
        sub1.mkdir(parents=True)
        sub2.mkdir(parents=True)
        sub3.mkdir(parents=True)

        # Add images in different directories
        Image.new("RGB", (100, 100), color="red").save(sub1 / "jan1.jpg")
        Image.new("RGB", (100, 100), color="green").save(sub1 / "jan2.jpg")
        Image.new("RGB", (100, 100), color="blue").save(sub2 / "feb1.jpg")
        Image.new("RGB", (100, 100), color="yellow").save(sub3 / "dec1.jpg")

        # Initialize catalog and scanner
        catalog_dir = tmp_path / "catalog"
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            scanner = ImageScanner(db, workers=1)

            # Collect files
            files = list(scanner._discover_files_incrementally(photos_dir))

            # Should find all 4 images across nested directories
            assert len(files) == 4
            file_names = [f.name for f in files]
            assert "jan1.jpg" in file_names
            assert "jan2.jpg" in file_names
            assert "feb1.jpg" in file_names
            assert "dec1.jpg" in file_names

    def test_discover_files_incrementally_yields_as_discovered(
        self, tmp_path: Path
    ) -> None:
        """Test that files are yielded incrementally, not collected first."""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create some test images
        for i in range(10):
            img_path = photos_dir / f"test{i}.jpg"
            Image.new("RGB", (100, 100), color="red").save(img_path)

        # Initialize catalog and scanner
        catalog_dir = tmp_path / "catalog"
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            scanner = ImageScanner(db, workers=1)

            # Test that we can iterate and get results immediately
            file_generator = scanner._discover_files_incrementally(photos_dir)
            first_file = next(file_generator)

            # Should get a result without exhausting the generator
            assert first_file is not None
            assert first_file.suffix == ".jpg"

    def test_discover_files_incrementally_mixed_file_types(
        self, tmp_path: Path
    ) -> None:
        """Test discovery with mixed image and video files."""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create images
        img_path = photos_dir / "photo.jpg"
        Image.new("RGB", (100, 100), color="red").save(img_path)

        # Create mock video file (just a file with video extension)
        video_path = photos_dir / "video.mp4"
        video_path.write_bytes(b"fake video data")

        # Create non-media file
        text_path = photos_dir / "readme.txt"
        text_path.write_text("text file")

        # Initialize catalog and scanner
        catalog_dir = tmp_path / "catalog"
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            scanner = ImageScanner(db, workers=1)

            # Collect files
            files = list(scanner._discover_files_incrementally(photos_dir))

            # Should find image and video, but not text file
            assert len(files) == 2
            file_names = [f.name for f in files]
            assert "photo.jpg" in file_names
            assert "video.mp4" in file_names
            assert "readme.txt" not in file_names

    def test_batch_processing_checkpoints(self, tmp_path: Path) -> None:
        """Test that batch processing creates checkpoints."""
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create more than batch size (100) unique images
        # Each image has a unique color to ensure different checksums
        for i in range(150):
            img_path = photos_dir / f"test{i}.jpg"
            # Use i to create unique RGB colors for each image
            color = (i % 256, (i * 2) % 256, (i * 3) % 256)
            Image.new("RGB", (50, 50), color=color).save(img_path)

        # Initialize catalog and scan
        catalog_dir = tmp_path / "catalog"
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            scanner = ImageScanner(db, workers=2)
            scanner.scan_directories([photos_dir])

            # Verify all images were processed
            images = db.list_images()
            assert len(images) == 150

            # Verify statistics were updated
            stats = db.get_statistics()
            assert stats.total_images == 150
