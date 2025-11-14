"""
Tests for FastAPI web API.
"""

from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from vam_tools.analysis.scanner import ImageScanner
from vam_tools.core.database import CatalogDatabase
from vam_tools.web.api import app, get_catalog, init_catalog


class TestAPI:
    """Tests for FastAPI endpoints."""

    def test_root_endpoint_without_static(self) -> None:
        """Test root endpoint when static files don't exist."""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200

    def test_api_root(self) -> None:
        """Test API root endpoint."""
        client = TestClient(app)
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "2.0.0"

    def test_get_catalog_not_initialized(self) -> None:
        """Test get_catalog raises when not initialized."""
        import vam_tools.web.api as api_module

        # Reset global state
        api_module._catalog = None
        api_module._catalog_path = None

        try:
            get_catalog()
            assert False, "Should have raised HTTPException"
        except Exception as e:
            assert "Catalog not initialized" in str(e)

    def test_init_catalog(self, tmp_path: Path) -> None:
        """Test catalog initialization."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create test catalog
        Image.new("RGB", (100, 100), color="red").save(photos_dir / "test.jpg")
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            db.execute(
                "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ("source_directories", str(photos_dir)),
            )
            # scanner = ImageScanner(db, workers=1)
            # scanner.scan_directories([photos_dir])

        # Initialize for API
        init_catalog(catalog_dir)

        # Should be able to get catalog
        catalog = get_catalog()
        assert catalog is not None

    def test_get_catalog_info(self, tmp_path: Path) -> None:
        """Test getting catalog info endpoint."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create catalog
        Image.new("RGB", (100, 100), color="blue").save(photos_dir / "test.jpg")
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            db.execute(
                "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ("source_directories", str(photos_dir)),
            )
            db.execute(
                "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ("catalog_id", "test-catalog-id"),
            )
            db.execute(
                "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ("created", datetime.now().isoformat()),
            )
            db.execute(
                "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ("last_updated", datetime.now().isoformat()),
            )
            db.execute(
                "INSERT OR REPLACE INTO catalog_config (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ("phase", "analyzed"),
            )
            # Insert a statistics snapshot
            db.execute(
                """
                INSERT INTO statistics (
                    timestamp, total_images, total_videos, total_size_bytes,
                    images_scanned, images_hashed, images_tagged,
                    duplicate_groups, duplicate_images, potential_savings_bytes,
                    high_quality_count, medium_quality_count, low_quality_count,
                    corrupted_count, unsupported_count,
                    processing_time_seconds, images_per_second,
                    no_date, suspicious_dates, problematic_files
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    1,
                    0,
                    1000,  # total_images, total_videos, total_size_bytes
                    1,
                    1,
                    0,  # images_scanned, images_hashed, images_tagged
                    0,
                    0,
                    0,  # duplicate_groups, duplicate_images, potential_savings_bytes
                    0,
                    0,
                    0,  # high_quality_count, medium_quality_count, low_quality_count
                    0,
                    0,  # corrupted_count, unsupported_count
                    1.0,
                    1.0,  # processing_time_seconds, images_per_second
                    0,
                    0,
                    0,  # no_date, suspicious_dates, problematic_files
                ),
            )
            # scanner = ImageScanner(db, workers=1)
            # scanner.scan_directories([photos_dir])

        # Init and test
        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/catalog/info")

        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "2.0.0"
        assert data["catalog_id"] == "test-catalog-id"
        assert "created" in data
        assert "last_updated" in data
        assert data["phase"] == "analyzed"
        assert "statistics" in data
        assert data["statistics"]["total_images"] == 1

    def test_list_images(self, tmp_path: Path) -> None:
        """Test listing images endpoint."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create multiple images
        for i in range(5):
            color = (i * 50, 0, 0)
            img_path = photos_dir / f"photo{i}.jpg"
            Image.new("RGB", (100, 100), color=color).save(img_path)
            with CatalogDatabase(catalog_dir) as db:
                db.initialize()
                db.execute(
                    """
                    INSERT INTO images (
                        id, source_path, file_size, file_hash, format,
                        width, height, created_at, modified_at, indexed_at,
                        date_taken, quality_score, is_corrupted, thumbnail_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"img{i}",
                        str(img_path),
                        1000 + i,
                        f"hash{i}",
                        "JPEG",
                        100,
                        100,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime(2023, 1, i + 1).isoformat(),
                        80,
                        0,
                        f"thumbnails/img{i}.jpg" if i % 2 == 0 else None,
                    ),
                )

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/images?sort_by=id&sort_order=asc")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 5
        assert "id" in data[0]
        assert "source_path" in data[0]
        assert "file_type" in data[0]
        assert "thumbnail_path" in data[0]
        assert data[0]["thumbnail_path"] == "thumbnails/img0.jpg"
        assert data[1]["thumbnail_path"] is None

    def test_list_images_pagination(self, tmp_path: Path) -> None:
        """Test image listing with pagination."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create 10 images
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            for i in range(10):
                img_path = photos_dir / f"photo{i}.jpg"
                Image.new("RGB", (100, 100), color=(i * 25, 0, 0)).save(img_path)
                db.execute(
                    """
                    INSERT INTO images (
                        id, source_path, file_size, file_hash, format,
                        width, height, created_at, modified_at, indexed_at,
                        date_taken, quality_score, is_corrupted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"img{i}",
                        str(img_path),
                        1000 + i,
                        f"hash{i}",
                        "JPEG",
                        100,
                        100,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime(2023, 1, i + 1).isoformat(),
                        80,
                        0,
                    ),
                )

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Get first page
        response = client.get("/api/images?skip=0&limit=5")
        assert response.status_code == 200
        page1 = response.json()
        assert len(page1) == 5

        # Get second page
        response = client.get("/api/images?skip=5&limit=5")
        assert response.status_code == 200
        page2 = response.json()
        assert len(page2) == 5

    def test_list_images_filter_type(self, tmp_path: Path) -> None:
        """Test filtering images by type."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create images
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            img_path_jpg = photos_dir / "photo.jpg"
            Image.new("RGB", (100, 100), color="green").save(img_path_jpg)
            db.execute(
                """
                INSERT INTO images (
                    id, source_path, file_size, file_hash, format,
                    width, height, created_at, modified_at, indexed_at,
                    date_taken, quality_score, is_corrupted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "img_jpg",
                    str(img_path_jpg),
                    1000,
                    "hash_jpg",
                    "JPEG",
                    100,
                    100,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime(2023, 1, 1).isoformat(),
                    80,
                    0,
                ),
            )

            img_path_mp4 = photos_dir / "video.mp4"
            img_path_mp4.touch()  # Create dummy video file
            db.execute(
                """
                INSERT INTO images (
                    id, source_path, file_size, file_hash, format,
                    width, height, created_at, modified_at, indexed_at,
                    date_taken, quality_score, is_corrupted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "vid_mp4",
                    str(img_path_mp4),
                    5000,
                    "hash_mp4",
                    "MP4",
                    1920,
                    1080,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime(2023, 1, 2).isoformat(),
                    70,
                    0,
                ),
            )

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Filter by image type
        response = client.get("/api/images?filter_type=image")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["file_type"] == "image"

        # Filter by video type
        response = client.get("/api/images?filter_type=video")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["file_type"] == "video"

    def test_list_images_sort_by(self, tmp_path: Path) -> None:
        """Test sorting images."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create images
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            for i in range(3):
                img_path = photos_dir / f"photo{i}.jpg"
                Image.new(
                    "RGB", (100 + i * 10, 100 + i * 10), color=(i * 80, 0, 0)
                ).save(img_path)
                db.execute(
                    """
                    INSERT INTO images (
                        id, source_path, file_size, file_hash, format,
                        width, height, created_at, modified_at, indexed_at,
                        date_taken, quality_score, is_corrupted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"img{i}",
                        str(img_path),
                        1000 + i * 100,  # Vary file size
                        f"hash{i}",
                        "JPEG",
                        100 + i * 10,
                        100 + i * 10,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime(2023, 1, 3 - i).isoformat(),  # Vary date for sorting
                        80,
                        0,
                    ),
                )

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Sort by path
        response = client.get("/api/images?sort_by=path")
        assert response.status_code == 200
        data = response.json()
        assert data[0]["id"] == "img0"  # photo0.jpg
        assert data[1]["id"] == "img1"  # photo1.jpg

        # Sort by size
        response = client.get("/api/images?sort_by=size")
        assert response.status_code == 200
        data = response.json()
        assert data[0]["id"] == "img2"  # Largest size
        assert data[2]["id"] == "img0"  # Smallest size

        # Sort by date (descending)
        response = client.get("/api/images?sort_by=date")
        assert response.status_code == 200
        data = response.json()
        assert data[0]["id"] == "img0"  # Latest date (2023-01-03)
        assert data[2]["id"] == "img2"  # Earliest date (2023-01-01)

    def test_get_image_detail(self, tmp_path: Path) -> None:
        """Test getting detailed image information."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create image
        img_path = photos_dir / "test.jpg"
        Image.new("RGB", (100, 100), color="purple").save(img_path)

        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            db.execute(
                """
                INSERT INTO images (
                    id, source_path, file_size, file_hash, format,
                    width, height, created_at, modified_at, indexed_at,
                    date_taken, quality_score, is_corrupted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "test_img",
                    str(img_path),
                    1000,
                    "testhash",
                    "JPEG",
                    100,
                    100,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime(2023, 1, 1).isoformat(),
                    80,
                    0,
                ),
            )
            image_id = "test_img"

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get(f"/api/images/{image_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == image_id
        assert "source_path" in data
        assert "checksum" in data
        assert "dates" in data
        assert "metadata" in data
        assert data["metadata"]["format"] == "JPEG"
        assert data["dates"]["selected_date"] == datetime(2023, 1, 1).isoformat()

    def test_get_image_detail_not_found(self, tmp_path: Path) -> None:
        """Test getting non-existent image."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize()

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/images/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_image_file(self, tmp_path: Path) -> None:
        """Test serving image files."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create image
        img_path = photos_dir / "test.jpg"
        Image.new("RGB", (100, 100), color="orange").save(img_path)

        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            db.execute(
                """
                INSERT INTO images (
                    id, source_path, file_size, file_hash, format,
                    width, height, created_at, modified_at, indexed_at,
                    date_taken, quality_score, is_corrupted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "test_img",
                    str(img_path),
                    1000,
                    "testhash",
                    "JPEG",
                    100,
                    100,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime(2023, 1, 1).isoformat(),
                    80,
                    0,
                ),
            )
            image_id = "test_img"

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get(f"/api/images/{image_id}/file")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/")

    def test_get_statistics_summary(self, tmp_path: Path) -> None:
        """Test statistics summary endpoint."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create images
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            for i in range(3):
                img_path = photos_dir / f"photo{i}.jpg"
                Image.new("RGB", (100, 100), color=(i * 80, 0, 0)).save(img_path)
                db.execute(
                    """
                    INSERT INTO images (
                        id, source_path, file_size, file_hash, format,
                        width, height, created_at, modified_at, indexed_at,
                        date_taken, quality_score, is_corrupted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"img{i}",
                        str(img_path),
                        1000 + i,
                        f"hash{i}",
                        "JPEG",
                        100,
                        100,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime(2023, 1, i + 1).isoformat(),
                        80,
                        0,
                    ),
                )
            # Insert a statistics snapshot
            db.execute(
                """
                INSERT INTO statistics (
                    timestamp, total_images, total_videos, total_size_bytes,
                    images_scanned, images_hashed, images_tagged,
                    duplicate_groups, duplicate_images, potential_savings_bytes,
                    high_quality_count, medium_quality_count, low_quality_count,
                    corrupted_count, unsupported_count,
                    processing_time_seconds, images_per_second,
                    no_date, suspicious_dates, problematic_files
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    3,
                    0,
                    3000,  # total_images, total_videos, total_size_bytes
                    3,
                    3,
                    0,  # images_scanned, images_hashed, images_tagged
                    0,
                    0,
                    0,  # duplicate_groups, duplicate_images, potential_savings_bytes
                    0,
                    0,
                    0,  # high_quality_count, medium_quality_count, low_quality_count
                    0,
                    0,  # corrupted_count, unsupported_count
                    1.0,
                    1.0,  # processing_time_seconds, images_per_second
                    0,
                    0,
                    0,  # no_date, suspicious_dates, problematic_files
                ),
            )

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/statistics/summary")

        assert response.status_code == 200
        data = response.json()
        # Statistics endpoint returns breakdown by category, format, etc.
        assert isinstance(data, dict)
        assert data["total"]["images"] == 3
        assert data["by_format"]["JPEG"] == 3

    def test_catalog_reload_on_change(self, tmp_path: Path) -> None:
        """Test that catalog reloads when file changes."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create initial catalog
        img1_path = photos_dir / "test1.jpg"
        Image.new("RGB", (100, 100), color="cyan").save(img1_path)
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            db.execute(
                """
                INSERT INTO images (
                    id, source_path, file_size, file_hash, format,
                    width, height, created_at, modified_at, indexed_at,
                    date_taken, quality_score, is_corrupted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "img1",
                    str(img1_path),
                    1000,
                    "hash1",
                    "JPEG",
                    100,
                    100,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime(2023, 1, 1).isoformat(),
                    80,
                    0,
                ),
            )

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Get initial count
        response = client.get("/api/images")
        initial_count = len(response.json())
        assert initial_count == 1

        # Add another image and update catalog
        img2_path = photos_dir / "test2.jpg"
        Image.new("RGB", (100, 100), color="magenta").save(img2_path)
        with CatalogDatabase(catalog_dir) as db:
            db.execute(
                """
                INSERT INTO images (
                    id, source_path, file_size, file_hash, format,
                    width, height, created_at, modified_at, indexed_at,
                    date_taken, quality_score, is_corrupted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "img2",
                    str(img2_path),
                    1000,
                    "hash2",
                    "JPEG",
                    100,
                    100,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime(2023, 1, 2).isoformat(),
                    80,
                    0,
                ),
            )

        # Should reload and show new image
        response = client.get("/api/images")
        new_count = len(response.json())
        assert new_count == 2

    def test_list_images_invalid_filter(self, tmp_path: Path) -> None:
        """Test invalid filter type."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize()

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/images?filter_type=invalid")

        assert response.status_code == 422  # Validation error

    def test_list_images_invalid_sort(self, tmp_path: Path) -> None:
        """Test invalid sort parameter."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize()

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/images?sort_by=invalid")

        assert response.status_code == 422  # Validation error

    def test_pagination_bounds(self, tmp_path: Path) -> None:
        """Test pagination boundary conditions."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        img_path = photos_dir / "test.jpg"
        Image.new("RGB", (100, 100), color="yellow").save(img_path)
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            db.execute(
                """
                INSERT INTO images (
                    id, source_path, file_size, file_hash, format,
                    width, height, created_at, modified_at, indexed_at,
                    date_taken, quality_score, is_corrupted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "test_img",
                    str(img_path),
                    1000,
                    "testhash",
                    "JPEG",
                    100,
                    100,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime(2023, 1, 1).isoformat(),
                    80,
                    0,
                ),
            )

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Test negative skip
        response = client.get("/api/images?skip=-1")
        assert response.status_code == 422

        # Test zero limit
        response = client.get("/api/images?limit=0")
        assert response.status_code == 422

        # Test limit above maximum
        response = client.get("/api/images?limit=2000")
        assert response.status_code == 422

    def test_empty_catalog(self, tmp_path: Path) -> None:
        """Test API with empty catalog."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize()

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Info endpoint should work
        response = client.get("/api/catalog/info")
        assert response.status_code == 200

        # Images endpoint should return empty list
        response = client.get("/api/images")
        assert response.status_code == 200
        assert response.json() == []


class TestAPIModels:
    """Tests for API Pydantic models."""

    def test_image_summary_model(self) -> None:
        """Test ImageSummary model."""
        from vam_tools.web.api import ImageSummary

        summary = ImageSummary(
            id="test123",
            source_path="/path/to/image.jpg",
            file_type="image",
            selected_date="2023-12-25T10:30:00",
            date_source="exif",
            confidence=95,
            suspicious=False,
            format="JPEG",
            resolution=(1920, 1080),
            size_bytes=1024000,
            thumbnail_path="thumbnails/test123.jpg",
        )
        assert summary.id == "test123"
        assert summary.file_type == "image"

    def test_catalog_stats_model(self) -> None:
        """Test CatalogStats model."""
        from vam_tools.web.api import CatalogStats

        stats = CatalogStats(
            total_images=100,
            total_videos=10,
            total_size_bytes=1024 * 1024 * 500,
            no_date=5,
            suspicious_dates=3,
        )
        assert stats.total_images == 100
        assert stats.suspicious_dates == 3


class TestDashboardAPI:
    """Tests for dashboard statistics API endpoints."""

    def test_get_dashboard_stats_empty_catalog(self, tmp_path: Path) -> None:
        """Test dashboard stats with empty catalog."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize()

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/dashboard/stats")

        assert response.status_code == 200
        data = response.json()

        # Check structure
        assert "catalog" in data
        assert "duplicates" in data
        assert "review" in data
        assert "hashes" in data

        # Check catalog section
        assert data["catalog"]["total_files"] == 0
        assert data["catalog"]["total_images"] == 0
        assert data["catalog"]["total_videos"] == 0
        assert data["catalog"]["total_size_bytes"] == 0
        assert data["catalog"]["total_size_gb"] == 0.0

        # Check duplicates section
        assert data["duplicates"]["total_groups"] == 0
        assert data["duplicates"]["needs_review"] == 0
        assert data["duplicates"]["total_duplicate_images"] == 0
        assert data["duplicates"]["potential_space_savings_bytes"] == 0
        assert data["duplicates"]["potential_space_savings_gb"] == 0.0

        # Check review section
        assert data["review"]["total_needing_review"] == 0
        assert data["review"]["date_conflicts"] == 0
        assert data["review"]["no_date"] == 0
        assert data["review"]["suspicious_dates"] == 0
        assert data["review"]["low_confidence"] == 0

        # Check hashes section
        assert data["hashes"]["images_with_dhash"] == 0
        assert data["hashes"]["images_with_ahash"] == 0
        assert data["hashes"]["images_with_whash"] == 0
        assert data["hashes"]["images_with_any_hash"] == 0
        assert data["hashes"]["coverage_percent"] == 0.0
        assert data["hashes"]["dhash_percent"] == 0.0
        assert data["hashes"]["ahash_percent"] == 0.0
        assert data["hashes"]["whash_percent"] == 0.0

    def test_get_dashboard_stats_with_images(self, tmp_path: Path) -> None:
        """Test dashboard stats with actual images."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create test images
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            for i in range(5):
                img_path = photos_dir / f"test{i}.jpg"
                Image.new("RGB", (100, 100), color="red").save(img_path)
                db.execute(
                    """
                    INSERT INTO images (
                        id, source_path, file_size, file_hash, format,
                        width, height, created_at, modified_at, indexed_at,
                        date_taken, quality_score, is_corrupted, perceptual_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"img{i}",
                        str(img_path),
                        1000 + i,
                        f"hash{i}",
                        "JPEG",
                        100,
                        100,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime(2023, 1, i + 1).isoformat(),
                        80,
                        0,
                        "1010101010101010",  # Add a perceptual hash
                    ),
                )
            # Insert a statistics snapshot
            db.execute(
                """
                INSERT INTO statistics (
                    timestamp, total_images, total_videos, total_size_bytes,
                    images_scanned, images_hashed, images_tagged,
                    duplicate_groups, duplicate_images, potential_savings_bytes,
                    high_quality_count, medium_quality_count, low_quality_count,
                    corrupted_count, unsupported_count,
                    processing_time_seconds, images_per_second,
                    no_date, suspicious_dates, problematic_files
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    5,
                    0,
                    5000,  # total_images, total_videos, total_size_bytes
                    5,
                    5,
                    0,  # images_scanned, images_hashed, images_tagged
                    0,
                    0,
                    0,  # duplicate_groups, duplicate_images, potential_savings_bytes
                    0,
                    0,
                    0,  # high_quality_count, medium_quality_count, low_quality_count
                    0,
                    0,  # corrupted_count, unsupported_count
                    1.0,
                    1.0,  # processing_time_seconds, images_per_second
                    0,
                    0,
                    0,  # no_date, suspicious_dates, problematic_files
                ),
            )

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/dashboard/stats")

        assert response.status_code == 200
        data = response.json()

        # Verify structure and types
        assert data["catalog"]["total_files"] == 5
        assert data["catalog"]["total_images"] == 5
        assert data["catalog"]["total_videos"] == 0
        assert isinstance(data["catalog"]["total_size_bytes"], int)
        assert isinstance(data["catalog"]["total_size_gb"], (int, float))

        # No duplicates yet
        assert data["duplicates"]["total_groups"] == 0

        # Hash stats should be present
        assert data["hashes"]["images_with_dhash"] == 5
        assert data["hashes"]["coverage_percent"] == 100.0

    def test_get_dashboard_stats_performance(self, tmp_path: Path) -> None:
        """Test dashboard stats completes in reasonable time."""
        import time

        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create 100 test images
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            for i in range(100):
                img_path = photos_dir / f"test{i}.jpg"
                Image.new("RGB", (100, 100), color="red").save(img_path)
                db.execute(
                    """
                    INSERT INTO images (
                        id, source_path, file_size, file_hash, format,
                        width, height, created_at, modified_at, indexed_at,
                        date_taken, quality_score, is_corrupted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"img{i}",
                        str(img_path),
                        1000 + i,
                        f"hash{i}",
                        "JPEG",
                        100,
                        100,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        datetime(2023, 1, i % 28 + 1).isoformat(),
                        80,
                        0,
                    ),
                )
            # Insert a statistics snapshot
            db.execute(
                """
                INSERT INTO statistics (
                    timestamp, total_images, total_videos, total_size_bytes,
                    images_scanned, images_hashed, images_tagged,
                    duplicate_groups, duplicate_images, potential_savings_bytes,
                    high_quality_count, medium_quality_count, low_quality_count,
                    corrupted_count, unsupported_count,
                    processing_time_seconds, images_per_second,
                    no_date, suspicious_dates, problematic_files
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    100,
                    0,
                    100000,  # total_images, total_videos, total_size_bytes
                    100,
                    100,
                    0,  # images_scanned, images_hashed, images_tagged
                    0,
                    0,
                    0,  # duplicate_groups, duplicate_images, potential_savings_bytes
                    0,
                    0,
                    0,  # high_quality_count, medium_quality_count, low_quality_count
                    0,
                    0,  # corrupted_count, unsupported_count
                    1.0,
                    1.0,  # processing_time_seconds, images_per_second
                    0,
                    0,
                    0,  # no_date, suspicious_dates, problematic_files
                ),
            )

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Measure response time
        start = time.time()
        response = client.get("/api/dashboard/stats")
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should complete in under 5 seconds even with 100 images
        assert elapsed < 5.0

        # Just verify we got valid data
        data = response.json()
        assert data["catalog"]["total_files"] == 100


class TestImageCaching:
    """Tests for image endpoint caching headers."""

    def test_image_file_has_cache_headers(self, tmp_path: Path) -> None:
        """Test that image files are served with cache headers."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create test image
        img_path = photos_dir / "test.jpg"
        Image.new("RGB", (100, 100), color="blue").save(img_path)

        # Create catalog
        with CatalogDatabase(catalog_dir) as db:
            db.initialize()
            db.execute(
                """
                INSERT INTO images (
                    id, source_path, file_size, file_hash, format,
                    width, height, created_at, modified_at, indexed_at,
                    date_taken, quality_score, is_corrupted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "test_img",
                    str(img_path),
                    1000,
                    "testhash",
                    "JPEG",
                    100,
                    100,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    datetime(2023, 1, 1).isoformat(),
                    80,
                    0,
                ),
            )
            image_id = "test_img"

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get(f"/api/images/{image_id}/file")

        assert response.status_code == 200
        # Check for cache-control header
        assert "cache-control" in response.headers
        cache_control = response.headers["cache-control"]
        assert "public" in cache_control
        assert "max-age" in cache_control
        # Should cache for at least 1 hour (3600 seconds)
        assert "3600" in cache_control


class TestDuplicateAPI:
    """Tests for duplicate review API endpoints."""

    def test_get_duplicate_stats_empty(self, tmp_path: Path) -> None:
        """Test duplicate stats with no duplicates."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize()

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/duplicates/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_groups"] == 0
        assert data["needs_review"] == 0
        assert data["total_duplicates"] == 0

    def test_list_duplicate_groups_empty(self, tmp_path: Path) -> None:
        """Test listing duplicate groups when none exist."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize()

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/duplicates/groups")

        assert response.status_code == 200
        assert response.json() == []

    def test_get_duplicate_group_not_found(self, tmp_path: Path) -> None:
        """Test getting non-existent duplicate group."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize()

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/duplicates/groups/nonexistent")

        assert response.status_code == 404
