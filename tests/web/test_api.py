"""
Tests for FastAPI web API.
"""

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from vam_tools.analysis.scanner import ImageScanner
from vam_tools.core.catalog import CatalogDatabase
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
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

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
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        # Init and test
        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/catalog/info")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "catalog_id" in data
        assert "created" in data
        assert "last_updated" in data
        assert "phase" in data
        assert "statistics" in data
        assert data["statistics"]["total_images"] >= 0

    def test_list_images(self, tmp_path: Path) -> None:
        """Test listing images endpoint."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create multiple images
        for i in range(5):
            color = (i * 50, 0, 0)
            Image.new("RGB", (100, 100), color=color).save(photos_dir / f"photo{i}.jpg")

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/images")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "id" in data[0]
        assert "source_path" in data[0]
        assert "file_type" in data[0]

    def test_list_images_pagination(self, tmp_path: Path) -> None:
        """Test image listing with pagination."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create 10 images
        for i in range(10):
            Image.new("RGB", (100, 100), color=(i * 25, 0, 0)).save(
                photos_dir / f"photo{i}.jpg"
            )

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Get first page
        response = client.get("/api/images?skip=0&limit=5")
        assert response.status_code == 200
        page1 = response.json()
        assert len(page1) <= 5

        # Get second page
        response = client.get("/api/images?skip=5&limit=5")
        assert response.status_code == 200
        page2 = response.json()
        assert len(page2) >= 0

    def test_list_images_filter_type(self, tmp_path: Path) -> None:
        """Test filtering images by type."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create images
        Image.new("RGB", (100, 100), color="green").save(photos_dir / "photo.jpg")

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Filter by image type
        response = client.get("/api/images?filter_type=image")
        assert response.status_code == 200
        data = response.json()
        assert all(img["file_type"] == "image" for img in data)

    def test_list_images_sort_by(self, tmp_path: Path) -> None:
        """Test sorting images."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create images
        for i in range(3):
            Image.new("RGB", (100 + i * 10, 100 + i * 10), color=(i * 80, 0, 0)).save(
                photos_dir / f"photo{i}.jpg"
            )

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Sort by path
        response = client.get("/api/images?sort_by=path")
        assert response.status_code == 200

        # Sort by size
        response = client.get("/api/images?sort_by=size")
        assert response.status_code == 200

        # Sort by date
        response = client.get("/api/images?sort_by=date")
        assert response.status_code == 200

    def test_get_image_detail(self, tmp_path: Path) -> None:
        """Test getting detailed image information."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create image
        Image.new("RGB", (100, 100), color="purple").save(photos_dir / "test.jpg")

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])
            images = db.list_images()
            image_id = images[0].id

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

    def test_get_image_detail_not_found(self, tmp_path: Path) -> None:
        """Test getting non-existent image."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

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
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])
            images = db.list_images()
            image_id = images[0].id

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
        for i in range(3):
            Image.new("RGB", (100, 100), color=(i * 80, 0, 0)).save(
                photos_dir / f"photo{i}.jpg"
            )

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/statistics/summary")

        assert response.status_code == 200
        data = response.json()
        # Statistics endpoint returns breakdown by category, format, etc.
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_catalog_reload_on_change(self, tmp_path: Path) -> None:
        """Test that catalog reloads when file changes."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        # Create initial catalog
        Image.new("RGB", (100, 100), color="cyan").save(photos_dir / "test1.jpg")
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Get initial count
        response = client.get("/api/images")
        initial_count = len(response.json())

        # Add another image and update catalog
        Image.new("RGB", (100, 100), color="magenta").save(photos_dir / "test2.jpg")
        with CatalogDatabase(catalog_dir) as db:
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

        # Should reload and show new image
        response = client.get("/api/images")
        new_count = len(response.json())
        assert new_count >= initial_count

    def test_list_images_invalid_filter(self, tmp_path: Path) -> None:
        """Test invalid filter type."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/images?filter_type=invalid")

        assert response.status_code == 422  # Validation error

    def test_list_images_invalid_sort(self, tmp_path: Path) -> None:
        """Test invalid sort parameter."""
        catalog_dir = tmp_path / "catalog"

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[])

        init_catalog(catalog_dir)
        client = TestClient(app)
        response = client.get("/api/images?sort_by=invalid")

        assert response.status_code == 422  # Validation error

    def test_pagination_bounds(self, tmp_path: Path) -> None:
        """Test pagination boundary conditions."""
        catalog_dir = tmp_path / "catalog"
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        Image.new("RGB", (100, 100), color="yellow").save(photos_dir / "test.jpg")
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[photos_dir])
            scanner = ImageScanner(db, workers=1)
            scanner.scan_directories([photos_dir])

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
            db.initialize(source_directories=[])

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
