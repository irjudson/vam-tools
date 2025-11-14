"""Tests for scanner functionality."""

import tempfile
import uuid
from pathlib import Path

import pytest
from PIL import Image

from vam_tools.db.catalog_schema import (
    create_schema,
    delete_catalog_data,
    get_image_count,
    schema_exists,
)
from vam_tools.tasks.scanner import compute_checksum, scan_directory


@pytest.fixture(autouse=True)
def ensure_schema_exists():
    """Ensure the main schema exists before each test."""
    if not schema_exists():
        create_schema()
    yield


@pytest.fixture
def test_catalog_id():
    """Create a test catalog ID for scanner tests."""
    from vam_tools.db.connection import SessionLocal
    from vam_tools.db.models import Catalog

    catalog_id = uuid.uuid4()

    # Create catalog record in database
    db = SessionLocal()
    try:
        catalog = Catalog(
            id=catalog_id,
            name="Test Scanner Catalog",
            schema_name=f"deprecated_{catalog_id}",
            source_directories=["/test"],
        )
        db.add(catalog)
        db.commit()
    finally:
        db.close()

    yield str(catalog_id)

    # Cleanup after test
    db = SessionLocal()
    try:
        delete_catalog_data(str(catalog_id))
        catalog = db.query(Catalog).filter(Catalog.id == catalog_id).first()
        if catalog:
            db.delete(catalog)
            db.commit()
    except:
        pass
    finally:
        db.close()


@pytest.fixture
def test_images_dir():
    """Create a temporary directory with test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create some test images
        for i in range(5):
            img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
            img.save(tmpdir / f"test_image_{i}.jpg")

        yield tmpdir


def test_compute_checksum():
    """Test checksum computation."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        temp_path = Path(f.name)

    try:
        checksum = compute_checksum(temp_path)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 is 64 hex characters

        # Verify checksum is consistent
        checksum2 = compute_checksum(temp_path)
        assert checksum == checksum2
    finally:
        temp_path.unlink()


def test_scan_empty_directory(test_catalog_id):
    """Test scanning an empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = scan_directory(Path(tmpdir), test_catalog_id)

        assert stats["files_found"] == 0
        assert stats["files_added"] == 0
        assert stats["files_skipped"] == 0
        assert stats["exact_duplicates"] == 0

        # Verify no images in database
        assert get_image_count(test_catalog_id) == 0


def test_scan_directory_with_images(test_catalog_id, test_images_dir):
    """Test scanning a directory with images."""
    stats = scan_directory(test_images_dir, test_catalog_id)

    assert stats["files_found"] == 5
    assert stats["files_added"] == 5
    assert stats["files_skipped"] == 0
    assert stats["exact_duplicates"] == 0

    # Verify images were added to database
    assert get_image_count(test_catalog_id) == 5


def test_scan_detects_duplicates(test_catalog_id, test_images_dir):
    """Test that scanning same directory twice detects duplicates."""
    # First scan
    stats1 = scan_directory(test_images_dir, test_catalog_id)
    assert stats1["files_added"] == 5

    # Second scan (same directory)
    stats2 = scan_directory(test_images_dir, test_catalog_id)
    assert stats2["files_found"] == 5
    assert stats2["files_added"] == 0
    assert stats2["exact_duplicates"] == 5

    # Total images should still be 5
    assert get_image_count(test_catalog_id) == 5


def test_scan_with_progress_callback(test_catalog_id, test_images_dir):
    """Test scanning with progress callback."""
    progress_calls = []

    def progress_cb(current, total, message):
        progress_calls.append((current, total, message))

    stats = scan_directory(
        test_images_dir, test_catalog_id, progress_callback=progress_cb
    )

    # Progress callback should have been called
    assert len(progress_calls) > 0
    assert stats["files_added"] == 5


def test_scan_mixed_file_types(test_catalog_id):
    """Test scanning directory with mixed file types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create images with different colors (different checksums)
        for i in range(3):
            color = (i * 80, i * 80, i * 80)
            img = Image.new("RGB", (100, 100), color=color)
            img.save(tmpdir / f"image_{i}.jpg")

        # Create non-image files (should be skipped)
        (tmpdir / "textfile.txt").write_text("test")
        (tmpdir / "document.pdf").write_bytes(b"fake pdf")

        stats = scan_directory(tmpdir, test_catalog_id)

        # Should only find and add images (3 JPGs)
        # files_found includes all .jpg files found
        assert stats["files_found"] >= 3
        assert stats["files_added"] == 3
        assert get_image_count(test_catalog_id) == 3


def test_scan_nested_directories(test_catalog_id):
    """Test scanning with nested subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create nested structure
        (tmpdir / "subdir1").mkdir()
        (tmpdir / "subdir2").mkdir()
        (tmpdir / "subdir1" / "nested").mkdir()

        # Create images in different directories
        img = Image.new("RGB", (100, 100))
        img.save(tmpdir / "root.jpg")
        img_sub = Image.new("RGB", (100, 100), color="red")
        img_sub.save(tmpdir / "subdir1" / "sub1.jpg")
        img_nested = Image.new("RGB", (100, 100), color="blue")
        img_nested.save(tmpdir / "subdir1" / "nested" / "nested1.jpg")

        stats = scan_directory(tmpdir, test_catalog_id)

        # Should find all images recursively
        assert stats["files_found"] == 3
        assert stats["files_added"] == 3
        assert get_image_count(test_catalog_id) == 3


def test_scan_stores_file_metadata(test_catalog_id, test_images_dir):
    """Test that scan stores file metadata correctly."""
    from sqlalchemy import text

    from vam_tools.db.connection import SessionLocal

    stats = scan_directory(test_images_dir, test_catalog_id)
    assert stats["files_added"] > 0

    # Check that images have metadata
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                "SELECT id, source_path, file_type, checksum, status FROM images WHERE catalog_id = :catalog_id LIMIT 1"
            ),
            {"catalog_id": test_catalog_id},
        ).fetchone()

        assert result is not None
        assert result[1].endswith(".jpg")  # source_path
        assert result[2] == "image"  # file_type
        assert len(result[3]) == 64  # checksum (SHA256)
        assert result[4] == "complete"  # status
    finally:
        db.close()
