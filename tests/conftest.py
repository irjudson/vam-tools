"""
Pytest configuration and fixtures for vam_tools tests.
"""

# ==============================================================================
# Test Database Isolation - MUST RUN BEFORE ANY IMPORTS
# ==============================================================================
import os
import sys

# Set environment variable BEFORE any vam_tools imports
os.environ["POSTGRES_DB"] = "vam-tools-test"

# Remove any already-imported vam_tools modules to force reload with test settings
modules_to_remove = [name for name in sys.modules if name.startswith("vam_tools")]
for module_name in modules_to_remove:
    del sys.modules[module_name]

import tempfile  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Generator  # noqa: E402

import pytest  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from sqlalchemy import create_engine, event, text  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

# Now import vam_tools with test environment variable set
from vam_tools.db.config import Settings  # noqa: E402

# Create test settings and verify it's using test database
test_settings = Settings()
assert test_settings.postgres_db == "vam-tools-test", (
    f"CRITICAL: Expected test database 'vam-tools-test', got '{test_settings.postgres_db}'. "
    f"Tests would write to production database!"
)

# Patch the global settings object
import vam_tools.db.config as config_module  # noqa: E402

config_module.settings = test_settings

# Create test database engine and session
test_engine = create_engine(
    test_settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=test_settings.sql_echo,
)


# Add safety check: verify we're NEVER connecting to production database
@event.listens_for(test_engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Safety check: ensure we're connecting to test database."""
    cursor = dbapi_conn.cursor()
    cursor.execute("SELECT current_database()")
    db_name = cursor.fetchone()[0]
    cursor.close()
    if db_name != "vam-tools-test":
        raise RuntimeError(
            f"CRITICAL SAFETY VIOLATION: Attempted to connect to '{db_name}' "
            f"instead of test database 'vam-tools-test'!"
        )


TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Patch connection module to use test database
import vam_tools.db.connection as connection_module  # noqa: E402

connection_module.engine = test_engine
connection_module.SessionLocal = TestSessionLocal

# Patch catalog_schema module to use test SessionLocal
import vam_tools.db.catalog_schema as catalog_schema_module  # noqa: E402

# Now import the rest
from vam_tools.db import Base, Catalog, CatalogDB  # noqa: E402

catalog_schema_module.SessionLocal = TestSessionLocal

# Try to import exiftool for setting EXIF data
try:
    import exiftool  # noqa: F401

    EXIFTOOL_AVAILABLE = True
except ImportError:
    EXIFTOOL_AVAILABLE = False


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a simple test image."""
    image_path = temp_dir / "test_image.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(image_path, "JPEG")
    return image_path


@pytest.fixture
def sample_images(temp_dir: Path) -> list[Path]:
    """Create multiple test images with different colors."""
    images = []

    colors = ["red", "green", "blue", "yellow"]
    for i, color in enumerate(colors):
        image_path = temp_dir / f"image_{i}.jpg"
        img = Image.new("RGB", (100, 100), color=color)
        img.save(image_path, "JPEG")
        images.append(image_path)

    return images


@pytest.fixture
def duplicate_images(temp_dir: Path) -> dict[str, list[Path]]:
    """
    Create sets of duplicate and similar images.

    Returns:
        Dictionary with 'exact' and 'similar' keys containing lists of image paths
    """
    # Create exact duplicates
    exact_dir = temp_dir / "exact"
    exact_dir.mkdir()

    original = Image.new("RGB", (100, 100), color="red")
    exact1 = exact_dir / "original.jpg"
    exact2 = exact_dir / "duplicate.jpg"

    original.save(exact1, "JPEG")
    original.save(exact2, "JPEG")

    # Create similar images (same but different sizes)
    similar_dir = temp_dir / "similar"
    similar_dir.mkdir()

    # Original blue image
    blue1 = Image.new("RGB", (100, 100), color="blue")
    similar1 = similar_dir / "blue_100.jpg"
    blue1.save(similar1, "JPEG")

    # Same blue image, different size
    blue2 = Image.new("RGB", (200, 200), color="blue")
    similar2 = similar_dir / "blue_200.jpg"
    blue2.save(similar2, "JPEG")

    # Slightly different blue image (with a small rectangle)
    blue3 = Image.new("RGB", (100, 100), color="blue")
    draw = ImageDraw.Draw(blue3)
    draw.rectangle([10, 10, 20, 20], fill="lightblue")
    similar3 = similar_dir / "blue_modified.jpg"
    blue3.save(similar3, "JPEG")

    return {
        "exact": [exact1, exact2],
        "similar": [similar1, similar2, similar3],
    }


@pytest.fixture
def dated_images(temp_dir: Path) -> dict[str, Path]:
    """
    Create images with dates in filenames.

    Returns:
        Dictionary mapping date strings to image paths
    """
    dates = {
        "2023-01-15": "IMG_2023-01-15.jpg",
        "2023-06-20": "photo_2023-06-20_120000.jpg",
        "2022-12-25": "20221225_family.jpg",
    }

    images = {}
    for date_str, filename in dates.items():
        image_path = temp_dir / filename
        img = Image.new("RGB", (100, 100), color="white")
        img.save(image_path, "JPEG")
        images[date_str] = image_path

    return images


@pytest.fixture
def directory_structure(temp_dir: Path) -> dict[str, Path]:
    """
    Create a directory structure with year/month folders.

    Returns:
        Dictionary mapping paths to their expected dates
    """
    structure = {
        "2023/01-15": "2023-01",
        "2023/06-20": "2023-06",
        "2022/12-25": "2022-12",
    }

    images = {}
    for path_str, expected_date in structure.items():
        dir_path = temp_dir / path_str
        dir_path.mkdir(parents=True, exist_ok=True)

        image_path = dir_path / "image.jpg"
        img = Image.new("RGB", (100, 100), color="white")
        img.save(image_path, "JPEG")
        images[expected_date] = image_path

    return images


@pytest.fixture
def non_image_files(temp_dir: Path) -> list[Path]:
    """Create some non-image files."""
    files = []

    # Text file
    txt_file = temp_dir / "readme.txt"
    txt_file.write_text("This is a text file")
    files.append(txt_file)

    # Python file
    py_file = temp_dir / "script.py"
    py_file.write_text("print('hello')")
    files.append(py_file)

    return files


@pytest.fixture
def mixed_directory(
    temp_dir: Path, sample_images: list[Path], non_image_files: list[Path]
) -> Path:
    """Create a directory with mixed file types."""
    # sample_images and non_image_files already created files in temp_dir
    return temp_dir


# ==============================================================================
# PostgreSQL Database Fixtures - Optimized with proper scoping
# ==============================================================================


@pytest.fixture(scope="session")
def session_engine():
    """
    Session-scoped database engine (created once for all tests).

    This is the most expensive operation - creating the database connection and
    tables. We only do this ONCE for the entire test run.
    """
    # The test_engine is already created at module level with safety checks
    # Create all tables once
    # Base.metadata.create_all() creates: catalogs, images, tags, image_tags,
    # duplicate_groups, duplicate_members, jobs, config, statistics
    Base.metadata.create_all(test_engine)

    yield test_engine

    # Cleanup: just dispose the engine, transactions handle data cleanup
    test_engine.dispose()


@pytest.fixture
def db_session(session_engine):
    """
    Function-scoped database session (one per test).

    Uses transactions to isolate tests:
    - Each test gets a fresh transaction
    - Changes are automatically rolled back after the test
    - Tests don't interfere with each other
    - Much faster than dropping/creating tables for every test
    """
    connection = session_engine.connect()
    transaction = connection.begin()

    # Create session bound to this specific transaction
    session = TestSessionLocal(bind=connection)

    yield session

    # Cleanup: rollback all changes and close
    session.close()
    transaction.rollback()
    connection.close()


# Keep backward compatibility with old fixture name
@pytest.fixture
def test_db_session(db_session):
    """Alias for db_session - for backward compatibility."""
    return db_session


# CatalogDB now handles both Path (for tests) and UUID str (for production) automatically
