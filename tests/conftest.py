"""
Pytest configuration and fixtures for vam_tools tests.

This version doesn't globally patch SessionLocal, avoiding deadlock issues.
"""

# ==============================================================================
# Test Database Isolation - MUST RUN BEFORE ANY IMPORTS
# ==============================================================================
import os
import sys

# Get worker ID from environment (set by pytest-xdist)
# This allows each xdist worker to use a separate database
worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")
test_db_name = f"vam-tools-test-{worker_id}"

# Set environment variable BEFORE any vam_tools imports
os.environ["POSTGRES_DB"] = test_db_name

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
assert test_settings.postgres_db == test_db_name, (
    f"CRITICAL: Expected test database '{test_db_name}', got '{test_settings.postgres_db}'. "
    f"Tests would write to production database!"
)

# ==============================================================================
# Database setup WITHOUT global patching
# ==============================================================================

# Import after settings are configured
from vam_tools.db import Base  # noqa: E402


def get_test_engine():
    """Create a test database engine."""
    engine = create_engine(
        test_settings.database_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=test_settings.sql_echo,
    )

    # Safety check: verify test database (accounting for worker-specific DB names)
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("SELECT current_database()")
        db_name = cursor.fetchone()[0]
        cursor.close()
        if not db_name.startswith("vam-tools-test"):
            raise RuntimeError(
                f"CRITICAL: Attempted to connect to '{db_name}' "
                f"instead of a vam-tools-test database!"
            )

    return engine


# Create a single test engine for the session
_test_engine = get_test_engine()


# ==============================================================================
# Session-scoped fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def engine():
    """Provide the test database engine."""
    return _test_engine


@pytest.fixture(scope="session")
def tables_created(engine):
    """
    Create all tables once for the test session.

    This creates tables in the current worker's database.
    Note: standalone_catalog_db creates its own connections and needs
    to ensure tables exist independently.
    """
    # Create pgvector extension before creating tables
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(bind=engine)

    # Populate ImageStatus lookup table FIRST (before adding foreign key)
    from vam_tools.db.models import ImageStatus
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # Check if statuses already exist
        existing_count = session.query(ImageStatus).count()
        if existing_count == 0:
            # Insert initial statuses
            statuses = [
                ImageStatus(id='active', name='Active', description='Normal visible image'),
                ImageStatus(id='rejected', name='Rejected', description='Rejected from burst/duplicate review'),
                ImageStatus(id='archived', name='Archived', description='Manually archived by user'),
                ImageStatus(id='flagged', name='Flagged', description='Flagged for review or special attention'),
            ]
            session.add_all(statuses)
            session.commit()
    finally:
        session.close()

    # Apply status_id column migration if not already applied
    # This is needed because existing test databases may have been created
    # before the status_id column was added to the Image model
    with engine.connect() as conn:
        # Check if status_id column exists
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'images' AND column_name = 'status_id'"
        ))
        if not result.fetchone():
            # Add the column
            conn.execute(text(
                "ALTER TABLE images ADD COLUMN status_id VARCHAR(50) "
                "DEFAULT 'active' NOT NULL"
            ))
            conn.execute(text(
                "ALTER TABLE images ADD CONSTRAINT fk_images_status_id "
                "FOREIGN KEY (status_id) REFERENCES image_statuses(id) "
                "ON DELETE RESTRICT"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_images_status_id ON images(status_id)"
            ))
        conn.commit()

    yield
    # Tables persist for the entire test session
    # Optionally drop them here if needed:
    # Base.metadata.drop_all(bind=engine)


# ==============================================================================
# Function-scoped fixtures for test isolation
# ==============================================================================


@pytest.fixture
def db_session(engine, tables_created):
    """
    Provide a transactional database session for tests.

    Each test gets its own SAVEPOINT that's rolled back after the test.
    This ensures complete isolation between tests and works with pytest-xdist.

    Uses SAVEPOINT (nested transactions) instead of full transaction rollback
    to avoid connection pool issues with parallel test execution.
    """
    connection = engine.connect()
    transaction = connection.begin()

    # Create a session bound to this specific transaction
    TestSession = sessionmaker(bind=connection, autoflush=False, autocommit=False)
    session = TestSession()

    # Create a SAVEPOINT for test isolation
    nested = connection.begin_nested()

    # If the application code calls session.commit(), it will only commit
    # the SAVEPOINT, not the outer transaction
    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(session, transaction):
        if transaction.nested and not transaction._parent.nested:
            # Restart the SAVEPOINT after each commit
            session.begin_nested()

    yield session

    # Rollback the SAVEPOINT and outer transaction to undo all changes
    session.close()
    if nested.is_active:
        nested.rollback()
    transaction.rollback()
    connection.close()


@pytest.fixture
def test_catalog_db(db_session):
    """
    Create a CatalogDB instance with an injected test session.

    This allows CatalogDB to work with pytest's transactional testing.
    """

    def _create_catalog_db(catalog_path: Path):
        """Factory function to create CatalogDB with injected session."""
        from vam_tools.db import CatalogDB

        # Pass the test session to CatalogDB
        return CatalogDB(catalog_path, session=db_session)

    return _create_catalog_db


@pytest.fixture
def standalone_catalog_db(engine):
    """
    Create a CatalogDB instance with its own connection.

    Use this for tests that need real database operations (like ImageScanner).
    These tests won't have transactional rollback, but will have test isolation
    through unique catalog IDs.

    Note: Ensures tables exist in the test database before creating CatalogDB.
    """
    # Create pgvector extension and ensure tables exist in this worker's database
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)

    def _create_catalog_db(catalog_path: Path):
        """Factory function to create standalone CatalogDB."""
        from vam_tools.db import CatalogDB

        # Don't inject a session - let CatalogDB manage its own
        return CatalogDB(catalog_path)

    return _create_catalog_db


# ==============================================================================
# File and directory fixtures
# ==============================================================================


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


@pytest.fixture(scope="module")
def shared_test_images(tmp_path_factory):
    """
    Create a shared directory with test images used by multiple tests.

    This fixture creates images ONCE for the entire module, making tests
    100x faster than creating images in each test.
    """
    images_dir = tmp_path_factory.mktemp("shared_images")

    # Basic colored images (10x10 for speed)
    Image.new("RGB", (10, 10), color="red").save(images_dir / "red.jpg")
    Image.new("RGB", (10, 10), color="green").save(images_dir / "green.jpg")
    Image.new("RGB", (10, 10), color="blue").save(images_dir / "blue.jpg")
    Image.new("RGB", (10, 10), color="purple").save(images_dir / "purple.jpg")
    Image.new("RGB", (10, 10), color="orange").save(images_dir / "orange.jpg")

    # Gradient images (for similarity testing)
    import numpy as np

    gradient1 = np.arange(0, 100, 10).reshape(10, 1) * np.ones((1, 10))
    Image.fromarray(gradient1.astype("uint8"), mode="L").save(
        images_dir / "gradient1.jpg"
    )

    gradient2 = (np.arange(0, 100, 10).reshape(10, 1) + 10) * np.ones((1, 10))
    gradient2 = np.clip(gradient2, 0, 255)
    Image.fromarray(gradient2.astype("uint8"), mode="L").save(
        images_dir / "gradient2.jpg"
    )

    # Different sizes (for quality testing)
    Image.new("RGB", (10, 10), color="red").save(images_dir / "small.jpg")
    Image.new("RGB", (20, 20), color="red").save(images_dir / "large.jpg")

    return images_dir
