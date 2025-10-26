"""
Pytest configuration and fixtures for vam_tools tests.
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image, ImageDraw

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
