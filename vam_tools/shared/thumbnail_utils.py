"""
Thumbnail generation utilities for VAM Tools.

Handles thumbnail creation for images and videos, supporting various formats
including HEIC, RAW, and common video formats.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Default thumbnail size
DEFAULT_THUMBNAIL_SIZE = (200, 200)
DEFAULT_THUMBNAIL_QUALITY = 85


def generate_thumbnail(
    source_path: Path,
    output_path: Path,
    size: Tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
    quality: int = DEFAULT_THUMBNAIL_QUALITY,
) -> bool:
    """
    Generate a thumbnail for an image or video file.

    Args:
        source_path: Path to the source image/video file
        output_path: Path where thumbnail should be saved (must end in .jpg)
        size: Thumbnail size as (width, height) tuple
        quality: JPEG quality (1-100, default 85)

    Returns:
        True if thumbnail was generated successfully, False otherwise

    The thumbnail will maintain aspect ratio and fit within the specified size.
    For videos, the first frame is extracted.
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if source exists
        if not source_path.exists():
            logger.error(f"Source file not found: {source_path}")
            return False

        # Try to open the image
        img: Optional[Image.Image] = None
        try:
            img = Image.open(source_path)
        except Exception as e:
            # If PIL can't open it, might be a video - try extracting first frame
            if is_video_file(source_path):
                img = extract_video_thumbnail(source_path)
                if img is None:
                    logger.error(f"Failed to extract video thumbnail: {source_path}")
                    return False
            else:
                logger.error(f"Failed to open image {source_path}: {e}")
                return False

        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")

        # Generate thumbnail (maintains aspect ratio)
        img.thumbnail(size, Image.Resampling.LANCZOS)

        # Save as JPEG
        img.save(output_path, "JPEG", quality=quality, optimize=True)

        logger.debug(f"Generated thumbnail: {output_path} (from {source_path})")
        return True

    except Exception as e:
        logger.error(f"Error generating thumbnail for {source_path}: {e}")
        return False


def extract_video_thumbnail(video_path: Path) -> Optional[Image.Image]:
    """
    Extract the first frame from a video file as a PIL Image.

    Args:
        video_path: Path to the video file

    Returns:
        PIL Image object of the first frame, or None if extraction fails

    Uses Pillow's video capabilities if available, otherwise returns None.
    For production use, consider using ffmpeg or opencv for better video support.
    """
    try:
        # Try using Pillow's built-in video support (limited)
        # For more robust video support, use ffmpeg via subprocess
        from PIL import ImageSequence

        with Image.open(video_path) as img:
            # Get the first frame
            frame = next(ImageSequence.Iterator(img))
            return frame.copy()

    except Exception as e:
        logger.debug(f"Pillow video extraction failed for {video_path}: {e}")

        # Fallback: try using ffmpeg if available
        try:
            import subprocess
            import tempfile

            # Create temporary file for extracted frame
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            # Extract first frame using ffmpeg
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",
                    str(tmp_path),
                ],
                capture_output=True,
                timeout=10,
            )

            if result.returncode == 0 and tmp_path.exists():
                img = Image.open(tmp_path)
                frame = img.copy()
                img.close()
                tmp_path.unlink()
                return frame
            else:
                if tmp_path.exists():
                    tmp_path.unlink()
                logger.error(
                    f"ffmpeg extraction failed for {video_path}: {result.stderr.decode()}"
                )
                return None

        except Exception as ffmpeg_error:
            logger.debug(f"ffmpeg fallback failed for {video_path}: {ffmpeg_error}")
            return None


def is_video_file(path: Path) -> bool:
    """
    Check if a file is a video based on extension.

    Args:
        path: Path to the file

    Returns:
        True if the file extension indicates a video format
    """
    video_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".m4v",
        ".mpg",
        ".mpeg",
        ".wmv",
        ".flv",
        ".webm",
    }
    return path.suffix.lower() in video_extensions


def get_thumbnail_path(
    image_id: str, thumbnails_dir: Path, create_dir: bool = True
) -> Path:
    """
    Get the path where a thumbnail should be stored for a given image ID.

    Args:
        image_id: Image ID (checksum)
        thumbnails_dir: Base thumbnails directory (catalog/thumbnails/)
        create_dir: Whether to create the directory if it doesn't exist

    Returns:
        Path object for the thumbnail file
    """
    if create_dir:
        thumbnails_dir.mkdir(parents=True, exist_ok=True)

    return thumbnails_dir / f"{image_id}.jpg"


def thumbnail_exists(image_id: str, thumbnails_dir: Path) -> bool:
    """
    Check if a thumbnail already exists for a given image ID.

    Args:
        image_id: Image ID (checksum)
        thumbnails_dir: Base thumbnails directory

    Returns:
        True if thumbnail file exists
    """
    thumb_path = get_thumbnail_path(image_id, thumbnails_dir, create_dir=False)
    return thumb_path.exists()
