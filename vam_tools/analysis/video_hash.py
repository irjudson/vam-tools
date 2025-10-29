"""
Video perceptual hashing for duplicate video detection.

Uses the videohash library to compute perceptual hashes of videos that are
robust to transcoding, resolution changes, and minor edits. The 64-bit hash
works seamlessly with the existing Hamming distance comparison infrastructure.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compute_video_hash(video_path: Path) -> Optional[str]:
    """
    Compute a perceptual hash for a video file.

    This uses videohash which extracts frames and creates a 64-bit hash that's
    robust to:
    - Video transcoding and re-encoding
    - Resolution changes (upscaling/downscaling)
    - Watermark additions/removals
    - Color modifications
    - Frame rate changes
    - Cropping operations
    - Black bar additions/removals

    Args:
        video_path: Path to the video file

    Returns:
        Hexadecimal hash string (64-bit), or None if hashing fails

    Note:
        Requires FFmpeg to be installed on the system.
        The hash is compatible with Hamming distance comparison (count bit differences).
    """
    try:
        from videohash import VideoHash

        # Create VideoHash object
        vh = VideoHash(path=str(video_path))

        # Get the hash as a hexadecimal string
        hash_hex = vh.hash_hex

        logger.debug(f"Computed video hash for {video_path}: {hash_hex}")
        return hash_hex

    except ImportError as e:
        logger.error(
            f"videohash library not installed. Install with: pip install videohash. Error: {e}"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to compute video hash for {video_path}: {e}")
        return None


def compute_video_hashes(video_path: Path) -> dict[str, Optional[str]]:
    """
    Compute video hash and return in format compatible with image hash storage.

    This returns a dict with 'dhash' key for compatibility with the existing
    catalog structure, even though video hashing uses a different algorithm.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with keys 'dhash', 'ahash', 'whash' where:
        - dhash contains the video hash
        - ahash and whash are None (videos don't support multiple algorithms)

    Example:
        >>> hashes = compute_video_hashes(Path("video.mp4"))
        >>> hashes['dhash']
        '1a2b3c4d5e6f7890'
    """
    video_hash = compute_video_hash(video_path)

    return {
        "dhash": video_hash,
        "ahash": None,  # Videos only support one hash type
        "whash": None,  # Videos only support one hash type
    }


def hamming_distance(hash1: str, hash2: str) -> Optional[int]:
    """
    Calculate Hamming distance between two video hashes.

    The Hamming distance is the number of bit positions where the hashes differ.
    Lower values indicate more similar videos.

    Args:
        hash1: First hexadecimal hash string
        hash2: Second hexadecimal hash string

    Returns:
        Number of differing bits (0-64), or None if hashes are invalid

    Example:
        >>> distance = hamming_distance("1a2b3c4d", "1a2b3c4e")
        >>> distance
        1
    """
    if not hash1 or not hash2:
        return None

    if len(hash1) != len(hash2):
        logger.warning(f"Hash length mismatch: {len(hash1)} vs {len(hash2)}")
        return None

    try:
        # Convert hex strings to integers
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)

        # XOR and count set bits
        xor = int1 ^ int2
        distance = bin(xor).count("1")

        return distance

    except ValueError as e:
        logger.error(f"Invalid hash format: {e}")
        return None


def are_videos_similar(
    hash1: str, hash2: str, threshold: int = 10
) -> tuple[bool, Optional[int]]:
    """
    Determine if two videos are similar based on their hashes.

    Args:
        hash1: First video hash (hex string)
        hash2: Second video hash (hex string)
        threshold: Maximum Hamming distance to consider similar (default: 10)

    Returns:
        Tuple of (is_similar: bool, distance: Optional[int])

    Example:
        >>> is_similar, distance = are_videos_similar("1a2b3c4d", "1a2b3c4e")
        >>> is_similar
        True
        >>> distance
        1
    """
    distance = hamming_distance(hash1, hash2)

    if distance is None:
        return (False, None)

    is_similar = distance <= threshold

    return (is_similar, distance)
