"""
Perceptual hashing for duplicate image detection.

Implements dHash (difference hash) and aHash (average hash) algorithms
to find visually similar images even if they differ in size, format, or compression.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


def dhash(image_path: Path, hash_size: int = 8) -> Optional[str]:
    """
    Calculate difference hash (dHash) for an image.

    dHash is robust to minor variations and works well for finding
    near-duplicates even across different sizes and formats.

    Args:
        image_path: Path to the image file
        hash_size: Size of the hash (default 8 = 64-bit hash)

    Returns:
        Hexadecimal string representation of the hash, or None on error
    """
    try:
        # Load and convert to grayscale
        img = Image.open(image_path).convert("L")

        # Resize to hash_size+1 x hash_size
        # We need +1 width to compute horizontal gradients
        img = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)

        # Convert to list of pixel values
        pixels = list(img.getdata())

        # Calculate horizontal gradient (difference between adjacent pixels)
        diff = []
        for row in range(hash_size):
            row_start = row * (hash_size + 1)
            for col in range(hash_size):
                left_pixel = pixels[row_start + col]
                right_pixel = pixels[row_start + col + 1]
                diff.append(left_pixel < right_pixel)

        # Convert boolean array to hexadecimal
        return _bits_to_hex(diff)

    except Exception as e:
        logger.error(f"Error computing dHash for {image_path}: {e}")
        return None


def ahash(image_path: Path, hash_size: int = 8) -> Optional[str]:
    """
    Calculate average hash (aHash) for an image.

    aHash compares each pixel to the average brightness.
    Less robust than dHash but faster to compute.

    Args:
        image_path: Path to the image file
        hash_size: Size of the hash (default 8 = 64-bit hash)

    Returns:
        Hexadecimal string representation of the hash, or None on error
    """
    try:
        # Load and convert to grayscale
        img = Image.open(image_path).convert("L")

        # Resize to hash_size x hash_size
        img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)

        # Convert to list of pixel values
        pixels = list(img.getdata())

        # Calculate average pixel value
        avg = sum(pixels) / len(pixels)

        # Create hash: 1 if pixel > average, 0 otherwise
        bits = [pixel > avg for pixel in pixels]

        # Convert boolean array to hexadecimal
        return _bits_to_hex(bits)

    except Exception as e:
        logger.error(f"Error computing aHash for {image_path}: {e}")
        return None


def combined_hash(image_path: Path, hash_size: int = 8) -> Optional[Tuple[str, str]]:
    """
    Calculate both dHash and aHash for an image.

    Using both hashes provides better duplicate detection:
    - dHash is better for finding near-duplicates
    - aHash is faster and good for exact duplicates

    Args:
        image_path: Path to the image file
        hash_size: Size of the hash (default 8 = 64-bit hash)

    Returns:
        Tuple of (dhash, ahash) as hex strings, or None on error
    """
    d = dhash(image_path, hash_size)
    a = ahash(image_path, hash_size)

    if d is None or a is None:
        return None

    return (d, a)


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Calculate Hamming distance between two hashes.

    Hamming distance is the number of bit positions where the hashes differ.
    Lower distance = more similar images.

    Similarity thresholds (for 64-bit hashes):
    - 0-5: Very similar (likely duplicates or minor edits)
    - 6-10: Similar (same subject, different crop/exposure)
    - 11-15: Somewhat similar
    - 16+: Likely different images

    Args:
        hash1: First hash (hex string)
        hash2: Second hash (hex string)

    Returns:
        Number of differing bits
    """
    if len(hash1) != len(hash2):
        raise ValueError("Hashes must be the same length")

    # Convert hex strings to integers and XOR them
    h1 = int(hash1, 16)
    h2 = int(hash2, 16)
    xor = h1 ^ h2

    # Count number of 1 bits (differing positions)
    return bin(xor).count("1")


def are_similar(
    hash1: str, hash2: str, threshold: int = 5, use_both: bool = False
) -> bool:
    """
    Check if two hashes represent similar images.

    Args:
        hash1: First hash (hex string)
        hash2: Second hash (hex string)
        threshold: Maximum Hamming distance to consider similar (default 5)
        use_both: If True, requires both dHash and aHash to match

    Returns:
        True if hashes are within threshold, False otherwise
    """
    try:
        distance = hamming_distance(hash1, hash2)
        return distance <= threshold
    except Exception as e:
        logger.error(f"Error comparing hashes: {e}")
        return False


def _bits_to_hex(bits: list) -> str:
    """
    Convert a list of boolean values to a hexadecimal string.

    Args:
        bits: List of boolean values

    Returns:
        Hexadecimal string representation
    """
    # Convert booleans to binary string
    binary_str = "".join("1" if b else "0" for b in bits)

    # Convert binary to hex
    hex_value = hex(int(binary_str, 2))[2:]  # [2:] removes '0x' prefix

    # Pad with zeros to maintain consistent length
    expected_length = (len(bits) + 3) // 4  # Round up to nearest hex digit
    return hex_value.zfill(expected_length)


def similarity_score(hash1: str, hash2: str, hash_size: int = 8) -> float:
    """
    Calculate similarity score between two hashes as a percentage.

    Args:
        hash1: First hash (hex string)
        hash2: Second hash (hex string)
        hash_size: Size of the hash used (for calculating max distance)

    Returns:
        Similarity percentage (0-100, where 100 is identical)
    """
    try:
        distance = hamming_distance(hash1, hash2)
        max_distance = hash_size * hash_size  # Total number of bits
        similarity = (1 - (distance / max_distance)) * 100
        return similarity
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0
