"""
Perceptual hashing for duplicate image detection.

Implements multiple perceptual hashing algorithms:
- dHash (difference hash): Gradient-based, robust to minor variations
- aHash (average hash): Mean-based, faster but less robust
- wHash (wavelet hash): DWT-based, most robust to transformations

Each algorithm finds visually similar images even if they differ in size,
format, compression, or have been edited.
"""

import io
import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pywt
from PIL import Image, ImageFile

# Increase PIL image size limit for large scanned images
# Default is ~179M pixels, but scanned documents can legitimately exceed this
# Set to None to disable the limit for trusted user files
Image.MAX_IMAGE_PIXELS = None

# Allow loading of truncated/partially corrupted images
# Better to get partial hash than skip the image entirely
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


# Corruption tracking
class CorruptionSeverity(str, Enum):
    """Severity levels for file corruption issues."""

    MINOR = "minor"  # Truncated but mostly readable
    MODERATE = "moderate"  # Broken data stream or partial corruption
    SEVERE = "severe"  # Completely unreadable or unsupported format


class CorruptionTracker:
    """Tracks corrupted/problematic files during hash computation."""

    def __init__(self) -> None:
        self.corrupted_files: List[Dict[str, Any]] = []

    def add(
        self,
        file_path: Path,
        error_message: str,
        severity: CorruptionSeverity,
        operation: str = "hashing",
    ) -> None:
        """Record a corrupted file."""
        self.corrupted_files.append(
            {
                "path": str(file_path),
                "error": error_message,
                "severity": severity,
                "operation": operation,
            }
        )

    def get_report(self) -> Dict[str, Any]:
        """Generate a corruption report."""
        if not self.corrupted_files:
            return {"total": 0, "by_severity": {}, "files": []}

        by_severity: Dict[CorruptionSeverity, Dict[str, Any]] = {}
        for severity in CorruptionSeverity:
            files = [f for f in self.corrupted_files if f["severity"] == severity]
            if files:
                by_severity[severity] = {"count": len(files), "files": files}

        return {
            "total": len(self.corrupted_files),
            "by_severity": by_severity,
            "files": self.corrupted_files,
        }

    def save_report(self, output_path: Path) -> None:
        """Save corruption report to a JSON file."""
        import json

        report = self.get_report()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Corruption report saved to {output_path}")

    def clear(self) -> None:
        """Clear all tracked corruption records."""
        self.corrupted_files.clear()


# Global corruption tracker instance
_corruption_tracker = CorruptionTracker()


class HashMethod(str, Enum):
    """Available perceptual hash methods."""

    DHASH = "dhash"  # Difference hash (gradient-based)
    AHASH = "ahash"  # Average hash (mean-based)
    WHASH = "whash"  # Wavelet hash (DWT-based)
    ALL = "all"  # Compute all methods


def _load_image_for_hashing(image_path: Path) -> Optional[Image.Image]:
    """
    Load an image in a format-aware way for hash computation.

    For RAW files, converts the full RAW data (not just preview) to ensure
    accurate duplicate detection and quality assessment.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object, or None if unable to load
    """
    file_ext = image_path.suffix.lower()

    # RAW formats need full conversion
    raw_formats = {
        ".arw",
        ".cr2",
        ".cr3",
        ".nef",
        ".dng",
        ".orf",
        ".rw2",
        ".pef",
        ".sr2",
        ".raf",
        ".raw",
    }

    # HEIC/TIFF can be converted with Pillow
    pillow_convertible = {".heic", ".heif", ".tif", ".tiff"}

    try:
        # Handle RAW files - convert full RAW data (not preview)
        if file_ext in raw_formats:
            # Try rawpy first (Python library for LibRaw)
            try:
                import rawpy

                with rawpy.imread(str(image_path)) as raw:
                    # Convert to RGB using default settings
                    # This processes the actual RAW data, not the preview
                    rgb = raw.postprocess(
                        use_camera_wb=True,  # Use camera white balance
                        half_size=True,  # Faster processing, still high quality
                        no_auto_bright=True,  # Don't auto-adjust brightness
                        output_bps=8,  # 8-bit output
                    )
                    return Image.fromarray(rgb)

            except ImportError:
                # rawpy not available, try dcraw
                logger.debug("rawpy not available, trying dcraw for RAW conversion")
                try:
                    # Use dcraw to convert RAW to PPM format
                    result = subprocess.run(
                        [
                            "dcraw",
                            "-c",  # Write to stdout
                            "-w",  # Use camera white balance
                            "-h",  # Half-size (faster, still good quality)
                            "-q",
                            "0",  # Highest quality interpolation
                            str(image_path),
                        ],
                        capture_output=True,
                        timeout=30,  # RAW conversion can take longer
                    )

                    if result.returncode == 0 and len(result.stdout) > 0:
                        return Image.open(io.BytesIO(result.stdout))

                    # Track RAW conversion failure
                    error_msg = (
                        result.stderr.decode() if result.stderr else "dcraw failed"
                    )
                    _corruption_tracker.add(
                        image_path,
                        error_msg,
                        CorruptionSeverity.SEVERE,
                        "RAW conversion",
                    )
                    logger.warning(f"dcraw failed for {image_path}: {error_msg}")
                    return None

                except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                    _corruption_tracker.add(
                        image_path,
                        str(e),
                        CorruptionSeverity.SEVERE,
                        "RAW conversion",
                    )
                    logger.warning(
                        f"RAW conversion failed for {image_path}: {e}. "
                        "Install rawpy or dcraw for RAW support."
                    )
                    return None

            except Exception as e:
                error_str = str(e)
                # Categorize RAW errors
                if "not RAW file" in error_str or "Unsupported" in error_str:
                    severity = CorruptionSeverity.SEVERE
                else:
                    severity = CorruptionSeverity.MODERATE

                _corruption_tracker.add(
                    image_path, error_str, severity, "RAW conversion"
                )
                logger.error(f"Error converting RAW file {image_path}: {e}")
                return None

        # Handle HEIC/TIFF with Pillow conversion
        elif file_ext in pillow_convertible:
            img = Image.open(image_path)
            # Convert to RGB if needed
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            return img

        # Standard formats (JPEG, PNG, GIF, WebP)
        else:
            return Image.open(image_path)

    except Exception as e:
        error_str = str(e)

        # Categorize error severity based on error message
        if "truncated" in error_str.lower():
            severity = CorruptionSeverity.MINOR
            log_level = logging.WARNING
        elif "broken data stream" in error_str.lower():
            severity = CorruptionSeverity.MODERATE
            log_level = logging.WARNING
        else:
            severity = CorruptionSeverity.SEVERE
            log_level = logging.ERROR

        _corruption_tracker.add(image_path, error_str, severity, "image loading")
        logger.log(log_level, f"Error loading image {image_path} for hashing: {e}")
        return None


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
        # Load image (handles RAW/HEIC/TIFF conversion)
        img = _load_image_for_hashing(image_path)
        if img is None:
            # Error already logged and tracked by _load_image_for_hashing
            return None

        # Convert to grayscale
        img = img.convert("L")

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
        # Computation error after successful load (rare)
        error_str = str(e)
        _corruption_tracker.add(
            image_path,
            f"dHash computation: {error_str}",
            CorruptionSeverity.MODERATE,
            "hashing",
        )
        logger.warning(f"Error computing dHash for {image_path}: {e}")
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
        # Load image (handles RAW/HEIC/TIFF conversion)
        img = _load_image_for_hashing(image_path)
        if img is None:
            # Error already logged and tracked by _load_image_for_hashing
            return None

        # Convert to grayscale
        img = img.convert("L")

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
        # Computation error after successful load (rare)
        error_str = str(e)
        _corruption_tracker.add(
            image_path,
            f"aHash computation: {error_str}",
            CorruptionSeverity.MODERATE,
            "hashing",
        )
        logger.warning(f"Error computing aHash for {image_path}: {e}")
        return None


def whash(image_path: Path, hash_size: int = 8, mode: str = "haar") -> Optional[str]:
    """
    Calculate wavelet hash (wHash) for an image.

    wHash uses Discrete Wavelet Transform (DWT) which makes it highly robust to:
    - Image scaling and rotation
    - Lighting/gamma corrections
    - Compression artifacts
    - Color adjustments
    - Cropping (to some extent)

    This is the most robust hash method but also the slowest to compute.

    Args:
        image_path: Path to the image file
        hash_size: Size of the hash (default 8 = 64-bit hash)
        mode: Wavelet mode ('haar', 'db1', etc.) - 'haar' is fastest

    Returns:
        Hexadecimal string representation of the hash, or None on error
    """
    try:
        # Load image (handles RAW/HEIC/TIFF conversion)
        img = _load_image_for_hashing(image_path)
        if img is None:
            # Error already logged and tracked by _load_image_for_hashing
            return None

        # Convert to grayscale
        img = img.convert("L")

        # Resize to hash_size x hash_size
        img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)

        # Convert to numpy array (normalized to 0-1)
        pixels = np.array(img, dtype=np.float32) / 255.0

        # Apply 2D Discrete Wavelet Transform
        # This decomposes the image into LL (approximation), LH, HL, HH (details)
        coeffs = pywt.dwt2(pixels, mode)
        ll, (lh, hl, hh) = coeffs

        # Use only the LL (low-frequency) component for the hash
        # This component is most robust to image transformations
        ll_flat = ll.flatten()

        # Calculate median of LL coefficients
        median = np.median(ll_flat)

        # Create hash: 1 if coefficient > median, 0 otherwise
        # Take only hash_size*hash_size coefficients
        num_coeffs = min(len(ll_flat), hash_size * hash_size)
        bits = [ll_flat[i] > median for i in range(num_coeffs)]

        # Pad if necessary
        while len(bits) < hash_size * hash_size:
            bits.append(False)

        # Convert boolean array to hexadecimal
        return _bits_to_hex(bits)

    except Exception as e:
        # Computation error after successful load (rare)
        error_str = str(e)
        _corruption_tracker.add(
            image_path,
            f"wHash computation: {error_str}",
            CorruptionSeverity.MODERATE,
            "hashing",
        )
        logger.warning(f"Error computing wHash for {image_path}: {e}")
        return None


def combined_hash(
    image_path: Path, hash_size: int = 8, methods: Optional[List[HashMethod]] = None
) -> Optional[Dict[str, Optional[str]]]:
    """
    Calculate multiple perceptual hashes for an image.

    Using multiple hash methods provides the best duplicate detection:
    - dHash: Best for finding near-duplicates (gradient-based)
    - aHash: Fastest, good for exact duplicates (mean-based)
    - wHash: Most robust to transformations (wavelet-based)

    Args:
        image_path: Path to the image file
        hash_size: Size of the hash (default 8 = 64-bit hash)
        methods: List of hash methods to compute (default: all methods)

    Returns:
        Dictionary mapping method name to hash hex string, or None on error
    """
    if methods is None:
        methods = [HashMethod.DHASH, HashMethod.AHASH, HashMethod.WHASH]
    elif HashMethod.ALL in methods:
        methods = [HashMethod.DHASH, HashMethod.AHASH, HashMethod.WHASH]

    hashes = {}

    for method in methods:
        if method == HashMethod.DHASH:
            h = dhash(image_path, hash_size)
            if h:
                hashes["dhash"] = h
        elif method == HashMethod.AHASH:
            h = ahash(image_path, hash_size)
            if h:
                hashes["ahash"] = h
        elif method == HashMethod.WHASH:
            h = whash(image_path, hash_size)
            if h:
                hashes["whash"] = h

    return hashes if hashes else None  # type: ignore[return-value]


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


def compare_hashes(
    hashes1: Dict[str, str],
    hashes2: Dict[str, str],
    threshold: int = 5,
    require_all: bool = False,
) -> bool:
    """
    Compare multiple perceptual hashes between two images.

    Args:
        hashes1: Dictionary of hash method -> hash for first image
        hashes2: Dictionary of hash method -> hash for second image
        threshold: Maximum Hamming distance to consider similar (default 5)
        require_all: If True, all common hash methods must match.
                    If False, any matching method is sufficient (default)

    Returns:
        True if images are similar according to the criteria, False otherwise

    Examples:
        >>> hashes1 = {"dhash": "abc123", "whash": "def456"}
        >>> hashes2 = {"dhash": "abc124", "whash": "def999"}
        >>> # dhash matches (distance=1), whash doesn't match
        >>> compare_hashes(hashes1, hashes2, threshold=5, require_all=False)
        True  # At least one matches
        >>> compare_hashes(hashes1, hashes2, threshold=5, require_all=True)
        False  # Not all match
    """
    # Find common hash methods
    common_methods = set(hashes1.keys()) & set(hashes2.keys())

    if not common_methods:
        return False

    matches = []
    for method in common_methods:
        try:
            distance = hamming_distance(hashes1[method], hashes2[method])
            matches.append(distance <= threshold)
        except Exception as e:
            logger.error(f"Error comparing {method} hashes: {e}")
            matches.append(False)

    if require_all:
        return all(matches)
    else:
        return any(matches)


def get_best_matches(
    hashes1: Dict[str, str], hashes2: Dict[str, str]
) -> Dict[str, Tuple[int, float]]:
    """
    Compare multiple hash methods and return detailed similarity metrics.

    Args:
        hashes1: Dictionary of hash method -> hash for first image
        hashes2: Dictionary of hash method -> hash for second image

    Returns:
        Dictionary mapping method name to (hamming_distance, similarity_percentage)

    Example:
        >>> hashes1 = {"dhash": "abc123", "whash": "def456"}
        >>> hashes2 = {"dhash": "abc124", "whash": "def999"}
        >>> get_best_matches(hashes1, hashes2)
        {'dhash': (1, 98.4), 'whash': (12, 81.2)}
    """
    results = {}
    common_methods = set(hashes1.keys()) & set(hashes2.keys())

    for method in common_methods:
        try:
            distance = hamming_distance(hashes1[method], hashes2[method])
            similarity = similarity_score(hashes1[method], hashes2[method])
            results[method] = (distance, similarity)
        except Exception as e:
            logger.error(f"Error comparing {method} hashes: {e}")

    return results


def get_recommended_threshold(method: HashMethod) -> int:
    """
    Get recommended similarity threshold for a hash method.

    Different hash methods have different sensitivity levels.

    Args:
        method: The hash method

    Returns:
        Recommended Hamming distance threshold
    """
    thresholds = {
        HashMethod.DHASH: 5,  # Good balance for gradient-based
        HashMethod.AHASH: 6,  # Slightly higher for mean-based
        HashMethod.WHASH: 4,  # Lower for wavelet (more robust)
    }
    return thresholds.get(method, 5)


def compute_hashes_batch(
    image_paths: List[Path],
    use_gpu: bool = False,
    batch_size: Optional[int] = None,
) -> List[Optional[Dict[str, Optional[str]]]]:
    """
    Compute perceptual hashes for multiple images efficiently.

    Uses GPU acceleration if available and enabled, otherwise processes
    on CPU.

    Args:
        image_paths: List of image file paths to process
        use_gpu: Whether to use GPU acceleration if available
        batch_size: Number of images to process in parallel (None = auto)

    Returns:
        List of hash dictionaries, one per image

    Example:
        >>> paths = [Path("img1.jpg"), Path("img2.jpg")]
        >>> hashes = compute_hashes_batch(paths, use_gpu=True)
        >>> len(hashes)
        2
        >>> hashes[0].keys()
        dict_keys(['dhash', 'ahash', 'whash'])
    """
    if use_gpu:
        try:
            from vam_tools.analysis.gpu_hash import GPUHashProcessor

            processor = GPUHashProcessor(batch_size=batch_size, enable_gpu=True)
            if processor.use_gpu:
                logger.info(f"Using GPU batch processing for {len(image_paths)} images")
                return processor.process_images(image_paths)  # type: ignore[return-value]
            else:
                logger.info("GPU not available, using CPU batch processing")
        except ImportError:
            logger.warning("GPU modules not available, using CPU batch processing")

    # CPU fallback - process sequentially
    logger.info(f"Using CPU batch processing for {len(image_paths)} images")
    return [combined_hash(path) for path in image_paths]


# Public corruption tracking API


def get_corruption_report() -> Dict[str, Any]:
    """
    Get a report of all corrupted/problematic files encountered during hashing.

    Returns:
        Dictionary with corruption statistics and file details:
        {
            "total": int,
            "by_severity": {
                "minor": {"count": int, "files": [...]},
                "moderate": {"count": int, "files": [...]},
                "severe": {"count": int, "files": [...]}
            },
            "files": [
                {
                    "path": str,
                    "error": str,
                    "severity": str,
                    "operation": str
                },
                ...
            ]
        }

    Example:
        >>> # After running hash operations
        >>> report = get_corruption_report()
        >>> print(f"Found {report['total']} corrupted files")
        >>> if 'severe' in report['by_severity']:
        ...     print(f"Severe: {report['by_severity']['severe']['count']}")
    """
    return _corruption_tracker.get_report()


def save_corruption_report(output_path: Path) -> None:
    """
    Save corruption report to a JSON file.

    Args:
        output_path: Path where the JSON report should be saved

    Example:
        >>> save_corruption_report(Path("corruption_report.json"))
    """
    _corruption_tracker.save_report(output_path)


def clear_corruption_tracking() -> None:
    """
    Clear all tracked corruption records.

    Useful when processing multiple batches and you want separate reports.

    Example:
        >>> # Process batch 1
        >>> hashes1 = compute_hashes_batch(batch1_paths)
        >>> report1 = get_corruption_report()
        >>> clear_corruption_tracking()
        >>> # Process batch 2
        >>> hashes2 = compute_hashes_batch(batch2_paths)
        >>> report2 = get_corruption_report()
    """
    _corruption_tracker.clear()


def get_corruption_summary() -> str:
    """
    Get a human-readable summary of corruption issues.

    Returns:
        Multi-line string summarizing corruption statistics

    Example:
        >>> print(get_corruption_summary())
        Corruption Summary:
        ------------------
        Total corrupted files: 5

        By severity:
          - Minor (truncated): 2 files
          - Moderate (broken data): 2 files
          - Severe (unreadable): 1 file
    """
    report = _corruption_tracker.get_report()

    if report["total"] == 0:
        return "No corruption detected."

    lines = [
        "Corruption Summary:",
        "------------------",
        f"Total corrupted files: {report['total']}",
        "",
        "By severity:",
    ]

    severity_desc = {
        CorruptionSeverity.MINOR: "truncated",
        CorruptionSeverity.MODERATE: "broken data",
        CorruptionSeverity.SEVERE: "unreadable",
    }

    by_severity: Dict[CorruptionSeverity, Dict[str, Any]] = report["by_severity"]
    for severity, desc in severity_desc.items():
        if severity in by_severity:
            count = by_severity[severity]["count"]
            lines.append(f"  - {severity.value.title()} ({desc}): {count} files")

    return "\n".join(lines)
