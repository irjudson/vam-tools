"""
File verification and corruption detection.

Uses multiple tools and methods to verify file integrity before
marking files as corrupt.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class VerificationResult:
    """Result of file verification."""

    def __init__(self) -> None:
        self.is_valid = True
        self.is_corrupt = False
        self.methods_tried: List[str] = []
        self.methods_succeeded: List[str] = []
        self.methods_failed: List[str] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_success(self, method: str) -> None:
        """Record successful verification method."""
        self.methods_tried.append(method)
        self.methods_succeeded.append(method)

    def add_failure(self, method: str, error: str) -> None:
        """Record failed verification method."""
        self.methods_tried.append(method)
        self.methods_failed.append(method)
        self.errors.append(f"{method}: {error}")

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def mark_corrupt(self) -> None:
        """Mark file as corrupt after all methods failed."""
        self.is_valid = False
        self.is_corrupt = True

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if self.is_corrupt:
            return f"File is CORRUPT - all {len(self.methods_tried)} verification methods failed"
        elif self.methods_succeeded:
            return f"File is valid - verified with: {', '.join(self.methods_succeeded)}"
        else:
            return "File verification incomplete"


def verify_file_integrity(file_path: Path) -> VerificationResult:
    """
    Comprehensive file integrity verification using multiple methods.

    Args:
        file_path: Path to file to verify

    Returns:
        VerificationResult with detailed findings
    """
    result = VerificationResult()

    # Check 1: Basic file existence and size
    if not _check_basic_file_properties(file_path, result):
        result.mark_corrupt()
        return result

    # Check 2: File extension and format
    file_ext = file_path.suffix.lower()

    # Check 3: Try to read file header
    if not _check_file_header(file_path, result):
        result.add_warning("File header check failed")

    # Check 4: Format-specific verification
    if file_ext in {
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
    }:
        _verify_raw_file(file_path, result)
    elif file_ext in {".jpg", ".jpeg"}:
        _verify_jpeg_file(file_path, result)
    elif file_ext in {".heic", ".heif"}:
        _verify_heic_file(file_path, result)
    elif file_ext in {".tif", ".tiff"}:
        _verify_tiff_file(file_path, result)
    elif file_ext in {".png"}:
        _verify_png_file(file_path, result)
    else:
        result.add_warning(f"No specific verification for {file_ext} format")

    # Final determination
    # Don't count basic_properties as format verification
    format_methods = [m for m in result.methods_succeeded if m != "basic_properties"]

    if not format_methods:
        # No format-specific verification succeeded
        result.mark_corrupt()
        logger.warning(f"File {file_path} failed all format verification methods")
    else:
        logger.debug(
            f"File {file_path} verified successfully with {len(format_methods)} format method(s)"
        )

    return result


def _check_basic_file_properties(file_path: Path, result: VerificationResult) -> bool:
    """Check basic file properties."""
    try:
        if not file_path.exists():
            result.add_failure("file_exists", "File does not exist")
            return False

        if not file_path.is_file():
            result.add_failure("is_file", "Path is not a file")
            return False

        size = file_path.stat().st_size
        if size == 0:
            result.add_failure("file_size", "File is empty (0 bytes)")
            return False

        # Check if file is suspiciously small for its type
        file_ext = file_path.suffix.lower()
        min_size = 1024  # 1KB minimum for most image files
        if file_ext in {".arw", ".cr2", ".cr3", ".nef", ".dng"}:
            min_size = 100 * 1024  # RAW files should be at least 100KB

        if size < min_size:
            result.add_warning(
                f"File size ({size} bytes) is suspiciously small for {file_ext}"
            )

        result.add_success("basic_properties")
        return True

    except Exception as e:
        result.add_failure("basic_properties", str(e))
        return False


def _check_file_header(file_path: Path, result: VerificationResult) -> bool:
    """Check if file has valid header/magic numbers."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)

        if len(header) < 4:
            result.add_failure("header_read", "Cannot read file header")
            return False

        # Check for common magic numbers
        file_ext = file_path.suffix.lower()

        magic_numbers = {
            ".jpg": [b"\xff\xd8\xff"],
            ".jpeg": [b"\xff\xd8\xff"],
            ".png": [b"\x89PNG"],
            ".tif": [b"II*\x00", b"MM\x00*"],  # Little/Big endian TIFF
            ".tiff": [b"II*\x00", b"MM\x00*"],
        }

        if file_ext in magic_numbers:
            valid = any(header.startswith(magic) for magic in magic_numbers[file_ext])
            if valid:
                result.add_success("magic_number")
                return True
            else:
                result.add_failure(
                    "magic_number", f"Invalid magic number for {file_ext}"
                )
                return False

        # For RAW files, magic numbers vary too much - skip this check
        return True

    except Exception as e:
        result.add_failure("header_read", str(e))
        return False


def _verify_raw_file(file_path: Path, result: VerificationResult) -> None:
    """Verify RAW file using multiple methods."""

    # Method 1: Try exiftool validation
    try:
        cmd = ["exiftool", "-validate", "-warning", "-a", str(file_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if proc.returncode == 0:
            output = proc.stdout.lower()
            if "error" in output or "corrupt" in output:
                result.add_failure(
                    "exiftool_validate",
                    f"ExifTool validation failed: {proc.stdout[:200]}",
                )
            else:
                result.add_success("exiftool_validate")
        else:
            result.add_failure(
                "exiftool_validate", f"ExifTool returned error code {proc.returncode}"
            )

    except FileNotFoundError:
        result.add_warning("ExifTool not available for validation")
    except Exception as e:
        result.add_failure("exiftool_validate", str(e))

    # Method 2: Try rawpy
    try:
        import rawpy  # type: ignore[import-untyped]

        with rawpy.imread(str(file_path)) as raw:
            # Try to access basic properties
            _ = raw.raw_image.shape
            _ = raw.sizes
            result.add_success("rawpy_read")

    except ImportError:
        result.add_warning("rawpy not available")
    except Exception as e:
        result.add_failure("rawpy_read", str(e))

    # Method 3: Try dcraw
    try:
        cmd = ["dcraw", "-i", str(file_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        if proc.returncode == 0:
            result.add_success("dcraw_identify")
        else:
            result.add_failure(
                "dcraw_identify", f"dcraw returned error code {proc.returncode}"
            )

    except FileNotFoundError:
        result.add_warning("dcraw not available")
    except Exception as e:
        result.add_failure("dcraw_identify", str(e))

    # Method 4: Try exiftool metadata extraction
    try:
        cmd = ["exiftool", "-j", "-Model", "-Make", str(file_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if proc.returncode == 0 and len(proc.stdout) > 10:
            result.add_success("exiftool_metadata")
        else:
            result.add_failure("exiftool_metadata", "Cannot extract metadata")

    except Exception as e:
        result.add_failure("exiftool_metadata", str(e))


def _verify_jpeg_file(file_path: Path, result: VerificationResult) -> None:
    """Verify JPEG file."""
    try:
        from PIL import Image

        img = Image.open(file_path)
        img.verify()  # Verify image integrity
        result.add_success("pil_verify")

        # Also try to load the image data
        img = Image.open(file_path)
        img.load()
        result.add_success("pil_load")

    except ImportError:
        result.add_warning("PIL not available")
    except Exception as e:
        result.add_failure("pil_verify", str(e))


def _verify_heic_file(file_path: Path, result: VerificationResult) -> None:
    """Verify HEIC/HEIF file."""
    try:
        import pillow_heif
        from PIL import Image

        pillow_heif.register_heif_opener()
        img = Image.open(file_path)
        img.verify()
        result.add_success("heic_verify")

        img = Image.open(file_path)
        img.load()
        result.add_success("heic_load")

    except ImportError:
        result.add_warning("HEIC support not available")
    except Exception as e:
        result.add_failure("heic_verify", str(e))


def _verify_tiff_file(file_path: Path, result: VerificationResult) -> None:
    """Verify TIFF file."""
    try:
        from PIL import Image

        img = Image.open(file_path)
        img.verify()
        result.add_success("tiff_verify")

        img = Image.open(file_path)
        img.load()
        result.add_success("tiff_load")

    except ImportError:
        result.add_warning("PIL not available")
    except Exception as e:
        result.add_failure("tiff_verify", str(e))


def _verify_png_file(file_path: Path, result: VerificationResult) -> None:
    """Verify PNG file."""
    try:
        from PIL import Image

        img = Image.open(file_path)
        img.verify()
        result.add_success("png_verify")

        img = Image.open(file_path)
        img.load()
        result.add_success("png_load")

    except ImportError:
        result.add_warning("PIL not available")
    except Exception as e:
        result.add_failure("png_verify", str(e))


def get_corruption_report(file_path: Path) -> Dict:
    """
    Get detailed corruption report for a file.

    Args:
        file_path: Path to file to check

    Returns:
        Dictionary with verification details
    """
    result = verify_file_integrity(file_path)

    return {
        "file": str(file_path),
        "is_valid": result.is_valid,
        "is_corrupt": result.is_corrupt,
        "methods_tried": result.methods_tried,
        "methods_succeeded": result.methods_succeeded,
        "methods_failed": result.methods_failed,
        "errors": result.errors,
        "warnings": result.warnings,
        "summary": result.get_summary(),
    }
