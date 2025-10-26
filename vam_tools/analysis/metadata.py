"""
Metadata extraction from images and videos.

Extracts EXIF data, dates, resolution, format, and other metadata.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import arrow
import exiftool
from PIL import Image

from ..core.types import DateInfo, FileType, ImageMetadata

logger = logging.getLogger(__name__)

# Register HEIC support for Pillow
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    logger.debug("HEIC support registered")
except ImportError:
    logger.warning(
        "pillow-heif not installed, HEIC files may not be processed correctly"
    )


class MetadataExtractor:
    """Extract comprehensive metadata from images and videos."""

    def __init__(self) -> None:
        """Initialize the metadata extractor."""
        self.exif_tool: Optional[exiftool.ExifToolHelper] = None

    def __enter__(self) -> "MetadataExtractor":
        """Context manager entry."""
        self.exif_tool = exiftool.ExifToolHelper()
        self.exif_tool.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        if self.exif_tool:
            self.exif_tool.__exit__(*args)

    def extract_metadata(self, file_path: Path, file_type: FileType) -> ImageMetadata:
        """
        Extract all metadata from a file.

        Args:
            file_path: Path to the image/video file
            file_type: Type of file (image or video)

        Returns:
            ImageMetadata object with extracted data
        """
        metadata = ImageMetadata()

        try:
            # Get file size
            metadata.size_bytes = os.path.getsize(file_path)

            # Extract EXIF data using ExifTool
            if self.exif_tool:
                exif_data = self._extract_exif(file_path)
                metadata.exif = exif_data

                # Extract camera information
                metadata.camera_make = exif_data.get("Make")
                metadata.camera_model = exif_data.get("Model")
                metadata.lens_model = exif_data.get("LensModel")

                # Extract camera settings
                metadata.focal_length = self._parse_float(exif_data.get("FocalLength"))
                metadata.aperture = self._parse_float(exif_data.get("FNumber"))
                metadata.shutter_speed = exif_data.get("ShutterSpeed") or exif_data.get(
                    "ExposureTime"
                )
                metadata.iso = self._parse_int(exif_data.get("ISO"))

                # Extract GPS information
                metadata.gps_latitude = self._parse_float(exif_data.get("GPSLatitude"))
                metadata.gps_longitude = self._parse_float(
                    exif_data.get("GPSLongitude")
                )

            # Get format and resolution
            if file_type == FileType.IMAGE:
                format_info = self._get_image_format(file_path)
                metadata.format = format_info[0]
                metadata.resolution = format_info[1]
                if format_info[1]:
                    metadata.width, metadata.height = format_info[1]
            elif file_type == FileType.VIDEO:
                metadata.format = self._get_video_format(file_path, exif_data)
                # Extract video resolution from EXIF
                resolution = self._get_video_resolution(exif_data)
                if resolution:
                    metadata.resolution = resolution
                    metadata.width, metadata.height = resolution

        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")

        return metadata

    def extract_dates(self, file_path: Path, metadata: ImageMetadata) -> DateInfo:
        """
        Extract all possible dates from a file.

        Args:
            file_path: Path to the file
            metadata: Extracted metadata containing EXIF data

        Returns:
            DateInfo object with all found dates
        """
        date_info = DateInfo()

        # Extract EXIF dates
        date_info.exif_dates = self._extract_exif_dates(metadata.exif)

        # Extract date from filename
        date_info.filename_date = self._extract_filename_date(file_path)

        # Extract date from directory structure
        date_info.directory_date = self._extract_directory_date(file_path)

        # Get filesystem dates
        try:
            stat = os.stat(file_path)
            date_info.filesystem_created = datetime.fromtimestamp(stat.st_ctime)
            date_info.filesystem_modified = datetime.fromtimestamp(stat.st_mtime)
        except Exception as e:
            logger.debug(f"Error getting filesystem dates: {e}")

        # Select the best date
        self._select_best_date(date_info)

        return date_info

    def _extract_exif(self, file_path: Path) -> Dict[str, Any]:
        """Extract EXIF data using ExifTool."""
        if not self.exif_tool:
            return {}

        try:
            metadata_list = self.exif_tool.get_metadata([str(file_path)])
            if metadata_list:
                return metadata_list[0]
        except Exception as e:
            logger.debug(f"Error extracting EXIF from {file_path}: {e}")

        return {}

    def _extract_exif_dates(
        self, exif: Dict[str, Any]
    ) -> Dict[str, Optional[datetime]]:
        """Extract date fields from EXIF data."""
        dates = {}

        # Common EXIF date fields
        date_fields = [
            "DateTimeOriginal",
            "CreateDate",
            "ModifyDate",
            "DateCreated",
            "DateTime",
            "FileModifyDate",
            "MediaCreateDate",
            "TrackCreateDate",
        ]

        for field in date_fields:
            if field in exif:
                try:
                    date_str = str(exif[field])
                    # Try to parse the date
                    parsed_date = self._parse_exif_date(date_str)
                    if parsed_date:
                        dates[field] = parsed_date
                except Exception as e:
                    logger.debug(f"Error parsing date {field}: {e}")

        return dates

    def _parse_exif_date(self, date_str: str) -> Optional[datetime]:
        """Parse EXIF date string to datetime."""
        # EXIF formats
        formats = [
            "YYYY:MM:DD HH:mm:ssZZ",
            "YYYY:MM:DD HH:mm:ss",
            "YYYY-MM-DD HH:mm:ss",
            "YYYY:MM:DD",
            "YYYY-MM-DD",
        ]

        for fmt in formats:
            try:
                parsed = arrow.get(date_str, fmt, normalize_whitespace=True)
                return parsed.datetime
            except (arrow.ParserError, ValueError):
                continue

        return None

    def _extract_filename_date(self, file_path: Path) -> Optional[datetime]:
        """Extract date from filename patterns."""
        filename = file_path.stem

        # Common date patterns
        patterns = [
            (
                r"(\d{4})-(\d{2})-(\d{2})[_\s](\d{2}):?(\d{2}):?(\d{2})",
                True,
            ),  # YYYY-MM-DD HH:MM:SS
            (r"(\d{4})-(\d{2})-(\d{2})", True),  # YYYY-MM-DD
            (r"(\d{4})_(\d{2})_(\d{2})", True),  # YYYY_MM_DD
            (r"(\d{4})(\d{2})(\d{2})", True),  # YYYYMMDD
            (r"(\d{2})-(\d{2})-(\d{4})", False),  # MM-DD-YYYY
        ]

        for pattern, is_year_first in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        # Date only
                        if is_year_first:
                            year, month, day = groups
                        else:
                            month, day, year = groups
                        return datetime(int(year), int(month), int(day))
                    elif len(groups) == 6:
                        # Date and time
                        year, month, day, hour, minute, second = groups
                        return datetime(
                            int(year),
                            int(month),
                            int(day),
                            int(hour),
                            int(minute),
                            int(second),
                        )
                except (ValueError, TypeError):
                    continue

        return None

    def _extract_directory_date(self, file_path: Path) -> Optional[str]:
        """Extract date pattern from directory structure."""
        path_parts = file_path.parts

        # Look for YYYY-MM pattern
        for part in reversed(path_parts):
            match = re.search(r"(\d{4})-(\d{2})", part)
            if match:
                year, month = match.groups()
                if 1900 <= int(year) <= 2100 and 1 <= int(month) <= 12:
                    return f"{year}-{month}"

            # Look for just year
            match = re.search(r"(\d{4})", part)
            if match:
                year = match.group(1)
                if 1900 <= int(year) <= 2100:
                    return year

        return None

    def _select_best_date(self, date_info: DateInfo) -> None:
        """Select the best date from all available sources."""
        # Priority order: EXIF > filename > directory > filesystem

        # Try EXIF dates (prefer DateTimeOriginal)
        priority_fields = ["DateTimeOriginal", "CreateDate", "MediaCreateDate"]
        for field in priority_fields:
            if field in date_info.exif_dates and date_info.exif_dates[field]:
                date_info.selected_date = date_info.exif_dates[field]
                date_info.selected_source = f"exif:{field}"
                date_info.confidence = 95
                break

        # If no EXIF date, try any EXIF date
        if not date_info.selected_date and date_info.exif_dates:
            for field, date in date_info.exif_dates.items():
                if date:
                    date_info.selected_date = date
                    date_info.selected_source = f"exif:{field}"
                    date_info.confidence = 85
                    break

        # Try filename date
        if not date_info.selected_date and date_info.filename_date:
            date_info.selected_date = date_info.filename_date
            date_info.selected_source = "filename"
            date_info.confidence = 70

        # Try directory date (partial date)
        if not date_info.selected_date and date_info.directory_date:
            # Parse directory date (might be just YYYY or YYYY-MM)
            try:
                if len(date_info.directory_date) == 4:  # Just year
                    date_info.selected_date = datetime(
                        int(date_info.directory_date), 1, 1
                    )
                else:  # YYYY-MM
                    year, month = date_info.directory_date.split("-")
                    date_info.selected_date = datetime(int(year), int(month), 1)
                date_info.selected_source = "directory"
                date_info.confidence = 50
            except ValueError:
                pass

        # Fall back to filesystem date
        if not date_info.selected_date:
            if date_info.filesystem_created:
                date_info.selected_date = date_info.filesystem_created
                date_info.selected_source = "filesystem"
                date_info.confidence = 30

        # Check for suspicious dates
        if date_info.selected_date:
            now = datetime.now()
            # Future date
            if date_info.selected_date > now:
                date_info.suspicious = True
                logger.warning(f"Future date detected: {date_info.selected_date}")

            # Very old date (before 1990)
            if date_info.selected_date.year < 1990:
                date_info.suspicious = True
                logger.warning(f"Very old date detected: {date_info.selected_date}")

            # Default camera dates
            default_dates = [
                datetime(2000, 1, 1),
                datetime(1970, 1, 1),
                datetime(1980, 1, 1),
            ]
            if any(date_info.selected_date.date() == d.date() for d in default_dates):
                date_info.suspicious = True
                logger.warning(
                    f"Suspicious default date detected: {date_info.selected_date}"
                )

    def _get_image_format(
        self, file_path: Path
    ) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """Get image format and resolution using Pillow."""
        try:
            with Image.open(file_path) as img:
                return img.format, img.size
        except Exception as e:
            logger.debug(f"Error getting image format: {e}")
            return None, None

    def _get_video_resolution(
        self, exif_data: Dict[str, Any]
    ) -> Optional[Tuple[int, int]]:
        """
        Extract video resolution from EXIF metadata.

        Args:
            exif_data: EXIF metadata dictionary

        Returns:
            Tuple of (width, height) or None if not found
        """
        # Video resolution can be stored in various EXIF fields
        # depending on the video format and container
        width_fields = [
            "ImageWidth",
            "SourceImageWidth",
            "VideoWidth",
            "ExifImageWidth",
        ]
        height_fields = [
            "ImageHeight",
            "SourceImageHeight",
            "VideoHeight",
            "ExifImageHeight",
        ]

        width = None
        height = None

        # Try to extract width
        for field in width_fields:
            if field in exif_data:
                width = self._parse_int(exif_data[field])
                if width:
                    break

        # Try to extract height
        for field in height_fields:
            if field in exif_data:
                height = self._parse_int(exif_data[field])
                if height:
                    break

        # Both width and height must be present
        if width and height and width > 0 and height > 0:
            return (width, height)

        return None

    def _get_video_format(
        self, file_path: Path, exif_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Get video format from EXIF metadata.

        Tries to extract the actual video codec/format from EXIF data.
        Falls back to file extension if not available.

        Args:
            file_path: Path to video file
            exif_data: EXIF metadata dictionary

        Returns:
            Video format string (e.g., "H.264", "HEVC", "mp4")
        """
        # Try to get codec information from EXIF
        # Different video containers store this in different fields
        codec_fields = [
            "CompressorName",  # QuickTime/MOV
            "VideoCodecID",  # MP4
            "VideoCodec",  # General
            "CompressorID",  # Alternative
            "FileType",  # Container format
        ]

        for field in codec_fields:
            if field in exif_data and exif_data[field]:
                codec = str(exif_data[field])
                # Clean up codec name
                if codec and codec.lower() not in ["unknown", "none"]:
                    return codec

        # Fall back to file extension
        return file_path.suffix.lower().lstrip(".")

    def _parse_float(self, value: Any) -> Optional[float]:
        """
        Parse a value to float.

        Args:
            value: Value to parse (could be string, int, float, etc.)

        Returns:
            Float value or None if parsing fails
        """
        if value is None:
            return None

        try:
            # Handle string values that might have units or extra text
            if isinstance(value, str):
                # Remove common units and extract first number
                import re

                match = re.search(r"[-+]?\d*\.?\d+", value)
                if match:
                    return float(match.group())
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    def _parse_int(self, value: Any) -> Optional[int]:
        """
        Parse a value to int.

        Args:
            value: Value to parse (could be string, int, float, etc.)

        Returns:
            Int value or None if parsing fails
        """
        if value is None:
            return None

        try:
            # Handle string values
            if isinstance(value, str):
                # Remove common units and extract first number
                import re

                match = re.search(r"[-+]?\d+", value)
                if match:
                    return int(match.group())
                return None
            return int(float(value))  # Convert via float to handle "100.0" strings
        except (ValueError, TypeError):
            return None
