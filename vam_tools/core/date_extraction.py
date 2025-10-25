"""
Date extraction from images.

This module provides functionality to extract dates from images using multiple sources:
- EXIF metadata (most reliable)
- Filename patterns
- Directory structure
- File system timestamps
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import arrow
import exiftool

logger = logging.getLogger(__name__)

# Date formats for parsing EXIF data
EXIF_DATE_FORMATS = [
    "YYYY:MM:DD HH:mm:ssZZ",
    "YYYY:MM:DD HH:mm:ss",
    "YYYY-MM-DD HH:mm:ss",
    "YYYY:MM:DD",
    "YYYY-MM-DD",
]

# Filename date patterns (pattern, is_year_first)
FILENAME_PATTERNS = [
    (r"(\d{4})-(\d{2})-(\d{2})", True),  # YYYY-MM-DD
    (r"(\d{4})_(\d{2})_(\d{2})", True),  # YYYY_MM_DD
    (r"(\d{4})(\d{2})(\d{2})", True),  # YYYYMMDD
    (r"(\d{2})-(\d{2})-(\d{4})", False),  # MM-DD-YYYY or DD-MM-YYYY
    (r"(\d{2})_(\d{2})_(\d{4})", False),  # MM_DD_YYYY or DD_MM_YYYY
    (r"(\d{2})(\d{2})(\d{4})", False),  # MMDDYYYY or DDMMYYYY
]


@dataclass
class DateInfo:
    """Information about a date extracted from an image."""

    date: arrow.Arrow
    source: str  # 'exif', 'filename', 'directory', 'filesystem'
    confidence: int  # 0-100, higher is more reliable


class DateExtractor:
    """Extract dates from images using multiple sources."""

    def __init__(self) -> None:
        """Initialize the date extractor."""
        self.exif_tool: Optional[exiftool.ExifToolHelper] = None

    def __enter__(self) -> "DateExtractor":
        """Context manager entry."""
        self.exif_tool = exiftool.ExifToolHelper()
        self.exif_tool.__enter__()
        return self

    def __exit__(self, *args: any) -> None:
        """Context manager exit."""
        if self.exif_tool:
            self.exif_tool.__exit__(*args)

    def extract_exif_date(self, image_path: Path) -> Optional[DateInfo]:
        """
        Extract the earliest date from EXIF metadata.

        Args:
            image_path: Path to the image file

        Returns:
            DateInfo object if a date was found, None otherwise
        """
        if not self.exif_tool:
            logger.warning("ExifTool not initialized, use as context manager")
            return None

        try:
            metadata_list = self.exif_tool.get_metadata([str(image_path)])
            if not metadata_list:
                logger.debug(f"No EXIF data found for {image_path}")
                return None

            metadata = metadata_list[0]
            earliest_date: Optional[arrow.Arrow] = None

            # Look for date fields in EXIF data
            for key, value in metadata.items():
                if "date" in key.lower() and isinstance(value, str):
                    try:
                        parsed_date = arrow.get(
                            value,
                            EXIF_DATE_FORMATS,
                            normalize_whitespace=True,
                        )
                        if (
                            earliest_date is None
                            or parsed_date.timestamp() < earliest_date.timestamp()
                        ):
                            earliest_date = parsed_date
                            logger.debug(
                                f"Found EXIF date in {key}: {parsed_date}"
                            )
                    except (arrow.ParserError, ValueError, TypeError):
                        logger.debug(f"Could not parse date from {key}: {value}")
                        continue

            if earliest_date:
                return DateInfo(
                    date=earliest_date, source="exif", confidence=95
                )

        except Exception as e:
            logger.debug(f"Error extracting EXIF date from {image_path}: {e}")

        return None

    def extract_filename_date(self, image_path: Path) -> Optional[DateInfo]:
        """
        Extract date from filename using common patterns.

        Args:
            image_path: Path to the image file

        Returns:
            DateInfo object if a date was found, None otherwise
        """
        filename = image_path.name

        for pattern, is_year_first in FILENAME_PATTERNS:
            match = re.search(pattern, filename)
            if match:
                try:
                    groups = match.groups()
                    if is_year_first:
                        year, month, day = groups
                    else:
                        # Assume MM-DD-YYYY for ambiguous formats
                        month, day, year = groups

                    parsed_date = arrow.get(
                        f"{year}-{month}-{day}", "YYYY-MM-DD"
                    )
                    logger.debug(
                        f"Extracted date from filename {filename}: {parsed_date}"
                    )
                    return DateInfo(
                        date=parsed_date, source="filename", confidence=70
                    )
                except (ValueError, arrow.ParserError):
                    continue

        return None

    def extract_directory_date(self, image_path: Path) -> Optional[DateInfo]:
        """
        Extract date from directory structure.

        Looks for year/month/day patterns in the directory path.

        Args:
            image_path: Path to the image file

        Returns:
            DateInfo object if a date was found, None otherwise
        """
        path_parts = image_path.parts

        # Look for year (4 digits) in path
        for part in reversed(path_parts):
            year_match = re.search(r"(\d{4})", part)
            if year_match:
                year = int(year_match.group(1))
                if 1900 <= year <= 2100:
                    # Look for month in the same or adjacent parts
                    month_match = re.search(r"(\d{1,2})", part)
                    month = 1  # Default to January
                    day = 1  # Default to first day

                    if month_match:
                        potential_month = int(month_match.group(1))
                        if 1 <= potential_month <= 12:
                            month = potential_month

                    try:
                        parsed_date = arrow.get(year, month, day)
                        logger.debug(
                            f"Extracted date from directory: {parsed_date}"
                        )
                        return DateInfo(
                            date=parsed_date, source="directory", confidence=50
                        )
                    except (ValueError, arrow.ParserError):
                        pass

        return None

    def extract_filesystem_date(self, image_path: Path) -> Optional[DateInfo]:
        """
        Extract date from filesystem metadata.

        Uses the file creation time as a last resort.

        Args:
            image_path: Path to the image file

        Returns:
            DateInfo object with the file creation date
        """
        try:
            stat = os.stat(image_path)
            # Use the earlier of creation time and modification time
            timestamp = min(stat.st_ctime, stat.st_mtime)
            parsed_date = arrow.get(timestamp)
            logger.debug(f"Using filesystem date: {parsed_date}")
            return DateInfo(
                date=parsed_date, source="filesystem", confidence=30
            )
        except Exception as e:
            logger.debug(
                f"Error getting filesystem date for {image_path}: {e}"
            )
            return None

    def extract_earliest_date(self, image_path: Path) -> Optional[DateInfo]:
        """
        Extract the earliest date from all available sources.

        Args:
            image_path: Path to the image file

        Returns:
            DateInfo object with the earliest date found, or None if no dates found
        """
        dates: List[DateInfo] = []

        # Try all extraction methods
        for method in [
            self.extract_exif_date,
            self.extract_filename_date,
            self.extract_directory_date,
            self.extract_filesystem_date,
        ]:
            date_info = method(image_path)
            if date_info:
                dates.append(date_info)

        if not dates:
            return None

        # Return the earliest date (prioritize by confidence in case of ties)
        return min(dates, key=lambda d: (d.date.timestamp(), -d.confidence))

    def analyze_images(
        self, image_paths: List[Path]
    ) -> Dict[Path, Optional[DateInfo]]:
        """
        Analyze multiple images and extract dates.

        Args:
            image_paths: List of paths to image files

        Returns:
            Dictionary mapping image paths to DateInfo objects
        """
        results: Dict[Path, Optional[DateInfo]] = {}

        for image_path in image_paths:
            try:
                date_info = self.extract_earliest_date(image_path)
                results[image_path] = date_info
            except Exception as e:
                logger.error(f"Error analyzing {image_path}: {e}")
                results[image_path] = None

        return results
