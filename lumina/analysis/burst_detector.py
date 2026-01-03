"""Burst detection for continuous shooting sequences.

Detects groups of images taken in rapid succession (bursts) based on
timestamps and camera metadata. No ML required - pure algorithmic approach.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """Information about an image for burst detection."""

    image_id: str
    timestamp: datetime
    camera_make: Optional[str]
    camera_model: Optional[str]
    quality_score: float = 0.0
    source_path: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    geohash: Optional[str] = None

    @property
    def camera_key(self) -> str:
        """Get a unique key for the camera."""
        make = self.camera_make or "unknown"
        model = self.camera_model or "unknown"
        return f"{make}:{model}"

    @property
    def filename(self) -> str:
        """Get just the filename from the source path."""
        if not self.source_path:
            return ""
        return self.source_path.split("/")[-1]

    @property
    def has_gps(self) -> bool:
        """Check if image has GPS coordinates."""
        return self.latitude is not None and self.longitude is not None


@dataclass
class BurstGroup:
    """A group of images forming a burst sequence."""

    images: List[ImageInfo] = field(default_factory=list)
    best_image_id: Optional[str] = None
    selection_method: str = "quality"

    @property
    def image_count(self) -> int:
        """Number of images in burst."""
        return len(self.images)

    @property
    def start_time(self) -> Optional[datetime]:
        """Start time of burst."""
        if not self.images:
            return None
        return min(img.timestamp for img in self.images)

    @property
    def end_time(self) -> Optional[datetime]:
        """End time of burst."""
        if not self.images:
            return None
        return max(img.timestamp for img in self.images)

    @property
    def duration_seconds(self) -> float:
        """Duration of burst in seconds."""
        if not self.images or len(self.images) < 2:
            return 0.0
        start = self.start_time
        end = self.end_time
        if start and end:
            return (end - start).total_seconds()
        return 0.0

    @property
    def camera_make(self) -> Optional[str]:
        """Camera make (should be same for all images)."""
        if self.images:
            return self.images[0].camera_make
        return None

    @property
    def camera_model(self) -> Optional[str]:
        """Camera model (should be same for all images)."""
        if self.images:
            return self.images[0].camera_model
        return None


class BurstDetector:
    """Detects burst sequences in image collections.

    A burst is defined as:
    1. Images from the same camera (make + model)
    2. Taken within gap_threshold_seconds of each other (< 1s)
    3. From the same location (GPS coordinates within tolerance)
    4. Sequential filenames (continuous numeric sequence)
    5. At least min_burst_size images in the sequence
    6. Total duration >= min_duration_seconds (filters identical timestamps)
    """

    def __init__(
        self,
        gap_threshold_seconds: float = 1.0,
        min_burst_size: int = 3,
        location_tolerance_meters: float = 10.0,
        min_duration_seconds: float = 0.5,
    ):
        """Initialize burst detector.

        Args:
            gap_threshold_seconds: Maximum gap between images (default 1.0s for bursts)
            min_burst_size: Minimum images required to form a burst
            location_tolerance_meters: Maximum distance between GPS coords (default 10m)
            min_duration_seconds: Minimum total duration for valid burst (default 0.5s)
                                 Filters out groups with identical timestamps
        """
        self.gap_threshold_seconds = gap_threshold_seconds
        self.min_burst_size = min_burst_size
        self.location_tolerance_meters = location_tolerance_meters
        self.min_duration_seconds = min_duration_seconds

    def detect_bursts(self, images: List[ImageInfo]) -> List[BurstGroup]:
        """Detect burst sequences in a list of images.

        Args:
            images: List of ImageInfo objects (will be sorted internally)

        Returns:
            List of detected BurstGroups
        """
        if len(images) < self.min_burst_size:
            return []

        # Group by camera first
        by_camera: dict = {}
        for img in images:
            key = img.camera_key
            if key not in by_camera:
                by_camera[key] = []
            by_camera[key].append(img)

        all_bursts: List[BurstGroup] = []

        # Process each camera's images
        for _camera_key, camera_images in by_camera.items():
            # Sort by timestamp
            sorted_images = sorted(camera_images, key=lambda x: x.timestamp)

            # Find burst sequences
            bursts = self._find_sequences(sorted_images)
            all_bursts.extend(bursts)

        # Sort bursts by start time
        all_bursts.sort(key=lambda b: b.start_time or datetime.min)

        logger.info(f"Detected {len(all_bursts)} bursts from {len(images)} images")
        return all_bursts

    def _is_same_location(self, img1: ImageInfo, img2: ImageInfo) -> bool:
        """Check if two images are from the same location.

        Uses geohash for fast comparison when available, falls back to
        Haversine distance for edge cases.

        Args:
            img1: First image
            img2: Second image

        Returns:
            True if same location or if GPS data missing
        """
        # If either image lacks GPS, can't verify location - allow it
        if not img1.has_gps or not img2.has_gps:
            return True

        # Fast path: check geohash match (precision 7 = ~153m cells)
        # If geohashes match, they're definitely within tolerance
        if img1.geohash and img2.geohash:
            # Use first 6 chars for ~610m cells (covers our 10m tolerance with margin)
            if img1.geohash[:6] == img2.geohash[:6]:
                return True
            # If geohashes differ, images might still be close (at cell boundary)
            # Fall through to Haversine check

        # Precise check: Calculate distance using Haversine formula
        from math import atan2, cos, radians, sin, sqrt

        # has_gps check above guarantees these are not None
        assert img1.latitude is not None and img1.longitude is not None
        assert img2.latitude is not None and img2.longitude is not None

        lat1, lon1 = radians(img1.latitude), radians(img1.longitude)
        lat2, lon2 = radians(img2.latitude), radians(img2.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance_m = 6371000 * c  # Earth radius in meters

        return distance_m <= self.location_tolerance_meters

    def _is_sequential_filename(self, img1: ImageInfo, img2: ImageInfo) -> bool:
        """Check if two images have sequential filenames.

        Args:
            img1: First image
            img2: Second image

        Returns:
            True if filenames are sequential or can't determine
        """
        import re

        name1 = img1.filename
        name2 = img2.filename

        if not name1 or not name2:
            return True  # Can't verify - allow it

        # Extract base name and number from filenames
        # Examples: IMG_1234.JPG -> IMG_, 1234
        #           DSC00123.ARW -> DSC, 123
        match1 = re.match(r"([A-Za-z_]+)(\d+)", name1)
        match2 = re.match(r"([A-Za-z_]+)(\d+)", name2)

        if not match1 or not match2:
            return True  # Can't parse - allow it

        base1, num1 = match1.groups()
        base2, num2 = match2.groups()

        # Must have same base prefix
        if base1 != base2:
            return False

        # Numbers should be sequential (within 5 to allow for some gaps)
        try:
            diff = abs(int(num2) - int(num1))
            return diff <= 5
        except ValueError:
            return True  # Can't parse numbers - allow it

    def _find_sequences(self, sorted_images: List[ImageInfo]) -> List[BurstGroup]:
        """Find burst sequences in time-sorted images from same camera.

        Applies all burst criteria:
        1. Same camera (already grouped)
        2. Time gap < threshold
        3. Same location (GPS)
        4. Sequential filenames

        Args:
            sorted_images: Images sorted by timestamp, all from same camera

        Returns:
            List of BurstGroups
        """
        if len(sorted_images) < self.min_burst_size:
            return []

        bursts: List[BurstGroup] = []
        current_sequence: List[ImageInfo] = [sorted_images[0]]

        for i in range(1, len(sorted_images)):
            current_img = sorted_images[i]
            prev_img = sorted_images[i - 1]

            # Check all burst criteria
            time_gap = (current_img.timestamp - prev_img.timestamp).total_seconds()
            same_location = self._is_same_location(prev_img, current_img)
            sequential_files = self._is_sequential_filename(prev_img, current_img)

            # All criteria must be met
            if (
                time_gap <= self.gap_threshold_seconds
                and same_location
                and sequential_files
            ):
                # Continue current sequence
                current_sequence.append(current_img)
            else:
                # Criteria not met - check if current sequence is a burst
                if len(current_sequence) >= self.min_burst_size:
                    burst = BurstGroup(images=list(current_sequence))
                    # Only include bursts with sufficient duration
                    # (filters out groups with identical timestamps)
                    if burst.duration_seconds >= self.min_duration_seconds:
                        burst.best_image_id = self.select_best_image(burst).image_id
                        bursts.append(burst)

                # Start new sequence
                current_sequence = [current_img]

        # Don't forget the last sequence
        if len(current_sequence) >= self.min_burst_size:
            burst = BurstGroup(images=list(current_sequence))
            # Only include bursts with sufficient duration
            if burst.duration_seconds >= self.min_duration_seconds:
                burst.best_image_id = self.select_best_image(burst).image_id
                bursts.append(burst)

        return bursts

    def select_best_image(
        self,
        group: BurstGroup,
        method: str = "quality",
    ) -> ImageInfo:
        """Select the best image from a burst group.

        Args:
            group: BurstGroup to select from
            method: Selection method ('quality', 'first', 'middle')

        Returns:
            The selected ImageInfo
        """
        if not group.images:
            raise ValueError("Cannot select from empty burst group")

        if method == "first":
            return group.images[0]
        elif method == "middle":
            return group.images[len(group.images) // 2]
        else:  # quality (default)
            return max(group.images, key=lambda img: img.quality_score)
