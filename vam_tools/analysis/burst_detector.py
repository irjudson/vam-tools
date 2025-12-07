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

    @property
    def camera_key(self) -> str:
        """Get a unique key for the camera."""
        make = self.camera_make or "unknown"
        model = self.camera_model or "unknown"
        return f"{make}:{model}"


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
    1. Images from the same camera
    2. Taken within gap_threshold_seconds of each other
    3. At least min_burst_size images in the sequence
    """

    def __init__(
        self,
        gap_threshold_seconds: float = 2.0,
        min_burst_size: int = 3,
    ):
        """Initialize burst detector.

        Args:
            gap_threshold_seconds: Maximum gap between images to be in same burst
            min_burst_size: Minimum images required to form a burst
        """
        self.gap_threshold_seconds = gap_threshold_seconds
        self.min_burst_size = min_burst_size

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

    def _find_sequences(self, sorted_images: List[ImageInfo]) -> List[BurstGroup]:
        """Find burst sequences in time-sorted images from same camera.

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

            gap = (current_img.timestamp - prev_img.timestamp).total_seconds()

            if gap <= self.gap_threshold_seconds:
                # Continue current sequence
                current_sequence.append(current_img)
            else:
                # Gap too large - check if current sequence is a burst
                if len(current_sequence) >= self.min_burst_size:
                    burst = BurstGroup(images=list(current_sequence))
                    burst.best_image_id = self.select_best_image(burst).image_id
                    bursts.append(burst)

                # Start new sequence
                current_sequence = [current_img]

        # Don't forget the last sequence
        if len(current_sequence) >= self.min_burst_size:
            burst = BurstGroup(images=list(current_sequence))
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
