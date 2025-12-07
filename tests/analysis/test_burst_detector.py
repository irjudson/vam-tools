"""Tests for burst detection algorithm."""

from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock

import pytest

from vam_tools.analysis.burst_detector import (
    BurstDetector,
    BurstGroup,
    ImageInfo,
)


class TestImageInfo:
    """Tests for ImageInfo dataclass."""

    def test_image_info_creation(self):
        """Test creating ImageInfo."""
        info = ImageInfo(
            image_id="img-001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            camera_make="Canon",
            camera_model="EOS R5",
            quality_score=0.85,
        )
        assert info.image_id == "img-001"
        assert info.camera_make == "Canon"


class TestBurstGroup:
    """Tests for BurstGroup dataclass."""

    def test_burst_group_duration(self):
        """Test burst group duration calculation."""
        images = [
            ImageInfo("img-001", datetime(2024, 1, 1, 12, 0, 0), "Canon", "R5", 0.8),
            ImageInfo("img-002", datetime(2024, 1, 1, 12, 0, 1), "Canon", "R5", 0.9),
            ImageInfo("img-003", datetime(2024, 1, 1, 12, 0, 2), "Canon", "R5", 0.7),
        ]
        group = BurstGroup(images=images)

        assert group.duration_seconds == 2.0
        assert group.image_count == 3


class TestBurstDetector:
    """Tests for BurstDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = BurstDetector(gap_threshold_seconds=2.0, min_burst_size=3)
        assert detector.gap_threshold_seconds == 2.0
        assert detector.min_burst_size == 3

    def test_detect_bursts_finds_sequences(self):
        """Test that detector finds burst sequences."""
        detector = BurstDetector(gap_threshold_seconds=2.0, min_burst_size=3)

        # Create images with burst pattern
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        images = [
            # Burst 1: 3 images, 0.5s apart
            ImageInfo("img-001", base_time, "Canon", "R5", 0.8),
            ImageInfo("img-002", base_time + timedelta(seconds=0.5), "Canon", "R5", 0.9),
            ImageInfo("img-003", base_time + timedelta(seconds=1.0), "Canon", "R5", 0.7),
            # Gap of 10 seconds
            # Burst 2: 4 images, 1s apart
            ImageInfo("img-004", base_time + timedelta(seconds=11), "Canon", "R5", 0.85),
            ImageInfo("img-005", base_time + timedelta(seconds=12), "Canon", "R5", 0.95),
            ImageInfo("img-006", base_time + timedelta(seconds=13), "Canon", "R5", 0.75),
            ImageInfo("img-007", base_time + timedelta(seconds=14), "Canon", "R5", 0.80),
            # Single image (not a burst)
            ImageInfo("img-008", base_time + timedelta(seconds=30), "Canon", "R5", 0.90),
        ]

        bursts = detector.detect_bursts(images)

        assert len(bursts) == 2
        assert bursts[0].image_count == 3
        assert bursts[1].image_count == 4

    def test_detect_bursts_respects_camera_boundaries(self):
        """Test that bursts don't cross camera boundaries."""
        detector = BurstDetector(gap_threshold_seconds=2.0, min_burst_size=3)

        base_time = datetime(2024, 1, 1, 12, 0, 0)
        images = [
            # Canon images
            ImageInfo("img-001", base_time, "Canon", "R5", 0.8),
            ImageInfo("img-002", base_time + timedelta(seconds=0.5), "Canon", "R5", 0.9),
            ImageInfo("img-003", base_time + timedelta(seconds=1.0), "Canon", "R5", 0.7),
            # Sony images at same time (different camera)
            ImageInfo("img-004", base_time + timedelta(seconds=1.5), "Sony", "A7", 0.85),
            ImageInfo("img-005", base_time + timedelta(seconds=2.0), "Sony", "A7", 0.95),
            ImageInfo("img-006", base_time + timedelta(seconds=2.5), "Sony", "A7", 0.75),
        ]

        bursts = detector.detect_bursts(images)

        assert len(bursts) == 2
        assert all(img.camera_make == "Canon" for img in bursts[0].images)
        assert all(img.camera_make == "Sony" for img in bursts[1].images)

    def test_detect_bursts_ignores_small_sequences(self):
        """Test that sequences smaller than min_burst_size are ignored."""
        detector = BurstDetector(gap_threshold_seconds=2.0, min_burst_size=3)

        base_time = datetime(2024, 1, 1, 12, 0, 0)
        images = [
            # Only 2 images - not a burst
            ImageInfo("img-001", base_time, "Canon", "R5", 0.8),
            ImageInfo("img-002", base_time + timedelta(seconds=0.5), "Canon", "R5", 0.9),
        ]

        bursts = detector.detect_bursts(images)

        assert len(bursts) == 0

    def test_select_best_image_uses_quality_score(self):
        """Test that best image is selected by quality score."""
        detector = BurstDetector()

        images = [
            ImageInfo("img-001", datetime(2024, 1, 1, 12, 0, 0), "Canon", "R5", 0.70),
            ImageInfo("img-002", datetime(2024, 1, 1, 12, 0, 1), "Canon", "R5", 0.95),  # Best
            ImageInfo("img-003", datetime(2024, 1, 1, 12, 0, 2), "Canon", "R5", 0.80),
        ]
        group = BurstGroup(images=images)

        best = detector.select_best_image(group)

        assert best.image_id == "img-002"
        assert best.quality_score == 0.95
