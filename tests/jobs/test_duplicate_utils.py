"""Tests for duplicate detection utility functions."""
from datetime import datetime, timedelta
import pytest
from vam_tools.jobs.duplicate_utils import (
    calculate_recommendation_score,
    find_recommended_image,
)


def test_calculate_recommendation_score_complete_data():
    """Test scoring with complete metadata."""
    # Create mock images
    oldest_date = datetime(2024, 1, 1)

    images = [
        {
            "image_id": "img1",
            "quality_score": 90,
            "dates": {"taken": oldest_date},
            "metadata_json": {"width": 4032, "height": 3024},
            "size_bytes": 8000000,
        },
        {
            "image_id": "img2",
            "quality_score": 85,
            "dates": {"taken": oldest_date + timedelta(days=1)},
            "metadata_json": {"width": 4032, "height": 3024},
            "size_bytes": 7000000,
        },
    ]

    # img1 should score higher (better quality, older)
    score1 = calculate_recommendation_score(images[0], images)
    score2 = calculate_recommendation_score(images[1], images)

    assert score1 > score2
    assert score1 >= 65  # Quality (45) + Age (30) + Size (20) = 95


def test_calculate_recommendation_score_missing_quality():
    """Test fallback when quality_score is None."""
    images = [
        {
            "image_id": "img1",
            "quality_score": None,
            "dates": {"taken": datetime(2024, 1, 1)},
            "metadata_json": {"width": 4032, "height": 3024},
            "size_bytes": 8000000,
        },
    ]

    score = calculate_recommendation_score(images[0], images)

    # Should use default quality score of 50
    assert score == 75.0  # Quality (25) + Age (30) + Size (20) = 75


def test_calculate_recommendation_score_missing_dates():
    """Test fallback when dates missing."""
    images = [
        {
            "image_id": "img1",
            "quality_score": 80,
            "dates": {},
            "metadata_json": {"width": 4032, "height": 3024},
            "size_bytes": 8000000,
            "created_at": datetime(2024, 1, 1),
        },
    ]

    score = calculate_recommendation_score(images[0], images)

    # Should still calculate score using created_at
    assert score > 0


def test_find_recommended_image_basic_selection():
    """Test basic selection - highest score wins."""
    images = [
        {
            "image_id": "img1",
            "quality_score": 60,
            "dates": {"taken": datetime(2024, 1, 1)},
            "metadata_json": {"width": 2000, "height": 1500},
            "size_bytes": 5000000,
        },
        {
            "image_id": "img2",
            "quality_score": 90,  # Highest quality
            "dates": {"taken": datetime(2024, 1, 1)},
            "metadata_json": {"width": 2000, "height": 1500},
            "size_bytes": 5000000,
        },
        {
            "image_id": "img3",
            "quality_score": 70,
            "dates": {"taken": datetime(2024, 1, 1)},
            "metadata_json": {"width": 2000, "height": 1500},
            "size_bytes": 5000000,
        },
    ]

    result = find_recommended_image(images)
    assert result == "img2"


def test_find_recommended_image_tiebreaker_oldest():
    """Test tie-breaking - oldest wins when scores are equal."""
    oldest_date = datetime(2023, 1, 1)
    newer_date = datetime(2024, 1, 1)

    images = [
        {
            "image_id": "img1",
            "quality_score": 80,
            "dates": {"taken": newer_date},
            "metadata_json": {"width": 2000, "height": 1500},
            "size_bytes": 5000000,
        },
        {
            "image_id": "img2",
            "quality_score": 80,  # Same quality
            "dates": {"taken": oldest_date},  # Oldest
            "metadata_json": {"width": 2000, "height": 1500},
            "size_bytes": 5000000,
        },
    ]

    result = find_recommended_image(images)
    assert result == "img2"


def test_find_recommended_image_tiebreaker_largest():
    """Test tie-breaking - largest file wins when score and date are equal."""
    same_date = datetime(2024, 1, 1)

    images = [
        {
            "image_id": "img1",
            "quality_score": 80,
            "dates": {"taken": same_date},
            "metadata_json": {"width": 2000, "height": 1500},
            "size_bytes": 5000000,
        },
        {
            "image_id": "img2",
            "quality_score": 80,  # Same quality
            "dates": {"taken": same_date},  # Same date
            "metadata_json": {"width": 2000, "height": 1500},
            "size_bytes": 8000000,  # Largest file
        },
        {
            "image_id": "img3",
            "quality_score": 80,
            "dates": {"taken": same_date},
            "metadata_json": {"width": 2000, "height": 1500},
            "size_bytes": 6000000,
        },
    ]

    result = find_recommended_image(images)
    assert result == "img2"


def test_find_recommended_image_empty_list():
    """Test empty list handling."""
    result = find_recommended_image([])
    assert result is None


def test_find_recommended_image_single_image():
    """Test single image returns that image."""
    images = [
        {
            "image_id": "img1",
            "quality_score": 80,
            "dates": {"taken": datetime(2024, 1, 1)},
            "metadata_json": {"width": 2000, "height": 1500},
            "size_bytes": 5000000,
        },
    ]

    result = find_recommended_image(images)
    assert result == "img1"
