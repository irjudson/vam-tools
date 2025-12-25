"""Utility functions for duplicate detection."""

from datetime import datetime
from typing import Any, Dict, List, Optional


def calculate_recommendation_score(
    image: Dict[str, Any], group_images: List[Dict[str, Any]]
) -> float:
    """
    Calculate recommendation score for an image in a duplicate group.

    Returns score 0-100 where higher = better keeper candidate.

    Scoring breakdown:
    - Quality: 50% weight (uses image.quality_score)
    - Age: 30% weight (older = better, preserves originals)
    - Size: 20% weight (larger resolution = better)

    Args:
        image: Image dict with quality_score, dates, metadata_json
        group_images: All images in the group for comparison

    Returns:
        Score from 0-100
    """
    # Quality component (50% weight)
    quality_score = image.get("quality_score") or 50  # Default if missing
    quality_component = quality_score * 0.5

    # Age component (30% weight) - older is better
    image_date = image.get("dates", {}).get("taken")
    if not image_date:
        image_date = image.get("created_at")

    # Find oldest date in group
    oldest_date = None
    for img in group_images:
        img_date = img.get("dates", {}).get("taken")
        if not img_date:
            img_date = img.get("created_at")
        if img_date:
            if oldest_date is None or img_date < oldest_date:
                oldest_date = img_date

    if image_date and oldest_date:
        if image_date == oldest_date:
            age_component = 30.0  # Full points for oldest
        else:
            # Proportional reduction based on days newer (caps at 50% for 1+ year)
            days_diff = (image_date - oldest_date).days
            age_component = 30.0 * (1 - min(days_diff / 365, 0.5))
    else:
        age_component = 15.0  # Default if no date data

    # Size component (20% weight) - larger resolution is better
    width = image.get("metadata_json", {}).get("width", 0)
    height = image.get("metadata_json", {}).get("height", 0)
    resolution = width * height

    max_resolution = 0
    for img in group_images:
        w = img.get("metadata_json", {}).get("width", 0)
        h = img.get("metadata_json", {}).get("height", 0)
        r = w * h
        if r > max_resolution:
            max_resolution = r

    if max_resolution > 0 and resolution > 0:
        size_component = (resolution / max_resolution) * 20.0
    else:
        size_component = 10.0  # Default if no resolution data

    return quality_component + age_component + size_component


def find_recommended_image(images: List[Dict[str, Any]]) -> Optional[str]:
    """
    Find the recommended image to keep from a duplicate group.

    Returns the image_id with the highest recommendation score.
    Tie-breaking: oldest > largest file > first by image_id.

    Args:
        images: List of image dicts

    Returns:
        image_id of recommended image, or None if images list is empty
    """
    if not images:
        return None

    # Calculate scores for all images
    scored = []
    for img in images:
        score = calculate_recommendation_score(img, images)
        scored.append((score, img))

    # Sort by score descending, then by date ascending, then by size descending
    scored.sort(
        key=lambda x: (
            -x[0],  # Higher score first
            x[1].get("dates", {}).get("taken")
            or x[1].get("created_at")
            or datetime.max,  # Older first
            -(x[1].get("size_bytes", 0)),  # Larger first
        )
    )

    return scored[0][1]["image_id"]
