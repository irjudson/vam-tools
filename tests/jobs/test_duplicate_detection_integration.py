"""Integration tests for duplicate detection with real-world scenarios."""

import pytest

from vam_tools.jobs.parallel_duplicates import _build_duplicate_groups


def test_prevents_mega_group_from_hash_chain():
    """
    Test that hash chains don't create mega-groups.

    Simulates the real bug: many images with similar hashes that form a chain
    where each pair is similar but ends are not similar.
    """
    # Simulate real hash collision scenario from investigation:
    # cc4cccceced062cf (118 images) --[dist 3]--> cc4cccceced0704f (92 images)
    # cc4cccceced0704f --[dist 4]--> cc4ccececed062cf (54 images)
    # cc4ccececed062cf --[dist 2]--> cc4cccceced060cf (43 images)

    # Create pairs representing this chain (simplified)
    pairs = [
        # Group 1 internally similar (simulate 118 images with same hash)
        {"image_1": "A1", "image_2": "A2", "distance": 0},
        {"image_1": "A1", "image_2": "A3", "distance": 0},
        {"image_1": "A2", "image_2": "A3", "distance": 0},
        # Group 2 internally similar (simulate 92 images)
        {"image_1": "B1", "image_2": "B2", "distance": 0},
        {"image_1": "B1", "image_2": "B3", "distance": 0},
        {"image_1": "B2", "image_2": "B3", "distance": 0},
        # Group 3 internally similar (simulate 54 images)
        {"image_1": "C1", "image_2": "C2", "distance": 0},
        # Cross-group connections (these create the chain)
        {"image_1": "A1", "image_2": "B1", "distance": 3},  # Within threshold
        {"image_1": "B1", "image_2": "C1", "distance": 4},  # Within threshold
        # Note: A1 and C1 are NOT similar (distance would be > 5)
    ]

    groups = _build_duplicate_groups(pairs)

    # Old behavior: 1 mega-group with all images
    # New behavior: Should create separate groups or at most merge A's with B's
    #               but NOT create one group with all A, B, and C

    # Verify we don't have a mega-group
    max_group_size = max(len(g) for g in groups) if groups else 0
    assert max_group_size <= 5, f"Created mega-group with {max_group_size} images"

    # Verify A's and C's are not in same group (no transitive closure)
    group_sets = [set(g) for g in groups]
    for group_set in group_sets:
        # Should not have both A and C members in same group
        has_a = any(img.startswith("A") for img in group_set)
        has_c = any(img.startswith("C") for img in group_set)
        assert not (
            has_a and has_c
        ), "A and C images in same group (transitive closure)"


def test_exact_duplicates_still_group():
    """Test that exact duplicates (distance 0) still group correctly."""
    pairs = [
        {"image_1": "A", "image_2": "B", "distance": 0},
        {"image_1": "B", "image_2": "C", "distance": 0},
        {"image_1": "A", "image_2": "C", "distance": 0},
    ]

    groups = _build_duplicate_groups(pairs)

    # Exact duplicates should all be in one group
    assert len(groups) == 1
    assert set(groups[0]) == {"A", "B", "C"}
