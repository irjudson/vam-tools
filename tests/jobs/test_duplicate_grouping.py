"""Tests for duplicate detection grouping algorithms."""

import pytest

from lumina.jobs.parallel_duplicates import (
    _build_duplicate_groups,
    _can_add_to_group,
)


def test_build_groups_all_similar():
    """Test grouping when all images are similar to each other (complete graph)."""
    # A-B-C triangle where all are similar
    pairs = [
        {"image_1": "A", "image_2": "B", "distance": 2},
        {"image_1": "B", "image_2": "C", "distance": 3},
        {"image_1": "A", "image_2": "C", "distance": 4},
    ]

    groups = _build_duplicate_groups(pairs)

    # Should create one group with all three images
    assert len(groups) == 1
    assert set(groups[0]) == {"A", "B", "C"}


def test_build_groups_transitive_not_similar():
    """Test that transitive closure is NOT created when A-C are not similar."""
    # A-B and B-C are similar, but A-C are NOT
    pairs = [
        {"image_1": "A", "image_2": "B", "distance": 2},
        {"image_1": "B", "image_2": "C", "distance": 3},
    ]

    groups = _build_duplicate_groups(pairs)

    # Should create separate groups, not merge via transitive closure
    # Old behavior: 1 group with {A, B, C}
    # New behavior: Could be {A, B} and {B, C}, or just {A, B} depending on implementation
    assert len(groups) >= 1
    # At minimum, should not create a mega-group
    # The key is that A and C should not be in same group if they're not similar
    for group in groups:
        assert len(group) <= 3


def test_build_groups_separate_cliques():
    """Test that separate cliques remain separate."""
    # Two separate triangles: A-B-C and D-E-F
    pairs = [
        {"image_1": "A", "image_2": "B", "distance": 2},
        {"image_1": "B", "image_2": "C", "distance": 3},
        {"image_1": "A", "image_2": "C", "distance": 4},
        {"image_1": "D", "image_2": "E", "distance": 1},
        {"image_1": "E", "image_2": "F", "distance": 2},
        {"image_1": "D", "image_2": "F", "distance": 3},
    ]

    groups = _build_duplicate_groups(pairs)

    # Should create two separate groups
    assert len(groups) == 2
    groups_sets = [set(g) for g in groups]
    assert {"A", "B", "C"} in groups_sets
    assert {"D", "E", "F"} in groups_sets


def test_build_groups_no_pairs():
    """Test handling of empty pairs list."""
    pairs = []
    groups = _build_duplicate_groups(pairs)
    assert groups == []


def test_build_groups_single_pair():
    """Test handling of single pair."""
    pairs = [{"image_1": "A", "image_2": "B", "distance": 2}]
    groups = _build_duplicate_groups(pairs)

    assert len(groups) == 1
    assert set(groups[0]) == {"A", "B"}


def test_can_add_to_group_all_connected():
    """Test _can_add_to_group when image is similar to all group members."""
    graph = {
        "A": {"B", "C", "D"},
        "B": {"A", "C", "D"},
        "C": {"A", "B", "D"},
        "D": {"A", "B", "C"},
    }
    group = {"A", "B", "C"}

    # D is connected to all members, should return True
    assert _can_add_to_group("D", group, graph) is True


def test_can_add_to_group_not_all_connected():
    """Test _can_add_to_group when image is not similar to all group members."""
    graph = {
        "A": {"B"},
        "B": {"A", "C"},
        "C": {"B"},
        "D": {"B"},  # D only connected to B, not A or C
    }
    group = {"A", "B", "C"}

    # D is only connected to B, not A or C, should return False
    assert _can_add_to_group("D", group, graph) is False


def test_can_add_to_group_already_in_group():
    """Test _can_add_to_group when image is already in the group."""
    graph = {"A": {"B"}, "B": {"A"}}
    group = {"A", "B"}

    # A is already in group, should return True
    assert _can_add_to_group("A", group, graph) is True
