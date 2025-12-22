# Fix Duplicate Detection Transitive Closure Bug

**Date:** 2025-12-21
**Status:** Design Approved
**Problem:** Duplicate detection creates mega-groups (96k+ images) due to transitive closure in Union-Find algorithm

## Problem Analysis

### Current Behavior (Broken)

The duplicate detection uses Union-Find to group similar images. This creates transitive closures:

```
Image A similar to B (distance 3) → merge
Image B similar to C (distance 4) → merge all three
Image C similar to D (distance 2) → merge all four
... continues ...
Result: One massive group of 96,221 images
```

Even when Image A and Image D are NOT similar to each other, they get grouped together because they're connected through intermediate images.

### Root Causes Identified

1. **Transitive closure problem:** Union-Find merges A-B-C-D even when A and D aren't similar
2. **Videos included:** 2,667 videos with NULL hashes (can't hash videos)
3. **Zero hashes:** 116 images with all-zero hashes (spacer GIFs, solid backgrounds) causing false matches
4. **Hash collisions:** 118 images share dhash `cc4cccceced062cf` but have different checksums (visually similar from same photo shoot, not duplicates)

### Evidence

Hash chain causing mega-group:
```
cc4cccceced062cf (118 images) --[dist 3]--> cc4cccceced0704f (92 images)
cc4cccceced0704f --[dist 4]--> cc4ccececed062cf (54 images)
cc4ccececed062cf --[dist 2]--> cc4cccceced060cf (43 images)
```

Each step ≤ threshold (5), so all merge into one group.

## Solution: Strict All-to-All Matching

### Approach

Replace Union-Find with **greedy maximal cliques** that only group images where EVERY image is similar to EVERY other image in the group.

**Rationale:**
- Prevents transitive closure mega-groups
- Fast enough for 96k images: O(n × m × k) where n=pairs, m=groups, k=avg_group_size
- Finds the most important duplicate groups (tight clusters)
- Simple to understand and validate

### Algorithm

```python
def _build_duplicate_groups(pairs: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Build duplicate groups using greedy maximal cliques.
    Only groups images where EVERY image is similar to EVERY other image.
    """
    # 1. Build similarity graph: image_id -> set of similar image_ids
    graph: Dict[str, Set[str]] = {}
    for pair in pairs:
        img1, img2 = pair["image_1"], pair["image_2"]
        graph.setdefault(img1, set()).add(img2)
        graph.setdefault(img2, set()).add(img1)

    # 2. Sort pairs by similarity (lowest distance = most similar first)
    sorted_pairs = sorted(pairs, key=lambda p: p["distance"])

    # 3. Greedy group building
    groups: List[Set[str]] = []
    assigned: Set[str] = set()

    for pair in sorted_pairs:
        img1, img2 = pair["image_1"], pair["image_2"]

        # Try to add to existing group where both are similar to ALL members
        added = False
        for group in groups:
            if _can_add_to_group(img1, group, graph) and \
               _can_add_to_group(img2, group, graph):
                group.add(img1)
                group.add(img2)
                assigned.update([img1, img2])
                added = True
                break

        # Create new group if couldn't extend existing
        if not added:
            groups.append({img1, img2})
            assigned.update([img1, img2])

    return [list(g) for g in groups if len(g) >= 2]


def _can_add_to_group(img: str, group: Set[str], graph: Dict[str, Set[str]]) -> bool:
    """Check if image is similar to ALL members of the group."""
    if img in group:
        return True
    neighbors = graph.get(img, set())
    return all(member in neighbors for member in group)
```

## Changes Required

### 1. Query Filtering (duplicates_comparison_phase_task, ~line 431)

**Before:**
```python
SELECT id, dhash, ahash, whash, checksum, source_path, quality_score
FROM images
WHERE catalog_id = :catalog_id
AND file_type = 'image'
AND (dhash IS NOT NULL OR ahash IS NOT NULL OR whash IS NOT NULL)
```

**After:**
```python
SELECT id, dhash, ahash, whash, checksum, source_path, quality_score
FROM images
WHERE catalog_id = :catalog_id
AND file_type = 'image'          -- exclude videos
AND dhash IS NOT NULL
AND dhash != ''
AND dhash != '0000000000000000'  -- exclude zero hashes
AND ahash IS NOT NULL
AND ahash != ''
AND whash IS NOT NULL
AND whash != ''
```

**Add logging:**
```python
filtered_count = total_images - len(images)
logger.info(
    f"[{task_id}] Filtered {filtered_count} images "
    f"(videos, null hashes, zero hashes). "
    f"Processing {len(images)} valid images."
)
```

### 2. Replace Grouping Algorithm (lines 1105-1151)

Replace entire `_build_duplicate_groups()` function with greedy clique implementation above.

Add helper function:
```python
def _can_add_to_group(img: str, group: Set[str], graph: Dict[str, Set[str]]) -> bool:
    """Check if image is similar to ALL members of the group."""
    if img in group:
        return True
    neighbors = graph.get(img, set())
    return all(member in neighbors for member in group)
```

### 3. Document Alternative Strategies (for future)

Add near line 1100:
```python
class DuplicateGroupingStrategy(str, Enum):
    """Different strategies for grouping similar images."""
    STRICT_CLIQUE = "strict_clique"      # Current: all-to-all matching
    STAR_GROUPS = "star_groups"          # Future: primary with duplicates
    PAIR_GROUPS = "pair_groups"          # Future: overlapping pairs
    TRANSITIVE = "transitive"            # Legacy: Union-find (creates mega-groups)

# Future: make grouping_strategy configurable in coordinator parameters
```

## Edge Cases

1. **No pairs found:** Return empty list (no groups)
2. **Single pairs that can't extend:** Become 2-image groups (correct)
3. **Exact duplicates (checksum match):** Distance=0, always group together
4. **Multiple hash types matching:** Already handled in comparison worker

## Validation

### Before deploying to production:

1. **Clear existing incorrect groups:**
   ```sql
   DELETE FROM duplicate_groups WHERE catalog_id = 'bd40ca52-c3f7-4877-9c97-1c227389c8c4';
   ```

2. **Test with small subset first:**
   Add optional parameter `max_images_for_testing: Optional[int] = None`

3. **Check group size distribution:**
   ```sql
   SELECT
       CASE
           WHEN member_count <= 5 THEN '2-5 images'
           WHEN member_count <= 10 THEN '6-10 images'
           WHEN member_count <= 20 THEN '11-20 images'
           WHEN member_count <= 50 THEN '21-50 images'
           ELSE '50+ images (investigate!)'
       END as size_range,
       COUNT(*) as group_count
   FROM (
       SELECT group_id, COUNT(*) as member_count
       FROM duplicate_members
       GROUP BY group_id
   ) sub
   GROUP BY size_range
   ORDER BY size_range;
   ```

### Expected Results

- **Group size distribution:** Mostly 2-10 images per group
- **Largest group:** Probably 20-50 images max (burst photos from same shoot)
- **Total groups:** Hundreds to thousands (not just 1!)
- **Unmatched images:** Images with unique hashes not grouped

### Performance

- **Query filtering:** Negligible overhead (reduces work)
- **Greedy clique algorithm:** O(n × m × k) complexity
  - n = number of pairs (~68M in test)
  - m = number of groups (expected: 1000s)
  - k = average group size (expected: 3-5)
- **Expected:** Faster than current due to fewer images processed

## Summary

| Component | Change | Location |
|-----------|--------|----------|
| Image loading query | Add filters for videos, NULL/zero hashes | `parallel_duplicates.py:431` |
| Grouping algorithm | Replace Union-Find with greedy cliques | `parallel_duplicates.py:1105-1151` |
| Helper function | Add `_can_add_to_group()` | `parallel_duplicates.py:~1153` |
| Future strategies | Document alternatives in comments | `parallel_duplicates.py:~1100` |
| Logging | Add filtered image count | `parallel_duplicates.py:~460` |

## Alternative Approaches Considered

1. **Star-shaped groups** - Primary image with duplicates, but A and C not duplicates of each other
   - Good for photo shoots with many similar shots
   - May implement later if needed

2. **Separate pair groups** - {A,B} and {B,C} are independent, images can appear in multiple groups
   - Good for finding all similar relationships
   - May implement later if needed

3. **Bron-Kerbosch clique finding** - Finds ALL maximal cliques
   - Most complete but exponential time complexity
   - Too slow for 96k images

**Decision:** Greedy maximal cliques provide the best balance of speed, accuracy, and simplicity.

## Implementation Completed

**Date:** 2025-12-22
**Branch:** `fix/duplicate-detection-transitive-closure`

### Changes Made

1. ✅ Added `_can_add_to_group()` helper function for strict similarity checking
2. ✅ Replaced `_build_duplicate_groups()` Union-Find with greedy clique algorithm
3. ✅ Added query filtering to exclude videos, NULL hashes, and zero hashes
4. ✅ Added logging for filtered image counts
5. ✅ Documented alternative grouping strategies with enum
6. ✅ Added comprehensive test suite (unit + integration tests)

### Test Results

All tests passing:
- `test_can_add_to_group_*` - Helper function validation
- `test_build_groups_*` - Greedy clique algorithm correctness
- `test_prevents_mega_group_from_hash_chain` - Real-world scenario validation
- `test_exact_duplicates_still_group` - Exact duplicate handling

**Total:** 10 tests passing

### Next Steps

1. Merge to main branch
2. Deploy to production
3. Run duplicate detection on test catalog
4. Verify group size distribution is reasonable (2-50 images per group)
5. Monitor for any performance issues

### Files Modified

- `vam_tools/jobs/parallel_duplicates.py` - Core algorithm changes
- `tests/jobs/test_duplicate_grouping.py` - Unit tests (new file)
- `tests/jobs/test_duplicate_detection_integration.py` - Integration tests (new file)
- `docs/plans/2025-12-21-fix-duplicate-detection-transitive-closure.md` - This doc
