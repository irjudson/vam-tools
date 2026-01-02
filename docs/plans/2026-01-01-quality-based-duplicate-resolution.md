# Quality-Based Duplicate Resolution

**Created:** 2026-01-01
**Status:** Design
**Priority:** HIGH

## Overview

Design and implement a quality scoring algorithm to automatically identify the "best" image in duplicate groups, allowing users to safely delete lower-quality versions.

## Problem Statement

After duplicate detection, we have groups of similar/identical images. Users need guidance on which version to keep. Currently, the primary image is selected by `quality_score` from analysis, but this doesn't account for all quality factors.

## Current State

**Database Schema:**
- `images.quality_score` - Overall quality from analysis (0.0-1.0)
- `images.width`, `images.height` - Dimensions
- `images.file_size` - Size in bytes
- `images.file_extension` - Format (.jpg, .raw, .dng, etc.)
- `duplicate_groups.primary_image_id` - Selected as highest quality_score
- `duplicate_members.similarity_score` - How similar to primary (0-100)

## Quality Ranking Criteria

Images should be ranked by multiple factors in priority order:

### 1. File Format (Weight: 40%)
Priority order:
1. **RAW formats** (highest): `.dng`, `.raw`, `.cr2`, `.nef`, `.arw`, `.orf`, `.rw2`
2. **Lossless formats**: `.png`, `.tiff`, `.tif`, `.bmp`
3. **High-quality lossy**: `.jpg`, `.jpeg` with quality > 90
4. **Standard lossy**: `.jpg`, `.jpeg` with quality 70-90
5. **Low-quality lossy**: `.jpg`, `.jpeg` with quality < 70
6. **Web formats** (lowest): `.webp`, `.heic`, `.avif`

### 2. Resolution (Weight: 30%)
- Megapixels = width × height
- Higher resolution = better quality
- Threshold: Differences < 5% considered equal

### 3. File Size (Weight: 20%)
- For same format/resolution, larger = less compression
- Threshold: Differences < 10% considered equal

### 4. Sharpness/Quality Score (Weight: 10%)
- Use existing `quality_score` from image analysis
- Laplacian variance-based sharpness metric

## Proposed Algorithm

```python
def calculate_quality_score(image: dict) -> float:
    """
    Calculate comprehensive quality score (0-100).

    Args:
        image: Dict with keys: file_extension, width, height,
               file_size, quality_score

    Returns:
        Quality score from 0-100 (higher is better)
    """
    score = 0.0

    # 1. Format score (0-40 points)
    format_scores = {
        # RAW formats
        '.dng': 40, '.raw': 40, '.cr2': 40, '.nef': 40,
        '.arw': 40, '.orf': 40, '.rw2': 40, '.raf': 40,
        # Lossless
        '.png': 35, '.tiff': 35, '.tif': 35, '.bmp': 30,
        # Lossy (will adjust based on quality)
        '.jpg': 25, '.jpeg': 25,
        # Web formats
        '.webp': 20, '.heic': 20, '.avif': 20,
    }
    ext = image['file_extension'].lower()
    score += format_scores.get(ext, 15)

    # 2. Resolution score (0-30 points)
    megapixels = (image['width'] * image['height']) / 1_000_000
    # Logarithmic scale: 1MP=15, 10MP=25, 50MP=30
    resolution_score = min(30, 15 + (math.log10(megapixels + 1) * 8))
    score += resolution_score

    # 3. File size score (0-20 points)
    # Normalize by megapixels to get bytes/pixel
    if megapixels > 0:
        bytes_per_megapixel = image['file_size'] / megapixels
        # 1-10 MB/MP typical range
        size_score = min(20, (bytes_per_megapixel / 10_000_000) * 20)
        score += size_score

    # 4. Sharpness/quality score (0-10 points)
    if image.get('quality_score'):
        score += image['quality_score'] * 10

    return round(score, 2)
```

## Database Changes

### Option A: Pre-compute and store (Recommended)

Add column to `images` table:
```sql
ALTER TABLE images ADD COLUMN composite_quality_score FLOAT;
CREATE INDEX idx_images_composite_quality ON images(composite_quality_score);
```

**Benefits:**
- Fast sorting/filtering
- Can update scores when algorithm improves
- Queryable for analytics

### Option B: Compute on-demand

Calculate scores in application layer when needed.

**Benefits:**
- No schema changes
- Always uses latest algorithm

**Drawbacks:**
- Slower for large result sets
- Can't use in SQL ORDER BY

**Recommendation:** Use Option A

## Implementation Phases

### Phase 1: Add Quality Calculation Function
- [ ] Create `vam_tools/analysis/quality_scorer.py`
- [ ] Implement `calculate_quality_score()` function
- [ ] Add unit tests with known good/bad image comparisons
- [ ] Add migration for `composite_quality_score` column

### Phase 2: Batch Compute Scores
- [ ] Create script to compute scores for all existing images
- [ ] Add progress tracking for large catalogs
- [ ] Update `duplicate_groups.primary_image_id` based on new scores

### Phase 3: Integrate with Analysis Pipeline
- [ ] Update `analyze_catalog_task` to compute composite scores
- [ ] Store scores during image insertion

### Phase 4: UI Integration
- [ ] Add "Keep Best" button in duplicate viewer
- [ ] Show quality scores and breakdown in UI
- [ ] Visual indicators (RAW badge, resolution, etc.)
- [ ] Bulk action: "Keep best in all groups"

### Phase 5: Automatic Resolution (Optional)
- [ ] Preference setting: auto-delete lower quality duplicates
- [ ] Safety threshold: only auto-delete if score difference > 20
- [ ] Confirmation UI with preview

## Example Use Cases

### Case 1: RAW + JPEG of same image
```
Image A: .CR2, 24MP, 25MB → Score: 40 + 25 + 12 + 8 = 85
Image B: .JPG, 24MP, 4MB  → Score: 25 + 25 + 3 + 8 = 61
Decision: Keep A (RAW)
```

### Case 2: Different resolutions
```
Image A: .JPG, 48MP, 12MB → Score: 25 + 28 + 10 + 8 = 71
Image B: .JPG, 12MP, 3MB  → Score: 25 + 22 + 5 + 8 = 60
Decision: Keep A (higher resolution)
```

### Case 3: Same format, different compression
```
Image A: .JPG, 24MP, 12MB (low compression) → Score: 25 + 25 + 18 + 8 = 76
Image B: .JPG, 24MP, 2MB (high compression) → Score: 25 + 25 + 3 + 8 = 61
Decision: Keep A (less compression)
```

## Testing Strategy

1. **Unit tests:** Known image pairs with expected winner
2. **Integration tests:** Duplicate groups with verified best image
3. **Real-world validation:** Sample 100 duplicate groups, manual review

## Success Metrics

- 90%+ of automatically selected "best" images match human judgment
- Users can confidently use "Keep best in all groups" bulk action
- Reduced manual review time for duplicates

## Future Enhancements

1. **ML-based quality assessment**
   - Train model on user preferences
   - Detect blur, noise, proper exposure

2. **Configurable weights**
   - Let users adjust format vs. resolution priorities
   - Per-catalog preferences

3. **Metadata consideration**
   - EXIF quality settings
   - Camera model (better sensor = better quality)
   - Edit history (edited version might be preferred)

## References

- Session notes: 29% burst contamination removed
- Consensus filtering: aHash AND dHash ≤ 5 bits
- 393 groups, 436 duplicates after cleanup
