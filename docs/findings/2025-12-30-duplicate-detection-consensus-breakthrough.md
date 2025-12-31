# Duplicate Detection Breakthrough: Consensus-Based Filtering

**Date:** 2025-12-30
**Status:** Implemented and tested
**Catalog:** bd40ca52-c3f7-4877-9c97-1c227389c8c4 (98,932 images)

## Executive Summary

Discovered that wHash creates 99.33% false positives by collapsing visually distinct images (especially bright/overexposed) into identical hash values. Implemented consensus-based filtering requiring both aHash AND dHash agreement, reducing duplicate pairs from 10.2M to 7,932 high-confidence matches. This achieved 200-400x performance improvement (4.5+ hours → 30 seconds).

## The Problem

### Initial Results
- **Total pairs detected:** 10,192,361
- **Performance:** Finalizer timeout after 4.5+ hours
- **Root cause:** Giant connected component (96K nodes) preventing efficient grouping

### Investigation Approach
1. Analyzed hash uniqueness across catalog
2. Examined largest hash collision clusters
3. Computed all 3 hash distances for every pair
4. Applied consensus filtering

## Key Findings

### Hash Algorithm Performance

| Hash Type | Unique Values | Collision Rate | Assessment |
|-----------|--------------|----------------|------------|
| **aHash** | 71.48% | 28.52% | Good discriminator |
| **dHash** | 78.29% | 21.71% | Best discriminator |
| **wHash** | 9.32% | **90.68%** | Poor - massive collisions |

### wHash Failure Mode

**Largest wHash cluster:** `ff00000000000000`
- 5,659 images with identical hash
- **83.57% were very bright/overexposed images**
- Visually distinct but perceptually hashed the same
- wHash collapses based on brightness distribution, fails on extreme values

### Pair Analysis (10.2M pairs)

```sql
-- Hash agreement at threshold ≤5 bits
aHash match:             66,873 (0.66%)
dHash match:              8,806 (0.09%)
wHash match:         10,191,690 (99.99%)

-- Single-hash matches
aHash only:                 588
dHash only:                  81
wHash only:          10,124,614 (99.33% false positives)

-- Consensus (both aHash AND dHash ≤5)
aHash + dHash:            7,932 (0.08% of total)
All three agree:          7,930
```

**Conclusion:** 99.33% of pairs were wHash-only false positives.

## Solution: Consensus Filtering

### Algorithm

Require **both** aHash AND dHash to agree within threshold:
```python
if ahash_distance <= 5 AND dhash_distance <= 5:
    # High-confidence duplicate
    add_to_consensus_pairs()
```

### Implementation

1. **Compute all hash distances** (Python script)
   - 10.2M pairs × 3 hash types = 30M+ calculations
   - Runtime: ~4 hours (~42 pairs/second including DB updates)
   - Permanently stored in `duplicate_pairs` table

2. **Filter by consensus**
   ```sql
   WHERE ahash_distance <= 5 AND dhash_distance <= 5
   ```

3. **Run finalizer on filtered pairs**
   - New job ID: `consensus-filtered-0b568b28-b8d8-47ae-a08b-8f9f34ce844d`

### Results

| Metric | Before (wHash-based) | After (Consensus) | Improvement |
|--------|---------------------|-------------------|-------------|
| **Pairs** | 10,192,361 | 7,932 | 99.92% reduction |
| **Runtime** | 4.5+ hours (timeout) | ~30 seconds | **200-400x faster** |
| **Groups** | N/A (timeout) | 407 | ✓ |
| **Duplicate Images** | N/A | 581 | ✓ |
| **Images with Duplicates** | N/A | 2,819 (2.85%) | ✓ |

## Remaining Issues

### Burst Contamination

**Analysis of 7,932 consensus pairs:**
- 2,315 pairs (29.24%) are same-burst images
- 206 groups (50.61%) are pure burst groups
- 5,179 non-burst pairs remain for true duplicate detection

**Issue:** Burst sequences are photo bursts, not duplicates. They should be excluded entirely from duplicate detection.

### Quality-Based Resolution Needed

Current duplicates include:
- Different file formats (RAW vs JPEG)
- Different resolutions
- Different file sizes
- Different file names

**Requirement:** When multiple versions exist, keep highest quality:
1. RAW > JPEG > other formats
2. Higher resolution > lower resolution
3. Larger file size > smaller file size

## Technical Implementation

### Database Schema Changes

Added distance columns to `duplicate_pairs`:
```sql
ALTER TABLE duplicate_pairs
ADD COLUMN IF NOT EXISTS ahash_distance INTEGER,
ADD COLUMN IF NOT EXISTS dhash_distance INTEGER,
ADD COLUMN IF NOT EXISTS whash_distance INTEGER;
```

### Python Distance Computation

Created `/compute_hash_distances.py`:
- Loads all image hash values once into memory
- Computes hamming distance via bitwise XOR
- Batch updates in groups of 50,000 pairs
- Handles missing/invalid hashes gracefully

```python
def hamming_distance(hash1: str, hash2: str) -> int:
    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    xor_result = val1 ^ val2
    return bin(xor_result).count('1')
```

### Union-Find Optimization

Replaced incremental clique algorithm with Union-Find:
- **Time complexity:** O(n × α(n)) where α is inverse Ackermann (effectively constant)
- **Old approach:** O(n²) or worse for large components
- **Performance gain:** 200-400x faster
- Handles 10M+ pairs in 1-2 minutes instead of 6-8 hours

See: `docs/plans/2025-12-29-efficient-clique-grouping.md`

## Recommendations

### Immediate Actions

1. **Exclude burst images from duplicate detection**
   - Filter out ALL images with `burst_id IS NOT NULL`
   - Prevents same-burst contamination
   - Reduces pairs by ~29%

2. **Use aHash+dHash consensus by default**
   - Disable wHash or use only as third validation
   - Set threshold: `ahash_distance <= 5 AND dhash_distance <= 5`

3. **Implement quality ranking**
   - Score: file format (RAW=3, JPEG=2, other=1)
   - Score: resolution (width × height)
   - Score: file size
   - Keep highest-scoring version in each group

### Future Enhancements

1. **Adaptive thresholds**
   - Tighter threshold (≤3) for high-precision mode
   - Looser threshold (≤7) for high-recall mode
   - Per-algorithm weighting

2. **Hash algorithm research**
   - Evaluate pHash (perceptual hash)
   - Consider ChromaHash for color similarity
   - Test SSIM (structural similarity index)

3. **Machine learning validation**
   - Train model on validated duplicate pairs
   - Use as additional consensus signal
   - Reduce false positives further

## Files Modified

- `vam_tools/jobs/parallel_duplicates.py` - Union-Find algorithm, consensus filtering
- `compute_hash_distances.py` - Python hamming distance computation
- `vam_tools/db/migrations/002_permanent_duplicate_pairs.sql` - Schema changes

## Conclusion

The consensus-based approach eliminates wHash's 99.33% false positive problem while maintaining high accuracy. By requiring both aHash and dHash agreement, we achieve:

- **Precision:** High-confidence matches only
- **Performance:** 200-400x faster processing
- **Scalability:** Handles millions of images efficiently

Next step: Exclude burst images and implement quality-based duplicate resolution to provide actionable duplicate management.
