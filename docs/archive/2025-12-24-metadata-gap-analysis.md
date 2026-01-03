# Metadata Gap Analysis - Lumina
**Generated**: 2025-12-24
**Total Images**: 96,228

## Summary

| Metadata Type | Coverage | Count | Gap | Priority |
|--------------|----------|-------|-----|----------|
| Perceptual Hashes (dhash/ahash/whash) | 99.99% | 96,222 | 6 | ✅ Complete |
| CLIP Embeddings | 99.13% | 95,390 | 838 | ✅ Nearly Complete |
| Auto-Tag Descriptions | 99.99% | 96,222 | 6 | ✅ Complete |
| GPS Coordinates | 32.36% | 31,133 | 65,095 | ⚠️ Expected (not all images have GPS) |
| Geohash (new metadata field) | 0.00% | 0 | 96,228 | 🔴 **MISSING** |
| Geohash Columns (legacy) | 32.34% | 31,131 | - | ✅ Populated for GPS images |
| Camera Make/Model | 93.09% | 89,578 | 6,650 | ✅ Good |
| Date Taken (dates.selected_date) | 100.00% | 96,228 | 0 | ✅ Complete |
| Quality Score | 0.00% | 0 | 96,228 | 🔴 **MISSING** |
| Thumbnails | 0.00% | 0 | 96,228 | 🔴 **MISSING** |
| Burst Detection | 49.21% | 47,351 | - | ⚠️ Partial (7,966 bursts found) |
| Tags (auto/manual) | 47.61% | 45,810 | 50,418 | ⚠️ Partial |

## Duplicate Detection Status

- **Duplicate Groups Found**: 0
- **Images in Duplicate Groups**: 0
- **Status**: ❌ Not run or no duplicates found

## Date Coverage Details

- **Total with Dates**: 96,228 (100%)
- **High Confidence (≥95%)**: 89,128 (92.6%)
- **From EXIF**: 90,309 (93.8%)
- **Suspicious Dates**: 621 (0.6%)
- **Status**: ✅ Fully populated in `dates.selected_date`

## Critical Missing Metadata

### 1. Geohash in Metadata (0% coverage)
- **Current State**: New `metadata.geohash` field not populated
- **Impact**: New geohash-based burst detection won't work optimally
- **Note**: Legacy columns (geohash_4/6/8) are populated for GPS images
- **Action Required**: Run metadata migration to populate from GPS coordinates
- **Command**: Database migration or re-analyze with new metadata extractor

### 2. Thumbnails (0% coverage)
- **Current State**: `thumbnail_path` is NULL for all images
- **Impact**: Slow UI performance, full images loaded for previews
- **Action Required**: Run thumbnail generation job
- **Priority**: **HIGH** for performance

### 3. Quality Score (0% coverage)
- **Current State**: `quality_score` column is NULL for all images
- **Impact**: Cannot rank/filter by quality, no quality-based recommendations
- **Action Required**: Run quality analysis job (if implemented)
- **Note**: Check if quality scoring is implemented in current analyze job

## Tag Coverage Details

- **Images with Tags**: 45,810 (47.61%)
- **Total Tag Assignments**: 74,489
- **Unique Tags**: 35
- **Top Tags**: birds (8,523), outdoor (7,421), candid (6,270)
- **Auto vs Manual**: All show source=manual (may need source tracking fix)

## Recommendations

### Immediate Actions (After current analyze completes)

1. **Wait for Current Analyze Job** ✅
   - Status: All 96,228 images in "analyzing" state
   - May populate: quality_score (if implemented)
   - Note: Dates already populated (100% coverage)
   - Let this complete before running other jobs

2. **Run Thumbnail Generation** 🔴 **HIGH PRIORITY**
   ```bash
   # Generate thumbnails for all images
   POST /api/jobs/thumbnails
   ```
   - Impact: Major UI performance improvement
   - Safe to run: No dependency on other metadata

3. **Populate Geohash Metadata Field** 🔴 **HIGH PRIORITY**
   ```sql
   -- Migration to populate metadata.geohash from gps_latitude/gps_longitude
   UPDATE images
   SET metadata = metadata || jsonb_build_object('geohash',
       -- Generate from gps_latitude/longitude
   )
   WHERE metadata->>'gps_latitude' IS NOT NULL;
   ```
   - Required for: Improved burst detection performance
   - Safe to run: After analyze completes

4. **Run Duplicate Detection** 🔴 **CRITICAL MISSING**
   ```bash
   POST /api/jobs/detect-duplicates
   {
     "catalog_id": "...",
     "similarity_threshold": 5
   }
   ```
   - Dependency: Perceptual hashes (✅ 99.99% coverage)
   - Impact: Find and merge duplicate images
   - Safe to run: After analyze completes

5. **Re-run Burst Detection** ⚠️ **RECOMMENDED**
   ```bash
   POST /api/jobs/detect-bursts
   {
     "catalog_id": "..."
   }
   ```
   - Why: New geohash-based location matching
   - Dependency: Populate geohash metadata first
   - Current: 47,351 images in 7,966 bursts (may improve with geohash)

6. **Complete Auto-Tagging** ⚠️ **OPTIONAL**
   - Currently: 47.61% coverage (45,810 / 96,228)
   - Missing: 50,418 images
   - Run: Auto-tag job for remaining images
   - Priority: LOW (already good coverage)

### Quality Analysis Investigation

Check if quality scoring is part of analyze job:
```bash
# Check logs for quality score calculation
docker compose logs celery-worker | grep -i quality

# Check if analyze job includes quality
# Review vam_tools/jobs/parallel_scan.py or analyze job code
```

If not implemented, quality_score will remain NULL (acceptable for now).

## Expected Metadata After Analyze

Once the current analyze job completes (all 96,228 images):

✅ **Already populated** (from previous scans):
- `dates.selected_date` - ✅ 100% coverage, 92.6% high confidence
- `dhash/ahash/whash` - ✅ 99.99% coverage
- `clip_embedding` - ✅ 99.13% coverage
- `description` - ✅ 99.99% coverage (auto-tags)

⚠️ **May be populated** (if implemented in analyze):
- `quality_score` - Check analyze job code

❌ **Will still be missing**:
- `thumbnail_path` - Requires separate thumbnail job
- `metadata.geohash` - Requires migration or metadata update
- Duplicate groups - Requires duplicate detection job

## Job Execution Order

```
1. ✅ Let current ANALYZE complete (in progress)
   ↓
2. 🔴 RUN THUMBNAIL generation (no dependencies)
   ↓
3. 🔴 POPULATE geohash metadata (SQL migration or metadata update)
   ↓
4. 🔴 RUN DUPLICATE DETECTION (depends on perceptual hashes ✅)
   ↓
5. ⚠️ RE-RUN BURST DETECTION (depends on geohash metadata)
   ↓
6. ⚠️ COMPLETE AUTO-TAGGING (optional, already 47% done)
```

## Notes

- **GPS Coverage**: 32.36% is expected - not all images have GPS EXIF data
- **Camera Info**: 93.09% is good - some images lack camera metadata
- **CLIP Embeddings**: 99.13% excellent - only 838 images missing (likely errors)
- **Status**: All images currently "analyzing" - wait for completion
- **Duplicates**: 0 groups means either not run or no duplicates found (unlikely with 96k images)
