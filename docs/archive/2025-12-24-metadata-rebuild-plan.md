# Metadata Rebuild Plan
**Created**: 2025-12-24
**Catalog**: Default Catalog (96,228 images)

## Current State

✅ **Complete**:
- Dates (100% - all from EXIF)
- Perceptual hashes (99.99% - dhash/ahash/whash)
- CLIP embeddings (99.13%)
- Camera metadata (93%)
- GPS coordinates (32% - not all images have GPS)

⏳ **In Progress**:
- Analyze job running (all 96,228 images in "analyzing" state)

🔴 **Missing**:
- Thumbnails (0%)
- Duplicate detection (0 groups found)
- Geohash metadata field (0%)
- Quality scores (0%)

⚠️ **Partial**:
- Burst detection (49% - 47,351 images in 7,966 bursts)
- Tags (48% - 45,810 images tagged)

## Execution Plan

### Phase 1: Let Current Jobs Complete ⏳

**Status**: Wait for analyze to finish
- **What**: Current analyze job processing all 96,228 images
- **Why**: May populate quality_score column
- **Duration**: Monitor job status
- **Action**: Wait before proceeding

### Phase 2: Generate Thumbnails 🔴 **HIGH PRIORITY**

**Status**: Queued ✅
- **What**: Create thumbnails for all 96,228 images
- **Why**: Critical for UI performance (loading full images is very slow)
- **API**: `POST /api/jobs/thumbnails`
- **Dependencies**: None
- **Duration**: ~2-4 hours for 96k images
- **Priority**: **CRITICAL** - Major performance impact

### Phase 3: Populate Geohash Metadata 🔴 **REQUIRED**

**Status**: Not started
- **What**: Populate `metadata.geohash` field from existing GPS coordinates
- **Why**: Required for improved burst detection performance
- **Method**: Database migration or metadata update
- **Dependencies**: None (GPS already extracted)
- **Duration**: ~5 minutes (SQL update)
- **Priority**: **HIGH** - Needed for burst detection

**SQL Migration**:
```sql
-- Update images with GPS to add geohash
UPDATE images
SET metadata = metadata || jsonb_build_object(
    'geohash',
    -- Generate precision-7 geohash from gps_latitude/longitude
    encode_geohash(
        (metadata->>'gps_latitude')::double precision,
        (metadata->>'gps_longitude')::double precision,
        7
    )
)
WHERE metadata->>'gps_latitude' IS NOT NULL
  AND metadata->>'gps_longitude' IS NOT NULL;
```

**Alternative**: Re-run metadata extraction with updated extractor (slower)

### Phase 4: Detect Duplicates 🔴 **CRITICAL**

**Status**: Not started
- **What**: Find duplicate/similar images across 96,228 images
- **Why**: 96k images very likely have duplicates
- **API**: `POST /api/jobs/detect-duplicates`
  ```json
  {
    "catalog_id": "bd40ca52-c3f7-4877-9c97-1c227389c8c4",
    "similarity_threshold": 5
  }
  ```
- **Dependencies**: Perceptual hashes ✅ (99.99% coverage)
- **Duration**: ~1-2 hours (depends on threshold)
- **Priority**: **HIGH** - Critical for deduplication

### Phase 5: Re-run Burst Detection ⚠️ **RECOMMENDED**

**Status**: Not started
- **What**: Detect burst sequences with improved geohash performance
- **Why**: New geohash-based location matching is faster
- **API**: `POST /api/jobs/detect-bursts`
  ```json
  {
    "catalog_id": "bd40ca52-c3f7-4877-9c97-1c227389c8c4"
  }
  ```
- **Dependencies**:
  - ✅ GPS coordinates (32% have GPS)
  - 🔴 Geohash metadata (Phase 3)
- **Duration**: ~30-60 minutes
- **Priority**: **MEDIUM** - Already 49% detected, this improves accuracy
- **Note**: Must run AFTER Phase 3 (geohash population)

### Phase 6: Complete Auto-Tagging ⚠️ **OPTIONAL**

**Status**: Not started
- **What**: Tag remaining 50,418 images (currently 48% coverage)
- **Why**: Improve search/filtering capabilities
- **API**: `POST /api/jobs/auto-tag`
- **Dependencies**: CLIP embeddings ✅ (99.13% coverage)
- **Duration**: ~2-3 hours for 50k images
- **Priority**: **LOW** - Already good coverage, not critical

### Phase 7: Quality Scoring 🔴 **IF AVAILABLE**

**Status**: Check if implemented
- **What**: Calculate quality scores for all images
- **Why**: Enable quality-based filtering and recommendations
- **Action**:
  1. Check if analyze job populates quality_score
  2. If not, check if quality scoring job exists
  3. Run if available
- **Dependencies**: None
- **Duration**: Unknown (depends on implementation)
- **Priority**: **MEDIUM** - Nice to have but not critical

## Execution Timeline

```
Day 1 (Today):
├─ Phase 1: Wait for ANALYZE to complete (⏳ in progress)
└─ Phase 2: THUMBNAILS (queued ✅)
   Duration: 2-4 hours

Day 1 (After thumbnails):
├─ Phase 3: GEOHASH migration (5 minutes)
├─ Phase 4: DUPLICATE DETECTION (1-2 hours)
└─ Phase 5: BURST DETECTION (30-60 minutes)

Day 2 (Optional):
└─ Phase 6: AUTO-TAGGING (2-3 hours)
```

**Total estimated time**: ~5-8 hours of job processing

## Job Submission Order

**Run immediately (parallel safe)**:
```bash
# 1. Thumbnails (already queued ✅)
POST /api/jobs/thumbnails

# 2. While thumbnails running, populate geohash
# Execute SQL migration (see Phase 3)
```

**Run after thumbnails complete**:
```bash
# 3. Duplicate detection
POST /api/jobs/detect-duplicates
{
  "catalog_id": "bd40ca52-c3f7-4877-9c97-1c227389c8c4",
  "similarity_threshold": 5
}

# 4. Burst detection (wait for geohash migration first!)
POST /api/jobs/detect-bursts
{
  "catalog_id": "bd40ca52-c3f7-4877-9c97-1c227389c8c4"
}
```

**Run later (optional)**:
```bash
# 5. Auto-tagging for remaining images
POST /api/jobs/auto-tag
{
  "catalog_id": "bd40ca52-c3f7-4877-9c97-1c227389c8c4"
}
```

## Critical Path

To get fully functional catalog with key features:

```
ANALYZE (in progress)
  → THUMBNAILS (queued) ✅
    → GEOHASH migration (5 min)
      → DUPLICATES (1-2 hrs) ← CRITICAL
        → BURSTS (30-60 min)
```

**Minimum viable**: Thumbnails + Duplicates = ~3-6 hours
**Full rebuild**: All phases = ~5-8 hours

## Monitoring

**Check job status**:
```bash
# Via API
GET /api/jobs

# Via database
docker exec postgres psql -U pg -d lumina -c "
SELECT id, job_type, status, created_at
FROM jobs
ORDER BY created_at DESC
LIMIT 10;
"
```

**Check worker capacity**:
```bash
# Should see 6 workers with concurrency=2 (12 slots total)
docker compose logs celery-worker | grep -i "ready"
```

## Expected Final State

After all phases complete:

| Metadata | Coverage | Priority |
|----------|----------|----------|
| Dates | 100% ✅ | Complete |
| Hashes | 100% ✅ | Complete |
| CLIP | 99% ✅ | Complete |
| **Thumbnails** | **100%** ✅ | **NEW** |
| **Duplicates** | **~80-90%** | **NEW** |
| Geohash | 32% ✅ | Complete (for GPS images) |
| **Bursts** | **~60-70%** | **IMPROVED** |
| Tags | 70-80% | Improved |
| Quality | TBD | If implemented |

## Notes

- **GPS Coverage (32%)**: Expected - not all images have GPS EXIF data
- **Worker Utilization**: All jobs run in parallel via Celery workers
- **Safe to interrupt**: Jobs use batch system, can resume
- **Geohash migration**: Can run as SQL (fast) or via job (slow but safer)
- **Progress tracking fix**: Applied today - progress should update smoothly

## Recovery from Failures

If any job fails:
1. Check logs: `docker compose logs celery-worker`
2. Check job status in UI or database
3. Failed batches can be retried via job recovery
4. Safe to re-run - jobs check for existing metadata

## Geohash Migration Script

**Option A: Fast SQL (5 minutes)**
```sql
-- Requires pygeohash Python package or PostgreSQL geohash extension
-- If not available, use Option B

-- Check current geohash coverage
SELECT COUNT(*) FILTER (WHERE metadata->>'geohash' IS NOT NULL) as has_geohash
FROM images;

-- Update (requires geohash function)
-- This is a placeholder - actual implementation needed
```

**Option B: Via Job (slower, ~30 min)**
- Create metadata update job
- Processes each image with GPS
- Generates geohash from coordinates
- Updates metadata.geohash field

**Recommendation**: Implement Option B as a job for safety and progress tracking
