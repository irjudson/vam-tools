# Catalog Health & Recovery Plan
**Generated:** 2025-12-25
**Total Images:** 98,889

## Current Status

### ✅ Complete (Good Coverage)
- **Dates:** 100% - All images have selected_date
- **Perceptual Hashes:** 97.3% - dhash/ahash/whash present for 96,222 images
- **CLIP Embeddings:** 96.5% - Present for 95,390 images
- **GPS Data:** 33.9% - 33,486 images have GPS coordinates (expected for camera mix)
- **Burst Detection:** Complete - 7,966 burst groups containing 48,024 images

### ❌ Missing (Needs Action)
- **Thumbnails:** 0% - No thumbnails generated
- **Quality Scores:** 0% - No quality analysis run
- **Duplicate Detection:** 0 groups - Not yet run (needs hashes first)
- **Processing Flags:** 0% - Not tracked (images loaded before flag system implemented)
- **Perceptual Hashes:** 2,667 images still need hashing (2.7%)
- **CLIP Embeddings:** 3,499 images need embeddings (3.5%)
- **Geohash:** Only 33,486 have geohash columns (should match GPS count - needs verification)

## Recommended Jobs (Priority Order)

### 1. Generate Thumbnails (HIGH PRIORITY)
**Reason:** UI depends on thumbnails for browsing
**Impact:** 98,889 images (100%)
**Command:**
```bash
docker compose exec web vam-web run-job thumbnails <catalog-id>
```
**Estimated Time:** ~2-3 hours (depending on hardware)

### 2. Finish Perceptual Hashing
**Reason:** Required for duplicate detection
**Impact:** 2,667 images (2.7%)
**Note:** This should have been done during analyze, investigate why these were skipped
**Command:**
```bash
# Re-run analyze with --recompute-hashes flag for missing images
docker compose exec web vam analyze <catalog-dir> --recompute-hashes
```

### 3. Run Duplicate Detection
**Reason:** Find and manage duplicate images
**Impact:** All images with hashes (~96,222)
**Prerequisites:** Perceptual hashes must be present
**Command:**
```bash
docker compose exec web vam-web run-job duplicates <catalog-id> --threshold 5
```
**Estimated Time:** ~30-60 minutes (depends on similarity threshold)

### 4. Generate Quality Scores
**Reason:** Helps identify best images in bursts/duplicates
**Impact:** 98,889 images (100%)
**Command:**
```bash
docker compose exec web vam-web run-job quality <catalog-id>
```
**Estimated Time:** ~1-2 hours

### 5. Complete CLIP Embeddings
**Reason:** Enable semantic search
**Impact:** 3,499 images (3.5%)
**Command:**
```bash
# Re-run embedding generation for missing images
docker compose exec web vam-web run-job clip <catalog-id>
```

### 6. Verify Geohash Coverage (Optional)
**Reason:** Ensure all GPS images have geohash for spatial queries
**Impact:** Should be 33,486 images
**Command:**
```bash
# Run geohash migration script to verify/fix
docker compose exec web python3 migrate_geohash.py
```

## Execution Plan

### Option A: Run All Jobs Sequentially (Recommended)
```bash
# 1. Thumbnails (critical for UI)
docker compose exec web vam-web run-job thumbnails <catalog-id>

# 2. Finish hashes (prerequisite for duplicates)
docker compose exec web vam analyze <catalog-dir> --recompute-hashes

# 3. Detect duplicates
docker compose exec web vam-web run-job duplicates <catalog-id> --threshold 5

# 4. Quality scores (helps with burst/duplicate review)
docker compose exec web vam-web run-job quality <catalog-id>

# 5. Complete embeddings
docker compose exec web vam-web run-job clip <catalog-id>
```

### Option B: Parallel Execution (Faster, More Resource Intensive)
```bash
# Start thumbnails in background
docker compose exec -d web vam-web run-job thumbnails <catalog-id>

# Start quality scoring in parallel
docker compose exec -d web vam-web run-job quality <catalog-id>

# Wait for both to complete, then run duplicate detection
```

## Why Processing Flags Show 0%

The `processing_flags` system was added after your images were loaded. The actual processing WAS done (evidenced by the data), but the flags weren't updated because:

1. Images loaded before flag system existed
2. Jobs completed before flag tracking implemented
3. Scanner didn't backfill flags for existing images

**This doesn't affect functionality** - it's just tracking metadata. The actual work (hashes, embeddings, etc.) is present in the data.

## Expected Results After Completion

After running all recommended jobs:
- **Thumbnails:** 100% (all images)
- **Quality Scores:** 100% (all images)
- **Perceptual Hashes:** 100% (all images)
- **CLIP Embeddings:** 100% (all images)
- **Duplicate Groups:** ~5,000-15,000 groups (typical for 99k images)
- **Total Processing Time:** 4-6 hours

## Monitoring Progress

Monitor job progress via:
```bash
# Watch job status
docker compose exec web vam-web jobs list

# Check specific job
docker compose exec web vam-web jobs status <job-id>

# View logs
docker compose logs -f web
```

## Notes

- All jobs are incremental and can be safely re-run
- Jobs will skip already-processed images
- Cancel jobs with: `docker compose exec web vam-web jobs cancel <job-id>`
- Database is automatically backed up before major operations
