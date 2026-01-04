# Thumbnail Data Loss Investigation
**Investigation Date**: 2025-12-24
**Audit Period**: 2025-12-09 to present

## Summary

✅ **Audit records exist**: 120,770 records from Dec 9 onwards
❌ **Thumbnails were generated but are now missing**
⚠️ **Data loss occurred through database cleanup operation**

## Timeline of Events

### December 9, 2025 (14:02 UTC)
- **Audit logging begins** - First audit record created
- System has existing images (already loaded before audit started)

### December 10, 2025 (04:24:16 UTC)
- **MASS DELETION EVENT**: 111,082 images deleted in single operation
  - Deleted by: `pg` user (PostgreSQL superuser)
  - Application: None (direct SQL operation)
  - Timestamp: All 111,082 deletions at exact same millisecond
  - Images had NO metadata: no thumbnails, no quality scores, no dhash values

### December 10, 2025 (12:56 UTC)
- **Catalog updated** - `Default Catalog` updated

### December 23, 2025 (17:11 UTC)
- **Multiple thumbnail jobs completed successfully** ✅
  - 3 separate `generate_thumbnails` jobs with SUCCESS status
  - These jobs were later deleted from jobs table (cleanup)
  - Evidence found in audit_log.old_data

### December 23, 2025 (17:11-17:12 UTC)
- **Jobs table cleanup**: 69 jobs deleted
  - Includes the successful thumbnail generation jobs
  - Part of routine job history cleanup

### December 24-25, 2025
- Multiple analyze, duplicate detection, burst detection jobs
- Current state: 96,228 images with NO thumbnails

## Key Findings

### 1. Thumbnails Were Generated
**Evidence from audit_log (deleted jobs table records):**
```
job_type: generate_thumbnails | status: SUCCESS | deleted_at: 2025-12-23 17:11:17
job_type: generate_thumbnails | status: SUCCESS | deleted_at: 2025-12-23 17:11:15
job_type: generate_thumbnails | status: SUCCESS | deleted_at: 2025-12-23 17:11:04
```

These jobs completed successfully but were later removed from the jobs table during cleanup.

### 2. Images Were Bulk Deleted
**From audit_log:**
- **Count**: 111,082 images
- **When**: 2025-12-10 04:24:16.940823
- **Who**: `pg` user (database admin)
- **How**: Single bulk DELETE operation (all same timestamp)
- **State**: Images had empty metadata (no thumbnails, no hashes, no quality)

This appears to be a cleanup of duplicate/incomplete records.

### 3. Current Images Not in Audit Log
**Critical finding**: Current 96,228 images have NO INSERT records in audit_log

**Why**: Images were loaded BEFORE audit logging was enabled (before Dec 9)

**Implication**: These images never had thumbnails generated - they're from a different dataset than the one that had thumbnails.

### 4. Missing INSERT Audit Records
```sql
SELECT COUNT(*) FROM audit_log
WHERE table_name = 'images' AND operation = 'INSERT';
-- Result: 0

SELECT COUNT(*) FROM audit_log
WHERE table_name = 'images' AND operation = 'DELETE';
-- Result: 111,082
```

**Only deletions are audited, no insertions.**

## What Happened

### Most Likely Scenario:

1. **Pre-Dec 9**: Original image set loaded (~207k images = 111k + 96k)
2. **Pre-Dec 9**: Audit triggers not yet enabled
3. **Dec 9**: Audit logging enabled
4. **Dec 10 04:24**: Cleanup job deletes 111,082 "bad" images (no metadata)
5. **Dec 23**: Thumbnails generated for remaining images
6. **Unknown date**: Images table rebuilt/reloaded with 96,228 images
7. **Dec 23**: Jobs table cleaned up (thumbnail job records deleted)
8. **Dec 24-25**: Current state - 96,228 images without thumbnails

### Alternative Scenario:

1. Thumbnails were generated for a different catalog
2. Catalog was deleted/reset
3. New catalog created with same images but no thumbnails

## Data Loss Root Cause

**The thumbnails weren't "lost" - they were never generated for the current dataset.**

The current 96,228 images either:
- Were loaded after the thumbnail generation
- Are from a catalog rebuild that didn't preserve thumbnails
- Never had the thumbnail_path column populated after insertion

## Evidence Gaps

❌ **No INSERT audit records**: Can't determine when current images were added
❌ **No UPDATE audit records for images**: Can't see if thumbnails were ever set
❌ **Audit started Dec 9**: Events before that date are not recorded

## Recommendations

### Immediate Actions

1. **Enable UPDATE auditing for images table**
   - Currently only INSERT/DELETE trigger audit_trigger_func()
   - Need to track thumbnail_path changes

2. **Run thumbnail generation NOW**
   - Current 96,228 images have never had thumbnails
   - No data to "recover" - must generate fresh

3. **Document catalog operations**
   - When was current image set loaded?
   - Was there a catalog rebuild?
   - Why were 111k images deleted on Dec 10?

### Prevention

1. **Add application_name to all operations**
   - Currently shows NULL for bulk operations
   - Set in connection string: `?application_name=lumina`

2. **Preserve critical job records**
   - Don't delete SUCCESS thumbnail jobs
   - Or extract completion info before deleting

3. **Track image table mutations**
   - Add UPDATE to audit trigger
   - Log bulk operations with metadata

## Questions for User

1. **Do you remember a catalog rebuild around Dec 10?**
2. **Were you running multiple catalogs?**
3. **Did you manually delete images or run a cleanup script?**
4. **Should we preserve all job records, or is cleanup OK?**

## Current State

```
Total Images:        96,228
Thumbnails:          0 (0.00%)
Last Thumbnail Job:  Dec 23 (deleted from jobs table)
Audit Coverage:      Dec 9 onwards
Image INSERTs:       Not audited
```

## Action Required

🔴 **Generate thumbnails for all 96,228 images**

The data wasn't "lost" - it was never created for this image set.
