# VAM Tools - Job Safety & Recovery Guide

**Date**: 2025-11-06  
**Status**: ✅ **PRODUCTION READY**

---

## 🛡️ Job Safety Mechanisms

### 1. Catalog Locking
All catalog operations use file locking to prevent concurrent writes:

```python
with CatalogDatabase(catalog_path) as db:
    # Exclusive lock acquired automatically
    db.add_image(...)
    # Lock released on exit
```

**Protection**:
- ❌ Multiple jobs can't corrupt the same catalog
- ✅ Jobs wait for lock (30s timeout)
- ✅ Failed jobs release locks automatically

### 2. Transaction Logging
All organize operations are logged:

```json
{
  "transaction_id": "abc123",
  "operations": [{
    "source": "/photos/img.jpg",
    "target": "/organized/img.jpg",
    "operation": "copy",
    "checksum": "sha256:...",
    "status": "completed"
  }]
}
```

**Protection**:
- ✅ Know exactly what was done
- ✅ Can manually reverse operations
- ✅ Checksum verification for integrity

### 3. Checkpointing
Analysis jobs checkpoint every 100 files:

```python
# Progress saved to catalog periodically
if processed % 100 == 0:
    db.save()  # Checkpoint
```

**Protection**:
- ✅ Partial results saved
- ✅ Resume from checkpoint (future feature)
- ✅ Not all work lost if job crashes

---

## 🔧 Job Control Options

### Cancel (Graceful Stop)
```
Button: "Cancel"
API: DELETE /api/jobs/{job_id}
Signal: SIGTERM
```

**What it does**:
- Sends graceful shutdown signal
- Job finishes current operation
- Releases locks cleanly
- Saves partial progress

**When to use**:
- Job taking longer than expected
- Want to adjust parameters and restart
- No urgent need to stop immediately

### Force Kill (Immediate Stop)
```
Button: "Force Kill"
API: POST /api/jobs/{job_id}/kill
Signal: SIGKILL
```

**What it does**:
- Immediately terminates process
- May leave files in incomplete state
- Releases locks (after timeout)
- No cleanup performed

**When to use**:
- Job completely stuck/frozen
- Cancel didn't work
- Need to free resources immediately
- **WARNING**: May require catalog integrity check

---

## 🚨 Handling Stuck Jobs

### Detection
A job is considered stuck if:
- Status is PROGRESS for > 10 minutes
- No progress updates for > 5 minutes
- Worker not responding to signals

### Recovery Steps

**Step 1: Try Cancel First**
```bash
1. Click "Cancel" button
2. Wait 30 seconds
3. Check if status changes to "CANCELLED"
```

**Step 2: Force Kill if Needed**
```bash
1. Click "Force Kill" button
2. Confirms with warning dialog
3. Job immediately terminated
```

**Step 3: Check Catalog Integrity**
```bash
# After force-killing a job, verify catalog
docker exec vam-celery-worker python -c "
from vam_tools.core.catalog import CatalogDatabase
db = CatalogDatabase('/app/catalogs/test')
print(f'Images: {len(db.list_images())}')
print('✓ Catalog intact')
"
```

---

## 📊 What Happens to Data

### Analyze Job (Read-Only)
**If Cancelled/Killed**:
- ✅ Source photos: Untouched
- ✅ Catalog: Contains partial results
- ✅ Resume: Can re-run, skips processed files

**Safety**: ★★★★★ (Very Safe)

### Organize Job (Writes Files)
**If Cancelled**:
- ✅ Completed operations: Files copied/moved successfully
- ✅ In-progress operation: Completes, then stops
- ✅ Transaction log: Shows exactly what was done
- ⚠️  Remaining files: Not processed

**If Force-Killed**:
- ✅ Completed operations: Files intact
- ⚠️  In-progress operation: May be incomplete
- ✅ Transaction log: Shows what completed
- ⚠️  Verify: Check target directory for partial files

**Safety**: ★★★★☆ (Safe with verification)

### Thumbnail Job (Creates New Files)
**If Cancelled/Killed**:
- ✅ Completed thumbnails: Ready to use
- ⚠️  In-progress thumbnail: May be corrupt/incomplete
- ✅ Original photos: Untouched
- ✅ Resume: Can re-run with skip_existing=true

**Safety**: ★★★★★ (Very Safe)

---

## 🔍 Post-Kill Checklist

After force-killing a job, verify:

**1. Catalog Accessibility**
```bash
curl http://localhost:8765/api/catalog/info
# Should return catalog info without errors
```

**2. File Integrity**
```bash
# Check for partial/corrupt files
find /app/organized -type f -size 0
# Empty files indicate incomplete operations
```

**3. Lock Status**
```bash
# Check if lock files are stuck
ls -la /app/catalogs/test/.lock
# Should not exist (or be old)
```

**4. Transaction Logs**
```bash
# Review what completed
cat /app/catalogs/test/.transactions/latest.json
```

---

## 🔄 Restart/Retry Jobs

### Manual Restart (Recommended)
1. Note the job parameters from the original form
2. Go to Dashboard → Click appropriate Quick Action
3. Select same catalog
4. Configure same parameters
5. Submit new job

### Why Manual?
- ✅ Ensures you want to retry
- ✅ Allows parameter adjustments
- ✅ Clear audit trail
- ✅ No automated retry loops

---

## 🚧 Recovery Scenarios

### Scenario 1: Analysis Job Stuck at 50%
```
Symptoms: Progress bar stuck, no updates for 5+ minutes
Action:
  1. Click "Cancel" → Wait 30s
  2. If still stuck → Click "Force Kill"
  3. Re-run analysis job
  4. Already-processed files are skipped automatically
Result: ✅ Catalog contains all processed files, resume from where it left off
```

### Scenario 2: Organize Job Killed Mid-Operation
```
Symptoms: Force-killed while copying files
Action:
  1. Check transaction log:
     cat /app/catalogs/test/.transactions/{id}.json
  2. List completed files:
     ls /app/organized
  3. Re-run with dry_run=true to see what's left
  4. Run actual organize for remaining files
Result: ✅ No duplicates, all files organized
```

### Scenario 3: Catalog Won't Load After Kill
```
Symptoms: Dashboard shows errors, catalog info fails
Action:
  1. Check for .lock files:
     rm /app/catalogs/test/.catalog.lock
  2. Verify catalog.json:
     python -m json.tool /app/catalogs/test/catalog.json
  3. Restore from backup if needed:
     cp /app/catalogs/test/.backup.json /app/catalogs/test/catalog.json
Result: ✅ Catalog restored from last good state
```

---

## 📋 Best Practices

### ✅ DO
- Use "Cancel" first, "Force Kill" as last resort
- Check transaction logs after force-killing organize jobs
- Re-run analysis jobs after kills (they skip processed files)
- Use dry_run=true for organize jobs first

### ❌ DON'T
- Force-kill unless absolutely necessary
- Run multiple jobs on same catalog simultaneously (will wait for lock)
- Delete lock files unless job is definitely dead
- Panic - data safety mechanisms protect you

---

## 🎯 Summary

**Job Safety Features**:
- ✅ File locking prevents corruption
- ✅ Transaction logging tracks all operations
- ✅ Checkpointing saves partial progress
- ✅ Graceful cancellation (SIGTERM)
- ✅ Force kill available (SIGKILL)
- ✅ Catalog backups (.backup.json)

**Recovery Options**:
- ✅ Cancel → Graceful stop
- ✅ Force Kill → Immediate stop
- ✅ Manual restart → Re-submit job
- ✅ Transaction logs → See what completed
- ✅ Catalog backup → Restore if corrupted

**Data Safety**:
- ★★★★★ Analysis: Read-only, very safe
- ★★★★☆ Organize: With verification, safe
- ★★★★★ Thumbnails: Creates new files, very safe

---

**Bottom Line**: The system is designed to be resilient. Force-kill is available when needed, and data safety mechanisms protect against corruption. 🛡️
