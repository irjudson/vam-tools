# File Reorganization System Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize entire image library into clean date-based directory structure with status-based separation (active vs rejected images)

**Architecture:** Parallel batch job using existing coordinator pattern, processes all images, updates database paths, preserves metadata for idempotency

**Tech Stack:** Python, Celery workers, PostgreSQL, existing OrganizationStrategy pattern, Vue 3 UI

---

## Overview

The file reorganization system allows users to reorganize their entire photo library (96K+ images) into a clean, consistent directory structure based on dates, while separating rejected images (from bursts and duplicates) into a distinct subtree for easy bulk management.

**Key Features:**
- Reorganizes all catalog images into date-based structure: `YYYY/MM-DD/HHMMSS_checksum.ext`
- Separates rejected images into `_rejected/` subtree using same structure
- Falls back to file modification time if EXIF date missing
- Handles conflicts with full checksum filenames
- Copy or Move operations with checksum verification
- Updates database source_path to track new locations
- Fully idempotent - safe to run multiple times
- Parallel processing with progress tracking
- Transaction log for rollback capability

---

## Architecture

### System Flow

```
User clicks "Organize Library"
  → Configuration modal (output path, copy/move, dry run)
  → API POST /api/jobs/start {job_type: "reorganize", ...}
  → reorganize_coordinator_task
      ├─ Queries all images from catalog
      ├─ Creates batches (500 images each)
      └─ Spawns reorganize_worker_task for each batch
          ├─ Determines target path (status-based routing)
          ├─ Uses EXIF date or mtime fallback
          ├─ Generates filename: HHMMSS_shortchecksum.ext
          ├─ Copies/moves with checksum verification
          └─ Updates source_path in database
  → reorganize_finalizer_task
      ├─ Aggregates results from all workers
      ├─ Creates master transaction log
      └─ Updates job status
```

### Directory Structure Output

```
/output_path/
  ├── 2023/
  │   ├── 06-15/
  │   │   ├── 143022_abc12345.jpg  (active image)
  │   │   └── 150330_def67890.jpg  (active image)
  │   ├── 06-16/
  │   │   └── 090145_fedcba98.jpg
  │   └── ...
  ├── 2024/
  │   └── ...
  └── _rejected/
      ├── 2023/
      │   ├── 06-15/
      │   │   └── 141230_98765fed.jpg  (rejected - burst or duplicate)
      │   └── ...
      └── 2024/
          └── ...
```

### Key Components

1. **Enhanced OrganizationStrategy** - New directory structure and naming patterns
2. **Reorganization Coordinator** - Parallel batch job coordinator
3. **Reorganization Workers** - Process batches of images
4. **Reorganization Finalizer** - Aggregates results, creates transaction log
5. **API Endpoint** - `/api/jobs/start` with `job_type: "reorganize"`
6. **UI Modal** - Configuration dialog in main toolbar
7. **Transaction Log** - Rollback support via filesystem logs

---

## Enhanced Organization Strategy

### New Enums

**Directory Structure:**
```python
class DirectoryStructure(str, Enum):
    YEAR_MONTH = "YYYY-MM"               # 2023-06
    YEAR_SLASH_MONTH = "YYYY/MM"         # 2023/06
    YEAR_MONTH_DAY = "YYYY-MM-DD"        # 2023-06-15
    YEAR_ONLY = "YYYY"                   # 2023
    YEAR_SLASH_MONTH_DAY = "YYYY/MM-DD"  # NEW: 2023/06-15 (for reorganization)
    FLAT = "FLAT"                        # All files in one directory
```

**Naming Strategy:**
```python
class NamingStrategy(str, Enum):
    DATE_TIME_CHECKSUM = "date_time_checksum"  # 2023-06-15_143022_abc123.jpg
    DATE_TIME_ORIGINAL = "date_time_original"  # 2023-06-15_143022_IMG_1234.jpg
    ORIGINAL = "original"                      # IMG_1234.jpg
    CHECKSUM = "checksum"                      # abc123def456.jpg
    TIME_CHECKSUM = "time_checksum"            # NEW: 143022_abc12345.jpg (for reorganization)
```

### Path Resolution Logic

**Input:** `ImageRecord` with metadata
**Output:** `Path` to target file

**Steps:**

1. **Get status-based base path:**
   ```python
   if image.status_id == 'rejected':
       base_path = output_directory / "_rejected"
   else:
       base_path = output_directory
   ```

2. **Get date for directory:**
   ```python
   # First try: EXIF date
   if image.dates and image.dates.selected_date:
       date = image.dates.selected_date
   else:
       # Fall back: file modification time
       date = datetime.fromtimestamp(os.path.getmtime(image.source_path))
   ```

3. **Build directory path:**
   ```python
   # For YEAR_SLASH_MONTH_DAY structure
   directory = base_path / date.strftime("%Y") / date.strftime("%m-%d")
   # Example: /organized/2023/06-15 or /organized/_rejected/2023/06-15
   ```

4. **Generate filename:**
   ```python
   # For TIME_CHECKSUM naming
   time_str = date.strftime("%H%M%S")
   short_checksum = image.checksum[:8]
   extension = image.source_path.suffix
   filename = f"{time_str}_{short_checksum}{extension}"
   # Example: 143022_abc12345.jpg
   ```

5. **Handle conflicts:**
   ```python
   target_path = directory / filename

   if target_path.exists():
       # Check if it's the same file (idempotent skip)
       target_checksum = compute_checksum(target_path)
       if target_checksum == image.checksum:
           return None  # Skip - already organized

       # Different file - use full checksum to resolve conflict
       filename = f"{time_str}_{image.checksum}{extension}"
       target_path = directory / filename

       # If still conflicts (impossibly rare), add suffix
       if target_path.exists():
           target_path = resolve_naming_conflict(target_path, image)

   return target_path
   ```

---

## Worker Implementation

### Coordinator Task

**Function:** `reorganize_coordinator_task(catalog_id, output_directory, operation, dry_run)`

**Responsibilities:**
1. Query all images from catalog (no filtering - process everything)
2. Create batches of 500 images each using `BatchManager`
3. Spawn `reorganize_worker_task` for each batch
4. Use chord pattern: `chord(worker_tasks)(finalizer_task)`
5. Track progress via `job_batches` table

**Parameters:**
- `catalog_id`: UUID of catalog to reorganize
- `output_directory`: Target path for organized files
- `operation`: `"copy"` or `"move"`
- `dry_run`: `True` to preview without executing

**Returns:**
```python
{
    "status": "dispatched",
    "total_images": 96228,
    "total_batches": 193,
    "output_directory": "/path/to/organized"
}
```

### Worker Task

**Function:** `reorganize_worker_task(catalog_id, batch_id, parent_job_id, output_directory, operation, dry_run)`

**Processes one batch (500 images):**

For each image:

1. **Load image data** from database:
   - `id`, `source_path`, `checksum`, `status_id`, `dates`

2. **Skip if already organized:**
   ```python
   # Check if source_path already in organized structure
   if image.source_path.startswith(output_directory):
       skip_count += 1
       continue
   ```

3. **Determine target path:**
   - Create `OrganizationStrategy` with `YEAR_SLASH_MONTH_DAY` and `TIME_CHECKSUM`
   - Inject mtime as selected_date if no EXIF date
   - Route to `_rejected/` if `status_id == 'rejected'`
   - Generate filename with short checksum

4. **Check for existing target:**
   ```python
   if target_path.exists():
       target_checksum = compute_checksum(target_path)
       if target_checksum == image.checksum:
           skip_count += 1  # Already organized
           continue
       else:
           # Conflict - use full checksum
           regenerate_filename_with_full_checksum()
   ```

5. **Execute file operation:**
   ```python
   if dry_run:
       log_operation(source, target, "would_copy" or "would_move")
   else:
       create_target_directory()
       if operation == "copy":
           shutil.copy2(source, target)  # Preserves mtime
       elif operation == "move":
           shutil.move(source, target)

       # Verify checksum
       if compute_checksum(target) != image.checksum:
           target.unlink()  # Delete corrupted file
           raise ValueError("Checksum mismatch")
   ```

6. **Update database:**
   ```python
   # Batch update at end of worker
   UPDATE images
   SET source_path = :new_path
   WHERE id = :image_id
     AND source_path != :new_path
   ```

7. **Record in batch results:**
   ```python
   return {
       "status": "completed",
       "batch_id": batch_id,
       "organized": 450,
       "skipped": 48,
       "failed": 2,
       "mtime_fallback_count": 12,
       "errors": ["path/to/file.jpg: permission denied", ...]
   }
   ```

**Error Handling:**
- Individual file failures don't stop batch
- Continue processing remaining files
- Track errors in results
- Report to finalizer

### Finalizer Task

**Function:** `reorganize_finalizer_task(worker_results, catalog_id, parent_job_id, output_directory)`

**Responsibilities:**

1. **Aggregate statistics:**
   ```python
   total_organized = sum(r['organized'] for r in worker_results)
   total_skipped = sum(r['skipped'] for r in worker_results)
   total_failed = sum(r['failed'] for r in worker_results)
   total_mtime_fallback = sum(r.get('mtime_fallback_count', 0) for r in worker_results)
   all_errors = [e for r in worker_results for e in r.get('errors', [])]
   ```

2. **Build master transaction log:**
   ```python
   transaction_log = {
       "transaction_id": parent_job_id,
       "catalog_id": catalog_id,
       "started_at": start_time,
       "completed_at": datetime.utcnow(),
       "operation": operation,
       "output_directory": output_directory,
       "statistics": {
           "total_files": total_organized + total_skipped + total_failed,
           "organized": total_organized,
           "skipped": total_skipped,
           "failed": total_failed,
           "mtime_fallback": total_mtime_fallback
       },
       "operations": [...]  # From all worker results
   }

   # Save to filesystem
   save_path = f"{output_directory}/.vam_transactions/{parent_job_id}.json"
   save_transaction_log(save_path, transaction_log)
   ```

3. **Update job status:**
   ```python
   if total_failed == 0:
       status = "SUCCESS"
   elif total_failed / total_files < 0.1:  # Less than 10% failed
       status = "SUCCESS"  # With warnings
   else:
       status = "FAILURE"

   update_job(parent_job_id, status, result={
       "status": "completed",
       "statistics": {...},
       "transaction_log": save_path,
       "errors": all_errors[:100]  # First 100 errors
   })
   ```

4. **Publish completion:**
   ```python
   publish_completion(parent_job_id, status, result=job_result)
   ```

---

## Idempotency Guarantees

### Metadata Preservation

**Critical:** All timestamps and metadata used for decisions must be preserved exactly.

**File Copy:**
- Use `shutil.copy2()` (already in existing code)
- Preserves: modification time, access time, permission bits
- **mtime preserved** → mtime fallback gives same date on re-run
- EXIF data embedded in file → never changes during copy/move

**Database:**
- Only update `source_path` if different (prevents unnecessary writes)
- `status_id` unchanged (used for routing only)
- All other metadata preserved

### Skip Logic

**Before processing each image:**

```python
# 1. Skip if already in organized structure
if image.source_path.startswith(output_directory):
    return False

# 2. Calculate target path
target = get_target_path(image)

# 3. Skip if target exists with matching checksum
if target.exists():
    if compute_checksum(target) == image.checksum:
        return False  # Already organized

return True
```

### Safe Re-run Scenarios

| Scenario | Behavior |
|----------|----------|
| Run copy twice | Second run skips all files (target exists, checksum matches) |
| Run copy, add images, run again | Only processes new images not yet organized |
| Run on already-organized catalog | Detects source_path in output_directory, skips all |
| Interrupted job, resume | Skips completed files, processes remaining |
| Run after changing status | Re-organizes images whose status changed (moves between active/_rejected) |
| Change status then re-run | Moves file to correct location based on new status |

### Database Update Idempotency

```sql
UPDATE images
SET source_path = :new_path
WHERE id = :image_id
  AND source_path != :new_path  -- Only update if different
```

This prevents unnecessary writes when source_path already correct.

### Transaction Log Handling

- Each run creates new transaction ID
- Previous transaction logs preserved in `.vam_transactions/`
- Can rollback any transaction independently
- Multiple reorganizations = multiple logs, all preserved

**This makes reorganization a "sync" operation** - brings organized directory in line with current catalog state, safe to run repeatedly.

---

## UI Integration

### Main Toolbar Button

Add button next to "Detect Bursts" and "Detect Duplicates":

```html
<button @click="showOrganizeModal = true" class="toolbar-button">
  📁 Organize Library
</button>
```

### Configuration Modal

**Fields:**

1. **Output Directory** (required)
   - Text input: `/mnt/storage/organized`
   - Validation:
     - Must be writable
     - Must not be inside catalog source directories
     - Must not be parent of source directories

2. **Operation Type** (radio buttons)
   - ○ Copy (default) - Leave originals untouched, recommended for first run
   - ○ Move - Free up space in original locations

3. **Dry Run** (checkbox, default: checked)
   - ☑ Preview changes without executing
   - Shows what would happen, no actual file operations

4. **Summary Display** (auto-calculated, read-only)
   - Total images: 96,228
   - Active images: 48,204 → `/organized/YYYY/MM-DD/`
   - Rejected images: 48,024 → `/organized/_rejected/YYYY/MM-DD/`
   - Images without EXIF dates: 1,234 (will use file modification time)
   - Estimated space needed: 450 GB (for copy operation)

5. **Action Buttons**
   - "Preview" (if dry run checked) - Shows sample paths for 10 images
   - "Start Reorganization" (primary, blue button)
   - "Cancel"

### Job Progress Display

Reuse existing job panel:

```html
<div class="job-panel">
  <div class="job-header">
    <span>Reorganizing Library</span>
    <span class="job-status">Running</span>
  </div>

  <div class="job-progress">
    <div class="progress-bar" :style="{width: progress + '%'}"></div>
    <span>45,000 / 96,228 images (47%)</span>
  </div>

  <div class="job-stats">
    <span>Organized: 44,800</span>
    <span>Skipped: 195</span>
    <span>Failed: 5</span>
  </div>

  <div class="job-actions">
    <button @click="cancelJob">Cancel</button>
    <button @click="viewLogs">View Logs</button>
  </div>
</div>
```

### Results View

After completion:

```html
<div class="job-results">
  <h3>✓ Reorganization Complete</h3>

  <div class="results-stats">
    <div class="stat">
      <span class="stat-label">Total Images</span>
      <span class="stat-value">96,228</span>
    </div>
    <div class="stat">
      <span class="stat-label">Organized</span>
      <span class="stat-value">96,120</span>
    </div>
    <div class="stat">
      <span class="stat-label">Skipped</span>
      <span class="stat-value">103</span>
    </div>
    <div class="stat">
      <span class="stat-label">Failed</span>
      <span class="stat-value error">5</span>
    </div>
  </div>

  <div class="results-actions">
    <button @click="openDirectory">Open Organized Directory</button>
    <button @click="viewTransactionLog">View Transaction Log</button>
    <button @click="rollback" class="secondary">Rollback Changes</button>
  </div>

  <!-- If failures -->
  <details class="error-details">
    <summary>5 files failed (click to expand)</summary>
    <ul>
      <li>/path/to/file1.jpg: Permission denied</li>
      <li>/path/to/file2.jpg: Checksum mismatch</li>
      ...
    </ul>
  </details>
</div>
```

---

## Implementation Considerations

### Edge Cases

**1. Checksum Mismatch After Copy:**
- Delete corrupted target file immediately
- Log error with source/target checksums
- Don't update database - leave source_path pointing to original
- Continue processing other files
- Report in finalizer results

**2. Disk Space:**
- Before starting: check available space on target volume
- Calculate needed: `SUM(size_bytes)` for all images to be copied
- If insufficient: abort with error before starting workers
- During execution: catch `OSError: [Errno 28] No space left on device`
- Pause job gracefully, report partial results

**3. Permission Errors:**
- Source unreadable: log error, skip file, continue batch
- Target unwritable: log error, fail batch but continue other batches
- Report all permission errors in final results with full paths

**4. Concurrent Access:**
- Source files might be modified during reorganization
- Checksum verification will fail if modified
- Don't retry automatically - flag for manual review
- Transaction log shows original checksum for investigation

**5. Database Update Failures:**
- If `UPDATE source_path` fails: file copied but DB not updated
- Mark operation as "partial_success" in transaction log
- Finalizer reports these separately
- On rollback: can clean up files even if DB update failed

### Performance Optimizations

**1. Batch Size Tuning:**
- Default: 500 images per batch
- Smaller (100) for large RAW files (50MB+ each)
- Larger (1000) for small JPEGs (2MB each)
- Auto-adjust based on average file size

**2. Checksum Verification:**
- Skip verification if `skip_existing=True` and checksums match
- Only verify after write operations
- Use cached checksum from database for source (don't recompute)

**3. Database Updates:**
- Batch updates per worker:
  ```sql
  UPDATE images
  SET source_path = CASE
    WHEN id = 'id1' THEN '/new/path1'
    WHEN id = 'id2' THEN '/new/path2'
    ...
  END
  WHERE id IN ('id1', 'id2', ...)
  ```
- Update 500 rows at once instead of individual queries
- Commit per batch, not per file

**4. Progress Reporting:**
- Update batch progress every 50 files (not every file)
- Reduces Redis writes from 96K to ~2K
- Still responsive enough for UI (updates every few seconds)

### Testing Strategy

**Unit Tests:**
- `OrganizationStrategy` with `YEAR_SLASH_MONTH_DAY` structure
- `TIME_CHECKSUM` naming strategy
- Status-based path routing (active vs rejected)
- mtime fallback logic
- Conflict resolution (short vs full checksum)
- Idempotent skip logic

**Integration Tests:**
- Small test catalog (100 images, 50 active + 50 rejected)
- Test copy operation end-to-end
- Verify directory structure created correctly
- Verify filenames match pattern
- Verify database source_path updated
- Test rollback functionality
- Test re-run (should skip all files)

**Manual Testing:**
1. Dry run with real catalog (96K images)
   - Review sample paths generated
   - Check statistics (active/rejected split)
   - Verify no files actually copied

2. Small batch copy (1000 images)
   - Pick specific date range from catalog
   - Copy to test directory
   - Verify structure: `YYYY/MM-DD/HHMMSS_checksum.ext`
   - Check active vs rejected separation
   - Verify database updates

3. Re-run same batch
   - Should skip all 1000 files (already organized)
   - Verify idempotency

4. UI testing
   - Open modal, configure settings
   - Start job, watch progress
   - Check results display
   - View transaction log

### Rollout Plan

**Phase 1: Core Implementation**
1. Add `YEAR_SLASH_MONTH_DAY` and `TIME_CHECKSUM` enums
2. Enhance `OrganizationStrategy.get_target_path()`:
   - Status-based routing
   - mtime fallback
   - Conflict resolution with full checksum
3. Add idempotent skip logic

**Phase 2: Worker Tasks**
4. Create `reorganize_coordinator_task`
5. Create `reorganize_worker_task`
6. Create `reorganize_finalizer_task`
7. Add to `JOB_TYPE_TO_TASK` mapping

**Phase 3: API Integration**
8. Add `reorganize` to job types in `/api/jobs/start`
9. Test API endpoint with Postman/curl
10. Verify batch creation and progress tracking

**Phase 4: UI**
11. Add "Organize Library" button to toolbar
12. Create configuration modal component
13. Wire up to API endpoint
14. Add job progress display
15. Add results view with rollback button

**Phase 5: Testing**
16. Run unit tests
17. Run integration tests
18. Dry run on real catalog
19. Small batch copy test
20. Full catalog reorganization (backup first!)

**Phase 6: Documentation**
21. Write user guide for reorganization feature
22. Document rollback procedure
23. Document transaction log format
24. Add troubleshooting guide

---

## Migration Considerations

**No Database Schema Changes Required:**
- `images` table already has `source_path` and `status_id`
- `jobs` and `job_batches` tables handle reorganization jobs
- Transaction logs stored in filesystem (not database)

**Backwards Compatibility:**
- Existing organization strategies still work
- New strategies are additive
- Old transaction logs remain valid

**Can Run On:**
- Existing catalogs without any migration
- Catalogs with mixed EXIF/no-EXIF images
- Catalogs with bursts and duplicates already detected

---

## Success Criteria

**Functional:**
- ✓ Reorganizes 96K+ images into `YYYY/MM-DD` structure
- ✓ Separates rejected images into `_rejected/` subtree
- ✓ Falls back to mtime for images without EXIF dates
- ✓ Handles filename conflicts with full checksum
- ✓ Updates database source_path to new locations
- ✓ Fully idempotent - safe to re-run
- ✓ Copy and Move operations both work
- ✓ Checksum verification prevents corruption
- ✓ Transaction log enables rollback

**Performance:**
- ✓ Processes 96K images in < 6 hours (6 workers)
- ✓ Shows progress updates every few seconds
- ✓ Handles large files (RAW 50MB+) without issues

**User Experience:**
- ✓ Clear configuration modal
- ✓ Preview mode (dry run) before execution
- ✓ Real-time progress tracking
- ✓ Clear results display with statistics
- ✓ Easy rollback if needed

**Reliability:**
- ✓ No data loss (checksums verified)
- ✓ Graceful handling of errors (permission, disk space)
- ✓ Can resume interrupted operations
- ✓ Preserves all file metadata

---

## Future Enhancements

**Not in Initial Implementation:**

1. **Incremental Reorganization**
   - Only reorganize images added/modified since last run
   - Track last reorganization timestamp per catalog

2. **Custom Directory Structures**
   - User-defined patterns: `{camera}/{year}/{month}/...`
   - Template language for flexibility

3. **Deduplication During Reorganization**
   - Skip copying if identical file already in target
   - Merge duplicate groups automatically

4. **Archive to External Storage**
   - Move rejected images to separate archive drive
   - Update catalog to mark as "archived" status

5. **Smart Batch Sizing**
   - Auto-adjust batch size based on file sizes
   - Optimize for SSD vs HDD target

6. **Parallel Checksum Verification**
   - Verify checksums in parallel thread pool
   - Speed up large file verification
