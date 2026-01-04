# Lumina - Safety Guarantees

## 🛡️ Non-Destructive Operations

Lumina is designed to be **safe by default**. Here's what protects your files:

---

## ✅ Analysis Operations (ALWAYS SAFE)

### `analyze_catalog_task` - 100% Read-Only

**What it does**:
- Reads files from source directories
- Extracts metadata
- Computes hashes
- Detects duplicates

**What it NEVER does**:
- ❌ Modify source files
- ❌ Delete files
- ❌ Move files
- ❌ Rename files
- ❌ Change permissions

**Safe to run multiple times**:
- ✅ Re-running analysis updates catalog data
- ✅ New files are added
- ✅ Changed files are re-analyzed
- ✅ Existing data is preserved (unless file changed)

**Catalog Storage**:
```
/path/to/catalog/
├── catalog.json          # Metadata database
├── .backup.json          # Auto-backup before writes
├── .transactions/        # Operation logs
└── thumbnails/           # Generated thumbnails
```

**Source Photos**:
```
/path/to/photos/          # NEVER MODIFIED
├── IMG_001.jpg           # Read-only
├── IMG_002.raw           # Read-only
└── ...                   # Read-only
```

---

## ⚠️ Organization Operations (COPY is Default)

### `organize_catalog_task` - Configurable Safety

**Default Settings (SAFE)**:
```json
{
  "operation": "copy",      // ✅ Keeps originals
  "dry_run": false,         // ⚠️ Set to true for preview
  "verify_checksums": true, // ✅ Ensures integrity
  "skip_existing": true     // ✅ Won't overwrite
}
```

**COPY Operation** (Default):
- ✅ Original files remain untouched
- ✅ New organized copies created
- ✅ Checksum verification after copy
- ✅ Transaction log for rollback

**MOVE Operation** (Requires Explicit Choice):
- ⚠️ Original files are moved (deleted from source)
- ✅ Transaction log allows rollback
- ✅ Checksum verification ensures no corruption
- ⚠️ Use with caution!

**Dry-Run Mode**:
```json
{
  "dry_run": true  // ✅ PREVIEW ONLY - No files touched
}
```

**ALWAYS use dry-run first**:
```bash
# 1. Preview (safe)
curl -X POST http://localhost:8000/api/jobs/organize \
  -d '{"dry_run": true, ...}'

# 2. Review results

# 3. Execute (if satisfied)
curl -X POST http://localhost:8000/api/jobs/organize \
  -d '{"dry_run": false, ...}'
```

---

## 🔒 Safety Features

### 1. Transaction Logging

Every operation is logged:
```json
{
  "transaction_id": "abc123...",
  "operations": [
    {
      "operation_id": "op001",
      "source_path": "/photos/IMG_001.jpg",
      "target_path": "/organized/2023-06/IMG_001.jpg",
      "operation_type": "copy",
      "checksum": "sha256:...",
      "status": "completed"
    }
  ]
}
```

**Rollback Capability**:
```bash
# Rollback a transaction
curl -X POST http://localhost:8000/api/jobs/organize \
  --rollback abc123...
```

### 2. Checksum Verification

**After every copy/move**:
```python
# Compute checksum before
original_checksum = compute_checksum(source)

# Copy/move file
copy_file(source, target)

# Verify after
target_checksum = compute_checksum(target)

if original_checksum != target_checksum:
    # Delete corrupted target
    target.unlink()
    raise ValueError("Checksum mismatch!")
```

**Protection against**:
- ❌ File corruption during copy
- ❌ Incomplete writes
- ❌ Disk errors

### 3. Naming Conflict Resolution

**When target exists**:
```python
# Original file
/organized/2023-06/IMG_001.jpg

# New file with same name
/photos/IMG_001.jpg

# Auto-resolution
/organized/2023-06/IMG_001_001.jpg  # Numbered suffix
```

**Protects against**:
- ❌ Accidental overwrites
- ❌ Data loss from duplicates

### 4. File Locking

**Catalog access is protected**:
```python
# Only one process can write at a time
with CatalogDatabase(catalog_path) as db:
    # Exclusive lock acquired
    db.add_image(...)
    # Lock released on exit
```

**Protects against**:
- ❌ Concurrent write corruption
- ❌ Race conditions
- ❌ Data inconsistency

### 5. Read-Only Photo Mounts

**Docker volumes are read-only by default**:
```yaml
volumes:
  - ${PHOTOS_PATH}:/app/photos:ro  # :ro = read-only
```

**Even if code has bugs**:
- ❌ Cannot delete source photos
- ❌ Cannot modify source photos
- ❌ Cannot rename source photos

---

## 🧪 Safety Testing

### Run Safety Tests

```bash
# 1. Create test directory
mkdir -p /tmp/vam-safety-test
cd /tmp/vam-safety-test

# 2. Create test photos
mkdir photos catalog organized
echo "test" > photos/test.jpg

# 3. Test analysis (safe)
curl -X POST http://localhost:8000/api/jobs/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "catalog_path": "/tmp/vam-safety-test/catalog",
    "source_directories": ["/tmp/vam-safety-test/photos"],
    "detect_duplicates": false
  }'

# 4. Verify source unchanged
ls -la photos/  # test.jpg should still exist
md5sum photos/test.jpg  # Checksum should match

# 5. Test organization dry-run (safe)
curl -X POST http://localhost:8000/api/jobs/organize \
  -H "Content-Type: application/json" \
  -d '{
    "catalog_path": "/tmp/vam-safety-test/catalog",
    "output_directory": "/tmp/vam-safety-test/organized",
    "dry_run": true,
    "operation": "copy"
  }'

# 6. Verify nothing moved/copied
ls -la organized/  # Should be empty or show preview
ls -la photos/     # test.jpg should still exist

# 7. Test actual copy (safe)
curl -X POST http://localhost:8000/api/jobs/organize \
  -H "Content-Type: application/json" \
  -d '{
    "catalog_path": "/tmp/vam-safety-test/catalog",
    "output_directory": "/tmp/vam-safety-test/organized",
    "dry_run": false,
    "operation": "copy"
  }'

# 8. Verify both copies exist
ls -la photos/     # Original should STILL exist
ls -la organized/  # Copy should exist
md5sum photos/test.jpg organized/*/test*.jpg  # Should match
```

### Expected Results

**After Analysis**:
```
photos/
└── test.jpg  ✅ (unchanged)

catalog/
├── catalog.json  ✅ (created)
└── .backup.json  ✅ (backup)
```

**After Dry-Run**:
```
photos/
└── test.jpg  ✅ (unchanged)

organized/
└── (empty)   ✅ (no changes)
```

**After Copy**:
```
photos/
└── test.jpg         ✅ (STILL exists)

organized/
└── 2023-11/
    └── test.jpg     ✅ (new copy)

Both files identical: ✅
```

---

## 🚨 What CAN Go Wrong (and how we protect)

### 1. Running MOVE Instead of COPY

**Risk**: Source files deleted

**Protection**:
- ⚠️ Web UI clearly labels "MOVE (deletes originals)"
- ⚠️ Confirmation prompt in UI
- ✅ Transaction log allows rollback
- ✅ Default is COPY

**Best Practice**:
- Always use COPY first
- Verify organized files work
- Manually delete originals if desired

### 2. Disk Full During Operation

**Risk**: Partial copy/corrupted file

**Protection**:
- ✅ Checksum verification detects corruption
- ✅ Corrupted file is deleted automatically
- ✅ Transaction marked as failed
- ✅ Original remains intact

### 3. Process Crash Mid-Operation

**Risk**: Incomplete organization

**Protection**:
- ✅ Transaction log shows what completed
- ✅ Resume capability (future enhancement)
- ✅ Checkpointing every 100 files
- ✅ Catalog has .backup.json

**Recovery**:
```bash
# Review transaction log
cat catalog/.transactions/{transaction_id}.json

# See what completed
grep "completed" catalog/.transactions/{transaction_id}.json

# Re-run with skip_existing=true
# Only unprocessed files will be handled
```

### 4. Multiple Jobs Running Simultaneously

**Risk**: Catalog corruption from concurrent writes

**Protection**:
- ✅ File locking prevents concurrent writes
- ✅ Jobs queued by Celery (sequential by default)
- ✅ Second job waits for lock (30s timeout)

### 5. Accidentally Deleting Catalog

**Risk**: Lose all metadata

**Protection**:
- ✅ Automatic .backup.json created
- ✅ Transaction logs preserved
- ✅ Re-running analysis rebuilds catalog
- 📝 Regular backups recommended (see docs)

**Recovery**:
```bash
# Restore from backup
cp catalog/.backup.json catalog/catalog.json

# Or re-analyze (safe, just slow)
vam-analyze /path/to/catalog --source /path/to/photos
```

---

## ✅ Summary: Is it Safe?

**Analysis**: ✅ **100% Safe** - Read-only, run as many times as you want

**Organization (COPY)**: ✅ **Safe** - Original files never touched

**Organization (MOVE)**: ⚠️ **Caution Required** - Original files deleted, but:
- Transaction logging
- Checksum verification
- Rollback capability
- Must be explicitly chosen

**Thumbnail Generation**: ✅ **Safe** - Only creates new files, never modifies originals

**Multiple Runs**: ✅ **Safe** - Re-running updates catalog, doesn't corrupt it

**Concurrent Jobs**: ✅ **Safe** - File locking prevents corruption

---

## 📋 Safety Checklist

Before running in production:

- [ ] Test with COPY operation first
- [ ] Use dry_run=true to preview
- [ ] Verify checksums match after copy
- [ ] Check transaction logs
- [ ] Backup catalog directory
- [ ] Mount photos as read-only in Docker
- [ ] Never use MOVE without testing COPY first
- [ ] Review organized files before deleting originals

---

## 🆘 Emergency Procedures

### If Something Goes Wrong

1. **Stop immediately**:
   ```bash
   docker-compose down
   ```

2. **Check what was done**:
   ```bash
   cat catalog/.transactions/latest.json
   ```

3. **Rollback if needed**:
   ```bash
   # Via API
   curl -X POST .../organize --rollback {transaction_id}

   # Or manually restore
   cp catalog/.backup.json catalog/catalog.json
   ```

4. **Verify source intact**:
   ```bash
   ls -la /path/to/photos  # All files should be there
   ```

5. **Report issue**:
   - GitHub: https://github.com/irjudson/lumina/issues
   - Include: transaction log, error messages, steps to reproduce

---

**Bottom Line**: Lumina is designed to be **safe by default**, with multiple layers of protection. The only destructive operation (MOVE) requires explicit opt-in and has rollback capability.
