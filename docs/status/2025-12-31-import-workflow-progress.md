# Import-Based Catalog Workflow - Status & Progress

**Last Updated:** 2025-12-31
**Status:** Design complete, implementation pending
**Priority:** High - fundamental architecture change

## Overview

Transitioning from source-based to destination-based catalog model. Catalogs will manage the destination directory where images are imported/organized, rather than pointing to original source directories.

## Design Documentation

**Primary design:** `docs/plans/2025-12-26-file-reorganization-implementation.md`

### Key Concepts

1. **Catalog = Destination Directory**
   - Each catalog owns a managed destination directory
   - Source paths become metadata (preserved in database)
   - Supports multiple import sources → single organized catalog

2. **Import Workflow**
   ```
   Source Directory → Import Analysis → User Review → Copy/Move to Catalog
   ```

3. **File Organization**
   - Structure: `YYYY/MM-DD/` date-based directories
   - Naming: `HHMMSSmmm_checksum.ext` (time + uniqueness)
   - Metadata: Source paths, original filenames preserved in DB

## Current Status

### ✅ Completed

1. **Design documentation**
   - Import workflow architecture
   - Database schema changes
   - File organization strategy
   - Migration plan

2. **File reorganization components**
   - Date-based directory structure
   - Time+checksum naming strategy
   - Duplicate handling via checksum
   - Branch: `feature/file-reorganization`

3. **Duplicate detection improvements**
   - Consensus-based filtering (99.92% false positive reduction)
   - Union-Find algorithm (200-400x performance improvement)
   - Hash distance analysis infrastructure

### 🚧 In Progress

1. **Burst view UI unification**
   - Unified burst/duplicate card grid pattern (✓ completed)
   - Committed but not yet pushed to GitHub

2. **Consensus duplicate detection push**
   - Tests running (88% complete)
   - Pre-push checks passed
   - Waiting for test completion

### 📋 Pending Implementation

#### Phase 1: Database Schema (Priority: HIGH)

**Add to catalogs table:**
```sql
ALTER TABLE catalogs
ADD COLUMN destination_path TEXT,      -- Managed directory for this catalog
ADD COLUMN import_mode TEXT DEFAULT 'source',  -- 'source' or 'import'
ADD COLUMN source_metadata JSONB;      -- Track multiple import sources
```

**Add import tracking:**
```sql
CREATE TABLE import_jobs (
    id UUID PRIMARY KEY,
    catalog_id UUID REFERENCES catalogs(id),
    source_path TEXT NOT NULL,
    status TEXT,  -- 'analyzing', 'ready', 'importing', 'completed'
    images_found INTEGER,
    images_imported INTEGER,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE import_items (
    id UUID PRIMARY KEY,
    import_job_id UUID REFERENCES import_jobs(id),
    source_file_path TEXT NOT NULL,
    destination_file_path TEXT,
    action TEXT,  -- 'copy', 'move', 'skip'
    status TEXT,  -- 'pending', 'completed', 'failed'
    error TEXT
);
```

**Update images table:**
```sql
ALTER TABLE images
ADD COLUMN original_source_path TEXT,  -- Preserve original location
ADD COLUMN import_job_id UUID REFERENCES import_jobs(id),
ADD COLUMN file_checksum TEXT;  -- For duplicate detection during import
```

#### Phase 2: Import Analysis Engine (Priority: HIGH)

**Location:** `vam_tools/import/analyzer.py`

**Responsibilities:**
- Scan source directory for images/videos
- Compute checksums (skip existing files)
- Extract EXIF metadata
- Detect duplicates (checksum + perceptual hash)
- Generate import plan with file conflicts

**API Endpoint:** `POST /api/catalogs/{id}/import/analyze`

#### Phase 3: Import Execution (Priority: HIGH)

**Location:** `vam_tools/import/executor.py`

**Responsibilities:**
- Execute import plan (copy/move files)
- Organize into date-based structure
- Update database with new image records
- Handle errors gracefully (transactional)
- Support progress tracking

**API Endpoint:** `POST /api/catalogs/{id}/import/execute`

#### Phase 4: UI Components (Priority: MEDIUM)

**Import Wizard:**
1. Select source directory
2. Review analysis results
3. Configure options (copy vs move, duplicate handling)
4. Execute import with progress
5. Review completion summary

**Catalog Creation:**
- New option: "Create Import-Based Catalog"
- Specify destination directory
- Initialize empty catalog for imports

#### Phase 5: Migration (Priority: LOW)

**Tool:** `vam_tools/tools/migrate_to_import_catalog.py`

Convert existing source-based catalogs:
1. Create new catalog with destination directory
2. Copy/move files to organized structure
3. Update database references
4. Preserve all metadata, tags, analysis results

## Design Decisions

### File Organization

**Directory structure:** `YYYY/MM-DD/`
- Balances browsability with scalability
- Groups by date for easy navigation
- Avoids deep nesting issues

**File naming:** `HHMMSSmmm_checksum.ext`
- Time component (millisecond precision) for ordering
- Checksum suffix ensures uniqueness
- Original filename preserved in database

### Duplicate Handling

**During import:**
1. Compute checksum of source file
2. Check if checksum exists in catalog
3. If exists: offer skip/keep both/replace options
4. If new: check perceptual hash for near-duplicates
5. Present user with duplicate resolution UI

**Quality ranking (for auto-resolution):**
1. File format: RAW > JPEG > other
2. Resolution: higher megapixels wins
3. File size: larger typically better quality

## Integration Points

### With Existing Features

1. **Duplicate Detection**
   - Run on import to prevent importing duplicates
   - Use consensus-based filtering (aHash+dHash)
   - Exclude burst images from duplicate detection

2. **Burst Detection**
   - Detect bursts during import analysis
   - Group burst sequences together
   - Preserve burst relationships in organized structure

3. **Tagging & Metadata**
   - All tags migrate to new organized location
   - EXIF metadata preserved
   - Source path recorded as metadata

4. **File Reorganization**
   - Reuse TIME_CHECKSUM naming strategy
   - Reuse YEAR_SLASH_MONTH_DAY directory structure
   - Already implemented on `feature/file-reorganization`

## Risks & Mitigations

### Risk: Data Loss During Import
**Mitigation:**
- Default to COPY (not move)
- Transactional import (rollback on failure)
- Verify checksums after copy
- Keep source metadata for recovery

### Risk: Disk Space
**Mitigation:**
- Warn user about space requirements
- Support incremental imports
- Offer cleanup tools for source files after verification

### Risk: Breaking Existing Workflows
**Mitigation:**
- Support both source and import modes
- Gradual migration, not forced
- Keep source-based catalogs working
- Provide migration tool, don't auto-migrate

## Next Steps

### Immediate (Before implementation)
1. ✅ Document duplicate detection findings
2. ✅ Document import workflow status
3. ⏸️ Pause to save tokens
4. Review and refine schema design
5. Create implementation plan with task breakdown

### Short-term (Next session)
1. Implement Phase 1: Database schema changes
2. Create import job models and API stubs
3. Build import analyzer (scan + analyze)
4. Create basic import UI (file picker + preview)

### Medium-term
1. Implement import executor
2. Build duplicate resolution UI
3. Add progress tracking
4. Test with real catalogs

### Long-term
1. Migration tool for existing catalogs
2. Advanced import options (filters, exclusions)
3. Scheduled/watched folder imports
4. Cloud storage import sources

## Questions to Resolve

1. **Naming conflicts:** How to handle files with identical timestamps?
   - Current: Add checksum suffix (ensures uniqueness)
   - Alternative: Add sequence number?

2. **Import modes:** Should catalogs support both source AND import modes?
   - Current: Yes, gradual transition
   - Alternative: Force migration?

3. **Source deletion:** Automatically clean up source files after import?
   - Current: Manual only, user decides
   - Alternative: Offer option during import?

4. **Burst handling:** Import bursts as subdirectories?
   - Current: No, keep flat structure
   - Alternative: `YYYY/MM-DD/burst_GUID/` for bursts?

## Success Metrics

- Import workflow reduces manual file organization by 90%
- No data loss during imports
- Duplicate detection during import reduces redundant storage by 20%+
- User satisfaction: Can find any image by date within seconds
- Performance: Import 10,000 images in < 10 minutes

## Related Documentation

- Design: `docs/plans/2025-12-26-file-reorganization-implementation.md`
- File org: `docs/plans/2025-12-26-file-reorganization-implementation.md`
- Duplicate detection: `docs/findings/2025-12-30-duplicate-detection-consensus-breakthrough.md`
- Burst management: `docs/plans/2025-12-24-burst-management.md`
