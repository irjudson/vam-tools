# Catalog to Import Model Design

**Date:** 2025-12-29
**Status:** Design Approved

## Overview

Transform the catalog system from a source-based scanning model to a destination-based import model. Catalogs will manage a collection of images (destination) with rich metadata about where they came from (sources), rather than being tied to specific source directories.

## Problem Statement

Current limitations:
- Catalogs are bound to source directories where images live
- No clear separation between "where files are" and "what we're managing"
- Difficult to track import history and lineage
- Can't easily support mixed storage (some managed, some referenced)
- Reorganization is tightly coupled to scanning

## Goals

1. **Catalog as destination**: Catalogs manage a collection, not scan sources
2. **Source tracking**: Track where images came from as metadata
3. **Import flexibility**: Support reference-in-place, copy, or move operations
4. **Migration safety**: Automatically migrate existing catalogs without data loss
5. **Import history**: Full lineage of what was imported when and from where

## Design Decisions

### Hybrid Storage Model
Support three import modes:
- **Reference**: Image stays at original location, catalog tracks it
- **Copy**: Image copied to catalog's managed directory
- **Move**: Image moved to catalog's managed directory

Rationale: Maximum flexibility for different use cases (external drives, camera cards, archives).

### Sources as First-Class Entities
Create explicit Source model representing external collections that can be imported from. Sources are persistent and can be re-scanned.

Rationale: Better than treating source paths as transient scan parameters. Enables import history, auto-import, and source management.

### Import Jobs Track Operations
Separate ImportJob model tracks each import operation with full metadata about what was imported and any errors.

Rationale: Complete audit trail of imports. Can answer "when did I import from X?" and "what happened during import Y?"

### Single Managed Directory Per Catalog
Each catalog has one optional `managed_directory` where copy/move imports go.

Rationale: Simpler than multiple storage locations. Referenced files can be anywhere, so distributed storage is still supported.

### Reuse Existing Reorganization Strategies
Import operations use existing YEAR_SLASH_MONTH_DAY, TIME_CHECKSUM, etc. strategies.

Rationale: Don't rebuild what works. Import is just reorganization with a source context.

## Data Model

### New: Source Model

Represents an external collection that can be imported from:

```python
class Source(Base):
    id: UUID
    catalog_id: UUID  # FK to Catalog
    name: str  # "Camera SD Card", "Google Photos 2024"
    source_type: str  # "directory", "device", "url", "cloud"
    source_path: str  # Original location
    auto_import: bool  # Auto-import on scan?
    default_import_mode: str  # "reference", "copy", "move"
    default_organization_strategy: str | None  # Override catalog default
    last_scanned_at: datetime | None
    created_at: datetime
    updated_at: datetime
```

### New: ImportJob Model

Tracks each import operation:

```python
class ImportJob(Base):
    id: UUID  # Also Celery task ID
    catalog_id: UUID  # FK to Catalog
    source_id: UUID  # FK to Source
    import_mode: str  # "reference", "copy", "move"
    organization_strategy: str
    skip_duplicates: bool  # Default: True
    files_discovered: int
    files_imported: int
    files_skipped: int  # Already in catalog
    started_at: datetime
    completed_at: datetime | None
    status: str  # PENDING, RUNNING, SUCCESS, FAILURE
    error: str | None
```

### Updated: Catalog Model

Catalog becomes destination-focused:

```python
class Catalog(Base):
    id: UUID
    name: str
    managed_directory: str | None  # Where imported files go (copy/move mode)
    default_organization_strategy: str  # YEAR_SLASH_MONTH_DAY, TIME_CHECKSUM, etc.
    source_directories: list[str]  # DEPRECATED - kept during migration
    organized_directory: str | None  # DEPRECATED - renamed to managed_directory
    created_at: datetime
    updated_at: datetime
```

### Updated: Image Model

Images track location, import history, and source:

```python
class Image(Base):
    id: UUID
    catalog_id: UUID

    # Location tracking
    current_path: str  # Absolute path (in managed_directory or external location)
    is_managed: bool  # True if in catalog's managed_directory
    file_missing: bool  # True if referenced file no longer exists

    # Import tracking
    source_id: UUID | None  # FK to Source that imported this image
    import_job_id: UUID | None  # FK to ImportJob that brought it in
    import_mode: str  # "reference", "copy", "move"
    original_source_path: str  # Path when first discovered/imported

    # Existing fields remain unchanged
    checksum: str
    file_type: str
    size_bytes: int
    captured_at: datetime | None
    dhash, ahash, whash: str
    # ... all other metadata fields ...
```

**Key behaviors:**
- `current_path` replaces `source_path` (renamed for clarity)
- When `mode=reference`: `current_path` = `original_source_path`, `is_managed` = False
- When `mode=copy/move`: File organized in `managed_directory`, `is_managed` = True
- `original_source_path` is immutable for duplicate detection
- Constraint: `unique(catalog_id, checksum)` prevents duplicate imports by default

## Import Workflow

### 1. Scan Source

```python
scan_source(source_id, catalog_id) -> ScanResult
```

- Discovers files at source location (recursive directory walk)
- Computes checksums for discovered files
- Compares against existing catalog images by checksum
- Returns preview:
  - `new_files`: Not in catalog yet
  - `duplicate_files`: Already in catalog (by checksum)
  - `error_files`: Unreadable, corrupted, etc.
  - `total_size`: Sum of new file sizes

### 2. Preview Results

User reviews scan results before import:
- See what would be imported
- Identify duplicates
- Estimate disk space needed (for copy/move)

### 3. Confirm Import

```python
create_import_job(
    catalog_id,
    source_id,
    import_mode: "reference" | "copy" | "move",
    organization_strategy: str,
    skip_duplicates: bool = True,
    files_to_import: List[UUID]  # From scan results
) -> ImportJob
```

User specifies:
- Import mode (reference/copy/move)
- Organization strategy (if copy/move)
- Whether to skip duplicates
- Which files to import (can exclude some)

### 4. Execute Import

Background job processes import:

**For reference mode:**
- Add Image records with `current_path` = original location
- Set `is_managed` = False

**For copy mode:**
- Copy files to `managed_directory`
- Organize by strategy (YEAR_SLASH_MONTH_DAY, etc.)
- Add Image records with `current_path` = new location
- Set `is_managed` = True
- Original files remain untouched

**For move mode:**
- Move files to `managed_directory`
- Organize by strategy
- Add Image records with `current_path` = new location
- Set `is_managed` = True
- Original files removed

**All modes:**
- If `skip_duplicates=True`: Skip files with existing checksum, increment `files_skipped`
- If `skip_duplicates=False`: Import anyway (different source context), checksum constraint will fail, log error
- Update `ImportJob` status and counts
- Handle errors gracefully (log, continue with other files)

### 5. Auto-Import (Optional)

Sources with `auto_import=True`:
- Scan triggers immediate import
- Uses source's `default_import_mode` and `default_organization_strategy`
- Skips preview/confirmation
- Good for watched folders, camera cards, dropbox

## Migration Strategy

Automatically migrate existing catalogs without data loss:

### 1. Detect Old Catalogs
- Check for catalogs with `source_directories[]` (old model)

### 2. Create Source Entities
For each directory in `catalog.source_directories`:
```python
Source.create(
    catalog_id=catalog.id,
    name=f"Imported from {directory}",
    source_type="directory",
    source_path=directory,
    auto_import=False,
    default_import_mode="reference"
)
```

### 3. Link Existing Images to Sources
For each image:
- Find Source whose `source_path` is parent of `image.source_path`
- Set `image.source_id` to matched Source
- Set `image.import_mode` = "reference" (original behavior)
- Set `image.original_source_path` = `image.source_path`
- Rename `image.source_path` to `image.current_path`
- Set `image.is_managed` = False (scanned in place)

### 4. Create Synthetic ImportJobs (Optional)
One ImportJob per Source representing original scan:
- `status` = SUCCESS
- `import_mode` = "reference"
- `files_imported` = count of images from that source

### 5. Update Catalog
- Set `managed_directory` = `catalog.organized_directory` (if exists)
- Keep `source_directories` for backward compatibility (deprecate later)

### 6. Handle Edge Cases
- Image `source_path` doesn't match any Source: Create "Unknown Source"
- Multiple sources could match: Pick most specific (longest matching path)
- Migration is logged for review

**Migration is idempotent**: Safe to run multiple times, won't duplicate data.

## Impact on Existing Operations

### Scan Job → Import Job
- **Old**: `scan_catalog(catalog_id, directories[])`
- **New**: `import_from_source(catalog_id, source_id, import_mode, skip_duplicates)`
- Scanning is now always in context of a Source
- Creates ImportJob instead of generic Job

### Reorganize Job - Two Operations

#### 1. Convert Referenced → Managed (New)
```python
convert_to_managed(catalog_id)
```
- Find all `is_managed=False` images
- Copy files to `managed_directory` using `default_organization_strategy`
- Update: `is_managed=True`, `current_path=new_location`
- Original files stay in place (copy for safety)
- `original_source_path` remains unchanged

#### 2. Re-organize Managed Files (Future)
- Find all `is_managed=True` images
- Apply new organization strategy
- Move within `managed_directory`
- Update `current_path`
- Not implemented initially - placeholder for future enhancement

### Duplicate Detection
- Exact duplicates: Uses `checksum` (unchanged)
- Perceptual duplicates: Uses dhash/ahash/whash (unchanged)
- Enhanced reporting: Show which sources duplicates came from
- New filters: Duplicates across sources vs within source

### File Operations - Safety First

**Delete Image:**
- Remove database record
- If `is_managed=True`: Optionally delete physical file (user confirms)
- If `is_managed=False`: Never delete physical file (external reference)
- Log all deletions for audit trail

**Move/Reorganize:**
- Only managed files can be moved
- Referenced files cannot be moved (would break reference)
- Always verify destination before moving
- Never overwrite existing files

**General Principle: When in doubt, keep the file**
- Failed operations leave files untouched
- Warnings before any destructive operation
- Referenced files are read-only from catalog's perspective

**Catalog Deletion:**
- Delete all database records (Sources, ImportJobs, Images via cascade)
- User chooses: keep or delete managed files
- Referenced files always remain (catalog doesn't own them)

## API Changes

### New Endpoints

```
# Source Management
POST   /catalogs/{catalog_id}/sources          # Create source
GET    /catalogs/{catalog_id}/sources          # List sources
PATCH  /sources/{source_id}                    # Update source settings
DELETE /sources/{source_id}                    # Remove source

# Import Workflow
POST   /sources/{source_id}/scan               # Scan and preview
POST   /sources/{source_id}/import             # Create import job
GET    /import-jobs/{job_id}                   # Get import status
GET    /catalogs/{catalog_id}/import-jobs      # List import history

# Catalog Operations (New)
POST   /catalogs/{catalog_id}/convert-to-managed  # Convert referenced→managed
GET    /catalogs/{catalog_id}/storage-info        # Show managed vs referenced breakdown
```

### Updated Endpoints

Existing scan/reorganize endpoints updated to use new models internally but maintain backward compatibility during transition.

## Error Handling

### Import Errors

**Source path doesn't exist:**
- Fail early before creating ImportJob
- Return clear error to user

**File unreadable:**
- Skip file, log in ImportJob.error_files
- Continue with other files

**Checksum collision with different file:**
- Skip file, log warning (potential data integrity issue)
- If `skip_duplicates=False`, this indicates same checksum but user wants to import anyway - allow but log

**Insufficient disk space (copy/move):**
- Check space before starting import
- If space runs out mid-import: Fail job, rollback partial imports

**File already exists at destination:**
- Never overwrite
- Generate unique name or skip (depending on context)

### Reference Integrity

**Periodic background job verifies referenced files:**
- Check that `is_managed=False` files still exist at `current_path`
- If missing: Set `file_missing=True`, don't delete record

**User options for missing files:**
- Relocate file (update `current_path`)
- Convert to managed (re-import from new location)
- Delete record

### Migration Errors

**Image source_path doesn't match any Source:**
- Create "Unknown Source" and link to it
- User can correct later

**Multiple sources could match:**
- Pick most specific (longest matching path)
- Log decision for review

**All migration operations logged:**
- Review which images got which sources
- Verify correctness

## Database Schema

### New Tables

```sql
CREATE TABLE sources (
    id UUID PRIMARY KEY,
    catalog_id UUID NOT NULL REFERENCES catalogs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL,  -- 'directory', 'device', 'url', 'cloud'
    source_path TEXT NOT NULL,
    auto_import BOOLEAN DEFAULT FALSE,
    default_import_mode TEXT DEFAULT 'reference',  -- 'reference', 'copy', 'move'
    default_organization_strategy TEXT,
    last_scanned_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE import_jobs (
    id UUID PRIMARY KEY,  -- Also Celery task ID
    catalog_id UUID NOT NULL REFERENCES catalogs(id) ON DELETE CASCADE,
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    import_mode TEXT NOT NULL,
    organization_strategy TEXT NOT NULL,
    skip_duplicates BOOLEAN DEFAULT TRUE,
    files_discovered INTEGER DEFAULT 0,
    files_imported INTEGER DEFAULT 0,
    files_skipped INTEGER DEFAULT 0,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status TEXT NOT NULL,  -- 'PENDING', 'RUNNING', 'SUCCESS', 'FAILURE'
    error TEXT
);
```

### Modified Tables

```sql
-- Catalogs: Add managed directory
ALTER TABLE catalogs
    ADD COLUMN managed_directory TEXT,
    ADD COLUMN default_organization_strategy TEXT DEFAULT 'YEAR_SLASH_MONTH_DAY';
-- Keep source_directories during migration, deprecate later

-- Images: Track location and import metadata
ALTER TABLE images
    RENAME COLUMN source_path TO current_path;

ALTER TABLE images
    ADD COLUMN is_managed BOOLEAN DEFAULT FALSE,
    ADD COLUMN source_id UUID REFERENCES sources(id) ON DELETE SET NULL,
    ADD COLUMN import_job_id UUID REFERENCES import_jobs(id) ON DELETE SET NULL,
    ADD COLUMN import_mode TEXT,
    ADD COLUMN original_source_path TEXT,
    ADD COLUMN file_missing BOOLEAN DEFAULT FALSE;
```

### Indexes

```sql
CREATE INDEX idx_sources_catalog_id ON sources(catalog_id);
CREATE INDEX idx_import_jobs_catalog_id ON import_jobs(catalog_id);
CREATE INDEX idx_import_jobs_source_id ON import_jobs(source_id);
CREATE INDEX idx_images_source_id ON images(source_id);
CREATE INDEX idx_images_is_managed ON images(is_managed);
CREATE INDEX idx_images_file_missing ON images(file_missing) WHERE file_missing = TRUE;
```

## Implementation Phases

### Phase 1: Database Schema
- Add new tables (sources, import_jobs)
- Add new columns to catalogs and images
- Migration script for existing data
- Verify migration with test catalog

### Phase 2: Core Import Logic
- Implement Source model and CRUD
- Implement ImportJob model
- Build scan → preview → import workflow
- Support all three import modes (reference/copy/move)

### Phase 3: Update Existing Operations
- Update reorganize to only touch managed files
- Add convert-to-managed operation
- Update duplicate detection reporting
- Update file operations safety checks

### Phase 4: API & UI
- New API endpoints for sources and imports
- Update scan UI to import workflow
- Source management interface
- Import history view

### Phase 5: Polish
- Auto-import for watched sources
- Reference integrity background job
- Enhanced duplicate detection filters
- Performance optimization

## Success Criteria

1. All existing catalogs migrate without data loss
2. Can import from new sources using all three modes
3. Import history tracked completely
4. Existing duplicate detection still works
5. File operations respect managed vs referenced
6. No performance regression on large catalogs

## Future Enhancements

- Cloud sources (S3, Google Photos, etc.)
- Smart import rules (auto-organize by metadata)
- Import presets/templates
- Bulk source management
- Import scheduling
- Advanced duplicate resolution during import
