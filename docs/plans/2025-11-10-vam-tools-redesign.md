# VAM Tools - Complete Redesign (PostgreSQL + Celery)

**Date**: 2025-11-10
**Status**: Approved - Ready for Implementation

## Overview

Complete redesign of VAM Tools as a production-grade photo management system with:
- PostgreSQL database (replacing JSON/SQLite)
- Celery + Redis for background job processing
- Modern web UI with real-time updates
- Multiple catalog support
- Hybrid organization (date + tags + manual)
- Safety-first design (no operations without user approval)

## Requirements Summary

### Scale & Performance
- Support 100,000+ photos per catalog
- Multiple independent catalogs
- Real-time progress tracking
- GPU acceleration where applicable

### Safety
- **CRITICAL**: No file deletion or movement without explicit user approval
- Dry-run previews for all operations
- Transaction-based file operations with rollback
- Audit logging

### Organization
- Date-based (flexible formats: YYYY-MM, YYYY/MM, YYYY-MM-DD, etc.)
- AI tag-based (people, places, subjects, scenes)
- Manual grouping (events, projects)
- Hybrid combinations (e.g., "Tags/People/YYYY-MM")

### User Experience
- Fast and responsive UI
- Beautiful, professional design
- Clear status and progress indicators
- Real-time updates via WebSocket

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Web Browser                         │
│            (Vue 3 SPA - Real-time UI)                   │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/WebSocket
┌────────────────────▼────────────────────────────────────┐
│                  FastAPI Server                         │
│  ┌──────────────┬──────────────┬──────────────┐        │
│  │ Catalog API  │  Jobs API    │  Review API  │        │
│  └──────┬───────┴──────┬───────┴──────┬───────┘        │
└─────────┼──────────────┼──────────────┼────────────────┘
          │              │              │
          ▼              ▼              ▼
┌─────────────────┐  ┌──────────────────────────┐
│   PostgreSQL    │  │   Celery Workers (3x)    │
│   vam-tools DB  │  │   - Scanner              │
│   Multiple      │  │   - Duplicate Detector   │
│   Schemas       │  │   - AI Tagger            │
└─────────────────┘  └──────────┬───────────────┘
                                │
                                ▼
                     ┌──────────────────┐
                     │ Redis (DB 2)     │
                     │ Task Queue       │
                     └──────────────────┘
```

### Technology Stack

**Database**
- PostgreSQL (localhost, database: vam-tools, user: pg, password: buffalo-jump)
- One schema per catalog for isolation
- Connection pooling via SQLAlchemy

**Job Queue**
- Celery for task management
- Redis broker (localhost:6379, database 2 - shared instance)
- PostgreSQL result backend

**Backend**
- FastAPI (async, WebSocket support)
- SQLAlchemy ORM + Alembic migrations
- Pydantic for validation

**Frontend**
- Vue 3 SPA (composition API)
- Pinia for state management
- Axios for HTTP, native WebSocket
- Tailwind CSS for styling

**Processing**
- ExifTool for metadata extraction
- Pillow + pillow-heif for image processing
- FAISS for fast similarity search (GPU optional)
- PyTorch for AI models (CLIP, YOLOv8)

## Database Schema

### Global Schema (public)

```sql
-- Catalog registry
CREATE TABLE catalogs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    schema_name TEXT NOT NULL UNIQUE,
    source_directories TEXT[] NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Per-Catalog Schema (catalog_<uuid>)

Each catalog gets an isolated schema with these tables:

```sql
-- Catalog configuration
CREATE TABLE config (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Images and videos
CREATE TABLE images (
    id TEXT PRIMARY KEY,              -- SHA256 checksum
    source_path TEXT NOT NULL,
    file_type TEXT NOT NULL,          -- 'image' or 'video'
    checksum TEXT NOT NULL,
    size_bytes BIGINT,

    -- Dates (JSONB for flexibility)
    dates JSONB NOT NULL,
    -- Format: {
    --   exif_date: "2023-06-15T14:30:00Z",
    --   filename_date: "2023-06-15",
    --   directory_date: "2023-06",
    --   filesystem_date: "2023-06-20T10:00:00Z",
    --   selected_date: "2023-06-15T14:30:00Z",
    --   source: "exif",
    --   confidence: 95
    -- }

    -- Metadata (JSONB for flexible schema)
    metadata JSONB NOT NULL,
    -- Format: {
    --   format: "JPEG",
    --   width: 4032,
    --   height: 3024,
    --   exif: {...},
    --   camera: {...},
    --   gps: {...}
    -- }

    -- Perceptual hashes
    dhash TEXT,
    ahash TEXT,

    -- Quality & status
    quality_score INTEGER,            -- 0-100
    status TEXT DEFAULT 'pending',    -- pending, analyzing, complete, error

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_images_status ON images(status);
CREATE INDEX idx_images_dates ON images USING GIN (dates);
CREATE INDEX idx_images_metadata ON images USING GIN (metadata);

-- AI-generated and manual tags
CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,          -- subject, scene, location, quality, technical, event
    confidence REAL,
    source TEXT DEFAULT 'ai',        -- ai, manual, exif
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, category)
);

-- Image-tag relationships (many-to-many)
CREATE TABLE image_tags (
    image_id TEXT REFERENCES images(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    confidence REAL,
    added_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (image_id, tag_id)
);

CREATE INDEX idx_image_tags_image ON image_tags(image_id);
CREATE INDEX idx_image_tags_tag ON image_tags(tag_id);

-- Duplicate groups
CREATE TABLE duplicate_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    primary_image_id TEXT REFERENCES images(id),
    similarity_type TEXT NOT NULL,    -- exact, perceptual
    confidence INTEGER,               -- 0-100
    status TEXT DEFAULT 'pending',    -- pending, reviewed, resolved
    created_at TIMESTAMP DEFAULT NOW(),
    reviewed_at TIMESTAMP
);

CREATE TABLE duplicate_members (
    group_id UUID REFERENCES duplicate_groups(id) ON DELETE CASCADE,
    image_id TEXT REFERENCES images(id) ON DELETE CASCADE,
    hamming_distance INTEGER,         -- Distance from primary
    PRIMARY KEY (group_id, image_id)
);

CREATE INDEX idx_duplicate_groups_status ON duplicate_groups(status);

-- Organization plans (must be reviewed before execution)
CREATE TABLE organization_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    strategy TEXT NOT NULL,          -- date_only, date_tags, tag_primary, custom
    rules JSONB NOT NULL,
    -- Format: {
    --   date_format: "YYYY/MM",
    --   primary_tag_category: "event",
    --   fallback_directory: "Unsorted",
    --   custom_rules: [...]
    -- }
    status TEXT DEFAULT 'draft',     -- draft, approved, executing, complete, failed
    created_at TIMESTAMP DEFAULT NOW(),
    approved_at TIMESTAMP,
    executed_at TIMESTAMP
);

CREATE TABLE plan_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id UUID REFERENCES organization_plans(id) ON DELETE CASCADE,
    image_id TEXT REFERENCES images(id),
    action_type TEXT NOT NULL,       -- move, copy, delete, skip
    source_path TEXT NOT NULL,
    target_path TEXT,
    reason TEXT,                     -- Why this action was chosen
    status TEXT DEFAULT 'pending',   -- pending, approved, rejected, complete, error
    error_message TEXT,
    executed_at TIMESTAMP,
    INDEX (plan_id, status)
);

CREATE INDEX idx_plan_actions_plan ON plan_actions(plan_id);
CREATE INDEX idx_plan_actions_status ON plan_actions(status);

-- Job tracking (Celery task metadata)
CREATE TABLE jobs (
    id UUID PRIMARY KEY,             -- Celery task ID
    job_type TEXT NOT NULL,          -- scan, duplicate_detect, ai_tag, organize
    status TEXT NOT NULL,            -- pending, running, success, failure, cancelled
    progress JSONB,
    -- Format: {
    --   current: 100,
    --   total: 1000,
    --   percent: 10,
    --   message: "Processing images...",
    --   rate: 15.5
    -- }
    result JSONB,                    -- Job results/summary
    error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type ON jobs(job_type);

-- Audit log for file operations
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    action TEXT NOT NULL,            -- scan, move, copy, delete, tag
    entity_type TEXT NOT NULL,       -- image, tag, plan
    entity_id TEXT NOT NULL,
    details JSONB,
    user_id TEXT,                    -- Future: user authentication
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_audit_log_entity ON audit_log(entity_type, entity_id);
CREATE INDEX idx_audit_log_created ON audit_log(created_at);
```

## Celery Tasks

### 1. Scan Directories Task

```python
@celery.task(bind=True)
def scan_directories(self, catalog_id: str, directories: List[str]) -> dict:
    """
    Scan directories for images/videos and populate catalog.

    Updates progress every 100 files.
    """
```

**Input**: catalog_id, source_directories[]

**Process**:
1. Set job status to 'running'
2. Recursively discover image/video files
3. For each file:
   - Compute SHA256 checksum
   - Check if already in catalog (skip if duplicate)
   - Extract EXIF metadata via ExifTool
   - Extract dates from EXIF, filename, directory, filesystem
   - Create image record in PostgreSQL
   - Update progress every 100 files
4. Set job status to 'success'

**Output**:
```json
{
  "files_found": 15000,
  "files_added": 12000,
  "files_skipped": 3000,
  "exact_duplicates": 500,
  "duration_seconds": 245.5
}
```

### 2. Detect Duplicates Task

```python
@celery.task(bind=True)
def detect_duplicates(self, catalog_id: str, similarity_threshold: int = 5) -> dict:
    """
    Find duplicate images using perceptual hashing.

    Uses FAISS for fast similarity search.
    """
```

**Input**: catalog_id, similarity_threshold (default: 5)

**Process**:
1. Load all images without hashes
2. Compute dHash and aHash for each image
3. Update image records with hashes
4. Build FAISS index from hashes
5. For each image, find similar images (Hamming distance ≤ threshold)
6. Score image quality (format, resolution, metadata)
7. Group similar images, select best as primary
8. Create duplicate_groups and duplicate_members records
9. Update progress throughout

**Output**:
```json
{
  "images_analyzed": 12000,
  "groups_found": 450,
  "potential_savings_bytes": 15000000000,
  "high_confidence_groups": 400,
  "low_confidence_groups": 50
}
```

### 3. Generate Tags Task

```python
@celery.task(bind=True)
def generate_tags(self, catalog_id: str, models: List[str] = ['clip']) -> dict:
    """
    Generate AI tags for images.

    Supports: CLIP (general), YOLOv8 (objects), custom models.
    """
```

**Input**: catalog_id, models_to_use[] (default: ['clip'])

**Process**:
1. Load specified ML models
2. Load images without tags
3. Batch process images (GPU accelerated if available)
4. Generate tags with confidence scores
5. Store in tags + image_tags tables
6. Update progress

**Output**:
```json
{
  "images_tagged": 12000,
  "tags_generated": 3500,
  "models_used": ["clip", "yolo"],
  "avg_tags_per_image": 5.2
}
```

### 4. Execute Organization Plan Task

```python
@celery.task(bind=True)
def execute_plan(self, plan_id: str) -> dict:
    """
    Execute approved organization plan.

    CRITICAL: Only runs if plan.status == 'approved'
    Uses transactions with rollback on any error.
    """
```

**Input**: plan_id (must have status='approved')

**Process**:
1. Verify plan is approved (fail if not)
2. Load all plan_actions with status='pending'
3. For each action:
   - Create target directory
   - Copy/move file (based on action_type)
   - Verify checksum after operation
   - Update image.source_path in database
   - Mark action as 'complete'
   - Log to audit_log
4. Use database transactions (rollback on any failure)
5. Update progress

**Output**:
```json
{
  "actions_total": 12000,
  "actions_executed": 12000,
  "actions_failed": 0,
  "bytes_moved": 150000000000,
  "duration_seconds": 1800
}
```

### Task Chaining

```python
# Example: Full analysis workflow
from celery import chain

workflow = chain(
    scan_directories.s(catalog_id, ['/photos']),
    detect_duplicates.s(catalog_id, threshold=5),
    generate_tags.s(catalog_id, models=['clip'])
)
workflow.apply_async()
```

## Web UI Design

### Screen 1: Catalogs Dashboard

**Features**:
- Grid/list view of all catalogs
- Each catalog shows:
  - Name
  - Total images/videos count
  - Total size
  - Last updated
  - Status (idle, scanning, analyzing)
- Actions:
  - Create new catalog (modal: name + source directories)
  - Delete catalog (confirmation required)
  - Open catalog (navigate to browse view)

### Screen 2: Analysis Tab

**Features**:
- Job controls:
  - "Scan Directories" button → modal to select/add directories
  - "Detect Duplicates" button → slider for threshold
  - "Generate Tags" button → checkboxes for models
- Active jobs section:
  - Live progress bars with stats
  - Current/total items
  - Processing rate (items/sec)
  - Estimated time remaining
  - Cancel button
- Job history table:
  - Type, status, started, duration, results
  - View details button

### Screen 3: Browse Tab

**Features**:
- Filters panel (left sidebar):
  - Date range picker
  - Tag multi-select (by category)
  - Camera/lens select
  - File type checkboxes
  - Quality score range slider
- Grid view (main area):
  - Lazy-loaded image thumbnails
  - Pagination (50 per page)
  - Hover: filename, date, size
  - Click: open detail modal
- Detail modal:
  - Full-size image preview
  - All EXIF data (organized tabs)
  - Tags (editable)
  - Duplicate info (if applicable)
  - File operations (future)

### Screen 4: Duplicates Tab

**Features**:
- Group list (left panel):
  - Each group shows thumbnail + count
  - Potential space savings
  - Confidence badge
  - Filter by status: pending, reviewed, resolved
- Comparison viewer (main area):
  - Side-by-side image comparison
  - Quality scores with explanation:
    - Format quality (40%)
    - Resolution (30%)
    - File size (20%)
    - Metadata completeness (10%)
  - Actions per image:
    - Keep this one
    - Delete this one
    - View full size
- Batch operations toolbar:
  - "Auto-resolve high confidence" (>95%)
  - "Keep all primaries"
  - "Mark all as reviewed"

### Screen 5: Organization Tab

**Features**:
- Plan list (left panel):
  - Existing plans with status
  - Create new plan button
- Plan creator/editor (main area):
  - Name and description fields
  - Strategy selector:
    - Date only: `YYYY/MM`
    - Date with tags: `Tags/People/YYYY-MM`
    - Tag primary: `Events/Wedding/YYYY-MM-DD`
    - Custom rules (JSON editor)
  - Preview button → generates plan_actions
  - Preview table:
    - Source path → Target path
    - Sortable, filterable
    - Stats: total files, total size, directory breakdown
  - Approve button (requires confirmation)
  - Execute button (only if approved, requires confirmation)
- Execution monitor:
  - Live progress bar
  - Success/failure counts
  - Error list (if any)
  - Rollback button (if failures occur)

### Screen 6: Tags Tab

**Features**:
- Tag browser (left panel):
  - Tree view by category
  - Count badges
  - Filter/search
- Tag editor (main area):
  - Rename tag
  - Merge tags
  - Delete tag (with confirmation)
  - View images with this tag
- Manual tagging interface:
  - Select images (multi-select)
  - Add/remove tags
  - Bulk operations

### Real-time Updates

**WebSocket Protocol**:
```javascript
// Client connects to ws://localhost:8000/ws/{catalog_id}
// Server sends updates:
{
  "type": "job_progress",
  "job_id": "uuid",
  "progress": {...}
}
{
  "type": "job_complete",
  "job_id": "uuid",
  "result": {...}
}
{
  "type": "catalog_updated",
  "stats": {...}
}
```

**UI Updates**:
- Progress bars update in real-time
- Toast notifications on job completion
- Auto-refresh affected views
- Optimistic UI updates for user actions

### Safety Features

1. **Confirmation dialogs** for:
   - Delete catalog
   - Delete images/duplicates
   - Approve organization plan
   - Execute organization plan

2. **Dry-run previews**:
   - Organization plan shows all actions before approval
   - Duplicate resolution previews space savings

3. **Status indicators**:
   - Clear visual states (draft, approved, executing, complete)
   - Warning badges for low-confidence operations

4. **Audit log viewer**:
   - View all file operations
   - Filter by date, type, entity
   - Export to CSV

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1)

**Database**:
- [x] PostgreSQL connection setup
- [x] Alembic migration system
- [x] Create global schema (catalogs table)
- [x] Create catalog schema template
- [x] Pydantic models for all tables

**Celery**:
- [x] Redis connection (db=2)
- [x] Celery app configuration
- [x] Task base class with progress tracking
- [x] Result backend in PostgreSQL

**FastAPI**:
- [x] Project structure
- [x] Database session management
- [x] API endpoints (basic CRUD):
  - `/api/catalogs` - list, create, delete
  - `/api/catalogs/{id}/images` - list, get
  - `/api/catalogs/{id}/jobs` - list, create, get
- [x] WebSocket endpoint `/ws/{catalog_id}`
- [x] Error handling & validation

**Deliverable**: Working API with database, can create catalogs and submit jobs

### Phase 2: Analysis Engine (Week 2)

**Scanner Task**:
- [x] Reuse existing file discovery code
- [x] Reuse metadata extraction (ExifTool)
- [x] Reuse date extraction logic
- [x] Adapt to PostgreSQL storage
- [x] Add progress tracking
- [x] Unit tests

**Duplicate Detection Task**:
- [x] Reuse perceptual hashing code
- [x] Reuse quality scoring code
- [x] Integrate FAISS for fast search
- [x] Adapt to PostgreSQL storage
- [x] Add progress tracking
- [x] Unit tests

**AI Tagging Task** (optional for MVP):
- [x] Model loader (CLIP)
- [x] Batch inference
- [x] Tag storage
- [x] Basic tests

**Deliverable**: Working background jobs that populate catalog

### Phase 3: Web UI (Week 3)

**Setup**:
- [x] Vue 3 + Vite project
- [x] Pinia store setup
- [x] API client with axios
- [x] WebSocket client
- [x] Tailwind CSS

**Core Components**:
- [x] Catalog dashboard
- [x] Analysis tab with job controls
- [x] Browse tab with grid view
- [x] Duplicates tab (basic)
- [x] Organization planner (basic)

**Real-time Features**:
- [x] WebSocket integration
- [x] Live progress updates
- [x] Toast notifications

**Deliverable**: Working web UI for end-to-end workflow

### Phase 4: Polish & Testing (Week 4)

**UI Polish**:
- [ ] Responsive design
- [ ] Loading states
- [ ] Error handling
- [ ] Accessibility
- [ ] Dark mode (optional)

**Testing**:
- [ ] E2E tests with real photo collection
- [ ] Performance testing (100k+ images)
- [ ] Error handling tests
- [ ] Rollback tests

**Documentation**:
- [ ] User guide
- [ ] API documentation
- [ ] Deployment guide

**Deliverable**: Production-ready application

## Code Reuse Plan

### Keep (Reuse as-is or with minor changes)
- `vam_tools/analysis/metadata.py` - EXIF extraction
- `vam_tools/analysis/perceptual_hash.py` - dHash/aHash
- `vam_tools/analysis/quality_scorer.py` - Quality scoring
- `vam_tools/shared/media_utils.py` - File type detection, checksums
- `vam_tools/shared/thumbnail_utils.py` - Thumbnail generation

### Adapt (Significant changes for PostgreSQL)
- `vam_tools/analysis/scanner.py` - Change from JSON to PostgreSQL
- `vam_tools/analysis/duplicate_detector.py` - New schema, FAISS integration
- Date extraction logic - Same algorithms, different storage

### Replace (New implementation)
- `vam_tools/core/catalog.py` - Replace with SQLAlchemy models
- `vam_tools/core/database.py` - Replace with PostgreSQL connection
- `vam_tools/cli/*` - Replace with web UI
- `vam_tools/web/api.py` - Rebuild for new schema

### Delete (No longer needed)
- `vam_tools/core/migrate_to_sqlite.py` - Not needed
- `tests/.old_*` - Clean up old tests
- JSON-based catalog code

### New Code Needed
- PostgreSQL schema setup
- Celery tasks
- Organization planner logic
- AI tagging system
- Vue 3 web UI
- WebSocket real-time updates

## Migration from Old System

**Strategy**: Fresh start with import tool

1. Create new PostgreSQL catalogs
2. Provide import tool:
   ```bash
   vam-import-json --json-file old_catalog.json --catalog-name "My Photos"
   ```
3. Import tool:
   - Creates new catalog
   - Reads JSON file
   - Populates PostgreSQL tables
   - Validates data
   - Reports success/errors

4. Keep old code in `.archive/` for reference

## Success Criteria

**MVP Complete When**:
1. Can create multiple catalogs
2. Can scan directories and extract metadata
3. Can detect duplicates with visual comparison
4. Can create and execute organization plans
5. Web UI works smoothly with 100k+ images
6. All file operations are safe (approval required)
7. Real-time progress tracking works

**Production Ready When**:
- All tests passing
- Documentation complete
- Performance validated with large collections
- Error handling robust
- Deployed and running reliably

## Next Steps

1. Review and approve this design
2. Create git branch: `feature/postgresql-redesign`
3. Start Phase 1 implementation
4. Iterate and refine as needed

---

**Design approved by**: [User]
**Implementation start date**: 2025-11-10
