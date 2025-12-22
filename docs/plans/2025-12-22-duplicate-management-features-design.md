# Duplicate Management Features Design

**Date:** 2025-12-22
**Status:** Design Approved

## Overview

Two features to improve duplicate image management:

1. **Auto-tag duplicate groups** - Automatically tag images with their duplicate group ID for easy navigation
2. **Duplicate review UI** - Interactive interface for reviewing and resolving duplicate groups

## Goals

- Make duplicate groups easily discoverable via tags
- Provide efficient workflow for reviewing and resolving duplicates
- Support safe "deletion" (soft delete/exclude) with undo capability
- Maintain consistency with existing catalog browse experience
- Create full audit trail for accountability

## Feature 1: Auto-Tag Duplicate Groups

### Tag Format

**Pattern:** `dup-{first_8_chars_of_primary_dhash}`

**Examples:**
- `dup-cc4cccce` (short version, displayed in UI)
- `dup-cc4cccceced062cf` (full hash, shown on hover)

**Rationale:**
- Similar duplicate groups have similar tag prefixes, enabling discovery of related groups
- Groups with hashes like `dup-cc4cccce` and `dup-cc4ccccd` sort together
- 8 characters balance readability vs collision avoidance
- Full hash in metadata allows precise identification on hover

### Tag Creation

**When:** Automatically during duplicate detection finalizer task

**Process:**
1. After creating `DuplicateGroup` records
2. For each group:
   - Extract first 8 characters of primary image's `dhash`
   - Create tag: `dup-{hash_prefix}`
   - Store in `Tag` table with:
     - `category = "system"` (system-generated)
     - `metadata = {full_hash, duplicate_group_id, similarity_type, created_by}`
3. Apply tag to all group members via `ImageTag` table
   - `confidence = 100` (system-generated)
   - `source = "duplicate_detection"`

**Tag Collision Handling:**

If two groups have same 8-char prefix (rare):
- Append group ID: `dup-cc4cccce-42`, `dup-cc4cccce-87`
- OR use 10 characters instead to reduce collisions

### Database Schema

**Tag model (existing):**
```python
class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True)
    catalog_id = Column(UUID, ForeignKey("catalogs.id"))
    name = Column(String, nullable=False)
    category = Column(String)  # Use "system" for duplicate tags
    metadata = Column(JSONB)  # Store full hash, group_id
```

**Metadata structure:**
```json
{
  "full_hash": "cc4cccceced062cf",
  "duplicate_group_id": 42,
  "similarity_type": "perceptual",
  "created_by": "duplicate_detection"
}
```

### UI Display

- **Tag list:** Show short version `dup-cc4cccce`
- **Hover tooltip:** Show full hash `dup-cc4cccceced062cf`
- **Clickable:** Filter to show all images with that tag (existing functionality)

## Feature 2: Duplicate Review UI

### Architecture

**Technology Stack:**
- Vue 3 (CDN, no build step) - matches existing `index.html`
- Axios for API calls
- Existing CSS (`dark-theme.css`, `styles.css`)
- New file: `/vam_tools/web/static/duplicates.html`

**Layout Pattern:**
Three-panel layout matching existing catalog browse:
- Left: Group navigator
- Center: Grid view or inspect view
- Right: Image details panel

### Database Schema Changes

#### 1. Image Model Updates

**Add exclusion metadata:**
```python
class Image(Base):
    # ... existing fields ...

    # Update status to support: "active", "excluded", "pending"
    status = Column(String, default="pending")

    # NEW: Track exclusion context
    exclusion_metadata = Column(JSONB, nullable=True)
```

**Exclusion metadata structure:**
```json
{
  "reason": "duplicate_resolution",
  "group_id": 42,
  "resolved_at": "2025-12-22T14:30:00Z",
  "action_id": "uuid-of-audit-entry"
}
```

#### 2. New Table: DuplicateAction

**Audit log for all duplicate resolution actions:**
```python
class DuplicateAction(Base):
    """Audit log for duplicate resolution actions with undo capability."""

    __tablename__ = "duplicate_actions"

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    catalog_id = Column(UUID, ForeignKey("catalogs.id", ondelete="CASCADE"), nullable=False)
    group_id = Column(Integer, ForeignKey("duplicate_groups.id", ondelete="SET NULL"))
    action_type = Column(String, nullable=False)  # "keep_recommended", "keep_all", "keep_selected", "split_group"
    affected_images = Column(JSONB, nullable=False)  # [{image_id, old_status, new_status}, ...]
    metadata = Column(JSONB)  # {kept_image_ids, excluded_image_ids, reason, etc.}
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    undone_at = Column(DateTime, nullable=True)

    # Relationships
    catalog = relationship("Catalog")
    group = relationship("DuplicateGroup")
```

**Indexes:**
```sql
CREATE INDEX idx_duplicate_actions_catalog ON duplicate_actions(catalog_id);
CREATE INDEX idx_duplicate_actions_group ON duplicate_actions(group_id);
CREATE INDEX idx_duplicate_actions_undone ON duplicate_actions(undone_at) WHERE undone_at IS NULL;
```

### API Endpoints

Add to `/vam_tools/api/routers/catalogs.py`:

#### 1. Get Duplicate Groups for Review

```python
@router.get("/{catalog_id}/duplicates/review")
def get_duplicates_for_review(
    catalog_id: uuid.UUID,
    reviewed: bool = None,
    min_group_size: int = None,
    similarity_type: str = None,  # "exact" or "perceptual"
    tag_prefix: str = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get duplicate groups with preview thumbnails for review UI.

    Returns:
        {
            groups: [{
                id: int,
                primary_image_id: str,
                similarity_type: str,
                confidence: int,
                member_count: int,
                preview_thumbnails: [urls],  # First 3-5 thumbnails
                tag: str,  # "dup-cc4cccce"
                reviewed: bool,
                created_at: datetime
            }],
            total_count: int,
            unreviewed_count: int
        }
    """
```

#### 2. Get Detailed Group View

```python
@router.get("/{catalog_id}/duplicates/{group_id}/detail")
def get_duplicate_group_detail(
    catalog_id: uuid.UUID,
    group_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all images in a group with metadata and recommendations.

    Returns:
        {
            group: {id, primary_image_id, tag, similarity_type, ...},
            members: [{
                image_id: str,
                thumbnail_url: str,
                metadata: {...},
                similarity_score: int,
                quality_score: int,
                file_size: int,
                resolution: {width, height},
                date_taken: datetime,
                is_recommended: bool,
                recommendation_score: float  # 0-100
            }],
            resolution_history: [past DuplicateAction records]
        }
    """
```

#### 3. Resolve Duplicate Group

```python
@router.post("/{catalog_id}/duplicates/{group_id}/resolve")
def resolve_duplicate_group(
    catalog_id: uuid.UUID,
    group_id: int,
    action: {
        "action_type": str,  # "keep_recommended" | "keep_all" | "keep_selected"
        "keep_image_ids": List[str],
        "exclude_image_ids": List[str]
    },
    db: Session = Depends(get_db)
):
    """
    Resolve a duplicate group by excluding unwanted images.

    Process:
    1. Validate at least one image kept
    2. Update excluded images: status = "excluded", add exclusion_metadata
    3. Create DuplicateAction audit entry
    4. Mark group as reviewed = True

    Returns:
        {
            success: bool,
            action_id: uuid,
            affected_images: int
        }
    """
```

#### 4. Undo Duplicate Action

```python
@router.post("/{catalog_id}/duplicates/actions/{action_id}/undo")
def undo_duplicate_action(
    catalog_id: uuid.UUID,
    action_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Undo a previous duplicate resolution action.

    Process:
    1. Load DuplicateAction record
    2. Verify not already undone
    3. Restore image statuses from affected_images
    4. Clear exclusion_metadata
    5. Mark action.undone_at = now()
    6. Mark group as reviewed = False

    Returns:
        {
            success: bool,
            restored_images: int,
            group_id: int
        }
    """
```

#### 5. Split Duplicate Group

```python
@router.post("/{catalog_id}/duplicates/{group_id}/split")
def split_duplicate_group(
    catalog_id: uuid.UUID,
    group_id: int,
    new_groups: List[List[str]],  # [[image_ids], [image_ids]]
    db: Session = Depends(get_db)
):
    """
    Split wrongly-grouped duplicates into separate groups.

    Process:
    1. Delete original DuplicateGroup
    2. Create new DuplicateGroup for each split
    3. Create new tags for new groups
    4. Update DuplicateMember records
    5. Create DuplicateAction audit entry

    Returns:
        {
            success: bool,
            new_group_ids: [int],
            action_id: uuid
        }
    """
```

### Image Recommendation Algorithm

**Weighted scoring to determine "best" image in each group:**

```python
def calculate_recommendation_score(image, group_images):
    """
    Returns score 0-100 where higher = better keeper candidate.

    Components:
    - Quality: 50% weight (uses image.quality_score)
    - Age: 30% weight (older = better, preserves originals)
    - Size: 20% weight (larger resolution = better)
    """
    # Quality component (50%)
    quality_score = image.quality_score or 50  # Default if missing
    quality_component = quality_score * 0.5

    # Age component (30%) - older is better
    oldest_date = min(img.dates.get('taken') for img in group_images)
    image_date = image.dates.get('taken')
    if image_date == oldest_date:
        age_component = 30  # Full points for oldest
    else:
        days_diff = (image_date - oldest_date).days
        age_component = max(0, 30 - (days_diff * 0.1))

    # Size component (20%) - larger resolution is better
    resolution = image.metadata_json.get('width', 0) * image.metadata_json.get('height', 0)
    max_resolution = max(
        img.metadata_json.get('width', 0) * img.metadata_json.get('height', 0)
        for img in group_images
    )
    if max_resolution > 0:
        size_component = (resolution / max_resolution) * 20
    else:
        size_component = 10  # Default if no resolution data

    return quality_component + age_component + size_component

# Tie-breaking rules:
# 1. If scores within 5 points: prefer oldest
# 2. If still tied: prefer largest file size
# 3. If still tied: prefer primary_image_id
```

**Edge cases:**
- Missing quality scores: default to 50
- Missing dates: use file `created_at` timestamp
- Missing resolution: use file size as proxy

### UI Layout & Views

#### View State 1: Grid View (Default)

```
┌─────────────────────┬────────────────────────────────┬─────────────────────┐
│ Group Navigator     │ Grid View                      │ Image Details       │
│                     │                                │                     │
│ ⚡ Unreviewed (142) │ Group: dup-cc4cccce (5 images)│ IMG_1234.jpg        │
│                     │ [Keep Rec] [Keep All] [Custom] │                     │
│ [Show Filters ▼]    │ ────────────────────────────── │ ⭐ Recommended: BEST│
│                     │                                │ Score: 92/100       │
│ ┌─────────────────┐ │ ┌────┐ ┌────┐ ┌────┐         │                     │
│ │ dup-cc4cccce   │ │ │ ⭐ │ │    │ │    │         │ Quality: 85         │
│ │ [3 thumbs]     │ │ │IMG1│ │IMG2│ │IMG3│         │ Date: 2024-03-15    │
│ │ 5 images       │ │ │ ✓  │ │    │ │    │         │ Size: 4032×3024     │
│ └─────────────────┘ │ └────┘ └────┘ └────┘         │ File: 8.2 MB        │
│                     │                                │                     │
│ ┌─────────────────┐ │ ┌────┐ ┌────┐                │ Tags:               │
│ │ dup-ab3d2e1f   │ │ │IMG4│ │IMG5│                │ • dup-cc4cccce...   │
│ │ [3 thumbs]     │ │ └────┘ └────┘                │ • vacation          │
│ │ 12 images      │ │                                │                     │
│ └─────────────────┘ │ Single click: details →       │ ☐ Keep this image  │
│                     │ Double click: inspect mode     │ [Compare with...]   │
└─────────────────────┴────────────────────────────────┴─────────────────────┘
```

**Interactions:**
- **Left sidebar:** Click group → Load grid view
- **Grid:** Single click → Show details in right panel
- **Grid:** Double click → Switch to inspect view
- **Checkboxes:** Toggle images to keep/exclude
- **Quick actions:**
  - "Keep Recommended" → Exclude all except recommended
  - "Keep All" → Mark reviewed, no exclusions
  - "Custom" → Use manual checkbox selection

#### View State 2: Inspect View (After Double-Click)

```
┌─────────────────────┬────────────────────────────────┬─────────────────────┐
│ Group Navigator     │ Single Image + Filmstrip       │ Image Details       │
│                     │                                │                     │
│ (same as grid view) │ [← Back to Grid] dup-cc4cccce │ (same as grid view) │
│                     │ ────────────────────────────── │                     │
│                     │                                │                     │
│                     │      ┌─────────────────┐      │                     │
│                     │      │                 │      │                     │
│                     │      │   Large Image   │      │                     │
│                     │      │    IMG_1234     │      │                     │
│                     │      │      ⭐ ✓       │      │                     │
│                     │      └─────────────────┘      │                     │
│                     │                                │                     │
│                     │ ┌───┐ ┌───┐ ┌───┐ ┌───┐      │                     │
│                     │ │IMG│ │IMG│ │IMG│ │IMG│      │                     │
│                     │ │ 1 │ │ 2 │ │ 3 │ │ 4 │      │                     │
│                     │ └───┘ └───┘ └───┘ └───┘      │                     │
│                     │  ⭐✓                           │                     │
│                     │                                │                     │
│                     │ [Keep This] [Exclude This]     │                     │
└─────────────────────┴────────────────────────────────┴─────────────────────┘
```

**Interactions:**
- **Back button:** Return to grid view (preserves selection state)
- **Filmstrip:** Click thumbnail → Switch image
- **Keyboard:** Arrow keys navigate, ESC returns to grid
- **Actions:** Keep/exclude current image

#### Progressive Disclosure Filters

**When "Show Filters" clicked:**
```
┌────────────────────────┐
│ Filters                │
│ ────────────────────── │
│ Status:                │
│ ○ Unreviewed (default) │
│ ○ Reviewed             │
│ ○ All                  │
│                        │
│ Group Size:            │
│ ☐ 2-5 images           │
│ ☐ 6-10 images          │
│ ☐ 11-20 images         │
│ ☐ 21+ images           │
│                        │
│ Similarity:            │
│ ○ Exact                │
│ ○ Perceptual           │
│ ○ All                  │
│                        │
│ Sort by:               │
│ ▼ Newest first         │
│                        │
│ Search tag:            │
│ [dup-cc4c____]         │
│                        │
│ [Apply] [Reset]        │
└────────────────────────┘
```

### Vue Component Structure

**Main app data structure:**
```javascript
{
  // View state
  currentView: 'grid' | 'inspect',

  // Panels
  showLeftPanel: true,
  showRightPanel: true,

  // Groups
  duplicateGroups: [],
  totalGroups: 0,
  unreviewedCount: 0,

  // Filters
  showFilters: false,
  filters: {reviewed, minGroupSize, similarityType, tagPrefix, sortBy},

  // Selected group (grid view)
  selectedGroupId: null,
  groupMembers: [],
  recommendedImageId: null,

  // Selected image (inspect view)
  selectedImageId: null,
  selectedImageIndex: 0,

  // Selection state (persists across views)
  imagesToKeep: Set<imageId>,

  // Right panel
  inspectedImage: null,

  // Undo
  lastAction: {action_id, description},

  // UI
  loading: false,
  error: null
}
```

**Key methods:**
- `selectGroup(groupId)` - Load group members, switch to grid view
- `onImageSingleClick(image)` - Show details in right panel
- `onImageDoubleClick(image)` - Switch to inspect view
- `backToGrid()` - Return to grid view
- `navigateFilmstrip(direction)` - Next/previous in filmstrip
- `toggleImageKeep(imageId)` - Toggle keep/exclude
- `resolveGroup(actionType)` - Exclude images, create audit log
- `undoLastAction()` - Restore previous state

**Keyboard shortcuts:**
- Arrow keys: Navigate filmstrip (in inspect mode)
- ESC: Return to grid view
- F6/F7: Toggle panels (existing pattern)

### Error Handling & Edge Cases

#### 1. Groups with No Valid Recommendation

**Scenario:** All images missing quality_score, dates, resolution

**Handling:**
- Fallback: Select `primary_image_id` as recommended
- Show warning in UI: "⚠ Unable to determine best image"
- User can still manually select

#### 2. User Excludes Recommended Image

**Scenario:** User disagrees with recommendation

**Handling:**
- Allow it (user knows best)
- Recalculate recommendation from remaining images
- Show new recommended image with note: "New recommendation (previous excluded)"

#### 3. Exclude All Images

**Scenario:** User tries to exclude all images in group

**Handling:**
- Validation: Prevent with error message
- UI error: "Must keep at least one image in group"
- Alternative: Suggest "Mark as not duplicates" action instead

#### 4. Concurrent Modifications

**Scenario:** User A and User B reviewing same group simultaneously

**Handling:**
- Optimistic locking: Check `group.updated_at` before resolving
- Return `409 Conflict` if group changed since loaded
- UI shows: "This group was modified. Reload to see current state."

#### 5. Undo After Group Deleted

**Scenario:** Group deleted after action created

**Handling:**
- Check if group exists before undo
- If deleted: Show error "Cannot undo: group was deleted"
- Audit log preserves enough metadata to describe what was undone

#### 6. Missing/Deleted Images

**Scenario:** Image file deleted from disk but record exists

**Handling:**
- Show placeholder thumbnail with warning icon
- Allow resolution (can still exclude missing images)
- Log warning in action metadata

#### 7. Tag Collisions

**Scenario:** Two groups have same 8-char hash prefix (rare)

**Handling:**
- Append group ID: `dup-cc4cccce-42`, `dup-cc4cccce-87`
- OR increase to 10 characters to reduce collisions

#### 8. Very Large Groups

**Scenario:** Group with 50+ images

**Handling:**
- Paginate filmstrip display (show 20 at a time with scroll)
- Lazy load thumbnails as user scrolls
- Warning in UI: "⚠ Large group - may take time to load"

**API Error Responses:**
```json
// Validation errors
400: {"error": "Must keep at least one image"}
400: {"error": "Invalid action_type"}

// Not found
404: {"error": "Duplicate group not found"}
404: {"error": "Action not found or already undone"}

// Conflicts
409: {"error": "Group modified since loaded", "reload": true}

// Server errors
500: {"error": "Failed to resolve group", "details": "..."}
```

### Testing Strategy

#### Unit Tests

**Test coverage:**
```python
# Recommendation algorithm
def test_calculate_recommendation_score_all_data():
    """Test scoring with complete metadata."""

def test_calculate_recommendation_score_missing_quality():
    """Test fallback when quality_score is None."""

def test_calculate_recommendation_score_missing_dates():
    """Test fallback to created_at when dates missing."""

def test_calculate_recommendation_score_tie_breaking():
    """Test tie-breaking rules."""

# Tag creation
def test_create_duplicate_tags_basic():
    """Test tag creation with valid dhash."""

def test_create_duplicate_tags_collision():
    """Test handling of hash prefix collisions."""

def test_create_duplicate_tags_missing_dhash():
    """Test skipping when primary image has no dhash."""

# Resolution actions
def test_resolve_group_keep_recommended():
    """Test excluding all except recommended."""

def test_resolve_group_keep_all():
    """Test marking reviewed without exclusions."""

def test_resolve_group_keep_selected():
    """Test custom selection."""

def test_resolve_group_validation_error():
    """Test error when excluding all images."""

# Undo functionality
def test_undo_duplicate_action():
    """Test restoring image statuses."""

def test_undo_already_undone():
    """Test preventing double undo."""

def test_undo_deleted_group():
    """Test handling when group no longer exists."""
```

#### Integration Tests

**Test scenarios:**
```python
def test_full_duplicate_workflow():
    """
    End-to-end test:
    1. Run duplicate detection
    2. Verify tags created
    3. Load review UI endpoint
    4. Resolve group
    5. Verify status changes
    6. Verify audit log
    7. Undo action
    8. Verify restoration
    """

def test_concurrent_resolution():
    """
    Test race condition handling:
    1. Load group in session A
    2. Resolve group in session B
    3. Attempt resolve in session A
    4. Verify 409 Conflict returned
    """

def test_backfill_existing_groups():
    """
    Test migration:
    1. Create groups without tags
    2. Run backfill script
    3. Verify all groups have tags
    """
```

#### Manual Testing Checklist

- [ ] Grid view shows all group members
- [ ] Recommended image has star indicator
- [ ] Single click loads details in right panel
- [ ] Double click switches to inspect view
- [ ] Filmstrip navigation works (click + keyboard)
- [ ] "Keep Recommended" excludes all others
- [ ] "Keep All" marks reviewed without exclusions
- [ ] Custom selection with checkboxes works
- [ ] Undo restores excluded images
- [ ] Tags show short version, full on hover
- [ ] Filters work (reviewed, size, similarity)
- [ ] Large groups (20+) handle gracefully
- [ ] Error messages show for validation failures
- [ ] Concurrent modification shows conflict error
- [ ] Selection state persists when switching views

### Performance Considerations

**Optimizations:**
- **Pagination:** Load 50 groups at a time
- **Thumbnail caching:** Browser caches existing thumbnail URLs
- **Lazy loading:** Intersection observer for off-screen thumbnails
- **API caching:** Cache group list for 30s, invalidate on resolution
- **Database indexes:** Already exist on duplicate_groups and duplicate_members

**Expected performance:**
- Group list load: < 500ms for 1000 groups
- Group detail load: < 200ms for 20 images
- Resolution action: < 300ms
- Undo action: < 200ms

### Migration & Deployment

#### Database Migration

```sql
-- 1. Add exclusion_metadata to images
ALTER TABLE images
ADD COLUMN IF NOT EXISTS exclusion_metadata JSONB DEFAULT NULL;

-- 2. Create duplicate_actions table
CREATE TABLE IF NOT EXISTS duplicate_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    catalog_id UUID NOT NULL REFERENCES catalogs(id) ON DELETE CASCADE,
    group_id INTEGER REFERENCES duplicate_groups(id) ON DELETE SET NULL,
    action_type VARCHAR NOT NULL,
    affected_images JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    undone_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_duplicate_actions_catalog
ON duplicate_actions(catalog_id);

CREATE INDEX IF NOT EXISTS idx_duplicate_actions_group
ON duplicate_actions(group_id);

CREATE INDEX IF NOT EXISTS idx_duplicate_actions_undone
ON duplicate_actions(undone_at)
WHERE undone_at IS NULL;

-- 3. Update Image status values (if needed)
-- Existing: "pending"
-- New: "active", "excluded"
UPDATE images SET status = 'active' WHERE status = 'pending';
```

#### Backfill Script

```python
def backfill_duplicate_tags(catalog_id: str):
    """
    One-time script to create tags for existing duplicate groups.
    Run after deploying tag creation code.
    """
    from vam_tools.db import CatalogDatabase
    from vam_tools.db.models import DuplicateGroup

    with CatalogDatabase(catalog_id) as db:
        groups = db.session.query(DuplicateGroup).all()
        print(f"Backfilling tags for {len(groups)} groups...")

        _create_duplicate_tags(catalog_id, groups, db.session)

        print(f"Created tags for {len(groups)} groups")
```

#### Rollout Plan

1. **Deploy database migration** (backward compatible)
   - Add `exclusion_metadata` column
   - Create `duplicate_actions` table
   - Add indexes

2. **Deploy backend changes**
   - Update `duplicates_finalizer_task` to create tags
   - Add new API endpoints
   - Deploy to production

3. **Backfill tags**
   - Run backfill script for each catalog
   - Monitor for errors

4. **Deploy UI**
   - Add `/duplicates` route to web server
   - Deploy `duplicates.html` file
   - Test with small catalog first

5. **Monitor**
   - Check for API errors
   - Verify performance metrics
   - Gather user feedback

6. **Enable for all catalogs**
   - Announce feature availability
   - Update documentation

## Summary

### Feature 1: Auto-Tag Duplicate Groups

- **Tag format:** `dup-{8-char-hash}` (full hash on hover)
- **Creation:** Automatic during duplicate detection
- **Storage:** Standard Tag/ImageTag models
- **UI:** Integrated with existing tag system

### Feature 2: Duplicate Review UI

- **Layout:** Three-panel (navigator, grid/inspect, details)
- **Views:** Grid view + inspect view with filmstrip
- **Actions:** Keep recommended, keep all, custom selection
- **Safety:** Soft delete (exclude) with full undo capability
- **Audit:** Complete action history in `duplicate_actions` table

### Key Design Decisions

1. **Soft delete** - Exclude images instead of deleting
2. **Audit log** - Full traceability with undo
3. **Weighted scoring** - Balanced recommendation algorithm
4. **Consistent UX** - Matches existing catalog browse pattern
5. **Progressive disclosure** - Simple by default, powerful when needed

### Files to Modify/Create

**Database:**
- Migration script for schema changes
- Backfill script for existing groups

**Backend:**
- `vam_tools/jobs/parallel_duplicates.py` - Add tag creation
- `vam_tools/api/routers/catalogs.py` - Add 5 new endpoints
- `vam_tools/db/models.py` - Add DuplicateAction model

**Frontend:**
- `vam_tools/web/static/duplicates.html` - New review UI
- `vam_tools/web/static/styles.css` - Minor additions for filmstrip

**Tests:**
- `tests/jobs/test_duplicate_tags.py` - Tag creation tests
- `tests/api/test_duplicate_review.py` - API endpoint tests
- `tests/integration/test_duplicate_workflow.py` - End-to-end tests

### Next Steps

Ready to set up for implementation?
