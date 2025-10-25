# VAM Tools V2 - Implementation Priorities

## Mission
**Get files organized with a single, correct copy of each asset first. Then enable curation and enrichment.**

## Core Philosophy

Visual Asset Management is about establishing a **single source of truth** for your visual assets. Before we can do sophisticated curation, AI-based selection, or advanced management features, we need:

1. **One correct copy** of each unique asset
2. **Correct metadata** (especially dates)
3. **Consistent organization** (chronological structure)
4. **Zero data loss** (every unique asset preserved)

Once we have this foundation, we can build value-added features on top.

## Phase 1: Core Organization (PRIORITY)

### Goal
Transform a chaotic collection of photos/videos into a clean, chronologically organized catalog with no duplicates.

### Critical Features (Must Have)

#### 1.1 Duplicate Detection & Resolution ⭐⭐⭐
- **Why**: Multiple copies waste space and create confusion
- **What**:
  - Perceptual hashing to find duplicates
  - Quality scoring to select the best copy
  - Handle different formats of the same image (RAW + JPEG, different sizes)
  - Merge EXIF metadata from all duplicates into the best copy
  - Delete or archive lower-quality duplicates
- **Status**: Not implemented
- **Iteration**: 2 (NEXT)

#### 1.2 Date Extraction & Correction ✅
- **Why**: Chronological organization requires accurate dates
- **What**:
  - Extract from EXIF, filenames, directories, filesystem
  - Confidence scoring
  - Flag suspicious dates for review
  - Human-in-the-loop for conflicts
- **Status**: COMPLETE
- **Iteration**: 1

#### 1.3 Catalog Database ✅
- **Why**: Track state, enable checkpointing, support rollback
- **What**:
  - JSON-based catalog for human readability
  - Checkpoint system for long-running operations
  - Transaction log for reversibility
  - File locking for safety
- **Status**: COMPLETE
- **Iteration**: 1

#### 1.4 Organization Execution ⭐⭐⭐
- **Why**: Actually organize the files into the target structure
- **What**:
  - YYYY-MM directory structure
  - Move/copy files to organized locations
  - Verify checksums after operations
  - Dry-run mode for safety
  - Rollback support if interrupted
- **Status**: Partially implemented (in V1, needs V2 integration)
- **Iteration**: 3

#### 1.5 Review Interface ⭐⭐
- **Why**: Human oversight for conflicts and edge cases
- **What**:
  - Web UI to review flagged issues
  - Conflict resolution (duplicate dates, missing dates)
  - Batch approval for safe operations
  - Progress monitoring
- **Status**: Basic web viewer exists, needs review features
- **Iteration**: 3

### Success Criteria
- ✅ All unique assets identified (one copy selected)
- ✅ All assets have dates (or flagged for manual review)
- ✅ Files organized into YYYY-MM structure
- ✅ Zero data loss (all unique content preserved)
- ✅ All operations logged and reversible

---

## Phase 2: Import & Maintenance (After Phase 1)

### Goal
Keep the organized catalog current as new assets arrive.

### Features

#### 2.1 Import System
- Watch import directory for new files
- Auto-analyze and integrate new assets
- Detect if new file is duplicate of existing
- Maintain organization as files arrive

#### 2.2 Incremental Updates
- Re-scan for changes in source directories
- Detect moved/renamed files
- Update catalog without re-processing everything

---

## Phase 3: Enhanced Curation (Future)

### Goal
Add value through intelligent selection and management.

### Features (Lower Priority)

#### 3.1 Burst Detection & Selection
- Group photos taken in rapid succession
- AI-based quality assessment
- Select best from burst
- Keep others in subfolder or archive

#### 3.2 AI Tagging & Classification
- Auto-tag subjects (people, places, things)
- Scene detection
- Quality scoring
- Smart collections based on AI metadata

#### 3.3 Advanced Features
- Face detection and clustering
- Location clustering
- Event detection (group by time/place)
- Smart search

---

## Implementation Roadmap

### ✅ Iteration 1: Foundation (COMPLETE)
- [x] Catalog database with checkpointing
- [x] Metadata extraction
- [x] Date extraction with confidence scoring
- [x] Basic web viewer
- [x] SHA256 checksums
- [x] Scanner with resume capability

### ⭐ Iteration 2: Duplicate Detection (CURRENT PRIORITY)
- [ ] Perceptual hashing (dHash/aHash/pHash)
- [ ] Duplicate grouping by hash similarity
- [ ] Quality scoring algorithm
  - Format (RAW > JPEG > compressed)
  - Resolution (higher is better)
  - File size (larger generally better)
  - EXIF completeness
- [ ] Primary selection logic
- [ ] EXIF metadata merging
- [ ] Duplicate visualization in web UI

### ⭐ Iteration 3: Plan & Execute (NEXT)
- [ ] Generate organization plan
  - Target paths for each file
  - Actions (move/delete/merge metadata)
  - Conflict detection
- [ ] Review UI enhancements
  - Show duplicates side-by-side
  - Manual override for selections
  - Date conflict resolution
  - Approve/reject plans
- [ ] Verification phase
  - Dry-run simulation
  - Report on operations
  - Space savings estimate
- [ ] Execution engine
  - Staged copy phase
  - Atomic operations with rollback
  - Progress tracking
  - Post-execution verification

### 🔄 Iteration 4: Import System
- [ ] Import directory watcher
- [ ] Scheduled import runs
- [ ] Incremental catalog updates
- [ ] Duplicate detection for new files

### 🎨 Iteration 5+: Enhanced Features (When Phase 1 is solid)
- [ ] Burst detection
- [ ] AI quality scoring
- [ ] Auto-tagging
- [ ] Smart collections

---

## Decision Matrix: When to Build What

| Feature | Phase | Priority | Rationale |
|---------|-------|----------|-----------|
| Duplicate detection | 1 | ⭐⭐⭐ | Can't have "one copy" without this |
| Date extraction | 1 | ⭐⭐⭐ | Already done, essential for organization |
| Organization execution | 1 | ⭐⭐⭐ | The actual goal - moving files |
| Review UI | 1 | ⭐⭐ | Human oversight needed for conflicts |
| Import system | 2 | ⭐⭐ | Needed after initial organization |
| Burst detection | 3 | ⭐ | Nice to have, not essential for organization |
| AI features | 3 | ⭐ | Value-add after core organization works |

---

## What We're NOT Building Yet

**Deliberately deferred until Phase 1 is complete:**

- ❌ Burst detection and selection
- ❌ AI-based quality assessment (beyond basic heuristics)
- ❌ Face detection
- ❌ Auto-tagging
- ❌ Smart collections
- ❌ Advanced search
- ❌ Editing/annotation features
- ❌ Sharing/export workflows

**Why**: These are all value-added features that depend on having a clean, organized foundation first.

---

## Success Metrics

### Phase 1 Complete When:
1. Can take 100,000 disorganized photos
2. Identify and eliminate duplicates
3. Establish correct dates for 95%+ of files
4. Organize into clean YYYY-MM structure
5. Complete in reasonable time (hours, not days)
6. Zero unique photos lost
7. All operations reversible

### Phase 2 Complete When:
1. Can add new photos continuously
2. Auto-detects duplicates vs existing catalog
3. Maintains organization automatically
4. Requires minimal manual intervention

### Phase 3 Complete When:
1. Curated collections based on AI analysis
2. Smart burst selection reduces storage
3. Rich metadata enables powerful search
4. User can find any photo quickly

---

## Current Status

**Phase 1 Progress: ~40%**
- ✅ Date extraction (100%)
- ✅ Catalog database (100%)
- ✅ Basic web UI (70%)
- ❌ Duplicate detection (0%)
- ❌ Organization execution (20% - needs V2 integration)
- ❌ Review UI (10% - needs conflict resolution)

**Next Up**: Duplicate detection (Iteration 2)
