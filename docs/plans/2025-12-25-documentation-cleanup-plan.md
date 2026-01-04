# Documentation Cleanup & Organization Plan
**Created:** 2025-12-25
**Status:** Proposed

## Overview

Consolidate, organize, and update all documentation using smart brevity principles. No code changes - purely documentation and file organization.

## Current State Analysis

### Root-Level Markdown Files (Should Be Moved)
```
catalog_health_plan.md                  → docs/plans/2025-12-25-catalog-health-plan.md
metadata_gap_analysis.md               → ARCHIVE (superseded by catalog_health_plan.md)
metadata_rebuild_plan.md               → ARCHIVE (superseded by catalog_health_plan.md)
thumbnail_audit_findings.md            → docs/plans/2025-12-24-thumbnail-audit-findings.md
PLAN-features-and-ui-redesign.md       → ARCHIVE (superseded by implemented features)
```

### Root-Level Scripts (Should Be Moved)
```
migrate_geohash.py                      → scripts/migrations/migrate_geohash.py
run_local.sh                           → scripts/dev/run_local.sh
```

### Misplaced Documentation
```
docs/ROADMAP.md                         → docs/roadmap/ROADMAP.md (consolidate with roadmap/)
docs/TASK_STATUS_DESIGN.md             → docs/plans/2025-12-12-task-status-design.md
docs/FEATURES.md                        → UPDATE (outdated Nov 6) or ARCHIVE
```

### Overlapping Directories
```
docs/improvements/                      → Merge with docs/roadmap/ (same purpose)
docs/development/                       → Only has status-filtering-pattern.md (move to plans/)
docs/features/                          → Only has burst-management.md (move to plans/)
```

### Documentation Issues in README.md
- References non-existent docs (USER_GUIDE.md, HOW_IT_WORKS.md, etc. in root)
- Test count outdated (580 vs 616 mentioned)
- Coverage outdated (78% vs 79%)
- Missing burst management and quality scoring features

## Proposed Actions

### Phase 1: Move Root-Level Files ✅ Priority

1. **Move to docs/plans/ (with timestamp rename)**
   ```bash
   mv catalog_health_plan.md → docs/plans/2025-12-25-catalog-health-plan.md
   mv thumbnail_audit_findings.md → docs/plans/2025-12-24-thumbnail-audit-findings.md
   ```

2. **Archive outdated plans**
   ```bash
   mv metadata_gap_analysis.md → docs/archive/2025-12-24-metadata-gap-analysis.md
   mv metadata_rebuild_plan.md → docs/archive/2025-12-24-metadata-rebuild-plan.md
   mv PLAN-features-and-ui-redesign.md → docs/archive/2025-11-xx-features-ui-redesign.md
   ```

3. **Move scripts to proper locations**
   ```bash
   mkdir -p scripts/migrations scripts/dev
   mv migrate_geohash.py → scripts/migrations/
   mv run_local.sh → scripts/dev/
   ```

### Phase 2: Reorganize docs/ Directory ✅ Priority

1. **Consolidate roadmap-related content**
   ```bash
   mv docs/ROADMAP.md → docs/roadmap/
   mv docs/improvements/* → docs/roadmap/improvements/
   rmdir docs/improvements/
   ```

2. **Move misplaced files**
   ```bash
   mv docs/TASK_STATUS_DESIGN.md → docs/plans/2025-12-12-task-status-design.md
   mv docs/development/status-filtering-pattern.md → docs/plans/2025-12-24-status-filtering-pattern.md
   mv docs/features/burst-management.md → docs/plans/2025-12-24-burst-management.md
   rmdir docs/development/ docs/features/
   ```

3. **Clean up FEATURES.md**
   - Option A: Update with current state (burst management, quality scoring, image status)
   - Option B: Archive if superseded by README features section
   - **Recommendation:** Archive - README has comprehensive feature list

### Phase 3: Update README.md 📝 High Priority

**Current Issues:**
- Doc links point to wrong paths (docs are in subdirectories, not root)
- Test/coverage stats outdated (580→616 tests, 78%→79% coverage)
- Missing new features (burst management, quality scoring, image status system)

**Updates Needed:**

1. **Fix Documentation Links**
   ```markdown
   OLD: [User Guide](./docs/USER_GUIDE.md)
   NEW: [User Guide](./docs/guides/USER_GUIDE.md)

   OLD: [How It Works](./docs/HOW_IT_WORKS.md)
   NEW: [How It Works](./docs/technical/HOW_IT_WORKS.md)
   ```

2. **Update Stats**
   ```markdown
   OLD: 580 passing tests, 78% coverage
   NEW: 616 passing tests, 79% coverage
   ```

3. **Add New Features**
   ```markdown
   - **Quality Scoring** - Multi-factor scoring (format, resolution, size, EXIF)
   - **Burst Management** - Detect continuous shooting sequences, select best from burst
   - **Image Status System** - Track active/archived/flagged/rejected/selected states
   - **Corruption Detection** - Identify and flag corrupt/empty files
   ```

4. **Simplify "Quick Links" section** (smart brevity)
   ```markdown
   OLD: Long list of links in Documentation section
   NEW: Consolidate into 4 categories:
        - Getting Started (guides/)
        - Technical Docs (technical/)
        - Plans & Roadmap (plans/, roadmap/)
        - Help (TROUBLESHOOTING, GitHub)
   ```

### Phase 4: Update docs/README.md 📝 Medium Priority

**Current State:**
- Well-organized structure
- Accurate directory layout
- Links work correctly

**Updates Needed:**

1. **Remove obsolete sections**
   ```markdown
   DELETE: docs/development/ (being removed)
   DELETE: docs/features/ (being removed)
   DELETE: docs/improvements/ (merging with roadmap/)
   ```

2. **Add new sections**
   ```markdown
   ADD: docs/roadmap/improvements/ (merged from improvements/)
   ```

3. **Update feature list** to match README.md new features

### Phase 5: Consolidate Duplicate/Overlapping Docs 🔄 Low Priority

**Candidates for consolidation:**

1. **Plans directory cleanup**
   - 14 design/plan documents spanning Nov-Dec 2025
   - Many implemented → move to archive/implemented/
   - Keep only: active plans, recent designs, reference docs

2. **Archive organization**
   - Add subdirectories: archive/summaries/, archive/migrations/, archive/implemented/
   - Move completion summaries to archive/summaries/
   - Move migration docs to archive/migrations/

## Final Directory Structure

```
lumina/
├── README.md                          # Updated: links, stats, features
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
│
├── docs/
│   ├── README.md                      # Updated: structure, removed obsolete
│   ├── guides/                        # User and setup guides (unchanged)
│   ├── technical/                     # Architecture and implementation (unchanged)
│   ├── plans/                         # Design docs (reorganized)
│   │   ├── 2025-12-25-catalog-health-plan.md
│   │   ├── 2025-12-24-thumbnail-audit-findings.md
│   │   ├── 2025-12-24-status-filtering-pattern.md
│   │   ├── 2025-12-24-burst-management.md
│   │   ├── 2025-12-22-duplicate-management-features-design.md
│   │   ├── 2025-12-12-task-status-design.md
│   │   ├── ... (other plans)
│   │
│   ├── roadmap/                       # Roadmaps and improvement proposals
│   │   ├── ROADMAP.md
│   │   ├── composition-quality-scoring.md
│   │   └── improvements/              # Merged from docs/improvements/
│   │       └── duplicate-detection-progress-aggregation.md
│   │
│   ├── research/                      # Research docs (unchanged)
│   └── archive/                       # Historical/completed docs
│       ├── summaries/                 # Project completion summaries
│       ├── migrations/                # Migration documentation
│       ├── implemented/               # Implemented plans
│       ├── 2025-12-24-metadata-gap-analysis.md
│       ├── 2025-12-24-metadata-rebuild-plan.md
│       └── ... (existing archive files)
│
├── scripts/
│   ├── README.md                      # Updated: mention new subdirs
│   ├── dev/                           # Development scripts
│   │   └── run_local.sh
│   ├── migrations/                    # One-off migration scripts
│   │   └── migrate_geohash.py
│   ├── run_tests.sh
│   ├── kill_tests.sh
│   └── iphone-mount.sh
│
└── (all other directories unchanged)
```

## Benefits

1. **Clarity** - All docs in proper locations, easy to find
2. **Maintenance** - Clear separation of active vs archived
3. **Onboarding** - New contributors find current info, not stale plans
4. **Smart Brevity** - README focuses on essentials, links to details
5. **History** - Archived docs preserve development journey

## Implementation Order

1. Phase 1: Move root files (5 min) ← **START HERE**
2. Phase 2: Reorganize docs/ (10 min)
3. Phase 3: Update README.md (15 min) ← **HIGHEST IMPACT**
4. Phase 4: Update docs/README.md (5 min)
5. Phase 5: Consolidate archives (15 min) ← Optional

**Total Time:** ~50 minutes
**Risk:** None (no code changes)
**Impact:** High (better navigation, accurate information)

## Rollback Plan

All moves are reversible with git:
```bash
git checkout HEAD -- <file>  # Restore individual file
git reset --hard HEAD        # Restore all changes
```

## Success Criteria

- [ ] No markdown files in project root (except README.md, LICENSE)
- [ ] No scripts in project root (except Makefile)
- [ ] README.md links all work
- [ ] README.md features section current
- [ ] docs/ structure matches docs/README.md
- [ ] All plans dated with YYYY-MM-DD prefix
- [ ] Obsolete docs in archive/
