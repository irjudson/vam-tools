# VAM Tools V2 - Complete Design Document

> **Note**: See [PRIORITIES.md](PRIORITIES.md) for the current implementation roadmap.
> **Current Focus**: Core organization (duplicates → organization → execution) before advanced features.

## Overview

A comprehensive photo catalog management system for organizing 100,000+ images with:
- Zero data loss guarantee
- Chronological organization
- Intelligent duplicate handling
- Conflict resolution with human oversight
- Plan-verify-execute workflow

**Advanced features** (burst detection, AI curation) are deferred until core organization is solid.

## Core Principles

1. **Never lose images** - All operations reversible until final execution
2. **Organize chronologically** - YYYY-MM directory structure
3. **One correct copy** - Identify and eliminate duplicates, keeping the best version
4. **Human in the loop** - Flag conflicts for review, auto-handle clear cases
5. **Foundation first** - Core organization before advanced curation features

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VAM Tools V2                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Analysis   │→ │    Review    │→ │   Verify     │     │
│  │    Engine    │  │   Web UI     │  │   & Execute  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         ↓                 ↓                  ↓              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Catalog Database (catalog.json)          │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                    ↓              │
│  ┌──────────────┐                    ┌──────────────┐     │
│  │   Import     │                    │   Execution  │     │
│  │   Watcher    │                    │   Engine     │     │
│  └──────────────┘                    └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Catalog Database Schema

### catalog.json Structure

```json
{
  "version": "2.0.0",
  "catalog_path": "/path/to/organized_catalog",
  "catalog_id": "uuid-1234-5678",
  "created": "2025-10-25T13:05:00Z",
  "last_updated": "2025-10-25T14:30:00Z",

  "configuration": {
    "source_directories": [
      "/external-drive-1/photos",
      "/external-drive-2/photos",
      "~/Pictures"
    ],
    "import_directory": "/import",
    "date_format": "YYYY-MM",
    "file_naming": "{date}_{time}_{checksum}.{ext}",
    "burst_threshold_seconds": 10,
    "burst_min_images": 3,
    "ai_model": "hybrid",
    "video_support": true
  },

  "state": {
    "phase": "analyzing|reviewing|verified|executing|complete",
    "last_checkpoint": "2025-10-25T14:25:00Z",
    "checkpoint_interval_seconds": 300,
    "images_processed": 45230,
    "images_total": 100000
  },

  "statistics": {
    "total_images": 100000,
    "total_videos": 5000,
    "total_size_bytes": 500000000000,
    "organized": 95000,
    "needs_review": 5000,
    "no_date": 500,
    "duplicate_groups": 15000,
    "duplicates_total": 30000,
    "burst_groups": 2000,
    "burst_images": 10000,
    "unique_images": 55000
  },

  "images": {
    "sha256:abc123...": {
      "id": "sha256:abc123...",
      "source_path": "/external-drive-1/photos/IMG001.jpg",
      "file_type": "image",
      "format": "JPEG",
      "size_bytes": 2500000,
      "resolution": [3840, 2160],
      "checksum": "sha256:abc123...",

      "dates": {
        "exif": {
          "DateTimeOriginal": "2023-06-15T12:00:00",
          "CreateDate": "2023-06-15T12:00:00",
          "ModifyDate": "2023-06-16T08:00:00"
        },
        "filename": null,
        "directory": "2023-06",
        "filesystem": {
          "created": "2024-01-01T00:00:00",
          "modified": "2024-01-01T00:00:00"
        },
        "selected_date": "2023-06-15T12:00:00",
        "selected_source": "exif",
        "confidence": 95,
        "suspicious": false,
        "user_verified": false
      },

      "metadata": {
        "exif": {
          "Make": "Canon",
          "Model": "EOS 5D Mark IV",
          "LensModel": "EF 24-70mm f/2.8L II USM",
          "FocalLength": "50mm",
          "FNumber": "f/2.8",
          "ISO": 400,
          "ExposureTime": "1/125",
          "GPS": {
            "Latitude": 37.7749,
            "Longitude": -122.4194
          }
        },
        "merged_from": ["sha256:def456...", "sha256:ghi789..."]
      },

      "duplicate_group_id": "dup_xyz",
      "duplicate_role": "primary|duplicate",
      "burst_group_id": null,
      "burst_role": null,

      "status": "pending|analyzing|needs_review|approved|executed",
      "issues": ["date_conflict", "suspicious_date"],

      "plan": {
        "action": "move|delete|skip",
        "target_path": "/catalog/2023-06/2023-06-15_120000.jpg",
        "target_exists": false,
        "target_checksum": null,
        "burst_folder": null
      },

      "execution": {
        "executed": false,
        "executed_at": null,
        "verified": false,
        "rollback_info": {
          "original_path": "/external-drive-1/photos/IMG001.jpg",
          "backup_checksum": "sha256:abc123..."
        }
      }
    },

    "sha256:def456...": {
      "id": "sha256:def456...",
      "source_path": "/external-drive-1/photos/IMG001_small.jpg",
      "duplicate_group_id": "dup_xyz",
      "duplicate_role": "duplicate",
      "plan": {
        "action": "delete",
        "reason": "lower_quality_duplicate"
      }
    }
  },

  "duplicate_groups": {
    "dup_xyz": {
      "id": "dup_xyz",
      "images": ["sha256:abc123...", "sha256:def456...", "sha256:ghi789..."],
      "primary": "sha256:abc123...",
      "perceptual_hash": "0101101...",
      "quality_scores": {
        "sha256:abc123...": {
          "format_score": 100,
          "resolution_score": 100,
          "size_score": 100,
          "exif_score": 95,
          "total_score": 98.75
        },
        "sha256:def456...": {
          "format_score": 100,
          "resolution_score": 50,
          "size_score": 40,
          "exif_score": 90,
          "total_score": 70
        }
      },
      "date_conflict": false,
      "needs_review": false,
      "user_override": null
    }
  },

  "burst_groups": {
    "burst_abc": {
      "id": "burst_abc",
      "images": ["sha256:111...", "sha256:222...", "sha256:333..."],
      "primary": "sha256:222...",
      "time_span_seconds": 3.5,
      "ai_scores": {
        "sha256:111...": {
          "sharpness": 85,
          "exposure": 90,
          "composition": 80,
          "total": 85
        },
        "sha256:222...": {
          "sharpness": 95,
          "exposure": 92,
          "composition": 90,
          "total": 92.3
        }
      },
      "needs_review": false,
      "user_override": null
    }
  },

  "review_queue": [
    {
      "id": "review_001",
      "type": "date_conflict",
      "priority": "high",
      "images": ["sha256:...", "sha256:..."],
      "description": "Duplicate group has conflicting EXIF dates",
      "details": {
        "image1_date": "2023-01-15",
        "image2_date": "2023-01-20"
      },
      "status": "pending|reviewing|resolved",
      "resolution": null,
      "resolved_at": null,
      "resolved_by": "user|auto"
    },
    {
      "id": "review_002",
      "type": "suspicious_date",
      "priority": "medium",
      "images": ["sha256:..."],
      "description": "Image has future date",
      "details": {
        "date_found": "2026-01-01",
        "current_date": "2025-10-25"
      },
      "status": "pending"
    },
    {
      "id": "review_003",
      "type": "no_date",
      "priority": "low",
      "images": ["sha256:..."],
      "description": "No date information found",
      "status": "pending"
    }
  ],

  "transactions": {
    "current": null,
    "history": [
      {
        "id": "tx_20251025_140000",
        "started": "2025-10-25T14:00:00Z",
        "completed": "2025-10-25T14:15:00Z",
        "phase": "execution",
        "operations": [
          {
            "seq": 1,
            "type": "move",
            "source": "/external/IMG001.jpg",
            "target": "/catalog/2023-06/2023-06-15_120000.jpg",
            "checksum_before": "sha256:abc...",
            "checksum_after": "sha256:abc...",
            "status": "complete",
            "timestamp": "2025-10-25T14:00:05Z"
          },
          {
            "seq": 2,
            "type": "delete",
            "source": "/external/IMG001_small.jpg",
            "checksum": "sha256:def...",
            "status": "complete",
            "timestamp": "2025-10-25T14:00:06Z"
          }
        ],
        "rollback_available": true
      }
    ]
  }
}
```

## Workflow Phases

### Phase 1: ANALYZE

**Command**: `vam-catalog analyze --source /photos --catalog /organized`

**Process**:
1. Scan all source directories recursively
2. For each image/video:
   - Compute SHA256 checksum
   - Extract all EXIF metadata
   - Extract dates from all sources (EXIF, filename, directory, filesystem)
   - Detect file format, resolution, size
   - Generate perceptual hash (for duplicates)
   - Extract video metadata and generate thumbnail
3. Group duplicates by perceptual hash
4. Rank duplicates by quality score
5. Detect bursts (images within 10s)
6. Run hybrid AI analysis on bursts:
   - Quick sharpness/exposure heuristics
   - AI model for close matches
7. Generate target paths
8. Detect conflicts:
   - Date conflicts within duplicate groups
   - Suspicious dates (future, very old, default)
   - Name collisions
   - No date available
9. Build review queue
10. Save to catalog.json with checkpoints every 5 minutes

**Output**:
- catalog.json with full analysis
- Statistics dashboard
- Review queue count

### Phase 2: REVIEW

**Command**: `vam-catalog review` (launches web UI)

**Web UI** (FastAPI + Vue.js):

**Dashboard View**:
```
┌─────────────────────────────────────────────────────────┐
│  Lightroom Catalog Review                               │
├─────────────────────────────────────────────────────────┤
│  Statistics:                                            │
│    Total Images:        100,000                         │
│    Unique Images:        55,000                         │
│    Duplicate Groups:     15,000                         │
│    Burst Groups:          2,000                         │
│    Needs Review:          5,000  [Review Queue →]       │
│    Ready to Execute:     95,000                         │
│                                                          │
│  Status: ●●●●●●●●○○ 80% Analyzed                       │
└─────────────────────────────────────────────────────────┘
```

**Review Queue View**:
```
┌─────────────────────────────────────────────────────────┐
│  Review Queue (5,000 items)                             │
│                                                          │
│  [High Priority] [Medium] [Low] [All]                   │
│                                                          │
│  □ Date Conflict (1,200 items)                          │
│  □ Suspicious Date (500 items)                          │
│  □ No Date Found (500 items)                            │
│  □ Burst Review (200 items)                             │
│  □ Name Collision (50 items)                            │
│                                                          │
│  [Next Item] [Auto-Approve Safe Items]                  │
└─────────────────────────────────────────────────────────┘
```

**Conflict Resolution View** (Date Conflict Example):
```
┌─────────────────────────────────────────────────────────┐
│  Date Conflict - Duplicate Group                        │
├─────────────────────────────────────────────────────────┤
│  These duplicates have different dates:                 │
│                                                          │
│  ┌───────────────┐  ┌───────────────┐                  │
│  │   [Image 1]   │  │   [Image 2]   │                  │
│  │  (Primary)    │  │  (Duplicate)  │                  │
│  │               │  │               │                  │
│  │  3840x2160    │  │  1920x1080    │                  │
│  │  2.5 MB       │  │  1.2 MB       │                  │
│  └───────────────┘  └───────────────┘                  │
│                                                          │
│  Image 1 Date: 2023-06-15 12:00:00 (EXIF)              │
│  Image 2 Date: 2023-06-20 10:30:00 (EXIF)              │
│                                                          │
│  Which date is correct?                                 │
│  ○ Use 2023-06-15 (from primary/best image)            │
│  ○ Use 2023-06-20                                       │
│  ○ Enter manually: [________]                           │
│                                                          │
│  [Apply to All Similar] [Skip] [Next]                   │
└─────────────────────────────────────────────────────────┘
```

**Burst Review View**:
```
┌─────────────────────────────────────────────────────────┐
│  Burst Group - 7 images in 3.5 seconds                  │
├─────────────────────────────────────────────────────────┤
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
│  │ [1] │ │ [2] │ │ [3] │ │ [4] │ │ [5] │ │ [6] │     │
│  │     │ │  ⭐ │ │     │ │     │ │     │ │     │     │
│  │ 85% │ │ 95% │ │ 88% │ │ 82% │ │ 90% │ │ 87% │     │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘     │
│                                                          │
│  AI Selected: Image 2 (95% quality score)               │
│  Sharpness: ████████████░░ 95%                         │
│  Exposure:  ███████████░░░ 92%                         │
│                                                          │
│  [Override Selection] [Keep All] [Delete All] [Next]    │
└─────────────────────────────────────────────────────────┘
```

**API Endpoints**:
```python
GET  /api/stats                    # Dashboard statistics
GET  /api/review/queue             # Get review queue
GET  /api/review/next              # Get next review item
POST /api/review/resolve           # Submit resolution
GET  /api/image/{id}/preview       # Get image preview
GET  /api/image/{id}/metadata      # Get full metadata
POST /api/review/auto-approve      # Auto-approve safe items
```

### Phase 3: VERIFY

**Command**: `vam-catalog verify`

**Process**:
1. Check all images have decisions (none pending)
2. Verify no images will be lost:
   - Every unique image has target path
   - All duplicates mapped to delete or keep
3. Check for path collisions
4. Verify checksums in database
5. Simulate file operations (dry-run)
6. Generate execution plan report

**Output Report**:
```
Verification Report
═══════════════════════════════════════════════════════

✓ All conflicts resolved
✓ No images will be lost
✓ No path collisions
✓ Checksums verified

Execution Plan Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Operations:
  - Move:   55,000 unique images
  - Delete: 30,000 duplicate images
  - Create: 2,000 burst folders
  - Merge:  15,000 EXIF metadata updates

Disk Space:
  - Current: 500 GB
  - After:   275 GB (saved 225 GB)

Target Structure:
  /catalog/
  ├── 2003-06/ (150 images)
  ├── 2003-07/ (200 images)
  ├── ...
  └── 2025-10/ (1,500 images)

⚠ WARNING: This will permanently delete 30,000 duplicate files
⚠ Ensure you have backups before proceeding

Ready to execute? [y/N]:
```

### Phase 4: EXECUTE

**Command**: `vam-catalog execute --confirm`

**Process** (with maximum safety):

1. **Pre-execution backup**:
   - Create transaction log
   - Save current catalog state

2. **Dry-run copy phase**:
   - Copy all files to temporary staging area
   - Verify all copies successful
   - Verify checksums match

3. **EXIF merge phase**:
   - Merge metadata from duplicates into primary images
   - Update EXIF data in staged files
   - Verify EXIF writes successful

4. **Execution phase** (one operation at a time):
   - Log operation to transaction
   - Perform operation
   - Verify operation
   - Mark complete
   - Checkpoint every 100 operations

5. **Operations sequence**:
   - Create target directories
   - Move primary images from staging to target
   - Create burst folders
   - Move burst images
   - Verify all moves successful
   - Delete duplicate files from source
   - Clean up staging

6. **Verification**:
   - Verify all target files exist
   - Verify all checksums match
   - Verify no source files lost
   - Update catalog status to "complete"

**Progress Display**:
```
Executing Catalog Organization
═══════════════════════════════════════════════════════

Phase 1/5: Staging files...
[████████████████████████████████████] 100% (55,000/55,000)

Phase 2/5: Merging EXIF metadata...
[████████████████████░░░░░░░░░░░░░░░] 60% (9,000/15,000)

Estimated time remaining: 45 minutes
Operations completed: 45,230 / 100,000
```

**Rollback Support**:
If execution is interrupted:
```bash
vam-catalog rollback --transaction tx_20251025_140000

# Or resume
vam-catalog execute --resume
```

### Phase 5: IMPORT (Ongoing)

**Manual Mode**:
```bash
vam-catalog import /import --catalog /organized
```

**Scheduled Mode** (cron/systemd):
```bash
# Every hour
0 * * * * vam-catalog import /import --catalog /organized --auto
```

**Watch Mode** (filesystem monitoring):
```bash
vam-catalog watch /import --catalog /organized
```

**Process**:
1. Scan import directory
2. Run analysis on new files
3. Auto-approve if no conflicts
4. Add to review queue if conflicts
5. Execute immediately (if auto) or queue for next execution

## Technology Stack

### Core
- **Python 3.8+**
- **Pillow**: Image processing
- **pyexiftool**: EXIF extraction/modification
- **imagehash**: Perceptual hashing
- **opencv-python**: Video processing, sharpness detection

### AI Model
- **torch** + **torchvision**: PyTorch for GPU
- **timm**: Image quality models
- **clip**: CLIP for hybrid approach
- Model size: ~2-3 GB
- GPU: NVIDIA CUDA support

### Web UI
- **FastAPI**: Backend API (async, fast, auto-docs)
- **Vue.js 3**: Frontend (simple, reactive)
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- Runs on `localhost:8000`

### Storage
- **JSON**: Catalog database (human-readable, git-friendly)
- **SQLite**: Optional for query performance on large catalogs

### Monitoring
- **watchdog**: Filesystem monitoring
- **schedule**: Scheduled tasks
- **rich**: Progress bars and console output

## File Organization Structure

```
/organized_catalog/
├── .catalog.json                    # Main database
├── .catalog.backup.json             # Last known good state
├── .transactions/                   # Transaction logs
│   ├── tx_20251025_140000.json
│   └── tx_20251025_150000.json
├── .staging/                        # Temporary during execution
│   └── (empty after completion)
├── no-date/                         # Images with no date
│   ├── IMG_unknown_a1b2c3.jpg
│   └── VID_unknown_d4e5f6.mp4
├── 2003-06/
│   ├── 2003-06-15_120000_abc123.jpg
│   ├── 2003-06-15_120000_abc123_1920x1080.jpg
│   ├── 2003-06-15_143022_def456.raw
│   └── 2003-06-15_143022_def456_burst/
│       ├── 2003-06-15_143022_001.raw
│       ├── 2003-06-15_143022_002.raw
│       └── 2003-06-15_143022_003.raw
├── 2023-12/
│   └── ...
└── import_history/                  # Optional: track imports
    └── import_20251025.json
```

## Implementation Plan (Updated - Organization First)

> **See [PRIORITIES.md](PRIORITIES.md) for detailed rationale and success criteria.**

### ✅ Iteration 1: Foundation (COMPLETE)
- [x] Catalog database manager with locking
- [x] Checksum computation (SHA256)
- [x] EXIF extraction using ExifTool
- [x] Transaction log foundation
- [x] Basic CLI structure (vam-analyze, vam-web)
- [x] Date extraction with confidence scoring
- [x] Image scanning with checkpoints
- [x] Basic web viewer

### ⭐ Iteration 2: Duplicate Detection (CURRENT PRIORITY)
- [ ] Perceptual hashing implementation (dHash, aHash, pHash)
- [ ] Duplicate grouping by hash similarity
- [ ] Quality scoring algorithm
  - Format priority (RAW > JPEG > compressed)
  - Resolution scoring
  - File size scoring
  - EXIF completeness
- [ ] Primary selection from duplicate groups
- [ ] EXIF metadata merging from duplicates to primary
- [ ] Web UI for duplicate visualization
- [ ] Tests for duplicate detection logic

### Iteration 3: Organization & Execution
- [ ] Organization plan generation
  - Target path calculation (YYYY-MM structure)
  - Action planning (move/delete/merge)
  - Conflict detection
- [ ] Enhanced review UI
  - Duplicate comparison view
  - Date conflict resolution
  - Manual override capability
  - Plan approval workflow
- [ ] Execution engine
  - Dry-run verification
  - Staged copy phase
  - Atomic operations with rollback
  - Progress tracking
  - Post-execution verification
- [ ] Integration tests for full workflow

### Iteration 4: Import & Maintenance
- [ ] Import directory processing
- [ ] Incremental catalog updates
- [ ] Filesystem watcher (optional)
- [ ] Scheduled import runs

### Future: Advanced Features (Phase 3)
- [ ] Burst detection (time-based grouping)
- [ ] AI quality scoring (sharpness, exposure)
- [ ] Auto-tagging and classification
- [ ] Smart collections

## Current Status

**Phase 1: Core Organization** (~40% complete)
- ✅ Infrastructure ready
- ⭐ Next: Duplicate detection
- ⏳ Then: Organization execution
- ⏳ Then: Import system
