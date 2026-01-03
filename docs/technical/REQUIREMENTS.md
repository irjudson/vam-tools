# Lumina - Requirements & Roadmap

## Mission

**Get files organized with a single, correct copy of each asset first. Then enable curation and enrichment.**

## Core Philosophy

Visual Asset Management is about establishing a **single source of truth** for your visual assets. Before we can do sophisticated curation, AI-based selection, or advanced management features, we need:

1. **One correct copy** of each unique asset
2. **Correct metadata** (especially dates)
3. **Consistent organization** (chronological structure)
4. **Zero data loss** (every unique asset preserved)

Once we have this foundation, we can build value-added features on top.

## Phase 1: Core Organization (CURRENT FOCUS)

### Goal
Transform a chaotic collection of photos/videos into a clean, chronologically organized catalog with no duplicates.

### Critical Features

#### 1.1 Duplicate Detection & Resolution ✅ COMPLETE
- Perceptual hashing to find duplicates (dHash + aHash)
- Quality scoring to select the best copy
- Handle different formats of the same image (RAW + JPEG, different sizes)
- Web UI for reviewing duplicate groups
- Side-by-side comparison interface
- **Status**: Implemented and tested

#### 1.2 Date Extraction & Correction ✅ COMPLETE
- Extract from EXIF, filenames, directories, filesystem
- Confidence scoring
- Flag suspicious dates for review
- Human-in-the-loop for conflicts
- **Status**: Implemented with multiple extraction strategies

#### 1.3 Catalog Database ✅ COMPLETE
- JSON-based catalog for human readability
- Checkpoint system for long-running operations
- File locking for safety
- Transaction support
- **Status**: Fully implemented with 73% test coverage

#### 1.4 Organization Execution 🚧 FUTURE
- YYYY-MM directory structure
- Move/copy files to organized locations
- Verify checksums after operations
- Dry-run mode for safety
- Rollback support if interrupted
- **Status**: Planned (from V1, needs V2 integration)

#### 1.5 Review Interface 🚧 IN PROGRESS
- Web UI to review flagged issues
- Conflict resolution (duplicate dates, missing dates)
- Batch approval for safe operations
- Progress monitoring
- **Status**: Basic web viewer + duplicate review complete, needs date conflict resolution

### Success Criteria
- ✅ All unique assets identified (one copy selected)
- ✅ All assets have dates (or flagged for manual review)
- ⏳ Files organized into YYYY-MM structure
- ✅ Zero data loss (all unique content preserved)

## Phase 2: Advanced Features (FUTURE)

### Burst Detection
- Group sequential images (burst mode, timelapse)
- Auto-select best shots based on focus, exposure, composition
- Flag representative images
- **Priority**: Medium
- **Status**: Designed, not implemented

### AI-Powered Curation
- Scene detection
- Face recognition
- Object detection
- Quality assessment (blur, exposure, composition)
- **Priority**: Low
- **Status**: Future consideration

### Auto-Tagging System
Automatically tag images with detected attributes to enable rich organization, search, and filtering capabilities. All tags include confidence scores, and low-confidence tags are queued for manual review.

#### Tag Categories

**Location Tags**:
- GPS coordinates → location names (city, region, country)
- Reverse geocoding using OpenStreetMap/Google Maps APIs
- Hierarchical tags: `location:country:usa`, `location:city:san-francisco`

**Time Tags**:
- Season (spring, summer, fall, winter)
- Time of day (morning, afternoon, evening, night)
- Special dates (holidays, events)

**Content Tags**:
- Scene type (landscape, portrait, street, architecture, nature)
- Objects detected (car, building, animal, food)
- Activities (hiking, swimming, dining, traveling)

**Technical Tags**:
- Camera settings (ISO, aperture, shutter speed ranges)
- Lens type (wide-angle, telephoto, macro)
- Flash usage (on, off, auto)
- File format categories (RAW, JPEG, HEIC)

**Quality Tags**:
- Image quality (sharp, blurry, overexposed, underexposed)
- Composition (rule-of-thirds, centered, symmetry)
- Aesthetic scores (highly-rated, keep, review, delete-candidate)

#### Implementation Approach

**Confidence Levels**:
- **High (90-100%)**: Auto-apply tag, no review needed
- **Medium (70-89%)**: Auto-apply tag, show in review queue
- **Low (<70%)**: Don't auto-apply, require manual review

**Tag Sources**:
1. **EXIF/Metadata** (100% confidence): GPS, camera model, lens, settings
2. **Pattern Recognition** (80-95%): Dates, seasons, time of day
3. **AI Vision** (60-90%): Scene detection, object recognition
4. **User Corrections** (100%): Manual tags, approved AI suggestions

**Review Interface**:
- Batch review of low-confidence tags
- Accept/reject/modify suggestions
- Learn from user corrections (improve AI thresholds)

**Priority**: Low (after core organization is solid)

## Design Principles

### Non-Destructive by Default
- All operations provide dry-run modes
- Destructive operations require explicit flags (off by default)
- Automatic backups before catalog modifications

### Minimal Functioning Code
- DRY (Don't Repeat Yourself)
- Clean architecture with clear separation of concerns
- Type-safe with Pydantic models throughout

### Quality Gates from Day One
- Tests and CI implemented at project start
- Pre-push hooks prevent broken code
- GitHub Actions run full test suite on every commit

## Technical Requirements

### Performance
- Multi-core parallel processing for large libraries (100k+ images)
- Efficient perceptual hashing algorithms
- Incremental scanning (only process new/changed files)
- Checkpoint system for resumable long-running operations

### Reliability
- File locking prevents concurrent modification
- Checksum verification after all file operations
- Transaction log for rollback capability
- Comprehensive error handling and recovery

### Usability
- Rich CLI with progress bars and formatted output
- Modern web UI for visual catalog browsing
- Clear error messages and troubleshooting guidance
- Comprehensive documentation

### Compatibility
- Python 3.9+ (supports 3.9, 3.10, 3.11, 3.12)
- Cross-platform (Linux, macOS, Windows via WSL)
- Support for all common image/video formats
- ExifTool integration for comprehensive metadata extraction
