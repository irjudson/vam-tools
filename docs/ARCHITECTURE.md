# VAM Tools - System Architecture

## Overview

VAM Tools is a comprehensive photo catalog management system designed to organize 100,000+ images with:
- Zero data loss guarantee
- Chronological organization
- Intelligent duplicate handling
- Conflict resolution with human oversight
- Plan-verify-execute workflow

## Core Principles

1. **Never lose images** - All operations reversible until final execution
2. **Organize chronologically** - YYYY-MM directory structure
3. **One correct copy** - Identify and eliminate duplicates, keeping the best version
4. **Human in the loop** - Flag conflicts for review, auto-handle clear cases
5. **Foundation first** - Core organization before advanced curation features

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      VAM Tools                               │
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
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Analysis Engine
**Module**: `vam_tools/analysis/`

Responsible for scanning source directories and extracting comprehensive metadata.

Components:
- **Scanner** (`scanner.py`): Multi-core parallel file processing
- **Metadata Extractor** (`metadata.py`): EXIF data extraction via ExifTool
- **Duplicate Detector** (`duplicate_detector.py`): Perceptual hashing (dHash + aHash)
- **Perceptual Hash** (`perceptual_hash.py`): Hash algorithm implementations
- **Quality Scorer** (`quality_scorer.py`): Image quality assessment for duplicate selection

### Catalog Database
**Module**: `vam_tools/core/`

Central data store with ACID-like properties.

Components:
- **Database Manager** (`catalog.py`): File locking, checkpointing, transactions
- **Type Definitions** (`types.py`): Pydantic models for type safety

Features:
- JSON-based storage (human-readable, version controllable)
- File locking (fcntl) for concurrent access safety
- Automatic checkpointing every 100 files
- Signal-based lock timeout (30s default)
- Path indexing for fast lookups

### Web Interface
**Module**: `vam_tools/web/`

FastAPI-based REST API with Vue.js frontend.

Components:
- **API Server** (`api.py`): REST endpoints for catalog browsing and duplicate review
- **Static Files**: Vue 3 SPAs for catalog viewing and duplicate comparison

Endpoints:
- `/api/catalog/info` - Catalog metadata and statistics
- `/api/images` - List images with pagination, filtering, sorting
- `/api/images/{id}` - Get image details
- `/api/images/{id}/file` - Serve image files
- `/api/duplicates/stats` - Duplicate statistics
- `/api/duplicates/groups` - List duplicate groups
- `/api/duplicates/groups/{id}` - Get duplicate group details
- `/api/statistics/summary` - Catalog statistics breakdown

### CLI
**Module**: `vam_tools/cli/`

Command-line interfaces built with Click.

Commands:
- `vam-analyze` - Scan directories and build catalog
- `vam-web` - Launch web UI server

## Data Model

### Catalog Structure

```json
{
  "version": "2.0.0",
  "catalog_id": "uuid-1234-5678",
  "created": "2025-10-26T00:00:00Z",
  "last_updated": "2025-10-26T08:00:00Z",

  "configuration": {
    "source_directories": ["/path/to/photos"],
    "similarity_threshold": 5,
    "checkpoint_interval_seconds": 300
  },

  "state": {
    "phase": "analyzing|complete",
    "last_checkpoint": "2025-10-26T07:55:00Z",
    "images_processed": 45230,
    "images_total": 50000,
    "progress_percentage": 90.46
  },

  "statistics": {
    "total_images": 45000,
    "total_videos": 5000,
    "total_size_bytes": 500000000000,
    "no_date": 500,
    "suspicious_dates": 50
  },

  "images": {
    "sha256:abc123...": {
      "id": "sha256:abc123...",
      "source_path": "/photos/IMG_1234.jpg",
      "file_type": "image",
      "checksum": "sha256:abc123...",

      "dates": {
        "exif_date": "2023-06-15T14:30:00Z",
        "filename_date": null,
        "directory_date": "2023-06",
        "filesystem_date": "2023-06-20T10:00:00Z",
        "selected_date": "2023-06-15T14:30:00Z",
        "source": "exif",
        "confidence": 95
      },

      "metadata": {
        "format": "JPEG",
        "resolution": [4032, 3024],
        "width": 4032,
        "height": 3024,
        "size_bytes": 3145728,
        "exif": {
          "Make": "Apple",
          "Model": "iPhone 12 Pro",
          "DateTimeOriginal": "2023:06:15 14:30:00"
        }
      },

      "hashes": {
        "dhash": "abcd1234ef567890",
        "ahash": "1234567890abcdef"
      },

      "quality_score": 85,
      "status": "complete"
    }
  },

  "duplicate_groups": [
    {
      "id": "dup-group-001",
      "primary_image_id": "sha256:abc123...",
      "duplicate_image_ids": ["sha256:def456...", "sha256:ghi789..."],
      "similarity_type": "perceptual",
      "confidence": 95
    }
  ],

  "review_queue": [
    {
      "id": "review-001",
      "type": "date_conflict|duplicate|no_date",
      "priority": "high|medium|low",
      "image_ids": ["sha256:abc123..."],
      "details": {},
      "status": "pending|resolved"
    }
  ]
}
```

### Image Record

**Type**: `ImageRecord` (Pydantic model)

Fields:
- `id`: Checksum (SHA-256) - unique identifier
- `source_path`: Original file location
- `file_type`: `image` or `video`
- `checksum`: SHA-256 hash for duplicate detection
- `dates`: `DateInfo` - extracted dates with confidence
- `metadata`: `ImageMetadata` - comprehensive metadata
- `hashes`: Optional perceptual hashes (dHash, aHash)
- `quality_score`: 0-100 quality assessment
- `status`: Processing status

### Date Information

**Type**: `DateInfo` (Pydantic model)

Date extraction sources (priority order):
1. **EXIF metadata** (95% confidence) - Camera timestamps
2. **Filename patterns** (70% confidence) - YYYY-MM-DD, YYYYMMDD, etc.
3. **Directory structure** (50% confidence) - Year/month from folder names
4. **Filesystem metadata** (30% confidence) - File creation/modification time

System selects the **earliest date** from all available sources.

### Duplicate Groups

**Type**: `DuplicateGroup` (Pydantic model)

Fields:
- `id`: Unique group identifier
- `primary_image_id`: Best quality image (selected automatically)
- `duplicate_image_ids`: List of duplicate images
- `similarity_type`: `exact` (checksum) or `perceptual` (hash)
- `confidence`: Similarity confidence (0-100)

## Processing Pipeline

### 1. Analysis Phase

```
┌─────────────┐
│   Scanner   │
└──────┬──────┘
       │
       ├──→ Collect files (recursive directory scan)
       │    └─→ Filter by extension (images/videos only)
       │
       ├──→ Parallel processing (multiprocessing pool)
       │    ├─→ Compute checksum (SHA-256)
       │    ├─→ Extract EXIF metadata (ExifTool)
       │    ├─→ Extract dates (multiple sources)
       │    ├─→ Generate perceptual hashes (if enabled)
       │    └─→ Score quality
       │
       ├──→ Add to catalog
       │    ├─→ Check for duplicates (by checksum)
       │    ├─→ Update statistics
       │    └─→ Create checkpoint (every 100 files)
       │
       └──→ Final checkpoint
            └─→ Save catalog
```

### 2. Duplicate Detection Phase

```
┌────────────────────┐
│ Duplicate Detector │
└─────────┬──────────┘
          │
          ├──→ Build hash index
          │    ├─→ Group by dHash
          │    └─→ Group by aHash
          │
          ├──→ Find similar images
          │    ├─→ Hamming distance comparison
          │    └─→ Filter by threshold (default: 5)
          │
          ├──→ Score and rank
          │    ├─→ Quality score (format, resolution, size)
          │    ├─→ Metadata completeness
          │    └─→ Select primary image
          │
          └──→ Create duplicate groups
               └─→ Save to catalog
```

### 3. Review Phase

```
┌─────────────┐
│   Web UI    │
└──────┬──────┘
       │
       ├──→ Browse catalog
       │    ├─→ View all images
       │    ├─→ Filter/sort
       │    └─→ View metadata
       │
       ├──→ Review duplicates
       │    ├─→ View duplicate groups
       │    ├─→ Side-by-side comparison
       │    ├─→ View recommendations
       │    └─→ Mark for action
       │
       └──→ Resolve conflicts
            ├─→ Date conflicts
            ├─→ No date images
            └─→ Manual overrides
```

## Performance Characteristics

### Multi-Core Scaling
- Linear scaling up to CPU core count
- 20-30x speedup on 32-core systems
- Chunk size: `len(files) // (workers * 4)`
- Default workers: `multiprocessing.cpu_count()`

### Memory Usage
- Catalog loaded entirely in memory
- ~500 bytes per image record
- 100k images ≈ 50 MB RAM
- Perceptual hashes add ~32 bytes per image

### Disk I/O
- Sequential reads for file scanning
- Random reads for checksum computation
- Checkpoint writes every 100 files (atomic JSON write)
- Single final catalog write

### Checkpointing
- Auto-checkpoint: Every 100 files processed
- Manual checkpoint: `catalog.checkpoint(force=True)`
- Lock timeout: 30 seconds default
- Recovery: Load from last checkpoint

## Technology Stack

### Core Dependencies
- **Python 3.9+**: Core language
- **Pydantic v2**: Type-safe data models
- **ExifTool**: Comprehensive metadata extraction
- **Pillow**: Image processing
- **pillow-heif**: HEIC/HEIF support

### CLI & Output
- **Click**: Command-line interface framework
- **Rich**: Beautiful terminal formatting

### Web Stack
- **FastAPI**: Modern async web framework
- **Uvicorn**: ASGI server
- **Vue.js 3**: Frontend framework (CDN-based)
- **Axios**: HTTP client

### Development
- **pytest**: Testing framework (213 tests, 84% coverage)
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

## Security Considerations

### File System Safety
- File locking (fcntl.flock) prevents concurrent writes
- Signal-based timeout prevents deadlocks
- Checksum verification after operations
- No modification of source files during analysis

### Input Validation
- Pydantic models validate all data
- Path traversal prevention
- Extension whitelist for file types
- Size limits for uploads (if implemented)

### Web API
- CORS disabled by default (localhost only)
- No authentication (designed for local use)
- Read-only operations (no file deletion via API)
- Static file serving from known directories only

## Future Architectural Considerations

### Scalability
- Consider SQLite for >500k images
- Implement lazy loading for catalog sections
- Add image thumbnail cache
- Stream large responses

### Distributed Processing
- Support remote workers
- Distributed hash table for deduplication
- Cloud storage integration

### Extensibility
- Plugin system for custom analyzers
- External AI model integration
- Custom organization rules
- Export to other catalog formats
