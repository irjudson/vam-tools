# VAM Tools Roadmap

This document outlines planned features and improvements for VAM Tools, organized by priority.

## High Priority

### Preview Caching System

**Goal**: Improve web UI performance for RAW files by caching extracted previews

**Features**:
- Cache extracted RAW file previews to avoid repeated extraction
- Implement LRU cache with configurable size limit (e.g., 10 GB)
- Background preview extraction during analysis phase
- Automatic cache cleanup for deleted/moved files

**Benefits**:
- Faster browsing in web UI (no 30s timeouts)
- Reduced CPU/disk load on repeated views
- Better experience with network drives


---

### Auto-Tagging System

**Goal**: Automatically tag images with AI-detected subjects and scenes

**Features**:
- Integration with ML models for automatic image tagging
- Subject detection (people, animals, objects)
- Scene classification (indoor, outdoor, landscape, portrait, etc.)
- Configurable tagging pipeline (enable/disable specific models)
- Tag confidence scores
- Manual tag editing and correction

**Models Considered**:
- CLIP for general image understanding
- YOLOv8 for object detection
- Face detection for people identification
- Custom fine-tuned models

**Benefits**:
- Searchable by content ("find photos of dogs")
- Better organization and discovery
- Automated album creation by tag


---

### Duplicate Resolution UI

**Goal**: Interactive UI for reviewing and resolving duplicates with confidence

**Features**:
- Side-by-side duplicate comparison with zoom
- Batch operations (keep/delete multiple at once)
- Undo/redo functionality with operation history
- Safe deletion with trash/backup before permanent removal
- Quality scoring visualization (why one copy is better)
- Metadata diff view (highlight differences)
- Preview before/after disk space savings

**Benefits**:
- Confident duplicate cleanup
- Recover from mistakes
- Clear space efficiently
- Understand quality scoring decisions


---

## Medium Priority

### Code Quality Improvements

**Goal**: Clean up technical debt and improve maintainability

**Tasks**:
- Review and resolve ignored flake8 warnings (docstrings, complexity)
- Add comprehensive docstrings to all modules and functions
- Reduce code complexity in flagged functions (C901 warnings)
- Clean up test warnings (deprecations, multiprocessing)
- Improve type coverage (currently good, but room for improvement)

**Benefits**:
- Easier onboarding for contributors
- Better IDE support and documentation
- Reduced bugs from complexity


---

### FAISS Index Persistence

**Goal**: Save and load FAISS indices to disk for instant startup

**Features**:
- Save/load FAISS indices to disk (`.faiss` file)
- Incremental index updates for new images (no full rebuild)
- Index versioning and automatic migration
- Fallback to rebuild if index corrupted

**Benefits**:
- Instant duplicate search (no rebuild wait)
- Faster catalog opening
- Lower memory usage (memory-mapped indices)


---

### Advanced Search

**Goal**: Powerful search capabilities across catalog

**Features**:
- Search by date range (start/end dates)
- Search by metadata (camera model, lens, focal length)
- Search by location (GPS coordinates, location names)
- Search by quality score (find highest quality images)
- Similar image search (reverse image search with uploaded reference)
- Combined filters (AND/OR logic)
- Saved search queries

**Benefits**:
- Find specific photos quickly
- Discover forgotten photos
- Quality-based selection for prints/sharing


---

### Batch Operations

**Goal**: Efficiently perform operations on multiple files

**Features**:
- Bulk metadata editing (set dates, add tags, etc.)
- Batch file operations (move, copy, delete)
- Transaction support with rollback (undo batch changes)
- Progress tracking for long operations
- Dry-run mode for safety

**Benefits**:
- Efficient catalog management
- Correct multiple files at once
- Safe operations with rollback


---

## Low Priority / Future Ideas


### Advanced Analytics

**Goal**: Visualize and understand photo collection

**Features**:
- Photo timeline visualization (interactive chart)
- Storage analysis by date/camera/format
- Quality distribution charts
- Duplicate savings projections
- Camera usage statistics
- Lens usage statistics
- Most photographed subjects/locations

**Benefits**:
- Understand collection better
- Identify storage optimization opportunities
- Track photography habits

**Estimated Effort**: 1 week

---

### Export Functionality

**Goal**: Export catalog data for external use

**Features**:
- Export catalog to CSV/JSON
- Export duplicate reports (which files, savings, quality scores)
- Export statistics and analytics
- Customizable export format
- Integration with other tools (Lightroom, etc.)

**Benefits**:
- Analyze data in Excel/other tools
- Share findings with others
- Backup catalog in portable format

---

### Performance Optimizations

**Goal**: Handle even larger catalogs more efficiently

**Features**:
- Distributed processing (multiple machines in cluster)
- Incremental FAISS index updates (don't rebuild everything)
- Smart caching strategies (predictive prefetch)
- Background workers for heavy operations (don't block UI)
- Streaming large results (don't load all in memory)

**Benefits**:
- Handle millions of photos
- Faster operations
- More responsive UI

**Estimated Effort**: Ongoing (continuous optimization)

---



## Documentation Improvements

### Planned Enhancements

- [ ] Video tutorials (YouTube series)
- [ ] Migration guides from other tools (Lightroom, Apple Photos, Google Photos)
- [ ] More examples and use cases
- [ ] Troubleshooting flowcharts
- [ ] Architecture decision records (ADRs)
- [ ] Performance tuning guide
- [ ] Contribution guide for specific areas (core, web UI, CLI)

**Estimated Effort**: Ongoing (as features are added)

---

## Community Requests

**Have an idea?** Open an issue on GitHub!

- [Submit Feature Request](https://github.com/irjudson/vam-tools/issues/new?template=feature_request.md)
- [Report Bug](https://github.com/irjudson/vam-tools/issues/new?template=bug_report.md)
- [Discuss Ideas](https://github.com/irjudson/vam-tools/discussions)

---

## Completed Features

For a history of completed features, see the [CHANGELOG.md](../CHANGELOG.md) (to be created).

---

## Priority Definitions

- **High Priority**: Essential features that significantly improve user experience or fill critical gaps
- **Medium Priority**: Important features that enhance functionality but aren't blockers
- **Low Priority**: Nice-to-have features that add convenience or advanced capabilities

Priorities may shift based on:
- Community feedback and requests
- Bug reports and pain points
- Technical dependencies (one feature may enable another)
- Development resources and expertise

---

## Contributing to Roadmap Items

Interested in implementing a feature from this roadmap?

1. Check if there's already an issue for it
2. Comment on the issue to express interest
3. Discuss approach with maintainers
4. Submit a PR with implementation and tests

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.
