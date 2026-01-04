# Lumina Roadmap

This document tracks the development progress of Lumina features. GitHub renders task lists as progress bars automatically.

> **Legend**: ✅ Complete | 🚧 In Progress | 📋 Planned | ⏸️ Deferred

---

## Overview

<!-- GitHub will render this as a progress bar -->
### Overall Progress: 16/22 features complete

- [x] Core scanning and analysis
- [x] Duplicate detection
- [x] Web UI with catalog browser
- [x] Docker deployment
- [x] Background job processing
- [x] Multi-catalog support
- [x] Preview caching
- [x] Advanced search/filtering
- [x] Export functionality
- [x] Performance monitoring
- [x] Map/timeline views
- [x] GPU acceleration (with fallback)
- [x] Adaptive batch sizing
- [x] Real-time progress tracking
- [x] Auto-tagging with AI
- [x] FAISS index persistence
- [ ] Duplicate resolution UI
- [ ] Batch edit operations
- [ ] Advanced analytics dashboard
- [ ] Job scheduling
- [ ] Notifications
- [ ] Video tutorials

---

## ✅ Completed Features

### Core Infrastructure
- [x] **High-performance scanning** - Multi-core parallel processing
- [x] **Metadata extraction** - EXIF, XMP, filenames, directory dates
- [x] **RAW file support** - Native metadata extraction (CR2, NEF, ARW, etc.)
- [x] **Duplicate detection** - Exact (checksum) and perceptual (hash) matching
- [x] **Quality scoring** - Automatic best-copy selection
- [x] **Corruption tracking** - Detect truncated/corrupted files

### Web Interface
- [x] **Modern Vue.js UI** - Dark theme, responsive design
- [x] **Catalog browser** - Grid view with thumbnails
- [x] **Advanced filtering** - By date, camera, lens, GPS, resolution
- [x] **Map view** - Geohash clustering, timeline slider
- [x] **Job management** - Submit, monitor, cancel background jobs

### Backend Services
- [x] **Docker deployment** - CUDA-enabled containers
- [x] **Celery job system** - Background processing with Redis
- [x] **PostgreSQL backend** - Scalable database with JSONB
- [x] **WebSocket progress** - Real-time job updates
- [x] **REST API** - Full CRUD for catalogs, images, jobs

### Performance
- [x] **Preview caching** - LRU cache for RAW previews ([`preview_cache.py`](../vam_tools/shared/preview_cache.py))
- [x] **GPU acceleration** - PyTorch CUDA with graceful CPU fallback
- [x] **Adaptive batching** - Auto-size batches based on timing history
- [x] **Performance snapshots** - Real-time CPU/GPU/throughput tracking

### Data Management
- [x] **Multi-catalog support** - Multiple catalogs with color coding
- [x] **Export functionality** - CSV/JSON export for images and duplicates
- [x] **Job persistence** - Job history stored in database
- [x] **Geohash indexing** - Fast spatial queries

### AI & Analysis
- [x] **Auto-tagging with AI** - OpenCLIP and Ollama-based image classification
- [x] **Tag taxonomy** - Hierarchical tags with categories (subject, scene, lighting, mood)
- [x] **Batch tagging** - Background Celery task with checkpointing for resumability
- [x] **Combined backend** - Weighted OpenCLIP + Ollama for high accuracy
- [x] **CLIP embeddings** - Stored for semantic search capability
- [x] **Tag API** - Full REST API for tag management and filtering

**Files**: [`image_tagger.py`](../vam_tools/analysis/image_tagger.py), [`tag_taxonomy.py`](../vam_tools/analysis/tag_taxonomy.py), [`tag_manager.py`](../vam_tools/analysis/tag_manager.py)

### Search & Indexing
- [x] **FAISS index persistence** - Save/load indices for instant startup
- [x] **Incremental updates** - Add images without full rebuild
- [x] **Index versioning** - Track version and hash method for compatibility
- [x] **Auto-validation** - Detect when rebuild is needed

**Files**: [`fast_search.py`](../vam_tools/analysis/fast_search.py), [`duplicate_detector.py`](../vam_tools/analysis/duplicate_detector.py)

---

## 📋 Planned Features

### High Priority

#### Duplicate Resolution UI
[Issue #12](https://github.com/irjudson/lumina/issues/12)

Interactive UI for reviewing and resolving duplicates with confidence.

- [ ] Side-by-side comparison with zoom
- [ ] Batch keep/delete operations
- [ ] Undo/redo with operation history
- [ ] Safe deletion (trash before permanent delete)
- [ ] Quality score visualization
- [ ] Metadata diff highlighting
- [ ] Space savings preview

**Why**: Currently duplicates are detected but resolution requires manual work.

---

### Medium Priority

#### Batch Edit Operations
[Issue #15](https://github.com/irjudson/lumina/issues/15)

Efficiently perform operations on multiple files.

- [ ] Bulk metadata editing (dates, tags)
- [ ] Batch move/copy/delete
- [ ] Transaction support with rollback
- [ ] Dry-run mode

**Why**: No way to edit multiple files at once in the UI.

---

#### Advanced Analytics Dashboard
[Issue #16](https://github.com/irjudson/lumina/issues/16)

Visualize and understand your photo collection.

- [ ] Photo timeline chart (interactive)
- [ ] Storage analysis by date/camera/format
- [ ] Quality distribution visualization
- [ ] Camera/lens usage statistics
- [ ] Duplicate savings projections

**Why**: Stats endpoint exists but no visualization.

---

### Low Priority / Deferred

#### Job Scheduling ⏸️

Cron-like scheduling for periodic analysis.

- [ ] Celery Beat integration
- [ ] Scheduled scan configuration
- [ ] Run history tracking

**Status**: Deferred - users typically run scans manually when adding photos.

---

#### Notifications ⏸️

Alerts when long jobs complete.

- [ ] Email notifications
- [ ] Webhook integration
- [ ] Browser push notifications

**Status**: Deferred - WebSocket progress streaming covers main use case.

---

## 🔧 Technical Debt

### Code Quality
- [ ] Resolve ignored flake8 warnings
- [ ] Add comprehensive docstrings
- [ ] Reduce complexity in flagged functions (C901)
- [ ] Clean up test warnings

### Documentation
- [ ] Video tutorials
- [ ] Migration guides from other tools
- [ ] Architecture decision records (ADRs)
- [ ] Performance tuning guide

---

## 📊 Feature Summary

| Category | Complete | In Progress | Planned | Deferred |
|----------|----------|-------------|---------|----------|
| Core | 6 | 0 | 0 | 0 |
| Web UI | 5 | 0 | 2 | 0 |
| Backend | 5 | 0 | 0 | 0 |
| Performance | 4 | 0 | 0 | 0 |
| Data | 4 | 0 | 2 | 0 |
| AI & Analysis | 6 | 0 | 0 | 0 |
| Search & Indexing | 4 | 0 | 0 | 0 |
| Analytics | 0 | 0 | 1 | 0 |
| Ops | 0 | 0 | 0 | 2 |
| **Total** | **34** | **0** | **5** | **2** |

---

## Contributing

Interested in implementing a feature?

1. Check if there's an issue for it (links above)
2. Comment on the issue to express interest
3. Discuss approach with maintainers
4. Submit a PR with implementation and tests

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## Request a Feature

Have an idea? Open an issue!

- [Submit Feature Request](https://github.com/irjudson/lumina/issues/new?template=feature_request.md)
- [Report Bug](https://github.com/irjudson/lumina/issues/new?template=bug_report.md)
- [Discuss Ideas](https://github.com/irjudson/lumina/discussions)
