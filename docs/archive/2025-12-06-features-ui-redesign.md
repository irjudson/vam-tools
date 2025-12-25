# VAM Tools: Features & UI Redesign Plan

## Executive Summary

This plan covers two major initiatives:
1. **New Features**: Face Recognition, Free-text/Semantic Search, and Burst Detection
2. **UI Redesign**: Lightroom/Excire Foto-inspired layout with better organization

---

## External Services vs Local Alternatives

### Current Architecture (100% Local)

VAM Tools currently runs entirely locally with one optional external service:
- **Ollama** (optional) - Local LLM for tag refinement, runs in Docker

### Proposed Architecture: Fully Local by Default

All new features are designed to run **completely offline** with no external API dependencies:

| Feature | Model/Service | Local? | Notes |
|---------|---------------|--------|-------|
| **Auto-Tagging** | OpenCLIP + Ollama | ✅ 100% Local | Already implemented |
| **Face Detection** | RetinaFace/MTCNN | ✅ 100% Local | Open-source models |
| **Face Recognition** | ArcFace (insightface) | ✅ 100% Local | Open-source, runs on GPU |
| **Semantic Search** | OpenCLIP | ✅ 100% Local | Same model as tagging |
| **Burst Detection** | EXIF analysis | ✅ 100% Local | No ML needed |
| **Quality Scoring** | BRISQUE/NIQE | ✅ 100% Local | Traditional CV algorithms |

### Model Details & Alternatives

#### 1. Face Detection

| Option | License | Size | GPU Required | Accuracy | Recommendation |
|--------|---------|------|--------------|----------|----------------|
| **RetinaFace** | MIT | ~100MB | Optional (faster) | Best | ✅ Primary |
| MTCNN | MIT | ~5MB | Optional | Good | Fallback for CPU |
| MediaPipe Face | Apache 2.0 | ~10MB | No | Good | Lightweight option |
| dlib HOG | Boost | ~100MB | No | Moderate | Legacy option |

**Recommendation**: RetinaFace with MTCNN fallback for CPU-only systems.

#### 2. Face Recognition (Embeddings)

| Option | License | Embedding Dim | GPU Required | Accuracy (LFW) | Recommendation |
|--------|---------|---------------|--------------|----------------|----------------|
| **ArcFace (insightface)** | MIT | 512 | Recommended | 99.83% | ✅ Primary |
| FaceNet (facenet-pytorch) | MIT | 512 | Recommended | 99.65% | Alternative |
| dlib face_recognition | MIT | 128 | No | 99.38% | CPU fallback |
| OpenFace | Apache 2.0 | 128 | No | 92.92% | Not recommended |

**Recommendation**: ArcFace via insightface (best accuracy, MIT license, GPU-accelerated).

#### 3. Semantic Search (CLIP Embeddings)

| Option | License | Embedding Dim | GPU Required | Quality | Recommendation |
|--------|---------|---------------|--------------|---------|----------------|
| **OpenCLIP ViT-L/14** | MIT | 768 | Recommended | Excellent | ✅ Already using |
| OpenCLIP ViT-B/32 | MIT | 512 | Optional | Good | Faster, smaller |
| CLIP (OpenAI) | MIT | 512/768 | Recommended | Excellent | Original |
| SigLIP | Apache 2.0 | 768 | Recommended | Excellent | Google's variant |

**Recommendation**: Keep using OpenCLIP ViT-L/14 (already loaded for tagging - zero additional cost).

#### 4. Image Quality Assessment

| Option | License | GPU Required | Type | Recommendation |
|--------|---------|--------------|------|----------------|
| **BRISQUE** | BSD | No | No-reference | ✅ Primary |
| NIQE | BSD | No | No-reference | Alternative |
| pyiqa (multiple) | Apache 2.0 | Optional | Various | Comprehensive |
| CLIP-IQA | MIT | Yes | Learning-based | Advanced option |

**Recommendation**: BRISQUE (already have quality_score field, likely using this).

### Cloud/API Alternatives (NOT Required)

For users who prefer cloud services (not recommended for privacy):

| Feature | Cloud Option | Cost | Privacy Risk |
|---------|-------------|------|--------------|
| Face Recognition | AWS Rekognition | ~$1/1000 images | High - images sent to AWS |
| Face Recognition | Azure Face API | ~$1/1000 images | High - images sent to Azure |
| Face Recognition | Google Vision | ~$1.50/1000 images | High - images sent to Google |
| Semantic Search | OpenAI CLIP API | ~$0.001/image | High - images sent to OpenAI |
| Tagging | Google Vision Labels | ~$1.50/1000 images | High - images sent to Google |

**VAM Tools Philosophy**: All processing happens locally. Your photos never leave your machine.

### Model Storage & Management

Models are downloaded once and cached locally:

```
~/.cache/
├── torch/                      # PyTorch models
│   └── hub/
├── huggingface/               # HuggingFace models
│   └── transformers/
├── insightface/               # Face recognition models
│   └── models/
│       └── buffalo_l/         # ArcFace model (~500MB)
└── open_clip/                 # CLIP models
    └── ViT-L-14-*.pt          # (~900MB)

Total disk space: ~2-3GB for all models
```

### GPU vs CPU Performance

| Feature | GPU (RTX 3060+) | CPU Only | Recommendation |
|---------|-----------------|----------|----------------|
| Face Detection | ~50 faces/sec | ~5 faces/sec | GPU preferred |
| Face Embedding | ~100 faces/sec | ~10 faces/sec | GPU preferred |
| CLIP Embedding | ~20 images/sec | ~1 image/sec | GPU strongly preferred |
| Burst Detection | N/A | ~10,000 images/sec | CPU fine |
| Quality Scoring | ~50 images/sec | ~10 images/sec | GPU helpful |

**Note**: All features work on CPU, but large catalogs benefit significantly from GPU acceleration.

### Docker Configuration for Models

```yaml
# docker-compose.yml
services:
  celery-worker:
    volumes:
      # Share model cache between host and container
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${HOME}/.cache/torch:/root/.cache/torch
      - ${HOME}/.cache/insightface:/root/.insightface
    environment:
      # Prevent re-downloading models
      - TRANSFORMERS_CACHE=/root/.cache/huggingface
      - TORCH_HOME=/root/.cache/torch
```

### Offline Installation

For air-gapped deployments:

```bash
# Download models on internet-connected machine
python -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)
"

python -c "
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
"

# Copy ~/.cache to offline machine
rsync -av ~/.cache/insightface user@offline:/home/user/.cache/
rsync -av ~/.cache/huggingface user@offline:/home/user/.cache/
```

---

## Part 1: UI Redesign (Lightroom-Style Layout)

### Current State Problems

The current UI has grown organically and now has:
- Top navigation with multiple buttons competing for attention
- Single main content area that changes entirely between views
- No persistent context (switching views loses browsing state)
- Filter panel that takes up vertical space above image grid
- No filmstrip for quick image navigation
- Limited panel customization

### Proposed Layout

```
+------------------------------------------------------------------+
|  [Logo]  [Catalog ▼]  [Library | Develop | Map | Jobs]   [⚙️]    | <- Module Picker
+------------------------------------------------------------------+
|        |                                          |               |
| LEFT   |           MAIN CONTENT                   |    RIGHT      |
| PANEL  |                                          |    PANEL      |
|        |    (Grid/Single Image/Map/Jobs)          |               |
| Folders|                                          |  Metadata     |
| Tags   |                                          |  Tags         |
| Faces  |                                          |  Info         |
| Search |                                          |  Actions      |
|        |                                          |               |
|        |                                          |               |
|  [◀]   |                                          |    [▶]        | <- Collapse buttons
+--------+------------------------------------------+---------------+
|                        FILMSTRIP                                  | <- Persistent
|  [◀] [thumb] [thumb] [thumb] [thumb] [thumb] [thumb] [thumb] [▶]  |
+------------------------------------------------------------------+
```

### Module-Based Navigation

Replace current view tabs with **modules** (like Lightroom):

| Module | Purpose | Left Panel | Right Panel |
|--------|---------|------------|-------------|
| **Library** | Browse & organize | Folders, Tags, Faces, Smart Collections | Metadata, Quick Tags, Histogram |
| **Develop** | View single image | Presets (future) | Edit tools (future) |
| **Map** | Geographic view | Location filters | Location details |
| **Jobs** | Background tasks | Job queue | Job details |

### Collapsible Panel System

- **Left Panel** (F7 toggle): Navigation and filtering
- **Right Panel** (F8 toggle): Context and actions
- **Filmstrip** (F6 toggle): Persistent thumbnail strip
- Panels remember state per module
- Smooth CSS transitions for collapse/expand

### Filmstrip Implementation

```javascript
// Persistent across all modules
// Shows current selection/filter results
// Click to jump to image
// Drag to reorder (if in collection)
// Shows current image indicator
```

### Implementation Tasks

1. **Restructure HTML layout**
   - Add CSS Grid for 3-column + filmstrip layout
   - Create collapsible panel components
   - Add module picker in top nav

2. **Create panel components**
   - `LeftPanel.js` - Folder tree, tag browser, face browser, search
   - `RightPanel.js` - Metadata display, quick actions, histogram
   - `Filmstrip.js` - Horizontal scrolling thumbnail strip

3. **Implement module routing**
   - Library module (main browse)
   - Map module (geographic view)
   - Jobs module (background tasks)

4. **Add keyboard shortcuts**
   - F5: Module picker
   - F6: Toggle filmstrip
   - F7: Toggle left panel
   - F8: Toggle right panel
   - G: Grid view
   - E: Single image view

5. **Migrate existing features**
   - Move filters to left panel
   - Move image details to right panel
   - Keep lightbox for full-screen view

---

## Part 2: Face Recognition Feature

### Overview

Implement face detection and recognition similar to Excire's "X-face AI" feature.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Face Detection │ --> │ Face Embedding  │ --> │ Face Clustering │
│  (RetinaFace)   │     │ (ArcFace)       │     │ (HDBSCAN)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      │                       v
         │                      │              ┌─────────────────┐
         │                      │              │  Person Groups  │
         │                      │              │  (Named/Unnamed)│
         │                      │              └─────────────────┘
         │                      │
         v                      v
    100% Local            100% Local
    MIT License           MIT License
    ~100MB model          ~500MB model
```

### Models Used

| Component | Model | License | Size | Source |
|-----------|-------|---------|------|--------|
| Detection | RetinaFace (buffalo_l) | MIT | ~100MB | insightface |
| Embedding | ArcFace (buffalo_l) | MIT | ~400MB | insightface |
| Clustering | HDBSCAN | BSD | N/A | scikit-learn-extra |

### Database Schema

```sql
-- Face detection results
CREATE TABLE faces (
    id UUID PRIMARY KEY,
    image_id UUID REFERENCES images(id),
    catalog_id UUID REFERENCES catalogs(id),

    -- Bounding box (normalized 0-1)
    bbox_x FLOAT NOT NULL,
    bbox_y FLOAT NOT NULL,
    bbox_width FLOAT NOT NULL,
    bbox_height FLOAT NOT NULL,

    -- Face quality metrics
    detection_confidence FLOAT,
    blur_score FLOAT,
    pose_yaw FLOAT,
    pose_pitch FLOAT,

    -- Embedding for recognition (512-dim ArcFace)
    embedding VECTOR(512),

    -- Clustering result
    person_id UUID REFERENCES persons(id),

    created_at TIMESTAMP DEFAULT NOW()
);

-- Person identities
CREATE TABLE persons (
    id UUID PRIMARY KEY,
    catalog_id UUID REFERENCES catalogs(id),

    name VARCHAR(255),  -- NULL for unnamed/unconfirmed
    is_confirmed BOOLEAN DEFAULT FALSE,

    -- Representative face (best quality)
    representative_face_id UUID REFERENCES faces(id),

    -- Metadata
    face_count INTEGER DEFAULT 0,
    image_count INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for similarity search
CREATE INDEX faces_embedding_idx ON faces
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### Implementation Tasks

1. **Face Detection Pipeline** (`vam_tools/analysis/face_detector.py`)
   - Use insightface with RetinaFace backbone
   - Extract face crops with padding
   - Compute quality metrics (blur, pose)
   - GPU acceleration support

2. **Face Embedding** (`vam_tools/analysis/face_embedder.py`)
   - Use ArcFace via insightface
   - Generate 512-dim embeddings
   - Normalize embeddings for cosine similarity

3. **Face Clustering** (`vam_tools/analysis/face_clusterer.py`)
   - HDBSCAN for automatic cluster discovery
   - Handle outliers (low-quality faces)
   - Incremental clustering for new images

4. **Celery Tasks** (`vam_tools/jobs/tasks.py`)
   - `detect_faces_task` - Run face detection on images
   - `cluster_faces_task` - Group faces into persons
   - `merge_persons_task` - Merge two person groups

5. **API Endpoints** (`vam_tools/web/faces_api.py`)
   ```
   GET    /api/catalogs/{id}/persons           - List all persons
   GET    /api/catalogs/{id}/persons/{id}      - Get person details
   PUT    /api/catalogs/{id}/persons/{id}      - Update person (name)
   POST   /api/catalogs/{id}/persons/merge     - Merge persons
   DELETE /api/catalogs/{id}/persons/{id}      - Delete/ungroup person

   GET    /api/catalogs/{id}/faces             - List faces (paginated)
   GET    /api/catalogs/{id}/faces/{id}        - Get face details
   PUT    /api/catalogs/{id}/faces/{id}        - Assign face to person

   GET    /api/catalogs/{id}/images?person_id= - Filter by person
   ```

6. **UI Components**
   - Face browser in left panel (grid of person thumbnails)
   - Person detail view (all faces of a person)
   - Face confirmation workflow (confirm/reject face matches)
   - Name assignment dialog

### Python Dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
faces = [
    "insightface>=0.7.3",      # MIT - Face detection & recognition
    "onnxruntime-gpu>=1.16.0", # MIT - ONNX runtime for models
    "hdbscan>=0.8.33",         # BSD - Clustering algorithm
]
```

---

## Part 3: Free-text/Semantic Search

### Overview

Enable natural language queries like "sunset over mountains" or "people at beach" using image embeddings.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Query    │ --> │ Text Embedding  │ --> │ Vector Search   │
│ "dogs playing"  │     │ (OpenCLIP)      │     │ (pgvector)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               │                        │
                          100% Local               PostgreSQL
                          MIT License              (local DB)
                          Already loaded!
                               │
        ┌──────────────────────┴────────────────────────┘
        v
┌─────────────────┐
│  Image Results  │ (ranked by similarity)
└─────────────────┘

Pre-computed (piggybacks on auto-tagging):
┌─────────────────┐     ┌─────────────────┐
│     Images      │ --> │ Image Embedding │ --> stored in DB
└─────────────────┘     │ (OpenCLIP)      │
                        │ Already loaded! │
                        └─────────────────┘
```

### Models Used

| Component | Model | License | Size | Notes |
|-----------|-------|---------|------|-------|
| Image Embedding | OpenCLIP ViT-L/14 | MIT | ~900MB | **Already loaded for tagging!** |
| Text Embedding | OpenCLIP ViT-L/14 | MIT | Same model | Zero additional memory |
| Vector Search | pgvector | PostgreSQL | N/A | Already have extension |

**Key Insight**: Since we already use OpenCLIP for auto-tagging, semantic search adds **zero additional model overhead**. We just need to save the embeddings we're already computing!

### Database Schema

```sql
-- Add embedding column to images table
ALTER TABLE images ADD COLUMN
    clip_embedding VECTOR(768);  -- OpenCLIP ViT-L/14

-- Index for fast similarity search
CREATE INDEX images_clip_embedding_idx ON images
USING ivfflat (clip_embedding vector_cosine_ops) WITH (lists = 100);

-- Search history (optional)
CREATE TABLE search_history (
    id UUID PRIMARY KEY,
    catalog_id UUID REFERENCES catalogs(id),
    query_text TEXT NOT NULL,
    result_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Implementation Tasks

1. **Modify Auto-Tag Task** (`vam_tools/jobs/tasks.py`)
   - Save CLIP embeddings during tagging (already computed!)
   - Update existing images with embeddings
   - Zero additional GPU time

2. **Search Service** (`vam_tools/analysis/semantic_search.py`)
   - Text-to-embedding conversion using same CLIP model
   - Vector similarity search via pgvector
   - Result ranking and filtering
   - Hybrid search (combine with tags/metadata)

3. **API Endpoints**
   ```
   GET  /api/catalogs/{id}/search?q=sunset    - Semantic search
   POST /api/catalogs/{id}/search             - Advanced search (JSON body)
   GET  /api/catalogs/{id}/similar/{image_id} - Find similar images
   ```

4. **UI Components**
   - Search bar in left panel
   - "Find Similar" button on image hover/lightbox
   - Search results view with relevance scores
   - Recent searches dropdown

### No Additional Dependencies

Semantic search uses only libraries we already have:
- `open-clip-torch` (already installed for tagging)
- `pgvector` (already installed for duplicate detection)

---

## Part 4: Burst Detection

### Overview

Automatically detect and group burst/continuous shooting sequences.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Sort by Time   │ --> │ Gap Detection   │ --> │ Group Creation  │
│  + Camera       │     │ (<2s threshold) │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      │                       v
    100% Local            No ML needed          ┌─────────────────┐
    Pure Python           Pure math             │  Burst Groups   │
    EXIF data only        Fast!                 └─────────────────┘
```

### Models Used

**None!** Burst detection uses only:
- EXIF timestamp data (already extracted)
- Camera make/model (already extracted)
- Simple time-gap algorithm

### Detection Criteria

A burst is a sequence of images where:
1. Same camera (EXIF Make/Model)
2. Timestamps within 2 seconds of each other
3. Similar exposure settings (optional)
4. At least 3 images in sequence

### Database Schema

```sql
-- Burst groups
CREATE TABLE bursts (
    id UUID PRIMARY KEY,
    catalog_id UUID REFERENCES catalogs(id),

    -- Group metadata
    image_count INTEGER NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds FLOAT,

    -- Camera info
    camera_make VARCHAR(255),
    camera_model VARCHAR(255),

    -- Best image selection
    best_image_id UUID REFERENCES images(id),
    selection_method VARCHAR(50),  -- 'quality', 'manual', 'first'

    created_at TIMESTAMP DEFAULT NOW()
);

-- Link images to bursts
ALTER TABLE images ADD COLUMN burst_id UUID REFERENCES bursts(id);
ALTER TABLE images ADD COLUMN burst_sequence INTEGER;  -- Position in burst
```

### Implementation Tasks

1. **Burst Detection Algorithm** (`vam_tools/analysis/burst_detector.py`)
   - Sort images by camera + timestamp
   - Sliding window to find sequences
   - Configurable threshold (default 2s gap)
   - Handle timezone edge cases

2. **Best Image Selection** (`vam_tools/analysis/burst_selector.py`)
   - Use existing quality_score
   - Consider sharpness (blur detection)
   - Face detection quality (if faces present)

3. **Celery Tasks**
   - `detect_bursts_task` - Find burst sequences
   - `select_burst_best_task` - Pick best from each burst

4. **API Endpoints**
   ```
   GET    /api/catalogs/{id}/bursts           - List burst groups
   GET    /api/catalogs/{id}/bursts/{id}      - Get burst details
   PUT    /api/catalogs/{id}/bursts/{id}      - Update (set best image)
   DELETE /api/catalogs/{id}/bursts/{id}      - Ungroup burst

   GET    /api/catalogs/{id}/images?burst_id= - Get images in burst
   GET    /api/catalogs/{id}/images?show_bursts=collapsed - Grid view mode
   ```

5. **UI Components**
   - Burst indicator on thumbnails (stack icon with count)
   - Burst expansion view (click to see all in burst)
   - "Best" badge on selected image
   - Manual best selection override
   - Grid view option: "Collapse bursts" (show only best)

### Grid View Modes

```
Normal:        [1] [2] [3] [4] [5] [6] [7] [8] [9]
                    └─burst─┘     └──burst──┘

Collapsed:     [1] [2*] [5] [6*] [9]
               (* = best of burst, shows stack indicator)
```

### No Additional Dependencies

Burst detection uses only standard Python:
- `datetime` for timestamp comparison
- Existing EXIF data from images table

---

## Summary: External Services & Privacy

### Privacy Commitment

| Feature | External API Calls | Data Leaves Machine |
|---------|-------------------|---------------------|
| Auto-Tagging | **None** | **No** |
| Face Recognition | **None** | **No** |
| Semantic Search | **None** | **No** |
| Burst Detection | **None** | **No** |
| Duplicate Detection | **None** | **No** |

**Your photos never leave your computer.**

### Model Download Summary

| Model | Size | Downloaded From | License |
|-------|------|-----------------|---------|
| OpenCLIP ViT-L/14 | ~900MB | HuggingFace | MIT |
| insightface buffalo_l | ~500MB | insightface servers | MIT |
| ONNX Runtime | ~200MB | PyPI | MIT |

Total: ~1.6GB one-time download, cached locally.

### Resource Requirements

| Configuration | RAM | VRAM | Disk | Processing Speed |
|---------------|-----|------|------|------------------|
| **Minimum** (CPU only) | 8GB | 0 | 5GB | ~1-5 images/sec |
| **Recommended** (GPU) | 16GB | 6GB+ | 10GB | ~20-50 images/sec |
| **Optimal** (GPU) | 32GB | 12GB+ | 20GB | ~50-100 images/sec |

---

## Implementation Priority

### Phase 1: UI Redesign Foundation
1. Implement collapsible panel layout
2. Add filmstrip component
3. Migrate existing browse view to Library module
4. Add keyboard shortcuts

### Phase 2: Semantic Search
1. Modify auto_tag_task to store CLIP embeddings (already computed!)
2. Add clip_embedding column to images table
3. Implement search API endpoint
4. Add search UI in left panel

### Phase 3: Burst Detection
1. Implement burst detection algorithm
2. Create burst database tables
3. Add burst API endpoints
4. Add burst UI indicators and collapse mode

### Phase 4: Face Recognition
1. Add insightface dependency
2. Set up face detection pipeline
3. Implement face embedding and clustering
4. Create person management API
5. Build face browser UI
6. Add person tagging workflow

---

## Technical Dependencies Summary

### New Python Packages

```toml
# pyproject.toml additions
[project.optional-dependencies]
faces = [
    "insightface>=0.7.3",       # MIT - Face detection & recognition
    "onnxruntime-gpu>=1.16.0",  # MIT - GPU inference
    "hdbscan>=0.8.33",          # BSD - Clustering
]

# Already have for search (no changes needed):
# open-clip-torch - already installed
# pgvector - already have extension
```

### PostgreSQL Extensions

```sql
-- Already have
CREATE EXTENSION IF NOT EXISTS vector;

-- Just need to add columns
ALTER TABLE images ADD COLUMN clip_embedding VECTOR(768);
```

---

## Success Metrics

1. **Face Recognition**
   - Detection accuracy >95% for frontal faces
   - Clustering precision >90% (few false merges)
   - <100ms per face embedding

2. **Semantic Search**
   - Relevant results in top 10 for >80% of queries
   - <500ms search response time
   - Zero additional processing (embeddings from tagging)

3. **Burst Detection**
   - >95% accuracy in burst grouping
   - Reasonable best-image selection (matches human preference >70%)
   - <1 second to detect bursts in 10k image catalog

4. **UI Redesign**
   - Panel transitions <200ms
   - Filmstrip scroll smooth at 60fps
   - All existing features accessible in new layout
