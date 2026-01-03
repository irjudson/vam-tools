# Auto-Tagging System Design

## Overview

Automatically tag images with AI-detected subjects, scenes, and content to enable semantic search and better organization.

## Goals

1. **Semantic Search**: Find images by content ("photos of dogs", "beach sunset")
2. **Better Organization**: Auto-create albums by detected content
3. **Discovery**: Uncover forgotten photos by content
4. **User Control**: Allow manual tag editing and model selection

## Architecture

### Model Selection

**Primary: CLIP (Contrastive Language-Image Pre-training)**
- **Pros**:
  - Zero-shot classification (no training needed)
  - Natural language queries ("sunset over ocean")
  - Excellent semantic understanding
  - Can detect scenes, objects, moods, styles
  - Lightweight inference with ViT models
- **Cons**:
  - Less precise than specialized models
  - Requires text prompts (but we can use predefined taxonomy)
- **Decision**: Use CLIP as primary model

**Secondary: YOLOv8 (for precise object detection)**
- **Use case**: When precise bounding boxes needed
- **Future enhancement**: Face detection, specific objects

### Tag Taxonomy

Hierarchical tag structure for better organization:

```
Subjects:
  - People
    - Portrait
    - Group
    - Candid
  - Animals
    - Dogs
    - Cats
    - Birds
    - Wildlife
  - Nature
    - Flowers
    - Trees
    - Landscapes
  - Objects
    - Food
    - Architecture
    - Vehicles
    - Technology

Scenes:
  - Indoor
  - Outdoor
  - Urban
  - Rural
  - Beach
  - Mountain
  - Forest
  - Water

Lighting:
  - Daylight
  - Golden Hour
  - Blue Hour
  - Night
  - Studio

Mood/Style:
  - Vibrant
  - Moody
  - Minimalist
  - Vintage
  - Abstract
```

### Database Schema

Add to `catalog.db`:

```sql
-- Tags table
CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL,  -- 'subject', 'scene', 'lighting', 'mood'
    parent_id INTEGER,        -- for hierarchical tags
    FOREIGN KEY (parent_id) REFERENCES tags(id)
);

-- Image tags (many-to-many)
CREATE TABLE image_tags (
    image_id TEXT NOT NULL,
    tag_id INTEGER NOT NULL,
    confidence REAL NOT NULL,  -- 0.0 to 1.0
    source TEXT NOT NULL,      -- 'clip', 'yolo', 'manual'
    created_at TEXT NOT NULL,
    PRIMARY KEY (image_id, tag_id),
    FOREIGN KEY (image_id) REFERENCES images(id),
    FOREIGN KEY (tag_id) REFERENCES tags(id)
);

-- Tag synonyms for search
CREATE TABLE tag_synonyms (
    tag_id INTEGER NOT NULL,
    synonym TEXT NOT NULL,
    PRIMARY KEY (tag_id, synonym),
    FOREIGN KEY (tag_id) REFERENCES tags(id)
);
```

### System Components

```
vam_tools/
├── analysis/
│   ├── auto_tagger.py        # Main tagging engine
│   ├── tag_models.py         # Model wrappers (CLIP, YOLO)
│   └── tag_taxonomy.py       # Tag definitions and hierarchy
├── core/
│   └── tag_manager.py        # Tag CRUD operations
└── web/
    └── api.py                # Tag endpoints (add later)
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. **Database schema** - Add tags tables to catalog
2. **Tag taxonomy** - Define comprehensive tag list
3. **Tag manager** - CRUD operations for tags
4. **Tests** - Full coverage for tag management

### Phase 2: CLIP Integration (Week 2)
1. **Model loader** - Load CLIP model (ViT-B/32 or ViT-L/14)
2. **Batch inference** - Process images in batches for efficiency
3. **GPU acceleration** - Use PyTorch CUDA
4. **Confidence thresholding** - Only keep high-confidence tags
5. **Tests** - Mock CLIP for testing, real inference tests

### Phase 3: Analysis Integration (Week 3)
1. **Scanner integration** - Run tagging during analysis
2. **Progress tracking** - Show tagging progress
3. **Incremental tagging** - Tag only new images
4. **Performance optimization** - Batch processing, caching
5. **Tests** - End-to-end analysis with tagging

### Phase 4: Search & UI (Week 4)
1. **Search API** - Search by tags, combine with other filters
2. **Tag management UI** - View, edit, remove tags
3. **Auto-complete** - Tag suggestions in search
4. **Tag clouds** - Visualize most common tags
5. **Tests** - Search functionality, UI interactions

## API Design

### CLI

```bash
# Run analysis with auto-tagging
vam-analyze /path/to/catalog -s /path/to/photos --auto-tag

# Tag existing catalog
vam-analyze /path/to/catalog --retag-all

# Configure tag models
vam-analyze /path/to/catalog --tag-model clip-vit-b32  # or clip-vit-l14
```

### Python API

```python
from lumina.analysis.auto_tagger import AutoTagger
from lumina.core.catalog import Catalog

# Initialize
catalog = Catalog("catalog.db")
tagger = AutoTagger(model="clip-vit-b32", device="cuda")

# Tag single image
tags = tagger.tag_image("photo.jpg", threshold=0.25)
# Returns: [
#     {"tag": "sunset", "confidence": 0.92, "category": "scene"},
#     {"tag": "beach", "confidence": 0.87, "category": "scene"},
#     {"tag": "ocean", "confidence": 0.78, "category": "subject"}
# ]

# Tag batch
results = tagger.tag_batch(image_paths, batch_size=32)

# Search by tags
images = catalog.search_by_tags(["sunset", "beach"], match_all=True)
```

### Web API

```javascript
// Get tags for image
GET /api/images/{id}/tags
Response: [
  {
    "tag": "sunset",
    "confidence": 0.92,
    "category": "scene",
    "source": "clip"
  }
]

// Add manual tag
POST /api/images/{id}/tags
Body: {"tag": "vacation", "source": "manual"}

// Remove tag
DELETE /api/images/{id}/tags/{tag_id}

// Search by tags
GET /api/search/tags?q=sunset,beach&match=all
```

## Performance Considerations

### Batch Processing
- Process 32-64 images per batch
- Preload and preprocess images in parallel
- Use DataLoader for efficient batching

### GPU Memory Management
- Monitor VRAM usage
- Adjust batch size dynamically
- Fall back to CPU if VRAM exhausted

### Caching
- Cache CLIP embeddings (512-dim vectors)
- Reuse embeddings for similarity search
- Store embeddings in catalog for future queries

### Parallelization
- Multi-GPU support for large catalogs
- Parallel preprocessing on CPU
- Async tag storage

## Testing Strategy

### Unit Tests
- Tag taxonomy validation
- Tag manager CRUD operations
- Model loading and inference (mocked)
- Confidence thresholding logic

### Integration Tests
- End-to-end tagging pipeline
- Database integration
- Search functionality
- Batch processing

### Performance Tests
- Benchmark inference speed
- Memory usage profiling
- Large catalog handling (100k+ images)

## Code Quality Standards

1. **Comprehensive Docstrings**
   - All classes and methods fully documented
   - Google-style docstrings with examples
   - Type hints for all parameters and returns

2. **Test Coverage**
   - Maintain ≥80% coverage throughout
   - Write tests before or alongside implementation (TDD)
   - Mock external dependencies (CLIP models)

3. **Error Handling**
   - Graceful degradation if model fails
   - Clear error messages
   - Retry logic for transient failures

4. **Logging**
   - Progress updates during tagging
   - Performance metrics (images/sec)
   - Warning for low-confidence tags

## Dependencies

```toml
[tool.poetry.dependencies]
# Existing dependencies...
transformers = "^4.30.0"      # For CLIP
torch = "^2.0.0"              # Already in project
torchvision = "^0.15.0"       # Image preprocessing
pillow = "^10.0.0"            # Already in project
```

## Migration Path

1. **Backward Compatibility**: Existing catalogs work without tags
2. **Opt-in**: Tagging is optional (`--auto-tag` flag)
3. **Incremental**: Tag only new images by default
4. **Upgrade Script**: Provide script to tag existing catalogs

## Success Metrics

1. **Accuracy**: ≥85% precision on common tags
2. **Performance**: ≥100 images/sec on GPU (RTX 3060+)
3. **Coverage**: ≥80% test coverage for all tagging code
4. **User Adoption**: Track usage via analytics (opt-in)

## Future Enhancements

1. **Face Recognition**: Identify and tag people
2. **Custom Models**: Fine-tune on user's collection
3. **Smart Albums**: Auto-create albums by tags
4. **Tag Suggestions**: Learn from manual edits
5. **Multi-modal Search**: Combine tags + similarity + metadata

---

**Status**: Design Complete ✅
**Next**: Begin Phase 1 Implementation
**Target**: Maintain 80%+ test coverage throughout
