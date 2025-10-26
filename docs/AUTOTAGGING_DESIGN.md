# Auto-Tagging System Design

## Overview

Automatically tag images with detected attributes to enable rich organization, search, and filtering capabilities. All tags include confidence scores, and low-confidence tags are queued for manual review.

## Tag Categories

### 1. Location Tags

**Source**: GPS EXIF data (GPSLatitude, GPSLongitude, GPSAltitude)

**Tags Generated**:
- Country (e.g., "USA", "France", "Japan")
- State/Region (e.g., "California", "Île-de-France")
- City (e.g., "San Francisco", "Paris")
- Location name (if near known landmark)
- Altitude category (e.g., "sea-level", "mountain")

**Implementation**:
- Extract GPS coordinates from EXIF
- Use reverse geocoding API (OpenStreetMap Nominatim or similar)
- Cache location lookups to avoid repeated API calls
- Confidence: 95% (GPS data is reliable when present)

**Example**:
```json
{
  "location": {
    "coordinates": [37.7749, -122.4194],
    "country": "USA",
    "state": "California",
    "city": "San Francisco",
    "confidence": 95
  }
}
```

### 2. Temporal Tags

**Source**: Date/time from EXIF or selected date

**Tags Generated**:
- Year, Month, Day
- Season (Spring, Summer, Fall, Winter)
- Time of day (Morning, Afternoon, Evening, Night)
- Day of week
- Holiday/Event (if near known holidays)

**Implementation**:
- Extract from DateTimeOriginal or selected_date
- Calculate season based on hemisphere and date
- Time of day from hour (morning: 5-11, afternoon: 12-17, evening: 18-20, night: 21-4)
- Confidence: Inherits from date confidence score

**Example**:
```json
{
  "temporal": {
    "year": 2023,
    "month": "June",
    "season": "Summer",
    "time_of_day": "Afternoon",
    "day_of_week": "Saturday",
    "confidence": 95
  }
}
```

### 3. Camera/Device Tags

**Source**: EXIF Make, Model, LensModel

**Tags Generated**:
- Camera make (e.g., "Canon", "Apple", "Sony")
- Camera model (e.g., "iPhone 14 Pro", "EOS 5D Mark IV")
- Camera type (e.g., "smartphone", "DSLR", "mirrorless", "point-and-shoot")
- Lens info (if available)

**Implementation**:
- Extract from EXIF Make/Model fields
- Normalize manufacturer names (e.g., "NIKON CORPORATION" → "Nikon")
- Classify camera type based on model patterns
- Confidence: 100% (EXIF data is authoritative)

**Example**:
```json
{
  "camera": {
    "make": "Apple",
    "model": "iPhone 14 Pro",
    "type": "smartphone",
    "lens": null,
    "confidence": 100
  }
}
```

### 4. Image Content Tags

**Source**: AI/ML image analysis

**Tags Generated**:
- **People**: Face count, face detection boxes, face recognition IDs
- **Objects**: Detected objects (car, dog, cat, tree, food, etc.)
- **Scenes**: Scene type (indoor, outdoor, beach, mountain, city, concert, party)
- **Activities**: Detected activities (sports, dining, travel, etc.)
- **Text**: OCR for receipts, signs, documents
- **Quality**: Blur detection, exposure issues, composition quality

**Implementation Options**:

#### Option A: Local Models (Privacy-focused, GPU-accelerated)
- **CLIP** (OpenAI): General image understanding, zero-shot classification
- **YOLO** or **Faster R-CNN**: Object detection
- **RetinaFace** or **MTCNN**: Face detection
- **EasyOCR** or **Tesseract**: Text recognition
- **BRISQUE**: Image quality assessment

**Pros**: Privacy, no API costs, works offline, GPU accelerated
**Cons**: Requires GPU, model downloads, more complex setup

#### Option B: Cloud APIs (Easier, but less private)
- **Google Cloud Vision API**
- **AWS Rekognition**
- **Azure Computer Vision**

**Pros**: Easy to use, very accurate, maintained
**Cons**: Privacy concerns, API costs, requires internet

**Recommendation**: Start with local models using GPU, add cloud API support as optional

**Example**:
```json
{
  "content": {
    "people": {
      "count": 3,
      "faces": [
        {"box": [100, 150, 200, 250], "confidence": 92, "person_id": null}
      ],
      "confidence": 92
    },
    "objects": [
      {"label": "dog", "confidence": 87},
      {"label": "bicycle", "confidence": 95}
    ],
    "scene": {
      "type": "outdoor",
      "sub_type": "park",
      "confidence": 81
    },
    "text": {
      "detected": false,
      "content": null,
      "confidence": 0
    },
    "quality": {
      "blur_score": 0.05,
      "exposure": "normal",
      "confidence": 90
    }
  }
}
```

### 5. Technical Tags

**Source**: EXIF technical metadata

**Tags Generated**:
- ISO speed
- Aperture (f-stop)
- Shutter speed
- Focal length
- Flash used (yes/no)
- Orientation (portrait/landscape)
- HDR/Panorama flags

**Implementation**:
- Extract from EXIF technical fields
- Normalize values for categorization
- Confidence: 100% (EXIF data is authoritative)

**Example**:
```json
{
  "technical": {
    "iso": 400,
    "aperture": 2.8,
    "shutter_speed": "1/500",
    "focal_length": 50,
    "flash": false,
    "orientation": "landscape",
    "hdr": false,
    "confidence": 100
  }
}
```

## Tag Confidence Levels

All tags include a confidence score (0-100):

- **95-100**: High confidence - Auto-approve (GPS, EXIF data)
- **80-94**: Medium-high confidence - Auto-approve with review available
- **60-79**: Medium confidence - Queue for review
- **Below 60**: Low confidence - Require manual review

## Tag Review Process

### Review Queue

Images with low-confidence tags are added to review queue:

```json
{
  "review_queue": [
    {
      "image_id": "sha256:abc123...",
      "review_type": "tag_confidence",
      "tags_for_review": [
        {
          "category": "content",
          "tag": "concert",
          "confidence": 65,
          "reason": "Scene classification uncertain"
        }
      ],
      "priority": "medium"
    }
  ]
}
```

### Review UI

Web interface includes tag review section:
- Show image with detected tags
- Display confidence scores
- Allow user to:
  - Approve tag
  - Reject tag
  - Modify tag
  - Add additional tags
- Learn from corrections (optional: improve model)

### Bulk Operations

Allow reviewing similar images together:
- "All images tagged 'concert' with confidence < 80"
- "All images with face detection"
- "All images with no content tags"

## Integration with Catalog

### Updated ImageRecord

```python
@dataclass
class TagInfo:
    """Tag information for an image."""
    location: Optional[Dict[str, any]] = None
    temporal: Optional[Dict[str, any]] = None
    camera: Optional[Dict[str, any]] = None
    content: Optional[Dict[str, any]] = None
    technical: Optional[Dict[str, any]] = None
    user_tags: List[str] = field(default_factory=list)  # Manual tags

@dataclass
class ImageRecord:
    # ... existing fields ...
    tags: Optional[TagInfo] = None
    tags_reviewed: bool = False
```

### Workflow Integration

**New Phase**: TAGGING (after ANALYZING, before REVIEWING)

```
ANALYZING → TAGGING → REVIEWING → PLANNING → VERIFYING → EXECUTING
```

The tagging phase:
1. Processes all images in catalog
2. Generates tags for each category
3. Scores confidence
4. Queues low-confidence tags for review
5. Updates catalog with tag data

## Use Cases

### 1. Smart Search

```
Find: "outdoor photos with dogs taken in summer with iPhone"

Filters:
- content.scene.type = "outdoor"
- content.objects contains "dog"
- temporal.season = "Summer"
- camera.make = "Apple"
```

### 2. Auto-Organization

Instead of just YYYY-MM, create richer structure:

```
2023/
├── 06-Summer/
│   ├── Vacation-Paris/          # Location tag
│   ├── Concert-Music-Festival/  # Content tag
│   └── Family-Gatherings/       # Content tag (people)
```

### 3. Quality Filtering

```
Find all blurry images: content.quality.blur_score > 0.3
Find all underexposed images: technical.exposure = "underexposed"
```

### 4. Receipt Management

```
Find all receipts: content.text.detected = true AND content.scene.type = "document"
Extract text from receipts for expense tracking
```

## Implementation Plan

### Phase 1: Foundation (Week 1-2)
- [ ] Add TagInfo to type system
- [ ] Create tag extractor base class
- [ ] Implement location tagging (GPS → geocoding)
- [ ] Implement temporal tagging
- [ ] Implement camera/device tagging
- [ ] Add tags to catalog schema

### Phase 2: Content Analysis (Week 3-4)
- [ ] Set up local AI model infrastructure (GPU support)
- [ ] Implement CLIP for scene classification
- [ ] Implement YOLO for object detection
- [ ] Implement face detection
- [ ] Implement OCR for text
- [ ] Implement quality assessment

### Phase 3: Review System (Week 5-6)
- [ ] Build tag review queue system
- [ ] Create review UI in web interface
- [ ] Add tag editing capabilities
- [ ] Implement bulk review operations
- [ ] Add tag confidence visualization

### Phase 4: Search & Organization (Week 7-8)
- [ ] Build tag-based search API
- [ ] Add search UI to web interface
- [ ] Integrate tags into plan generation
- [ ] Support tag-based folder organization
- [ ] Add tag statistics dashboard

## Dependencies

### Python Packages

```python
# Location tagging
geopy>=2.3.0           # Geocoding

# Image content analysis
torch>=2.0.0           # PyTorch for GPU
torchvision>=0.15.0    # Vision models
transformers>=4.30.0   # CLIP and other models
ultralytics>=8.0.0     # YOLOv8
easyocr>=1.7.0         # Text recognition
opencv-python>=4.8.0   # Image processing

# Image quality
scikit-image>=0.21.0   # Quality metrics
```

### Models to Download

- **CLIP** (ViT-B/32): ~350 MB - General image understanding
- **YOLOv8n**: ~6 MB - Fast object detection
- **RetinaFace**: ~1.5 MB - Face detection
- **EasyOCR** English model: ~50 MB - Text recognition

**Total**: ~400-500 MB

### Hardware Requirements

- **Minimum**: CPU-only (slower, ~5-10 images/sec)
- **Recommended**: NVIDIA GPU with 4GB+ VRAM (~50-100 images/sec)
- **Optimal**: NVIDIA GPU with 8GB+ VRAM (~100-200 images/sec)

## Privacy Considerations

- All processing done locally by default
- No data sent to cloud unless user opts in
- Face recognition IDs are local only (not names)
- User can disable specific tag categories
- Catalog file is local and encrypted (optional)

## Future Enhancements

- **Face clustering**: Group photos by person without identification
- **Duplicate content**: Find visually similar images even if not exact duplicates
- **Video tagging**: Extend to video content analysis
- **Audio tagging**: Detect music, speech, ambient sounds in videos
- **Smart albums**: Auto-create albums based on tag combinations
- **Tag learning**: Improve models based on user corrections
- **Custom tags**: User-defined tag categories with ML training

## Notes

This feature transforms the tool from simple chronological organization to intelligent content-aware catalog management. Combined with the existing duplicate and burst detection, it creates a comprehensive photo management solution.

The confidence-based review system ensures accuracy while maintaining high automation. Users only review what needs attention.
