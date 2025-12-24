# Burst Management

## Overview

Burst Management is an intelligent feature that automatically detects and organizes continuous shooting sequences (bursts) in your photo catalog. When photographers use continuous shooting mode or rapid capture, they often end up with multiple nearly-identical shots taken within seconds of each other. Burst Management helps you identify these sequences, automatically select the best image from each burst, and efficiently manage your photo library.

## What is a Burst?

A **burst** is a sequence of images that meet the following criteria:

1. **Same Camera**: All images were taken with the same camera (same make and model)
2. **Rapid Succession**: Images were taken within a configurable time threshold (default: 2 seconds between consecutive shots)
3. **Minimum Size**: The sequence contains at least a minimum number of images (default: 3 images)

### Example Burst Scenarios

- **Action Photography**: Capturing a soccer player kicking a ball - 15 frames in 2 seconds
- **Portrait Sessions**: Taking multiple shots to get the perfect smile - 5-10 frames in 3 seconds
- **Wildlife Photography**: Bird in flight sequence - 20 frames in 1 second
- **Event Coverage**: Group photo with multiple takes - 4-6 frames in 5 seconds

## How It Works

### 1. Detection Algorithm

Burst detection uses a pure Python algorithm (no machine learning required) that:

1. **Groups by Camera**: First separates images by camera make and model to prevent mixing bursts from different cameras
2. **Sorts by Time**: Orders images chronologically based on EXIF timestamps
3. **Identifies Sequences**: Scans through sorted images to find consecutive shots within the time threshold
4. **Creates Burst Groups**: Forms burst groups when sequences meet the minimum size requirement

### 2. Best Image Selection

For each detected burst, the system automatically selects the "best" image using:

- **Quality Score**: The primary selection method uses the image's quality score (based on sharpness, exposure, composition)
- **Manual Override**: Users can manually select a different image as the best if they prefer
- **Selection Methods**:
  - `quality` (default): Highest quality score
  - `first`: First image in the sequence
  - `middle`: Middle image in the sequence
  - `manual`: User-selected image

### 3. Parallel Processing

Burst detection is optimized for large catalogs using the coordinator-worker pattern:

- **Coordinator Task**: Divides images into time-based batches
- **Worker Tasks**: Process batches in parallel across multiple Celery workers
- **Finalizer Task**: Merges bursts that span batch boundaries and writes results to database
- **Progress Tracking**: Real-time progress updates via Redis pub/sub

## Database Schema

### Bursts Table

```sql
CREATE TABLE bursts (
    id UUID PRIMARY KEY,
    catalog_id UUID NOT NULL REFERENCES catalogs(id) ON DELETE CASCADE,
    image_count INTEGER NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds FLOAT,
    camera_make VARCHAR(255),
    camera_model VARCHAR(255),
    best_image_id UUID,
    selection_method VARCHAR(50) DEFAULT 'quality',
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Image Columns

The `images` table includes burst-related columns:

```sql
ALTER TABLE images
ADD COLUMN burst_id UUID REFERENCES bursts(id) ON DELETE SET NULL,
ADD COLUMN burst_sequence INTEGER;
```

- `burst_id`: Links the image to its parent burst group (NULL if not in a burst)
- `burst_sequence`: Position of the image within the burst (0-indexed)

## API Endpoints

### Start Burst Detection

```http
POST /api/catalogs/{catalog_id}/jobs
Content-Type: application/json

{
  "job_type": "detect_bursts",
  "params": {
    "gap_threshold": 2.0,
    "min_burst_size": 3,
    "batch_size": 5000
  }
}
```

**Parameters:**
- `gap_threshold`: Maximum seconds between consecutive images in a burst (default: 2.0)
- `min_burst_size`: Minimum images required to form a burst (default: 3)
- `batch_size`: Number of images to process per worker batch (default: 5000)

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Burst detection job started"
}
```

### List Bursts

```http
GET /api/catalogs/{catalog_id}/bursts?limit=100&offset=0
```

**Response:**
```json
{
  "bursts": [
    {
      "id": "burst-uuid",
      "image_count": 12,
      "start_time": "2024-12-24T14:30:45.123Z",
      "end_time": "2024-12-24T14:30:47.890Z",
      "duration_seconds": 2.767,
      "camera_make": "Canon",
      "camera_model": "EOS R5",
      "best_image_id": "image-uuid",
      "selection_method": "quality",
      "best_thumbnail_url": "/api/catalogs/{catalog_id}/images/{image_id}/thumbnail"
    }
  ],
  "total": 47,
  "limit": 100,
  "offset": 0
}
```

### Get Burst Details

```http
GET /api/catalogs/{catalog_id}/bursts/{burst_id}
```

**Response:**
```json
{
  "id": "burst-uuid",
  "image_count": 12,
  "start_time": "2024-12-24T14:30:45.123Z",
  "end_time": "2024-12-24T14:30:47.890Z",
  "duration_seconds": 2.767,
  "camera_make": "Canon",
  "camera_model": "EOS R5",
  "best_image_id": "image-uuid",
  "selection_method": "quality",
  "images": [
    {
      "id": "image-uuid-1",
      "source_path": "/photos/2024/IMG_1234.CR3",
      "sequence": 0,
      "quality_score": 0.82,
      "is_best": false,
      "thumbnail_url": "/api/catalogs/{catalog_id}/images/{image_id}/thumbnail"
    },
    {
      "id": "image-uuid-2",
      "source_path": "/photos/2024/IMG_1235.CR3",
      "sequence": 1,
      "quality_score": 0.95,
      "is_best": true,
      "thumbnail_url": "/api/catalogs/{catalog_id}/images/{image_id}/thumbnail"
    }
  ]
}
```

### Update Best Image

```http
PUT /api/catalogs/{catalog_id}/bursts/{burst_id}
Content-Type: application/json

{
  "best_image_id": "new-best-image-uuid"
}
```

**Response:**
```json
{
  "status": "updated"
}
```

### Apply Burst Selection

Mark all non-best images in a burst for deletion:

```http
POST /api/catalogs/{catalog_id}/bursts/{burst_id}/apply
```

**Response:**
```json
{
  "burst_id": "burst-uuid",
  "images_marked": 11,
  "best_image_id": "image-uuid"
}
```

### Batch Apply Burst Selections

Apply burst selections to multiple bursts at once:

```http
POST /api/catalogs/{catalog_id}/bursts/batch-apply
Content-Type: application/json

{
  "burst_ids": ["burst-uuid-1", "burst-uuid-2", "burst-uuid-3"]
}
```

**Response:**
```json
{
  "bursts_processed": 3,
  "total_images_marked": 34
}
```

## Web UI Features

### Burst Indicators

In the catalog grid view, images that are part of a burst show:
- Camera icon badge in the top-right corner
- Burst count (number of images in the burst)
- Click the indicator to open the burst detail modal

### Burst Detail Modal

The burst modal displays:
- **Burst Information**:
  - Camera make and model
  - Duration in seconds
  - Start timestamp
  - Total image count

- **Image Grid**: All images in the burst with:
  - Thumbnail previews
  - Quality score percentage
  - "BEST" badge on the selected best image
  - Green border on the best image
  - Click any image to set it as the new best

### Burst Actions

- **Detect Bursts**: Button in the catalog actions menu to start burst detection
- **Collapse Bursts**: Grid view option to show only the best image from each burst
- **Apply Selection**: Mark non-best images for deletion (batch or individual bursts)

## Usage Workflows

### Initial Burst Detection

After scanning a catalog with images:

1. Click "Detect Bursts" in the catalog actions menu
2. (Optional) Adjust gap threshold and minimum burst size
3. Monitor job progress in the Jobs panel
4. Once complete, burst indicators appear on images in the grid

### Reviewing Bursts

1. Click the camera icon on any burst-marked image
2. Review all images in the burst sequence
3. Check the automatically selected best image (marked with "BEST" badge)
4. (Optional) Click a different image to change the best selection
5. View quality scores to understand the automatic selection

### Managing Storage

To clean up burst duplicates:

1. Navigate to the bursts list view
2. Review bursts to ensure best images are correctly selected
3. Use "Apply Selection" on individual bursts or "Batch Apply" for multiple bursts
4. This marks non-best images for deletion
5. Review marked images before final deletion
6. Execute deletion through the catalog management interface

### Customizing Detection

For different photography styles, adjust parameters:

**Sports/Action Photography** (fast bursts):
```json
{
  "gap_threshold": 0.5,
  "min_burst_size": 5
}
```

**Portrait Sessions** (slower bursts):
```json
{
  "gap_threshold": 5.0,
  "min_burst_size": 3
}
```

**Wedding/Events** (mixed shooting):
```json
{
  "gap_threshold": 3.0,
  "min_burst_size": 4
}
```

## Performance Characteristics

### Processing Speed

- **Small Catalogs** (<10,000 images): 5-15 seconds
- **Medium Catalogs** (10,000-100,000 images): 30-90 seconds
- **Large Catalogs** (>100,000 images): 2-5 minutes

Speed depends on:
- Number of images with timestamps
- Number of different cameras in the catalog
- Database performance
- Number of available Celery workers

### Scalability

The parallel processing architecture scales linearly with worker count:
- 1 worker: Baseline performance
- 4 workers: ~3.5x faster
- 8 workers: ~6-7x faster
- 16 workers: ~10-12x faster

### Memory Usage

Memory-efficient design:
- Streams images from database (doesn't load all into memory)
- Processes in batches to limit memory footprint
- Worker memory usage: ~50-200 MB per worker

## Best Practices

### When to Run Burst Detection

- **After initial catalog scan**: Detect all bursts in your library
- **After adding new photos**: Re-run to detect new bursts
- **Before storage cleanup**: Identify duplicates before deleting images
- **After event shoots**: Process recent event coverage

### Optimizing Parameters

1. **Review Sample Results**: Start with defaults and review a few bursts
2. **Adjust Gap Threshold**: If sequences are being split, increase; if unrelated images are grouped, decrease
3. **Set Minimum Size**: Higher values reduce false positives but may miss small bursts
4. **Test and Iterate**: Run detection, review results, adjust parameters, repeat

### Quality Score Considerations

The automatic best image selection relies on quality scores:
- Ensure quality scoring has been run on your catalog
- Higher quality scores indicate sharper, better-exposed images
- Manual review is recommended for important bursts (portraits, key moments)
- Consider camera-specific quality score calibration

### Storage Management Strategy

1. **Don't Auto-Delete**: Always review burst selections before deletion
2. **Archive First**: Consider archiving non-best images rather than deleting
3. **Keep Originals**: For critical events, keep all burst images as backup
4. **Test on Sample**: Test burst selection on a small subset before batch applying

## Technical Details

### Burst Detection Algorithm Complexity

- **Time Complexity**: O(n log n) where n is number of images
  - Sorting images by timestamp: O(n log n)
  - Linear scan for burst detection: O(n)
  - Overall: dominated by sorting

- **Space Complexity**: O(n) for storing image metadata in memory during processing

### Database Indexing

For optimal performance, ensure these indexes exist:

```sql
CREATE INDEX idx_images_catalog_date ON images(catalog_id, date_taken);
CREATE INDEX idx_images_burst_id ON images(burst_id);
CREATE INDEX idx_bursts_catalog_id ON bursts(catalog_id);
```

### Batch Boundary Handling

Bursts that span batch boundaries are detected and merged:
1. Each worker processes its time range independently
2. Finalizer task receives all worker results
3. Identifies bursts with overlapping time ranges
4. Merges adjacent bursts if they meet burst criteria
5. Writes final merged bursts to database

## Troubleshooting

### No Bursts Detected

**Possible Causes:**
- Images don't have EXIF timestamps
- Gap threshold is too small for your shooting style
- Minimum burst size is too large
- Images are from different cameras (check camera metadata)

**Solutions:**
1. Verify images have `date_taken` values in database
2. Increase `gap_threshold` to 5-10 seconds
3. Decrease `min_burst_size` to 2
4. Check camera make/model consistency

### Too Many False Positives

**Possible Causes:**
- Gap threshold is too large
- Different shooting scenarios mixed together

**Solutions:**
1. Decrease `gap_threshold` to 1.0 seconds or less
2. Increase `min_burst_size` to 5 or more
3. Run detection separately for different time periods

### Wrong Image Selected as Best

**Possible Causes:**
- Quality scores not calculated or inaccurate
- Specific artistic preference differs from quality metrics

**Solutions:**
1. Re-run quality scoring on catalog
2. Manually select the preferred image using the UI
3. Review and adjust quality scoring parameters if needed

### Slow Processing

**Possible Causes:**
- Limited Celery workers
- Large batch size causing memory issues
- Database performance bottlenecks

**Solutions:**
1. Increase number of Celery workers
2. Decrease `batch_size` to 2000-3000
3. Ensure database indexes are present
4. Monitor database and Redis performance

## Future Enhancements

Planned improvements for burst management:

1. **ML-Based Selection**: Use computer vision to detect focus, facial expressions, and composition
2. **Burst Similarity**: Detect similar bursts across different time periods
3. **Smart Archiving**: Automatically archive non-best images to separate storage
4. **Burst Analytics**: Statistics on shooting patterns and burst usage
5. **Video Export**: Create time-lapse videos from burst sequences
6. **Focus Stacking**: Automatically stack bursts for extended depth of field

## Related Features

- **Quality Scoring**: Provides the quality metrics used for best image selection
- **Duplicate Detection**: Complementary feature for finding identical/similar images across the catalog
- **Metadata Extraction**: Required for timestamp and camera information
- **Job Management**: Monitors burst detection progress and handles errors

## See Also

- [Quality Scoring Documentation](./quality-scoring.md) (if exists)
- [Duplicate Detection Documentation](./duplicate-detection.md) (if exists)
- [API Reference](../api/README.md) (if exists)
- [Database Schema](../technical/database-schema.md) (if exists)
