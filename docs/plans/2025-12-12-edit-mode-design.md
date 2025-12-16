# Edit Mode Design

## Overview

Add an Image Edit Mode for viewing images at full resolution with zoom/pan, displaying metadata and histogram, and performing non-destructive transforms (rotate, flip). Edits are stored in the database with XMP sidecar export for Darktable compatibility.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| View type | Separate Edit Mode (not lightbox) | More space for tools, cleaner separation |
| V1 tools | View + rotate/flip | Simple transforms, defer complex image processing |
| Storage | Database + XMP export | Fast queries, portable when needed |
| Histogram | Server-side on-demand | Works for all formats including RAW/HEIC |

## Layout

```
+-------------------------------------------------------------+
| [<- Back] [Prev] [Next]     Image: DSC_1234.jpg     [Save]  |  <- Toolbar
+-------------------------------------------+-----------------+
|                                           | FILE INFO       |
|                                           | - Filename      |
|                                           | - Path          |
|           MAIN IMAGE AREA                 | - Dimensions    |
|           (zoomable/pannable)             | - File size     |
|                                           | - Format        |
|                                           +-----------------+
|                                           | HISTOGRAM       |
|                                           | [RGB graph]     |
|                                           +-----------------+
|                                           | METADATA        |
|                                           | - Camera/Lens   |
|                                           | - Exposure      |
|                                           | - Date taken    |
|                                           +-----------------+
|                                           | TRANSFORMS      |
|                                           | [R90] [L90]     |
|                                           | [FlipH] [FlipV] |
+-------------------------------------------+-----------------+
| [Zoom: 100%] [Fit] [1:1]                     [Reset Edits]  |  <- Footer
+-------------------------------------------------------------+
```

## Data Model

### Database Changes

Add `edit_data` JSONB column to `images` table:

```python
edit_data = Column(JSONB, nullable=True, default=None)
```

Structure:
```json
{
    "version": 1,
    "transforms": {
        "rotation": 0,
        "flip_h": false,
        "flip_v": false
    }
}
```

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/images/{id}/full` | Serve full-size image (with transforms applied) |
| `GET` | `/api/images/{id}/histogram` | Return RGB histogram data as JSON |
| `GET` | `/api/images/{id}/edit` | Get current edit_data |
| `PUT` | `/api/images/{id}/edit` | Update edit_data |
| `DELETE` | `/api/images/{id}/edit` | Reset to original (clear edit_data) |
| `POST` | `/api/images/{id}/export-xmp` | Write XMP sidecar file |

### Histogram Response

```json
{
    "red": [0, 5, 12, ...],
    "green": [0, 4, 11, ...],
    "blue": [0, 6, 10, ...],
    "luminance": [0, 5, 11, ...]
}
```

## Frontend Implementation

### Navigation

- Click image in grid -> enters Edit Mode
- `currentView: 'edit'` state variable
- URL updates to `#edit/{image_id}` for direct linking

### Image Viewer

- CSS `transform` for zoom/pan (hardware accelerated)
- Mouse wheel = zoom in/out (centered on cursor)
- Click+drag = pan when zoomed
- Double-click = toggle Fit/100%
- Keyboard: `+`/`-` zoom, arrows pan, `0` fit, `1` actual size

### Transform Application

Transforms applied client-side via CSS for instant preview:
```css
.edit-image {
    transform: rotate(90deg) scaleX(-1);
}
```

Server applies transforms when serving `/api/images/{id}/full?apply_transforms=true`

### State Management

```javascript
editMode: {
    imageId: null,
    zoom: 'fit',
    panX: 0,
    panY: 0,
    transforms: {
        rotation: 0,
        flip_h: false,
        flip_v: false
    },
    hasChanges: false,
    loading: false
}
```

## XMP Sidecar Export

### File Location

Same directory as source: `<filename>.xmp`

### XMP Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
      <tiff:Orientation>6</tiff:Orientation>
      <dc:subject>
        <rdf:Bag>
          <rdf:li>landscape</rdf:li>
        </rdf:Bag>
      </dc:subject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
```

### Orientation Mapping (EXIF Standard)

| Transform | EXIF Orientation |
|-----------|------------------|
| 0, no flip | 1 |
| 0, flip H | 2 |
| 180 | 3 |
| 180, flip H | 4 |
| 90 CW, flip H | 5 |
| 90 CW | 6 |
| 90 CCW, flip H | 7 |
| 90 CCW | 8 |

## Future Roadmap

### V2 - Light Editing
- Crop tool with aspect ratio presets
- Exposure/brightness slider
- Contrast slider
- Saturation slider
- Real-time preview

### Other Features Noted
- File system organize (move images to user-specified directory structure)
- Image classification (documents, icons, previews, regular images)
- Star rating (1-5)
