# Lumina UI Redesign Plan
## Darktable/Lightroom-Inspired Interface

### Overview
Transform Lumina into a professional photo management application with a dark, photographer-friendly interface inspired by Darktable and Adobe Lightroom.

---

## Phase 1: Dark Theme & Base Styling

### 1.1 Color Palette
```css
/* Primary Colors */
--bg-primary: #1e1e1e;        /* Main background */
--bg-secondary: #2d2d2d;      /* Panels, cards */
--bg-tertiary: #3a3a3a;       /* Hover states */
--bg-elevated: #404040;       /* Modals, dropdowns */

/* Text Colors */
--text-primary: #e0e0e0;      /* Primary text */
--text-secondary: #b0b0b0;    /* Secondary text */
--text-muted: #808080;        /* Muted text */

/* Accent Colors */
--accent-primary: #4a9eff;    /* Primary actions */
--accent-success: #6abf40;    /* Success states */
--accent-warning: #ff9500;    /* Warnings */
--accent-danger: #ff3b30;     /* Errors, delete */

/* UI Elements */
--border-color: #505050;      /* Borders, dividers */
--selection-bg: #0066cc40;    /* Selection overlay */
--thumbnail-bg: #252525;      /* Image background */
```

### 1.2 Typography
- **Font Family**: System fonts (SF Pro, Segoe UI, Roboto)
- **Sizes**: 11px (small), 13px (body), 15px (headers), 18px+ (titles)
- **Weights**: 400 (normal), 500 (medium), 600 (semibold)

### 1.3 Layout Structure
```
┌─────────────────────────────────────────────────────┐
│  Top Bar (catalog selector, search, settings)       │
├──────┬──────────────────────────────────────────────┤
│      │                                              │
│ Side │          Main Content Area                   │
│ Nav  │          (Grid/Table View)                   │
│      │                                              │
│      │                                              │
├──────┴──────────────────────────────────────────────┤
│  Bottom Bar (status, info, view controls)           │
└─────────────────────────────────────────────────────┘
```

---

## Phase 2: Navigation & Layout

### 2.1 Sidebar Navigation
**Location**: Left side, collapsible
**Width**: 240px normal, 60px collapsed
**Sections**:
- Library (Catalogs, Collections)
- Folders (Directory tree)
- Jobs (Active/Recent)
- Review Queue
- Tags/Keywords

**Design**:
- Icon + text (or icon-only when collapsed)
- Active state highlighting
- Badge counts for pending items

### 2.2 Top Bar
**Components**:
- Catalog selector (dropdown)
- Global search
- View mode toggle (Grid/Table/Map)
- Sort/Filter controls
- User menu

### 2.3 Bottom Info Bar
**Components**:
- Selected images count
- Total images in view
- Zoom slider (for grid view)
- Quick metadata display
- Job status indicator

---

## Phase 3: Browse View - Image Grid

### 3.1 Grid Layout
**Features**:
- Responsive grid (adjust columns based on window width)
- Thumbnail sizes: Small (150px), Medium (250px), Large (350px)
- Infinite scroll or pagination
- Selection mode (single/multiple)

**Thumbnail Design**:
```
┌────────────────┐
│                │
│   IMAGE        │  ← Actual image thumbnail
│                │
├────────────────┤
│ ⭐ filename.jpg│  ← Rating + filename
│ 📅 2024-01-15  │  ← Date info
└────────────────┘
```

### 3.2 Thumbnail Overlay States
- **Hover**: Show metadata overlay (camera, lens, settings)
- **Selected**: Blue border + checkmark
- **Loading**: Skeleton/spinner
- **Error**: Error icon + message

### 3.3 Quick Actions
- Click: Select image
- Double-click: Open lightbox viewer
- Right-click: Context menu (rate, tag, delete, etc.)
- Shift+click: Multi-select range
- Ctrl/Cmd+click: Toggle selection

---

## Phase 4: Lightbox Viewer

### 4.1 Full-Screen Image Viewer
**Layout**:
```
┌─────────────────────────────────────────────────┐
│  [<] Image 15 of 2,422              [✕]        │  ← Header
├─────────────────────────────────────────────────┤
│                                                 │
│                                                 │
│              FULL IMAGE DISPLAY                 │
│                                                 │
│                                                 │
├─────────────────────────────────────────────────┤
│  Metadata Panel (collapsible)                   │  ← Footer
└─────────────────────────────────────────────────┘
```

**Controls**:
- Left/Right arrows: Navigate images
- Escape: Close viewer
- Space: Toggle metadata panel
- +/-: Zoom in/out
- 1-5 keys: Quick rating

### 4.2 Metadata Panel
**Sections**:
- File Info (name, size, format, dimensions)
- Camera Settings (ISO, aperture, shutter, focal length)
- Date/Time (with confidence indicator)
- GPS Location (if available, show map)
- Tags/Keywords
- Duplicate info (if part of group)

---

## Phase 5: Image Serving & Caching

### 5.1 Thumbnail API Endpoint
**New Backend Route**: `/api/catalogs/{id}/images/{image_id}/thumbnail`

**Parameters**:
- `size`: small (150px), medium (250px), large (350px)
- `quality`: low (60), medium (80), high (95)

**Implementation**:
- Generate thumbnails on-the-fly using PIL/Pillow
- Cache generated thumbnails to disk
- Serve with proper cache headers

**Example**:
```python
@router.get("/{catalog_id}/images/{image_id}/thumbnail")
def get_image_thumbnail(
    catalog_id: uuid.UUID,
    image_id: str,
    size: str = "medium",
    quality: int = 80
):
    # Check cache
    # Generate if missing
    # Return image with cache headers
```

### 5.2 Full Image Endpoint
**Route**: `/api/catalogs/{id}/images/{image_id}/full`

**Features**:
- Serve original file (if accessible)
- Or serve high-quality preview
- Support byte-range requests for large files

---

## Phase 6: Enhanced Features

### 6.1 Filtering & Sorting
**Filter Options**:
- Date range
- Camera/Lens
- Rating (1-5 stars)
- File type (image/video)
- Has GPS data
- Needs review
- In duplicate group

**Sort Options**:
- Date (newest/oldest)
- Filename
- Rating
- File size
- Camera model

### 6.2 Rating System
- 0-5 stars
- Keyboard shortcuts (0-5 keys)
- Color coding (unrated, 1★ red → 5★ gold)
- Filter by minimum rating

### 6.3 Collections
- Smart collections (saved filters)
- Manual collections (drag-and-drop)
- Collection colors/icons

### 6.4 Keyboard Shortcuts
```
Navigation:
- ←/→     Previous/Next image
- ↑/↓     Scroll grid
- Space   Toggle lightbox
- Esc     Close lightbox/dialog

Selection:
- Ctrl+A  Select all
- Ctrl+D  Deselect all
- Shift+Click  Range select

Rating:
- 0-5     Set rating
- [/]     Decrease/Increase rating

View:
- G       Grid view
- T       Table view
- +/-     Zoom in/out
```

---

## Phase 7: Jobs Dashboard Improvements

### 7.1 Better Job Display
**Active Jobs**:
- Large progress circle
- Real-time status updates
- Estimated time remaining
- Ability to pause/cancel

**Completed Jobs**:
- Summary cards with results
- Click to view details
- "Run again" button
- Export results

### 7.2 Job History
- Searchable job history
- Filter by type/status/date
- Detailed logs available
- Performance metrics

---

## Implementation Priorities

### Must-Have (MVP)
1. ✅ Dark theme CSS
2. ✅ Thumbnail API endpoint
3. ✅ Grid view with image thumbnails
4. ✅ Basic lightbox viewer
5. ✅ Navigation structure

### Should-Have (V2)
6. ⬜ Advanced filtering/sorting
7. ⬜ Rating system
8. ⬜ Collections
9. ⬜ Keyboard shortcuts
10. ⬜ Metadata editing

### Nice-to-Have (V3)
11. ⬜ Map view (GPS)
12. ⬜ Timeline view
13. ⬜ Batch operations
14. ⬜ Export presets
15. ⬜ Compare mode (side-by-side)

---

## Technical Stack

**Frontend**:
- Vue 3 (already in use)
- CSS Variables for theming
- CSS Grid/Flexbox for layouts
- Intersection Observer for lazy loading

**Backend**:
- FastAPI (already in use)
- PIL/Pillow for thumbnail generation
- File system caching for thumbnails

**No Additional Dependencies Required** - use what's already there!

---

## Next Steps

1. Create dark theme CSS file (`dark-theme.css`)
2. Update HTML structure for new layout
3. Implement thumbnail generation endpoint
4. Build grid view component
5. Add lightbox viewer
6. Wire up keyboard shortcuts
7. Test with real catalog data

---

## Estimated Timeline

- **Phase 1-2** (Theme & Layout): 2-3 hours
- **Phase 3** (Grid View): 3-4 hours
- **Phase 4** (Lightbox): 2-3 hours
- **Phase 5** (Image Serving): 2-3 hours
- **Phase 6** (Enhanced Features): 4-6 hours
- **Phase 7** (Jobs Dashboard): 2-3 hours

**Total**: 15-22 hours for full implementation

---

## Design References

**Darktable**:
- Dark interface with subtle highlights
- Filmstrip at bottom
- Metadata panel on right
- Modular panel system

**Lightroom**:
- Grid/Loupe/Compare/Survey views
- Histogram display
- Quick develop controls
- Star rating system

**Key Principles**:
- Dark UI reduces eye strain
- High contrast for images
- Keyboard-first workflow
- Fast, responsive interaction
- Clean, minimal chrome
