# VAM Tools UI Implementation Plan (TDD-Focused)
## Agent-Executable Tasks with Test-Driven Development

---

## Implementation Strategy

Each task follows TDD principles:
1. **RED**: Write tests that fail (define expected behavior)
2. **GREEN**: Implement minimum code to pass tests
3. **REFACTOR**: Clean up implementation
4. **VERIFY**: Manual verification steps

Tasks are designed to be:
- **Independent**: Can be executed in parallel by different agents
- **Testable**: Clear acceptance criteria and verification steps
- **Incremental**: Each task delivers working functionality

---

## Task 1: Thumbnail Generation API

### Objective
Create API endpoint to serve image thumbnails with caching.

### Prerequisites
- PIL/Pillow installed (`pip install Pillow`)
- Images table has `source_path` field
- Catalog images API working (already done)

### Acceptance Criteria
- [ ] Endpoint: `GET /api/catalogs/{catalog_id}/images/{image_id}/thumbnail?size=medium&quality=80`
- [ ] Generates thumbnail on first request
- [ ] Caches thumbnail to disk (`.thumbnails/` directory)
- [ ] Serves from cache on subsequent requests
- [ ] Returns proper content-type headers (`image/jpeg`)
- [ ] Returns 404 if image not found or inaccessible
- [ ] Supports sizes: small (150px), medium (250px), large (350px)
- [ ] Supports quality: 60-95

### Test Cases (Write First)

**File**: `tests/api/test_thumbnails.py`

```python
import pytest
from pathlib import Path

def test_thumbnail_generation_creates_file(client, sample_catalog_with_images):
    """Test that requesting a thumbnail creates a cached file"""
    catalog_id, image_id = sample_catalog_with_images

    response = client.get(
        f"/api/catalogs/{catalog_id}/images/{image_id}/thumbnail?size=medium"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0

    # Verify cache file exists
    cache_path = Path(f".thumbnails/{catalog_id}/{image_id}_medium_80.jpg")
    assert cache_path.exists()

def test_thumbnail_served_from_cache(client, sample_catalog_with_images):
    """Test that second request uses cache"""
    catalog_id, image_id = sample_catalog_with_images

    # First request
    response1 = client.get(
        f"/api/catalogs/{catalog_id}/images/{image_id}/thumbnail?size=medium"
    )

    # Get cache file modified time
    cache_path = Path(f".thumbnails/{catalog_id}/{image_id}_medium_80.jpg")
    mtime1 = cache_path.stat().st_mtime

    # Second request
    response2 = client.get(
        f"/api/catalogs/{catalog_id}/images/{image_id}/thumbnail?size=medium"
    )

    # Cache file should not have been regenerated
    mtime2 = cache_path.stat().st_mtime
    assert mtime1 == mtime2

def test_thumbnail_not_found(client):
    """Test 404 for non-existent image"""
    response = client.get(
        f"/api/catalogs/{uuid.uuid4()}/images/nonexistent/thumbnail"
    )
    assert response.status_code == 404

def test_thumbnail_sizes(client, sample_catalog_with_images):
    """Test different thumbnail sizes"""
    catalog_id, image_id = sample_catalog_with_images

    for size in ['small', 'medium', 'large']:
        response = client.get(
            f"/api/catalogs/{catalog_id}/images/{image_id}/thumbnail?size={size}"
        )
        assert response.status_code == 200
```

### Implementation Steps

1. **Create thumbnail utility function**
   - File: `vam_tools/api/utils/thumbnails.py`
   - Function: `generate_thumbnail(source_path: Path, output_path: Path, size: int, quality: int) -> bool`
   - Use PIL to open, resize, and save image
   - Handle errors gracefully

2. **Add thumbnail endpoint**
   - File: `vam_tools/api/routers/catalogs.py`
   - Add route: `@router.get("/{catalog_id}/images/{image_id}/thumbnail")`
   - Query params: size (default: medium), quality (default: 80)
   - Check cache first
   - Generate if missing
   - Return FileResponse with cache headers

3. **Create cache directory structure**
   - `.thumbnails/{catalog_id}/{image_id}_{size}_{quality}.jpg`
   - Add `.thumbnails/` to `.gitignore`

### Verification Steps
```bash
# Start server
docker compose restart web

# Test endpoint
curl -o test.jpg "http://localhost:8765/api/catalogs/{CATALOG_ID}/images/{IMAGE_ID}/thumbnail?size=medium"

# Verify image
file test.jpg  # Should show: JPEG image data

# Check cache
ls -la .thumbnails/  # Should contain cached thumbnails

# Test performance (should be fast on second request)
time curl "http://localhost:8765/api/catalogs/{CATALOG_ID}/images/{IMAGE_ID}/thumbnail?size=medium" > /dev/null
```

### Success Metrics
- ✅ All tests pass
- ✅ Endpoint returns valid JPEG images
- ✅ Thumbnails cached to disk
- ✅ Second request <50ms (cache hit)
- ✅ First request <500ms (generation)

---

## Task 2: Grid View Component (Frontend)

### Objective
Create a responsive image grid view that displays catalog images with thumbnails.

### Prerequisites
- Task 1 completed (thumbnail API working)
- Catalog images API working (already done)
- Vue 3 app structure exists

### Acceptance Criteria
- [ ] Grid displays image thumbnails in responsive columns
- [ ] Lazy loading (images load as user scrolls)
- [ ] Click to select image (single selection)
- [ ] Ctrl/Cmd+click for multi-selection
- [ ] Selected images show blue border + checkmark
- [ ] Shows loading spinner while fetching
- [ ] Shows image count (e.g., "2,422 images")
- [ ] Thumbnail size adjustable (small/medium/large)
- [ ] Infinite scroll or "Load More" button

### Test Cases (Manual Verification)

**Checklist**: `tests/manual/grid_view_checklist.md`

```markdown
# Grid View Manual Test Checklist

## Display Tests
- [ ] Grid shows when catalog selected and "Browse" tab clicked
- [ ] Images display in even grid (4-6 columns depending on size)
- [ ] Each thumbnail has filename below it
- [ ] Placeholder/spinner shows while thumbnail loading
- [ ] Broken image icon shows if thumbnail fails to load

## Interaction Tests
- [ ] Click image -> image gets blue border
- [ ] Click different image -> previous unselected, new selected
- [ ] Ctrl+click -> multiple images selected
- [ ] Click selected image again -> deselects

## Performance Tests
- [ ] Initial 50 images load in <2 seconds
- [ ] Scrolling smooth with 100+ images loaded
- [ ] Memory usage reasonable (<200MB with 500 images)

## Responsive Tests
- [ ] Narrow window (800px): 2-3 columns
- [ ] Medium window (1200px): 4-5 columns
- [ ] Wide window (1920px): 6-8 columns
```

### Implementation Steps

1. **Add grid view data to Vue app**
   - File: `vam_tools/web/static/app.js`
   - Add to `data()`:
     ```javascript
     gridImages: [],
     gridLoading: false,
     gridPage: 0,
     gridPageSize: 50,
     gridTotalImages: 0,
     gridThumbnailSize: 'medium',
     selectedImages: new Set(),
     ```

2. **Add methods to load images**
   ```javascript
   async loadGridImages() {
       this.gridLoading = true;
       try {
           const response = await axios.get(
               `/api/catalogs/${this.currentCatalog.id}/images`,
               {
                   params: {
                       limit: this.gridPageSize,
                       offset: this.gridPage * this.gridPageSize
                   }
               }
           );

           this.gridImages.push(...response.data.images);
           this.gridTotalImages = response.data.total;
           this.gridPage++;
       } catch (error) {
           console.error('Failed to load images:', error);
           this.addNotification('Failed to load images', 'error');
       } finally {
           this.gridLoading = false;
       }
   },

   toggleImageSelection(imageId, event) {
       if (event.ctrlKey || event.metaKey) {
           // Multi-select
           if (this.selectedImages.has(imageId)) {
               this.selectedImages.delete(imageId);
           } else {
               this.selectedImages.add(imageId);
           }
       } else {
           // Single select
           this.selectedImages.clear();
           this.selectedImages.add(imageId);
       }
   },

   getThumbnailUrl(catalogId, imageId) {
       return `/api/catalogs/${catalogId}/images/${imageId}/thumbnail?size=${this.gridThumbnailSize}`;
   }
   ```

3. **Add HTML for grid view**
   - File: `vam_tools/web/static/index.html`
   - In browse view section:
   ```html
   <div v-if="currentView === 'browse'" class="browse-view">
       <div class="browse-header">
           <h2>{{ currentCatalog ? currentCatalog.name : 'Select a catalog' }}</h2>
           <div class="browse-controls">
               <span class="image-count">{{ gridTotalImages }} images</span>
               <select v-model="gridThumbnailSize" @change="reloadGrid">
                   <option value="small">Small</option>
                   <option value="medium">Medium</option>
                   <option value="large">Large</option>
               </select>
           </div>
       </div>

       <div class="image-grid">
           <div
               v-for="image in gridImages"
               :key="image.id"
               class="grid-item"
               :class="{ selected: selectedImages.has(image.id) }"
               @click="toggleImageSelection(image.id, $event)"
           >
               <div class="thumbnail-wrapper">
                   <img
                       :src="getThumbnailUrl(currentCatalog.id, image.id)"
                       :alt="image.source_path"
                       loading="lazy"
                       @error="handleImageError"
                   />
                   <div v-if="selectedImages.has(image.id)" class="selection-indicator">
                       ✓
                   </div>
               </div>
               <div class="image-info">
                   <span class="filename">{{ getFilename(image.source_path) }}</span>
               </div>
           </div>
       </div>

       <div v-if="gridLoading" class="loading-more">
           Loading...
       </div>

       <button
           v-if="gridImages.length < gridTotalImages && !gridLoading"
           @click="loadGridImages"
           class="load-more-btn"
       >
           Load More ({{ gridTotalImages - gridImages.length }} remaining)
       </button>
   </div>
   ```

4. **Add CSS for grid layout**
   - File: `vam_tools/web/static/styles.css`
   - Add grid styles:
   ```css
   .browse-view {
       padding: 2rem;
   }

   .browse-header {
       display: flex;
       justify-content: space-between;
       align-items: center;
       margin-bottom: 2rem;
   }

   .image-grid {
       display: grid;
       grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
       gap: 1rem;
   }

   .image-grid.size-small {
       grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
   }

   .image-grid.size-large {
       grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
   }

   .grid-item {
       cursor: pointer;
       border: 3px solid transparent;
       border-radius: 4px;
       transition: border-color 0.2s;
   }

   .grid-item:hover {
       border-color: var(--border-color);
   }

   .grid-item.selected {
       border-color: var(--accent-primary);
   }

   .thumbnail-wrapper {
       position: relative;
       aspect-ratio: 4/3;
       background: var(--thumbnail-bg);
       overflow: hidden;
   }

   .thumbnail-wrapper img {
       width: 100%;
       height: 100%;
       object-fit: cover;
   }

   .selection-indicator {
       position: absolute;
       top: 8px;
       right: 8px;
       width: 24px;
       height: 24px;
       background: var(--accent-primary);
       color: white;
       border-radius: 50%;
       display: flex;
       align-items: center;
       justify-content: center;
       font-size: 14px;
   }

   .image-info {
       padding: 0.5rem;
       font-size: 12px;
       color: var(--text-secondary);
       overflow: hidden;
       text-overflow: ellipsis;
       white-space: nowrap;
   }
   ```

### Verification Steps
```bash
# Open browser
open http://localhost:8765

# Manual tests:
1. Select catalog from dropdown
2. Click "Browse" tab
3. Verify grid displays
4. Click image -> should select
5. Ctrl+click another -> both selected
6. Scroll down -> click "Load More"
7. Change thumbnail size -> grid adjusts
```

### Success Metrics
- ✅ Grid displays thumbnails correctly
- ✅ Selection works (single and multi)
- ✅ Load more fetches next batch
- ✅ Performance acceptable (<2s initial load)
- ✅ Responsive (adjusts to window width)

---

## Task 3: Lightbox Viewer

### Objective
Full-screen image viewer activated by double-clicking a grid image.

### Prerequisites
- Task 2 completed (grid view working)
- Full image serving endpoint (can reuse thumbnail API with size=original)

### Acceptance Criteria
- [ ] Double-click grid image opens lightbox
- [ ] Lightbox displays full-size image
- [ ] Escape key closes lightbox
- [ ] Left/Right arrow keys navigate images
- [ ] Close button (X) in top-right
- [ ] Shows current position (e.g., "15 of 2,422")
- [ ] Shows filename and basic metadata
- [ ] Click outside image closes lightbox
- [ ] Smooth transitions between images

### Test Cases (Manual Verification)

**Checklist**: `tests/manual/lightbox_checklist.md`

```markdown
# Lightbox Manual Test Checklist

## Opening/Closing
- [ ] Double-click grid image -> lightbox opens
- [ ] ESC key -> lightbox closes
- [ ] Click X button -> lightbox closes
- [ ] Click outside image (on dark overlay) -> lightbox closes

## Navigation
- [ ] Right arrow -> next image
- [ ] Left arrow -> previous image
- [ ] At last image, right arrow disabled/wraps to first
- [ ] At first image, left arrow disabled/wraps to last

## Display
- [ ] Image centered and scaled to fit screen
- [ ] Counter shows "N of TOTAL"
- [ ] Filename displayed
- [ ] Metadata panel toggleable (spacebar)

## Performance
- [ ] Image loads within 1 second
- [ ] Navigation between images smooth (<300ms)
- [ ] No memory leaks (can navigate 50+ images)
```

### Implementation Steps

1. **Add lightbox data to Vue app**
   ```javascript
   data() {
       return {
           // ... existing data
           lightboxOpen: false,
           lightboxCurrentIndex: 0,
           lightboxShowMeta: false,
       }
   }
   ```

2. **Add lightbox methods**
   ```javascript
   openLightbox(imageId) {
       const index = this.gridImages.findIndex(img => img.id === imageId);
       if (index === -1) return;

       this.lightboxCurrentIndex = index;
       this.lightboxOpen = true;

       // Keyboard event listeners
       document.addEventListener('keydown', this.handleLightboxKeyboard);
   },

   closeLightbox() {
       this.lightboxOpen = false;
       document.removeEventListener('keydown', this.handleLightboxKeyboard);
   },

   nextImage() {
       if (this.lightboxCurrentIndex < this.gridImages.length - 1) {
           this.lightboxCurrentIndex++;
       }
   },

   prevImage() {
       if (this.lightboxCurrentIndex > 0) {
           this.lightboxCurrentIndex--;
       }
   },

   handleLightboxKeyboard(event) {
       switch(event.key) {
           case 'Escape':
               this.closeLightbox();
               break;
           case 'ArrowLeft':
               this.prevImage();
               break;
           case 'ArrowRight':
               this.nextImage();
               break;
           case ' ':
               this.lightboxShowMeta = !this.lightboxShowMeta;
               event.preventDefault();
               break;
       }
   }
   ```

3. **Update grid item to open lightbox**
   ```html
   <div
       v-for="image in gridImages"
       :key="image.id"
       class="grid-item"
       :class="{ selected: selectedImages.has(image.id) }"
       @click="toggleImageSelection(image.id, $event)"
       @dblclick="openLightbox(image.id)"
   >
   ```

4. **Add lightbox HTML**
   ```html
   <!-- Lightbox Modal -->
   <div v-if="lightboxOpen" class="lightbox-overlay" @click.self="closeLightbox">
       <div class="lightbox-container">
           <!-- Header -->
           <div class="lightbox-header">
               <span class="lightbox-counter">
                   {{ lightboxCurrentIndex + 1 }} of {{ gridImages.length }}
               </span>
               <button @click="closeLightbox" class="lightbox-close">×</button>
           </div>

           <!-- Navigation Buttons -->
           <button
               v-if="lightboxCurrentIndex > 0"
               @click="prevImage"
               class="lightbox-nav lightbox-nav-prev"
           >
               ‹
           </button>

           <button
               v-if="lightboxCurrentIndex < gridImages.length - 1"
               @click="nextImage"
               class="lightbox-nav lightbox-nav-next"
           >
               ›
           </button>

           <!-- Main Image -->
           <div class="lightbox-image-container">
               <img
                   :src="getThumbnailUrl(currentCatalog.id, gridImages[lightboxCurrentIndex].id)"
                   :alt="gridImages[lightboxCurrentIndex].source_path"
                   class="lightbox-image"
               />
           </div>

           <!-- Metadata Footer -->
           <div v-if="lightboxShowMeta" class="lightbox-metadata">
               <div class="meta-item">
                   <strong>File:</strong>
                   {{ getFilename(gridImages[lightboxCurrentIndex].source_path) }}
               </div>
               <div class="meta-item">
                   <strong>Size:</strong>
                   {{ formatFileSize(gridImages[lightboxCurrentIndex].size_bytes) }}
               </div>
               <div class="meta-item">
                   <strong>Date:</strong>
                   {{ formatDate(gridImages[lightboxCurrentIndex].dates.selected_date) }}
               </div>
           </div>
       </div>
   </div>
   ```

5. **Add lightbox CSS**
   ```css
   .lightbox-overlay {
       position: fixed;
       top: 0;
       left: 0;
       right: 0;
       bottom: 0;
       background: rgba(0, 0, 0, 0.95);
       z-index: 9999;
       display: flex;
       align-items: center;
       justify-content: center;
   }

   .lightbox-container {
       position: relative;
       width: 100%;
       height: 100%;
       display: flex;
       align-items: center;
       justify-content: center;
   }

   .lightbox-header {
       position: absolute;
       top: 0;
       left: 0;
       right: 0;
       padding: 1rem;
       display: flex;
       justify-content: space-between;
       align-items: center;
       background: linear-gradient(to bottom, rgba(0,0,0,0.7), transparent);
       z-index: 10;
   }

   .lightbox-counter {
       color: white;
       font-size: 14px;
   }

   .lightbox-close {
       background: none;
       border: none;
       color: white;
       font-size: 48px;
       cursor: pointer;
       padding: 0;
       width: 48px;
       height: 48px;
       line-height: 1;
   }

   .lightbox-close:hover {
       color: var(--accent-primary);
   }

   .lightbox-nav {
       position: absolute;
       top: 50%;
       transform: translateY(-50%);
       background: rgba(0, 0, 0, 0.5);
       border: none;
       color: white;
       font-size: 48px;
       cursor: pointer;
       padding: 2rem 1rem;
       z-index: 10;
   }

   .lightbox-nav:hover {
       background: rgba(0, 0, 0, 0.8);
   }

   .lightbox-nav-prev {
       left: 0;
   }

   .lightbox-nav-next {
       right: 0;
   }

   .lightbox-image-container {
       max-width: 90%;
       max-height: 90vh;
       display: flex;
       align-items: center;
       justify-content: center;
   }

   .lightbox-image {
       max-width: 100%;
       max-height: 90vh;
       object-fit: contain;
   }

   .lightbox-metadata {
       position: absolute;
       bottom: 0;
       left: 0;
       right: 0;
       padding: 1rem;
       background: rgba(0, 0, 0, 0.8);
       color: white;
       display: flex;
       gap: 2rem;
       font-size: 13px;
   }
   ```

### Verification Steps
```bash
# Open browser
open http://localhost:8765

# Manual tests:
1. Double-click any image in grid
2. Verify lightbox opens with large image
3. Press right arrow -> next image
4. Press left arrow -> previous image
5. Press ESC -> lightbox closes
6. Press space -> metadata toggles
```

### Success Metrics
- ✅ Lightbox opens on double-click
- ✅ Keyboard navigation works
- ✅ Image displays full-size
- ✅ Smooth transitions (<300ms)
- ✅ No memory leaks

---

## Task 4: Dark Theme Polish

### Objective
Apply professional dark theme colors throughout the application.

### Prerequisites
- Grid view working (Task 2)
- Lightbox working (Task 3)

### Acceptance Criteria
- [ ] All text readable on dark backgrounds
- [ ] Consistent color scheme across all views
- [ ] Proper contrast ratios (WCAG AA minimum)
- [ ] Interactive elements have hover/active states
- [ ] No bright white flashes on page load
- [ ] Icons/buttons visible and clear

### Implementation Steps

1. **Add CSS variables**
   - File: `vam_tools/web/static/styles.css`
   - Add to top of file:
   ```css
   :root {
       /* Primary Colors */
       --bg-primary: #1e1e1e;
       --bg-secondary: #2d2d2d;
       --bg-tertiary: #3a3a3a;
       --bg-elevated: #404040;

       /* Text Colors */
       --text-primary: #e0e0e0;
       --text-secondary: #b0b0b0;
       --text-muted: #808080;

       /* Accent Colors */
       --accent-primary: #4a9eff;
       --accent-success: #6abf40;
       --accent-warning: #ff9500;
       --accent-danger: #ff3b30;

       /* UI Elements */
       --border-color: #505050;
       --selection-bg: #0066cc40;
       --thumbnail-bg: #252525;
   }

   body {
       background: var(--bg-primary);
       color: var(--text-primary);
   }
   ```

2. **Update component styles to use variables**
   - Replace hardcoded colors with CSS variables
   - Ensure all backgrounds use dark colors
   - Ensure all text uses light colors

3. **Test color contrast**
   - Use browser dev tools accessibility checker
   - Verify all text meets WCAG AA standards (4.5:1 ratio)

### Verification Steps
```bash
# Open browser
open http://localhost:8765

# Visual inspection:
1. Check all text is readable
2. Check hover states are visible
3. Check buttons have clear states
4. Check no bright white backgrounds
5. Use browser accessibility checker
```

### Success Metrics
- ✅ Consistent dark theme across all views
- ✅ All text readable (WCAG AA contrast)
- ✅ Professional appearance
- ✅ No accessibility warnings

---

## Execution Order

**Parallel Execution** (agents can work simultaneously):
- Task 1 (Backend) + Task 4 (CSS)

**Sequential Execution**:
1. Task 1: Thumbnail API
2. Task 2: Grid View
3. Task 3: Lightbox
4. Task 4: Theme Polish

**Estimated Timeline**:
- Task 1: 2-3 hours
- Task 2: 3-4 hours
- Task 3: 2-3 hours
- Task 4: 1-2 hours

**Total: 8-12 hours** for full MVP

---

## Definition of Done

**MVP Complete When**:
- ✅ All acceptance criteria met for Tasks 1-4
- ✅ All tests passing (automated + manual checklists)
- ✅ User can browse catalog images in dark-themed grid
- ✅ User can view full-size images in lightbox
- ✅ Thumbnails load quickly (<500ms)
- ✅ UI looks professional and polished
- ✅ No console errors
- ✅ Docker containers restart without issues

**Ready for Next Iteration**:
- Rating system
- Advanced filtering
- Collections
- Metadata editing
