# Catalog Browsing Optimization Plan

## Executive Summary

Transform the catalog viewer into a high-performance, tab-based interface optimized for browsing catalogs with 100K+ files.

**Key Goals:**
- Reduce initial load time from seconds to <500ms
- Support seamless browsing of 100K+ image catalogs
- Organize content by type/issue/duplicate for better UX
- Maintain responsive UI even under heavy load

---

## Current Bottlenecks (Analysis)

### 1. **Single View Overload**
- All content types mixed in one view (images, videos, issues)
- Filtering requires full page reload
- No content-specific optimizations

### 2. **Inefficient Loading**
- Loads 50-200 items per request (configurable pageSize)
- Each item includes full metadata in response
- No caching between filter changes
- Thumbnail generation on-demand (can timeout on RAW files)

### 3. **DOM Rendering Performance**
- All loaded images rendered in DOM simultaneously
- No virtual scrolling (1000+ DOM nodes with infinite scroll)
- Heavy re-renders on filter/sort changes

### 4. **API Response Size**
- Returns full ImageSummary objects (~500 bytes each)
- 200 items = ~100KB JSON per request
- Includes metadata users rarely view in grid

---

## Proposed Solution: Multi-Faceted Optimization

### Phase 1: Tab-Based Navigation Architecture

#### **Tab Structure**

```
┌─────────────────────────────────────────────────┐
│ [Overview] [All Files] [Duplicates] [Issues]   │
├─────────────────────────────────────────────────┤
│                                                 │
│   Tab-specific content with optimized loading  │
│                                                 │
└─────────────────────────────────────────────────┘
```

#### **Tabs:**

1. **Overview Tab** (Dashboard)
   - Statistics cards (already implemented)
   - Performance charts (already implemented)
   - Quick insights (recent files, top issues, duplicate count)
   - No heavy image loading

2. **All Files Tab** (Current grid view)
   - Combined images + videos
   - Sub-filters: [All] [Images] [Videos] [RAW] [JPEG]
   - Virtual scrolling implementation
   - Lazy thumbnail loading

3. **Duplicates Tab**
   - Group-based view (one card per duplicate group)
   - Shows best copy highlighted
   - Side-by-side comparison
   - Batch resolution UI
   - Only loads when tab activated

4. **Issues Tab**
   - Organized by issue type:
     - No Date Detected
     - Suspicious Dates
     - Corrupt Files
     - Missing Metadata
   - Collapsible issue type sections
   - Fix/ignore actions per file

---

### Phase 2: Virtual Scrolling Implementation

#### **Current vs Proposed:**

| Metric | Current | Proposed Virtual Scroll |
|--------|---------|-------------------------|
| DOM nodes (1000 images) | 1000+ | ~20-30 |
| Memory usage | High | Low |
| Scroll performance | Degraded | Smooth 60fps |
| Initial render | Slow | Fast |

#### **Implementation Strategy:**

Use **Intersection Observer API** + **Virtual Scrolling**:

```javascript
// Pseudo-code
class VirtualScroller {
  constructor(container, itemHeight, buffer = 5) {
    this.visibleStart = 0;
    this.visibleEnd = 0;
    this.buffer = buffer; // Items to render outside viewport
  }

  render(allItems) {
    // Only render visible items + buffer
    const start = Math.max(0, this.visibleStart - this.buffer);
    const end = Math.min(allItems.length, this.visibleEnd + this.buffer);
    return allItems.slice(start, end);
  }
}
```

**Benefits:**
- Render only 20-30 images at a time (viewport + buffer)
- Recycle DOM nodes as user scrolls
- Maintains scroll position and feel

**Libraries to Consider:**
- `vue-virtual-scroller` - Vue 3 compatible, battle-tested
- `vue-virtual-scroll-grid` - Grid-specific implementation
- Custom implementation (more control)

---

### Phase 3: API Optimizations

#### **3.1 Lightweight Response Format**

**Current ImageSummary:**
```json
{
  "id": "abc123...",
  "file_path": "/full/path/to/image.jpg",
  "media_type": "image",
  "file_extension": ".jpg",
  "file_size_bytes": 12345678,
  "date_taken": "2024-01-15T10:30:00",
  "best_date_source": "exif",
  "camera_make": "Canon",
  "camera_model": "EOS R5",
  "width": 8192,
  "height": 5464,
  "quality_score": 95.5,
  "has_issues": false,
  "is_duplicate": false
}
```

**Proposed GridItem (for grid views):**
```json
{
  "id": "abc123",
  "thumb": "/api/images/abc123/thumbnail",
  "preview": "/api/images/abc123/preview",
  "type": "image",
  "ext": ".jpg",
  "size": 12345678,
  "date": "2024-01-15",
  "score": 95.5,
  "flags": ["dup", "issue"]  // Only if applicable
}
```

**Savings:** ~500 bytes → ~150 bytes per item (70% reduction)

#### **3.2 New Endpoint: `/api/images/grid`**

```python
@app.get("/api/images/grid")
async def get_images_grid(
    skip: int = 0,
    limit: int = 100,
    tab: str = "all",  # all, duplicates, issues
    filter_type: Optional[str] = None,
    issue_type: Optional[str] = None,
    sort_by: str = "date"
):
    """Optimized endpoint for grid view with minimal response size."""
    # Returns GridItem[] instead of ImageSummary[]
```

#### **3.3 Prefetch/Caching Strategy**

**Browser-Side:**
- Cache API responses in memory (Vue reactive store)
- Prefetch next page when user reaches 80% of current view
- Cache thumbnails in browser storage (IndexedDB)

**Server-Side:**
- Redis cache for frequently accessed metadata
- Pre-generate thumbnails during analysis (already planned in TODO)
- ETag support for conditional requests

---

### Phase 4: Thumbnail Optimization

#### **Current Issues:**
- RAW files can timeout (20-30 seconds for large files)
- Thumbnails generated on-demand
- No progressive loading

#### **Proposed Solutions:**

1. **Background Thumbnail Generation**
   ```python
   # During analysis phase
   async def generate_thumbnails(image_path):
       # 3 sizes: grid (200px), preview (800px), full
       await generate_thumbnail(image_path, size="grid")
       await generate_thumbnail(image_path, size="preview")
   ```

2. **Fallback Strategy**
   ```
   1. Try embedded JPEG from RAW (fast, instant)
   2. Try cached thumbnail from disk (fast, <100ms)
   3. Generate on-demand with timeout (slow, 5-20s)
   4. Show placeholder if timeout
   ```

3. **Progressive Loading**
   ```javascript
   // Load low-quality placeholder first (base64 inline)
   // Then swap to full thumbnail when ready
   <img :src="image.placeholder"
        @load="loadFullThumbnail(image.id)" />
   ```

---

### Phase 5: Duplicate Group Optimization

#### **Current Approach:**
- Duplicates mixed in main grid
- No group-based view
- Difficult to review and resolve

#### **Proposed Duplicate Tab:**

```
┌─────────────────────────────────────┐
│ Group 1 (4 duplicates) - 45.2 MB   │
│ ┌────┬────┬────┬────┐              │
│ │ ⭐ │    │    │    │  [Keep] [Delete Others] │
│ └────┴────┴────┴────┘              │
│ Best: IMG_001.ARW (8K, RAW, EXIF)  │
├─────────────────────────────────────┤
│ Group 2 (3 duplicates) - 12.1 MB   │
│ ┌────┬────┬────┐                   │
│ │ ⭐ │    │    │  [Keep] [Delete Others] │
│ └────┴────┴────┘                   │
└─────────────────────────────────────┘
```

**Features:**
- Load groups on-demand (lazy load)
- Best copy pre-selected based on quality_score
- Batch operations (keep best, delete all duplicates)
- Side-by-side comparison modal
- Show total space savings

**API Endpoint:**
```python
@app.get("/api/duplicates/groups")
async def get_duplicate_groups(
    skip: int = 0,
    limit: int = 20,  # Groups, not individual files
    min_size_mb: Optional[float] = None,
    similarity_threshold: Optional[int] = None
):
    """Returns duplicate groups with best copy identified."""
```

---

### Phase 6: Issues Tab Organization

#### **Proposed Structure:**

```
Issues Tab
├── No Date Detected (142 files) ▼
│   ├── IMG_0001.jpg (2.3 MB)
│   ├── IMG_0002.jpg (2.1 MB)
│   └── ... [Load More]
│
├── Suspicious Dates (23 files) ▼
│   ├── _DSC1234.NEF (45 MB) - Date: 1970-01-01
│   └── ... [Load More]
│
├── Corrupt/Unreadable (5 files) ▼
│   └── ... [Load More]
│
└── Missing Metadata (89 files) ▼
    └── ... [Load More]
```

**Features:**
- Collapsible sections per issue type
- Counts in section headers
- Lazy load files within each section
- Bulk actions (ignore all, fix all)
- Jump to file in main view

---

## Performance Targets

| Metric | Current | Target | Strategy |
|--------|---------|--------|----------|
| Initial page load | 2-5s | <500ms | Lazy tabs, virtual scroll |
| Time to first image | 1-3s | <200ms | Lightweight API, prefetch |
| Scroll FPS (1000 images) | 20-30 fps | 60 fps | Virtual scrolling |
| Memory usage (10K images) | ~500 MB | <100 MB | Virtual scrolling, GridItem |
| API response (200 items) | ~100 KB | ~30 KB | GridItem format |
| Thumbnail load time (RAW) | 5-20s | <1s | Pre-generation + fallback |

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Add tab navigation UI
- [ ] Implement tab state management
- [ ] Create Overview tab (dashboard)
- [ ] Move current grid to "All Files" tab

### Phase 2: Virtual Scrolling (Week 1-2)
- [ ] Install/implement virtual scroller
- [ ] Integrate with existing pagination
- [ ] Test with large datasets (10K+ images)
- [ ] Performance benchmarking

### Phase 3: API Optimization (Week 2)
- [ ] Create GridItem model
- [ ] Implement `/api/images/grid` endpoint
- [ ] Add caching layer (browser + server)
- [ ] Implement prefetch strategy

### Phase 4: Duplicates Tab (Week 2-3)
- [ ] Design duplicate group UI
- [ ] Implement `/api/duplicates/groups` endpoint
- [ ] Add batch resolution actions
- [ ] Side-by-side comparison modal

### Phase 5: Issues Tab (Week 3)
- [ ] Design collapsible issue sections
- [ ] Implement issue type filtering
- [ ] Add bulk actions
- [ ] Lazy loading per section

### Phase 6: Thumbnails (Week 3-4)
- [ ] Background thumbnail generation during analysis
- [ ] Fallback strategy implementation
- [ ] Progressive loading (placeholder → full)
- [ ] Cache thumbnails in IndexedDB

### Phase 7: Polish & Testing (Week 4)
- [ ] Loading states and skeletons
- [ ] Error handling
- [ ] Mobile responsiveness
- [ ] Load testing with 100K+ catalogs
- [ ] Documentation

---

## Technical Considerations

### Libraries/Dependencies

1. **Virtual Scrolling:**
   - `vue-virtual-scroller` (23KB gzipped)
   - Alternative: `vue-virtual-scroll-grid`

2. **State Management:**
   - Consider Pinia for complex tab state
   - Or continue with Vue 3 reactive data

3. **Caching:**
   - `localforage` for IndexedDB (thumbnails)
   - Native `Map` for in-memory cache

### Backward Compatibility

- Keep existing `/api/images` endpoint for v1 clients
- Add `/api/images/grid` as new optimized endpoint
- Feature flag for tab-based UI (gradual rollout)

### Testing Strategy

- Load testing with catalogs of varying sizes:
  - Small: 1K images
  - Medium: 10K images
  - Large: 100K images
  - Extreme: 500K+ images
- Performance profiling (Chrome DevTools)
- Memory leak detection
- Mobile device testing

---

## Alternative Approaches Considered

### 1. **Server-Side Pagination Only**
❌ **Rejected** - Still requires full page loads, poor UX

### 2. **Infinite Scroll Without Virtual Scrolling**
❌ **Rejected** - DOM bloat, memory issues at scale

### 3. **Separate Pages for Each View**
❌ **Rejected** - Loses context, requires navigation

### 4. **WebGL/Canvas-Based Rendering**
❌ **Rejected** - Overkill, accessibility concerns, complexity

---

## Success Metrics

1. **Performance:**
   - 60 FPS scroll with 10K+ images loaded
   - <500ms initial page load
   - <200ms tab switching

2. **User Experience:**
   - Users can find duplicates in <30 seconds
   - Issues organized and actionable
   - No perception of "loading" during normal use

3. **Scalability:**
   - Smooth performance up to 100K images
   - Reasonable performance up to 500K images
   - Memory usage stays under 200 MB

---

## Questions for Review

1. **Priority Order:** Which tab should we implement first after Overview?
   - All Files (refinement)
   - Duplicates (new functionality)
   - Issues (new functionality)

2. **Virtual Scrolling Library:** Prefer existing library or custom implementation?
   - `vue-virtual-scroller` (proven, maintained)
   - Custom (more control, learning curve)

3. **Thumbnail Strategy:** Should we:
   - Generate all thumbnails during analysis (slower analysis, faster browsing)
   - Generate on-demand with cache (faster analysis, slower first browse)
   - Hybrid approach (generate grid size during analysis, preview on-demand)

4. **API Changes:** Breaking changes acceptable or maintain v1 compatibility?
   - New `/api/v2/` endpoints (cleaner, versioned)
   - Extend existing endpoints (backward compatible)

5. **UI Framework:** Stay with vanilla Vue 3 or add component library?
   - Current approach (custom components, full control)
   - Add Vuetify/PrimeVue (faster dev, larger bundle)

---

## Next Steps

After review and approval:
1. Create detailed technical spec for approved approach
2. Set up development branch
3. Implement Phase 1 (Foundation + Tabs)
4. Benchmark and iterate

---

**Document Version:** 1.0
**Date:** 2025-10-28
**Status:** DRAFT - Awaiting Review
