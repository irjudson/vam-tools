# Unify Burst and Duplicate Views

**Date:** 2025-12-31
**Status:** Approved
**Author:** Claude

## Overview

Replace the burst view's two-panel layout with the duplicate view's card-grid + modal pattern to provide visual consistency and code reuse across the application.

## Problem Statement

The current burst view uses a unique two-panel layout (sidebar list + main content viewer) that differs significantly from the duplicate view's card-grid approach. This creates:

1. **Visual inconsistency** - Users experience different interaction patterns for similar group-viewing tasks
2. **Code duplication** - Separate layouts, styles, and patterns for similar functionality
3. **Navigation inefficiency** - Left panel takes valuable screen space

The duplicate view's card-grid + modal pattern is preferred and should be extended to bursts.

## Design Principles

### DRY (Don't Repeat Yourself)

Create a unified group viewer pattern that both duplicates and bursts reuse:

- Shared CSS classes (`.group-cards-grid`, `.group-card`, etc.)
- Same modal structure
- Configuration-driven differences (badges, actions, member display)

### Progressive Disclosure

- **Overview level:** Grid of burst cards showing thumbnails and key metadata
- **Detail level:** Modal with large preview, filmstrip, and actions

## Architecture

### Current State (Bursts)

```
Bursts View
├── Left Panel
│   ├── Filters
│   └── Burst Groups List (scrollable)
└── Main Content (Two-Panel)
    └── Burst Viewer (always visible)
        ├── Header (metadata)
        ├── Large Preview
        └── Filmstrip
```

### New State (Bursts)

```
Bursts View
├── Stats Toolbar (unchanged)
├── Filters Bar (inline, moved from left panel)
├── Burst Cards Grid
│   └── Burst Card (4 thumbnails + metadata)
└── Burst Detail Modal (on click)
    ├── Modal Header
    ├── Large Preview (selected image)
    ├── Filmstrip (all burst images)
    └── Actions (Set as Best, View Full Size)
```

### Shared Components

Both views use identical structure:

```html
view-container
├── stats/toolbar
├── filter-bar (inline)
├── group-cards-grid
│   └── group-card
│       ├── group-header (badges, counts)
│       ├── group-thumbnails (4 preview images)
│       └── group-footer (metadata)
└── modal (detail view)
    ├── modal-header
    ├── modal-content
    │   ├── detail-header (metadata)
    │   ├── preview/members section
    │   └── actions
    └── modal-footer
```

## Implementation Plan

### 1. HTML Changes (index.html)

#### Remove
- Lines 263-297: Left panel burst groups section
- Lines 927-997: Two-panel burst layout (`bursts-two-panel`, `bursts-content-panel`)

#### Add
- Inline filter bar (moved from left panel)
- Burst cards grid using `.group-cards-grid`
- Burst detail modal using duplicate modal pattern

#### Preserve
- Lines 895-907: Stats toolbar
- Burst-specific features: filmstrip, "Set as Best", quality scores, sequence numbers

### 2. CSS Changes (styles.css)

#### Rename for Reuse
```css
/* Before */
.duplicates-grid { ... }
.duplicate-group-card { ... }

/* After */
.group-cards-grid { ... }  /* Used by both duplicates and bursts */
.group-card { ... }
```

#### Add Burst-Specific Styles
```css
.burst-preview-section { ... }
.burst-large-preview { ... }
.burst-filmstrip { ... }
.filmstrip-item { ... }
```

### 3. JavaScript Changes (app.js)

#### Data Structure
```javascript
// Keep existing burst data structure
bursts: [],
selectedBurst: null,
selectedBurstImage: null,
burstImages: {},  // Cache: { burstId: [images...] }
```

#### New Methods
```javascript
// Close modal
closeSelectedBurst() {
    this.selectedBurst = null;
    this.selectedBurstImage = null;
}

// Ensure burst images loaded for grid preview
async ensureBurstImagesLoaded(burstId) {
    if (!this.burstImages[burstId]) {
        const images = await this.fetchBurstImages(burstId);
        this.burstImages[burstId] = images;
    }
}
```

#### Modified Methods
- `selectBurst(burst)` - Now opens modal instead of updating panel
- `loadBursts()` - Pre-fetch first 4 images for grid thumbnails

### 4. API Changes

No backend changes required. All existing burst APIs remain unchanged.

## Visual Specification

### Burst Card (Grid View)

```
┌─────────────────────────┐
│ Canon EOS R5      8 imgs│ ← header
├─────────────────────────┤
│ [img] [img] [img] [img] │ ← 4 thumbnails (best has ★)
│                   +4    │ ← overflow indicator if >4
├─────────────────────────┤
│ 2.3s        14:32:15    │ ← footer (duration, time)
└─────────────────────────┘
```

### Burst Modal (Detail View)

```
┌────────────────────────────────────────┐
│ Burst Sequence - 8 Images          [×] │
├────────────────────────────────────────┤
│ Canon EOS R5  Duration: 2.3s  14:32:15 │
├────────────────────────────────────────┤
│                                        │
│         [Large Preview Image]          │
│    filename.jpg  #3  Q: 87.3%         │
│    [Set as Best] [View Full Size]     │
│                                        │
├────────────────────────────────────────┤
│ [★#1] [#2] [#3] [#4] [#5] [#6] [#7]   │ ← filmstrip
├────────────────────────────────────────┤
│                           [Close]      │
└────────────────────────────────────────┘
```

## Benefits

1. **Visual Consistency** - Same interaction pattern for viewing grouped content
2. **Code Reuse** - ~60% code reduction through shared classes and structure
3. **Better Space Usage** - No permanent left panel, more room for content
4. **Maintainability** - Changes to group viewing pattern benefit both views
5. **User Experience** - Familiar pattern reduces cognitive load

## Preserved Functionality

All existing burst features remain:
- ✓ Burst detection
- ✓ Camera/size filtering
- ✓ Quality scores
- ✓ "Set as Best" image selection
- ✓ Sequence numbers
- ✓ Duration/timing display
- ✓ Filmstrip navigation
- ✓ Full-size image viewing

## Migration Notes

### For Users
- Burst list moves from left sidebar to main grid
- Click burst card to view details (was: auto-selected)
- Modal for details (was: always-visible panel)
- All features preserved, just reorganized

### For Developers
- Shared CSS classes reduce maintenance
- Modal pattern consistent across features
- Left panel freed for future uses

## Future Enhancements

With unified structure, both views can benefit from:
- Keyboard navigation (arrow keys in modal)
- Bulk actions on selected groups
- Export/share group functionality
- Advanced filtering options

## Conclusion

This change unifies the visual language of VAM Tools while maintaining all burst-specific functionality. The DRY approach reduces code and ensures future improvements benefit both duplicates and bursts.
