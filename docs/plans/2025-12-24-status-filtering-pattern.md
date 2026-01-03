# Status Filtering Pattern

This document describes the pattern for filtering images by status across the Lumina API.

## Overview

Images in Lumina can have different status values (`active`, `rejected`, etc.). Most API endpoints should exclude rejected images by default to provide a clean user experience, while offering an option to include all statuses when needed.

## Default Behavior

**By default, API endpoints should exclude rejected images.**

The standard SQL pattern for filtering out rejected images:

```sql
WHERE (status_id IS NULL OR status_id != 'rejected')
```

This handles both cases:
- Images with no status set (`status_id IS NULL`) - included
- Images with status set to anything except 'rejected' - included
- Images with status set to 'rejected' - excluded

## Including All Statuses

Endpoints that need to show all images (including rejected ones) should accept a `show_rejected` query parameter:

```python
@router.get("/api/catalogs/{catalog_id}/images")
async def list_images(
    catalog_id: str,
    show_rejected: bool = Query(False, description="Include rejected images"),
    db: Session = Depends(get_db),
):
    """List images in a catalog."""
    # Build base query
    query = "SELECT * FROM images WHERE catalog_id = :catalog_id"

    # Apply status filter unless show_rejected=True
    if not show_rejected:
        query += " AND (status_id IS NULL OR status_id != 'rejected')"

    # Execute query...
```

## Query Parameter Pattern

For consistency, always use this parameter definition:

```python
show_rejected: bool = Query(False, description="Include rejected images")
```

Or for burst-specific endpoints:

```python
show_rejected: bool = Query(False, description="Include bursts where all images are rejected")
```

## Endpoints Needing Updates

The following endpoints should be updated to follow this pattern:

### High Priority
- [ ] `GET /api/catalogs/{catalog_id}/images` - List images
- [ ] `GET /api/catalogs/{catalog_id}/thumbnails` - Grid view
- [ ] `GET /api/catalogs/{catalog_id}/search` - Search results
- [ ] `GET /api/catalogs/{catalog_id}/similar/{image_id}` - Similar images

### Medium Priority
- [ ] `GET /api/catalogs/{catalog_id}/tags` - Filter counts should respect status
- [ ] `GET /api/catalogs/{catalog_id}/duplicates` - Duplicate detection
- [ ] `GET /api/catalogs/{catalog_id}/random` - Random image selection

### Already Implemented
- [x] `GET /api/catalogs/{catalog_id}/bursts` - Has `show_rejected` parameter

## CSS Rendering Pattern

When rejected images ARE shown (via `show_rejected=true`), they should be visually distinguished in the UI.

### Recommended CSS Pattern

```css
/* Rejected image styling */
.image-item.rejected {
    opacity: 0.5;
    position: relative;
}

.image-item.rejected::after {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        45deg,
        transparent,
        transparent 10px,
        rgba(255, 0, 0, 0.1) 10px,
        rgba(255, 0, 0, 0.1) 20px
    );
    pointer-events: none;
}

.image-item.rejected .status-badge {
    background: var(--color-danger);
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75em;
    text-transform: uppercase;
}
```

### Vue.js Template Pattern

```vue
<div
    class="image-item"
    :class="{ rejected: image.status_id === 'rejected' }"
>
    <img :src="image.thumbnail_url" :alt="image.name">
    <div v-if="image.status_id === 'rejected'" class="status-badge">
        Rejected
    </div>
</div>
```

## Testing Requirements

Each endpoint that implements status filtering should have tests covering:

1. **Default behavior** - Verify rejected images are excluded by default
2. **show_rejected=false** - Explicitly test that rejected images are excluded
3. **show_rejected=true** - Verify rejected images are included when requested
4. **Mixed statuses** - Test with a mix of active, rejected, and null status images

Example test structure:

```python
class TestImageStatusFiltering:
    """Tests for image status filtering."""

    def test_list_images_excludes_rejected_by_default(self, client, catalog_with_mixed_statuses):
        """Verify rejected images are excluded by default."""
        response = client.get(f"/api/catalogs/{catalog_with_mixed_statuses.id}/images")
        assert response.status_code == 200
        data = response.json()
        # Verify no rejected images in response
        assert all(img["status_id"] != "rejected" for img in data["images"])

    def test_list_images_includes_rejected_when_requested(self, client, catalog_with_mixed_statuses):
        """Verify rejected images are included when show_rejected=true."""
        response = client.get(
            f"/api/catalogs/{catalog_with_mixed_statuses.id}/images",
            params={"show_rejected": True}
        )
        assert response.status_code == 200
        data = response.json()
        # Verify rejected images are present
        has_rejected = any(img["status_id"] == "rejected" for img in data["images"])
        assert has_rejected
```

## Implementation Checklist

When adding status filtering to an endpoint:

1. [ ] Add `show_rejected` query parameter with default `False`
2. [ ] Apply status filter in SQL: `AND (status_id IS NULL OR status_id != 'rejected')`
3. [ ] Conditionally apply filter based on `show_rejected` parameter
4. [ ] Update API documentation/docstring
5. [ ] Add tests for both modes (default excluded, explicitly included)
6. [ ] Update frontend to pass `show_rejected` when needed
7. [ ] Add CSS styling for rejected images when shown
8. [ ] Update this checklist in the endpoint tracking section above

## Migration Strategy

To avoid breaking changes, implement this pattern gradually:

1. **Phase 1**: Add `show_rejected` parameter to all relevant endpoints (default `False`)
2. **Phase 2**: Update frontend components to use the parameter
3. **Phase 3**: Add visual indicators for rejected images in UI
4. **Phase 4**: Add user preferences for default behavior

## Related Documentation

- See `vam_tools/api/routers/catalogs.py` lines 2441-2492 for reference implementation
- Image status values are defined in the database schema
- Status can be set via `PUT /api/catalogs/{catalog_id}/images/{image_id}/status`
