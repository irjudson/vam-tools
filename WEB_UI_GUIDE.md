# Web UI Guide

## Overview

The web-based catalog viewer provides a visual interface to explore and review your analyzed photo catalog. It displays statistics, allows filtering and sorting, and shows image previews with metadata.

## Features

✅ **Dashboard**
- Total images and videos count
- Total storage used
- Issues count (no date, suspicious dates)

✅ **Image Browser**
- Grid view with thumbnails
- Filter by type (images, videos, no date, suspicious dates)
- Sort by date, path, or size
- Pagination support

✅ **Image Details**
- Click any image to see full details
- View larger preview
- See all date information
- View EXIF metadata
- See confidence scores

## Quick Start

### 1. Install the package

First, make sure you're in your virtual environment and install with the new web dependencies:

```bash
source venv/bin/activate
pip install -e .
```

This will install FastAPI and Uvicorn along with the other dependencies.

### 2. Make sure you have a catalog

If you haven't run the analysis yet:

```bash
vam-analyze /path/to/catalog --source /path/to/photos
```

### 3. Start the web server

```bash
vam-web /path/to/catalog
```

For example:
```bash
vam-web /mnt/synology/shared/test-catalog
```

The server will start on `http://127.0.0.1:8765` by default.

### 4. Open in your browser

Navigate to:
```
http://localhost:8765
```

You should see the catalog viewer with your statistics and images!

## Command Options

```bash
vam-web --help
```

Available options:
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 8765)
- `--reload`: Enable auto-reload for development

Examples:
```bash
# Start on different port
vam-web /path/to/catalog --port 8080

# Make accessible on network
vam-web /path/to/catalog --host 0.0.0.0 --port 8765

# Development mode with auto-reload
vam-web /path/to/catalog --reload
```

## Using the Interface

### Dashboard

The top section displays interactive charts showing catalog statistics:

**File Types Chart** 📊
- Toggle between two views:
  - **Category View**: Images vs Videos distribution
  - **Extension View**: Breakdown by file extension (.jpg, .png, .heic, etc.)
- Click the button to switch views
- Bar chart shows count of each type

**Size Distribution Chart** 📈
- Histogram showing file size distribution across buckets:
  - < 100 KB
  - 100 KB - 1 MB
  - 1 MB - 5 MB
  - 5 MB - 10 MB
  - 10 MB - 50 MB
  - \> 50 MB
- Helps identify large files or unusual size patterns

**Issues Chart** ⚠️
- Doughnut chart showing breakdown of issues:
  - **No Date**: Files with no date information
  - **Suspicious Date**: Files with questionable dates (future, very old, defaults)
  - **Low Confidence**: Files with date confidence < 70%
- Interactive legend on the right

**Overview Card** 📋
- Total file count (images + videos)
- Total catalog size in human-readable format

All charts update automatically when auto-refresh is enabled!

### Navigation Modes

**Scroll Mode (Default)** 📜
- Images load continuously as you scroll down
- Shows "X of Y images" counter
- Automatically loads more when you reach bottom
- Can also click "Load More Images" button
- Perfect for browsing large catalogs

**Page Mode** 📄
- Traditional pagination with Previous/Next buttons
- Shows "Page X of ~Y" indicator
- Navigate with arrow keys (← →) or H/L keys
- Better for jumping to specific sections

Toggle between modes with the **Scroll/Page Mode** button or press **S** key.

### Filtering

Use the dropdown menus to filter images:
- **All Images**: Show everything
- **Images Only**: Only image files (no videos)
- **Videos Only**: Only video files
- **No Date**: Files where no date could be extracted
- **Suspicious Dates**: Files with questionable dates (future, very old, or default camera dates)

### Sorting

Sort images by:
- **Date**: Chronological order (newest first)
- **Path**: Alphabetical by file path
- **Size**: Largest files first

### Keyboard Shortcuts

Make browsing faster with keyboard shortcuts:
- **R** - Refresh data now
- **S** - Switch between Scroll/Page mode
- **A** - Toggle auto-refresh on/off
- **← / →** - Previous/Next page (Page mode only)
- **H / L** - Previous/Next page (Vim-style, Page mode only)
- **Escape** - Close image detail modal

### Image Cards

Each image card shows:
- Thumbnail preview
- Filename
- Format (JPEG, PNG, HEIC, etc.)
- Resolution
- File size
- Date badge (or "No Date" if missing)
- Warning badge for suspicious dates

### Image Details

Click any image to see full details:
- Larger preview
- Full file path
- All extracted dates (EXIF, filename, directory, filesystem)
- Date source and confidence score
- Complete metadata

## API Endpoints

The backend provides a REST API that the frontend uses. You can also access these directly:

### Get catalog info
```
GET http://localhost:8765/api/catalog/info
```

### List images with filtering
```
GET http://localhost:8765/api/images?skip=0&limit=50&filter_type=no_date&sort_by=date
```

Parameters:
- `skip`: Number of images to skip (pagination)
- `limit`: Max images to return (1-1000)
- `filter_type`: Filter type (no_date, suspicious, image, video)
- `sort_by`: Sort order (date, path, size)

### Get image details
```
GET http://localhost:8765/api/images/{image_id}
```

### Get image file
```
GET http://localhost:8765/api/images/{image_id}/file
```

### Get statistics summary
```
GET http://localhost:8765/api/statistics/summary
```

Returns detailed breakdown by format, date source, year, etc.

## Troubleshooting

### "Catalog not initialized"

Make sure you've run `vam-analyze` first to create the catalog:
```bash
vam-analyze /path/to/catalog --source /path/to/photos
```

### Images not loading

Check that:
1. The catalog path is correct
2. The source files still exist at their original paths
3. You have read permissions for the source files

### Port already in use

If port 8765 is already in use:
```bash
vam-web /path/to/catalog --port 8080
```

### CORS errors

If accessing from a different host, you may need to modify the CORS settings in `vam_tools/v2/web/api.py`.

## Next Steps

While the web UI is running, you can:
- Browse all your images
- Review images with no dates
- Check suspicious dates
- Identify issues that need manual review
- **Watch analysis progress in real-time** - Just refresh your browser while `vam-analyze` is running to see updated counts!

### Live Progress Monitoring

You can run the analysis and web viewer simultaneously:

```bash
# Terminal 1: Start the analysis
vam-analyze /path/to/catalog --source /path/to/photos

# Terminal 2: Start the web viewer
vam-web /path/to/catalog
```

Then simply refresh your browser periodically to see:
- Updated image counts as files are processed
- New images appearing in the grid
- Statistics updating in real-time
- Checkpoint progress

The web server automatically reloads the catalog whenever it detects the file has changed on disk.

Future features will include:
- Duplicate group visualization
- Burst sequence viewer
- Manual date correction
- Conflict resolution UI
- Plan review and approval

## Performance Notes

- The web UI loads images in batches of 50 for performance
- Thumbnails are loaded on-demand with lazy loading
- Large catalogs (10,000+ images) may take a moment to load statistics
- The catalog file is automatically reloaded when it changes on disk
- Simply refresh your browser to see updates from ongoing analysis scans

## Development

To work on the web UI:

1. Start in reload mode:
```bash
vam-web /path/to/catalog --reload
```

2. Edit files in `vam_tools/v2/web/`:
   - `api.py`: Backend API endpoints
   - `static/index.html`: Frontend UI

3. Changes to Python code will auto-reload
4. Refresh browser to see frontend changes

## Architecture

```
Frontend (Vue.js)
    ↓ HTTP requests
Backend (FastAPI)
    ↓ Reads
Catalog Database (.catalog.json)
    ↓ References
Source Files (images/videos)
```

The web UI is read-only - it won't modify your catalog or files. All changes happen through the analysis and execution phases.
