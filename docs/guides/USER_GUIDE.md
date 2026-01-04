# Lumina - User Guide

## Table of Contents

- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
- [Web Interface](#web-interface)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Getting Started

### Installation

```bash
# Clone or navigate to the project
cd lumina

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### System Requirements

**Required:**
- Python 3.9+ (supports 3.9, 3.10, 3.11, 3.12)
- ExifTool for metadata extraction

**Install ExifTool:**
```bash
# macOS
brew install exiftool

# Ubuntu/Debian
sudo apt-get install libimage-exiftool-perl

# Verify installation
which exiftool
```

**Optional:**
- pillow-heif for HEIC/HEIF image support (included in requirements)

## Quick Start

### Basic Workflow

1. **Analyze your photos** to build a catalog
2. **Browse the catalog** using the web interface
3. **Review duplicates** and conflicts
4. **(Future) Execute organization** to reorganize files

### Your First Scan

```bash
# Activate environment
source venv/bin/activate

# Scan a directory (start small)
vam-analyze ~/my-catalog --source ~/Pictures/vacation-2023

# View results in web browser
vam-web ~/my-catalog
```

Then open http://localhost:8765 in your browser.

## Command Reference

### vam-analyze

Build or update a photo catalog by scanning source directories.

```bash
vam-analyze CATALOG_PATH --source SOURCE_DIR [OPTIONS]
```

**Arguments:**
- `CATALOG_PATH`: Directory where catalog database will be stored

**Options:**
- `-s, --source PATH`: Source directory to scan (can specify multiple)
- `-v, --verbose`: Enable detailed logging
- `-w, --workers N`: Number of parallel workers (default: CPU count)
- `--detect-duplicates`: Enable perceptual duplicate detection
- `--similarity-threshold N`: Hamming distance threshold for duplicates (default: 5)

**Examples:**

```bash
# Simple scan
vam-analyze /path/to/catalog --source /path/to/photos

# Scan multiple directories
vam-analyze /path/to/catalog \
  --source ~/Pictures \
  --source /mnt/external/photos \
  --source /mnt/backup/photos

# With duplicate detection
vam-analyze /path/to/catalog \
  --source ~/Pictures \
  --detect-duplicates \
  --similarity-threshold 3

# Verbose with custom workers
vam-analyze /path/to/catalog \
  --source ~/Pictures \
  --workers 16 \
  --verbose
```

**Output:**
```
Lumina - Analysis

Catalog: /home/user/my-catalog
Sources: /home/user/Pictures

Starting scan...

Processing files... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

✓ Scan complete!

               Catalog Statistics
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric                ┃       Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Total Images          │         523 │
│ Total Videos          │          12 │
│ Total Size            │     2.34 GB │
│ Images with no date   │          15 │
└───────────────────────┴─────────────┘
```

### vam-web

Launch the web interface to browse and review your catalog.

```bash
vam-web CATALOG_PATH [OPTIONS]
```

**Arguments:**
- `CATALOG_PATH`: Path to catalog directory

**Options:**
- `--host HOST`: Host to bind to (default: 127.0.0.1)
- `--port PORT`: Port to bind to (default: 8765)
- `--reload`: Enable auto-reload for development

**Examples:**

```bash
# Start server on default port
vam-web /path/to/catalog

# Custom port
vam-web /path/to/catalog --port 8080

# Make accessible on network
vam-web /path/to/catalog --host 0.0.0.0 --port 8765

# Development mode with auto-reload
vam-web /path/to/catalog --reload
```

Then open http://localhost:8765 in your browser.

## Web Interface

### Dashboard

The dashboard displays catalog overview and statistics:

**File Types Chart** 📊
- Toggle between Category View (Images/Videos) and Extension View (.jpg, .png, etc.)
- Bar chart shows count of each type

**Size Distribution Chart** 📈
- Histogram showing file size distribution:
  - < 100 KB
  - 100 KB - 1 MB
  - 1 MB - 5 MB
  - 5 MB - 10 MB
  - 10 MB - 50 MB
  - > 50 MB

**Issues Chart** ⚠️
- Doughnut chart showing:
  - **No Date**: Files with no date information
  - **Suspicious Date**: Questionable dates (future, very old, defaults)
  - **Low Confidence**: Date confidence < 70%

**Overview Card** 📋
- Total file count
- Total catalog size

### Navigation

**Scroll Mode (Default)** 📜
- Images load continuously as you scroll
- Shows "X of Y images" counter
- Automatically loads more at bottom
- Click "Load More Images" button
- Perfect for browsing large catalogs

**Page Mode** 📄
- Traditional pagination with Previous/Next buttons
- Shows "Page X of ~Y" indicator
- Navigate with arrow keys (← →) or H/L keys
- Better for jumping to specific sections

Toggle with the **Scroll/Page Mode** button or press **S** key.

### Filtering and Sorting

**Filter by:**
- **All Images**: Show everything
- **Images Only**: Only image files (no videos)
- **Videos Only**: Only video files
- **No Date**: Files where no date could be extracted
- **Suspicious Dates**: Files with questionable dates

**Sort by:**
- **Date**: Chronological order (newest first)
- **Path**: Alphabetical by file path
- **Size**: Largest files first

### Keyboard Shortcuts

- **R** - Refresh data now
- **S** - Switch between Scroll/Page mode
- **A** - Toggle auto-refresh on/off
- **← / →** - Previous/Next page (Page mode only)
- **H / L** - Previous/Next page (Vim-style, Page mode only)
- **Escape** - Close image detail modal

### Image Details

Click any image card to view:
- Larger preview
- Full file path
- All extracted dates (EXIF, filename, directory, filesystem)
- Date source and confidence score
- Resolution, format, file size
- Complete EXIF metadata

### Duplicate Review

When duplicates are detected (`--detect-duplicates` flag):

**Duplicate Groups Page:**
- View all detected duplicate groups
- See quality scores for each image
- Side-by-side comparison
- Recommended primary selection
- Manual override controls (future)

## Common Tasks

### Initial Catalog Scan

Start with a small test directory to verify everything works:

```bash
# Scan a small test set (10-100 images)
vam-analyze ~/test-catalog --source ~/Pictures/test-batch

# View results
vam-web ~/test-catalog
```

### Incremental Updates

Add new photos to an existing catalog:

```bash
# Scan the same catalog with updated source directories
vam-analyze /path/to/catalog --source /path/to/photos

# The scanner will:
# - Skip already-processed files (by checksum)
# - Add new files
# - Update statistics
```

### Large Library Scan

For 10,000+ images:

```bash
# Use all CPU cores, enable verbose logging
vam-analyze /path/to/catalog \
  --source /path/to/large-library \
  --workers 32 \
  --verbose
```

**Performance:**
- ~1-5 images/second depending on file size and disk speed
- Linear scaling up to CPU core count
- Checkpoints every 100 files (safe to interrupt)

### Monitoring Progress

You can run analysis and web viewer simultaneously:

```bash
# Terminal 1: Start the analysis
vam-analyze /path/to/catalog --source /path/to/photos

# Terminal 2: Start the web viewer
vam-web /path/to/catalog
```

Then refresh your browser periodically to see:
- Updated image counts
- New images in the grid
- Real-time statistics
- Checkpoint progress

The web server automatically reloads the catalog when it detects file changes.

### Finding Duplicates

```bash
# Enable duplicate detection
vam-analyze /path/to/catalog \
  --source /path/to/photos \
  --detect-duplicates

# Adjust sensitivity (lower = more sensitive)
vam-analyze /path/to/catalog \
  --source /path/to/photos \
  --detect-duplicates \
  --similarity-threshold 3
```

**Similarity Threshold:**
- **0-3**: Very similar (strict matching)
- **4-6**: Similar (default: 5)
- **7-10**: Somewhat similar (loose matching)

### Reviewing Images with Issues

1. **No Date**: Filter by "No Date" in web UI to review files that need manual date assignment
2. **Suspicious Dates**: Review files with future dates, very old dates, or default camera dates
3. **Low Confidence**: Check files where date confidence is below 70%

## Troubleshooting

### "Catalog not initialized"

Make sure you've run `vam-analyze` first:
```bash
vam-analyze /path/to/catalog --source /path/to/photos
```

### "ExifTool not found"

Install ExifTool:
```bash
# macOS
brew install exiftool

# Ubuntu/Debian
sudo apt-get install libimage-exiftool-perl

# Verify
which exiftool
```

### Images not loading in web UI

Check that:
1. The catalog path is correct
2. Source files still exist at their original paths
3. You have read permissions for the source files

### Port already in use

If port 8765 is already in use:
```bash
vam-web /path/to/catalog --port 8080
```

### Scan is slow

Normal performance: ~1-5 images/second depending on:
- File size (larger files take longer)
- Disk speed (network drives are slower)
- CPU speed (EXIF extraction is CPU-bound)

To improve performance:
- Use local disk instead of network drive
- Increase workers: `--workers 32`
- Close other applications

### Out of memory

For very large catalogs (100k+ images):
- Try smaller batches
- Close other applications
- The checkpoint system prevents data loss

### HEIC files not processing

Ensure pillow-heif is installed:
```bash
pip install pillow-heif
```

If still failing, check that files aren't corrupted:
```bash
# Try opening with another tool
exiftool /path/to/image.heic
```

### Permission denied

Check directory permissions:
```bash
# Make sure you have read access
ls -la /path/to/photos

# If needed, fix permissions
chmod -R u+r /path/to/photos
```

### Scan interrupted (Ctrl+C)

The checkpoint system saves progress every 100 files. To resume:
```bash
# Just re-run the same command
vam-analyze /path/to/catalog --source /path/to/photos

# The scanner will:
# - Load last checkpoint
# - Skip already-processed files
# - Continue from where it left off
```

## Advanced Usage

### REST API

The web interface provides a REST API that you can access directly:

**Get catalog info:**
```
GET http://localhost:8765/api/catalog/info
```

**List images with filtering:**
```
GET http://localhost:8765/api/images?skip=0&limit=50&filter_type=no_date&sort_by=date
```

Parameters:
- `skip`: Number of images to skip (pagination)
- `limit`: Max images to return (1-1000)
- `filter_type`: Filter type (no_date, suspicious, image, video)
- `sort_by`: Sort order (date, path, size)

**Get image details:**
```
GET http://localhost:8765/api/images/{image_id}
```

**Get image file:**
```
GET http://localhost:8765/api/images/{image_id}/file
```

**Get statistics summary:**
```
GET http://localhost:8765/api/statistics/summary
```

Returns detailed breakdown by format, date source, year, etc.

**Duplicate statistics:**
```
GET http://localhost:8765/api/duplicates/stats
```

**Duplicate groups:**
```
GET http://localhost:8765/api/duplicates/groups
```

**Duplicate group details:**
```
GET http://localhost:8765/api/duplicates/groups/{group_id}
```

### Inspecting the Catalog

The catalog is stored as JSON (human-readable):

```bash
# View the entire catalog
cat ~/my-catalog/.catalog.json | jq .

# View just statistics
cat ~/my-catalog/.catalog.json | jq '.statistics'

# View images with no date
cat ~/my-catalog/.catalog.json | jq '.images[] | select(.dates.selected_date == null)'

# Count images by format
cat ~/my-catalog/.catalog.json | jq '.images[].metadata.format' | sort | uniq -c
```

### Re-running Analysis

If you need to rebuild your catalog (e.g., after updates):

```bash
# Backup old catalog (optional)
mv /path/to/catalog/.catalog.json /path/to/catalog/.catalog.json.old

# Delete checkpoint files
rm -f /path/to/catalog/.catalog.*.json
rm -f /path/to/catalog/.catalog.lock

# Re-run analysis
vam-analyze /path/to/catalog --source /path/to/photos --verbose
```

### Development Mode

For working on the web UI:

```bash
# Start in reload mode
vam-web /path/to/catalog --reload

# Edit files in vam_tools/web/
# - api.py: Backend API endpoints
# - static/index.html: Frontend UI

# Changes to Python auto-reload
# Refresh browser to see frontend changes
```

### Network Access

By default, the web server only accepts connections from localhost. To make it accessible on your network:

```bash
vam-web /path/to/catalog --host 0.0.0.0 --port 8765
```

Then access from other devices at: `http://YOUR_IP:8765`

**Warning**: This has no authentication. Only use on trusted networks.

### CORS Configuration

If accessing from a different host and encountering CORS errors, you may need to modify CORS settings in `vam_tools/web/api.py`.

## Performance Notes

### Memory Usage
- Catalog loaded entirely in memory
- ~500 bytes per image record
- 100k images ≈ 50 MB RAM
- Perceptual hashes add ~32 bytes per image

### Disk I/O
- Sequential reads for file scanning
- Random reads for checksum computation
- Checkpoint writes every 100 files
- Single final catalog write

### Scalability
- Linear scaling up to CPU core count
- 20-30x speedup on 32-core systems
- Tested with 100,000+ images
- For >500k images, consider SQLite in future versions

## What's Next

### Planned Features

**Organization Execution** (Phase 1):
- Move files to YYYY-MM directory structure
- Dry-run mode for safety
- Rollback support
- Checksum verification

**Burst Detection** (Phase 2):
- Group sequential images (burst mode, timelapse)
- Auto-select best shots
- Representative image selection

**AI-Powered Curation** (Phase 3):
- Scene detection
- Face recognition
- Object detection
- Quality assessment (blur, exposure, composition)

**Auto-Tagging** (Phase 3):
- Location tags from GPS
- Time-based tags (season, time of day)
- Content tags (scene type, objects)
- Technical tags (camera settings, lens type)
- Quality tags (sharp, rule-of-thirds)

See `REQUIREMENTS.md` for the complete roadmap.

## Getting Help

- **Documentation**: See `README.md` and `/docs` folder
- **Issues**: [GitHub Issues](https://github.com/irjudson/lumina/issues)
- **Discussions**: [GitHub Discussions](https://github.com/irjudson/lumina/discussions)

## Safety and Data Protection

Lumina is designed with safety in mind:

1. **Non-destructive by default**: Analysis phase never modifies source files
2. **Checkpoint system**: Progress saved every 100 files (safe to interrupt)
3. **File locking**: Prevents concurrent writes to catalog
4. **Dry-run modes**: Test operations before execution
5. **Atomic writes**: Catalog updates are atomic with backup
6. **Read-only web UI**: Web interface can't modify files

Your original files are never modified during the analysis phase. Future execution phases will require explicit confirmation for any file operations.
