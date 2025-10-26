# V2 Testing Guide

## What's Ready to Test

The analysis phase is complete and ready for testing on your real data!

### Implemented Features

✅ **Catalog Database Manager**
- File locking for safe concurrent access
- Atomic writes with automatic backup
- Checkpoint system (saves every 5 minutes or 100 images)
- Transaction support (prepared for execution phase)

✅ **Metadata Extraction**
- Full EXIF extraction using ExifTool
- Date extraction from multiple sources:
  - EXIF dates (DateTimeOriginal, CreateDate, etc.)
  - Filename patterns (YYYY-MM-DD, YYYYMMDD, etc.)
  - Directory structure (YYYY-MM folders)
  - Filesystem timestamps
- Automatic date selection with confidence scoring
- Suspicious date detection (future dates, very old dates, default dates)
- Image format and resolution detection
- Video file support (format detection, metadata extraction)

✅ **Image Scanner**
- Recursive directory scanning
- SHA256 checksum computation for all files
- Progress tracking with rich console output
- Automatic checkpointing during long scans
- Resume capability (won't re-process existing checksums)

✅ **Catalog JSON Database**
- Human-readable JSON format
- Complete metadata storage
- Backup before every write
- Recovery from backup on corruption

### File Structure Created

```
vam_tools/v2/
├── __init__.py
├── core/
│   ├── types.py          # Complete type system
│   ├── utils.py          # Utility functions
│   └── catalog.py        # Database manager
├── analysis/
│   ├── metadata.py       # Metadata extractor
│   └── scanner.py        # Directory scanner
└── cli_analyze.py        # CLI for testing
```

## How to Test

### 1. Install/Update the Package

```bash
cd vam-tools
pip install -e .
```

This will install the `vam-analyze` command.

### 2. Run Analysis on a Small Test Set

Start with a small directory to test (e.g., 10-100 images):

```bash
vam-analyze /path/to/test_catalog --source /path/to/test_photos
```

Example:
```bash
vam-analyze ~/photo-catalog-test --source ~/Pictures/test-batch
```

### 3. Review the Output

The tool will:
1. Create `/path/to/test_catalog/.catalog.json`
2. Scan all images in the source directory
3. Extract metadata and dates
4. Show progress with a progress bar
5. Display statistics when complete

### 4. Inspect the Catalog

```bash
# View the catalog JSON (it's human-readable)
cat ~/photo-catalog-test/.catalog.json | jq .

# Or just look at statistics
cat ~/photo-catalog-test/.catalog.json | jq '.statistics'
```

### 5. Test on Larger Dataset

Once the small test works:

```bash
vam-analyze /path/to/full_catalog \
  --source /external-drive-1/photos \
  --source /external-drive-2/photos \
  --source ~/Pictures
```

### 6. Monitor Progress

The scanner:
- Shows progress bar with time remaining
- Checkpoints every 100 images (saves state)
- Can be interrupted (Ctrl+C) and resumed
- Logs to console (use `-v` for verbose)

## What to Look For

### ✅ Success Indicators

1. **Scan completes without errors**
2. **Statistics look reasonable**:
   ```
   Total Images: 100
   Total Videos: 5
   Total Size: 2.5 GB
   Images with no date: 10
   ```

3. **Catalog JSON is created**:
   ```bash
   ls -lh ~/photo-catalog-test/.catalog.json
   ```

4. **Dates are extracted**:
   ```bash
   cat ~/photo-catalog-test/.catalog.json | jq '.images[] | .dates'
   ```

### ⚠️ Things to Check

1. **Images with no dates** - Check the count in statistics
2. **Suspicious dates** - Look for `"suspicious": true` in the catalog
3. **Scan performance** - How long does it take per image?
4. **Memory usage** - Monitor with `htop` or Activity Monitor
5. **Checkpoint/resume** - Interrupt scan (Ctrl+C) and restart - should continue

### 🐛 Known Limitations (Not Yet Implemented)

- ❌ No duplicate detection yet
- ❌ No burst detection yet
- ❌ No plan generation yet
- ❌ No execution (files stay in place)
- ❌ No web UI for review

These will be added in the next iteration.

## Sample Output

```
VAM Tools V2 - Analysis

Catalog: /home/user/photo-catalog-test
Sources: /home/user/Pictures/vacation

Initializing new catalog...

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

## Testing Checklist

- [ ] Install package: `pip install -e .`
- [ ] Run on small test set (10-100 images)
- [ ] Verify catalog.json is created
- [ ] Check statistics match expected counts
- [ ] Inspect a few image records for correct metadata
- [ ] Check date extraction for files with different formats
- [ ] Test with videos (if you have any)
- [ ] Test interrupt/resume (Ctrl+C and restart)
- [ ] Run on larger dataset (1000+ images)
- [ ] Check for memory leaks on large scans
- [ ] Verify checkpoint files are created

## Troubleshooting

### "ExifTool not found"
```bash
# Install ExifTool first
brew install exiftool  # macOS
# or
sudo apt install libimage-exiftool-perl  # Linux
```

### "Permission denied"
- Check directory permissions
- Make sure you have read access to source directories

### Scan is slow
- Normal: ~1-5 images/second depending on file size
- Slow disk or network drives will be slower
- Check CPU/disk usage

### Out of memory
- Try smaller batches
- Close other applications
- The checkpoint system prevents data loss

## Next Steps

Once you've tested the analysis:

1. **Review the catalog JSON** - Is the data structure useful?
2. **Check date extraction accuracy** - Are dates being found correctly?
3. **Identify issues** - Any missing metadata or incorrect dates?
4. **Report findings** - What works? What needs adjustment?

Then we'll build:
- Duplicate detection
- Burst detection
- Review UI
- Plan generation
- Execution engine

## Catalog JSON Structure

Example of what gets stored:

```json
{
  "version": "2.0.0",
  "catalog_id": "uuid-...",
  "created": "2025-10-25T13:00:00",
  "last_updated": "2025-10-25T13:15:00",

  "statistics": {
    "total_images": 523,
    "total_videos": 12,
    "total_size_bytes": 2500000000,
    "no_date": 15
  },

  "images": {
    "sha256:abc123...": {
      "id": "sha256:abc123...",
      "source_path": "/path/to/IMG_001.jpg",
      "file_type": "image",
      "checksum": "sha256:abc123...",
      "dates": {
        "exif_dates": {
          "DateTimeOriginal": "2023-06-15T12:00:00"
        },
        "filename_date": null,
        "directory_date": "2023-06",
        "selected_date": "2023-06-15T12:00:00",
        "selected_source": "exif:DateTimeOriginal",
        "confidence": 95,
        "suspicious": false
      },
      "metadata": {
        "format": "JPEG",
        "resolution": [3840, 2160],
        "size_bytes": 2500000,
        "exif": {
          "Make": "Canon",
          "Model": "EOS 5D Mark IV",
          ...
        }
      }
    }
  }
}
```

Ready to test!
