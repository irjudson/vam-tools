# Re-running Analysis

## Why Re-run?

Your existing catalog was created before we fixed the serialization code. The catalog JSON only has basic image info (path, checksum, type) but is missing:
- ✗ Date information (EXIF dates, filename dates, selected dates)
- ✗ Metadata (format, resolution, size_bytes, EXIF data)
- ✗ Statistics (total images, videos, size counts)

The fixes have been applied, so re-running will capture everything correctly.

## How to Re-run

### Option 1: Fresh Start (Recommended)

```bash
# Backup old catalog (optional)
mv /mnt/synology/shared/test-catalog/.catalog.json /mnt/synology/shared/test-catalog/.catalog.json.old

# Delete checkpoint files
rm -f /mnt/synology/shared/test-catalog/.catalog.*.json
rm -f /mnt/synology/shared/test-catalog/.catalog.lock

# Re-run analysis
vam-analyze /mnt/synology/shared/test-catalog \
  --source "/mnt/synology/shared/Possible Duplicates/" \
  --verbose
```

### Option 2: New Catalog Directory

```bash
# Create new catalog directory
mkdir /mnt/synology/shared/photo-catalog

# Run analysis
vam-analyze /mnt/synology/shared/photo-catalog \
  --source "/mnt/synology/shared/Possible Duplicates/" \
  --verbose
```

## What's Fixed

The re-run will now properly save:

✅ **Statistics**
- Total images count
- Total videos count
- Total size in bytes
- Images with no date count

✅ **Complete Metadata**
- Image format (JPEG, PNG, HEIC, etc.)
- Resolution (width x height)
- File size in bytes
- Full EXIF data

✅ **Complete Date Information**
- All EXIF dates (DateTimeOriginal, CreateDate, etc.)
- Filename-extracted dates
- Directory-extracted dates
- Filesystem dates (created, modified)
- Selected best date with confidence score
- Suspicious date flagging

## Expected Results

After re-running, the web UI will show:
- ✅ Correct statistics (not all 0s)
- ✅ File sizes (not "0 B")
- ✅ Dates for images
- ✅ Format information
- ✅ Resolution info
- ✅ Full EXIF metadata in detail view

## Timing

- Your 26,900 files will take approximately 1.5-4 hours depending on:
  - Disk speed (network drive is slower)
  - ExifTool processing time
  - Whether files need HEIC conversion

## Monitoring Progress

The scanner will:
- Show progress bar with % complete
- Display estimated time remaining
- Checkpoint every 100 images (can interrupt and resume)
- Log statistics when complete

## After Analysis Completes

1. Check the results:
   ```bash
   vam-web /mnt/synology/shared/test-catalog
   ```

2. Open browser to `http://localhost:8765`

3. You should see:
   - Dashboard with correct counts
   - Images showing sizes
   - Dates displayed
   - Filters working (No Date, Suspicious, etc.)

## If You Get Interrupted

The checkpoint system saves progress every 100 files. If interrupted:
- Just re-run the same command
- It will skip already-processed files
- Continue from where it left off

## Troubleshooting

### Still seeing 0s?
- Make sure you deleted the old catalog files
- Check that the analysis completed (not interrupted)
- Verify the catalog.json was updated (check last modified time)

### Missing metadata?
- Ensure ExifTool is installed: `which exiftool`
- Check verbose logs for errors during scan
- Some files may genuinely have no EXIF data

### HEIC files not displaying?
- Ensure pillow-heif is installed: `pip list | grep pillow-heif`
- Check web server logs for conversion errors
- Some HEIC files may be corrupted

## Questions?

Check the logs with `--verbose` flag to see detailed processing info.
