# Date Extraction Guide

Lumina uses a sophisticated multi-source approach to extract the most accurate capture date for each photo or video.

## Date Sources and Confidence Levels

Dates are extracted from multiple sources, each with an assigned confidence level based on reliability:

### 1. EXIF Metadata (95% confidence)

**Most Reliable** - Camera-generated timestamps from EXIF/XMP data

**Fields Checked** (in priority order):
1. `DateTimeOriginal` - When the photo was taken (preferred)
2. `CreateDate` - File creation date
3. `DateTimeDigitized` - When the image was digitized
4. `ModifyDate` - Last modification date (least preferred)

**Why High Confidence:**
- Written by camera at capture time
- Typically accurate unless camera clock was wrong
- Preserved across file copies and edits (in sidecar files)

**Example EXIF Data:**
```
Date/Time Original: 2023:04:15 14:32:18
Create Date: 2023:04:15 14:32:18
```

### 2. Filename Patterns (70% confidence)

**Reliable** - Date embedded in filename by camera or user

**Patterns Recognized:**
- `YYYY-MM-DD` - 2023-04-15
- `YYYYMMDD` - 20230415
- `YYYY_MM_DD` - 2023_04_15
- `IMG_YYYYMMDD` - IMG_20230415
- `YYYYMMDD_HHMMSS` - 20230415_143218
- `DSC_YYYYMMDD` - DSC_20230415 (Nikon)
- `_DSC####` with date in path - _DSC1234 (Sony)

**Why Medium-High Confidence:**
- Often set by camera or smartphone
- May be user-renamed with correct date
- Can be incorrect if file copied/renamed with wrong pattern

**Example Filenames:**
```
IMG_20230415_143218.jpg    → 2023-04-15 14:32:18
2023-04-15_Beach_Sunset.jpg → 2023-04-15
DSC_20230415.ARW           → 2023-04-15
```

### 3. Directory Structure (50% confidence)

**Moderate** - Date inferred from folder organization

**Patterns Recognized:**
- `YYYY/MM/` - 2023/04/
- `YYYY-MM/` - 2023-04/
- `YYYY/Month Name/` - 2023/April/
- `Photos/YYYY/` - Photos/2023/

**Why Medium Confidence:**
- Users often organize photos by date
- May reflect organization date, not capture date
- Folders can be renamed or files moved

**Example Paths:**
```
/Photos/2023/04/vacation.jpg  → 2023-04 (no day)
/Pictures/2023-04-15/beach/   → 2023-04-15
/Media/April 2023/img001.jpg  → 2023-04 (no day)
```

### 4. Filesystem Metadata (30% confidence)

**Least Reliable** - File creation/modification time from filesystem

**Fields Used:**
- File creation time (`stat.st_ctime` or `stat.st_birthtime`)
- File modification time (`stat.st_mtime`)

**Why Low Confidence:**
- Changes when file is copied
- Changes when file is edited
- Timezone may be incorrect
- Can be completely wrong after system restores

**When Useful:**
- Last resort when no other metadata available
- Can provide approximate timeframe
- Better than no date at all

## Date Selection Algorithm

Lumina uses an **earliest date wins** strategy with confidence weighting:

```python
def select_date(candidates):
    """Select the best date from all candidates."""

    # Filter out invalid dates (future, too old, etc.)
    valid = [d for d in candidates if is_valid(d)]

    # Sort by: 1) confidence (desc), 2) date (asc = earliest)
    sorted_dates = sorted(valid, key=lambda d: (-d.confidence, d.date))

    # Return highest confidence, earliest date
    return sorted_dates[0] if sorted_dates else None
```

### Example Date Selection

**Scenario**: Photo with multiple date sources

```
EXIF DateTimeOriginal:  2023-04-15 14:32:18  (95% confidence)
Filename:               2023-04-15           (70% confidence)
Directory:              2023/04              (50% confidence)
File Modified:          2024-01-10 08:15:22  (30% confidence)
```

**Selected**: 2023-04-15 14:32:18 (EXIF, highest confidence and earliest)

### Edge Cases

**Multiple EXIF Dates Differ:**
```
DateTimeOriginal: 2023-04-15 14:32:18
ModifyDate:       2024-01-10 08:15:22
```
→ Selects `DateTimeOriginal` (earliest and more reliable)

**Filename vs EXIF Conflict:**
```
EXIF:     2023-04-15  (95% confidence)
Filename: 2023-12-25  (70% confidence)
```
→ Selects EXIF date (higher confidence)
→ **Flags as suspicious** due to conflict

**Only Filesystem Date Available:**
```
File Modified: 2024-01-10
```
→ Uses filesystem date (30% confidence)
→ **Flags as low confidence** for user review

## Suspicious Date Detection

Dates are marked **suspicious** when:

1. **Conflicting Sources** - EXIF and filename differ by >30 days
2. **Future Date** - Date is in the future
3. **Implausible Date** - Before 1970 or >100 years old
4. **Filesystem Only** - No metadata, only filesystem date
5. **Low Confidence** - Selected date has <50% confidence

**Review Workflow:**
```bash
# Launch web UI
vam-web /path/to/catalog

# Navigate to: Review Queue → Suspicious Dates
# Manually correct or accept suggested dates
```

## Date Sources by File Type

### JPEG/PNG/TIFF
- ✅ EXIF metadata (primary)
- ✅ Filename patterns
- ✅ Directory structure
- ✅ Filesystem metadata

### RAW Files (ARW, NEF, CR2, etc.)
- ✅ EXIF metadata (primary, via ExifTool)
- ✅ Filename patterns
- ✅ Directory structure
- ✅ Filesystem metadata

### HEIC/HEIF
- ✅ EXIF metadata (via ExifTool)
- ✅ Filename patterns
- ✅ Directory structure
- ✅ Filesystem metadata

### Videos (MP4, MOV, etc.)
- ✅ Metadata (via ExifTool)
- ✅ Filename patterns
- ✅ Directory structure
- ✅ Filesystem metadata

## Manual Date Correction

For files with no date or suspicious dates, you can manually set the correct date:

**Via Web UI:**
1. Open web interface: `vam-web /path/to/catalog`
2. Navigate to image details
3. Click "Edit Date"
4. Enter correct date (YYYY-MM-DD HH:MM:SS)
5. Save (sets confidence to 100% and source to "manual")

**Via API:**
```bash
curl -X PATCH "http://localhost:8765/api/images/{image_id}/date?date_str=2023-04-15%2014:32:18"
```

## Timezone Handling

**Current Behavior:**
- All dates stored as-is (no timezone conversion)
- EXIF dates typically in camera's local time
- Filesystem dates in system's local time

**Best Practices:**
1. Set camera clock to correct timezone
2. Keep consistent timezone across devices
3. Manually correct dates after timezone changes
4. Consider timezone when organizing by date

**Future Enhancement:**
- Timezone detection from GPS data
- Timezone correction based on location
- UTC storage with timezone display

## Common Date Extraction Issues

### Issue: "No date found"

**Cause**: File has no EXIF data, filename has no date, in root directory

**Solutions:**
1. Check if file has EXIF: `exiftool photo.jpg | grep Date`
2. Rename file with date: `mv photo.jpg 2023-04-15_photo.jpg`
3. Move to dated folder: `mv photo.jpg 2023/04/15/`
4. Manually set date via web UI

### Issue: "Wrong date selected"

**Cause**: Multiple conflicting dates, filesystem date incorrect

**Solutions:**
1. Review in web UI under "Suspicious Dates"
2. Check EXIF data: `exiftool photo.jpg`
3. Verify filename pattern is correct
4. Manually correct date if needed

### Issue: "Future date"

**Cause**: Camera clock was wrong, file timestamp incorrect

**Solutions:**
1. Marked as suspicious automatically
2. Review in web UI
3. Batch correct with date offset if many files affected
4. Manually set correct dates

### Issue: "All photos showing same date"

**Cause**: Only filesystem dates available (files copied without preserving timestamps)

**Solutions:**
1. Restore from backup with correct timestamps
2. Extract dates from original filenames if available
3. Manually set dates for important photos
4. Consider using photo management tools to restore metadata

## Technical Implementation

### ExifTool Integration

Lumina uses ExifTool for comprehensive metadata extraction:

```python
# Extract all date fields
cmd = ["exiftool", "-j", "-DateTimeOriginal", "-CreateDate",
       "-DateTimeDigitized", "-ModifyDate", "/path/to/image.jpg"]

result = subprocess.run(cmd, capture_output=True, text=True)
metadata = json.loads(result.stdout)[0]

# Parse and validate dates
dates = {}
for field in ["DateTimeOriginal", "CreateDate", ...]:
    if field in metadata:
        dates[field] = parse_exif_date(metadata[field])
```

### Date Parsing

Handles multiple date formats:

- ISO 8601: `2023-04-15T14:32:18`
- EXIF format: `2023:04:15 14:32:18`
- Date only: `2023-04-15`
- Unix timestamp: `1681567938`

### Validation Rules

```python
def is_valid_date(date):
    """Check if date is plausible."""
    now = datetime.now()
    min_date = datetime(1970, 1, 1)  # Unix epoch
    max_date = now + timedelta(days=365)  # 1 year future

    return min_date <= date <= max_date
```

## See Also

- [How It Works](./HOW_IT_WORKS.md) - Overall analysis pipeline
- [User Guide](./USER_GUIDE.md) - Date correction workflows
- [Troubleshooting](./TROUBLESHOOTING.md) - Date extraction problems
