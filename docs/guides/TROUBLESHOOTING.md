# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with VAM Tools.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Analysis Issues](#analysis-issues)
- [Web Interface Issues](#web-interface-issues)
- [GPU Issues](#gpu-issues)
- [Performance Issues](#performance-issues)
- [Data Issues](#data-issues)
- [Getting Help](#getting-help)

---

## Installation Issues

### "ExifTool not found"

**Symptom**: Error message "ExifTool not found" when running analysis

**Cause**: ExifTool is not installed or not in system PATH

**Solutions**:

1. **Verify ExifTool is installed**:
   ```bash
   exiftool -ver
   ```
   Should output version number (e.g., `12.40`)

2. **Install ExifTool**:

   **macOS (Homebrew)**:
   ```bash
   brew install exiftool
   ```

   **Ubuntu/Debian**:
   ```bash
   sudo apt-get install libimage-exiftool-perl
   ```

   **Windows**:
   - Download from [exiftool.org](https://exiftool.org/)
   - Extract and add to PATH

3. **Check PATH**:
   ```bash
   which exiftool  # Unix
   where exiftool  # Windows
   ```

4. **Restart terminal** after installation

---

### "Permission denied"

**Symptom**: Cannot read or write files during analysis or web viewing

**Cause**: Insufficient file system permissions

**Solutions**:

1. **Check file permissions** (Unix/Linux/macOS):
   ```bash
   ls -la /path/to/photos
   ```

2. **Ensure read access to source directories**:
   ```bash
   chmod -R +r /path/to/photos
   ```

3. **Ensure write access to catalog directory**:
   ```bash
   chmod -R +w /path/to/catalog
   ```

4. **Check ownership**:
   ```bash
   ls -l /path/to/catalog
   # If owned by different user:
   sudo chown -R $USER:$USER /path/to/catalog
   ```

5. **Windows**: Right-click folder → Properties → Security → Edit permissions

---

### "Module not found" or Import Errors

**Symptom**: Python import errors when running commands

**Cause**: Package not installed or virtual environment not activated

**Solutions**:

1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate  # Unix
   venv\Scripts\activate     # Windows
   ```

2. **Reinstall package**:
   ```bash
   pip install -e .
   # Or with dev dependencies:
   pip install -e ".[dev]"
   ```

3. **Check Python version**:
   ```bash
   python --version  # Should be 3.8+
   ```

4. **Clear pip cache** if installation fails:
   ```bash
   pip cache purge
   pip install -e . --no-cache-dir
   ```

---

## Analysis Issues

### "Catalog corrupted"

**Symptom**: Error loading catalog, invalid JSON, or unexpected data structure

**Cause**: Interrupted write, disk full, or software bug

**Solutions**:

1. **Use repair command**:
   ```bash
   vam-analyze /path/to/catalog --repair
   ```

2. **Restore from backup**:
   ```bash
   # Backups are in .catalog.json.backup.TIMESTAMP
   cp .catalog.json.backup.20240115_143000 .catalog.json
   ```

3. **Start fresh** (creates backup first):
   ```bash
   vam-analyze /path/to/catalog --clear -s /path/to/photos
   ```

4. **Check disk space**:
   ```bash
   df -h /path/to/catalog
   ```

---

### "Dates not being extracted"

**Symptom**: Images show "no date" or wrong dates

**Cause**: Missing EXIF data, filename doesn't match patterns, or filesystem dates incorrect

**Solutions**:

1. **Check if image has EXIF data**:
   ```bash
   exiftool image.jpg | grep -i date
   ```

2. **Enable verbose logging**:
   ```bash
   vam-analyze /path/to/catalog -s /path/to/photos -v
   ```
   Look for date extraction details in output

3. **Try different date sources**:
   - Rename files with date: `2023-04-15_photo.jpg`
   - Organize in dated folders: `2023/04/photos/`
   - Manually set dates via web UI

4. **Check file modification dates**:
   ```bash
   ls -l image.jpg  # Unix
   ```

5. **See [Date Extraction Guide](./DATE_EXTRACTION_GUIDE.md)** for detailed troubleshooting

---

### "Analysis hanging or very slow"

**Symptom**: Analysis progress stalls or takes much longer than expected

**Cause**: Network drive latency, too many workers, or system resource exhaustion

**Solutions**:

1. **Check if it's actually progressing**:
   - Watch output for new files being processed
   - Check catalog size: `ls -lh .catalog.json`

2. **Reduce worker count**:
   ```bash
   vam-analyze /path/to/catalog -s /path/to/photos --workers 4
   ```

3. **Check system resources**:
   ```bash
   htop  # or top
   ```
   Look for high CPU/memory usage

4. **Network drives**: Consider local copy first:
   ```bash
   rsync -av /network/photos /local/temp/
   vam-analyze /local/catalog -s /local/temp/
   ```

5. **Enable verbose mode** to see where it's stuck:
   ```bash
   vam-analyze /path/to/catalog -s /path/to/photos -v
   ```

---

## Web Interface Issues

### "Timeout extracting preview from ARW/RAW files"

**Symptom**: 504 Gateway Timeout error when viewing RAW files in web UI

**Cause**: ExifTool extraction taking >30 seconds (common on network drives or large files)

**Expected Behavior**: This is a known limitation for slow storage

**Solutions**:

1. **Accept the timeout**: It's a safety mechanism
   - Browse other images while RAW previews are skipped
   - Use JPEG copies for web browsing

2. **Enable preview caching** (planned feature):
   - Will cache extracted previews for instant viewing
   - Track progress: [Issue #TBD]

3. **Convert RAWs to JPEG for browsing**:
   ```bash
   # Create browse copies with dcraw or rawtherapee
   dcraw -w -T -q 3 *.ARW
   ```

4. **Use SSD for better performance**:
   - Network drives and HDDs are slow for RAW extraction

---

### "Web interface not loading"

**Symptom**: Browser shows "Connection refused" or "Cannot connect"

**Cause**: Server not running, wrong port, or firewall blocking

**Solutions**:

1. **Verify server is running**:
   ```bash
   vam-web /path/to/catalog
   ```
   Should show: `Server: http://127.0.0.1:8765`

2. **Check correct URL**:
   - Default: `http://localhost:8765`
   - Or: `http://127.0.0.1:8765`

3. **Try different port** if 8765 is in use:
   ```bash
   vam-web /path/to/catalog --port 8080
   ```

4. **Check firewall**:
   ```bash
   sudo ufw allow 8765  # Linux
   ```

5. **Check process is listening**:
   ```bash
   lsof -i :8765  # Should show Python process
   ```

---

### "Images not displaying in web interface"

**Symptom**: Image placeholders or broken image icons

**Cause**: Missing files, incorrect paths, or permission issues

**Solutions**:

1. **Check if files exist**:
   ```bash
   # From catalog, find a file path and verify:
   ls -l /path/from/catalog/image.jpg
   ```

2. **Check file permissions**:
   ```bash
   # Ensure web server can read files:
   chmod +r /path/to/images/*
   ```

3. **Verify paths in catalog are correct**:
   - If files were moved, paths in catalog are now incorrect
   - Rerun analysis to update paths

4. **Check browser console** (F12) for error messages

---

## GPU Issues

### "GPU not detected"

**Symptom**: Analysis shows "GPU: Not available" or uses CPU-only

**Cause**: CUDA not installed, incompatible GPU, or PyTorch not built with CUDA

**Solutions**:

1. **Verify GPU exists**:
   ```bash
   nvidia-smi
   ```
   Should show GPU model and driver version

2. **Check CUDA compatibility**:
   ```bash
   nvcc --version  # Should match PyTorch requirements
   ```

3. **Test PyTorch CUDA**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))
   ```

4. **Reinstall PyTorch with CUDA**:
   ```bash
   pip uninstall torch torchvision
   # Get correct command from https://pytorch.org/get-started/locally/
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Check GPU compute capability**:
   - RTX 5060 Ti (sm_120): **Not supported by PyTorch 2.6.0**
   - Requires PyTorch with sm_120 support
   - See [GPU Setup Guide](./GPU_SETUP_GUIDE.md)

---

### "CUDA out of memory"

**Symptom**: GPU memory error during analysis

**Cause**: GPU memory exhausted by large batches or other processes

**Solutions**:

1. **Reduce batch size**:
   ```bash
   # Modify in code or wait for CLI option
   # Current default: 32 images per batch
   ```

2. **Close other GPU applications**:
   ```bash
   nvidia-smi  # Check what's using GPU
   # Kill other processes if needed
   ```

3. **Use CPU fallback**:
   - VAM Tools automatically falls back to CPU if GPU fails
   - Slower but will complete

4. **Upgrade GPU** or reduce workload

---

## Performance Issues

### "Analysis is slower than expected"

**Symptom**: Lower throughput than benchmarks suggest

**Cause**: I/O bottleneck, insufficient workers, or CPU contention

**Solutions**:

1. **Check storage speed**:
   ```bash
   # Test read speed:
   dd if=/path/to/large/file of=/dev/null bs=1M count=1000
   ```
   - HDD: 50-150 MB/s
   - SSD: 200-500 MB/s
   - NVMe: 1000-3000 MB/s

2. **Optimize worker count**:
   ```bash
   # Try different values:
   vam-analyze /path/to/catalog -s /path/to/photos --workers 16
   ```
   Optimal: Usually 1-2x CPU core count

3. **Use SSD for catalog** even if photos on HDD:
   ```bash
   vam-analyze /fast/ssd/catalog -s /slow/hdd/photos
   ```

4. **Enable GPU acceleration** if available:
   ```bash
   # See GPU Setup Guide
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Check system resources**:
   ```bash
   htop  # Look for CPU/memory/I/O bottlenecks
   ```

---

### "High memory usage"

**Symptom**: System runs out of RAM during analysis

**Cause**: Too many workers, large catalog, or memory leak

**Solutions**:

1. **Reduce worker count**:
   ```bash
   vam-analyze /path/to/catalog -s /path/to/photos --workers 4
   ```

2. **Checkpoint more frequently** (planned feature):
   - Currently checkpoints every 100 files
   - Will be configurable

3. **Close other applications**:
   ```bash
   free -h  # Check available memory
   ```

4. **Increase swap space** (Linux):
   ```bash
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

## Data Issues

### "Duplicate detection not finding duplicates"

**Symptom**: Known duplicates not grouped together

**Cause**: Similarity threshold too strict, perceptual hashes not computed

**Solutions**:

1. **Check if hashing is enabled**:
   ```bash
   vam-analyze /path/to/catalog -s /path/to/photos --detect-duplicates
   ```

2. **Adjust similarity threshold**:
   ```bash
   # Default: 5 (stricter)
   # Try higher for more matches:
   vam-analyze /path/to/catalog -s /path/to/photos \
     --detect-duplicates --similarity-threshold 10
   ```

3. **Check catalog for hashes**:
   ```bash
   # Look for perceptual_hash fields:
   grep -i "perceptual_hash" .catalog.json
   ```

4. **Verify images are actually similar**:
   - Different crops/rotations may not match
   - Heavy edits may exceed threshold

---

### "All dates showing as filesystem dates"

**Symptom**: Low confidence dates, all from filesystem

**Cause**: Images have no EXIF data (stripped or never had it)

**Solutions**:

1. **Restore from original files** if available

2. **Use filename/directory dates**:
   - Rename files: `2023-04-15_photo.jpg`
   - Organize folders: `2023/04/`

3. **Manually set dates** via web UI

4. **Recover EXIF** from backups if files were edited without preserving metadata

---

## Getting Help

If you're still stuck after trying these solutions:

### 1. Search Existing Issues

Check if others have reported the same problem:
- [GitHub Issues](https://github.com/irjudson/vam-tools/issues)

### 2. Gather Diagnostic Information

Before reporting an issue, collect:

```bash
# System info:
uname -a
python --version
pip list | grep -i vam

# ExifTool version:
exiftool -ver

# GPU info (if applicable):
nvidia-smi

# Run with verbose logging:
vam-analyze /path/to/catalog -s /path/to/photos -v > debug.log 2>&1
```

### 3. Open an Issue

[Create a new issue](https://github.com/irjudson/vam-tools/issues/new) with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Diagnostic information from above
- Relevant log excerpts (use code blocks)

### 4. Community Discussion

For questions and general discussion:
- [GitHub Discussions](https://github.com/irjudson/vam-tools/discussions)

---

## See Also

- [User Guide](./USER_GUIDE.md) - General usage instructions
- [GPU Setup Guide](./GPU_SETUP_GUIDE.md) - GPU troubleshooting
- [Date Extraction Guide](./DATE_EXTRACTION_GUIDE.md) - Date-specific issues
- [Performance & GPU Summary](./PERFORMANCE_AND_GPU_SUMMARY.md) - Optimization tips
