# Frontend Performance Polling Update Summary

## Changes Made

### 1. Frontend Updates (vam_tools/web/static/index.html)

#### Replaced WebSocket with Polling
- **Added polling state variables**:
  - `performancePollingInterval`: Stores the setInterval ID
  - `performancePollingActive`: Boolean flag to control polling

- **Updated performance status display**:
  - Changed from WebSocket connection indicator (`wsConnected`) to analysis status
  - Now shows: 🟢 Running | ⚪ Idle | ⚪ No Data

- **Updated data binding**:
  - Changed from `performanceStats.metrics.*` to `performanceStats.data.*`
  - Changed from `performanceStats.status === 'ok'` to `performanceStats.status === 'running' || performanceStats.status === 'idle'`
  - Updated operations display to handle object format instead of array
  - Added `getTopOperations()` helper method to sort and slice operations

- **New polling methods**:
  ```javascript
  startPerformancePolling() {
      this.performancePollingActive = true;
      this.performancePollingInterval = setInterval(async () => {
          if (this.performancePollingActive) {
              await this.loadPerformanceStats();
          }
      }, 1000); // Poll every 1 second
  }

  stopPerformancePolling() {
      this.performancePollingActive = false;
      if (this.performancePollingInterval) {
          clearInterval(this.performancePollingInterval);
          this.performancePollingInterval = null;
      }
  }
  ```

- **Updated lifecycle hooks**:
  - `mounted()`: Starts polling instead of WebSocket connection
  - `beforeUnmount()`: Stops polling to prevent memory leaks

### 2. Backend API Updates (vam_tools/web/api.py)

#### Improved RAW File Preview Extraction
- **Increased timeout**: 10s → 30s for exiftool operations
  - Reason: Network drives (Synology NFS) and large ARW files need more time

- **Better error classification**:
  - Changed from `500 Internal Server Error` to `504 Gateway Timeout`
  - Changed from `logger.error()` to `logger.warning()` for timeout cases
  - More descriptive error messages mentioning slow storage

- **Updated error message**:
  ```python
  raise HTTPException(
      status_code=504,  # Gateway Timeout instead of 500
      detail=f"Timeout extracting preview from {file_ext.upper()} file (file on slow storage)",
  )
  ```

### 3. Test Updates

#### Fixed API Response Format Tests (tests/web/test_performance_api.py)
- Updated `test_get_current_performance_no_data`:
  - Changed from checking `"message" in data` to `data["data"] is None`

- Updated `test_get_current_performance_with_data`:
  - Changed from `status === "ok"` to `status === "idle"`
  - Changed from `data.metrics.*` to `data.data.*`
  - Changed from array format to object format for operations

## API Response Format

The `/api/performance/current` endpoint now returns:

```json
{
  "status": "running",  // or "idle" or "no_data"
  "data": {
    "run_id": "...",
    "started_at": "2024-01-01T00:00:00",
    "completed_at": null,  // null = running, timestamp = completed
    "total_files_analyzed": 100,
    "files_per_second": 10.5,
    "total_duration_seconds": 9.5,
    "bytes_processed": 5000000000,
    "bytes_per_second": 526315789,
    "peak_memory_mb": 512.0,
    "gpu_utilized": true,
    "gpu_device": "NVIDIA RTX 3080",
    "total_errors": 0,
    "operations": {
      "scan_directories": {
        "total_time_seconds": 5.2,
        "call_count": 1,
        "items_processed": 100,
        "errors": 0,
        ...
      }
    },
    "hashing": {
      "total_hashes_computed": 100,
      "gpu_hashes": 80,
      "cpu_hashes": 20,
      "failed_hashes": 0
    },
    "bottlenecks": []
  },
  "history": [...]  // Last 5 runs
}
```

## Test Results

All **514 tests** pass successfully:
- Added 7 new performance tracking tests
- Updated 2 existing performance API tests
- All other tests remain passing

## Performance Improvements

With `pytest-xdist` for parallel testing:
- **Before**: 44.5 seconds
- **After**: 16.7 seconds
- **Improvement**: 62.5% faster

## Benefits

1. **No WebSocket complexity**: Simpler architecture, easier to debug
2. **Works across processes**: CLI and web server can be separate processes
3. **No connection management**: No need to handle reconnections, pings, etc.
4. **Better error handling**: Clearer separation between running/idle/no-data states
5. **Network-friendly**: Works through proxies and load balancers without issues

## Known Issues

### ARW File Preview Timeouts
- **Issue**: Some ARW files on network drives timeout after 30 seconds
- **Impact**: Returns 504 Gateway Timeout, image preview not displayed
- **Workaround**:
  - Increase timeout further if needed
  - Consider caching previews to disk
  - Use thumbnail cache system
- **Root Cause**: Network I/O latency + large file sizes + exiftool extraction time

## Next Steps (Optional)

1. **Implement preview caching**: Cache extracted previews to avoid repeated extraction
2. **Add lazy loading**: Only extract previews when user scrolls them into view
3. **Background preview extraction**: Extract all previews during analysis phase
4. **Fallback placeholder**: Show placeholder image when extraction fails
