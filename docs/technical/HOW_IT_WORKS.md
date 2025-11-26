# How VAM Tools Works

This document explains the technical details of how VAM Tools analyzes and catalogs your photo library.

## Analysis Pipeline

The analysis process performs the following steps:

### 1. File Discovery

Scans directories recursively for image and video files. Supports all common formats including RAW, JPEG, PNG, HEIC, and video files.

### 2. Parallel Processing

Uses a worker pool to process files in parallel across multiple CPU cores:

- **Checksum Computation** - SHA-256 hash for duplicate detection
- **Metadata Extraction** - Comprehensive EXIF/XMP data via ExifTool
- **Date Extraction** - Multiple sources (EXIF, filename, directory, filesystem)
- **Quality Analysis** - Resolution, format, and metadata completeness scoring

Worker processes operate independently on separate files, maximizing CPU utilization.

### 3. GPU Acceleration (Optional)

When a compatible NVIDIA GPU is detected:

- **PyTorch-based Hashing** - Perceptual hash computation (dHash, aHash, wHash) runs on GPU
- **Batch Processing** - Multiple images processed simultaneously
- **20-30x Speedup** - Typical acceleration over CPU-based hashing
- **FAISS Integration** - GPU-accelerated similarity search for large catalogs

See [GPU Setup Guide](./GPU_SETUP_GUIDE.md) for detailed configuration.

### 4. Catalog Building

Creates a comprehensive catalog database with:

- **Image/Video Records** - Indexed by checksum for fast lookups
- **Metadata Storage** - All EXIF/XMP data preserved
- **Date Information** - Selected date with confidence level and all candidate dates
- **Quality Scores** - Format, resolution, and metadata completeness ratings
- **Perceptual Hashes** - For similarity-based duplicate detection
- **Performance Statistics** - Timing data and throughput metrics

The catalog is stored as a JSON file (`.catalog.json`) with automatic backups.

### 5. Real-Time Monitoring

During analysis, the system tracks and displays:

- **Throughput** - Files processed per second
- **Progress** - Files completed and remaining
- **GPU Utilization** - GPU memory and compute usage
- **Memory Usage** - System memory consumption
- **Operation Timing** - Time spent in each operation (metadata, hashing, etc.)
- **Data Throughput** - Gigabytes per second processed
- **Bottleneck Analysis** - Identifies slowest operations

Performance data is written to the catalog every 5 seconds and available via the web API.

### 6. Incremental Updates

Subsequent scans intelligently update the catalog:

- **Checksum Comparison** - Only processes files not already in catalog
- **Changed File Detection** - Re-analyzes files with modified timestamps
- **New File Addition** - Adds newly discovered files
- **Duplicate Recomputation** - Updates duplicate groups when new files added
- **Fast Rescan** - Typically 10-100x faster than initial scan

## Processing Flow Diagram

```
┌──────────────┐
│ File Scanner │ ──┐
└──────────────┘   │
                   ▼
        ┌──────────────────┐
        │  Worker Pool      │
        │  (N workers)      │
        └──────────────────┘
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Worker 1│  │ Worker 2│  │ Worker N│
│ Process │  │ Process │  │ Process │
└─────────┘  └─────────┘  └─────────┘
     │             │             │
     │  Checksum   │  Metadata   │  Quality
     │  Hashing    │  Extraction │  Scoring
     │  Dates      │             │
     │             │             │
     └─────────────┼─────────────┘
                   ▼
        ┌──────────────────┐
        │  Catalog Builder │
        └──────────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │  .catalog.json   │
        └──────────────────┘
```

## Performance Characteristics

### Single-Threaded (1 worker)
- **Throughput**: ~5-10 files/second
- **Bottleneck**: CPU-bound metadata extraction

### Multi-Threaded (32 workers)
- **Throughput**: ~100-150 files/second (20-30x speedup)
- **Bottleneck**: I/O for reading files, especially on HDD

### With GPU Acceleration
- **Throughput**: ~150-200 files/second
- **Perceptual Hashing**: 20-30x faster than CPU
- **Bottleneck**: I/O for large files (RAW, video)

### Optimal Configuration
- **Workers**: Set to CPU core count or 2x core count
- **Storage**: SSD significantly improves throughput
- **GPU**: NVIDIA GPU with CUDA support for perceptual hashing
- **RAM**: 4-8 GB minimum, more for large catalogs

## Technical Implementation

### Concurrency Model
- **Multiprocessing** - Multiple worker processes for parallelism
- **Process Pool** - Managed by Python's `multiprocessing.Pool`
- **Queue-based** - Work items distributed via process-safe queue
- **Lock-free Reads** - Catalog reads don't require locks
- **Write Serialization** - Catalog writes are serialized with file locking

### Catalog Storage
- **Format**: JSON with pretty printing for human readability
- **Schema**: Pydantic v2 models for type safety
- **Versioning**: Schema version tracked for future migrations
- **Backup**: Automatic backup before any destructive operation
- **Checkpoints**: Periodic saves during analysis (every 100 files)

### Error Handling
- **Graceful Degradation** - Failed files logged but don't stop analysis
- **Retry Logic** - Transient errors (network, permissions) retried
- **Validation** - All data validated with Pydantic before catalog write
- **Recovery** - Interrupted scans can be resumed from last checkpoint

## See Also

- [Architecture](./ARCHITECTURE.md) - System design and components
- [Performance & GPU Summary](./PERFORMANCE_AND_GPU_SUMMARY.md) - Optimization guide
- [Date Extraction Guide](./DATE_EXTRACTION_GUIDE.md) - Date extraction details
