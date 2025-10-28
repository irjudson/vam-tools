# VAM Tools (Visual Asset Management)

A professional collection of Python tools for managing and organizing photo/video libraries with GPU acceleration and real-time performance monitoring.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-518%20passing-success.svg)](https://github.com/irjudson/vam-tools)
[![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen.svg)](https://github.com/irjudson/vam-tools)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents

- [Documentation](#documentation)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Common Workflows](#common-workflows)
- [How It Works](#how-it-works)
- [Date Extraction](#date-extraction)
- [Supported Formats](#supported-formats)
- [Development](#development)
- [TODO](#todo)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Documentation

Comprehensive documentation is available in the [docs](./docs) directory:

### User Documentation
- **[User Guide](./docs/USER_GUIDE.md)** - Complete user documentation, tutorials, and common tasks
- **[Requirements](./docs/REQUIREMENTS.md)** - Product requirements, features, and roadmap

### Technical Documentation
- **[Architecture](./docs/ARCHITECTURE.md)** - Technical design, system components, and implementation details
- **[GPU Setup Guide](./docs/GPU_SETUP_GUIDE.md)** - GPU acceleration setup and configuration
- **[GPU Acceleration Plan](./docs/GPU_ACCELERATION_PLAN.md)** - GPU implementation details
- **[Performance & GPU Summary](./docs/PERFORMANCE_AND_GPU_SUMMARY.md)** - Performance optimization guide

### Development Documentation
- **[Contributing Guide](./docs/CONTRIBUTING.md)** - Development setup, testing, and contribution guidelines
- **[Project Notes](./docs/NOTES.md)** - Historical notes and implementation summaries
- **[Frontend Polling Update](./docs/FRONTEND_POLLING_UPDATE.md)** - Real-time performance monitoring implementation
- **[Performance Widget Fix](./docs/PERFORMANCE_WIDGET_FIX.md)** - Multi-process communication solution

## Features

### Core Functionality
- **High-Performance Scanning** - Multi-core parallel processing for fast catalog analysis
- **Comprehensive Metadata Extraction** - Extract dates from EXIF, XMP, filenames, and directory structure using ExifTool
- **Duplicate Detection** - Find exact and similar duplicates using checksums and perceptual hashing (dHash/aHash/wHash)
- **Quality Scoring** - Automatically select the best copy among duplicates based on format, resolution, and metadata
- **Date-Based Reorganization** - Reorganize photo/video catalogs into date-based directory structures

### Advanced Features
- **GPU Acceleration** - PyTorch-based GPU acceleration for perceptual hashing (20-30x faster on compatible GPUs)
- **Web Interface** - Modern Vue.js web UI with real-time performance monitoring
- **Real-Time Performance Tracking** - Live throughput, GPU utilization, and bottleneck analysis
- **FAISS Similarity Search** - GPU-accelerated similarity search for large catalogs (millions of images)
- **Beautiful CLI** - Rich terminal interface with progress bars and formatted output

### Quality & Testing
- **Fully Tested** - Comprehensive test suite with **518 passing tests** and **84% coverage**
- **Type Safe** - Full type hints throughout the codebase with Pydantic v2
- **Fast Tests** - Parallel test execution with pytest-xdist (62.5% faster)

## Installation

### Prerequisites

- Python 3.8 or higher
- [ExifTool](https://exiftool.org/) must be installed on your system
- (Optional) NVIDIA GPU with CUDA support for GPU acceleration

#### Installing ExifTool

**macOS (Homebrew):**
```bash
brew install exiftool
```

**Ubuntu/Debian:**
```bash
sudo apt-get install exiftool
```

**Windows:**
Download from [exiftool.org](https://exiftool.org/) and add to PATH.

### Install VAM Tools

```bash
# Clone the repository
git clone https://github.com/irjudson/vam-tools.git
cd vam-tools

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Optional: GPU Acceleration Setup

For GPU-accelerated perceptual hashing (20-30x faster):

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install FAISS for GPU similarity search
pip install faiss-gpu
```

See [GPU Setup Guide](./docs/GPU_SETUP_GUIDE.md) for detailed instructions.

## Quick Start

### Basic Workflow

1. **Analyze your photo library** to build a catalog
2. **Launch the web interface** to browse and review with real-time performance monitoring
3. **Find duplicates** and identify the best copies
4. **Reorganize** your library with date-based structure (optional)

### Analyze and Build Catalog

Start by scanning your photo directories to build a catalog:

```bash
# Analyze photos from one or more source directories
vam-analyze /path/to/catalog -s /path/to/photos

# Recommended: Use all CPU cores with duplicate detection
vam-analyze /path/to/catalog -s /path/to/photos --detect-duplicates -v

# For large libraries: specify worker count for multi-core systems
vam-analyze /path/to/catalog -s /path/to/photos --workers 32 --detect-duplicates
```

**Performance:**
- On a 32-core system, expect 20-30x speedup compared to single-threaded processing
- With GPU acceleration, perceptual hashing is 20-30x faster than CPU
- Real-time performance monitoring shows throughput, GPU utilization, and bottlenecks

### Browse Your Catalog

Launch the web interface to explore your catalog:

```bash
# Start web server (opens at http://localhost:5000)
vam-web /path/to/catalog

# Custom port or allow external access
vam-web /path/to/catalog --port 8080 --host 0.0.0.0
```

The web interface provides:
- Browse all images and videos with metadata
- View duplicate groups with side-by-side comparison
- See date extraction results and confidence levels
- Review statistics and storage analysis
- **Real-time performance monitoring** - Live throughput, memory usage, GPU utilization
- Beautiful charts and dashboard

## Common Workflows

### Find and Review Duplicates

```bash
# Analyze with duplicate detection
vam-analyze /path/to/catalog -s /path/to/photos --detect-duplicates

# Adjust similarity threshold (default: 5, lower = more strict)
vam-analyze /path/to/catalog -s /path/to/photos \
  --detect-duplicates \
  --similarity-threshold 3

# Launch web UI to review duplicates
vam-web /path/to/catalog
# Navigate to "View Duplicates" to see groups and recommended deletions
```

**Duplicate Detection:** Uses perceptual hashing (dHash, aHash, and wHash) with quality scoring to identify duplicates and automatically select the best copy based on format (RAW > JPEG), resolution, file size, and metadata completeness.

### Incremental Updates

Add new photos to an existing catalog without reprocessing:

```bash
# Scan will skip files already in catalog
vam-analyze /path/to/catalog -s /path/to/photos
```

### Manage Your Catalog

```bash
# Start fresh (clears existing catalog, creates backup first)
vam-analyze /path/to/catalog -s /path/to/photos --clear

# Repair corrupted catalog
vam-analyze /path/to/catalog --repair

# Verbose logging for troubleshooting
vam-analyze /path/to/catalog -s /path/to/photos -v
```

## How It Works

The analysis process performs the following steps:

1. **File Discovery** - Scans directories for image and video files
2. **Parallel Processing** - Uses worker pool to process files in parallel:
   - Computes checksums (for duplicate detection)
   - Extracts comprehensive metadata via ExifTool
   - Extracts dates from multiple sources
3. **GPU Acceleration** (if available) - Accelerates perceptual hashing with PyTorch
4. **Catalog Building** - Creates a catalog database with:
   - Image/video records indexed by checksum
   - Comprehensive metadata for each file
   - Date information with confidence levels
   - Performance statistics and timing data
5. **Real-Time Monitoring** - Tracks and displays:
   - Files processed per second
   - GPU utilization and memory usage
   - Operation timing and bottlenecks
   - Data throughput (GB/s)
6. **Incremental Updates** - Rescan to add new files without reprocessing existing ones

## Date Extraction

Dates are extracted from multiple sources with confidence levels:

1. **EXIF Metadata** (95% confidence) - Camera timestamps from EXIF/XMP data
2. **Filename Patterns** (70% confidence) - Dates in filenames (YYYY-MM-DD, YYYYMMDD, etc.)
3. **Directory Structure** (50% confidence) - Year/month from folder names
4. **Filesystem Metadata** (30% confidence) - File creation/modification time

The system selects the **earliest date** from all available sources.

## Supported Formats

- **Images:** JPEG, PNG, TIFF, BMP, GIF, WEBP, HEIC/HEIF
- **RAW Formats:** CR2, CR3, NEF, ARW, DNG, ORF, RW2, PEF, SR2, RAF, and more
- **Videos:** MP4, MOV, AVI, MKV, and more

## Development

### Setup Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests (parallel execution)
pytest -n auto

# Run all tests (sequential)
pytest

# Run with coverage
pytest --cov=vam_tools --cov-report=html

# Run specific test file
pytest tests/core/test_catalog.py -v

# Run performance benchmark tests
pytest tests/analysis/test_perceptual_hash_performance.py --benchmark-only
```

### Code Quality

The project uses pre-commit hooks to ensure code quality:

```bash
# Format code
black vam_tools/ tests/

# Sort imports
isort vam_tools/ tests/

# Lint code
flake8 vam_tools/ tests/

# Type checking
mypy vam_tools/
```

### Project Structure

```
vam-tools/
├── vam_tools/
│   ├── analysis/             # Scanner, metadata, duplicate detection
│   │   ├── scanner.py            # Multi-core file scanner (80% coverage)
│   │   ├── metadata.py           # ExifTool metadata extraction (91% coverage)
│   │   ├── duplicate_detector.py # Perceptual hash duplicate detection (69% coverage)
│   │   ├── perceptual_hash.py    # dHash, aHash, wHash algorithms (81% coverage)
│   │   ├── quality_scorer.py     # Quality scoring for duplicates (94% coverage)
│   │   ├── gpu_hash.py           # GPU-accelerated hashing (54% coverage)
│   │   └── fast_search.py        # FAISS similarity search (93% coverage)
│   ├── core/                 # Catalog database and types
│   │   ├── catalog.py            # Catalog database with locking (76% coverage)
│   │   ├── types.py              # Pydantic models (100% coverage)
│   │   ├── performance_stats.py  # Performance tracking (96% coverage)
│   │   └── gpu_utils.py          # GPU utilities (76% coverage)
│   ├── cli/                  # Command-line interfaces
│   │   ├── analyze.py            # Analysis CLI (89% coverage)
│   │   └── web.py                # Web server CLI (96% coverage)
│   ├── web/                  # Web interface (FastAPI + Vue.js)
│   │   ├── api.py                # REST API endpoints (82% coverage)
│   │   └── static/index.html     # Vue.js frontend with real-time monitoring
│   └── shared/               # Shared utilities
│       └── media_utils.py        # Image/video utilities (95% coverage)
├── tests/                    # Test suite (518 tests, 84% coverage)
│   ├── analysis/             # Analysis module tests
│   ├── core/                 # Core module tests
│   ├── cli/                  # CLI tests
│   ├── web/                  # Web API tests
│   └── shared/               # Shared utilities tests
├── docs/                     # Documentation
├── pyproject.toml
└── README.md
```

## TODO

### High Priority

- [ ] **Preview Caching System**
  - Cache extracted RAW file previews to avoid repeated extraction
  - Implement LRU cache with configurable size limit
  - Background preview extraction during analysis phase

- [ ] **Auto-Tagging System**
  - Integration with ML models for automatic image tagging
  - Subject detection (people, animals, objects)
  - Scene classification (indoor, outdoor, landscape, etc.)
  - Configurable tagging pipeline

- [ ] **Duplicate Resolution UI**
  - Interactive UI for reviewing and resolving duplicates
  - Batch operations (keep/delete)
  - Undo/redo functionality
  - Safe deletion with trash/backup

### Medium Priority

- [ ] **FAISS Index Persistence**
  - Save/load FAISS indices to disk
  - Incremental index updates for new images
  - Index versioning and migration

- [ ] **Advanced Search**
  - Search by date range
  - Search by metadata (camera, lens, location)
  - Search by quality score
  - Similar image search (reverse image search)

- [ ] **Batch Operations**
  - Bulk metadata editing
  - Batch file operations (move, copy, delete)
  - Transaction support with rollback

- [ ] **Export Functionality**
  - Export catalog to CSV/JSON
  - Export duplicate reports
  - Export statistics and analytics

### Low Priority / Future Ideas

- [ ] **Cloud Integration**
  - Google Photos sync
  - iCloud Photos sync
  - Dropbox/OneDrive integration

- [ ] **Mobile App**
  - React Native mobile viewer
  - Remote catalog access
  - Photo upload from mobile

- [ ] **Advanced Analytics**
  - Photo timeline visualization
  - Storage analysis by date/camera/format
  - Quality distribution charts
  - Duplicate savings projections

- [ ] **Plugin System**
  - Custom metadata extractors
  - Custom duplicate detection algorithms
  - Custom quality scorers
  - Event hooks for automation

- [ ] **Performance Optimizations**
  - Distributed processing (multiple machines)
  - Incremental FAISS index updates
  - Smart caching strategies
  - Background workers for heavy operations

### Documentation Improvements

- [ ] Add video tutorials
- [ ] Create migration guides from other tools (Lightroom, etc.)
- [ ] Add more examples and use cases
- [ ] Create troubleshooting flowcharts

## Best Practices

1. **Always backup your photos before reorganizing**
2. **Use dry-run mode** when testing reorganization
3. **Start with multiple workers** for large catalogs (--workers option)
4. **Review results** in the web UI before making changes
5. **Enable verbose logging** (-v) for troubleshooting
6. **Use incremental scanning** to add new files to existing catalogs
7. **Enable GPU acceleration** if available for 20-30x faster processing
8. **Monitor performance** in real-time via the web dashboard

## Troubleshooting

### Common Issues

**"ExifTool not found"**
- Make sure ExifTool is installed and in your PATH
- Test by running `exiftool -ver` in your terminal

**"Permission denied"**
- Ensure you have read/write permissions for the directories
- On Unix systems, check with `ls -la`

**"Catalog corrupted"**
- Use `vam-analyze /path/to/catalog --repair` to fix
- Or use `--clear` to start fresh (creates backup first)

**"Dates not being extracted"**
- Check that images actually have EXIF data: `exiftool image.jpg`
- Try verbose mode (-v) to see what's being detected

**"GPU not detected"**
- Verify GPU with `nvidia-smi`
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- See [GPU Setup Guide](./docs/GPU_SETUP_GUIDE.md) for detailed troubleshooting

**"Timeout extracting preview from ARW files"**
- Some RAW files on network drives may timeout (30s limit)
- Returns 504 Gateway Timeout (expected for slow storage)
- Consider enabling preview caching (planned feature)

## Performance Tips

- **Use multiprocessing:** Specify `--workers N` where N is your CPU core count
- **Enable GPU acceleration:** 20-30x faster perceptual hashing with compatible GPU
- **Use FAISS for large catalogs:** GPU-accelerated similarity search for millions of images
- **SSD recommended:** Faster I/O significantly improves scanning speed
- **Incremental updates:** Rerun analysis to add only new files
- **Monitor in real-time:** Use web dashboard to identify bottlenecks

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure tests pass (`pytest -n auto`)
6. Format code (`black`, `isort`)
7. Commit your changes
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Uses [Pillow](https://python-pillow.org/) for image processing
- Uses [pillow-heif](https://github.com/bigcat88/pillow_heif) for HEIC support
- Uses [ExifTool](https://exiftool.org/) for comprehensive metadata extraction
- Uses [Click](https://click.palletsprojects.com/) for CLI framework
- Uses [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- Uses [Arrow](https://arrow.readthedocs.io/) for date/time handling
- Uses [FastAPI](https://fastapi.tiangolo.com/) for web interface
- Uses [Pydantic](https://docs.pydantic.dev/) for data validation
- Uses [PyTorch](https://pytorch.org/) for GPU acceleration
- Uses [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- Uses [Vue.js](https://vuejs.org/) for frontend

## Author

Ivan R. Judson - [irjudson@gmail.com](mailto:irjudson@gmail.com)

## Development Approach

This project was developed using human-AI pair programming with Claude. The collaboration followed established engineering principles to ensure code quality without requiring exhaustive human review of every line.

### Core Principles

1. **Non-Destructive by Default**
   - All operations provide dry-run modes for safe testing
   - Destructive operations require explicit flags (off by default)
   - Automatic backups before any catalog modifications

2. **Minimal Functioning Code**
   - DRY (Don't Repeat Yourself) - shared utilities, no duplication
   - Clean architecture with clear separation of concerns
   - Type-safe with Pydantic models throughout

3. **Quality Gates from Day One**
   - Tests and CI implemented at project start, not added later
   - Pre-push hooks prevent broken code from reaching repository
   - GitHub Actions run full test suite on every commit

### Development Cycle

Each feature followed this iterative cycle:

1. **Prototype** - Initial implementation with core functionality
2. **Validate** - Human review of architecture and approach
3. **Develop** - Complete implementation with error handling
4. **Test** - Comprehensive test coverage (518 tests, 84% coverage)
5. **Refactor** - Clean up, optimize, ensure DRY principles

### How Code Quality Was Ensured

Rather than reviewing every line of AI-generated code, quality assurance came from:

- **Continuous Integration**: Every commit runs Black, isort, flake8, pytest, and coverage checks
- **Test Coverage Requirements**: 80%+ coverage enforced, comprehensive test suite validates correctness
- **Pre-Push Hooks**: Local quality gates catch issues before they reach GitHub
- **Type Safety**: Full type hints with Pydantic v2 catch type errors at development time
- **Incremental Development**: Small commits with focused changes, easy to validate

### Efficiency Gains

This approach enabled building a production-ready tool in a single weekend—a timeline that would typically require weeks:

- **Automated Boilerplate**: AI handled repetitive code patterns
- **Parallel Development**: Tests written simultaneously with implementation
- **Instant Documentation**: README and docstrings generated from implementation
- **Rapid Iteration**: Quick prototype-validate-refactor cycles

### Key Takeaway

**The AI accelerated development; the test suite ensured quality.** Instead of manually reviewing every line, automated testing and CI gates provided confidence in code correctness.

## Project Links

- **Repository**: https://github.com/irjudson/vam-tools
- **Issues**: https://github.com/irjudson/vam-tools/issues
