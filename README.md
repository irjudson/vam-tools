# VAM Tools (Visual Asset Management)

A professional collection of Python tools for managing and organizing photo/video libraries.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-194%20passing-success.svg)](https://github.com/irjudson/vam-tools)
[![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen.svg)](https://github.com/irjudson/vam-tools)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

- **High-Performance Scanning** - Multi-core parallel processing for fast catalog analysis
- **Comprehensive Metadata Extraction** - Extract dates from EXIF, XMP, filenames, and directory structure using ExifTool
- **Duplicate Detection** - Find exact and similar duplicates using checksums and perceptual hashing (dHash/aHash)
- **Quality Scoring** - Automatically select the best copy among duplicates based on format, resolution, and metadata
- **Date-Based Reorganization** - Reorganize photo/video catalogs into date-based directory structures
- **Web Interface** - Modern web UI for reviewing and managing your catalog
- **Beautiful CLI** - Rich terminal interface with progress bars and formatted output
- **Fully Tested** - Comprehensive test suite with 194 passing tests and 84% coverage
- **Type Safe** - Full type hints throughout the codebase with Pydantic v2

## Installation

### Prerequisites

- Python 3.8 or higher
- [ExifTool](https://exiftool.org/) must be installed on your system

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

## Usage

VAM Tools provides two main interfaces:

### 1. V2 Catalog Analysis (Recommended)

The V2 system provides high-performance catalog scanning with multiprocessing support.

#### Analyze and Build Catalog

```bash
# Analyze photos from one or more source directories
vam-analyze /path/to/catalog -s /path/to/photos -s /path/to/more/photos

# Use all CPU cores with duplicate detection (recommended)
vam-analyze /path/to/catalog -s /path/to/photos --detect-duplicates -v

# Specify number of workers (great for multi-core systems)
vam-analyze /path/to/catalog -s /path/to/photos --workers 32 --detect-duplicates

# Customize similarity threshold (default: 5, lower = more strict)
vam-analyze /path/to/catalog -s /path/to/photos \
  --detect-duplicates \
  --similarity-threshold 3

# Start fresh (clears existing catalog, creates backup)
vam-analyze /path/to/catalog -s /path/to/photos --clear

# Repair corrupted catalog
vam-analyze /path/to/catalog --repair

# Verbose logging
vam-analyze /path/to/catalog -s /path/to/photos -v
```

**Performance:** On a 32-core system, expect 20-30x speedup compared to single-threaded processing.

**Duplicate Detection:** Uses both perceptual hashing (dHash and aHash) and quality scoring to identify duplicates and automatically select the best copy based on format (RAW > JPEG), resolution, file size, and metadata completeness.

#### Web Interface

Launch the web UI to browse and manage your catalog:

```bash
# Start web server
vam-web /path/to/catalog

# Custom port
vam-web /path/to/catalog --port 8080

# Allow external access
vam-web /path/to/catalog --host 0.0.0.0
```

Then open your browser to http://localhost:5000 to view your catalog.

### 2. V1 Legacy Tools

The V1 tools provide the original date analysis, duplicate detection, and reorganization features via an interactive menu:

```bash
# Launch interactive menu
vam-v1
```

The menu provides access to:
- **Image Date Analyzer** - Extract and analyze dates from images
- **Duplicate Image Finder** - Find duplicate and similar images
- **Catalog Reorganizer** - Reorganize photos into date-based structure

## Catalog Analysis Process

The V2 system performs the following steps:

1. **File Discovery** - Scans directories for image and video files
2. **Parallel Processing** - Uses worker pool to process files in parallel:
   - Computes checksums (for duplicate detection)
   - Extracts comprehensive metadata via ExifTool
   - Extracts dates from multiple sources
3. **Catalog Building** - Creates a catalog database with:
   - Image/video records indexed by checksum
   - Comprehensive metadata for each file
   - Date information with confidence levels
   - Statistics (total images, videos, size, etc.)
4. **Incremental Updates** - Rescan to add new files without reprocessing existing ones

## Date Extraction

Dates are extracted from multiple sources with confidence levels:

1. **EXIF Metadata** (95% confidence) - Camera timestamps from EXIF/XMP data
2. **Filename Patterns** (70% confidence) - Dates in filenames (YYYY-MM-DD, YYYYMMDD, etc.)
3. **Directory Structure** (50% confidence) - Year/month from folder names
4. **Filesystem Metadata** (30% confidence) - File creation/modification time

The system selects the **earliest date** from all available sources.

## Supported Formats

- **Images:** JPEG, PNG, TIFF, BMP, GIF, WEBP, HEIC/HEIF
- **RAW Formats:** CR2, NEF, ARW, DNG, and more
- **Videos:** MP4, MOV, AVI, MKV, and more

## Development

### Setup Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vam_tools --cov-report=html

# Run specific test file
pytest tests/core/test_date_extraction.py

# Run with verbose output
pytest -v
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
│   │   ├── scanner.py            # Multi-core file scanner (69% coverage)
│   │   ├── metadata.py           # ExifTool metadata extraction (80% coverage)
│   │   ├── duplicate_detector.py # Perceptual hash duplicate detection (89% coverage)
│   │   ├── perceptual_hash.py    # dHash and aHash algorithms (91% coverage)
│   │   └── quality_scorer.py     # Quality scoring for duplicates (89% coverage)
│   ├── core/                 # Catalog database and types
│   │   ├── catalog.py            # Catalog database with locking (75% coverage)
│   │   └── types.py              # Pydantic models (100% coverage)
│   ├── cli/                  # Command-line interfaces
│   │   ├── analyze.py            # Analysis CLI (79% coverage)
│   │   └── web.py                # Web server CLI (96% coverage)
│   ├── web/                  # Web interface (FastAPI)
│   │   └── api.py                # REST API endpoints (80% coverage)
│   └── shared/               # Shared utilities
│       └── media_utils.py        # Image/video utilities (95% coverage)
├── tests/                    # Test suite (194 tests, 84% coverage)
│   ├── analysis/             # Analysis module tests
│   ├── core/                 # Core module tests
│   ├── cli/                  # CLI tests
│   ├── web/                  # Web API tests
│   └── shared/               # Shared utilities tests
├── docs/                     # Documentation
├── pyproject.toml
└── README.md
```

## Best Practices

1. **Always backup your photos before reorganizing**
2. **Use dry-run mode** when testing reorganization
3. **Start with multiple workers** for large catalogs (--workers option)
4. **Review results** in the web UI before making changes
5. **Enable verbose logging** (-v) for troubleshooting
6. **Use incremental scanning** to add new files to existing catalogs

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

## Performance Tips

- **Use multiprocessing:** Specify `--workers N` where N is your CPU core count
- **Process in batches:** For very large libraries, process subdirectories separately
- **SSD recommended:** Faster I/O significantly improves scanning speed
- **Incremental updates:** Rerun analysis to add only new files

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure tests pass (`pytest`)
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

## Author

Ivan R. Judson - [irjudson@gmail.com](mailto:irjudson@gmail.com)

## Development Approach

This project was developed over a weekend using human-AI pair programming with Claude. The collaboration followed established engineering principles to ensure code quality without requiring exhaustive human review of every line.

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
4. **Test** - Comprehensive test coverage (213 tests, 84% coverage)
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
