# VAM Tools (Visual Asset Management)

A professional collection of Python tools for managing and organizing photo/video libraries with GPU acceleration and real-time performance monitoring.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-580%20passing-success.svg)](https://github.com/irjudson/vam-tools)
[![Coverage](https://img.shields.io/badge/coverage-78%25-green.svg)](https://github.com/irjudson/vam-tools)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Quick Links

📖 **[Full Documentation](./docs/)** | 🚀 **[Quick Start](#quick-start)** | 🐛 **[Issues](https://github.com/irjudson/vam-tools/issues)** | 💬 **[Discussions](https://github.com/irjudson/vam-tools/discussions)**

---

## Documentation

Comprehensive documentation is available in the **[docs](./docs)** directory:

### Getting Started
- **[User Guide](./docs/USER_GUIDE.md)** - Complete user documentation and tutorials
- **[Installation](#installation)** - Get up and running (see below)
- **[Quick Start](#quick-start)** - Basic workflow in 5 minutes

### Technical Guides
- **[How It Works](./docs/HOW_IT_WORKS.md)** - Analysis pipeline and processing details
- **[Date Extraction Guide](./docs/DATE_EXTRACTION_GUIDE.md)** - Date detection and confidence levels
- **[Architecture](./docs/ARCHITECTURE.md)** - System design and components
- **[GPU Setup Guide](./docs/GPU_SETUP_GUIDE.md)** - GPU acceleration configuration
- **[Performance & GPU Summary](./docs/PERFORMANCE_AND_GPU_SUMMARY.md)** - Optimization guide

### Help & Reference
- **[Troubleshooting](./docs/TROUBLESHOOTING.md)** - Common problems and solutions
- **[Roadmap](./docs/ROADMAP.md)** - Planned features and priorities
- **[Contributing Guide](./docs/CONTRIBUTING.md)** - Development setup and guidelines
- **[Development Approach](./docs/DEVELOPMENT_APPROACH.md)** - Human-AI collaboration story

---

## Features

### Core Functionality
- **High-Performance Scanning** - Multi-core parallel processing with incremental file discovery for network filesystems
- **Comprehensive Metadata Extraction** - Extract dates from EXIF, XMP, filenames, and directory structure
- **RAW File Support** - Native RAW metadata extraction during scanning (no conversion required)
- **Duplicate Detection** - Find exact and similar duplicates using checksums and perceptual hashing
- **Quality Scoring** - Automatically select the best copy among duplicates
- **Corruption Tracking** - Automatically detect and report corrupted/truncated image files
- **Date-Based Reorganization** - Reorganize libraries into date-based directory structures

### Advanced Features
- **GPU Acceleration** - PyTorch-based GPU acceleration (20-30x faster on compatible GPUs)
- **Web Interface** - Modern Vue.js web UI with progressive phase-based performance monitoring
- **Real-Time Performance Tracking** - Live throughput, GPU utilization, and bottleneck analysis
- **FAISS Similarity Search** - GPU-accelerated similarity search for large catalogs
- **Network Filesystem Optimization** - Incremental file discovery prevents blocking on slow NAS mounts
- **Beautiful CLI** - Rich terminal interface with progress bars and formatted output

### Quality & Testing
- **Fully Tested** - Comprehensive test suite with **580 passing tests** and **78% coverage**
- **Type Safe** - Full type hints throughout the codebase with Pydantic v2
- **Fast Tests** - Parallel test execution with pytest-xdist (62.5% faster)

---

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

See **[GPU Setup Guide](./docs/GPU_SETUP_GUIDE.md)** for detailed instructions.

---

## Quick Start

### 1. Analyze Your Photo Library

Start by scanning your photo directories to build a catalog:

```bash
# Basic analysis
vam-analyze /path/to/catalog -s /path/to/photos

# Recommended: Use all CPU cores with duplicate detection
vam-analyze /path/to/catalog -s /path/to/photos --detect-duplicates -v

# For large libraries: specify worker count
vam-analyze /path/to/catalog -s /path/to/photos --workers 32 --detect-duplicates
```

**Performance:**
- 32-core system: 20-30x speedup vs single-threaded
- With GPU: 20-30x faster perceptual hashing
- Real-time monitoring shows throughput, GPU utilization, and bottlenecks

### 2. Browse Your Catalog

Launch the web interface to explore your catalog:

```bash
# Start web server (opens at http://localhost:8765)
vam-web /path/to/catalog

# Custom port or allow external access
vam-web /path/to/catalog --port 8080 --host 0.0.0.0
```

The web interface provides:
- Browse all images and videos with metadata
- View duplicate groups with side-by-side comparison
- See date extraction results and confidence levels
- Review statistics and storage analysis
- **Real-time performance monitoring** - Live charts and dashboard

### 3. Find and Review Duplicates

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

**Duplicate Detection:** Uses perceptual hashing (dHash, aHash, wHash) with quality scoring to identify duplicates and automatically select the best copy.

---

## Common Workflows

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

For detailed workflows, see the **[User Guide](./docs/USER_GUIDE.md)**.

---

## Supported Formats

- **Images:** JPEG, PNG, TIFF, BMP, GIF, WEBP, HEIC/HEIF
- **RAW Formats:** CR2, CR3, NEF, ARW, DNG, ORF, RW2, PEF, SR2, RAF, and more
- **Videos:** MP4, MOV, AVI, MKV, and more

---

## Contributing

We welcome contributions! Please see our **[Contributing Guide](./docs/CONTRIBUTING.md)** for details on:

- Setting up your development environment
- Running tests
- Code style and quality standards
- Submitting pull requests

### Quick Contribution Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests (skips integration tests by default)
pytest -n auto

# Run integration tests (requires docker-compose services running)
pytest -m integration

# Run code quality checks
black vam_tools/ tests/
isort vam_tools/ tests/
flake8 vam_tools/ tests/
mypy vam_tools/
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

---

## Acknowledgments

VAM Tools builds on excellent open-source projects:

- [Pillow](https://python-pillow.org/) for image processing
- [ExifTool](https://exiftool.org/) for metadata extraction
- [Click](https://click.palletsprojects.com/) for CLI framework
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- [FastAPI](https://fastapi.tiangolo.com/) for web interface
- [Pydantic](https://docs.pydantic.dev/) for data validation
- [PyTorch](https://pytorch.org/) for GPU acceleration
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Vue.js](https://vuejs.org/) for frontend

---

## Author

**Ivan R. Judson** - [irjudson@gmail.com](mailto:irjudson@gmail.com)

---

## Project Links

- **Repository**: https://github.com/irjudson/vam-tools
- **Issues**: https://github.com/irjudson/vam-tools/issues
- **Discussions**: https://github.com/irjudson/vam-tools/discussions
- **Documentation**: [./docs](./docs)

---

## Development Story

This project was developed using human-AI pair programming with Claude. The collaboration followed established engineering principles to ensure code quality without requiring exhaustive human review. Read more about the **[Development Approach](./docs/DEVELOPMENT_APPROACH.md)**.

**Result**: Production-ready tool with continuous improvements, 616 passing tests and 79% coverage.
