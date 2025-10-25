# VAM Tools (Visual Asset Management)

A professional collection of Python tools for managing and organizing photo/video libraries.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Image Date Analyzer** - Extract earliest dates from EXIF data, filenames, and directory structure using ExifTool
- **Duplicate Image Finder** - Find duplicate images across different formats and sizes using perceptual hashing (dHash + aHash)
- **Catalog Reorganizer** - Reorganize photo/video catalog files into date-based directory structures with dry-run support
- **Beautiful CLI** - Rich terminal interface with progress bars and formatted output
- **Fully Tested** - Comprehensive test suite with fixtures
- **Type Safe** - Full type hints throughout the codebase

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
sudo apt-get install libimage-exiftool-perl
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

### Interactive Mode

Launch the interactive menu interface:

```bash
vam-tools
```

This provides an easy-to-use menu for accessing all tools.

### Command Line Tools

Each tool can be run directly from the command line:

#### 1. Image Date Analyzer

Extract and analyze dates from images:

```bash
# Analyze dates in a directory
vam-dates /path/to/photos

# Analyze with custom output file
vam-dates /path/to/photos -o dates_report.txt

# Non-recursive scan
vam-dates /path/to/photos --no-recursive

# Verbose output
vam-dates /path/to/photos -v

# Sort by source instead of date
vam-dates /path/to/photos --sort-by source
```

**Output Example:**
```
Image Date Analysis Results
================================================================================

/photos/vacation - IMG_1234.jpg - 2023-06-15 14:30:22 (from exif, confidence: 95%)
/photos/vacation - photo.jpg - 2023-06-15 00:00:00 (from filename, confidence: 70%)
/photos/old - scan.jpg - 2020-01-01 00:00:00 (from directory, confidence: 50%)
```

#### 2. Duplicate Image Finder

Find duplicate and similar images:

```bash
# Find duplicates with default threshold
vam-duplicates /path/to/photos

# Strict matching (lower threshold = more strict)
vam-duplicates /path/to/photos -t 3

# Loose matching
vam-duplicates /path/to/photos -t 15

# Custom output file
vam-duplicates /path/to/photos -o duplicates.txt

# Verbose mode
vam-duplicates /path/to/photos -v
```

**Similarity Thresholds:**
- `0-5`: Very similar images only (recommended for finding true duplicates)
- `6-15`: Similar images (good for finding variations)
- `16-30`: Somewhat similar
- `31-64`: Very loose matching

**Output Example:**
```
DUPLICATE IMAGE ANALYSIS RESULTS
================================================================================

GROUP 1 - EXACT
----------------------------------------
File: /photos/IMG_001.jpg
  Size: 2.5 MB
  Dimensions: 3840x2160
  Format: JPEG

File: /photos/backup/IMG_001.jpg
  Size: 2.5 MB
  Dimensions: 3840x2160
  Format: JPEG

GROUP 2 - PERCEPTUAL
Similarity distance: 3
----------------------------------------
File: /photos/vacation.jpg
  Size: 3.1 MB
  Dimensions: 4032x3024
  Format: JPEG

File: /photos/vacation_resized.jpg
  Size: 1.2 MB
  Dimensions: 1920x1440
  Format: JPEG
```

#### 3. Catalog Reorganizer

Reorganize photos into date-based directory structure:

```bash
# Dry-run (preview without making changes)
vam-catalog /path/to/photos -o /path/to/organized --dry-run

# Actually reorganize (move files)
vam-catalog /path/to/photos -o /path/to/organized

# Copy instead of move
vam-catalog /path/to/photos -o /path/to/organized --copy

# Different organization strategies
vam-catalog /path/to/photos -o /output -s year/month-day  # 2023/12-25/
vam-catalog /path/to/photos -o /output -s year/month      # 2023/12/
vam-catalog /path/to/photos -o /output -s year            # 2023/
vam-catalog /path/to/photos -o /output -s flat            # 2023-12-25/

# Handle file conflicts
vam-catalog /path/to/photos -o /output --conflict rename    # Add counter (default)
vam-catalog /path/to/photos -o /output --conflict skip      # Skip existing
vam-catalog /path/to/photos -o /output --conflict overwrite # Overwrite
```

**Organization Strategies:**

| Strategy | Example Structure |
|----------|------------------|
| `year/month-day` | `2023/12-25/2023-12-25_143022_image.jpg` |
| `year/month` | `2023/12/2023-12-25_143022_image.jpg` |
| `year` | `2023/2023-12-25_143022_image.jpg` |
| `flat` | `2023-12-25/2023-12-25_143022_image.jpg` |

## Supported Image Formats

- JPEG/JPG
- PNG
- TIFF/TIF
- BMP
- GIF
- WEBP
- HEIC/HEIF
- RAW formats: CR2, NEF, ARW, DNG

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
pytest tests/core/test_image_utils.py

# Run with verbose output
pytest -v
```

### Code Quality

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
│   ├── core/                 # Core business logic
│   │   ├── image_utils.py
│   │   ├── date_extraction.py
│   │   ├── duplicate_detection.py
│   │   └── catalog_reorganization.py
│   ├── cli/                  # Command-line interfaces
│   │   ├── date_cli.py
│   │   ├── duplicate_cli.py
│   │   ├── catalog_cli.py
│   │   └── main.py
│   └── __init__.py
├── tests/                    # Test suite
│   ├── core/
│   ├── cli/
│   ├── fixtures/
│   └── conftest.py
├── pyproject.toml
└── README.md
```

## How It Works

### Date Extraction

The date analyzer examines multiple sources to find the earliest date:

1. **EXIF Metadata** (95% confidence) - Camera timestamps from EXIF data
2. **Filename Patterns** (70% confidence) - Dates in filenames (YYYY-MM-DD, YYYYMMDD, etc.)
3. **Directory Structure** (50% confidence) - Year/month from folder names
4. **Filesystem Metadata** (30% confidence) - File creation/modification time

### Duplicate Detection

Uses multiple algorithms to find duplicates:

1. **MD5 Hash** - Identifies exact file duplicates
2. **dHash (Difference Hash)** - Compares adjacent pixels for perceptual similarity
3. **aHash (Average Hash)** - Compares pixels to average for perceptual similarity
4. **Hamming Distance** - Measures similarity between perceptual hashes

### Catalog Reorganization

1. Extracts date from each image using all available sources
2. Generates new path based on organization strategy
3. Handles filename conflicts according to chosen strategy
4. Supports dry-run mode for safe testing
5. Can copy or move files

## Best Practices

1. **Always backup your photos before reorganizing**
2. **Use dry-run mode first** to preview changes
3. **Start with strict thresholds** for duplicate detection (3-5)
4. **Review duplicate results** carefully before deleting files
5. **Use copy mode** when unsure about reorganization
6. **Enable verbose logging** (`-v`) for troubleshooting

## Troubleshooting

### Common Issues

**"ExifTool not found"**
- Make sure ExifTool is installed and in your PATH
- Test by running `exiftool -ver` in your terminal

**"Permission denied"**
- Ensure you have read/write permissions for the directories
- On Unix systems, check with `ls -la`

**"Memory issues with large directories"**
- Process directories in smaller batches
- Use `--no-recursive` to limit scope

**"Dates not being extracted"**
- Check that images actually have EXIF data: `exiftool image.jpg`
- Try verbose mode (`-v`) to see what's being detected

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure tests pass (`pytest`)
6. Format code (`black`, `isort`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses [Pillow](https://python-pillow.org/) for image processing
- Uses [ExifTool](https://exiftool.org/) for comprehensive EXIF data extraction
- Uses [Click](https://click.palletsprojects.com/) for CLI framework
- Uses [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- Uses [Arrow](https://arrow.readthedocs.io/) for date/time handling

## Author

Ivan R. Judson - [irjudson@gmail.com](mailto:irjudson@gmail.com)

## Project Links

- **Homepage**: https://github.com/irjudson/vam-tools
- **Issues**: https://github.com/irjudson/vam-tools/issues
- **Repository**: https://github.com/irjudson/vam-tools
