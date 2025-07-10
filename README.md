# Lightroom Tools

A collection of Python tools for managing and organizing Lightroom photo libraries.

## Features

- **Image Date Analyzer** - Extract earliest dates from EXIF data, filenames, and directory structure
- **Duplicate Image Finder** - Find duplicate images across different formats and sizes using perceptual hashing
- **Lightroom Catalog Reorganizer** - Reorganize Lightroom catalog files and directories

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/irjudson/lightroom-tools.git
   cd lightroom-tools
   ```

2. **Create a virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install requirements**
   ```bash
   # Install from requirements.txt
   pip install -r requirements.txt
   
   # OR install the package in development mode
   pip install -e .
   ```

## Usage

### Interactive CLI Tool

The easiest way to use Lightroom Tools is through the interactive command-line interface:

```bash
lightroom-tools
```

This will launch an interactive menu where you can:
- Analyze image dates
- Find duplicate images  
- Run quick analysis (both tools)
- Get help and documentation

### Command Line Usage

You can also run tools directly from the command line:

```bash
# Interactive mode
lightroom-tools

# Analyze dates only
lightroom-tools --dates /path/to/photos

# Find duplicates only
lightroom-tools --duplicates /path/to/photos --threshold 3

# Quick analysis (both tools)
lightroom-tools --quick /path/to/photos
```

### Individual Tools

### Image Date Analyzer

Analyze images to find the earliest date from EXIF data, filename patterns, and directory structure.

```bash
python lightroom_tools/image_date_analyzer.py /path/to/images
python lightroom_tools/image_date_analyzer.py /path/to/images -r  # recursive
python lightroom_tools/image_date_analyzer.py /path/to/images -o output.txt
```

**Options:**
- `-r, --recursive` - Scan directories recursively
- `-o, --output` - Output file (default: image_dates.txt)

### Duplicate Image Finder

Find duplicate images that may have different sizes, formats, or filenames.

```bash
python lightroom_tools/duplicate_image_finder.py /path/to/images
python lightroom_tools/duplicate_image_finder.py /path/to/images -o duplicates.txt
python lightroom_tools/duplicate_image_finder.py /path/to/images -t 3  # stricter similarity
```

**Options:**
- `-r, --recursive` - Scan directories recursively (default: True)
- `-o, --output` - Output file (default: duplicate_images.txt)
- `-t, --threshold` - Similarity threshold (0-64, lower = more similar, default: 5)

### Lightroom Catalog Reorganizer

Reorganize Lightroom catalog files and directories.

```bash
python lightroom_tools/reorganize_lightroom_catalog.py /path/to/catalog
```

## Output Formats

### Image Date Analyzer Output
```
/path/to/directory - filename.jpg - 2023-08-15 14:30:22
/path/to/directory - another.png - 2023-08-16 09:15:45
```

### Duplicate Image Finder Output
```
DUPLICATE IMAGE ANALYSIS RESULTS
==================================================

GROUP 1 - EXACT_FILE_DUPLICATE
----------------------------------------
File: /path/to/image1.jpg
  Size: 2.5MB
  Dimensions: 3840x2160
  Format: JPEG
  ...

GROUP 2 - PERCEPTUAL_DUPLICATE
----------------------------------------
File: /path/to/similar1.jpg
  Size: 1.8MB
  Dimensions: 1920x1080
  Format: JPEG
  ...
```

## Development

### Setting up development environment

1. **Create and activate virtual environment** (see Installation steps above)

2. **Install development dependencies**
   ```bash
   pip install -e .[dev]
   ```

3. **Run tests**
   ```bash
   pytest
   ```

4. **Format code**
   ```bash
   black lightroom_tools/
   ```

5. **Type checking**
   ```bash
   mypy lightroom_tools/
   ```

## Dependencies

- **Pillow** - Image processing and EXIF data extraction
- **argparse** - Command-line argument parsing (built-in)
- **pathlib** - Path manipulation (built-in)
- **hashlib** - File hashing (built-in)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **"No module named 'PIL'"**
   - Make sure you've activated your virtual environment
   - Install Pillow: `pip install Pillow`

2. **Permission errors**
   - Make sure you have read permissions for the directories you're scanning
   - Run with appropriate user permissions

3. **Memory issues with large directories**
   - Process directories in smaller batches
   - Consider using the non-recursive option for very large directory trees

### Getting Help

- Check the [Issues](https://github.com/irjudson/lightroom-tools/issues) page
- Create a new issue if you encounter problems
- Include your Python version, OS, and error messages when reporting issues