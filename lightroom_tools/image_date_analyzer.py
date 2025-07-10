#!/usr/bin/env python3
"""
Image Date Analyzer

Analyzes all images in a specified directory and determines the earliest date
from EXIF data, filename, and directory structure.
"""

import os
import sys
import re
from datetime import datetime
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import argparse


def extract_exif_date(image_path):
    """Extract date from EXIF data."""
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if exif:
                for tag, value in exif.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                        try:
                            return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                        except ValueError:
                            continue
    except Exception:
        pass
    return None


def extract_filename_date(filename):
    """Extract date from filename using common patterns."""
    # Common date patterns in filenames
    patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
        r'(\d{4})_(\d{2})_(\d{2})',  # YYYY_MM_DD
        r'(\d{4})(\d{2})(\d{2})',    # YYYYMMDD
        r'(\d{2})-(\d{2})-(\d{4})',  # MM-DD-YYYY
        r'(\d{2})_(\d{2})_(\d{4})',  # MM_DD_YYYY
        r'(\d{2})(\d{2})(\d{4})',    # MMDDYYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                groups = match.groups()
                if len(groups[0]) == 4:  # Year first
                    year, month, day = groups
                else:  # Year last
                    month, day, year = groups
                return datetime(int(year), int(month), int(day))
            except ValueError:
                continue
    return None


def extract_directory_date(directory_path):
    """Extract date from directory structure."""
    path_parts = Path(directory_path).parts
    
    # Look for year/month/day patterns in directory structure
    for part in reversed(path_parts):
        # Try to find year (4 digits)
        year_match = re.search(r'(\d{4})', part)
        if year_match:
            year = int(year_match.group(1))
            if 1900 <= year <= 2100:
                # Look for month in the same part or adjacent parts
                month_match = re.search(r'(\d{1,2})', part)
                if month_match:
                    month = int(month_match.group(1))
                    if 1 <= month <= 12:
                        try:
                            return datetime(year, month, 1)
                        except ValueError:
                            pass
                # If no month found, use January as default
                try:
                    return datetime(year, 1, 1)
                except ValueError:
                    pass
    return None


def get_file_creation_date(file_path):
    """Get file creation date as fallback."""
    try:
        stat = os.stat(file_path)
        return datetime.fromtimestamp(stat.st_ctime)
    except Exception:
        return None


def analyze_image(image_path):
    """Analyze a single image and return the earliest date found."""
    dates = []
    
    # Extract EXIF date
    exif_date = extract_exif_date(image_path)
    if exif_date:
        dates.append(exif_date)
    
    # Extract filename date
    filename = os.path.basename(image_path)
    filename_date = extract_filename_date(filename)
    if filename_date:
        dates.append(filename_date)
    
    # Extract directory date
    directory = os.path.dirname(image_path)
    directory_date = extract_directory_date(directory)
    if directory_date:
        dates.append(directory_date)
    
    # Get file creation date as fallback
    creation_date = get_file_creation_date(image_path)
    if creation_date:
        dates.append(creation_date)
    
    # Return the earliest date
    if dates:
        return min(dates)
    return None


def is_image_file(filename):
    """Check if file is an image."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    return Path(filename).suffix.lower() in image_extensions


def main():
    parser = argparse.ArgumentParser(description='Analyze images for earliest date')
    parser.add_argument('directory', help='Directory to analyze')
    parser.add_argument('-o', '--output', default='image_dates.txt', help='Output file')
    parser.add_argument('-r', '--recursive', action='store_true', help='Scan recursively')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        sys.exit(1)
    
    results = []
    
    # Scan directory
    if args.recursive:
        for root, dirs, files in os.walk(args.directory):
            for file in files:
                if is_image_file(file):
                    image_path = os.path.join(root, file)
                    earliest_date = analyze_image(image_path)
                    if earliest_date:
                        results.append((root, file, earliest_date))
    else:
        for file in os.listdir(args.directory):
            if is_image_file(file):
                image_path = os.path.join(args.directory, file)
                if os.path.isfile(image_path):
                    earliest_date = analyze_image(image_path)
                    if earliest_date:
                        results.append((args.directory, file, earliest_date))
    
    # Write results to file
    with open(args.output, 'w', encoding='utf-8') as f:
        for directory, filename, date in sorted(results, key=lambda x: x[2]):
            f.write(f"{directory} - {filename} - {date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Analyzed {len(results)} images. Results written to '{args.output}'")


if __name__ == '__main__':
    main()