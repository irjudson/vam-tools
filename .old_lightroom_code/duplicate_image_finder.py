#!/usr/bin/env python3
"""
Duplicate Image Finder

Scans a directory for duplicate images that may have different sizes, formats, etc.
Uses perceptual hashing to find visually similar images.
"""

import os
import sys
import hashlib
from collections import defaultdict
from pathlib import Path
from PIL import Image
import argparse


def calculate_file_hash(file_path):
    """Calculate MD5 hash of the file for exact duplicates."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def calculate_dhash(image_path, hash_size=8):
    """Calculate difference hash (dHash) for perceptual similarity."""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize
            img = img.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
            
            # Calculate horizontal gradient
            hash_bits = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = img.getpixel((col, row))
                    pixel_right = img.getpixel((col + 1, row))
                    hash_bits.append(pixel_left > pixel_right)
            
            # Convert to hex string
            return ''.join('1' if bit else '0' for bit in hash_bits)
    except Exception:
        return None


def calculate_average_hash(image_path, hash_size=8):
    """Calculate average hash (aHash) for perceptual similarity."""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize
            img = img.convert('L').resize((hash_size, hash_size), Image.LANCZOS)
            
            # Calculate average pixel value
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            
            # Generate hash based on average
            hash_bits = ['1' if pixel > avg else '0' for pixel in pixels]
            return ''.join(hash_bits)
    except Exception:
        return None


def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two hash strings."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return float('inf')
    
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def get_image_info(image_path):
    """Get basic image information."""
    try:
        with Image.open(image_path) as img:
            return {
                'dimensions': img.size,
                'format': img.format,
                'mode': img.mode,
                'file_size': os.path.getsize(image_path)
            }
    except Exception:
        return None


def is_image_file(filename):
    """Check if file is an image."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    return Path(filename).suffix.lower() in image_extensions


def find_duplicates(directory, recursive=True, similarity_threshold=5):
    """Find duplicate images in directory."""
    print(f"Scanning directory: {directory}")
    
    # Collect all image files
    image_files = []
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if is_image_file(file):
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if is_image_file(file):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    image_files.append(file_path)
    
    print(f"Found {len(image_files)} image files")
    
    # Calculate hashes and metadata for all images
    image_data = {}
    file_hash_groups = defaultdict(list)
    dhash_groups = defaultdict(list)
    ahash_groups = defaultdict(list)
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Calculate various hashes
        file_hash = calculate_file_hash(image_path)
        dhash = calculate_dhash(image_path)
        ahash = calculate_average_hash(image_path)
        info = get_image_info(image_path)
        
        image_data[image_path] = {
            'file_hash': file_hash,
            'dhash': dhash,
            'ahash': ahash,
            'info': info
        }
        
        # Group by hashes
        if file_hash:
            file_hash_groups[file_hash].append(image_path)
        if dhash:
            dhash_groups[dhash].append(image_path)
        if ahash:
            ahash_groups[ahash].append(image_path)
    
    # Find duplicate groups
    duplicate_groups = []
    processed = set()
    
    # First, find exact file duplicates
    for file_hash, files in file_hash_groups.items():
        if len(files) > 1:
            duplicate_groups.append({
                'type': 'exact_file_duplicate',
                'files': files,
                'similarity': 'identical'
            })
            processed.update(files)
    
    # Then find perceptual duplicates among remaining files
    unprocessed_files = [f for f in image_files if f not in processed]
    
    for i, file1 in enumerate(unprocessed_files):
        if file1 in processed:
            continue
            
        current_group = [file1]
        processed.add(file1)
        
        for file2 in unprocessed_files[i+1:]:
            if file2 in processed:
                continue
                
            data1 = image_data[file1]
            data2 = image_data[file2]
            
            # Check dhash similarity
            dhash_dist = hamming_distance(data1['dhash'], data2['dhash'])
            ahash_dist = hamming_distance(data1['ahash'], data2['ahash'])
            
            # Consider similar if either hash is within threshold
            if dhash_dist <= similarity_threshold or ahash_dist <= similarity_threshold:
                current_group.append(file2)
                processed.add(file2)
        
        if len(current_group) > 1:
            duplicate_groups.append({
                'type': 'perceptual_duplicate',
                'files': current_group,
                'similarity': 'similar'
            })
    
    return duplicate_groups, image_data


def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"


def write_results(duplicate_groups, image_data, output_file):
    """Write results to output file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DUPLICATE IMAGE ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        if not duplicate_groups:
            f.write("No duplicate images found.\n")
            return
        
        f.write(f"Found {len(duplicate_groups)} groups of duplicate images\n\n")
        
        for i, group in enumerate(duplicate_groups, 1):
            f.write(f"GROUP {i} - {group['type'].upper()}\n")
            f.write("-" * 40 + "\n")
            
            for file_path in group['files']:
                data = image_data[file_path]
                info = data['info']
                
                f.write(f"File: {file_path}\n")
                if info:
                    f.write(f"  Size: {format_file_size(info['file_size'])}\n")
                    f.write(f"  Dimensions: {info['dimensions'][0]}x{info['dimensions'][1]}\n")
                    f.write(f"  Format: {info['format']}\n")
                    f.write(f"  Mode: {info['mode']}\n")
                if data['file_hash']:
                    f.write(f"  File Hash: {data['file_hash'][:16]}...\n")
                if data['dhash']:
                    f.write(f"  dHash: {data['dhash'][:16]}...\n")
                f.write("\n")
            
            f.write("\n")
        
        # Summary section for script parsing
        f.write("SUMMARY FOR SCRIPT PARSING\n")
        f.write("=" * 30 + "\n")
        for i, group in enumerate(duplicate_groups, 1):
            f.write(f"GROUP_{i}|{group['type']}|{len(group['files'])}|")
            f.write("|".join(group['files']) + "\n")


def main():
    parser = argparse.ArgumentParser(description='Find duplicate images in directory')
    parser.add_argument('directory', help='Directory to scan for duplicates')
    parser.add_argument('-o', '--output', default='duplicate_images.txt', 
                       help='Output file (default: duplicate_images.txt)')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='Scan recursively (default: True)', default=True)
    parser.add_argument('-t', '--threshold', type=int, default=5,
                       help='Similarity threshold (0-64, lower = more similar)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        sys.exit(1)
    
    print("Starting duplicate image analysis...")
    duplicate_groups, image_data = find_duplicates(
        args.directory, 
        recursive=args.recursive,
        similarity_threshold=args.threshold
    )
    
    write_results(duplicate_groups, image_data, args.output)
    
    print(f"\nAnalysis complete!")
    print(f"Found {len(duplicate_groups)} groups of duplicate images")
    print(f"Results written to: {args.output}")


if __name__ == '__main__':
    main()