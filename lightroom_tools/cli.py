#!/usr/bin/env python3
"""
Lightroom Tools CLI

Interactive command-line interface for Lightroom photo management tools.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

from . import image_date_analyzer
from . import duplicate_image_finder


def print_banner():
    """Print the application banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                            LIGHTROOM TOOLS                                   ║
║                                                                               ║
║            A collection of tools for managing photo libraries                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")


def print_menu():
    """Print the main menu."""
    print("\n" + "="*80)
    print("MAIN MENU")
    print("="*80)
    print("1. Analyze Image Dates")
    print("   └─ Extract earliest dates from EXIF, filenames, and directory structure")
    print()
    print("2. Find Duplicate Images")
    print("   └─ Identify duplicate photos across different formats and sizes")
    print()
    print("3. Reorganize Lightroom Catalog")
    print("   └─ Reorganize catalog files and directories")
    print()
    print("4. Quick Analysis")
    print("   └─ Run both date analysis and duplicate detection")
    print()
    print("5. Help & Documentation")
    print("   └─ Show detailed help for each tool")
    print()
    print("0. Exit")
    print("="*80)


def get_directory_input(prompt: str = "Enter directory path: ") -> Optional[str]:
    """Get and validate directory input from user."""
    while True:
        try:
            directory = input(prompt).strip()
            if not directory:
                print("❌ Please enter a directory path.")
                continue
            
            # Expand user home directory
            directory = os.path.expanduser(directory)
            
            if not os.path.exists(directory):
                print(f"❌ Directory '{directory}' does not exist.")
                continue
            
            if not os.path.isdir(directory):
                print(f"❌ '{directory}' is not a directory.")
                continue
            
            return directory
        
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled.")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


def get_output_file(default_name: str) -> str:
    """Get output file path from user."""
    output = input(f"Output file (default: {default_name}): ").strip()
    if not output:
        output = default_name
    
    # Expand user home directory
    output = os.path.expanduser(output)
    
    # Check if file exists and confirm overwrite
    if os.path.exists(output):
        confirm = input(f"File '{output}' exists. Overwrite? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Operation cancelled.")
            return None
    
    return output


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{prompt} ({default_str}): ").strip().lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes']


def analyze_image_dates():
    """Run the image date analyzer tool."""
    print("\n" + "="*60)
    print("IMAGE DATE ANALYZER")
    print("="*60)
    print("This tool analyzes images to find the earliest date from:")
    print("• EXIF metadata")
    print("• Filename patterns (YYYY-MM-DD, YYYYMMDD, etc.)")
    print("• Directory structure")
    print("• File creation dates")
    print()
    
    directory = get_directory_input("Enter directory containing images: ")
    if not directory:
        return
    
    recursive = get_yes_no("Scan subdirectories recursively?", True)
    
    output_file = get_output_file("image_dates.txt")
    if not output_file:
        return
    
    print(f"\n🔍 Analyzing images in: {directory}")
    print(f"📁 Recursive: {'Yes' if recursive else 'No'}")
    print(f"📄 Output file: {output_file}")
    print("\nStarting analysis...")
    
    try:
        # Call the image date analyzer
        sys.argv = ['image_date_analyzer.py', directory]
        if recursive:
            sys.argv.append('-r')
        sys.argv.extend(['-o', output_file])
        
        image_date_analyzer.main()
        print(f"\n✅ Analysis complete! Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")


def find_duplicate_images():
    """Run the duplicate image finder tool."""
    print("\n" + "="*60)
    print("DUPLICATE IMAGE FINDER")
    print("="*60)
    print("This tool finds duplicate images that may have:")
    print("• Different file formats (JPG, PNG, TIFF, etc.)")
    print("• Different sizes or resolutions")
    print("• Different filenames")
    print("• Similar visual content using perceptual hashing")
    print()
    
    directory = get_directory_input("Enter directory containing images: ")
    if not directory:
        return
    
    recursive = get_yes_no("Scan subdirectories recursively?", True)
    
    print("\nSimilarity threshold (0-64, lower = more strict):")
    print("• 0-5: Very similar images only")
    print("• 6-15: Similar images (recommended)")
    print("• 16-30: Somewhat similar images")
    print("• 31+: Very loose matching")
    
    while True:
        try:
            threshold = input("Enter threshold (default: 5): ").strip()
            if not threshold:
                threshold = 5
            else:
                threshold = int(threshold)
            
            if 0 <= threshold <= 64:
                break
            else:
                print("❌ Threshold must be between 0 and 64.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    output_file = get_output_file("duplicate_images.txt")
    if not output_file:
        return
    
    print(f"\n🔍 Searching for duplicates in: {directory}")
    print(f"📁 Recursive: {'Yes' if recursive else 'No'}")
    print(f"🎯 Similarity threshold: {threshold}")
    print(f"📄 Output file: {output_file}")
    print("\nStarting duplicate detection...")
    
    try:
        # Call the duplicate image finder
        sys.argv = ['duplicate_image_finder.py', directory]
        if recursive:
            sys.argv.append('-r')
        sys.argv.extend(['-t', str(threshold)])
        sys.argv.extend(['-o', output_file])
        
        duplicate_image_finder.main()
        print(f"\n✅ Analysis complete! Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during duplicate detection: {e}")


def reorganize_catalog():
    """Run the catalog reorganizer tool."""
    print("\n" + "="*60)
    print("LIGHTROOM CATALOG REORGANIZER")
    print("="*60)
    print("This tool helps reorganize Lightroom catalog files and directories.")
    print()
    print("⚠️  WARNING: This feature is under development.")
    print("Please backup your Lightroom catalog before using this tool.")
    print()
    
    proceed = get_yes_no("Do you want to continue?", False)
    if not proceed:
        return
    
    directory = get_directory_input("Enter Lightroom catalog directory: ")
    if not directory:
        return
    
    print(f"\n🔧 This feature will be implemented in a future version.")
    print(f"📁 Target directory: {directory}")


def quick_analysis():
    """Run both date analysis and duplicate detection."""
    print("\n" + "="*60)
    print("QUICK ANALYSIS")
    print("="*60)
    print("This will run both image date analysis and duplicate detection.")
    print()
    
    directory = get_directory_input("Enter directory containing images: ")
    if not directory:
        return
    
    recursive = get_yes_no("Scan subdirectories recursively?", True)
    
    print("\n🚀 Starting quick analysis...")
    
    # Run date analysis
    print("\n📅 Step 1: Analyzing image dates...")
    try:
        sys.argv = ['image_date_analyzer.py', directory]
        if recursive:
            sys.argv.append('-r')
        sys.argv.extend(['-o', 'quick_analysis_dates.txt'])
        
        image_date_analyzer.main()
        print("✅ Date analysis complete!")
        
    except Exception as e:
        print(f"❌ Error in date analysis: {e}")
    
    # Run duplicate detection
    print("\n🔍 Step 2: Finding duplicate images...")
    try:
        sys.argv = ['duplicate_image_finder.py', directory]
        if recursive:
            sys.argv.append('-r')
        sys.argv.extend(['-t', '5'])
        sys.argv.extend(['-o', 'quick_analysis_duplicates.txt'])
        
        duplicate_image_finder.main()
        print("✅ Duplicate detection complete!")
        
    except Exception as e:
        print(f"❌ Error in duplicate detection: {e}")
    
    print("\n🎉 Quick analysis complete!")
    print("📄 Results saved to:")
    print("   • quick_analysis_dates.txt")
    print("   • quick_analysis_duplicates.txt")


def show_help():
    """Show detailed help information."""
    print("\n" + "="*60)
    print("HELP & DOCUMENTATION")
    print("="*60)
    
    print("""
📖 TOOL DESCRIPTIONS:

1. Image Date Analyzer
   • Extracts the earliest date from multiple sources
   • Checks EXIF metadata for camera timestamps
   • Parses common date patterns in filenames
   • Analyzes directory structure for date information
   • Falls back to file creation dates
   • Output: Text file with directory, filename, and earliest date

2. Duplicate Image Finder
   • Uses perceptual hashing to find visually similar images
   • Detects duplicates across different file formats
   • Finds images with different sizes or resolutions
   • Groups exact duplicates and similar images separately
   • Output: Text file with grouped duplicate information

3. Lightroom Catalog Reorganizer
   • [Under Development] Reorganizes catalog structure
   • Helps maintain organized photo libraries
   • Backup recommendations and safety checks

📝 SUPPORTED IMAGE FORMATS:
   • JPEG/JPG • PNG • TIFF/TIF • BMP • GIF • WEBP

💡 TIPS:
   • Always backup your photos before reorganizing
   • Use recursive scanning for complete analysis
   • Lower similarity thresholds find more exact matches
   • Review duplicate results before deleting any files
   • EXIF data is the most reliable source for image dates

🔧 TECHNICAL DETAILS:
   • Uses Pillow library for image processing
   • Implements dHash and aHash algorithms for similarity
   • Supports Unicode filenames and paths
   • Cross-platform compatible (Windows, macOS, Linux)
""")


def interactive_mode():
    """Run the interactive CLI mode."""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("Select an option (0-5): ").strip()
            
            if choice == '0':
                print("\n👋 Thank you for using Lightroom Tools!")
                break
            elif choice == '1':
                analyze_image_dates()
            elif choice == '2':
                find_duplicate_images()
            elif choice == '3':
                reorganize_catalog()
            elif choice == '4':
                quick_analysis()
            elif choice == '5':
                show_help()
            else:
                print("❌ Invalid option. Please select 0-5.")
            
            if choice != '0' and choice != '5':
                input("\nPress Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("Press Enter to continue...")


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description='Lightroom Tools - Photo library management utilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Interactive mode
  %(prog)s --dates /photos    # Analyze dates only
  %(prog)s --duplicates /photos --threshold 3
  %(prog)s --quick /photos    # Quick analysis
        """
    )
    
    parser.add_argument('--dates', metavar='DIR', 
                       help='Run image date analysis on directory')
    parser.add_argument('--duplicates', metavar='DIR',
                       help='Run duplicate detection on directory')
    parser.add_argument('--quick', metavar='DIR',
                       help='Run quick analysis (dates + duplicates)')
    parser.add_argument('--threshold', type=int, default=5,
                       help='Similarity threshold for duplicates (default: 5)')
    parser.add_argument('--output', '-o', 
                       help='Output file for results')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Scan directories recursively')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')
    
    args = parser.parse_args()
    
    # If no arguments provided, run interactive mode
    if not any([args.dates, args.duplicates, args.quick]):
        interactive_mode()
        return
    
    # Command-line mode
    if args.dates:
        if not os.path.exists(args.dates):
            print(f"❌ Directory '{args.dates}' does not exist.")
            sys.exit(1)
        
        output = args.output or 'image_dates.txt'
        sys.argv = ['image_date_analyzer.py', args.dates]
        if args.recursive:
            sys.argv.append('-r')
        sys.argv.extend(['-o', output])
        
        image_date_analyzer.main()
    
    elif args.duplicates:
        if not os.path.exists(args.duplicates):
            print(f"❌ Directory '{args.duplicates}' does not exist.")
            sys.exit(1)
        
        output = args.output or 'duplicate_images.txt'
        sys.argv = ['duplicate_image_finder.py', args.duplicates]
        if args.recursive:
            sys.argv.append('-r')
        sys.argv.extend(['-t', str(args.threshold)])
        sys.argv.extend(['-o', output])
        
        duplicate_image_finder.main()
    
    elif args.quick:
        if not os.path.exists(args.quick):
            print(f"❌ Directory '{args.quick}' does not exist.")
            sys.exit(1)
        
        # Run both tools
        print("🚀 Running quick analysis...")
        
        # Date analysis
        sys.argv = ['image_date_analyzer.py', args.quick]
        if args.recursive:
            sys.argv.append('-r')
        sys.argv.extend(['-o', 'quick_analysis_dates.txt'])
        image_date_analyzer.main()
        
        # Duplicate detection
        sys.argv = ['duplicate_image_finder.py', args.quick]
        if args.recursive:
            sys.argv.append('-r')
        sys.argv.extend(['-t', str(args.threshold)])
        sys.argv.extend(['-o', 'quick_analysis_duplicates.txt'])
        duplicate_image_finder.main()
        
        print("✅ Quick analysis complete!")


if __name__ == '__main__':
    main()