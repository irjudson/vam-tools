"""
CLI for duplicate image detection.

Finds duplicate and similar images using perceptual hashing.
"""

import sys
from pathlib import Path

import click

from vam_tools.shared import (
    collect_image_files,
)
from vam_tools.shared import format_bytes as format_file_size
from vam_tools.shared import (
    get_image_info,
)

from ..core.duplicate_detection import DuplicateDetector
from .base import CLIDisplay, common_options, file_scan_options, init_logging


@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="duplicate_images.txt",
    help="Output file for results",
)
@click.option(
    "-t",
    "--threshold",
    type=int,
    default=5,
    help="Similarity threshold (0-64, lower = more strict)",
)
@file_scan_options
@common_options
@init_logging
def cli(
    directory: str,
    output: str,
    threshold: int,
    recursive: bool,
    no_recursive: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """
    Find duplicate and similar images.

    Uses perceptual hashing to find images that are visually similar, even if
    they have different file sizes, formats, or resolutions.

    DIRECTORY: Path to directory containing images to analyze
    """
    # Handle recursive flag
    if no_recursive:
        recursive = False

    # Validate threshold
    if not 0 <= threshold <= 64:
        display = CLIDisplay(quiet=False)
        display.print_error("Error: Threshold must be between 0 and 64")
        sys.exit(1)

    directory_path = Path(directory).resolve()
    output_path = Path(output).resolve()

    # Initialize display helper
    display = CLIDisplay(quiet=quiet)

    # Display header and configuration
    display.print_header("Duplicate Image Finder")
    display.print_config(
        {
            "Directory": str(directory_path),
            "Recursive": str(recursive),
            "Threshold": f"{threshold} (0=exact, 64=very loose)",
            "Output": str(output_path),
        }
    )

    # Collect image files
    with display.spinner_progress("Collecting image files...") as progress:
        task = progress.add_task("Collecting image files...", total=None)
        image_files = collect_image_files(directory_path, recursive=recursive)
        progress.update(task, completed=True)

    if not image_files:
        display.print_warning("No image files found.")
        sys.exit(0)

    if len(image_files) < 2:
        display.print_warning("Need at least 2 images to find duplicates.")
        sys.exit(0)

    display.print_info(f"Found [bold]{len(image_files)}[/bold] image files\n")

    # Find duplicates
    detector = DuplicateDetector()

    with display.bar_progress("Finding duplicates...") as progress:
        task = progress.add_task("Finding duplicates...", total=len(image_files))

        # Process images with progress updates
        detector.image_hashes = {}
        for image_path in image_files:
            try:
                hashes = detector.calculate_hashes(image_path)
                detector.image_hashes[image_path] = hashes
                progress.advance(task)
            except Exception as e:
                if verbose:
                    display.print_warning(f"Error processing {image_path}: {e}")
                progress.advance(task)

    # Find duplicates
    display.print_info("\n[cyan]Analyzing for duplicates...[/cyan]\n")

    exact_duplicates = detector.find_exact_duplicates()
    perceptual_duplicates = detector.find_perceptual_duplicates(threshold)
    all_duplicates = exact_duplicates + perceptual_duplicates

    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("DUPLICATE IMAGE ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")

        if not all_duplicates:
            f.write("No duplicate images found.\n")
        else:
            f.write(f"Found {len(all_duplicates)} groups of duplicate images\n\n")

            for i, group in enumerate(all_duplicates, 1):
                f.write(f"\nGROUP {i} - {group.similarity_type.upper()}\n")
                f.write("-" * 80 + "\n")

                if group.similarity_type == "perceptual":
                    f.write(f"Similarity distance: {group.hash_distance}\n\n")

                for image_path in group.images:
                    info = get_image_info(image_path)
                    f.write(f"File: {image_path}\n")

                    if info:
                        f.write(f"  Size: {format_file_size(info['file_size'])}\n")
                        f.write(
                            f"  Dimensions: {info['dimensions'][0]}x{info['dimensions'][1]}\n"
                        )
                        f.write(f"  Format: {info['format']}\n")

                    f.write("\n")

    # Display summary
    total_images = len(image_files)
    total_groups = len(all_duplicates)
    exact_groups = len(exact_duplicates)
    perceptual_groups = len(perceptual_duplicates)
    images_in_groups = sum(len(g.images) for g in all_duplicates)

    metrics = {
        "Total images scanned": total_images,
        "Images in duplicate groups": images_in_groups,
        "Unique images": total_images - images_in_groups,
        "": "",
        "Total duplicate groups": total_groups,
        "Exact duplicates": exact_groups,
        "Perceptual duplicates": perceptual_groups,
    }
    display.print_summary(metrics, title="Duplicate Detection Summary")

    if total_groups > 0:
        display.print_warning(
            "\nReview the results carefully before deleting any files!"
        )

    display.print_success(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    cli()
