"""
CLI for duplicate image detection.

Finds duplicate and similar images using perceptual hashing.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from ..core.duplicate_detection import DuplicateDetector
from ..core.image_utils import (
    collect_image_files,
    format_file_size,
    get_image_info,
    setup_logging,
)

console = Console()


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
    "-r",
    "--recursive",
    is_flag=True,
    default=True,
    help="Scan directories recursively",
)
@click.option(
    "--no-recursive",
    is_flag=True,
    help="Disable recursive scanning",
)
@click.option(
    "-t",
    "--threshold",
    type=int,
    default=5,
    help="Similarity threshold (0-64, lower = more strict)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress all output except errors",
)
def cli(
    directory: str,
    output: str,
    recursive: bool,
    no_recursive: bool,
    threshold: int,
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
        console.print("[red]Error: Threshold must be between 0 and 64[/red]")
        sys.exit(1)

    # Setup logging
    setup_logging(verbose=verbose, quiet=quiet)

    directory_path = Path(directory).resolve()
    output_path = Path(output).resolve()

    if not quiet:
        console.print(f"\n[bold cyan]Duplicate Image Finder[/bold cyan]\n")
        console.print(f"Directory: {directory_path}")
        console.print(f"Recursive: {recursive}")
        console.print(f"Threshold: {threshold} (0=exact, 64=very loose)")
        console.print(f"Output: {output_path}\n")

    # Collect image files
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=quiet,
    ) as progress:
        task = progress.add_task("Collecting image files...", total=None)
        image_files = collect_image_files(directory_path, recursive=recursive)
        progress.update(task, completed=True)

    if not image_files:
        console.print("[yellow]No image files found.[/yellow]")
        sys.exit(0)

    if len(image_files) < 2:
        console.print(
            "[yellow]Need at least 2 images to find duplicates.[/yellow]"
        )
        sys.exit(0)

    if not quiet:
        console.print(f"Found [bold]{len(image_files)}[/bold] image files\n")

    # Find duplicates
    detector = DuplicateDetector()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
        disable=quiet,
    ) as progress:
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
                    console.print(f"[yellow]Error processing {image_path}: {e}[/yellow]")
                progress.advance(task)

    # Find duplicates
    if not quiet:
        console.print("\n[cyan]Analyzing for duplicates...[/cyan]\n")

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
    if not quiet:
        table = Table(title="Duplicate Detection Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        total_images = len(image_files)
        total_groups = len(all_duplicates)
        exact_groups = len(exact_duplicates)
        perceptual_groups = len(perceptual_duplicates)

        images_in_groups = sum(len(g.images) for g in all_duplicates)

        table.add_row("Total images scanned", str(total_images))
        table.add_row("Images in duplicate groups", str(images_in_groups))
        table.add_row("Unique images", str(total_images - images_in_groups))
        table.add_row("", "")
        table.add_row("Total duplicate groups", str(total_groups))
        table.add_row("Exact duplicates", str(exact_groups))
        table.add_row("Perceptual duplicates", str(perceptual_groups))

        console.print(table)

        if total_groups > 0:
            console.print(
                f"\n[yellow]Review the results carefully before deleting any files![/yellow]"
            )

        console.print(f"\n[green]Results written to:[/green] {output_path}")


if __name__ == "__main__":
    cli()
