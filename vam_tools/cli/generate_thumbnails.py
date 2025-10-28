"""
CLI tool for generating thumbnails for catalog images.

This standalone tool can be run to generate thumbnails for all images in a catalog.
It can also be used to regenerate thumbnails or generate them for newly added images.
"""

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from ..core.catalog import CatalogDatabase
from ..shared.thumbnail_utils import generate_thumbnail, get_thumbnail_path
from ..version import get_version_string

console = Console()
logger = logging.getLogger(__name__)


@click.command()
@click.argument("catalog_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Regenerate thumbnails even if they already exist",
)
@click.option(
    "--size",
    "-s",
    type=int,
    default=200,
    help="Thumbnail size in pixels (default: 200)",
)
@click.option(
    "--quality",
    "-q",
    type=int,
    default=85,
    help="JPEG quality 1-100 (default: 85)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def generate(
    catalog_dir: Path,
    force: bool,
    size: int,
    quality: int,
    verbose: bool,
) -> None:
    """
    Generate thumbnails for all images in a catalog.

    CATALOG_DIR: Path to the catalog directory (containing catalog.json)

    This tool will:
    - Load the catalog
    - Generate thumbnails for all images (skipping existing ones unless --force)
    - Store thumbnails in catalog/thumbnails/ directory
    - Update the catalog with thumbnail paths
    - Show progress and statistics

    Examples:
        # Generate thumbnails for all images
        vam-generate-thumbnails /path/to/catalog

        # Regenerate all thumbnails at 300x300 pixels
        vam-generate-thumbnails /path/to/catalog --force --size 300

        # Generate with lower quality for faster processing
        vam-generate-thumbnails /path/to/catalog --quality 70
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    version_str = get_version_string()
    console.print(
        f"[bold cyan]VAM Tools - Thumbnail Generator {version_str}[/bold cyan]\n"
    )

    console.print(f"Catalog: {catalog_dir}")
    console.print(f"Thumbnail size: {size}x{size}px")
    console.print(f"JPEG quality: {quality}")
    if force:
        console.print("[yellow]Force mode: Regenerating all thumbnails[/yellow]")
    console.print()

    # Load catalog
    try:
        catalog = CatalogDatabase(catalog_dir)
        catalog.load()
        console.print("[green]✓ Catalog loaded successfully[/green]\n")
    except Exception as e:
        console.print(f"[red]Error loading catalog: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)

    # Get all images
    images = catalog.list_images()
    total_images = len(images)

    if total_images == 0:
        console.print("[yellow]No images found in catalog[/yellow]")
        return

    console.print(f"Found {total_images:,} images in catalog\n")

    # Generate thumbnails with progress bar
    thumbnails_dir = catalog_dir / "thumbnails"
    thumbnails_dir.mkdir(exist_ok=True)

    generated_count = 0
    skipped_count = 0
    error_count = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating thumbnails...", total=total_images)

        for image in images:
            # Get thumbnail path
            thumb_path = get_thumbnail_path(image.id, thumbnails_dir)
            relative_thumb_path = thumb_path.relative_to(catalog_dir)

            # Skip if thumbnail exists and not forcing
            if thumb_path.exists() and not force:
                skipped_count += 1
                progress.update(task, advance=1)
                continue

            # Generate thumbnail
            success = generate_thumbnail(
                source_path=image.source_path,
                output_path=thumb_path,
                size=(size, size),
                quality=quality,
            )

            if success:
                # Update image record with thumbnail path
                image.thumbnail_path = relative_thumb_path
                catalog.add_image(image)  # This updates existing record
                generated_count += 1
            else:
                error_count += 1

            progress.update(task, advance=1)

    # Save catalog with updated thumbnail paths
    try:
        catalog.save()
        console.print("\n[green]✓ Catalog saved with thumbnail paths[/green]")
    except Exception as e:
        console.print(f"\n[red]Error saving catalog: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)

    # Print summary
    console.print("\n[bold cyan]Summary:[/bold cyan]")
    console.print(f"  Total images: {total_images:,}")
    console.print(f"  [green]Generated: {generated_count:,}[/green]")
    console.print(f"  [yellow]Skipped: {skipped_count:,}[/yellow]")
    if error_count > 0:
        console.print(f"  [red]Errors: {error_count:,}[/red]")

    console.print("\n[green]✓ Thumbnail generation complete![/green]")


if __name__ == "__main__":
    generate()
