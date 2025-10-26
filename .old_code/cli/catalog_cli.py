"""
CLI for catalog reorganization.

Reorganizes photo files into a date-based directory structure.
"""

import sys
from pathlib import Path

import click
from rich.panel import Panel

from vam_tools.shared import collect_image_files

from ..core.catalog_reorganization import (
    CatalogReorganizer,
    ConflictResolution,
    OrganizationStrategy,
)
from .base import CLIDisplay, common_options, file_scan_options, init_logging


@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="Output directory for reorganized photos",
)
@click.option(
    "-s",
    "--strategy",
    type=click.Choice(
        ["year/month-day", "year/month", "year", "flat"],
        case_sensitive=False,
    ),
    default="year/month-day",
    help="Directory organization strategy",
)
@click.option(
    "--conflict",
    type=click.Choice(["skip", "rename", "overwrite"], case_sensitive=False),
    default="rename",
    help="How to handle filename conflicts",
)
@click.option(
    "--copy",
    is_flag=True,
    help="Copy files instead of moving them",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Simulate reorganization without moving files",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip confirmation prompt",
)
@file_scan_options
@common_options
@init_logging
def cli(
    directory: str,
    output: str,
    strategy: str,
    conflict: str,
    copy: bool,
    dry_run: bool,
    yes: bool,
    recursive: bool,
    no_recursive: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """
    Reorganize photo catalog into date-based structure.

    Organizes photos into directories based on their dates extracted from
    EXIF metadata, filenames, or directory structure.

    DIRECTORY: Path to directory containing images to reorganize
    """
    # Handle recursive flag
    if no_recursive:
        recursive = False

    directory_path = Path(directory).resolve()
    output_path = Path(output).resolve()

    # Initialize display helper
    display = CLIDisplay(quiet=quiet)

    # Convert string strategy to enum
    strategy_map = {
        "year/month-day": OrganizationStrategy.YEAR_MONTH_DAY,
        "year/month": OrganizationStrategy.YEAR_MONTH,
        "year": OrganizationStrategy.YEAR,
        "flat": OrganizationStrategy.FLAT_DATE,
    }
    org_strategy = strategy_map[strategy]

    # Convert conflict resolution to enum
    conflict_map = {
        "skip": ConflictResolution.SKIP,
        "rename": ConflictResolution.RENAME,
        "overwrite": ConflictResolution.OVERWRITE,
    }
    conflict_resolution = conflict_map[conflict]

    # Display header and configuration
    display.print_header("Catalog Reorganizer")
    display.print_config(
        {
            "Source directory": str(directory_path),
            "Output directory": str(output_path),
            "Organization strategy": strategy,
            "Conflict resolution": conflict,
            "Mode": "copy" if copy else "move",
            "Recursive scan": str(recursive),
            "Dry run": str(dry_run),
        }
    )

    # Safety check: prevent organizing into source directory
    if output_path == directory_path or output_path in directory_path.parents:
        display.print_error(
            "Error: Output directory cannot be the same as or "
            "parent of source directory"
        )
        sys.exit(1)

    # Collect image files
    with display.spinner_progress("Collecting image files...") as progress:
        task = progress.add_task("Collecting image files...", total=None)
        image_files = collect_image_files(directory_path, recursive=recursive)
        progress.update(task, completed=True)

    if not image_files:
        display.print_warning("No image files found.")
        sys.exit(0)

    display.print_info(f"Found [bold]{len(image_files)}[/bold] image files\n")

    # Confirmation prompt
    if not dry_run and not yes and not quiet:
        warning_text = (
            f"About to {'copy' if copy else 'move'} {len(image_files)} files.\n"
            f"This operation {'will NOT modify' if copy else 'WILL MODIFY'} "
            f"the source directory.\n\n"
            f"{'[yellow]Consider running with --dry-run first![/yellow]' if not copy else ''}"
        )

        display.console.print(
            Panel(
                warning_text,
                title="⚠️  Confirmation Required",
                border_style="yellow",
            )
        )

        if not click.confirm("Do you want to continue?", default=False):
            display.print_warning("Operation cancelled.")
            sys.exit(0)

    # Create reorganizer
    reorganizer = CatalogReorganizer(
        output_directory=output_path,
        strategy=org_strategy,
        conflict_resolution=conflict_resolution,
        copy_mode=copy,
    )

    # Execute reorganization
    mode_text = "[DRY RUN] " if dry_run else ""
    display.print_info(f"\n{mode_text}[cyan]Starting reorganization...[/cyan]\n")

    stats = reorganizer.reorganize(image_files, dry_run=dry_run)

    # Display results
    metrics = {
        "Copied" if copy else "Moved": stats.get("copied" if copy else "moved", 0),
        "Skipped": stats.get("skipped", 0),
        "Errors": stats.get("errors", 0),
    }
    display.print_summary(metrics, title="Reorganization Results")

    if dry_run:
        display.print_warning("\nThis was a dry run. No files were actually modified.")
        display.print_warning("Run without --dry-run to perform the reorganization.")
    else:
        display.print_success("\nReorganization complete!")
        display.print_success(f"Files are in: {output_path}")

        if stats.get("errors", 0) > 0:
            display.print_error(
                f"\nWarning: {stats['errors']} errors occurred. "
                f"Check logs for details."
            )


if __name__ == "__main__":
    cli()
