"""
CLI for catalog reorganization.

Reorganizes photo files into a date-based directory structure.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..core.catalog_reorganization import (
    CatalogReorganizer,
    ConflictResolution,
    OrganizationStrategy,
)
from ..core.image_utils import collect_image_files, setup_logging

console = Console()


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
    "--dry-run",
    is_flag=True,
    help="Simulate reorganization without moving files",
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
@click.option(
    "--yes",
    is_flag=True,
    help="Skip confirmation prompt",
)
def cli(
    directory: str,
    output: str,
    strategy: str,
    conflict: str,
    copy: bool,
    recursive: bool,
    no_recursive: bool,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
    yes: bool,
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

    # Setup logging
    setup_logging(verbose=verbose, quiet=quiet)

    directory_path = Path(directory).resolve()
    output_path = Path(output).resolve()

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

    # Display configuration
    if not quiet:
        console.print(f"\n[bold cyan]Catalog Reorganizer[/bold cyan]\n")

        config_table = Table(show_header=False, box=None)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Source directory", str(directory_path))
        config_table.add_row("Output directory", str(output_path))
        config_table.add_row("Organization strategy", strategy)
        config_table.add_row("Conflict resolution", conflict)
        config_table.add_row("Mode", "copy" if copy else "move")
        config_table.add_row("Recursive scan", str(recursive))
        config_table.add_row("Dry run", str(dry_run))

        console.print(config_table)
        console.print()

    # Safety check: prevent organizing into source directory
    if output_path == directory_path or output_path in directory_path.parents:
        console.print(
            "[red]Error: Output directory cannot be the same as or "
            "parent of source directory[/red]"
        )
        sys.exit(1)

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

    if not quiet:
        console.print(f"Found [bold]{len(image_files)}[/bold] image files\n")

    # Confirmation prompt
    if not dry_run and not yes and not quiet:
        warning_text = (
            f"About to {'copy' if copy else 'move'} {len(image_files)} files.\n"
            f"This operation {'will NOT modify' if copy else 'WILL MODIFY'} "
            f"the source directory.\n\n"
            f"{'[yellow]Consider running with --dry-run first![/yellow]' if not copy else ''}"
        )

        console.print(
            Panel(
                warning_text,
                title="⚠️  Confirmation Required",
                border_style="yellow",
            )
        )

        if not click.confirm("Do you want to continue?", default=False):
            console.print("[yellow]Operation cancelled.[/yellow]")
            sys.exit(0)

    # Create reorganizer
    reorganizer = CatalogReorganizer(
        output_directory=output_path,
        strategy=org_strategy,
        conflict_resolution=conflict_resolution,
        copy_mode=copy,
    )

    # Execute reorganization
    if not quiet:
        mode_text = "[DRY RUN]" if dry_run else ""
        console.print(f"\n{mode_text} [cyan]Starting reorganization...[/cyan]\n")

    stats = reorganizer.reorganize(image_files, dry_run=dry_run)

    # Display results
    if not quiet:
        result_table = Table(title="Reorganization Results")
        result_table.add_column("Status", style="cyan")
        result_table.add_column("Count", style="green")

        if copy:
            result_table.add_row("Copied", str(stats.get("copied", 0)))
        else:
            result_table.add_row("Moved", str(stats.get("moved", 0)))

        result_table.add_row("Skipped", str(stats.get("skipped", 0)))
        result_table.add_row("Errors", str(stats.get("errors", 0)))

        console.print(result_table)

        if dry_run:
            console.print(
                "\n[yellow]This was a dry run. No files were actually modified.[/yellow]"
            )
            console.print(
                "[yellow]Run without --dry-run to perform the reorganization.[/yellow]"
            )
        else:
            console.print(f"\n[green]Reorganization complete![/green]")
            console.print(f"[green]Files are in:[/green] {output_path}")

            if stats.get("errors", 0) > 0:
                console.print(
                    f"\n[red]Warning: {stats['errors']} errors occurred. "
                    f"Check logs for details.[/red]"
                )


if __name__ == "__main__":
    cli()
