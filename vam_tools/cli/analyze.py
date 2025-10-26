"""
CLI for running catalog analysis.

This is a temporary CLI for testing the V2 analysis system.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from vam_tools.analysis.scanner import ImageScanner
from vam_tools.core.catalog import CatalogDatabase
from vam_tools.core.types import Statistics
from vam_tools.shared import format_bytes

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@click.command()
@click.argument("catalog_path", type=click.Path())
@click.option(
    "--source",
    "-s",
    multiple=True,
    type=click.Path(exists=True),
    help="Source directory to scan (can specify multiple)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--clear",
    is_flag=True,
    help="Clear existing catalog and start from scratch",
)
@click.option(
    "--repair",
    is_flag=True,
    help="Repair corrupted catalog database",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count)",
)
@click.option(
    "--detect-duplicates",
    is_flag=True,
    help="Detect duplicate and similar images after scanning",
)
@click.option(
    "--similarity-threshold",
    type=int,
    default=5,
    help="Hamming distance threshold for similar images (default: 5, lower is more strict)",
)
def analyze(
    catalog_path: str,
    source: tuple,
    verbose: bool,
    clear: bool,
    repair: bool,
    workers: int,
    detect_duplicates: bool,
    similarity_threshold: int,
) -> None:
    """
    Analyze images and build catalog database.

    By default, resumes from existing catalog if present.
    Use --clear to start fresh or --repair to fix corruption.

    CATALOG_PATH: Path to the organized catalog directory
    """
    setup_logging(verbose)

    catalog_dir = Path(catalog_path)

    # Validate source directories are provided when not in repair mode
    if not repair and not source:
        console.print("[red]Error: --source/-s is required unless using --repair[/red]")
        console.print(
            "Example: vam-analyze /path/to/catalog -s /path/to/photos -s /path/to/more/photos"
        )
        sys.exit(1)

    source_dirs = [Path(s) for s in source] if source else []

    console.print("\n[bold cyan]VAM Tools V2 - Analysis[/bold cyan]\n")
    console.print(f"Catalog: {catalog_dir}")
    if source_dirs:
        console.print(f"Sources: {', '.join(str(s) for s in source_dirs)}")
    console.print()

    # Check if repair mode is requested
    if repair:
        console.print("[yellow]🔧 Repair mode enabled[/yellow]\n")
        catalog_file = catalog_dir / ".catalog.json"
        if not catalog_file.exists():
            console.print("[red]Error: No catalog found to repair[/red]")
            console.print(f"  Catalog file: {catalog_file}")
            sys.exit(1)

        try:
            with CatalogDatabase(catalog_dir) as db:
                console.print("[green]Repairing catalog...[/green]")
                db.repair()
                console.print("[green]✓ Catalog repaired successfully![/green]\n")

                # Show repaired catalog info
                state = db.get_state()
                stats = db.get_statistics()
                console.print(
                    f"Catalog contains: {stats.total_images:,} images, {stats.total_videos:,} videos"
                )
        except Exception as e:
            console.print(f"[red]Error during repair: {e}[/red]")
            if verbose:
                console.print_exception()
            sys.exit(1)
        return

    # Check if clear mode is requested
    catalog_file = catalog_dir / ".catalog.json"
    if clear and catalog_file.exists():
        console.print("[yellow]⚠ Clearing existing catalog...[/yellow]")
        backup_file = catalog_dir / ".catalog.backup.json"

        # Create backup before clearing
        try:
            import shutil

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = catalog_dir / f".catalog.backup.{timestamp}.json"
            shutil.copy2(catalog_file, backup_name)
            console.print(f"[dim]Backup saved to: {backup_name.name}[/dim]")

            # Remove current catalog
            catalog_file.unlink()
            if backup_file.exists():
                backup_file.unlink()
            console.print("[green]✓ Catalog cleared[/green]\n")
        except Exception as e:
            console.print(f"[red]Error clearing catalog: {e}[/red]")
            sys.exit(1)

    # Initialize or load catalog
    try:
        with CatalogDatabase(catalog_dir) as db:
            # Check if catalog exists
            catalog_exists = (catalog_dir / ".catalog.json").exists() and not clear

            if not catalog_exists:
                console.print("[yellow]Initializing new catalog...[/yellow]")
                db.initialize(source_directories=source_dirs)
            else:
                console.print("[green]Loading existing catalog...[/green]")
                state = db.get_state()
                stats = db.get_statistics()

                console.print(f"[dim]Catalog ID: {state.catalog_id}[/dim]")
                console.print(
                    f"[dim]Created: {state.created.strftime('%Y-%m-%d %H:%M:%S') if state.created else 'Unknown'}[/dim]"
                )
                console.print(
                    f"[dim]Last updated: {state.last_updated.strftime('%Y-%m-%d %H:%M:%S') if state.last_updated else 'Unknown'}[/dim]"
                )
                console.print(f"[dim]Current phase: {state.phase.value}[/dim]")

                total_existing = stats.total_images + stats.total_videos
                if total_existing > 0:
                    console.print("\n[cyan]Existing catalog contains:[/cyan]")
                    console.print(f"  • {stats.total_images:,} images")
                    console.print(f"  • {stats.total_videos:,} videos")
                    console.print(
                        f"  • {format_bytes(stats.total_size_bytes)} total size"
                    )
                    console.print(
                        "\n[yellow]Scanning for new/changed files...[/yellow]"
                    )

            # Run scanner
            if workers:
                console.print(
                    f"\n[cyan]Starting scan with {workers} worker processes...[/cyan]\n"
                )
            else:
                import multiprocessing

                workers_count = multiprocessing.cpu_count()
                console.print(
                    f"\n[cyan]Starting scan with {workers_count} worker processes (auto-detected)...[/cyan]\n"
                )

            scanner = ImageScanner(db, workers=workers)
            scanner.scan_directories(source_dirs)

            # Display results
            console.print("\n[green]✓ Scan complete![/green]\n")

            # Show what happened during this scan
            if catalog_exists:
                console.print(f"[dim]Files added: {scanner.files_added:,}[/dim]")
                console.print(
                    f"[dim]Files skipped: {scanner.files_skipped:,} (already in catalog)[/dim]\n"
                )

            # Run duplicate detection if requested
            if detect_duplicates:
                console.print("\n[cyan]Starting duplicate detection...[/cyan]\n")

                from vam_tools.analysis.duplicate_detector import DuplicateDetector

                detector = DuplicateDetector(
                    db, similarity_threshold=similarity_threshold
                )

                # Detect duplicates
                detector.detect_duplicates()

                # Save to catalog
                detector.save_duplicate_groups()

                # Display duplicate detection results
                console.print("\n[green]✓ Duplicate detection complete![/green]\n")

                dup_stats = detector.get_statistics()
                console.print(
                    f"[cyan]Found {dup_stats['total_groups']:,} duplicate groups[/cyan]"
                )
                console.print(
                    f"  • {dup_stats['total_images_in_groups']:,} images in duplicate groups"
                )
                console.print(
                    f"  • {dup_stats['total_unique']:,} unique images (keeping best quality)"
                )
                console.print(
                    f"  • {dup_stats['total_redundant']:,} redundant copies (can be removed)"
                )
                if dup_stats["groups_needing_review"] > 0:
                    console.print(
                        f"  • [yellow]{dup_stats['groups_needing_review']:,} groups need manual review[/yellow]"
                    )
                console.print()

            stats = db.get_statistics()
            display_statistics(stats)
    except Exception as e:
        console.print(f"\n[red]✗ Error loading/processing catalog: {e}[/red]")
        console.print(
            "\n[yellow]The catalog may be corrupted. Try running with --repair to fix it:[/yellow]"
        )
        console.print(f"  [cyan]vam-analyze {catalog_path} --repair[/cyan]")
        console.print(
            "\n[yellow]Or use --clear to start fresh (existing data will be backed up):[/yellow]"
        )
        source_args = (
            " -s ".join(str(s) for s in source_dirs)
            if source_dirs
            else "[your source directories]"
        )
        if source_dirs:
            console.print(
                f"  [cyan]vam-analyze {catalog_path} -s {source_args} --clear[/cyan]"
            )
        else:
            console.print(
                f"  [cyan]vam-analyze {catalog_path} -s [your source directories] --clear[/cyan]"
            )
        if verbose:
            console.print("\n[dim]Full error details:[/dim]")
            console.print_exception()
        sys.exit(1)


def display_statistics(stats: "Statistics") -> None:
    """Display statistics in a nice table."""
    table = Table(title="Catalog Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total Images", str(stats.total_images))
    table.add_row("Total Videos", str(stats.total_videos))
    table.add_row("Total Size", format_bytes(stats.total_size_bytes))
    table.add_row("Images with no date", str(stats.no_date))

    console.print(table)


if __name__ == "__main__":
    try:
        analyze()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            console.print_exception()
        sys.exit(1)
