"""
CLI for running catalog analysis.

This is a temporary CLI for testing the V2 analysis system.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from vam_tools.analysis.perceptual_hash import HashMethod
from vam_tools.analysis.scanner import ImageScanner
from vam_tools.core.catalog import CatalogDatabase
from vam_tools.core.performance_stats import PerformanceTracker
from vam_tools.core.types import Statistics
from vam_tools.shared import format_bytes
from vam_tools.version import get_version_string

console = Console()
logger = logging.getLogger(__name__)


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
@click.option(
    "--hash-methods",
    type=click.Choice(["dhash", "ahash", "whash", "all"], case_sensitive=False),
    multiple=True,
    default=["dhash", "ahash"],
    help="Perceptual hash methods to use (default: dhash+ahash, can specify multiple)",
)
@click.option(
    "--hash-size",
    type=int,
    default=8,
    help="Size of perceptual hash in bits (default: 8 = 64-bit hash)",
)
@click.option(
    "--recompute-hashes",
    is_flag=True,
    help="Recompute perceptual hashes even if they already exist",
)
@click.option(
    "--gpu",
    is_flag=True,
    help="Enable GPU acceleration for hash computation (requires PyTorch with CUDA)",
)
@click.option(
    "--gpu-batch-size",
    type=int,
    default=None,
    help="GPU batch size for hash computation (default: auto-detect based on VRAM)",
)
@click.option(
    "--use-faiss",
    is_flag=True,
    help="Force enable FAISS for fast similarity search (auto-detects by default)",
)
@click.option(
    "--no-thumbnails",
    is_flag=True,
    help="Skip thumbnail generation during analysis (can generate later with vam-generate-thumbnails)",
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
    hash_methods: tuple,
    hash_size: int,
    recompute_hashes: bool,
    gpu: bool,
    gpu_batch_size: int,
    use_faiss: bool,
    no_thumbnails: bool,
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

    version_str = get_version_string()
    console.print(
        f"\n[bold cyan]VAM Tools V2 - Analysis[/bold cyan] [dim]v{version_str}[/dim]\n"
    )
    console.print(f"Catalog: {catalog_dir}")
    if source_dirs:
        console.print(f"Sources: {', '.join(str(s) for s in source_dirs)}")
    console.print()

    # Check if repair mode is requested
    if repair:
        console.print("[yellow]🔧 Repair mode enabled[/yellow]\n")
        catalog_file = catalog_dir / "catalog.json"
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
    catalog_file = catalog_dir / "catalog.json"
    if clear and catalog_file.exists():
        console.print("[yellow]⚠ Clearing existing catalog...[/yellow]")
        backup_file = catalog_dir / ".backup.json"

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
            catalog_exists = (catalog_dir / "catalog.json").exists() and not clear

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

            # Create performance tracker with catalog-writing callback
            # This enables real-time performance updates via polling
            def performance_update_callback(stats_data: Dict) -> None:
                """Write performance stats to catalog for polling endpoint."""
                try:
                    # Get existing stats or create new structure
                    analysis_stats = db.get_performance_statistics() or {
                        "last_run": None,
                        "history": [],
                        "total_runs": 0,
                        "total_files_analyzed": 0,
                        "total_time_seconds": 0.0,
                        "average_throughput": 0.0,
                    }
                    # Update last_run with current metrics
                    analysis_stats["last_run"] = stats_data
                    db.store_performance_statistics(analysis_stats)
                    # Save to disk so polling can read it
                    db.save(create_backup=False)
                except Exception as e:
                    # Don't break analysis if performance tracking fails
                    logger.debug(f"Failed to write performance stats: {e}")

            perf_tracker = PerformanceTracker(
                update_callback=performance_update_callback,
                update_interval=5.0,  # Write to catalog every 5 seconds
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

            scanner = ImageScanner(
                db,
                workers=workers,
                perf_tracker=perf_tracker,
                generate_thumbnails=not no_thumbnails,
            )
            scanner.scan_directories(source_dirs)

            # Store intermediate performance stats
            metrics = perf_tracker.metrics
            analysis_stats = db.get_performance_statistics() or {}
            if "last_run" not in analysis_stats or analysis_stats["last_run"] is None:
                analysis_stats = {
                    "last_run": None,
                    "history": [],
                    "total_runs": 0,
                    "total_files_analyzed": 0,
                    "total_time_seconds": 0.0,
                    "average_throughput": 0.0,
                }
            analysis_stats["last_run"] = metrics.model_dump(mode="json")
            db.store_performance_statistics(analysis_stats)
            db.save()

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

                # Convert CLI hash method strings to HashMethod enum
                selected_methods = []
                if "all" in hash_methods:
                    selected_methods = [
                        HashMethod.DHASH,
                        HashMethod.AHASH,
                        HashMethod.WHASH,
                    ]
                else:
                    method_map = {
                        "dhash": HashMethod.DHASH,
                        "ahash": HashMethod.AHASH,
                        "whash": HashMethod.WHASH,
                    }
                    selected_methods = [
                        method_map[m.lower()]
                        for m in hash_methods
                        if m.lower() in method_map
                    ]

                # Show selected configuration
                console.print(
                    f"[dim]Using hash methods: {', '.join(m.value for m in selected_methods)}[/dim]"
                )
                console.print(f"[dim]Hash size: {hash_size}-bit[/dim]")
                console.print(
                    f"[dim]Similarity threshold: {similarity_threshold}[/dim]"
                )
                if recompute_hashes:
                    console.print("[dim]Recomputing existing hashes[/dim]")
                if gpu:
                    console.print("[green]GPU acceleration enabled[/green]")
                    if gpu_batch_size:
                        console.print(f"[dim]GPU batch size: {gpu_batch_size}[/dim]")
                # Note: FAISS is now auto-enabled if available, no need to print unless explicitly requested
                console.print()

                detector = DuplicateDetector(
                    db,
                    similarity_threshold=similarity_threshold,
                    hash_size=hash_size,
                    hash_methods=selected_methods,
                    use_gpu=gpu,
                    gpu_batch_size=gpu_batch_size,
                    use_faiss=use_faiss,
                    perf_tracker=perf_tracker,
                )

                # Detect duplicates
                detector.detect_duplicates(recompute_hashes=recompute_hashes)

                # Save to catalog
                detector.save_duplicate_groups()
                detector.save_problematic_files()

                # Store intermediate performance stats after duplicate detection
                metrics = perf_tracker.metrics
                analysis_stats = db.get_performance_statistics() or {}
                if "last_run" not in analysis_stats:
                    analysis_stats = {
                        "last_run": None,
                        "history": [],
                        "total_runs": 0,
                        "total_files_analyzed": 0,
                        "total_time_seconds": 0.0,
                        "average_throughput": 0.0,
                    }
                analysis_stats["last_run"] = metrics.model_dump(mode="json")
                db.store_performance_statistics(analysis_stats)
                db.save()

                # Update statistics with problematic files count
                stats = db.get_statistics()
                stats.problematic_files = len(detector.problematic_files)
                db.update_statistics(stats)

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

                # Display problematic files statistics
                if detector.problematic_files:
                    console.print(
                        f"\n[yellow]⚠ {len(detector.problematic_files):,} files had processing issues[/yellow]"
                    )
                    # Count by category
                    from collections import Counter

                    categories = Counter(
                        f.category.value for f in detector.problematic_files
                    )
                    for category, count in categories.most_common():
                        console.print(f"  • {category}: {count:,}")
                    console.print(
                        "[dim]View these files with: vam-web /path/to/catalog (Problematic Files tab)[/dim]"
                    )

                console.print()

            # Finalize performance tracking
            final_metrics = perf_tracker.finalize()

            # Store final performance statistics in catalog
            perf_stats_data = db.get_performance_statistics() or {}

            # Initialize AnalysisStatistics if needed
            if not perf_stats_data or "history" not in perf_stats_data:
                perf_stats_data = {
                    "last_run": None,
                    "history": [],
                    "total_runs": 0,
                    "total_files_analyzed": 0,
                    "total_time_seconds": 0.0,
                    "average_throughput": 0.0,
                }

            # Add current run to history
            perf_stats_data["last_run"] = final_metrics.model_dump(mode="json")
            perf_stats_data["history"].insert(0, final_metrics.model_dump(mode="json"))
            perf_stats_data["history"] = perf_stats_data["history"][:10]  # Keep last 10
            perf_stats_data["total_runs"] = perf_stats_data.get("total_runs", 0) + 1
            perf_stats_data["total_files_analyzed"] = (
                perf_stats_data.get("total_files_analyzed", 0)
                + final_metrics.total_files_analyzed
            )
            perf_stats_data["total_time_seconds"] = (
                perf_stats_data.get("total_time_seconds", 0.0)
                + final_metrics.total_duration_seconds
            )
            if perf_stats_data["total_time_seconds"] > 0:
                perf_stats_data["average_throughput"] = (
                    perf_stats_data["total_files_analyzed"]
                    / perf_stats_data["total_time_seconds"]
                )

            db.store_performance_statistics(perf_stats_data)
            db.save()

            stats = db.get_statistics()
            display_statistics(stats)

            # Display performance summary
            console.print("\n[bold cyan]Performance Summary[/bold cyan]\n")
            summary = final_metrics.get_summary_report()
            console.print(summary)
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
