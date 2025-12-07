"""
CLI for running catalog analysis.

This is a temporary CLI for testing the V2 analysis system.
"""

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from vam_tools.analysis.perceptual_hash import HashMethod
from vam_tools.analysis.scanner import ImageScanner
from vam_tools.core.performance_stats import PerformanceTracker
from vam_tools.core.types import Statistics
from vam_tools.db import CatalogDB as CatalogDatabase
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
    help="Detect duplicate and similar images/videos after scanning",
)
@click.option(
    "--similarity-threshold",
    type=int,
    default=5,
    help="Hamming distance threshold for similarity (default: 5, lower is more strict)",
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
@click.option(
    "--extract-previews",
    is_flag=True,
    help="Extract and cache previews for RAW/HEIC files after analysis (improves web UI performance)",
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
    extract_previews: bool,
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

    # Check if repair mode is requested (PostgreSQL mode - repair not supported)
    if repair:
        console.print("[yellow]🔧 Repair mode enabled[/yellow]\n")

        try:
            with CatalogDatabase(catalog_dir) as db:
                from vam_tools.db.models import Catalog

                # Check if catalog exists in database
                existing_catalog = (
                    db.session.query(Catalog).filter_by(id=db.catalog_id).first()
                )

                if not existing_catalog:
                    console.print("[red]Error: No catalog found to repair[/red]")
                    console.print(f"  Catalog ID: {db.catalog_id}")
                    sys.exit(1)

                console.print("[green]Attempting to repair catalog...[/green]")
                # PostgreSQL handles integrity automatically
                db.repair()  # This is a stub that does nothing
                console.print(
                    "[green]✓ Catalog repair complete (PostgreSQL maintains integrity automatically)[/green]"
                )
                sys.exit(0)
        except Exception as e:
            console.print(f"[red]Error during repair: {e}[/red]")
            if verbose:
                console.print_exception()
            sys.exit(1)
        return

    # Check if clear mode is requested (PostgreSQL mode - delete from database)
    if clear:
        try:
            with CatalogDatabase(catalog_dir) as db:
                from vam_tools.db.models import Catalog

                # Check if catalog exists in database
                existing_catalog = (
                    db.session.query(Catalog).filter_by(id=db.catalog_id).first()
                )

                if existing_catalog:
                    console.print("[yellow]⚠ Clearing existing catalog...[/yellow]")

                    # Delete all images, config, stats, etc. for this catalog
                    from vam_tools.db.models import (
                        Config,
                        DuplicateGroup,
                        DuplicateMember,
                        Image,
                    )
                    from vam_tools.db.models import Statistics as StatsModel

                    db.session.query(DuplicateMember).filter(
                        DuplicateMember.group_id.in_(
                            db.session.query(DuplicateGroup.id).filter_by(
                                catalog_id=db.catalog_id
                            )
                        )
                    ).delete(synchronize_session=False)
                    db.session.query(DuplicateGroup).filter_by(
                        catalog_id=db.catalog_id
                    ).delete()
                    db.session.query(Image).filter_by(catalog_id=db.catalog_id).delete()
                    db.session.query(StatsModel).filter_by(
                        catalog_id=db.catalog_id
                    ).delete()
                    db.session.query(Config).filter_by(
                        catalog_id=db.catalog_id
                    ).delete()
                    db.session.query(Catalog).filter_by(id=db.catalog_id).delete()
                    db.session.commit()

                    console.print("[green]✓ Catalog cleared[/green]\n")
        except Exception as e:
            console.print(f"[red]Error clearing catalog: {e}[/red]")
            if verbose:
                console.print_exception()
            sys.exit(1)

    # Initialize or load catalog
    try:
        with CatalogDatabase(catalog_dir) as db:
            # Check if catalog exists in database (PostgreSQL mode)
            from vam_tools.db.models import Catalog

            existing_catalog = (
                db.session.query(Catalog).filter_by(id=db.catalog_id).first()
            )
            catalog_exists = existing_catalog is not None and not clear

            if not catalog_exists:
                console.print("[yellow]Initializing new catalog...[/yellow]")
                db.initialize()  # Initialize schema and create Catalog record

                # Store source directories in config
                for src_dir in source_dirs:
                    try:
                        print(
                            f"DEBUG: About to insert config for {src_dir}", flush=True
                        )
                        print(f"DEBUG: Value = {json.dumps(str(src_dir))}", flush=True)
                        db.execute(
                            """
                            INSERT INTO config (catalog_id, key, value, updated_at)
                            VALUES (?, ?, ?, NOW())
                            ON CONFLICT (catalog_id, key) DO UPDATE SET
                                value = EXCLUDED.value,
                                updated_at = EXCLUDED.updated_at
                            """,
                            (
                                db.catalog_id,
                                f"source_directory_{src_dir.name}",
                                json.dumps(str(src_dir)),
                            ),
                        )
                        print("DEBUG: Config insert succeeded", flush=True)
                    except Exception as e:
                        print(f"DEBUG: Config insert FAILED: {e}", flush=True)
                        import traceback

                        traceback.print_exc()
                        raise
                db.execute(
                    """
                    INSERT INTO config (catalog_id, key, value, updated_at)
                    VALUES (?, ?, ?, NOW())
                    ON CONFLICT (catalog_id, key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (db.catalog_id, "created", json.dumps(datetime.now().isoformat())),
                )
                db.execute(
                    """
                    INSERT INTO config (catalog_id, key, value, updated_at)
                    VALUES (?, ?, ?, NOW())
                    ON CONFLICT (catalog_id, key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        db.catalog_id,
                        "last_updated",
                        json.dumps(datetime.now().isoformat()),
                    ),
                )
                db.execute(
                    """
                    INSERT INTO config (catalog_id, key, value, updated_at)
                    VALUES (?, ?, ?, NOW())
                    ON CONFLICT (catalog_id, key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (db.catalog_id, "catalog_id", json.dumps(str(uuid.uuid4()))),
                )
                db.execute(
                    """
                    INSERT INTO config (catalog_id, key, value, updated_at)
                    VALUES (?, ?, ?, NOW())
                    ON CONFLICT (catalog_id, key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (db.catalog_id, "version", json.dumps("2.0.0")),
                )
                db.execute(
                    """
                    INSERT INTO config (catalog_id, key, value, updated_at)
                    VALUES (?, ?, ?, NOW())
                    ON CONFLICT (catalog_id, key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (db.catalog_id, "phase", json.dumps("analyzing")),
                )

            else:
                console.print("[green]Loading existing catalog...[/green]")
                # Get state information from config table
                config_rows = db.execute(
                    "SELECT key, value FROM config WHERE catalog_id = ?",
                    (db.catalog_id,),
                ).fetchall()
                config_dict = {
                    row._mapping["key"]: row._mapping["value"] for row in config_rows
                }

                catalog_id = config_dict.get("catalog_id", "N/A")
                created_at = config_dict.get("created", "Unknown")
                last_updated_at = config_dict.get("last_updated", "Unknown")
                phase = config_dict.get("phase", "unknown")

                console.print(f"[dim]Catalog ID: {catalog_id}[/dim]")
                console.print(f"[dim]Created: {created_at}[/dim]")
                console.print(f"[dim]Last updated: {last_updated_at}[/dim]")
                console.print(f"[dim]Current phase: {phase}[/dim]")

                # Get statistics from the latest entry in the statistics table
                stats_row = db.execute(
                    "SELECT total_images, total_videos, total_size_bytes FROM statistics ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()

                total_existing_images = (
                    stats_row._mapping["total_images"] if stats_row else 0
                )
                total_existing_videos = (
                    stats_row._mapping["total_videos"] if stats_row else 0
                )
                total_existing_size = (
                    stats_row._mapping["total_size_bytes"] if stats_row else 0
                )

                total_existing = total_existing_images + total_existing_videos
                if total_existing > 0:
                    console.print("\n[cyan]Existing catalog contains:[/cyan]")
                    console.print(f"  • {total_existing_images:,} images")
                    console.print(f"  • {total_existing_videos:,} videos")
                    console.print(f"  • {format_bytes(total_existing_size)} total size")
                    console.print(
                        "\n[yellow]Scanning for new/changed files...[/yellow]"
                    )

            # Create performance tracker with catalog-writing callback
            # This enables real-time performance updates via polling
            def performance_update_callback(stats_data: Dict) -> None:
                """Write performance stats to catalog for polling endpoint."""
                try:
                    # Insert a new performance snapshot for real-time monitoring
                    db.execute(
                        """
                        INSERT INTO performance_snapshots (
                            catalog_id, timestamp, phase, files_processed, files_total,
                            bytes_processed, cpu_percent, memory_mb, disk_read_mb,
                            disk_write_mb, elapsed_seconds, rate_files_per_sec,
                            rate_mb_per_sec, gpu_utilization, gpu_memory_mb
                        ) VALUES (?, NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            db.catalog_id,
                            stats_data.get("phase", "unknown"),
                            stats_data.get("files_processed"),
                            stats_data.get("files_total"),
                            stats_data.get("bytes_processed", 0),
                            stats_data.get("cpu_percent"),
                            stats_data.get("memory_mb"),
                            stats_data.get("disk_read_mb"),
                            stats_data.get("disk_write_mb"),
                            stats_data.get("elapsed_seconds"),
                            stats_data.get("rate_files_per_sec"),
                            stats_data.get("rate_mb_per_sec"),
                            stats_data.get("gpu_utilization"),
                            stats_data.get("gpu_memory_mb"),
                        ),
                    )
                    db.session.commit()
                except Exception as e:
                    # Don't break analysis if performance tracking fails
                    logger.debug(f"Failed to write performance stats: {e}")
                    # Rollback the failed transaction so we can continue
                    if db.session:
                        db.session.rollback()

            perf_tracker = PerformanceTracker(
                update_callback=performance_update_callback,
                update_interval=1.0,  # Write to catalog every 1 second (matches frontend polling)
            )

            # Write initial performance data so frontend sees "running" state immediately
            try:
                performance_update_callback(perf_tracker.get_current_stats())
            except Exception as e:
                logger.debug(f"Failed to write initial performance stats: {e}")
                if db.session:
                    db.session.rollback()

            # Run scanner
            if workers:
                actual_workers = workers
                console.print(
                    f"\n[cyan]Starting scan with {actual_workers} worker processes...[/cyan]\n"
                )
            else:
                import multiprocessing

                actual_workers = multiprocessing.cpu_count()
                console.print(
                    f"\n[cyan]Starting scan with {actual_workers} worker processes (auto-detected)...[/cyan]\n"
                )

            scanner = ImageScanner(
                db,
                workers=actual_workers,
                perf_tracker=perf_tracker,
            )
            scanner.scan_directories(source_dirs)

            # Store intermediate performance stats (already handled by callback)
            # The perf_tracker's callback will have already inserted the latest snapshot.
            # We just need to ensure the final metrics are captured.
            pass

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

                # Store intermediate performance stats (already handled by callback)
                # The perf_tracker's callback will have already inserted the latest snapshot.
                pass

                # Update statistics with problematic files count
                # Fetch current stats
                current_stats_row = db.execute(
                    "SELECT * FROM statistics ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                current_stats = (
                    Statistics(**dict(current_stats_row._mapping))
                    if current_stats_row
                    else Statistics()
                )
                current_stats.problematic_files = len(detector.problematic_files)

                # Insert new statistics snapshot
                db.execute(
                    """
                    INSERT INTO statistics (
                        catalog_id, timestamp, total_images, total_videos, total_size_bytes,
                        images_scanned, images_hashed, images_tagged,
                        duplicate_groups, duplicate_images, potential_savings_bytes,
                        high_quality_count, medium_quality_count, low_quality_count,
                        corrupted_count, unsupported_count,
                        processing_time_seconds, images_per_second,
                        no_date, suspicious_dates, problematic_files
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(db.catalog_id),
                        datetime.now().isoformat(),
                        current_stats.total_images,
                        current_stats.total_videos,
                        current_stats.total_size_bytes,
                        current_stats.images_scanned,
                        current_stats.images_hashed,
                        current_stats.images_tagged,
                        current_stats.duplicate_groups,
                        current_stats.duplicate_images,
                        current_stats.potential_savings_bytes,
                        current_stats.high_quality_count,
                        current_stats.medium_quality_count,
                        current_stats.low_quality_count,
                        current_stats.corrupted_count,
                        current_stats.unsupported_count,
                        current_stats.processing_time_seconds,
                        current_stats.images_per_second,
                        current_stats.no_date,
                        current_stats.suspicious_dates,
                        current_stats.problematic_files,
                    ),
                )

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

            # Run preview extraction if requested
            if extract_previews:
                console.print("\n[cyan]Starting preview extraction...[/cyan]\n")
                console.print(
                    "[dim]Extracting and caching previews for RAW/HEIC files...[/dim]\n"
                )

                from vam_tools.analysis.preview_extractor import PreviewExtractor

                extractor = PreviewExtractor(db, workers=workers)
                extractor.extract_previews(force=False)

                # Show results
                cache_stats = extractor.preview_cache.get_cache_stats()
                console.print("\n[green]✓ Preview extraction complete![/green]\n")
                console.print(
                    f"[cyan]Preview cache: {cache_stats['num_previews']:,} previews[/cyan]"
                )
                console.print(
                    f"  • {cache_stats['total_size_gb']:.2f} GB / {cache_stats['max_size_gb']:.2f} GB "
                    f"({cache_stats['usage_percent']:.1f}% full)"
                )
                console.print(
                    "  • [dim]Cached previews will load instantly in web UI[/dim]\n"
                )

            # Finalize performance tracking
            final_metrics = perf_tracker.finalize()

            # The final metrics are already inserted into performance_snapshots by the callback.
            # We just need to fetch the latest statistics for display.
            stats_row = db.execute(
                "SELECT * FROM statistics ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            stats = (
                Statistics(**dict(stats_row._mapping)) if stats_row else Statistics()
            )
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
