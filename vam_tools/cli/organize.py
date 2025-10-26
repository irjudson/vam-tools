"""
CLI command for organizing files.

Reorganizes files from catalog into a clean chronological structure.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..core.catalog import CatalogDatabase
from ..organization import (
    DirectoryStructure,
    FileOrganizer,
    NamingStrategy,
    OrganizationOperation,
    OrganizationResult,
    OrganizationStrategy,
)

console = Console()


@click.command()
@click.argument("catalog_path", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
@click.option(
    "--operation",
    type=click.Choice(["copy", "move"], case_sensitive=False),
    default="copy",
    help="Operation type: copy (safe) or move (deletes originals)",
)
@click.option(
    "--structure",
    type=click.Choice(
        ["YYYY-MM", "YYYY/MM", "YYYY-MM-DD", "YYYY", "FLAT"],
        case_sensitive=False,
    ),
    default="YYYY-MM",
    help="Directory structure pattern",
)
@click.option(
    "--naming",
    type=click.Choice(
        ["date_time_checksum", "date_time_original", "original", "checksum"],
        case_sensitive=False,
    ),
    default="date_time_checksum",
    help="File naming strategy",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview changes without executing (RECOMMENDED FIRST)",
)
@click.option(
    "--no-verify",
    is_flag=True,
    default=False,
    help="Skip checksum verification after copy/move",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing files (default: skip)",
)
@click.option(
    "--rollback",
    type=str,
    help="Rollback a previous transaction by ID",
)
@click.option(
    "--resume",
    type=str,
    help="Resume an interrupted transaction by ID",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Verbose output",
)
def organize(
    catalog_path: str,
    output_directory: str,
    operation: str,
    structure: str,
    naming: str,
    dry_run: bool,
    no_verify: bool,
    overwrite: bool,
    rollback: str,
    resume: str,
    verbose: bool,
) -> None:
    """
    Organize files from CATALOG_PATH into OUTPUT_DIRECTORY.

    This command reorganizes your files into a clean, chronological structure
    based on the dates extracted during analysis.

    \b
    Examples:
        # DRY RUN (preview changes - always do this first!)
        vam-organize /path/to/catalog /path/to/organized --dry-run

        # Copy files to organized structure (safe, keeps originals)
        vam-organize /path/to/catalog /path/to/organized --operation copy

        # Move files (deletes originals - be careful!)
        vam-organize /path/to/catalog /path/to/organized --operation move

        # Custom directory structure
        vam-organize /path/to/catalog /path/to/organized \\
            --structure YYYY/MM --naming date_time_original

        # Rollback a transaction
        vam-organize /path/to/catalog /path/to/organized \\
            --rollback abc123...

    \b
    Directory Structures:
        YYYY-MM:     2023-06/
        YYYY/MM:     2023/06/
        YYYY-MM-DD:  2023-06-15/
        YYYY:        2023/
        FLAT:        (all in output_directory)

    \b
    Naming Strategies:
        date_time_checksum:  2023-06-15_143022_abc123.jpg
        date_time_original:  2023-06-15_143022_IMG_1234.jpg
        original:            IMG_1234.jpg (keep original name)
        checksum:            abc123def456.jpg

    \b
    Safety Features:
        • Dry-run mode previews changes without executing
        • Checksum verification ensures file integrity
        • Transaction logging enables rollback
        • File locking prevents corruption
        • Automatic backup before destructive operations

    \b
    IMPORTANT:
        • ALWAYS run with --dry-run first to preview changes
        • Use --operation copy (default) to keep original files
        • Only use --operation move after verifying with dry-run
        • Transactions can be rolled back if something goes wrong
    """
    # Setup logging
    import logging

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s - %(message)s",
    )

    catalog_dir = Path(catalog_path)
    output_dir = Path(output_directory)

    # Load catalog
    try:
        with CatalogDatabase(catalog_dir) as db:
            # Create strategy
            strategy = OrganizationStrategy(
                directory_structure=DirectoryStructure(structure),
                naming_strategy=NamingStrategy(naming),
                handle_duplicates=True,
            )

            # Create organizer
            organizer = FileOrganizer(
                catalog=db,
                strategy=strategy,
                output_directory=output_dir,
                operation=OrganizationOperation(operation),
            )

            # Handle rollback
            if rollback:
                console.print(
                    f"[yellow]Rolling back transaction {rollback}...[/yellow]"
                )
                try:
                    organizer.rollback(rollback)
                    console.print("[green]✓ Rollback complete[/green]")
                except Exception as e:
                    console.print(f"[red]✗ Rollback failed: {e}[/red]")
                    sys.exit(1)
                return

            # Handle resume
            if resume:
                console.print(f"[yellow]Resuming transaction {resume}...[/yellow]")
                try:
                    result = organizer.resume(resume)
                    _display_result(result)
                except Exception as e:
                    console.print(f"[red]✗ Resume failed: {e}[/red]")
                    sys.exit(1)
                return

            # Show configuration
            console.print("\n[cyan]Organization Configuration:[/cyan]")
            console.print(f"  Catalog: {catalog_dir}")
            console.print(f"  Output: {output_dir}")
            console.print(f"  Operation: {operation}")
            console.print(f"  Structure: {structure}")
            console.print(f"  Naming: {naming}")
            console.print(f"  Dry run: {'YES' if dry_run else 'NO'}")
            console.print(f"  Verify checksums: {'NO' if no_verify else 'YES'}")
            console.print(f"  Overwrite existing: {'YES' if overwrite else 'NO'}")

            if dry_run:
                console.print(
                    "\n[yellow]⚠ DRY RUN MODE - No files will be modified[/yellow]"
                )
            elif operation == "move":
                console.print(
                    "\n[red]⚠ WARNING: MOVE operation will delete original files![/red]"
                )
                console.print(
                    "[yellow]Make sure you've tested with --dry-run first![/yellow]"
                )
                if not click.confirm("Continue with MOVE operation?"):
                    console.print("[yellow]Cancelled[/yellow]")
                    return

            console.print()

            # Run organization
            result = organizer.organize(
                dry_run=dry_run,
                verify_checksums=not no_verify,
                skip_existing=not overwrite,
            )

            # Display results
            _display_result(result)

            # Show transaction ID for rollback
            if not dry_run and result.transaction_id:
                console.print(f"\n[dim]Transaction ID: {result.transaction_id}[/dim]")
                console.print("[dim]You can rollback this operation with:[/dim]")
                console.print(
                    f"[dim]  vam-organize {catalog_path} {output_directory} "
                    f"--rollback {result.transaction_id}[/dim]"
                )

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _display_result(result: "OrganizationResult") -> None:
    """Display organization result."""
    console.print("\n[green]✓ Organization complete![/green]\n")

    # Create results table
    table = Table(title="Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Total files", str(result.total_files))
    table.add_row("Organized", str(result.organized))
    table.add_row("Skipped", str(result.skipped))
    table.add_row("Failed", str(result.failed))
    table.add_row("No date", str(result.no_date))

    console.print(table)

    if result.dry_run:
        console.print("\n[yellow]This was a DRY RUN - no files were modified[/yellow]")
        console.print("Run without --dry-run to execute the organization.")

    # Show errors if any
    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for error in result.errors[:10]:  # Show first 10
            console.print(f"  [red]• {error}[/red]")
        if len(result.errors) > 10:
            console.print(f"  [dim]... and {len(result.errors) - 10} more[/dim]")


if __name__ == "__main__":
    organize()
