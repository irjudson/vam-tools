"""
Main CLI entry point for Lightroom Tools.

Provides an interactive menu interface for accessing all tools.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .. import __version__
from . import catalog_cli, date_cli, duplicate_cli

console = Console()


def print_banner() -> None:
    """Print the application banner."""
    banner = f"""
[bold cyan]╔═══════════════════════════════════════════════════════════════════════════════╗
║                           LIGHTROOM TOOLS v{__version__:<32}║
║                                                                               ║
║              A collection of tools for managing photo libraries               ║
╚═══════════════════════════════════════════════════════════════════════════════╝[/bold cyan]
"""
    console.print(banner)


def print_menu() -> None:
    """Print the main menu."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style="bold cyan", width=5)
    table.add_column("Tool", style="bold")
    table.add_column("Description")

    table.add_row(
        "1",
        "Analyze Image Dates",
        "Extract earliest dates from EXIF, filenames, and directories",
    )
    table.add_row(
        "2",
        "Find Duplicate Images",
        "Identify duplicate photos across different formats and sizes",
    )
    table.add_row(
        "3",
        "Reorganize Catalog",
        "Reorganize photos into date-based directory structure",
    )
    table.add_row(
        "4",
        "Help & Documentation",
        "Show detailed help for each tool",
    )
    table.add_row("0", "Exit", "Quit the application")

    console.print(Panel(table, title="Main Menu", border_style="cyan"))


def get_directory_input(prompt_text: str = "Enter directory path") -> Path:
    """Get and validate directory input from user."""
    while True:
        path_str = Prompt.ask(prompt_text)

        # Expand home directory
        path = Path(path_str).expanduser().resolve()

        if not path.exists():
            console.print(f"[red]Directory '{path}' does not exist.[/red]")
            if not Confirm.ask("Try again?", default=True):
                sys.exit(0)
            continue

        if not path.is_dir():
            console.print(f"[red]'{path}' is not a directory.[/red]")
            if not Confirm.ask("Try again?", default=True):
                sys.exit(0)
            continue

        return path


def analyze_dates() -> None:
    """Run the image date analyzer."""
    console.print("\n[bold cyan]Image Date Analyzer[/bold cyan]\n")
    console.print("This tool analyzes images to find the earliest date from:")
    console.print("  • EXIF metadata")
    console.print("  • Filename patterns (YYYY-MM-DD, YYYYMMDD, etc.)")
    console.print("  • Directory structure")
    console.print("  • File creation dates\n")

    directory = get_directory_input("Enter directory containing images")
    recursive = Confirm.ask("Scan subdirectories recursively?", default=True)
    output = Prompt.ask("Output file", default="image_dates.txt")

    # Build arguments for the CLI
    args = [str(directory), "-o", output]
    if recursive:
        args.append("-r")

    # Call the date CLI
    ctx = click.Context(date_cli.cli)
    ctx.invoke(date_cli.cli, directory=str(directory), output=output, recursive=recursive, no_recursive=not recursive, verbose=False, quiet=False, sort_by="date")


def find_duplicates() -> None:
    """Run the duplicate image finder."""
    console.print("\n[bold cyan]Duplicate Image Finder[/bold cyan]\n")
    console.print("This tool finds duplicate images that may have:")
    console.print("  • Different file formats (JPG, PNG, TIFF, etc.)")
    console.print("  • Different sizes or resolutions")
    console.print("  • Different filenames")
    console.print("  • Similar visual content using perceptual hashing\n")

    directory = get_directory_input("Enter directory containing images")
    recursive = Confirm.ask("Scan subdirectories recursively?", default=True)

    console.print("\n[cyan]Similarity threshold:[/cyan]")
    console.print("  • 0-5: Very similar images only")
    console.print("  • 6-15: Similar images (recommended)")
    console.print("  • 16-30: Somewhat similar images")
    console.print("  • 31+: Very loose matching")

    threshold = IntPrompt.ask(
        "\nEnter threshold",
        default=5,
        show_default=True,
    )

    if not 0 <= threshold <= 64:
        console.print("[red]Threshold must be between 0 and 64. Using default of 5.[/red]")
        threshold = 5

    output = Prompt.ask("Output file", default="duplicate_images.txt")

    # Call the duplicate CLI
    ctx = click.Context(duplicate_cli.cli)
    ctx.invoke(
        duplicate_cli.cli,
        directory=str(directory),
        output=output,
        recursive=recursive,
        no_recursive=not recursive,
        threshold=threshold,
        verbose=False,
        quiet=False,
    )


def reorganize_catalog() -> None:
    """Run the catalog reorganizer."""
    console.print("\n[bold cyan]Catalog Reorganizer[/bold cyan]\n")
    console.print("[yellow]⚠️  This tool will reorganize your photo files![/yellow]\n")
    console.print("It reorganizes photos into a date-based directory structure.")
    console.print("You can choose to copy or move files, and test with --dry-run first.\n")

    if not Confirm.ask("Do you want to continue?", default=False):
        return

    source = get_directory_input("Enter source directory (photos to reorganize)")
    output = get_directory_input("Enter output directory (where to organize photos)")

    console.print("\n[cyan]Organization strategies:[/cyan]")
    console.print("  1. year/month-day  (e.g., 2023/12-25/)")
    console.print("  2. year/month      (e.g., 2023/12/)")
    console.print("  3. year            (e.g., 2023/)")
    console.print("  4. flat            (e.g., 2023-12-25/)")

    strategy_choice = IntPrompt.ask(
        "\nSelect organization strategy",
        choices=["1", "2", "3", "4"],
        default=1,
    )

    strategies = {
        1: "year/month-day",
        2: "year/month",
        3: "year",
        4: "flat",
    }
    strategy = strategies[strategy_choice]

    copy_mode = Confirm.ask("\nCopy files instead of moving them?", default=False)
    recursive = Confirm.ask("Scan subdirectories recursively?", default=True)
    dry_run = Confirm.ask(
        "\n[yellow]Run in dry-run mode first (recommended)?[/yellow]",
        default=True,
    )

    # Call the catalog CLI
    ctx = click.Context(catalog_cli.cli)
    ctx.invoke(
        catalog_cli.cli,
        directory=str(source),
        output=str(output),
        strategy=strategy,
        conflict="rename",
        copy=copy_mode,
        recursive=recursive,
        no_recursive=not recursive,
        dry_run=dry_run,
        verbose=False,
        quiet=False,
        yes=True,  # We already confirmed above
    )


def show_help() -> None:
    """Show detailed help information."""
    console.print("\n[bold cyan]Help & Documentation[/bold cyan]\n")

    help_sections = [
        (
            "Tool Descriptions",
            [
                ("Image Date Analyzer", [
                    "Extracts the earliest date from multiple sources",
                    "Checks EXIF metadata for camera timestamps",
                    "Parses common date patterns in filenames",
                    "Analyzes directory structure for date information",
                    "Falls back to file creation dates",
                    "Output: Text file with directory, filename, and earliest date",
                ]),
                ("Duplicate Image Finder", [
                    "Uses perceptual hashing to find visually similar images",
                    "Detects duplicates across different file formats",
                    "Finds images with different sizes or resolutions",
                    "Groups exact duplicates and similar images separately",
                    "Output: Text file with grouped duplicate information",
                ]),
                ("Catalog Reorganizer", [
                    "Reorganizes photos into date-based directory structure",
                    "Supports multiple organization strategies",
                    "Can copy or move files",
                    "Dry-run mode for safe testing",
                    "Handles filename conflicts intelligently",
                ]),
            ],
        ),
        (
            "Supported Image Formats",
            ["JPEG/JPG", "PNG", "TIFF/TIF", "BMP", "GIF", "WEBP", "HEIC", "RAW formats (CR2, NEF, ARW, DNG)"],
        ),
        (
            "Tips",
            [
                "Always backup your photos before reorganizing",
                "Use recursive scanning for complete analysis",
                "Lower similarity thresholds find more exact matches",
                "Review duplicate results before deleting any files",
                "EXIF data is the most reliable source for image dates",
                "Use dry-run mode to preview changes before applying them",
            ],
        ),
    ]

    for title, content in help_sections:
        console.print(f"\n[bold]{title}[/bold]\n")

        if isinstance(content, list):
            if isinstance(content[0], tuple):
                # Subsections
                for subtitle, items in content:
                    console.print(f"[cyan]{subtitle}:[/cyan]")
                    for item in items:
                        console.print(f"  • {item}")
                    console.print()
            else:
                # Simple list
                for item in content:
                    console.print(f"  • {item}")


def interactive_mode() -> None:
    """Run the interactive CLI mode."""
    print_banner()

    while True:
        console.print()
        print_menu()
        console.print()

        try:
            choice = Prompt.ask(
                "Select an option",
                choices=["0", "1", "2", "3", "4"],
                default="0",
            )

            if choice == "0":
                console.print("\n[cyan]Thank you for using Lightroom Tools![/cyan]")
                break
            elif choice == "1":
                analyze_dates()
            elif choice == "2":
                find_duplicates()
            elif choice == "3":
                reorganize_catalog()
            elif choice == "4":
                show_help()

            if choice != "0":
                console.print()
                Prompt.ask("Press Enter to continue", default="")

        except KeyboardInterrupt:
            console.print("\n\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"\n[red]Unexpected error: {e}[/red]")
            if Confirm.ask("Continue?", default=True):
                continue
            else:
                break


@click.command()
@click.option(
    "--version",
    is_flag=True,
    help="Show version and exit",
)
def cli(version: bool) -> None:
    """
    Lightroom Tools - Photo library management utilities.

    Launches an interactive menu to access all tools.
    """
    if version:
        console.print(f"Lightroom Tools version {__version__}")
        sys.exit(0)

    interactive_mode()


if __name__ == "__main__":
    cli()
