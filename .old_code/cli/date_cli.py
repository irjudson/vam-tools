"""
CLI for image date analysis.

Extracts and reports the earliest dates found in images from multiple sources.
"""

import sys
from pathlib import Path

import click
from rich.table import Table

from vam_tools.shared import collect_image_files

from ..core.date_extraction import DateExtractor
from .base import CLIDisplay, common_options, file_scan_options, init_logging


@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="image_dates.txt",
    help="Output file for results",
)
@file_scan_options
@common_options
@click.option(
    "--sort-by",
    type=click.Choice(["date", "path", "source"], case_sensitive=False),
    default="date",
    help="Sort output by date, path, or source",
)
@init_logging
def cli(
    directory: str,
    output: str,
    recursive: bool,
    no_recursive: bool,
    verbose: bool,
    quiet: bool,
    sort_by: str,
) -> None:
    """
    Analyze images to extract earliest dates.

    Examines EXIF metadata, filenames, directory structure, and file timestamps
    to determine the earliest date associated with each image.

    DIRECTORY: Path to directory containing images to analyze
    """
    # Handle recursive flag
    if no_recursive:
        recursive = False

    directory_path = Path(directory).resolve()
    output_path = Path(output).resolve()

    # Initialize display helper
    display = CLIDisplay(quiet=quiet)

    # Display header and configuration
    display.print_header("Image Date Analyzer")
    display.print_config(
        {
            "Directory": str(directory_path),
            "Recursive": str(recursive),
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

    display.print_info(f"Found [bold]{len(image_files)}[/bold] image files\n")

    # Analyze images
    results = []
    with DateExtractor() as extractor:
        with display.percent_progress("Analyzing images...") as progress:
            task = progress.add_task("Analyzing images...", total=len(image_files))

            for image_path in image_files:
                date_info = extractor.extract_earliest_date(image_path)
                results.append((image_path, date_info))
                progress.advance(task)

    # Sort results
    if sort_by == "date":
        results.sort(
            key=lambda x: (
                x[1].date.timestamp() if x[1] else float("inf"),
                str(x[0]),
            )
        )
    elif sort_by == "path":
        results.sort(key=lambda x: str(x[0]))
    elif sort_by == "source":
        results.sort(key=lambda x: (x[1].source if x[1] else "none", str(x[0])))

    # Write results to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Image Date Analysis Results\n")
        f.write("=" * 80 + "\n\n")

        for image_path, date_info in results:
            if date_info:
                date_str = date_info.date.format("YYYY-MM-DD HH:mm:ss")
                f.write(
                    f"{image_path.parent} - {image_path.name} - "
                    f"{date_str} (from {date_info.source}, "
                    f"confidence: {date_info.confidence}%)\n"
                )
            else:
                f.write(f"{image_path.parent} - {image_path.name} - NO DATE FOUND\n")

    # Display summary
    if not quiet:
        table = Table(title="Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        total = len(results)
        with_dates = sum(1 for _, date_info in results if date_info)
        without_dates = total - with_dates

        # Count by source
        source_counts = {}
        for _, date_info in results:
            if date_info:
                source_counts[date_info.source] = (
                    source_counts.get(date_info.source, 0) + 1
                )

        table.add_row("Total images", str(total))
        table.add_row("With dates", str(with_dates))
        table.add_row("Without dates", str(without_dates))
        table.add_row("", "")

        for source, count in sorted(source_counts.items()):
            table.add_row(f"From {source}", str(count))

        display.console.print(table)
        display.console.print(f"\n[green]Results written to:[/green] {output_path}")


if __name__ == "__main__":
    cli()
