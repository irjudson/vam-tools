"""
Base CLI framework with common patterns.

Provides reusable decorators, display helpers, and progress utilities
to eliminate duplication across CLI commands.
"""

from functools import wraps
from typing import Callable, Dict, Optional

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

from vam_tools.shared import setup_logging


def common_options(f: Callable) -> Callable:
    """Apply standard CLI options (verbose, quiet)."""
    decorators = [
        click.option(
            "-v",
            "--verbose",
            is_flag=True,
            help="Enable verbose logging",
        ),
        click.option(
            "-q",
            "--quiet",
            is_flag=True,
            help="Suppress all output except errors",
        ),
    ]
    for decorator in reversed(decorators):
        f = decorator(f)
    return f


def file_scan_options(f: Callable) -> Callable:
    """Apply file scanning options (recursive)."""
    decorators = [
        click.option(
            "-r",
            "--recursive",
            is_flag=True,
            default=True,
            help="Scan directories recursively",
        ),
        click.option(
            "--no-recursive",
            is_flag=True,
            help="Disable recursive scanning",
        ),
    ]
    for decorator in reversed(decorators):
        f = decorator(f)
    return f


def init_logging(f: Callable) -> Callable:
    """Decorator to setup logging from verbose/quiet flags."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        verbose = kwargs.get("verbose", False)
        quiet = kwargs.get("quiet", False)
        setup_logging(verbose=verbose, quiet=quiet)
        return f(*args, **kwargs)

    return wrapper


class CLIDisplay:
    """Standardized CLI output helpers."""

    def __init__(self, console: Optional[Console] = None, quiet: bool = False):
        """Initialize display helper.

        Args:
            console: Rich console instance (creates new if None)
            quiet: Suppress all output except errors
        """
        self.console = console or Console()
        self.quiet = quiet

    def print_header(self, title: str, subtitle: Optional[str] = None) -> None:
        """Print standardized header.

        Args:
            title: Main title
            subtitle: Optional subtitle
        """
        if self.quiet:
            return

        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")
        if subtitle:
            self.console.print(f"[dim]{subtitle}[/dim]")
        self.console.print()

    def print_config(self, config: Dict[str, str]) -> None:
        """Print configuration table.

        Args:
            config: Dictionary of setting name -> value
        """
        if self.quiet:
            return

        table = Table(show_header=False, box=None)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        for key, value in config.items():
            table.add_row(key, str(value))

        self.console.print(table)
        self.console.print()

    def print_summary(self, metrics: Dict[str, int], title: str = "Summary") -> None:
        """Print summary table.

        Args:
            metrics: Dictionary of metric name -> value
            title: Table title
        """
        if self.quiet:
            return

        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        for key, value in metrics.items():
            if isinstance(value, int):
                table.add_row(key, f"{value:,}")
            else:
                table.add_row(key, str(value))

        self.console.print(table)

    def print_info(self, message: str) -> None:
        """Print info message."""
        if not self.quiet:
            self.console.print(message)

    def print_success(self, message: str) -> None:
        """Print success message."""
        if not self.quiet:
            self.console.print(f"[green]{message}[/green]")

    def print_warning(self, message: str) -> None:
        """Print warning message (shown even in quiet mode)."""
        self.console.print(f"[yellow]{message}[/yellow]")

    def print_error(self, message: str) -> None:
        """Print error message (shown even in quiet mode)."""
        self.console.print(f"[red]{message}[/red]")

    def spinner_progress(self, description: str):
        """Create spinner progress bar (indeterminate).

        Args:
            description: Progress description

        Returns:
            Context manager for progress bar
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            disable=self.quiet,
        )

    def bar_progress(self, description: str):
        """Create bar progress (for determinate tasks).

        Args:
            description: Progress description

        Returns:
            Context manager for progress bar with percentage and ETA
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            disable=self.quiet,
        )

    def percent_progress(self, description: str):
        """Create percentage progress (lighter than full bar).

        Args:
            description: Progress description

        Returns:
            Context manager for progress bar with percentage only
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            disable=self.quiet,
        )
