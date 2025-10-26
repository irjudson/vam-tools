"""
CLI for launching the web-based catalog viewer.
"""

from pathlib import Path

import click
import uvicorn
from rich.console import Console

from vam_tools.web.api import init_catalog

console = Console()


@click.command()
@click.argument("catalog_path", type=click.Path(exists=True))
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
)
@click.option(
    "--port",
    default=8765,
    type=int,
    help="Port to bind to (default: 8765)",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
def web(catalog_path: str, host: str, port: int, reload: bool) -> None:
    """
    Launch web-based catalog viewer.

    CATALOG_PATH: Path to the catalog directory (containing .catalog.json)
    """
    catalog_path = Path(catalog_path)
    catalog_file = catalog_path / ".catalog.json"

    if not catalog_file.exists():
        console.print(f"[red]Error: Catalog not found at {catalog_file}[/red]")
        console.print("\nRun [cyan]vam-analyze[/cyan] first to create a catalog.")
        return

    console.print("\n[bold cyan]VAM Tools - Catalog Viewer[/bold cyan]\n")
    console.print(f"Catalog: {catalog_path}")
    console.print(f"Server: http://{host}:{port}\n")

    # Initialize catalog
    init_catalog(catalog_path)

    console.print("[green]Starting web server...[/green]")
    console.print(f"[yellow]Open http://{host}:{port} in your browser[/yellow]\n")

    # Start server
    uvicorn.run(
        "vam_tools.web.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    web()
