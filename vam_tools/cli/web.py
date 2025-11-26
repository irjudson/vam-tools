"""
CLI for launching the web-based catalog viewer.
"""

from pathlib import Path

import click
import uvicorn
from rich.console import Console

from vam_tools.version import get_version_string
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

    CATALOG_PATH: Path to the catalog directory
    """
    catalog_dir = Path(catalog_path)

    # Check if catalog exists in PostgreSQL database
    try:
        from vam_tools.db import CatalogDB

        with CatalogDB(catalog_dir) as db:
            # Try to query the catalog to see if it exists
            from vam_tools.db.models import Catalog

            catalog = db.session.query(Catalog).filter_by(id=db.catalog_id).first()

            if not catalog:
                console.print("[red]Error: Catalog not found in database[/red]")
                console.print(
                    "\nRun [cyan]vam-analyze[/cyan] first to create a catalog."
                )
                return
    except Exception as e:
        console.print(f"[red]Error: Could not connect to catalog database: {e}[/red]")
        console.print("\nRun [cyan]vam-analyze[/cyan] first to create a catalog.")
        return

    version_str = get_version_string()
    console.print(
        f"\n[bold cyan]VAM Tools - Catalog Viewer[/bold cyan] [dim]v{version_str}[/dim]\n"
    )
    console.print(f"Catalog: {catalog_dir}")
    console.print(f"Server: http://{host}:{port}\n")

    # Initialize catalog
    init_catalog(catalog_dir)

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
