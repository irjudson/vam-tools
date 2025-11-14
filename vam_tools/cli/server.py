"""Start the VAM Tools API server."""

import click
import uvicorn


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def server(host: str, port: int, reload: bool):
    """Start the VAM Tools API server."""
    uvicorn.run(
        "vam_tools.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    server()
