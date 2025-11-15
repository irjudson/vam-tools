"""FastAPI application factory."""

import logging

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..db import init_db
from .routers import catalogs, jobs

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="VAM Tools API",
        description="Visual Asset Management - Photo/Video Catalog API",
        version="2.0.0",
    )

    # CORS middleware (allow all origins for local development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize database on startup
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting VAM Tools API...")
        init_db()
        logger.info("Database initialized")

    # Include routers
    app.include_router(catalogs.router, prefix="/api/catalogs", tags=["catalogs"])
    app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # Serve static files and root endpoint
    static_dir = Path(__file__).parent.parent / "web" / "static"
    if static_dir.exists():
        # Serve index.html at root
        from fastapi.responses import FileResponse

        @app.get("/")
        async def root():
            return FileResponse(static_dir / "index.html")

        # Mount static files at /static prefix
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    else:
        # Fallback: redirect to docs if no static files
        @app.get("/")
        async def root():
            return RedirectResponse(url="/docs")

    return app


# Create app instance for uvicorn
app = create_app()
