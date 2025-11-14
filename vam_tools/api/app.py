"""FastAPI application factory."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    return app
