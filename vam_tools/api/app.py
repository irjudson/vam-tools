"""FastAPI application factory."""

import logging
from pathlib import Path
from typing import Any

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
    async def startup_event() -> None:
        logger.info("Starting VAM Tools API...")
        init_db()
        logger.info("Database initialized")

    # Graceful shutdown
    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        logger.info("Shutting down VAM Tools API...")
        # Give WebSocket connections a moment to close gracefully
        import asyncio

        await asyncio.sleep(0.5)
        logger.info("Shutdown complete")

    # Include routers
    app.include_router(catalogs.router, prefix="/api/catalogs", tags=["catalogs"])
    app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "healthy"}

    # Worker health monitoring endpoint
    @app.get("/api/workers/status")
    async def worker_status() -> dict[str, Any]:
        """Get status of all Celery workers."""
        from datetime import datetime, timedelta

        from ..celery_app import app as celery_app

        try:
            inspect = celery_app.control.inspect(timeout=3.0)

            # Get worker stats
            stats = inspect.stats()
            active_tasks = inspect.active()
            registered = inspect.registered()

            workers_info = []

            if stats:
                for worker_name, worker_stats in stats.items():
                    # Get active tasks for this worker
                    tasks = active_tasks.get(worker_name, []) if active_tasks else []

                    # Check for stuck tasks
                    stuck_tasks = []
                    for task in tasks:
                        time_start = task.get("time_start")
                        if time_start:
                            start_time = datetime.fromtimestamp(time_start)
                            duration = datetime.now() - start_time
                            if duration > timedelta(hours=1):
                                stuck_tasks.append(
                                    {
                                        "id": task.get("id"),
                                        "name": task.get("name"),
                                        "duration_hours": duration.total_seconds()
                                        / 3600,
                                    }
                                )

                    workers_info.append(
                        {
                            "name": worker_name,
                            "healthy": True,
                            "active_tasks": len(tasks),
                            "stuck_tasks": stuck_tasks,
                            "total_tasks_processed": worker_stats.get("total", {}).get(
                                "celery.tasks", 0
                            ),
                            "registered_tasks": (
                                len(registered.get(worker_name, []))
                                if registered
                                else 0
                            ),
                        }
                    )

            return {
                "total_workers": len(workers_info),
                "healthy_workers": sum(1 for w in workers_info if w["healthy"]),
                "workers": workers_info,
            }

        except Exception as e:
            return {
                "error": str(e),
                "total_workers": 0,
                "healthy_workers": 0,
                "workers": [],
            }

    # Serve static files and root endpoint
    static_dir = Path(__file__).parent.parent / "web" / "static"
    if static_dir.exists():
        # Serve index.html at root
        from fastapi.responses import FileResponse

        @app.get("/")
        async def serve_index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

        # Mount static files at /static prefix
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    else:
        # Fallback: redirect to docs if no static files
        @app.get("/")
        async def redirect_to_docs() -> RedirectResponse:
            return RedirectResponse(url="/docs")

    return app


# Create app instance for uvicorn
app = create_app()
