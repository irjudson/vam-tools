"""
FastAPI backend for catalog review UI.
"""

import io
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from ..analysis.semantic_search import SearchResult, SemanticSearchService
from ..core.types import ImageRecord
from ..db import CatalogDB as CatalogDatabase
from ..shared.preview_cache import PreviewCache
from .catalogs_api import router as catalogs_router
from .jobs_api import router as jobs_router

# Import burst detection task for job submission
try:
    from ..jobs.tasks import detect_bursts_task
except ImportError:
    detect_bursts_task = None

logger = logging.getLogger(__name__)

# Register HEIC support for Pillow
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    logger.debug("HEIC support registered for web viewer")
except ImportError:
    logger.warning(
        "pillow-heif not installed, HEIC files cannot be displayed in web viewer"
    )

app = FastAPI(title="VAM Tools Catalog Viewer", version="2.0.0")

# Configure CORS - restricted to localhost by default for security
# Set VAM_CORS_ORIGINS environment variable to customize (comma-separated)
cors_origins_str = os.getenv(
    "VAM_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
)
cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Restrict to needed methods
    allow_headers=["*"],
)

# Include routers
app.include_router(jobs_router)
app.include_router(catalogs_router)

# Mount static files directory
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global catalog instance
_catalog: Optional[CatalogDatabase] = None
_catalog_path: Optional[Path] = None
_catalog_mtime: Optional[float] = None  # Track last modification time
_preview_cache: Optional[PreviewCache] = None

# Global semantic search service (lazy loaded)
_search_service: Optional[SemanticSearchService] = None


# WebSocket connection manager for real-time updates
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and store a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        self.active_connections.remove(websocket)
        logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)


# Global WebSocket manager
ws_manager = ConnectionManager()


# Pydantic models for API responses
class ImageSummary(BaseModel):
    """Summary of an image for list views."""

    id: str
    source_path: str
    file_type: str
    selected_date: Optional[str]
    date_source: Optional[str]
    confidence: int
    suspicious: bool
    format: Optional[str]
    resolution: Optional[tuple]
    size_bytes: Optional[int]
    thumbnail_path: Optional[str]


class ImageDetail(BaseModel):
    """Full image details."""

    id: str
    source_path: str
    file_type: str
    checksum: str
    status: str
    dates: Dict
    metadata: Dict


class CatalogStats(BaseModel):
    """Catalog statistics."""

    total_images: int
    total_videos: int
    total_size_bytes: int
    no_date: int
    suspicious_dates: int = 0
    problematic_files: int = 0


class CatalogInfo(BaseModel):
    """Overall catalog information."""

    version: str
    catalog_id: str
    created: str
    last_updated: str
    phase: str
    statistics: CatalogStats


class ImageCountResponse(BaseModel):
    """Counts of images by filter type."""

    total: int
    images: int
    videos: int
    no_date: int
    suspicious: int
    problematic: int


class ProblematicFileSummary(BaseModel):
    """Summary of a problematic file."""

    id: str
    source_path: str
    category: str
    error_message: Optional[str]
    detected_at: str
    file_type: Optional[str]
    retries: int
    resolved: bool


def init_catalog(catalog_path: Path) -> None:
    """Initialize the catalog for API access."""
    global _catalog, _catalog_path, _catalog_mtime, _preview_cache
    _catalog_path = catalog_path
    _catalog = CatalogDatabase(catalog_path)
    _catalog.connect()  # Establish connection
    _catalog.initialize()  # Initialize schema

    # Track modification time
    db_file = catalog_path / "catalog.db"
    if db_file.exists():
        _catalog_mtime = db_file.stat().st_mtime

    # Initialize preview cache
    _preview_cache = PreviewCache(catalog_path)
    cache_stats = _preview_cache.get_cache_stats()
    logger.info(
        f"Preview cache initialized: {cache_stats['num_previews']} previews, "
        f"{cache_stats['total_size_gb']:.2f} GB / {cache_stats['max_size_gb']:.2f} GB"
    )

    logger.info(f"Catalog loaded from {catalog_path}")


def get_catalog() -> CatalogDatabase:
    """Get the current catalog instance, reloading if file has changed."""
    global _catalog_mtime

    if _catalog is None or _catalog_path is None:
        raise HTTPException(status_code=500, detail="Catalog not initialized")

    # Check if catalog file has been modified
    db_file = _catalog_path / "catalog.db"
    if db_file.exists():
        current_mtime = db_file.stat().st_mtime
        if _catalog_mtime is None or current_mtime > _catalog_mtime:
            logger.info("Catalog file changed, reconnecting...")
            _catalog.close()
            _catalog.connect()
            _catalog_mtime = current_mtime

    return _catalog


def get_preview_cache() -> PreviewCache:
    """Get the preview cache instance."""
    if _preview_cache is None:
        raise HTTPException(status_code=500, detail="Preview cache not initialized")
    return _preview_cache


def get_search_service() -> SemanticSearchService:
    """Get or create the semantic search service."""
    global _search_service
    if _search_service is None:
        _search_service = SemanticSearchService()
    return _search_service


def get_catalog_db(catalog_id: str) -> CatalogDatabase:
    """Get a CatalogDatabase instance for a specific catalog ID.

    Args:
        catalog_id: Catalog UUID

    Returns:
        CatalogDatabase instance
    """
    return CatalogDatabase(catalog_id=catalog_id)


@app.get("/", response_model=None)
async def root() -> Union[HTMLResponse, Dict[str, Any]]:
    """Serve the frontend UI."""
    static_dir = Path(__file__).parent / "static"
    index_file = static_dir / "index.html"

    if index_file.exists():
        with open(index_file, "r") as f:
            return HTMLResponse(content=f.read())

    return {"message": "VAM Tools Catalog API", "version": "2.0.0"}


@app.get("/api")
async def api_root() -> Dict[str, Any]:
    """API root endpoint."""
    return {"message": "VAM Tools Catalog API", "version": "2.0.0"}


@app.get("/api/catalog/info", response_model=CatalogInfo)
async def get_catalog_info() -> CatalogInfo:
    """Get overall catalog information."""
    catalog = get_catalog()

    # Get state information from config table
    config_rows = catalog.execute(
        "SELECT key, value FROM config WHERE catalog_id = ?", (catalog.catalog_id,)
    ).fetchall()
    config_dict = {row[0]: row[1] for row in config_rows}

    # Default values if not found in config
    version = config_dict.get("version", "2.0.0")
    catalog_id = config_dict.get("catalog_id", "N/A")
    created = config_dict.get("created", datetime.min.isoformat())
    last_updated = config_dict.get("last_updated", datetime.min.isoformat())
    phase = config_dict.get("phase", "unknown")

    # Get statistics from the latest entry in the statistics table
    stats_row = catalog.execute(
        "SELECT * FROM statistics WHERE catalog_id = ? ORDER BY timestamp DESC LIMIT 1",
        (str(catalog.catalog_id),),
    ).fetchone()
    if stats_row:
        stats_dict = (
            dict(stats_row._mapping)
            if hasattr(stats_row, "_mapping")
            else dict(stats_row)
        )
        stats = CatalogStats(
            total_images=stats_dict.get("total_images", 0),
            total_videos=stats_dict.get("total_videos", 0),
            total_size_bytes=stats_dict.get("total_size_bytes", 0),
            no_date=stats_dict.get("no_date", 0),
            suspicious_dates=stats_dict.get("suspicious_dates", 0),
            problematic_files=(stats_dict.get("corrupted_count", 0) or 0)
            + (stats_dict.get("unsupported_count", 0) or 0),  # Placeholder for now
        )
    else:
        stats = CatalogStats(
            total_images=0,
            total_videos=0,
            total_size_bytes=0,
            no_date=0,
            suspicious_dates=0,
            problematic_files=0,
        )

    # Count suspicious dates (this will be done via query later)
    suspicious_count = 0  # Placeholder for now

    return CatalogInfo(
        version=version,
        catalog_id=catalog_id,
        created=created,
        last_updated=last_updated,
        phase=phase,
        statistics=stats,
    )


@app.get("/api/images", response_model=List[ImageSummary])
async def list_images(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    filter_type: Optional[str] = Query(
        None, pattern="^(no_date|suspicious|image|video)$"
    ),
    sort_by: str = Query("date", pattern="^(date|path|size|id)$"),
) -> List[ImageSummary]:
    """
    List images with pagination and filtering using direct SQL queries.
    """
    catalog = get_catalog()

    query = "SELECT id, source_path, file_type, (metadata->>'width')::int as width, (metadata->>'height')::int as height, size_bytes as file_size, (dates->>'taken') as date_taken, quality_score, thumbnail_path, (metadata->>'format') as format FROM images WHERE catalog_id = ?"
    params: List[Union[str, int]] = [str(catalog.catalog_id)]

    # Apply filters
    if filter_type == "no_date":
        query += " AND date_taken IS NULL"
    elif filter_type == "suspicious":
        # This requires more complex logic, for now, we'll skip or use a placeholder
        # In a real scenario, 'suspicious' would be a flag in the DB or derived from other fields
        logger.warning(
            "Filtering by 'suspicious' is not fully implemented with direct SQL yet."
        )
        pass  # For now, no direct SQL filter for suspicious
    elif filter_type == "image":
        query += " AND file_type = 'image'"
    elif filter_type == "video":
        query += " AND file_type = 'video'"

    # Apply sorting
    if sort_by == "date":
        query += " ORDER BY date_taken DESC, id ASC"
    elif sort_by == "path":
        query += " ORDER BY source_path ASC"
    elif sort_by == "size":
        query += " ORDER BY file_size DESC"
    elif sort_by == "id":
        query += " ORDER BY id ASC"

    # Apply pagination
    query += " LIMIT ? OFFSET ?"
    params.append(limit)
    params.append(skip)

    rows = catalog.execute(query, tuple(params)).fetchall()

    summaries = []
    for row in rows:
        # Determine file_type based on format
        file_type = "unknown"
        if row[9] in ["JPEG", "PNG", "GIF", "BMP", "WEBP", "TIFF", "HEIC"]:
            file_type = "image"
        elif row[9] in ["MP4", "MOV", "AVI", "MKV"]:
            file_type = "video"

        summaries.append(
            ImageSummary(
                id=row[0],
                source_path=row[1],
                file_type=file_type,
                selected_date=row[6],
                date_source="db" if row[6] else None,  # Placeholder
                confidence=100 if row[6] else 0,  # Placeholder
                suspicious=False,  # Placeholder
                format=row[9],
                resolution=([row[3], row[4]] if row[3] and row[4] else None),
                size_bytes=row[5],
                thumbnail_path=row[8],
            )
        )

    return summaries


@app.get("/api/images/count", response_model=ImageCountResponse)
async def get_image_counts() -> ImageCountResponse:
    """
    Get counts of images by filter type using direct SQL queries.
    """
    catalog = get_catalog()

    # Total count
    total_count = catalog.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    # Images count
    images_count = catalog.execute(
        "SELECT COUNT(*) FROM images WHERE file_type = 'image'"
    ).fetchone()[0]

    # Videos count
    videos_count = catalog.execute(
        "SELECT COUNT(*) FROM images WHERE file_type = 'video'"
    ).fetchone()[0]

    # No date count
    no_date_count = catalog.execute(
        "SELECT COUNT(*) FROM images WHERE date_taken IS NULL"
    ).fetchone()[0]

    # Suspicious count (placeholder for now, needs proper DB field)
    suspicious_count = 0  # This would need a dedicated column or more complex logic

    # Problematic files count
    problematic_count = catalog.execute(
        "SELECT COUNT(*) FROM problematic_files"
    ).fetchone()[0]

    return ImageCountResponse(
        total=total_count,
        images=images_count,
        videos=videos_count,
        no_date=no_date_count,
        suspicious=suspicious_count,
        problematic=problematic_count,
    )


@app.get("/api/problematic", response_model=List[ProblematicFileSummary])
async def list_problematic_files(
    category: Optional[str] = Query(None),
    resolved: bool = Query(False),
) -> List[ProblematicFileSummary]:
    """
    Get list of problematic files using direct SQL queries.
    """
    catalog = get_catalog()

    query = "SELECT id, file_path, category, error_message, detected_at, resolved_at FROM problematic_files WHERE 1=1"
    params: List[Union[str, bool]] = []

    if category:
        query += " AND category = ?"
        params.append(category)

    if not resolved:
        query += " AND resolved_at IS NULL"

    rows = catalog.execute(query, tuple(params)).fetchall()

    summaries = []
    for row in rows:
        summaries.append(
            ProblematicFileSummary(
                id=str(
                    row[0]
                ),  # id is INTEGER in DB, convert to str for Pydantic model
                source_path=row[1],
                category=row[2],
                error_message=row[3],
                detected_at=row[4],
                file_type="unknown",  # Not stored in problematic_files table directly
                retries=0,  # Not stored in problematic_files table directly
                resolved=bool(row[5]),
            )
        )

    return summaries


from ..core.types import (  # Added imports
    DateInfo,
    FileType,
    ImageMetadata,
    ImageRecord,
    ImageStatus,
)


@app.get("/api/images/{image_id}", response_model=ImageDetail)
async def get_image_detail(image_id: str) -> ImageDetail:
    """Get detailed information about a specific image."""
    catalog = get_catalog()

    # Use catalog.get_image() which returns an ImageRecord
    image = catalog.get_image(image_id)

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Build dates dict
    dates_dict = {}
    if image.dates:
        dates_dict = {
            "selected_date": (
                image.dates.selected_date.isoformat()
                if image.dates.selected_date
                else None
            ),
            "selected_source": image.dates.selected_source,
            "confidence": image.dates.confidence,
            "suspicious": image.dates.suspicious,
        }
    else:
        dates_dict = {
            "selected_date": None,
            "selected_source": None,
            "confidence": 0,
            "suspicious": False,
        }

    # Build metadata dict
    metadata_dict = {}
    if image.metadata:
        metadata_dict = {
            "format": image.metadata.format,
            "resolution": image.metadata.resolution,
            "size_bytes": image.metadata.size_bytes,
            "exif": image.metadata.exif or {},
            "gps_latitude": image.metadata.gps_latitude,
            "gps_longitude": image.metadata.gps_longitude,
        }
    else:
        metadata_dict = {
            "format": None,
            "resolution": None,
            "size_bytes": None,
            "exif": {},
            "gps": None,
        }

    return ImageDetail(
        id=image.id,
        source_path=str(image.source_path),
        file_type=image.file_type.value,
        checksum=image.checksum,
        status=image.status.value,
        dates=dates_dict,
        metadata=metadata_dict,
    )


@app.get("/api/images/{image_id}/file", response_model=None)
async def get_image_file(image_id: str) -> Union[FileResponse, StreamingResponse]:
    """Serve the actual image file for preview."""
    catalog = get_catalog()
    preview_cache = get_preview_cache()

    # Use catalog.get_image() which returns an ImageRecord
    image = catalog.get_image(image_id)

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = Path(image.source_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    # Check if file needs conversion for browser display
    file_ext = file_path.suffix.lower()

    # RAW formats need special handling (extract embedded preview)
    raw_formats = [
        ".arw",
        ".cr2",
        ".cr3",
        ".nef",
        ".dng",
        ".orf",
        ".rw2",
        ".pef",
        ".sr2",
        ".raf",
        ".raw",
    ]

    # HEIC/TIFF can be converted with Pillow
    pillow_convertible = [".heic", ".heif", ".tif", ".tiff"]

    # Check if this file needs extraction/conversion (RAW or HEIC/TIFF)
    needs_extraction = file_ext in raw_formats or file_ext in pillow_convertible

    # If file needs extraction, check cache first
    if needs_extraction:
        cached_preview_path = preview_cache.get_preview_path(image_id)
        if cached_preview_path and cached_preview_path.exists():
            # Serve cached preview (fast!)
            logger.debug(f"Serving cached preview for {image_id}")
            return FileResponse(
                cached_preview_path,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": f"inline; filename={file_path.stem}_preview.jpg",
                    "Cache-Control": "public, max-age=3600",
                },
            )

    # Handle RAW files - extract embedded JPEG preview using ExifTool
    if file_ext in raw_formats:
        try:
            # Use ExifTool to extract preview image
            # Most RAW files have embedded JPEG previews
            # Increase timeout to 30s for large/slow files on network drives
            result = subprocess.run(
                ["exiftool", "-b", "-PreviewImage", str(file_path)],
                capture_output=True,
                timeout=30,
            )

            if result.returncode == 0 and len(result.stdout) > 0:
                # Successfully extracted preview - store in cache
                preview_cache.store_preview(image_id, result.stdout)
                logger.debug(f"Extracted and cached RAW preview for {image_id}")

                buffer = io.BytesIO(result.stdout)
                return StreamingResponse(
                    buffer,
                    media_type="image/jpeg",
                    headers={
                        "Content-Disposition": f"inline; filename={file_path.stem}_preview.jpg",
                        "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                    },
                )
            else:
                # No preview found, try different preview tags
                result = subprocess.run(
                    ["exiftool", "-b", "-JpgFromRaw", str(file_path)],
                    capture_output=True,
                    timeout=30,
                )

                if result.returncode == 0 and len(result.stdout) > 0:
                    # Successfully extracted preview - store in cache
                    preview_cache.store_preview(image_id, result.stdout)
                    logger.debug(
                        f"Extracted and cached RAW preview (JpgFromRaw) for {image_id}"
                    )

                    buffer = io.BytesIO(result.stdout)
                    return StreamingResponse(
                        buffer,
                        media_type="image/jpeg",
                        headers={
                            "Content-Disposition": f"inline; filename={file_path.stem}_preview.jpg",
                            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                        },
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"No embedded preview found in {file_ext.upper()} file",
                    )
        except subprocess.TimeoutExpired:
            # Warn instead of error - some network drives or large files may timeout
            logger.warning(
                f"Timeout extracting preview from {file_path} (file on slow storage or very large)"
            )
            raise HTTPException(
                status_code=504,  # Gateway Timeout instead of 500
                detail=f"Timeout extracting preview from {file_ext.upper()} file (file on slow storage)",
            )
        except FileNotFoundError:
            logger.error("ExifTool not found - required for RAW file previews")
            raise HTTPException(
                status_code=500,
                detail="ExifTool not installed (required for RAW previews)",
            )
        except Exception as e:
            logger.error(f"Error extracting preview from {file_path} ({file_ext}): {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error extracting preview from {file_ext.upper()}: {e}",
            )

    # Handle HEIC/TIFF with Pillow conversion
    elif file_ext in pillow_convertible:
        try:
            # Convert to JPEG on-the-fly for browser display
            with Image.open(file_path) as img:
                # Convert to RGB if needed (HEIC can have different color modes)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                # Save to bytes buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                buffer.seek(0)

                # Store in cache
                preview_bytes = buffer.getvalue()
                preview_cache.store_preview(image_id, preview_bytes)
                logger.debug(
                    f"Converted and cached {file_ext.upper()} preview for {image_id}"
                )

                # Reset buffer for streaming
                buffer.seek(0)

                return StreamingResponse(
                    buffer,
                    media_type="image/jpeg",
                    headers={
                        "Content-Disposition": f"inline; filename={file_path.stem}.jpg",
                        "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                    },
                )
        except Exception as e:
            logger.error(f"Error converting image {file_path} ({file_ext}): {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error converting {file_ext.upper()} image: {e}",
            )

    # For browser-native formats (JPEG, PNG, GIF, WebP), serve directly
    return FileResponse(
        file_path,
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
        },
    )


@app.get("/api/images/{image_id}/thumbnail", response_model=None)
async def get_image_thumbnail(image_id: str) -> Union[FileResponse, StreamingResponse]:
    """
    Serve thumbnail for an image.
    """
    catalog = get_catalog()

    row = catalog.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    # Manually construct ImageRecord from row (reusing logic from get_image_detail)
    image = ImageRecord(
        id=row["id"],
        source_path=Path(row["source_path"]),
        file_type=(
            FileType.IMAGE
            if row["format"] in ["JPEG", "PNG", "GIF", "BMP", "WEBP", "TIFF", "HEIC"]
            else FileType.VIDEO
        ),  # Infer from format
        checksum=row["file_hash"],
        status=ImageStatus.COMPLETE,  # Placeholder, needs proper status tracking
        file_size=row["file_size"],
        created_at=(
            datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
        ),
        modified_at=(
            datetime.fromisoformat(row["modified_at"]) if row["modified_at"] else None
        ),
        dates=DateInfo(
            selected_date=(
                datetime.fromisoformat(row["date_taken"]) if row["date_taken"] else None
            ),
            selected_source="db",  # Placeholder
            confidence=100 if row["date_taken"] else 0,  # Placeholder
            suspicious=False,  # Placeholder
        ),
        metadata=ImageMetadata(
            format=row["format"],
            resolution=(
                [row["width"], row["height"]]
                if row["width"] and row["height"]
                else None
            ),
            size_bytes=row["file_size"],
            exif={
                "Make": row["camera_make"],
                "Model": row["camera_model"],
                "LensModel": row["lens_model"],
                "FocalLength": row["focal_length"],
                "ApertureValue": row["aperture"],
                "ExposureTime": row["shutter_speed"],
                "ISO": row["iso"],
            },
            gps=(
                GPSInfo(
                    latitude=row["gps_latitude"],
                    longitude=row["gps_longitude"],
                )
                if row["gps_latitude"] and row["gps_longitude"]
                else None
            ),
        ),
        quality_score=row["quality_score"],
        is_corrupted=bool(row["is_corrupted"]),
        thumbnail_path=Path(
            f"thumbnails/{row['id']}.jpg"
        ),  # Placeholder: assuming thumbnails are stored in a predictable path
    )

    # Try to serve thumbnail if it exists
    if image.thumbnail_path and _catalog_path is not None:
        thumbnail_path = _catalog_path / image.thumbnail_path
        if thumbnail_path.exists():
            return FileResponse(
                thumbnail_path,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
                },
            )

    # Fallback: thumbnail doesn't exist, redirect to full image endpoint
    # (This handles edge cases where thumbnails weren't generated)
    # For now, we'll just serve the full image directly if thumbnail not found
    file_path = Path(image.source_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return FileResponse(
        file_path,
        headers={
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/statistics/summary")
async def get_statistics_summary() -> Dict[str, Any]:
    """Get various statistics about the catalog using direct SQL queries."""
    catalog = get_catalog()

    # Get latest statistics snapshot
    stats_row = catalog.execute(
        "SELECT * FROM statistics ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()

    if not stats_row:
        return {
            "total": {"images": 0, "videos": 0, "size_bytes": 0},
            "issues": {"no_date": 0, "suspicious_dates": 0},
            "by_format": {},
            "by_extension": {},
            "by_category": {},
            "by_size_bucket": {},
            "by_issue_type": {},
            "by_date_source": {},
            "by_year": {},
        }

    # Convert row to dict for easier access
    stats_dict = (
        dict(stats_row._mapping)
        if hasattr(stats_row, "_mapping")
        else dict(zip(stats_row.keys(), stats_row))
    )

    # Basic catalog stats from the latest snapshot
    total_images = stats_dict.get("total_images", 0)
    total_videos = stats_dict.get("total_videos", 0)
    total_size_bytes = stats_dict.get("total_size_bytes", 0)
    no_date = stats_dict.get("no_date", 0)
    suspicious_dates = stats_dict.get("suspicious_dates", 0)

    # Calculate additional statistics using SQL
    # By format
    by_format_rows = catalog.execute(
        "SELECT file_type, COUNT(*) as count FROM images GROUP BY file_type"
    ).fetchall()
    by_format = {}
    for row in by_format_rows:
        row_dict = (
            dict(row._mapping)
            if hasattr(row, "_mapping")
            else dict(zip(row.keys(), row))
        )
        by_format[row_dict.get("file_type", "unknown")] = row_dict.get("count", 0)

    # By extension
    # This is tricky with SQL directly, as extension is part of source_path.
    # For now, we'll approximate or skip. A more robust solution would involve
    # storing extension in the DB or using a view.
    by_extension = {}  # Placeholder

    # By category (image/video)
    by_category = {"image": total_images, "video": total_videos}

    # By size bucket (approximate with SQL, or fetch all and process in Python)
    # For simplicity, we'll do a basic approximation here.
    by_size_bucket = {}
    size_buckets = [
        (0, 100_000, "< 100 KB"),
        (100_000, 1_000_000, "100 KB - 1 MB"),
        (1_000_000, 5_000_000, "1 MB - 5 MB"),
        (5_000_000, 10_000_000, "5 MB - 10 MB"),
        (10_000_000, 50_000_000, "10 MB - 50 MB"),
        (50_000_000, float("inf"), "> 50 MB"),
    ]
    for bucket_min, bucket_max, bucket_label in size_buckets:
        if bucket_max == float("inf"):
            count = catalog.execute(
                "SELECT COUNT(*) FROM images WHERE size_bytes >= ?", (bucket_min,)
            ).fetchone()[0]
        else:
            count = catalog.execute(
                "SELECT COUNT(*) FROM images WHERE size_bytes >= ? AND size_bytes < ?",
                (bucket_min, bucket_max),
            ).fetchone()[0]
        by_size_bucket[bucket_label] = count

    # By date source (not directly in DB, needs to be inferred or stored)
    by_date_source = {"db": total_images + total_videos}  # Placeholder

    # By issue type
    by_issue_type = {
        "no_date": no_date,
        "suspicious_date": suspicious_dates,
        "corrupted": stats_dict.get("corrupted_count", 0),
        "unsupported": stats_dict.get("unsupported_count", 0),
    }

    # By year - extract year from JSONB dates->>'selected_date'
    by_year_rows = catalog.execute(
        """SELECT EXTRACT(YEAR FROM (dates->>'selected_date')::timestamp) as year, COUNT(*) as count
           FROM images
           WHERE dates->>'selected_date' IS NOT NULL
           GROUP BY year
           ORDER BY year"""
    ).fetchall()
    by_year = {}
    for row in by_year_rows:
        row_dict = (
            dict(row._mapping)
            if hasattr(row, "_mapping")
            else dict(zip(row.keys(), row))
        )
        year = row_dict.get("year")
        if year:
            by_year[int(year)] = row_dict.get("count", 0)

    return {
        "total": {
            "images": total_images,
            "videos": total_videos,
            "size_bytes": total_size_bytes,
        },
        "issues": {
            "no_date": no_date,
            "suspicious_dates": suspicious_dates,
        },
        "by_format": by_format,
        "by_extension": by_extension,
        "by_category": by_category,
        "by_size_bucket": by_size_bucket,
        "by_issue_type": by_issue_type,
        "by_date_source": by_date_source,
        "by_year": dict(sorted(by_year.items())),
    }


# Pydantic models for duplicate groups
class DuplicateGroupSummary(BaseModel):
    """Summary of a duplicate group."""

    id: str
    primary_image_id: str
    duplicate_count: int
    total_size_bytes: int
    format_types: List[str]
    needs_review: bool


class SimilarityMetricsResponse(BaseModel):
    """Similarity metrics for a pair of images."""

    image1_id: str
    image2_id: str
    dhash_distance: Optional[int] = None
    ahash_distance: Optional[int] = None
    whash_distance: Optional[int] = None
    dhash_similarity: Optional[float] = None
    ahash_similarity: Optional[float] = None
    whash_similarity: Optional[float] = None
    overall_similarity: float = 0.0


class DuplicateGroupDetail(BaseModel):
    """Detailed duplicate group information."""

    id: str
    primary_image_id: str
    duplicate_image_ids: List[str]
    similarity_score: float
    # Pairwise similarity metrics for all images in group
    similarity_metrics: List[SimilarityMetricsResponse] = []
    needs_review: bool
    review_reason: Optional[str] = None


@app.get("/api/dashboard/stats")
async def get_dashboard_stats() -> Dict[str, Any]:
    """
    Get comprehensive statistics for dashboard display using direct SQL queries.
    """
    catalog = get_catalog()

    # Get latest statistics snapshot
    stats_row = catalog.execute(
        "SELECT * FROM statistics ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()

    if not stats_row:
        # Return default empty stats if no data
        return {
            "catalog": {
                "total_files": 0,
                "total_images": 0,
                "total_videos": 0,
                "total_size_bytes": 0,
                "total_size_gb": 0.0,
            },
            "duplicates": {
                "total_groups": 0,
                "needs_review": 0,
                "total_duplicate_images": 0,
                "potential_space_savings_bytes": 0,
                "potential_space_savings_gb": 0.0,
            },
            "review": {
                "total_needing_review": 0,
                "date_conflicts": 0,
                "no_date": 0,
                "suspicious_dates": 0,
                "low_confidence": 0,
            },
            "hashes": {
                "images_with_dhash": 0,
                "images_with_ahash": 0,
                "images_with_whash": 0,
                "images_with_any_hash": 0,
                "coverage_percent": 0.0,
                "dhash_percent": 0.0,
                "ahash_percent": 0.0,
                "whash_percent": 0.0,
            },
        }

    # Convert row to dict for easier access
    stats_dict = (
        dict(stats_row._mapping)
        if hasattr(stats_row, "_mapping")
        else dict(zip(stats_row.keys(), stats_row))
    )

    # Basic catalog stats from the latest snapshot
    total_images = stats_dict.get("total_images", 0)
    total_videos = stats_dict.get("total_videos", 0)
    total_files = total_images + total_videos
    total_size_bytes = stats_dict.get("total_size_bytes", 0)

    # Duplicate stats
    total_groups = catalog.execute("SELECT COUNT(*) FROM duplicate_groups").fetchone()[
        0
    ]
    needs_review_groups = catalog.execute(
        "SELECT COUNT(*) FROM duplicate_groups WHERE reviewed = 0"
    ).fetchone()[0]
    total_duplicate_images = catalog.execute(
        "SELECT COUNT(*) FROM duplicate_group_images"
    ).fetchone()[0]

    # Potential space savings (sum of all but the primary image's size in each group)
    potential_space_savings_bytes = (
        catalog.execute(
            """
        SELECT SUM(i.size_bytes)
        FROM duplicate_group_images dgi
        JOIN images i ON dgi.image_id = i.id
        WHERE dgi.is_primary = 0
        """
        ).fetchone()[0]
        or 0
    )

    # Review queue stats
    no_date = stats_dict.get("no_date", 0)
    suspicious_dates = stats_dict.get("suspicious_dates", 0)
    # date_conflicts and low_confidence are not directly in statistics table,
    # would need to query review_queue or images table with more complex logic
    date_conflicts = 0  # Placeholder
    low_confidence = 0  # Placeholder
    total_needing_review = no_date + suspicious_dates + date_conflicts + low_confidence

    # Hash performance
    images_with_dhash = catalog.execute(
        "SELECT COUNT(*) FROM images WHERE perceptual_hash IS NOT NULL"  # Assuming perceptual_hash is dhash for now
    ).fetchone()[0]
    images_with_ahash = 0  # Not directly in schema
    images_with_whash = 0  # Not directly in schema
    images_with_any_hash = images_with_dhash  # Assuming dhash is the only one for now

    hash_coverage_percent = (
        (images_with_any_hash / total_files * 100) if total_files > 0 else 0
    )
    dhash_percent = (images_with_dhash / total_files * 100) if total_files > 0 else 0
    ahash_percent = 0.0
    whash_percent = 0.0

    return {
        "catalog": {
            "total_files": total_files,
            "total_images": total_images,
            "total_videos": total_videos,
            "total_size_bytes": total_size_bytes,
            "total_size_gb": round(total_size_bytes / (1024**3), 2),
        },
        "duplicates": {
            "total_groups": total_groups,
            "needs_review": needs_review_groups,
            "total_duplicate_images": total_duplicate_images,
            "potential_space_savings_bytes": potential_space_savings_bytes,
            "potential_space_savings_gb": round(
                potential_space_savings_bytes / (1024**3), 2
            ),
        },
        "review": {
            "total_needing_review": total_needing_review,
            "date_conflicts": date_conflicts,
            "no_date": no_date,
            "suspicious_dates": suspicious_dates,
            "low_confidence": low_confidence,
        },
        "hashes": {
            "images_with_dhash": images_with_dhash,
            "images_with_ahash": images_with_ahash,
            "images_with_whash": images_with_whash,
            "images_with_any_hash": images_with_any_hash,
            "coverage_percent": round(hash_coverage_percent, 1),
            "dhash_percent": round(dhash_percent, 1),
            "ahash_percent": round(ahash_percent, 1),
            "whash_percent": round(whash_percent, 1),
        },
    }


async def list_duplicate_groups(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    needs_review_only: bool = Query(False),
) -> List[DuplicateGroupSummary]:
    """
    List duplicate groups with pagination using direct SQL queries.
    """
    catalog = get_catalog()

    query = """
        SELECT
            dg.id,
            dgi_primary.image_id AS primary_image_id,
            COUNT(dgi.image_id) AS duplicate_count,
            SUM(i.size_bytes) AS total_size_bytes,
            STRING_AGG(DISTINCT i.file_type, ',') AS format_types,
            dg.reviewed AS needs_review
        FROM duplicate_groups dg
        JOIN duplicate_group_images dgi ON dg.id = dgi.group_id
        JOIN images i ON dgi.image_id = i.id
        LEFT JOIN duplicate_group_images dgi_primary ON dg.id = dgi_primary.group_id AND dgi_primary.is_primary = 1
        WHERE 1=1
    """
    params: List[Union[str, int, bool]] = []

    if needs_review_only:
        query += " AND dg.reviewed = 0"  # Assuming 'reviewed' column indicates if it needs review

    query += " GROUP BY dg.id, dgi_primary.image_id"
    query += " ORDER BY dg.created_at DESC"  # Or some other relevant sorting

    # Apply pagination
    query += " LIMIT ? OFFSET ?"
    params.append(limit)
    params.append(skip)

    rows = catalog.execute(query, tuple(params)).fetchall()

    summaries = []
    for row in rows:
        summaries.append(
            DuplicateGroupSummary(
                id=str(row["id"]),
                primary_image_id=row["primary_image_id"] or "",
                duplicate_count=row["duplicate_count"],
                total_size_bytes=row["total_size_bytes"] or 0,
                format_types=(
                    row["format_types"].split(",") if row["format_types"] else []
                ),
                needs_review=not bool(
                    row["needs_review"]
                ),  # Assuming 0 means needs review
            )
        )

    return summaries


@app.get("/api/duplicates/groups/{group_id}", response_model=DuplicateGroupDetail)
async def get_duplicate_group(group_id: str) -> DuplicateGroupDetail:
    """Get detailed information about a specific duplicate group."""
    catalog = get_catalog()

    group_row = catalog.execute(
        "SELECT id, similarity_score, reviewed FROM duplicate_groups WHERE id = ?",
        (group_id,),
    ).fetchone()

    if not group_row:
        raise HTTPException(status_code=404, detail="Duplicate group not found")

    # Get images in this group
    image_rows = catalog.execute(
        "SELECT image_id, is_primary FROM duplicate_group_images WHERE group_id = ?",
        (group_id,),
    ).fetchall()

    duplicate_image_ids = [row["image_id"] for row in image_rows]
    primary_image_id = next(
        (row["image_id"] for row in image_rows if row["is_primary"]), ""
    )

    # Similarity metrics are not yet migrated to the database
    similarity_responses: List[SimilarityMetricsResponse] = []
    avg_similarity = (
        group_row["similarity_score"] if group_row["similarity_score"] else 0.0
    )

    return DuplicateGroupDetail(
        id=str(group_row["id"]),
        primary_image_id=primary_image_id,
        duplicate_image_ids=duplicate_image_ids,
        similarity_score=avg_similarity,
        similarity_metrics=similarity_responses,  # Placeholder
        needs_review=not bool(group_row["reviewed"]),
        review_reason="Needs review" if not bool(group_row["reviewed"]) else None,
    )


@app.get("/api/duplicates/stats")
async def get_duplicate_stats() -> Dict[str, Any]:
    """Get statistics about duplicate groups using direct SQL queries."""
    catalog = get_catalog()

    total_groups = catalog.execute("SELECT COUNT(*) FROM duplicate_groups").fetchone()[
        0
    ]
    needs_review = catalog.execute(
        "SELECT COUNT(*) FROM duplicate_groups WHERE reviewed = 0"
    ).fetchone()[0]
    total_duplicates = catalog.execute(
        "SELECT COUNT(*) FROM duplicate_group_images"
    ).fetchone()[0]
    total_unique = total_groups  # One primary per group

    # Calculate potential space savings
    total_duplicate_size = (
        catalog.execute(
            """
        SELECT SUM(i.size_bytes)
        FROM duplicate_group_images dgi
        JOIN images i ON dgi.image_id = i.id
        WHERE dgi.is_primary = 0
        """
        ).fetchone()[0]
        or 0
    )

    return {
        "total_groups": total_groups,
        "needs_review": needs_review,
        "total_duplicates": total_duplicates,
        "total_unique": total_unique,
        "potential_space_savings_bytes": total_duplicate_size,
    }


# ============================================================================
# Review Queue Endpoints
# ============================================================================


@app.get("/api/review/queue")
async def get_review_queue(
    filter_type: Optional[str] = Query(
        None,
        description="Filter by type: date_conflict, no_date, suspicious_date",
    )
) -> Dict[str, Any]:
    """
    Get review queue items using direct SQL queries.
    """
    catalog = get_catalog()

    review_items = []

    # Base query for review queue items
    base_query = """
        SELECT
            rq.id,
            rq.image_id,
            rq.reason,
            rq.priority,
            rq.created_at,
            i.source_path,
            i.date_taken AS current_date,
            i.quality_score AS confidence
        FROM review_queue rq
        JOIN images i ON rq.image_id = i.id
        WHERE rq.reviewed_at IS NULL
    """
    params: List[str] = []

    if filter_type:
        base_query += " AND rq.reason = ?"
        params.append(filter_type)

    rows = catalog.execute(base_query, tuple(params)).fetchall()

    for row in rows:
        item_type = row["reason"]
        if item_type == "date_conflict":
            # For date conflicts, we might need to fetch group_id if it's relevant
            # For now, we'll just use the basic info
            review_items.append(
                {
                    "id": row["image_id"],
                    "type": item_type,
                    "source_path": row["source_path"],
                    "current_date": row["current_date"],
                    "confidence": row["confidence"],
                    "group_id": None,  # Placeholder for now
                }
            )
        elif item_type == "no_date":
            review_items.append(
                {
                    "id": row["image_id"],
                    "type": item_type,
                    "source_path": row["source_path"],
                    "current_date": None,
                    "confidence": 0,
                    "filesystem_created": None,  # Not directly in review_queue
                }
            )
        elif item_type == "suspicious_date":
            review_items.append(
                {
                    "id": row["image_id"],
                    "type": item_type,
                    "source_path": row["source_path"],
                    "current_date": row["current_date"],
                    "confidence": row["confidence"],
                    "selected_source": "db",  # Placeholder
                }
            )
        else:
            # Generic review item
            review_items.append(
                {
                    "id": row["image_id"],
                    "type": item_type,
                    "source_path": row["source_path"],
                    "current_date": row["current_date"],
                    "confidence": row["confidence"],
                }
            )

    return {"items": review_items, "total": len(review_items)}


@app.patch("/api/images/{image_id}/date")
async def update_image_date(
    image_id: str, date_str: str = Query(...)
) -> Dict[str, Any]:
    """
    Update the date for an image using direct SQL.
    """
    catalog = get_catalog()

    # Check if image exists
    existing_image = catalog.execute(
        "SELECT id FROM images WHERE id = ?", (image_id,)
    ).fetchone()
    if not existing_image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Parse date
    try:
        from datetime import datetime

        # Validate date format
        datetime.fromisoformat(date_str.replace("Z", "+00:00"))  # Allow Z for UTC
        new_date_iso = date_str
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SSZ)",
        )

    # Update image date in the database
    catalog.execute(
        "UPDATE images SET date_taken = ?, modified_at = datetime('now') WHERE id = ?",
        (new_date_iso, image_id),
    )

    return {"success": True, "image_id": image_id, "new_date": date_str}


@app.get("/api/review/stats")
async def get_review_stats() -> Dict[str, Any]:
    """Get statistics about items needing review using direct SQL queries."""
    catalog = get_catalog()

    stats = {
        "date_conflicts": 0,
        "no_date": 0,
        "suspicious_dates": 0,
        "low_confidence": 0,
        "ready_to_organize": 0,
        "total_needing_review": 0,
    }

    # Count date conflicts (from review_queue where reason is 'date_conflict')
    stats["date_conflicts"] = catalog.execute(
        "SELECT COUNT(*) FROM review_queue WHERE reason = 'date_conflict' AND reviewed_at IS NULL"
    ).fetchone()[0]

    # Count no date (from review_queue where reason is 'no_date')
    stats["no_date"] = catalog.execute(
        "SELECT COUNT(*) FROM review_queue WHERE reason = 'no_date' AND reviewed_at IS NULL"
    ).fetchone()[0]

    # Count suspicious dates (from review_queue where reason is 'suspicious_date')
    stats["suspicious_dates"] = catalog.execute(
        "SELECT COUNT(*) FROM review_queue WHERE reason = 'suspicious_date' AND reviewed_at IS NULL"
    ).fetchone()[0]

    # Count low confidence (this would require a dedicated field in images or review_queue)
    # For now, we'll leave it at 0 or derive from a more complex query if schema allows
    stats["low_confidence"] = 0  # Placeholder

    # Total needing review
    stats["total_needing_review"] = (
        stats["date_conflicts"] + stats["no_date"] + stats["suspicious_dates"]
    )

    # Ready to organize (images not in review queue and have a date)
    # This is a more complex query, for now, we'll approximate or leave as 0
    stats["ready_to_organize"] = 0  # Placeholder

    return stats


# ============================================================================
# Performance Statistics Endpoints
# ============================================================================


class PerformanceStatsResponse(BaseModel):
    """Response model for performance statistics."""

    run_id: str
    started_at: Optional[str]
    completed_at: Optional[str]
    total_duration_seconds: float
    total_files_analyzed: int
    files_per_second: float
    bytes_processed: int
    bytes_per_second: float
    peak_memory_mb: float
    gpu_utilized: bool
    gpu_device: Optional[str]
    total_errors: int


class OperationStatsResponse(BaseModel):
    """Response model for operation statistics."""

    operation_name: str
    total_time_seconds: float
    call_count: int
    items_processed: int
    errors: int
    average_time_per_item: float
    min_time_seconds: Optional[float]
    max_time_seconds: Optional[float]


class HashingStatsResponse(BaseModel):
    """Response model for hashing statistics."""

    dhash_time_seconds: float
    ahash_time_seconds: float
    whash_time_seconds: float
    total_hashes_computed: int
    gpu_hashes: int
    cpu_hashes: int
    failed_hashes: int
    raw_conversions: int
    raw_conversion_time_seconds: float


class PerformanceDetailResponse(BaseModel):
    """Detailed performance statistics response."""

    metrics: PerformanceStatsResponse
    operations: List[OperationStatsResponse]
    hashing: HashingStatsResponse
    bottlenecks: List[str]
    slowest_operations: List[Dict[str, Any]]


@app.get("/api/performance/history")
async def get_performance_history() -> Dict[str, Any]:
    """
    Get historical performance statistics from the database.
    """
    catalog = get_catalog()

    # Fetch all performance snapshots filtered by catalog_id
    try:
        rows = catalog.execute(
            "SELECT * FROM performance_snapshots WHERE catalog_id = ? ORDER BY timestamp DESC",
            (str(catalog.catalog_id),),
        ).fetchall()
    except Exception:
        # Table doesn't exist - return no_data
        return {
            "status": "no_data",
            "history": [],
            "total_runs": 0,
            "average_throughput": 0,
        }

    if not rows:
        return {
            "status": "no_data",
            "history": [],
            "total_runs": 0,
            "average_throughput": 0,
        }

    history = []
    total_runs = len(rows)
    total_files_analyzed = 0
    total_time_seconds = 0.0

    for row in rows:
        row_dict = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
        history.append(
            {
                "timestamp": row_dict["timestamp"],
                "phase": row_dict["phase"],
                "files_processed": row_dict["files_processed"],
                "files_total": row_dict["files_total"],
                "elapsed_seconds": row_dict["elapsed_seconds"],
                "rate_files_per_sec": row_dict["rate_files_per_sec"],
                "cpu_percent": row_dict["cpu_percent"],
                "memory_mb": row_dict["memory_mb"],
                "gpu_utilization": row_dict["gpu_utilization"],
                "gpu_memory_mb": row_dict["gpu_memory_mb"],
            }
        )
        total_files_analyzed += row_dict["files_processed"] or 0
        total_time_seconds += row_dict["elapsed_seconds"] or 0

    average_throughput = (
        (total_files_analyzed / total_time_seconds) if total_time_seconds > 0 else 0
    )

    return {
        "status": "ok",
        "history": history,
        "total_runs": total_runs,
        "total_files_analyzed": total_files_analyzed,
        "total_time_seconds": total_time_seconds,
        "average_throughput": average_throughput,
    }


@app.get("/api/performance/summary")
async def get_performance_summary() -> Dict[str, Any]:
    """
    Get a summary report of current performance statistics from the database.
    """
    catalog = get_catalog()

    # Fetch the latest performance snapshot filtered by catalog_id
    try:
        last_run_row = catalog.execute(
            "SELECT * FROM performance_snapshots WHERE catalog_id = ? ORDER BY timestamp DESC LIMIT 1",
            (str(catalog.catalog_id),),
        ).fetchone()
    except Exception:
        # Table doesn't exist - return no_data
        return {
            "status": "no_data",
            "summary": "No performance statistics available",
        }

    if not last_run_row:
        return {
            "status": "no_data",
            "summary": "No performance statistics available",
        }

    last_run = (
        dict(last_run_row._mapping)
        if hasattr(last_run_row, "_mapping")
        else dict(last_run_row)
    )

    # Build summary similar to PerformanceMetrics.get_summary_report()
    lines = [
        "=== Performance Analysis Summary ===",
        f"Total Duration: {last_run.get('elapsed_seconds', 0):.2f}s",
        f"Files Analyzed: {last_run.get('files_processed', 0)}",
        f"Throughput: {last_run.get('rate_files_per_sec', 0):.2f} files/sec",
        f"Data Processed: {last_run.get('bytes_processed', 0) / (1024**3):.2f} GB",
        "",
    ]

    # Top operations (not directly available in performance_snapshots table)
    # This would require a separate table for operation-level stats
    # For now, we'll skip this section or use placeholders
    lines.append("Top Operations: (Not available in current schema)")
    lines.append("")

    # Hashing stats (not directly available in performance_snapshots table)
    # This would require dedicated columns or a separate table
    lines.append("Hashing Statistics: (Not available in current schema)")

    return {
        "status": "ok",
        "summary": "\n".join(lines),
        "run_id": last_run.get("id"),  # Assuming 'id' is the run_id
        "completed_at": last_run.get("timestamp"),  # Assuming timestamp is completed_at
    }


# Cache for tracking last catalog modification time
_last_catalog_mtime = 0.0


@app.get("/api/performance/current")
async def get_current_performance_stats() -> Dict[str, Any]:
    """
    Get current performance statistics from the database.
    """
    catalog = get_catalog()

    # Fetch from performance_snapshots table filtered by catalog_id
    try:
        last_run_row = catalog.execute(
            "SELECT * FROM performance_snapshots WHERE catalog_id = ? ORDER BY timestamp DESC LIMIT 1",
            (str(catalog.catalog_id),),
        ).fetchone()
    except Exception:
        # Table doesn't exist - return no_data
        return {"status": "no_data", "data": None}

    if not last_run_row:
        return {"status": "no_data", "data": None}

    last_run = (
        dict(last_run_row._mapping)
        if hasattr(last_run_row, "_mapping")
        else dict(last_run_row)
    )

    # Check if analysis is currently running by comparing timestamps
    # If elapsed_seconds is 0 or very small, it might be running or just started
    # A more robust check would involve a 'status' column in the performance_snapshots table
    status = "idle"
    if (
        last_run.get("elapsed_seconds", 0) < 1.0
    ):  # Heuristic: if duration is very short, might be running
        status = "running"

    # Fetch history (last 5 runs)
    try:
        history_rows = catalog.execute(
            "SELECT * FROM performance_snapshots WHERE catalog_id = ? ORDER BY timestamp DESC LIMIT 5",
            (str(catalog.catalog_id),),
        ).fetchall()
        history = [
            dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
            for row in history_rows
        ]
    except Exception:
        history = []

    return {
        "status": status,
        "data": last_run,
        "history": history,
    }


@app.websocket("/ws/performance")
async def websocket_performance_updates(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time performance updates.

    NOTE: Due to multi-process architecture (CLI runs separately from web server),
    WebSocket updates are not sent during analysis. Use GET /api/performance/current
    with polling instead for real-time updates.

    This endpoint is kept for backwards compatibility and future enhancements.
    """
    await ws_manager.connect(websocket)
    try:
        # Keep connection alive and send periodic updates
        while True:
            # Wait for messages from client (ping/pong)
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ws_manager.disconnect(websocket)


async def broadcast_performance_update(stats_data: Dict[str, Any]) -> None:
    """
    Broadcast performance statistics update to all connected WebSocket clients.

    This function should be called during analysis to send real-time updates.

    Args:
        stats_data: Performance statistics data to broadcast
    """
    message = {
        "type": "performance_update",
        "timestamp": datetime.now().isoformat(),
        "data": stats_data,
    }
    await ws_manager.broadcast(message)


def sync_broadcast_performance_update(stats_data: Dict[str, Any]) -> None:  # noqa: C901
    """
    Synchronous wrapper for broadcasting performance updates.

    This can be used as a callback in PerformanceTracker for real-time WebSocket updates.

    **IMPORTANT LIMITATIONS:**
    - This function bridges synchronous analysis code with async WebSocket broadcasting
    - In production, WebSocket connections must be managed by the FastAPI server process
    - CLI analysis runs in a separate process and cannot directly push to WebSockets
    - For production deployments, consider a message queue (Redis, RabbitMQ) to bridge processes
    - This implementation is fire-and-forget; errors in broadcasting won't be reported
    - Updates are throttled by PerformanceTracker (default: 1 update/second) to avoid flooding

    **USE CASES:**
    - Development/testing with web UI running alongside analysis
    - Single-process deployments where FastAPI and analysis share the same process
    - Monitoring dashboards where occasional missed updates are acceptable

    **NOT RECOMMENDED FOR:**
    - Production multi-process deployments (CLI + separate web server)
    - Critical monitoring where every update must be guaranteed
    - High-frequency updates (use update_interval parameter in PerformanceTracker)

    Example:
        ```python
        from vam_tools.web.api import sync_broadcast_performance_update
        from vam_tools.core.performance_stats import PerformanceTracker

        # Create tracker with broadcast callback (throttled to 1 update/sec by default)
        tracker = PerformanceTracker(
            update_callback=sync_broadcast_performance_update,
            update_interval=1.0  # Minimum 1 second between broadcasts
        )

        # Use tracker during analysis - updates will be broadcast automatically
        with tracker.track_operation("scan_files"):
            # ... do work ...
            pass
        ```

    Args:
        stats_data: Performance statistics data to broadcast
    """
    import asyncio

    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a task
            asyncio.create_task(broadcast_performance_update(stats_data))
        else:
            # If no loop is running, run the coroutine
            loop.run_until_complete(broadcast_performance_update(stats_data))
    except RuntimeError:
        # No event loop available, create one
        asyncio.run(broadcast_performance_update(stats_data))


# ============================================================================
# Burst Detection Endpoints
# ============================================================================


@app.get("/api/catalogs/{catalog_id}/bursts")
async def list_bursts(
    catalog_id: str,
    limit: int = 100,
    offset: int = 0,
):
    """List burst groups for a catalog.

    Args:
        catalog_id: Catalog ID
        limit: Maximum bursts to return
        offset: Pagination offset
    """
    from sqlalchemy import text

    db = get_catalog_db(catalog_id)

    try:
        result = db.session.execute(
            text(
                """
                SELECT
                    b.id, b.image_count, b.start_time, b.end_time,
                    b.duration_seconds, b.camera_make, b.camera_model,
                    b.best_image_id, b.selection_method
                FROM bursts b
                WHERE b.catalog_id = :catalog_id
                ORDER BY b.start_time DESC
                LIMIT :limit OFFSET :offset
            """
            ),
            {"catalog_id": catalog_id, "limit": limit, "offset": offset},
        )

        bursts = [
            {
                "id": str(row[0]),
                "image_count": row[1],
                "start_time": row[2].isoformat() if row[2] else None,
                "end_time": row[3].isoformat() if row[3] else None,
                "duration_seconds": row[4],
                "camera_make": row[5],
                "camera_model": row[6],
                "best_image_id": str(row[7]) if row[7] else None,
                "selection_method": row[8],
                "best_thumbnail_url": (
                    f"/api/catalogs/{catalog_id}/images/{row[7]}/thumbnail"
                    if row[7]
                    else None
                ),
            }
            for row in result.fetchall()
        ]

        # Get total count
        count_result = db.session.execute(
            text("SELECT COUNT(*) FROM bursts WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id},
        )
        total = count_result.scalar() or 0

        return {
            "bursts": bursts,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    finally:
        db.close()


@app.get("/api/catalogs/{catalog_id}/bursts/{burst_id}")
async def get_burst(catalog_id: str, burst_id: str):
    """Get burst details including all images.

    Args:
        catalog_id: Catalog ID
        burst_id: Burst ID
    """
    from sqlalchemy import text

    db = get_catalog_db(catalog_id)

    try:
        # Get burst info
        result = db.session.execute(
            text(
                """
                SELECT
                    id, image_count, start_time, end_time,
                    duration_seconds, camera_make, camera_model,
                    best_image_id, selection_method
                FROM bursts
                WHERE id = :burst_id AND catalog_id = :catalog_id
            """
            ),
            {"burst_id": burst_id, "catalog_id": catalog_id},
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Burst not found")

        # Get images in burst
        images_result = db.session.execute(
            text(
                """
                SELECT id, source_path, burst_sequence, quality_score
                FROM images
                WHERE burst_id = :burst_id
                ORDER BY burst_sequence
            """
            ),
            {"burst_id": burst_id},
        )

        images = [
            {
                "id": str(img[0]),
                "source_path": img[1],
                "sequence": img[2],
                "quality_score": img[3],
                "is_best": str(img[0]) == str(row[7]),
                "thumbnail_url": f"/api/catalogs/{catalog_id}/images/{img[0]}/thumbnail",
            }
            for img in images_result.fetchall()
        ]

        return {
            "id": str(row[0]),
            "image_count": row[1],
            "start_time": row[2].isoformat() if row[2] else None,
            "end_time": row[3].isoformat() if row[3] else None,
            "duration_seconds": row[4],
            "camera_make": row[5],
            "camera_model": row[6],
            "best_image_id": str(row[7]) if row[7] else None,
            "selection_method": row[8],
            "images": images,
        }
    finally:
        db.close()


@app.put("/api/catalogs/{catalog_id}/bursts/{burst_id}")
async def update_burst(catalog_id: str, burst_id: str, data: dict):
    """Update burst (e.g., change best image).

    Args:
        catalog_id: Catalog ID
        burst_id: Burst ID
        data: Update data (best_image_id)
    """
    from sqlalchemy import text

    db = get_catalog_db(catalog_id)

    try:
        if "best_image_id" in data:
            db.session.execute(
                text(
                    """
                    UPDATE bursts
                    SET best_image_id = :best_id, selection_method = 'manual'
                    WHERE id = :burst_id AND catalog_id = :catalog_id
                """
                ),
                {
                    "best_id": data["best_image_id"],
                    "burst_id": burst_id,
                    "catalog_id": catalog_id,
                },
            )
            db.session.commit()

        return {"status": "updated"}
    finally:
        db.close()


@app.post("/api/catalogs/{catalog_id}/detect-bursts", status_code=202)
async def start_burst_detection(
    catalog_id: str,
    gap_threshold: float = 2.0,
    min_burst_size: int = 3,
):
    """Start burst detection job.

    Args:
        catalog_id: Catalog ID to process
        gap_threshold: Maximum seconds between burst images
        min_burst_size: Minimum images to form a burst
    """
    if detect_bursts_task is None:
        raise HTTPException(
            status_code=500, detail="Burst detection task not available"
        )

    task = detect_bursts_task.delay(
        catalog_id=catalog_id,
        gap_threshold=gap_threshold,
        min_burst_size=min_burst_size,
    )

    return {
        "job_id": task.id,
        "status": "queued",
        "message": "Burst detection job started",
    }


# ============================================================================
# Semantic Search Endpoints
# ============================================================================


@app.get("/api/catalogs/{catalog_id}/search")
async def search_images(
    catalog_id: str,
    q: str,
    limit: int = 50,
    threshold: float = 0.2,
):
    """Search for images using natural language query.

    Args:
        catalog_id: Catalog to search in
        q: Search query (e.g., "sunset over mountains")
        limit: Maximum results to return
        threshold: Minimum similarity score (0-1)
    """
    db = get_catalog_db(catalog_id)
    service = get_search_service()

    try:
        results = service.search(
            session=db.session,
            catalog_id=catalog_id,
            query=q,
            limit=limit,
            threshold=threshold,
        )

        return {
            "query": q,
            "results": [
                {
                    "image_id": r.image_id,
                    "source_path": r.source_path,
                    "similarity_score": r.similarity_score,
                    "thumbnail_url": f"/api/catalogs/{catalog_id}/images/{r.image_id}/thumbnail",
                }
                for r in results
            ],
            "count": len(results),
        }
    finally:
        db.close()


@app.get("/api/catalogs/{catalog_id}/similar/{image_id}")
async def find_similar_images(
    catalog_id: str,
    image_id: str,
    limit: int = 20,
    threshold: float = 0.5,
):
    """Find images similar to a given image.

    Args:
        catalog_id: Catalog to search in
        image_id: Source image ID
        limit: Maximum results to return
        threshold: Minimum similarity score (0-1)
    """
    db = get_catalog_db(catalog_id)
    service = get_search_service()

    try:
        results = service.find_similar(
            session=db.session,
            catalog_id=catalog_id,
            image_id=image_id,
            limit=limit,
            threshold=threshold,
        )

        return {
            "source_image_id": image_id,
            "results": [
                {
                    "image_id": r.image_id,
                    "source_path": r.source_path,
                    "similarity_score": r.similarity_score,
                    "thumbnail_url": f"/api/catalogs/{catalog_id}/images/{r.image_id}/thumbnail",
                }
                for r in results
            ],
            "count": len(results),
        }
    finally:
        db.close()


# ============================================================================
# Edit Mode Endpoints
# ============================================================================


class EditData(BaseModel):
    """Edit data for non-destructive transforms."""

    version: int = 1
    transforms: Dict[str, Any] = {"rotation": 0, "flip_h": False, "flip_v": False}


class HistogramResponse(BaseModel):
    """RGB histogram data."""

    red: List[int]
    green: List[int]
    blue: List[int]
    luminance: List[int]


@app.get("/api/catalogs/{catalog_id}/images/{image_id}/histogram")
async def get_image_histogram(catalog_id: str, image_id: str) -> HistogramResponse:
    """Generate RGB histogram for an image.

    Args:
        catalog_id: Catalog ID
        image_id: Image ID

    Returns:
        Histogram data with red, green, blue, and luminance channels (256 bins each)
    """
    import numpy as np

    db = get_catalog_db(catalog_id)

    try:
        # Get image record
        from sqlalchemy import text

        from ..db.models import Image as ImageModel

        result = db.session.execute(
            text(
                "SELECT source_path FROM images WHERE id = :id AND catalog_id = :catalog_id"
            ),
            {"id": image_id, "catalog_id": catalog_id},
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Image not found")

        source_path = Path(row[0])
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found on disk")

        # Load image and compute histogram
        try:
            with Image.open(source_path) as img:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Get image data as numpy array
                img_array = np.array(img)

                # Compute histograms for each channel
                red_hist = np.histogram(img_array[:, :, 0], bins=256, range=(0, 256))[0]
                green_hist = np.histogram(img_array[:, :, 1], bins=256, range=(0, 256))[
                    0
                ]
                blue_hist = np.histogram(img_array[:, :, 2], bins=256, range=(0, 256))[
                    0
                ]

                # Compute luminance histogram (using standard weights)
                luminance = (
                    0.299 * img_array[:, :, 0]
                    + 0.587 * img_array[:, :, 1]
                    + 0.114 * img_array[:, :, 2]
                )
                lum_hist = np.histogram(luminance, bins=256, range=(0, 256))[0]

                return HistogramResponse(
                    red=red_hist.tolist(),
                    green=green_hist.tolist(),
                    blue=blue_hist.tolist(),
                    luminance=lum_hist.tolist(),
                )
        except Exception as e:
            logger.error(f"Error computing histogram for {source_path}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error computing histogram: {e}"
            )
    finally:
        db.close()


@app.get("/api/catalogs/{catalog_id}/images/{image_id}/edit")
async def get_image_edit_data(catalog_id: str, image_id: str) -> Dict[str, Any]:
    """Get current edit data for an image.

    Args:
        catalog_id: Catalog ID
        image_id: Image ID

    Returns:
        Edit data or default empty structure
    """
    from sqlalchemy import text

    db = get_catalog_db(catalog_id)

    try:
        result = db.session.execute(
            text(
                "SELECT edit_data FROM images WHERE id = :id AND catalog_id = :catalog_id"
            ),
            {"id": image_id, "catalog_id": catalog_id},
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Image not found")

        edit_data = row[0]
        if edit_data is None:
            # Return default structure
            return {
                "version": 1,
                "transforms": {"rotation": 0, "flip_h": False, "flip_v": False},
            }

        return edit_data
    finally:
        db.close()


@app.put("/api/catalogs/{catalog_id}/images/{image_id}/edit")
async def update_image_edit_data(
    catalog_id: str, image_id: str, edit_data: EditData
) -> Dict[str, Any]:
    """Update edit data for an image.

    Args:
        catalog_id: Catalog ID
        image_id: Image ID
        edit_data: New edit data

    Returns:
        Success status and updated edit data
    """
    from sqlalchemy import text

    db = get_catalog_db(catalog_id)

    try:
        # Verify image exists
        result = db.session.execute(
            text("SELECT id FROM images WHERE id = :id AND catalog_id = :catalog_id"),
            {"id": image_id, "catalog_id": catalog_id},
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Image not found")

        # Update edit_data
        import json

        edit_dict = edit_data.model_dump()
        db.session.execute(
            text(
                "UPDATE images SET edit_data = :edit_data, updated_at = NOW() "
                "WHERE id = :id AND catalog_id = :catalog_id"
            ),
            {
                "edit_data": json.dumps(edit_dict),
                "id": image_id,
                "catalog_id": catalog_id,
            },
        )
        db.session.commit()

        return {"success": True, "edit_data": edit_dict}
    finally:
        db.close()


@app.delete("/api/catalogs/{catalog_id}/images/{image_id}/edit")
async def reset_image_edit_data(catalog_id: str, image_id: str) -> Dict[str, Any]:
    """Reset edit data for an image (clear all edits).

    Args:
        catalog_id: Catalog ID
        image_id: Image ID

    Returns:
        Success status
    """
    from sqlalchemy import text

    db = get_catalog_db(catalog_id)

    try:
        # Verify image exists
        result = db.session.execute(
            text("SELECT id FROM images WHERE id = :id AND catalog_id = :catalog_id"),
            {"id": image_id, "catalog_id": catalog_id},
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Image not found")

        # Clear edit_data
        db.session.execute(
            text(
                "UPDATE images SET edit_data = NULL, updated_at = NOW() "
                "WHERE id = :id AND catalog_id = :catalog_id"
            ),
            {"id": image_id, "catalog_id": catalog_id},
        )
        db.session.commit()

        return {"success": True, "message": "Edit data reset to original"}
    finally:
        db.close()


@app.get("/api/catalogs/{catalog_id}/images/{image_id}/full")
async def get_full_image(
    catalog_id: str,
    image_id: str,
    apply_transforms: bool = Query(
        False, description="Apply stored transforms to image"
    ),
) -> Union[FileResponse, StreamingResponse]:
    """Serve full-size image, optionally with transforms applied.

    Args:
        catalog_id: Catalog ID
        image_id: Image ID
        apply_transforms: Whether to apply stored rotation/flip transforms

    Returns:
        Image file (JPEG for processed images, original format otherwise)
    """
    from sqlalchemy import text

    db = get_catalog_db(catalog_id)

    try:
        result = db.session.execute(
            text(
                "SELECT source_path, edit_data FROM images "
                "WHERE id = :id AND catalog_id = :catalog_id"
            ),
            {"id": image_id, "catalog_id": catalog_id},
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Image not found")

        source_path = Path(row[0])
        edit_data = row[1]

        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found on disk")

        # If no transforms requested or no edit_data, serve original
        if not apply_transforms or edit_data is None:
            return FileResponse(
                source_path,
                headers={"Cache-Control": "public, max-age=3600"},
            )

        # Apply transforms
        transforms = edit_data.get("transforms", {})
        rotation = transforms.get("rotation", 0)
        flip_h = transforms.get("flip_h", False)
        flip_v = transforms.get("flip_v", False)

        # If no actual transforms, serve original
        if rotation == 0 and not flip_h and not flip_v:
            return FileResponse(
                source_path,
                headers={"Cache-Control": "public, max-age=3600"},
            )

        # Load and transform image
        try:
            with Image.open(source_path) as img:
                # Convert to RGB if needed
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                # Apply rotation (PIL rotates counter-clockwise, so negate for clockwise)
                if rotation != 0:
                    img = img.rotate(-rotation, expand=True)

                # Apply flips
                if flip_h:
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if flip_v:
                    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                # Save to buffer
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=95)
                buffer.seek(0)

                return StreamingResponse(
                    buffer,
                    media_type="image/jpeg",
                    headers={
                        "Content-Disposition": f"inline; filename={source_path.stem}_edited.jpg",
                        "Cache-Control": "no-cache",  # Don't cache transformed images
                    },
                )
        except Exception as e:
            logger.error(f"Error applying transforms to {source_path}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error applying transforms: {e}"
            )
    finally:
        db.close()


@app.post("/api/catalogs/{catalog_id}/images/{image_id}/export-xmp")
async def export_xmp_sidecar(catalog_id: str, image_id: str) -> Dict[str, Any]:
    """Export edit data as XMP sidecar file for Darktable compatibility.

    The XMP file will be created in the same directory as the source image
    with the same filename but .xmp extension.

    Args:
        catalog_id: Catalog ID
        image_id: Image ID

    Returns:
        Success status and path to created XMP file
    """
    from sqlalchemy import text

    db = get_catalog_db(catalog_id)

    try:
        result = db.session.execute(
            text(
                "SELECT source_path, edit_data FROM images "
                "WHERE id = :id AND catalog_id = :catalog_id"
            ),
            {"id": image_id, "catalog_id": catalog_id},
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Image not found")

        source_path = Path(row[0])
        edit_data = row[1]

        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found on disk")

        # Calculate XMP file path
        xmp_path = source_path.with_suffix(".xmp")

        # Get transforms
        transforms = edit_data.get("transforms", {}) if edit_data else {}
        rotation = transforms.get("rotation", 0)
        flip_h = transforms.get("flip_h", False)
        flip_v = transforms.get("flip_v", False)

        # Map transforms to EXIF orientation value
        # Standard EXIF orientation mapping:
        # 1 = normal, 2 = flip H, 3 = 180, 4 = 180 + flip H
        # 5 = 90 CW + flip H, 6 = 90 CW, 7 = 90 CCW + flip H, 8 = 90 CCW
        orientation = 1  # Default: normal

        if rotation == 0:
            orientation = 2 if flip_h else 1
        elif rotation == 90:
            orientation = 5 if flip_h else 6
        elif rotation == 180:
            orientation = 4 if flip_h else 3
        elif rotation == 270 or rotation == -90:
            orientation = 7 if flip_h else 8

        # Generate XMP content
        xmp_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
      <tiff:Orientation>{orientation}</tiff:Orientation>
      <xmp:CreatorTool>VAM Tools</xmp:CreatorTool>
      <xmp:ModifyDate>{datetime.now().isoformat()}</xmp:ModifyDate>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""

        # Write XMP file
        try:
            xmp_path.write_text(xmp_content, encoding="utf-8")
            logger.info(f"Exported XMP sidecar: {xmp_path}")

            return {
                "success": True,
                "xmp_path": str(xmp_path),
                "orientation": orientation,
            }
        except PermissionError:
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: Cannot write to {xmp_path.parent}",
            )
        except Exception as e:
            logger.error(f"Error writing XMP file {xmp_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Error writing XMP file: {e}")
    finally:
        db.close()
