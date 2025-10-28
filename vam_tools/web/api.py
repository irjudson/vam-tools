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
from PIL import Image
from pydantic import BaseModel

from ..core.catalog import CatalogDatabase

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

# Global catalog instance
_catalog: Optional[CatalogDatabase] = None
_catalog_path: Optional[Path] = None
_catalog_mtime: Optional[float] = None  # Track last modification time


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


class CatalogInfo(BaseModel):
    """Overall catalog information."""

    version: str
    catalog_id: str
    created: str
    last_updated: str
    phase: str
    statistics: CatalogStats


def init_catalog(catalog_path: Path) -> None:
    """Initialize the catalog for API access."""
    global _catalog, _catalog_path, _catalog_mtime
    _catalog_path = catalog_path
    _catalog = CatalogDatabase(catalog_path)
    _catalog.load()

    # Track modification time
    db_file = catalog_path / ".catalog.json"
    if db_file.exists():
        _catalog_mtime = db_file.stat().st_mtime

    logger.info(f"Catalog loaded from {catalog_path}")


def get_catalog() -> CatalogDatabase:
    """Get the current catalog instance, reloading if file has changed."""
    global _catalog_mtime

    if _catalog is None or _catalog_path is None:
        raise HTTPException(status_code=500, detail="Catalog not initialized")

    # Check if catalog file has been modified
    db_file = _catalog_path / ".catalog.json"
    if db_file.exists():
        current_mtime = db_file.stat().st_mtime
        if _catalog_mtime is None or current_mtime > _catalog_mtime:
            logger.info("Catalog file changed, reloading...")
            _catalog.load()
            _catalog_mtime = current_mtime

    return _catalog


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
    state = catalog.get_state()
    stats = catalog.get_statistics()

    # Count suspicious dates
    suspicious_count = 0
    for image in catalog.list_images():
        if image.dates and image.dates.suspicious:
            suspicious_count += 1

    return CatalogInfo(
        version=state.version,
        catalog_id=state.catalog_id,
        created=state.created.isoformat() if state.created else "",
        last_updated=state.last_updated.isoformat() if state.last_updated else "",
        phase=state.phase.value,
        statistics=CatalogStats(
            total_images=stats.total_images,
            total_videos=stats.total_videos,
            total_size_bytes=stats.total_size_bytes,
            no_date=stats.no_date,
            suspicious_dates=suspicious_count,
        ),
    )


@app.get("/api/images", response_model=List[ImageSummary])
async def list_images(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    filter_type: Optional[str] = Query(
        None, pattern="^(no_date|suspicious|image|video)$"
    ),
    sort_by: str = Query("date", pattern="^(date|path|size)$"),
) -> List[ImageSummary]:
    """
    List images with pagination and filtering.

    - skip: Number of images to skip (for pagination)
    - limit: Maximum number of images to return
    - filter_type: Filter by type (no_date, suspicious, image, video)
    - sort_by: Sort order (date, path, size)
    """
    catalog = get_catalog()
    images = catalog.list_images()

    # Apply filters
    if filter_type == "no_date":
        images = [img for img in images if not (img.dates and img.dates.selected_date)]
    elif filter_type == "suspicious":
        images = [img for img in images if img.dates and img.dates.suspicious]
    elif filter_type == "image":
        images = [img for img in images if img.file_type.value == "image"]
    elif filter_type == "video":
        images = [img for img in images if img.file_type.value == "video"]

    # Sort
    if sort_by == "date":
        images.sort(
            key=lambda x: (
                x.dates.selected_date
                if (x.dates and x.dates.selected_date)
                else datetime.min
            ),
            reverse=True,
        )
    elif sort_by == "path":
        images.sort(key=lambda x: str(x.source_path))
    elif sort_by == "size":
        images.sort(
            key=lambda x: (
                x.metadata.size_bytes if (x.metadata and x.metadata.size_bytes) else 0
            ),
            reverse=True,
        )

    # Paginate
    images = images[skip : skip + limit]

    # Convert to summaries
    summaries = []
    for img in images:
        summaries.append(
            ImageSummary(
                id=img.id,
                source_path=str(img.source_path),
                file_type=img.file_type.value,
                selected_date=(
                    img.dates.selected_date.isoformat()
                    if (img.dates and img.dates.selected_date)
                    else None
                ),
                date_source=img.dates.selected_source if img.dates else None,
                confidence=img.dates.confidence if img.dates else 0,
                suspicious=img.dates.suspicious if img.dates else False,
                format=img.metadata.format if img.metadata else None,
                resolution=img.metadata.resolution if img.metadata else None,
                size_bytes=img.metadata.size_bytes if img.metadata else None,
            )
        )

    return summaries


@app.get("/api/images/{image_id}", response_model=ImageDetail)
async def get_image_detail(image_id: str) -> ImageDetail:
    """Get detailed information about a specific image."""
    catalog = get_catalog()
    image = catalog.get_image(image_id)

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Build dates dict
    dates_dict = {}
    if image.dates:
        dates_dict = {
            "exif_dates": {
                k: v.isoformat() if v else None
                for k, v in image.dates.exif_dates.items()
            },
            "filename_date": (
                image.dates.filename_date.isoformat()
                if image.dates.filename_date
                else None
            ),
            "directory_date": image.dates.directory_date,
            "filesystem_created": (
                image.dates.filesystem_created.isoformat()
                if image.dates.filesystem_created
                else None
            ),
            "filesystem_modified": (
                image.dates.filesystem_modified.isoformat()
                if image.dates.filesystem_modified
                else None
            ),
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
            "exif_dates": {},
            "filename_date": None,
            "directory_date": None,
            "filesystem_created": None,
            "filesystem_modified": None,
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
        }
    else:
        metadata_dict = {
            "format": None,
            "resolution": None,
            "size_bytes": None,
            "exif": {},
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
                # Successfully extracted preview
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


@app.get("/api/statistics/summary")
async def get_statistics_summary() -> Dict[str, Any]:
    """Get various statistics about the catalog."""
    catalog = get_catalog()
    stats = catalog.get_statistics()
    images = catalog.list_images()

    # Calculate additional statistics
    by_format: Dict[str, int] = {}
    by_extension: Dict[str, int] = {}
    by_category = {"image": 0, "video": 0}
    by_date_source: Dict[str, int] = {}
    by_size_bucket: Dict[str, int] = {}
    by_issue_type: Dict[str, int] = {}
    suspicious = 0
    no_date = 0
    by_year: Dict[int, int] = {}

    # Define size buckets (in bytes)
    size_buckets = [
        (0, 100_000, "< 100 KB"),
        (100_000, 1_000_000, "100 KB - 1 MB"),
        (1_000_000, 5_000_000, "1 MB - 5 MB"),
        (5_000_000, 10_000_000, "5 MB - 10 MB"),
        (10_000_000, 50_000_000, "10 MB - 50 MB"),
        (50_000_000, float("inf"), "> 50 MB"),
    ]

    for bucket_min, bucket_max, bucket_label in size_buckets:
        by_size_bucket[bucket_label] = 0

    for image in images:
        # Format distribution
        fmt = (image.metadata.format if image.metadata else None) or "unknown"
        by_format[fmt] = by_format.get(fmt, 0) + 1

        # Extension distribution
        ext = Path(image.source_path).suffix.lower() or "no extension"
        by_extension[ext] = by_extension.get(ext, 0) + 1

        # Category distribution
        category = image.file_type.value if image.file_type else "unknown"
        by_category[category] = by_category.get(category, 0) + 1

        # Size distribution
        size = (image.metadata.size_bytes if image.metadata else None) or 0
        for bucket_min, bucket_max, bucket_label in size_buckets:
            if bucket_min <= size < bucket_max:
                by_size_bucket[bucket_label] += 1
                break

        # Date source distribution
        source = (image.dates.selected_source if image.dates else None) or "none"
        by_date_source[source] = by_date_source.get(source, 0) + 1

        # Issues distribution
        if image.dates and image.dates.suspicious:
            suspicious += 1
            by_issue_type["suspicious_date"] = (
                by_issue_type.get("suspicious_date", 0) + 1
            )
        if not (image.dates and image.dates.selected_date):
            no_date += 1
            by_issue_type["no_date"] = by_issue_type.get("no_date", 0) + 1
        if image.dates and image.dates.confidence < 70:
            by_issue_type["low_confidence_date"] = (
                by_issue_type.get("low_confidence_date", 0) + 1
            )

        # Year distribution
        if image.dates and image.dates.selected_date:
            year = image.dates.selected_date.year
            by_year[year] = by_year.get(year, 0) + 1

    return {
        "total": {
            "images": stats.total_images,
            "videos": stats.total_videos,
            "size_bytes": stats.total_size_bytes,
        },
        "issues": {
            "no_date": no_date,
            "suspicious_dates": suspicious,
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
    Get comprehensive statistics for dashboard display.

    Combines catalog overview, duplicates, review queue, and hash performance.
    Optimized to avoid loading all images into memory.
    """
    catalog = get_catalog()
    stats = catalog.get_statistics()
    groups = catalog.get_duplicate_groups()

    # Basic catalog stats - use existing statistics
    total_files = stats.total_images + stats.total_videos
    total_size_bytes = stats.total_size_bytes

    # Duplicate stats
    total_groups = len(groups)
    duplicates_needing_review = sum(1 for g in groups if g.needs_review)
    total_duplicate_images = sum(len(g.images) for g in groups)

    # Calculate potential space savings (only load images in duplicate groups)
    potential_space_savings = 0
    for group in groups:
        images_in_group = [catalog.get_image(img_id) for img_id in group.images]
        sizes = sorted(
            [
                (img.metadata.size_bytes or 0)
                for img in images_in_group
                if img and img.metadata
            ],
            reverse=True,
        )
        if len(sizes) > 1:
            potential_space_savings += sum(sizes[1:])  # All but largest

    # Review queue stats - use existing statistics where available
    date_conflicts = 0
    images_in_conflict_groups = set()
    for group in groups:
        if group.date_conflict:
            images_in_conflict_groups.update(group.images)
    date_conflicts = len(images_in_conflict_groups)

    # Use stats from catalog statistics
    no_date = stats.no_date or 0

    # Count suspicious dates, low confidence, and hash coverage
    # Since catalog data is already in memory (JSON), iterate efficiently
    suspicious_dates = 0
    low_confidence = 0
    images_with_dhash = 0
    images_with_ahash = 0
    images_with_whash = 0
    images_with_any_hash = 0

    # Get all images as dict and iterate
    all_images = catalog.get_all_images()
    for img in all_images.values():
        # Count hash coverage
        if img.metadata:
            has_dhash = bool(img.metadata.perceptual_hash_dhash)
            has_ahash = bool(img.metadata.perceptual_hash_ahash)
            has_whash = bool(img.metadata.perceptual_hash_whash)

            if has_dhash:
                images_with_dhash += 1
            if has_ahash:
                images_with_ahash += 1
            if has_whash:
                images_with_whash += 1
            if has_dhash or has_ahash or has_whash:
                images_with_any_hash += 1

        # Count review queue items (only if not already counted in no_date)
        if img.dates:
            if img.dates.selected_date and img.dates.suspicious:
                suspicious_dates += 1
            elif img.dates.selected_date and img.dates.confidence < 70:
                low_confidence += 1

    total_needing_review = date_conflicts + no_date + suspicious_dates

    hash_coverage_percent = (
        (images_with_any_hash / total_files * 100) if total_files > 0 else 0
    )

    return {
        # Catalog overview
        "catalog": {
            "total_files": total_files,
            "total_images": stats.total_images,
            "total_videos": stats.total_videos,
            "total_size_bytes": total_size_bytes,
            "total_size_gb": round(total_size_bytes / (1024**3), 2),
        },
        # Duplicate detection
        "duplicates": {
            "total_groups": total_groups,
            "needs_review": duplicates_needing_review,
            "total_duplicate_images": total_duplicate_images,
            "potential_space_savings_bytes": potential_space_savings,
            "potential_space_savings_gb": round(potential_space_savings / (1024**3), 2),
        },
        # Review queue
        "review": {
            "total_needing_review": total_needing_review,
            "date_conflicts": date_conflicts,
            "no_date": no_date,
            "suspicious_dates": suspicious_dates,
            "low_confidence": low_confidence,
        },
        # Hash performance
        "hashes": {
            "images_with_dhash": images_with_dhash,
            "images_with_ahash": images_with_ahash,
            "images_with_whash": images_with_whash,
            "images_with_any_hash": images_with_any_hash,
            "coverage_percent": round(hash_coverage_percent, 1),
            "dhash_percent": round(
                (images_with_dhash / total_files * 100) if total_files > 0 else 0, 1
            ),
            "ahash_percent": round(
                (images_with_ahash / total_files * 100) if total_files > 0 else 0, 1
            ),
            "whash_percent": round(
                (images_with_whash / total_files * 100) if total_files > 0 else 0, 1
            ),
        },
    }


@app.get("/api/duplicates/groups", response_model=List[DuplicateGroupSummary])
async def list_duplicate_groups(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    needs_review_only: bool = Query(False),
) -> List[DuplicateGroupSummary]:
    """
    List duplicate groups with pagination.

    Args:
        skip: Number of groups to skip (for pagination)
        limit: Maximum number of groups to return
        needs_review_only: Only return groups that need review

    Returns:
        List of duplicate group summaries
    """
    catalog = get_catalog()
    groups = catalog.get_duplicate_groups()

    # Filter if requested
    if needs_review_only:
        groups = [g for g in groups if g.needs_review]

    # Paginate
    groups = groups[skip : skip + limit]

    # Convert to summaries
    summaries = []
    for group in groups:
        # Get images for this group
        images_optional = [catalog.get_image(img_id) for img_id in group.images]
        images = [img for img in images_optional if img is not None]  # Filter out None

        # Calculate stats
        sizes: List[int] = [
            (img.metadata.size_bytes or 0) for img in images if img.metadata
        ]
        total_size = sum(sizes)
        formats = list(
            set(
                img.metadata.format
                for img in images
                if img.metadata and img.metadata.format
            )
        )

        summaries.append(
            DuplicateGroupSummary(
                id=group.id,
                primary_image_id=group.primary or "",
                duplicate_count=len(group.images),
                total_size_bytes=total_size,
                format_types=formats,
                needs_review=group.needs_review,
            )
        )

    return summaries


@app.get("/api/duplicates/groups/{group_id}", response_model=DuplicateGroupDetail)
async def get_duplicate_group(group_id: str) -> DuplicateGroupDetail:
    """Get detailed information about a specific duplicate group."""
    catalog = get_catalog()
    groups = catalog.get_duplicate_groups()

    # Find the group
    group = next((g for g in groups if g.id == group_id), None)
    if not group:
        raise HTTPException(status_code=404, detail="Duplicate group not found")

    # Convert similarity metrics to response format
    similarity_responses = []
    for pair_key, metrics in group.similarity_metrics.items():
        # Parse the pair key to get image IDs
        image_ids = pair_key.split(":")
        if len(image_ids) == 2:
            similarity_responses.append(
                SimilarityMetricsResponse(
                    image1_id=image_ids[0],
                    image2_id=image_ids[1],
                    dhash_distance=metrics.dhash_distance,
                    ahash_distance=metrics.ahash_distance,
                    whash_distance=metrics.whash_distance,
                    dhash_similarity=metrics.dhash_similarity,
                    ahash_similarity=metrics.ahash_similarity,
                    whash_similarity=metrics.whash_similarity,
                    overall_similarity=metrics.overall_similarity,
                )
            )

    # Calculate overall group similarity score
    if similarity_responses:
        avg_similarity = sum(m.overall_similarity for m in similarity_responses) / len(
            similarity_responses
        )
    else:
        avg_similarity = 0.0

    return DuplicateGroupDetail(
        id=group.id,
        primary_image_id=group.primary or "",
        duplicate_image_ids=group.images,
        similarity_score=avg_similarity,
        similarity_metrics=similarity_responses,
        needs_review=group.needs_review,
        review_reason="Date conflict detected" if group.date_conflict else None,
    )


@app.get("/api/duplicates/stats")
async def get_duplicate_stats() -> Dict[str, Any]:
    """Get statistics about duplicate groups."""
    catalog = get_catalog()
    groups = catalog.get_duplicate_groups()

    total_groups = len(groups)
    needs_review = sum(1 for g in groups if g.needs_review)
    total_duplicates = sum(len(g.images) for g in groups)
    total_unique = total_groups  # One primary per group

    # Calculate potential space savings
    total_duplicate_size = 0
    for group in groups:
        images_optional = [catalog.get_image(img_id) for img_id in group.images]
        images = [
            img
            for img in images_optional
            if img is not None and img.metadata is not None
        ]

        if len(images) > 1:
            # Keep largest, count rest as savings
            sizes = sorted(
                [img.metadata.size_bytes or 0 for img in images], reverse=True
            )
            total_duplicate_size += sum(sizes[1:])  # All but the largest

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
    Get review queue items.

    Returns items that need manual review, optionally filtered by type.
    """
    catalog = get_catalog()
    images = catalog.list_images()

    review_items = []

    for image in images:
        # Check for date conflicts (in duplicate groups)
        if filter_type in [None, "date_conflict"]:
            # Find if image is in a duplicate group with date conflicts
            groups = catalog.get_duplicate_groups()
            for group in groups:
                if image.id in group.images and group.date_conflict:
                    review_items.append(
                        {
                            "id": image.id,
                            "type": "date_conflict",
                            "source_path": str(image.source_path),
                            "current_date": (
                                image.dates.selected_date.isoformat()
                                if image.dates and image.dates.selected_date
                                else None
                            ),
                            "confidence": (
                                image.dates.confidence if image.dates else 0
                            ),
                            "group_id": group.id,
                        }
                    )
                    break  # Only add once per image

        # Check for no date
        if filter_type in [None, "no_date"]:
            if not image.dates or not image.dates.selected_date:
                review_items.append(
                    {
                        "id": image.id,
                        "type": "no_date",
                        "source_path": str(image.source_path),
                        "current_date": None,
                        "confidence": 0,
                        "filesystem_created": (
                            image.dates.filesystem_created.isoformat()
                            if image.dates and image.dates.filesystem_created
                            else None
                        ),
                    }
                )

        # Check for suspicious dates
        if filter_type in [None, "suspicious_date"]:
            if image.dates and image.dates.suspicious:
                review_items.append(
                    {
                        "id": image.id,
                        "type": "suspicious_date",
                        "source_path": str(image.source_path),
                        "current_date": (
                            image.dates.selected_date.isoformat()
                            if image.dates.selected_date
                            else None
                        ),
                        "confidence": image.dates.confidence,
                        "selected_source": image.dates.selected_source,
                    }
                )

    return {"items": review_items, "total": len(review_items)}


@app.patch("/api/images/{image_id}/date")
async def update_image_date(
    image_id: str, date_str: str = Query(...)
) -> Dict[str, Any]:
    """
    Update the date for an image.

    Args:
        image_id: Image ID
        date_str: New date in ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
    """
    catalog = get_catalog()
    image = catalog.get_image(image_id)

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Parse date
    try:
        from datetime import datetime

        if len(date_str) == 10:  # YYYY-MM-DD
            new_date = datetime.strptime(date_str, "%Y-%m-%d")
        else:  # YYYY-MM-DD HH:MM:SS
            new_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS",
        )

    # Update image date
    if not image.dates:
        from ..core.types import DateInfo

        image.dates = DateInfo()

    image.dates.selected_date = new_date
    image.dates.selected_source = "manual"
    image.dates.confidence = 100  # Manual dates have 100% confidence

    # Save to catalog
    catalog.update_image(image)
    catalog.save()

    return {"success": True, "image_id": image_id, "new_date": date_str}


@app.get("/api/review/stats")
async def get_review_stats() -> Dict[str, Any]:
    """Get statistics about items needing review."""
    catalog = get_catalog()
    images = catalog.list_images()

    stats = {
        "date_conflicts": 0,
        "no_date": 0,
        "suspicious_dates": 0,
        "low_confidence": 0,
        "ready_to_organize": 0,
    }

    # Count images in duplicate groups with date conflicts
    groups = catalog.get_duplicate_groups()
    images_in_conflict_groups = set()
    for group in groups:
        if group.date_conflict:
            images_in_conflict_groups.update(group.images)
    stats["date_conflicts"] = len(images_in_conflict_groups)

    # Count other issues
    for image in images:
        if not image.dates or not image.dates.selected_date:
            stats["no_date"] += 1
        elif image.dates.suspicious:
            stats["suspicious_dates"] += 1
        elif image.dates.confidence < 70:
            stats["low_confidence"] += 1
        else:
            stats["ready_to_organize"] += 1

    stats["total_needing_review"] = (
        stats["date_conflicts"] + stats["no_date"] + stats["suspicious_dates"]
    )

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
    Get historical performance statistics.

    Returns performance trends across the last 10 analysis runs.
    """
    catalog = get_catalog()

    perf_data = cast(Dict[str, Any], catalog._data.get("performance_statistics", {}))  # type: ignore[union-attr]

    if not perf_data or not isinstance(perf_data, dict):
        return {
            "status": "no_data",
            "history": [],
            "total_runs": 0,
            "average_throughput": 0,
        }

    history = perf_data.get("history", [])
    total_runs = perf_data.get("total_runs", 0)
    total_files_analyzed = perf_data.get("total_files_analyzed", 0)
    total_time_seconds = perf_data.get("total_time_seconds", 0)
    average_throughput = perf_data.get("average_throughput", 0)

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
    Get a summary report of current performance statistics.

    Returns a human-readable summary of the last analysis run.
    """
    catalog = get_catalog()

    perf_data = cast(Dict[str, Any], catalog._data.get("performance_statistics", {}))  # type: ignore[union-attr]

    if not perf_data or not isinstance(perf_data, dict) or "last_run" not in perf_data:
        return {
            "status": "no_data",
            "summary": "No performance statistics available",
        }

    last_run = perf_data["last_run"]

    # Build summary similar to PerformanceMetrics.get_summary_report()
    lines = [
        "=== Performance Analysis Summary ===",
        f"Total Duration: {last_run.get('total_duration_seconds', 0):.2f}s",
        f"Files Analyzed: {last_run.get('total_files_analyzed', 0)}",
        f"Throughput: {last_run.get('files_per_second', 0):.2f} files/sec",
        f"Data Processed: {last_run.get('bytes_processed', 0) / (1024**3):.2f} GB",
        "",
    ]

    # Get top operations
    operations = last_run.get("operations", {})
    if operations:
        lines.append("Top Operations:")
        # Sort by total time
        sorted_ops = sorted(
            operations.items(),
            key=lambda x: x[1].get("total_time_seconds", 0),
            reverse=True,
        )
        for name, stats in sorted_ops[:5]:
            duration = stats.get("total_time_seconds", 0)
            total_duration = last_run.get("total_duration_seconds", 1)
            percent = (duration / total_duration) * 100
            lines.append(f"  {name}: {duration:.2f}s ({percent:.1f}%)")
        lines.append("")

    # Hashing stats
    hashing = last_run.get("hashing", {})
    if hashing:
        lines.append("Hashing Statistics:")
        lines.append(f"  Total Hashes: {hashing.get('total_hashes_computed', 0)}")
        lines.append(f"  GPU Hashes: {hashing.get('gpu_hashes', 0)}")
        lines.append(f"  CPU Hashes: {hashing.get('cpu_hashes', 0)}")
        lines.append(f"  Failed: {hashing.get('failed_hashes', 0)}")

    return {
        "status": "ok",
        "summary": "\n".join(lines),
        "run_id": last_run.get("run_id"),
        "completed_at": last_run.get("completed_at"),
    }


@app.get("/api/performance/current")
async def get_current_performance_stats() -> Dict[str, Any]:
    """
    Get current performance statistics from catalog.

    This endpoint supports polling for real-time updates during analysis.
    The CLI writes performance stats to the catalog periodically, and
    this endpoint reads them.

    Returns:
        Dictionary with status and performance data:
        - status: "running" if analysis is active, "idle" if not, "no_data" if never run
        - data: Performance metrics if available, None otherwise
    """
    catalog = get_catalog()
    perf_stats = catalog.get_performance_statistics()

    if (
        not perf_stats
        or not isinstance(perf_stats, dict)
        or not perf_stats.get("last_run")
    ):
        return {"status": "no_data", "data": None}

    last_run = perf_stats.get("last_run")
    if not last_run or not isinstance(last_run, dict):
        return {"status": "no_data", "data": None}

    # Check if analysis is currently running by comparing timestamps
    # If completed_at is None, it's still running
    if last_run.get("completed_at") is None:
        status = "running"
    else:
        status = "idle"

    return {
        "status": status,
        "data": last_run,
        "history": perf_stats.get("history", [])[:5],  # Last 5 runs
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
