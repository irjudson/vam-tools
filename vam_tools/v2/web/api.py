"""
FastAPI backend for catalog review UI.
"""

import io
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from ..core.catalog import CatalogDatabase
from ..core.types import ImageRecord, Statistics

logger = logging.getLogger(__name__)

# Register HEIC support for Pillow
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    logger.debug("HEIC support registered for web viewer")
except ImportError:
    logger.warning("pillow-heif not installed, HEIC files cannot be displayed in web viewer")

app = FastAPI(title="VAM Tools Catalog Viewer", version="2.0.0")

# Enable CORS for Vue.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global catalog instance
_catalog: Optional[CatalogDatabase] = None
_catalog_path: Optional[Path] = None
_catalog_mtime: Optional[float] = None  # Track last modification time


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
    global _catalog, _catalog_path, _catalog_mtime

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


@app.get("/")
async def root():
    """Serve the frontend UI."""
    static_dir = Path(__file__).parent / "static"
    index_file = static_dir / "index.html"

    if index_file.exists():
        with open(index_file, "r") as f:
            return HTMLResponse(content=f.read())

    return {"message": "VAM Tools Catalog API", "version": "2.0.0"}


@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {"message": "VAM Tools Catalog API", "version": "2.0.0"}


@app.get("/api/catalog/info", response_model=CatalogInfo)
async def get_catalog_info():
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
    filter_type: Optional[str] = Query(None, regex="^(no_date|suspicious|image|video)$"),
    sort_by: str = Query("date", regex="^(date|path|size)$"),
):
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
            key=lambda x: x.dates.selected_date if (x.dates and x.dates.selected_date) else datetime.min,
            reverse=True,
        )
    elif sort_by == "path":
        images.sort(key=lambda x: str(x.source_path))
    elif sort_by == "size":
        images.sort(
            key=lambda x: x.metadata.size_bytes if (x.metadata and x.metadata.size_bytes) else 0,
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
                selected_date=img.dates.selected_date.isoformat()
                if (img.dates and img.dates.selected_date)
                else None,
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
async def get_image_detail(image_id: str):
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
            "filename_date": image.dates.filename_date.isoformat()
            if image.dates.filename_date
            else None,
            "directory_date": image.dates.directory_date,
            "filesystem_created": image.dates.filesystem_created.isoformat()
            if image.dates.filesystem_created
            else None,
            "filesystem_modified": image.dates.filesystem_modified.isoformat()
            if image.dates.filesystem_modified
            else None,
            "selected_date": image.dates.selected_date.isoformat()
            if image.dates.selected_date
            else None,
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


@app.get("/api/images/{image_id}/file")
async def get_image_file(image_id: str):
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
    raw_formats = ['.arw', '.cr2', '.cr3', '.nef', '.dng', '.orf', '.rw2', '.pef', '.sr2', '.raf', '.raw']

    # HEIC/TIFF can be converted with Pillow
    pillow_convertible = ['.heic', '.heif', '.tif', '.tiff']

    # Handle RAW files - extract embedded JPEG preview using ExifTool
    if file_ext in raw_formats:
        try:
            # Use ExifTool to extract preview image
            # Most RAW files have embedded JPEG previews
            result = subprocess.run(
                ['exiftool', '-b', '-PreviewImage', str(file_path)],
                capture_output=True,
                timeout=10
            )

            if result.returncode == 0 and len(result.stdout) > 0:
                # Successfully extracted preview
                buffer = io.BytesIO(result.stdout)
                return StreamingResponse(
                    buffer,
                    media_type="image/jpeg",
                    headers={"Content-Disposition": f"inline; filename={file_path.stem}_preview.jpg"}
                )
            else:
                # No preview found, try different preview tags
                result = subprocess.run(
                    ['exiftool', '-b', '-JpgFromRaw', str(file_path)],
                    capture_output=True,
                    timeout=10
                )

                if result.returncode == 0 and len(result.stdout) > 0:
                    buffer = io.BytesIO(result.stdout)
                    return StreamingResponse(
                        buffer,
                        media_type="image/jpeg",
                        headers={"Content-Disposition": f"inline; filename={file_path.stem}_preview.jpg"}
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"No embedded preview found in {file_ext.upper()} file"
                    )
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout extracting preview from {file_path}")
            raise HTTPException(status_code=500, detail=f"Timeout extracting preview from {file_ext.upper()} file")
        except FileNotFoundError:
            logger.error("ExifTool not found - required for RAW file previews")
            raise HTTPException(status_code=500, detail="ExifTool not installed (required for RAW previews)")
        except Exception as e:
            logger.error(f"Error extracting preview from {file_path} ({file_ext}): {e}")
            raise HTTPException(status_code=500, detail=f"Error extracting preview from {file_ext.upper()}: {e}")

    # Handle HEIC/TIFF with Pillow conversion
    elif file_ext in pillow_convertible:
        try:
            # Convert to JPEG on-the-fly for browser display
            with Image.open(file_path) as img:
                # Convert to RGB if needed (HEIC can have different color modes)
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                # Save to bytes buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)

                return StreamingResponse(
                    buffer,
                    media_type="image/jpeg",
                    headers={"Content-Disposition": f"inline; filename={file_path.stem}.jpg"}
                )
        except Exception as e:
            logger.error(f"Error converting image {file_path} ({file_ext}): {e}")
            raise HTTPException(status_code=500, detail=f"Error converting {file_ext.upper()} image: {e}")

    # For browser-native formats (JPEG, PNG, GIF, WebP), serve directly
    return FileResponse(file_path)


@app.get("/api/statistics/summary")
async def get_statistics_summary():
    """Get various statistics about the catalog."""
    catalog = get_catalog()
    stats = catalog.get_statistics()
    images = catalog.list_images()

    # Calculate additional statistics
    by_format = {}
    by_extension = {}
    by_category = {"image": 0, "video": 0}
    by_date_source = {}
    by_size_bucket = {}
    by_issue_type = {}
    suspicious = 0
    no_date = 0
    by_year = {}

    # Define size buckets (in bytes)
    size_buckets = [
        (0, 100_000, "< 100 KB"),
        (100_000, 1_000_000, "100 KB - 1 MB"),
        (1_000_000, 5_000_000, "1 MB - 5 MB"),
        (5_000_000, 10_000_000, "5 MB - 10 MB"),
        (10_000_000, 50_000_000, "10 MB - 50 MB"),
        (50_000_000, float('inf'), "> 50 MB"),
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
            by_issue_type["suspicious_date"] = by_issue_type.get("suspicious_date", 0) + 1
        if not (image.dates and image.dates.selected_date):
            no_date += 1
            by_issue_type["no_date"] = by_issue_type.get("no_date", 0) + 1
        if image.dates and image.dates.confidence < 70:
            by_issue_type["low_confidence_date"] = by_issue_type.get("low_confidence_date", 0) + 1

        # Year distribution
        if image.dates and image.dates.selected_date:
            year = str(image.dates.selected_date.year)
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
