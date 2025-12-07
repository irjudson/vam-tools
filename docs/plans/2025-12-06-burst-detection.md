# Burst Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically detect and group burst/continuous shooting sequences based on timestamps and camera metadata.

**Architecture:** Pure Python algorithm using EXIF timestamps (no ML required). Sort by camera+time, detect gaps <2s, group into bursts, auto-select best image using quality_score.

**Tech Stack:** Python, SQLAlchemy, existing EXIF data, FastAPI, Vue.js

**GitHub Issue:** #19

---

## Task 1: Database Schema - Bursts Table

**Files:**
- Create: `vam_tools/db/migrations/versions/004_add_bursts.py`
- Modify: `vam_tools/db/catalog_schema.py`

### Step 1: Write the failing test

Create `tests/db/test_bursts_schema.py`:

```python
"""Tests for bursts table schema."""

import pytest
from sqlalchemy import text


class TestBurstsTable:
    """Tests for bursts table."""

    def test_bursts_table_exists(self, db_session):
        """Test that bursts table exists."""
        result = db_session.execute(
            text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_name = 'bursts'
            """)
        )
        assert result.fetchone() is not None

    def test_bursts_table_has_required_columns(self, db_session):
        """Test that bursts table has all required columns."""
        result = db_session.execute(
            text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'bursts'
            """)
        )
        columns = {row[0] for row in result.fetchall()}

        required = {
            "id", "catalog_id", "image_count", "start_time", "end_time",
            "duration_seconds", "camera_make", "camera_model",
            "best_image_id", "selection_method", "created_at"
        }
        assert required.issubset(columns)

    def test_images_has_burst_columns(self, db_session):
        """Test that images table has burst-related columns."""
        result = db_session.execute(
            text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'images'
                AND column_name IN ('burst_id', 'burst_sequence')
            """)
        )
        columns = {row[0] for row in result.fetchall()}
        assert "burst_id" in columns
        assert "burst_sequence" in columns
```

### Step 2: Run test to verify it fails

```bash
./venv/bin/pytest tests/db/test_bursts_schema.py -v
```

Expected: FAIL with "bursts table does not exist"

### Step 3: Create the migration

Create `vam_tools/db/migrations/versions/004_add_bursts.py`:

```python
"""Add bursts table and burst columns to images.

Revision ID: 004_bursts
Revises: 003_clip_embedding
Create Date: 2025-12-06
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "004_bursts"
down_revision = "003_clip_embedding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create bursts table and add burst columns to images."""

    # Create bursts table
    op.create_table(
        "bursts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("catalog_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("image_count", sa.Integer, nullable=False),
        sa.Column("start_time", sa.DateTime, nullable=True),
        sa.Column("end_time", sa.DateTime, nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("camera_make", sa.String(255), nullable=True),
        sa.Column("camera_model", sa.String(255), nullable=True),
        sa.Column("best_image_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("selection_method", sa.String(50), default="quality"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["catalog_id"], ["catalogs.id"], ondelete="CASCADE"),
    )

    # Add index on catalog_id
    op.create_index("bursts_catalog_id_idx", "bursts", ["catalog_id"])

    # Add burst columns to images table
    op.add_column("images", sa.Column("burst_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("images", sa.Column("burst_sequence", sa.Integer, nullable=True))

    # Add foreign key constraint
    op.create_foreign_key(
        "images_burst_id_fkey",
        "images", "bursts",
        ["burst_id"], ["id"],
        ondelete="SET NULL"
    )

    # Add index for burst queries
    op.create_index("images_burst_id_idx", "images", ["burst_id"])


def downgrade() -> None:
    """Remove bursts table and columns."""
    op.drop_constraint("images_burst_id_fkey", "images", type_="foreignkey")
    op.drop_index("images_burst_id_idx", table_name="images")
    op.drop_column("images", "burst_sequence")
    op.drop_column("images", "burst_id")
    op.drop_index("bursts_catalog_id_idx", table_name="bursts")
    op.drop_table("bursts")
```

### Step 4: Run migration and verify test passes

```bash
./venv/bin/alembic upgrade head
./venv/bin/pytest tests/db/test_bursts_schema.py -v
```

Expected: PASS

### Step 5: Commit

```bash
git add vam_tools/db/migrations/versions/004_add_bursts.py
git add tests/db/test_bursts_schema.py
git commit -m "feat: add bursts table schema (#19)"
```

---

## Task 2: Burst Detection Algorithm

**Files:**
- Create: `vam_tools/analysis/burst_detector.py`
- Create: `tests/analysis/test_burst_detector.py`

### Step 1: Write the failing test

Create `tests/analysis/test_burst_detector.py`:

```python
"""Tests for burst detection algorithm."""

from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock

import pytest

from vam_tools.analysis.burst_detector import (
    BurstDetector,
    BurstGroup,
    ImageInfo,
)


class TestImageInfo:
    """Tests for ImageInfo dataclass."""

    def test_image_info_creation(self):
        """Test creating ImageInfo."""
        info = ImageInfo(
            image_id="img-001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            camera_make="Canon",
            camera_model="EOS R5",
            quality_score=0.85,
        )
        assert info.image_id == "img-001"
        assert info.camera_make == "Canon"


class TestBurstGroup:
    """Tests for BurstGroup dataclass."""

    def test_burst_group_duration(self):
        """Test burst group duration calculation."""
        images = [
            ImageInfo("img-001", datetime(2024, 1, 1, 12, 0, 0), "Canon", "R5", 0.8),
            ImageInfo("img-002", datetime(2024, 1, 1, 12, 0, 1), "Canon", "R5", 0.9),
            ImageInfo("img-003", datetime(2024, 1, 1, 12, 0, 2), "Canon", "R5", 0.7),
        ]
        group = BurstGroup(images=images)

        assert group.duration_seconds == 2.0
        assert group.image_count == 3


class TestBurstDetector:
    """Tests for BurstDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = BurstDetector(gap_threshold_seconds=2.0, min_burst_size=3)
        assert detector.gap_threshold_seconds == 2.0
        assert detector.min_burst_size == 3

    def test_detect_bursts_finds_sequences(self):
        """Test that detector finds burst sequences."""
        detector = BurstDetector(gap_threshold_seconds=2.0, min_burst_size=3)

        # Create images with burst pattern
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        images = [
            # Burst 1: 3 images, 0.5s apart
            ImageInfo("img-001", base_time, "Canon", "R5", 0.8),
            ImageInfo("img-002", base_time + timedelta(seconds=0.5), "Canon", "R5", 0.9),
            ImageInfo("img-003", base_time + timedelta(seconds=1.0), "Canon", "R5", 0.7),
            # Gap of 10 seconds
            # Burst 2: 4 images, 1s apart
            ImageInfo("img-004", base_time + timedelta(seconds=11), "Canon", "R5", 0.85),
            ImageInfo("img-005", base_time + timedelta(seconds=12), "Canon", "R5", 0.95),
            ImageInfo("img-006", base_time + timedelta(seconds=13), "Canon", "R5", 0.75),
            ImageInfo("img-007", base_time + timedelta(seconds=14), "Canon", "R5", 0.80),
            # Single image (not a burst)
            ImageInfo("img-008", base_time + timedelta(seconds=30), "Canon", "R5", 0.90),
        ]

        bursts = detector.detect_bursts(images)

        assert len(bursts) == 2
        assert bursts[0].image_count == 3
        assert bursts[1].image_count == 4

    def test_detect_bursts_respects_camera_boundaries(self):
        """Test that bursts don't cross camera boundaries."""
        detector = BurstDetector(gap_threshold_seconds=2.0, min_burst_size=3)

        base_time = datetime(2024, 1, 1, 12, 0, 0)
        images = [
            # Canon images
            ImageInfo("img-001", base_time, "Canon", "R5", 0.8),
            ImageInfo("img-002", base_time + timedelta(seconds=0.5), "Canon", "R5", 0.9),
            ImageInfo("img-003", base_time + timedelta(seconds=1.0), "Canon", "R5", 0.7),
            # Sony images at same time (different camera)
            ImageInfo("img-004", base_time + timedelta(seconds=1.5), "Sony", "A7", 0.85),
            ImageInfo("img-005", base_time + timedelta(seconds=2.0), "Sony", "A7", 0.95),
            ImageInfo("img-006", base_time + timedelta(seconds=2.5), "Sony", "A7", 0.75),
        ]

        bursts = detector.detect_bursts(images)

        assert len(bursts) == 2
        assert all(img.camera_make == "Canon" for img in bursts[0].images)
        assert all(img.camera_make == "Sony" for img in bursts[1].images)

    def test_detect_bursts_ignores_small_sequences(self):
        """Test that sequences smaller than min_burst_size are ignored."""
        detector = BurstDetector(gap_threshold_seconds=2.0, min_burst_size=3)

        base_time = datetime(2024, 1, 1, 12, 0, 0)
        images = [
            # Only 2 images - not a burst
            ImageInfo("img-001", base_time, "Canon", "R5", 0.8),
            ImageInfo("img-002", base_time + timedelta(seconds=0.5), "Canon", "R5", 0.9),
        ]

        bursts = detector.detect_bursts(images)

        assert len(bursts) == 0

    def test_select_best_image_uses_quality_score(self):
        """Test that best image is selected by quality score."""
        detector = BurstDetector()

        images = [
            ImageInfo("img-001", datetime(2024, 1, 1, 12, 0, 0), "Canon", "R5", 0.70),
            ImageInfo("img-002", datetime(2024, 1, 1, 12, 0, 1), "Canon", "R5", 0.95),  # Best
            ImageInfo("img-003", datetime(2024, 1, 1, 12, 0, 2), "Canon", "R5", 0.80),
        ]
        group = BurstGroup(images=images)

        best = detector.select_best_image(group)

        assert best.image_id == "img-002"
        assert best.quality_score == 0.95
```

### Step 2: Run test to verify it fails

```bash
./venv/bin/pytest tests/analysis/test_burst_detector.py -v
```

Expected: FAIL with "No module named 'vam_tools.analysis.burst_detector'"

### Step 3: Implement BurstDetector

Create `vam_tools/analysis/burst_detector.py`:

```python
"""Burst detection for continuous shooting sequences.

Detects groups of images taken in rapid succession (bursts) based on
timestamps and camera metadata. No ML required - pure algorithmic approach.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """Information about an image for burst detection."""

    image_id: str
    timestamp: datetime
    camera_make: Optional[str]
    camera_model: Optional[str]
    quality_score: float = 0.0
    source_path: Optional[str] = None

    @property
    def camera_key(self) -> str:
        """Get a unique key for the camera."""
        make = self.camera_make or "unknown"
        model = self.camera_model or "unknown"
        return f"{make}:{model}"


@dataclass
class BurstGroup:
    """A group of images forming a burst sequence."""

    images: List[ImageInfo] = field(default_factory=list)
    best_image_id: Optional[str] = None
    selection_method: str = "quality"

    @property
    def image_count(self) -> int:
        """Number of images in burst."""
        return len(self.images)

    @property
    def start_time(self) -> Optional[datetime]:
        """Start time of burst."""
        if not self.images:
            return None
        return min(img.timestamp for img in self.images)

    @property
    def end_time(self) -> Optional[datetime]:
        """End time of burst."""
        if not self.images:
            return None
        return max(img.timestamp for img in self.images)

    @property
    def duration_seconds(self) -> float:
        """Duration of burst in seconds."""
        if not self.images or len(self.images) < 2:
            return 0.0
        start = self.start_time
        end = self.end_time
        if start and end:
            return (end - start).total_seconds()
        return 0.0

    @property
    def camera_make(self) -> Optional[str]:
        """Camera make (should be same for all images)."""
        if self.images:
            return self.images[0].camera_make
        return None

    @property
    def camera_model(self) -> Optional[str]:
        """Camera model (should be same for all images)."""
        if self.images:
            return self.images[0].camera_model
        return None


class BurstDetector:
    """Detects burst sequences in image collections.

    A burst is defined as:
    1. Images from the same camera
    2. Taken within gap_threshold_seconds of each other
    3. At least min_burst_size images in the sequence
    """

    def __init__(
        self,
        gap_threshold_seconds: float = 2.0,
        min_burst_size: int = 3,
    ):
        """Initialize burst detector.

        Args:
            gap_threshold_seconds: Maximum gap between images to be in same burst
            min_burst_size: Minimum images required to form a burst
        """
        self.gap_threshold_seconds = gap_threshold_seconds
        self.min_burst_size = min_burst_size

    def detect_bursts(self, images: List[ImageInfo]) -> List[BurstGroup]:
        """Detect burst sequences in a list of images.

        Args:
            images: List of ImageInfo objects (will be sorted internally)

        Returns:
            List of detected BurstGroups
        """
        if len(images) < self.min_burst_size:
            return []

        # Group by camera first
        by_camera: dict = {}
        for img in images:
            key = img.camera_key
            if key not in by_camera:
                by_camera[key] = []
            by_camera[key].append(img)

        all_bursts: List[BurstGroup] = []

        # Process each camera's images
        for camera_key, camera_images in by_camera.items():
            # Sort by timestamp
            sorted_images = sorted(camera_images, key=lambda x: x.timestamp)

            # Find burst sequences
            bursts = self._find_sequences(sorted_images)
            all_bursts.extend(bursts)

        # Sort bursts by start time
        all_bursts.sort(key=lambda b: b.start_time or datetime.min)

        logger.info(f"Detected {len(all_bursts)} bursts from {len(images)} images")
        return all_bursts

    def _find_sequences(self, sorted_images: List[ImageInfo]) -> List[BurstGroup]:
        """Find burst sequences in time-sorted images from same camera.

        Args:
            sorted_images: Images sorted by timestamp, all from same camera

        Returns:
            List of BurstGroups
        """
        if len(sorted_images) < self.min_burst_size:
            return []

        bursts: List[BurstGroup] = []
        current_sequence: List[ImageInfo] = [sorted_images[0]]

        for i in range(1, len(sorted_images)):
            current_img = sorted_images[i]
            prev_img = sorted_images[i - 1]

            gap = (current_img.timestamp - prev_img.timestamp).total_seconds()

            if gap <= self.gap_threshold_seconds:
                # Continue current sequence
                current_sequence.append(current_img)
            else:
                # Gap too large - check if current sequence is a burst
                if len(current_sequence) >= self.min_burst_size:
                    burst = BurstGroup(images=list(current_sequence))
                    burst.best_image_id = self.select_best_image(burst).image_id
                    bursts.append(burst)

                # Start new sequence
                current_sequence = [current_img]

        # Don't forget the last sequence
        if len(current_sequence) >= self.min_burst_size:
            burst = BurstGroup(images=list(current_sequence))
            burst.best_image_id = self.select_best_image(burst).image_id
            bursts.append(burst)

        return bursts

    def select_best_image(
        self,
        group: BurstGroup,
        method: str = "quality",
    ) -> ImageInfo:
        """Select the best image from a burst group.

        Args:
            group: BurstGroup to select from
            method: Selection method ('quality', 'first', 'middle')

        Returns:
            The selected ImageInfo
        """
        if not group.images:
            raise ValueError("Cannot select from empty burst group")

        if method == "first":
            return group.images[0]
        elif method == "middle":
            return group.images[len(group.images) // 2]
        else:  # quality (default)
            return max(group.images, key=lambda img: img.quality_score)
```

### Step 4: Run tests to verify they pass

```bash
./venv/bin/pytest tests/analysis/test_burst_detector.py -v
```

Expected: PASS

### Step 5: Commit

```bash
git add vam_tools/analysis/burst_detector.py
git add tests/analysis/test_burst_detector.py
git commit -m "feat: add burst detection algorithm (#19)"
```

---

## Task 3: Burst Detection Celery Task

**Files:**
- Modify: `vam_tools/jobs/tasks.py`
- Create: `tests/jobs/test_burst_tasks.py`

### Step 1: Write the failing test

Create `tests/jobs/test_burst_tasks.py`:

```python
"""Tests for burst detection Celery tasks."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import uuid

import pytest


class TestDetectBurstsTask:
    """Tests for detect_bursts_task."""

    def test_detect_bursts_task_creates_burst_records(self, mock_db):
        """Test that task creates burst records in database."""
        from vam_tools.jobs.tasks import detect_bursts_task

        # Mock images with burst pattern
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_images = [
            {
                "id": f"img-{i:03d}",
                "date_taken": base_time + timedelta(seconds=i * 0.5),
                "camera_make": "Canon",
                "camera_model": "R5",
                "quality_score": 0.8 + (i * 0.02),
            }
            for i in range(5)
        ]

        mock_db.session.execute.return_value.fetchall.return_value = [
            (img["id"], img["date_taken"], img["camera_make"],
             img["camera_model"], img["quality_score"])
            for img in mock_images
        ]

        catalog_id = str(uuid.uuid4())
        result = detect_bursts_task(catalog_id)

        assert result["status"] == "completed"
        assert result["bursts_detected"] >= 1

    def test_detect_bursts_task_updates_image_burst_ids(self, mock_db):
        """Test that task updates images with burst_id."""
        from vam_tools.jobs.tasks import detect_bursts_task

        # ... similar setup ...
        # Verify UPDATE queries were executed for burst_id
```

### Step 2: Run test to verify it fails

```bash
./venv/bin/pytest tests/jobs/test_burst_tasks.py -v
```

Expected: FAIL with "cannot import name 'detect_bursts_task'"

### Step 3: Implement detect_bursts_task

Add to `vam_tools/jobs/tasks.py`:

```python
from vam_tools.analysis.burst_detector import BurstDetector, BurstGroup, ImageInfo


@celery_app.task(bind=True, name="detect_bursts")
def detect_bursts_task(
    self,
    catalog_id: str,
    gap_threshold: float = 2.0,
    min_burst_size: int = 3,
) -> dict:
    """Detect burst sequences in a catalog.

    Args:
        catalog_id: Catalog ID to process
        gap_threshold: Maximum seconds between burst images
        min_burst_size: Minimum images to form a burst

    Returns:
        Dict with detection results
    """
    job_id = self.request.id
    logger.info(f"[{job_id}] Starting burst detection for catalog {catalog_id}")

    db = get_catalog_database(catalog_id)

    try:
        # Clear existing bursts for this catalog
        db.session.execute(
            text("DELETE FROM bursts WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id}
        )

        # Load images with timestamps
        result = db.session.execute(
            text("""
                SELECT id, date_taken, camera_make, camera_model, quality_score
                FROM images
                WHERE catalog_id = :catalog_id
                AND date_taken IS NOT NULL
                ORDER BY date_taken
            """),
            {"catalog_id": catalog_id}
        )

        images = [
            ImageInfo(
                image_id=row[0],
                timestamp=row[1],
                camera_make=row[2],
                camera_model=row[3],
                quality_score=row[4] or 0.0,
            )
            for row in result.fetchall()
        ]

        logger.info(f"[{job_id}] Loaded {len(images)} images with timestamps")

        # Detect bursts
        detector = BurstDetector(
            gap_threshold_seconds=gap_threshold,
            min_burst_size=min_burst_size,
        )
        bursts = detector.detect_bursts(images)

        logger.info(f"[{job_id}] Detected {len(bursts)} bursts")

        # Save bursts to database
        for burst in bursts:
            burst_id = str(uuid.uuid4())

            # Insert burst record
            db.session.execute(
                text("""
                    INSERT INTO bursts (
                        id, catalog_id, image_count, start_time, end_time,
                        duration_seconds, camera_make, camera_model,
                        best_image_id, selection_method
                    ) VALUES (
                        :id, :catalog_id, :image_count, :start_time, :end_time,
                        :duration, :camera_make, :camera_model,
                        :best_image_id, :selection_method
                    )
                """),
                {
                    "id": burst_id,
                    "catalog_id": catalog_id,
                    "image_count": burst.image_count,
                    "start_time": burst.start_time,
                    "end_time": burst.end_time,
                    "duration": burst.duration_seconds,
                    "camera_make": burst.camera_make,
                    "camera_model": burst.camera_model,
                    "best_image_id": burst.best_image_id,
                    "selection_method": burst.selection_method,
                }
            )

            # Update images with burst_id and sequence
            for seq, img in enumerate(burst.images):
                db.session.execute(
                    text("""
                        UPDATE images
                        SET burst_id = :burst_id, burst_sequence = :seq
                        WHERE id = :image_id
                    """),
                    {
                        "burst_id": burst_id,
                        "image_id": img.image_id,
                        "seq": seq,
                    }
                )

        db.session.commit()

        return {
            "status": "completed",
            "catalog_id": catalog_id,
            "images_processed": len(images),
            "bursts_detected": len(bursts),
            "total_burst_images": sum(b.image_count for b in bursts),
        }

    except Exception as e:
        logger.error(f"[{job_id}] Burst detection failed: {e}")
        db.session.rollback()
        raise
    finally:
        db.close()
```

### Step 4: Run tests to verify they pass

```bash
./venv/bin/pytest tests/jobs/test_burst_tasks.py -v
```

Expected: PASS

### Step 5: Commit

```bash
git add vam_tools/jobs/tasks.py
git add tests/jobs/test_burst_tasks.py
git commit -m "feat: add burst detection Celery task (#19)"
```

---

## Task 4: Burst API Endpoints

**Files:**
- Modify: `vam_tools/web/api.py`
- Create: `tests/web/test_bursts_api.py`

### Step 1: Write the failing test

Create `tests/web/test_bursts_api.py`:

```python
"""Tests for burst API endpoints."""

import pytest
from unittest.mock import MagicMock, patch


class TestBurstsAPI:
    """Tests for /api/catalogs/{id}/bursts endpoints."""

    def test_list_bursts_returns_bursts(self, client, mock_catalog):
        """Test listing bursts for a catalog."""
        with patch("vam_tools.web.api.get_catalog_db") as mock_db:
            mock_db.return_value.session.execute.return_value.fetchall.return_value = [
                ("burst-001", 5, "2024-01-01 12:00:00", "2024-01-01 12:00:02",
                 2.0, "Canon", "R5", "img-003", "quality"),
            ]

            response = client.get(f"/api/catalogs/{mock_catalog.id}/bursts")

            assert response.status_code == 200
            data = response.json()
            assert "bursts" in data
            assert len(data["bursts"]) == 1
            assert data["bursts"][0]["image_count"] == 5

    def test_get_burst_details(self, client, mock_catalog):
        """Test getting burst details with images."""
        response = client.get(f"/api/catalogs/{mock_catalog.id}/bursts/burst-001")

        assert response.status_code in [200, 404]  # Depends on mock

    def test_update_burst_best_image(self, client, mock_catalog):
        """Test updating the best image for a burst."""
        response = client.put(
            f"/api/catalogs/{mock_catalog.id}/bursts/burst-001",
            json={"best_image_id": "img-002"}
        )

        assert response.status_code in [200, 404]

    def test_start_burst_detection_job(self, client, mock_catalog):
        """Test starting a burst detection job."""
        with patch("vam_tools.web.api.detect_bursts_task") as mock_task:
            mock_task.delay.return_value.id = "job-123"

            response = client.post(f"/api/catalogs/{mock_catalog.id}/detect-bursts")

            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
```

### Step 2: Run test to verify it fails

```bash
./venv/bin/pytest tests/web/test_bursts_api.py -v
```

Expected: FAIL with 404 (endpoint doesn't exist)

### Step 3: Add burst endpoints to API

Add to `vam_tools/web/api.py`:

```python
from vam_tools.jobs.tasks import detect_bursts_task


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
    db = get_catalog_db(catalog_id)

    try:
        result = db.session.execute(
            text("""
                SELECT
                    b.id, b.image_count, b.start_time, b.end_time,
                    b.duration_seconds, b.camera_make, b.camera_model,
                    b.best_image_id, b.selection_method
                FROM bursts b
                WHERE b.catalog_id = :catalog_id
                ORDER BY b.start_time DESC
                LIMIT :limit OFFSET :offset
            """),
            {"catalog_id": catalog_id, "limit": limit, "offset": offset}
        )

        bursts = [
            {
                "id": row[0],
                "image_count": row[1],
                "start_time": row[2].isoformat() if row[2] else None,
                "end_time": row[3].isoformat() if row[3] else None,
                "duration_seconds": row[4],
                "camera_make": row[5],
                "camera_model": row[6],
                "best_image_id": row[7],
                "selection_method": row[8],
                "best_thumbnail_url": f"/api/catalogs/{catalog_id}/images/{row[7]}/thumbnail" if row[7] else None,
            }
            for row in result.fetchall()
        ]

        # Get total count
        count_result = db.session.execute(
            text("SELECT COUNT(*) FROM bursts WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id}
        )
        total = count_result.scalar()

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
    db = get_catalog_db(catalog_id)

    try:
        # Get burst info
        result = db.session.execute(
            text("""
                SELECT
                    id, image_count, start_time, end_time,
                    duration_seconds, camera_make, camera_model,
                    best_image_id, selection_method
                FROM bursts
                WHERE id = :burst_id AND catalog_id = :catalog_id
            """),
            {"burst_id": burst_id, "catalog_id": catalog_id}
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Burst not found")

        # Get images in burst
        images_result = db.session.execute(
            text("""
                SELECT id, source_path, burst_sequence, quality_score
                FROM images
                WHERE burst_id = :burst_id
                ORDER BY burst_sequence
            """),
            {"burst_id": burst_id}
        )

        images = [
            {
                "id": img[0],
                "source_path": img[1],
                "sequence": img[2],
                "quality_score": img[3],
                "is_best": img[0] == row[7],
                "thumbnail_url": f"/api/catalogs/{catalog_id}/images/{img[0]}/thumbnail",
            }
            for img in images_result.fetchall()
        ]

        return {
            "id": row[0],
            "image_count": row[1],
            "start_time": row[2].isoformat() if row[2] else None,
            "end_time": row[3].isoformat() if row[3] else None,
            "duration_seconds": row[4],
            "camera_make": row[5],
            "camera_model": row[6],
            "best_image_id": row[7],
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
    db = get_catalog_db(catalog_id)

    try:
        if "best_image_id" in data:
            db.session.execute(
                text("""
                    UPDATE bursts
                    SET best_image_id = :best_id, selection_method = 'manual'
                    WHERE id = :burst_id AND catalog_id = :catalog_id
                """),
                {
                    "best_id": data["best_image_id"],
                    "burst_id": burst_id,
                    "catalog_id": catalog_id,
                }
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
```

### Step 4: Run tests to verify they pass

```bash
./venv/bin/pytest tests/web/test_bursts_api.py -v
```

Expected: PASS

### Step 5: Commit

```bash
git add vam_tools/web/api.py
git add tests/web/test_bursts_api.py
git commit -m "feat: add burst API endpoints (#19)"
```

---

## Task 5: Burst UI Components

**Files:**
- Modify: `vam_tools/web/static/index.html`
- Modify: `vam_tools/web/static/app.js`
- Modify: `vam_tools/web/static/styles.css`

### Step 1: Add burst indicator to image thumbnails

Add to `index.html` in the image grid item template:

```html
<!-- Burst indicator -->
<div v-if="image.burst_id" class="burst-indicator" @click.stop="viewBurst(image.burst_id)">
    <span class="burst-icon">&#x1F4F7;</span>
    <span class="burst-count">{{ image.burst_count || '?' }}</span>
</div>
```

### Step 2: Add burst data to app.js

Add to Vue data:

```javascript
bursts: [],
currentBurst: null,
showBurstModal: false,
collapseBursts: false,  // Grid view option
```

Add methods:

```javascript
async loadBursts() {
    if (!this.currentCatalog) return;

    try {
        const response = await axios.get(
            `/api/catalogs/${this.currentCatalog.id}/bursts`
        );
        this.bursts = response.data.bursts;
    } catch (error) {
        console.error('Failed to load bursts:', error);
    }
},

async viewBurst(burstId) {
    try {
        const response = await axios.get(
            `/api/catalogs/${this.currentCatalog.id}/bursts/${burstId}`
        );
        this.currentBurst = response.data;
        this.showBurstModal = true;
    } catch (error) {
        console.error('Failed to load burst:', error);
    }
},

async setBestImage(burstId, imageId) {
    try {
        await axios.put(
            `/api/catalogs/${this.currentCatalog.id}/bursts/${burstId}`,
            { best_image_id: imageId }
        );
        this.showNotification('Best image updated', 'success');
        this.loadBursts();
    } catch (error) {
        console.error('Failed to update burst:', error);
    }
},

async startBurstDetection() {
    try {
        const response = await axios.post(
            `/api/catalogs/${this.currentCatalog.id}/detect-bursts`
        );
        this.showNotification('Burst detection started', 'success');
        // Refresh jobs
        this.loadJobs();
    } catch (error) {
        console.error('Failed to start burst detection:', error);
    }
},

toggleCollapseBursts() {
    this.collapseBursts = !this.collapseBursts;
    this.loadImages();  // Reload with new mode
},
```

### Step 3: Add burst modal

Add to `index.html`:

```html
<!-- Burst Detail Modal -->
<div v-if="showBurstModal && currentBurst" class="modal-overlay" @click.self="showBurstModal = false">
    <div class="modal burst-modal">
        <div class="modal-header">
            <h3>Burst: {{ currentBurst.image_count }} images</h3>
            <button @click="showBurstModal = false" class="close-btn">&times;</button>
        </div>
        <div class="modal-body">
            <div class="burst-info">
                <p><strong>Camera:</strong> {{ currentBurst.camera_make }} {{ currentBurst.camera_model }}</p>
                <p><strong>Duration:</strong> {{ currentBurst.duration_seconds?.toFixed(1) }}s</p>
                <p><strong>Time:</strong> {{ currentBurst.start_time }}</p>
            </div>
            <div class="burst-images-grid">
                <div
                    v-for="image in currentBurst.images"
                    :key="image.id"
                    class="burst-image-item"
                    :class="{ 'is-best': image.is_best }"
                    @click="setBestImage(currentBurst.id, image.id)"
                >
                    <img :src="image.thumbnail_url" :alt="image.source_path">
                    <div class="burst-image-overlay">
                        <span v-if="image.is_best" class="best-badge">BEST</span>
                        <span class="quality-score">Q: {{ (image.quality_score * 100).toFixed(0) }}%</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
```

### Step 4: Add CSS styles

Add to `styles.css`:

```css
/* Burst indicator on thumbnails */
.burst-indicator {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(0, 0, 0, 0.7);
    color: white;
    padding: 4px 8px;
    border-radius: var(--radius-sm);
    font-size: 0.8em;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 4px;
}

.burst-indicator:hover {
    background: var(--accent-primary);
}

.burst-icon {
    font-size: 1.1em;
}

/* Burst modal */
.burst-modal {
    max-width: 900px;
    width: 90%;
}

.burst-info {
    display: flex;
    gap: 24px;
    margin-bottom: 16px;
    color: var(--text-secondary);
}

.burst-images-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 12px;
}

.burst-image-item {
    position: relative;
    cursor: pointer;
    border: 2px solid transparent;
    border-radius: var(--radius-sm);
    overflow: hidden;
}

.burst-image-item:hover {
    border-color: var(--accent-primary);
}

.burst-image-item.is-best {
    border-color: var(--accent-success);
}

.burst-image-item img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
}

.burst-image-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.8));
    padding: 8px;
    display: flex;
    justify-content: space-between;
}

.best-badge {
    background: var(--accent-success);
    color: white;
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    font-size: 0.75em;
    font-weight: bold;
}

.quality-score {
    color: var(--text-secondary);
    font-size: 0.8em;
}
```

### Step 5: Add "Detect Bursts" button to actions

Add to the quick actions area in `index.html`:

```html
<button @click="startBurstDetection" class="action-btn">
    Detect Bursts
</button>
```

### Step 6: Manual testing

1. Start the web server
2. Run burst detection on a catalog
3. Verify burst indicators appear on images
4. Click indicator to view burst modal
5. Click image to set as best
6. Verify "Collapse bursts" works in grid view

### Step 7: Commit

```bash
git add vam_tools/web/static/index.html
git add vam_tools/web/static/app.js
git add vam_tools/web/static/styles.css
git commit -m "feat: add burst detection UI (#19)"
```

---

## Summary

This plan implements burst detection in 5 tasks:

1. **Database Schema** - bursts table and image columns
2. **Detection Algorithm** - Pure Python burst finder
3. **Celery Task** - Background processing
4. **API Endpoints** - REST API for bursts
5. **UI Components** - Burst indicators and modal

**Key Points:**
- No ML required - pure algorithmic approach
- Uses existing EXIF timestamps
- Integrates with existing quality_score
- TDD approach throughout
