# File Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement parallel file reorganization system that organizes 96K+ images into date-based directory structure with status-based separation

**Architecture:** Extends existing OrganizationStrategy with new patterns, adds parallel coordinator/worker/finalizer tasks using existing batch pattern, integrates with API and UI for job management

**Tech Stack:** Python, Celery, PostgreSQL, FastAPI, Vue 3 (CDN), existing coordinator pattern

---

## Task 1: Add New Directory Structure Enum

**Files:**
- Modify: `vam_tools/organization/strategy.py:16-24`
- Test: `tests/organization/test_strategy.py`

**Step 1: Write the failing test**

Create test file if it doesn't exist, or add to existing:

```python
def test_year_slash_month_day_structure():
    """Test YYYY/MM-DD directory structure."""
    from datetime import datetime
    from vam_tools.organization.strategy import DirectoryStructure, OrganizationStrategy
    from vam_tools.core.types import ImageRecord, ImageDates
    from pathlib import Path

    strategy = OrganizationStrategy(
        directory_structure=DirectoryStructure.YEAR_SLASH_MONTH_DAY
    )

    # Create test image with date
    dates = ImageDates(selected_date=datetime(2023, 6, 15, 14, 30, 22))
    image = ImageRecord(
        id="test123",
        source_path=Path("/source/IMG_1234.jpg"),
        checksum="abc123",
        dates=dates
    )

    base_path = Path("/organized")
    target_dir = strategy.get_target_directory(base_path, image)

    assert target_dir == Path("/organized/2023/06-15")
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/organization/test_strategy.py::test_year_slash_month_day_structure -v`

Expected: FAIL with "YEAR_SLASH_MONTH_DAY" not found in DirectoryStructure enum

**Step 3: Add enum value**

In `vam_tools/organization/strategy.py`, add to DirectoryStructure enum:

```python
class DirectoryStructure(str, Enum):
    """Directory structure patterns for organizing files."""

    YEAR_MONTH = "YYYY-MM"  # 2023-06
    YEAR_SLASH_MONTH = "YYYY/MM"  # 2023/06
    YEAR_MONTH_DAY = "YYYY-MM-DD"  # 2023-06-15
    YEAR_ONLY = "YYYY"  # 2023
    YEAR_SLASH_MONTH_DAY = "YYYY/MM-DD"  # NEW: 2023/06-15
    FLAT = "FLAT"  # All files in one directory
```

**Step 4: Add path generation logic**

In `get_target_directory()` method, add new case:

```python
def get_target_directory(
    self, base_path: Path, image: ImageRecord
) -> Optional[Path]:
    """Get target directory for an image based on the strategy."""
    if not image.dates or not image.dates.selected_date:
        return None

    date = image.dates.selected_date

    if self.directory_structure == DirectoryStructure.YEAR_MONTH:
        return base_path / date.strftime("%Y-%m")
    elif self.directory_structure == DirectoryStructure.YEAR_SLASH_MONTH:
        return base_path / date.strftime("%Y") / date.strftime("%m")
    elif self.directory_structure == DirectoryStructure.YEAR_MONTH_DAY:
        return base_path / date.strftime("%Y-%m-%d")
    elif self.directory_structure == DirectoryStructure.YEAR_ONLY:
        return base_path / date.strftime("%Y")
    elif self.directory_structure == DirectoryStructure.YEAR_SLASH_MONTH_DAY:
        return base_path / date.strftime("%Y") / date.strftime("%m-%d")
    elif self.directory_structure == DirectoryStructure.FLAT:
        return base_path

    return base_path
```

**Step 5: Run test to verify it passes**

Run: `./venv/bin/pytest tests/organization/test_strategy.py::test_year_slash_month_day_structure -v`

Expected: PASS

**Step 6: Commit**

```bash
git add vam_tools/organization/strategy.py tests/organization/test_strategy.py
git commit -m "feat: add YEAR_SLASH_MONTH_DAY directory structure"
```

---

## Task 2: Add TIME_CHECKSUM Naming Strategy

**Files:**
- Modify: `vam_tools/organization/strategy.py:26-33`
- Test: `tests/organization/test_strategy.py`

**Step 1: Write the failing test**

```python
def test_time_checksum_naming():
    """Test TIME_CHECKSUM naming strategy: HHMMSS_shortchecksum.ext"""
    from datetime import datetime
    from vam_tools.organization.strategy import NamingStrategy, OrganizationStrategy
    from vam_tools.core.types import ImageRecord, ImageDates
    from pathlib import Path

    strategy = OrganizationStrategy(
        naming_strategy=NamingStrategy.TIME_CHECKSUM
    )

    dates = ImageDates(selected_date=datetime(2023, 6, 15, 14, 30, 22))
    image = ImageRecord(
        id="test123",
        source_path=Path("/source/IMG_1234.jpg"),
        checksum="abc123def456",
        dates=dates
    )

    filename = strategy.get_target_filename(image)

    assert filename == "143022_abc123de.jpg"
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/organization/test_strategy.py::test_time_checksum_naming -v`

Expected: FAIL with "TIME_CHECKSUM" not found in NamingStrategy enum

**Step 3: Add enum value**

In `vam_tools/organization/strategy.py`, add to NamingStrategy enum:

```python
class NamingStrategy(str, Enum):
    """File naming strategies."""

    DATE_TIME_CHECKSUM = "date_time_checksum"  # 2023-06-15_143022_abc123.jpg
    DATE_TIME_ORIGINAL = "date_time_original"  # 2023-06-15_143022_IMG_1234.jpg
    ORIGINAL = "original"  # IMG_1234.jpg (keep original name)
    CHECKSUM = "checksum"  # abc123def456.jpg
    TIME_CHECKSUM = "time_checksum"  # NEW: 143022_abc12345.jpg
```

**Step 4: Add filename generation logic**

In `get_target_filename()` method, add new case:

```python
def get_target_filename(self, image: ImageRecord) -> str:
    """Get target filename for an image based on the strategy."""
    original_name = image.source_path.name
    stem = image.source_path.stem
    suffix = image.source_path.suffix

    if self.naming_strategy == NamingStrategy.ORIGINAL:
        return original_name

    elif self.naming_strategy == NamingStrategy.CHECKSUM:
        return f"{image.checksum}{suffix}"

    elif self.naming_strategy == NamingStrategy.TIME_CHECKSUM:
        if image.dates and image.dates.selected_date:
            time_str = image.dates.selected_date.strftime("%H%M%S")
            checksum_short = image.checksum[:8]
            return f"{time_str}_{checksum_short}{suffix}"
        else:
            # Fall back to checksum if no date
            return f"{image.checksum}{suffix}"

    elif self.naming_strategy == NamingStrategy.DATE_TIME_CHECKSUM:
        if image.dates and image.dates.selected_date:
            date_str = image.dates.selected_date.strftime("%Y-%m-%d_%H%M%S")
            checksum_short = image.checksum[:8]
            return f"{date_str}_{checksum_short}{suffix}"
        else:
            return f"{image.checksum}{suffix}"

    elif self.naming_strategy == NamingStrategy.DATE_TIME_ORIGINAL:
        if image.dates and image.dates.selected_date:
            date_str = image.dates.selected_date.strftime("%Y-%m-%d_%H%M%S")
            return f"{date_str}_{stem}{suffix}"
        else:
            return original_name

    return original_name
```

**Step 5: Run test to verify it passes**

Run: `./venv/bin/pytest tests/organization/test_strategy.py::test_time_checksum_naming -v`

Expected: PASS

**Step 6: Commit**

```bash
git add vam_tools/organization/strategy.py tests/organization/test_strategy.py
git commit -m "feat: add TIME_CHECKSUM naming strategy"
```

---

## Task 3: Add Status-Based Path Routing

**Files:**
- Modify: `vam_tools/organization/strategy.py:132-148`
- Test: `tests/organization/test_strategy.py`

**Step 1: Write the failing test**

```python
def test_status_based_routing_rejected():
    """Test that rejected images route to _rejected/ subdirectory."""
    from datetime import datetime
    from vam_tools.organization.strategy import OrganizationStrategy, DirectoryStructure, NamingStrategy
    from vam_tools.core.types import ImageRecord, ImageDates
    from pathlib import Path

    strategy = OrganizationStrategy(
        directory_structure=DirectoryStructure.YEAR_SLASH_MONTH_DAY,
        naming_strategy=NamingStrategy.TIME_CHECKSUM
    )

    dates = ImageDates(selected_date=datetime(2023, 6, 15, 14, 30, 22))
    image = ImageRecord(
        id="test123",
        source_path=Path("/source/IMG_1234.jpg"),
        checksum="abc123def456",
        dates=dates,
        status_id="rejected"  # Rejected image
    )

    base_path = Path("/organized")
    target_path = strategy.get_target_path(base_path, image)

    # Should route to _rejected/ subdirectory
    assert str(target_path) == "/organized/_rejected/2023/06-15/143022_abc123de.jpg"


def test_status_based_routing_active():
    """Test that active images route to main directory."""
    from datetime import datetime
    from vam_tools.organization.strategy import OrganizationStrategy, DirectoryStructure, NamingStrategy
    from vam_tools.core.types import ImageRecord, ImageDates
    from pathlib import Path

    strategy = OrganizationStrategy(
        directory_structure=DirectoryStructure.YEAR_SLASH_MONTH_DAY,
        naming_strategy=NamingStrategy.TIME_CHECKSUM
    )

    dates = ImageDates(selected_date=datetime(2023, 6, 15, 14, 30, 22))
    image = ImageRecord(
        id="test123",
        source_path=Path("/source/IMG_1234.jpg"),
        checksum="abc123def456",
        dates=dates,
        status_id="active"  # Active image
    )

    base_path = Path("/organized")
    target_path = strategy.get_target_path(base_path, image)

    # Should route to main directory
    assert str(target_path) == "/organized/2023/06-15/143022_abc123de.jpg"
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/organization/test_strategy.py::test_status_based_routing_rejected -v`

Expected: FAIL - ImageRecord doesn't have status_id field, or path doesn't include _rejected/

**Step 3: Check if ImageRecord has status_id field**

Check `vam_tools/core/types.py` to see if ImageRecord has status_id. If not, skip this task and add status_id to the model first.

If it exists, modify `get_target_path()` to check status:

```python
def get_target_path(self, base_path: Path, image: ImageRecord) -> Optional[Path]:
    """Get complete target path for an image.

    Routes rejected images to _rejected/ subdirectory.
    """
    # Adjust base path for rejected images
    if hasattr(image, 'status_id') and image.status_id == 'rejected':
        base_path = base_path / "_rejected"

    target_dir = self.get_target_directory(base_path, image)
    if not target_dir:
        return None

    filename = self.get_target_filename(image)
    return target_dir / filename
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/organization/test_strategy.py::test_status_based_routing_rejected -v`
Run: `./venv/bin/pytest tests/organization/test_strategy.py::test_status_based_routing_active -v`

Expected: PASS on both

**Step 5: Commit**

```bash
git add vam_tools/organization/strategy.py tests/organization/test_strategy.py
git commit -m "feat: add status-based routing for rejected images"
```

---

## Task 4: Add mtime Fallback for Missing EXIF Dates

**Files:**
- Modify: `vam_tools/organization/strategy.py:60-90`
- Test: `tests/organization/test_strategy.py`

**Step 1: Write the failing test**

```python
def test_mtime_fallback():
    """Test that mtime is used when EXIF date is missing."""
    import os
    from datetime import datetime
    from vam_tools.organization.strategy import OrganizationStrategy, DirectoryStructure
    from vam_tools.core.types import ImageRecord
    from pathlib import Path
    import tempfile

    strategy = OrganizationStrategy(
        directory_structure=DirectoryStructure.YEAR_SLASH_MONTH_DAY
    )

    # Create temp file with known mtime
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = Path(f.name)

    # Set mtime to specific timestamp (June 15, 2023 14:30:22)
    target_time = datetime(2023, 6, 15, 14, 30, 22).timestamp()
    os.utime(temp_path, (target_time, target_time))

    try:
        # Create image without dates (EXIF missing)
        image = ImageRecord(
            id="test123",
            source_path=temp_path,
            checksum="abc123",
            dates=None  # No EXIF date
        )

        base_path = Path("/organized")
        target_dir = strategy.get_target_directory(base_path, image, use_mtime_fallback=True)

        # Should use mtime and create 2023/06-15 directory
        assert target_dir == Path("/organized/2023/06-15")
    finally:
        temp_path.unlink()
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/organization/test_strategy.py::test_mtime_fallback -v`

Expected: FAIL - get_target_directory returns None for images without dates, or use_mtime_fallback parameter doesn't exist

**Step 3: Add mtime fallback parameter**

Modify `get_target_directory()` signature and logic:

```python
def get_target_directory(
    self,
    base_path: Path,
    image: ImageRecord,
    use_mtime_fallback: bool = False
) -> Optional[Path]:
    """Get target directory for an image based on the strategy.

    Args:
        base_path: Base output directory
        image: Image record with metadata
        use_mtime_fallback: If True, fall back to file mtime when EXIF date missing

    Returns:
        Target directory path, or None if image has no date and fallback disabled
    """
    import os
    from datetime import datetime

    # Try to get date from EXIF
    if image.dates and image.dates.selected_date:
        date = image.dates.selected_date
    elif use_mtime_fallback and image.source_path.exists():
        # Fall back to file modification time
        mtime = os.path.getmtime(image.source_path)
        date = datetime.fromtimestamp(mtime)
    else:
        return None

    if self.directory_structure == DirectoryStructure.YEAR_MONTH:
        return base_path / date.strftime("%Y-%m")
    elif self.directory_structure == DirectoryStructure.YEAR_SLASH_MONTH:
        return base_path / date.strftime("%Y") / date.strftime("%m")
    elif self.directory_structure == DirectoryStructure.YEAR_MONTH_DAY:
        return base_path / date.strftime("%Y-%m-%d")
    elif self.directory_structure == DirectoryStructure.YEAR_ONLY:
        return base_path / date.strftime("%Y")
    elif self.directory_structure == DirectoryStructure.YEAR_SLASH_MONTH_DAY:
        return base_path / date.strftime("%Y") / date.strftime("%m-%d")
    elif self.directory_structure == DirectoryStructure.FLAT:
        return base_path

    return base_path
```

**Step 4: Update get_target_path to pass through parameter**

```python
def get_target_path(
    self,
    base_path: Path,
    image: ImageRecord,
    use_mtime_fallback: bool = False
) -> Optional[Path]:
    """Get complete target path for an image."""
    if hasattr(image, 'status_id') and image.status_id == 'rejected':
        base_path = base_path / "_rejected"

    target_dir = self.get_target_directory(base_path, image, use_mtime_fallback)
    if not target_dir:
        return None

    filename = self.get_target_filename(image)
    return target_dir / filename
```

**Step 5: Run test to verify it passes**

Run: `./venv/bin/pytest tests/organization/test_strategy.py::test_mtime_fallback -v`

Expected: PASS

**Step 6: Commit**

```bash
git add vam_tools/organization/strategy.py tests/organization/test_strategy.py
git commit -m "feat: add mtime fallback for images without EXIF dates"
```

---

## Task 5: Add Full Checksum Conflict Resolution

**Files:**
- Modify: `vam_tools/organization/strategy.py:150-179`
- Test: `tests/organization/test_strategy.py`

**Step 1: Write the failing test**

```python
def test_full_checksum_conflict_resolution():
    """Test that conflicts are resolved using full checksum."""
    from datetime import datetime
    from vam_tools.organization.strategy import OrganizationStrategy, NamingStrategy
    from vam_tools.core.types import ImageRecord, ImageDates
    from pathlib import Path
    import tempfile
    import os

    strategy = OrganizationStrategy(
        naming_strategy=NamingStrategy.TIME_CHECKSUM
    )

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Create first image at specific time
        dates = ImageDates(selected_date=datetime(2023, 6, 15, 14, 30, 22))
        image1 = ImageRecord(
            id="test123",
            source_path=Path("/source/IMG_1234.jpg"),
            checksum="abc123def456",
            dates=dates
        )

        # Create conflicting image - same time, different checksum
        image2 = ImageRecord(
            id="test456",
            source_path=Path("/source/IMG_5678.jpg"),
            checksum="xyz789fedcba",
            dates=dates
        )

        # Create first file to simulate conflict
        target_dir = base_path / "2023" / "06-15"
        target_dir.mkdir(parents=True, exist_ok=True)
        existing_file = target_dir / "143022_abc123de.jpg"
        existing_file.write_text("existing")

        # Resolve conflict for second image
        resolved_path = strategy.resolve_conflict_with_full_checksum(
            base_path, image2, existing_file
        )

        # Should use full checksum
        assert resolved_path.name == "143022_xyz789fedcba.jpg"
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/organization/test_strategy.py::test_full_checksum_conflict_resolution -v`

Expected: FAIL - resolve_conflict_with_full_checksum method doesn't exist

**Step 3: Add conflict resolution method**

Add new method to OrganizationStrategy class:

```python
def resolve_conflict_with_full_checksum(
    self,
    base_path: Path,
    image: ImageRecord,
    conflicting_path: Path
) -> Path:
    """Resolve naming conflict by using full checksum instead of short checksum.

    Args:
        base_path: Base output directory
        image: Image record
        conflicting_path: Path that already exists

    Returns:
        New path with full checksum
    """
    # Get directory and extension
    target_dir = conflicting_path.parent
    suffix = image.source_path.suffix

    # Generate filename with full checksum
    if image.dates and image.dates.selected_date:
        time_str = image.dates.selected_date.strftime("%H%M%S")
        return target_dir / f"{time_str}_{image.checksum}{suffix}"
    else:
        # No date - use checksum only
        return target_dir / f"{image.checksum}{suffix}"
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/organization/test_strategy.py::test_full_checksum_conflict_resolution -v`

Expected: PASS

**Step 5: Commit**

```bash
git add vam_tools/organization/strategy.py tests/organization/test_strategy.py
git commit -m "feat: add full checksum conflict resolution"
```

---

## Task 6: Add Idempotent Skip Logic

**Files:**
- Create: `vam_tools/organization/reorganizer.py`
- Test: `tests/organization/test_reorganizer.py`

**Step 1: Write the failing test**

Create new test file:

```python
def test_skip_already_organized():
    """Test that files already in organized structure are skipped."""
    from pathlib import Path
    from vam_tools.organization.reorganizer import should_reorganize_image
    from vam_tools.core.types import ImageRecord

    output_dir = Path("/organized")

    # Image already in organized structure
    image = ImageRecord(
        id="test123",
        source_path=Path("/organized/2023/06-15/143022_abc123de.jpg"),
        checksum="abc123def456"
    )

    assert should_reorganize_image(image, output_dir) is False


def test_skip_matching_checksum():
    """Test that files with matching checksum at target are skipped."""
    from pathlib import Path
    from vam_tools.organization.reorganizer import should_reorganize_image
    from vam_tools.core.types import ImageRecord
    from datetime import datetime
    import tempfile

    # Create temp directory with existing file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create target file
        target_dir = output_dir / "2023" / "06-15"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "143022_abc123de.jpg"
        target_file.write_bytes(b"test content")

        # Calculate checksum of target
        from vam_tools.shared.media_utils import compute_checksum
        target_checksum = compute_checksum(target_file)

        # Image with same checksum
        image = ImageRecord(
            id="test123",
            source_path=Path("/source/IMG_1234.jpg"),
            checksum=target_checksum
        )

        # Should skip
        assert should_reorganize_image(image, output_dir, target_file) is False
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/organization/test_reorganizer.py -v`

Expected: FAIL - module or function doesn't exist

**Step 3: Create reorganizer module**

Create `vam_tools/organization/reorganizer.py`:

```python
"""Helper functions for file reorganization."""

import logging
from pathlib import Path
from typing import Optional

from ..core.types import ImageRecord
from ..shared.media_utils import compute_checksum

logger = logging.getLogger(__name__)


def should_reorganize_image(
    image: ImageRecord,
    output_directory: Path,
    target_path: Optional[Path] = None
) -> bool:
    """Determine if an image should be reorganized.

    Args:
        image: Image record
        output_directory: Target organization directory
        target_path: Calculated target path (optional, for checksum check)

    Returns:
        True if image should be reorganized, False if should be skipped
    """
    # Skip if source_path already in organized structure
    if str(image.source_path).startswith(str(output_directory)):
        logger.debug(f"Skipping {image.id}: already in organized structure")
        return False

    # Skip if target exists with matching checksum
    if target_path and target_path.exists():
        target_checksum = compute_checksum(target_path)
        if target_checksum == image.checksum:
            logger.debug(f"Skipping {image.id}: already organized (checksum match)")
            return False

    return True
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/organization/test_reorganizer.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add vam_tools/organization/reorganizer.py tests/organization/test_reorganizer.py
git commit -m "feat: add idempotent skip logic for reorganization"
```

---

## Task 7: Create Reorganization Coordinator Task

**Files:**
- Create: `vam_tools/jobs/reorganize.py`
- Test: `tests/jobs/test_reorganize.py`

**Step 1: Write the failing test**

Create test file:

```python
def test_reorganize_coordinator_creates_batches():
    """Test that coordinator creates batches for images."""
    from vam_tools.jobs.reorganize import reorganize_coordinator_task
    from vam_tools.db import get_db_context
    from unittest.mock import patch, MagicMock

    catalog_id = "test-catalog-id"
    output_directory = "/organized"

    # Mock the task to run synchronously
    with patch('vam_tools.jobs.reorganize.chord') as mock_chord:
        with patch('vam_tools.jobs.reorganize.group') as mock_group:
            # Mock database to return 1000 images
            with patch('vam_tools.jobs.reorganize.CatalogDatabase') as mock_db:
                mock_db_instance = MagicMock()
                mock_db.return_value.__enter__.return_value = mock_db_instance

                # Mock query to return 1000 image IDs
                mock_result = MagicMock()
                mock_result.fetchall.return_value = [(f"id{i}",) for i in range(1000)]
                mock_db_instance.session.execute.return_value = mock_result

                # Run coordinator
                result = reorganize_coordinator_task(
                    catalog_id=catalog_id,
                    output_directory=output_directory,
                    operation="copy",
                    dry_run=False
                )

                # Should create 2 batches (500 images each)
                assert result["status"] == "dispatched"
                assert result["total_images"] == 1000
                assert result["total_batches"] == 2
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/jobs/test_reorganize.py::test_reorganize_coordinator_creates_batches -v`

Expected: FAIL - module doesn't exist

**Step 3: Create coordinator task**

Create `vam_tools/jobs/reorganize.py`:

```python
"""Reorganization job using coordinator pattern."""

import logging
from typing import Any, Dict

from celery import chord, group
from sqlalchemy import text

from ..db import CatalogDB as CatalogDatabase
from ..db.models import Job
from .celery_app import app
from .coordinator import BatchManager
from .progress_publisher import publish_completion, publish_progress
from .tasks import ProgressTask

logger = logging.getLogger(__name__)


@app.task(bind=True, base=ProgressTask, name="reorganize_coordinator")
def reorganize_coordinator_task(
    self: ProgressTask,
    catalog_id: str,
    output_directory: str,
    operation: str = "copy",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Coordinator task for file reorganization.

    Args:
        catalog_id: UUID of catalog to reorganize
        output_directory: Target directory for organized files
        operation: "copy" or "move"
        dry_run: If True, preview without executing

    Returns:
        Status and batch information
    """
    parent_job_id = self.request.id or "unknown"
    logger.info(f"[{parent_job_id}] Starting reorganization for catalog {catalog_id}")

    try:
        self.update_progress(0, 1, "Querying images...", {"phase": "init"})

        # Get all images from catalog
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None
            result = db.session.execute(
                text("""
                    SELECT id FROM images
                    WHERE catalog_id = :catalog_id
                """),
                {"catalog_id": catalog_id}
            )
            image_ids = [row[0] for row in result.fetchall()]

        total_images = len(image_ids)
        logger.info(f"[{parent_job_id}] Found {total_images} images to reorganize")

        if total_images == 0:
            publish_completion(
                parent_job_id,
                "SUCCESS",
                result={"status": "completed", "message": "No images in catalog"}
            )
            return {"status": "completed", "message": "No images in catalog"}

        # Create batches
        self.update_progress(
            0,
            total_images,
            f"Creating batches for {total_images} images...",
            {"phase": "batching"}
        )

        batch_manager = BatchManager(catalog_id, parent_job_id, "reorganize")
        batch_size = 500

        with CatalogDatabase(catalog_id) as db:
            batch_ids = batch_manager.create_batches(
                work_items=[(img_id,) for img_id in image_ids],
                batch_size=batch_size,
                db=db
            )

        num_batches = len(batch_ids)
        logger.info(f"[{parent_job_id}] Created {num_batches} batches")

        # Spawn worker tasks
        self.update_progress(
            0,
            total_images,
            f"Spawning {num_batches} worker tasks...",
            {"phase": "spawning"}
        )

        worker_tasks = group(
            reorganize_worker_task.s(
                catalog_id=catalog_id,
                batch_id=batch_id,
                parent_job_id=parent_job_id,
                output_directory=output_directory,
                operation=operation,
                dry_run=dry_run
            )
            for batch_id in batch_ids
        )

        # Finalizer collects results
        finalizer = reorganize_finalizer_task.s(
            catalog_id=catalog_id,
            parent_job_id=parent_job_id,
            output_directory=output_directory
        )

        chord(worker_tasks)(finalizer)

        logger.info(f"[{parent_job_id}] Dispatched {num_batches} workers → finalizer")

        # Update job to STARTED
        from ..db import get_db_context
        with get_db_context() as session:
            job = session.query(Job).filter(Job.id == parent_job_id).first()
            if job:
                job.status = "STARTED"
                job.result = {
                    "status": "processing",
                    "total_images": total_images,
                    "message": f"Processing {total_images} images"
                }
                session.commit()

        publish_progress(
            parent_job_id,
            "PROGRESS",
            current=0,
            total=total_images,
            message=f"Processing {total_images} images",
            extra={"phase": "reorganizing"}
        )

        return {
            "status": "dispatched",
            "total_images": total_images,
            "total_batches": num_batches,
            "output_directory": output_directory
        }

    except Exception as e:
        logger.error(f"[{parent_job_id}] Coordinator failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        raise
```

**Step 4: Add worker and finalizer task stubs**

Add temporary stubs so coordinator doesn't fail:

```python
@app.task(bind=True, base=ProgressTask, name="reorganize_worker")
def reorganize_worker_task(
    self: ProgressTask,
    catalog_id: str,
    batch_id: str,
    parent_job_id: str,
    output_directory: str,
    operation: str,
    dry_run: bool
) -> Dict[str, Any]:
    """Worker task - processes one batch of images."""
    # TODO: Implement in next task
    return {"status": "completed", "batch_id": batch_id}


@app.task(bind=True, base=ProgressTask, name="reorganize_finalizer")
def reorganize_finalizer_task(
    self: ProgressTask,
    worker_results: list,
    catalog_id: str,
    parent_job_id: str,
    output_directory: str
) -> Dict[str, Any]:
    """Finalizer task - aggregates results."""
    # TODO: Implement in Task 9
    return {"status": "completed"}
```

**Step 5: Run test to verify it passes**

Run: `./venv/bin/pytest tests/jobs/test_reorganize.py::test_reorganize_coordinator_creates_batches -v`

Expected: PASS

**Step 6: Commit**

```bash
git add vam_tools/jobs/reorganize.py tests/jobs/test_reorganize.py
git commit -m "feat: add reorganization coordinator task"
```

---

## Task 8: Implement Reorganization Worker Task

**Files:**
- Modify: `vam_tools/jobs/reorganize.py` (worker task)
- Test: `tests/jobs/test_reorganize.py`

**Step 1: Write the failing test**

Add to test file:

```python
def test_reorganize_worker_processes_batch():
    """Test that worker processes a batch of images."""
    from vam_tools.jobs.reorganize import reorganize_worker_task
    from unittest.mock import patch, MagicMock
    import tempfile
    from pathlib import Path

    catalog_id = "test-catalog-id"
    batch_id = "batch-1"
    parent_job_id = "job-123"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_directory = tmpdir

        # Mock batch manager to return test images
        with patch('vam_tools.jobs.reorganize.BatchManager') as mock_batch_mgr:
            mock_mgr_instance = MagicMock()
            mock_batch_mgr.return_value = mock_mgr_instance

            # Mock batch with 3 test images
            mock_mgr_instance.get_batch_work_items.return_value = [
                ("img1",), ("img2",), ("img3",)
            ]

            # Mock database to return image records
            with patch('vam_tools.jobs.reorganize.CatalogDatabase') as mock_db:
                # TODO: Mock image loading

                # Run worker
                result = reorganize_worker_task(
                    catalog_id=catalog_id,
                    batch_id=batch_id,
                    parent_job_id=parent_job_id,
                    output_directory=output_directory,
                    operation="copy",
                    dry_run=True
                )

                assert result["status"] == "completed"
                assert result["batch_id"] == batch_id
                # In dry run, nothing actually copied
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/jobs/test_reorganize.py::test_reorganize_worker_processes_batch -v`

Expected: FAIL - worker returns stub result, not actually processing

**Step 3: Implement worker task**

Replace stub worker implementation:

```python
@app.task(bind=True, base=ProgressTask, name="reorganize_worker")
def reorganize_worker_task(
    self: ProgressTask,
    catalog_id: str,
    batch_id: str,
    parent_job_id: str,
    output_directory: str,
    operation: str,
    dry_run: bool
) -> Dict[str, Any]:
    """Worker task - processes one batch of images.

    Args:
        catalog_id: UUID of catalog
        batch_id: Batch ID from batch manager
        parent_job_id: Parent job ID
        output_directory: Target directory
        operation: "copy" or "move"
        dry_run: If True, preview without executing

    Returns:
        Batch processing results
    """
    import os
    import shutil
    from datetime import datetime
    from pathlib import Path

    from ..organization.strategy import OrganizationStrategy, DirectoryStructure, NamingStrategy
    from ..organization.reorganizer import should_reorganize_image
    from ..shared.media_utils import compute_checksum

    worker_id = self.request.id or "unknown"
    logger.info(f"[{worker_id}] Processing batch {batch_id}")

    try:
        # Load batch
        batch_manager = BatchManager(catalog_id, parent_job_id, "reorganize")
        work_items = batch_manager.get_batch_work_items(batch_id)
        image_ids = [item[0] for item in work_items]

        # Create strategy
        strategy = OrganizationStrategy(
            directory_structure=DirectoryStructure.YEAR_SLASH_MONTH_DAY,
            naming_strategy=NamingStrategy.TIME_CHECKSUM
        )

        output_path = Path(output_directory)

        # Counters
        organized = 0
        skipped = 0
        failed = 0
        mtime_fallback_count = 0
        errors = []

        # Load images from database
        with CatalogDatabase(catalog_id) as db:
            assert db.session is not None

            for image_id in image_ids:
                try:
                    # Load image
                    result = db.session.execute(
                        text("""
                            SELECT id, source_path, checksum, status_id, dates, metadata
                            FROM images
                            WHERE id = :image_id
                        """),
                        {"image_id": image_id}
                    )
                    row = result.fetchone()
                    if not row:
                        logger.warning(f"Image {image_id} not found")
                        skipped += 1
                        continue

                    # Build ImageRecord
                    from ..core.types import ImageRecord, ImageDates

                    # Parse dates
                    dates_dict = row.dates if row.dates else {}
                    selected_date = None
                    if dates_dict and 'selected_date' in dates_dict:
                        from dateutil.parser import parse
                        selected_date = parse(dates_dict['selected_date'])

                    dates = ImageDates(selected_date=selected_date) if selected_date else None

                    image = ImageRecord(
                        id=row.id,
                        source_path=Path(row.source_path),
                        checksum=row.checksum,
                        status_id=row.status_id or 'active',
                        dates=dates
                    )

                    # Check if should reorganize
                    if not should_reorganize_image(image, output_path):
                        skipped += 1
                        continue

                    # Get target path (with mtime fallback)
                    used_mtime = False
                    if not image.dates:
                        used_mtime = True
                        mtime_fallback_count += 1

                    target_path = strategy.get_target_path(
                        output_path,
                        image,
                        use_mtime_fallback=True
                    )

                    if not target_path:
                        logger.warning(f"Could not determine target path for {image_id}")
                        skipped += 1
                        continue

                    # Check for conflict
                    if target_path.exists():
                        target_checksum = compute_checksum(target_path)
                        if target_checksum == image.checksum:
                            # Already organized
                            skipped += 1
                            continue
                        else:
                            # Conflict - use full checksum
                            target_path = strategy.resolve_conflict_with_full_checksum(
                                output_path,
                                image,
                                target_path
                            )

                    # Execute operation
                    if dry_run:
                        logger.info(f"[DRY RUN] Would {operation} {image.source_path} → {target_path}")
                        organized += 1
                    else:
                        # Create target directory
                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        # Copy or move
                        if operation == "copy":
                            shutil.copy2(image.source_path, target_path)
                        elif operation == "move":
                            shutil.move(str(image.source_path), str(target_path))

                        # Verify checksum
                        new_checksum = compute_checksum(target_path)
                        if new_checksum != image.checksum:
                            target_path.unlink()
                            raise ValueError(f"Checksum mismatch: expected {image.checksum}, got {new_checksum}")

                        # Update database
                        db.session.execute(
                            text("""
                                UPDATE images
                                SET source_path = :new_path
                                WHERE id = :image_id
                                  AND source_path != :new_path
                            """),
                            {"image_id": image_id, "new_path": str(target_path)}
                        )

                        organized += 1
                        logger.info(f"Reorganized {image.source_path} → {target_path}")

                except Exception as e:
                    logger.error(f"Error processing {image_id}: {e}")
                    failed += 1
                    errors.append(f"{image_id}: {str(e)}")

            # Commit database updates
            if not dry_run:
                db.session.commit()

        # Update batch status
        batch_manager.update_batch_status(batch_id, "SUCCESS")

        logger.info(f"[{worker_id}] Batch {batch_id} complete: {organized} organized, {skipped} skipped, {failed} failed")

        return {
            "status": "completed",
            "batch_id": batch_id,
            "organized": organized,
            "skipped": skipped,
            "failed": failed,
            "mtime_fallback_count": mtime_fallback_count,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"[{worker_id}] Batch {batch_id} failed: {e}", exc_info=True)
        batch_manager.update_batch_status(batch_id, "FAILED", str(e))
        raise
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/jobs/test_reorganize.py::test_reorganize_worker_processes_batch -v`

Expected: PASS

**Step 5: Commit**

```bash
git add vam_tools/jobs/reorganize.py tests/jobs/test_reorganize.py
git commit -m "feat: implement reorganization worker task"
```

---

## Task 9: Implement Reorganization Finalizer Task

**Files:**
- Modify: `vam_tools/jobs/reorganize.py` (finalizer task)
- Test: `tests/jobs/test_reorganize.py`

**Step 1: Write the failing test**

Add to test file:

```python
def test_reorganize_finalizer_aggregates_results():
    """Test that finalizer aggregates worker results."""
    from vam_tools.jobs.reorganize import reorganize_finalizer_task
    from unittest.mock import patch

    catalog_id = "test-catalog-id"
    parent_job_id = "job-123"
    output_directory = "/organized"

    # Mock worker results
    worker_results = [
        {
            "status": "completed",
            "batch_id": "batch-1",
            "organized": 450,
            "skipped": 48,
            "failed": 2,
            "mtime_fallback_count": 5,
            "errors": ["img1: error"]
        },
        {
            "status": "completed",
            "batch_id": "batch-2",
            "organized": 490,
            "skipped": 10,
            "failed": 0,
            "mtime_fallback_count": 3,
            "errors": []
        }
    ]

    with patch('vam_tools.jobs.reorganize.get_db_context'):
        result = reorganize_finalizer_task(
            worker_results=worker_results,
            catalog_id=catalog_id,
            parent_job_id=parent_job_id,
            output_directory=output_directory
        )

        assert result["status"] == "completed"
        assert result["total_organized"] == 940
        assert result["total_skipped"] == 58
        assert result["total_failed"] == 2
        assert result["mtime_fallback_count"] == 8
        assert len(result["errors"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/jobs/test_reorganize.py::test_reorganize_finalizer_aggregates_results -v`

Expected: FAIL - finalizer returns stub result

**Step 3: Implement finalizer task**

Replace stub finalizer implementation:

```python
@app.task(bind=True, base=ProgressTask, name="reorganize_finalizer")
def reorganize_finalizer_task(
    self: ProgressTask,
    worker_results: list,
    catalog_id: str,
    parent_job_id: str,
    output_directory: str
) -> Dict[str, Any]:
    """Finalizer task - aggregates results from all workers.

    Args:
        worker_results: List of results from worker tasks
        catalog_id: UUID of catalog
        parent_job_id: Parent job ID
        output_directory: Target directory

    Returns:
        Aggregated results
    """
    import json
    from datetime import datetime
    from pathlib import Path

    finalizer_id = self.request.id or "unknown"
    logger.info(f"[{finalizer_id}] Starting finalizer for job {parent_job_id}")

    try:
        # Aggregate statistics
        total_organized = sum(r.get('organized', 0) for r in worker_results)
        total_skipped = sum(r.get('skipped', 0) for r in worker_results)
        total_failed = sum(r.get('failed', 0) for r in worker_results)
        mtime_fallback_count = sum(r.get('mtime_fallback_count', 0) for r in worker_results)
        all_errors = [e for r in worker_results for e in r.get('errors', [])]

        total_files = total_organized + total_skipped + total_failed

        logger.info(
            f"[{finalizer_id}] Aggregated: {total_organized} organized, "
            f"{total_skipped} skipped, {total_failed} failed"
        )

        # Build transaction log
        transaction_log = {
            "transaction_id": parent_job_id,
            "catalog_id": catalog_id,
            "completed_at": datetime.utcnow().isoformat(),
            "output_directory": output_directory,
            "statistics": {
                "total_files": total_files,
                "organized": total_organized,
                "skipped": total_skipped,
                "failed": total_failed,
                "mtime_fallback": mtime_fallback_count
            },
            "errors": all_errors[:100]  # First 100 errors
        }

        # Save transaction log
        log_dir = Path(output_directory) / ".vam_transactions"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{parent_job_id}.json"

        with open(log_path, 'w') as f:
            json.dump(transaction_log, f, indent=2)

        logger.info(f"[{finalizer_id}] Saved transaction log to {log_path}")

        # Determine final status
        if total_failed == 0:
            status = "SUCCESS"
        elif total_failed / total_files < 0.1:  # Less than 10% failed
            status = "SUCCESS"  # With warnings
        else:
            status = "FAILURE"

        # Update job in database
        from ..db import get_db_context
        with get_db_context() as session:
            job = session.query(Job).filter(Job.id == parent_job_id).first()
            if job:
                job.status = status
                job.result = {
                    "status": "completed",
                    "statistics": transaction_log["statistics"],
                    "transaction_log": str(log_path),
                    "errors": all_errors[:100]
                }
                session.commit()

        # Publish completion
        publish_completion(
            parent_job_id,
            status,
            result={
                "status": "completed",
                "total_organized": total_organized,
                "total_skipped": total_skipped,
                "total_failed": total_failed,
                "mtime_fallback_count": mtime_fallback_count,
                "transaction_log": str(log_path)
            }
        )

        logger.info(f"[{finalizer_id}] Finalizer complete")

        return {
            "status": "completed",
            "total_organized": total_organized,
            "total_skipped": total_skipped,
            "total_failed": total_failed,
            "mtime_fallback_count": mtime_fallback_count,
            "errors": all_errors,
            "transaction_log": str(log_path)
        }

    except Exception as e:
        logger.error(f"[{finalizer_id}] Finalizer failed: {e}", exc_info=True)
        publish_completion(parent_job_id, "FAILURE", error=str(e))
        raise
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/jobs/test_reorganize.py::test_reorganize_finalizer_aggregates_results -v`

Expected: PASS

**Step 5: Commit**

```bash
git add vam_tools/jobs/reorganize.py tests/jobs/test_reorganize.py
git commit -m "feat: implement reorganization finalizer task"
```

---

## Task 10: Add Reorganization to Job Router

**Files:**
- Modify: `vam_tools/api/routers/jobs.py`

**Step 1: Add reorganize to JOB_TYPE_TO_TASK mapping**

Find the `JOB_TYPE_TO_TASK` dictionary and add:

```python
from ..jobs.reorganize import reorganize_coordinator_task

JOB_TYPE_TO_TASK = {
    "scan": scan_coordinator_task,
    "analyze": analyze_coordinator_task,
    "detect_duplicates": duplicates_coordinator_task,
    "detect_bursts": burst_detector_task,
    "quality": quality_coordinator_task,
    "auto_tag": tagging_coordinator_task,
    "generate_thumbnails": thumbnail_coordinator_task,
    "organize": organize_catalog_task,
    "reorganize": reorganize_coordinator_task,  # NEW
}
```

**Step 2: Add reorganize to COORDINATOR_JOB_TYPES**

Find where COORDINATOR_JOB_TYPES is defined and add:

```python
COORDINATOR_JOB_TYPES = {"detect_duplicates", "detect_bursts", "quality", "reorganize"}
```

**Step 3: Update start_job docstring**

Update the docstring to include reorganize:

```python
@router.post("/start", response_model=JobResponse, status_code=202)
def start_job(request: GenericJobRequest, db: Session = Depends(get_db)):
    """Start a job by type.

    Supported job types:
    - generate_thumbnails: Generate thumbnails for all images (parallel)
    - detect_duplicates: Detect duplicate images (parallel)
    - auto_tag: Auto-tag images using AI (parallel)
    - detect_bursts: Detect burst photo sequences (parallel)
    - quality: Compute quality scores for images (parallel)
    - reorganize: Reorganize library into date-based structure (parallel)
    """
```

**Step 4: Create GenericJobRequest extension for reorganize parameters**

Add new request model before start_job function:

```python
class ReorganizeJobRequest(BaseModel):
    """Request to start reorganization job."""

    catalog_id: uuid.UUID
    output_directory: str
    operation: str = "copy"  # "copy" or "move"
    dry_run: bool = False
```

**Step 5: Add dedicated endpoint for reorganize (optional but clearer)**

Add after start_job function:

```python
@router.post("/reorganize", response_model=JobResponse, status_code=202)
def start_reorganize(request: ReorganizeJobRequest, db: Session = Depends(get_db)):
    """Start a file reorganization job.

    Args:
        request: Reorganization parameters

    Returns:
        Job response with job ID
    """
    logger.info(
        f"Starting reorganize for catalog {request.catalog_id} "
        f"to {request.output_directory} ({request.operation}, dry_run={request.dry_run})"
    )

    # Validate output directory
    from pathlib import Path
    output_path = Path(request.output_directory)
    if not output_path.is_absolute():
        raise HTTPException(
            status_code=400,
            detail="Output directory must be an absolute path"
        )

    # Start task
    task = reorganize_coordinator_task.delay(
        catalog_id=str(request.catalog_id),
        output_directory=request.output_directory,
        operation=request.operation,
        dry_run=request.dry_run
    )

    # Save job to database
    job = Job(
        id=task.id,
        catalog_id=request.catalog_id,
        job_type="reorganize",
        status="PENDING",
        parameters={
            "catalog_id": str(request.catalog_id),
            "output_directory": request.output_directory,
            "operation": request.operation,
            "dry_run": request.dry_run,
            "parallel": True
        }
    )
    db.add(job)
    db.commit()

    return JobResponse(
        job_id=task.id,
        status="pending",
        progress={"parallel": True},
        result={}
    )
```

**Step 6: Test API endpoint**

Run: `docker compose restart web`

Test with curl:

```bash
curl -X POST http://localhost:8765/api/jobs/reorganize \
  -H "Content-Type: application/json" \
  -d '{
    "catalog_id": "your-catalog-id",
    "output_directory": "/tmp/organized-test",
    "operation": "copy",
    "dry_run": true
  }'
```

Expected: Returns job_id and status

**Step 7: Commit**

```bash
git add vam_tools/api/routers/jobs.py
git commit -m "feat: add reorganization API endpoint"
```

---

## Task 11: Create UI Modal Component

**Files:**
- Modify: `vam_tools/web/static/index.html`
- Modify: `vam_tools/web/static/app.js`
- Modify: `vam_tools/web/static/styles.css`

**Step 1: Add "Organize Library" button to toolbar**

In `index.html`, find the main toolbar and add button:

```html
<!-- In toolbar, after Detect Bursts button -->
<button @click="showOrganizeModal = true" class="toolbar-button" title="Reorganize library into date-based structure">
    📁 Organize Library
</button>
```

**Step 2: Add modal HTML**

Add modal at end of `index.html` before closing `</div>` of `#app`:

```html
<!-- Reorganize Modal -->
<div v-if="showOrganizeModal" class="modal-overlay" @click.self="showOrganizeModal = false">
    <div class="modal-content reorganize-modal">
        <div class="modal-header">
            <h2>📁 Organize Library</h2>
            <button @click="showOrganizeModal = false" class="close-button">×</button>
        </div>

        <div class="modal-body">
            <div class="form-group">
                <label for="organize-output-dir">Output Directory *</label>
                <input
                    id="organize-output-dir"
                    v-model="organizeForm.outputDirectory"
                    type="text"
                    placeholder="/mnt/storage/organized"
                    class="form-input"
                />
                <small class="form-help">Target directory for organized files (must be absolute path)</small>
            </div>

            <div class="form-group">
                <label>Operation</label>
                <div class="radio-group">
                    <label class="radio-label">
                        <input type="radio" v-model="organizeForm.operation" value="copy" />
                        <span>Copy</span>
                        <small>Leave originals untouched (recommended)</small>
                    </label>
                    <label class="radio-label">
                        <input type="radio" v-model="organizeForm.operation" value="move" />
                        <span>Move</span>
                        <small>Free up space in original locations</small>
                    </label>
                </div>
            </div>

            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" v-model="organizeForm.dryRun" />
                    <span>Dry Run (Preview Only)</span>
                </label>
                <small class="form-help">Preview changes without executing file operations</small>
            </div>

            <div v-if="organizeForm.summary" class="summary-box">
                <h3>Summary</h3>
                <div class="summary-stats">
                    <div class="stat">
                        <span class="stat-label">Total Images</span>
                        <span class="stat-value">{{ formatNumber(organizeForm.summary.total) }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Active Images</span>
                        <span class="stat-value">{{ formatNumber(organizeForm.summary.active) }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Rejected Images</span>
                        <span class="stat-value">{{ formatNumber(organizeForm.summary.rejected) }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">No EXIF Date</span>
                        <span class="stat-value">{{ formatNumber(organizeForm.summary.no_date) }}</span>
                    </div>
                </div>
                <p class="summary-note">
                    Active → <code>/organized/YYYY/MM-DD/</code><br>
                    Rejected → <code>/organized/_rejected/YYYY/MM-DD/</code><br>
                    Images without EXIF dates will use file modification time
                </p>
            </div>
        </div>

        <div class="modal-footer">
            <button @click="showOrganizeModal = false" class="button-secondary">Cancel</button>
            <button @click="loadOrganizeSummary" class="button-primary" :disabled="!isOrganizeFormValid">
                Load Summary
            </button>
            <button @click="startReorganize" class="button-primary" :disabled="!isOrganizeFormValid || !organizeForm.summary">
                {{ organizeForm.dryRun ? 'Preview' : 'Start Reorganization' }}
            </button>
        </div>
    </div>
</div>
```

**Step 3: Add Vue data properties**

In `app.js`, add to `data()`:

```javascript
data() {
    return {
        // ... existing properties ...

        showOrganizeModal: false,
        organizeForm: {
            outputDirectory: '',
            operation: 'copy',
            dryRun: true,
            summary: null
        }
    }
}
```

**Step 4: Add computed properties and methods**

In `app.js`, add to `computed`:

```javascript
computed: {
    // ... existing computed properties ...

    isOrganizeFormValid() {
        return this.organizeForm.outputDirectory &&
               this.organizeForm.outputDirectory.startsWith('/') &&
               this.currentCatalog;
    }
}
```

Add to `methods`:

```javascript
methods: {
    // ... existing methods ...

    async loadOrganizeSummary() {
        if (!this.currentCatalog) return;

        try {
            // Query database for summary stats
            const response = await axios.get(`/api/catalogs/${this.currentCatalog.id}/stats`);
            const stats = response.data;

            this.organizeForm.summary = {
                total: stats.total_images || 0,
                active: stats.active_images || 0,
                rejected: stats.rejected_images || 0,
                no_date: stats.images_without_dates || 0
            };
        } catch (error) {
            console.error('Failed to load summary:', error);
            this.addNotification('Failed to load summary', 'error');
        }
    },

    async startReorganize() {
        if (!this.isOrganizeFormValid || !this.currentCatalog) return;

        try {
            const response = await axios.post('/api/jobs/reorganize', {
                catalog_id: this.currentCatalog.id,
                output_directory: this.organizeForm.outputDirectory,
                operation: this.organizeForm.operation,
                dry_run: this.organizeForm.dryRun
            });

            const jobType = this.organizeForm.dryRun ? 'Preview' : 'Reorganization';
            this.addNotification(`${jobType} job started`, 'success');

            // Add job to tracked jobs
            if (response.data.job_id) {
                this.allJobs.unshift(response.data.job_id);
                this.jobMetadata[response.data.job_id] = {
                    name: `Reorganize: ${this.currentCatalog.name}`,
                    started: new Date().toISOString()
                };
            }

            // Close modal
            this.showOrganizeModal = false;

            // Reset form
            this.organizeForm = {
                outputDirectory: '',
                operation: 'copy',
                dryRun: true,
                summary: null
            };
        } catch (error) {
            console.error('Failed to start reorganization:', error);
            const detail = error.response?.data?.detail || error.message;
            this.addNotification(`Failed to start: ${detail}`, 'error');
        }
    }
}
```

**Step 5: Add CSS styles**

In `styles.css`, add at end:

```css
/* Reorganize Modal */
.reorganize-modal {
    max-width: 600px;
}

.reorganize-modal .form-group {
    margin-bottom: 1.5rem;
}

.reorganize-modal .radio-group {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.reorganize-modal .radio-label {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.75rem;
    border: 1px solid #334155;
    border-radius: 0.5rem;
    cursor: pointer;
    transition: all 0.2s;
}

.reorganize-modal .radio-label:hover {
    background: #1e293b;
    border-color: #38bdf8;
}

.reorganize-modal .radio-label input[type="radio"] {
    margin-top: 0.25rem;
}

.reorganize-modal .radio-label > span {
    font-weight: 500;
    flex: 1;
}

.reorganize-modal .radio-label small {
    display: block;
    color: #94a3b8;
    margin-top: 0.25rem;
}

.reorganize-modal .checkbox-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 500;
}

.reorganize-modal .summary-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-top: 1rem;
}

.reorganize-modal .summary-box h3 {
    margin: 0 0 1rem 0;
    font-size: 1rem;
    color: #e2e8f0;
}

.reorganize-modal .summary-stats {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.reorganize-modal .stat {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.reorganize-modal .stat-label {
    font-size: 0.875rem;
    color: #94a3b8;
}

.reorganize-modal .stat-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: #38bdf8;
}

.reorganize-modal .summary-note {
    font-size: 0.875rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.6;
}

.reorganize-modal .summary-note code {
    background: #0f172a;
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    color: #38bdf8;
    font-family: monospace;
    font-size: 0.8125rem;
}
```

**Step 6: Test UI**

Run: `docker compose restart web`

Open browser, click "Organize Library" button, verify:
- Modal opens
- Form fields work
- Validation prevents invalid input
- "Load Summary" fetches stats (may need to implement stats endpoint first)

**Step 7: Commit**

```bash
git add vam_tools/web/static/index.html vam_tools/web/static/app.js vam_tools/web/static/styles.css
git commit -m "feat: add reorganization UI modal and integration"
```

---

## Task 12: Add Stats Endpoint for Summary

**Files:**
- Modify: `vam_tools/api/routers/catalogs.py`

**Step 1: Add stats endpoint**

Find the catalogs router and add:

```python
@router.get("/{catalog_id}/stats")
def get_catalog_stats(catalog_id: str, db: Session = Depends(get_db)):
    """Get catalog statistics for reorganization summary.

    Returns counts of total, active, rejected, and no-date images.
    """
    from sqlalchemy import text, func

    with CatalogDatabase(catalog_id) as catalog_db:
        assert catalog_db.session is not None

        # Total images
        result = catalog_db.session.execute(
            text("SELECT COUNT(*) FROM images WHERE catalog_id = :catalog_id"),
            {"catalog_id": catalog_id}
        )
        total_images = result.scalar() or 0

        # Active images
        result = catalog_db.session.execute(
            text("""
                SELECT COUNT(*) FROM images
                WHERE catalog_id = :catalog_id
                  AND status_id != 'rejected'
            """),
            {"catalog_id": catalog_id}
        )
        active_images = result.scalar() or 0

        # Rejected images
        result = catalog_db.session.execute(
            text("""
                SELECT COUNT(*) FROM images
                WHERE catalog_id = :catalog_id
                  AND status_id = 'rejected'
            """),
            {"catalog_id": catalog_id}
        )
        rejected_images = result.scalar() or 0

        # Images without dates (EXIF dates missing)
        result = catalog_db.session.execute(
            text("""
                SELECT COUNT(*) FROM images
                WHERE catalog_id = :catalog_id
                  AND (dates IS NULL OR dates->>'selected_date' IS NULL)
            """),
            {"catalog_id": catalog_id}
        )
        images_without_dates = result.scalar() or 0

        return {
            "total_images": total_images,
            "active_images": active_images,
            "rejected_images": rejected_images,
            "images_without_dates": images_without_dates
        }
```

**Step 2: Test endpoint**

Run: `docker compose restart web`

Test with curl:

```bash
curl http://localhost:8765/api/catalogs/YOUR-CATALOG-ID/stats
```

Expected: JSON with total_images, active_images, rejected_images, images_without_dates

**Step 3: Commit**

```bash
git add vam_tools/api/routers/catalogs.py
git commit -m "feat: add catalog stats endpoint for reorganization summary"
```

---

## Summary

This implementation plan creates a complete file reorganization system with:

1. ✅ Enhanced OrganizationStrategy (Tasks 1-5)
   - YEAR_SLASH_MONTH_DAY directory structure
   - TIME_CHECKSUM naming strategy
   - Status-based routing to _rejected/
   - mtime fallback for missing EXIF dates
   - Full checksum conflict resolution
   - Idempotent skip logic

2. ✅ Parallel Job Implementation (Tasks 6-9)
   - Coordinator task creates batches
   - Worker tasks process images in parallel
   - Finalizer aggregates results and creates transaction log

3. ✅ API Integration (Tasks 10, 12)
   - /api/jobs/reorganize endpoint
   - /api/catalogs/{id}/stats endpoint

4. ✅ UI Integration (Task 11)
   - "Organize Library" button in toolbar
   - Configuration modal with summary
   - Job progress display (reuses existing)

**Next Steps:**
1. Test end-to-end with small catalog (dry run)
2. Test with real catalog (copy operation)
3. Verify idempotency (run twice, should skip all)
4. Test UI flow
5. Add documentation

**Execution Options:**

Use `superpowers:executing-plans` to implement this plan task-by-task in batches with review checkpoints.
