"""
Job metrics tracking for adaptive batch sizing.

Tracks per-operation timing data to estimate job durations and
automatically size batches to complete within reasonable timeouts.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Target batch duration in seconds (aim for batches that complete in ~2 minutes)
TARGET_BATCH_DURATION_SECONDS = 120

# Minimum and maximum batch sizes
MIN_BATCH_SIZE = 10
MAX_BATCH_SIZE = 5000

# Default timing estimates (seconds per item) when no historical data exists
DEFAULT_TIMINGS = {
    "hash_computation": 0.5,  # ~0.5s per image for perceptual hash on CPU
    "hash_computation_gpu": 0.05,  # ~0.05s per image with GPU acceleration
    "duplicate_comparison": 0.001,  # ~1ms per comparison
    "quality_scoring": 0.01,  # ~10ms per image for quality scoring
    "thumbnail_generation": 0.2,  # ~200ms per thumbnail
    "metadata_extraction": 0.1,  # ~100ms per file for EXIF extraction
}


@dataclass
class TimingMetric:
    """A single timing measurement."""

    operation: str
    items_processed: int
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    use_gpu: bool = False

    @property
    def seconds_per_item(self) -> float:
        """Calculate average time per item."""
        if self.items_processed == 0:
            return 0.0
        return self.duration_seconds / self.items_processed


@dataclass
class BatchPlan:
    """Plan for processing items in batches."""

    total_items: int
    batch_size: int
    estimated_batch_duration: float
    estimated_total_duration: float
    num_batches: int
    seconds_per_item: float

    def get_batch_ranges(self) -> List[tuple]:
        """Get list of (start, end) tuples for each batch."""
        ranges = []
        for i in range(0, self.total_items, self.batch_size):
            end = min(i + self.batch_size, self.total_items)
            ranges.append((i, end))
        return ranges


class JobMetricsTracker:
    """
    Tracks job timing metrics and provides adaptive batch sizing.

    Stores timing data in the database for persistence across restarts.
    Uses exponential moving average to weight recent measurements more heavily.
    """

    def __init__(self, session: Any, catalog_id: Optional[str] = None):
        """
        Initialize metrics tracker.

        Args:
            session: SQLAlchemy session for database access
            catalog_id: Optional catalog ID for catalog-specific metrics
        """
        self.session = session
        self.catalog_id = catalog_id
        self._timing_cache: Dict[str, float] = {}
        self._load_cached_timings()

    def _load_cached_timings(self) -> None:
        """Load cached timing data from database."""
        try:
            result = self.session.execute(
                text(
                    """
                    SELECT key, value FROM config
                    WHERE key LIKE 'job_timing_%'
                    AND (catalog_id IS NULL OR catalog_id = :catalog_id)
                    ORDER BY updated_at DESC
                """
                ),
                {"catalog_id": self.catalog_id},
            )

            for row in result:
                key = row[0].replace("job_timing_", "")
                try:
                    self._timing_cache[key] = float(json.loads(row[1]))
                except (json.JSONDecodeError, ValueError):
                    pass

            logger.debug(f"Loaded {len(self._timing_cache)} cached timing metrics")

        except Exception as e:
            logger.warning(f"Failed to load cached timings: {e}")

    def _save_timing(self, operation: str, seconds_per_item: float) -> None:
        """Save timing data to database."""
        key = f"job_timing_{operation}"
        try:
            self.session.execute(
                text(
                    """
                    INSERT INTO config (catalog_id, key, value, updated_at)
                    VALUES (:catalog_id, :key, :value, NOW())
                    ON CONFLICT (catalog_id, key) DO UPDATE SET
                        value = :value,
                        updated_at = NOW()
                """
                ),
                {
                    "catalog_id": self.catalog_id,
                    "key": key,
                    "value": json.dumps(seconds_per_item),
                },
            )
            self.session.commit()
        except Exception as e:
            logger.warning(f"Failed to save timing for {operation}: {e}")

    def get_timing_estimate(self, operation: str, use_gpu: bool = False) -> float:
        """
        Get estimated seconds per item for an operation.

        Args:
            operation: Operation name (e.g., 'hash_computation')
            use_gpu: Whether GPU acceleration is being used

        Returns:
            Estimated seconds per item
        """
        cache_key = f"{operation}_gpu" if use_gpu else operation

        # Check cache first
        if cache_key in self._timing_cache:
            return self._timing_cache[cache_key]

        # Fall back to defaults
        if cache_key in DEFAULT_TIMINGS:
            return DEFAULT_TIMINGS[cache_key]
        if operation in DEFAULT_TIMINGS:
            return DEFAULT_TIMINGS[operation]

        # Ultimate fallback
        return 0.1

    def record_timing(self, metric: TimingMetric) -> None:
        """
        Record a timing measurement.

        Uses exponential moving average to blend with existing data.

        Args:
            metric: Timing measurement to record
        """
        cache_key = f"{metric.operation}_gpu" if metric.use_gpu else metric.operation

        # Get current estimate
        current_estimate = self.get_timing_estimate(metric.operation, metric.use_gpu)

        # Blend with new measurement (alpha = 0.3 for ~30% weight to new data)
        alpha = 0.3
        new_estimate = (alpha * metric.seconds_per_item) + (
            (1 - alpha) * current_estimate
        )

        # Update cache and persist
        self._timing_cache[cache_key] = new_estimate
        self._save_timing(cache_key, new_estimate)

        logger.debug(
            f"Recorded timing for {cache_key}: {metric.seconds_per_item:.4f}s/item "
            f"(new estimate: {new_estimate:.4f}s/item)"
        )

    def plan_batches(
        self,
        operation: str,
        total_items: int,
        use_gpu: bool = False,
        target_duration: float = TARGET_BATCH_DURATION_SECONDS,
    ) -> BatchPlan:
        """
        Plan batch sizes for an operation.

        Args:
            operation: Operation name
            total_items: Total number of items to process
            use_gpu: Whether GPU acceleration is being used
            target_duration: Target duration per batch in seconds

        Returns:
            BatchPlan with recommended batch sizes
        """
        seconds_per_item = self.get_timing_estimate(operation, use_gpu)

        # Calculate ideal batch size to hit target duration
        if seconds_per_item > 0:
            ideal_batch_size = int(target_duration / seconds_per_item)
        else:
            ideal_batch_size = MAX_BATCH_SIZE

        # Clamp to min/max
        batch_size = max(MIN_BATCH_SIZE, min(ideal_batch_size, MAX_BATCH_SIZE))

        # If total items is small enough, just do it in one batch
        if total_items <= batch_size:
            batch_size = total_items

        num_batches = (total_items + batch_size - 1) // batch_size
        estimated_batch_duration = batch_size * seconds_per_item
        estimated_total_duration = total_items * seconds_per_item

        plan = BatchPlan(
            total_items=total_items,
            batch_size=batch_size,
            estimated_batch_duration=estimated_batch_duration,
            estimated_total_duration=estimated_total_duration,
            num_batches=num_batches,
            seconds_per_item=seconds_per_item,
        )

        logger.info(
            f"Batch plan for {operation}: {num_batches} batches of ~{batch_size} items, "
            f"estimated {estimated_total_duration:.1f}s total"
        )

        return plan

    def timed_operation(
        self, operation: str, items_count: int, use_gpu: bool = False
    ) -> "TimedOperationContext":
        """
        Context manager for timing an operation.

        Usage:
            with metrics.timed_operation("hash_computation", 100) as timer:
                # do work
            # timing is automatically recorded

        Args:
            operation: Operation name
            items_count: Number of items being processed
            use_gpu: Whether GPU is being used

        Returns:
            Context manager that records timing on exit
        """
        return TimedOperationContext(self, operation, items_count, use_gpu)


class TimedOperationContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        tracker: JobMetricsTracker,
        operation: str,
        items_count: int,
        use_gpu: bool,
    ):
        self.tracker = tracker
        self.operation = operation
        self.items_count = items_count
        self.use_gpu = use_gpu
        self.start_time: Optional[float] = None

    def __enter__(self) -> "TimedOperationContext":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None and exc_type is None:
            duration = time.perf_counter() - self.start_time
            metric = TimingMetric(
                operation=self.operation,
                items_processed=self.items_count,
                duration_seconds=duration,
                use_gpu=self.use_gpu,
            )
            self.tracker.record_timing(metric)


def check_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_info() -> Optional[Dict[str, Any]]:
    """Get GPU information if available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        return {
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(0),
        }
    except ImportError:
        return None
