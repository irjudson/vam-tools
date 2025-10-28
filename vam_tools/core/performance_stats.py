"""
Performance statistics tracking for analysis operations.

Tracks timing, throughput, and resource usage to identify optimization opportunities.
"""

import logging
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class OperationStats(BaseModel):
    """Statistics for a single operation type."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation_name: str
    total_time_seconds: float = 0.0
    call_count: int = 0
    items_processed: int = 0
    errors: int = 0
    average_time_per_item: float = 0.0
    min_time_seconds: Optional[float] = None
    max_time_seconds: Optional[float] = None

    def record_execution(
        self, duration_seconds: float, items: int = 1, error: bool = False
    ) -> None:
        """Record an operation execution."""
        self.total_time_seconds += duration_seconds
        self.call_count += 1
        self.items_processed += items

        if error:
            self.errors += 1

        # Update min/max
        if self.min_time_seconds is None or duration_seconds < self.min_time_seconds:
            self.min_time_seconds = duration_seconds
        if self.max_time_seconds is None or duration_seconds > self.max_time_seconds:
            self.max_time_seconds = duration_seconds

        # Recalculate average
        if self.items_processed > 0:
            self.average_time_per_item = self.total_time_seconds / self.items_processed


class HashingStats(BaseModel):
    """Statistics for hash computation."""

    dhash_time_seconds: float = 0.0
    ahash_time_seconds: float = 0.0
    whash_time_seconds: float = 0.0
    total_hashes_computed: int = 0
    gpu_hashes: int = 0
    cpu_hashes: int = 0
    failed_hashes: int = 0
    raw_conversions: int = 0
    raw_conversion_time_seconds: float = 0.0


class FileFormatStats(BaseModel):
    """Statistics broken down by file format."""

    format: str
    count: int = 0
    total_time_seconds: float = 0.0
    total_size_bytes: int = 0
    average_time_per_file: float = 0.0


class PerformanceMetrics(BaseModel):
    """Performance metrics for analysis run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Run identification
    run_id: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0

    # Overall statistics
    total_files_analyzed: int = 0
    files_per_second: float = 0.0
    bytes_processed: int = 0
    bytes_per_second: float = 0.0

    # Operation-specific statistics
    operations: Dict[str, OperationStats] = Field(default_factory=dict)

    # Hashing statistics
    hashing: HashingStats = Field(default_factory=HashingStats)

    # File format breakdown
    formats: Dict[str, FileFormatStats] = Field(default_factory=dict)

    # Resource usage
    peak_memory_mb: float = 0.0
    gpu_utilized: bool = False
    gpu_device: Optional[str] = None

    # Error tracking
    total_errors: int = 0
    error_types: Dict[str, int] = Field(default_factory=dict)

    def finalize(self) -> None:
        """Finalize metrics after analysis completes."""
        if self.started_at and self.completed_at:
            self.total_duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()

        if self.total_duration_seconds > 0:
            self.files_per_second = (
                self.total_files_analyzed / self.total_duration_seconds
            )
            self.bytes_per_second = self.bytes_processed / self.total_duration_seconds

    def get_slowest_operations(self, n: int = 5) -> List[tuple[str, float]]:
        """Get the N slowest operations by total time."""
        ops = [
            (name, stats.total_time_seconds) for name, stats in self.operations.items()
        ]
        return sorted(ops, key=lambda x: x[1], reverse=True)[:n]

    def get_bottlenecks(self, threshold_percent: float = 10.0) -> List[str]:
        """Identify operations taking more than threshold percent of total time."""
        if self.total_duration_seconds == 0:
            return []

        bottlenecks = []
        for name, stats in self.operations.items():
            percent = (stats.total_time_seconds / self.total_duration_seconds) * 100
            if percent >= threshold_percent:
                bottlenecks.append(f"{name} ({percent:.1f}%)")

        return bottlenecks

    def get_summary_report(self) -> str:
        """Get human-readable summary report."""
        lines = [
            "=== Performance Analysis Summary ===",
            f"Total Duration: {self.total_duration_seconds:.2f}s",
            f"Files Analyzed: {self.total_files_analyzed}",
            f"Throughput: {self.files_per_second:.2f} files/sec",
            f"Data Processed: {self.bytes_processed / (1024**3):.2f} GB",
            "",
            "Top 5 Slowest Operations:",
        ]

        for name, duration in self.get_slowest_operations(5):
            percent = (duration / self.total_duration_seconds) * 100
            lines.append(f"  {name}: {duration:.2f}s ({percent:.1f}%)")

        lines.append("")
        lines.append("Hashing Statistics:")
        lines.append(f"  Total Hashes: {self.hashing.total_hashes_computed}")
        lines.append(f"  GPU Hashes: {self.hashing.gpu_hashes}")
        lines.append(f"  CPU Hashes: {self.hashing.cpu_hashes}")
        lines.append(
            f"  Failed: {self.hashing.failed_hashes} ({self.hashing.failed_hashes / max(1, self.hashing.total_hashes_computed) * 100:.1f}%)"
        )

        if self.hashing.raw_conversions > 0:
            lines.append(f"  RAW Conversions: {self.hashing.raw_conversions}")
            avg_raw_time = (
                self.hashing.raw_conversion_time_seconds / self.hashing.raw_conversions
            )
            lines.append(f"  Avg RAW Time: {avg_raw_time:.3f}s")

        if self.total_errors > 0:
            lines.append("")
            lines.append(f"Total Errors: {self.total_errors}")
            for error_type, count in sorted(
                self.error_types.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {error_type}: {count}")

        bottlenecks = self.get_bottlenecks(10.0)
        if bottlenecks:
            lines.append("")
            lines.append("Bottlenecks (>10% of total time):")
            for b in bottlenecks:
                lines.append(f"  • {b}")

        return "\n".join(lines)


class PerformanceTracker:
    """Tracker for performance metrics during analysis."""

    def __init__(
        self,
        update_callback: Optional[Callable[[Dict], None]] = None,
        update_interval: float = 1.0,
    ):
        """
        Initialize performance tracker.

        Args:
            update_callback: Optional callback function to call with performance updates.
                           Useful for broadcasting real-time updates via WebSocket.
                           Note: Callbacks are throttled to avoid overwhelming receivers.
            update_interval: Minimum seconds between callback invocations (default: 1.0).
                           Set to 0 to disable throttling (not recommended for WebSocket).
        """
        self.metrics = PerformanceMetrics()
        self.metrics.run_id = f"run_{int(time.time())}"
        self.metrics.started_at = datetime.now()
        self.update_callback = update_callback
        self.update_interval = update_interval
        self._last_callback_time = 0.0

    @contextmanager
    def track_operation(
        self, operation_name: str, items: int = 1, record_error: bool = False
    ) -> Generator[None, None, None]:
        """Context manager to track operation timing."""
        start_time = time.time()
        error_occurred = False

        try:
            yield
        except Exception as e:
            error_occurred = True
            self.record_error(operation_name, str(e))
            raise
        finally:
            duration = time.time() - start_time

            # Get or create operation stats
            if operation_name not in self.metrics.operations:
                self.metrics.operations[operation_name] = OperationStats(
                    operation_name=operation_name
                )

            self.metrics.operations[operation_name].record_execution(
                duration, items, error_occurred or record_error
            )

            # Broadcast update if callback is set (with throttling)
            if self.update_callback:
                current_time = time.time()
                # Only call callback if enough time has passed (throttling)
                if (
                    self.update_interval == 0
                    or current_time - self._last_callback_time >= self.update_interval
                ):
                    try:
                        update_data = self.get_current_stats()
                        self.update_callback(update_data)
                        self._last_callback_time = current_time
                    except Exception as e:
                        # Don't let callback errors break the tracking
                        logger.debug(
                            f"Performance update callback failed for '{operation_name}': {e}"
                        )

    def record_hash_computation(
        self,
        hash_type: str,
        duration: float,
        success: bool,
        used_gpu: bool = False,
        is_raw: bool = False,
    ) -> None:
        """Record hash computation statistics."""
        if success:
            self.metrics.hashing.total_hashes_computed += 1
            if used_gpu:
                self.metrics.hashing.gpu_hashes += 1
            else:
                self.metrics.hashing.cpu_hashes += 1

            if hash_type == "dhash":
                self.metrics.hashing.dhash_time_seconds += duration
            elif hash_type == "ahash":
                self.metrics.hashing.ahash_time_seconds += duration
            elif hash_type == "whash":
                self.metrics.hashing.whash_time_seconds += duration

            if is_raw:
                self.metrics.hashing.raw_conversions += 1
                self.metrics.hashing.raw_conversion_time_seconds += duration
        else:
            self.metrics.hashing.failed_hashes += 1

    def record_file_format(
        self, format: str, size_bytes: int, processing_time: float
    ) -> None:
        """Record statistics for a file format."""
        if format not in self.metrics.formats:
            self.metrics.formats[format] = FileFormatStats(format=format)

        stats = self.metrics.formats[format]
        stats.count += 1
        stats.total_time_seconds += processing_time
        stats.total_size_bytes += size_bytes

        if stats.count > 0:
            stats.average_time_per_file = stats.total_time_seconds / stats.count

    def record_error(self, error_type: str, error_message: str) -> None:
        """Record an error."""
        self.metrics.total_errors += 1
        self.metrics.error_types[error_type] = (
            self.metrics.error_types.get(error_type, 0) + 1
        )

    def set_gpu_info(self, device_name: str) -> None:
        """Record GPU information."""
        self.metrics.gpu_utilized = True
        self.metrics.gpu_device = device_name

    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics as a dictionary.

        Returns:
            Dictionary containing current performance metrics
        """
        return {
            "run_id": self.metrics.run_id,
            "started_at": (
                self.metrics.started_at.isoformat() if self.metrics.started_at else None
            ),
            "total_files_analyzed": self.metrics.total_files_analyzed,
            "files_per_second": self.metrics.files_per_second,
            "bytes_processed": self.metrics.bytes_processed,
            "operations": {
                name: {
                    "total_time": stats.total_time_seconds,
                    "call_count": stats.call_count,
                    "items_processed": stats.items_processed,
                }
                for name, stats in self.metrics.operations.items()
            },
        }

    def finalize(self) -> PerformanceMetrics:
        """Finalize and return metrics."""
        self.metrics.completed_at = datetime.now()
        self.metrics.finalize()
        return self.metrics


class AnalysisStatistics(BaseModel):
    """Complete analysis statistics including history."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Last run metrics
    last_run: Optional[PerformanceMetrics] = None

    # Historical runs (keep last 10)
    history: List[PerformanceMetrics] = Field(default_factory=list)

    # Aggregate statistics
    total_runs: int = 0
    total_files_analyzed: int = 0
    total_time_seconds: float = 0.0
    average_throughput: float = 0.0

    def add_run(self, metrics: PerformanceMetrics) -> None:
        """Add a new run's metrics."""
        self.last_run = metrics
        self.history.insert(0, metrics)

        # Keep only last 10 runs
        self.history = self.history[:10]

        # Update aggregates
        self.total_runs += 1
        self.total_files_analyzed += metrics.total_files_analyzed
        self.total_time_seconds += metrics.total_duration_seconds

        if self.total_time_seconds > 0:
            self.average_throughput = (
                self.total_files_analyzed / self.total_time_seconds
            )

    def get_trends(self) -> Dict[str, Any]:
        """Get performance trends across runs."""
        if len(self.history) < 2:
            return {"status": "insufficient_data"}

        recent_throughputs = [
            run.files_per_second for run in self.history[:5] if run.files_per_second > 0
        ]

        if not recent_throughputs:
            return {"status": "no_throughput_data"}

        avg_recent = sum(recent_throughputs) / len(recent_throughputs)

        return {
            "status": "ok",
            "average_recent_throughput": avg_recent,
            "throughput_trend": (
                "improving" if recent_throughputs[0] > avg_recent else "declining"
            ),
            "runs_analyzed": len(recent_throughputs),
        }
