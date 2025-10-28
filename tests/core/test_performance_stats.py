"""
Tests for performance statistics tracking.
"""

import time
from datetime import datetime

import pytest

from vam_tools.core.performance_stats import (
    AnalysisStatistics,
    FileFormatStats,
    HashingStats,
    OperationStats,
    PerformanceMetrics,
    PerformanceTracker,
)


class TestOperationStats:
    """Tests for OperationStats."""

    def test_operation_stats_initialization(self) -> None:
        """Test default initialization."""
        stats = OperationStats(operation_name="test_op")

        assert stats.operation_name == "test_op"
        assert stats.total_time_seconds == 0.0
        assert stats.call_count == 0
        assert stats.items_processed == 0
        assert stats.errors == 0
        assert stats.average_time_per_item == 0.0
        assert stats.min_time_seconds is None
        assert stats.max_time_seconds is None

    def test_record_single_execution(self) -> None:
        """Test recording a single execution."""
        stats = OperationStats(operation_name="scan")
        stats.record_execution(duration_seconds=1.5, items=10)

        assert stats.total_time_seconds == 1.5
        assert stats.call_count == 1
        assert stats.items_processed == 10
        assert stats.average_time_per_item == 0.15
        assert stats.min_time_seconds == 1.5
        assert stats.max_time_seconds == 1.5

    def test_record_multiple_executions(self) -> None:
        """Test recording multiple executions."""
        stats = OperationStats(operation_name="hash")

        stats.record_execution(duration_seconds=1.0, items=5)
        stats.record_execution(duration_seconds=2.0, items=5)
        stats.record_execution(duration_seconds=0.5, items=5)

        assert stats.total_time_seconds == 3.5
        assert stats.call_count == 3
        assert stats.items_processed == 15
        assert abs(stats.average_time_per_item - (3.5 / 15)) < 0.001
        assert stats.min_time_seconds == 0.5
        assert stats.max_time_seconds == 2.0

    def test_record_execution_with_error(self) -> None:
        """Test recording execution with error."""
        stats = OperationStats(operation_name="convert")
        stats.record_execution(duration_seconds=1.0, items=1, error=True)

        assert stats.errors == 1
        assert stats.call_count == 1


class TestHashingStats:
    """Tests for HashingStats."""

    def test_hashing_stats_initialization(self) -> None:
        """Test default initialization."""
        stats = HashingStats()

        assert stats.dhash_time_seconds == 0.0
        assert stats.ahash_time_seconds == 0.0
        assert stats.whash_time_seconds == 0.0
        assert stats.total_hashes_computed == 0
        assert stats.gpu_hashes == 0
        assert stats.cpu_hashes == 0
        assert stats.failed_hashes == 0
        assert stats.raw_conversions == 0


class TestFileFormatStats:
    """Tests for FileFormatStats."""

    def test_file_format_stats_initialization(self) -> None:
        """Test initialization with format."""
        stats = FileFormatStats(format="jpg")

        assert stats.format == "jpg"
        assert stats.count == 0
        assert stats.total_time_seconds == 0.0
        assert stats.total_size_bytes == 0


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""

    def test_metrics_initialization(self) -> None:
        """Test default initialization."""
        metrics = PerformanceMetrics()

        assert metrics.total_files_analyzed == 0
        assert metrics.files_per_second == 0.0
        assert metrics.total_errors == 0
        assert len(metrics.operations) == 0
        assert len(metrics.error_types) == 0

    def test_finalize_calculates_duration(self) -> None:
        """Test finalize calculates duration from timestamps."""
        metrics = PerformanceMetrics()
        metrics.started_at = datetime.now()
        time.sleep(0.1)
        metrics.completed_at = datetime.now()
        metrics.total_files_analyzed = 100
        metrics.bytes_processed = 1000000

        metrics.finalize()

        assert metrics.total_duration_seconds > 0.1
        assert metrics.files_per_second > 0
        assert metrics.bytes_per_second > 0

    def test_get_slowest_operations(self) -> None:
        """Test getting slowest operations."""
        metrics = PerformanceMetrics()

        # Add some operations
        metrics.operations["fast_op"] = OperationStats(
            operation_name="fast_op", total_time_seconds=1.0
        )
        metrics.operations["slow_op"] = OperationStats(
            operation_name="slow_op", total_time_seconds=10.0
        )
        metrics.operations["medium_op"] = OperationStats(
            operation_name="medium_op", total_time_seconds=5.0
        )

        slowest = metrics.get_slowest_operations(2)

        assert len(slowest) == 2
        assert slowest[0][0] == "slow_op"
        assert slowest[0][1] == 10.0
        assert slowest[1][0] == "medium_op"

    def test_get_bottlenecks(self) -> None:
        """Test identifying bottleneck operations."""
        metrics = PerformanceMetrics()
        metrics.total_duration_seconds = 100.0

        # Add operations - one taking >10% of time
        metrics.operations["small_op"] = OperationStats(
            operation_name="small_op", total_time_seconds=5.0
        )
        metrics.operations["bottleneck"] = OperationStats(
            operation_name="bottleneck", total_time_seconds=25.0
        )

        bottlenecks = metrics.get_bottlenecks(threshold_percent=10.0)

        assert len(bottlenecks) == 1
        assert "bottleneck" in bottlenecks[0]
        assert "25.0%" in bottlenecks[0]

    def test_get_summary_report(self) -> None:
        """Test generating summary report."""
        metrics = PerformanceMetrics()
        metrics.total_duration_seconds = 100.0
        metrics.total_files_analyzed = 1000
        metrics.files_per_second = 10.0
        metrics.bytes_processed = 1000000000

        metrics.operations["scan"] = OperationStats(
            operation_name="scan", total_time_seconds=50.0
        )

        metrics.hashing.total_hashes_computed = 1000
        metrics.hashing.gpu_hashes = 800
        metrics.hashing.cpu_hashes = 200

        report = metrics.get_summary_report()

        assert "100.00s" in report
        assert "1000" in report
        assert "scan" in report
        assert "1000" in report  # Total hashes
        assert "800" in report  # GPU hashes


class TestPerformanceTracker:
    """Tests for PerformanceTracker."""

    def test_tracker_initialization(self) -> None:
        """Test tracker initializes with default metrics."""
        tracker = PerformanceTracker()

        assert tracker.metrics is not None
        assert tracker.metrics.started_at is not None
        assert "run_" in tracker.metrics.run_id

    def test_track_operation_context_manager(self) -> None:
        """Test tracking operation with context manager."""
        tracker = PerformanceTracker()

        with tracker.track_operation("test_op", items=5):
            time.sleep(0.05)

        assert "test_op" in tracker.metrics.operations
        op_stats = tracker.metrics.operations["test_op"]
        assert op_stats.call_count == 1
        assert op_stats.items_processed == 5
        assert op_stats.total_time_seconds >= 0.05

    def test_track_operation_with_exception(self) -> None:
        """Test tracking records error when exception occurs."""
        tracker = PerformanceTracker()

        with pytest.raises(ValueError):
            with tracker.track_operation("failing_op"):
                raise ValueError("Test error")

        assert "failing_op" in tracker.metrics.operations
        op_stats = tracker.metrics.operations["failing_op"]
        assert op_stats.errors == 1

    def test_track_multiple_operations(self) -> None:
        """Test tracking multiple different operations."""
        tracker = PerformanceTracker()

        with tracker.track_operation("op1", items=10):
            time.sleep(0.01)

        with tracker.track_operation("op2", items=20):
            time.sleep(0.02)

        with tracker.track_operation("op1", items=5):
            time.sleep(0.01)

        assert len(tracker.metrics.operations) == 2
        assert tracker.metrics.operations["op1"].call_count == 2
        assert tracker.metrics.operations["op1"].items_processed == 15
        assert tracker.metrics.operations["op2"].call_count == 1

    def test_record_hash_computation_success(self) -> None:
        """Test recording successful hash computation."""
        tracker = PerformanceTracker()

        tracker.record_hash_computation(
            hash_type="dhash",
            duration=0.1,
            success=True,
            used_gpu=False,
            is_raw=False,
        )

        assert tracker.metrics.hashing.total_hashes_computed == 1
        assert tracker.metrics.hashing.cpu_hashes == 1
        assert tracker.metrics.hashing.gpu_hashes == 0
        assert tracker.metrics.hashing.dhash_time_seconds == 0.1

    def test_record_hash_computation_gpu(self) -> None:
        """Test recording GPU hash computation."""
        tracker = PerformanceTracker()

        tracker.record_hash_computation(
            hash_type="ahash", duration=0.05, success=True, used_gpu=True
        )

        assert tracker.metrics.hashing.gpu_hashes == 1
        assert tracker.metrics.hashing.cpu_hashes == 0
        assert tracker.metrics.hashing.ahash_time_seconds == 0.05

    def test_record_hash_computation_raw(self) -> None:
        """Test recording RAW file hash computation."""
        tracker = PerformanceTracker()

        tracker.record_hash_computation(
            hash_type="dhash", duration=2.0, success=True, is_raw=True
        )

        assert tracker.metrics.hashing.raw_conversions == 1
        assert tracker.metrics.hashing.raw_conversion_time_seconds == 2.0

    def test_record_hash_computation_failure(self) -> None:
        """Test recording failed hash computation."""
        tracker = PerformanceTracker()

        tracker.record_hash_computation(hash_type="dhash", duration=0.1, success=False)

        assert tracker.metrics.hashing.failed_hashes == 1
        assert tracker.metrics.hashing.total_hashes_computed == 0

    def test_record_file_format(self) -> None:
        """Test recording file format statistics."""
        tracker = PerformanceTracker()

        tracker.record_file_format("jpg", size_bytes=1000000, processing_time=0.5)
        tracker.record_file_format("jpg", size_bytes=2000000, processing_time=1.0)
        tracker.record_file_format("png", size_bytes=500000, processing_time=0.3)

        assert len(tracker.metrics.formats) == 2

        jpg_stats = tracker.metrics.formats["jpg"]
        assert jpg_stats.count == 2
        assert jpg_stats.total_size_bytes == 3000000
        assert jpg_stats.total_time_seconds == 1.5
        assert jpg_stats.average_time_per_file == 0.75

        png_stats = tracker.metrics.formats["png"]
        assert png_stats.count == 1

    def test_record_error(self) -> None:
        """Test recording errors."""
        tracker = PerformanceTracker()

        tracker.record_error("FileNotFound", "File missing")
        tracker.record_error("FileNotFound", "Another missing file")
        tracker.record_error("CorruptFile", "Bad header")

        assert tracker.metrics.total_errors == 3
        assert tracker.metrics.error_types["FileNotFound"] == 2
        assert tracker.metrics.error_types["CorruptFile"] == 1

    def test_set_gpu_info(self) -> None:
        """Test setting GPU information."""
        tracker = PerformanceTracker()

        tracker.set_gpu_info("NVIDIA RTX 3080")

        assert tracker.metrics.gpu_utilized is True
        assert tracker.metrics.gpu_device == "NVIDIA RTX 3080"

    def test_finalize_tracker(self) -> None:
        """Test finalizing tracker."""
        tracker = PerformanceTracker()

        with tracker.track_operation("test"):
            time.sleep(0.05)

        tracker.metrics.total_files_analyzed = 100
        tracker.metrics.bytes_processed = 1000000

        metrics = tracker.finalize()

        assert metrics.completed_at is not None
        assert metrics.total_duration_seconds > 0
        assert metrics.files_per_second > 0
        assert metrics.bytes_per_second > 0


class TestAnalysisStatistics:
    """Tests for AnalysisStatistics."""

    def test_analysis_statistics_initialization(self) -> None:
        """Test default initialization."""
        stats = AnalysisStatistics()

        assert stats.last_run is None
        assert len(stats.history) == 0
        assert stats.total_runs == 0
        assert stats.total_files_analyzed == 0
        assert stats.total_time_seconds == 0.0

    def test_add_run(self) -> None:
        """Test adding a run's metrics."""
        stats = AnalysisStatistics()

        metrics1 = PerformanceMetrics()
        metrics1.total_files_analyzed = 100
        metrics1.total_duration_seconds = 10.0

        stats.add_run(metrics1)

        assert stats.last_run is metrics1
        assert len(stats.history) == 1
        assert stats.total_runs == 1
        assert stats.total_files_analyzed == 100
        assert stats.total_time_seconds == 10.0

    def test_add_multiple_runs(self) -> None:
        """Test adding multiple runs."""
        stats = AnalysisStatistics()

        for i in range(5):
            metrics = PerformanceMetrics()
            metrics.total_files_analyzed = 100 + i * 10
            metrics.total_duration_seconds = 10.0 + i
            stats.add_run(metrics)

        assert len(stats.history) == 5
        assert stats.total_runs == 5
        assert stats.total_files_analyzed == 600
        # Last run should be most recent
        assert stats.last_run.total_files_analyzed == 140

    def test_history_limit(self) -> None:
        """Test history is limited to 10 runs."""
        stats = AnalysisStatistics()

        # Add 15 runs
        for i in range(15):
            metrics = PerformanceMetrics()
            metrics.run_id = f"run_{i}"
            stats.add_run(metrics)

        # Should only keep last 10
        assert len(stats.history) == 10
        # Most recent should be run_14
        assert stats.history[0].run_id == "run_14"
        # Oldest in history should be run_5
        assert stats.history[-1].run_id == "run_5"

    def test_average_throughput_calculation(self) -> None:
        """Test average throughput is calculated correctly."""
        stats = AnalysisStatistics()

        metrics1 = PerformanceMetrics()
        metrics1.total_files_analyzed = 100
        metrics1.total_duration_seconds = 10.0
        stats.add_run(metrics1)

        metrics2 = PerformanceMetrics()
        metrics2.total_files_analyzed = 200
        metrics2.total_duration_seconds = 20.0
        stats.add_run(metrics2)

        # Average throughput: 300 files / 30 seconds = 10 files/sec
        assert abs(stats.average_throughput - 10.0) < 0.01

    def test_get_trends_insufficient_data(self) -> None:
        """Test trends with insufficient data."""
        stats = AnalysisStatistics()

        trends = stats.get_trends()

        assert trends["status"] == "insufficient_data"

    def test_get_trends_with_data(self) -> None:
        """Test trends calculation with sufficient data."""
        stats = AnalysisStatistics()

        # Add multiple runs with improving throughput
        for i in range(5):
            metrics = PerformanceMetrics()
            metrics.files_per_second = 10.0 + i * 2  # Improving
            stats.add_run(metrics)

        trends = stats.get_trends()

        assert trends["status"] == "ok"
        assert "average_recent_throughput" in trends
        assert trends["throughput_trend"] in ["improving", "declining"]


class TestIntegration:
    """Integration tests for performance tracking."""

    def test_full_analysis_workflow(self) -> None:
        """Test complete workflow of tracking analysis."""
        tracker = PerformanceTracker()

        # Simulate scanning phase
        with tracker.track_operation("scan_files", items=100):
            time.sleep(0.05)

        # Simulate hashing phase
        for i in range(50):
            tracker.record_hash_computation(
                "dhash", 0.001, success=True, used_gpu=(i % 2 == 0)
            )

        # Simulate some file format processing
        tracker.record_file_format("jpg", 1000000, 0.01)
        tracker.record_file_format("arw", 20000000, 0.5)

        # Record an error
        tracker.record_error("CorruptFile", "Bad RAW file")

        # Finalize
        tracker.metrics.total_files_analyzed = 100
        metrics = tracker.finalize()

        # Verify all data was captured
        assert "scan_files" in metrics.operations
        assert metrics.hashing.total_hashes_computed == 50
        assert metrics.hashing.gpu_hashes == 25
        assert metrics.hashing.cpu_hashes == 25
        assert len(metrics.formats) == 2
        assert metrics.total_errors == 1

        # Get summary
        summary = metrics.get_summary_report()
        assert "scan_files" in summary
        assert "50" in summary  # Total hashes

    def test_statistics_accumulation(self) -> None:
        """Test accumulation across multiple analysis runs."""
        stats = AnalysisStatistics()

        # Run 3 analyses
        for run_num in range(3):
            tracker = PerformanceTracker()

            with tracker.track_operation("process", items=100):
                time.sleep(0.02)

            tracker.metrics.total_files_analyzed = 100
            metrics = tracker.finalize()

            stats.add_run(metrics)

        # Verify accumulation
        assert stats.total_runs == 3
        assert stats.total_files_analyzed == 300
        assert stats.average_throughput > 0
        assert len(stats.history) == 3
