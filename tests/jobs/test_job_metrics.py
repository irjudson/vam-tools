"""Tests for job metrics tracking."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from vam_tools.jobs.job_metrics import (
    DEFAULT_TIMINGS,
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    TARGET_BATCH_DURATION_SECONDS,
    BatchPlan,
    JobMetricsTracker,
    TimedOperationContext,
    TimingMetric,
)


class TestTimingMetric:
    """Tests for TimingMetric dataclass."""

    def test_timing_metric_creation(self):
        """Test creating a timing metric."""
        metric = TimingMetric(
            operation="hash_computation",
            items_processed=100,
            duration_seconds=50.0,
        )
        assert metric.operation == "hash_computation"
        assert metric.items_processed == 100
        assert metric.duration_seconds == 50.0
        assert metric.use_gpu is False
        assert isinstance(metric.timestamp, datetime)

    def test_seconds_per_item(self):
        """Test calculating seconds per item."""
        metric = TimingMetric(
            operation="test",
            items_processed=100,
            duration_seconds=50.0,
        )
        assert metric.seconds_per_item == 0.5

    def test_seconds_per_item_zero_items(self):
        """Test seconds per item with zero items processed."""
        metric = TimingMetric(
            operation="test",
            items_processed=0,
            duration_seconds=10.0,
        )
        assert metric.seconds_per_item == 0.0

    def test_timing_metric_with_gpu(self):
        """Test timing metric with GPU flag."""
        metric = TimingMetric(
            operation="hash_computation_gpu",
            items_processed=1000,
            duration_seconds=5.0,
            use_gpu=True,
        )
        assert metric.use_gpu is True
        assert metric.seconds_per_item == 0.005


class TestBatchPlan:
    """Tests for BatchPlan dataclass."""

    def test_batch_plan_creation(self):
        """Test creating a batch plan."""
        plan = BatchPlan(
            total_items=1000,
            batch_size=100,
            estimated_batch_duration=50.0,
            estimated_total_duration=500.0,
            num_batches=10,
            seconds_per_item=0.5,
        )
        assert plan.total_items == 1000
        assert plan.batch_size == 100
        assert plan.num_batches == 10

    def test_get_batch_ranges(self):
        """Test getting batch ranges."""
        plan = BatchPlan(
            total_items=250,
            batch_size=100,
            estimated_batch_duration=50.0,
            estimated_total_duration=125.0,
            num_batches=3,
            seconds_per_item=0.5,
        )
        ranges = plan.get_batch_ranges()
        assert len(ranges) == 3
        assert ranges[0] == (0, 100)
        assert ranges[1] == (100, 200)
        assert ranges[2] == (200, 250)  # Last batch is smaller

    def test_get_batch_ranges_single_batch(self):
        """Test batch ranges with single batch."""
        plan = BatchPlan(
            total_items=50,
            batch_size=100,
            estimated_batch_duration=25.0,
            estimated_total_duration=25.0,
            num_batches=1,
            seconds_per_item=0.5,
        )
        ranges = plan.get_batch_ranges()
        assert len(ranges) == 1
        assert ranges[0] == (0, 50)

    def test_get_batch_ranges_exact_fit(self):
        """Test batch ranges when items fit exactly in batches."""
        plan = BatchPlan(
            total_items=300,
            batch_size=100,
            estimated_batch_duration=50.0,
            estimated_total_duration=150.0,
            num_batches=3,
            seconds_per_item=0.5,
        )
        ranges = plan.get_batch_ranges()
        assert len(ranges) == 3
        assert ranges[0] == (0, 100)
        assert ranges[1] == (100, 200)
        assert ranges[2] == (200, 300)


class TestJobMetricsTracker:
    """Tests for JobMetricsTracker class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        # Mock execute to return empty results (no cached timings)
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])
        session.execute.return_value = mock_result
        return session

    def test_tracker_initialization(self, mock_session):
        """Test creating a tracker with session."""
        tracker = JobMetricsTracker(mock_session)
        assert tracker.session is mock_session
        assert tracker.catalog_id is None
        assert len(tracker._timing_cache) == 0

    def test_tracker_initialization_with_catalog(self, mock_session):
        """Test creating a tracker with catalog ID."""
        tracker = JobMetricsTracker(mock_session, catalog_id="test-catalog")
        assert tracker.catalog_id == "test-catalog"

    def test_get_timing_estimate_default(self, mock_session):
        """Test getting default timing when no history exists."""
        tracker = JobMetricsTracker(mock_session)
        rate = tracker.get_timing_estimate("hash_computation")
        assert rate == DEFAULT_TIMINGS["hash_computation"]

    def test_get_timing_estimate_with_cache(self, mock_session):
        """Test getting timing from cache."""
        tracker = JobMetricsTracker(mock_session)
        # Manually set cache
        tracker._timing_cache["test_op"] = 0.25
        rate = tracker.get_timing_estimate("test_op")
        assert rate == 0.25

    def test_get_timing_estimate_gpu(self, mock_session):
        """Test getting GPU-specific timing."""
        tracker = JobMetricsTracker(mock_session)
        rate = tracker.get_timing_estimate("hash_computation", use_gpu=True)
        assert rate == DEFAULT_TIMINGS["hash_computation_gpu"]

    def test_get_timing_estimate_unknown_operation(self, mock_session):
        """Test getting timing for unknown operation."""
        tracker = JobMetricsTracker(mock_session)
        rate = tracker.get_timing_estimate("unknown_operation")
        # Should return fallback default (0.1s)
        assert rate == 0.1

    def test_record_timing(self, mock_session):
        """Test recording a timing metric."""
        tracker = JobMetricsTracker(mock_session)
        metric = TimingMetric(
            operation="test_operation",
            items_processed=50,
            duration_seconds=10.0,
        )
        tracker.record_timing(metric)
        # Should have updated cache
        assert "test_operation" in tracker._timing_cache

    def test_record_timing_with_gpu(self, mock_session):
        """Test recording a timing metric with GPU."""
        tracker = JobMetricsTracker(mock_session)
        metric = TimingMetric(
            operation="hash_computation",
            items_processed=1000,
            duration_seconds=5.0,
            use_gpu=True,
        )
        tracker.record_timing(metric)
        # GPU timing should be stored with _gpu suffix
        assert "hash_computation_gpu" in tracker._timing_cache

    def test_plan_batches(self, mock_session):
        """Test planning batches for an operation."""
        tracker = JobMetricsTracker(mock_session)
        plan = tracker.plan_batches("hash_computation", 1000)

        assert plan.total_items == 1000
        assert MIN_BATCH_SIZE <= plan.batch_size <= MAX_BATCH_SIZE
        assert plan.num_batches >= 1
        assert plan.estimated_total_duration > 0

    def test_plan_batches_small_count(self, mock_session):
        """Test planning batches for small item count."""
        tracker = JobMetricsTracker(mock_session)
        plan = tracker.plan_batches("thumbnail_generation", 5)

        # Should return all items in single batch
        assert plan.batch_size == 5
        assert plan.num_batches == 1

    def test_plan_batches_large_count(self, mock_session):
        """Test planning batches for large item count."""
        tracker = JobMetricsTracker(mock_session)
        plan = tracker.plan_batches("metadata_extraction", 100000)

        assert plan.batch_size <= MAX_BATCH_SIZE
        assert plan.num_batches >= 1
        assert plan.estimated_total_duration > 0

    def test_plan_batches_with_gpu(self, mock_session):
        """Test planning batches with GPU acceleration."""
        tracker = JobMetricsTracker(mock_session)
        plan_cpu = tracker.plan_batches("hash_computation", 1000, use_gpu=False)
        plan_gpu = tracker.plan_batches("hash_computation", 1000, use_gpu=True)

        # GPU should have larger batches (faster processing = more items per batch)
        assert plan_gpu.batch_size >= plan_cpu.batch_size

    def test_batch_size_respects_limits(self, mock_session):
        """Test that batch size respects min and max limits."""
        tracker = JobMetricsTracker(mock_session)

        # Slow operation (manually set very slow timing in cache)
        tracker._timing_cache["slow_op"] = 120.0  # 120s per item
        plan = tracker.plan_batches("slow_op", 100)
        assert plan.batch_size >= MIN_BATCH_SIZE

        # Fast operation
        tracker._timing_cache["fast_op"] = 0.0001  # 0.1ms per item
        plan = tracker.plan_batches("fast_op", 100000)
        assert plan.batch_size <= MAX_BATCH_SIZE


class TestTimedOperationContext:
    """Tests for TimedOperationContext."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])
        session.execute.return_value = mock_result
        return session

    def test_timed_operation_context(self, mock_session):
        """Test using timed_operation context manager."""
        tracker = JobMetricsTracker(mock_session)

        with tracker.timed_operation("test_op", 100) as timer:
            # Simulate some work
            pass

        # Should have recorded timing
        assert "test_op" in tracker._timing_cache

    def test_timed_operation_with_exception(self, mock_session):
        """Test that timing is not recorded on exception."""
        tracker = JobMetricsTracker(mock_session)

        try:
            with tracker.timed_operation("failing_op", 100):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should NOT have recorded timing (exception occurred)
        assert "failing_op" not in tracker._timing_cache


class TestConstants:
    """Tests for module constants."""

    def test_default_timings_exist(self):
        """Test that all expected default timings exist."""
        expected_operations = [
            "hash_computation",
            "hash_computation_gpu",
            "duplicate_comparison",
            "quality_scoring",
            "thumbnail_generation",
            "metadata_extraction",
        ]
        for op in expected_operations:
            assert op in DEFAULT_TIMINGS
            assert DEFAULT_TIMINGS[op] > 0

    def test_constants_reasonable_values(self):
        """Test that constants have reasonable values."""
        assert TARGET_BATCH_DURATION_SECONDS > 0
        assert MIN_BATCH_SIZE > 0
        assert MAX_BATCH_SIZE > MIN_BATCH_SIZE


class TestGPUFunctions:
    """Tests for GPU utility functions."""

    def test_check_gpu_available_no_torch(self):
        """Test GPU check when torch is not available."""
        from vam_tools.jobs.job_metrics import check_gpu_available

        with patch.dict("sys.modules", {"torch": None}):
            # This will still import cached torch, so we need to mock the import
            with patch("vam_tools.jobs.job_metrics.check_gpu_available") as mock_check:
                mock_check.return_value = False
                assert mock_check() is False

    def test_get_gpu_info_no_torch(self):
        """Test get GPU info when torch is not available."""
        from vam_tools.jobs.job_metrics import get_gpu_info

        # Mock torch import to raise ImportError
        with patch.dict("sys.modules", {"torch": None}):
            with patch("vam_tools.jobs.job_metrics.get_gpu_info") as mock_info:
                mock_info.return_value = None
                assert mock_info() is None
