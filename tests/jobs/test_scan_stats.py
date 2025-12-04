"""Tests for scan statistics tracking."""

import time

from vam_tools.jobs.scan_stats import ScanStatistics


class TestScanStatistics:
    """Tests for ScanStatistics class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        stats = ScanStatistics()
        assert stats.files_discovered == 0
        assert stats.files_added == 0
        assert stats.total_skipped == 0
        assert stats.total_errors == 0
        assert stats.start_time > 0

    def test_total_skipped(self) -> None:
        """Test total_skipped property aggregates all skip reasons."""
        stats = ScanStatistics()
        stats.skipped_already_in_catalog = 10
        stats.skipped_duplicate_checksum = 5
        stats.skipped_hidden_file = 3
        stats.skipped_synology_metadata = 2
        stats.skipped_unsupported_format = 20
        stats.skipped_file_not_accessible = 1

        assert stats.total_skipped == 41

    def test_total_errors(self) -> None:
        """Test total_errors property aggregates all error types."""
        stats = ScanStatistics()
        stats.errors_metadata_extraction = 5
        stats.errors_checksum_computation = 2
        stats.errors_thumbnail_generation = 3
        stats.errors_database = 1
        stats.errors_other = 4

        assert stats.total_errors == 15

    def test_total_files_processed(self) -> None:
        """Test total_files_processed includes all outcomes."""
        stats = ScanStatistics()
        stats.files_added = 100
        stats.files_updated = 10
        stats.skipped_already_in_catalog = 50
        stats.errors_metadata_extraction = 5

        # 100 + 10 + 50 + 5 = 165
        assert stats.total_files_processed == 165

    def test_duration_and_throughput(self) -> None:
        """Test duration calculation and throughput."""
        stats = ScanStatistics()
        stats.start_time = time.time() - 10  # 10 seconds ago
        stats.files_added = 100

        duration = stats.duration_seconds
        assert 9.5 < duration < 10.5  # Allow some tolerance

        stats.finish()
        fps = stats.files_per_second
        assert fps > 0  # Should be around 10 fps

    def test_record_error_samples(self) -> None:
        """Test error recording with samples."""
        stats = ScanStatistics()

        stats.record_error("/path/to/file1.jpg", "metadata", "Failed to extract")
        stats.record_error("/path/to/file2.jpg", "database", "Connection error")

        assert len(stats.error_samples) == 2
        assert stats.error_samples[0]["file"] == "/path/to/file1.jpg"
        assert stats.error_samples[0]["type"] == "metadata"

    def test_record_error_limit(self) -> None:
        """Test error samples are limited to max_error_samples."""
        stats = ScanStatistics()
        stats.max_error_samples = 5

        for i in range(10):
            stats.record_error(f"/path/to/file{i}.jpg", "error", "Message")

        assert len(stats.error_samples) == 5

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        stats = ScanStatistics()
        stats.start_time = time.time() - 1  # 1 second ago
        stats.files_discovered = 1000
        stats.files_added = 800
        stats.skipped_already_in_catalog = 150
        stats.skipped_unsupported_format = 30
        stats.errors_metadata_extraction = 20
        stats.images_processed = 750
        stats.videos_processed = 50
        stats.total_bytes_processed = 10 * 1024**3  # 10 GB
        stats.thumbnails_generated = 800
        stats.finish()

        result = stats.to_dict()

        assert result["status"] == "completed"
        assert result["files_discovered"] == 1000
        assert result["files_added"] == 800
        assert result["total_skipped"] == 180
        assert result["total_errors"] == 20
        assert result["skip_reasons"]["already_in_catalog"] == 150
        assert result["skip_reasons"]["unsupported_format"] == 30
        assert result["error_reasons"]["metadata_extraction"] == 20
        assert result["file_types"]["images"] == 750
        assert result["file_types"]["videos"] == 50
        assert result["size_stats"]["total_gb"] == 10.0
        assert result["thumbnails"]["generated"] == 800
        assert result["duration_seconds"] >= 0  # Could be 0 on very fast runs
        assert result["started_at"] is not None
        assert result["completed_at"] is not None

    def test_to_summary(self) -> None:
        """Test human-readable summary generation."""
        stats = ScanStatistics()
        stats.files_discovered = 100
        stats.files_added = 80
        stats.skipped_already_in_catalog = 15
        stats.skipped_hidden_file = 5
        stats.errors_metadata_extraction = 3
        stats.images_processed = 75
        stats.videos_processed = 5
        stats.total_bytes_processed = 1024**3  # 1 GB
        stats.finish()

        summary = stats.to_summary()

        assert "Files discovered: 100" in summary
        assert "Files added to catalog: 80" in summary
        assert "Already in catalog: 15" in summary
        assert "Hidden files: 5" in summary
        assert "Metadata extraction: 3" in summary
        assert "75 images" in summary
        assert "5 videos" in summary
        assert "1.00 GB" in summary
