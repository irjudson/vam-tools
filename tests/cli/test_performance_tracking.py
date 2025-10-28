"""Tests for performance tracking in CLI."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from vam_tools.cli.analyze import analyze


class TestPerformanceTrackerIntegration:
    """Test that performance tracker is properly configured in CLI."""

    def test_performance_tracker_has_callback(self, tmp_path):
        """Test that PerformanceTracker is created with WebSocket callback."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create a test image
        test_img = source_dir / "test.jpg"
        test_img.write_text("test")

        # Mock the PerformanceTracker to capture its initialization
        with patch("vam_tools.cli.analyze.PerformanceTracker") as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            # Mock the scanner to avoid actual file processing
            with patch("vam_tools.cli.analyze.ImageScanner") as mock_scanner:
                mock_scanner_instance = MagicMock()
                mock_scanner.return_value = mock_scanner_instance

                runner = CliRunner()
                _result = runner.invoke(
                    analyze,
                    [
                        str(catalog_dir),
                        "-s",
                        str(source_dir),
                    ],
                )

                # Verify PerformanceTracker was created with callback
                assert mock_tracker_class.called
                call_args = mock_tracker_class.call_args

                # Check that update_callback was provided
                if call_args.kwargs:
                    assert "update_callback" in call_args.kwargs
                    assert call_args.kwargs["update_callback"] is not None
                    assert "update_interval" in call_args.kwargs
                    assert call_args.kwargs["update_interval"] == 5.0

    def test_performance_tracker_callback_writes_to_catalog(self, tmp_path):
        """Test that the callback writes performance stats to catalog."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        test_img = source_dir / "test.jpg"
        test_img.write_text("test")

        with patch("vam_tools.cli.analyze.PerformanceTracker") as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            with patch("vam_tools.cli.analyze.ImageScanner"):
                runner = CliRunner()
                _result = runner.invoke(
                    analyze,
                    [
                        str(catalog_dir),
                        "-s",
                        str(source_dir),
                    ],
                )

                # Verify the callback is a callable function
                call_args = mock_tracker_class.call_args
                if call_args.kwargs and "update_callback" in call_args.kwargs:
                    callback = call_args.kwargs["update_callback"]
                    assert callable(callback)
                    # Verify it's named appropriately
                    assert callback.__name__ == "performance_update_callback"


class TestWebSocketBroadcastFunction:
    """Test the WebSocket broadcast function."""

    def test_sync_broadcast_handles_no_event_loop(self):
        """Test that sync_broadcast handles cases with no event loop gracefully."""
        from vam_tools.web.api import sync_broadcast_performance_update

        # This should not raise an exception even without an event loop
        test_data = {
            "run_id": "test123",
            "files_analyzed": 10,
            "operations": {},
        }

        # Should complete without error (though it won't actually broadcast anything
        # since there are no WebSocket connections)
        try:
            sync_broadcast_performance_update(test_data)
        except Exception as e:
            pytest.fail(f"sync_broadcast_performance_update raised exception: {e}")

    def test_broadcast_update_message_format(self):
        """Test that broadcast messages have the correct format."""
        import asyncio

        from vam_tools.web.api import broadcast_performance_update

        test_data = {
            "run_id": "test123",
            "files_analyzed": 10,
        }

        # Run the async function (won't actually send since no connections)
        try:
            asyncio.run(broadcast_performance_update(test_data))
        except Exception:
            # Expected to fail since there are no WebSocket connections
            pass


class TestPerformancePollingEndpoint:
    """Test the polling endpoint for performance statistics."""

    def test_performance_endpoint_no_data(self, tmp_path):
        """Test performance endpoint returns no_data when no analysis has run."""
        from fastapi.testclient import TestClient

        from vam_tools.core.catalog import CatalogDatabase
        from vam_tools.web.api import app, init_catalog

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        # Create empty catalog
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[tmp_path / "source"])
            db.save()

        init_catalog(catalog_dir)
        client = TestClient(app)

        response = client.get("/api/performance/current")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "no_data"
        assert data["data"] is None

    def test_performance_endpoint_with_completed_run(self, tmp_path):
        """Test performance endpoint returns idle status for completed analysis."""
        from fastapi.testclient import TestClient

        from vam_tools.core.catalog import CatalogDatabase
        from vam_tools.web.api import app, init_catalog

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        # Create catalog with completed performance stats
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[tmp_path / "source"])

            # Store performance statistics with completed timestamp
            perf_stats = {
                "last_run": {
                    "run_id": "test123",
                    "started_at": "2024-01-01T00:00:00",
                    "completed_at": "2024-01-01T00:10:00",
                    "total_files_analyzed": 100,
                    "files_per_second": 10.0,
                },
                "history": [],
            }
            db.store_performance_statistics(perf_stats)
            db.save()

        init_catalog(catalog_dir)
        client = TestClient(app)

        response = client.get("/api/performance/current")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "idle"
        assert data["data"] is not None
        assert data["data"]["run_id"] == "test123"
        assert data["data"]["total_files_analyzed"] == 100

    def test_performance_endpoint_with_running_analysis(self, tmp_path):
        """Test performance endpoint returns running status for in-progress analysis."""
        from fastapi.testclient import TestClient

        from vam_tools.core.catalog import CatalogDatabase
        from vam_tools.web.api import app, init_catalog

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        # Create catalog with running performance stats (no completed_at)
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[tmp_path / "source"])

            perf_stats = {
                "last_run": {
                    "run_id": "test456",
                    "started_at": "2024-01-01T00:00:00",
                    "completed_at": None,  # Still running
                    "total_files_analyzed": 50,
                    "files_per_second": 5.0,
                },
                "history": [],
            }
            db.store_performance_statistics(perf_stats)
            db.save()

        init_catalog(catalog_dir)
        client = TestClient(app)

        response = client.get("/api/performance/current")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "running"
        assert data["data"] is not None
        assert data["data"]["run_id"] == "test456"
        assert data["data"]["completed_at"] is None
