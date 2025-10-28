"""
Tests for performance statistics API endpoints.
"""

from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vam_tools.core.catalog import CatalogDatabase
from vam_tools.core.performance_stats import (
    HashingStats,
    OperationStats,
    PerformanceMetrics,
)
from vam_tools.web.api import app, init_catalog


class TestPerformanceAPI:
    """Tests for performance statistics endpoints."""

    @pytest.fixture
    def client(self, tmp_path: Path):
        """Create test client with initialized catalog."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        # Initialize catalog
        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[tmp_path / "photos"])

        # Initialize API with catalog
        init_catalog(catalog_dir)

        yield TestClient(app)

    def test_get_current_performance_no_data(self, client: TestClient):
        """Test getting current performance stats when no data exists."""
        response = client.get("/api/performance/current")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_data"
        assert data["data"] is None

    def test_get_current_performance_with_data(
        self, client: TestClient, tmp_path: Path
    ):
        """Test getting current performance stats with data."""
        # Initialize catalog and add performance stats
        catalog_dir = tmp_path / "catalog"
        with CatalogDatabase(catalog_dir) as db:
            perf_metrics = PerformanceMetrics(
                run_id="test_run_1",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                total_duration_seconds=120.5,
                total_files_analyzed=1000,
                files_per_second=8.3,
                bytes_processed=5000000000,
                bytes_per_second=41494144.0,
                peak_memory_mb=512.0,
                gpu_utilized=True,
                gpu_device="NVIDIA RTX 3080",
                total_errors=5,
            )

            # Add some operations
            perf_metrics.operations["scan_directories"] = OperationStats(
                operation_name="scan_directories",
                total_time_seconds=10.5,
                call_count=1,
                items_processed=5,
                errors=0,
                average_time_per_item=2.1,
                min_time_seconds=10.5,
                max_time_seconds=10.5,
            )

            # Add hashing stats
            perf_metrics.hashing = HashingStats(
                dhash_time_seconds=25.0,
                ahash_time_seconds=22.0,
                whash_time_seconds=28.0,
                total_hashes_computed=1000,
                gpu_hashes=800,
                cpu_hashes=200,
                failed_hashes=10,
                raw_conversions=50,
                raw_conversion_time_seconds=15.0,
            )

            perf_stats_data = {
                "last_run": perf_metrics.model_dump(mode="json"),
                "history": [],
                "total_runs": 1,
                "total_files_analyzed": 1000,
                "total_time_seconds": 120.5,
                "average_throughput": 8.3,
            }

            db.store_performance_statistics(perf_stats_data)
            db.save()

        # Reinitialize API with updated catalog
        init_catalog(catalog_dir)
        response = client.get("/api/performance/current")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"  # completed_at is set, so status is idle
        assert data["data"] is not None
        assert data["data"]["run_id"] == "test_run_1"
        assert data["data"]["total_files_analyzed"] == 1000
        assert data["data"]["files_per_second"] == 8.3
        assert data["data"]["gpu_utilized"] is True
        assert data["data"]["gpu_device"] == "NVIDIA RTX 3080"
        assert "scan_directories" in data["data"]["operations"]
        assert data["data"]["hashing"]["total_hashes_computed"] == 1000
        assert data["data"]["hashing"]["gpu_hashes"] == 800

    def test_get_performance_history_no_data(self, client: TestClient):
        """Test getting performance history when no data exists."""
        response = client.get("/api/performance/history")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_data"
        assert data["total_runs"] == 0
        assert data["history"] == []

    def test_get_performance_history_with_data(
        self, client: TestClient, tmp_path: Path
    ):
        """Test getting performance history with multiple runs."""
        catalog_dir = tmp_path / "catalog"
        with CatalogDatabase(catalog_dir) as db:
            # Create multiple runs
            runs = []
            for i in range(3):
                run = PerformanceMetrics(
                    run_id=f"test_run_{i}",
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    total_duration_seconds=100.0 + i * 10,
                    total_files_analyzed=500 + i * 100,
                    files_per_second=5.0 + i,
                    bytes_processed=1000000000,
                    bytes_per_second=10000000.0,
                    peak_memory_mb=256.0,
                    gpu_utilized=False,
                    total_errors=i,
                )
                runs.append(run.model_dump(mode="json"))

            perf_stats_data = {
                "last_run": runs[-1],
                "history": runs,
                "total_runs": 3,
                "total_files_analyzed": 2100,
                "total_time_seconds": 330.0,
                "average_throughput": 6.36,
            }

            db.store_performance_statistics(perf_stats_data)
            db.save()

        # Reinitialize API
        init_catalog(catalog_dir)
        response = client.get("/api/performance/history")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["total_runs"] == 3
        assert data["total_files_analyzed"] == 2100
        assert len(data["history"]) == 3
        assert data["history"][0]["run_id"] == "test_run_0"

    def test_get_performance_summary_no_data(self, client: TestClient):
        """Test getting performance summary when no data exists."""
        response = client.get("/api/performance/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_data"
        assert "summary" in data

    def test_get_performance_summary_with_data(
        self, client: TestClient, tmp_path: Path
    ):
        """Test getting performance summary with data."""
        catalog_dir = tmp_path / "catalog"
        with CatalogDatabase(catalog_dir) as db:
            perf_metrics = PerformanceMetrics(
                run_id="test_run_summary",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                total_duration_seconds=150.0,
                total_files_analyzed=2000,
                files_per_second=13.3,
                bytes_processed=10000000000,  # 10 GB
                bytes_per_second=66666666.0,
                peak_memory_mb=1024.0,
                gpu_utilized=True,
                gpu_device="NVIDIA RTX 4090",
                total_errors=2,
            )

            # Add operations
            perf_metrics.operations["scan_directories"] = OperationStats(
                operation_name="scan_directories",
                total_time_seconds=50.0,
                call_count=5,
                items_processed=2000,
                errors=0,
                average_time_per_item=0.025,
                min_time_seconds=8.0,
                max_time_seconds=12.0,
            )

            perf_metrics.operations["compute_hashes"] = OperationStats(
                operation_name="compute_hashes",
                total_time_seconds=80.0,
                call_count=1,
                items_processed=2000,
                errors=2,
                average_time_per_item=0.04,
                min_time_seconds=80.0,
                max_time_seconds=80.0,
            )

            # Add hashing stats
            perf_metrics.hashing = HashingStats(
                total_hashes_computed=2000,
                gpu_hashes=1900,
                cpu_hashes=100,
                failed_hashes=5,
            )

            perf_stats_data = {
                "last_run": perf_metrics.model_dump(mode="json"),
                "history": [],
                "total_runs": 1,
                "total_files_analyzed": 2000,
                "total_time_seconds": 150.0,
                "average_throughput": 13.3,
            }

            db.store_performance_statistics(perf_stats_data)
            db.save()

        # Reinitialize API
        init_catalog(catalog_dir)
        response = client.get("/api/performance/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["run_id"] == "test_run_summary"
        assert "summary" in data
        # Check that summary contains expected text
        summary = data["summary"]
        assert "Performance Analysis Summary" in summary
        assert "150.00s" in summary
        assert "2000" in summary  # Files analyzed
        assert "13.3" in summary  # Throughput


class TestPerformanceWebSocket:
    """Tests for performance WebSocket endpoint."""

    def test_websocket_connection(self, tmp_path: Path):
        """Test WebSocket connection for performance updates."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[tmp_path / "photos"])

        init_catalog(catalog_dir)
        client = TestClient(app)

        # Test WebSocket connection
        with client.websocket_connect("/ws/performance") as websocket:
            # Send ping
            websocket.send_text("ping")

            # Receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"


class TestCatalogPerformanceStatsMethods:
    """Tests for catalog performance statistics storage methods."""

    def test_store_performance_statistics(self, tmp_path: Path):
        """Test storing performance statistics in catalog."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[tmp_path / "photos"])

            perf_data = {
                "last_run": {
                    "run_id": "test_1",
                    "total_files_analyzed": 100,
                    "total_duration_seconds": 50.0,
                },
                "history": [],
                "total_runs": 1,
                "total_files_analyzed": 100,
                "total_time_seconds": 50.0,
                "average_throughput": 2.0,
            }

            db.store_performance_statistics(perf_data)
            db.save()

        # Reload and verify
        with CatalogDatabase(catalog_dir) as db:
            retrieved = db.get_performance_statistics()
            assert retrieved is not None
            assert retrieved["last_run"]["run_id"] == "test_1"
            assert retrieved["total_runs"] == 1

    def test_get_performance_statistics_not_exist(self, tmp_path: Path):
        """Test getting performance stats when they don't exist."""
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        with CatalogDatabase(catalog_dir) as db:
            db.initialize(source_directories=[tmp_path / "photos"])

            stats = db.get_performance_statistics()
            # Should return None if not set
            assert stats is None or "last_run" not in stats
