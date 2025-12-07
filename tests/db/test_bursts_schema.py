"""Tests for bursts table schema."""

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


class TestBurstsTable:
    """Tests for bursts table."""

    def test_bursts_table_exists(self, db_session):
        """Test that bursts table exists."""
        result = db_session.execute(
            text(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_name = 'bursts'
            """
            )
        )
        assert result.fetchone() is not None

    def test_bursts_table_has_required_columns(self, db_session):
        """Test that bursts table has all required columns."""
        result = db_session.execute(
            text(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'bursts'
            """
            )
        )
        columns = {row[0] for row in result.fetchall()}

        required = {
            "id",
            "catalog_id",
            "image_count",
            "start_time",
            "end_time",
            "duration_seconds",
            "camera_make",
            "camera_model",
            "best_image_id",
            "selection_method",
            "created_at",
        }
        assert required.issubset(columns)

    def test_images_has_burst_columns(self, db_session):
        """Test that images table has burst-related columns."""
        result = db_session.execute(
            text(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'images'
                AND column_name IN ('burst_id', 'burst_sequence')
            """
            )
        )
        columns = {row[0] for row in result.fetchall()}
        assert "burst_id" in columns
        assert "burst_sequence" in columns
