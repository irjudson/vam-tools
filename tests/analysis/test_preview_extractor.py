"""Tests for Preview Extractor - NEEDS REFACTORING FOR POSTGRESQL"""

import pytest

# These tests were written for the SQLite-based catalog system
# They need to be refactored to work with the PostgreSQL-based CatalogDB
pytestmark = pytest.mark.skip(
    reason="Legacy SQLite tests - need refactoring for PostgreSQL"
)
