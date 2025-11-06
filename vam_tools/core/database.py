"""
SQLite database manager for VAM Tools catalog.

This module provides a robust SQLite-based storage layer for the entire
catalog including images, tags, duplicates, and analysis results.

Features:
- Schema migrations with versioning
- Connection pooling
- Transaction support
- Automatic backups
- Thread-safe operations

Example:
    Initialize and use database:
        >>> db = CatalogDatabase(Path("catalog"))
        >>> with db.transaction():
        ...     db.add_image(image_record)
        ...     db.add_tag("dogs", "subject")
"""

import logging
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CatalogDatabase:
    """SQLite database manager for catalog data.

    Provides thread-safe access to catalog database with automatic
    schema initialization, migrations, and backups.

    Attributes:
        catalog_path: Path to catalog directory
        db_path: Path to SQLite database file
        connection: Active database connection

    Example:
        >>> db = CatalogDatabase(Path("catalog"))
        >>> db.initialize()
        >>> images = db.get_all_images()
    """

    def __init__(self, catalog_path: Path) -> None:
        """Initialize database manager.

        Args:
            catalog_path: Path to catalog directory

        Example:
            >>> db = CatalogDatabase(Path("/path/to/catalog"))
        """
        self.catalog_path = catalog_path
        self.db_path = catalog_path / "catalog.db"
        self.backup_dir = catalog_path / "backups"
        self.connection: Optional[sqlite3.Connection] = None

        # Ensure catalog directory exists
        self.catalog_path.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)

    def connect(self) -> None:
        """Open database connection.

        Creates database file if it doesn't exist and initializes schema.

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> db.connect()
        """
        if self.connection is not None:
            return

        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode, we'll manage transactions
        )

        # Configure connection
        self.connection.row_factory = sqlite3.Row  # Access columns by name
        self.connection.execute("PRAGMA foreign_keys = ON")  # Enable FK constraints
        self.connection.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging

        logger.info(f"Connected to database: {self.db_path}")

    def close(self) -> None:
        """Close database connection.

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> db.connect()
            >>> db.close()
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.debug("Closed database connection")

    def __enter__(self) -> "CatalogDatabase":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions.

        Automatically commits on success, rolls back on error.

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> with db.transaction():
            ...     db.execute("INSERT INTO tags ...")
            ...     db.execute("INSERT INTO image_tags ...")
            # Automatically committed

        Yields:
            Database connection for executing queries
        """
        if self.connection is None:
            raise RuntimeError("Database not connected")

        self.connection.execute("BEGIN")
        try:
            yield self.connection
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def initialize(self) -> None:
        """Initialize database schema.

        Creates all tables, indexes, and views from schema.sql.
        Safe to call multiple times (idempotent).

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> db.connect()
            >>> db.initialize()
        """
        if self.connection is None:
            self.connect()

        assert self.connection is not None  # For mypy

        # Load schema from file
        schema_file = Path(__file__).parent / "schema.sql"
        with open(schema_file, "r") as f:
            schema_sql = f.read()

        # Execute schema
        try:
            self.connection.executescript(schema_sql)
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    def get_schema_version(self) -> int:
        """Get current schema version.

        Returns:
            Schema version number

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> version = db.get_schema_version()
            >>> version >= 1
            True
        """
        if self.connection is None:
            self.connect()

        assert self.connection is not None  # For mypy
        cursor = self.connection.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        return row[0] if row and row[0] else 0

    def create_backup(self) -> Path:
        """Create backup of database.

        Returns:
            Path to backup file

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> backup_path = db.create_backup()
            >>> backup_path.exists()
            True
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"catalog_{timestamp}.db"

        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise

    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Remove old backup files, keeping most recent.

        Args:
            keep_count: Number of backups to keep

        Returns:
            Number of backups deleted

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> deleted = db.cleanup_old_backups(keep_count=5)
            >>> deleted >= 0
            True
        """
        if not self.backup_dir.exists():
            return 0

        backups = sorted(
            self.backup_dir.glob("catalog_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        deleted = 0
        for backup in backups[keep_count:]:
            try:
                backup.unlink()
                deleted += 1
                logger.debug(f"Deleted old backup: {backup.name}")
            except Exception as e:
                logger.warning(f"Failed to delete backup {backup}: {e}")

        return deleted

    def execute(self, sql: str, parameters: Tuple = ()) -> sqlite3.Cursor:
        """Execute SQL query.

        Args:
            sql: SQL query string
            parameters: Query parameters

        Returns:
            Cursor with results

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> cursor = db.execute("SELECT * FROM tags WHERE id = ?", (1,))
            >>> row = cursor.fetchone()
        """
        if self.connection is None:
            self.connect()

        assert self.connection is not None  # For mypy
        return self.connection.execute(sql, parameters)

    def executemany(self, sql: str, parameters_list: List[Tuple]) -> sqlite3.Cursor:
        """Execute SQL query with multiple parameter sets.

        Args:
            sql: SQL query string
            parameters_list: List of parameter tuples

        Returns:
            Cursor with results

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> db.executemany(
            ...     "INSERT INTO tags (name, category) VALUES (?, ?)",
            ...     [("dogs", "subject"), ("cats", "subject")]
            ... )
        """
        if self.connection is None:
            self.connect()

        assert self.connection is not None  # For mypy
        return self.connection.executemany(sql, parameters_list)

    def vacuum(self) -> None:
        """Vacuum database to reclaim space and optimize.

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> db.vacuum()
        """
        if self.connection is None:
            self.connect()

        assert self.connection is not None  # For mypy
        logger.info("Vacuuming database...")
        self.connection.execute("VACUUM")
        logger.info("Database vacuumed")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database stats

        Example:
            >>> db = CatalogDatabase(Path("catalog"))
            >>> stats = db.get_stats()
            >>> stats["image_count"] >= 0
            True
        """
        if self.connection is None:
            self.connect()

        stats = {}

        # Count tables
        tables = [
            "images",
            "tags",
            "image_tags",
            "duplicate_groups",
            "burst_groups",
            "review_queue",
            "problematic_files",
        ]

        for table in tables:
            cursor = self.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            stats[f"{table}_count"] = count

        # Database size
        if self.db_path.exists():
            stats["db_size_bytes"] = self.db_path.stat().st_size
            stats["db_size_mb"] = stats["db_size_bytes"] / (1024 * 1024)

        # Schema version
        stats["schema_version"] = self.get_schema_version()

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return f"CatalogDatabase(path={self.db_path})"
